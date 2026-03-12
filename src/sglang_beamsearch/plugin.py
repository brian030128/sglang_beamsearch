"""Monkeypatch plugin for SGLang beam search.

Four patches following the FastTree pattern:
  1. TpModelWorker.__init__  — disable CUDA graphs
  2. Scheduler.__init__      — attach BeamSearchManager
  3. process_batch_result_prefill — detect beam search reqs, expand 1→K beams
  4. process_batch_result_decode  — beam scoring, forking, pruning
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang_beamsearch.beam_state import (
    BeamInfo,
    BeamSearchConfig,
    BeamSearchManager,
    BeamSearchState,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Custom key on SamplingParams.custom_params to signal beam search
BEAM_SEARCH_KEY = "__beam_search__"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_beam_rid(group_id: str, idx: int) -> str:
    return f"{group_id}_b{idx}_{uuid.uuid4().hex[:8]}"


def _is_beam_search_req(req: "Req") -> bool:
    """Check if a request has beam search config in its custom_params."""
    cp = getattr(req.sampling_params, "custom_params", None)
    if cp is None or not isinstance(cp, dict):
        return False
    return BEAM_SEARCH_KEY in cp


def _get_beam_config(req: "Req") -> BeamSearchConfig:
    """Extract BeamSearchConfig from a request's custom_params."""
    cfg = req.sampling_params.custom_params[BEAM_SEARCH_KEY]
    return BeamSearchConfig(
        beam_width=cfg["beam_width"],
        max_new_tokens=cfg["max_new_tokens"],
    )


def _fork_req(
    parent_req: "Req",
    new_rid: str,
    new_token_id: int,
    req_to_token_pool,
) -> "Req":
    """Create a new Req sharing the parent's KV cache via index copying.

    The new req gets its own req_pool_idx but points to the same physical
    KV cache slots as the parent (no data copy).
    """
    from sglang.srt.managers.schedule_batch import Req

    child = Req(
        rid=new_rid,
        origin_input_text=parent_req.origin_input_text,
        origin_input_ids=list(parent_req.origin_input_ids),
        sampling_params=copy.copy(parent_req.sampling_params),
        return_logprob=parent_req.return_logprob,
        top_logprobs_num=getattr(parent_req, "top_logprobs_num", 0),
        stream=False,
        eos_token_ids=getattr(parent_req, "eos_token_ids", None),
    )

    # Copy decode state from parent
    child.output_ids = list(parent_req.output_ids)
    child.fill_ids = list(parent_req.fill_ids)
    child.tokenizer = parent_req.tokenizer
    child.kv_committed_len = parent_req.kv_committed_len
    child.kv_allocated_len = parent_req.kv_allocated_len

    # Allocate a new req_pool_idx
    free_slots = req_to_token_pool.free_slots
    if not free_slots:
        raise RuntimeError("No free req_pool slots for beam fork")
    new_pool_idx = free_slots[0]
    req_to_token_pool.free_slots = free_slots[1:]
    child.req_pool_idx = new_pool_idx

    # Copy the parent's token→KV index mapping (shares physical KV slots)
    parent_idx = parent_req.req_pool_idx
    seq_len = len(parent_req.origin_input_ids) + len(parent_req.output_ids)
    req_to_token_pool.req_to_token[new_pool_idx, :seq_len] = (
        req_to_token_pool.req_to_token[parent_idx, :seq_len]
    )

    # Append the new diverging token
    child.output_ids.append(new_token_id)
    child.fill_ids.append(new_token_id)

    return child


def _prune_beam_req(req: "Req", req_to_token_pool):
    """Remove a pruned beam's req_pool_idx (but NOT its KV slots).

    KV slots are shared with siblings, so we only free the pool index.
    """
    if req.req_pool_idx is not None:
        req_to_token_pool.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    req.finished_reason = FINISH_ABORT()


def _add_reqs_to_batch(batch: "ScheduleBatch", new_reqs: list["Req"]):
    """Add forked beam reqs to the running batch's tensor structures.

    Also extends sampling_info tensors so the forward pass sees the right
    batch size. Beam reqs use temperature=1, top_p=1, top_k=-1 (raw logits).
    """
    device = batch.req_pool_indices.device
    n = len(new_reqs)
    if n == 0:
        return

    new_pool_indices = []
    new_seq_lens = []

    for req in new_reqs:
        batch.reqs.append(req)
        new_pool_indices.append(req.req_pool_idx)
        seq_len = len(req.origin_input_ids) + len(req.output_ids)
        new_seq_lens.append(seq_len)

    pool_t = torch.tensor(
        new_pool_indices, dtype=batch.req_pool_indices.dtype, device=device
    )
    seq_t = torch.tensor(
        new_seq_lens, dtype=batch.seq_lens.dtype, device=device
    )

    batch.req_pool_indices = torch.cat([batch.req_pool_indices, pool_t])
    batch.seq_lens = torch.cat([batch.seq_lens, seq_t])

    if hasattr(batch, "seq_lens_cpu") and batch.seq_lens_cpu is not None:
        cpu_t = torch.tensor(new_seq_lens, dtype=batch.seq_lens_cpu.dtype)
        batch.seq_lens_cpu = torch.cat([batch.seq_lens_cpu, cpu_t])

    if hasattr(batch, "orig_seq_lens") and batch.orig_seq_lens is not None:
        batch.orig_seq_lens = torch.cat(
            [batch.orig_seq_lens, seq_t.clone()]
        )

    if batch.output_ids is not None:
        # output_ids tensor tracks the last output token for each req
        new_out = torch.tensor(
            [req.output_ids[-1] if req.output_ids else 0 for req in new_reqs],
            dtype=batch.output_ids.dtype,
            device=batch.output_ids.device,
        )
        batch.output_ids = torch.cat([batch.output_ids, new_out])

    batch.seq_lens_sum = batch.seq_lens.sum().item()

    # Extend sampling_info tensors (beam reqs: temp=1, top_p=1, top_k=-1, min_p=0)
    si = getattr(batch, "sampling_info", None)
    if si is not None:
        for attr, fill_val in [
            ("temperatures", 1.0),
            ("top_ps", 1.0),
            ("top_ks", -1),
            ("min_ps", 0.0),
        ]:
            t = getattr(si, attr, None)
            if t is not None:
                ext = torch.full((n,), fill_val, dtype=t.dtype, device=t.device)
                setattr(si, attr, torch.cat([t, ext]))

        if hasattr(si, "sampling_seed") and si.sampling_seed is not None:
            ext = torch.zeros(n, dtype=si.sampling_seed.dtype, device=si.sampling_seed.device)
            si.sampling_seed = torch.cat([si.sampling_seed, ext])

        if hasattr(si, "logit_bias") and si.logit_bias is not None:
            ext = torch.zeros(
                (n, si.logit_bias.shape[1]),
                dtype=si.logit_bias.dtype,
                device=si.logit_bias.device,
            )
            si.logit_bias = torch.cat([si.logit_bias, ext])


# --------------------------------------------------------------------------- #
# Patch 1: TpModelWorker.__init__ — disable CUDA graphs
# --------------------------------------------------------------------------- #


def _patch_tp_worker_init():
    from sglang.srt.managers.tp_worker import TpModelWorker

    _old_init = TpModelWorker.__init__

    def _new_init(self, server_args, *args, **kwargs):
        server_args.disable_cuda_graph = True
        _old_init(self, server_args, *args, **kwargs)

    TpModelWorker.__init__ = _new_init
    logger.info("Patched TpModelWorker.__init__ (CUDA graphs disabled)")


# --------------------------------------------------------------------------- #
# Patch 2: Scheduler.__init__ — attach BeamSearchManager
# --------------------------------------------------------------------------- #


def _patch_scheduler_init():
    from sglang.srt.managers.scheduler import Scheduler

    _old_init = Scheduler.__init__

    def _new_init(self, *args, **kwargs):
        _old_init(self, *args, **kwargs)
        self.beam_manager = BeamSearchManager()
        logger.info("BeamSearchManager attached to Scheduler")

    Scheduler.__init__ = _new_init
    logger.info("Patched Scheduler.__init__")


# --------------------------------------------------------------------------- #
# Patch 3: process_batch_result_prefill — beam expansion after prefill
# --------------------------------------------------------------------------- #


def _patch_process_batch_result_prefill():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    _old_process = SchedulerOutputProcessorMixin.process_batch_result_prefill

    def _new_process_batch_result_prefill(
        self: "Scheduler",
        batch: "ScheduleBatch",
        result,
    ):
        beam_manager: BeamSearchManager = getattr(self, "beam_manager", None)

        # Identify beam search requests BEFORE calling the original handler
        beam_search_reqs = []
        if beam_manager is not None:
            for req in batch.reqs:
                if _is_beam_search_req(req) and not beam_manager.is_beam_search_req(req.rid):
                    beam_search_reqs.append(req)

        # Call original prefill processing — this appends one sampled token
        _old_process(self, batch, result)

        if not beam_search_reqs or beam_manager is None:
            return

        # After prefill, each beam search req has 1 token in output_ids.
        # We need to:
        # 1. Create BeamSearchState for the req
        # 2. Get logits and compute top-K tokens
        # 3. Fork the req into K beams (the original req becomes beam 0)

        logits_output = result.logits_output
        if logits_output is None or logits_output.next_token_logits is None:
            return

        for req in beam_search_reqs:
            if req.finished():
                continue

            config = _get_beam_config(req)
            K = config.beam_width

            # Find this req's index in the batch to get its logits
            try:
                req_idx = batch.reqs.index(req)
            except ValueError:
                continue

            # Get logits for this req (already on GPU from forward pass)
            req_logits = logits_output.next_token_logits[req_idx]  # [vocab]
            log_probs = F.log_softmax(req_logits.float(), dim=-1)

            # Top-K tokens from prefill logits
            topk_log_probs, topk_ids = log_probs.topk(K, dim=-1)
            topk_log_probs = topk_log_probs.tolist()
            topk_ids = topk_ids.tolist()

            # Create beam search state
            state = beam_manager.create_beam_search(config, req.rid)

            # Beam 0: reuse the original req (it already has the sampled token).
            # Replace its output_ids[-1] with the top-1 token from beam search.
            req.output_ids[-1] = topk_ids[0]
            if req.fill_ids:
                req.fill_ids[-1] = topk_ids[0]

            beam0 = BeamInfo(
                rid=req.rid,
                cum_log_prob=topk_log_probs[0],
                token_ids=[topk_ids[0]],
                prompt_group_id=state.prompt_group_id,
            )
            state.beams.append(beam0)
            beam_manager.register_beam(beam0)

            # Beams 1..K-1: fork from the original req (before token append)
            # We need to "undo" the token from the parent's output_ids temporarily
            # Actually, the parent already has topk_ids[0]. For beams 1..K-1,
            # we fork from a state that has the prompt KV but NOT the new token KV.
            # Since the parent's KV cache includes the token just appended by prefill,
            # all forks should share the prompt KV. We copy mapping up to seq_len-1
            # (prompt only) then add the divergent token.

            # Forked beams need to share the prompt KV but have different first tokens.
            # The parent's req_to_token mapping includes the prompt + 1st token.
            # For fork, we copy just the prompt part and add a different token.
            for i in range(1, K):
                new_rid = _make_beam_rid(state.prompt_group_id, i)

                # Fork from parent, but we need to handle the token replacement.
                # The simplest approach: fork the req (which copies parent's full
                # KV mapping including token 0), then overwrite the last token.
                child = _fork_req(
                    parent_req=req,
                    new_rid=new_rid,
                    new_token_id=topk_ids[i],
                    req_to_token_pool=batch.req_to_token_pool,
                )
                # The child's output_ids = parent.output_ids + [topk_ids[i]]
                # But parent.output_ids already has [topk_ids[0]], so child has
                # [topk_ids[0], topk_ids[i]] — that's wrong.
                # Fix: child should have [topk_ids[i]] only.
                child.output_ids = [topk_ids[i]]
                child.fill_ids = list(req.origin_input_ids) + [topk_ids[i]]

                beam_info = BeamInfo(
                    rid=new_rid,
                    cum_log_prob=topk_log_probs[i],
                    token_ids=[topk_ids[i]],
                    prompt_group_id=state.prompt_group_id,
                )
                state.beams.append(beam_info)
                beam_manager.register_beam(beam_info)

                # Add the child to the running batch (it will join decode next step)
                # We need to add it to self.running_batch, not the prefill batch
                if not hasattr(self, "_beam_pending_reqs"):
                    self._beam_pending_reqs = []
                self._beam_pending_reqs.append(child)

            state.step = 1  # We've generated 1 token
            logger.info(
                f"Beam search initialized for {req.rid}: "
                f"group={state.prompt_group_id}, K={K}, "
                f"top tokens={topk_ids}"
            )

    SchedulerOutputProcessorMixin.process_batch_result_prefill = (
        _new_process_batch_result_prefill
    )
    logger.info("Patched process_batch_result_prefill for beam expansion")


# --------------------------------------------------------------------------- #
# Patch 4: process_batch_result_decode — beam scoring + forking + pruning
# --------------------------------------------------------------------------- #


def _patch_process_batch_result_decode():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    _old_process = SchedulerOutputProcessorMixin.process_batch_result_decode

    def _new_process_batch_result_decode(
        self: "Scheduler",
        batch: "ScheduleBatch",
        result: "GenerationBatchResult",
    ):
        beam_manager: BeamSearchManager = getattr(self, "beam_manager", None)
        if beam_manager is None:
            return _old_process(self, batch, result)

        # Partition reqs into beam-search vs normal
        beam_indices = []
        normal_indices = []
        for i, req in enumerate(batch.reqs):
            if beam_manager.is_beam_search_req(req.rid):
                beam_indices.append(i)
            else:
                normal_indices.append(i)

        if not beam_indices:
            return _old_process(self, batch, result)

        # --- Synchronize GPU→CPU copy ---
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output = result.logits_output
        next_token_ids = result.next_token_ids

        # --- Handle normal (non-beam) requests with original logic ---
        self.token_to_kv_pool_allocator.free_group_begin()

        if normal_indices:
            normal_ids = next_token_ids.tolist()
            for batch_idx in normal_indices:
                req = batch.reqs[batch_idx]
                tid = normal_ids[batch_idx]
                req.output_ids.append(tid)
                req.check_finished()
                if req.finished():
                    from sglang.srt.mem_cache.common import release_kv_cache

                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()

        # --- Beam search scoring ---
        beam_device = logits_output.next_token_logits.device
        beam_dev_indices = torch.tensor(beam_indices, device=beam_device)
        beam_logits = logits_output.next_token_logits[beam_dev_indices]
        beam_log_probs = F.log_softmax(beam_logits.float(), dim=-1)
        vocab_size = beam_log_probs.shape[-1]

        # Group beam reqs by prompt_group_id
        groups: dict[str, list[tuple[int, int, "Req"]]] = {}
        # group_id -> [(batch_idx, position_in_beam_logits, req), ...]
        for pos_in_beam, batch_idx in enumerate(beam_indices):
            req = batch.reqs[batch_idx]
            state = beam_manager.get_state_for_rid(req.rid)
            if state is None:
                continue
            gid = state.prompt_group_id
            if gid not in groups:
                groups[gid] = []
            groups[gid].append((batch_idx, pos_in_beam, req))

        reqs_to_add: list["Req"] = []

        for gid, group_entries in groups.items():
            state = beam_manager.states[gid]
            K = state.config.beam_width
            K_current = len(group_entries)

            # Collect per-beam log probs and cumulative scores
            group_log_probs = torch.stack(
                [beam_log_probs[pos] for _, pos, _ in group_entries]
            )  # [K_current, vocab]

            cum_probs = torch.tensor(
                [state.get_beam_by_rid(req.rid).cum_log_prob for _, _, req in group_entries],
                device=beam_device,
                dtype=torch.float32,
            )  # [K_current]

            # Score: cum_log_prob + log_prob for each (beam, token)
            scores = cum_probs[:, None] + group_log_probs  # [K_current, vocab]
            flat_scores = scores.reshape(-1)

            # Top-K selection across all beams × vocab
            topk_scores, topk_flat_ids = flat_scores.topk(K, dim=-1)
            parent_local_ids = (topk_flat_ids // vocab_size).tolist()
            new_tids = (topk_flat_ids % vocab_size).tolist()
            topk_scores_list = topk_scores.tolist()

            # Count parent usage for forking/pruning
            parent_usage = [0] * K_current
            for pid in parent_local_ids:
                parent_usage[pid] += 1

            # Build mapping: which parent reqs can we reuse in-place?
            # Strategy: for each parent that's used at least once, reuse it for
            # the first child. For additional children, fork.
            parent_claimed = [False] * K_current
            new_beams: list[BeamInfo] = []
            new_req_objects: list["Req"] = []  # parallel to new_beams

            for i in range(K):
                pid = parent_local_ids[i]
                batch_idx, _, parent_req = group_entries[pid]
                new_tid = new_tids[i]
                new_score = topk_scores_list[i]
                parent_beam = state.get_beam_by_rid(parent_req.rid)

                if not parent_claimed[pid]:
                    # Reuse parent req in-place
                    parent_claimed[pid] = True
                    parent_req.output_ids.append(new_tid)
                    parent_req.fill_ids.append(new_tid)

                    parent_beam.cum_log_prob = new_score
                    parent_beam.token_ids.append(new_tid)
                    new_beams.append(parent_beam)
                    new_req_objects.append(parent_req)
                else:
                    # Fork: new req sharing parent's KV cache
                    new_rid = _make_beam_rid(gid, i)
                    child_req = _fork_req(
                        parent_req=parent_req,
                        new_rid=new_rid,
                        new_token_id=new_tid,
                        req_to_token_pool=batch.req_to_token_pool,
                    )
                    new_beam = BeamInfo(
                        rid=new_rid,
                        cum_log_prob=new_score,
                        token_ids=list(parent_beam.token_ids) + [new_tid],
                        prompt_group_id=gid,
                    )
                    beam_manager.register_beam(new_beam)
                    new_beams.append(new_beam)
                    new_req_objects.append(child_req)
                    reqs_to_add.append(child_req)

            # Prune: parent reqs not claimed by any child
            for local_idx in range(K_current):
                if not parent_claimed[local_idx]:
                    _, _, req = group_entries[local_idx]
                    _prune_beam_req(req, batch.req_to_token_pool)
                    beam_manager.unregister_beam(req.rid)

            state.beams = new_beams
            state.step += 1

            # Check if beam search is complete
            if state.step >= state.config.max_new_tokens:
                state.finished = True
                logger.info(
                    f"Beam search {gid} done after {state.step} steps. "
                    f"Best: score={new_beams[0].cum_log_prob:.4f}, "
                    f"len={len(new_beams[0].token_ids)}"
                )
                for req_obj in new_req_objects:
                    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

                    req_obj.finished_reason = FINISH_LENGTH(
                        length=len(req_obj.output_ids)
                    )
            else:
                # Check per-beam EOS (don't end prematurely)
                for req_obj in new_req_objects:
                    req_obj.check_finished()

        # Add forked reqs to batch
        if reqs_to_add:
            _add_reqs_to_batch(batch, reqs_to_add)

        # Stream partial output
        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool_allocator.free_group_end()
        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)

    SchedulerOutputProcessorMixin.process_batch_result_decode = (
        _new_process_batch_result_decode
    )
    logger.info("Patched process_batch_result_decode for beam search")


# --------------------------------------------------------------------------- #
# Patch 5: get_next_batch_to_run — inject pending beam reqs into running batch
# --------------------------------------------------------------------------- #


def _patch_get_next_batch_to_run():
    from sglang.srt.managers.scheduler import Scheduler

    _old_get_next = Scheduler.get_next_batch_to_run

    def _new_get_next_batch_to_run(self):
        batch = _old_get_next(self)

        # Inject any pending beam reqs that were created during prefill
        pending = getattr(self, "_beam_pending_reqs", None)
        if pending and batch is not None and not batch.is_empty():
            _add_reqs_to_batch(batch, pending)
            self._beam_pending_reqs = []

        return batch

    Scheduler.get_next_batch_to_run = _new_get_next_batch_to_run
    logger.info("Patched get_next_batch_to_run for beam req injection")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def apply_beam_search_patches():
    """Apply all beam search monkeypatches to SGLang."""
    _patch_tp_worker_init()
    _patch_scheduler_init()
    _patch_process_batch_result_prefill()
    _patch_process_batch_result_decode()
    _patch_get_next_batch_to_run()
    logger.info("All beam search patches applied successfully")


# --------------------------------------------------------------------------- #
# Request-side API: mark a request for beam search
# --------------------------------------------------------------------------- #


def configure_beam_search_request(
    sampling_params,
    beam_width: int = 4,
    max_new_tokens: int = 128,
):
    """Configure a SamplingParams object to trigger beam search.

    Call this before sending the request to SGLang. The plugin will detect
    the custom_params key and route the request through beam search.
    """
    if sampling_params.custom_params is None:
        sampling_params.custom_params = {}
    sampling_params.custom_params[BEAM_SEARCH_KEY] = {
        "beam_width": beam_width,
        "max_new_tokens": max_new_tokens,
    }
    # Beam search manages its own stopping — set max_new_tokens high enough
    sampling_params.max_new_tokens = max_new_tokens + 1
    # Need raw logits — temperature=1 gives unscaled log_softmax
    sampling_params.temperature = 1.0
    sampling_params.top_p = 1.0
    sampling_params.top_k = -1
