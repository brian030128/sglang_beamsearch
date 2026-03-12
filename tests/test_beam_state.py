"""Unit tests for beam state management and scoring logic.

These tests don't require a GPU or running SGLang server.
They verify the pure-Python beam state tracking and the
beam scoring algorithm (top-K selection, forking, pruning).
"""

import copy

import torch
import torch.nn.functional as F
import pytest

from sglang_beamsearch.beam_state import (
    BeamInfo,
    BeamSearchConfig,
    BeamSearchManager,
    BeamSearchState,
)


# --------------------------------------------------------------------------- #
# BeamSearchState tests
# --------------------------------------------------------------------------- #


class TestBeamSearchState:
    def test_create_state(self):
        cfg = BeamSearchConfig(beam_width=4, max_new_tokens=10)
        state = BeamSearchState(config=cfg, prompt_group_id="test_group")
        assert state.step == 0
        assert state.finished is False
        assert len(state.beams) == 0

    def test_add_and_get_beam(self):
        cfg = BeamSearchConfig(beam_width=2, max_new_tokens=10)
        state = BeamSearchState(config=cfg, prompt_group_id="g1")

        b0 = BeamInfo(rid="r0", cum_log_prob=-1.0, token_ids=[42], prompt_group_id="g1")
        b1 = BeamInfo(rid="r1", cum_log_prob=-2.0, token_ids=[99], prompt_group_id="g1")
        state.beams = [b0, b1]

        assert state.get_beam_by_rid("r0") is b0
        assert state.get_beam_by_rid("r1") is b1
        assert state.get_beam_by_rid("nonexistent") is None

    def test_ref_counting(self):
        cfg = BeamSearchConfig(beam_width=2, max_new_tokens=10)
        state = BeamSearchState(config=cfg, prompt_group_id="g1")

        # Add refs for shared KV slots
        state.add_ref(100, count=3)
        assert state.kv_slot_refs[100] == 3

        # Remove one ref — still alive
        freed = state.remove_ref(100, count=1)
        assert freed is False
        assert state.kv_slot_refs[100] == 2

        # Remove remaining refs — should free
        freed = state.remove_ref(100, count=2)
        assert freed is True
        assert 100 not in state.kv_slot_refs


# --------------------------------------------------------------------------- #
# BeamSearchManager tests
# --------------------------------------------------------------------------- #


class TestBeamSearchManager:
    def test_create_and_lookup(self):
        mgr = BeamSearchManager()
        cfg = BeamSearchConfig(beam_width=4, max_new_tokens=10)

        state = mgr.create_beam_search(cfg, initial_rid="req_0")

        assert mgr.is_beam_search_req("req_0")
        assert not mgr.is_beam_search_req("unknown")
        assert mgr.get_state_for_rid("req_0") is state

    def test_register_and_unregister(self):
        mgr = BeamSearchManager()
        cfg = BeamSearchConfig(beam_width=2, max_new_tokens=5)
        state = mgr.create_beam_search(cfg, initial_rid="req_0")

        beam = BeamInfo(
            rid="fork_1", cum_log_prob=-1.0,
            token_ids=[10], prompt_group_id=state.prompt_group_id,
        )
        mgr.register_beam(beam)
        assert mgr.is_beam_search_req("fork_1")
        assert mgr.get_state_for_rid("fork_1") is state

        mgr.unregister_beam("fork_1")
        assert not mgr.is_beam_search_req("fork_1")

    def test_remove_beam_search_cleans_all(self):
        mgr = BeamSearchManager()
        cfg = BeamSearchConfig(beam_width=2, max_new_tokens=5)
        state = mgr.create_beam_search(cfg, initial_rid="req_0")

        b0 = BeamInfo(rid="req_0", cum_log_prob=-1.0, token_ids=[1], prompt_group_id=state.prompt_group_id)
        b1 = BeamInfo(rid="fork_1", cum_log_prob=-2.0, token_ids=[2], prompt_group_id=state.prompt_group_id)
        state.beams = [b0, b1]
        mgr.register_beam(b0)
        mgr.register_beam(b1)

        mgr.remove_beam_search(state.prompt_group_id)
        assert not mgr.is_beam_search_req("req_0")
        assert not mgr.is_beam_search_req("fork_1")
        assert state.prompt_group_id not in mgr.states

    def test_multiple_concurrent_searches(self):
        mgr = BeamSearchManager()
        cfg = BeamSearchConfig(beam_width=2, max_new_tokens=5)

        s1 = mgr.create_beam_search(cfg, initial_rid="a_0")
        s2 = mgr.create_beam_search(cfg, initial_rid="b_0")

        assert mgr.get_state_for_rid("a_0") is s1
        assert mgr.get_state_for_rid("b_0") is s2
        assert s1.prompt_group_id != s2.prompt_group_id


# --------------------------------------------------------------------------- #
# Beam scoring algorithm tests (pure tensor math, no SGLang deps)
# --------------------------------------------------------------------------- #


def beam_score_step(
    log_probs: torch.Tensor,  # [K, vocab]
    cum_log_probs: list[float],  # [K]
    beam_width: int,
) -> tuple[list[int], list[int], list[float]]:
    """Pure implementation of one beam search scoring step.

    Returns (parent_ids, new_token_ids, new_cum_log_probs).
    """
    K_current = log_probs.shape[0]
    vocab_size = log_probs.shape[1]

    cum = torch.tensor(cum_log_probs, dtype=torch.float32)
    scores = cum[:, None] + log_probs  # [K, vocab]
    flat = scores.reshape(-1)

    topk_scores, topk_flat = flat.topk(beam_width, dim=-1)
    parent_ids = (topk_flat // vocab_size).tolist()
    new_tids = (topk_flat % vocab_size).tolist()
    new_cum = topk_scores.tolist()
    return parent_ids, new_tids, new_cum


class TestBeamScoring:
    def test_greedy_single_beam(self):
        """With beam_width=1, should pick the max-prob token each step."""
        vocab = 10
        logits = torch.randn(1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        expected_token = logits.argmax(dim=-1).item()

        parents, tids, scores = beam_score_step(log_probs, [0.0], beam_width=1)
        assert tids[0] == expected_token
        assert parents[0] == 0

    def test_beam_width_2_from_single(self):
        """Expand from 1 beam to 2: should pick top-2 tokens."""
        vocab = 5
        # Hand-crafted logits so we know the answer
        logits = torch.tensor([[10.0, 5.0, 1.0, 0.0, -5.0]])
        log_probs = F.log_softmax(logits, dim=-1)

        parents, tids, scores = beam_score_step(log_probs, [0.0], beam_width=2)
        # Token 0 (logit=10) and token 1 (logit=5) should be selected
        assert tids[0] == 0
        assert tids[1] == 1
        assert parents == [0, 0]  # both from the single parent
        assert scores[0] > scores[1]  # first is better

    def test_beam_width_2_from_two(self):
        """Two existing beams, select top-2 across both x vocab."""
        vocab = 4
        # Beam 0 (cum=-1.0): logits favor token 2
        # Beam 1 (cum=-2.0): logits favor token 0
        logits = torch.tensor([
            [0.0, 0.0, 10.0, 0.0],   # beam 0
            [10.0, 0.0, 0.0, 0.0],   # beam 1
        ])
        log_probs = F.log_softmax(logits, dim=-1)

        parents, tids, scores = beam_score_step(
            log_probs, [-1.0, -2.0], beam_width=2
        )
        # Beam 0 + token 2: -1.0 + ~0.0 = -1.0
        # Beam 1 + token 0: -2.0 + ~0.0 = -2.0
        # So beam 0/token 2 should be first
        assert parents[0] == 0
        assert tids[0] == 2
        assert scores[0] > scores[1]

    def test_forking_one_parent_multiple_children(self):
        """One parent can produce multiple children."""
        vocab = 3
        # Beam 0 has good tokens, beam 1 has terrible tokens.
        # Use extreme gap so beam 1 can't compete even with equal cum_log_prob.
        logits = torch.tensor([
            [100.0, 99.0, 98.0],   # beam 0: all great
            [-100.0, -100.0, -100.0],  # beam 1: all terrible
        ])
        log_probs = F.log_softmax(logits, dim=-1)

        parents, tids, scores = beam_score_step(
            log_probs, [0.0, 0.0], beam_width=3
        )
        # All 3 children should come from beam 0
        assert all(p == 0 for p in parents)
        assert tids == [0, 1, 2]

    def test_pruning_beam_not_selected(self):
        """Verify that parent usage can identify pruned beams."""
        vocab = 3
        # Beam 0 is clearly best, beam 1 is clearly worst, beam 2 is mid.
        # Give beam 1 a big cum_log_prob penalty so it gets pruned.
        logits = torch.tensor([
            [10.0, 9.0, 8.0],      # beam 0: all good
            [-100.0, -100.0, -100.0],  # beam 1: all terrible
            [5.0, 4.0, 3.0],       # beam 2: mediocre
        ])
        log_probs = F.log_softmax(logits, dim=-1)

        parents, tids, scores = beam_score_step(
            log_probs, [0.0, -50.0, 0.0], beam_width=3
        )

        # Count parent usage
        parent_usage = [0, 0, 0]
        for p in parents:
            parent_usage[p] += 1

        # Beam 1 should be pruned (0 usage)
        assert parent_usage[1] == 0
        # Beam 0 should be used
        assert parent_usage[0] >= 1

    def test_scores_are_cumulative(self):
        """Verify cum_log_prob accumulates correctly."""
        vocab = 3
        logits = torch.tensor([[5.0, 0.0, 0.0]])
        log_probs = F.log_softmax(logits, dim=-1)

        # Step 1: start from 0
        _, tids1, scores1 = beam_score_step(log_probs, [0.0], beam_width=1)
        assert tids1[0] == 0
        step1_log_prob = log_probs[0, 0].item()
        assert abs(scores1[0] - step1_log_prob) < 1e-5

        # Step 2: accumulate
        _, tids2, scores2 = beam_score_step(log_probs, scores1, beam_width=1)
        assert abs(scores2[0] - 2 * step1_log_prob) < 1e-5

    def test_multi_step_beam_search_simulation(self):
        """Simulate a full 3-step beam search with K=2, vocab=4."""
        K = 2
        vocab = 4
        torch.manual_seed(42)

        # Simulate 3 decode steps with random logits
        beams = [{"parent": -1, "tokens": [], "score": 0.0}]

        for step in range(3):
            K_cur = len(beams)
            logits = torch.randn(K_cur, vocab)
            log_probs = F.log_softmax(logits, dim=-1)
            cum = [b["score"] for b in beams]

            parents, tids, scores = beam_score_step(log_probs, cum, beam_width=K)

            new_beams = []
            for i in range(K):
                new_beams.append({
                    "parent": parents[i],
                    "tokens": beams[parents[i]]["tokens"] + [tids[i]],
                    "score": scores[i],
                })
            beams = new_beams

        # After 3 steps: should have K beams, each with 3 tokens
        assert len(beams) == K
        for b in beams:
            assert len(b["tokens"]) == 3
        # Scores should be sorted (best first from topk)
        assert beams[0]["score"] >= beams[1]["score"]


# --------------------------------------------------------------------------- #
# Plugin helper tests (mock SGLang objects)
# --------------------------------------------------------------------------- #


class MockSamplingParams:
    def __init__(self, custom_params=None):
        self.custom_params = custom_params
        self.max_new_tokens = 100
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = -1


class MockReq:
    def __init__(self, rid, sampling_params=None):
        self.rid = rid
        self.sampling_params = sampling_params or MockSamplingParams()
        self.output_ids = []
        self.fill_ids = []
        self.origin_input_ids = [1, 2, 3]
        self.origin_input_text = ""
        self.tokenizer = None
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.req_pool_idx = None
        self.return_logprob = False
        self.top_logprobs_num = 0
        self.finished_reason = None
        self.eos_token_ids = None

    def finished(self):
        return self.finished_reason is not None


class TestPluginHelpers:
    def test_is_beam_search_req(self):
        from sglang_beamsearch.plugin import _is_beam_search_req, BEAM_SEARCH_KEY

        normal = MockReq("r1")
        assert not _is_beam_search_req(normal)

        beam = MockReq("r2", MockSamplingParams(
            custom_params={BEAM_SEARCH_KEY: {"beam_width": 4, "max_new_tokens": 10}}
        ))
        assert _is_beam_search_req(beam)

    def test_get_beam_config(self):
        from sglang_beamsearch.plugin import _get_beam_config, BEAM_SEARCH_KEY

        req = MockReq("r1", MockSamplingParams(
            custom_params={BEAM_SEARCH_KEY: {"beam_width": 8, "max_new_tokens": 64}}
        ))
        cfg = _get_beam_config(req)
        assert cfg.beam_width == 8
        assert cfg.max_new_tokens == 64

    def test_configure_beam_search_request(self):
        from sglang_beamsearch.plugin import configure_beam_search_request, BEAM_SEARCH_KEY

        sp = MockSamplingParams()
        configure_beam_search_request(sp, beam_width=4, max_new_tokens=128)

        assert BEAM_SEARCH_KEY in sp.custom_params
        assert sp.custom_params[BEAM_SEARCH_KEY]["beam_width"] == 4
        assert sp.custom_params[BEAM_SEARCH_KEY]["max_new_tokens"] == 128
        assert sp.max_new_tokens == 129  # +1 so SGLang doesn't stop early
        assert sp.temperature == 1.0
        assert sp.top_p == 1.0
        assert sp.top_k == -1


# --------------------------------------------------------------------------- #
# Fork / prune logic test with mock req_to_token_pool
# --------------------------------------------------------------------------- #


class MockReqToTokenPool:
    def __init__(self, size=16, max_seq_len=128):
        self.req_to_token = torch.zeros(size, max_seq_len, dtype=torch.int32)
        self.free_slots = list(range(size))


def _mock_fork_req(parent_req, new_rid, new_token_id, req_to_token_pool):
    """Fork logic extracted from plugin, but constructs MockReq instead of real Req."""
    child = MockReq(new_rid, copy.copy(parent_req.sampling_params))
    child.origin_input_ids = list(parent_req.origin_input_ids)
    child.origin_input_text = parent_req.origin_input_text
    child.output_ids = list(parent_req.output_ids)
    child.fill_ids = list(parent_req.fill_ids)
    child.tokenizer = parent_req.tokenizer
    child.kv_committed_len = parent_req.kv_committed_len
    child.kv_allocated_len = parent_req.kv_allocated_len
    child.return_logprob = parent_req.return_logprob
    child.eos_token_ids = parent_req.eos_token_ids

    free_slots = req_to_token_pool.free_slots
    if not free_slots:
        raise RuntimeError("No free req_pool slots for beam fork")
    new_pool_idx = free_slots[0]
    req_to_token_pool.free_slots = free_slots[1:]
    child.req_pool_idx = new_pool_idx

    parent_idx = parent_req.req_pool_idx
    seq_len = len(parent_req.origin_input_ids) + len(parent_req.output_ids)
    req_to_token_pool.req_to_token[new_pool_idx, :seq_len] = (
        req_to_token_pool.req_to_token[parent_idx, :seq_len]
    )

    child.output_ids.append(new_token_id)
    child.fill_ids.append(new_token_id)
    return child


class TestForkAndPrune:
    def test_fork_req_copies_kv_mapping(self):
        pool = MockReqToTokenPool(size=8, max_seq_len=32)

        # Set up parent
        parent = MockReq("parent_0")
        parent.req_pool_idx = pool.free_slots.pop(0)  # slot 0
        parent.output_ids = [10, 20]
        parent.fill_ids = [1, 2, 3, 10, 20]
        parent.kv_committed_len = 5
        parent.kv_allocated_len = 5

        # Write fake KV indices for parent
        seq_len = len(parent.origin_input_ids) + len(parent.output_ids)  # 3 + 2 = 5
        pool.req_to_token[parent.req_pool_idx, :seq_len] = torch.arange(100, 100 + seq_len)

        child = _mock_fork_req(parent, "child_0", new_token_id=30, req_to_token_pool=pool)

        # Child should have its own pool idx
        assert child.req_pool_idx != parent.req_pool_idx
        assert child.req_pool_idx is not None

        # Child's KV mapping should match parent's for shared prefix
        parent_kv = pool.req_to_token[parent.req_pool_idx, :seq_len]
        child_kv = pool.req_to_token[child.req_pool_idx, :seq_len]
        assert torch.equal(parent_kv, child_kv)

        # Child output_ids should be parent's + new token
        assert child.output_ids == [10, 20, 30]
        assert child.fill_ids == [1, 2, 3, 10, 20, 30]
        assert child.rid == "child_0"

    def test_fork_consumes_free_slot(self):
        pool = MockReqToTokenPool(size=4, max_seq_len=16)
        initial_free = len(pool.free_slots)

        parent = MockReq("p")
        parent.req_pool_idx = pool.free_slots.pop(0)

        _mock_fork_req(parent, "c1", 1, pool)
        assert len(pool.free_slots) == initial_free - 2  # parent + child

    def test_fork_raises_when_no_slots(self):
        pool = MockReqToTokenPool(size=2, max_seq_len=16)
        parent = MockReq("p")
        parent.req_pool_idx = pool.free_slots.pop(0)
        pool.free_slots.pop(0)  # exhaust remaining

        with pytest.raises(RuntimeError, match="No free req_pool slots"):
            _mock_fork_req(parent, "c1", 1, pool)

    def test_prune_frees_pool_idx(self):
        from sglang_beamsearch.plugin import _prune_beam_req

        pool = MockReqToTokenPool(size=4, max_seq_len=16)
        req = MockReq("victim")
        req.req_pool_idx = pool.free_slots.pop(0)
        initial_free = len(pool.free_slots)

        _prune_beam_req(req, pool)

        assert req.req_pool_idx is None
        assert req.finished()
        assert len(pool.free_slots) == initial_free + 1

    def test_multiple_forks_share_kv(self):
        """Fork 3 children from same parent — all share same KV indices."""
        pool = MockReqToTokenPool(size=8, max_seq_len=32)

        parent = MockReq("p")
        parent.req_pool_idx = pool.free_slots.pop(0)
        parent.output_ids = [10]
        parent.fill_ids = [1, 2, 3, 10]
        seq_len = 4
        pool.req_to_token[parent.req_pool_idx, :seq_len] = torch.arange(200, 200 + seq_len)

        children = []
        for i in range(3):
            c = _mock_fork_req(parent, f"c{i}", new_token_id=50 + i, req_to_token_pool=pool)
            children.append(c)

        # All children share parent's prefix KV
        parent_kv = pool.req_to_token[parent.req_pool_idx, :seq_len]
        for c in children:
            child_kv = pool.req_to_token[c.req_pool_idx, :seq_len]
            assert torch.equal(parent_kv, child_kv)

        # Each child has unique pool idx
        all_idxs = [parent.req_pool_idx] + [c.req_pool_idx for c in children]
        assert len(set(all_idxs)) == 4

        # Each child has different last token
        assert [c.output_ids[-1] for c in children] == [50, 51, 52]
