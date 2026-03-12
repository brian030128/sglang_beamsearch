"""Beam search state management for SGLang beam search plugin."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchConfig:
    beam_width: int
    max_new_tokens: int


@dataclass
class BeamInfo:
    rid: str  # SGLang request ID for this beam
    cum_log_prob: float  # Cumulative log probability
    token_ids: list[int]  # Generated tokens (decode only, no prompt)
    prompt_group_id: str  # Groups beams from same original prompt

    def __repr__(self):
        return (
            f"BeamInfo(rid={self.rid!r}, cum_log_prob={self.cum_log_prob:.4f}, "
            f"tokens={len(self.token_ids)}, group={self.prompt_group_id!r})"
        )


@dataclass
class BeamSearchState:
    """Tracks all beams for one original prompt."""

    config: BeamSearchConfig
    prompt_group_id: str
    beams: list[BeamInfo] = field(default_factory=list)
    step: int = 0
    finished: bool = False

    # Ref-counting for shared KV slots.
    # Maps kv_slot_index -> ref_count. When a beam is forked, shared slots
    # get their ref count bumped. When a beam is pruned, ref counts are
    # decremented; slots are only actually freed when ref_count reaches 0.
    kv_slot_refs: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Maps rid -> list of kv_slot_indices owned by that beam
    # (slots where this beam diverged from parent and wrote new KV)
    rid_to_unique_slots: dict[str, list[int]] = field(default_factory=dict)

    def add_ref(self, slot: int, count: int = 1):
        self.kv_slot_refs[slot] += count

    def remove_ref(self, slot: int, count: int = 1) -> bool:
        """Decrement ref count. Returns True if slot should be freed (ref_count hit 0)."""
        self.kv_slot_refs[slot] -= count
        if self.kv_slot_refs[slot] <= 0:
            del self.kv_slot_refs[slot]
            return True
        return False

    def get_beam_by_rid(self, rid: str) -> Optional[BeamInfo]:
        for beam in self.beams:
            if beam.rid == rid:
                return beam
        return None


class BeamSearchManager:
    """Manages all active beam searches across requests.

    Attached to the Scheduler via monkeypatch. Provides lookup from
    individual beam rids to their BeamSearchState group.
    """

    def __init__(self):
        # prompt_group_id -> BeamSearchState
        self.states: dict[str, BeamSearchState] = {}
        # Individual beam rid -> prompt_group_id (for fast lookup)
        self.rid_to_group: dict[str, str] = {}

    def create_beam_search(
        self, config: BeamSearchConfig, initial_rid: str
    ) -> BeamSearchState:
        """Create a new beam search for a prompt. Called when a beam search request arrives."""
        group_id = f"beam_{uuid.uuid4().hex[:12]}"
        state = BeamSearchState(config=config, prompt_group_id=group_id)
        self.states[group_id] = state
        self.rid_to_group[initial_rid] = group_id
        return state

    def get_state_for_rid(self, rid: str) -> Optional[BeamSearchState]:
        group_id = self.rid_to_group.get(rid)
        if group_id is None:
            return None
        return self.states.get(group_id)

    def register_beam(self, beam: BeamInfo):
        """Register a new beam (e.g., after forking)."""
        self.rid_to_group[beam.rid] = beam.prompt_group_id

    def unregister_beam(self, rid: str):
        """Remove a beam from tracking (e.g., after pruning)."""
        self.rid_to_group.pop(rid, None)

    def remove_beam_search(self, group_id: str):
        """Clean up a completed beam search."""
        state = self.states.pop(group_id, None)
        if state:
            for beam in state.beams:
                self.rid_to_group.pop(beam.rid, None)

    def is_beam_search_req(self, rid: str) -> bool:
        return rid in self.rid_to_group

    def get_all_group_rids(self, group_id: str) -> list[str]:
        """Get all beam rids belonging to a group."""
        state = self.states.get(group_id)
        if state is None:
            return []
        return [b.rid for b in state.beams]
