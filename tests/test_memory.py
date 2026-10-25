import unittest
import numpy as np
from project_override.memory.action_ledger import ActionLedger, compute_arena_hash
from project_override.memory.hypothesis_store import HypothesisStore
from project_override.perception.observation_processor import Entity, ObservationProcessor

def test_arena_hash_and_dead_signature_pruning() -> None:
    proc = ObservationProcessor()
    ledger = ActionLedger()
    g0 = np.zeros((64, 64), dtype=np.int8)
    g0[10, 10] = 3
    h0 = compute_arena_hash(g0, proc.hud_masker)
    ledger.record_transition(h0, 1, h0, delta_omega=0, level_idx=0)
    assert ledger.is_dead(h0, 1, 0, 0, level_idx=0)
    assert not ledger.is_dead(h0, 2, 0, 0, level_idx=0)
    assert ledger.govern_action(h0, 1, 0, 0, level_idx=0) is False
    assert ledger.govern_action(h0, 2, 0, 0, level_idx=0) is True
    obj_sig = (3, 4, 2, 2)
    ledger.record_transition(
        h0, 6, h0, delta_omega=0, coord_x=10, coord_y=10, level_idx=0, clicked_obj_sig=obj_sig
    )
    assert ledger.is_dead(h0, 6, 25, 25, level_idx=0, obj_sig=obj_sig)

def test_epistemic_hypothesis_and_orthogonal_completion() -> None:
    store = HypothesisStore()
    e_prev = Entity(id=0, color=2, pixels=[(10, 10)], mass=1, centroid=(10.0, 10.0), bbox=(10, 10, 10, 10))
    e_curr = Entity(id=0, color=2, pixels=[(9, 10)], mass=1, centroid=(9.0, 10.0), bbox=(9, 10, 9, 10))
    store.observe_entity_motions(1, [(e_prev, e_curr, (-1.0, 0.0))], enable_orthogonal=True)
    eff = store.get_effective_vectors(enable_orthogonal=True)
    assert eff.get(1) == (-1, 0)
    assert eff.get(2) == (1, 0)
    store.persistent_notes["key_color"] = "yellow_4"
    store.add_reasoning_trace(step_idx=3, level_idx=0, thought="Action 1 moves avatar up by 1px")
    store.reset_for_new_level(cleared_previous=True, prev_level=0)
    mem_text = store.format_epistemic_memory()
    assert "A1:(-1, 0)" in mem_text
    assert "key_color=yellow_4" in mem_text
    assert "Cleared Level 0" in mem_text
    sig = (4, 9, 3, 3)
    p_untested = store.bayesian_tracker.exploration_priority(sig, 6)
    store.bayesian_tracker.record_effect(sig, 6, "NO_OP")
    p_noop = store.bayesian_tracker.exploration_priority(sig, 6)
    assert p_untested > p_noop

class TestMemory(unittest.TestCase):
    def test_memory_suite(self) -> None:
        test_arena_hash_and_dead_signature_pruning()
        test_epistemic_hypothesis_and_orthogonal_completion()
