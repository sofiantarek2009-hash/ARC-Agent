"""Tests for FiveTierMemoryManager."""
import pytest
from inference.utils.five_tier_memory import FiveTierMemoryManager


def test_five_tier_memory_transitions_and_ledgers():
    mem = FiveTierMemoryManager(current_game_id="game_alpha")
    s0 = 100
    s1 = 101

    # Normal state change
    mem.record_transition(turn_idx=1, state_hash=s0, action="ACTION1", next_state_hash=s1, gameplay_changed=True)
    assert not mem.is_known_noop(s0, "ACTION1")
    assert not mem.is_known_fatal(s0, "ACTION1")
    assert len(mem.episodic_tape) == 1
    assert s0 in mem.visited_state_graph
    assert s1 in mem.visited_state_graph

    # No-op action
    mem.record_transition(turn_idx=2, state_hash=s1, action="ACTION2", next_state_hash=s1, gameplay_changed=False)
    assert mem.is_known_noop(s1, "ACTION2")

    # Fatal action
    mem.record_transition(turn_idx=3, state_hash=s1, action="ACTION3", next_state_hash=0, gameplay_changed=True, is_death=True)
    assert mem.is_known_fatal(s1, "ACTION3")


def test_cross_level_transfer_decay():
    mem = FiveTierMemoryManager(current_game_id="game_alpha")
    mem.register_rule("rule_1", "Pushing crate moves it along action direction", status="VALIDATED")
    mem.record_transition(1, 100, "ACTION1", 100, gameplay_changed=False)
    mem.record_transition(2, 100, "ACTION2", 0, gameplay_changed=True, is_death=True)

    assert mem.semantic_rules["rule_1"].status == "VALIDATED"
    assert mem.semantic_rules["rule_1"].confidence == 1.0
    assert mem.is_known_noop(100, "ACTION1")
    assert mem.is_known_fatal(100, "ACTION2")

    # Advance to Level 2
    mem.advance_level(new_level=2)
    assert mem.current_level == 2
    # Rule decays to PROBABLE with 0.85 confidence
    assert mem.semantic_rules["rule_1"].status == "PROBABLE"
    assert mem.semantic_rules["rule_1"].confidence == 0.85
    # NoOpLedger is reset per level
    assert not mem.is_known_noop(100, "ACTION1")
    # FatalLedger is preserved
    assert mem.is_known_fatal(100, "ACTION2")


def test_wipe_cross_game():
    mem = FiveTierMemoryManager(current_game_id="game_alpha")
    mem.register_rule("rule_1", "Rule A", status="VALIDATED")
    mem.record_transition(1, 100, "ACTION1", 101, gameplay_changed=True)
    mem.record_transition(2, 101, "ACTION2", 0, gameplay_changed=True, is_death=True)

    # Wipe for new game
    mem.wipe_cross_game(new_game_id="game_beta")
    assert mem.current_game_id == "game_beta"
    assert mem.current_level == 1
    assert len(mem.episodic_tape) == 0
    assert len(mem.visited_state_graph) == 0
    assert len(mem.semantic_rules) == 0
    assert not mem.is_known_fatal(101, "ACTION2")


def test_render_ledger_prompt():
    mem = FiveTierMemoryManager(current_game_id="game_alpha")
    mem.register_rule("rule_1", "Moving right moves avatar", status="VALIDATED")
    mem.record_transition(1, 100, "ACTION1", 100, gameplay_changed=False)
    mem.record_transition(2, 100, "ACTION2", 0, gameplay_changed=True, is_death=True)

    rendered = mem.render_ledger_prompt()
    assert "[FIVE-TIER MEMORY HIERARCHY]" in rendered
    assert "Level: 1" in rendered
    assert "rule_1: Moving right moves avatar" in rendered
    assert "Fatal Ledger: 1 terminal actions" in rendered
    assert "NoOp Ledger: 1 no-ops" in rendered

