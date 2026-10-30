"""Comprehensive test suite for Archimedes-3 core subsystems.

Validates:
1. NVIDIA AVO Supervisory Intervention (trapped in cycle -> directive injected).
2. Kepler Pre-Action Commit Gate (unexpected no-op -> early batch termination).
3. Tycho Complete-History Retrodiction (transition function evaluated against tape).
4. Spatiotemporal Object-Centric State Estimator (SOCSE) component extraction & monotonic HUD detection.
5. Deterministic State-Equivalence A* Planner.
"""
from __future__ import annotations

import numpy as np
import pytest

from inference.utils.socse_perception import SOCSEPerceptionEngine, mask_monotonic_hud
from inference.utils.avo_and_commit import AVOSupervisor, KeplerCommitGate
from inference.utils.tycho_workbench import ProgrammaticWorldModelWorkbench
from inference.utils.symbolic_planner import StateEquivalenceAStarPlanner
from inference.agent.tool_agent import ToolAgent, _interior_state_hash


def test_avo_supervisor_stagnation_injection():
    """Verify that NVIDIA AVO Supervisor detects cyclic loops and emits prompt directive."""
    avo = AVOSupervisor(stagnation_threshold=4)
    # Simulate 4 turns without state progression
    for turn in range(4):
        avo.observe_turn(
            current_state_hash="cyclic_hash_123",
            gameplay_changed=False,
            level=1,
        )

    assert avo.is_stagnating() is True
    directive = avo.get_supervisory_directive()
    assert directive is not None
    assert "SUPERVISORY INTERVENTION - NVIDIA AVO" in directive
    assert "MANDATE" in directive


def test_kepler_commit_gate_surprise_detection():
    """Verify that unexpected no-ops in multi-action batch trigger surprise abort."""
    before = [[0, 1], [0, 0]]
    after = [[0, 1], [0, 0]]  # identical -> 0 change
    surprise = KeplerCommitGate.evaluate_step_surprise(before, after, expected_change=True)
    assert surprise is True


def test_tycho_complete_history_retrodiction():
    """Verify that candidate transition functions are retrodicted against historical tape."""
    wb = ProgrammaticWorldModelWorkbench()
    # Record two turns
    wb.record_transition(
        turn_idx=1,
        before_grid=[[1, 0], [0, 0]],
        action="RIGHT",
        action_data={},
        after_grid=[[0, 1], [0, 0]],
        gameplay_changed=True,
    )
    wb.record_transition(
        turn_idx=2,
        before_grid=[[0, 1], [0, 0]],
        action="DOWN",
        action_data={},
        after_grid=[[0, 0], [0, 1]],
        gameplay_changed=True,
    )

    # Correct hypothesis
    def correct_hyp(grid, action, data):
        if action == "RIGHT" and grid == [[1, 0], [0, 0]]:
            return [[0, 1], [0, 0]]
        if action == "DOWN" and grid == [[0, 1], [0, 0]]:
            return [[0, 0], [0, 1]]
        return grid

    res = wb.retrodict_hypothesis(correct_hyp)
    assert res["accuracy"] == 1.0
    assert res["dynamic_recall"] == 1.0
    assert res["first_failure"] is None

    # Flawed hypothesis
    def flawed_hyp(grid, action, data):
        return grid  # predicts no movement

    res_flawed = wb.retrodict_hypothesis(flawed_hyp)
    assert res_flawed["accuracy"] == 0.0
    assert res_flawed["first_failure"] is not None
    assert res_flawed["first_failure"]["turn"] == 1


def test_socse_and_monotonic_hud_masking():
    """Verify that SOCSE identifies monotonic bars and masks them for canonical hashing."""
    socse = SOCSEPerceptionEngine()
    g1 = np.zeros((64, 64), dtype=int)
    g1[0, :10] = 3  # row 0 has 10 bar pixels
    g2 = np.zeros((64, 64), dtype=int)
    g2[0, :8] = 3   # row 0 shrunk to 8 pixels
    g3 = np.zeros((64, 64), dtype=int)
    g3[0, :6] = 3   # row 0 shrunk to 6 pixels

    monotonic_strips = socse.detect_monotonic_counters([g1, g2, g3])
    assert ('row', 0) in monotonic_strips

    # Masking test
    grid_list = g1.tolist()
    masked = mask_monotonic_hud(grid_list, monotonic_strips)
    assert all(val == -1 for val in masked[0])


def test_state_equivalence_astar_planner():
    """Verify deterministic shortest path search finds minimal action sequence."""
    initial = [[1, 0, 0]]
    valid_actions = ["LEFT", "RIGHT"]

    def transition(grid, action):
        g = [list(r) for r in grid]
        idx = g[0].index(1) if 1 in g[0] else -1
        if action == "RIGHT" and idx < 2:
            g[0][idx] = 0
            g[0][idx + 1] = 1
        elif action == "LEFT" and idx > 0:
            g[0][idx] = 0
            g[0][idx - 1] = 1
        return g

    def goal(grid):
        return grid[0][2] == 1

    path = StateEquivalenceAStarPlanner.solve(
        initial_grid=initial,
        valid_actions=valid_actions,
        transition_fn=transition,
        goal_predicate=goal,
    )
    assert path == ["RIGHT", "RIGHT"]
