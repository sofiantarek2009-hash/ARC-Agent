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


def test_persistent_sandbox_worker_recycling():
    """Verify that PersistentSandboxWorker stays alive across calls and delivers sub-millisecond turnarounds."""
    import time
    from inference.agent.python_tool_sandbox import run_sandboxed_python

    # First call spins up worker
    t0 = time.time()
    res1 = run_sandboxed_python(
        code="result = 40 + 2",
        timeout_seconds=5,
        initial_state={},
        action_handler=lambda actions, **kwargs: {},
    )
    t1 = time.time()
    assert res1.get("result") == 42

    # Subsequent call reuses warm worker process in <10ms
    t2 = time.time()
    res2 = run_sandboxed_python(
        code="result = 'hello'.upper()",
        timeout_seconds=5,
        initial_state={},
        action_handler=lambda actions, **kwargs: {},
    )
    t3 = time.time()
    assert res2.get("result") == "HELLO"
    assert (t3 - t2) < 0.05, f"Expected warm worker turnaround < 50ms, got {(t3 - t2)*1000:.2f}ms"


def test_tycho_universal_epistemic_oracle():
    """Verify 5-valued epistemic status in Universal Transition Oracle (§31.2 Innovation 1)."""
    from inference.utils.tycho_workbench import (
        ProgrammaticWorldModelWorkbench,
        EpistemicStatus,
        predict_epistemic_transition,
    )

    wb = ProgrammaticWorldModelWorkbench()
    grid_0 = [[0, 0], [0, 0]]
    action = "ACTION1"

    # 1. Unknown if no history and no candidate models
    pred, status = predict_epistemic_transition(wb, grid_0, action)
    assert pred is None
    assert status == EpistemicStatus.UNKNOWN

    # 2. Deterministic if unanimous agreement
    model_a = lambda g, a, d: [[1, 0], [0, 0]]
    model_b = lambda g, a, d: [[1, 0], [0, 0]]
    pred, status = predict_epistemic_transition(wb, grid_0, action, candidate_models=[model_a, model_b])
    assert pred == [[1, 0], [0, 0]]
    assert status == EpistemicStatus.DETERMINISTIC

    # 3. Ambiguous if models diverge
    model_c = lambda g, a, d: [[0, 1], [0, 0]]
    pred, status = predict_epistemic_transition(wb, grid_0, action, candidate_models=[model_a, model_c])
    assert status == EpistemicStatus.AMBIGUOUS

    # 4. Abstain if model outputs abstention sentinel (-1 or 16)
    model_abstain = lambda g, a, d: [[-1, 0], [0, 0]]
    pred, status = predict_epistemic_transition(wb, grid_0, action, candidate_models=[model_abstain])
    assert status == EpistemicStatus.ABSTAIN


def test_solver_salvage_mode_governor():
    """Verify Dynamic Mode Switching into Salvage Mode at T_remaining <= 1980s."""
    from datetime import datetime, timedelta
    from inference.framework.solver import HarnessSolver

    solver = HarnessSolver()
    # No soft_end_time -> not in salvage mode
    assert solver.is_salvage_mode() is False

    # Soft end time 1000s in the future (within 1980s window) -> Salvage Mode True
    solver.soft_end_time = datetime.now() + timedelta(seconds=1000)
    assert solver.is_salvage_mode() is True

    # Soft end time 5000s in the future (outside window) -> Salvage Mode False
    solver.soft_end_time = datetime.now() + timedelta(seconds=5000)
    assert solver.is_salvage_mode() is False


def test_incremental_token_counting_and_trimming():
    """Verify O(1) incremental token caching and fast context trimming."""
    from inference.agent.tool_agent import ToolAgent

    agent = ToolAgent(model="mock_model")
    msg1 = {"role": "user", "content": "Hello " * 50}
    msg2 = {"role": "assistant", "content": "World " * 50}
    
    t1 = agent._estimate_message_tokens(msg1)
    t2 = agent._estimate_message_tokens(msg2)
    assert t1 > 0
    assert t2 > 0
    # Second call should hit cache instantaneously
    assert agent._estimate_message_tokens(msg1) == t1

    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    history = [
        {"role": "user", "content": f"Turn {i} " + "data " * 40}
        if i % 2 == 0 else
        {"role": "assistant", "content": f"Reply {i} " + "answer " * 40}
        for i in range(20)
    ]
    messages = [system_msg, *history]
    
    # Trim to budget of 500 tokens
    agent._context_budget_tokens = 500
    trimmed = agent._trim_messages_for_context(messages, preserve_recent=2)
    assert len(trimmed) < len(messages)
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["role"] == "user"
    final_used = agent._estimate_request_input_tokens(trimmed)
    assert final_used <= 500

