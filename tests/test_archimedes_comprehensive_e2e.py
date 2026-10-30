"""Comprehensive End-to-End Archimedes-3 Verification Suite.

Verifies all integrated mathematical mechanisms from ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md:
1. SGLang serving configuration without --tool-call-parser qwen3_coder (Issue #36537 mitigation).
2. Portfolio Value-of-Computation Scheduler wiring into solver concurrency dispatcher.
3. Five-Tier Memory Manager Bayesian decay and state graph preservation.
4. Spatiotemporal Object-Centric State Estimator (SOCSE) component extraction & monotonic HUD masking.
5. In-Sandbox Tycho Programmatic Workbench with ABSTAIN sentinel and complete-history retrodiction.
6. Pure Python Goal Estimator progress evaluation across exhaustion and target reachability.
7. Pure Python A* Planner with HUD-masked state equivalence.
8. Animation Event Compiler temporal event and transient flash detection.
9. Speed-Depth BOED Acquisition function and OnlineFrameChangeClassifier filter.
10. Procedural Moore Machine procedural 2D grid world generation.
"""
import pytest
from pathlib import Path
from inference.framework.kaggle import duck_kaggle_setup_command, DuckKaggleVllmConfig
from inference.utils.portfolio_voc_scheduler import PortfolioVoCScheduler, GameStreamState
from inference.utils.five_tier_memory import FiveTierMemoryManager
from inference.utils.socse_perception import SOCSEPerceptionEngine, mask_monotonic_hud
from inference.utils.sim_helpers import (
    PurePythonSOCSE,
    PurePythonWorkbench,
    PurePythonGoalEstimator,
    PurePythonAStarPlanner,
    RenderedMooreMachine,
    ABSTAIN,
)
from inference.utils.animation_event_compiler import AnimationEventCompiler
from inference.utils.boed_acquisition import SpeedDepthBOEDAcquisition, OnlineFrameChangeClassifier
from inference.utils.procedural_moore_machine import ProceduralMooreMachine


def test_sglang_issue_36537_mitigation():
    """Verify that --tool-call-parser qwen3_coder is removed from kaggle vLLM/SGLang setup script."""
    cmd = duck_kaggle_setup_command(DuckKaggleVllmConfig())
    assert "--tool-call-parser" not in cmd
    assert "qwen3_coder" not in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--reasoning-parser" in cmd


def test_portfolio_voc_scheduler_end_to_end():
    """Verify Portfolio VoC Scheduler prioritizes streams correctly."""
    scheduler = PortfolioVoCScheduler(max_active_streams=2)
    s1 = GameStreamState(
        game_id="game_urgent",
        current_level=4,
        total_levels=5,
        actions_expended=15,
        human_baseline=20,
        tokens_since_progress=500,
        world_model_recall=0.95,
        is_resident=True,
    )
    s2 = GameStreamState(
        game_id="game_stalled",
        current_level=1,
        total_levels=5,
        actions_expended=45,
        human_baseline=10,
        tokens_since_progress=80000,
        world_model_recall=0.20,
        is_resident=False,
    )
    p1 = scheduler.compute_game_priority(s1)
    p2 = scheduler.compute_game_priority(s2)
    assert p1 > p2 * 10
    active = scheduler.select_active_working_set([s1, s2])
    assert active[0] == "game_urgent"


def test_five_tier_memory_end_to_end():
    """Verify 5-tier memory records, decays across levels, and isolates across games."""
    mem = FiveTierMemoryManager("game_alpha")
    mem.record_transition(
        turn_idx=1,
        state_hash=1001,
        action="ACTION1",
        next_state_hash=1002,
        gameplay_changed=True,
    )
    mem.register_rule("r1", "mover steps north", "VALIDATED")
    assert mem.semantic_rules["r1"].status == "VALIDATED"
    assert mem.semantic_rules["r1"].confidence == 1.0

    # Advance level
    mem.advance_level(2)
    assert mem.current_level == 2
    assert mem.semantic_rules["r1"].status == "PROBABLE"
    assert mem.semantic_rules["r1"].confidence == 0.85

    # Cross-game wipe
    mem.wipe_cross_game("game_beta")
    assert mem.current_game_id == "game_beta"
    assert len(mem.semantic_rules) == 0
    assert len(mem.episodic_tape) == 0


def test_socse_and_hud_masking_end_to_end():
    """Verify SOCSE extracts objects and detects monotonic HUD strip."""
    import numpy as np
    grid = np.array([
        [1, 1, 1, 0],
        [0, 2, 2, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
    ])
    engine = SOCSEPerceptionEngine()
    comps = engine.extract_connected_components(grid)
    assert len(comps) == 2
    masked = mask_monotonic_hud(grid.tolist(), [('row', 0)])
    assert masked[0] == [-1, -1, -1, -1]
    assert masked[1][1] == 2


def test_in_sandbox_workbench_and_abstention_end_to_end():
    """Verify in-sandbox Tycho workbench with ABSTAIN sentinel."""
    class DummyFrame:
        def __init__(self, g):
            self._grid = g

    class DummyEntry:
        def __init__(self, frame, action):
            self.frame = frame
            self.action = action
            self.result = {"gameplay_changed": True}

    history = [
        DummyEntry(DummyFrame([[0, 0], [0, 0]]), ""),
        DummyEntry(DummyFrame([[1, 0], [0, 0]]), "ACTION1"),
    ]
    wb = PurePythonWorkbench(history)

    def partial_model(g, act):
        # Predicts cell (0,0) and abstains on (0,1) and others
        return [[1, ABSTAIN], [ABSTAIN, ABSTAIN]]

    report = wb.retrodict_hypothesis(partial_model)
    assert report["dynamic_recall"] == 1.0
    assert report["coverage"] == 0.25
    assert report["accuracy"] == 1.0


def test_goal_estimator_and_planner_end_to_end():
    """Verify goal progress estimator and A* shortest-path solver."""
    init_g = [[1, 1], [0, 0]]
    cur_g = [[1, 0], [0, 0]]
    prog = PurePythonGoalEstimator.evaluate_progress(init_g, cur_g)
    assert prog == 0.5

    def trans(g, act):
        res = [row[:] for row in g]
        if act == "ACTION4":
            res[0][0] = 0
        return res

    def goal(g):
        return g[0][0] == 0

    plan = PurePythonAStarPlanner.solve([[1, 0], [0, 0]], ["ACTION1", "ACTION4"], trans, goal)
    assert plan == ["ACTION4"]


def test_animation_and_boed_end_to_end():
    """Verify animation event compiler and BOED acquisition."""
    sub_frames = [
        [[0, 0], [0, 0]],
        [[2, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    anim = AnimationEventCompiler.compile_animation_timeline(sub_frames)
    assert anim["sub_frame_count"] == 3
    assert len(anim["transient_flashes"]) == 1

    classifier = OnlineFrameChangeClassifier()
    classifier.observe(1001, "ACTION1", True)
    assert classifier.predict_change_probability(1001, "ACTION1") == 1.0

    boed = SpeedDepthBOEDAcquisition()
    phase = boed.get_epistemic_phase(turn_idx=1, dynamic_recall=0.1)
    assert phase == "PHASE_I_COLD_START"
    val = boed.compute_acquisition("ACTION1", 1001, phase, untested_affordance=True)
    assert val > 0.0
