"""Unit tests for Archimedes-3 Abductive Fault Localizer (§16.1 & §18.1)."""
from __future__ import annotations

import pytest
from inference.utils.abductive_fault_localizer import AbductiveFaultLocalizer, FaultDiagnosis
from inference.utils.avo_and_commit import KeplerCommitGate
from inference.utils.sim_helpers import PurePythonFaultLocalizer, PurePythonCommitGate
from inference.utils.procedural_moore_machine import NVARC3SyntheticGenerator, ProceduralMooreMachine


def test_type_a_unexpected_noop():
    localizer = AbductiveFaultLocalizer()
    before = [[0, 4, 1], [0, 0, 0]]
    actual = [[0, 4, 1], [0, 0, 0]]  # zero change
    predicted = [[0, 0, 4], [0, 0, 0]]  # predicted player moved right

    diag = localizer.classify_surprise(
        before_grid=before,
        action="ACTION4",
        actual_grid=actual,
        predicted_grid=predicted,
        actual_status="PLAYABLE",
        predicted_status="PLAYABLE",
    )
    assert diag.error_type == "TYPE_A"
    assert diag.root_cause == "UNEXPECTED_NO_OP"
    assert "0 interior grid mutations" in diag.diagnostic_message


def test_type_b_localized_kinematics_error():
    localizer = AbductiveFaultLocalizer()
    before = [
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    # Prediction: moved 1 cell right to (0, 2)
    predicted = [
        [0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    # Actual: slid 3 cells right to (0, 4)
    actual = [
        [0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0],
    ]

    diag = localizer.classify_surprise(
        before_grid=before,
        action="ACTION4",
        actual_grid=actual,
        predicted_grid=predicted,
        actual_status="PLAYABLE",
        predicted_status="PLAYABLE",
    )
    assert diag.error_type == "TYPE_B"
    assert diag.root_cause == "LOCALIZED_KINEMATICS_ERROR"


def test_type_c_non_local_contact_effect():
    localizer = AbductiveFaultLocalizer()
    h, w = 10, 10
    before = [[0] * w for _ in range(h)]
    before[1][1] = 4  # player
    before[8][8] = 6  # closed door far away

    predicted = [[0] * w for _ in range(h)]
    predicted[1][2] = 4  # player moved to switch
    predicted[8][8] = 6  # model thought door stayed closed

    actual = [[0] * w for _ in range(h)]
    actual[1][2] = 4  # player moved
    actual[8][8] = 0  # door opened at opposite end!

    diag = localizer.classify_surprise(
        before_grid=before,
        action="ACTION4",
        actual_grid=actual,
        predicted_grid=predicted,
        actual_status="PLAYABLE",
        predicted_status="PLAYABLE",
    )
    assert diag.error_type == "TYPE_C"
    assert diag.root_cause == "NON_LOCAL_CONTACT_EFFECT"


def test_type_d_latent_state_aliasing():
    localizer = AbductiveFaultLocalizer()
    before = [[0, 4, 0], [0, 0, 0]]
    actual = [[0, 0, 4], [0, 0, 0]]

    # History shows identical before frame and action previously produced [[4, 0, 0], [0, 0, 0]]
    past_history = [
        {
            "turn_idx": 2,
            "before_grid": [[0, 4, 0], [0, 0, 0]],
            "action": "ACTION4",
            "after_grid": [[4, 0, 0], [0, 0, 0]],  # different outcome!
        }
    ]

    diag = localizer.classify_surprise(
        before_grid=before,
        action="ACTION4",
        actual_grid=actual,
        history_transitions=past_history,
    )
    assert diag.error_type == "TYPE_D"
    assert diag.root_cause == "LATENT_STATE_ALIASING"


def test_type_e_fatal_collision_and_budget_timeout():
    localizer = AbductiveFaultLocalizer()
    before = [[0, 4, 1], [0, 0, 0]]
    # Collision: interior cells mutated
    actual_collision = [[0, 0, 9], [0, 0, 0]]

    diag_col = localizer.classify_surprise(
        before_grid=before,
        action="ACTION4",
        actual_grid=actual_collision,
        actual_status="DEATH",
        predicted_status="PLAYABLE",
    )
    assert diag_col.error_type == "TYPE_E"
    assert diag_col.root_cause == "LETHAL_HAZARD_COLLISION"

    # Timeout: no interior changes, but status is DEATH
    actual_timeout = [[0, 4, 1], [0, 0, 0]]
    diag_time = localizer.classify_surprise(
        before_grid=before,
        action="ACTION4",
        actual_grid=actual_timeout,
        actual_status="DEATH",
        predicted_status="PLAYABLE",
    )
    assert diag_time.error_type == "TYPE_E"
    assert diag_time.root_cause == "BUDGET_EXHAUSTION"


def test_kepler_commit_gate_abort_and_diagnose():
    gate = KeplerCommitGate(abstain_val=-1)
    plan = ["ACTION4", "ACTION4", "ACTION2"]
    pred_frames = [
        [[0, 0, 4], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 4]],
        [[0, 0, 0], [0, 0, 0]],
    ]
    gate.commit_plan(plan, expected_frames=pred_frames)

    # Step 0: Actual diverges from prediction
    before = [[0, 4, 0], [0, 0, 0]]
    actual = [[0, 4, 0], [0, 0, 0]]  # no-op!
    surprise, diag = gate.evaluate_step(before, "ACTION4", actual, step_idx=0)

    assert surprise is True
    assert diag is not None
    assert diag.error_type == "TYPE_A"
    # Verify plan was aborted immediately
    assert len(gate.active_plan) == 0


def test_pure_python_fault_localizer_and_gate():
    localizer = PurePythonFaultLocalizer()
    before = [[0, 4, 1], [0, 0, 0]]
    actual = [[0, 4, 1], [0, 0, 0]]
    pred = [[0, 0, 4], [0, 0, 0]]
    diag = localizer.classify_surprise(before, "ACTION4", actual, predicted_grid=pred)
    assert diag["error_type"] == "TYPE_A"

    gate = PurePythonCommitGate()
    gate.commit_plan(["ACTION4", "ACTION2"], expected_frames=[pred, [[0, 0, 0], [0, 0, 4]]])
    surprise, d = gate.evaluate_step(before, "ACTION4", actual, step_idx=0)
    assert surprise is True
    assert d["error_type"] == "TYPE_A"
    assert len(gate.active_plan) == 0


def test_nvarc3_procedural_generator():
    task = NVARC3SyntheticGenerator.generate_task(seed=42)
    assert task is not None
    assert 5 <= task.optimal_path_length <= 40
    assert len(task.initial_grid) == 10
    assert len(task.optimal_path) == task.optimal_path_length
    assert len(task.exploratory_trace) > 0
    assert "class SyntheticGroundTruthWorldModel" in task.ground_truth_world_model_code
