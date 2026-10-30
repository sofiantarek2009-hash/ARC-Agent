"""Unit tests for OnlineFrameChangeClassifier and SpeedDepthBOEDAcquisition."""
import pytest
from inference.utils.boed_acquisition import (
    OnlineFrameChangeClassifier,
    SpeedDepthBOEDAcquisition,
)


def test_frame_change_classifier():
    classifier = OnlineFrameChangeClassifier()
    state_h = 123456789
    action = "ACTION1"

    # Initially neutral prior
    assert classifier.predict_change_probability(state_h, action) == 0.5
    assert not classifier.is_known_noop(state_h, action)

    # Observe no-op
    classifier.observe(state_h, action, gameplay_changed=False)
    assert classifier.is_known_noop(state_h, action)
    assert classifier.predict_change_probability(state_h, action) == 0.0

    # Observe change in different state
    state_h2 = 987654321
    classifier.observe(state_h2, action, gameplay_changed=True)
    assert not classifier.is_known_noop(state_h2, action)
    assert classifier.predict_change_probability(state_h2, action) == 1.0


def test_boed_acquisition():
    boed = SpeedDepthBOEDAcquisition()
    state_h = 12345

    # Phase I (Cold start) prioritizes novel information
    phase = boed.get_epistemic_phase(turn_idx=1, dynamic_recall=0.1)
    assert phase == "PHASE_I_COLD_START"

    score_novel = boed.compute_acquisition("ACTION1", state_h, phase, untested_affordance=True)
    score_tested = boed.compute_acquisition("ACTION1", state_h, phase, untested_affordance=False)
    assert score_novel > score_tested

    # Lethal action hard rejection
    score_deadly = boed.compute_acquisition("ACTION2", state_h, phase, is_known_deadly=True)
    assert score_deadly <= -1000.0

    # Known no-op hard rejection
    boed.frame_change_classifier.observe(state_h, "ACTION3", gameplay_changed=False)
    score_noop = boed.compute_acquisition("ACTION3", state_h, phase)
    assert score_noop <= -500.0

    # Phase III (Exploitation) when recall is high
    phase_iii = boed.get_epistemic_phase(turn_idx=10, dynamic_recall=0.95, hypothesis_entropy=0.05)
    assert phase_iii == "PHASE_III_EXPLOITATION_SHORTEST_PATH"
