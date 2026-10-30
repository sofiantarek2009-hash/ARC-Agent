"""Tests verifying RenderedMooreMachine and abstention retrodiction."""
import pytest
from inference.utils.sim_helpers import (
    RenderedMooreMachine,
    PurePythonWorkbench,
    ABSTAIN,
)


def test_rendered_moore_machine_class_retrodiction():
    class DummyFrame:
        def __init__(self, g):
            self._grid = g

    class DummyEntry:
        def __init__(self, frame, action):
            self.frame = frame
            self.action = action
            self.result = {"gameplay_changed": True}

    g0 = [[0, 0], [0, 0]]
    g1 = [[1, 0], [0, 0]]
    history = [
        DummyEntry(DummyFrame(g0), ""),
        DummyEntry(DummyFrame(g1), "ACTION1"),
    ]

    wb = PurePythonWorkbench(history)

    # Candidate Moore machine that models Action 1 setting (0, 0) to 1
    class CorrectMachine(RenderedMooreMachine):
        def transition(self, state, action, action_data=None):
            nxt = super().transition(state, action, action_data)
            if action == "ACTION1":
                nxt["grid"][0][0] = 1
            return nxt

    res = wb.retrodict_hypothesis(CorrectMachine)
    assert res["accuracy"] == 1.0
    assert res["dynamic_recall"] == 1.0
    assert res["coverage"] == 1.0
    assert res["first_failure"] is None


def test_abstention_partial_render():
    class DummyFrame:
        def __init__(self, g):
            self._grid = g

    class DummyEntry:
        def __init__(self, frame, action):
            self.frame = frame
            self.action = action
            self.result = {"gameplay_changed": True}

    # Turn changes (0, 0), leaves (0, 1) unchanged
    g0 = [[0, 2], [0, 0]]
    g1 = [[1, 2], [0, 0]]
    history = [
        DummyEntry(DummyFrame(g0), ""),
        DummyEntry(DummyFrame(g1), "ACTION1"),
    ]
    wb = PurePythonWorkbench(history)

    # Machine that abstains on (0, 1) because it doesn't know its rule yet
    def partial_model(grid, act):
        return [
            [1, ABSTAIN],
            [0, 0],
        ]

    res = wb.retrodict_hypothesis(partial_model)
    # 3 cells covered out of 4: cell (0, 1) is abstained (-1)
    assert res["accuracy"] == 1.0
    assert res["dynamic_recall"] == 1.0
    assert res["coverage"] == 0.75
