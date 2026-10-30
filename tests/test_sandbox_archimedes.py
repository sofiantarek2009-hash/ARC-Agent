"""Tests verifying in-sandbox Archimedes-3 neurosymbolic tools: socse, workbench, planner."""
import pytest
from inference.utils.sim_helpers import PurePythonSOCSE, PurePythonWorkbench, PurePythonAStarPlanner


def test_pure_python_socse():
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    bg = PurePythonSOCSE.infer_background_color(grid)
    assert bg == 0

    components = PurePythonSOCSE.get_connected_components(grid)
    assert len(components) == 1
    assert components[0]["color"] == 1
    assert components[0]["size"] == 3
    assert components[0]["bbox"] == (1, 2, 1, 2)


def test_pure_python_workbench_retrodiction():
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
    g2 = [[1, 1], [0, 0]]
    history = [
        DummyEntry(DummyFrame(g0), ""),
        DummyEntry(DummyFrame(g1), "ACTION1"),
        DummyEntry(DummyFrame(g2), "ACTION4"),
    ]

    wb = PurePythonWorkbench(history)
    assert len(wb.transitions) == 2

    # Correct transition function
    def correct_model(grid, act):
        res = [row[:] for row in grid]
        if act == "ACTION1":
            res[0][0] = 1
        elif act == "ACTION4":
            res[0][1] = 1
        return res

    res = wb.retrodict_hypothesis(correct_model)
    assert res["accuracy"] == 1.0
    assert res["dynamic_recall"] == 1.0
    assert res["first_failure"] is None

    # Broken transition function
    def broken_model(grid, act):
        return [row[:] for row in grid]

    res_broken = wb.retrodict_hypothesis(broken_model)
    assert res_broken["dynamic_recall"] == 0.0
    assert res_broken["first_failure"] is not None


def test_pure_python_planner():
    initial_grid = [
        [0, 0, 0],
        [0, 0, 0],
    ]
    valid_actions = ["ACTION1", "ACTION2", "ACTION4"]

    # Action 4 shifts right: (0,0) -> (0,1) -> (0,2)
    def trans(grid, act):
        res = [row[:] for row in grid]
        if act == "ACTION4":
            for c in range(2, 0, -1):
                res[0][c] = res[0][c - 1]
            res[0][0] = 0
            if grid[0][0] == 0:
                res[0][0] = 1
        return res

    # Goal: reaching grid[0][2] == 1
    def goal(grid):
        return grid[0][2] == 1

    plan = PurePythonAStarPlanner.solve(initial_grid, valid_actions, trans, goal)
    assert plan is not None
    assert len(plan) == 3
    assert plan == ["ACTION4", "ACTION4", "ACTION4"]


def test_pure_python_goal_estimator():
    from inference.utils.sim_helpers import PurePythonGoalEstimator

    init_grid = [
        [1, 1, 0],
        [1, 0, 0],
    ]
    cur_grid_progress = [
        [1, 0, 0],
        [0, 0, 0],
    ]
    progress = PurePythonGoalEstimator.evaluate_progress(init_grid, cur_grid_progress)
    assert progress > 0.5  # 2 of 3 tokens cleared

    target_cells = [(0, 2)]
    cur_grid_align = [
        [0, 1, 0],
        [0, 0, 0],
    ]
    progress_align = PurePythonGoalEstimator.evaluate_progress(init_grid, cur_grid_align, target_cells=target_cells)
    assert progress_align > 0.0
