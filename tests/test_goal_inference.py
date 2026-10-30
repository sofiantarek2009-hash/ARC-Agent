"""Tests for GoalPosteriorEstimator."""
import pytest
from inference.utils.goal_inference import GoalPosteriorEstimator


def test_goal_posterior_initial_priors():
    est = GoalPosteriorEstimator()
    assert "G_exhaust" in est.candidate_hypotheses
    assert "G_align" in est.candidate_hypotheses
    assert sum(est.candidate_hypotheses.values()) == pytest.approx(1.0)


def test_goal_progress_exhaustion():
    est = GoalPosteriorEstimator()
    initial_grid = [
        [1, 1, 1],
        [0, 0, 0],
    ]
    # 2 tokens remaining out of 3
    current_grid = [
        [1, 1, 0],
        [0, 0, 0],
    ]
    progress = est.evaluate_state_goal_progress(initial_grid, current_grid)
    assert progress > 0.0


def test_observe_terminal_win():
    est = GoalPosteriorEstimator()
    g_before = [[1, 0], [0, 0]]
    g_win = [[0, 0], [0, 0]]
    est.observe_terminal_win(g_before, g_win)
    assert len(est.confirmed_win_predicates) == 1
