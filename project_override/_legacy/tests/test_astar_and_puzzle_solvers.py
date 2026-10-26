"""Unit tests for A* navigation, symbolic puzzle solver, and DPA physics predictor.
Project Override (ARC-AGI-3)
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project_override package is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_override.core.astar_navigation import (
    AStarNavigator,
    CollisionHull,
    NavAction,
)
from project_override.core.symbolic_puzzle_solver import (
    SymbolicPuzzleSolver,
    ClickCandidate,
)
from project_override.core.dpa_physics_predictor import (
    DPAPhysicsPredictor,
    AutomataClass,
)


class TestAStarNavigation(unittest.TestCase):
    """Tests for obstacle-avoiding A* pathfinding and Sokoban push mechanics."""

    def setUp(self):
        self.nav = AStarNavigator(max_dim=32)

    def test_simple_straight_path(self):
        # 10x10 empty grid, all passable
        passable = np.ones((10, 10), dtype=bool)
        res = self.nav.find_path(passable, start=(2, 2), goals=[(2, 6)])
        self.assertTrue(res.success)
        self.assertEqual(len(res.actions), 4)
        # All actions should be RIGHT (4)
        self.assertEqual(res.actions, [NavAction.RIGHT] * 4)

    def test_u_shaped_obstacle_navigation(self):
        # 10x10 grid with U-shaped wall
        passable = np.ones((10, 10), dtype=bool)
        # Wall at row 4, cols 2 to 6
        passable[4, 2:7] = False
        res = self.nav.find_path(passable, start=(2, 4), goals=[(6, 4)])
        self.assertTrue(res.success)
        # Verify path does not step on wall
        for r, c in res.path_coords:
            self.assertTrue(passable[r, c] or (r, c) == (6, 4))

    def test_multi_cell_hull_collision(self):
        # 2x2 player hull
        hull = CollisionHull(offsets=[(0, 0), (0, 1), (1, 0), (1, 1)], height=2, width=2, mass=4)
        passable = np.ones((10, 10), dtype=bool)
        # 1-cell narrow gap at (3, 3) (impassable for 2x2 hull)
        passable[3, :] = False
        passable[3, 3] = True  # only 1 cell wide!
        res = self.nav.find_path(passable, start=(1, 1), goals=[(6, 6)], hull=hull)
        # Cannot squeeze 2x2 hull through 1-cell gap
        self.assertFalse(res.success)

    def test_sokoban_push_planning(self):
        passable = np.ones((10, 10), dtype=bool)
        player_pos = (5, 2)
        box_pos = (5, 5)
        goal_pos = (5, 8)
        # Push box from (5,5) rightward toward (5,8)
        plan = self.nav.plan_sokoban_push(passable, player_pos, box_pos, goal_pos)
        self.assertIsNotNone(plan)
        # Last action should be push RIGHT (4)
        self.assertEqual(plan[-1], NavAction.RIGHT)


class TestSymbolicPuzzleSolver(unittest.TestCase):
    """Tests for ACTION6 coordinate targeting and GF(2) linear toggle solver."""

    def setUp(self):
        self.solver = SymbolicPuzzleSolver()

    def test_click_candidate_extraction(self):
        grid = np.zeros((16, 16), dtype=np.int32)
        # Background is 0
        # Button 1 at (4, 4), color 2
        grid[4, 4] = 2
        # Button 2 at (10, 10), color 3
        grid[10, 10] = 3

        candidates = self.solver.extract_click_candidates(grid)
        self.assertEqual(len(candidates), 2)
        candidate_coords = {(c.y, c.x) for c in candidates}
        self.assertIn((4, 4), candidate_coords)
        self.assertIn((10, 10), candidate_coords)

    def test_gf2_lights_out_solver(self):
        # 3x3 Lights Out system
        # Toggle matrix for 3 cells where clicking cell flips self and adjacent
        A = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=np.int8)
        # Target state: all 1s
        b = np.array([1, 1, 1], dtype=np.int8)
        x = self.solver.solve_gf2_linear_system(A, b)
        self.assertIsNotNone(x)
        # Verify A * x == b (mod 2)
        result = (A @ x) % 2
        np.testing.assert_array_equal(result, b)


class TestDPAPhysicsPredictor(unittest.TestCase):
    """Tests for discrete cellular automata bouncer and intercept geometry."""

    def test_bouncer_trajectory_reflection(self):
        # Moving right from col 8 in 10-wide grid
        traj = DPAPhysicsPredictor.predict_bouncer_trajectory(
            start_pos=(2, 8),
            velocity=(0, 1),
            steps=4,
            grid_shape=(10, 10),
        )
        self.assertEqual(len(traj), 5)
        # After hitting wall col 9, should reflect back left
        self.assertEqual(traj[0], (2, 8))
        self.assertEqual(traj[1], (2, 9))
        self.assertEqual(traj[2], (2, 8))  # reflected!
        self.assertEqual(traj[3], (2, 7))

    def test_deterministic_intercept(self):
        avatar = (0, 0)
        target_pos = (5, 5)
        target_vel = (0, 0)  # stationary
        sol = DPAPhysicsPredictor.compute_deterministic_intercept(
            avatar_pos=avatar,
            avatar_speed=1.0,
            target_start_pos=target_pos,
            target_velocity=target_vel,
            max_horizon=20,
        )
        self.assertTrue(sol.is_feasible)
        self.assertEqual(sol.target_pos, (5, 5))
        self.assertEqual(sol.intercept_time, 10)  # 5 down + 5 right = 10 steps


if __name__ == "__main__":
    unittest.main()
