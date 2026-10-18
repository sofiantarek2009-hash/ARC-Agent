"""Unit tests for ExactCoverDLX, SplitPanelD4Solver, and EMPAObjectCuriosityPlanner."""

import unittest
import numpy as np

from project_override.core.empirical_planner import (
    ExactCoverDLX,
    SplitPanelD4Solver,
    EMPAObjectCuriosityPlanner,
)


class TestEmpiricalPlanner(unittest.TestCase):

    def test_exact_cover_dlx(self):
        # Classic exact cover problem:
        # Rows:
        # 0: {1, 4, 7}
        # 1: {1, 4}
        # 2: {4, 5, 7}
        # 3: {3, 5, 6}
        # 4: {2, 3, 6, 7}
        # 5: {2, 7}
        # Desired columns: {1, 2, 3, 4, 5, 6, 7}
        # Solution: rows 1 {1, 4}, 3 {3, 5, 6}, 5 {2, 7} covers all exactly once.
        matrix = {
            0: {1, 4, 7},
            1: {1, 4},
            2: {4, 5, 7},
            3: {3, 5, 6},
            4: {2, 3, 6, 7},
            5: {2, 7},
        }
        solution = ExactCoverDLX.solve(matrix, primary_columns={1, 2, 3, 4, 5, 6, 7})
        self.assertIsNotNone(solution)
        self.assertEqual(sorted(solution), [1, 3, 5])

        # Test empty / impossible cover
        impossible = {0: {1, 2}, 1: {2, 3}}
        self.assertIsNone(ExactCoverDLX.solve(impossible, primary_columns={1, 2, 3, 4}))

    def test_split_panel_d4_solver(self):
        # 64x64 grid with a vertical divider line at col 32 of color 4 (frame/divider)
        grid = np.zeros((64, 64), dtype=np.int32)
        grid[4:60, 32] = 4  # Divider

        # Left panel (reference): pixel at (10, 15) is color 2
        grid[10, 15] = 2

        # Right panel (interactive): mirror of col 15 is 32 + (32 - 15) = 49
        # But (10, 49) is missing (still 0), so mismatch should be detected!
        match = SplitPanelD4Solver.analyze_symmetry(grid, border_margin=4)
        self.assertIsNotNone(match)
        self.assertEqual(match.split_type, "VERTICAL")
        self.assertEqual(match.divider_coord, 32)
        self.assertIn(match.symmetry_op, ["IDENTITY", "FLIP_H", "FLIP_V"])

    def test_empa_curiosity_planner(self):
        planner = EMPAObjectCuriosityPlanner(border_margin=4)
        grid = np.zeros((64, 64), dtype=np.int32)
        # Background = 0, Walls = 5
        grid[4:60, 4] = 5
        # Avatar at (10, 10) color 1
        grid[10, 10] = 1
        # Object A (color 2) at (12, 10)
        grid[12, 10] = 2
        # Object B (color 3) at (30, 30)
        grid[30, 30] = 3

        targets = planner.extract_candidate_targets(
            grid=grid,
            avatar_pos=(10, 10),
            avatar_color=1,
            bg_color=0,
            wall_colors={5},
        )
        self.assertEqual(len(targets), 2)
        # Target A should be first because it is closest and untested
        self.assertEqual(targets[0].color, 2)
        self.assertEqual(targets[1].color, 3)

        # Record contact with Object A
        planner.record_contact(color=2, coord=(12, 10), delta_observed=1)
        self.assertIn(2, planner.tested_color_classes)
        self.assertEqual(planner.affordances[2], "COLLECTIBLE_OR_SWITCH")


if __name__ == "__main__":
    unittest.main()
