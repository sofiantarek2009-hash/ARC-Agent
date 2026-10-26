"""Unit Tests for Combinatorial Engine, Passive Conveyor Engine, and Online World Model Synthesizer."""

import itertools
import unittest
import numpy as np

from project_override.core.combinatorial_engine import (
    UniversalGF2LinearSolver,
    UniversalBooleanSATSolver,
    UniversalIndependentSetSolver,
    InteractiveMatrixInducer,
)
from project_override.core.passive_conveyor_engine import (
    PassiveVelocityField,
    solve_conveyor_astar,
)
from project_override.core.online_world_model_synthesizer import (
    OnlineWorldModelSynthesizer,
    MultiStrideKinematicsSimulator,
)


class TestCombinatorialEngine(unittest.TestCase):
    def test_gf2_square_system(self):
        # A = [[1, 1], [0, 1]], b = [1, 0]
        # x_1 + x_2 = 1, x_2 = 0 => x_1 = 1, x_2 = 0
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        sol = UniversalGF2LinearSolver.solve(A, b)
        self.assertIsNotNone(sol)
        self.assertTrue(np.array_equal(sol, [1, 0]))

    def test_gf2_min_weight_free_variables(self):
        # Underdetermined: x_1 + x_2 + x_3 = 1
        # Solutions: [1, 0, 0] (weight 1), [0, 1, 0] (weight 1), [1, 1, 1] (weight 3)
        A = np.array([[1, 1, 1]], dtype=np.uint8)
        b = np.array([1], dtype=np.uint8)
        sol = UniversalGF2LinearSolver.solve(A, b, find_min_weight=True)
        self.assertIsNotNone(sol)
        self.assertEqual(int(np.sum(sol)), 1)
        self.assertEqual(int((A @ sol % 2)[0]), 1)

    def test_gf2_inconsistent_system(self):
        # x_1 = 0, x_1 = 1 (contradiction)
        A = np.array([[1], [1]], dtype=np.uint8)
        b = np.array([0, 1], dtype=np.uint8)
        sol = UniversalGF2LinearSolver.solve(A, b)
        self.assertIsNone(sol)

    def test_boolean_2sat_satisfiable(self):
        # (x_1 or x_2) and (~x_1 or x_2)
        clauses = [[1, 2], [-1, 2]]
        sol = UniversalBooleanSATSolver.solve_2sat(2, clauses)
        self.assertIsNotNone(sol)
        self.assertTrue(sol[2])  # x_2 must be True

    def test_boolean_2sat_unsatisfiable(self):
        # (x_1) and (~x_1)
        clauses = [[1], [-1]]
        sol = UniversalBooleanSATSolver.solve_2sat(1, clauses)
        self.assertIsNone(sol)

    def test_boolean_dpll_3sat(self):
        # (x_1 or x_2 or x_3) and (~x_1 or ~x_2) and (~x_3)
        clauses = [[1, 2, 3], [-1, -2], [-3]]
        sol = UniversalBooleanSATSolver.solve_dpll(3, clauses)
        self.assertIsNotNone(sol)
        self.assertFalse(sol[3])  # x_3 must be False
        # One of x_1 or x_2 must be True, but not both
        self.assertTrue(sol[1] != sol[2])

    def test_independent_set_triangle_and_edge(self):
        # Graph: triangle (0, 1, 2) + isolated node 3
        # Max independent set is {3, one of (0, 1, 2)} -> size 2
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 0)]
        ind_set = UniversalIndependentSetSolver.solve(nodes, edges)
        self.assertEqual(len(ind_set), 2)
        self.assertIn(3, ind_set)
        other = [n for n in ind_set if n != 3][0]
        self.assertIn(other, [0, 1, 2])

        # Minimum vertex cover is the complement
        vert_cover = UniversalIndependentSetSolver.solve_vertex_cover(nodes, edges)
        self.assertEqual(len(vert_cover), 2)
        for u, v in edges:
            self.assertTrue(u in vert_cover or v in vert_cover)

    def test_extract_graph_from_grid(self):
        # Create a 64x64 grid with 3 vertices connected by edges
        grid = np.zeros((64, 64), dtype=np.uint8)
        # Vertex 0: block at (10..18, 10..18)
        grid[10:19, 10:19] = 2
        # Vertex 1: block at (10..18, 40..48)
        grid[10:19, 40:49] = 2
        # Vertex 2: block at (40..48, 25..33)
        grid[40:49, 25:34] = 2
        # Edge between Vertex 0 and Vertex 1: horizontal line on row 14
        grid[14, 19:40] = 2
        # Edge between Vertex 0 and Vertex 2: diagonal line
        for step in range(len(range(19, 40))):
            r = 19 + step
            c = int(round(14 + step * (29 - 14) / 21))
            grid[r, c] = 2

        graph_data = UniversalIndependentSetSolver.extract_graph_from_grid(
            grid, bg_color=0, frame_colors=set(), submit_btn=(1, 0)
        )
        self.assertIsNotNone(graph_data)
        vertices, edges = graph_data
        self.assertEqual(len(vertices), 3)
        self.assertGreaterEqual(len(edges), 1)

    def test_option_subset_combinations(self):
        avail_options = [1, 2, 3, 4]
        all_subsets = []
        for k in range(1, len(avail_options) + 1):
            for combo in itertools.combinations(avail_options, k):
                all_subsets.append(tuple(sorted(combo)))
        self.assertEqual(len(all_subsets), 15)
        # First 4 are singletons
        self.assertEqual(all_subsets[:4], [(1,), (2,), (3,), (4,)])
        # Next 6 are pairs
        self.assertEqual(len([s for s in all_subsets if len(s) == 2]), 6)
        # Next 4 are triples
        self.assertEqual(len([s for s in all_subsets if len(s) == 3]), 4)
        # Final is quadruple
        self.assertEqual(all_subsets[-1], (1, 2, 3, 4))


class TestPassiveConveyorEngine(unittest.TestCase):
    def test_conveyor_step_simulation(self):
        field = PassiveVelocityField()
        # Conveyor at (5, 2) and (5, 1) pushes LEFT by (-1 column)
        field.cell_velocity[(5, 2)] = (0, -1)
        field.cell_velocity[(5, 1)] = (0, -1)

        # Player at (4, 2) moves DOWN (dr=1, dc=0) to (5, 2)
        # Conveyor at (5, 2) pushes to (5, 1), then conveyor at (5, 1) pushes to (5, 0)
        end_pos = field.simulate_step(
            pos=(4, 2),
            action_vector=(1, 0),
            walls=set(),
            gh=10,
            gw=10,
        )
        self.assertEqual(end_pos, (5, 0))

    def test_conveyor_astar_navigation(self):
        field = PassiveVelocityField()
        # Conveyor at row 2 moves player RIGHT by (+2 cols)
        for c in range(1, 5):
            field.cell_velocity[(2, c)] = (0, 1)

        # Start at (2, 0), goal at (2, 5)
        path = solve_conveyor_astar(
            p_start=(2, 0),
            goals={(2, 5)},
            walls=set(),
            gh=10,
            gw=10,
            field=field,
            displacement_vectors={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)},
        )
        self.assertIsNotNone(path)
        # Should reach goal very fast thanks to conveyor boost
        self.assertLessEqual(len(path), 3)


class TestOnlineWorldModelSynthesizer(unittest.TestCase):
    def test_push_chain_success(self):
        # Player at (2, 2), Crate 1 at (2, 3), Crate 2 at (2, 4)
        # Push RIGHT along action vector (0, 1)
        p = (2, 2)
        boxes = {(2, 3), (2, 4)}
        np_pos, new_boxes, success = MultiStrideKinematicsSimulator.simulate_push_chain(
            p_pos=p,
            action_vector=(0, 1),
            boxes=boxes,
            walls=set(),
            gh=10,
            gw=10,
        )
        self.assertTrue(success)
        self.assertEqual(np_pos, (2, 3))
        self.assertEqual(new_boxes, {(2, 4), (2, 5)})

    def test_push_chain_blocked_by_wall(self):
        # Wall at (2, 5) blocks the train
        p = (2, 2)
        boxes = {(2, 3), (2, 4)}
        walls = {(2, 5)}
        np_pos, new_boxes, success = MultiStrideKinematicsSimulator.simulate_push_chain(
            p_pos=p,
            action_vector=(0, 1),
            boxes=boxes,
            walls=walls,
            gh=10,
            gw=10,
        )
        self.assertFalse(success)
        self.assertEqual(np_pos, p)
        self.assertEqual(new_boxes, boxes)

    def test_satisfaction_bar_detection(self):
        from project_override.core.combinatorial_engine import (
            UniversalSatisfactionBarDetector,
            UniversalSubmitButtonDetector,
        )
        grid = np.zeros((64, 64), dtype=np.int32)
        # Put an 8-bit satisfaction bar on row 63 with 6 green (11) and 2 red (4)
        grid[63, 1:7] = 11
        grid[63, 7:9] = 4
        bar = UniversalSatisfactionBarDetector.detect(grid, bg_color=0)
        self.assertIsNotNone(bar)
        self.assertEqual(bar.row, 63)
        self.assertEqual(len(bar.col_indices), 8)
        self.assertFalse(bar.all_satisfied)

        # Make all 8 slots green
        grid[63, 1:9] = 11
        bar2 = UniversalSatisfactionBarDetector.detect(grid, bg_color=0)
        self.assertIsNotNone(bar2)
        self.assertTrue(bar2.all_satisfied)

    def test_submit_button_detection(self):
        from project_override.core.combinatorial_engine import UniversalSubmitButtonDetector
        grid = np.zeros((64, 64), dtype=np.int32)
        # Put a button rectangle at rows 4..7, cols 46..53
        grid[4:8, 46:54] = 6
        btn = UniversalSubmitButtonDetector.detect_submit_button(grid, bg_color=0, frame_colors=set())
        self.assertIsNotNone(btn)
        cx, cy = btn
        self.assertTrue(46 <= cx <= 53)
        self.assertTrue(4 <= cy <= 7)

    def test_sat_solver_arbitrary_variables(self):
        from project_override.core.combinatorial_engine import (
            UniversalSatisfactionBarDetector,
            UniversalSATClauseSolver,
        )
        grid = np.zeros((64, 64), dtype=np.int32)
        # Satisfaction bar with 4 slots, 1 unsatisfied
        grid[63, 1:4] = 11
        grid[63, 4] = 4
        # Add 10 variable boxes (5x5 each)
        for i in range(10):
            r = 10 + (i // 5) * 12
            c = 10 + (i % 5) * 10
            grid[r:r+4, c:c+4] = 3

        boxes = UniversalSATClauseSolver.extract_variable_boxes(grid, bg_color=0, frame_colors=set())
        self.assertEqual(len(boxes), 10)

        # Plan actions should return the first untoggled box even for V=10
        plan = UniversalSATClauseSolver.plan_sat_actions(
            grid=grid,
            prev_grid=None,
            bg_color=0,
            frame_colors=set(),
            click_history=[],
            submit_btn=(50, 5),
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan, [boxes[0]])

        # If all satisfied, plan should return submit button
        grid[63, 4] = 11
        plan_sat = UniversalSATClauseSolver.plan_sat_actions(
            grid=grid,
            prev_grid=None,
            bg_color=0,
            frame_colors=set(),
            click_history=[],
            submit_btn=(50, 5),
        )
        self.assertEqual(plan_sat, [(50, 5)])

    def test_option_cursor_ignores_top_rows(self):
        from project_override.core.combinatorial_engine import UniversalOptionSelectorEngine
        grid = np.zeros((64, 64), dtype=np.int32)
        # Single pixel on row 1 or 2 should NOT be detected as selector cursor
        grid[1, 10] = 7
        grid[2, 20] = 7
        cursor = UniversalOptionSelectorEngine.detect_selector_cursor(grid, bg_color=0)
        self.assertIsNone(cursor)

        # But cursor on row 62 (h-2) SHOULD be detected
        grid[62, 15] = 7
        cursor2 = UniversalOptionSelectorEngine.detect_selector_cursor(grid, bg_color=0)
        self.assertIsNotNone(cursor2)
        self.assertEqual(cursor2, (62, 15, 7))

    def test_extract_discrete_board_conveyor_exemption(self):
        from project_override.core.universal_engine import extract_discrete_board
        grid = np.zeros((64, 64), dtype=np.int32)
        # Avatar at (10, 10) of size 4x4
        grid[10:14, 10:14] = 1
        # Conveyor track of color 8 along rows 20..24
        grid[20:24, 10:50] = 8
        # Pass conveyor_colors={8}
        world = extract_discrete_board(
            grid,
            avatar_color=1,
            conveyor_colors={8},
        )
        self.assertIsNotNone(world)
        # Color 8 cells should NOT be in walls!
        for w in world["walls"]:
            self.assertNotEqual(int(world["board"][w[0], w[1]]), 8)


    def test_single_token_indicator_and_bottom_submit(self):
        from project_override.core.combinatorial_engine import (
            UniversalSatisfactionBarDetector,
            UniversalSubmitButtonDetector,
        )
        grid = np.zeros((64, 64), dtype=np.int32)
        # Indicator token at (1, 0)
        grid[0, 1] = 14
        # Submit button token at (1, 63)
        grid[63, 1] = 6

        bar = UniversalSatisfactionBarDetector.detect(grid, bg_color=0)
        self.assertIsNotNone(bar)
        self.assertEqual(bar.row, 0)
        self.assertEqual(bar.col_indices, [1])
        self.assertTrue(bar.all_satisfied)

        submit_btn = UniversalSubmitButtonDetector.detect_submit_button(grid, bg_color=0, frame_colors=set())
        self.assertEqual(submit_btn, (1, 63))

    def test_gray_code_circuit_toggle(self):
        from project_override.core.combinatorial_engine import (
            UniversalSatisfactionBarDetector,
            UniversalSATClauseSolver,
        )
        grid = np.zeros((64, 64), dtype=np.int32)
        # Indicator token at (1, 0) starts unsatisfied (color 4)
        grid[0, 1] = 4
        # Submit button at (1, 63)
        grid[63, 1] = 6
        # 3 variable boxes
        for i, c_pos in enumerate([10, 25, 40]):
            grid[20:25, c_pos:c_pos+5] = 3

        boxes = UniversalSATClauseSolver.extract_variable_boxes(grid, bg_color=0, frame_colors=set())
        self.assertEqual(len(boxes), 3)

        # Step 0: Gray code flips first variable
        plan0 = UniversalSATClauseSolver.plan_sat_actions(
            grid=grid,
            prev_grid=None,
            bg_color=0,
            frame_colors=set(),
            click_history=[],
            submit_btn=(1, 63),
        )
        self.assertIsNotNone(plan0)
        self.assertEqual(len(plan0), 1)

        # When indicator turns to color 14, plan should return submit button immediately
        grid[0, 1] = 14
        plan_win = UniversalSATClauseSolver.plan_sat_actions(
            grid=grid,
            prev_grid=None,
            bg_color=0,
            frame_colors=set(),
            click_history=[plan0[0]],
            submit_btn=(1, 63),
        )
        self.assertEqual(plan_win, [(1, 63)])


if __name__ == "__main__":
    unittest.main()

