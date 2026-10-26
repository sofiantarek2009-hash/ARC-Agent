"""Unit tests for Circuit, Delay Pipeline, Hold Mechanics, and Beacon Triangulation engines."""

import unittest
import numpy as np

from project_override.core.circuit_engine import (
    CardinalDir,
    rot90_cw,
    rot90_ccw,
    get_mask_orbit,
    has_port,
    UniversalCircuitPerception,
    UniversalCircuitSolver,
    UniversalCircuitEngineFacade,
    TerminalNode,
)
from project_override.core.delay_hold_engine import (
    ActionPipelineTracker,
    solve_delayed_pipeline_astar,
    HoldMechanicInducer,
    synthesize_holding_action,
    solve_inertial_slide_to_hold,
)
from project_override.core.triangulation_engine import (
    ManhattanTriangulator,
    PerceptualBeaconSignalExtractor,
    UniversalBeaconPlanner,
)


class TestCircuitEngine(unittest.TestCase):
    def test_port_rotation_orbits(self):
        # North-East elbow mask: North (1) | East (2) = 3
        m = 3
        cw1 = rot90_cw(m)
        self.assertEqual(cw1, 6)  # East (2) | South (4) = 6
        cw2 = rot90_cw(cw1)
        self.assertEqual(cw2, 12) # South (4) | West (8) = 12
        cw3 = rot90_cw(cw2)
        self.assertEqual(cw3, 9)  # West (8) | North (1) = 9
        cw4 = rot90_cw(cw3)
        self.assertEqual(cw4, 3)

        # Orbit of a straight wire N-S: 5
        orbit_straight = get_mask_orbit(5)
        self.assertEqual(len(orbit_straight), 2)
        self.assertEqual(set(orbit_straight), {5, 10})

    def test_point_to_point_path_solver(self):
        # 3x3 grid where (0, 0) needs to connect to (0, 2)
        # Row 0: Elbow(N-E: 3), Straight(E-W: 10), Elbow(S-W: 12)
        # All randomly rotated initially
        grid_masks = [
            [rot90_cw(3), rot90_cw(10), rot90_cw(12)],
            [0, 0, 0],
            [0, 0, 0],
        ]
        res = UniversalCircuitSolver.solve_point_to_point_path(
            grid_masks,
            start_pos=(0, 0),
            start_in_dir=CardinalDir.NORTH,
            goal_pos=(0, 2),
            goal_out_dir=CardinalDir.SOUTH,
        )
        self.assertIsNotNone(res)
        min_clicks, moves = res
        self.assertGreater(min_clicks, 0)
        self.assertTrue(len(moves) > 0)


class TestDelayHoldEngine(unittest.TestCase):
    def test_pipeline_tracker_induction(self):
        tracker = ActionPipelineTracker(max_k=4)
        # Simulate a 3-step action delay: actions executed are a_{t-3}
        actions = [4, 2, 1, 3, 4, 1]
        p = (10, 10)
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

        for t, act in enumerate(actions):
            p_prev = p
            if t < 3:
                dp = (0, 0)
            else:
                exec_act = actions[t - 3]
                dp = deltas[exec_act]
            p = (p_prev[0] + dp[0], p_prev[1] + dp[1])
            k = tracker.record_transition(act, p_prev, p)

        self.assertEqual(tracker.inferred_k, 3)
        self.assertTrue(tracker.confirmed)

    def test_delayed_pipeline_astar(self):
        # Simple straight path with latency K=2
        start = (5, 5)
        goals = {(5, 8)}
        walls = set()
        queue = (0, 0)
        plan = solve_delayed_pipeline_astar(start, queue, goals, walls, gh=10, gw=10)
        self.assertIsNotNone(plan)
        # Must reach goal in at least 3 actions (Right, Right, Right)
        self.assertGreaterEqual(len(plan), 3)

    def test_hold_mechanic_inducer(self):
        inducer = HoldMechanicInducer(max_probe_ticks=5)
        goal = (4, 4)
        f_prev = np.zeros((64, 64), dtype=np.int32)
        f_curr = np.zeros((64, 64), dtype=np.int32)
        f_curr[1, 1] = 2  # HUD progress pixel

        # Step 1: land on goal plate
        res1 = inducer.on_step((4, 4), {goal}, f_prev, f_curr, level_won=False)
        self.assertTrue(res1['probe_hold'])
        self.assertEqual(res1['dwell_ticks'], 1)
        self.assertTrue(res1['hud_feedback'])

        # Step 2: dwell tick 2
        res2 = inducer.on_step((4, 4), {goal}, f_prev, f_curr, level_won=True)
        self.assertEqual(res2['status'], 'WON')
        self.assertEqual(res2['hold_ticks'], 2)
        self.assertTrue(inducer.hold_confirmed)


class TestTriangulationEngine(unittest.TestCase):
    def test_manhattan_triangulation_exact(self):
        target = (17, 23)
        obs1 = (10, 10, abs(target[0] - 10) + abs(target[1] - 10))
        obs2 = (30, 20, abs(target[0] - 30) + abs(target[1] - 20))
        obs3 = (15, 35, abs(target[0] - 15) + abs(target[1] - 35))

        res = ManhattanTriangulator.triangulate([obs1, obs2, obs3])
        self.assertTrue(res.is_unique)
        self.assertEqual(res.target, target)

    def test_beacon_planner_execution(self):
        planner = UniversalBeaconPlanner(grid_shape=(30, 30))
        target = (12, 18)
        walls = set()

        # Step 0: at spawn (5, 5)
        spawn = (5, 5)
        d0 = abs(target[0] - spawn[0]) + abs(target[1] - spawn[1])
        act0, _ = planner.plan_next_action(spawn, walls, d0, legal_actions=[1, 2, 3, 4, 5])
        self.assertIn(act0, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
