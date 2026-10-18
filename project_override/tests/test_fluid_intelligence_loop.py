"""Unit tests for ARC-AGI-3 Frontier Fluid Intelligence Loop
Verifies:
1. Universal Semantic Delta Engine & World Model Transition Graph
2. Delta-Driven Goal Induction Engine & Solved-Level Memory Transfer
3. Simulated Model Planner, Deadlock Pruning & Epistemic Prober
"""

from __future__ import annotations

import unittest
import numpy as np

from project_override.core.world_model_engine import (
    UniversalSemanticDeltaEngine,
    WorldModelTransitionGraph,
    FunctionalRole,
    KinematicModality,
)
from project_override.core.goal_induction_engine import (
    DeltaGoalInductionEngine,
    GoalCategory,
    GoalHypothesis,
    CausalWinInvariant,
    SolvedLevelMemory,
)
from project_override.core.simulated_planner import (
    SimulatedModelPlanner,
    KinematicTransitionModel,
    SimulatedState,
    DeadlockDetector,
    EpistemicProber,
    VerificationDeltaController,
)


class TestFluidIntelligenceLoop(unittest.TestCase):

    def setUp(self) -> None:
        self.delta_engine = UniversalSemanticDeltaEngine()
        self.world_model = WorldModelTransitionGraph(self.delta_engine)
        self.goal_engine = DeltaGoalInductionEngine()
        self.planner = SimulatedModelPlanner(border_margin=1)

    def test_semantic_delta_and_role_induction(self) -> None:
        """Verifies that the delta engine accurately induces functional roles."""
        grid0 = np.zeros((10, 10), dtype=int)
        grid0[0, :] = 5; grid0[-1, :] = 5; grid0[:, 0] = 5; grid0[:, -1] = 5
        grid0[1, 1:4] = 8   # HUD indicator
        grid0[5, 5] = 2     # Mover
        grid0[5, 4] = 7     # Co-mover
        grid0[5, 6] = 3     # Pushable
        grid0[5, 8] = 4     # Static Obstacle

        # Step 1: Move right (Action 4)
        grid1 = grid0.copy()
        grid1[1, 3] = 5     # HUD decrement
        grid1[5, 4] = 0
        grid1[5, 5] = 7     # Co-mover moved to (5, 5)
        grid1[5, 6] = 2     # Mover moved to (5, 6)
        grid1[5, 7] = 3     # Pushable displaced to (5, 7)

        d1 = self.delta_engine.process_transition(grid0, grid1, action=4, step_idx=1)
        self.world_model.record_transition(d1)

        self.assertEqual(self.delta_engine.discovered_roles.get(2), FunctionalRole.MOVER)
        self.assertEqual(self.delta_engine.discovered_roles.get(7), FunctionalRole.CO_MOVER)
        self.assertEqual(self.delta_engine.discovered_roles.get(3), FunctionalRole.PUSHABLE)
        self.assertEqual(self.delta_engine.discovered_roles.get(8), FunctionalRole.HUD_INDICATOR)

    def test_goal_induction_consumption_and_level_transfer(self) -> None:
        """Verifies consumption goal induction and zero-shot invariant transfer."""
        self.goal_engine.reset_for_game("test_consumption_game")

        g0 = np.zeros((8, 8), dtype=int)
        g0[2, 2] = 2  # Mover
        g0[2, 3] = 6  # Collectible 1
        g0[2, 4] = 6  # Collectible 2

        # Step 1: Collect item 1
        g1 = g0.copy()
        g1[2, 2] = 0
        g1[2, 3] = 2
        hyps = self.goal_engine.observe_transition(g0, g1, action=4, mover_prev=(2, 2), mover_curr=(2, 3))
        hyp = next(h for h in hyps if h.category == GoalCategory.CONSUMPTION)
        self.assertEqual(hyp.current_metric, 1.0)

        # Step 2: Collect item 2 -> LEVEL_CLEAR
        g2 = g1.copy()
        g2[2, 3] = 0
        g2[2, 4] = 2
        hyps = self.goal_engine.observe_transition(
            g1, g2, action=4, mover_prev=(2, 3), mover_curr=(2, 4), level_cleared=True, current_level=0
        )
        inv = self.goal_engine.memory.get_invariant(0)
        self.assertIsNotNone(inv)
        self.assertEqual(inv.category, GoalCategory.CONSUMPTION)
        self.assertIn(6, inv.target_colors)

        # Zero-shot transfer to Level 1
        g_lvl1 = np.zeros((10, 10), dtype=int)
        g_lvl1[5, 5] = 2
        g_lvl1[5, 7] = 6
        g_lvl1[8, 8] = 6
        guidance = self.goal_engine.plan_next_subgoal(g_lvl1, mover_pos=(5, 5), current_level=1)
        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.category, GoalCategory.CONSUMPTION)
        self.assertEqual(len(guidance.target_coordinates), 2)

    def test_simulated_planner_offline_geodesic_and_deadlocks(self) -> None:
        """Verifies offline A* simulation and deadlock prevention."""
        walls = {(1, 4), (2, 5)}
        # (2, 4) is formed by walls above and right -> concave corner
        is_dead = DeadlockDetector.is_concave_corner((2, 4), walls, goal_plates=set(), gh=10, gw=10)
        self.assertTrue(is_dead)

        # If (2, 4) is a goal plate, not a deadlock
        is_safe = DeadlockDetector.is_concave_corner((2, 4), walls, goal_plates={(2, 4)}, gh=10, gw=10)
        self.assertFalse(is_safe)

        # Sokoban forward simulation to goal
        model = KinematicTransitionModel(
            walls={(0, 0), (0, 1), (0, 2), (0, 3)},
            goal_plates={(2, 5)},
            grid_h=8,
            grid_w=8,
        )
        init_st = SimulatedState(
            player_pos=(2, 2),
            boxes=frozenset({(2, 3)}),
        )
        plan = self.planner.plan_offline(init_st, model)
        self.assertIsNotNone(plan)
        # Pushing box right twice: [4, 4]
        self.assertEqual(plan, [4, 4])

    def test_epistemic_prober_minimal_pair(self) -> None:
        """Verifies shortest reversible probe generation."""
        plan = EpistemicProber.plan_probe(
            player_pos=(3, 3),
            target_pos=(3, 4),
            walls=set(),
            gh=8,
            gw=8,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.probe_action, 4)    # Probe Right
        self.assertEqual(plan.reverse_action, 3)  # Reverse Left

    def test_verification_delta_divergence(self) -> None:
        """Verifies verification delta divergence detection."""
        g = np.zeros((8, 8), dtype=int)
        report = VerificationDeltaController.verify_step(
            actual_grid=g,
            actual_player_pos=(2, 3),
            expected_player_pos=(2, 4),
            expected_hash=None,
        )
        self.assertTrue(report.is_diverged)
        self.assertIn("Player pos diverged", report.divergence_reason)


if __name__ == "__main__":
    unittest.main()
