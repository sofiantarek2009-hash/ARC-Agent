"""Project Override: Comprehensive Test Suite for Semantic Evaluator,
Dead Signature, and Epistemic MCTS Planner.
Uses standard library unittest for zero-dependency execution.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

import numpy as np

from project_override.core.semantic_evaluator import (
    D4SymmetryScorer,
    BoundaryEnclosureScorer,
    CompactionScorer,
    StructuralEntropyScorer,
    SinkhornKnoppOptimalTransport,
    SemanticEvaluator,
)
from project_override.core.dead_signature import (
    ActionSignature,
    DeadSignatureFilter,
    pack_signature,
    unpack_signature,
    compute_context_hash,
)
from project_override.core.mcts_planner import (
    DiscreteActionType,
    EpistemicMCTSNode,
    EpistemicMCTSPlanner,
    simulate_grid_action,
)


class TestSemanticEvaluator(unittest.TestCase):
    """Test suite for Semantic Evaluator components."""

    # -----------------------------------------------------------------
    # 1. D4 Symmetry Scorer Tests
    # -----------------------------------------------------------------
    def test_d4_symmetry_horizontal(self):
        """Test horizontal reflection symmetry (flipud)."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
        ])
        scorer = D4SymmetryScorer()
        res = scorer.evaluate(grid)
        self.assertEqual(res.horizontal_reflection, 1.0)
        self.assertTrue(res.is_symmetric)
        self.assertEqual(res.dominant_symmetry, "horizontal_reflection")

    def test_d4_symmetry_vertical(self):
        """Test vertical reflection symmetry (fliplr)."""
        grid = np.array([
            [1, 2, 1],
            [3, 4, 3],
            [5, 6, 5],
        ])
        scorer = D4SymmetryScorer()
        res = scorer.evaluate(grid)
        self.assertEqual(res.vertical_reflection, 1.0)
        self.assertTrue(res.is_symmetric)

    def test_d4_symmetry_rotational_180(self):
        """Test 180-degree rotational symmetry."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 4],
            [3, 2, 1],
        ])
        scorer = D4SymmetryScorer()
        res = scorer.evaluate(grid)
        self.assertEqual(res.rotational_180, 1.0)
        self.assertTrue(res.is_symmetric)

    def test_d4_symmetry_diagonal(self):
        """Test main diagonal reflection symmetry (transpose)."""
        grid = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ])
        scorer = D4SymmetryScorer()
        res = scorer.evaluate(grid)
        self.assertEqual(res.diagonal_reflection_main, 1.0)
        self.assertTrue(res.is_symmetric)

    def test_d4_symmetry_rectangular_padding(self):
        """Test that rectangular grids (H != W) are properly evaluated with pad_to_square."""
        grid = np.array([
            [1, 2, 1],
            [3, 4, 3],
        ])
        scorer = D4SymmetryScorer(pad_to_square=True)
        res = scorer.evaluate(grid)
        self.assertEqual(res.vertical_reflection, 1.0)
        self.assertTrue(0.0 <= res.rotational_90 <= 1.0)

    # -----------------------------------------------------------------
    # 2. Boundary Enclosure & Connectivity Closure Tests
    # -----------------------------------------------------------------
    def test_boundary_enclosure_sealed_cavity(self):
        """Test detection of sealed loop enclosing a goal."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[1:4, 1] = 1
        grid[1:4, 3] = 1
        grid[1, 1:4] = 1
        grid[3, 1:4] = 1
        grid[2, 2] = 2  # Goal inside cavity

        scorer = BoundaryEnclosureScorer()
        res = scorer.compute_enclosure(grid, barrier_colors={1}, goal_color=2)

        self.assertTrue(res.has_sealed_loop)
        self.assertEqual(res.sealed_cavity_count, 1)
        self.assertTrue(res.is_goal_enclosed)
        self.assertIn((2, 2), res.enclosed_goal_cells)

    def test_boundary_enclosure_leaky_loop(self):
        """Test that an open / leaky border is NOT considered a sealed loop."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[1:4, 1] = 1
        grid[1:4, 3] = 1
        grid[1, 1:4] = 1
        # Missing bottom wall at row 3 (open leak)
        grid[2, 2] = 2

        scorer = BoundaryEnclosureScorer()
        res = scorer.compute_enclosure(grid, barrier_colors={1}, goal_color=2)

        self.assertFalse(res.has_sealed_loop)
        self.assertFalse(res.is_goal_enclosed)

    def test_ray_cast_containment(self):
        """Test 4-way ray-cast containment."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[1, 1:4] = 1
        grid[3, 1:4] = 1
        grid[1:4, 1] = 1
        grid[1:4, 3] = 1

        scorer = BoundaryEnclosureScorer()
        self.assertTrue(scorer.is_point_enclosed_raycast(grid, 2, 2))
        self.assertFalse(scorer.is_point_enclosed_raycast(grid, 0, 0))

    def test_path_connectivity_bfs(self):
        """Test BFS path connectivity."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[:, 2] = 1  # Wall
        scorer = BoundaryEnclosureScorer()

        connected, path = scorer.check_path_connectivity(
            grid, (2, 0), (2, 4), traversable_colors={0}
        )
        self.assertFalse(connected)

        grid[2, 2] = 0  # Door
        connected, path = scorer.check_path_connectivity(
            grid, (2, 0), (2, 4), traversable_colors={0}
        )
        self.assertTrue(connected)
        self.assertGreater(len(path), 0)

    # -----------------------------------------------------------------
    # 3. Compaction Metrics Tests
    # -----------------------------------------------------------------
    def test_compaction_metrics(self):
        """Test cluster density and perimeter-to-area compactness."""
        grid_compact = np.zeros((6, 6), dtype=np.int32)
        grid_compact[2:4, 2:4] = 1

        grid_line = np.zeros((6, 6), dtype=np.int32)
        grid_line[2, 1:5] = 1

        scorer = CompactionScorer()
        res_compact = scorer.evaluate(grid_compact)
        res_line = scorer.evaluate(grid_line)

        self.assertGreater(res_compact.compactness_score, res_line.compactness_score)
        self.assertEqual(res_compact.mean_cluster_density, 1.0)
        self.assertLess(res_compact.mean_radius_of_gyration, res_line.mean_radius_of_gyration)

    # -----------------------------------------------------------------
    # 4. Structural Shannon Entropy Tests
    # -----------------------------------------------------------------
    def test_structural_entropy_and_reward(self):
        """Test that orderly transition reducing entropy yields a reward."""
        rng = np.random.RandomState(42)
        noisy_state = rng.randint(1, 5, size=(6, 6))
        ordered_state = np.ones((6, 6), dtype=np.int32) * 1

        scorer = StructuralEntropyScorer()
        e_noisy = scorer.evaluate(noisy_state)
        e_ordered = scorer.evaluate(ordered_state)

        self.assertLess(e_ordered.total_entropy, e_noisy.total_entropy)

        trans = scorer.evaluate_transition(noisy_state, ordered_state)
        self.assertLess(trans.delta_h, 0.0)
        self.assertTrue(trans.is_rewarded)
        self.assertGreater(trans.reward, 0.0)

    # -----------------------------------------------------------------
    # 5. Discrete Optimal Transport (Sinkhorn-Knopp) Tests
    # -----------------------------------------------------------------
    def test_optimal_transport_identity(self):
        """Test optimal transport between identical point clouds (W2 ~ 0)."""
        pts = np.array([[2.0, 3.0], [5.0, 5.0]], dtype=np.float64)
        masses = np.array([4.0, 4.0], dtype=np.float64)

        ot = SinkhornKnoppOptimalTransport()
        res = ot.compute_wasserstein_w2(pts, pts, masses, masses)

        self.assertTrue(res.converged)
        self.assertLess(res.w2_distance, 1e-3)

    def test_optimal_transport_fission(self):
        """Test object fission: 1 parent object shattering into 4 shards."""
        src_pts = np.array([[10.0, 10.0]], dtype=np.float64)
        src_m = np.array([4.0], dtype=np.float64)

        tgt_pts = np.array([
            [8.0, 10.0],
            [12.0, 10.0],
            [10.0, 8.0],
            [10.0, 12.0],
        ], dtype=np.float64)
        tgt_m = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        ot = SinkhornKnoppOptimalTransport()
        res = ot.compute_wasserstein_w2(src_pts, tgt_pts, src_m, tgt_m)

        self.assertTrue(res.converged)
        self.assertTrue(math.isclose(res.w2_distance, 2.0, rel_tol=0.05))

    def test_optimal_transport_fusion(self):
        """Test object fusion: 4 shards merging into 1 merged centroid."""
        src_pts = np.array([
            [8.0, 10.0],
            [12.0, 10.0],
            [10.0, 8.0],
            [10.0, 12.0],
        ], dtype=np.float64)
        src_m = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        tgt_pts = np.array([[10.0, 10.0]], dtype=np.float64)
        tgt_m = np.array([4.0], dtype=np.float64)

        ot = SinkhornKnoppOptimalTransport()
        res = ot.compute_wasserstein_w2(src_pts, tgt_pts, src_m, tgt_m)

        self.assertTrue(res.converged)
        self.assertTrue(math.isclose(res.w2_distance, 2.0, rel_tol=0.05))


class TestDeadSignatureAndMCTS(unittest.TestCase):
    """Test suite for Dead Signature and Epistemic MCTS Planner."""

    # -----------------------------------------------------------------
    # 6. Dead Signature Encoder & O(1) Culling Tests
    # -----------------------------------------------------------------
    def test_action_signature_packing_unpacking(self):
        """Verify bitwise packing and unpacking preserves all fields."""
        action = 12
        color = 9
        mass = 54321
        context_hash = 0xA1B2C3D4
        flags = 0x5A

        sig_int = pack_signature(action, color, mass, context_hash, flags)
        sig_obj = unpack_signature(sig_int)

        self.assertEqual(sig_obj.action, action)
        self.assertEqual(sig_obj.color, color)
        self.assertEqual(sig_obj.mass, mass)
        self.assertEqual(sig_obj.context_hash, context_hash)
        self.assertEqual(sig_obj.flags, flags)
        self.assertEqual(sig_obj.to_uint64(), sig_int)

        hex_str = sig_obj.to_hex()
        from_hex_obj = ActionSignature.from_hex(hex_str)
        self.assertEqual(from_hex_obj, sig_obj)

    def test_dead_signature_culling(self):
        """Verify O(1) dead signature recording and culling."""
        filt = DeadSignatureFilter()
        sig = pack_signature(action=3, color=2, mass=100, context_hash=12345)

        self.assertFalse(filt.is_dead(sig))

        was_dead = filt.record_transition(sig, delta_omega=0)
        self.assertTrue(was_dead)
        self.assertTrue(filt.is_dead(sig))
        self.assertEqual(len(filt), 1)

        sig_live = pack_signature(action=4, color=2, mass=100, context_hash=12345)
        was_dead_live = filt.record_transition(sig_live, delta_omega=5)
        self.assertFalse(was_dead_live)
        self.assertFalse(filt.is_dead(sig_live))

    # -----------------------------------------------------------------
    # 7. Epistemic MCTS Planner Tests
    # -----------------------------------------------------------------
    def test_epistemic_uct_formula(self):
        """Verify UCT formula with epistemic damping:

        UCT(s, a) = Q(s, a) + C_p * exp(-lambda * V(s)) * sqrt(ln(N) / (1 + n))
        """
        node = EpistemicMCTSNode(
            state=np.zeros((3, 3), dtype=np.int32),
            prior_variance=0.0,
        )
        node.visit_count = 10
        node.action_visits[0] = 2
        node.action_values[0] = 0.5
        node.epistemic_variance = 0.0

        c_p = 1.4142
        lambd = 1.0

        score = node.compute_uct(0, c_p=c_p, lambd=lambd)
        expected_bonus = c_p * 1.0 * math.sqrt(math.log(11.0) / 3.0)
        expected_score = 0.5 + expected_bonus

        self.assertTrue(math.isclose(score, expected_score, rel_tol=1e-4))

        node.epistemic_variance = 2.0
        score_damped = node.compute_uct(0, c_p=c_p, lambd=lambd)
        expected_damped_bonus = c_p * math.exp(-2.0) * math.sqrt(math.log(11.0) / 3.0)
        expected_damped_score = 0.5 + expected_damped_bonus

        self.assertTrue(math.isclose(score_damped, expected_damped_score, rel_tol=1e-4))
        self.assertLess(score_damped, score)

    def test_mcts_planner_execution(self):
        """Verify end-to-end MCTS planning with dead-signature culling and semantic rewards."""
        initial_grid = np.zeros((5, 5), dtype=np.int32)
        initial_grid[2, 2] = 3

        target_grid = np.zeros((5, 5), dtype=np.int32)
        target_grid[1, 2] = 3

        planner = EpistemicMCTSPlanner(
            c_p=1.4142,
            lambd=1.0,
            max_depth=5,
        )

        plan_res = planner.search(
            root_state=initial_grid,
            target_state=target_grid,
            num_simulations=50,
            timeout_seconds=5.0,
        )

        self.assertGreater(plan_res.completed_iterations, 0)
        self.assertIn(plan_res.best_action, DiscreteActionType.NAMES)
        self.assertIsInstance(plan_res.macro_action_sequence, list)
        self.assertGreaterEqual(len(plan_res.macro_action_sequence), 1)
        self.assertGreaterEqual(plan_res.root_epistemic_variance, 0.0)

    def test_mcts_macro_queue_pipeline_integration(self):
        """Verify seamless pipeline: MCTS -> MacroPlanSteps -> MacroActionQueue."""
        from project_override.orchestration.macro_queue import MacroActionQueue

        initial_grid = np.zeros((5, 5), dtype=np.int32)
        initial_grid[2, 2] = 3

        planner = EpistemicMCTSPlanner(max_depth=4)
        plan_res = planner.search(root_state=initial_grid, num_simulations=30)

        # Convert to macro plan steps
        steps = planner.to_macro_plan_steps(plan_res, initial_grid, color=3)
        self.assertGreater(len(steps), 0)

        queue = MacroActionQueue(max_burst_size=4)
        queue.load_plan(steps)
        self.assertFalse(queue.is_empty)

        # Simulate live environment execution
        curr_state = np.copy(initial_grid)

        def env_step(action: int, action_data: dict):
            nonlocal curr_state
            curr_state = simulate_grid_action(curr_state, action, color=action_data.get("color", 1))
            return curr_state, False, {}

        result = queue.execute_burst(
            env_step_fn=env_step,
            hash_fn=compute_context_hash,
        )

        self.assertGreater(result.actions_executed, 0)
        self.assertTrue(queue.is_empty)


if __name__ == "__main__":
    unittest.main()
