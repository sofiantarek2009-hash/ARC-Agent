"""Project Override: Test Suite for Perception (Zero-NN CCL) and Physics Engine."""

from __future__ import annotations

import unittest
import numpy as np

from project_override.core.zero_nn_ccl import (
    ZeroNNPerception,
    BoundingBox,
    Component,
)
from project_override.core.physics_engine import (
    Action,
    EntityType,
    PhysicsEngine,
    compute_deterministic_intercept,
)


class TestZeroNNPerception(unittest.TestCase):
    """Tests for Zero-NN Connected Component Labeling, Border Mask, and Topological Hashing."""

    def test_empty_grid(self):
        empty = np.zeros((64, 64), dtype=int)
        comps = ZeroNNPerception.extract_components(empty)
        self.assertEqual(len(comps), 0)
        h, _ = ZeroNNPerception.hash_scene(comps)
        self.assertIsInstance(h, str)

    def test_translation_invariance(self):
        grid1 = np.zeros((64, 64), dtype=int)
        grid1[10:15, 10:15] = 3
        comps1 = ZeroNNPerception.extract_components(grid1)
        h1, _ = ZeroNNPerception.hash_scene(comps1)

        grid2 = np.zeros((64, 64), dtype=int)
        grid2[40:45, 50:55] = 3
        comps2 = ZeroNNPerception.extract_components(grid2)
        h2, _ = ZeroNNPerception.hash_scene(comps2)

        self.assertEqual(h1, h2, "Hashing must be invariant under translation")

    def test_border_exclusion_mask(self):
        timer_grid = np.zeros((64, 64), dtype=int)
        timer_grid[:4, :] = 8  # Border timer bar
        timer_grid[20:25, 20:25] = 2  # Interior player

        interior, border = ZeroNNPerception.separate_grid(timer_grid, border_width=4)
        self.assertEqual(np.sum(interior == 8), 0, "Interior must not contain border timer")
        self.assertEqual(np.sum(border == 2), 0, "Border must not contain interior player")


class TestPhysicsEngine(unittest.TestCase):
    """Tests for Kinematic Grid Simulation, Sokoban Pushes, ACTION6 Intercept, and Undo."""

    def setUp(self):
        self.engine = PhysicsEngine(width=15, height=15)

    def test_movement_and_wall_collision(self):
        # Place player at (5, 5) and wall at (4, 5)
        self.engine.spawn_player(row=5, col=5, push_strength=1.0)
        self.engine.spawn_entity(EntityType.WALL, pos=(4, 5))
        self.engine.checkpoint_root()

        # Attempt to move UP (collision with wall)
        res = self.engine.step(Action.ACTION1_UP)
        self.assertFalse(res.success, "Movement into wall must fail")
        self.assertEqual(self.engine.player_pos, (5, 5))

        # Move DOWN (clear space)
        res = self.engine.step(Action.ACTION2_DOWN)
        self.assertTrue(res.success, "Movement into empty space must succeed")
        self.assertEqual(self.engine.player_pos, (6, 5))

    def test_sokoban_push(self):
        # Player at (5, 5), Box at (5, 6), empty at (5, 7)
        self.engine.spawn_player(row=5, col=5, push_strength=2.0)
        self.engine.spawn_entity(EntityType.MOVABLE, pos=(5, 6), mass=1.0)
        self.engine.checkpoint_root()

        # Step RIGHT: pushes box to (5, 7), player to (5, 6)
        res = self.engine.step(Action.ACTION4_RIGHT)
        self.assertTrue(res.success)
        self.assertEqual(self.engine.player_pos, (5, 6))

    def test_undo_stack(self):
        self.engine.spawn_player(row=5, col=5, push_strength=1.0)
        self.engine.checkpoint_root()

        # Move RIGHT to (5, 6)
        self.engine.step(Action.ACTION4_RIGHT)
        self.assertEqual(self.engine.player_pos, (5, 6))

        # ACTION7: Undo back to (5, 5)
        res = self.engine.step(Action.ACTION7_UNDO)
        self.assertTrue(res.success)
        self.assertEqual(self.engine.player_pos, (5, 5))

    def test_deterministic_intercept_action6(self):
        # Stationary target at (10, 10)
        pos, t, ok = compute_deterministic_intercept(
            source_pos=(2, 2),
            target_pos=(10, 10),
            target_vel=(0, 0),
            speed=1.0,
            metric="manhattan",
        )
        self.assertTrue(ok)
        self.assertEqual(pos, (10, 10))


if __name__ == "__main__":
    unittest.main()
