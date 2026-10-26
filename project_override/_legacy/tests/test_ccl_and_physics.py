"""
Unit tests for zero_nn_ccl.py and physics_engine.py
Project Override (ARC-AGI-3)
"""

import math
import sys
from pathlib import Path
import unittest

import numpy as np

# Ensure project_override package is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_override.core.zero_nn_ccl import (
    BoundingBox,
    Component,
    SceneGraph,
    ZeroNNPerception,
    compute_convex_hull_2d,
    rasterize_convex_polygon,
)

from project_override.core.physics_engine import (
    Action,
    EntityType,
    Entity,
    PhysicsEvent,
    StepResult,
    DiscretePhysicsEngine,
    compute_deterministic_intercept,
    CPPKinematicsCore,
)


# =============================================================================
# Test Suite 1: zero_nn_ccl.py
# =============================================================================

class TestZeroNNPerception(unittest.TestCase):
    """Thorough tests for Zero-NN Connected Component Labeling & Feature Extraction."""

    def test_ccl_connectivity_4_vs_8(self):
        """Verify difference between 4-way orthogonal and 8-way diagonal connectivity."""
        # Diagonal checkerboard pattern of 2 cells
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2, 2] = 5
        grid[3, 3] = 5

        # 4-way connectivity: diagonally adjacent pixels are separate components
        comps_4 = ZeroNNPerception.extract_components(grid, connectivity=4)
        self.assertEqual(len(comps_4), 2)
        self.assertEqual(comps_4[0].mass, 1)
        self.assertEqual(comps_4[1].mass, 1)

        # 8-way connectivity: diagonally adjacent pixels merge into 1 component
        comps_8 = ZeroNNPerception.extract_components(grid, connectivity=8)
        self.assertEqual(len(comps_8), 1)
        self.assertEqual(comps_8[0].mass, 2)
        self.assertEqual(comps_8[0].bbox.height, 2)
        self.assertEqual(comps_8[0].bbox.width, 2)

    def test_empty_grid(self):
        """Empty grid returns 0 components and deterministic empty scene hash."""
        empty_grid = np.zeros((64, 64), dtype=np.int32)
        comps = ZeroNNPerception.extract_components(empty_grid)
        self.assertEqual(len(comps), 0)

        hash_hex, hash_u64 = ZeroNNPerception.hash_scene(comps)
        self.assertIsInstance(hash_hex, str)
        self.assertEqual(len(hash_hex), 16)
        self.assertIsInstance(hash_u64, int)

    def test_multi_color_partitioning(self):
        """Distinct colors are partitioned into distinct components."""
        grid = np.zeros((20, 20), dtype=np.int32)
        # Red square (color 1)
        grid[2:5, 2:5] = 1
        # Blue line (color 2) touching the red square
        grid[2:5, 5] = 2

        comps = ZeroNNPerception.extract_components(grid, connectivity=4, per_color=True)
        self.assertEqual(len(comps), 2)
        colors = {c.color for c in comps}
        self.assertEqual(colors, {1, 2})

        red_comp = [c for c in comps if c.color == 1][0]
        blue_comp = [c for c in comps if c.color == 2][0]

        self.assertEqual(red_comp.mass, 9)
        self.assertEqual(blue_comp.mass, 3)

    def test_feature_extraction_single_pixel(self):
        """Feature extraction for an isolated 1-pixel component."""
        grid = np.zeros((30, 30), dtype=np.int32)
        grid[15, 12] = 4

        comps = ZeroNNPerception.extract_components(grid)
        self.assertEqual(len(comps), 1)
        c = comps[0]

        self.assertEqual(c.color, 4)
        self.assertEqual(c.mass, 1)
        self.assertEqual(c.bbox.min_row, 15)
        self.assertEqual(c.bbox.min_col, 12)
        self.assertEqual(c.bbox.max_row, 15)
        self.assertEqual(c.bbox.max_col, 12)
        self.assertEqual(c.bbox.height, 1)
        self.assertEqual(c.bbox.width, 1)
        self.assertEqual(c.bbox.area, 1)
        self.assertEqual(c.centroid, (15.0, 12.0))
        self.assertEqual(c.perimeter_pixels, 1)
        self.assertEqual(c.perimeter_edges, 4)
        self.assertEqual(c.solidity, 1.0)
        self.assertEqual(c.density, 1.0)
        self.assertTrue(c.is_interior)
        self.assertFalse(c.is_border)

    def test_feature_extraction_solid_block(self):
        """Feature extraction for a 2x3 solid rectangle."""
        grid = np.zeros((30, 30), dtype=np.int32)
        grid[10:12, 10:13] = 7

        comps = ZeroNNPerception.extract_components(grid)
        self.assertEqual(len(comps), 1)
        c = comps[0]

        self.assertEqual(c.mass, 6)
        self.assertEqual(c.bbox.height, 2)
        self.assertEqual(c.bbox.width, 3)
        self.assertEqual(c.bbox.area, 6)
        self.assertEqual(c.centroid, (10.5, 11.0))
        self.assertEqual(c.perimeter_pixels, 6)
        self.assertEqual(c.perimeter_edges, 10)  # 2*2 + 2*3 = 10
        self.assertEqual(c.convex_hull_area, 6)
        self.assertEqual(c.solidity, 1.0)
        self.assertEqual(c.density, 1.0)

    def test_feature_extraction_hollow_donut(self):
        """Feature extraction for a 3x3 hollow square with a hole in the center."""
        grid = np.zeros((30, 30), dtype=np.int32)
        grid[5:8, 5:8] = 3
        grid[6, 6] = 0  # Hole in the center

        comps = ZeroNNPerception.extract_components(grid, connectivity=4)
        self.assertEqual(len(comps), 1)
        c = comps[0]

        self.assertEqual(c.mass, 8)
        self.assertEqual(c.bbox.height, 3)
        self.assertEqual(c.bbox.width, 3)
        self.assertEqual(c.bbox.area, 9)
        self.assertEqual(c.centroid, (6.0, 6.0))
        self.assertEqual(c.perimeter_pixels, 8)
        self.assertEqual(c.convex_hull_area, 9)
        self.assertAlmostEqual(c.solidity, 8.0 / 9.0, places=4)
        self.assertAlmostEqual(c.density, 8.0 / 9.0, places=4)

    def test_border_exclusion_mask(self):
        """Separation of 4-cell outer timer border from interior gameplay grid."""
        h, w = 64, 64
        grid = np.zeros((h, w), dtype=np.int32)

        # Action-budget timer bar on top border
        grid[:4, :] = 9
        # Interior player avatar
        grid[20:23, 20:23] = 2

        interior_grid, border_grid = ZeroNNPerception.separate_grid(grid, border_width=4)

        # Interior must have no border timer pixels
        self.assertEqual(np.sum(interior_grid == 9), 0)
        self.assertEqual(np.sum(interior_grid == 2), 9)

        # Border grid must have only border timer pixels
        self.assertEqual(np.sum(border_grid == 2), 0)
        self.assertEqual(np.sum(border_grid == 9), 64 * 4)

        # Test gameplay change detection
        # Shrinking timer: reduce top border from 64 to 32 pixels
        next_grid = grid.copy()
        next_grid[:4, 32:] = 0  # Timer bar depleted by half
        # Interior player moved right by 1 cell
        next_grid[20:23, 20:23] = 0
        next_grid[20:23, 21:24] = 2

        gameplay_diff = ZeroNNPerception.detect_gameplay_changes(grid, next_grid, border_width=4)

        # Must ignore timer changes in outer 4-cell border
        self.assertFalse(np.any(gameplay_diff[:4, :]))
        # Must detect interior gameplay changes
        self.assertTrue(np.any(gameplay_diff[20:23, 20:24]))

        # Test border timer state telemetry
        timer_state = ZeroNNPerception.extract_border_timer_state(grid, border_width=4)
        self.assertEqual(timer_state["active_pixel_count"], 256)
        self.assertEqual(timer_state["colors"], [9])
        self.assertGreater(timer_state["budget_ratio"], 0.0)

    def test_relational_topological_invariant_hashing(self):
        """
        Verify that Blake2b-64 scene hashing is strictly invariant
        under arbitrary absolute translation shifts.
        """
        # Scene 1: Triangle of 3 components at (10, 10)
        grid1 = np.zeros((64, 64), dtype=np.int32)
        grid1[10:13, 10:13] = 1  # 3x3 red block
        grid1[18:20, 15:18] = 2  # 2x3 blue block
        grid1[12:15, 25:27] = 3  # 3x2 green block

        comps1 = ZeroNNPerception.extract_components(grid1)
        hash1, u64_1 = ZeroNNPerception.hash_scene(comps1)

        # Scene 2: The exact same configuration translated by (+25, +20)
        grid2 = np.zeros((64, 64), dtype=np.int32)
        grid2[35:38, 30:33] = 1  # 3x3 red block
        grid2[43:45, 35:38] = 2  # 2x3 blue block
        grid2[37:40, 45:47] = 3  # 3x2 green block

        comps2 = ZeroNNPerception.extract_components(grid2)
        hash2, u64_2 = ZeroNNPerception.hash_scene(comps2)

        self.assertEqual(hash1, hash2, "Translation invariance failed across scene shift!")
        self.assertEqual(u64_1, u64_2)

        # Scene 3: Perturb relative distance between components -> hash MUST change
        grid3 = grid1.copy()
        grid3[12:15, 25:27] = 0
        grid3[12:15, 28:30] = 3  # Shifted relative to other two objects

        comps3 = ZeroNNPerception.extract_components(grid3)
        hash3, _ = ZeroNNPerception.hash_scene(comps3)
        self.assertNotEqual(hash1, hash3, "Hash should be sensitive to relational topological changes!")


# =============================================================================
# Test Suite 2: physics_engine.py
# =============================================================================

class TestDiscretePhysicsEngine(unittest.TestCase):
    """Thorough tests for Discrete Grid Kinematics, Sokoban Dynamics, and Reversible Mechanics."""

    def test_64x64_canvas_simulation_and_movement(self):
        """Test basic canvas dimensions, player movement, and boundary limits."""
        engine = DiscretePhysicsEngine(width=64, height=64)
        self.assertEqual(engine.render().shape, (64, 64))

        player = engine.get_player()
        self.assertIsNotNone(player)
        start_r, start_c = player.pos

        # Move UP
        res1 = engine.step(Action.ACTION1_UP)
        self.assertTrue(res1.success)
        self.assertEqual(res1.player_pos, (start_r - 1, start_c))

        # Move DOWN
        res2 = engine.step(Action.ACTION2_DOWN)
        self.assertTrue(res2.success)
        self.assertEqual(res2.player_pos, (start_r, start_c))

        # Move LEFT
        res3 = engine.step(Action.ACTION3_LEFT)
        self.assertTrue(res3.success)
        self.assertEqual(res3.player_pos, (start_r, start_c - 1))

        # Move RIGHT
        res4 = engine.step(Action.ACTION4_RIGHT)
        self.assertTrue(res4.success)
        self.assertEqual(res4.player_pos, (start_r, start_c))

    def test_collision_detection_against_boundaries_and_walls(self):
        """Test collision prevention against grid limits and static walls."""
        engine = DiscretePhysicsEngine(width=10, height=10)
        # Position player at top border (0, 5)
        engine.spawn_player(row=0, col=5)

        # Move UP into grid boundary
        res_boundary = engine.step(Action.ACTION1_UP)
        self.assertFalse(res_boundary.success)
        self.assertEqual(engine.get_player().pos, (0, 5))

        # Place wall below at (1, 5)
        engine.spawn_entity(EntityType.WALL, (1, 5))
        res_wall = engine.step(Action.ACTION2_DOWN)
        self.assertFalse(res_wall.success)
        self.assertEqual(engine.get_player().pos, (0, 5))

    def test_sokoban_mass_proportional_pushing(self):
        """Test pushing chains of movable blocks with mass-proportional limits."""
        engine = DiscretePhysicsEngine(width=20, height=20)
        # Player with push strength = 2.0
        engine.spawn_player(row=10, col=5, push_strength=2.0)

        # Spawn chain of 2 boxes of mass 1.0 each (total mass = 2.0)
        b1 = engine.spawn_entity(EntityType.MOVABLE, (10, 6), mass=1.0)
        b2 = engine.spawn_entity(EntityType.MOVABLE, (10, 7), mass=1.0)

        # Push RIGHT: total mass 2.0 <= strength 2.0 -> SUCCEEDS
        res_push = engine.step(Action.ACTION4_RIGHT)
        self.assertTrue(res_push.success)
        self.assertEqual(engine.get_player().pos, (10, 6))
        self.assertEqual(b1.pos, (10, 7))
        self.assertEqual(b2.pos, (10, 8))

        # Add a 3rd box in front: total mass is now 3.0 > strength 2.0
        b3 = engine.spawn_entity(EntityType.MOVABLE, (10, 9), mass=1.0)
        res_push_fail = engine.step(Action.ACTION4_RIGHT)
        self.assertFalse(res_push_fail.success)
        # Positions must remain unchanged
        self.assertEqual(engine.get_player().pos, (10, 6))
        self.assertEqual(b1.pos, (10, 7))
        self.assertEqual(b2.pos, (10, 8))
        self.assertEqual(b3.pos, (10, 9))

        # Push blocked by static wall at end of chain
        engine.get_player().push_strength = 10.0  # Increase strength
        engine.spawn_entity(EntityType.WALL, (10, 10))  # Wall right after b3
        res_push_wall = engine.step(Action.ACTION4_RIGHT)
        self.assertFalse(res_push_wall.success)

    def test_action5_interact_toggle_and_collect(self):
        """Test collecting items and toggling switches to open doors."""
        engine = DiscretePhysicsEngine(width=20, height=20)
        engine.spawn_player(row=5, col=5)

        collectible = engine.spawn_entity(EntityType.COLLECTIBLE, (5, 6), color=5)
        door = engine.spawn_entity(EntityType.DOOR, (5, 8), color=6, is_passable=False)
        toggle = engine.spawn_entity(EntityType.TOGGLE, (5, 7), color=7, linked_doors=[door.id])

        # Walk onto collectible (5, 6)
        res_col = engine.step(Action.ACTION4_RIGHT)
        self.assertTrue(res_col.success)
        self.assertTrue(collectible.is_collected)
        self.assertIn(5, engine.inventory)
        self.assertGreater(engine.score, 0)

        # Walk onto toggle (5, 7)
        res_walk_toggle = engine.step(Action.ACTION4_RIGHT)
        self.assertTrue(res_walk_toggle.success)
        self.assertFalse(door.is_passable)

        # Interact with toggle (ACTION5)
        res_toggle = engine.step(Action.ACTION5_INTERACT)
        self.assertTrue(res_toggle.success)
        self.assertTrue(door.is_passable)
        self.assertTrue(toggle.is_active)

        # Player can now walk through the opened door at (5, 8)
        res_through_door = engine.step(Action.ACTION4_RIGHT)
        self.assertTrue(res_through_door.success)
        self.assertEqual(engine.get_player().pos, (5, 8))

    def test_action6_deterministic_intercept_geometry(self):
        """Test deterministic intercept geometry for stationary and moving targets."""
        # 1. Stationary target
        intercept_coord, t_steps, ok = compute_deterministic_intercept(
            source_pos=(10, 10),
            target_pos=(10, 30),
            target_vel=(0, 0),
            speed=1,
        )
        self.assertTrue(ok)
        self.assertEqual(intercept_coord, (10, 30))
        self.assertEqual(t_steps, 20)

        # 2. Moving target with velocity (1, 1), pursuer speed 2
        # Target pos at time t: (10 + t, 10 + t). Source: (0, 0).
        # Manhattan dist: (10 + t) + (10 + t) = 20 + 2t <= 2 * t -> no intercept under Manhattan!
        # With speed = 3: 20 + 2t <= 3t => t >= 20!
        coord_dyn, t_dyn, ok_dyn = compute_deterministic_intercept(
            source_pos=(0, 0),
            target_pos=(10, 10),
            target_vel=(1, 1),
            speed=3,
            metric="manhattan",
        )
        self.assertTrue(ok_dyn)
        self.assertEqual(t_dyn, 20)
        self.assertEqual(coord_dyn, (30, 30))

        # 3. Obstacle-aware BFS intercept
        obstacles = np.zeros((20, 20), dtype=bool)
        obstacles[5, 0:15] = True  # Barrier wall
        coord_bfs, t_bfs, ok_bfs = compute_deterministic_intercept(
            source_pos=(0, 0),
            target_pos=(10, 0),
            target_vel=(0, 0),
            speed=1,
            metric="bfs",
            grid_shape=(20, 20),
            obstacles=obstacles,
        )
        self.assertTrue(ok_bfs)
        self.assertEqual(coord_bfs, (10, 0))
        # Path must go around barrier at col 15 -> distance > 10
        self.assertGreater(t_bfs, 10)

    def test_action7_undo_and_reset(self):
        """Test state stack unwinding (ACTION7 Undo) and root baseline restoration (RESET)."""
        engine = DiscretePhysicsEngine(width=30, height=30)
        engine.spawn_player(row=15, col=15)
        box = engine.spawn_entity(EntityType.MOVABLE, (15, 16))
        engine.checkpoint_root()

        # Step 1: Push box right
        engine.step(Action.ACTION4_RIGHT)
        self.assertEqual(engine.get_player().pos, (15, 16))
        self.assertEqual(box.pos, (15, 17))

        # Step 2: Move down
        engine.step(Action.ACTION2_DOWN)
        self.assertEqual(engine.get_player().pos, (16, 16))

        # Step 3: Move left
        engine.step(Action.ACTION3_LEFT)
        self.assertEqual(engine.get_player().pos, (16, 15))

        # Undo Step 3 -> back to (16, 16)
        res_u1 = engine.undo()
        self.assertTrue(res_u1.success)
        self.assertEqual(engine.get_player().pos, (16, 16))

        # Undo Step 2 -> back to (15, 16)
        res_u2 = engine.undo()
        self.assertTrue(res_u2.success)
        self.assertEqual(engine.get_player().pos, (15, 16))

        # Undo Step 1 -> player back to (15, 15), box back to (15, 16)
        res_u3 = engine.undo()
        self.assertTrue(res_u3.success)
        self.assertEqual(engine.get_player().pos, (15, 15))
        self.assertEqual(box.pos, (15, 16))

        # Undo at root state -> should return False cleanly
        res_u4 = engine.undo()
        self.assertFalse(res_u4.success)
        self.assertEqual(engine.get_player().pos, (15, 15))

        # Move around again then test RESET
        engine.step(Action.ACTION1_UP)
        engine.step(Action.ACTION1_UP)
        self.assertEqual(engine.get_player().pos, (13, 15))

        res_reset = engine.reset()
        self.assertTrue(res_reset.success)
        self.assertEqual(engine.get_player().pos, (15, 15))
        self.assertEqual(box.pos, (15, 16))


if __name__ == "__main__":
    unittest.main()
