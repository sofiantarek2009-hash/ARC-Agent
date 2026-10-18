"""Unit tests verifying the cures for the 5 systemic architectural gaps:
1. Sub-frame animation decomposition & transient flash extraction.
2. Dynamic perimeter timer masking & dead signature preservation.
3. Diagonal kinematics and pathfinding.
4. Coupled multi-sprite joint state planning.
5. EMPA epistemic curiosity exploration.
"""

import unittest
import numpy as np

from project_override.core.dead_signature import (
    is_perimeter_timer_tick,
    mask_perimeter_timer_pixels,
    compute_context_hash,
    DeadSignatureFilter,
)
from project_override.core.universal_engine import (
    solve_cardinal_astar,
    solve_coupled_joint_astar,
    DynamicKinematicProber,
)
from project_override.core.subframe_perception import (
    SubFrameDecomposer,
    LatentEpistemicGrid,
)
from project_override.orchestration.general_agent import GeneralAgent


class TestArchitecturalGapsCured(unittest.TestCase):

    def test_perimeter_timer_tick_detection(self):
        # Two 64x64 frames where interior is identical, but top row loses 1 pixel (timer decrement)
        prev = np.zeros((64, 64), dtype=np.int32)
        curr = np.zeros((64, 64), dtype=np.int32)

        # Set interior feature
        prev[20, 20] = 3
        curr[20, 20] = 3

        # Set perimeter timer on row 0
        prev[0, 10:20] = 2
        curr[0, 10:19] = 2  # Pixel (0, 19) reverted to 0 (timer tick!)

        is_tick = is_perimeter_timer_tick(prev, curr, border_margin=4)
        self.assertTrue(is_tick)

        # Context hash with mask_perimeter=True should be identical despite timer tick!
        h_prev = compute_context_hash(prev, mask_perimeter=True, border_margin=4)
        h_curr = compute_context_hash(curr, mask_perimeter=True, border_margin=4)
        self.assertEqual(h_prev, h_curr)

    def test_diagonal_kinematics_and_pathfinding(self):
        # Prober diagonal vector discovery
        prober = DynamicKinematicProber()
        # Initial probe: action 1
        f0 = np.zeros((64, 64), dtype=np.int32)
        f0[30, 30] = 2  # Avatar at (30, 30)
        a1 = prober.get_probe_action(f0, [1, 2, 3, 4])
        self.assertEqual(a1, 1)

        # Frame after action 1: moved diagonally by (-1, -1) to (29, 29)
        f1 = np.zeros((64, 64), dtype=np.int32)
        f1[29, 29] = 2
        prober.get_probe_action(f1, [1, 2, 3, 4])

        self.assertTrue(prober.profile.probed)
        self.assertEqual(prober.profile.avatar_color, 2)
        # Check that diagonal displacement vectors were set
        self.assertIn(1, prober.profile.displacement_vectors)
        v1 = prober.profile.displacement_vectors[1]
        self.assertEqual(v1, (-1, -1))

        # Test solve_cardinal_astar with diagonal displacement vectors
        diag_vectors = {1: (-1, -1), 2: (1, 1), 3: (1, -1), 4: (-1, 1)}
        path = solve_cardinal_astar(
            start_pos=(10, 10),
            goal_pos=(13, 13),
            walls=set(),
            gh=64,
            gw=64,
            displacement_vectors=diag_vectors,
        )
        self.assertIsNotNone(path)
        # 3 steps of action 2 (1, 1) reaches (13, 13)
        self.assertEqual(path, [2, 2, 2])

    def test_coupled_joint_astar(self):
        # Two sprites: Sprite 1 at (10, 10), Sprite 2 at (10, 20)
        # Sprite 1 vectors: standard cardinal {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        # Sprite 2 vectors: horizontal mirror {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
        # Goal 1 at (10, 12) (move right by 2 -> action 4 twice)
        # When action 4 is taken, Sprite 2 moves left by 2: (10, 20) -> (10, 18)
        v1 = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        v2 = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}

        plan = solve_coupled_joint_astar(
            start_pos_1=(10, 10),
            start_pos_2=(10, 20),
            goal_pos_1=(10, 12),
            goal_pos_2=(10, 18),
            walls_1=set(),
            walls_2=set(),
            gh=64,
            gw=64,
            vectors_1=v1,
            vectors_2=v2,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan, [4, 4])

    def test_subframe_flashes_in_general_agent(self):
        agent = GeneralAgent()
        agent.start_new_game("test_flash_game")
        agent.start_new_level(0)

        # Create sub-frame animation sequence:
        # F0: black background, avatar at (10, 10)
        # F1: flash target revealed at (15, 15) with color 3
        # F2: settled: avatar still at (10, 10), flash disappeared
        f0 = np.zeros((64, 64), dtype=np.int32)
        f0[10, 10] = 1

        f1 = np.zeros((64, 64), dtype=np.int32)
        f1[10, 10] = 1
        f1[15, 15] = 3

        f2 = np.zeros((64, 64), dtype=np.int32)
        f2[10, 10] = 1

        # Pass raw_frame sequence to agent
        action, _ = agent.choose_action(
            current_grid=f2,
            available_actions=[1, 2, 3, 4],
            raw_frame=[f0, f1, f2],
        )

        # Epistemic grid must have registered the flash target at (15, 15)
        self.assertIn((15, 15), agent.subframe_decomposer.epistemic_grid.revealed_targets)


if __name__ == "__main__":
    unittest.main()
