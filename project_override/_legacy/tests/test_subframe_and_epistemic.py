"""Unit tests for SubFrameDecomposer, TransientFlashExtractor, and LatentEpistemicGrid."""

import unittest
import numpy as np

from project_override.core.subframe_perception import (
    SubFrameDecomposer,
    LatentEpistemicGrid,
    TransientFlashEvent,
    SubFrameAnalysis,
)


class TestSubFramePerception(unittest.TestCase):

    def setUp(self):
        self.decomposer = SubFrameDecomposer(border_margin=4)

    def test_single_frame_passthrough(self):
        frame = np.zeros((64, 64), dtype=np.int32)
        frame[10, 10] = 3
        res = self.decomposer.analyze(frame)
        self.assertFalse(res.has_animation)
        self.assertEqual(res.num_subframes, 1)
        self.assertEqual(res.settled_frame[10, 10], 3)
        self.assertEqual(len(res.flash_events), 0)

    def test_transient_flash_detection(self):
        # 3-frame sequence:
        # F0: black background (0), avatar at (10, 10)
        # F1: beacon flash at (20, 25), (20, 26) with color 2
        # F2: settled: avatar moved to (10, 11), beacon disappeared back to 0
        f0 = np.zeros((64, 64), dtype=np.int32)
        f0[10, 10] = 1

        f1 = np.zeros((64, 64), dtype=np.int32)
        f1[10, 11] = 1
        f1[20, 25] = 2
        f1[20, 26] = 2

        f2 = np.zeros((64, 64), dtype=np.int32)
        f2[10, 11] = 1

        seq = [f0, f1, f2]
        res = self.decomposer.analyze(seq, bg_color=0, current_step=1)

        self.assertTrue(res.has_animation)
        self.assertEqual(res.num_subframes, 3)
        self.assertEqual(len(res.flash_events), 1)
        self.assertIn((20, 25), res.flashed_coords)
        self.assertIn((20, 26), res.flashed_coords)
        self.assertEqual(res.flashed_color_map[(20, 25)], 2)
        self.assertEqual(res.flashed_color_map[(20, 26)], 2)

        # Verify epistemic grid memory
        unvisited = self.decomposer.epistemic_grid.get_unvisited_targets()
        self.assertIn((20, 25), unvisited)
        self.assertIn((20, 26), unvisited)

        # Mark one as visited
        self.decomposer.epistemic_grid.mark_visited(20, 25)
        rem = self.decomposer.epistemic_grid.get_unvisited_targets()
        self.assertNotIn((20, 25), rem)
        self.assertIn((20, 26), rem)

    def test_3d_numpy_parsing(self):
        # Shape (2, 64, 64)
        arr = np.zeros((2, 64, 64), dtype=np.int32)
        arr[0, 5, 5] = 7
        arr[1, 5, 6] = 7
        res = self.decomposer.analyze(arr)
        self.assertTrue(res.has_animation)
        self.assertEqual(res.num_subframes, 2)
        self.assertEqual(res.settled_frame[5, 6], 7)


if __name__ == "__main__":
    unittest.main()
