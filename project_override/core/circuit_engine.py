from typing import List, Optional, Tuple
import numpy as np


class AC3CircuitSolver:
    """Bezel-guarded Periodic Lattice & AC-3 Circuit Tile Rotation Detector."""

    @staticmethod
    def detect_periodic_circuit_clicks(
        grid: np.ndarray, bg_color: int, bezel_px: int = 4
    ) -> Optional[List[Tuple[int, int]]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        core = arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px]
        fg = core != bg_color
        if int(np.sum(fg)) < 24:
            return None

        best_k = 0
        for k in (6, 8, 10, 12):
            if core.shape[0] >= 3 * k and core.shape[1] >= 3 * k:
                row_lines = sum(1 for r in range(k, core.shape[0], k) if np.mean(core[r, :] == core[0, :]) > 0.7)
                if row_lines >= 2:
                    best_k = k
                    break
        if best_k == 0:
            return None

        clicks: List[Tuple[int, int]] = []
        for r in range(bezel_px + best_k // 2, h - bezel_px, best_k):
            for c in range(bezel_px + best_k // 2, w - bezel_px, best_k):
                sub = arr[r - 2:r + 3, c - 2:c + 3]
                if sub.size > 0 and int(np.sum(sub != bg_color)) >= 3:
                    clicks.append((int(r), int(c)))
        return clicks if len(clicks) >= 4 else None
