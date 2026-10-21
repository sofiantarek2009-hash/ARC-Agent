from typing import Dict, List, Optional, Tuple
import numpy as np
from project_override.core.visual_goal_induction import DihedralSymmetrySolver


class GF2ToggleSolver:
    """GF(2) Gauss-Jordan Linear Toggle / Lights-Out Solver with Minimum Hamming Weight."""

    @staticmethod
    def solve_toggle_matrix(
        effect_matrix: np.ndarray, target_vector: np.ndarray
    ) -> Optional[np.ndarray]:
        A = (np.asarray(effect_matrix, dtype=np.int8) & 1).copy()
        b = (np.asarray(target_vector, dtype=np.int8) & 1).copy()
        n_eq, n_vars = A.shape
        aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)

        pivot_row = 0
        pivot_cols = []
        for col in range(n_vars):
            if pivot_row >= n_eq:
                break
            cand = np.where(aug[pivot_row:, col] == 1)[0]
            if len(cand) == 0:
                continue
            swap_r = pivot_row + int(cand[0])
            if swap_r != pivot_row:
                aug[[pivot_row, swap_r]] = aug[[swap_r, pivot_row]]
            for r in range(n_eq):
                if r != pivot_row and aug[r, col] == 1:
                    aug[r] ^= aug[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1

        for r in range(pivot_row, n_eq):
            if aug[r, -1] == 1:
                return None

        free_cols = [c for c in range(n_vars) if c not in pivot_cols]
        x_part = np.zeros(n_vars, dtype=np.int8)
        for idx, p_col in enumerate(pivot_cols):
            x_part[p_col] = aug[idx, -1]

        if not free_cols or len(free_cols) > 10:
            return x_part

        null_basis = []
        for f_col in free_cols:
            v = np.zeros(n_vars, dtype=np.int8)
            v[f_col] = 1
            for idx, p_col in enumerate(pivot_cols):
                v[p_col] = aug[idx, f_col]
            null_basis.append(v)

        best_x = x_part
        best_weight = int(np.sum(best_x))
        for mask in range(1, 1 << len(free_cols)):
            cand_x = x_part.copy()
            for bit_idx, basis_vec in enumerate(null_basis):
                if (mask >> bit_idx) & 1:
                    cand_x ^= basis_vec
            w = int(np.sum(cand_x))
            if w < best_weight:
                best_weight = w
                best_x = cand_x
        return best_x


class SplitPanelSymmetrySolver(DihedralSymmetrySolver):
    @classmethod
    def find_asymmetric_clicks(cls, grid: np.ndarray, bg_color: int, bezel_px: int = 4) -> List[Tuple[int, int]]:
        return cls.find_symmetry_clicks(grid, bg_color=bg_color, bezel_px=bezel_px)
