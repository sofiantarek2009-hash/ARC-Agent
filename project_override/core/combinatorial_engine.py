from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from project_override.core.visual_goal_induction import DihedralSymmetrySolver


class GF2ToggleSolver:
    """GF(2) Gauss-Jordan Linear Toggle / Lights-Out Solver with Live Auto-Solve Queueing."""

    def __init__(self):
        self.click_effects: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        self.cell_stride: int = 8

    def reset(self) -> None:
        self.click_effects.clear()

    def record_click_effect(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray, click_xy: Tuple[int, int], cell_stride: int = 8
    ) -> None:
        self.cell_stride = cell_stride
        diff = np.asarray(prev_grid, dtype=np.int8) != np.asarray(curr_grid, dtype=np.int8)
        rs, cs = np.where(diff)
        if len(rs) == 0 or len(rs) > 400:
            return
        toggled_cells = {(int(r // cell_stride), int(c // cell_stride)) for r, c in zip(rs, cs)}
        if toggled_cells:
            self.click_effects[(int(click_xy[0]), int(click_xy[1]))] = toggled_cells

    def try_solve_current_board(self, grid: np.ndarray, bg_color: int) -> Optional[List[Tuple[int, int]]]:
        if len(self.click_effects) < 3:
            return None
        s = self.cell_stride
        arr = np.asarray(grid, dtype=np.int8)
        active_cells: Set[Tuple[int, int]] = set()
        for r in range(4, arr.shape[0] - 4):
            for c in range(4, arr.shape[1] - 4):
                if int(arr[r, c]) != bg_color:
                    active_cells.add((r // s, c // s))
        if not active_cells or len(active_cells) > 36:
            return None

        all_cells = sorted(set().union(*self.click_effects.values()) | active_cells)
        cell_to_idx = {cell: idx for idx, cell in enumerate(all_cells)}
        buttons = list(self.click_effects.keys())
        A = np.zeros((len(all_cells), len(buttons)), dtype=np.int8)
        b = np.zeros(len(all_cells), dtype=np.int8)

        for j, btn in enumerate(buttons):
            for cell in self.click_effects[btn]:
                if cell in cell_to_idx:
                    A[cell_to_idx[cell], j] = 1
        for cell in active_cells:
            if cell in cell_to_idx:
                b[cell_to_idx[cell]] = 1

        sol = self.solve_toggle_matrix(A, b)
        if sol is None:
            return None
        clicks = [buttons[j] for j in range(len(buttons)) if int(sol[j]) == 1]
        return clicks if clicks else None

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
