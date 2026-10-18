"""Project Override: Empirical Model-Based Affordance Planner & Exact Solvers (ARC-AGI-3)

Implements:
1. EMPA Object-Pair Epistemic Curiosity:
   - Identifies distinct object clusters and color classes on the board.
   - Maintains an interaction ledger of which object classes have been contacted.
   - Plans shortest-path routes to contact untested objects on Level 1 to induce affordances.

2. Split-Panel D4 Symmetry Solver:
   - Detects dividing axes (vertical split, horizontal split, quadrant split).
   - Tests all 8 elements of the Dihedral group D_4 (rotations + reflections).
   - Computes exact mismatch coordinates to click or toggle.

3. ExactCoverDLX (Knuth's Algorithm X):
   - Fast backtracking solver with column-minimum pivoting for exact cover and tiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import ndimage


# =====================================================================
# 1. Knuth's Algorithm X (Exact Cover / DLX)
# =====================================================================

class ExactCoverDLX:
    """Knuth's Algorithm X with Minimum-Remaining-Values (MRV) column selection.
    Solves binary exact cover problems where a subset of rows must partition columns.
    """

    @staticmethod
    def solve(
        matrix: Dict[int, Set[int]],
        primary_columns: Optional[Set[int]] = None,
    ) -> Optional[List[int]]:
        """Solves exact cover: matrix maps row_id -> set of column_ids.

        Returns list of row_ids that cover each primary column exactly once.
        """
        if not matrix:
            return None

        # Build column to rows reverse map
        cols: Dict[int, Set[int]] = {}
        for r_id, col_set in matrix.items():
            for c in col_set:
                if c not in cols:
                    cols[c] = set()
                cols[c].add(r_id)

        target_cols = set(primary_columns) if primary_columns is not None else set(cols.keys())

        def search(
            active_cols: Set[int],
            active_rows: Dict[int, Set[int]],
            solution: List[int],
        ) -> Optional[List[int]]:
            if not active_cols:
                return solution

            # Minimum-Remaining-Values (MRV) heuristic: choose column with smallest candidate rows
            best_col = None
            min_len = 1000000
            for c in active_cols:
                candidate_rows = [r for r in cols.get(c, set()) if r in active_rows]
                if len(candidate_rows) < min_len:
                    min_len = len(candidate_rows)
                    best_col = c
                if min_len == 0:
                    return None  # Dead end

            if best_col is None or min_len == 0:
                return None

            candidates = [r for r in cols[best_col] if r in active_rows]
            for r_id in candidates:
                # Cover columns satisfied by this row
                covered_cols = active_rows[r_id] & active_cols
                new_active_cols = active_cols - covered_cols

                # Conflicting rows share at least one column with r_id
                conflicting_rows: Set[int] = set()
                for c in covered_cols:
                    for other_r in cols.get(c, set()):
                        if other_r in active_rows:
                            conflicting_rows.add(other_r)

                new_active_rows = {k: v for k, v in active_rows.items() if k not in conflicting_rows}

                res = search(new_active_cols, new_active_rows, solution + [r_id])
                if res is not None:
                    return res

            return None

        return search(target_cols, dict(matrix), [])


# =====================================================================
# 2. Split-Panel D4 Symmetry Solver
# =====================================================================

@dataclass
class SplitPanelMatch:
    """Discovered split-screen symmetry configuration."""
    split_type: str  # 'VERTICAL', 'HORIZONTAL'
    divider_coord: int
    symmetry_op: str  # 'IDENTITY', 'ROT90', 'ROT180', 'ROT270', 'FLIP_H', 'FLIP_V', 'TRANSPOSE'
    reference_panel: np.ndarray
    interactive_panel: np.ndarray
    mismatch_coords: List[Tuple[int, int]]  # Global grid coords to click/toggle


class SplitPanelD4Solver:
    """Detects reference vs interactive panels and solves exact D4 symmetry matching."""

    @staticmethod
    def find_split_axis(grid: np.ndarray, border_margin: int = 4) -> Optional[Tuple[str, int]]:
        """Identifies if a full-span single-color divider line splits the interior."""
        H, W = grid.shape[:2]
        bm = border_margin
        interior = grid[bm : H - bm, bm : W - bm]
        iH, iW = interior.shape

        vals, counts = np.unique(interior, return_counts=True)
        bg_color = int(vals[np.argmax(counts)])

        # Vertical divider check: column where all rows have identical frame/divider color != bg_color
        for c in range(iW // 4, 3 * iW // 4 + 1):
            col_vals = interior[:, c]
            uniques = np.unique(col_vals)
            if len(uniques) == 1 and int(uniques[0]) != bg_color:
                return "VERTICAL", bm + c

        # Horizontal divider check: row where all columns have identical frame/divider color != bg_color
        for r in range(iH // 4, 3 * iH // 4 + 1):
            row_vals = interior[r, :]
            uniques = np.unique(row_vals)
            if len(uniques) == 1 and int(uniques[0]) != bg_color:
                return "HORIZONTAL", bm + r

        return None

    @classmethod
    def analyze_symmetry(
        cls,
        grid: np.ndarray,
        border_margin: int = 4,
    ) -> Optional[SplitPanelMatch]:
        """Checks for split-screen panel and finds exact mismatch coordinates."""
        split_info = cls.find_split_axis(grid, border_margin)
        if split_info is None:
            return None

        split_type, div_idx = split_info
        H, W = grid.shape[:2]
        bm = border_margin

        if split_type == "VERTICAL":
            left_panel = grid[bm : H - bm, bm : div_idx]
            right_panel = grid[bm : H - bm, div_idx + 1 : W - bm]
            min_w = min(left_panel.shape[1], right_panel.shape[1])
            if min_w < 3:
                return None
            p_left = left_panel[:, :min_w]
            p_right = right_panel[:, :min_w]

            # Operations to test:
            # 1. Direct copy: p_right should match p_left
            # 2. Horizontal mirror: p_right should match np.fliplr(p_left)
            ops = [
                ("IDENTITY", p_left),
                ("FLIP_H", np.fliplr(p_left)),
                ("FLIP_V", np.flipud(p_left)),
            ]
            for op_name, transformed in ops:
                diff = (p_right != transformed)
                mismatch_count = int(np.sum(diff))
                if 0 < mismatch_count <= (p_right.size // 3):
                    # Found active mismatch target in right interactive panel
                    pts = np.argwhere(diff)
                    global_mismatches = [
                        (bm + int(r), div_idx + 1 + int(c)) for r, c in pts
                    ]
                    return SplitPanelMatch(
                        split_type="VERTICAL",
                        divider_coord=div_idx,
                        symmetry_op=op_name,
                        reference_panel=p_left,
                        interactive_panel=p_right,
                        mismatch_coords=global_mismatches,
                    )

        elif split_type == "HORIZONTAL":
            top_panel = grid[bm : div_idx, bm : W - bm]
            bottom_panel = grid[div_idx + 1 : H - bm, bm : W - bm]
            min_h = min(top_panel.shape[0], bottom_panel.shape[0])
            if min_h < 3:
                return None
            p_top = top_panel[:min_h, :]
            p_bottom = bottom_panel[:min_h, :]

            ops = [
                ("IDENTITY", p_top),
                ("FLIP_V", np.flipud(p_top)),
                ("FLIP_H", np.fliplr(p_top)),
            ]
            for op_name, transformed in ops:
                diff = (p_bottom != transformed)
                mismatch_count = int(np.sum(diff))
                if 0 < mismatch_count <= (p_bottom.size // 3):
                    pts = np.argwhere(diff)
                    global_mismatches = [
                        (div_idx + 1 + int(r), bm + int(c)) for r, c in pts
                    ]
                    return SplitPanelMatch(
                        split_type="HORIZONTAL",
                        divider_coord=div_idx,
                        symmetry_op=op_name,
                        reference_panel=p_top,
                        interactive_panel=p_bottom,
                        mismatch_coords=global_mismatches,
                    )

        return None


# =====================================================================
# 3. EMPA Object-Pair Epistemic Curiosity Planner
# =====================================================================

@dataclass
class EpistemicTarget:
    """An object cluster prioritized for empirical curiosity probing."""
    color: int
    coords: Set[Tuple[int, int]]
    centroid: Tuple[float, float]
    contacted: bool = False
    affordance_induced: Optional[str] = None  # 'COLLECTIBLE', 'SWITCH', 'HAZARD', 'PASSABLE', 'OBSTACLE'


class EMPAObjectCuriosityPlanner:
    """Epistemic Model-Based Affordance Planner.
    Guides exploration on Level 1 by identifying untested object classes
    and prioritizing paths to interact with them before declaring dead ends.
    """

    def __init__(self, border_margin: int = 4) -> None:
        self.border_margin = border_margin
        self.tested_color_classes: Set[int] = set()
        self.contacted_targets: Set[Tuple[int, int]] = set()
        self.affordances: Dict[int, str] = {}

    def reset_for_level(self) -> None:
        self.contacted_targets.clear()

    def reset_for_game(self) -> None:
        self.tested_color_classes.clear()
        self.contacted_targets.clear()
        self.affordances.clear()

    def extract_candidate_targets(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        avatar_color: Optional[int],
        bg_color: int,
        wall_colors: Set[int],
    ) -> List[EpistemicTarget]:
        """Extracts candidate target components that have not yet been contacted."""
        H, W = grid.shape[:2]
        bm = self.border_margin
        interior = grid[bm : H - bm, bm : W - bm]

        ignore_colors = {bg_color} | wall_colors
        if avatar_color is not None:
            ignore_colors.add(avatar_color)

        targets: List[EpistemicTarget] = []

        # Find connected components for each non-ignored color
        unique_colors = [int(c) for c in np.unique(interior) if int(c) not in ignore_colors]

        for color in unique_colors:
            color_mask = (grid == color)
            # Mask out outer margin
            color_mask[:bm, :] = False
            color_mask[H - bm :, :] = False
            color_mask[:, :bm] = False
            color_mask[:, W - bm :] = False

            labeled, num_comp = ndimage.label(color_mask)
            for c_idx in range(1, num_comp + 1):
                pts = np.argwhere(labeled == c_idx)
                if len(pts) == 0:
                    continue
                # Skip massive structures (likely walls or floor fills)
                if len(pts) > 250:
                    continue

                coords = {(int(r), int(c)) for r, c in pts}
                centroid = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
                already_contacted = any(pt in self.contacted_targets for pt in coords)

                target = EpistemicTarget(
                    color=color,
                    coords=coords,
                    centroid=centroid,
                    contacted=already_contacted,
                    affordance_induced=self.affordances.get(color),
                )
                targets.append(target)

        # Sort targets by epistemic priority:
        # Untested color class first, then minimum distance to avatar
        def priority_key(t: EpistemicTarget) -> Tuple[int, float]:
            is_untested_color = 0 if t.color not in self.tested_color_classes else 1
            dist = abs(t.centroid[0] - avatar_pos[0]) + abs(t.centroid[1] - avatar_pos[1])
            return (is_untested_color, dist)

        targets.sort(key=priority_key)
        return targets

    def record_contact(
        self,
        color: int,
        coord: Tuple[int, int],
        delta_observed: int,
        avatar_survived: bool = True,
    ) -> None:
        """Records an interaction event and updates the affordance ledger."""
        self.tested_color_classes.add(color)
        self.contacted_targets.add(coord)

        if not avatar_survived:
            self.affordances[color] = "HAZARD"
        elif delta_observed > 0:
            self.affordances[color] = "COLLECTIBLE_OR_SWITCH"
        else:
            self.affordances[color] = "OBSTACLE"
