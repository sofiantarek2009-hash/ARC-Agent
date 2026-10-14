"""Project Override: Symbolic Puzzle and Click/Toggle Solver Engine
Handles ACTION6 coordinate click targeting, state-delta classification,
topological entropy minimization, and GF(2) linear toggle solvers for ARC-AGI-3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import ndimage


@dataclass
class ClickCandidate:
    x: int  # Column coordinate (0-63)
    y: int  # Row coordinate (0-63)
    color: int
    mass: int
    priority: float
    description: str


@dataclass
class TransitionDelta:
    click_coords: Tuple[int, int]
    changed_pixel_count: int
    affected_bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    is_null: bool
    is_toggle: bool
    is_shift: bool
    entropy_delta: float


class SymbolicPuzzleSolver:
    """Symbolic reasoning and coordinate targeting solver for ARC-AGI-3."""

    def __init__(self) -> None:
        self.dead_clicks: Set[Tuple[int, int]] = set()
        self.history_deltas: List[TransitionDelta] = []
        self.toggle_matrix: Optional[np.ndarray] = None
        self.tile_memory: Dict[Tuple[int, int], int] = {}
        self.cleared_tiles: Set[Tuple[int, int]] = set()
        self.tile_size: Optional[int] = None
        self.grid_ox: int = 0
        self.grid_oy: int = 0
        self.centers: List[Tuple[int, int]] = []
        self.open_cards: List[Tuple[Tuple[int, int], int]] = []
        self.wait_ticks: int = 0

    def clear(self) -> None:
        self.dead_clicks.clear()
        self.history_deltas.clear()
        self.toggle_matrix = None
        self.tile_memory.clear()
        self.cleared_tiles.clear()
        self.tile_size = None
        self.grid_ox = 0
        self.grid_oy = 0
        self.centers.clear()
        self.open_cards.clear()
        self.wait_ticks = 0

    def clear_level(self) -> None:
        self.dead_clicks.clear()
        self.tile_memory.clear()
        self.cleared_tiles.clear()
        self.tile_size = None
        self.grid_ox = 0
        self.grid_oy = 0
        self.centers.clear()
        self.open_cards.clear()
        self.wait_ticks = 0

    def deduce_centers(self, card_area_size: int, min_r: int, min_c: int, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Dynamically deduces all card centers using square root of card area and grid purity."""
        ts = int(round(math.sqrt(max(16, card_area_size))))
        self.tile_size = ts
        canvas_bg = int(grid[0, 0])

        best_centers: List[Tuple[int, int]] = []
        best_ox, best_oy = 0, 0
        max_cards = 0

        for rows in range(1, 8):
            for cols in range(1, 9):
                gw = cols * ts
                gh = rows * ts
                if gw <= 58 and gh <= 58:
                    ox = (64 - gw) // 2
                    oy = (64 - gh) // 2
                    c_idx = round((min_c - ox) / ts)
                    r_idx = round((min_r - oy) / ts)
                    if 0 <= c_idx < cols and 0 <= r_idx < rows:
                        if abs(ox + c_idx * ts - min_c) <= 2 and abs(oy + r_idx * ts - min_r) <= 2:
                            # 100% purity check: no candidate center can be canvas background
                            is_pure = True
                            for r in range(rows):
                                for c in range(cols):
                                    cy_c = oy + r * ts + ts // 2
                                    cx_c = ox + c * ts + ts // 2
                                    if 0 <= cy_c < grid.shape[0] and 0 <= cx_c < grid.shape[1]:
                                        if grid[cy_c, cx_c] == canvas_bg:
                                            is_pure = False
                                            break
                                if not is_pure:
                                    break
                            if is_pure and rows * cols > max_cards:
                                max_cards = rows * cols
                                centers = []
                                for r in range(rows):
                                    for c in range(cols):
                                        cx_c = ox + c * ts + ts // 2
                                        cy_c = oy + r * ts + ts // 2
                                        centers.append((cx_c, cy_c))
                                best_centers = centers
                                best_ox, best_oy = ox, oy

        self.centers = best_centers
        self.grid_ox = best_ox
        self.grid_oy = best_oy
        return self.centers

    def get_canonical(self, x: int, y: int) -> Tuple[int, int]:
        """Maps any coordinate within a tile to the tile's canonical center coordinate."""
        if self.tile_size is None or not self.centers:
            return (x, y)
        ts = self.tile_size
        c = max(0, (x - self.grid_ox) // ts)
        r = max(0, (y - self.grid_oy) // ts)
        cx = self.grid_ox + c * ts + ts // 2
        cy = self.grid_oy + r * ts + ts // 2
        return (int(cx), int(cy))

    def get_tile_centers(self, grid: np.ndarray, game_id: str = "", level_num: int = 0) -> List[Tuple[int, int]]:
        """Extracts discrete tile centers in memory / grid games."""
        if self.centers:
            return list(self.centers)

        # Fallback to scanning for contiguous unrevealed cards (often color 3 or 2)
        for tile_color in [3, 2]:
            rows, cols = np.where(grid == tile_color)
            mask = (rows >= 4) & (rows <= 60) & (cols >= 4) & (cols <= 60)
            r_tile, c_tile = rows[mask], cols[mask]
            if len(r_tile) > 0:
                min_r, max_r = int(r_tile.min()), int(r_tile.max())
                min_c, max_c = int(c_tile.min()), int(c_tile.max())
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                for k in range(2, 8):
                    ts = 56 // k
                    if ts > 0:
                        r_cnt = round(h / ts)
                        c_cnt = round(w / ts)
                        if abs(r_cnt * ts - h) <= 3 and abs(c_cnt * ts - w) <= 3 and r_cnt >= 1 and c_cnt >= 1:
                            centers = []
                            for r in range(r_cnt):
                                for c in range(c_cnt):
                                    cy = min_r + int((r + 0.5) * ts)
                                    cx = min_c + int((c + 0.5) * ts)
                                    centers.append((cx, cy))
                            self.centers = centers
                            self.tile_size = ts
                            return centers
        return []

    def update_tile_observation(self, last_click: Tuple[int, int], grid: np.ndarray, bg_color: int = 5) -> None:
        """Caches revealed tile color at clicked location, sampling offset to avoid UI click markers."""
        lx, ly = self.get_canonical(last_click[0], last_click[1])
        if 0 <= ly < grid.shape[0] and 0 <= lx < grid.shape[1]:
            # Sample multiple offsets within card boundary to avoid central 3x3 click indicator
            card_c = None
            for dy, dx in [(4, 4), (-4, -4), (4, -4), (-4, 4), (2, 2), (0, 0)]:
                sy = min(grid.shape[0] - 1, max(0, ly + dy))
                sx = min(grid.shape[1] - 1, max(0, lx + dx))
                c = int(grid[sy, sx])
                if c not in [0, 3, bg_color]:
                    card_c = c
                    break
            if card_c is not None:
                self.tile_memory[(lx, ly)] = card_c

    def find_pending_match(self, match_size: int = 2) -> Optional[List[Tuple[int, int]]]:
        """Returns coordinates of a newly formed complete match set if available."""
        color_groups: Dict[int, List[Tuple[int, int]]] = {}
        for pos, color in self.tile_memory.items():
            if pos not in self.cleared_tiles:
                color_groups.setdefault(color, []).append(pos)

        for color, positions in color_groups.items():
            if len(positions) >= match_size:
                match_set = positions[:match_size]
                return match_set
        return None

    @staticmethod
    def compute_topological_entropy(grid: np.ndarray) -> float:
        """Calculates Shannon structural entropy H(G) = H_color + 0.5 * H_spatial.

        Lower entropy indicates higher order, symmetry, and goal state alignment.
        """
        h, w = grid.shape
        # Exclude border timer bar (outer 4 cells)
        interior = grid[4 : h - 4, 4 : w - 4] if (h > 8 and w > 8) else grid

        # 1. Color distribution entropy
        counts = np.bincount(interior.flatten(), minlength=16)[1:]  # Exclude background 0
        total = np.sum(counts)
        if total == 0:
            return 0.0

        p_c = counts[counts > 0] / total
        h_color = float(-np.sum(p_c * np.log2(p_c + 1e-9)))

        # 2. Spatial dispersion entropy
        structure = ndimage.generate_binary_structure(2, 2)
        centroids = []
        for c in np.unique(interior):
            if c == 0:
                continue
            labeled, num_features = ndimage.label(interior == c, structure=structure)
            for i in range(1, num_features + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) > 0:
                    centroids.append(coords.mean(axis=0))

        if len(centroids) <= 1:
            h_spatial = 0.0
        else:
            pts = np.array(centroids)
            center = pts.mean(axis=0)
            dists = np.linalg.norm(pts - center, axis=1)
            sum_d = np.sum(dists)
            if sum_d > 0:
                p_d = dists / sum_d
                h_spatial = float(-np.sum(p_d * np.log2(p_d + 1e-9)))
            else:
                h_spatial = 0.0

        return h_color + 0.5 * h_spatial

    @staticmethod
    def extract_click_candidates(grid: np.ndarray) -> List[ClickCandidate]:
        """Extracts prioritized candidate coordinates for ACTION6."""
        h, w = grid.shape
        candidates: List[ClickCandidate] = []

        # Find unique colors and connected components
        unique_colors, counts = np.unique(grid, return_counts=True)
        # Background is largest component
        bg_color = unique_colors[np.argmax(counts)]

        structure = ndimage.generate_binary_structure(2, 2)

        for color in unique_colors:
            if color == bg_color:
                continue

            color_mask = (grid == color)
            labeled, num_features = ndimage.label(color_mask, structure=structure)

            for feat_idx in range(1, num_features + 1):
                coords = np.argwhere(labeled == feat_idx)
                mass = len(coords)
                if mass == 0:
                    continue

                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)
                cr, cc = coords.mean(axis=0)

                # Prioritize small, distinct, centered components
                center_dist = math.hypot(cr - h / 2.0, cc - w / 2.0)
                # Smaller mass and closer to center = higher priority
                priority = 100.0 / (mass + 1.0) + (50.0 / (center_dist + 1.0))

                candidates.append(ClickCandidate(
                    x=int(round(cc)),
                    y=int(round(cr)),
                    color=int(color),
                    mass=mass,
                    priority=priority,
                    description=f"color_{color}_mass_{mass}_at_({int(cr)},{int(cc)})",
                ))

        # Sort by priority descending
        candidates.sort(key=lambda c: c.priority, reverse=True)
        return candidates

    def analyze_click_transition(
        self,
        prev_grid: np.ndarray,
        next_grid: np.ndarray,
        click_coords: Tuple[int, int],
    ) -> TransitionDelta:
        """Classifies the environmental response to a click action."""
        diff = (prev_grid != next_grid)
        diff_coords = np.argwhere(diff)
        changed_count = len(diff_coords)

        if changed_count == 0:
            self.dead_clicks.add(click_coords)
            delta = TransitionDelta(
                click_coords=click_coords,
                changed_pixel_count=0,
                affected_bbox=(0, 0, 0, 0),
                is_null=True,
                is_toggle=False,
                is_shift=False,
                entropy_delta=0.0,
            )
            self.history_deltas.append(delta)
            return delta

        min_r, min_c = diff_coords.min(axis=0)
        max_r, max_c = diff_coords.max(axis=0)
        bbox = (int(min_r), int(min_c), int(max_r), int(max_c))

        h_prev = self.compute_topological_entropy(prev_grid)
        h_next = self.compute_topological_entropy(next_grid)
        delta_h = h_next - h_prev

        is_toggle = (changed_count <= 9 and (max_r - min_r) <= 3 and (max_c - min_c) <= 3)
        is_shift = (changed_count > 9 and ((max_r - min_r == 0) or (max_c - min_c == 0)))

        delta = TransitionDelta(
            click_coords=click_coords,
            changed_pixel_count=changed_count,
            affected_bbox=bbox,
            is_null=False,
            is_toggle=is_toggle,
            is_shift=is_shift,
            entropy_delta=delta_h,
        )
        self.history_deltas.append(delta)
        return delta

    @staticmethod
    def solve_gf2_linear_system(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Solves A * x = b (mod 2) using Gaussian elimination over GF(2).

        Returns solution vector x (binary numpy array) if solvable, else None.
        """
        n, m = A.shape
        # Augmented matrix [A | b]
        aug = np.column_stack([A % 2, b % 2]).astype(np.int8)

        row = 0
        pivot_cols = []
        for col in range(m):
            if row >= n:
                break
            # Find pivot in this column
            pivot_idx = -1
            for r in range(row, n):
                if aug[r, col] == 1:
                    pivot_idx = r
                    break
            if pivot_idx == -1:
                continue

            # Swap pivot row into place
            if pivot_idx != row:
                aug[[row, pivot_idx]] = aug[[pivot_idx, row]]

            # Eliminate below and above
            for r in range(n):
                if r != row and aug[r, col] == 1:
                    aug[r] ^= aug[row]

            pivot_cols.append(col)
            row += 1

        # Check for inconsistency: row with all zeros in A but 1 in b
        for r in range(row, n):
            if aug[r, m] == 1:
                return None  # Inconsistent

        # Back-substitution
        x = np.zeros(m, dtype=np.int8)
        for r in range(len(pivot_cols)):
            col = pivot_cols[r]
            x[col] = aug[r, m]

        return x
