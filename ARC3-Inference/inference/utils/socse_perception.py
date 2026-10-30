"""Archimedes-3 Spatiotemporal Object-Centric State Estimator (SOCSE) & Invariant Detector.

100% General, zero hardcoded game IDs, color roles, or coordinate constants.
Formalized from Section 6 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


@dataclass
class EntitySlot:
    """Persistent object entity slot tracked across frames."""
    entity_id: int
    colors: set[int]
    mask: list[tuple[int, int]]  # list of (r, c)
    bbox: tuple[int, int, int, int]  # (rmin, rmax, cmin, cmax)
    centroid: tuple[float, float]
    canonical_hash: str
    is_hud_counter: bool = False
    velocity: tuple[int, int] = (0, 0)


class SOCSEPerceptionEngine:
    """Spatiotemporal Object-Centric State Estimator.
    
    Decomposes 64x64 grids into invariant background, monotonic HUD bars,
    and distinct cohesive entity slots with amodal spatiotemporal tracking.
    """

    def __init__(self) -> None:
        self.next_entity_id: int = 1
        self.active_slots: dict[int, EntitySlot] = {}
        self.last_grid: np.ndarray | None = None
        self.monotonic_strips_detected: set[tuple[str, int]] = set()

    def infer_background_color(self, grid: np.ndarray) -> int:
        """Infers background color via frame perimeter mode and area support."""
        # Top, bottom, left, right perimeter slices
        border_pixels = np.concatenate([
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        ])
        values, counts = np.unique(border_pixels, return_counts=True)
        return int(values[np.argmax(counts)])

    def detect_monotonic_counters(self, history_grids: Sequence[np.ndarray]) -> set[tuple[str, int]]:
        """Identifies 1D linear strips (rows or cols) that mutate monotonically across turns.
        
        Section 6.2 Tier 3: HUD and Monotonic Counter Invariant Separation.
        Returns set of ('row'|'col', index).
        """
        if len(history_grids) < 3:
            return self.monotonic_strips_detected

        h, w = history_grids[0].shape
        # Check rows
        for r in range(h):
            non_bg_counts = []
            bg = self.infer_background_color(history_grids[0])
            for g in history_grids:
                non_bg_counts.append(int(np.sum(g[r, :] != bg)))
            diffs = [non_bg_counts[i+1] - non_bg_counts[i] for i in range(len(non_bg_counts)-1)]
            # Monotonic: strictly all non-negative or all non-positive, with at least one non-zero
            if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
                self.monotonic_strips_detected.add(('row', r))
            elif all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
                self.monotonic_strips_detected.add(('row', r))

        # Check cols
        for c in range(w):
            non_bg_counts = []
            bg = self.infer_background_color(history_grids[0])
            for g in history_grids:
                non_bg_counts.append(int(np.sum(g[:, c] != bg)))
            diffs = [non_bg_counts[i+1] - non_bg_counts[i] for i in range(len(non_bg_counts)-1)]
            if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
                self.monotonic_strips_detected.add(('col', c))
            elif all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
                self.monotonic_strips_detected.add(('col', c))

        return self.monotonic_strips_detected

    def extract_connected_components(
        self, grid: np.ndarray, bg_color: int | None = None
    ) -> list[EntitySlot]:
        """Extracts monochrome and cohesive connected components."""
        if bg_color is None:
            bg_color = self.infer_background_color(grid)

        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        slots: list[EntitySlot] = []

        for r in range(h):
            for c in range(w):
                val = int(grid[r, c])
                if val == bg_color or visited[r, c]:
                    continue

                # BFS 4-connected component
                comp_cells: list[tuple[int, int]] = []
                queue = deque([(r, c)])
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    comp_cells.append((curr_r, curr_c))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr, nc] and grid[nr, nc] == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                # Compute attributes
                rmin = min(cell[0] for cell in comp_cells)
                rmax = max(cell[0] for cell in comp_cells)
                cmin = min(cell[1] for cell in comp_cells)
                cmax = max(cell[1] for cell in comp_cells)

                centroid = (
                    sum(cell[0] for cell in comp_cells) / len(comp_cells),
                    sum(cell[1] for cell in comp_cells) / len(comp_cells),
                )

                # Canonical translation-invariant bitmask hash
                rel_coords = sorted([(cr - rmin, cc - cmin) for cr, cc in comp_cells])
                canon_hash = hashlib.sha1(str(rel_coords).encode("utf-8")).hexdigest()[:12]

                # Check if part of monotonic counter
                is_hud = False
                for kind, idx in self.monotonic_strips_detected:
                    if kind == 'row' and rmin <= idx <= rmax and (rmax - rmin == 0):
                        is_hud = True
                        break
                    elif kind == 'col' and cmin <= idx <= cmax and (cmax - cmin == 0):
                        is_hud = True
                        break

                slot = EntitySlot(
                    entity_id=self.next_entity_id,
                    colors={val},
                    mask=comp_cells,
                    bbox=(rmin, rmax, cmin, cmax),
                    centroid=centroid,
                    canonical_hash=canon_hash,
                    is_hud_counter=is_hud,
                )
                self.next_entity_id += 1
                slots.append(slot)

        return slots

    def compute_frame_difference(
        self, before: np.ndarray, after: np.ndarray
    ) -> dict[str, Any]:
        """Calculates precise cell mutation masks, separating HUD changes from interior."""
        diff_mask = (before != after)
        total_mutated = int(np.sum(diff_mask))
        if total_mutated == 0:
            return {"mutated_count": 0, "interior_mutated": 0, "hud_mutated": 0}

        mutated_coords = list(zip(*np.where(diff_mask)))
        hud_coords = []
        interior_coords = []

        for r, c in mutated_coords:
            is_hud = False
            for kind, idx in self.monotonic_strips_detected:
                if (kind == 'row' and r == idx) or (kind == 'col' and c == idx):
                    is_hud = True
                    break
            if is_hud:
                hud_coords.append((r, c))
            else:
                interior_coords.append((r, c))

        return {
            "mutated_count": total_mutated,
            "interior_mutated": len(interior_coords),
            "hud_mutated": len(hud_coords),
            "interior_coords": interior_coords[:20],  # sample for fast inspection
        }


def mask_monotonic_hud(grid: list[list[int]], monotonic_strips: set[tuple[str, int]]) -> list[list[int]]:
    """Masks out detected monotonic HUD cells with a constant sentinel (-1) for canonical state hashing."""
    if not monotonic_strips:
        return grid
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    masked = [list(row) for row in grid]
    for kind, idx in monotonic_strips:
        if kind == 'row' and 0 <= idx < h:
            for c in range(w):
                masked[idx][c] = -1
        elif kind == 'col' and 0 <= idx < w:
            for r in range(h):
                masked[r][idx] = -1
    return masked
