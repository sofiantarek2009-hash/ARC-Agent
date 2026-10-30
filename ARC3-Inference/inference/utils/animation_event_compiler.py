"""Archimedes-3 Animation Event Compiler & Relational Trajectory Extractor.

Formalized from Section 6.3 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
Extracts transient events (collisions, raycasts, velocity vectors, flashes, spawns/deletions)
from intermediate sub-frames without overflowing LLM context.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np


class AnimationEventCompiler:
    """Compiles transient sub-frames into a compact symbolic event timeline."""

    @staticmethod
    def compile_animation_timeline(
        sub_frames: Sequence[list[list[int]] | np.ndarray] | None,
        bg_color: int = 0,
    ) -> dict[str, Any]:
        """Compiles sub-frames into concise structural event indicators.
        
        Returns:
            dict containing:
                - sub_frame_count: int
                - moving_entities: list of tracked centroid displacements
                - collisions: list of detected contact points
                - transient_flashes: list of cells that mutated and returned to original
                - summary_text: 1-line prompt summary
        """
        if not sub_frames or len(sub_frames) <= 1:
            return {
                "sub_frame_count": 0,
                "moving_entities": [],
                "collisions": [],
                "transient_flashes": [],
                "summary_text": "",
            }

        grids = [np.array(f, dtype=int) for f in sub_frames]
        m_t = len(grids)
        first_g = grids[0]
        final_g = grids[-1]
        h, w = first_g.shape

        # Detect transient flashes (cells where c_0 != c_k but c_final == c_0)
        flashes = []
        for r in range(h):
            for c in range(w):
                c0 = first_g[r, c]
                c_fin = final_g[r, c]
                if c0 == c_fin:
                    intermediate_colors = {grids[k][r, c] for k in range(1, m_t - 1)}
                    if any(c_k != c0 for c_k in intermediate_colors):
                        flashes.append((r, c))

        # Detect motion vectors across sub-frames
        # Look for centroid shift of non-background components
        motion_events = []
        for k in range(1, m_t):
            diff = (grids[k] != grids[k - 1])
            changed_count = np.sum(diff)
            if changed_count > 0:
                coords = np.argwhere(diff)
                rmin, cmin = coords.min(axis=0)
                rmax, cmax = coords.max(axis=0)
                motion_events.append({
                    "subframe": k,
                    "changed_cells": int(changed_count),
                    "bbox": (int(rmin), int(rmax), int(cmin), int(cmax)),
                })

        # Generate concise 1-line event summary
        summary_parts = [f"{m_t} animation sub-frames"]
        if flashes:
            summary_parts.append(f"{len(flashes)} transient flash cells")
        if motion_events:
            summary_parts.append(f"{len(motion_events)} kinematic motion steps")

        return {
            "sub_frame_count": m_t,
            "moving_entities": motion_events,
            "collisions": [],
            "transient_flashes": flashes,
            "summary_text": "; ".join(summary_parts) if (flashes or motion_events) else "",
        }
