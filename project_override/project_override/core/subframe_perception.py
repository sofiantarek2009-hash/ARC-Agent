"""Project Override: Sub-Frame Animation Decomposer & Transient Flash Extractor (ARC-AGI-3)

Implements:
1. SubFrameDecomposer:
   - Ingests multi-frame animation sequences [F_{t,0}, ..., F_{t,K}] from raw observations.
   - Identifies the settled final frame F_K, intermediate animation frames, and initial frame.
   - Distinguishes instantaneous direct action effects from autonomous environment ticks.

2. TransientFlashExtractor:
   - Computes transient flash masks T_t(r, c) = val for pixels that light up/reveal
     during intermediate animation steps but revert to background/default in the final frame.
   - Extracts flash coordinates, revealed colors, and flash duration for memory/beacon games.

3. LatentEpistemicGrid:
   - Persistent per-level / per-game latent spatial memory.
   - Retains transient flashed targets, hidden identities, and revealed path waypoints
     even after they disappear from raw optical observation.
   - Provides verified target coordinates to planning modules (A*, DLX, Click Solver).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


@dataclass
class TransientFlashEvent:
    """A transient visual flash or reveal detected across sub-frames."""
    frame_index: int
    coords: Set[Tuple[int, int]]
    colors: Dict[Tuple[int, int], int]
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    centroid: Tuple[float, float]
    reverted_to_bg: bool = True


@dataclass
class SubFrameAnalysis:
    """Decomposition result of a multi-frame animation observation."""
    settled_frame: np.ndarray
    initial_frame: np.ndarray
    num_subframes: int
    has_animation: bool
    flash_events: List[TransientFlashEvent] = field(default_factory=list)
    flashed_coords: Set[Tuple[int, int]] = field(default_factory=set)
    flashed_color_map: Dict[Tuple[int, int], int] = field(default_factory=dict)
    motion_trajectory: List[Tuple[int, int]] = field(default_factory=list)
    direct_agent_delta: int = 0
    autonomous_ticks: int = 0


class LatentEpistemicGrid:
    """Persistent spatial memory retaining transiently revealed information across steps."""

    def __init__(self, shape: Tuple[int, int] = (64, 64)) -> None:
        self.shape = shape
        self.revealed_targets: Set[Tuple[int, int]] = set()
        self.revealed_colors: Dict[Tuple[int, int], int] = {}
        self.revealed_sequences: List[Tuple[int, int]] = []
        self.visited_reveals: Set[Tuple[int, int]] = set()
        self.latent_grid: np.ndarray = np.full(shape, -1, dtype=np.int32)
        self.step_discovered: Dict[Tuple[int, int], int] = {}

    @property
    def grid(self) -> np.ndarray:
        return self.latent_grid

    def reset_for_level(self) -> None:
        """Resets dynamic level memory while retaining invariant geometry."""
        self.revealed_targets.clear()
        self.revealed_colors.clear()
        self.revealed_sequences.clear()
        self.visited_reveals.clear()
        self.latent_grid.fill(-1)
        self.step_discovered.clear()

    def register_flash(
        self,
        coords: Set[Tuple[int, int]],
        color_map: Dict[Tuple[int, int], int],
        step: int = 0,
    ) -> None:
        """Stores newly flashed/revealed coordinates into latent memory."""
        for r, c in coords:
            if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
                self.revealed_targets.add((r, c))
                color = color_map.get((r, c), 0)
                self.revealed_colors[(r, c)] = color
                self.latent_grid[r, c] = color
                if (r, c) not in self.revealed_sequences:
                    self.revealed_sequences.append((r, c))
                self.step_discovered[(r, c)] = step

    def mark_visited(self, r: int, c: int) -> None:
        """Marks a revealed target as fulfilled/visited."""
        self.visited_reveals.add((r, c))

    def get_unvisited_targets(self) -> List[Tuple[int, int]]:
        """Returns revealed targets in discovery order that have not yet been visited."""
        return [pt for pt in self.revealed_sequences if pt not in self.visited_reveals]

    @property
    def has_active_targets(self) -> bool:
        return len(self.get_unvisited_targets()) > 0


class SubFrameDecomposer:
    """Processes raw observation frames and extracts sub-frame physics and flashes."""

    def __init__(self, border_margin: int = 4) -> None:
        self.border_margin = border_margin
        self.epistemic_grid = LatentEpistemicGrid()

    @staticmethod
    def parse_raw_frame(raw_frame: Any) -> List[np.ndarray]:
        """Normalizes any raw frame input into a list of 2D integer numpy arrays [F_0, ..., F_K]."""
        if raw_frame is None:
            return [np.zeros((64, 64), dtype=np.int32)]

        if isinstance(raw_frame, np.ndarray):
            if raw_frame.ndim == 3:
                # Shape (K, H, W)
                return [np.asarray(raw_frame[i], dtype=np.int32) for i in range(raw_frame.shape[0])]
            elif raw_frame.ndim == 2:
                return [np.asarray(raw_frame, dtype=np.int32)]
            else:
                return [np.zeros((64, 64), dtype=np.int32)]

        if isinstance(raw_frame, list):
            if len(raw_frame) == 0:
                return [np.zeros((64, 64), dtype=np.int32)]
            first = raw_frame[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
                # Could be a single 2D grid as list of lists
                # Check if first element is a list of integers
                if isinstance(first[0][0], (int, np.integer)):
                    # Single 2D frame: list of rows
                    return [np.array(raw_frame, dtype=np.int32)]
                elif isinstance(first[0][0], list):
                    # List of 2D frames
                    return [np.array(f, dtype=np.int32) for f in raw_frame]
            elif isinstance(first, np.ndarray):
                return [np.asarray(f, dtype=np.int32) for f in raw_frame]
            else:
                # Fallback: convert directly
                arr = np.array(raw_frame, dtype=np.int32)
                if arr.ndim == 3:
                    return [arr[i] for i in range(arr.shape[0])]
                elif arr.ndim == 2:
                    return [arr]

        arr = np.array(raw_frame, dtype=np.int32)
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 2:
            return [arr]
        return [np.zeros((64, 64), dtype=np.int32)]

    def analyze(
        self,
        raw_frame: Any,
        bg_color: Optional[int] = None,
        current_step: int = 0,
    ) -> SubFrameAnalysis:
        """Decomposes a frame sequence, extracts transient flashes, and updates epistemic memory."""
        frames = self.parse_raw_frame(raw_frame)
        num_frames = len(frames)
        settled = frames[-1]
        initial = frames[0]

        if num_frames <= 1:
            return SubFrameAnalysis(
                settled_frame=settled,
                initial_frame=initial,
                num_subframes=1,
                has_animation=False,
                flash_events=[],
                flashed_coords=set(),
                flashed_color_map={},
                motion_trajectory=[],
                direct_agent_delta=0,
                autonomous_ticks=0,
            )

        # Multi-frame animation detected!
        H, W = settled.shape[:2]
        bm = self.border_margin
        interior_slice = (slice(bm, H - bm), slice(bm, W - bm))

        # Infer background if not provided
        if bg_color is None:
            interior_settled = settled[interior_slice]
            vals, counts = np.unique(interior_settled, return_counts=True)
            bg_color = int(vals[np.argmax(counts)])

        flash_events: List[TransientFlashEvent] = []
        all_flashed_coords: Set[Tuple[int, int]] = set()
        all_flashed_colors: Dict[Tuple[int, int], int] = {}

        # Scan intermediate sub-frames 1 .. K-1 against settled frame F_K
        start_idx = 1 if num_frames >= 3 else 0
        for idx in range(start_idx, num_frames - 1):
            f_k = frames[idx]
            # Difference in interior between sub-frame f_k and settled frame
            diff_mask = (f_k != settled)
            # Exclude outer border
            diff_mask[:bm, :] = False
            diff_mask[H - bm :, :] = False
            diff_mask[:, :bm] = False
            diff_mask[:, W - bm :] = False

            if not np.any(diff_mask):
                continue

            diff_pts = np.argwhere(diff_mask)
            event_coords: Set[Tuple[int, int]] = set()
            event_colors: Dict[Tuple[int, int], int] = {}

            for r_idx, c_idx in diff_pts:
                r, c = int(r_idx), int(c_idx)
                c_val = int(f_k[r, c])
                settled_val = int(settled[r, c])

                # Check if this pixel showed non-background in f_k and reverted to bg in settled
                if c_val != bg_color and settled_val == bg_color:
                    event_coords.add((r, c))
                    event_colors[(r, c)] = c_val
                    all_flashed_coords.add((r, c))
                    all_flashed_colors[(r, c)] = c_val
                elif c_val != settled_val and c_val != bg_color:
                    # Revealed a non-background feature that changed or vanished
                    event_coords.add((r, c))
                    event_colors[(r, c)] = c_val
                    all_flashed_coords.add((r, c))
                    all_flashed_colors[(r, c)] = c_val

            if event_coords:
                rs = [p[0] for p in event_coords]
                cs = [p[1] for p in event_coords]
                bbox = (min(rs), min(cs), max(rs), max(cs))
                centroid = (float(np.mean(rs)), float(np.mean(cs)))
                flash_events.append(TransientFlashEvent(
                    frame_index=idx,
                    coords=event_coords,
                    colors=event_colors,
                    bounding_box=bbox,
                    centroid=centroid,
                    reverted_to_bg=True,
                ))

        # Register flash events with persistent epistemic memory
        if all_flashed_coords:
            self.epistemic_grid.register_flash(
                coords=all_flashed_coords,
                color_map=all_flashed_colors,
                step=current_step,
            )

        # Compute trajectory of moving entities across sub-frames
        trajectory: List[Tuple[int, int]] = []
        for f in frames:
            # Find center of mass of non-bg differences from initial
            diff = (f != initial)
            diff[:bm, :] = False
            diff[H - bm :, :] = False
            diff[:, :bm] = False
            diff[:, W - bm :] = False
            pts = np.argwhere(diff)
            if len(pts) > 0:
                trajectory.append((int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))))

        direct_delta = int(np.sum(frames[0][interior_slice] != frames[1][interior_slice])) if num_frames > 1 else 0
        autonomous_ticks = max(0, num_frames - 2)

        return SubFrameAnalysis(
            settled_frame=settled,
            initial_frame=initial,
            num_subframes=num_frames,
            has_animation=True,
            flash_events=flash_events,
            flashed_coords=all_flashed_coords,
            flashed_color_map=all_flashed_colors,
            motion_trajectory=trajectory,
            direct_agent_delta=direct_delta,
            autonomous_ticks=autonomous_ticks,
        )
