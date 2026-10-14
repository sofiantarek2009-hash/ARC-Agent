"""Project Override: Visual Goal & Invariant Induction Engine (ARC-AGI-3)

Strictly general, color-agnostic, zero-hardcoding mathematical induction of:
1. Split-Screen / Reference Panel & Reflection Symmetry Invariants
2. Topological Enclosure (Jordan Curve & Exterior Flood-Fill) Invariants
3. HUD Sequence / Highlighted Token Indicator Invariants
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class DiscrepancyCell:
    """Represents a single pixel or logical cell discrepancy in the playfield."""
    global_row: int
    global_col: int
    local_row: int
    local_col: int
    current_color: int
    target_color: int
    discrepancy_type: str  # "MISSING_FOREGROUND", "EXCESS_FOREGROUND", "COLOR_MISMATCH"
    patch_radius: int = 0


@dataclass
class SplitScreenInvariant:
    """Induced split-screen or reflection symmetry invariant."""
    axis: str  # "VERTICAL" (left/right panels) or "HORIZONTAL" (top/bottom panels)
    separator_range: Tuple[int, int]  # Inclusive (start_idx, end_idx) in global coords
    separator_color: Optional[int]  # None if implicit midpoint axis
    transform_type: str  # "TRANSLATION_COPY" or "REFLECTION_MIRROR"
    reference_side: str  # "LEFT", "RIGHT", "TOP", "BOTTOM", or "BOTH_SYMMETRIC"
    playfield_side: str  # "LEFT", "RIGHT", "TOP", "BOTTOM", or "BOTH_SYMMETRIC"
    reference_bbox: Tuple[int, int, int, int]  # (r0, c0, r1_excl, c1_excl)
    playfield_bbox: Tuple[int, int, int, int]  # (r0, c0, r1_excl, c1_excl)
    scale_ratio: Tuple[int, int]  # (play_scale, ref_scale)
    color_mapping: Dict[int, int]  # Maps reference color -> target playfield color
    alignment_score: float  # Structural IoU / match ratio in [0.0, 1.0]
    target_playfield_grid: np.ndarray  # Reconstructed goal state for playfield_bbox
    discrepancy_mask: np.ndarray  # Boolean mask over playfield_bbox where state != goal
    discrepancies: List[DiscrepancyCell] = field(default_factory=list)

    @property
    def total_discrepancy_count(self) -> int:
        return len(self.discrepancies)

    @property
    def is_satisfied(self) -> bool:
        return len(self.discrepancies) == 0


@dataclass
class TopologicalCavity:
    """Represents an interior cavity enclosed by a closed Jordan curve on the grid."""
    cavity_id: int
    enclosing_color: Optional[int]  # Specific loop color, or None if multi-color loop
    is_4_watertight: bool  # True if sealed against 8-way flood (4-connected loop)
    is_8_closed: bool  # True if sealed against 4-way flood (8-connected loop)
    bbox: Tuple[int, int, int, int]  # Global (min_r, min_c, max_r_incl, max_c_incl)
    centroid: Tuple[float, float]  # Global (row, col)
    total_area: int
    uncolored_cells: List[Tuple[int, int]]  # Global (r, c) where grid == bg_color
    colored_cells: List[Tuple[int, int]]  # Global (r, c) where grid != bg_color
    enclosed_colors: Dict[int, int]  # Color -> count inside cavity
    boundary_loop_coords: List[Tuple[int, int]]  # Global (r, c) of the Jordan curve
    induced_fill_color: Optional[int]  # Target fill color if partially seeded or loop-colored


@dataclass
class OpenLoopCandidate:
    """Represents an almost-closed Jordan curve that can be sealed by filling gap cells."""
    loop_color: int
    gap_cells: List[Tuple[int, int]]  # Global (r, c) coordinates that seal the loop
    potential_cavity_area: int
    potential_cavity_centroid: Tuple[float, float]


@dataclass
class TopologicalEnclosureInvariant:
    """Induced topological enclosure invariant over the playfield."""
    background_color: int
    exterior_reachable_mask: np.ndarray  # Full-frame boolean mask of exterior background
    cavity_mask: np.ndarray  # Full-frame boolean mask of all enclosed interior cavities
    cavities: List[TopologicalCavity] = field(default_factory=list)
    open_loop_candidates: List[OpenLoopCandidate] = field(default_factory=list)

    @property
    def uncolored_cavity_cell_count(self) -> int:
        return sum(len(c.uncolored_cells) for c in self.cavities)


@dataclass
class HUDToken:
    """A single token inside a 1D HUD sequence strip."""
    index: int
    color: int
    center: Tuple[int, int]  # Global (row, col)
    bbox: Tuple[int, int, int, int]  # Global (min_r, min_c, max_r_incl, max_c_incl)
    mass: int
    is_highlighted: bool
    frame_color: Optional[int] = None
    frame_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class PlayfieldMatchObject:
    """An object in the interactive playfield matching the active HUD token color."""
    component_id: int
    color: int
    centroid: Tuple[int, int]  # Global (row, col) rounded to int
    exact_centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # Global (min_r, min_c, max_r_incl, max_c_incl)
    mass: int
    coords: List[Tuple[int, int]]
    distance_to_avatar: Optional[int] = None


@dataclass
class HUDSequenceInvariant:
    """Induced 1D HUD token sequence and active playfield target invariant."""
    strip_location: str  # "TOP", "BOTTOM", "LEFT", "RIGHT"
    strip_slice: Tuple[int, int, int, int]  # (r0, c0, r1_excl, c1_excl)
    strip_background_color: int
    frame_color: int
    tokens: List[HUDToken]
    active_token_index: int
    active_token_color: int
    matched_playfield_objects: List[PlayfieldMatchObject] = field(default_factory=list)


# =============================================================================
# Core Utilities: Viewport & Background Induction
# =============================================================================

def induce_playfield_viewport(
    grid: np.ndarray,
    default_border: int = 4,
) -> Tuple[Tuple[int, int, int, int], int]:
    """Dynamically induces the active interior playfield bounding box (r0, c0, r1, c1)
    and its modal background color without hardcoded color constants.
    """
    h, w = grid.shape
    b = min(default_border, h // 8, w // 8)

    interior = grid[b : h - b, b : w - b] if (h > 2 * b and w > 2 * b) else grid
    vals, cnts = np.unique(interior, return_counts=True)
    bg_color = int(vals[np.argmax(cnts)])

    r0, c0, r1, c1 = b, b, h - b, w - b
    corner_color = int(grid[r0, c0])
    if (
        corner_color != bg_color
        and np.all(grid[r0, c0:c1] == corner_color)
        and np.all(grid[r1 - 1, c0:c1] == corner_color)
    ):
        non_pad_rows = np.where(np.any(grid[b : h - b, b : w - b] != corner_color, axis=1))[0]
        non_pad_cols = np.where(np.any(grid[b : h - b, b : w - b] != corner_color, axis=0))[0]
        if len(non_pad_rows) >= 4 and len(non_pad_cols) >= 4:
            r0 = b + int(non_pad_rows[0])
            r1 = b + int(non_pad_rows[-1]) + 1
            c0 = b + int(non_pad_cols[0])
            c1 = b + int(non_pad_cols[-1]) + 1
            sub = grid[r0:r1, c0:c1]
            s_vals, s_cnts = np.unique(sub, return_counts=True)
            bg_color = int(s_vals[np.argmax(s_cnts)])

    return (r0, c0, r1, c1), bg_color


def _infer_fundamental_scale(panel: np.ndarray, bg_color: int) -> int:
    """Induces the discrete pixel block scale s >= 1 of a grid panel using run-length quantiles."""
    mask = (panel != bg_color)
    if not np.any(mask):
        return 1

    run_lengths: List[int] = []
    for axis in (0, 1):
        padded = np.pad(mask.astype(np.int8), 1, mode="constant", constant_values=0)
        diffs = np.diff(padded, axis=axis)
        starts = np.argwhere(diffs == 1)
        ends = np.argwhere(diffs == -1)
        if len(starts) == len(ends) and len(starts) > 0:
            lengths = ends[:, axis] - starts[:, axis]
            run_lengths.extend(int(x) for x in lengths if x >= 1)

    if not run_lengths:
        return 1

    min_run = int(np.percentile(run_lengths, 10))
    if min_run <= 1:
        return 1

    for cand_s in range(min_run, 1, -1):
        divisible_ratio = sum(1 for L in run_lengths if L % cand_s == 0) / len(run_lengths)
        if divisible_ratio >= 0.80:
            return cand_s
    return 1


def _downsample_panel(panel: np.ndarray, scale: int) -> np.ndarray:
    """Downsamples a 2D panel by integer factor `scale` using modal block pooling."""
    if scale <= 1:
        return panel.copy()
    h, w = panel.shape
    gh, gw = h // scale, w // scale
    if gh == 0 or gw == 0:
        return panel.copy()
    out = np.zeros((gh, gw), dtype=panel.dtype)
    for r in range(gh):
        for c in range(gw):
            patch = panel[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            vals, cnts = np.unique(patch, return_counts=True)
            out[r, c] = vals[np.argmax(cnts)]
    return out


# =============================================================================
# 1. Split-Screen / Reference Panel & Symmetry Invariant Detector
# =============================================================================

class SplitScreenDetector:
    """Detects full-span vertical/horizontal separator lines and reflection symmetry axes,
    identifies static reference vs. interactive playfield, and computes exact cell discrepancies.
    """

    def __init__(
        self,
        min_panel_span: int = 6,
        max_separator_thickness: int = 4,
        span_purity_threshold: float = 0.90,
        min_alignment_score: float = 0.25,
    ) -> None:
        self.min_panel_span = min_panel_span
        self.max_separator_thickness = max_separator_thickness
        self.span_purity_threshold = span_purity_threshold
        self.min_alignment_score = min_alignment_score

    def detect(
        self,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray] = None,
        avatar_color: Optional[int] = None,
        viewport: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[SplitScreenInvariant]:
        """Induces a SplitScreenInvariant if the frame is partitioned into reference/playfield
        or symmetric halves.
        """
        if viewport is None:
            viewport, bg_color = induce_playfield_viewport(grid)
        else:
            r0, c0, r1, c1 = viewport
            vals, cnts = np.unique(grid[r0:r1, c0:c1], return_counts=True)
            bg_color = int(vals[np.argmax(cnts)])

        r0, c0, r1, c1 = viewport
        sub = grid[r0:r1, c0:c1]
        sh, sw = sub.shape
        if sh < 2 * self.min_panel_span or sw < 2 * self.min_panel_span:
            return None

        candidates: List[SplitScreenInvariant] = []

        # 1. Scan for explicit Vertical & Horizontal full-span separator lines
        v_seps = self._find_separators(sub, bg_color, axis=0)
        h_seps = self._find_separators(sub, bg_color, axis=1)

        for sep_start, sep_end, sep_color in v_seps:
            inv = self._evaluate_partition(
                grid=grid,
                prev_grid=prev_grid,
                viewport=viewport,
                bg_color=bg_color,
                axis="VERTICAL",
                sep_local=(sep_start, sep_end),
                sep_color=sep_color,
                avatar_color=avatar_color,
            )
            if inv is not None:
                candidates.append(inv)

        for sep_start, sep_end, sep_color in h_seps:
            inv = self._evaluate_partition(
                grid=grid,
                prev_grid=prev_grid,
                viewport=viewport,
                bg_color=bg_color,
                axis="HORIZONTAL",
                sep_local=(sep_start, sep_end),
                sep_color=sep_color,
                avatar_color=avatar_color,
            )
            if inv is not None:
                candidates.append(inv)

        # 2. If no explicit separator line found, test implicit midpoint symmetry / split axis
        if not candidates:
            if sw % 2 == 0 and sw >= 2 * self.min_panel_span:
                mid_c = sw // 2
                inv = self._evaluate_partition(
                    grid=grid,
                    prev_grid=prev_grid,
                    viewport=viewport,
                    bg_color=bg_color,
                    axis="VERTICAL",
                    sep_local=(mid_c, mid_c - 1),
                    sep_color=None,
                    avatar_color=avatar_color,
                    min_score_override=0.45,
                )
                if inv is not None:
                    candidates.append(inv)

            if sh % 2 == 0 and sh >= 2 * self.min_panel_span:
                mid_r = sh // 2
                inv = self._evaluate_partition(
                    grid=grid,
                    prev_grid=prev_grid,
                    viewport=viewport,
                    bg_color=bg_color,
                    axis="HORIZONTAL",
                    sep_local=(mid_r, mid_r - 1),
                    sep_color=None,
                    avatar_color=avatar_color,
                    min_score_override=0.45,
                )
                if inv is not None:
                    candidates.append(inv)

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (x.separator_color is not None, x.alignment_score, -x.total_discrepancy_count),
            reverse=True,
        )
        return candidates[0]

    def _find_separators(
        self,
        sub: np.ndarray,
        bg_color: int,
        axis: int,
    ) -> List[Tuple[int, int, int]]:
        """Finds contiguous 1D separator bands along `axis` (0 = vertical col line, 1 = horizontal row line)."""
        sh, sw = sub.shape
        length = sw if axis == 0 else sh
        span = sh if axis == 0 else sw

        valid_lines: List[Tuple[int, int]] = []
        for idx in range(self.min_panel_span, length - self.min_panel_span):
            line = sub[:, idx] if axis == 0 else sub[idx, :]
            vals, cnts = np.unique(line, return_counts=True)
            dom_i = int(np.argmax(cnts))
            dom_c = int(vals[dom_i])
            purity = float(cnts[dom_i]) / float(span)
            if dom_c != bg_color and purity >= self.span_purity_threshold:
                valid_lines.append((idx, dom_c))

        if not valid_lines:
            return []

        bands: List[Tuple[int, int, int]] = []
        cur_s, cur_e, cur_c = valid_lines[0][0], valid_lines[0][0], valid_lines[0][1]
        for idx, c in valid_lines[1:]:
            if idx == cur_e + 1 and c == cur_c:
                cur_e = idx
            else:
                bands.append((cur_s, cur_e, cur_c))
                cur_s, cur_e, cur_c = idx, idx, c
        bands.append((cur_s, cur_e, cur_c))

        filtered: List[Tuple[int, int, int]] = []
        for s_idx, e_idx, c_val in bands:
            thickness = e_idx - s_idx + 1
            if thickness > self.max_separator_thickness:
                continue
            prev_line = sub[:, s_idx - 1] if axis == 0 else sub[s_idx - 1, :]
            next_line = sub[:, e_idx + 1] if axis == 0 else sub[e_idx + 1, :]
            prev_ratio = float(np.mean(prev_line == c_val))
            next_ratio = float(np.mean(next_line == c_val))
            if prev_ratio < 0.50 and next_ratio < 0.50:
                filtered.append((s_idx, e_idx, c_val))

        return filtered

    def _evaluate_partition(
        self,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        viewport: Tuple[int, int, int, int],
        bg_color: int,
        axis: str,
        sep_local: Tuple[int, int],
        sep_color: Optional[int],
        avatar_color: Optional[int],
        min_score_override: Optional[float] = None,
    ) -> Optional[SplitScreenInvariant]:
        r0, c0, r1, c1 = viewport
        s_start, s_end = sep_local

        if axis == "VERTICAL":
            bbox_a = (r0, c0, r1, c0 + s_start)
            bbox_b = (r0, c0 + s_end + 1, r1, c1)
            side_a_name, side_b_name = "LEFT", "RIGHT"
            sep_global = (c0 + s_start, c0 + s_end)
        else:
            bbox_a = (r0, c0, r0 + s_start, c1)
            bbox_b = (r0 + s_end + 1, c0, r1, c1)
            side_a_name, side_b_name = "TOP", "BOTTOM"
            sep_global = (r0 + s_start, r0 + s_end)

        panel_a = grid[bbox_a[0] : bbox_a[2], bbox_a[1] : bbox_a[3]]
        panel_b = grid[bbox_b[0] : bbox_b[2], bbox_b[1] : bbox_b[3]]
        if panel_a.size == 0 or panel_b.size == 0:
            return None

        bg_a = int(collections.Counter(panel_a.flatten()).most_common(1)[0][0])
        bg_b = int(collections.Counter(panel_b.flatten()).most_common(1)[0][0])

        fg_a_count = int(np.sum(panel_a != bg_a))
        fg_b_count = int(np.sum(panel_b != bg_b))
        if fg_a_count == 0 and fg_b_count == 0:
            return None

        ref_is_a: Optional[bool] = None
        both_symmetric = False

        if prev_grid is not None and prev_grid.shape == grid.shape:
            diff_a = int(np.sum(prev_grid[bbox_a[0] : bbox_a[2], bbox_a[1] : bbox_a[3]] != panel_a))
            diff_b = int(np.sum(prev_grid[bbox_b[0] : bbox_b[2], bbox_b[1] : bbox_b[3]] != panel_b))
            if diff_b > 0 and diff_a == 0:
                ref_is_a = True
            elif diff_a > 0 and diff_b == 0:
                ref_is_a = False

        if ref_is_a is None and avatar_color is not None:
            has_av_a = bool(np.any(panel_a == avatar_color))
            has_av_b = bool(np.any(panel_b == avatar_color))
            if has_av_b and not has_av_a:
                ref_is_a = True
            elif has_av_a and not has_av_b:
                ref_is_a = False

        if ref_is_a is None:
            if panel_a.shape == panel_b.shape:
                both_symmetric = (sep_color is None)
                ref_is_a = True if fg_a_count >= fg_b_count else False
            else:
                ref_is_a = (panel_a.size <= panel_b.size)

        if ref_is_a:
            ref_panel, play_panel = panel_a, panel_b
            ref_bbox, play_bbox = bbox_a, bbox_b
            ref_bg, play_bg = bg_a, bg_b
            ref_side = "BOTH_SYMMETRIC" if both_symmetric else side_a_name
            play_side = "BOTH_SYMMETRIC" if both_symmetric else side_b_name
        else:
            ref_panel, play_panel = panel_b, panel_a
            ref_bbox, play_bbox = bbox_b, bbox_a
            ref_bg, play_bg = bg_b, bg_a
            ref_side = "BOTH_SYMMETRIC" if both_symmetric else side_b_name
            play_side = "BOTH_SYMMETRIC" if both_symmetric else side_a_name

        play_clean = play_panel.copy()
        if avatar_color is not None and avatar_color != play_bg:
            play_clean[play_clean == avatar_color] = play_bg

        scale_ref = 1
        scale_play = 1
        if ref_panel.shape != play_clean.shape:
            s_ref = _infer_fundamental_scale(ref_panel, ref_bg)
            s_play = _infer_fundamental_scale(play_clean, play_bg)
            ds_ref = _downsample_panel(ref_panel, s_ref)
            ds_play = _downsample_panel(play_clean, s_play)
            if ds_ref.shape != ds_play.shape:
                min_h = min(ref_panel.shape[0], play_clean.shape[0])
                min_w = min(ref_panel.shape[1], play_clean.shape[1])
                if abs(ref_panel.shape[0] - play_clean.shape[0]) <= 2 and abs(ref_panel.shape[1] - play_clean.shape[1]) <= 2:
                    ref_eval = ref_panel[:min_h, :min_w]
                    play_eval = play_clean[:min_h, :min_w]
                    play_bbox = (play_bbox[0], play_bbox[1], play_bbox[0] + min_h, play_bbox[1] + min_w)
                    ref_bbox = (ref_bbox[0], ref_bbox[1], ref_bbox[0] + min_h, ref_bbox[1] + min_w)
                else:
                    return None
            else:
                ref_eval, play_eval = ds_ref, ds_play
                scale_ref, scale_play = s_ref, s_play
        else:
            ref_eval, play_eval = ref_panel, play_clean

        t_copy = ref_eval
        t_mirror = np.fliplr(ref_eval) if axis == "VERTICAL" else np.flipud(ref_eval)

        score_copy, map_copy = self._score_alignment(t_copy, play_eval, ref_bg, play_bg)
        score_mirror, map_mirror = self._score_alignment(t_mirror, play_eval, ref_bg, play_bg)

        if sep_color is None:
            has_asymmetric_action = (prev_grid is not None and ref_is_a is not None)
            has_canvas_contrast = (fg_a_count >= 2 * max(1, fg_b_count)) or (fg_b_count >= 2 * max(1, fg_a_count))
            if not has_asymmetric_action and not has_canvas_contrast:
                return None
            if max(score_copy, score_mirror) < 0.70:
                return None

        thresh = min_score_override if min_score_override is not None else self.min_alignment_score
        if max(score_copy, score_mirror) < thresh and sep_color is None:
            return None

        if score_mirror > score_copy + 1e-4:
            best_transform = "REFLECTION_MIRROR"
            best_score = score_mirror
            best_map = map_mirror
            transformed_ref = t_mirror
        else:
            best_transform = "TRANSLATION_COPY"
            best_score = score_copy
            best_map = map_copy
            transformed_ref = t_copy

        mapped_target = np.vectorize(lambda c: best_map.get(int(c), int(c)))(transformed_ref)
        if scale_play > 1 and mapped_target.shape != play_clean.shape:
            target_grid = np.kron(mapped_target, np.ones((scale_play, scale_play), dtype=np.int32))
            target_grid = target_grid[: play_clean.shape[0], : play_clean.shape[1]]
        else:
            target_grid = mapped_target

        disc_mask = (play_clean != target_grid)
        discrepancies: List[DiscrepancyCell] = []

        step = max(1, _infer_fundamental_scale(target_grid, play_bg) if scale_play == 1 else scale_play)
        gh = play_clean.shape[0] // step
        gw = play_clean.shape[1] // step

        for gr in range(gh):
            for gc in range(gw):
                lr0, lc0 = gr * step, gc * step
                patch_curr = play_clean[lr0 : lr0 + step, lc0 : lc0 + step]
                patch_targ = target_grid[lr0 : lr0 + step, lc0 : lc0 + step]
                if not np.array_equal(patch_curr, patch_targ):
                    c_curr = int(collections.Counter(patch_curr.flatten()).most_common(1)[0][0])
                    c_targ = int(collections.Counter(patch_targ.flatten()).most_common(1)[0][0])
                    if c_curr == c_targ:
                        diff_pts = np.argwhere(patch_curr != patch_targ)
                        dr, dc = int(diff_pts[0, 0]), int(diff_pts[0, 1])
                        c_curr = int(patch_curr[dr, dc])
                        c_targ = int(patch_targ[dr, dc])
                        center_lr, center_lc = lr0 + dr, lc0 + dc
                    else:
                        center_lr, center_lc = lr0 + step // 2, lc0 + step // 2

                    if c_curr == play_bg and c_targ != play_bg:
                        d_type = "MISSING_FOREGROUND"
                    elif c_curr != play_bg and c_targ == play_bg:
                        d_type = "EXCESS_FOREGROUND"
                    else:
                        d_type = "COLOR_MISMATCH"

                    discrepancies.append(
                        DiscrepancyCell(
                            global_row=play_bbox[0] + center_lr,
                            global_col=play_bbox[1] + center_lc,
                            local_row=center_lr,
                            local_col=center_lc,
                            current_color=c_curr,
                            target_color=c_targ,
                            discrepancy_type=d_type,
                            patch_radius=step // 2,
                        )
                    )

        return SplitScreenInvariant(
            axis=axis,
            separator_range=sep_global,
            separator_color=sep_color,
            transform_type=best_transform,
            reference_side=ref_side,
            playfield_side=play_side,
            reference_bbox=ref_bbox,
            playfield_bbox=play_bbox,
            scale_ratio=(scale_play, scale_ref),
            color_mapping=best_map,
            alignment_score=best_score,
            target_playfield_grid=target_grid,
            discrepancy_mask=disc_mask,
            discrepancies=discrepancies,
        )

    @staticmethod
    def _score_alignment(
        ref: np.ndarray,
        play: np.ndarray,
        ref_bg: int,
        play_bg: int,
    ) -> Tuple[float, Dict[int, int]]:
        """Computes foreground structural IoU and optimal color mapping between transformed ref and play."""
        fg_ref = (ref != ref_bg)
        fg_play = (play != play_bg)
        union = int(np.sum(fg_ref | fg_play))
        if union == 0:
            return 1.0, {ref_bg: play_bg}

        inter = int(np.sum(fg_ref & fg_play))
        mask_iou = float(inter) / float(union)

        color_map: Dict[int, int] = {ref_bg: play_bg}
        ref_fg_colors = [int(c) for c in np.unique(ref[fg_ref])]
        play_fg_colors = [int(c) for c in np.unique(play[fg_play])]

        for rc in ref_fg_colors:
            rc_mask = (ref == rc)
            overlap_play = play[rc_mask & fg_play]
            if overlap_play.size > 0:
                vals, cnts = np.unique(overlap_play, return_counts=True)
                color_map[rc] = int(vals[np.argmax(cnts)])
            elif rc in play_fg_colors:
                color_map[rc] = rc
            elif len(ref_fg_colors) == 1 and len(play_fg_colors) == 1:
                color_map[rc] = play_fg_colors[0]
            else:
                color_map[rc] = rc

        mapped_ref = np.vectorize(lambda c: color_map.get(int(c), int(c)))(ref)
        exact_fg_matches = int(np.sum((mapped_ref == play) & (fg_ref | fg_play)))
        exact_score = float(exact_fg_matches) / float(union)

        return 0.5 * mask_iou + 0.5 * exact_score, color_map


# =============================================================================
# 2. Topological Enclosure (Jordan Curve & Flood-Fill) Detector
# =============================================================================

class TopologicalEnclosureDetector:
    """Detects closed Jordan boundary curves and interior cavities vs. exterior background
    via topological flood-fill from the frame boundary.
    """

    def detect(
        self,
        grid: np.ndarray,
        viewport: Optional[Tuple[int, int, int, int]] = None,
        background_color: Optional[int] = None,
        detect_open_loops: bool = True,
    ) -> TopologicalEnclosureInvariant:
        """Runs exterior boundary flood-fill and extracts all enclosed topological cavities."""
        if viewport is None:
            viewport, induced_bg = induce_playfield_viewport(grid)
        else:
            r0, c0, r1, c1 = viewport
            vals, cnts = np.unique(grid[r0:r1, c0:c1], return_counts=True)
            induced_bg = int(vals[np.argmax(cnts)])

        bg_color = induced_bg if background_color is None else background_color
        r0, c0, r1, c1 = viewport
        sub = grid[r0:r1, c0:c1]

        full_ext_mask = np.ones(grid.shape, dtype=bool)
        full_cavity_mask = np.zeros(grid.shape, dtype=bool)
        cavities: List[TopologicalCavity] = []
        seen_cavity_masks: List[np.ndarray] = []
        cavity_id_counter = 1

        # 1. Per-Color Jordan Curve Enclosure Analysis
        fg_colors = [int(c) for c in np.unique(sub) if int(c) != bg_color]
        for loop_color in fg_colors:
            if loop_color in (0, 4):
                continue
            barrier_mask = (sub == loop_color)
            if int(np.sum(barrier_mask)) < 4:
                continue

            _, cav_4 = self._flood_fill_exterior(barrier_mask, connectivity=4)
            _, cav_8 = self._flood_fill_exterior(barrier_mask, connectivity=8)

            if not np.any(cav_4):
                continue

            labeled, num_cavs = ndimage.label(cav_4)
            for idx in range(1, num_cavs + 1):
                c_mask = (labeled == idx)
                if any(np.array_equal(c_mask, prev_m) for prev_m in seen_cavity_masks):
                    continue
                seen_cavity_masks.append(c_mask)

                cav_obj = self._build_cavity_object(
                    cavity_id=cavity_id_counter,
                    c_mask=c_mask,
                    cav_8_mask=cav_8,
                    barrier_mask=barrier_mask,
                    sub=sub,
                    r0=r0,
                    c0=c0,
                    bg_color=bg_color,
                    enclosing_color=loop_color,
                )
                cavities.append(cav_obj)
                cavity_id_counter += 1
                full_cavity_mask[r0:r1, c0:c1] |= c_mask

        # 2. Composite Multi-Color Foreground Enclosure Analysis
        all_fg_barrier = (sub != bg_color) & (sub != 4) & (sub != 0)
        ext_all_4, cav_all_4 = self._flood_fill_exterior(all_fg_barrier, connectivity=4)
        _, cav_all_8 = self._flood_fill_exterior(all_fg_barrier, connectivity=8)

        full_ext_mask[r0:r1, c0:c1] = ext_all_4

        if np.any(cav_all_4):
            labeled_all, num_all = ndimage.label(cav_all_4)
            for idx in range(1, num_all + 1):
                c_mask = (labeled_all == idx)
                if any(np.all(c_mask <= prev_m) for prev_m in seen_cavity_masks):
                    continue
                seen_cavity_masks.append(c_mask)

                cav_obj = self._build_cavity_object(
                    cavity_id=cavity_id_counter,
                    c_mask=c_mask,
                    cav_8_mask=cav_all_8,
                    barrier_mask=all_fg_barrier,
                    sub=sub,
                    r0=r0,
                    c0=c0,
                    bg_color=bg_color,
                    enclosing_color=None,
                )
                cavities.append(cav_obj)
                cavity_id_counter += 1
                full_cavity_mask[r0:r1, c0:c1] |= c_mask

        # 3. Open-Loop / 1-Gap Jordan Curve Closure Induction
        open_candidates: List[OpenLoopCandidate] = []
        if detect_open_loops:
            open_candidates = self._detect_almost_closed_loops(
                sub=sub,
                r0=r0,
                c0=c0,
                bg_color=bg_color,
                existing_cavity_mask=full_cavity_mask[r0:r1, c0:c1],
            )

        return TopologicalEnclosureInvariant(
            background_color=bg_color,
            exterior_reachable_mask=full_ext_mask,
            cavity_mask=full_cavity_mask,
            cavities=cavities,
            open_loop_candidates=open_candidates,
        )

    @staticmethod
    def _flood_fill_exterior(
        barrier_mask: np.ndarray,
        connectivity: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pads `barrier_mask` with a 1-pixel traversable exterior border and flood-fills
        from (0, 0) to identify all exterior-reachable cells vs. enclosed interior cavities.
        """
        sh, sw = barrier_mask.shape
        padded_barrier = np.pad(barrier_mask, pad_width=1, mode="constant", constant_values=False)
        traversable = ~padded_barrier

        struct = (
            ndimage.generate_binary_structure(2, 1)
            if connectivity == 4
            else ndimage.generate_binary_structure(2, 2)
        )
        labeled, _ = ndimage.label(traversable, structure=struct)
        exterior_label = labeled[0, 0]
        ext_reachable = (labeled[1 : sh + 1, 1 : sw + 1] == exterior_label)
        cavity_mask = (~barrier_mask) & (~ext_reachable)
        return ext_reachable, cavity_mask

    @staticmethod
    def _build_cavity_object(
        cavity_id: int,
        c_mask: np.ndarray,
        cav_8_mask: np.ndarray,
        barrier_mask: np.ndarray,
        sub: np.ndarray,
        r0: int,
        c0: int,
        bg_color: int,
        enclosing_color: Optional[int],
    ) -> TopologicalCavity:
        coords_local = np.argwhere(c_mask)
        min_lr, min_lc = coords_local.min(axis=0)
        max_lr, max_lc = coords_local.max(axis=0)
        cent_r = float(r0 + np.mean(coords_local[:, 0]))
        cent_c = float(c0 + np.mean(coords_local[:, 1]))

        uncolored: List[Tuple[int, int]] = []
        colored: List[Tuple[int, int]] = []
        enclosed_counts: Dict[int, int] = collections.defaultdict(int)

        for lr, lc in coords_local:
            val = int(sub[lr, lc])
            gr, gc = int(r0 + lr), int(c0 + lc)
            if val == bg_color:
                uncolored.append((gr, gc))
            else:
                colored.append((gr, gc))
                enclosed_counts[val] += 1

        struct_4 = ndimage.generate_binary_structure(2, 1)
        dilated = ndimage.binary_dilation(c_mask, structure=struct_4)
        loop_mask = dilated & barrier_mask
        loop_coords = [(int(r0 + lr), int(c0 + lc)) for lr, lc in np.argwhere(loop_mask)]

        if enclosed_counts:
            induced_fill = max(enclosed_counts.items(), key=lambda kv: kv[1])[0]
        else:
            induced_fill = enclosing_color

        is_4_watertight = bool(np.all(cav_8_mask[c_mask]))

        return TopologicalCavity(
            cavity_id=cavity_id,
            enclosing_color=enclosing_color,
            is_4_watertight=is_4_watertight,
            is_8_closed=True,
            bbox=(int(r0 + min_lr), int(c0 + min_lc), int(r0 + max_lr), int(c0 + max_lc)),
            centroid=(round(cent_r, 4), round(cent_c, 4)),
            total_area=int(len(coords_local)),
            uncolored_cells=uncolored,
            colored_cells=colored,
            enclosed_colors=dict(enclosed_counts),
            boundary_loop_coords=loop_coords,
            induced_fill_color=induced_fill,
        )

    def _detect_almost_closed_loops(
        self,
        sub: np.ndarray,
        r0: int,
        c0: int,
        bg_color: int,
        existing_cavity_mask: np.ndarray,
    ) -> List[OpenLoopCandidate]:
        """Identifies open boundary curves where bridging a 1-cell gap seals a new topological cavity."""
        candidates: List[OpenLoopCandidate] = []

        for c_val in np.unique(sub):
            color = int(c_val)
            if color == bg_color:
                continue
            b_mask = (sub == color)
            if int(np.sum(b_mask)) < 7:
                continue

            nbr_count = (
                np.pad(b_mask, 1, mode="constant")[0:-2, 1:-1].astype(int)
                + np.pad(b_mask, 1, mode="constant")[2:, 1:-1].astype(int)
                + np.pad(b_mask, 1, mode="constant")[1:-1, 0:-2].astype(int)
                + np.pad(b_mask, 1, mode="constant")[1:-1, 2:].astype(int)
            )
            gap_cands = np.argwhere((~b_mask) & (~existing_cavity_mask) & (nbr_count >= 2))
            if len(gap_cands) == 0 or len(gap_cands) > 64:
                continue

            _, base_cav = self._flood_fill_exterior(b_mask, connectivity=4)

            for gr, gc in gap_cands:
                test_mask = b_mask.copy()
                test_mask[gr, gc] = True
                _, new_cav = self._flood_fill_exterior(test_mask, connectivity=4)
                added_cav = new_cav & (~base_cav)
                added_area = int(np.sum(added_cav))
                if added_area >= 1:
                    pts = np.argwhere(added_cav)
                    cr = float(r0 + np.mean(pts[:, 0]))
                    cc = float(c0 + np.mean(pts[:, 1]))
                    candidates.append(
                        OpenLoopCandidate(
                            loop_color=color,
                            gap_cells=[(int(r0 + gr), int(c0 + gc))],
                            potential_cavity_area=added_area,
                            potential_cavity_centroid=(round(cr, 4), round(cc, 4)),
                        )
                    )
        return candidates


# =============================================================================
# 3. HUD Sequence / Highlighted Indicator Invariant Detector
# =============================================================================

class HUDSequenceDetector:
    """Detects a 1D sequence of colored tokens in the border strip (e.g., rows 0..4 or 60..63)
    where one active token is highlighted by a surrounding bounding frame, and matches
    that token color to target objects in the playfield.
    """

    def __init__(self, border_strip_height: int = 5) -> None:
        self.border_strip_height = border_strip_height

    def detect(
        self,
        grid: np.ndarray,
        viewport: Optional[Tuple[int, int, int, int]] = None,
        avatar_pos: Optional[Tuple[int, int]] = None,
    ) -> Optional[HUDSequenceInvariant]:
        """Scans top and bottom (plus left/right) border strips for a highlighted 1D token sequence."""
        h, w = grid.shape
        b = min(self.border_strip_height, h // 4)
        if viewport is None:
            viewport, play_bg = induce_playfield_viewport(grid, default_border=b)
        else:
            r0, c0, r1, c1 = viewport
            vals, cnts = np.unique(grid[r0:r1, c0:c1], return_counts=True)
            play_bg = int(vals[np.argmax(cnts)])

        strips = [
            ("TOP", (0, 0, b, w), grid[0:b, :], True),
            ("BOTTOM", (h - b, 0, h, w), grid[h - b : h, :], True),
            ("LEFT", (0, 0, h, b), grid[:, 0:b], False),
            ("RIGHT", (0, w - b, h, w), grid[:, w - b : w], False),
        ]

        best_hud: Optional[HUDSequenceInvariant] = None

        for loc_name, (sr0, sc0, sr1, sc1), strip_arr, is_horizontal in strips:
            work_strip = strip_arr if is_horizontal else strip_arr.T
            hud_res = self._parse_strip_sequence(
                work_strip=work_strip,
                loc_name=loc_name,
                strip_slice=(sr0, sc0, sr1, sc1),
                is_horizontal=is_horizontal,
            )
            if hud_res is not None:
                matched_objs = self._match_playfield_objects(
                    grid=grid,
                    viewport=viewport,
                    play_bg=play_bg,
                    target_color=hud_res.active_token_color,
                    avatar_pos=avatar_pos,
                )
                hud_res.matched_playfield_objects = matched_objs
                if best_hud is None or len(hud_res.tokens) > len(best_hud.tokens):
                    best_hud = hud_res

        return best_hud

    def _parse_strip_sequence(
        self,
        work_strip: np.ndarray,
        loc_name: str,
        strip_slice: Tuple[int, int, int, int],
        is_horizontal: bool,
    ) -> Optional[HUDSequenceInvariant]:
        """Analyzes a (b x L) border strip for a 1D array of tokens with one framed active token."""
        sh, _ = work_strip.shape
        vals, cnts = np.unique(work_strip, return_counts=True)
        if len(vals) < 3:
            return None

        strip_bg = int(vals[np.argmax(cnts)])
        sr0, sc0, _, _ = strip_slice

        struct_8 = ndimage.generate_binary_structure(2, 2)
        fg_colors = [int(c) for c in vals if int(c) != strip_bg]

        frame_candidates: List[Tuple[int, Tuple[int, int, int, int], int, Tuple[int, int, int, int]]] = []

        for c_frame in fg_colors:
            f_mask = (work_strip == c_frame)
            labeled_f, num_f = ndimage.label(f_mask, structure=struct_8)
            for f_idx in range(1, num_f + 1):
                comp_f = (labeled_f == f_idx)
                f_pts = np.argwhere(comp_f)
                f_mass = len(f_pts)
                r_min, c_min = f_pts.min(axis=0)
                r_max, c_max = f_pts.max(axis=0)
                fh = r_max - r_min + 1
                fw = c_max - c_min + 1

                if not (2 <= fh <= sh and 3 <= fw <= 14 and f_mass >= 4):
                    continue

                sub_box = work_strip[r_min : r_max + 1, c_min : c_max + 1]
                f_sub_mask = comp_f[r_min : r_max + 1, c_min : c_max + 1]

                if f_mass >= fh * fw:
                    continue

                inner_colors = [
                    int(c)
                    for c in np.unique(sub_box[~f_sub_mask])
                    if int(c) not in (strip_bg, c_frame)
                ]
                if len(inner_colors) != 1:
                    continue

                t_color = inner_colors[0]
                t_pts_local = np.argwhere((sub_box == t_color) & (~f_sub_mask))
                if len(t_pts_local) == 0:
                    continue

                tr_min, tc_min = t_pts_local.min(axis=0) + np.array([r_min, c_min])
                tr_max, tc_max = t_pts_local.max(axis=0) + np.array([r_min, c_min])

                has_left = bool(np.any(comp_f[r_min : r_max + 1, :tc_min]))
                has_right = bool(np.any(comp_f[r_min : r_max + 1, tc_max + 1 :]))
                has_top = bool(np.any(comp_f[:tr_min, c_min : c_max + 1]))
                has_bottom = bool(np.any(comp_f[tr_max + 1 :, c_min : c_max + 1]))

                if has_left and has_right and (has_top or has_bottom or fh >= 3):
                    frame_candidates.append(
                        (
                            c_frame,
                            (int(r_min), int(c_min), int(r_max), int(c_max)),
                            t_color,
                            (int(tr_min), int(tc_min), int(tr_max), int(tc_max)),
                        )
                    )

        if not frame_candidates:
            return None

        for c_frame, f_bbox, active_t_color, active_t_bbox in frame_candidates:
            fr_min, fc_min, fr_max, fc_max = f_bbox
            tr_min, tc_min, tr_max, tc_max = active_t_bbox
            t_height = tr_max - tr_min + 1
            t_width = tc_max - tc_min + 1

            struct_4 = ndimage.generate_binary_structure(2, 1)
            raw_tokens: List[Tuple[int, Tuple[int, int, int, int], int, bool]] = []

            active_mass = int(np.sum(work_strip[tr_min : tr_max + 1, tc_min : tc_max + 1] == active_t_color))
            raw_tokens.append((active_t_color, active_t_bbox, active_mass, True))

            rem_strip = work_strip.copy()
            rem_strip[fr_min : fr_max + 1, fc_min : fc_max + 1] = strip_bg

            for c_val in np.unique(rem_strip):
                c_int = int(c_val)
                if c_int == strip_bg:
                    continue
                c_labeled, c_num = ndimage.label(rem_strip == c_int, structure=struct_4)
                for idx in range(1, c_num + 1):
                    pts = np.argwhere(c_labeled == idx)
                    mass = len(pts)
                    r0_t, c0_t = pts.min(axis=0)
                    r1_t, c1_t = pts.max(axis=0)
                    h_t = r1_t - r0_t + 1
                    w_t = c1_t - c0_t + 1
                    if 1 <= h_t <= max(3, t_height + 1) and 1 <= w_t <= max(4, t_width + 2) and mass <= 16:
                        if abs((r0_t + r1_t) / 2.0 - (tr_min + tr_max) / 2.0) <= 1.5:
                            raw_tokens.append((c_int, (int(r0_t), int(c0_t), int(r1_t), int(c1_t)), mass, False))

            if len(raw_tokens) < 2:
                continue

            raw_tokens.sort(key=lambda item: item[1][1])

            hud_tokens: List[HUDToken] = []
            active_idx = 0
            for idx, (t_col, (lr0, lc0, lr1, lc1), t_mass, is_hi) in enumerate(raw_tokens):
                if is_horizontal:
                    g_bbox = (sr0 + lr0, sc0 + lc0, sr0 + lr1, sc0 + lc1)
                    g_f_bbox = (sr0 + fr_min, sc0 + fc_min, sr0 + fr_max, sc0 + fc_max) if is_hi else None
                else:
                    g_bbox = (sr0 + lc0, sc0 + lr0, sr0 + lc1, sc0 + lr1)
                    g_f_bbox = (sr0 + fc_min, sc0 + fr_min, sr0 + fc_max, sc0 + fr_max) if is_hi else None

                g_cent = ((g_bbox[0] + g_bbox[2]) // 2, (g_bbox[1] + g_bbox[3]) // 2)
                if is_hi:
                    active_idx = idx

                hud_tokens.append(
                    HUDToken(
                        index=idx,
                        color=t_col,
                        center=g_cent,
                        bbox=g_bbox,
                        mass=t_mass,
                        is_highlighted=is_hi,
                        frame_color=c_frame if is_hi else None,
                        frame_bbox=g_f_bbox,
                    )
                )

            return HUDSequenceInvariant(
                strip_location=loc_name,
                strip_slice=strip_slice,
                strip_background_color=strip_bg,
                frame_color=c_frame,
                tokens=hud_tokens,
                active_token_index=active_idx,
                active_token_color=active_t_color,
            )

        return None

    @staticmethod
    def _match_playfield_objects(
        grid: np.ndarray,
        viewport: Tuple[int, int, int, int],
        play_bg: int,
        target_color: int,
        avatar_pos: Optional[Tuple[int, int]] = None,
    ) -> List[PlayfieldMatchObject]:
        """Finds all objects in the playfield matching `target_color` and ranks them."""
        r0, c0, r1, c1 = viewport
        sub = grid[r0:r1, c0:c1]
        if target_color == play_bg:
            return []

        mask = (sub == target_color)
        if not np.any(mask):
            return []

        struct_4 = ndimage.generate_binary_structure(2, 1)
        labeled, num_objs = ndimage.label(mask, structure=struct_4)
        matches: List[PlayfieldMatchObject] = []

        for idx in range(1, num_objs + 1):
            pts = np.argwhere(labeled == idx)
            mass = int(len(pts))
            lr_min, lc_min = pts.min(axis=0)
            lr_max, lc_max = pts.max(axis=0)
            exact_r = float(r0 + np.mean(pts[:, 0]))
            exact_c = float(c0 + np.mean(pts[:, 1]))
            cent_r = int(round(exact_r))
            cent_c = int(round(exact_c))

            dist = (
                abs(cent_r - avatar_pos[0]) + abs(cent_c - avatar_pos[1])
                if avatar_pos is not None
                else None
            )

            global_coords = [(int(r0 + lr), int(c0 + lc)) for lr, lc in pts]
            matches.append(
                PlayfieldMatchObject(
                    component_id=idx,
                    color=target_color,
                    centroid=(cent_r, cent_c),
                    exact_centroid=(round(exact_r, 4), round(exact_c, 4)),
                    bbox=(int(r0 + lr_min), int(c0 + lc_min), int(r0 + lr_max), int(c0 + lc_max)),
                    mass=mass,
                    coords=global_coords,
                    distance_to_avatar=dist,
                )
            )

        matches.sort(
            key=lambda m: (
                m.distance_to_avatar if m.distance_to_avatar is not None else 0,
                m.centroid[0],
                m.centroid[1],
            )
        )
        return matches


# =============================================================================
# Unified Visual Goal & Invariant Induction Engine
# =============================================================================

@dataclass
class InducedVisualGoalState:
    """Unified output of all induced visual invariants for a 64x64 frame."""
    playfield_viewport: Tuple[int, int, int, int]
    background_color: int
    split_screen: Optional[SplitScreenInvariant]
    enclosure: TopologicalEnclosureInvariant
    hud_sequence: Optional[HUDSequenceInvariant]

    def get_prioritized_click_targets(self) -> List[Tuple[int, int, str]]:
        """Returns prioritized (x_col, y_row, reason) click targets induced from invariants."""
        targets: List[Tuple[int, int, str]] = []

        # 1. Split-screen / symmetry discrepancy cells
        if self.split_screen is not None and not self.split_screen.is_satisfied:
            for d in self.split_screen.discrepancies:
                targets.append((d.global_col, d.global_row, f"SPLIT_{d.discrepancy_type}"))

        # 2. HUD active token matched playfield objects
        if self.hud_sequence is not None and self.hud_sequence.matched_playfield_objects:
            for obj in self.hud_sequence.matched_playfield_objects:
                targets.append(
                    (obj.centroid[1], obj.centroid[0], f"HUD_MATCH_COLOR_{obj.color}")
                )

        # 3. Topological enclosure uncolored cavities
        # Completely untouched (100% uncolored) first, then by descending area
        sorted_cavs = sorted(
            self.enclosure.cavities,
            key=lambda c: (
                1 if len(c.uncolored_cells) == c.total_area else 0,
                c.total_area,
            ),
            reverse=True,
        )
        for cav in sorted_cavs:
            if cav.uncolored_cells:
                cr, cc = int(round(cav.centroid[0])), int(round(cav.centroid[1]))
                if (cr, cc) in cav.uncolored_cells:
                    targets.append((cc, cr, f"FILL_CAVITY_{cav.cavity_id}"))
                else:
                    r_first, c_first = cav.uncolored_cells[0]
                    targets.append((c_first, r_first, f"FILL_CAVITY_{cav.cavity_id}"))

        # 4. Open-loop 1-cell Jordan gaps (secondary heuristic)
        for gap_cand in self.enclosure.open_loop_candidates:
            for gr, gc in gap_cand.gap_cells:
                targets.append((gc, gr, f"SEAL_JORDAN_LOOP_COLOR_{gap_cand.loop_color}"))

        return targets


class VisualGoalInductionEngine:
    """Master zero-hardcoding engine combining Split-Screen, Topological Enclosure,
    and HUD Sequence Invariant detectors.
    """

    def __init__(self) -> None:
        self.split_detector = SplitScreenDetector()
        self.enclosure_detector = TopologicalEnclosureDetector()
        self.hud_detector = HUDSequenceDetector()

    def induce(
        self,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray] = None,
        avatar_color: Optional[int] = None,
        avatar_pos: Optional[Tuple[int, int]] = None,
    ) -> InducedVisualGoalState:
        viewport, bg_color = induce_playfield_viewport(grid)

        hud_inv = self.hud_detector.detect(
            grid=grid,
            viewport=viewport,
            avatar_pos=avatar_pos,
        )
        split_inv = self.split_detector.detect(
            grid=grid,
            prev_grid=prev_grid,
            avatar_color=avatar_color,
            viewport=viewport,
        )
        enclosure_inv = self.enclosure_detector.detect(
            grid=grid,
            viewport=viewport,
            background_color=bg_color,
        )

        return InducedVisualGoalState(
            playfield_viewport=viewport,
            background_color=bg_color,
            split_screen=split_inv,
            enclosure=enclosure_inv,
            hud_sequence=hud_inv,
        )
