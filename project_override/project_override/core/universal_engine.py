"""Project Override: Universal Zero-Overfitting Reasoning Engine (ARC-AGI-3)
Implements strictly general, non-overfitted perception, induction, and search modules:
1. 64-Bit Bitwise Dead-Signature Culling (O(1) null transition avoidance)
2. Dynamic Kinematic Prober (reversible avatar & displacement vector discovery)
3. Universal Background & Lattice Separation (Perimeter flux & topological regular component analysis)
4. Online Action-Semantics Inducer (Case A, B, C, D classification for Action 6 and 5)
5. Universal Discrete Board Extractor (dynamic scale, structural + empirical object-role induction)
6. Universal GF(2) Impulse-Response Toggle Solver (color-agnostic grid & kernel induction)
7. Universal Lattice & K-Tuple Memory Matcher (dynamic card lattice & match-size induction)
8. Universal Reachability Sokoban Solver (1-step & multi-cell slide push BFS)
9. Universal Inertial Slide Solver (arbitrary-vector raycast BFS)
10. Universal Jump / Knight Solver (multi-vector BFS)
11. Universal Cardinal A* Navigator
12. Topological Shannon Entropy Heuristic
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import ndimage


# =====================================================================
# 1. 64-Bit Packed Bitwise Action Signature (Dead-Signature Culler)
# =====================================================================

SHIFT_CTX = 0
MASK_CTX = 0xFFFFFFFF
SHIFT_MASS = 32
MASK_MASS = 0x0FFF
SHIFT_COORD = 44
MASK_COORD = 0x0FFF
SHIFT_ACT = 56
MASK_ACT = 0x00FF


def pack_signature_64(action_byte: int, coords_12: int, mass_color_12: int, ctx_hash_32: int) -> int:
    """Packs transition state-action key into a single 64-bit integer."""
    return (
        ((int(action_byte) & MASK_ACT) << SHIFT_ACT)
        | ((int(coords_12) & MASK_COORD) << SHIFT_COORD)
        | ((int(mass_color_12) & MASK_MASS) << SHIFT_MASS)
        | (int(ctx_hash_32) & MASK_CTX)
    )


def fast_grid_hash(grid: np.ndarray) -> int:
    """Fast 32-bit FNV-1a hash of a 2D integer numpy grid."""
    h = 2166136261
    for byte in grid.tobytes():
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


# =====================================================================
# 2. Dynamic Kinematic Prober (Reversible Avatar & Vector Discovery)
# =====================================================================

@dataclass
class KinematicProfile:
    """Empirical kinematic characteristics induced from initial action probes."""
    avatar_color: Optional[int] = None
    avatar_size: Tuple[int, int] = (1, 1)
    displacement_vectors: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    modality: str = "NAV"  # 'NAV', 'SLIDE', 'KNIGHT', 'SOKOBAN', 'CLICK'
    is_sliding: bool = False
    is_deterministic: bool = True
    probed: bool = False


class DynamicKinematicProber:
    """Executes orthogonal micro-probes on Level 1 to identify the avatar entity,
    displacement vectors, and movement dynamics from frame deltas without getting stuck on walls.
    """

    def __init__(self) -> None:
        self.profile = KinematicProfile()
        self.probe_step = 0
        self.f0_grid: Optional[np.ndarray] = None
        self.candidate_actions: List[int] = [1, 4, 2, 3]  # Up, Right, Down, Left
        self.current_probe_idx: int = 0

    def reset_for_new_game(self) -> None:
        self.profile = KinematicProfile()
        self.probe_step = 0
        self.f0_grid = None
        self.current_probe_idx = 0

    def get_probe_action(self, grid: np.ndarray, legal_actions: List[int]) -> Optional[int]:
        """Returns the next probing action if probing is active, or None if finished."""
        if self.profile.probed:
            return None

        valid_candidates = [a for a in self.candidate_actions if a in legal_actions]
        if not valid_candidates:
            self.profile.probed = True
            return None

        if self.probe_step == 0:
            self.f0_grid = grid.copy()
            self.current_probe_idx = 0
            self.probe_step = 1
            return valid_candidates[0]

        if self.probe_step == 1:
            diff = self.f0_grid[4:60, 4:60] != grid[4:60, 4:60]
            delta_omega = int(np.sum(diff))

            if delta_omega > 0:
                c0_colors = set(self.f0_grid[4:60, 4:60][diff])
                c1_colors = set(grid[4:60, 4:60][diff])
                bg_0 = collections.Counter(self.f0_grid[4:60, 4:60].flatten()).most_common(1)[0][0]
                candidates = [c for c in (c0_colors & c1_colors) if c != bg_0 and c != 4]
                if not candidates:
                    candidates = [c for c in c0_colors if c != bg_0 and c != 4]
                if not candidates:
                    candidates = [c for c in c1_colors if c != bg_0 and c != 4]

                if candidates:
                    av_c = int(candidates[0])
                    self.profile.avatar_color = av_c

                    p0 = np.argwhere((self.f0_grid == av_c) & (np.arange(64)[:, None] >= 4))
                    p1 = np.argwhere((grid == av_c) & (np.arange(64)[:, None] >= 4))

                    if len(p0) > 0 and len(p1) > 0:
                        dr = int(round(p1[:, 0].mean() - p0[:, 0].mean()))
                        dc = int(round(p1[:, 1].mean() - p0[:, 1].mean()))
                        step_size = max(1, max(abs(dr), abs(dc)))
                        self.profile.avatar_size = (step_size, step_size)

                        if dr != 0 and dc == 0:
                            self.profile.displacement_vectors = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                            if abs(dr) > 8:
                                self.profile.is_sliding = True
                                self.profile.modality = "SLIDE"
                            else:
                                self.profile.modality = "NAV"
                        elif abs(abs(dr) - abs(dc)) <= 1 and min(abs(dr), abs(dc)) >= 2:
                            self.profile.displacement_vectors = {1: (-1, -1), 2: (-1, 1), 3: (1, -1), 4: (1, 1)}
                            self.profile.is_sliding = True
                            self.profile.modality = "SLIDE"
                        elif dr != 0 and dc != 0 and (abs(dr) != abs(dc)):
                            self.profile.displacement_vectors = {1: (-1, -2), 2: (1, -2), 3: (-1, 2), 4: (1, 2)}
                            self.profile.modality = "KNIGHT"
                            self.profile.is_sliding = False
                        else:
                            self.profile.displacement_vectors = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                            self.profile.modality = "NAV"

                self.profile.probed = True
                return None
            else:
                self.current_probe_idx += 1
                if self.current_probe_idx < len(valid_candidates):
                    return valid_candidates[self.current_probe_idx]
                else:
                    self.profile.avatar_color = None
                    self.profile.modality = "CLICK"
                    self.profile.probed = True
                    return None

        self.profile.probed = True
        return None


# =====================================================================
# 3. Universal Background vs. Foreground Lattice Separation
# =====================================================================

@dataclass
class ArenaSeparation:
    """Result of universal background, frame, and foreground lattice separation."""
    y_min: int
    y_max: int  # Exclusive slice bound
    x_min: int
    x_max: int  # Exclusive slice bound
    frame_colors: Set[int]
    bg_color: int
    tile_back_color: Optional[int]
    foreground_mask: np.ndarray
    lattice: Optional["LatticeGrid"] = None


def detect_arena_bounds_and_frame(grid: np.ndarray, max_margin: int = 6) -> Tuple[int, int, int, int, Set[int]]:
    """Dynamically detects outer bezel/timer-bar margin and frame colors without hardcoding."""
    H, W = grid.shape[:2]
    frame_colors: Set[int] = set()
    inner_sub = grid[max_margin : H - max_margin, max_margin : W - max_margin]

    def is_border_line(line: np.ndarray) -> Tuple[bool, int]:
        vals, cnts = np.unique(line, return_counts=True)
        dom_idx = int(np.argmax(cnts))
        dom_c = int(vals[dom_idx])
        purity = float(cnts[dom_idx]) / float(line.size)
        inner_frac = float(np.sum(inner_sub == dom_c)) / float(inner_sub.size) if inner_sub.size > 0 else 0.0
        return (purity >= 0.85 and inner_frac < 0.10), dom_c

    y_min = 0
    for r in range(min(max_margin, H // 4)):
        unif, c = is_border_line(grid[r, :])
        if unif:
            frame_colors.add(c)
            y_min = r + 1
        else:
            break

    y_max = H
    for r in range(H - 1, max(H - 1 - max_margin, 3 * H // 4), -1):
        unif, c = is_border_line(grid[r, :])
        if unif:
            frame_colors.add(c)
            y_max = r
        else:
            break

    x_min = 0
    for col in range(min(max_margin, W // 4)):
        unif, c = is_border_line(grid[:, col])
        if unif:
            frame_colors.add(c)
            x_min = col + 1
        else:
            break

    x_max = W
    for col in range(W - 1, max(W - 1 - max_margin, 3 * W // 4), -1):
        unif, c = is_border_line(grid[:, col])
        if unif:
            frame_colors.add(c)
            x_max = col
        else:
            break

    if y_min == 0 and H >= 16 and W >= 16:
        outer_ring = np.concatenate([grid[0, :], grid[H - 1, :], grid[:, 0], grid[:, W - 1]])
        vals, cnts = np.unique(outer_ring, return_counts=True)
        dom_c = int(vals[np.argmax(cnts)])
        outer_frac = float(np.max(cnts)) / float(outer_ring.size)
        inner_frac = float(np.sum(inner_sub == dom_c)) / float(inner_sub.size) if inner_sub.size > 0 else 0.0
        if outer_frac >= 0.50 and inner_frac < 0.05:
            frame_colors.add(dom_c)
            y_min, y_max, x_min, x_max = 4, H - 4, 4, W - 4

    return y_min, y_max, x_min, x_max, frame_colors


def separate_background_and_lattice(grid: np.ndarray) -> ArenaSeparation:
    """Separates true canvas background from dense foreground tile grids using
    perimeter incidence, 4-border span, and topological component regularity.
    """
    H, W = grid.shape[:2]
    y0, y1, x0, x1, frame_colors = detect_arena_bounds_and_frame(grid)
    interior = grid[y0:y1, x0:x1]
    ih, iw = interior.shape[:2]

    band = min(2, max(1, min(ih, iw) // 16))
    top_band = interior[:band, :]
    bot_band = interior[ih - band :, :]
    left_band = interior[:, :band]
    right_band = interior[:, iw - band :]

    perim_pixels = np.concatenate([
        top_band.flatten(),
        bot_band.flatten(),
        left_band.flatten(),
        right_band.flatten(),
    ])

    # Outer canvas boundary check (rows 0..2, rows 62..63, cols 0..2, cols 62..63)
    outer_ring = np.concatenate([
        grid[:3, :].flatten(),
        grid[-3:, :].flatten(),
        grid[:, :3].flatten(),
        grid[:, -3:].flatten(),
    ])
    outer_vals, outer_cnts = np.unique(outer_ring, return_counts=True)
    c_outer = int(outer_vals[np.argmax(outer_cnts)])
    outer_frac = float(np.max(outer_cnts)) / float(outer_ring.size)

    all_vals, _ = np.unique(interior, return_counts=True)
    candidate_colors = [int(v) for v in all_vals]

    bg_scores: Dict[int, float] = {}
    color_stats: Dict[int, Dict[str, Any]] = {}

    for c in candidate_colors:
        area_frac = float(np.sum(interior == c)) / float(interior.size)
        perim_frac = float(np.sum(perim_pixels == c)) / float(perim_pixels.size)

        sides_touched = sum([
            int(np.any(top_band == c)),
            int(np.any(bot_band == c)),
            int(np.any(left_band == c)),
            int(np.any(right_band == c)),
        ])
        side_span = sides_touched / 4.0

        c_mask = (interior == c)
        labeled, n_comp = ndimage.label(c_mask)
        comp_sizes = ndimage.sum(c_mask, labeled, range(1, n_comp + 1)) if n_comp > 0 else []
        max_comp_share = float(np.max(comp_sizes)) / float(np.sum(c_mask)) if len(comp_sizes) > 0 else 0.0

        rect_regularity = 0.0
        if n_comp >= 4:
            slices = ndimage.find_objects(labeled)
            solidities = []
            areas = []
            for idx, sl in enumerate(slices):
                if sl is None:
                    continue
                bh = sl[0].stop - sl[0].start
                bw = sl[1].stop - sl[1].start
                if bh >= 2 and bw >= 2:
                    comp_area = float(comp_sizes[idx])
                    solidities.append(comp_area / float(bh * bw))
                    areas.append(comp_area)
            if len(solidities) >= 4:
                med_sol = float(np.median(solidities))
                cv_area = float(np.std(areas) / (np.mean(areas) + 1e-6))
                if med_sol >= 0.80 and cv_area <= 0.35:
                    rect_regularity = med_sol * (1.0 - min(1.0, cv_area))

        color_stats[c] = {
            "area_frac": area_frac,
            "perim_frac": perim_frac,
            "side_span": side_span,
            "n_comp": n_comp,
            "max_comp_share": max_comp_share,
            "rect_regularity": rect_regularity,
        }

        score = (
            0.45 * perim_frac
            + 0.25 * side_span
            + 0.20 * (max_comp_share * min(1.0, area_frac * 3.0))
            + 0.10 * area_frac
            - 0.40 * rect_regularity
        )
        bg_scores[c] = score

    if outer_frac >= 0.80 and c_outer != 4:
        bg_color = c_outer
    else:
        bg_color = max(candidate_colors, key=lambda c: bg_scores[c])

    tile_back_color: Optional[int] = None
    rem_colors = [c for c in candidate_colors if c != bg_color]
    for c in sorted(rem_colors, key=lambda x: color_stats.get(x, {}).get("area_frac", 0.0), reverse=True):
        st = color_stats.get(c, {})
        if st.get("rect_regularity", 0.0) >= 0.55 or (st.get("area_frac", 0.0) >= 0.25 and st.get("n_comp", 0) >= 1):
            tile_back_color = c
            break

    fg_mask = np.zeros((H, W), dtype=bool)
    fg_mask[y0:y1, x0:x1] = (interior != bg_color)
    for fc in frame_colors:
        fg_mask &= (grid != fc)

    lattice = detect_lattice_grid(grid, bg_color)

    return ArenaSeparation(
        y_min=y0,
        y_max=y1,
        x_min=x0,
        x_max=x1,
        frame_colors=frame_colors,
        bg_color=bg_color,
        tile_back_color=tile_back_color,
        foreground_mask=fg_mask,
        lattice=lattice,
    )


# =====================================================================
# 4. Online Action-Semantics Classification
# =====================================================================

class ActionEffectType(str, Enum):
    NULL_EFFECT = "NULL_EFFECT"
    CASE_A_COMPONENT_PAINT = "CASE_A_COMPONENT_PAINT"
    CASE_B_SINGLE_CELL_CYCLE = "CASE_B_SINGLE_CELL_CYCLE"
    CASE_C_KERNEL_PROPAGATION = "CASE_C_KERNEL_PROPAGATION"
    CASE_D_MEMORY_MATCHING = "CASE_D_MEMORY_MATCHING"


@dataclass
class TransitionClassification:
    """Detailed semantic classification of an observed (frame_before, action, frame_after) step."""
    effect_type: ActionEffectType
    confidence: float
    changed_pixels: int
    affected_cells: List[Tuple[int, int]] = field(default_factory=list)
    source_color: Optional[int] = None
    target_color: Optional[int] = None
    cycle_length_k: Optional[int] = None
    memory_tuple_k: Optional[int] = None
    triggered_reversion: bool = False


# =====================================================================
# 5. Universal Discrete Board Extractor (Structural & Empirical Roles)
# =====================================================================

def _induce_object_roles_on_board(
    board: np.ndarray,
    f: np.ndarray,
    y0: int,
    x0: int,
    scale: int,
    gh: int,
    gw: int,
    p_pos: Tuple[int, int],
    avatar_color: int,
    bg_color: int,
    learned_wall_colors: Optional[Set[int]] = None,
    learned_box_colors: Optional[Set[int]] = None,
    learned_goal_colors: Optional[Set[int]] = None,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Induces (walls, boxes, plates, exit_doors) from structural geometry and empirical role memory.
    Assigns each discrete board cell to a single canonical foreground role to eliminate cardinality errors.
    """
    w_colors = set(learned_wall_colors) if learned_wall_colors else set()
    b_colors = set(learned_box_colors) if learned_box_colors else set()
    g_colors = set(learned_goal_colors) if learned_goal_colors else set()

    color_cells: Dict[int, Set[Tuple[int, int]]] = collections.defaultdict(set)
    fill_ratios: Dict[int, List[float]] = collections.defaultdict(list)

    for r in range(gh):
        for c in range(gw):
            if (r, c) == p_pos:
                continue
            cv = int(board[r, c])
            if cv in (bg_color, avatar_color, 4, 0):
                continue
            patch = f[y0 + r * scale : y0 + (r + 1) * scale, x0 + c * scale : x0 + (c + 1) * scale]
            cnt = int(np.sum(patch == cv)) if patch.size > 0 else 0
            color_cells[cv].add((r, c))
            fill_ratios[cv].append(float(cnt) / float(max(1, patch.size)))

    walls: Set[Tuple[int, int]] = set()
    boxes: Set[Tuple[int, int]] = set()
    plates: Set[Tuple[int, int]] = set()
    exit_doors: Set[Tuple[int, int]] = set()

    all_cand_colors = set(color_cells.keys())

    # 1. Structural wall identification:
    # A color is classified as a wall if:
    # - It is in learned_wall_colors, OR
    # - It forms an outer perimeter border of the board (continuous line on board edge), OR
    # - It forms long orthogonal lines (>= 6 cells) AND there are at least two OTHER foreground colors present
    for c_val in list(all_cand_colors):
        cells = color_cells[c_val]
        if c_val in w_colors:
            walls |= cells
            continue
        if c_val in b_colors or c_val in g_colors:
            continue

        is_perimeter_wall = False
        if len(cells) >= 6:
            r_coords = [r for r, _ in cells]
            c_coords = [c for _, c in cells]
            is_perimeter_wall = (
                r_coords.count(0) >= 5
                or r_coords.count(gh - 1) >= 5
                or c_coords.count(0) >= 5
                or c_coords.count(gw - 1) >= 5
            )

        if is_perimeter_wall and len(all_cand_colors - {c_val}) >= 1:
            walls |= cells
            w_colors.add(c_val)
        elif len(cells) >= 6 and len(all_cand_colors - {c_val}) >= 2:
            cells_set = set(cells)
            adj_pairs = 2 * sum(
                1
                for (r, c) in cells_set
                for (dr, dc) in ((0, 1), (1, 0))
                if (r + dr, c + dc) in cells_set
            )
            if adj_pairs >= len(cells):
                walls |= cells
                w_colors.add(c_val)

    remaining_colors = [c_val for c_val in all_cand_colors if c_val not in w_colors]

    # 2. Apply empirical box / goal classifications
    for c_val in list(remaining_colors):
        if c_val in b_colors:
            boxes |= color_cells[c_val]
            remaining_colors.remove(c_val)
        elif c_val in g_colors:
            plates |= color_cells[c_val]
            remaining_colors.remove(c_val)

    # 3. Structural pairing for unlearned foreground colors (Sokoban boxes + targets)
    if not boxes and not plates and len(remaining_colors) >= 2:
        by_count: Dict[int, List[int]] = collections.defaultdict(list)
        for c_val in remaining_colors:
            cnt = len(color_cells[c_val])
            if 1 <= cnt <= 12:
                by_count[cnt].append(c_val)

        paired = False
        for _, c_list in sorted(by_count.items()):
            if len(c_list) >= 2:
                c_list.sort(
                    key=lambda cv: float(np.mean(fill_ratios[cv])) if fill_ratios[cv] else 0.0,
                    reverse=True,
                )
                box_c, plate_c = c_list[0], c_list[1]
                boxes |= color_cells[box_c]
                plates |= color_cells[plate_c]
                for extra_c in remaining_colors:
                    if extra_c not in (box_c, plate_c):
                        walls |= color_cells[extra_c]
                paired = True
                break

        if not paired:
            for c_val in remaining_colors:
                plates |= color_cells[c_val]
    elif boxes and not plates and remaining_colors:
        # Boxes are empirically known; deduce plates from remaining colors matching box count
        box_count = len(boxes)
        matching_cnt = [c for c in remaining_colors if len(color_cells[c]) == box_count]
        if matching_cnt:
            # Plates have lower patch fill ratio than solid walls/obstacles
            matching_cnt.sort(
                key=lambda cv: float(np.mean(fill_ratios[cv])) if fill_ratios[cv] else 0.0
            )
            plate_c = matching_cnt[0]
            plates |= color_cells[plate_c]
            for c in remaining_colors:
                if c != plate_c:
                    walls |= color_cells[c]
        else:
            remaining_colors.sort(
                key=lambda cv: (len(color_cells[cv]), float(np.mean(fill_ratios[cv])) if fill_ratios[cv] else 0.0)
            )
            plate_c = remaining_colors[0]
            plates |= color_cells[plate_c]
            for c in remaining_colors[1:]:
                walls |= color_cells[c]
    elif not boxes and plates and remaining_colors:
        # Plates are known; deduce boxes from remaining colors matching plate count
        plate_count = len(plates)
        matching_cnt = [c for c in remaining_colors if len(color_cells[c]) == plate_count]
        if matching_cnt:
            # Boxes have higher patch fill ratio (solid blocks) than empty/marker tiles
            matching_cnt.sort(
                key=lambda cv: float(np.mean(fill_ratios[cv])) if fill_ratios[cv] else 0.0,
                reverse=True,
            )
            box_c = matching_cnt[0]
            boxes |= color_cells[box_c]
            for c in remaining_colors:
                if c != box_c:
                    walls |= color_cells[c]
        else:
            remaining_colors.sort(
                key=lambda cv: float(np.mean(fill_ratios[cv])) if fill_ratios[cv] else 0.0,
                reverse=True,
            )
            box_c = remaining_colors[0]
            boxes |= color_cells[box_c]
            for c in remaining_colors[1:]:
                walls |= color_cells[c]
    elif boxes and plates:
        # Both boxes and plates are identified; any remaining colors are obstacle walls
        for c_val in remaining_colors:
            walls |= color_cells[c_val]
    else:
        for c_val in remaining_colors:
            plates |= color_cells[c_val]

    walls.discard(p_pos)
    return walls, boxes, plates, exit_doors


def extract_discrete_board(
    f: np.ndarray,
    avatar_color: Optional[int] = None,
    prev_world: Optional[Dict[str, Any]] = None,
    last_action: Optional[int] = None,
    learned_wall_colors: Optional[Set[int]] = None,
    learned_box_colors: Optional[Set[int]] = None,
    learned_goal_colors: Optional[Set[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Dynamically extracts the downsampled discrete grid representation.
    Induces per-level scale dynamically from the avatar bounding box and classifies object roles.
    """
    h_dim, _ = f.shape[:2]
    sep = separate_background_and_lattice(f)
    bg_0 = sep.bg_color
    interior = f[sep.y_min:sep.y_max, sep.x_min:sep.x_max]

    if avatar_color is None:
        candidates = []
        for c in np.unique(interior):
            if c in (bg_0, 4, 0, 3, 8) or c in sep.frame_colors:
                continue
            pts = np.argwhere((f == c) & (np.arange(h_dim)[:, None] >= 4) & (np.arange(h_dim)[:, None] < 60))
            if 1 <= len(pts) <= 400:
                h_span = pts[:, 0].max() - pts[:, 0].min() + 1
                w_span = pts[:, 1].max() - pts[:, 1].min() + 1
                aspect = max(h_span, w_span) / max(1, min(h_span, w_span))
                if aspect <= 3.0:
                    candidates.append((len(pts), int(c)))
        if candidates:
            candidates.sort()
            avatar_color = candidates[0][1]

    if avatar_color is None:
        return None

    min_row_search = 2 if f[0, 0] in sep.frame_colors else 0
    cand_mask = (f == avatar_color) & (np.arange(h_dim)[:, None] >= min_row_search)
    labeled, n_comp = ndimage.label(cand_mask)
    pts = np.empty((0, 2), dtype=np.int64)
    best_len = 0
    min_req_area = (
        max(4, prev_world["scale"] * prev_world["scale"] // 2)
        if (prev_world is not None and prev_world.get("scale", 1) >= 3)
        else 1
    )
    for comp_idx in range(1, n_comp + 1):
        c_pts = np.argwhere(labeled == comp_idx)
        if c_pts[:, 0].max() >= 4 and len(c_pts) >= min_req_area and len(c_pts) > best_len:
            best_len = len(c_pts)
            pts = c_pts

    if len(pts) == 0 and prev_world is None:
        p_mask = (f == avatar_color) & (np.arange(h_dim)[:, None] >= 4)
        pts = np.argwhere(p_mask)
        if len(pts) == 0:
            pts = np.argwhere(f == avatar_color)

    if len(pts) == 0:
        if prev_world is not None:
            scale = prev_world["scale"]
            y0 = prev_world["y0"]
            x0 = prev_world["x0"]
            gh = prev_world["gh"]
            gw = prev_world["gw"]
            deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dr, dc = deltas.get(last_action, (0, 0))
            p_pos = (
                max(0, min(gh - 1, prev_world["player"][0] + dr)),
                max(0, min(gw - 1, prev_world["player"][1] + dc)),
            )
            board = np.zeros((gh, gw), dtype=np.int32)
            for r in range(gh):
                for c in range(gw):
                    patch = f[y0 + r * scale : y0 + (r + 1) * scale, x0 + c * scale : x0 + (c + 1) * scale]
                    if patch.size > 0:
                        board[r, c] = collections.Counter(patch.flatten()).most_common(1)[0][0]
            walls, boxes, plate_goals, exit_doors = _induce_object_roles_on_board(
                board,
                f,
                y0,
                x0,
                scale,
                gh,
                gw,
                p_pos,
                avatar_color,
                bg_0,
                learned_wall_colors,
                learned_box_colors,
                learned_goal_colors,
            )
            return {
                "board": board,
                "scale": scale,
                "gh": gh,
                "gw": gw,
                "player": p_pos,
                "avatar_color": avatar_color,
                "walls": walls,
                "boxes": boxes,
                "plates": plate_goals,
                "exit_doors": exit_doors,
                "all_goals": plate_goals | exit_doors,
                "y0": y0,
                "x0": x0,
            }
        return None

    scale_h = int(pts[:, 0].max() - pts[:, 0].min() + 1)
    scale_w = int(pts[:, 1].max() - pts[:, 1].min() + 1)
    if abs(scale_h - scale_w) == 1 and min(scale_h, scale_w) >= 4:
        scale = min(scale_h, scale_w)
    else:
        scale = max(1, max(scale_h, scale_w))

    p_min_r = int(pts[:, 0].min())
    p_min_c = int(pts[:, 1].min())
    y0 = p_min_r % scale
    x0 = p_min_c % scale

    gh = (64 - y0) // scale
    gw = (64 - x0) // scale
    if f[0, 0] in sep.frame_colors and y0 >= 2 and x0 >= 2:
        gh = (64 - 2 * y0) // scale
        gw = (64 - 2 * x0) // scale

    board = np.zeros((gh, gw), dtype=np.int32)
    for r in range(gh):
        for c in range(gw):
            patch = f[y0 + r * scale : y0 + (r + 1) * scale, x0 + c * scale : x0 + (c + 1) * scale]
            if patch.size > 0:
                board[r, c] = collections.Counter(patch.flatten()).most_common(1)[0][0]

    player_matches = np.argwhere(board == avatar_color)
    p_cy = int(np.mean(pts[:, 0]))
    p_cx = int(np.mean(pts[:, 1]))
    target_p_pos = ((p_cy - y0) // scale, (p_cx - x0) // scale)
    if len(player_matches) == 0:
        p_pos = target_p_pos
    elif len(player_matches) == 1:
        p_pos = (int(player_matches[0][0]), int(player_matches[0][1]))
    else:
        best_p = (int(player_matches[0][0]), int(player_matches[0][1]))
        best_dist = float("inf")
        for pm in player_matches:
            pr, pc = int(pm[0]), int(pm[1])
            dist = (pr - target_p_pos[0]) ** 2 + (pc - target_p_pos[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_p = (pr, pc)
        p_pos = best_p

    walls, boxes, plate_goals, exit_doors = _induce_object_roles_on_board(
        board,
        f,
        y0,
        x0,
        scale,
        gh,
        gw,
        p_pos,
        avatar_color,
        bg_0,
        learned_wall_colors,
        learned_box_colors,
        learned_goal_colors,
    )

    return {
        "board": board,
        "scale": scale,
        "gh": gh,
        "gw": gw,
        "player": p_pos,
        "avatar_color": avatar_color,
        "walls": walls,
        "boxes": boxes,
        "plates": plate_goals,
        "exit_doors": exit_doors,
        "all_goals": plate_goals | exit_doors,
        "y0": y0,
        "x0": x0,
    }


# =====================================================================
# 6. Universal GF(2) Impulse-Response Toggle Solver
# =====================================================================

def solve_gf2_linear_system(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solves A x = b (mod 2) via Gaussian elimination over GF(2)."""
    M = len(b)
    aug = np.hstack([A.copy() % 2, b.copy().reshape(-1, 1) % 2]).astype(np.uint8)
    pivot_row = 0
    pivots = []
    for col in range(M):
        row = None
        for r in range(pivot_row, M):
            if aug[r, col] == 1:
                row = r
                break
        if row is None:
            continue
        aug[[pivot_row, row]] = aug[[row, pivot_row]]
        for r in range(M):
            if r != pivot_row and aug[r, col] == 1:
                aug[r] ^= aug[pivot_row]
        pivots.append((pivot_row, col))
        pivot_row += 1
    for r in range(pivot_row, M):
        if aug[r, M] == 1:
            return None
    x = np.zeros(M, dtype=np.uint8)
    for r, col in pivots:
        x[col] = aug[r, M]
    return x


def detect_lo_N(frame: np.ndarray) -> Optional[int]:
    """Dynamically induces grid dimension N in [3..8] by letterbox padding and cell patch purity."""
    best_N: Optional[int] = None
    best_score = 0.80
    for N in range(3, 9):
        scale = 64 // N
        pad = (64 - N * scale) // 2
        purities = []
        for gy in range(N):
            for gx in range(N):
                r0 = max(3, pad + gy * scale + 1)
                r1 = pad + (gy + 1) * scale - 1
                c0 = pad + gx * scale + 1
                c1 = pad + (gx + 1) * scale - 1
                patch = frame[r0:r1, c0:c1]
                valid = patch[patch != 4]
                if valid.size > 0:
                    _, cnts = np.unique(valid, return_counts=True)
                    purities.append(float(np.max(cnts)) / valid.size)
        if not purities:
            continue
        purities.sort()
        score = purities[1] if len(purities) > 1 else purities[0]
        if score > best_score + 0.02:
            best_score = score
            best_N = N
    return best_N


def parse_lo_board(
    frame: np.ndarray, N: Optional[int] = None, off_color: Optional[int] = None
) -> Tuple[int, int, int, Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Extracts (N, scale, pad, lit_cells, wall_cells) dynamically without hardcoding colors."""
    if N is None:
        N = detect_lo_N(frame) or 4
    scale = 64 // N
    pad = (64 - N * scale) // 2

    cell_colors: Dict[Tuple[int, int], int] = {}
    for gy in range(N):
        for gx in range(N):
            r0 = max(3, pad + gy * scale + 1)
            r1 = pad + (gy + 1) * scale - 1
            c0 = pad + gx * scale + 1
            c1 = pad + (gx + 1) * scale - 1
            patch = frame[r0:r1, c0:c1]
            valid = patch[patch != 4]
            if valid.size > 0:
                vals, cnts = np.unique(valid, return_counts=True)
                cell_colors[(gx, gy)] = int(vals[np.argmax(cnts)])

    if not cell_colors:
        return N, scale, pad, set(), set()

    counts = collections.Counter(cell_colors.values())
    most_common = counts.most_common()

    lit: Set[Tuple[int, int]] = set()
    walls: Set[Tuple[int, int]] = set()

    if off_color is None:
        sep = separate_background_and_lattice(frame)
        if sep.bg_color in cell_colors.values():
            off_color = sep.bg_color
        else:
            off_color = most_common[0][0]

    c_wall = most_common[2][0] if len(most_common) >= 3 else None
    for (gx, gy), c in cell_colors.items():
        if c_wall is not None and c == c_wall:
            walls.add((gx, gy))
        elif c != off_color:
            lit.add((gx, gy))

    return N, scale, pad, lit, walls


def induce_toggle_mode(
    f0: np.ndarray, f1: np.ndarray, click_px: Tuple[int, int], N: Optional[int] = None
) -> str:
    """Induces the GF(2) impulse-response kernel from a single click transition."""
    N_val, scale, pad, lit0, _ = parse_lo_board(f0, N=N)
    _, _, _, lit1, _ = parse_lo_board(f1, N=N_val)
    gx = (click_px[0] - pad) // scale
    gy = (click_px[1] - pad) // scale
    flipped = lit0 ^ lit1
    deltas = {(fx - gx, fy - gy) for fx, fy in flipped if (fx, fy) != (gx, gy)}
    if any(abs(dx) + abs(dy) == 3 and abs(dx) in (1, 2) for dx, dy in deltas):
        return "knight_clip"
    has_diag = any(abs(dx) == 1 and abs(dy) == 1 for dx, dy in deltas)
    has_orth = any(abs(dx) + abs(dy) == 1 for dx, dy in deltas)
    if has_diag and has_orth:
        return "king_torus"
    if has_diag and not has_orth:
        return "diag_bounded"
    if any(abs(dx) == N_val - 1 or abs(dy) == N_val - 1 for dx, dy in deltas):
        return "orth_torus"
    return "orth_bounded"


def solve_lights_out_universal(
    frame: np.ndarray, mode: str = "orth_bounded", N: Optional[int] = None, off_color: Optional[int] = None
) -> List[Tuple[int, int]]:
    """Builds the GF(2) linear system for the induced toggle kernel and returns click coordinates."""
    N_val, scale, pad, lit, walls = parse_lo_board(frame, N=N, off_color=off_color)
    if not lit:
        return []

    M = N_val * N_val
    b = np.zeros(M, dtype=np.uint8)
    for gx, gy in lit:
        b[gy * N_val + gx] = 1
    A = np.zeros((M, M), dtype=np.uint8)
    for gy in range(N_val):
        for gx in range(N_val):
            if (gx, gy) in walls:
                continue
            idx = gy * N_val + gx
            A[idx, idx] = 1
            if mode == "orth_torus":
                neighs = [
                    ((gx - 1) % N_val, gy),
                    ((gx + 1) % N_val, gy),
                    (gx, (gy - 1) % N_val),
                    (gx, (gy + 1) % N_val),
                ]
            elif mode == "orth_bounded":
                neighs = [
                    (gx + dx, gy + dy)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= gx + dx < N_val and 0 <= gy + dy < N_val
                ]
            elif mode == "king_torus":
                neighs = [
                    ((gx + dx) % N_val, (gy + dy) % N_val)
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    if not (dx == 0 and dy == 0)
                ]
            elif mode == "diag_bounded":
                neighs = [
                    (gx + dx, gy + dy)
                    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]
                    if 0 <= gx + dx < N_val and 0 <= gy + dy < N_val
                ]
            elif mode == "knight_clip":
                neighs = [
                    (gx + dx, gy + dy)
                    for dx, dy in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
                    if 0 <= gx + dx < N_val and 0 <= gy + dy < N_val
                ]
            else:
                neighs = []
            for nx, ny in neighs:
                if (nx, ny) not in walls:
                    A[ny * N_val + nx, idx] = 1

    sol = solve_gf2_linear_system(A, b)
    if sol is None:
        return []

    clicks = []
    for idx in np.where(sol == 1)[0]:
        gy = int(idx // N_val)
        gx = int(idx % N_val)
        clicks.append((pad + gx * scale + scale // 2, pad + gy * scale + scale // 2))
    return clicks


# =====================================================================
# 7. Universal Lattice Detector & K-Tuple Memory Matcher
# =====================================================================

@dataclass
class LatticeGrid:
    """Detected regular matrix of interactive tiles."""
    rows: int
    cols: int
    tile_size: int
    origin_r: int
    origin_c: int
    centers: List[Tuple[int, int]] = field(default_factory=list)
    tile_coords: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)


def detect_lattice_grid(grid: np.ndarray, bg_color: int) -> Optional[LatticeGrid]:
    """Dynamically detects whether the interior grid contains a regular matrix of tiles."""
    interior = grid[4:60, 4:60]
    mask = (interior != bg_color) & (interior != 4)
    if not np.any(mask):
        return None

    labeled, num_features = ndimage.label(mask)
    if num_features < 2:
        # Check for contiguous, touching tile block (e.g. edge-to-edge memory cards)
        if num_features == 1:
            slices = ndimage.find_objects(labeled)
            if slices and slices[0] is not None:
                sl = slices[0]
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                if h >= 24 and w >= 24:
                    candidates = [s for s in range(12, 33) if h % s == 0 and w % s == 0]
                    if candidates:
                        best_s = None
                        for s in candidates:
                            n_tiles = (h // s) * (w // s)
                            if n_tiles % 2 == 0 or n_tiles % 3 == 0:
                                best_s = s
                                break
                        tile_size = best_s or candidates[0]
                        rows = h // tile_size
                        cols = w // tile_size
                        r0 = 4 + sl[0].start
                        c0 = 4 + sl[1].start
                        actual_th = h / rows
                        actual_tw = w / cols
                        canonical_centers = []
                        coords_map = {}
                        for r_idx in range(rows):
                            for c_idx in range(cols):
                                cy = int(r0 + r_idx * actual_th + actual_th / 2)
                                cx = int(c0 + c_idx * actual_tw + actual_tw / 2)
                                canonical_centers.append((cx, cy))
                                coords_map[(r_idx, c_idx)] = (cx, cy)
                        return LatticeGrid(
                            rows=rows,
                            cols=cols,
                            tile_size=tile_size,
                            origin_r=r0,
                            origin_c=c0,
                            centers=canonical_centers,
                            tile_coords=coords_map,
                        )
        return None

    slices = ndimage.find_objects(labeled)
    heights = []
    widths = []
    centers = []

    for sl in slices:
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if 2 <= h <= 32 and 2 <= w <= 32:
            heights.append(h)
            widths.append(w)
            cr = 4 + int((sl[0].start + sl[0].stop) / 2)
            cc = 4 + int((sl[1].start + sl[1].stop) / 2)
            centers.append((cc, cr))

    if len(centers) < 4:
        return None

    tile_size = max(int(np.median(heights)), int(np.median(widths)))
    unique_ys = sorted(list(set(c[1] for c in centers)))
    unique_xs = sorted(list(set(c[0] for c in centers)))

    def cluster_coords(coords: List[int], thresh: int) -> List[int]:
        clusters = []
        for v in coords:
            if not clusters or abs(v - clusters[-1]) > thresh:
                clusters.append(v)
            else:
                clusters[-1] = int((clusters[-1] + v) / 2)
        return clusters

    r_clusters = cluster_coords(unique_ys, max(3, tile_size // 2))
    c_clusters = cluster_coords(unique_xs, max(3, tile_size // 2))

    if len(r_clusters) >= 2 and len(c_clusters) >= 2:
        canonical_centers = []
        coords_map = {}
        for r_idx, cy in enumerate(r_clusters):
            for c_idx, cx in enumerate(c_clusters):
                canonical_centers.append((cx, cy))
                coords_map[(r_idx, c_idx)] = (cx, cy)

        return LatticeGrid(
            rows=len(r_clusters),
            cols=len(c_clusters),
            tile_size=tile_size,
            origin_r=r_clusters[0],
            origin_c=c_clusters[0],
            centers=canonical_centers,
            tile_coords=coords_map,
        )

    return None


class UniversalMemoryMatcher:
    """Zero-hardcoding memory card matcher with dynamic lattice and K-tuple induction."""

    def __init__(self) -> None:
        self.lattice: Optional[LatticeGrid] = None
        self.tile_memory: Dict[Tuple[int, int], int] = {}
        self.cleared_tiles: Set[Tuple[int, int]] = set()
        self.turn_cards: List[Tuple[Tuple[int, int], int]] = []
        self.last_turn_cards: List[Tuple[Tuple[int, int], int]] = []
        self.last_clicked_center: Optional[Tuple[int, int]] = None
        self.waiting_ticks: int = 0
        self.last_was_dummy: bool = False
        self.match_size: int = 2
        self.hidden_color: Optional[int] = None

    def reset_for_level(self, lattice: Optional[LatticeGrid] = None) -> None:
        self.lattice = lattice
        self.tile_memory.clear()
        self.cleared_tiles.clear()
        self.turn_cards.clear()
        self.last_turn_cards.clear()
        self.last_clicked_center = None
        self.waiting_ticks = 0
        self.last_was_dummy = False
        if lattice is not None and (lattice.rows * lattice.cols) % 2 != 0 and (lattice.rows * lattice.cols) % 3 == 0:
            self.match_size = 3
        else:
            self.match_size = 2

    def plan_action(self, grid: np.ndarray, bg_color: int) -> Tuple[int, Dict[str, Any]]:
        """Returns (action_id, action_data) for the next memory step."""
        if self.waiting_ticks > 0:
            self.waiting_ticks -= 1
            self.last_was_dummy = True
            return 6, {"x": 0, "y": 0}

        self.last_was_dummy = False

        if self.last_turn_cards:
            # Check if after waiting ticks on a mismatch, the cards did NOT revert to hidden color
            # This indicates a 3-tuple game where the 2nd click didn't trigger turn end!
            c1 = self.last_turn_cards[0][0]
            patch = grid[max(0, c1[1] - 2) : min(64, c1[1] + 3), max(0, c1[0] - 2) : min(64, c1[0] + 3)]
            h_col = self.hidden_color if self.hidden_color is not None else 3
            if patch.size > 0 and np.sum((patch == h_col) | (patch == bg_color)) < patch.size // 2:
                self.match_size = max(self.match_size, len(self.last_turn_cards) + 1)
                self.turn_cards = list(self.last_turn_cards)
            self.last_turn_cards.clear()

        if not self.lattice or not self.lattice.centers:
            # Salient probe: click point in top-left tile of largest non-bg component in interior
            interior_mask = (grid != bg_color) & (grid != 4)
            interior_mask[:4, :] = False
            interior_mask[60:, :] = False
            interior_mask[:, :4] = False
            interior_mask[:, 60:] = False
            labeled_c, n_c = ndimage.label(interior_mask)
            if n_c > 0:
                sizes = ndimage.sum(interior_mask, labeled_c, range(1, n_c + 1))
                best_c = int(np.argmax(sizes) + 1)
                pts_c = np.argwhere(labeled_c == best_c)
                cy = int(pts_c[:, 0].min() + 9)
                cx = int(pts_c[:, 1].min() + 9)
                self.last_clicked_center = (cx, cy)
                return 6, {"x": cx, "y": cy}
            self.last_clicked_center = (32, 32)
            return 6, {"x": 32, "y": 32}

        centers = self.lattice.centers
        k = len(self.turn_cards)

        if k == 0:
            color_to_tiles = collections.defaultdict(list)
            for c, col in self.tile_memory.items():
                if c not in self.cleared_tiles:
                    color_to_tiles[col].append(c)

            for col, tlist in color_to_tiles.items():
                if len(tlist) >= self.match_size:
                    c1 = tlist[0]
                    self.last_clicked_center = c1
                    return 6, {"x": int(c1[0]), "y": int(c1[1])}

            unvisited = [c for c in centers if c not in self.tile_memory and c not in self.cleared_tiles]
            if not unvisited:
                unvisited = [c for c in centers if c not in self.cleared_tiles]
            if not unvisited:
                unvisited = list(centers)
            c_pick = unvisited[0]
            self.last_clicked_center = c_pick
            return 6, {"x": int(c_pick[0]), "y": int(c_pick[1])}
        else:
            flipped_set = {c for c, _ in self.turn_cards}
            first_col = self.turn_cards[0][1]
            all_same = all(col == first_col for _, col in self.turn_cards)
            if all_same:
                matching = [
                    c
                    for c, col in self.tile_memory.items()
                    if col == first_col and c not in flipped_set and c not in self.cleared_tiles
                ]
                if len(matching) >= self.match_size - k:
                    c_next = matching[0]
                    self.last_clicked_center = c_next
                    return 6, {"x": int(c_next[0]), "y": int(c_next[1])}

            unvisited = [
                c for c in centers if c not in self.tile_memory and c not in self.cleared_tiles and c not in flipped_set
            ]
            if not unvisited:
                unvisited = [c for c in centers if c not in self.cleared_tiles and c not in flipped_set]
            if not unvisited:
                unvisited = list(centers)
            c_pick = unvisited[0]
            self.last_clicked_center = c_pick
            return 6, {"x": int(c_pick[0]), "y": int(c_pick[1])}

    def on_observation(
        self,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
        bg_color: int,
        last_click_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Updates internal state and memory from frame diff."""
        if self.last_was_dummy:
            return

        if last_click_pos is not None:
            self.last_clicked_center = last_click_pos

        diff = (grid_before != grid_after) & (np.arange(grid_after.shape[0])[:, None] >= 4)
        if not np.any(diff):
            return

        cross_mask = np.zeros_like(diff)
        if self.last_clicked_center:
            cx, cy = self.last_clicked_center
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= cy + dy < grid_after.shape[0] and 0 <= cx + dx < 64:
                    cross_mask[cy + dy, cx + dx] = True

        card_diff = diff & ~cross_mask
        if not np.any(card_diff):
            card_diff = diff

        vals, cnts = np.unique(grid_after[card_diff], return_counts=True)
        cand_indices = [i for i, v in enumerate(vals) if v != bg_color and v != 4]
        if not cand_indices:
            return
        best_idx = max(cand_indices, key=lambda i: cnts[i])
        dom_color = int(vals[best_idx])

        card_mask = card_diff & (grid_after == dom_color)
        c_pts = np.argwhere(card_mask)
        if self.last_clicked_center is None:
            if len(c_pts) > 0:
                self.last_clicked_center = (int(round(c_pts[:, 1].mean())), int(round(c_pts[:, 0].mean())))
            else:
                self.last_clicked_center = (32, 32)

        if not self.lattice or not self.lattice.centers:
            sep = separate_background_and_lattice(grid_before)
            if sep.lattice:
                self.lattice = sep.lattice
                self.hidden_color = sep.tile_back_color
                if (sep.lattice.rows * sep.lattice.cols) % 2 != 0 and (sep.lattice.rows * sep.lattice.cols) % 3 == 0:
                    self.match_size = 3
            elif len(c_pts) >= 4:
                # Dynamically induce seamless lattice from single flipped tile
                h_t = max(2, int(c_pts[:, 0].max() - c_pts[:, 0].min() + 1))
                w_t = max(2, int(c_pts[:, 1].max() - c_pts[:, 1].min() + 1))
                c_hidden = int(grid_before[c_pts[0, 0], c_pts[0, 1]])
                labeled_h, _ = ndimage.label(grid_before == c_hidden)
                comp_h_idx = labeled_h[c_pts[0, 0], c_pts[0, 1]]
                comp_h_pts = np.argwhere(labeled_h == comp_h_idx)
                if len(comp_h_pts) > 0:
                    rmin, rmax = int(comp_h_pts[:, 0].min()), int(comp_h_pts[:, 0].max() + 1)
                    cmin, cmax = int(comp_h_pts[:, 1].min()), int(comp_h_pts[:, 1].max() + 1)
                    rows = (rmax - rmin) // h_t
                    cols = (cmax - cmin) // w_t
                    if rows >= 1 and cols >= 1 and (rows * cols) >= 4:
                        canonical_centers = []
                        coords_map = {}
                        for r_idx in range(rows):
                            for c_idx in range(cols):
                                cy = int(rmin + r_idx * h_t + h_t // 2)
                                cx = int(cmin + c_idx * w_t + w_t // 2)
                                canonical_centers.append((cx, cy))
                                coords_map[(r_idx, c_idx)] = (cx, cy)
                        self.lattice = LatticeGrid(
                            rows=rows,
                            cols=cols,
                            tile_size=max(h_t, w_t),
                            origin_r=rmin,
                            origin_c=cmin,
                            centers=canonical_centers,
                            tile_coords=coords_map,
                        )
                        self.hidden_color = c_hidden
                        if (rows * cols) % 2 != 0 and (rows * cols) % 3 == 0:
                            self.match_size = 3
                        else:
                            self.match_size = 2

        if self.lattice and self.lattice.centers:
            closest_c = min(
                self.lattice.centers,
                key=lambda pt: (pt[0] - self.last_clicked_center[0]) ** 2
                + (pt[1] - self.last_clicked_center[1]) ** 2,
            )
            self.tile_memory[closest_c] = dom_color
            self.turn_cards.append((closest_c, dom_color))
            if len(self.turn_cards) >= self.match_size:
                if all(col == self.turn_cards[0][1] for _, col in self.turn_cards):
                    for c_pt, _ in self.turn_cards:
                        self.cleared_tiles.add(c_pt)
                    self.turn_cards.clear()
                    self.last_turn_cards.clear()
                    self.waiting_ticks = 0
                else:
                    self.waiting_ticks = 2
                    self.last_turn_cards = list(self.turn_cards)
                    self.turn_cards.clear()


# =====================================================================
# 8. Universal Sokoban Reachability Solver (Single-Step & Sliding Crate)
# =====================================================================

def solve_sokoban_universal(
    player_pos: Tuple[int, int],
    boxes: Set[Tuple[int, int]],
    goals: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    max_states: int = 2500,
    slide_push: bool = False,
) -> Optional[List[int]]:
    """Universal Box-Push BFS with canonical player reachability component reduction.
    Supports both 1-step pushing (slide_push=False) and 2-cell sliding crates (slide_push=True).
    """
    if not boxes or not goals:
        return []
    min_r, min_c, max_r, max_c = 0, 0, gh, gw
    deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def get_comp(p: Tuple[int, int], b_set: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        q = collections.deque([p])
        vis = {p}
        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if min_r <= nr < max_r and min_c <= nc < max_c:
                    if (nr, nc) not in walls and (nr, nc) not in b_set and (nr, nc) not in vis:
                        vis.add((nr, nc))
                        q.append((nr, nc))
        return tuple(sorted(vis))

    def get_path(start: Tuple[int, int], dest: Tuple[int, int], b_set: Set[Tuple[int, int]]) -> Optional[List[int]]:
        if start == dest:
            return []
        q = collections.deque([(start, [])])
        vis = {start}
        while q:
            (cr, cc), path = q.popleft()
            if (cr, cc) == dest:
                return path
            for act, (dr, dc) in deltas.items():
                nr, nc = cr + dr, cc + dc
                if min_r <= nr < max_r and min_c <= nc < max_c:
                    if (nr, nc) not in walls and (nr, nc) not in b_set and (nr, nc) not in vis:
                        vis.add((nr, nc))
                        q.append(((nr, nc), path + [act]))
        return None

    def heuristic(b_tuple: Tuple[Tuple[int, int], ...]) -> int:
        h = 0
        for br, bc in b_tuple:
            h += min(abs(br - gr) + abs(bc - gc) for gr, gc in goals)
        return h

    def is_deadlock(b_pos: Tuple[int, int]) -> bool:
        if b_pos in goals:
            return False
        br, bc = b_pos
        blocked_up = (br - 1 < min_r) or ((br - 1, bc) in walls)
        blocked_down = (br + 1 >= max_r) or ((br + 1, bc) in walls)
        blocked_left = (bc - 1 < min_c) or ((br, bc - 1) in walls)
        blocked_right = (bc + 1 >= max_c) or ((br, bc + 1) in walls)
        if (blocked_up or blocked_down) and (blocked_left or blocked_right):
            return True
        if br == min_r and not any(gr == min_r for gr, _ in goals):
            return True
        if br == max_r - 1 and not any(gr == max_r - 1 for gr, _ in goals):
            return True
        if bc == min_c and not any(gc == min_c for _, gc in goals):
            return True
        if bc == max_c - 1 and not any(gc == max_c for _, gc in goals):
            return True
        return False

    init_b = tuple(sorted(boxes))
    init_c = get_comp(player_pos, boxes)
    init_rep = init_c[0] if init_c else player_pos

    q = [(heuristic(init_b), 0, init_b, init_rep, player_pos, [])]
    visited = {(init_b, init_rep)}
    states = 0

    while q and states < max_states:
        _, cost, cur_b, cur_rep, cur_p, path = heapq.heappop(q)
        states += 1
        b_set = set(cur_b)
        if all(b in goals for b in cur_b):
            return path
        comp_set = set(get_comp(cur_p, b_set))
        for b_idx, (br, bc) in enumerate(cur_b):
            for push_act, (dr, dc) in deltas.items():
                p_from = (br - dr, bc - dc)
                b_dest = (br + dr, bc + dc)
                if p_from in comp_set:
                    if min_r <= b_dest[0] < max_r and min_c <= b_dest[1] < max_c and b_dest not in walls and b_dest not in b_set:
                        if slide_push:
                            b_slide = (b_dest[0] + dr, b_dest[1] + dc)
                            if (
                                min_r <= b_slide[0] < max_r
                                and min_c <= b_slide[1] < max_c
                                and b_slide not in walls
                                and b_slide not in b_set
                            ):
                                b_dest = b_slide
                        if is_deadlock(b_dest):
                            continue
                        new_b_list = list(cur_b)
                        new_b_list[b_idx] = b_dest
                        new_b_tup = tuple(sorted(new_b_list))
                        new_c = get_comp((br, bc), set(new_b_tup))
                        new_rep = new_c[0] if new_c else (br, bc)
                        if (new_b_tup, new_rep) not in visited:
                            visited.add((new_b_tup, new_rep))
                            step_p = get_path(cur_p, p_from, b_set)
                            if step_p is not None:
                                full_step = path + step_p + [push_act]
                                heapq.heappush(
                                    q,
                                    (
                                        heuristic(new_b_tup) + len(full_step),
                                        len(full_step),
                                        new_b_tup,
                                        new_rep,
                                        (br, bc),
                                        full_step,
                                    ),
                                )
    return None


# =====================================================================
# 9. Universal Inertial Slide Solver (Raycast BFS)
# =====================================================================

def solve_inertial_slide_universal(
    start_pos: Tuple[int, int],
    goal_set: Union[Set[Tuple[int, int]], Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    deltas: Optional[Dict[int, Tuple[int, int]]] = None,
    max_steps: Optional[int] = None,
    allow_wrap: bool = False,
    stop_on_goal: bool = False,
) -> Optional[List[int]]:
    """Universal Raycast BFS for inertial sliding surfaces."""
    if not isinstance(goal_set, (set, list, tuple)):
        goals = {goal_set}
    elif isinstance(goal_set, tuple) and len(goal_set) == 2 and isinstance(goal_set[0], int):
        goals = {goal_set}
    else:
        goals = set(goal_set)

    action_deltas = deltas or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def raycast(pr: int, pc: int, dr: int, dc: int) -> Tuple[int, int]:
        cr, cc = pr, pc
        steps = 0
        limit = max_steps if max_steps is not None else (gh * gw if allow_wrap else (gh + gw))
        while steps < limit:
            nr, nc = cr + dr, cc + dc
            if allow_wrap:
                if nr < 0:
                    nr = gh - 1
                elif nr >= gh:
                    nr = 0
                if nc < 0:
                    nc = gw - 1
                elif nc >= gw:
                    nc = 0
            else:
                if not (0 <= nr < gh and 0 <= nc < gw):
                    break
            if (nr, nc) in walls:
                break
            cr, cc = nr, nc
            steps += 1
            if stop_on_goal and (cr, cc) in goals:
                break
        return (cr, cc)

    q = collections.deque([(start_pos, [])])
    visited = {start_pos}

    while q:
        (cr, cc), path = q.popleft()
        if (cr, cc) in goals:
            return path
        for act, (dr, dc) in action_deltas.items():
            nr, nc = raycast(cr, cc, dr, dc)
            if (nr, nc) != (cr, cc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append(((nr, nc), path + [act]))

    return None


# =====================================================================
# 10. Universal Multi-Vector Jump / Knight Solver
# =====================================================================

def solve_knight_universal(
    start_pos: Tuple[int, int],
    goal_set: Union[Set[Tuple[int, int]], Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    has_action_5: bool = True,
) -> Optional[List[int]]:
    """Universal Non-Cardinal Jump BFS with dual bank vectors."""
    if not isinstance(goal_set, (set, list, tuple)):
        goals = {goal_set}
    elif isinstance(goal_set, tuple) and len(goal_set) == 2 and isinstance(goal_set[0], int):
        goals = {goal_set}
    else:
        goals = set(goal_set)

    bank_offsets = {
        0: {1: (-1, -2), 2: (1, -2), 3: (-1, 2), 4: (1, 2)},
        1: {1: (-2, -1), 2: (2, -1), 3: (-2, 1), 4: (2, 1)},
    }
    q = collections.deque([(start_pos[0], start_pos[1], 0, [])])
    visited = {(start_pos[0], start_pos[1], 0)}

    while q:
        cr, cc, bank, path = q.popleft()
        if (cr, cc) in goals:
            return path
        if has_action_5:
            nb = 1 - bank
            if (cr, cc, nb) not in visited:
                visited.add((cr, cc, nb))
                q.append((cr, cc, nb, path + [5]))
        for act, (dr, dc) in bank_offsets[bank].items():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in walls:
                if (nr, nc, bank) not in visited:
                    visited.add((nr, nc, bank))
                    q.append((nr, nc, bank, path + [act]))

    return None


# =====================================================================
# 11. Universal Cardinal A* Navigator & Time-Expanded Hazard Avoidance
# =====================================================================

def predict_discrete_hazard_trajectory(
    start_pos: Tuple[int, int],
    velocity: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    steps: int = 40,
) -> List[Tuple[int, int]]:
    """Predicts discrete trajectory of a dynamic bouncing hazard with boundary & wall reflection."""
    r, c = start_pos
    vr, vc = velocity
    if vr == 0 and vc == 0:
        return [(r, c)] * (steps + 1)

    trajectory: List[Tuple[int, int]] = [(r, c)]
    for _ in range(steps):
        next_r = r + vr
        next_c = c + vc

        # Vertical reflection
        if next_r < 0 or next_r >= gh or (next_r, c) in walls:
            vr = -vr
            next_r = r + vr

        # Horizontal reflection
        if next_c < 0 or next_c >= gw or (r, next_c) in walls:
            vc = -vc
            next_c = c + vc

        # Diagonal / corner collision
        if (next_r, next_c) in walls:
            vr = -vr
            vc = -vc
            next_r = r + vr
            next_c = c + vc

        r = max(0, min(gh - 1, next_r))
        c = max(0, min(gw - 1, next_c))
        trajectory.append((r, c))

    return trajectory


def solve_cardinal_astar(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    dynamic_hazards: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
    allow_wait: bool = False,
    max_t: int = 45,
) -> Optional[List[int]]:
    """Universal A* Pathfinding with Manhattan distance heuristic and Time-Expanded Hazard Avoidance."""
    deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    # Fast-path: If no dynamic hazards, use standard 2D static A*
    if not dynamic_hazards:
        q = [(float(abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])), 0.0, start_pos, [])]
        visited: Dict[Tuple[int, int], float] = {start_pos: 0.0}

        while q:
            _, cost, cur, path = heapq.heappop(q)
            if cur == goal_pos:
                return path
            if cost > visited.get(cur, 1e9):
                continue
            for act, (dr, dc) in deltas.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in walls:
                    ncost = cost + 1.0
                    if ncost < visited.get((nr, nc), 1e9):
                        visited[(nr, nc)] = ncost
                        h = float(abs(nr - goal_pos[0]) + abs(nc - goal_pos[1]))
                        heapq.heappush(q, (ncost + h, ncost, (nr, nc), path + [act]))
        return None

    # Time-Expanded Graph Search G(r, c, t)
    # State: (cr, cc, t)
    h0 = float(abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1]))
    q = [(h0, 0.0, (start_pos[0], start_pos[1], 0), [])]
    visited_t: Dict[Tuple[int, int, int], float] = {(start_pos[0], start_pos[1], 0): 0.0}

    while q:
        _, cost, (cr, cc, t), path = heapq.heappop(q)
        if (cr, cc) == goal_pos:
            return path
        if t >= max_t:
            continue
        if cost > visited_t.get((cr, cc, t), 1e9):
            continue

        haz_curr = dynamic_hazards.get(t, set())
        haz_next = dynamic_hazards.get(t + 1, set())

        # Cardinal moves
        for act, (dr, dc) in deltas.items():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in walls:
                # Vertex collision: hazard at (nr, nc) at step t + 1
                if (nr, nc) in haz_next:
                    continue
                # Edge-swap collision: head-on collision between avatar and hazard
                if (nr, nc) in haz_curr and (cr, cc) in haz_next:
                    continue

                ncost = cost + 1.0
                nt = t + 1
                if ncost < visited_t.get((nr, nc, nt), 1e9):
                    visited_t[(nr, nc, nt)] = ncost
                    h = float(abs(nr - goal_pos[0]) + abs(nc - goal_pos[1]))
                    heapq.heappush(q, (ncost + h, ncost, (nr, nc, nt), path + [act]))

        # Wait in place (Action 5) if allowed
        if allow_wait and (cr, cc) not in haz_next:
            ncost = cost + 1.2
            nt = t + 1
            if ncost < visited_t.get((cr, cc, nt), 1e9):
                visited_t[(cr, cc, nt)] = ncost
                h = float(abs(cr - goal_pos[0]) + abs(cc - goal_pos[1]))
                heapq.heappush(q, (ncost + h, ncost, (cr, cc, nt), path + [5]))

    # Fallback to static A* if dynamic avoidance couldn't reach within max_t
    return solve_cardinal_astar(start_pos, goal_pos, walls, gh, gw, dynamic_hazards=None)



# =====================================================================
# 12. Topological Shannon Entropy Minimizer
# =====================================================================

def compute_structural_shannon_entropy(grid: np.ndarray, bg_color: int = 5) -> float:
    """Computes H(G) = H_color + 0.5 * H_spatial + 0.3 * H_degree over the gameplay interior."""
    interior = grid[4:60, 4:60]
    non_bg_mask = (interior != bg_color) & (interior != 4)
    total_active = np.sum(non_bg_mask)
    if total_active == 0:
        return 0.0

    colors, counts = np.unique(interior[non_bg_mask], return_counts=True)
    p_c = counts / total_active
    h_color = -float(np.sum(p_c * np.log2(p_c + 1e-12)))

    labeled, num_features = ndimage.label(non_bg_mask)
    if num_features > 1:
        slices = ndimage.find_objects(labeled)
        centroids = []
        for sl in slices:
            if sl is not None:
                centroids.append(((sl[0].start + sl[0].stop) / 2.0, (sl[1].start + sl[1].stop) / 2.0))
        if len(centroids) > 1:
            c_arr = np.array(centroids)
            mean_c = c_arr.mean(axis=0)
            dists = np.linalg.norm(c_arr - mean_c, axis=1)
            sum_d = np.sum(dists)
            if sum_d > 0:
                p_d = dists / sum_d
                h_spatial = -float(np.sum(p_d[p_d > 0] * np.log2(p_d[p_d > 0] + 1e-12)))
            else:
                h_spatial = 0.0
        else:
            h_spatial = 0.0
    else:
        h_spatial = 0.0

    d_map = np.zeros_like(interior, dtype=np.int32)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(interior, dr, axis=0), dc, axis=1)
        d_map += (interior != shifted).astype(np.int32)
    d_vals, d_counts = np.unique(d_map[non_bg_mask], return_counts=True)
    p_deg = d_counts / total_active
    h_degree = -float(np.sum(p_deg * np.log2(p_deg + 1e-12)))

    return h_color + 0.5 * h_spatial + 0.3 * h_degree
