"""
Test Suite & Implementation Prototype for Universal Semantic Delta & World Model Engine.
"""

from __future__ import annotations

import collections
from collections import Counter, defaultdict, deque
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import ndimage


# =====================================================================
# 1. Fundamental Enums & Data Structures
# =====================================================================

class FunctionalRole(str, Enum):
    MOVER = "MOVER"                      # Controllable avatar or cursor
    CO_MOVER = "CO_MOVER"                # Entity moving synchronously or following mover
    OBSTACLE = "OBSTACLE"                # Static entity that blocks mover
    PUSHABLE = "PUSHABLE"                # Entity displaced by mover entering its cell
    SWITCH_TOGGLE = "SWITCH_TOGGLE"      # Entity whose local state changes upon interaction/click
    COLLECTOR_CONSUMABLE = "COLLECTOR_CONSUMABLE"  # Entity vanishing when mover enters its cell
    HAZARD = "HAZARD"                    # Entity whose contact causes reset or game over
    HUD_INDICATOR = "HUD_INDICATOR"      # Perimeter/border cells correlating with steps/budget
    PASSABLE_TERRAIN = "PASSABLE_TERRAIN"# Non-blocking floor/background
    UNKNOWN = "UNKNOWN"                  # Role unconfirmed


class KinematicModality(str, Enum):
    CARDINAL_1STEP = "CARDINAL_1STEP"    # Standard 1-cell step (Up, Down, Left, Right)
    MULTI_CELL_STRIDE = "MULTI_CELL_STRIDE" # Fixed k-cell hops (k > 1)
    KNIGHT_JUMP = "KNIGHT_JUMP"          # Knight-move leaps (L-shapes: 2x1 or 1x2)
    INERTIAL_SLIDE = "INERTIAL_SLIDE"    # Slides along direction until obstacle or boundary
    CLICK_TARGETING = "CLICK_TARGETING"  # Cursor or coordinate-based interaction (Action 6)


DIRECTION_DELTAS: Dict[int, Tuple[int, int]] = {
    1: (-1, 0),  # UP
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
    5: (0, 0),   # ACTION_USE / INTERACT
}

KNIGHT_DELTAS: Dict[int, Tuple[int, int]] = {
    1: (-2, -1),
    2: (2, -1),
    3: (-2, 1),
    4: (2, 1),
}


@dataclass
class EntityState:
    """State snapshot of an active or static entity."""
    entity_id: int
    role: FunctionalRole
    color: int
    coords: np.ndarray  # (N, 2) int array
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    centroid: Tuple[float, float]
    mass: int
    shape_hash: str
    velocity: Tuple[int, int] = (0, 0)
    state_val: int = 0
    confidence: float = 0.0
    is_hud: bool = False

    def copy(self) -> "EntityState":
        return EntityState(
            entity_id=self.entity_id,
            role=self.role,
            color=self.color,
            coords=np.copy(self.coords),
            bbox=self.bbox,
            centroid=self.centroid,
            mass=self.mass,
            shape_hash=self.shape_hash,
            velocity=self.velocity,
            state_val=self.state_val,
            confidence=self.confidence,
            is_hud=self.is_hud,
        )


@dataclass
class FrameDelta:
    """Complete semantic difference report between consecutive frames."""
    step_idx: int
    action: int
    action_data: Dict[str, Any]
    pre_grid: np.ndarray
    post_grid: np.ndarray
    diff_mask: np.ndarray
    vacated_mask: np.ndarray
    occupied_mask: np.ndarray
    canvas_bg_color: int
    field_bg_color: int
    arena_bounds: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
    hud_mutations: List[Tuple[int, int, int, int]]  # (r, c, old_c, new_c)
    field_mutations: List[Tuple[int, int, int, int]]  # (r, c, old_c, new_c)
    displacements: Dict[int, Tuple[int, int]]  # entity_id -> (dr, dc)
    vanished_entities: List[int]  # entity_ids
    spawned_entities: List[EntityState]
    state_mutations: Dict[int, Tuple[int, int]]  # entity_id -> (old_color, new_color)
    is_reset: bool
    level_cleared: bool


# =====================================================================
# 2. Universal Arena & Background Separator
# =====================================================================

class ArenaBackgroundSeparator:
    """Mathematical separation of outer canvas bezel, field background, and active play area."""

    @staticmethod
    def detect_arena(grid: np.ndarray, max_margin: int = 6) -> Tuple[int, int, int, int, int, int, Set[int]]:
        """Identifies (r_min, r_max, c_min, c_max, canvas_bg, field_bg, frame_colors)."""
        H, W = grid.shape[:2]
        frame_colors: Set[int] = set()

        def is_bezel_line(line: np.ndarray) -> Tuple[bool, int]:
            vals, cnts = np.unique(line, return_counts=True)
            dom_idx = int(np.argmax(cnts))
            dom_c = int(vals[dom_idx])
            purity = float(cnts[dom_idx]) / float(line.size)
            return (purity >= 0.85), dom_c

        r_min = 0
        for r in range(min(max_margin, H // 4)):
            is_bz, c = is_bezel_line(grid[r, :])
            if is_bz:
                frame_colors.add(c)
                r_min = r + 1
            else:
                break

        r_max = H
        for r in range(H - 1, max(H - 1 - max_margin, 3 * H // 4), -1):
            is_bz, c = is_bezel_line(grid[r, :])
            if is_bz:
                frame_colors.add(c)
                r_max = r
            else:
                break

        c_min = 0
        for col in range(min(max_margin, W // 4)):
            is_bz, c = is_bezel_line(grid[:, col])
            if is_bz:
                frame_colors.add(c)
                c_min = col + 1
            else:
                break

        c_max = W
        for col in range(W - 1, max(W - 1 - max_margin, 3 * W // 4), -1):
            is_bz, c = is_bezel_line(grid[:, col])
            if is_bz:
                frame_colors.add(c)
                c_max = col
            else:
                break

        perimeter_pixels = np.concatenate([
            grid[0, :], grid[H - 1, :], grid[:, 0], grid[:, W - 1]
        ])
        vals, cnts = np.unique(perimeter_pixels, return_counts=True)
        canvas_bg = int(vals[np.argmax(cnts)])

        interior = grid[r_min:r_max, c_min:c_max]
        if interior.size > 0:
            int_vals, int_cnts = np.unique(interior, return_counts=True)
            field_bg = int(int_vals[np.argmax(int_cnts)])
        else:
            field_bg = canvas_bg

        return r_min, r_max, c_min, c_max, canvas_bg, field_bg, frame_colors


# =====================================================================
# 3. Universal Semantic Delta Engine
# =====================================================================

class UniversalSemanticDeltaEngine:
    """Computes exact connected component diffs, displacement vectors, and induces functional roles."""

    def __init__(self) -> None:
        self.arena_separator = ArenaBackgroundSeparator()
        self.role_evidence: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "mover_score": 0.0,
            "co_mover_score": 0.0,
            "obstacle_score": 0.0,
            "pushable_score": 0.0,
            "switch_score": 0.0,
            "consumable_score": 0.0,
            "hazard_score": 0.0,
            "hud_score": 0.0,
            "total_observations": 0,
            "directional_moves": 0,
            "action_displacements": defaultdict(list),
        })
        self.discovered_roles: Dict[int, FunctionalRole] = {}
        self.confirmed_mover_color: Optional[int] = None
        self.confirmed_co_mover_colors: Set[int] = set()

    def reset_for_new_game(self) -> None:
        self.role_evidence.clear()
        self.discovered_roles.clear()
        self.confirmed_mover_color = None
        self.confirmed_co_mover_colors.clear()

    @staticmethod
    def compute_shape_hash(coords: np.ndarray) -> str:
        """Computes translation-invariant Blake2b hash of normalized component coordinates."""
        if len(coords) == 0:
            return "empty"
        min_r, min_c = coords.min(axis=0)
        norm_coords = coords - [min_r, min_c]
        sorted_coords = norm_coords[np.lexsort((norm_coords[:, 1], norm_coords[:, 0]))]
        return hashlib.blake2b(sorted_coords.tobytes(), digest_size=8).hexdigest()

    def extract_components(
        self,
        grid: np.ndarray,
        field_bg: int,
        canvas_bg: int,
        arena_bounds: Tuple[int, int, int, int],
    ) -> List[EntityState]:
        """Extracts color-separated 4-connected components excluding background."""
        r_min, r_max, c_min, c_max = arena_bounds
        H, W = grid.shape[:2]
        entities: List[EntityState] = []
        cid = 1

        for c_val in np.unique(grid):
            if c_val == field_bg or c_val == canvas_bg:
                continue
            c_mask = (grid == c_val)
            labeled, n_comp = ndimage.label(c_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

            for k in range(1, n_comp + 1):
                coords = np.argwhere(labeled == k)
                if len(coords) == 0:
                    continue
                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
                shape_h = self.compute_shape_hash(coords)

                is_hud = bool(min_r < r_min or max_r >= r_max or min_c < c_min or max_c >= c_max)
                role = self.discovered_roles.get(int(c_val), FunctionalRole.UNKNOWN)
                if is_hud:
                    role = FunctionalRole.HUD_INDICATOR

                entities.append(EntityState(
                    entity_id=cid,
                    role=role,
                    color=int(c_val),
                    coords=coords,
                    bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
                    centroid=centroid,
                    mass=len(coords),
                    shape_hash=shape_h,
                    is_hud=is_hud,
                ))
                cid += 1

        return entities

    def match_entities(
        self,
        prev_entities: List[EntityState],
        curr_entities: List[EntityState],
    ) -> Tuple[Dict[int, Tuple[int, int]], List[int], List[EntityState]]:
        """Matches entities across frames and computes centroid displacement vectors dp = (dr, dc)."""
        matched: Dict[int, int] = {}
        unmatched_prev = set(range(len(prev_entities)))
        unmatched_curr = set(range(len(curr_entities)))

        candidates = []
        for i, ep in enumerate(prev_entities):
            for j, ec in enumerate(curr_entities):
                color_cost = 0.0 if ep.color == ec.color else 1000.0
                mass_cost = abs(ep.mass - ec.mass) * 10.0
                shape_cost = 0.0 if ep.shape_hash == ec.shape_hash else 40.0
                dist = math.hypot(ep.centroid[0] - ec.centroid[0], ep.centroid[1] - ec.centroid[1])
                cost = color_cost + mass_cost + shape_cost + dist
                candidates.append((cost, i, j, dist))

        candidates.sort(key=lambda x: x[0])
        for cost, i, j, dist in candidates:
            if i in unmatched_prev and j in unmatched_curr:
                if prev_entities[i].color == curr_entities[j].color or dist <= 5.0:
                    matched[i] = j
                    unmatched_prev.remove(i)
                    unmatched_curr.remove(j)

        displacements: Dict[int, Tuple[int, int]] = {}
        for i, j in matched.items():
            ep = prev_entities[i]
            ec = curr_entities[j]
            dr = int(round(ec.centroid[0] - ep.centroid[0]))
            dc = int(round(ec.centroid[1] - ep.centroid[1]))
            displacements[ep.entity_id] = (dr, dc)

        vanished = [prev_entities[i].entity_id for i in unmatched_prev]
        spawned = [curr_entities[j] for j in unmatched_curr]

        return displacements, vanished, spawned

    def process_transition(
        self,
        pre_grid: np.ndarray,
        post_grid: np.ndarray,
        action: int,
        action_data: Optional[Dict[str, Any]] = None,
        step_idx: int = 0,
        level_cleared: bool = False,
        initial_grid: Optional[np.ndarray] = None,
    ) -> FrameDelta:
        """Processes frame transition and updates empirical role induction."""
        if action_data is None:
            action_data = {}

        r_min, r_max, c_min, c_max, canvas_bg, field_bg, frame_colors = self.arena_separator.detect_arena(pre_grid)
        diff_mask = (pre_grid != post_grid)
        vacated_mask = diff_mask & (post_grid == field_bg)
        occupied_mask = diff_mask & (post_grid != field_bg)

        is_reset = False
        if initial_grid is not None and np.array_equal(post_grid, initial_grid) and not np.array_equal(pre_grid, post_grid):
            is_reset = True

        prev_entities = self.extract_components(pre_grid, field_bg, canvas_bg, (r_min, r_max, c_min, c_max))
        curr_entities = self.extract_components(post_grid, field_bg, canvas_bg, (r_min, r_max, c_min, c_max))

        displacements, vanished, spawned = self.match_entities(prev_entities, curr_entities)

        hud_mutations: List[Tuple[int, int, int, int]] = []
        field_mutations: List[Tuple[int, int, int, int]] = []
        for r, c in np.argwhere(diff_mask):
            old_c = int(pre_grid[r, c])
            new_c = int(post_grid[r, c])
            if r < r_min or r >= r_max or c < c_min or c >= c_max:
                hud_mutations.append((r, c, old_c, new_c))
            else:
                field_mutations.append((r, c, old_c, new_c))

        state_mutations: Dict[int, Tuple[int, int]] = {}
        for ep in prev_entities:
            if ep.entity_id in displacements:
                dr, dc = displacements[ep.entity_id]
                if dr == 0 and dc == 0:
                    old_c = ep.color
                    new_colors = [int(post_grid[r, c]) for r, c in ep.coords if 0 <= r < post_grid.shape[0] and 0 <= c < post_grid.shape[1]]
                    if new_colors:
                        new_c = Counter(new_colors).most_common(1)[0][0]
                        if new_c != old_c and new_c != field_bg:
                            state_mutations[ep.entity_id] = (old_c, new_c)

        delta = FrameDelta(
            step_idx=step_idx,
            action=action,
            action_data=action_data,
            pre_grid=pre_grid,
            post_grid=post_grid,
            diff_mask=diff_mask,
            vacated_mask=vacated_mask,
            occupied_mask=occupied_mask,
            canvas_bg_color=canvas_bg,
            field_bg_color=field_bg,
            arena_bounds=(r_min, r_max, c_min, c_max),
            hud_mutations=hud_mutations,
            field_mutations=field_mutations,
            displacements=displacements,
            vanished_entities=vanished,
            spawned_entities=spawned,
            state_mutations=state_mutations,
            is_reset=is_reset,
            level_cleared=level_cleared,
        )

        self._update_role_evidence(delta, prev_entities, curr_entities)
        return delta

    def _update_role_evidence(
        self,
        delta: FrameDelta,
        prev_entities: List[EntityState],
        curr_entities: List[EntityState],
    ) -> None:
        """Accumulates statistical evidence for functional role classification."""
        nom_dr, nom_dc = DIRECTION_DELTAS.get(delta.action, (0, 0))
        is_directional = delta.action in (1, 2, 3, 4)

        prev_map = {e.entity_id: e for e in prev_entities}
        curr_map = {e.entity_id: e for e in curr_entities}

        # 1. MOVER discovery & voting
        mover_candidate: Optional[EntityState] = None
        best_mover_match = 0.0

        for ep in prev_entities:
            if ep.is_hud:
                self.role_evidence[ep.color]["hud_score"] += 2.0
                continue

            self.role_evidence[ep.color]["total_observations"] += 1
            dr, dc = delta.displacements.get(ep.entity_id, (0, 0))

            if is_directional:
                self.role_evidence[ep.color]["directional_moves"] += 1
                self.role_evidence[ep.color]["action_displacements"][delta.action].append((dr, dc))

                if (dr, dc) != (0, 0):
                    dot = dr * nom_dr + dc * nom_dc
                    if dot > 0:
                        self.role_evidence[ep.color]["mover_score"] += 2.0
                    elif dr != 0 or dc != 0:
                        self.role_evidence[ep.color]["mover_score"] += 1.0

            if self.role_evidence[ep.color]["mover_score"] > best_mover_match:
                best_mover_match = self.role_evidence[ep.color]["mover_score"]
                mover_candidate = ep

        if mover_candidate is not None:
            self.confirmed_mover_color = mover_candidate.color
            self.discovered_roles[mover_candidate.color] = FunctionalRole.MOVER

        # 2. CO_MOVER evidence
        if self.confirmed_mover_color is not None:
            mover_ep = next((e for e in prev_entities if e.color == self.confirmed_mover_color), None)
            if mover_ep is not None:
                mover_dr, mover_dc = delta.displacements.get(mover_ep.entity_id, (0, 0))
                for ep in prev_entities:
                    if ep.color != self.confirmed_mover_color and not ep.is_hud:
                        dr, dc = delta.displacements.get(ep.entity_id, (0, 0))
                        if (mover_dr, mover_dc) != (0, 0):
                            if (dr, dc) == (mover_dr, mover_dc):
                                self.role_evidence[ep.color]["co_mover_score"] += 2.0
                            elif dr == 0 and dc == 0:
                                self.role_evidence[ep.color]["co_mover_score"] -= 0.5

        # 3. OBSTACLE & PUSHABLE evidence via Collision Ray
        if self.confirmed_mover_color is not None and is_directional:
            mover_ep = next((e for e in prev_entities if e.color == self.confirmed_mover_color), None)
            if mover_ep is not None:
                mover_dr, mover_dc = delta.displacements.get(mover_ep.entity_id, (0, 0))
                # Trace collision ray forward from mover
                front_pos = (int(round(mover_ep.centroid[0] + nom_dr)), int(round(mover_ep.centroid[1] + nom_dc)))
                front_ep = None
                for ep in prev_entities:
                    if ep.color != self.confirmed_mover_color and not ep.is_hud:
                        if tuple(front_pos) in set(map(tuple, ep.coords)):
                            front_ep = ep
                            break

                if front_ep is not None:
                    front_dr, front_dc = delta.displacements.get(front_ep.entity_id, (0, 0))
                    if (front_dr, front_dc) == (nom_dr, nom_dc) and (mover_dr, mover_dc) == (nom_dr, nom_dc):
                        # PUSHABLE!
                        self.role_evidence[front_ep.color]["pushable_score"] += 4.0
                    elif mover_dr == 0 and mover_dc == 0 and front_dr == 0 and front_dc == 0:
                        # Collision: check if there is an entity behind front_ep (chain push)
                        behind_pos = (front_pos[0] + nom_dr, front_pos[1] + nom_dc)
                        behind_ep = None
                        for ep in prev_entities:
                            if ep.color not in (self.confirmed_mover_color, front_ep.color) and not ep.is_hud:
                                if tuple(behind_pos) in set(map(tuple, ep.coords)):
                                    behind_ep = ep
                                    break
                        if behind_ep is not None:
                            # behind_ep blocked front_ep!
                            self.role_evidence[behind_ep.color]["obstacle_score"] += 4.0
                        else:
                            # front_ep directly blocked mover!
                            self.role_evidence[front_ep.color]["obstacle_score"] += 4.0

        # 4. COLLECTOR_CONSUMABLE evidence
        if self.confirmed_mover_color is not None:
            mover_ep = next((e for e in prev_entities if e.color == self.confirmed_mover_color), None)
            mover_ec = next((e for e in curr_entities if e.color == self.confirmed_mover_color), None)
            if mover_ep is not None and mover_ec is not None:
                curr_mover_cells = set(map(tuple, mover_ec.coords))
                for vid in delta.vanished_entities:
                    ve = prev_map.get(vid)
                    if ve is not None and ve.color != self.confirmed_mover_color:
                        ve_cells = set(map(tuple, ve.coords))
                        if curr_mover_cells & ve_cells:
                            self.role_evidence[ve.color]["consumable_score"] += 5.0

        # 5. SWITCH_TOGGLE evidence
        click_xy = delta.action_data.get("click_xy") or delta.action_data.get("click_coords")
        is_interact = (delta.action == 5 or click_xy is not None)
        if is_interact:
            for eid, (old_c, new_c) in delta.state_mutations.items():
                ep = prev_map.get(eid)
                if ep is not None and not ep.is_hud:
                    if click_xy is not None:
                        cr, cc = click_xy
                        if ep.bbox[0] <= cr <= ep.bbox[2] and ep.bbox[1] <= cc <= ep.bbox[3]:
                            self.role_evidence[ep.color]["switch_score"] += 4.0
                    else:
                        self.role_evidence[ep.color]["switch_score"] += 3.0

        # 6. HAZARD evidence
        if delta.is_reset and self.confirmed_mover_color is not None:
            mover_ep = next((e for e in prev_entities if e.color == self.confirmed_mover_color), None)
            if mover_ep is not None:
                target_coords = mover_ep.coords + [nom_dr, nom_dc]
                target_cells = set(map(tuple, target_coords))
                for ep in prev_entities:
                    if ep.color != self.confirmed_mover_color:
                        if target_cells & set(map(tuple, ep.coords)):
                            self.role_evidence[ep.color]["hazard_score"] += 5.0

        # 7. HUD_INDICATOR evidence from perimeter mutations remote from mover
        r_min, r_max, c_min, c_max = delta.arena_bounds
        for r, c, old_c, new_c in delta.hud_mutations + delta.field_mutations:
            if r <= r_min or r >= r_max - 1 or c <= c_min or c >= c_max - 1:
                is_remote = True
                if self.confirmed_mover_color is not None:
                    mover_ep = next((e for e in prev_entities if e.color == self.confirmed_mover_color), None)
                    if mover_ep is not None:
                        if any(abs(r - mr) <= 1 and abs(c - mc) <= 1 for mr, mc in mover_ep.coords):
                            is_remote = False
                if is_remote:
                    if old_c != delta.field_bg_color and old_c != delta.canvas_bg_color:
                        self.role_evidence[old_c]["hud_score"] += 3.0
                    if new_c != delta.field_bg_color and new_c != delta.canvas_bg_color:
                        self.role_evidence[new_c]["hud_score"] += 3.0

        self._resolve_roles()

    def _resolve_roles(self) -> None:
        """Resolves most probable FunctionalRole for each color."""
        for c, ev in self.role_evidence.items():
            mover_score = ev["mover_score"]
            if self.confirmed_mover_color is not None and c != self.confirmed_mover_color:
                mover_score = 0.0

            scores = {
                FunctionalRole.MOVER: mover_score,
                FunctionalRole.CO_MOVER: ev["co_mover_score"],
                FunctionalRole.OBSTACLE: ev["obstacle_score"],
                FunctionalRole.PUSHABLE: ev["pushable_score"],
                FunctionalRole.SWITCH_TOGGLE: ev["switch_score"],
                FunctionalRole.COLLECTOR_CONSUMABLE: ev["consumable_score"],
                FunctionalRole.HAZARD: ev["hazard_score"],
                FunctionalRole.HUD_INDICATOR: ev["hud_score"],
            }
            best_role, best_val = max(scores.items(), key=lambda x: x[1])
            if best_val >= 2.0:
                self.discovered_roles[c] = best_role
                if best_role == FunctionalRole.CO_MOVER:
                    self.confirmed_co_mover_colors.add(c)


# =====================================================================
# 4. Pipeline Latency & Inertia Trackers
# =====================================================================

class ActionPipelineTracker:
    """Detects and maintains FIFO queue for action pipeline latency K (actions delayed by K steps)."""

    def __init__(self, max_k: int = 4) -> None:
        self.max_k = max_k
        self.action_history: List[int] = []
        self.disp_history: List[Tuple[int, int]] = []
        self.inferred_k: int = 0
        self.confirmed: bool = False
        self.pipeline_queue: deque = deque()

    def reset_for_level(self) -> None:
        self.action_history.clear()
        self.disp_history.clear()
        self.pipeline_queue.clear()
        for _ in range(self.inferred_k):
            self.pipeline_queue.append(0)

    def record_step(self, action: int, mover_dp: Tuple[int, int]) -> int:
        self.action_history.append(action)
        self.disp_history.append(mover_dp)

        if not self.confirmed and len(self.action_history) >= 3:
            self._induce_k()

        return self.inferred_k

    def _induce_k(self) -> None:
        N = len(self.disp_history)
        best_k = 0
        best_score = -1e9

        for k in range(min(self.max_k + 1, N - 1)):
            score = 0.0
            matches = 0
            for t in range(k, N):
                act = self.action_history[t - k]
                dp = self.disp_history[t]
                nom_dp = DIRECTION_DELTAS.get(act, (0, 0))
                if dp != (0, 0):
                    if dp == nom_dp:
                        score += 3.0
                        matches += 1
                    else:
                        score -= 2.0
                elif nom_dp == (0, 0):
                    score += 1.0

            if score > best_score:
                best_score = score
                best_k = k

        if best_score >= 5.0:
            self.inferred_k = best_k
            self.confirmed = True

    def get_effective_action(self, action: int) -> int:
        """Returns the action that executes at current step given pipeline latency."""
        if self.inferred_k == 0:
            return action
        self.pipeline_queue.append(action)
        if len(self.pipeline_queue) > self.inferred_k:
            return self.pipeline_queue.popleft()
        return 0


class InertiaDynamicsTracker:
    """Tracks continuous/discrete velocity vectors and momentum under friction/collisions."""

    def __init__(self) -> None:
        self.velocity: Tuple[int, int] = (0, 0)
        self.is_inertial: bool = False

    def reset(self) -> None:
        self.velocity = (0, 0)
        self.is_inertial = False

    def update_velocity(
        self,
        applied_action: int,
        collision_occurred: bool,
    ) -> Tuple[int, int]:
        nom_dr, nom_dc = DIRECTION_DELTAS.get(applied_action, (0, 0))
        if collision_occurred:
            self.velocity = (0, 0)
        elif (nom_dr, nom_dc) != (0, 0):
            self.velocity = (nom_dr, nom_dc)
        return self.velocity


# =====================================================================
# 5. Local World Model Transition Graph & Simulator
# =====================================================================

@dataclass
class VerificationReport:
    total_steps: int
    exact_matches: int
    exact_match_ratio: float
    mean_jaccard: float
    counterexamples: List[Dict[str, Any]]


class WorldModelTransitionGraph:
    """Empirical transition graph and high-performance forward simulator."""

    def __init__(self, delta_engine: Optional[UniversalSemanticDeltaEngine] = None) -> None:
        self.delta_engine = delta_engine or UniversalSemanticDeltaEngine()
        self.pipeline_tracker = ActionPipelineTracker()
        self.inertia_tracker = InertiaDynamicsTracker()
        self.modality: KinematicModality = KinematicModality.CARDINAL_1STEP
        self.stride_k: int = 1

        self.transition_graph: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
        self.transition_history: List[FrameDelta] = []

    def reset_for_level(self) -> None:
        self.pipeline_tracker.reset_for_level()
        self.inertia_tracker.reset()
        self.transition_history.clear()

    def record_transition(self, delta: FrameDelta) -> None:
        """Ingests a verified transition and updates modality and transition statistics."""
        self.transition_history.append(delta)

        mover_c = self.delta_engine.confirmed_mover_color
        if mover_c is not None:
            for action in (1, 2, 3, 4):
                disps = self.delta_engine.role_evidence[mover_c]["action_displacements"][action]
                for dr, dc in disps:
                    if (dr, dc) == (0, 0):
                        continue
                    step_size = max(abs(dr), abs(dc))
                    if abs(dr) == 1 and abs(dc) == 2 or abs(dr) == 2 and abs(dc) == 1:
                        self.modality = KinematicModality.KNIGHT_JUMP
                    elif step_size >= 2:
                        if abs(dr) == 0 or abs(dc) == 0:
                            self.modality = KinematicModality.INERTIAL_SLIDE
                            self.stride_k = step_size
                    else:
                        self.modality = KinematicModality.CARDINAL_1STEP
                        self.stride_k = 1

            mover_dp = (0, 0)
            for eid, dp in delta.displacements.items():
                old_colors = delta.pre_grid[delta.diff_mask]
                if mover_c in old_colors:
                    mover_dp = dp
                    break
            self.pipeline_tracker.record_step(delta.action, mover_dp)

    @staticmethod
    def raycast_slide(
        start_pos: Tuple[int, int],
        direction: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        """Calculates destination of continuous slide until collision with obstacle or boundary."""
        r_min, r_max, c_min, c_max = bounds
        dr, dc = direction
        curr_r, curr_c = start_pos

        while True:
            next_r = curr_r + dr
            next_c = curr_c + dc
            if not (r_min <= next_r < r_max and c_min <= next_c < c_max):
                break
            if (next_r, next_c) in obstacles:
                break
            curr_r, curr_c = next_r, next_c

        return curr_r, curr_c

    def simulate_step(
        self,
        grid: np.ndarray,
        action: int,
        action_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulates next frame grid and state transition under current empirical world model."""
        if action_data is None:
            action_data = {}

        next_grid = np.copy(grid)
        r_min, r_max, c_min, c_max, canvas_bg, field_bg, frame_colors = self.delta_engine.arena_separator.detect_arena(grid)
        bounds = (r_min, r_max, c_min, c_max)

        mover_c = self.delta_engine.confirmed_mover_color
        if mover_c is None:
            return next_grid, {"status": "NO_MOVER"}

        # Extract object roles
        obstacles: Set[Tuple[int, int]] = set()
        pushables: Dict[Tuple[int, int], int] = {}
        consumables: Set[Tuple[int, int]] = set()
        switches: Set[Tuple[int, int]] = set()
        hazards: Set[Tuple[int, int]] = set()

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                val = int(grid[r, c])
                if val == field_bg or val == canvas_bg or val in frame_colors:
                    continue
                role = self.delta_engine.discovered_roles.get(val, FunctionalRole.UNKNOWN)
                if role == FunctionalRole.OBSTACLE:
                    obstacles.add((r, c))
                elif role == FunctionalRole.PUSHABLE:
                    pushables[(r, c)] = val
                elif role == FunctionalRole.COLLECTOR_CONSUMABLE:
                    consumables.add((r, c))
                elif role == FunctionalRole.SWITCH_TOGGLE:
                    switches.add((r, c))
                elif role == FunctionalRole.HAZARD:
                    hazards.add((r, c))

        mover_pts = np.argwhere(grid == mover_c)
        if len(mover_pts) == 0:
            return next_grid, {"status": "MOVER_NOT_FOUND"}

        mover_r = int(round(mover_pts[:, 0].mean()))
        mover_c_col = int(round(mover_pts[:, 1].mean()))
        mover_pos = (mover_r, mover_c_col)

        eff_action = self.pipeline_tracker.get_effective_action(action)
        dr, dc = DIRECTION_DELTAS.get(eff_action, (0, 0))

        if (dr, dc) != (0, 0):
            if self.modality == KinematicModality.INERTIAL_SLIDE:
                dest_pos = self.raycast_slide(mover_pos, (dr, dc), obstacles | set(pushables.keys()), bounds)
                actual_dp = (dest_pos[0] - mover_pos[0], dest_pos[1] - mover_pos[1])
            elif self.modality == KinematicModality.KNIGHT_JUMP:
                k_dr, k_dc = KNIGHT_DELTAS.get(eff_action, (dr, dc))
                cand_pos = (mover_pos[0] + k_dr, mover_pos[1] + k_dc)
                if r_min <= cand_pos[0] < r_max and c_min <= cand_pos[1] < c_max and cand_pos not in obstacles:
                    dest_pos = cand_pos
                    actual_dp = (k_dr, k_dc)
                else:
                    dest_pos = mover_pos
                    actual_dp = (0, 0)
            else:
                cand_pos = (mover_pos[0] + dr * self.stride_k, mover_pos[1] + dc * self.stride_k)
                if not (r_min <= cand_pos[0] < r_max and c_min <= cand_pos[1] < c_max) or cand_pos in obstacles:
                    dest_pos = mover_pos
                    actual_dp = (0, 0)
                elif cand_pos in pushables:
                    push_c = pushables[cand_pos]
                    push_dest = (cand_pos[0] + dr, cand_pos[1] + dc)
                    if r_min <= push_dest[0] < r_max and c_min <= push_dest[1] < c_max and push_dest not in obstacles and push_dest not in pushables:
                        next_grid[push_dest[0], push_dest[1]] = push_c
                        dest_pos = cand_pos
                        actual_dp = (dr, dc)
                    else:
                        dest_pos = mover_pos
                        actual_dp = (0, 0)
                else:
                    dest_pos = cand_pos
                    actual_dp = (dr * self.stride_k, dc * self.stride_k)

            if actual_dp != (0, 0):
                for pr, pc in mover_pts:
                    next_grid[pr, pc] = field_bg
                for pr, pc in mover_pts:
                    nr, nc = pr + actual_dp[0], pc + actual_dp[1]
                    if 0 <= nr < next_grid.shape[0] and 0 <= nc < next_grid.shape[1]:
                        next_grid[nr, nc] = mover_c

                if dest_pos in consumables:
                    consumables.remove(dest_pos)

                if dest_pos in hazards:
                    return next_grid, {"status": "HAZARD_RESET", "reset": True}

        click_xy = action_data.get("click_xy") or action_data.get("click_coords")
        if eff_action == 6 and click_xy is not None:
            cr, cc = click_xy
            if 0 <= cr < next_grid.shape[0] and 0 <= cc < next_grid.shape[1]:
                if (cr, cc) in switches:
                    next_grid[cr, cc] = (int(next_grid[cr, cc]) % 9) + 1

        return next_grid, {"status": "SUCCESS", "dest_pos": dest_pos if (dr, dc) != (0, 0) else mover_pos}

    def verify_history(self, history: List[FrameDelta]) -> VerificationReport:
        """Replays world model forward simulation against recorded historical transitions."""
        if not history:
            return VerificationReport(0, 0, 1.0, 1.0, [])

        exact_matches = 0
        jaccard_scores = []
        counterexamples = []

        for delta in history:
            sim_grid, info = self.simulate_step(delta.pre_grid, delta.action, delta.action_data)
            is_exact = np.array_equal(sim_grid, delta.post_grid)
            if is_exact:
                exact_matches += 1

            diff_actual = (delta.pre_grid != delta.post_grid)
            diff_sim = (delta.pre_grid != sim_grid)
            intersection = np.sum(diff_actual & diff_sim)
            union = np.sum(diff_actual | diff_sim)
            jaccard = 1.0 if union == 0 else float(intersection) / float(union)
            jaccard_scores.append(jaccard)

            if not is_exact:
                counterexamples.append({
                    "step_idx": delta.step_idx,
                    "action": delta.action,
                    "jaccard": jaccard,
                    "diff_cells": int(np.sum(sim_grid != delta.post_grid)),
                })

        return VerificationReport(
            total_steps=len(history),
            exact_matches=exact_matches,
            exact_match_ratio=float(exact_matches) / float(len(history)),
            mean_jaccard=float(np.mean(jaccard_scores)),
            counterexamples=counterexamples,
        )


# =====================================================================
# 6. Verification Test Suite
# =====================================================================

def run_verification_tests():
    print("Running Universal World Model Verification Tests...")

    # Test 1: Mover, Obstacle, Pushable, Consumable
    grid0 = np.zeros((16, 16), dtype=int)
    grid0[0, :] = 5; grid0[-1, :] = 5; grid0[:, 0] = 5; grid0[:, -1] = 5
    grid0[5, 5] = 2  # Mover
    grid0[5, 6] = 3  # Pushable
    grid0[5, 8] = 4  # Obstacle
    grid0[7, 5] = 6  # Consumable

    # Step 1: Move Right (Action 4) -> Mover pushes 3 from (5,6) to (5,7)
    grid1 = np.copy(grid0)
    grid1[5, 5] = 0
    grid1[5, 6] = 2
    grid1[5, 7] = 3

    # Step 2: Move Right again (Action 4) -> Mover at (5,6) tries to push 3 into (5,8), but 4 is obstacle! Blocked!
    grid2 = np.copy(grid1) # No change

    # Step 3: Move Down (Action 2) -> Mover moves to (6, 6)
    grid3 = np.copy(grid2)
    grid3[5, 6] = 0
    grid3[6, 6] = 2

    # Step 4: Move Left (Action 3) -> Mover moves to (6, 5)
    grid4 = np.copy(grid3)
    grid4[6, 6] = 0
    grid4[6, 5] = 2

    # Step 5: Move Down (Action 2) -> Mover moves to (7, 5) and collects 6 (Consumable vanishes)
    grid5 = np.copy(grid4)
    grid5[6, 5] = 0
    grid5[7, 5] = 2

    engine = UniversalSemanticDeltaEngine()
    wm = WorldModelTransitionGraph(engine)

    d1 = engine.process_transition(grid0, grid1, 4, step_idx=1)
    wm.record_transition(d1)

    d2 = engine.process_transition(grid1, grid2, 4, step_idx=2)
    wm.record_transition(d2)

    d3 = engine.process_transition(grid2, grid3, 2, step_idx=3)
    wm.record_transition(d3)

    d4 = engine.process_transition(grid3, grid4, 3, step_idx=4)
    wm.record_transition(d4)

    d5 = engine.process_transition(grid4, grid5, 2, step_idx=5)
    wm.record_transition(d5)

    print("Roles induced:")
    for c, r in engine.discovered_roles.items():
        print(f"  Color {c} -> {r.value}")

    assert engine.discovered_roles.get(2) == FunctionalRole.MOVER, f"Expected MOVER for color 2, got {engine.discovered_roles.get(2)}"
    assert engine.discovered_roles.get(3) == FunctionalRole.PUSHABLE, f"Expected PUSHABLE for color 3, got {engine.discovered_roles.get(3)}"
    assert engine.discovered_roles.get(4) == FunctionalRole.OBSTACLE, f"Expected OBSTACLE for color 4, got {engine.discovered_roles.get(4)}"
    assert engine.discovered_roles.get(6) == FunctionalRole.COLLECTOR_CONSUMABLE, f"Expected COLLECTOR_CONSUMABLE for color 6, got {engine.discovered_roles.get(6)}"

    # Replay Verification
    report = wm.verify_history([d1, d2, d3, d4, d5])
    print(f"Verification report: exact_match_ratio={report.exact_match_ratio:.2f}, mean_jaccard={report.mean_jaccard:.2f}")
    assert report.exact_match_ratio == 1.0, f"Expected 100% exact match replay, got {report.exact_match_ratio}"

    # Test 2: Inertial Slide
    slide_start = (5, 5)
    obstacles = {(5, 12)}
    bounds = (1, 15, 1, 15)
    dest = wm.raycast_slide(slide_start, (0, 1), obstacles, bounds)
    assert dest == (5, 11), f"Expected slide destination (5, 11), got {dest}"
    print("Inertial slide test passed:", dest)

    # Test 3: Pipeline Latency
    pipe = ActionPipelineTracker(max_k=3)
    pipe.record_step(1, (0, 0))
    pipe.record_step(4, (-1, 0))  # Latency K=1 detected: move at step 2 was from step 1 (action 1)
    pipe.record_step(2, (0, 1))   # Move at step 3 was from step 2 (action 4)
    pipe.record_step(3, (1, 0))   # Move at step 4 was from step 3 (action 2)
    assert pipe.inferred_k == 1, f"Expected K=1 latency, got {pipe.inferred_k}"
    print("Pipeline latency induction passed: K =", pipe.inferred_k)

    print("\nALL WORLD MODEL VERIFICATION TESTS PASSED SUCCESSFULLY (100% ACCURACY)!")


if __name__ == "__main__":
    run_verification_tests()
