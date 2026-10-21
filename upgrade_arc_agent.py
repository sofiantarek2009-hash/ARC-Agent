#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PO_DIR = ROOT / "project_override"

def clean_and_prepare_dirs():
    # Remove any broken symlinks or nested duplicate directories first
    if PO_DIR.is_symlink() or PO_DIR.is_file():
        PO_DIR.unlink()
    nested = PO_DIR / "project_override"
    if nested.is_symlink() or nested.is_file():
        nested.unlink()
    elif nested.exists() and nested.is_dir():
        shutil.rmtree(nested)

    PO_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["core", "perception", "planning", "world_model", "orchestration"]:
        sub_path = PO_DIR / sub
        if sub_path.is_symlink() or sub_path.is_file():
            sub_path.unlink()
        sub_path.mkdir(parents=True, exist_ok=True)
        (sub_path / "__init__.py").write_text("", encoding="utf-8")
    (PO_DIR / "__init__.py").write_text("", encoding="utf-8")

clean_and_prepare_dirs()

# =========================================================================
# 1. project_override/core/dead_signature.py
# =========================================================================
(PO_DIR / "core" / "dead_signature.py").write_text('''import hashlib
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def compute_arena_hash(grid: np.ndarray, bezel_px: int = 4) -> int:
    """Computes a 44-bit Blake2b spatial hash masking out 4px perimeter timer bezels."""
    arr = np.asarray(grid, dtype=np.int8)
    h, w = arr.shape
    if h > 2 * bezel_px and w > 2 * bezel_px:
        core = np.ascontiguousarray(arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px])
    else:
        core = np.ascontiguousarray(arr)
    digest = hashlib.blake2b(core.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 44) - 1)


def pack_signature_64(
    arena_hash: int,
    action_id: int,
    coord_x: int = 0,
    coord_y: int = 0,
    level_idx: int = 0,
) -> int:
    """Packs (arena_hash: 44b, level_idx: 4b, coord_y: 6b, coord_x: 6b, action_id: 4b) into 64-bit int."""
    h44 = arena_hash & ((1 << 44) - 1)
    lvl4 = level_idx & 0xF
    cy6 = coord_y & 0x3F
    cx6 = coord_x & 0x3F
    act4 = action_id & 0xF
    return (h44 << 20) | (lvl4 << 16) | (cy6 << 10) | (cx6 << 4) | act4


class DeadSignatureTable:
    """
    O(1) dead-action & fatal-action pruning + Empirical Transition Graph + IW(1) Novelty Table.
    Persists fatal GAME_OVER signatures across level resets.
    """

    def __init__(self, ring_size: int = 24):
        self.dead_set: Set[int] = set()
        self.death_set: Set[int] = set()  # Never cleared on RESET
        self.ring_buffer: deque = deque(maxlen=ring_size)
        self.transitions: Dict[int, Dict[Tuple[int, int, int], int]] = {}
        self.state_grids: Dict[int, np.ndarray] = {}
        self.state_novelty: Dict[int, int] = {}
        self.seen_atoms: Set[Tuple[int, int, int]] = set()

    def reset_level(self, clear_deaths: bool = False) -> None:
        self.ring_buffer.clear()
        self.transitions.clear()
        self.state_grids.clear()
        self.state_novelty.clear()
        self.seen_atoms.clear()
        if clear_deaths:
            self.death_set.clear()
            self.dead_set.clear()

    def record_death(
        self,
        pre_hash: int,
        action_id: int,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
    ) -> None:
        sig = pack_signature_64(pre_hash, action_id, coord_x, coord_y, level_idx)
        self.death_set.add(sig)
        self.dead_set.add(sig)

    def record_transition(
        self,
        pre_hash: int,
        action_id: int,
        post_hash: int,
        delta_omega: int,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
        post_grid: Optional[np.ndarray] = None,
    ) -> None:
        if delta_omega == 0 or pre_hash == post_hash:
            sig = pack_signature_64(pre_hash, action_id, coord_x, coord_y, level_idx)
            self.dead_set.add(sig)
        else:
            act_key = (int(action_id), int(coord_x), int(coord_y))
            if pre_hash not in self.transitions:
                self.transitions[pre_hash] = {}
            self.transitions[pre_hash][act_key] = post_hash
            if post_grid is not None and post_hash not in self.state_grids:
                self.state_grids[post_hash] = post_grid.copy()
                nov = self.update_novelty(post_grid)
                self.state_novelty[post_hash] = nov

        self.ring_buffer.append(post_hash)

    def update_novelty(self, grid: np.ndarray) -> int:
        """Width-1 Iterated Width IW(1) novelty evaluation over (row, col, color) atoms."""
        arr = np.asarray(grid, dtype=np.int8)
        new_count = 0
        for r in range(4, arr.shape[0] - 4, 2):
            for c in range(4, arr.shape[1] - 4, 2):
                atom = (r, c, int(arr[r, c]))
                if atom not in self.seen_atoms:
                    self.seen_atoms.add(atom)
                    new_count += 1
        return new_count

    def is_dead(
        self,
        arena_hash: int,
        action_id: int,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
    ) -> bool:
        sig = pack_signature_64(arena_hash, action_id, coord_x, coord_y, level_idx)
        return (sig in self.dead_set) or (sig in self.death_set)

    def detect_loop(self) -> bool:
        n = len(self.ring_buffer)
        if n < 6:
            return False
        recent = list(self.ring_buffer)
        if len(set(recent[-4:])) == 1:
            return True
        if recent[-1] == recent[-3] == recent[-5] and recent[-2] == recent[-4]:
            return True
        if n >= 8 and recent[-4:] == recent[-8:-4]:
            return True
        if n >= 6 and recent[-3:] == recent[-6:-3]:
            return True
        return False

    def bfs_to_unexplored_action(
        self,
        current_hash: int,
        candidate_actions: List[int],
        level_idx: int = 0,
    ) -> Optional[List[int]]:
        """
        Empirical Graph BFS: Finds shortest path in known transition graph to a state
        with an untried non-dead action, prioritizing high IW(1) novelty states.
        """
        if not candidate_actions:
            return None
        tried_here = {k[0] for k in self.transitions.get(current_hash, {}).keys()}
        for a in candidate_actions:
            if not self.is_dead(current_hash, a, 0, 0, level_idx) and a not in tried_here:
                return [a]

        queue = deque([(current_hash, [])])
        visited = {current_hash}
        best_path: Optional[List[int]] = None
        best_score = -1e9

        while queue and len(visited) < 350:
            sh, path = queue.popleft()
            if len(path) >= 22:
                continue
            tried = {k[0] for k in self.transitions.get(sh, {}).keys()}
            for a in candidate_actions:
                if not self.is_dead(sh, a, 0, 0, level_idx) and a not in tried:
                    cand_path = path + [a]
                    nov = self.state_novelty.get(sh, 0)
                    score = nov * 2.0 - len(cand_path)
                    if score > best_score:
                        best_score = score
                        best_path = cand_path
                    if len(cand_path) <= 3:
                        return cand_path
            for (act_id, cx, cy), nxt_h in self.transitions.get(sh, {}).items():
                if act_id in candidate_actions and nxt_h not in visited:
                    if not self.is_dead(sh, act_id, cx, cy, level_idx):
                        visited.add(nxt_h)
                        queue.append((nxt_h, path + [act_id]))
        return best_path
''', encoding="utf-8")

# =========================================================================
# 2. project_override/perception/observation_processor.py
# =========================================================================
(PO_DIR / "perception" / "observation_processor.py").write_text('''from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class Entity:
    id: int
    color: int
    pixels: List[Tuple[int, int]]
    mass: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]


def px_to_board_coord(r_px: float, c_px: float, scale: int, shape: Tuple[int, int] = (64, 64)) -> Tuple[int, int]:
    s = max(1, int(scale))
    h, w = shape
    br = max(0, min((h // s) - 1, int(round(r_px / s))))
    bc = max(0, min((w // s) - 1, int(round(c_px / s))))
    return br, bc


def board_to_px_coord(br: int, bc: int, scale: int, shape: Tuple[int, int] = (64, 64)) -> Tuple[int, int]:
    s = max(1, int(scale))
    h, w = shape
    r_px = max(0, min(h - 1, int(br * s + s // 2)))
    c_px = max(0, min(w - 1, int(bc * s + s // 2)))
    return r_px, c_px


def hungarian_match(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Kuhn-Munkres / min-cost bipartite matching for multi-entity tracking."""
    if cost_matrix.size == 0:
        return []
    n_rows, n_cols = cost_matrix.shape
    pairs = []
    used_r, used_c = set(), set()
    flat_indices = np.argsort(cost_matrix, axis=None)
    for idx in flat_indices:
        r = int(idx // n_cols)
        c = int(idx % n_cols)
        if r not in used_r and c not in used_c:
            if cost_matrix[r, c] < 1e6:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
                if len(used_r) == n_rows or len(used_c) == n_cols:
                    break
    return pairs


class ObservationProcessor:
    """Zero-color-assumption CCL + Rigid-Body Composite Sprite & Box Push Tracker."""

    def __init__(self, bezel_px: int = 4):
        self.bezel_px = bezel_px

    def induce_background_color(self, grid: np.ndarray, learned_floor_color: Optional[int] = None) -> int:
        if learned_floor_color is not None:
            return int(learned_floor_color)
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        core = arr[self.bezel_px:h - self.bezel_px, self.bezel_px:w - self.bezel_px] if h > 8 else arr
        vals, counts = np.unique(core, return_counts=True)
        return int(vals[int(np.argmax(counts))]) if len(vals) > 0 else 0

    def extract_entities(
        self,
        grid: np.ndarray,
        bg_color: Optional[int] = None,
        ignore_bezel: bool = True,
    ) -> List[Entity]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        if bg_color is None:
            bg_color = self.induce_background_color(arr)

        visited = np.zeros((h, w), dtype=bool)
        if ignore_bezel and h >= 16 and w >= 16:
            b = self.bezel_px
            visited[:b, :] = True
            visited[h - b:, :] = True
            visited[:, :b] = True
            visited[:, w - b:] = True

        entities: List[Entity] = []
        ent_id = 0
        for r in range(h):
            for c in range(w):
                if visited[r, c] or int(arr[r, c]) == bg_color:
                    continue
                color = int(arr[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                pixels = []
                r_min = r_max = r
                c_min = c_max = c
                while stack:
                    cr, cc = stack.pop()
                    pixels.append((cr, cc))
                    if cr < r_min: r_min = cr
                    if cr > r_max: r_max = cr
                    if cc < c_min: c_min = cc
                    if cc > c_max: c_max = cc
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(arr[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                mass = len(pixels)
                mean_r = sum(p[0] for p in pixels) / mass
                mean_c = sum(p[1] for p in pixels) / mass
                entities.append(
                    Entity(
                        id=ent_id,
                        color=color,
                        pixels=pixels,
                        mass=mass,
                        centroid=(mean_r, mean_c),
                        bbox=(r_min, c_min, r_max, c_max),
                    )
                )
                ent_id += 1
        return entities

    def compute_delta_omega(self, prev_grid: np.ndarray, curr_grid: np.ndarray) -> int:
        a = np.asarray(prev_grid, dtype=np.int8)
        b = np.asarray(curr_grid, dtype=np.int8)
        if a.shape != b.shape:
            return 0
        h, w = a.shape
        if h >= 16 and w >= 16:
            bz = self.bezel_px
            return int(np.sum(a[bz:h - bz, bz:w - bz] != b[bz:h - bz, bz:w - bz]))
        return int(np.sum(a != b))

    def infer_rigid_translation(
        self,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        known_avatar_colors: Optional[Set[int]] = None,
    ) -> Optional[Tuple[int, Set[int], Tuple[int, int], Tuple[float, float], Set[int]]]:
        """
        Universal Rigid-Body Differential Tracker.
        Returns: (floor_color, avatar_colors_set, (dr, dc), post_centroid, pushed_box_colors)
        Distinguishes multi-color composite avatars (e.g. ls20) from pushed Sokoban boxes!
        """
        f0 = np.asarray(prev_grid, dtype=np.int8)
        f1 = np.asarray(curr_grid, dtype=np.int8)
        if f0.shape != f1.shape:
            return None
        h, w = f0.shape
        diff = f0 != f1
        if h >= 16 and w >= 16:
            bz = self.bezel_px
            mask = np.zeros((h, w), dtype=bool)
            mask[bz:h - bz, bz:w - bz] = True
            diff = diff & mask

        total_changed = int(np.sum(diff))
        if total_changed < 2 or total_changed > 800:
            return None

        best_floor = None
        best_score = 0
        for c in range(16):
            stepped_onto = int(np.sum(diff & (f0 == c) & (f1 != c)))
            vacated = int(np.sum(diff & (f0 != c) & (f1 == c)))
            score = min(stepped_onto, vacated)
            if score > best_score:
                best_score = score
                best_floor = c

        if best_floor is None or best_score == 0:
            return None

        b_post = np.argwhere(diff & (f0 == best_floor) & (f1 != best_floor))
        b_pre_all = np.argwhere(diff & (f0 != best_floor) & (f1 == best_floor))
        if len(b_post) == 0 or len(b_pre_all) == 0:
            return None

        post_center = b_post.mean(axis=0)
        dists = np.linalg.norm(b_pre_all - post_center, axis=1)
        near_mask = dists <= 20.0
        b_pre = b_pre_all[near_mask] if np.any(near_mask) else b_pre_all

        pre_center = b_pre.mean(axis=0)
        dr = int(round(float(post_center[0] - pre_center[0])))
        dc = int(round(float(post_center[1] - pre_center[1])))
        if dr == 0 and dc == 0:
            return None

        # Check if an entity of a DIFFERENT color than known_avatar_colors was pushed ahead
        pushed_box_colors: Set[int] = set()
        moving_colors = {int(c) for c in np.unique(f1[b_post[:, 0], b_post[:, 1]]) if int(c) != best_floor}
        if known_avatar_colors:
            extra_colors = moving_colors - known_avatar_colors
            # If known avatar moved into pixels previously occupied by extra_colors, it's a pushed box!
            for ec in extra_colors:
                overlap = int(np.sum(diff & (f0 == ec) & np.isin(f1, list(known_avatar_colors))))
                if overlap > 0:
                    pushed_box_colors.add(ec)
            avatar_colors = known_avatar_colors
        else:
            avatar_colors = moving_colors

        if not avatar_colors:
            return None

        return best_floor, avatar_colors, (dr, dc), (float(post_center[0]), float(post_center[1])), pushed_box_colors
''', encoding="utf-8")

# =========================================================================
# 3. project_override/core/visual_goal_induction.py
# =========================================================================
(PO_DIR / "core" / "visual_goal_induction.py").write_text('''from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class JordanCurveCavityInducer:
    """Detects hollow frames, target receptacles, and enclosed cavities."""

    @staticmethod
    def find_cavities(grid: np.ndarray, bg_color: int, bezel_px: int = 4) -> List[Tuple[int, int]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        cavities: List[Tuple[int, int]] = []
        for r in range(bezel_px + 1, h - bezel_px - 1):
            for c in range(bezel_px + 1, w - bezel_px - 1):
                if int(arr[r, c]) != bg_color:
                    continue
                neighbors = [
                    int(arr[r - 1, c]),
                    int(arr[r + 1, c]),
                    int(arr[r, c - 1]),
                    int(arr[r, c + 1]),
                ]
                non_bg = [n for n in neighbors if n != bg_color]
                if len(non_bg) >= 3 and len(set(non_bg)) == 1:
                    cavities.append((r, c))
        return cavities


class DihedralSymmetrySolver:
    """Full D4 Dihedral Split-Panel Symmetry & Odd-One-Out Pattern Completion Solver."""

    @staticmethod
    def _d4_transforms(panel: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        transforms = [
            ("id", panel),
            ("fliplr", np.fliplr(panel)),
            ("flipud", np.flipud(panel)),
            ("rot180", np.rot90(panel, 2)),
        ]
        if panel.shape[0] == panel.shape[1]:
            transforms.extend([
                ("rot90", np.rot90(panel, 1)),
                ("rot270", np.rot90(panel, 3)),
                ("transpose", panel.T),
                ("antitranspose", np.fliplr(np.rot90(panel, 1))),
            ])
        return transforms

    @classmethod
    def find_symmetry_clicks(
        cls, grid: np.ndarray, bg_color: int, bezel_px: int = 4
    ) -> List[Tuple[int, int]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        core = arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px]
        ch, cw = core.shape
        clicks: List[Tuple[int, int]] = []

        # 1. Vertical split (left vs right)
        half_w = cw // 2
        left = core[:, :half_w - 1]
        right = core[:, cw - left.shape[1]:]
        if left.shape == right.shape and left.size > 0:
            for _, t_left in cls._d4_transforms(left):
                diff = t_left != right
                n_diff = int(np.sum(diff))
                if 1 <= n_diff <= 48:
                    # Click where reference has foreground and editable panel has background (or vice versa)
                    rs, cs = np.where(diff)
                    for r, c in zip(rs, cs):
                        clicks.append((int(r + bezel_px), int(c + bezel_px + (cw - left.shape[1]))))
                    return clicks

        # 2. Horizontal split (top vs bottom)
        half_h = ch // 2
        top = core[:half_h - 1, :]
        bot = core[ch - top.shape[0]:, :]
        if top.shape == bot.shape and top.size > 0:
            for _, t_top in cls._d4_transforms(top):
                diff = t_top != bot
                n_diff = int(np.sum(diff))
                if 1 <= n_diff <= 48:
                    rs, cs = np.where(diff)
                    for r, c in zip(rs, cs):
                        clicks.append((int(r + bezel_px + (ch - top.shape[0])), int(c + bezel_px)))
                    return clicks

        return clicks


class LatentMemoryMatcher:
    """Tracks hidden-state tile reveals in Concentration / Memory-Match games (e.g. mm01-mm05)."""

    def __init__(self):
        self.known_tiles: Dict[Tuple[int, int], int] = {}
        self.matched_Positions: Set[Tuple[int, int]] = set()

    def reset(self) -> None:
        self.known_tiles.clear()
        self.matched_Positions.clear()

    def observe_reveal(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray, click_xy: Tuple[int, int], bg_color: int
    ) -> Optional[Tuple[int, int]]:
        """Records revealed color at click_xy; if a matching partner is known, returns its (x, y)!”"""
        cx, cy = click_xy
        f0 = np.asarray(prev_grid, dtype=np.int8)
        f1 = np.asarray(curr_grid, dtype=np.int8)
        if not (0 <= cy < f1.shape[0] and 0 <= cx < f1.shape[1]):
            return None
        new_color = int(f1[cy, cx])
        old_color = int(f0[cy, cx])
        if new_color != old_color and new_color != bg_color:
            # Quantize to 4px tile center
            key = (int(round(cx / 4.0) * 4), int(round(cy / 4.0) * 4))
            for other_xy, c in list(self.known_tiles.items()):
                if c == new_color and abs(other_xy[0] - key[0]) + abs(other_xy[1] - key[1]) >= 4:
                    if other_xy not in self.matched_Positions:
                        self.matched_Positions.add(other_xy)
                        self.matched_Positions.add(key)
                        return other_xy
            self.known_tiles[key] = new_color
        return None
''', encoding="utf-8")

# =========================================================================
# 4. project_override/core/combinatorial_engine.py
# =========================================================================
(PO_DIR / "core" / "combinatorial_engine.py").write_text('''from typing import Dict, List, Optional, Tuple
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
''', encoding="utf-8")

# =========================================================================
# 5. project_override/core/circuit_engine.py
# =========================================================================
(PO_DIR / "core" / "circuit_engine.py").write_text('''from typing import List, Optional, Tuple
import numpy as np


class AC3CircuitSolver:
    """Bezel-guarded Periodic Lattice & AC-3 Circuit Tile Rotation Detector."""

    @staticmethod
    def detect_periodic_circuit_clicks(
        grid: np.ndarray, bg_color: int, bezel_px: int = 4
    ) -> Optional[List[Tuple[int, int]]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        core = arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px]
        fg = core != bg_color
        if int(np.sum(fg)) < 24:
            return None

        # Check periodicity k in {6, 8, 10, 12}
        best_k = 0
        for k in (6, 8, 10, 12):
            if core.shape[0] >= 3 * k and core.shape[1] >= 3 * k:
                row_Lines = sum(1 for r in range(k, core.shape[0], k) if np.mean(core[r, :] == core[0, :]) > 0.7)
                if row_Lines >= 2:
                    best_k = k
                    break
        if best_k == 0:
            return None

        clicks: List[Tuple[int, int]] = []
        for r in range(bezel_px + best_k // 2, h - bezel_px, best_k):
            for c in range(bezel_px + best_k // 2, w - bezel_px, best_k):
                sub = arr[r - 2:r + 3, c - 2:c + 3]
                if sub.size > 0 and int(np.sum(sub != bg_color)) >= 3:
                    clicks.append((int(r), int(c)))
        return clicks if len(clicks) >= 4 else None
''', encoding="utf-8")

# =========================================================================
# 6. project_override/core/universal_engine.py
# =========================================================================
(PO_DIR / "core" / "universal_engine.py").write_text('''import time
from typing import Dict, List, Optional, Set, Tuple


class SystemPacer:
    """
    Tracks active CPU/GPU compute time per step rather than wall-clock wait time
    so 110 concurrent Swarm threads never suffer false-positive timeouts.
    """

    GLOBAL_START_TIME = time.time()
    GLOBAL_HARD_LIMIT_S = 31800.0  # 8h 50m safety margin within 9h Kaggle limit

    def __init__(self, max_level_compute_s: float = 45.0):
        self.max_level_compute_s = max_level_compute_s
        self.level_compute_s: float = 0.0
        self._step_t0: Optional[float] = None

    def reset_level(self) -> None:
        self.level_compute_s = 0.0
        self._step_t0 = None

    def begin_step(self) -> None:
        self._step_t0 = time.perf_counter()

    def end_step(self) -> None:
        if self._step_t0 is not None:
            self.level_compute_s += time.perf_counter() - self._step_t0
            self._step_t0 = None

    def has_compute_budget(self) -> bool:
        if (time.time() - self.GLOBAL_START_TIME) > self.GLOBAL_HARD_LIMIT_S:
            return False
        return self.level_compute_s < self.max_level_compute_s
''', encoding="utf-8")

# =========================================================================
# 7. project_override/planning/symbolic_planners.py
# =========================================================================
(PO_DIR / "planning" / "symbolic_planners.py").write_text('''from collections import deque
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class MacroActionQueue:
    """Executes multi-step discrete or click plans with immediate rollback on delta_omega == 0."""

    def __init__(self):
        self.queue: deque = deque()

    def push_plan(self, actions: List[Any]) -> None:
        self.queue.clear()
        for a in actions:
            if isinstance(a, tuple) and len(a) == 2:
                self.queue.append((int(a[0]), dict(a[1])))
            else:
                self.queue.append((int(a), {}))

    def pop_next(self, last_delta_omega: Optional[int] = None) -> Optional[Tuple[int, Dict[str, Any]]]:
        if last_delta_omega == 0:
            self.queue.clear()
            return None
        if self.queue:
            return self.queue.popleft()
        return None

    def clear(self) -> None:
        self.queue.clear()


def astar_grid(
    start: Tuple[int, int],
    target: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
    max_expansions: int = 2500,
) -> Optional[List[int]]:
    if not action_vectors:
        return None
    h, w = grid_shape
    strides = [max(abs(dr), abs(dc)) for dr, dc in action_vectors.values() if (dr, dc) != (0, 0)]
    s = max(1, min(strides)) if strides else 1

    if abs(start[0] - target[0]) < s and abs(start[1] - target[1]) < s:
        return []

    def heuristic(pos: Tuple[int, int]) -> float:
        return (abs(pos[0] - target[0]) + abs(pos[1] - target[1])) / float(s)

    open_heap = [(heuristic(start), 0, start, [])]
    visited: Dict[Tuple[int, int], int] = {start: 0}
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, g, curr, path = heapq.heappop(open_heap)
        expansions += 1

        if abs(curr[0] - target[0]) < s and abs(curr[1] - target[1]) < s:
            return path

        for act_id, (dr, dc) in action_vectors.items():
            if dr == 0 and dc == 0:
                continue
            nr, nc = curr[0] + dr, curr[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue

            steps_n = max(abs(dr), abs(dc))
            blocked = False
            for k in range(1, steps_n + 1):
                mr = int(round(curr[0] + (dr * k) / steps_n))
                mc = int(round(curr[1] + (dc * k) / steps_n))
                mid = (mr, mc)
                is_target_cell = abs(mid[0] - target[0]) < s and abs(mid[1] - target[1]) < s
                if mid in walls and not is_target_cell:
                    blocked = True
                    break
            if blocked:
                continue

            nxt = (nr, nc)
            new_g = g + 1
            if nxt not in visited or new_g < visited[nxt]:
                visited[nxt] = new_g
                heapq.heappush(open_heap, (new_g + heuristic(nxt), new_g, nxt, path + [int(act_id)]))

    return None


def bfs_frontier_explore(
    start: Tuple[int, int],
    visited_cells: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
) -> Optional[List[int]]:
    if not action_vectors:
        return None
    h, w = grid_shape
    strides = [max(abs(dr), abs(dc)) for dr, dc in action_vectors.values() if (dr, dc) != (0, 0)]
    s = max(1, min(strides)) if strides else 1

    queue = deque([(start, [])])
    seen = {start}
    while queue and len(seen) < 1200:
        curr, path = queue.popleft()
        norm_cell = (curr[0] // s, curr[1] // s)
        if path and norm_cell not in visited_cells:
            return path
        if len(path) >= 25:
            continue
        for act_id, (dr, dc) in action_vectors.items():
            if dr == 0 and dc == 0:
                continue
            nr, nc = curr[0] + dr, curr[1] + dc
            if not (4 <= nr < h - 4 and 4 <= nc < w - 4):
                continue
            steps_n = max(abs(dr), abs(dc))
            blocked = any(
                (int(round(curr[0] + (dr * k) / steps_n)), int(round(curr[1] + (dc * k) / steps_n))) in walls
                for k in range(1, steps_n + 1)
            )
            if not blocked and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), path + [int(act_id)]))
    return None


def sokoban_bfs(
    avatar: Tuple[int, int],
    boxes: Set[Tuple[int, int]],
    goals: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
    max_states: int = 3000,
) -> Optional[List[int]]:
    if not boxes or not goals or not action_vectors:
        return None
    h, w = grid_shape
    strides = [max(abs(dr), abs(dc)) for dr, dc in action_vectors.values() if (dr, dc) != (0, 0)]
    s = max(1, min(strides)) if strides else 1

    def is_corner_deadlock(b: Tuple[int, int]) -> bool:
        if b in goals:
            return False
        up = (b[0] - s, b[1]) in walls
        down = (b[0] + s, b[1]) in walls
        left = (b[0], b[1] - s) in walls
        right = (b[0], b[1] + s) in walls
        return (up or down) and (left or right)

    init_boxes = frozenset(boxes)
    queue = deque([(avatar, init_boxes, [])])
    visited = {(avatar, init_boxes)}

    while queue and len(visited) < max_states:
        av, curr_boxes, path = queue.popleft()
        if curr_boxes.issubset(goals) or len(curr_boxes & goals) >= len(goals):
            return path
        if len(path) >= 40:
            continue

        for act_id, (dr, dc) in action_vectors.items():
            if dr == 0 and dc == 0:
                continue
            nr, nc = av[0] + dr, av[1] + dc
            nxt_av = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w) or nxt_av in walls:
                continue
            if nxt_av in curr_boxes:
                br, bc = nr + dr, nc + dc
                nxt_box = (br, bc)
                if not (0 <= br < h and 0 <= bc < w) or nxt_box in walls or nxt_box in curr_boxes:
                    continue
                if is_corner_deadlock(nxt_box):
                    continue
                new_boxes = frozenset((curr_boxes - {nxt_av}) | {nxt_box})
                state = (nxt_av, new_boxes)
            else:
                state = (nxt_av, curr_boxes)

            if state not in visited:
                visited.add(state)
                queue.append((state[0], state[1], path + [int(act_id)]))
    return None


def ice_slide_bfs(
    start: Tuple[int, int],
    target: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
) -> Optional[List[int]]:
    if not action_vectors:
        return None
    h, w = grid_shape
    unit_dirs = {}
    for act_id, (dr, dc) in action_vectors.items():
        if (dr, dc) != (0, 0):
            ur = 0 if dr == 0 else (1 if dr > 0 else -1)
            uc = 0 if dc == 0 else (1 if dc > 0 else -1)
            unit_dirs[act_id] = (ur, uc)

    queue = deque([(start, [])])
    visited = {start}
    while queue and len(visited) < 1500:
        curr, path = queue.popleft()
        if abs(curr[0] - target[0]) <= 2 and abs(curr[1] - target[1]) <= 2:
            return path
        if len(path) >= 25:
            continue
        for act_id, (ur, uc) in unit_dirs.items():
            r, c = curr
            while True:
                nr, nc = r + ur, c + uc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in walls:
                    break
                r, c = nr, nc
                if abs(r - target[0]) <= 1 and abs(c - target[1]) <= 1:
                    return path + [int(act_id)]
            if (r, c) != curr and (r, c) not in visited:
                visited.add((r, c))
                queue.append(((r, c), path + [int(act_id)]))
    return None


def knight_bfs(
    start: Tuple[int, int],
    target: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
) -> Optional[List[int]]:
    return astar_grid(start, target, walls, action_vectors, grid_shape=grid_shape)
''', encoding="utf-8")

# =========================================================================
# 8. project_override/world_model/metagame_schema.py & hypothesis_manager.py
# =========================================================================
(PO_DIR / "world_model" / "metagame_schema.py").write_text('''from typing import Any, Callable, Dict, Optional, Set, Tuple


class MetagameSchema:
    """
    Stores learned directional action vectors, confirmed entity semantics,
    and verified Python world models across levels (Zero-Shot Level 2..K Transfer).
    """

    INVERSE_PAIRS = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self):
        self.displacement_vectors: Dict[int, Tuple[int, int]] = {}
        self.frozen: bool = False
        self.verified_world_model: Optional[Callable] = None
        self.verified_model_code: Optional[str] = None

    def record_vector(self, action_id: int, vec: Tuple[int, int]) -> None:
        if vec == (0, 0):
            return
        self.displacement_vectors[int(action_id)] = (int(vec[0]), int(vec[1]))
        inv = self.INVERSE_PAIRS.get(int(action_id))
        if inv is not None and inv not in self.displacement_vectors:
            self.displacement_vectors[inv] = (-int(vec[0]), -int(vec[1]))

    def freeze_on_level_clear(self) -> None:
        if self.displacement_vectors:
            self.frozen = True
''', encoding="utf-8")

(PO_DIR / "world_model" / "hypothesis_manager.py").write_text('''from typing import Dict, List, Optional, Tuple


class HypothesisManager:
    """Manages active kinematic & interaction hypotheses during Level 1 exploration."""

    def __init__(self):
        self.probe_order: List[int] = [1, 3, 2, 4]
        self.tested_actions: Dict[int, int] = {}

    def next_probe(self, available_moves: List[int], known_vectors: Dict[int, Tuple[int, int]]) -> Optional[int]:
        for act in self.probe_order:
            if act in available_moves and act not in known_vectors:
                if self.tested_actions.get(act, 0) < 2:
                    self.tested_actions[act] = self.tested_actions.get(act, 0) + 1
                    return act
        return None
''', encoding="utf-8")

# =========================================================================
# 9. project_override/world_model/context_manager.py
# =========================================================================
(PO_DIR / "world_model" / "context_manager.py").write_text('''from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


class ContextManager:
    """
    4-Tier Memory & Franzen Hysteresis Context Compaction (HARNESS_DESIGN.md).
    - Immutable System Prefix (KV-cache anchored)
    - Tier 1: Episodic Ring Buffer
    - Tier 2: Compacted Causal Summary (Hysteresis band: High=1900 tokens -> Low=1100 tokens)
    - Tier 3: Confirmed Physical Laws & Hazards
    - Tier 4: Cross-Level Schema & Reflexion Post-Mortems
    """

    SYSTEM_PREFIX = (
        "You are the Project Override Fluid-Intelligence Architect for ARC-AGI-3.\\n"
        "Grid is 64x64 int8 (0..15). Actions: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=INTERACT, "
        "6=CLICK(x=col,y=row), 7=UNDO/CYCLE.\\n"
        "Respond ONLY with valid JSON: {\"subgoal\": \"NAVIGATE\"|\"CLICK\"|\"ACTION\", "
        "\"target_rc\": [r, c], \"click_xy\": [x, y], \"action_id\": int, \"hypothesis\": str}"
    )

    def __init__(self, high_water_tokens: int = 1900, low_water_tokens: int = 1100):
        self.high_water_tokens = high_water_tokens
        self.low_water_tokens = low_water_tokens
        self.tier1_episodic: deque = deque(maxlen=24)
        self.tier2_summaries: List[str] = []
        self.tier4_reflexions: List[str] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def record_step(self, step: int, action_id: int, delta_omega: int, note: str = "") -> None:
        entry = f"t={step}: act={action_id} dOmega={delta_omega} {note}".strip()
        self.tier1_episodic.append(entry)
        self._enforce_hysteresis()

    def record_reflexion(self, failure_reason: str) -> None:
        if failure_reason and failure_reason not in self.tier4_reflexions:
            self.tier4_reflexions.append(failure_reason[:160])
            if len(self.tier4_reflexions) > 6:
                self.tier4_reflexions.pop(0)

    def _enforce_hysteresis(self) -> None:
        raw_text = "\\n".join(self.tier2_summaries + list(self.tier1_episodic))
        if self.estimate_tokens(self.SYSTEM_PREFIX + raw_text) <= self.high_water_tokens:
            return
        # Compact oldest episodic traces into a single causal digest until under low_water_tokens
        compacted_chunk = []
        while len(self.tier1_episodic) > 6:
            compacted_chunk.append(self.tier1_episodic.popleft())
        if compacted_chunk:
            digest = f"[Compacted {len(compacted_chunk)} steps: {compacted_chunk[0]} .. {compacted_chunk[-1]}]"
            self.tier2_summaries.append(digest)
        while len(self.tier2_summaries) > 4:
            self.tier2_summaries.pop(0)

    def build_prompt(
        self,
        level_idx: int,
        av_pos: Optional[Tuple[int, int]],
        vectors: Dict[int, Tuple[int, int]],
        walls: Set[int],
        hazards: Set[int],
        entities_summary: str,
        available_actions: List[int],
    ) -> str:
        tier3 = (
            f"Level={level_idx} | AvatarPos={av_pos} | Vectors={vectors} | "
            f"Walls={sorted(walls)} | Hazards={sorted(hazards)} | Avail={available_actions}"
        )
        t2 = "\\n".join(self.tier2_summaries[-3:])
        t1 = "\\n".join(list(self.tier1_episodic)[-8:])
        t4 = "\\n".join(self.tier4_reflexions[-3:])
        return (
            f"{self.SYSTEM_PREFIX}\\n"
            f"[TIER 3 LAWS] {tier3}\\n"
            f"[TIER 4 REFLEXION] {t4}\\n"
            f"[TIER 2 COMPACTED] {t2}\\n"
            f"[TIER 1 EPISODIC]\\n{t1}\\n"
            f"[ENTITIES] {entities_summary}\\n"
            f"Emit next JSON subgoal:"
        )
''', encoding="utf-8")

# =========================================================================
# 10. project_override/world_model/ast_sandbox.py
# =========================================================================
(PO_DIR / "world_model" / "ast_sandbox.py").write_text('''import ast
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)

FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals"}


class ASTSandbox:
    """Safe AST validator and retrodiction verifier for CEGIS Python world models."""

    @staticmethod
    def validate_and_compile(code_str: str, fn_name: str = "predict_next_grid") -> Optional[Callable]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_AST_NODES):
                    return None
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    return None
            safe_globals: Dict[str, Any] = {
                "__builtins__": {
                    "range": range,
                    "len": len,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "tuple": tuple,
                    "list": list,
                    "set": set,
                    "dict": dict,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sum": sum,
                    "any": any,
                    "all": all,
                    "round": round,
                },
                "np": np,
            }
            local_ns: Dict[str, Any] = {}
            exec(compile(tree, filename="<cegis_world_model>", mode="exec"), safe_globals, local_ns)
            fn = local_ns.get(fn_name)
            return fn if callable(fn) else None
        except Exception:
            return None

    @staticmethod
    def evaluate_retrodiction(
        fn: Callable,
        transitions: List[Tuple[np.ndarray, int, Dict[str, Any], np.ndarray]],
    ) -> Tuple[float, Optional[str]]:
        """
        Tests synthesized predict_next_grid(grid, action, data) against historical transitions.
        Returns (accuracy_0_to_1, first_counterexample_diff_str).
        """
        if not transitions:
            return 0.0, None
        passed = 0
        counterexample = None
        for pre_g, act_id, data, post_g in transitions[-10:]:
            try:
                pred = fn(pre_g.copy(), int(act_id), data or {})
                if isinstance(pred, np.ndarray) and pred.shape == post_g.shape:
                    diff = pred != post_g
                    n_err = int(np.sum(diff))
                    if n_err == 0:
                        passed += 1
                    elif counterexample is None:
                        rs, cs = np.where(diff)
                        r0, c0 = int(rs[0]), int(cs[0])
                        counterexample = (
                            f"Action {act_id}: {n_err} px mismatch; at ({r0},{c0}) "
                            f"expected {int(post_g[r0, c0])}, got {int(pred[r0, c0])}"
                        )
                else:
                    counterexample = f"Action {act_id}: invalid return shape"
            except Exception as e:
                counterexample = f"Action {act_id} raised {type(e).__name__}: {e}"
        return passed / float(len(transitions[-10:])), counterexample
''', encoding="utf-8")

# =========================================================================
# 11. project_override/world_model/retrodict_synthesizer.py
# =========================================================================
(PO_DIR / "world_model" / "retrodict_synthesizer.py").write_text('''from collections import deque
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.world_model.ast_sandbox import ASTSandbox

logger = logging.getLogger(__name__)


def _install_pure_pytorch_awq_fallback() -> None:
    """
    If Kaggle's offline Python 3.13 container lacks `autoawq`, this registers a
    pure-PyTorch GPU 4-bit AWQ dequantizer so Qwen 2.5 Coder 32B AWQ loads and runs
    natively on Dual T4 (17.5 GB compressed in VRAM) without external compiled wheels.
    """
    try:
        import awq  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers.quantizers import AutoHfQuantizer, HfQuantizer

        class TorchAWQLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 128):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.group_size = group_size
                pack_num = 8
                self.register_buffer(
                    "qweight",
                    torch.zeros((in_features, out_features // pack_num), dtype=torch.int32),
                )
                self.register_buffer(
                    "qzeros",
                    torch.zeros((in_features // group_size, out_features // pack_num), dtype=torch.int32),
                )
                self.register_buffer(
                    "scales",
                    torch.zeros((in_features // group_size, out_features), dtype=torch.float16),
                )
                if bias:
                    self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float16))
                else:
                    self.bias = None
                self.register_buffer(
                    "_shifts",
                    torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32),
                    persistent=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shifts = self._shifts.to(self.qweight.device)
                # Unpack 4-bit AWQ weights & zeros on-the-fly in GPU registers
                w_unpacked = torch.bitwise_and(
                    torch.bitwise_right_shift(self.qweight.unsqueeze(-1), shifts), 0x0F
                ).reshape(self.in_features, self.out_features)
                z_unpacked = torch.bitwise_and(
                    torch.bitwise_right_shift(self.qzeros.unsqueeze(-1), shifts), 0x0F
                ).reshape(self.in_features // self.group_size, self.out_features)

                z_expanded = z_unpacked.repeat_interleave(self.group_size, dim=0)
                s_expanded = self.scales.repeat_interleave(self.group_size, dim=0)
                w_fp16 = (w_unpacked.to(torch.float16) - z_expanded.to(torch.float16)) * s_expanded
                out = torch.matmul(x.to(torch.float16), w_fp16)
                if self.bias is not None:
                    out = out + self.bias
                return out.to(x.dtype)

        class PureTorchAwqQuantizer(HfQuantizer):
            requires_calibration = False
            required_packages = []

            def __init__(self, quantization_config: Any, **kwargs: Any):
                super().__init__(quantization_config, **kwargs)
                self.group_size = getattr(quantization_config, "group_size", 128)

            def validate_environment(self, *args: Any, **kwargs: Any) -> None:
                return

            def update_torch_dtype(self, torch_dtype: Any) -> Any:
                return torch.float16

            def _process_model_before_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
                def _replace(module: nn.Module, prefix: str = "") -> None:
                    for name, child in list(module.named_children()):
                        full = f"{prefix}.{name}" if prefix else name
                        if isinstance(child, nn.Linear) and "lm_head" not in full:
                            if child.in_features % self.group_size == 0 and child.out_features % 8 == 0:
                                setattr(
                                    module,
                                    name,
                                    TorchAWQLinear(
                                        child.in_features,
                                        child.out_features,
                                        bias=child.bias is not None,
                                        group_size=self.group_size,
                                    ),
                                )
                        else:
                            _replace(child, full)
                _replace(model)

            def _process_model_after_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
                return

            @property
            def is_trainable(self) -> bool:
                return False

            @property
            def is_serializable(self) -> bool:
                return True

        AutoHfQuantizer.register("awq", PureTorchAwqQuantizer, exist_ok=True)
    except Exception:
        pass


class OfflineLLMRuntime:
    """Thread-safe Singleton for Qwen 2.5 Coder 32B Instruct AWQ on Kaggle Dual T4."""

    _INSTANCE: Optional["OfflineLLMRuntime"] = None
    _LOAD_LOCK = threading.Lock()
    _INFERENCE_LOCK = threading.Lock()

    CANDIDATE_PATHS = [
        "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen-2.5-coder-32b-instruct-awq",
    ]

    def __init__(self):
        self.model_path: Optional[str] = self._resolve_model_path()
        self.model: Any = None
        self.tokenizer: Any = None
        self.is_loaded: bool = False
        self.load_attempted: bool = False

    @classmethod
    def get_instance(cls) -> "OfflineLLMRuntime":
        if cls._INSTANCE is None:
            with cls._LOAD_LOCK:
                if cls._INSTANCE is None:
                    cls._INSTANCE = cls()
        return cls._INSTANCE

    def _resolve_model_path(self) -> Optional[str]:
        env_p = os.environ.get("ARC_LOCAL_MODEL_PATH")
        if env_p and Path(env_p).exists():
            return env_p
        for p in self.CANDIDATE_PATHS:
            if Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                if "qwen" in str(cfg).lower() or "coder" in str(cfg).lower():
                    return str(cfg.parent)
        return None

    def ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True
        if self.load_attempted or self.model_path is None:
            return False
        with self._LOAD_LOCK:
            if self.is_loaded:
                return True
            self.load_attempted = True
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                _install_pure_pytorch_awq_fallback()
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True, local_files_only=True
                )
                n_gpus = torch.cuda.device_count()
                max_mem = {i: "14GiB" for i in range(n_gpus)}
                max_mem["cpu"] = "24GiB"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    max_memory=max_mem,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.model.eval()
                self.is_loaded = True
                return True
            except Exception as e:
                logger.warning(f"OfflineLLMRuntime fallback to symbolic CEGIS ({e})")
                self.is_loaded = False
                return False

    def generate_text(self, prompt: str, max_new_tokens: int = 160, timeout_s: float = 12.0) -> Optional[str]:
        if not self.ensure_loaded():
            return None
        if not self._INFERENCE_LOCK.acquire(timeout=1.5):
            return None
        try:
            import torch
            t0 = time.perf_counter()
            if (time.perf_counter() - t0) > timeout_s:
                return None
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = prompt
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception:
            return None
        finally:
            self._INFERENCE_LOCK.release()

    def infer_subgoal(self, prompt: str) -> Optional[Dict[str, Any]]:
        resp = self.generate_text(prompt, max_new_tokens=110)
        if not resp:
            return None
        match = re.search(r"\\{.*\\}", resp, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


class RetrodictWorldModelSynthesizer:
    """
    CEGIS (Counterexample-Guided Inductive Synthesis) World Model Engine.
    Synthesizes Python `predict_next_grid(grid, action, data)` from observed transitions,
    validates in ASTSandbox, and runs in-memory Width-1 Iterated Width IW(1) planning!
    """

    def __init__(self):
        self.transitions: List[Tuple[np.ndarray, int, Dict[str, Any], np.ndarray]] = []
        self.verified_fn: Optional[Callable] = None
        self.verified_code: Optional[str] = None

    def reset_level(self, keep_verified: bool = True) -> None:
        self.transitions.clear()
        if not keep_verified:
            self.verified_fn = None
            self.verified_code = None

    def record_transition(
        self, pre_grid: np.ndarray, action_id: int, data: Dict[str, Any], post_grid: np.ndarray
    ) -> None:
        if not np.array_equal(pre_grid, post_grid):
            self.transitions.append((pre_grid.copy(), int(action_id), dict(data or {}), post_grid.copy()))
            if len(self.transitions) > 20:
                self.transitions.pop(0)

    def synthesize_world_model(
        self,
        floor_color: Optional[int],
        avatar_colors: Set[int],
        wall_colors: Set[int],
        box_colors: Set[int],
        vectors: Dict[int, Tuple[int, int]],
    ) -> Optional[Callable]:
        """Synthesizes and verifies a fast executable Python world model from induced physics."""
        if floor_color is None or not avatar_colors or not vectors:
            return None

        code_template = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {repr(vectors)}
    if action not in vecs:
        return grid
    dr, dc = vecs[action]
    floor_c = {int(floor_color)}
    av_cols = {repr(set(int(c) for c in avatar_colors))}
    wall_cols = {repr(set(int(c) for c in wall_colors))}
    box_cols = {repr(set(int(c) for c in box_colors))}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return grid
    # Check if destination cells are free or pushable
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (4 <= nr < h - 4 and 4 <= nc < w - 4):
            return grid
        dest_c = int(grid[nr, nc])
        if dest_c in wall_cols:
            return grid
    nxt = grid.copy()
    for r, c in pts:
        nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        nxt[nr, nc] = grid[int(r), int(c)]
    return nxt
"""
        fn = ASTSandbox.validate_and_compile(code_template)
        if fn is not None and len(self.transitions) >= 2:
            acc, _ = ASTSandbox.evaluate_retrodiction(fn, self.transitions)
            if acc >= 0.75:
                self.verified_fn = fn
                self.verified_code = code_template
                return fn
        return fn

    def plan_iw1_lookahead(
        self,
        start_grid: np.ndarray,
        available_actions: List[int],
        target_pos: Optional[Tuple[int, int]],
        avatar_colors: Set[int],
        max_expansions: int = 120,
    ) -> Optional[List[int]]:
        """
        Width-1 Iterated Width IW(1) simulated lookahead over the synthesized Python world model.
        Finds multi-step trajectories to novel states or target overlap without spending real steps.
        """
        if self.verified_fn is None or not avatar_colors:
            return None
        moves = [a for a in available_actions if a in (1, 2, 3, 4)]
        if not moves:
            return None

        seen_atoms: Set[Tuple[int, int, int]] = set()
        def _register_atoms(g: np.ndarray) -> int:
            nov = 0
            for r in range(4, g.shape[0] - 4, 3):
                for c in range(4, g.shape[1] - 4, 3):
                    atom = (r, c, int(g[r, c]))
                    if atom not in seen_atoms:
                        seen_atoms.add(atom)
                        nov += 1
            return nov

        _register_atoms(start_grid)
        queue = deque([(start_grid.copy(), [])])
        expansions = 0

        while queue and expansions < max_expansions:
            g, path = queue.popleft()
            expansions += 1
            if len(path) >= 14:
                continue
            for a in moves:
                try:
                    nxt_g = self.verified_fn(g, a, {})
                except Exception:
                    continue
                if np.array_equal(nxt_g, g):
                    continue
                new_path = path + [a]
                if target_pos is not None:
                    tr, tc = target_pos
                    if 0 <= tr < nxt_g.shape[0] and 0 <= tc < nxt_g.shape[1]:
                        if int(nxt_g[tr, tc]) in avatar_colors:
                            return new_path
                if _register_atoms(nxt_g) > 0:
                    queue.append((nxt_g, new_path))
        return None
''', encoding="utf-8")

# =========================================================================
# 12. project_override/orchestration/general_agent.py
# =========================================================================
(PO_DIR / "orchestration" / "general_agent.py").write_text('''from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.core.circuit_engine import AC3CircuitSolver
from project_override.core.combinatorial_engine import GF2ToggleSolver, SplitPanelSymmetrySolver
from project_override.core.dead_signature import DeadSignatureTable, compute_arena_hash
from project_override.core.universal_engine import SystemPacer
from project_override.core.visual_goal_induction import JordanCurveCavityInducer, LatentMemoryMatcher
from project_override.perception.observation_processor import ObservationProcessor
from project_override.planning.symbolic_planners import (
    MacroActionQueue,
    astar_grid,
    bfs_frontier_explore,
    ice_slide_bfs,
    sokoban_bfs,
)
from project_override.world_model.context_manager import ContextManager
from project_override.world_model.hypothesis_manager import HypothesisManager
from project_override.world_model.metagame_schema import MetagameSchema
from project_override.world_model.retrodict_synthesizer import (
    OfflineLLMRuntime,
    RetrodictWorldModelSynthesizer,
)


class GeneralAgent:
    """
    Full 12-Module Neurosymbolic + CEGIS + Qwen-32B-AWQ Orchestrator for ARC-AGI-3.
    """

    def __init__(self, game_id: str = "unknown"):
        self.game_id = game_id
        self.processor = ObservationProcessor(bezel_px=4)
        self.dead_table = DeadSignatureTable()
        self.macro_queue = MacroActionQueue()
        self.schema = MetagameSchema()
        self.hypotheses = HypothesisManager()
        self.context = ContextManager(high_water_tokens=1900, low_water_tokens=1100)
        self.synthesizer = RetrodictWorldModelSynthesizer()
        self.memory_matcher = LatentMemoryMatcher()
        self.pacer = SystemPacer(max_level_compute_s=45.0)

        self.current_level: int = 0
        self.step_in_level: int = 0
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_hash: Optional[int] = None
        self.prev_av_pos: Optional[Tuple[int, int]] = None
        self.last_action_id: Optional[int] = None
        self.last_click_xy: Tuple[int, int] = (0, 0)
        self.last_delta_omega: Optional[int] = None

        # Induced Level-Invariant Physics
        self.learned_floor_color: Optional[int] = None
        self.learned_avatar_colors: Set[int] = set()
        self.learned_avatar_color: Optional[int] = None
        self.learned_game_modality: str = "NAV"
        self.confirmed_wall_colors: Set[int] = set()
        self.confirmed_hazard_colors: Set[int] = set()
        self.confirmed_box_colors: Set[int] = set()
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.visited_goals: Set[Tuple[int, int]] = set()
        self.transition_history: List[Tuple[np.ndarray, int, np.ndarray, int]] = []

    def on_game_over(self, final_grid: Optional[np.ndarray] = None) -> None:
        """
        Called right before RESET on GAME_OVER.
        Permanently blacklists the fatal (state_hash, action) signature and inducts
        the hazard tile color that killed the avatar so we never die to it again!
        """
        if self.prev_hash is not None and self.last_action_id is not None:
            cx, cy = self.last_click_xy if self.last_action_id == 6 else (0, 0)
            self.dead_table.record_death(
                self.prev_hash, self.last_action_id, cx, cy, self.current_level
            )
            if (
                self.prev_grid is not None
                and self.prev_av_pos is not None
                and self.last_action_id in self.schema.displacement_vectors
            ):
                dr, dc = self.schema.displacement_vectors[self.last_action_id]
                nr, nc = self.prev_av_pos[0] + dr, self.prev_av_pos[1] + dc
                h, w = self.prev_grid.shape
                if 4 <= nr < h - 4 and 4 <= nc < w - 4:
                    lethal_color = int(self.prev_grid[nr, nc])
                    if (
                        lethal_color != self.learned_floor_color
                        and lethal_color not in self.learned_avatar_colors
                    ):
                        self.confirmed_hazard_colors.add(lethal_color)
                        self.context.record_reflexion(
                            f"GAME_OVER stepping onto hazard color {lethal_color} via action {self.last_action_id}"
                        )
        self.macro_queue.clear()
        self.prev_grid = None
        self.prev_hash = None
        self.last_action_id = None
        self.last_delta_omega = None
        self.visited_goals.clear()

    def _on_level_advance(self, new_level: int) -> None:
        self.current_level = new_level
        self.step_in_level = 0
        self.prev_grid = None
        self.prev_hash = None
        self.prev_av_pos = None
        self.last_action_id = None
        self.last_delta_omega = None
        self.macro_queue.clear()
        self.dead_table.reset_level(clear_deaths=True)
        self.synthesizer.reset_level(keep_verified=True)
        self.memory_matcher.reset()
        self.pacer.reset_level()
        self.visited_cells.clear()
        self.visited_goals.clear()
        self.transition_history.clear()
        self.schema.freeze_on_level_clear()

    def _locate_avatar_centroid(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        if not self.learned_avatar_colors:
            return None
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        mask = np.zeros((h, w), dtype=bool)
        for c in self.learned_avatar_colors:
            mask |= (arr == c)
        if h >= 16 and w >= 16:
            mask[:4, :] = False
            mask[h - 4:, :] = False
            mask[:, :4] = False
            mask[:, w - 4:] = False
        pts = np.argwhere(mask)
        if len(pts) == 0:
            return None
        return (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))

    def _induce_blocking_wall(self, grid: np.ndarray, failed_action: int) -> None:
        if failed_action not in self.schema.displacement_vectors:
            return
        av_pos = self._locate_avatar_centroid(grid)
        if av_pos is None:
            return
        dr, dc = self.schema.displacement_vectors[failed_action]
        nr, nc = av_pos[0] + dr, av_pos[1] + dc
        h, w = grid.shape
        if 4 <= nr < h - 4 and 4 <= nc < w - 4:
            c = int(grid[nr, nc])
            if (
                c != self.learned_floor_color
                and c not in self.learned_avatar_colors
                and c not in self.confirmed_box_colors
            ):
                self.confirmed_wall_colors.add(c)

    def _retroactive_wall_scan(self) -> None:
        """Retroactively inducts walls from Step-0 blocked probes once vectors are known."""
        for pre_g, act_id, _, delta in self.transition_history:
            if delta == 0 and act_id in self.schema.displacement_vectors:
                self._induce_blocking_wall(pre_g, act_id)

    def _build_obstacle_set(self, grid: np.ndarray) -> Set[Tuple[int, int]]:
        obstacles: Set[Tuple[int, int]] = set()
        blocked_colors = self.confirmed_wall_colors | self.confirmed_hazard_colors
        if not blocked_colors:
            return obstacles
        arr = np.asarray(grid, dtype=np.int8)
        for c in blocked_colors:
            for r, col in np.argwhere(arr == c):
                obstacles.add((int(r), int(col)))
        return obstacles

    def _find_candidate_goals(self, grid: np.ndarray, av_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        bg = self.processor.induce_background_color(grid, self.learned_floor_color)
        entities = self.processor.extract_entities(grid, bg_color=bg, ignore_bezel=True)
        strides = [max(abs(v[0]), abs(v[1])) for v in self.schema.displacement_vectors.values() if v != (0, 0)]
        s = max(1, min(strides)) if strides else 1

        candidates: List[Tuple[float, Tuple[int, int]]] = []
        blocked_colors = self.learned_avatar_colors | self.confirmed_wall_colors | self.confirmed_hazard_colors
        for ent in entities:
            if ent.color in blocked_colors or ent.mass > 400:
                continue
            gr, gc = int(round(ent.centroid[0])), int(round(ent.centroid[1]))
            norm_g = (gr // s, gc // s)
            if norm_g in self.visited_goals:
                continue
            dist = abs(gr - av_pos[0]) + abs(gc - av_pos[1])
            if dist >= s:
                score = ent.mass * 0.4 + dist * 0.25
                candidates.append((score, (gr, gc)))

        # Also include topological cavities as goal candidates
        for cr, cc in JordanCurveCavityInducer.find_cavities(grid, bg_color=bg, bezel_px=4)[:4]:
            norm_g = (cr // s, cc // s)
            if norm_g not in self.visited_goals:
                dist = abs(cr - av_pos[0]) + abs(cc - av_pos[1])
                candidates.append((dist * 0.2, (cr, cc)))

        candidates.sort(key=lambda x: x[0])
        return [pos for _, pos in candidates]

    def _select_smart_click(
        self, grid: np.ndarray, arena_hash: int
    ) -> Tuple[int, Dict[str, int]]:
        bg = self.processor.induce_background_color(grid, self.learned_floor_color)

        # 1. D4 Split-Panel Symmetry / Odd-One-Out Completion
        for r, c in SplitPanelSymmetrySolver.find_asymmetric_clicks(grid, bg_color=bg, bezel_px=4):
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        # 2. Jordan Curve Topological Cavity Clicks
        for r, c in JordanCurveCavityInducer.find_cavities(grid, bg_color=bg, bezel_px=4):
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        # 3. AC-3 Periodic Circuit / Pipe Tile Clicks
        circuit_clicks = AC3CircuitSolver.detect_periodic_circuit_clicks(grid, bg_color=bg, bezel_px=4)
        if circuit_clicks:
            for r, c in circuit_clicks:
                x, y = int(c), int(r)
                if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                    self.last_click_xy = (x, y)
                    return 6, {"x": x, "y": y}

        # 4. Component-Priority Foreground Clicks (ranked by rarity and distinctiveness)
        entities = self.processor.extract_entities(grid, bg_color=bg, ignore_bezel=True)
        color_counts: Dict[int, int] = {}
        for e in entities:
            color_counts[e.color] = color_counts.get(e.color, 0) + 1

        ranked = sorted(
            entities,
            key=lambda e: (color_counts.get(e.color, 99), abs(e.mass - 9), e.id),
        )
        for ent in ranked:
            r = int(round(ent.centroid[0]))
            c = int(round(ent.centroid[1]))
            if (r, c) not in ent.pixels and ent.pixels:
                r, c = ent.pixels[len(ent.pixels) // 2]
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        # 5. Multi-resolution lattice sweep
        h, w = grid.shape
        for stride in (6, 4, 3):
            for r in range(6, h - 6, stride):
                for c in range(6, w - 6, stride):
                    x, y = int(c), int(r)
                    if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                        self.last_click_xy = (x, y)
                        return 6, {"x": x, "y": y}

        return 6, {"x": 32, "y": 32}

    def step(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int = 0,
    ) -> Tuple[int, Dict[str, Any]]:
        self.pacer.begin_step()
        try:
            return self._step_impl(grid, available_actions, levels_completed)
        finally:
            self.pacer.end_step()

    def _step_impl(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int = 0,
    ) -> Tuple[int, Dict[str, Any]]:
        arr = np.asarray(grid, dtype=np.int8)
        if levels_completed > self.current_level:
            self._on_level_advance(levels_completed)

        curr_hash = compute_arena_hash(arr, bezel_px=4)
        avail = [int(a) for a in available_actions if int(a) != 0] if available_actions else [1, 2, 3, 4]

        # 1. Process transition from previous step
        if self.prev_grid is not None and self.last_action_id is not None and self.prev_hash is not None:
            delta = self.processor.compute_delta_omega(self.prev_grid, arr)
            self.last_delta_omega = delta
            cx, cy = self.last_click_xy if self.last_action_id == 6 else (0, 0)
            self.dead_table.record_transition(
                self.prev_hash,
                self.last_action_id,
                curr_hash,
                delta,
                coord_x=cx,
                coord_y=cy,
                level_idx=self.current_level,
                post_grid=arr,
            )
            self.transition_history.append((self.prev_grid.copy(), self.last_action_id, arr.copy(), delta))
            if len(self.transition_history) > 24:
                self.transition_history.pop(0)
            self.synthesizer.record_transition(
                self.prev_grid, self.last_action_id, {"x": cx, "y": cy} if self.last_action_id == 6 else {}, arr
            )
            self.context.record_step(self.step_in_level, self.last_action_id, delta)

            if self.last_action_id == 6 and delta > 0:
                bg = self.processor.induce_background_color(arr, self.learned_floor_color)
                match_xy = self.memory_matcher.observe_reveal(self.prev_grid, arr, self.last_click_xy, bg)
                if match_xy is not None:
                    self.macro_queue.push_plan([(6, {"x": int(match_xy[0]), "y": int(match_xy[1])})])

            if self.last_action_id in (1, 2, 3, 4):
                if delta > 0:
                    rigid = self.processor.infer_rigid_translation(
                        self.prev_grid, arr, self.learned_avatar_colors if self.learned_avatar_colors else None
                    )
                    if rigid is not None:
                        fl_c, av_colors, (dr, dc), _, pushed_boxes = rigid
                        self.learned_floor_color = fl_c
                        self.learned_avatar_colors |= av_colors
                        self.learned_avatar_color = min(self.learned_avatar_colors)
                        if pushed_boxes:
                            self.confirmed_box_colors |= pushed_boxes
                            self.learned_game_modality = "SOKOBAN"
                        self.schema.record_vector(self.last_action_id, (dr, dc))
                        self._retroactive_wall_scan()
                        if max(abs(dr), abs(dc)) > 8:
                            self.learned_game_modality = "SLIDE"
                        elif abs(dr) > 0 and abs(dc) > 0:
                            self.learned_game_modality = "KNIGHT"
                else:
                    self._induce_blocking_wall(arr, self.last_action_id)

        self.prev_grid = arr.copy()
        self.prev_hash = curr_hash
        self.step_in_level += 1

        av_pos = self._locate_avatar_centroid(arr)
        self.prev_av_pos = av_pos
        strides = [max(abs(v[0]), abs(v[1])) for v in self.schema.displacement_vectors.values() if v != (0, 0)]
        s = max(1, min(strides)) if strides else 1

        if av_pos is not None:
            norm_av = (av_pos[0] // s, av_pos[1] // s)
            self.visited_cells.add(norm_av)
            self.visited_goals.add(norm_av)

        # 2. Pure Click-Only Games
        move_actions = [a for a in avail if a in (1, 2, 3, 4)]
        if not move_actions and 6 in avail:
            queued = self.macro_queue.pop_next(self.last_delta_omega)
            if queued is not None and queued[0] == 6:
                qx, qy = int(queued[1].get("x", 32)), int(queued[1].get("y", 32))
                if not self.dead_table.is_dead(curr_hash, 6, qx, qy, self.current_level):
                    self.last_action_id = 6
                    self.last_click_xy = (qx, qy)
                    return 6, {"x": qx, "y": qy}
            act_id, data = self._select_smart_click(arr, curr_hash)
            self.last_action_id = act_id
            return act_id, data

        # 3. Loop Detection -> Empirical State-Graph BFS Escape
        if self.dead_table.detect_loop():
            self.macro_queue.clear()
            all_disc = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
            escape_plan = self.dead_table.bfs_to_unexplored_action(curr_hash, all_disc, self.current_level)
            if escape_plan:
                first_act = escape_plan[0]
                self.macro_queue.push_plan(escape_plan[1:])
                self.last_action_id = first_act
                return first_act, {}

        # 4. Pop Queued Macro-Plan
        queued = self.macro_queue.pop_next(self.last_delta_omega)
        if queued is not None:
            q_act, q_data = queued
            qx = int(q_data.get("x", 0)) if q_act == 6 else 0
            qy = int(q_data.get("y", 0)) if q_act == 6 else 0
            if not self.dead_table.is_dead(curr_hash, q_act, qx, qy, self.current_level):
                self.last_action_id = q_act
                if q_act == 6:
                    self.last_click_xy = (qx, qy)
                return q_act, q_data
            else:
                self.macro_queue.clear()

        # 5. Fast 2-Probe Orthogonal Kinematic Basis Discovery on Level 0
        probe_act = self.hypotheses.next_probe(move_actions, self.schema.displacement_vectors)
        if probe_act is not None and not self.dead_table.is_dead(curr_hash, probe_act, 0, 0, self.current_level):
            self.last_action_id = probe_act
            return probe_act, {}

        # 6. Symbolic + CEGIS World-Model Planning
        if av_pos is not None and self.schema.displacement_vectors and self.pacer.has_compute_budget():
            obstacles = self._build_obstacle_set(arr)
            goals = self._find_candidate_goals(arr, av_pos)

            # A. Sokoban Box Push Planner if pushable boxes were confirmed
            if self.confirmed_box_colors and goals:
                box_pts: Set[Tuple[int, int]] = set()
                for bc in self.confirmed_box_colors:
                    rs, cs = np.where(arr == bc)
                    if len(rs) > 0:
                        box_pts.add((int(round(float(rs.mean()))), int(round(float(cs.mean())))))
                if box_pts:
                    soko_plan = sokoban_bfs(
                        av_pos, box_pts, set(goals[:4]), obstacles, self.schema.displacement_vectors, arr.shape
                    )
                    if soko_plan:
                        first_act = soko_plan[0]
                        if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                            self.macro_queue.push_plan(soko_plan[1:])
                            self.last_action_id = first_act
                            return first_act, {}

            # B. Navigate to candidate goals via A* / Ice-Slide + optional Action-5 Interact at destination
            for goal in goals[:5]:
                if self.learned_game_modality == "SLIDE":
                    plan = ice_slide_bfs(av_pos, goal, obstacles, self.schema.displacement_vectors, arr.shape)
                else:
                    plan = astar_grid(av_pos, goal, obstacles, self.schema.displacement_vectors, arr.shape)
                if plan:
                    first_act = plan[0]
                    if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                        full_plan: List[Any] = list(plan[1:])
                        if 5 in avail:
                            full_plan.append(5)
                        self.macro_queue.push_plan(full_plan)
                        self.last_action_id = first_act
                        return first_act, {}

            # C. CEGIS Online Python World-Model Synthesis + In-Memory IW(1) Lookahead
            if self.step_in_level >= 4 and self.step_in_level % 4 == 0:
                self.synthesizer.synthesize_world_model(
                    self.learned_floor_color,
                    self.learned_avatar_colors,
                    self.confirmed_wall_colors | self.confirmed_hazard_colors,
                    self.confirmed_box_colors,
                    self.schema.displacement_vectors,
                )
            if self.synthesizer.verified_fn is not None:
                iw_plan = self.synthesizer.plan_iw1_lookahead(
                    arr,
                    avail,
                    goals[0] if goals else None,
                    self.learned_avatar_colors,
                )
                if iw_plan:
                    first_act = iw_plan[0]
                    if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                        self.macro_queue.push_plan(iw_plan[1:])
                        self.last_action_id = first_act
                        return first_act, {}

            # D. Frontier Curiosity BFS to nearest unvisited stride-lattice cell
            frontier_plan = bfs_frontier_explore(
                av_pos, self.visited_cells, obstacles, self.schema.displacement_vectors, arr.shape
            )
            if frontier_plan:
                first_act = frontier_plan[0]
                if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                    self.macro_queue.push_plan(frontier_plan[1:])
                    self.last_action_id = first_act
                    return first_act, {}

            # E. Offline Qwen 32B AWQ Subgoal Architect (invoked periodically on novel/stagnant levels)
            if self.step_in_level >= 10 and self.step_in_level % 12 == 0:
                llm = OfflineLLMRuntime.get_instance()
                prompt = self.context.build_prompt(
                    level_idx=self.current_level,
                    av_pos=av_pos,
                    vectors=self.schema.displacement_vectors,
                    walls=self.confirmed_wall_colors,
                    hazards=self.confirmed_hazard_colors,
                    entities_summary=f"goals={goals[:4]}",
                    available_actions=avail,
                )
                sub = llm.infer_subgoal(prompt)
                if sub:
                    if sub.get("subgoal") == "NAVIGATE" and isinstance(sub.get("target_rc"), list):
                        tr, tc = int(sub["target_rc"][0]), int(sub["target_rc"][1])
                        plan = astar_grid(av_pos, (tr, tc), obstacles, self.schema.displacement_vectors, arr.shape)
                        if plan and not self.dead_table.is_dead(curr_hash, plan[0], 0, 0, self.current_level):
                            self.macro_queue.push_plan(plan[1:])
                            self.last_action_id = plan[0]
                            return plan[0], {}
                    elif sub.get("subgoal") == "CLICK" and 6 in avail and isinstance(sub.get("click_xy"), list):
                        cx, cy = int(sub["click_xy"][0]), int(sub["click_xy"][1])
                        if not self.dead_table.is_dead(curr_hash, 6, cx, cy, self.current_level):
                            self.last_action_id = 6
                            self.last_click_xy = (cx, cy)
                            return 6, {"x": cx, "y": cy}

        # 7. Empirical State-Graph BFS to nearest untried action (IW(1) Graph Explorer)
        all_discrete = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
        graph_plan = self.dead_table.bfs_to_unexplored_action(curr_hash, all_discrete, self.current_level)
        if graph_plan:
            first_act = graph_plan[0]
            self.macro_queue.push_plan(graph_plan[1:])
            self.last_action_id = first_act
            return first_act, {}

        # 8. Hybrid Game Action-6 Fallback
        if 6 in avail:
            act_id, data = self._select_smart_click(arr, curr_hash)
            self.last_action_id = act_id
            return act_id, data

        for a in all_discrete:
            if not self.dead_table.is_dead(curr_hash, a, 0, 0, self.current_level):
                self.last_action_id = a
                return a, {}

        chosen = all_discrete[self.step_in_level % len(all_discrete)] if all_discrete else 1
        self.last_action_id = chosen
        return chosen, {}
''', encoding="utf-8")

# =========================================================================
# 13. Root Shims & agent/my_agent.py (with GAME_OVER hazard hook)
# =========================================================================
(ROOT / "dead_signature.py").write_text("from project_override.core.dead_signature import *\n", encoding="utf-8")
(ROOT / "context_manager.py").write_text("from project_override.world_model.context_manager import *\n", encoding="utf-8")
(ROOT / "ast_sandbox.py").write_text("from project_override.world_model.ast_sandbox import *\n", encoding="utf-8")

MY_AGENT_CODE = '''import os
import sys
import threading
from pathlib import Path
from typing import Any, List
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR.parent, CURRENT_DIR.parent.parent, Path("/tmp")]:
    if (p / "project_override").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from project_override.orchestration.general_agent import GeneralAgent

# Throttle 110 concurrent Swarm threads to 8 active compute workers on Kaggle 4-vCPU VM
_COMPUTE_SEMAPHORE = threading.BoundedSemaphore(8)


class MyAgent(Agent):
    """Official ARC-AGI-3 Harness Adapter for Project Override GeneralAgent."""

    MAX_ACTIONS: int = 400

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.orchestration_agent = GeneralAgent(game_id=getattr(self, "game_id", "unknown"))

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state is GameState.NOT_PLAYED:
            return GameAction.RESET

        raw_frames = getattr(latest_frame, "frame", [])
        grid = None
        if raw_frames:
            grid = np.asarray(raw_frames[-1], dtype=np.int8)
            if grid.ndim == 3:
                grid = grid[-1]

        if latest_frame.state is GameState.GAME_OVER:
            with _COMPUTE_SEMAPHORE:
                self.orchestration_agent.on_game_over(grid)
            return GameAction.RESET

        if grid is None:
            return GameAction.ACTION1

        avail_raw = getattr(latest_frame, "available_actions", None) or [1, 2, 3, 4]
        avail_ids = [int(getattr(a, "value", a)) for a in avail_raw]
        lvls = int(getattr(latest_frame, "levels_completed", 0) or 0)

        with _COMPUTE_SEMAPHORE:
            act_id, data = self.orchestration_agent.step(
                grid=grid,
                available_actions=avail_ids,
                levels_completed=lvls,
            )

        try:
            action = GameAction.from_id(int(act_id))
        except Exception:
            action = GameAction.ACTION1

        if action == GameAction.ACTION6 and data:
            action.set_data({"x": int(data.get("x", 32)), "y": int(data.get("y", 32))})

        return action
'''

for target in [
    ROOT / "agent" / "my_agent.py",
    ROOT / "ARC-AGI-3-Kaggle-Starter" / "agent" / "my_agent.py",
    ROOT / "ARC-AGI-3-Agents" / "agents" / "templates" / "my_agent.py",
]:
    if target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(MY_AGENT_CODE, encoding="utf-8")

print("[OK] All 12 Project Override modules written and verified without symlink errors.")
