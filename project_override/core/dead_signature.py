import hashlib
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class DynamicHUDMasker:
    """
    Tycho Abstained-Cell HUD & Timer-Bar Separator (arXiv:2607.28287).
    Identifies border/corner HUD cells that change monotonically on every action
    (such as 64-cell move-budget bars or life counters) so they never corrupt
    arena hashes, delta_omega, or world-model retrodiction.
    """

    def __init__(self, shape: Tuple[int, int] = (64, 64), default_bezel: int = 4):
        self.shape = shape
        self.default_bezel = default_bezel
        self.hud_mask = np.zeros(shape, dtype=bool)
        self.change_counts = np.zeros(shape, dtype=np.int32)
        self.total_transitions = 0
        self._init_bezel()

    def _init_bezel(self) -> None:
        h, w = self.shape
        b = self.default_bezel
        if h > 2 * b and w > 2 * b:
            self.hud_mask[:b, :] = True
            self.hud_mask[h - b:, :] = True
            self.hud_mask[:, :b] = True
            self.hud_mask[:, w - b:] = True

    def reset_level(self) -> None:
        self.change_counts.fill(0)
        self.total_transitions = 0

    def observe_transition(self, pre_grid: np.ndarray, post_grid: np.ndarray) -> None:
        if pre_grid.shape != self.shape or post_grid.shape != self.shape:
            return
        diff = pre_grid != post_grid
        changed_px = int(np.sum(diff))
        if 0 < changed_px <= 120:
            self.total_transitions += 1
            self.change_counts[diff] += 1
            if self.total_transitions >= 3:
                h, w = self.shape
                border_band = np.zeros((h, w), dtype=bool)
                border_band[:8, :] = True
                border_band[h - 8:, :] = True
                border_band[:, :8] = True
                border_band[:, w - 8:] = True
                frequent = (self.change_counts >= max(2, self.total_transitions - 1)) & border_band
                self.hud_mask |= frequent

    def mask_grid(self, grid: np.ndarray) -> np.ndarray:
        arr = np.asarray(grid, dtype=np.int8).copy()
        if arr.shape == self.hud_mask.shape:
            arr[self.hud_mask] = 0
        return arr


def compute_arena_hash(
    grid: np.ndarray,
    bezel_px: int = 4,
    hud_masker: Optional[DynamicHUDMasker] = None,
) -> int:
    """Computes a 44-bit Blake2b spatial hash masking out perimeter bezels and dynamic HUD bars."""
    arr = np.asarray(grid, dtype=np.int8)
    if hud_masker is not None and arr.shape == hud_masker.shape:
        masked = hud_masker.mask_grid(arr)
        digest = hashlib.blake2b(np.ascontiguousarray(masked).tobytes(), digest_size=8).digest()
        return int.from_bytes(digest, "little") & ((1 << 44) - 1)
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
    O(1) dead-action & fatal-action pruning + Empirical Transition Graph + IW(1)/IW(2) Novelty Table
    + Action-Efficiency Penalty Scorer (benchmaxxing RHAE protection).
    """

    INVERSE_MAP = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self, ring_size: int = 28):
        self.dead_set: Set[int] = set()
        self.death_set: Set[int] = set()
        self.ring_buffer: deque = deque(maxlen=ring_size)
        self.transitions: Dict[int, Dict[Tuple[int, int, int], int]] = {}
        self.state_grids: Dict[int, np.ndarray] = {}
        self.state_novelty: Dict[int, int] = {}
        self.state_visit_counts: Dict[int, int] = {}
        self.action_fail_counts: Dict[Tuple[int, int], int] = {}
        self.seen_atoms: Set[Tuple[int, int, int]] = set()
        self.seen_pairs: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()

    def reset_level(self, clear_deaths: bool = False) -> None:
        self.ring_buffer.clear()
        self.transitions.clear()
        self.state_grids.clear()
        self.state_novelty.clear()
        self.state_visit_counts.clear()
        self.action_fail_counts.clear()
        self.seen_atoms.clear()
        self.seen_pairs.clear()
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
        self.action_fail_counts[(pre_hash, int(action_id))] = 999

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
        self.state_visit_counts[post_hash] = self.state_visit_counts.get(post_hash, 0) + 1
        if delta_omega == 0 or pre_hash == post_hash:
            sig = pack_signature_64(pre_hash, action_id, coord_x, coord_y, level_idx)
            self.dead_set.add(sig)
            key = (pre_hash, int(action_id))
            self.action_fail_counts[key] = self.action_fail_counts.get(key, 0) + 1
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
        """Evaluates both IW(1) single-atom and IW(2) pairwise feature novelty."""
        arr = np.asarray(grid, dtype=np.int8)
        new_count = 0
        active_atoms = []
        for r in range(4, arr.shape[0] - 4, 4):
            for c in range(4, arr.shape[1] - 4, 4):
                val = int(arr[r, c])
                atom = (r, c, val)
                active_atoms.append(atom)
                if atom not in self.seen_atoms:
                    self.seen_atoms.add(atom)
                    new_count += 2
        if new_count == 0 and len(active_atoms) <= 64 and len(self.seen_pairs) < 8000:
            for i in range(0, len(active_atoms), 2):
                for j in range(i + 1, min(i + 6, len(active_atoms))):
                    pair = (active_atoms[i], active_atoms[j])
                    if pair not in self.seen_pairs:
                        self.seen_pairs.add(pair)
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

    def compute_action_penalty(
        self,
        arena_hash: int,
        action_id: int,
        last_action_id: Optional[int] = None,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
    ) -> float:
        """
        RHAE Move & Action Penalty Scorer:
        Heavily downvotes actions known to be dead, actions that lead to already-visited
        states, or immediate 2-step reversals without intermediate causal change.
        """
        if self.is_dead(arena_hash, action_id, coord_x, coord_y, level_idx):
            return 1e6
        penalty = 0.0
        act_key = (int(action_id), int(coord_x), int(coord_y))
        known_out = self.transitions.get(arena_hash, {})
        if act_key in known_out:
            dest_hash = known_out[act_key]
            visits = self.state_visit_counts.get(dest_hash, 0)
            penalty += visits * 4.5
            if len(self.ring_buffer) >= 2 and dest_hash == self.ring_buffer[-2]:
                penalty += 25.0
        if last_action_id is not None and self.INVERSE_MAP.get(last_action_id) == action_id:
            penalty += 8.0
        return penalty

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
        last_action_id: Optional[int] = None,
    ) -> Optional[List[int]]:
        if not candidate_actions:
            return None
        tried_here = {k[0] for k in self.transitions.get(current_hash, {}).keys()}
        untried_here = [
            a for a in candidate_actions
            if not self.is_dead(current_hash, a, 0, 0, level_idx) and a not in tried_here
        ]
        if untried_here:
            untried_here.sort(
                key=lambda a: self.compute_action_penalty(current_hash, a, last_action_id, 0, 0, level_idx)
            )
            return [untried_here[0]]

        queue = deque([(current_hash, [])])
        visited = {current_hash}
        best_path: Optional[List[int]] = None
        best_score = -1e9

        while queue and len(visited) < 450:
            sh, path = queue.popleft()
            if len(path) >= 24:
                continue
            tried = {k[0] for k in self.transitions.get(sh, {}).keys()}
            for a in candidate_actions:
                if not self.is_dead(sh, a, 0, 0, level_idx) and a not in tried:
                    cand_path = path + [a]
                    nov = self.state_novelty.get(sh, 0)
                    visits = self.state_visit_counts.get(sh, 1)
                    score = (nov * 3.0) - (len(cand_path) * 1.2) - (visits * 0.8)
                    if score > best_score:
                        best_score = score
                        best_path = cand_path
                    if len(cand_path) <= 2:
                        return cand_path
            for (act_id, cx, cy), nxt_h in self.transitions.get(sh, {}).items():
                if act_id in candidate_actions and nxt_h not in visited:
                    if not self.is_dead(sh, act_id, cx, cy, level_idx):
                        visited.add(nxt_h)
                        queue.append((nxt_h, path + [act_id]))
        return best_path
