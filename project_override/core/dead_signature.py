import hashlib
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
        self.death_set: Set[int] = set()
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
