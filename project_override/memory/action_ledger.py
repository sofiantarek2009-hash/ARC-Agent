from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from project_override.perception.observation_processor import DynamicHUDMasker

@dataclass
class TransitionRecord:
    step_idx: int
    level_idx: int
    pre_hash: int
    action_id: int
    x: int
    y: int
    post_hash: int
    delta_omega: int
    diff_summary: str
    pre_grid: Optional[np.ndarray] = None
    post_grid: Optional[np.ndarray] = None

def compute_arena_hash(
    grid: np.ndarray,
    hud_masker: Optional[DynamicHUDMasker] = None,
) -> int:
    """Computes a 44-bit BLAKE2b state hash using explicit little-endian byte order."""
    arr = np.asarray(grid, dtype=np.int8)
    if hud_masker is not None:
        arr = hud_masker.mask_grid(arr)
    digest = hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 44) - 1)

def pack_signature_64(
    arena_hash: int,
    action_id: int,
    coord_x: int = 0,
    coord_y: int = 0,
    level_idx: int = 0,
) -> int:
    h44 = arena_hash & ((1 << 44) - 1)
    lvl4 = level_idx & 0xF
    cy6 = coord_y & 0x3F
    cx6 = coord_x & 0x3F
    act4 = action_id & 0xF
    return (h44 << 20) | (lvl4 << 16) | (cy6 << 10) | (cx6 << 4) | act4

class ActionLedger:
    """
    Tracks state-action-result transitions, dead signatures, object click no-ops,
    PRO-LONG programmatic event history (arXiv:2607.20064), and Ledger Inform/Govern
    online execution state mediation (Wang et al., arXiv:2608.00808).
    """
    INVERSE_MAP = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self, ring_size: int = 40, enable_dead_signatures: bool = True):
        self.enable_dead_signatures = enable_dead_signatures
        self.dead_set: Set[int] = set()
        self.death_set: Set[int] = set()
        self.dead_object_clicks: Set[Tuple[int, Tuple[int, int, int, int]]] = set()
        self.ring_buffer: deque = deque(maxlen=ring_size)
        self.transitions: Dict[int, Dict[Tuple[int, int, int], int]] = {}
        self.state_grids: Dict[int, np.ndarray] = {}
        self.state_novelty: Dict[int, int] = {}
        self.state_visit_counts: Dict[int, int] = {}
        self.action_fail_counts: Dict[Tuple[int, int], int] = {}
        self.seen_atoms: Set[Tuple[int, int, int]] = set()
        self.seen_pairs: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
        self.history: List[TransitionRecord] = []

    def reset_level(self, clear_deaths: bool = False) -> None:
        self.ring_buffer.clear()
        self.transitions.clear()
        self.state_grids.clear()
        self.state_novelty.clear()
        self.state_visit_counts.clear()
        self.action_fail_counts.clear()
        self.seen_atoms.clear()
        self.seen_pairs.clear()
        self.dead_object_clicks.clear()
        self.history.clear()
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
        pre_grid: Optional[np.ndarray] = None,
        post_grid: Optional[np.ndarray] = None,
        diff_summary: str = "",
        clicked_obj_sig: Optional[Tuple[int, int, int, int]] = None,
        step_idx: int = 0,
    ) -> None:
        self.state_visit_counts[post_hash] = self.state_visit_counts.get(post_hash, 0) + 1
        if pre_grid is not None and pre_hash not in self.state_grids:
            self.state_grids[pre_hash] = pre_grid.copy()
        if delta_omega == 0 or pre_hash == post_hash:
            if self.enable_dead_signatures:
                sig = pack_signature_64(pre_hash, action_id, coord_x, coord_y, level_idx)
                self.dead_set.add(sig)
                if int(action_id) == 6 and clicked_obj_sig is not None:
                    self.dead_object_clicks.add((pre_hash, clicked_obj_sig))
            key = (pre_hash, int(action_id))
            self.action_fail_counts[key] = self.action_fail_counts.get(key, 0) + 1
        else:
            act_key = (int(action_id), int(coord_x), int(coord_y))
            if pre_hash not in self.transitions:
                self.transitions[pre_hash] = {}
            self.transitions[pre_hash][act_key] = post_hash
            if post_grid is not None and post_hash not in self.state_grids:
                self.state_grids[post_hash] = post_grid.copy()
                self.state_novelty[post_hash] = self.update_novelty(post_grid)
        self.ring_buffer.append(post_hash)
        self.history.append(
            TransitionRecord(
                step_idx=step_idx,
                level_idx=level_idx,
                pre_hash=pre_hash,
                action_id=int(action_id),
                x=int(coord_x),
                y=int(coord_y),
                post_hash=post_hash,
                delta_omega=int(delta_omega),
                diff_summary=diff_summary,
                pre_grid=pre_grid.copy() if pre_grid is not None else None,
                post_grid=post_grid.copy() if post_grid is not None else None,
            )
        )
        if len(self.history) > 250:
            self.history = self.history[-250:]

    def query_history(
        self, action_id: Optional[int] = None, min_delta: int = 0
    ) -> List[Dict[str, Any]]:
        """PRO-LONG programmatic search helper over recorded transitions."""
        out: List[Dict[str, Any]] = []
        for rec in self.history:
            if action_id is not None and rec.action_id != int(action_id):
                continue
            if rec.delta_omega < min_delta:
                continue
            out.append(
                {
                    "step": rec.step_idx,
                    "level": rec.level_idx,
                    "action": rec.action_id,
                    "x": rec.x,
                    "y": rec.y,
                    "delta": rec.delta_omega,
                    "diff": rec.diff_summary,
                }
            )
        return out

    def render_execution_state(self, current_hash: int, avail_actions: List[int], level_idx: int) -> str:
        """Ledger Inform Path (arXiv:2608.00808): distills current state visits, tried actions, and dead actions."""
        visits = self.state_visit_counts.get(current_hash, 1)
        tried_map = self.transitions.get(current_hash, {})
        tried_str = ", ".join(f"A{k[0]}({k[1]},{k[2]})->h{v & 0xFFFF:04x}" for k, v in list(tried_map.items())[:6])
        dead_here = [
            a for a in avail_actions if a in (1, 2, 3, 4, 5, 7) and self.is_dead(current_hash, a, 0, 0, level_idx)
        ]
        return f"STATE_VISITS={visits} | TRIED_FROM_HERE=[{tried_str}] | DEAD_HERE={dead_here}"

    def govern_action(
        self,
        current_hash: int,
        action_id: int,
        x: int = 0,
        y: int = 0,
        level_idx: int = 0,
        obj_sig: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Ledger Govern Path (arXiv:2608.00808): mediates proposed commands before environment execution."""
        return not self.is_dead(current_hash, action_id, x, y, level_idx, obj_sig=obj_sig)

    def update_novelty(self, grid: np.ndarray) -> int:
        arr = np.asarray(grid, dtype=np.int8)
        new_count = 0
        active_atoms: List[Tuple[int, int, int]] = []
        for r in range(0, arr.shape[0], 4):
            for c in range(0, arr.shape[1], 4):
                atom = (r, c, int(arr[r, c]))
                active_atoms.append(atom)
                if atom not in self.seen_atoms:
                    self.seen_atoms.add(atom)
                    new_count += 2
        if new_count == 0 and len(active_atoms) <= 256 and len(self.seen_pairs) < 10000:
            for i in range(0, len(active_atoms), 4):
                for j in range(i + 1, min(i + 5, len(active_atoms))):
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
        obj_sig: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        if not self.enable_dead_signatures:
            return False
        sig = pack_signature_64(arena_hash, action_id, coord_x, coord_y, level_idx)
        if (sig in self.dead_set) or (sig in self.death_set):
            return True
        if int(action_id) == 6 and obj_sig is not None:
            if (arena_hash, obj_sig) in self.dead_object_clicks:
                return True
        return False

    def is_in_loop(self) -> bool:
        if len(self.ring_buffer) < 4:
            return False
        recent = list(self.ring_buffer)[-6:]
        return len(set(recent)) <= 2

    def get_untried_discrete_actions(
        self, arena_hash: int, available_actions: List[int], level_idx: int = 0
    ) -> List[int]:
        tried = {k[0] for k in self.transitions.get(arena_hash, {}).keys()}
        return [
            int(a)
            for a in available_actions
            if int(a) in (1, 2, 3, 4, 5)
            and int(a) not in tried
            and not self.is_dead(arena_hash, int(a), 0, 0, level_idx)
        ]

    def bfs_to_unexplored_action(
        self,
        start_hash: int,
        available_discrete: List[int],
        level_idx: int = 0,
        last_action_id: Optional[int] = None,
    ) -> Optional[List[int]]:
        if not available_discrete:
            return None
        untried_here = self.get_untried_discrete_actions(
            start_hash, available_discrete, level_idx
        )
        if untried_here:
            if (
                last_action_id is not None
                and len(untried_here) > 1
                and self.INVERSE_MAP.get(last_action_id) in untried_here
            ):
                non_inv = [
                    a for a in untried_here if a != self.INVERSE_MAP.get(last_action_id)
                ]
                return [non_inv[0]]
            return [untried_here[0]]
        queue: deque = deque([(start_hash, [])])
        visited: Set[int] = {start_hash}
        while queue and len(visited) < 500:
            curr_h, path = queue.popleft()
            if len(path) >= 18:
                continue
            for (act_id, cx, cy), next_h in self.transitions.get(curr_h, {}).items():
                if cx != 0 or cy != 0 or act_id not in available_discrete:
                    continue
                if self.is_dead(curr_h, act_id, 0, 0, level_idx):
                    continue
                new_path = path + [act_id]
                if self.get_untried_discrete_actions(next_h, available_discrete, level_idx):
                    nxt_untried = self.get_untried_discrete_actions(
                        next_h, available_discrete, level_idx
                    )
                    return new_path + [nxt_untried[0]]
                if next_h not in visited:
                    visited.add(next_h)
                    queue.append((next_h, new_path))
        return None

    def compute_action_penalty(
        self,
        arena_hash: int,
        action_id: int,
        last_action_id: Optional[int] = None,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
    ) -> float:
        if self.is_dead(arena_hash, action_id, coord_x, coord_y, level_idx):
            return 1e6
        penalty = float(self.action_fail_counts.get((arena_hash, int(action_id)), 0) * 20.0)
        act_key = (int(action_id), int(coord_x), int(coord_y))
        known_out = self.transitions.get(arena_hash, {})
        if act_key in known_out:
            dest_hash = known_out[act_key]
            visits = self.state_visit_counts.get(dest_hash, 0)
            novelty = self.state_novelty.get(dest_hash, 0)
            penalty += visits * 5.0 - min(12.0, novelty * 1.5)
        if last_action_id is not None and self.INVERSE_MAP.get(last_action_id) == int(action_id):
            penalty += 3.5
        return penalty
