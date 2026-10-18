"""Universal Coupled-Delay & Hold Kinematics Engines for ARC-AGI-3.
Zero-overfitting implementations for:
1. ActionPipelineTracker: Online K-latency induction & FIFO queue tracking
2. solve_delayed_pipeline_astar: Augmented-state A* with pipeline lookahead
3. HoldMechanicInducer: Frame diff HUD/charge discrimination & dwell tracking
4. solve_inertial_slide_to_hold: Sliding raycast + obstacle-bump dwell planning
"""

from __future__ import annotations

import collections
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class ActionPipelineTracker:
    """Induces pipeline latency K in 3-5 steps and maintains FIFO queue state."""

    def __init__(self, max_k: int = 5) -> None:
        self.max_k = max_k
        self.action_history: List[int] = []
        self.pos_history: List[Tuple[int, int]] = []
        self.disp_history: List[Tuple[int, int]] = []
        self.deltas: Dict[int, Tuple[int, int]] = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
            5: (0, 0),
        }
        self.inferred_k: int = 0
        self.confirmed: bool = False

    def set_deltas(self, deltas: Dict[int, Tuple[int, int]]) -> None:
        """Sets the expected displacement vectors per action for latency induction."""
        if deltas:
            self.deltas = dict(deltas)

    def reset_for_level(self) -> None:
        self.action_history.clear()
        self.pos_history.clear()
        self.disp_history.clear()
        # Keep inferred_k across levels as strong prior, but allow re-confirmation if needed
        self.confirmed = (self.inferred_k > 0)

    def record_transition(
        self,
        action: int,
        p_before: Tuple[int, int],
        p_after: Tuple[int, int],
    ) -> int:
        dp = (p_after[0] - p_before[0], p_after[1] - p_before[1])
        self.action_history.append(action)
        self.pos_history.append(p_before)
        self.disp_history.append(dp)

        if len(self.action_history) >= 2 and dp != (0, 0):
            prev_act = self.action_history[-2]
            curr_act = self.action_history[-1]
            if prev_act != curr_act:
                prev_dp = self.deltas.get(prev_act, (0, 0))
                curr_dp = self.deltas.get(curr_act, (0, 0))
                if dp == prev_dp and dp != curr_dp:
                    self.inferred_k = 1
                    self.confirmed = True
                    return 1

        if not self.confirmed:
            best_k, matching_moves = self._induce_k()
            has_action_change = any(self.action_history[i] != self.action_history[i-1] for i in range(1, len(self.action_history)))
            if best_k > 0:
                self.inferred_k = best_k
                if matching_moves >= 2:
                    self.confirmed = True
            elif best_k == 0 and has_action_change and matching_moves >= 2:
                self.inferred_k = 0
                self.confirmed = True

        return self.inferred_k

    def _induce_k(self) -> Tuple[int, int]:
        n = len(self.disp_history)
        if n < 1:
            return 0, 0
        best_k = 0
        best_score = -1e9
        best_matches = 0

        for k in range(min(self.max_k + 1, n)):
            score = 0.0
            matches = 0
            for t in range(n):
                dp = self.disp_history[t]
                if t < k:
                    if dp == (0, 0):
                        score += 3.0
                    else:
                        score -= 6.0
                else:
                    act = self.action_history[t - k]
                    expected_dp = self.deltas.get(act, (0, 0))
                    if dp != (0, 0):
                        if dp == expected_dp:
                            score += 5.0
                            matches += 1
                        else:
                            score -= 12.0
            if score > best_score:
                best_score = score
                best_k = k
                best_matches = matches
        return best_k, best_matches

    def get_current_pipeline(self) -> Tuple[int, ...]:
        q = list(self.action_history)
        while len(q) < self.inferred_k:
            q.insert(0, 0)
        return tuple(q[-self.inferred_k:]) if self.inferred_k > 0 else ()


def solve_delayed_pipeline_astar(
    start_pos: Tuple[int, int],
    initial_queue: Tuple[int, ...],
    goals: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    deltas: Optional[Dict[int, Tuple[int, int]]] = None,
    legal_actions: Optional[List[int]] = None,
    max_expansions: int = 40000,
) -> Optional[List[int]]:
    """A* search over state tuple (avatar_pos, queue_tuple) with pipeline lookahead."""
    if not goals:
        return None
    if start_pos in goals:
        return []

    act_deltas = deltas or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    k_lat = len(initial_queue)

    # Precompute obstacle-aware Dijkstra distances from all cells to nearest goal
    dist_map: Dict[Tuple[int, int], int] = {g: 0 for g in goals}
    queue_bfs = collections.deque(list(goals))
    while queue_bfs:
        cr, cc = queue_bfs.popleft()
        d = dist_map[(cr, cc)]
        for dr, dc in act_deltas.values():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in walls and (nr, nc) not in dist_map:
                dist_map[(nr, nc)] = d + 1
                queue_bfs.append((nr, nc))

    def step_pos(pos: Tuple[int, int], act: int) -> Tuple[int, int]:
        dr, dc = act_deltas.get(act, (0, 0))
        if dr == 0 and dc == 0:
            return pos
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in walls:
            return (nr, nc)
        return pos

    def heuristic(pos: Tuple[int, int], q: Tuple[int, ...]) -> int:
        p = pos
        for idx, act in enumerate(q):
            p = step_pos(p, act)
            if p in goals:
                return idx + 1
        d = dist_map.get(p, 9999)
        return len(q) + d

    start_state = (start_pos[0], start_pos[1], initial_queue)
    h0 = heuristic(start_pos, initial_queue)
    open_set = [(h0, 0, start_pos[0], start_pos[1], initial_queue, [])]
    visited: Dict[Tuple[int, int, Tuple[int, ...]], int] = {start_state: 0}
    expansions = 0

    move_actions = [a for a, d in act_deltas.items() if d != (0, 0)]
    if not move_actions:
        move_actions = [1, 2, 3, 4]

    while open_set and expansions < max_expansions:
        _, g, r, c, q_curr, path = heapq.heappop(open_set)
        expansions += 1

        if (r, c) in goals:
            return path

        if g > visited.get((r, c, q_curr), 1000000):
            continue

        exec_act = q_curr[0] if k_lat > 0 else 0
        nr, nc = step_pos((r, c), exec_act) if k_lat > 0 else (r, c)

        if (nr, nc) in goals:
            flush_act = path[-1] if path else (move_actions[0] if move_actions else 1)
            return path + [flush_act]  # Flush trigger action

        for act in move_actions:
            nxt_q = (q_curr[1:] + (act,)) if k_lat > 0 else ()
            nxt_state = (nr, nc, nxt_q)
            ng = g + 1
            if ng < visited.get(nxt_state, 1000000):
                visited[nxt_state] = ng
                h_val = heuristic((nr, nc), nxt_q) if k_lat > 0 else dist_map.get((nr, nc), 9999)
                heapq.heappush(open_set, (ng + h_val, ng, nr, nc, nxt_q, path + [act]))

    return None


class HoldMechanicInducer:
    """Discriminates 1-tick goal vs T-tick hold plates via stationary frame diffs."""

    def __init__(self, max_probe_ticks: int = 8) -> None:
        self.max_probe_ticks = max_probe_ticks
        self.dwell_ticks: int = 0
        self.active_plate: Optional[Tuple[int, int]] = None
        self.hold_confirmed: bool = False
        self.hold_rejected: bool = False
        self.required_hold_ticks: int = 1
        self.hud_accumulator_detected: bool = False
        self.charging_plate_detected: bool = False

    def reset_for_game(self) -> None:
        self.dwell_ticks = 0
        self.active_plate = None
        self.hold_confirmed = False
        self.hold_rejected = False
        self.required_hold_ticks = 1
        self.hud_accumulator_detected = False
        self.charging_plate_detected = False

    def reset_for_level(self) -> None:
        self.dwell_ticks = 0
        self.active_plate = None
        self.hud_accumulator_detected = False
        self.charging_plate_detected = False
        if not self.hold_confirmed:
            self.hold_rejected = False

    def on_step(
        self,
        player_pos: Tuple[int, int],
        candidate_goals: Set[Tuple[int, int]],
        prev_frame: Optional[np.ndarray],
        curr_frame: np.ndarray,
        level_won: bool,
    ) -> Dict[str, Any]:
        if self.hold_rejected:
            return {'action_needed': False, 'probe_hold': False}

        if player_pos in candidate_goals:
            if self.active_plate != player_pos:
                self.active_plate = player_pos
                self.dwell_ticks = 1
            else:
                self.dwell_ticks += 1

            if level_won:
                if self.dwell_ticks > 1:
                    self.hold_confirmed = True
                    self.required_hold_ticks = self.dwell_ticks
                return {'action_needed': False, 'status': 'WON', 'hold_ticks': self.required_hold_ticks}

            # Inspect frame delta for HUD progress or plate charging
            if prev_frame is not None:
                diff_mask = (prev_frame != curr_frame)
                r, c = player_pos
                diff_mask_ex = diff_mask.copy()
                diff_mask_ex[r, c] = False
                num_changed = int(np.sum(diff_mask_ex))
                if num_changed > 0:
                    changed_coords = np.argwhere(diff_mask_ex)
                    is_hud = any(cr < 6 or cr >= 56 or cc < 6 or cc >= 58 for cr, cc in changed_coords)
                    if is_hud:
                        self.hud_accumulator_detected = True
                    elif self.dwell_ticks > 1:
                        self.charging_plate_detected = True

            # If hold is confirmed or feedback was observed, continue holding
            if self.hold_confirmed or self.hud_accumulator_detected or self.charging_plate_detected:
                if self.dwell_ticks < 25:
                    return {
                        'action_needed': True,
                        'probe_hold': True,
                        'dwell_ticks': self.dwell_ticks,
                        'hud_feedback': self.hud_accumulator_detected,
                        'charge_feedback': self.charging_plate_detected,
                    }
                else:
                    return {'action_needed': False, 'probe_hold': False, 'status': 'TIMEOUT'}

            # Initial probe up to max_probe_ticks if no immediate feedback
            if self.dwell_ticks <= self.max_probe_ticks:
                return {
                    'action_needed': True,
                    'probe_hold': True,
                    'dwell_ticks': self.dwell_ticks,
                    'hud_feedback': self.hud_accumulator_detected,
                    'charge_feedback': self.charging_plate_detected,
                }
            else:
                # No feedback after max_probe_ticks -> not a hold mechanic
                self.hold_rejected = True
                return {'action_needed': False, 'probe_hold': False, 'status': 'REJECTED'}
        else:
            self.active_plate = None
            self.dwell_ticks = 0
            return {'action_needed': False, 'probe_hold': False}


def synthesize_holding_action(
    player_pos: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    legal_actions: List[int],
    deltas: Optional[Dict[int, Tuple[int, int]]] = None,
) -> int:
    """Synthesizes an action that keeps the avatar stationary on its current tile."""
    if 5 in legal_actions:
        return 5  # Neutral Interact / Wait

    act_deltas = deltas or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    r, c = player_pos
    for act in [1, 2, 3, 4]:
        if act in legal_actions and act in act_deltas:
            dr, dc = act_deltas[act]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < gh and 0 <= nc < gw) or (nr, nc) in walls:
                return act  # Stationary bump into wall/boundary

    return legal_actions[0]


def solve_inertial_slide_to_hold(
    start_pos: Tuple[int, int],
    hold_plate: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    hold_ticks: int = 1,
    deltas: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Optional[List[int]]:
    """Inertial sliding raycast BFS requiring terminal stop on plate + collision bumps."""
    action_deltas = deltas or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def raycast(pr: int, pc: int, dr: int, dc: int) -> Tuple[int, int]:
        cr, cc = pr, pc
        while True:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < gh and 0 <= nc < gw) or (nr, nc) in walls:
                break
            cr, cc = nr, nc
        return (cr, cc)

    q = collections.deque([(start_pos, [], 0)])
    visited = {start_pos}

    while q:
        pos, path, last_act = q.popleft()
        if pos == hold_plate:
            hold_act = last_act
            if hold_act == 0 or raycast(pos[0], pos[1], action_deltas[hold_act][0], action_deltas[hold_act][1]) != pos:
                for act, (dr, dc) in action_deltas.items():
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if not (0 <= nr < gh and 0 <= nc < gw) or (nr, nc) in walls:
                        hold_act = act
                        break

            full_plan = list(path)
            for _ in range(max(0, hold_ticks - 1)):
                full_plan.append(hold_act)
            return full_plan

        for act, (dr, dc) in action_deltas.items():
            landing = raycast(pos[0], pos[1], dr, dc)
            if landing != pos and landing not in visited:
                visited.add(landing)
                q.append((landing, path + [act], act))

    return None
