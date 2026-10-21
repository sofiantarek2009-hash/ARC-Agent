from collections import deque
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class MacroActionQueue:
    def __init__(self):
        self.queue: deque = deque()
        self.active_subagent: str = "NONE"

    def push_plan(self, actions: List[Any], subagent_tag: str = "PLANNER") -> None:
        self.queue.clear()
        self.active_subagent = subagent_tag
        for a in actions:
            if isinstance(a, tuple) and len(a) == 2:
                self.queue.append((int(a[0]), dict(a[1])))
            else:
                self.queue.append((int(a), {}))

    def pop_next(self, last_delta_omega: Optional[int] = None) -> Optional[Tuple[int, Dict[str, Any]]]:
        if last_delta_omega == 0:
            self.queue.clear()
            self.active_subagent = "NONE"
            return None
        if self.queue:
            return self.queue.popleft()
        self.active_subagent = "NONE"
        return None

    def clear(self) -> None:
        self.queue.clear()
        self.active_subagent = "NONE"


def astar_grid(
    start: Tuple[int, int],
    target: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
    portals: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
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
        d_direct = (abs(pos[0] - target[0]) + abs(pos[1] - target[1])) / float(s)
        if portals:
            for p_in, p_out in portals.items():
                d_portal = (
                    abs(pos[0] - p_in[0]) + abs(pos[1] - p_in[1])
                    + abs(p_out[0] - target[0]) + abs(p_out[1] - target[1])
                ) / float(s)
                if d_portal + 1.0 < d_direct:
                    d_direct = d_portal + 1.0
        return d_direct

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
            if portals:
                for p_in, p_out in portals.items():
                    if abs(nxt[0] - p_in[0]) < s and abs(nxt[1] - p_in[1]) < s:
                        nxt = p_out
                        break

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
    max_states: int = 3500,
) -> Optional[List[int]]:
    if not boxes or not goals or not action_vectors:
        return None
    h, w = grid_shape
    strides = [max(abs(dr), abs(dc)) for dr, dc in action_vectors.values() if (dr, dc) != (0, 0)]
    s = max(1, min(strides)) if strides else 1

    def is_goal_satisfied(b: Tuple[int, int]) -> bool:
        return any(abs(b[0] - g[0]) < s and abs(b[1] - g[1]) < s for g in goals)

    def is_corner_deadlock(b: Tuple[int, int]) -> bool:
        if is_goal_satisfied(b):
            return False
        up = (b[0] - s, b[1]) in walls
        down = (b[0] + s, b[1]) in walls
        left = (b[0], b[1] - s) in walls
        right = (b[0], b[1] + s) in walls
        return (up or down) and (left or right)

    def box_heuristic(curr_boxes: frozenset) -> float:
        total = 0.0
        for b in curr_boxes:
            d_min = min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in goals)
            total += d_min / float(s)
        return total

    init_boxes = frozenset(boxes)
    open_heap = [(box_heuristic(init_boxes), 0, avatar, init_boxes, [])]
    visited = {(avatar, init_boxes): 0}

    while open_heap and len(visited) < max_states:
        _, g_cost, av, curr_boxes, path = heapq.heappop(open_heap)
        if all(is_goal_satisfied(b) for b in curr_boxes):
            return path
        if len(path) >= 42:
            continue

        for act_id, (dr, dc) in action_vectors.items():
            if dr == 0 and dc == 0:
                continue
            nr, nc = av[0] + dr, av[1] + dc
            nxt_av = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w) or nxt_av in walls:
                continue
            hit_box = None
            for b in curr_boxes:
                if abs(nxt_av[0] - b[0]) < s and abs(nxt_av[1] - b[1]) < s:
                    hit_box = b
                    break
            if hit_box is not None:
                br, bc = hit_box[0] + dr, hit_box[1] + dc
                nxt_box = (br, bc)
                if not (0 <= br < h and 0 <= bc < w) or nxt_box in walls:
                    continue
                if any(abs(nxt_box[0] - ob[0]) < s and abs(nxt_box[1] - ob[1]) < s for ob in curr_boxes if ob != hit_box):
                    continue
                if is_corner_deadlock(nxt_box):
                    continue
                new_boxes = frozenset((curr_boxes - {hit_box}) | {nxt_box})
                state = (nxt_av, new_boxes)
            else:
                new_boxes = curr_boxes
                state = (nxt_av, new_boxes)

            new_g = g_cost + 1
            if state not in visited or new_g < visited[state]:
                visited[state] = new_g
                f_score = new_g + box_heuristic(new_boxes)
                heapq.heappush(open_heap, (f_score, new_g, state[0], state[1], path + [int(act_id)]))
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
