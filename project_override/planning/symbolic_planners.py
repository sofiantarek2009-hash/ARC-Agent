from collections import deque
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
