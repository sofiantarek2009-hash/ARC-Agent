"""Fast in-sandbox simulation, pathfinding, and retrodiction helpers.

This module is intentionally self-contained (pure standard library) so its source
can be spliced verbatim into the Python-tool sandbox bootstrap.
"""

from collections import deque

_DELTAS = {
    1: (-1, 0),  # UP
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
}

def find_shortest_path(start, goals, passable_cells, grid_shape=(64, 64)):
    """General BFS shortest pathfinder over passable grid cells.
    
    Args:
        start: tuple (r, c)
        goals: set/list of tuples {(r, c), ...} or single tuple (r, c)
        passable_cells: set/iterable of tuples {(r, c), ...}
        grid_shape: (height, width)
        
    Returns:
        list of integer actions [1..4]
    """
    if isinstance(goals, tuple):
        goals = {goals}
    else:
        goals = set(goals)
        
    if start in goals:
        return []
    
    passable = set(passable_cells)
    queue = deque([(start, [])])
    visited = {start}
    h, w = grid_shape
    
    while queue:
        curr, path = queue.popleft()
        if curr in goals:
            return path
        
        r, c = curr
        for act, (dr, dc) in _DELTAS.items():
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w:
                if (nxt in passable or nxt in goals) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [act]))
    return []

def verify_retrodiction(simulator_func, transition_history):
    """Verifies that a candidate transition simulator retrodicts past transitions.
    
    Args:
        simulator_func: callable (state, action) -> next_state
        transition_history: list of dicts [{'state': s_t, 'action': a_t, 'next_state': s_{t+1}}]
        
    Returns:
        tuple (bool, dict): (is_valid, failure_detail)
    """
    for idx, step in enumerate(transition_history):
        s_t = step.get('state')
        a_t = step.get('action')
        expected = step.get('next_state')
        try:
            predicted = simulator_func(s_t, a_t)
            if predicted != expected:
                return False, {
                    'step_index': idx,
                    'state': s_t,
                    'action': a_t,
                    'predicted': predicted,
                    'expected': expected,
                }
        except Exception as exc:
            return False, {
                'step_index': idx,
                'state': s_t,
                'action': a_t,
                'error': str(exc),
            }
    return True, {}

def solve_pushable_crates(start_player, boxes, targets, walls, grid_shape=(64, 64)):
    """General BFS solver for pushable crate / Sokoban mechanics.
    
    Args:
        start_player: tuple (r, c)
        boxes: frozenset/set of tuples {(r, c), ...}
        targets: frozenset/set of tuples {(r, c), ...}
        walls: set of tuples {(r, c), ...}
        grid_shape: (height, width)
        
    Returns:
        list of integer actions [1..4]
    """
    targets = frozenset(targets)
    boxes = frozenset(boxes)
    walls = set(walls)
    h, w = grid_shape
    
    if boxes == targets:
        return []
    
    initial_state = (start_player, boxes)
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    
    # Simple concavity deadlock check
    def is_deadlocked(b_set):
        for br, bc in b_set:
            if (br, bc) not in targets:
                # Check corner: (N/S wall) and (E/W wall)
                north_wall = (br - 1, bc) in walls or br == 0
                south_wall = (br + 1, bc) in walls or br == h - 1
                west_wall = (br, bc - 1) in walls or bc == 0
                east_wall = (br, bc + 1) in walls or bc == w - 1
                if (north_wall or south_wall) and (west_wall or east_wall):
                    return True
        return False

    max_evals = 15000
    evals = 0
    
    while queue and evals < max_evals:
        evals += 1
        (p_pos, b_set), path = queue.popleft()
        if b_set == targets:
            return path
            
        pr, pc = p_pos
        for act, (dr, dc) in _DELTAS.items():
            np_pos = (pr + dr, pc + dc)
            npr, npc = np_pos
            if not (0 <= npr < h and 0 <= npc < w) or np_pos in walls:
                continue
                
            if np_pos in b_set:
                # Pushing box
                nb_pos = (npr + dr, npc + dc)
                nbr, nbc = nb_pos
                if not (0 <= nbr < h and 0 <= nbc < w) or nb_pos in walls or nb_pos in b_set:
                    continue
                next_boxes = (b_set - {np_pos}) | {nb_pos}
                if is_deadlocked(next_boxes):
                    continue
                next_state = (np_pos, frozenset(next_boxes))
            else:
                next_state = (np_pos, b_set)
                
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [act]))
    return []

def solve_bipartite_assignment(sources, targets):
    """Hungarian bipartite minimum Manhattan distance matching.
    
    Pairs each source with the optimal target to minimize total distance.
    Returns:
        dict {source_tuple: target_tuple}
    """
    sources = list(sources)
    targets = list(targets)
    if len(sources) != len(targets):
        # Fallback to greedy nearest neighbor
        available = set(targets)
        pairing = {}
        for s in sources:
            if not available:
                break
            best = min(available, key=lambda t: abs(s[0] - t[0]) + abs(s[1] - t[1]))
            pairing[s] = best
            available.remove(best)
        return pairing
        
    n = len(sources)
    cost_matrix = [
        [abs(sources[i][0] - targets[j][0]) + abs(sources[i][1] - targets[j][1]) for j in range(n)]
        for i in range(n)
    ]
    
    # Hungarian algorithm (Kuhn-Munkres) for O(n^3) matching
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
                
    result = {}
    for j in range(1, n + 1):
        if p[j] > 0:
            result[sources[p[j] - 1]] = targets[j - 1]
    return result


class PurePythonSOCSE:
    """Zero-dependency Spatiotemporal Object-Centric State Estimator for Sandbox."""

    @staticmethod
    def infer_background_color(grid):
        """Infers background color from perimeter pixels."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        if h == 0 or w == 0:
            return 0
        counts = {}
        for c in range(w):
            for r in (0, h - 1):
                val = grid[r][c]
                counts[val] = counts.get(val, 0) + 1
        for r in range(h):
            for c in (0, w - 1):
                val = grid[r][c]
                counts[val] = counts.get(val, 0) + 1
        return max(counts, key=counts.get) if counts else 0

    @staticmethod
    def get_connected_components(grid, bg_color=None):
        """Extracts 4-connected components excluding background."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        if h == 0 or w == 0:
            return []
        if bg_color is None:
            bg_color = PurePythonSOCSE.infer_background_color(grid)

        visited = [[False] * w for _ in range(h)]
        components = []

        for r in range(h):
            for c in range(w):
                if visited[r][c] or grid[r][c] == bg_color:
                    continue
                color = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                cells = []

                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))

                rmin = min(r for r, _ in cells)
                rmax = max(r for r, _ in cells)
                cmin = min(c for _, c in cells)
                cmax = max(c for _, c in cells)
                cr = sum(r for r, _ in cells) / len(cells)
                cc = sum(c for _, c in cells) / len(cells)

                # Canonical translation-invariant bitmask relative to (rmin, cmin)
                norm_cells = tuple(sorted((r - rmin, c - cmin) for r, c in cells))

                components.append({
                    "color": color,
                    "cells": cells,
                    "size": len(cells),
                    "bbox": (rmin, rmax, cmin, cmax),
                    "centroid": (cr, cc),
                    "canonical_shape": hash(norm_cells),
                })
        return components

    @staticmethod
    def detect_monotonic_counters(history_grids):
        """Detects 1D linear strips whose non-bg count changes monotonically."""
        if len(history_grids) < 3:
            return set()
        h = len(history_grids[0])
        w = len(history_grids[0][0]) if h > 0 else 0
        bg = PurePythonSOCSE.infer_background_color(history_grids[0])
        strips = set()

        # Check rows
        for r in range(h):
            counts = [sum(1 for c in range(w) if g[r][c] != bg) for g in history_grids]
            if len(set(counts)) >= 3:
                diffs = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
                if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
                    strips.add(('row', r))
                elif all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
                    strips.add(('row', r))

        # Check cols
        for c in range(w):
            counts = [sum(1 for r in range(h) if g[r][c] != bg) for g in history_grids]
            if len(set(counts)) >= 3:
                diffs = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
                if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
                    strips.add(('col', c))
                elif all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
                    strips.add(('col', c))
        return strips


class PurePythonWorkbench:
    """In-Sandbox Complete-History Retrodiction Workbench."""

    def __init__(self, history=None):
        self.transitions = []
        if history:
            for i in range(1, len(history)):
                b = history[i - 1].frame._grid if hasattr(history[i - 1], 'frame') and history[i - 1].frame else None
                a = history[i].frame._grid if hasattr(history[i], 'frame') and history[i].frame else None
                act = getattr(history[i], 'action', "")
                res = getattr(history[i], 'result', {})
                if b is not None and a is not None:
                    self.transitions.append({
                        "turn": i,
                        "before_grid": b,
                        "action": str(act),
                        "after_grid": a,
                        "gameplay_changed": bool(res.get("gameplay_changed", True)),
                    })

    def retrodict_hypothesis(self, transition_fn):
        """Tests candidate transition function against all recorded history transitions.
        
        Args:
            transition_fn: callable (grid, action_str) -> predicted_grid
            
        Returns:
            dict with accuracy, dynamic_recall, and first_failure if any.
        """
        if not self.transitions:
            return {"accuracy": 1.0, "dynamic_recall": 1.0, "transitions_checked": 0}

        correct_cells = 0
        total_cells = 0
        dyn_correct = 0
        dyn_total = 0
        first_failure = None

        for t in self.transitions:
            b_grid = t["before_grid"]
            act = t["action"]
            a_grid = t["after_grid"]
            h = len(b_grid)
            w = len(b_grid[0]) if h > 0 else 0

            try:
                pred_grid = transition_fn(b_grid, act)
            except Exception as exc:
                if first_failure is None:
                    first_failure = {
                        "turn": t["turn"],
                        "action": act,
                        "error": str(exc),
                    }
                continue

            for r in range(h):
                for c in range(w):
                    actual_changed = (b_grid[r][c] != a_grid[r][c])
                    if actual_changed:
                        dyn_total += 1
                        if pred_grid and pred_grid[r][c] == a_grid[r][c]:
                            dyn_correct += 1

                    total_cells += 1
                    if pred_grid and pred_grid[r][c] == a_grid[r][c]:
                        correct_cells += 1
                    elif first_failure is None:
                        first_failure = {
                            "turn": t["turn"],
                            "action": act,
                            "r": r,
                            "c": c,
                            "expected": a_grid[r][c],
                            "predicted": pred_grid[r][c] if pred_grid else None,
                        }

        acc = (correct_cells / total_cells) if total_cells > 0 else 1.0
        rec_delta = (dyn_correct / dyn_total) if dyn_total > 0 else 1.0

        return {
            "accuracy": acc,
            "dynamic_recall": rec_delta,
            "transitions_checked": len(self.transitions),
            "first_failure": first_failure,
        }


class PurePythonAStarPlanner:
    """Deterministic State-Equivalence A* Shortest-Path Search with HUD Masking."""

    @staticmethod
    def solve(initial_grid, valid_actions, transition_fn, goal_fn, masked_strips=None, max_depth=35):
        """Finds minimal sequence of valid actions to reach goal_fn(grid) == True."""
        def hash_grid(g):
            h = len(g)
            w = len(g[0]) if h > 0 else 0
            rows = []
            for r in range(h):
                r_vals = []
                for c in range(w):
                    if masked_strips and (('row', r) in masked_strips or ('col', c) in masked_strips):
                        r_vals.append(-1)
                    else:
                        r_vals.append(g[r][c])
                rows.append(tuple(r_vals))
            return hash(tuple(rows))

        if goal_fn(initial_grid):
            return []

        init_h = hash_grid(initial_grid)
        visited = {init_h: 0}
        q = deque([(initial_grid, [])])

        while q:
            curr_grid, path = q.popleft()
            if len(path) >= max_depth:
                continue

            for act in valid_actions:
                try:
                    nxt_grid = transition_fn(curr_grid, act)
                except Exception:
                    continue
                if nxt_grid is None:
                    continue

                if goal_fn(nxt_grid):
                    return path + [act]

                nh = hash_grid(nxt_grid)
                n_cost = len(path) + 1
                if nh not in visited or n_cost < visited[nh]:
                    visited[nh] = n_cost
                    q.append((nxt_grid, path + [act]))

        return None

