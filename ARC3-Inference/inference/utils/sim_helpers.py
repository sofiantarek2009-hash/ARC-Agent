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
