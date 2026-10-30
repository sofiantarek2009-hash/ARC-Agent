"""Fast in-sandbox simulation, pathfinding, and retrodiction helpers.

This module is intentionally self-contained (pure standard library) so its source
can be spliced verbatim into the Python-tool sandbox bootstrap.
"""

from collections import deque
import math

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


ABSTAIN = -1


class RenderedMooreMachine:
    """Deterministic Rendered Moore Machine base class in sandbox."""

    def __init__(self, frame_0, level_idx=1):
        self.state = self.init_state(frame_0, level_idx)

    def init_state(self, frame_0, level_idx=1):
        return {
            "grid": [list(r) for r in frame_0],
            "level": level_idx,
            "step": 0,
            "latent_flags": {},
        }

    def transition(self, state, action, action_data=None):
        import copy
        nxt = copy.deepcopy(state)
        nxt["step"] += 1
        return nxt

    def render(self, state):
        return state.get("grid", [])

    def predict_outcome(self, state):
        return "PLAYABLE"


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

    def retrodict_hypothesis(self, transition_fn_or_class):
        """Tests candidate transition function or RenderedMooreMachine class against all recorded history transitions.
        
        Args:
            transition_fn_or_class: callable (grid, action_str) -> predicted_grid
                                    OR RenderedMooreMachine class
            
        Returns:
            dict with accuracy, dynamic_recall, coverage, and first_failure if any.
        """
        if not self.transitions:
            return {"accuracy": 1.0, "dynamic_recall": 1.0, "coverage": 1.0, "transitions_checked": 0}

        # Check if argument is a Moore machine class or instance
        is_class = isinstance(transition_fn_or_class, type)

        correct_cells = 0
        total_cells = 0
        dyn_correct = 0
        dyn_total = 0
        covered_cells = 0
        first_failure = None

        for t in self.transitions:
            b_grid = t["before_grid"]
            act = t["action"]
            a_grid = t["after_grid"]
            h = len(b_grid)
            w = len(b_grid[0]) if h > 0 else 0

            try:
                if is_class:
                    machine = transition_fn_or_class(b_grid, level_idx=1)
                    nxt_state = machine.transition(machine.state, act)
                    pred_grid = machine.render(nxt_state)
                elif callable(transition_fn_or_class):
                    pred_grid = transition_fn_or_class(b_grid, act)
                else:
                    pred_grid = None
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
                    pred_val = pred_grid[r][c] if pred_grid and r < len(pred_grid) and c < len(pred_grid[r]) else None
                    if pred_val != -1:  # Not abstained
                        covered_cells += 1
                        actual_changed = (b_grid[r][c] != a_grid[r][c])
                        if actual_changed:
                            dyn_total += 1
                            if pred_val == a_grid[r][c]:
                                dyn_correct += 1

                        total_cells += 1
                        if pred_val == a_grid[r][c]:
                            correct_cells += 1
                        elif first_failure is None:
                            first_failure = {
                                "turn": t["turn"],
                                "action": act,
                                "r": r,
                                "c": c,
                                "expected": a_grid[r][c],
                                "predicted": pred_val,
                            }

        acc = (correct_cells / total_cells) if total_cells > 0 else 1.0
        rec_delta = (dyn_correct / dyn_total) if dyn_total > 0 else 1.0
        cov = (covered_cells / max(1, len(self.transitions) * h * w))

        return {
            "accuracy": acc,
            "dynamic_recall": rec_delta,
            "coverage": cov,
            "transitions_checked": len(self.transitions),
            "first_failure": first_failure,
        }


class PurePythonGoalEstimator:
    """Core-Knowledge Goal Progress & Terminal Invariant Evaluator."""

    @staticmethod
    def evaluate_progress(initial_grid, current_grid, target_cells=None):
        """Calculates normalized progress toward goal invariants (exhaustion & alignment)."""
        h = len(current_grid)
        w = len(current_grid[0]) if h > 0 else 0
        if h == 0 or w == 0:
            return 0.0

        scores = []
        init_nonzero = sum(1 for r in range(h) for c in range(w) if initial_grid[r][c] != 0)
        curr_nonzero = sum(1 for r in range(h) for c in range(w) if current_grid[r][c] != 0)
        if init_nonzero > 0 and curr_nonzero < init_nonzero:
            scores.append((init_nonzero - curr_nonzero) / float(init_nonzero))

        if target_cells:
            # Check min distance to target
            target_set = set(target_cells)
            min_dist = float("inf")
            for r in range(h):
                for c in range(w):
                    if current_grid[r][c] != 0 and (r, c) not in target_set:
                        for tr, tc in target_set:
                            dist = abs(r - tr) + abs(c - tc)
                            if dist < min_dist:
                                min_dist = dist
            if min_dist != float("inf"):
                scores.append(1.0 / (1.0 + min_dist))

        return float(sum(scores) / max(1, len(scores))) if scores else 0.0


class PurePythonBOEDProber:
    """Speed-Depth Bayesian Optimal Experiment Design Acquisition in Sandbox."""

    @staticmethod
    def get_phase(turn_idx, dynamic_recall, expended_actions=0, human_baseline=20):
        if turn_idx <= 3 or dynamic_recall < 0.30:
            return "PHASE_I_COLD_START"
        if expended_actions >= int(3.5 * human_baseline):
            return "PHASE_IV_BUDGET_CRITICAL"
        if dynamic_recall >= 0.90:
            return "PHASE_III_EXPLOITATION_SHORTEST_PATH"
        return "PHASE_II_TARGETED_DISAMBIGUATION"

    @staticmethod
    def score_probe(action, is_untested=False, is_known_noop=False, is_known_deadly=False):
        if is_known_deadly:
            return -1000.0
        if is_known_noop:
            return -500.0
        return 10.0 if is_untested else 1.0


class PurePythonMacroOptions:
    """In-Sandbox Two-Level Hierarchical Macro-Options (§10.2, §22.1 Mechanism 22)."""

    @staticmethod
    def option_nav(start_pos, target_pos, passable_cells, fatal_cells=None, grid_shape=(64, 64)):
        if start_pos == target_pos:
            return []
        h, w = grid_shape
        fatal = set(fatal_cells) if fatal_cells else set()
        safe_passable = set(passable_cells) - fatal
        safe_passable.add(target_pos)

        deltas = {
            "ACTION1": (-1, 0),
            "ACTION2": (1, 0),
            "ACTION3": (0, -1),
            "ACTION4": (0, 1),
        }
        q = deque([(start_pos, [])])
        visited = {start_pos}

        while q:
            curr, path = q.popleft()
            if curr == target_pos:
                return path
            r, c = curr
            for act, (dr, dc) in deltas.items():
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w:
                    if nxt not in visited and (nxt in safe_passable or nxt == target_pos):
                        visited.add(nxt)
                        q.append((nxt, path + [act]))
        return []

    @staticmethod
    def option_contact(start_pos, target_obj_cells, approach_dir, passable_cells, fatal_cells=None, grid_shape=(64, 64)):
        dir_offsets = {
            "UP": (1, 0, "ACTION1"),
            "DOWN": (-1, 0, "ACTION2"),
            "LEFT": (0, 1, "ACTION3"),
            "RIGHT": (0, -1, "ACTION4"),
        }
        if approach_dir not in dir_offsets:
            return []
        dr, dc, final_act = dir_offsets[approach_dir]
        valid_stand_positions = set()
        for tr, tc in target_obj_cells:
            sr, sc = tr + dr, tc + dc
            if 0 <= sr < grid_shape[0] and 0 <= sc < grid_shape[1]:
                if (sr, sc) in passable_cells:
                    valid_stand_positions.add((sr, sc))
        if not valid_stand_positions:
            return []

        best_path = None
        for stand_pos in valid_stand_positions:
            p = PurePythonMacroOptions.option_nav(start_pos, stand_pos, passable_cells, fatal_cells, grid_shape)
            if p is not None and (best_path is None or len(p) < len(best_path)):
                best_path = p + [final_act]
        return best_path or []


class PurePythonAStarPlanner:
    """Deterministic State-Equivalence A* & Belief-Space PUCT Planner with HUD Masking (§10.2)."""

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

    @classmethod
    def solve_puct(
        cls,
        initial_grid,
        valid_actions,
        transition_models,
        goal_fn,
        masked_strips=None,
        max_depth=20,
        num_simulations=60,
        c_puct=1.414,
    ):
        """Belief-Space PUCT Planner over an ensemble of competing transition hypotheses (§10.2)."""
        if not transition_models or not valid_actions:
            return None

        if goal_fn(initial_grid):
            return []

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

        h_init = hash_grid(initial_grid)
        N = {}
        W = {}
        state_grids = {h_init: initial_grid}

        for _ in range(num_simulations):
            curr_h = h_init
            search_path = []
            depth = 0
            curr_val = 0.0

            while depth < max_depth:
                curr_grid = state_grids[curr_h]
                if goal_fn(curr_grid):
                    curr_val = 1.0
                    break

                total_state_visits = sum(N.get((curr_h, a), 0) for a in valid_actions) + 1

                best_action = None
                best_score = -float('inf')

                for a in valid_actions:
                    n_a = N.get((curr_h, a), 0)
                    w_a = W.get((curr_h, a), 0.0)
                    q_a = (w_a / n_a) if n_a > 0 else 0.0
                    p_prior = 1.0 / len(valid_actions)
                    u_a = c_puct * p_prior * (math.sqrt(total_state_visits) / (1 + n_a))
                    score = q_a + u_a
                    if score > best_score:
                        best_score = score
                        best_action = a

                if best_action is None:
                    break

                search_path.append((curr_h, best_action))

                candidate_next_grids = []
                for model in transition_models:
                    try:
                        nxt = model([list(r) for r in curr_grid], best_action)
                        if nxt is not None:
                            candidate_next_grids.append(nxt)
                    except Exception:
                        continue

                if not candidate_next_grids:
                    break

                next_grid = candidate_next_grids[0]
                next_h = hash_grid(next_grid)
                state_grids[next_h] = next_grid

                is_unseen = ((curr_h, best_action) not in N)
                curr_h = next_h
                depth += 1

                if goal_fn(next_grid):
                    curr_val = 1.0
                    break

                if is_unseen:
                    rollout_g = [list(r) for r in state_grids[curr_h]]
                    for _r in range(min(10, max_depth - depth)):
                        if goal_fn(rollout_g):
                            curr_val = 1.0
                            break
                        candidate_acts = [act for act in valid_actions if act in ("ACTION2", "ACTION4")] or valid_actions
                        act = candidate_acts[hash((curr_h, depth, _r)) % len(candidate_acts)]
                        try:
                            rollout_g = transition_models[0](rollout_g, act)
                        except Exception:
                            break
                    else:
                        curr_val = 1.0 if goal_fn(rollout_g) else 0.0
                    break

            for s_h, a in search_path:
                N[(s_h, a)] = N.get((s_h, a), 0) + 1
                W[(s_h, a)] = W.get((s_h, a), 0.0) + curr_val

        curr_h = h_init
        plan = []
        visited_in_extract = {curr_h}

        for _ in range(max_depth):
            curr_grid = state_grids[curr_h]
            if goal_fn(curr_grid):
                return plan

            best_a = None
            max_n = -1
            for a in valid_actions:
                visits = N.get((curr_h, a), 0)
                if visits > max_n:
                    max_n = visits
                    best_a = a

            if best_a is None or max_n <= 0:
                break

            plan.append(best_a)
            try:
                nxt = transition_models[0]([list(r) for r in curr_grid], best_a)
                next_h = hash_grid(nxt)
                if next_h in visited_in_extract:
                    break
                visited_in_extract.add(next_h)
                curr_h = next_h
                state_grids[curr_h] = nxt
            except Exception:
                break

        return plan if plan else None


class PurePythonFaultLocalizer:
    """Sandbox-safe 5-Type Abductive Fault Localizer (§16.1 & §18.1).
    
    Classifies surprises into Type A (No-Op), Type B (Kinematics), Type C (Non-local),
    Type D (Latent state aliasing), or Type E (Fatal termination).
    """

    def __init__(self, abstain_val=-1):
        self.abstain_val = abstain_val

    def classify_surprise(
        self,
        before_grid,
        action,
        actual_grid,
        predicted_grid=None,
        actual_status="PLAYABLE",
        predicted_status="PLAYABLE",
        history_transitions=None,
        hud_rows=None,
        hud_cols=None,
    ):
        hud_r_set = set(hud_rows or [])
        hud_c_set = set(hud_cols or [])
        h = len(actual_grid)
        w = len(actual_grid[0]) if h > 0 else 0

        # Type E Check: Fatal Termination
        if actual_status == "DEATH" and predicted_status != "DEATH":
            colliding = []
            for r in range(h):
                if r in hud_r_set:
                    continue
                for c in range(w):
                    if c in hud_c_set:
                        continue
                    if before_grid[r][c] != actual_grid[r][c]:
                        colliding.append((r, c, before_grid[r][c], actual_grid[r][c]))
            is_timer = len(colliding) == 0
            cause = "BUDGET_EXHAUSTION" if is_timer else "LETHAL_HAZARD_COLLISION"
            msg = (
                f"Action '{action}' caused GAME_OVER via timer exhaustion."
                if is_timer
                else f"Action '{action}' caused GAME_OVER via interior hazard collision at {colliding[:4]}."
            )
            return {
                "error_type": "TYPE_E",
                "root_cause": cause,
                "diagnostic_message": msg,
                "colliding_cells": colliding[:8],
            }

        # Check interior changes between before and actual
        interior_changes = []
        for r in range(h):
            if r in hud_r_set:
                continue
            for c in range(w):
                if c in hud_c_set:
                    continue
                if before_grid[r][c] != actual_grid[r][c]:
                    interior_changes.append((r, c, before_grid[r][c], actual_grid[r][c]))

        has_change = len(interior_changes) > 0

        # Discrepancies with prediction
        discrepancies = []
        predicts_change = False
        if predicted_grid is not None:
            for r in range(h):
                if r in hud_r_set:
                    continue
                for c in range(w):
                    if c in hud_c_set:
                        continue
                    pv = predicted_grid[r][c]
                    bv = before_grid[r][c]
                    av = actual_grid[r][c]
                    if pv != self.abstain_val and pv != bv:
                        predicts_change = True
                    if pv != self.abstain_val and pv != av:
                        discrepancies.append((r, c, pv, av))

        # Type A: Unexpected No-Op
        if not has_change and (predicts_change or len(discrepancies) > 0):
            return {
                "error_type": "TYPE_A",
                "root_cause": "UNEXPECTED_NO_OP",
                "diagnostic_message": (
                    f"Action '{action}' produced 0 interior grid mutations despite predicting movement. "
                    f"Root cause: Unmodeled solid obstacle or impassable boundary."
                ),
            }

        # Type D: Latent State Aliasing (identical past observation produced different outcome)
        if history_transitions:
            b_hash = hash(tuple(before_grid[r][c] for r in range(h) if r not in hud_r_set for c in range(w) if c not in hud_c_set))
            a_hash = hash(tuple(actual_grid[r][c] for r in range(h) if r not in hud_r_set for c in range(w) if c not in hud_c_set))
            for prev in history_transitions:
                pb = prev.get("before_grid")
                pa = prev.get("action")
                pafter = prev.get("after_grid")
                if pb and pa == action:
                    pb_h = hash(tuple(pb[r][c] for r in range(len(pb)) if r not in hud_r_set for c in range(len(pb[0])) if c not in hud_c_set))
                    if pb_h == b_hash:
                        pafter_h = hash(tuple(pafter[r][c] for r in range(len(pafter)) if r not in hud_r_set for c in range(len(pafter[0])) if c not in hud_c_set))
                        if pafter_h != a_hash:
                            return {
                                "error_type": "TYPE_D",
                                "root_cause": "LATENT_STATE_ALIASING",
                                "diagnostic_message": (
                                    f"Non-injective transition for action '{action}'. Identical visible state at turn "
                                    f"{prev.get('turn_idx', '?')} produced different result. Root cause: Hidden internal state h_t."
                                ),
                            }

        if not discrepancies:
            return {
                "error_type": "NONE",
                "root_cause": "CONCORDANT",
                "diagnostic_message": "Actual grid concordant with prediction.",
            }

        # Distinguish Type B (Kinematics) vs Type C (Non-local side effect)
        rows = [r for (r, c, _, _) in discrepancies]
        cols = [c for (r, c, _, _) in discrepancies]
        span = max(max(rows) - min(rows), max(cols) - min(cols)) if rows else 0

        action_origin_coords = []
        for r in range(h):
            if r in hud_r_set:
                continue
            for c in range(w):
                if c in hud_c_set:
                    continue
                if before_grid[r][c] != actual_grid[r][c] and (r, c) not in [(dr, dc) for dr, dc, _, _ in discrepancies]:
                    action_origin_coords.append((r, c))

        dist_from_origin = 0
        if action_origin_coords:
            for dr, dc, _, _ in discrepancies:
                min_d = min(abs(dr - or_r) + abs(dc - or_c) for or_r, or_c in action_origin_coords)
                dist_from_origin = max(dist_from_origin, min_d)

        if span <= 4 and dist_from_origin <= 3:
            return {
                "error_type": "TYPE_B",
                "root_cause": "LOCALIZED_KINEMATICS_ERROR",
                "diagnostic_message": (
                    f"Kinematic divergence for '{action}': entity displacement span ({span}) differs from prediction. "
                    f"Likely multi-cell slide, bounce, or diagonal move."
                ),
                "mismatches": discrepancies[:5],
            }
        else:
            return {
                "error_type": "TYPE_C",
                "root_cause": "NON_LOCAL_CONTACT_EFFECT",
                "diagnostic_message": (
                    f"Non-local effect for '{action}': changes span {span} cells across board (dist from origin={dist_from_origin}). "
                    f"Likely switch/door connection, portal, or global field effect."
                ),
                "mismatches": discrepancies[:6],
            }


class PurePythonCommitGate:
    """Pre-Action Commit Verifier & Surprise Gate in Sandbox (§10.3 & §16.1)."""

    def __init__(self, abstain_val=-1):
        self.localizer = PurePythonFaultLocalizer(abstain_val=abstain_val)
        self.active_plan = []
        self.expected_frames = []
        self.expected_outcomes = []
        self.last_fault = None

    def commit_plan(self, action_seq, expected_frames=None, expected_outcomes=None):
        self.active_plan = list(action_seq)
        self.expected_frames = list(expected_frames or [])
        self.expected_outcomes = list(expected_outcomes or [])
        self.last_fault = None

    def evaluate_step(
        self,
        before_grid,
        action,
        actual_grid,
        step_idx=0,
        actual_status="PLAYABLE",
        predicted_frame=None,
        predicted_outcome="PLAYABLE",
        history_transitions=None,
        hud_rows=None,
        hud_cols=None,
    ):
        pred_f = predicted_frame if predicted_frame is not None else (
            self.expected_frames[step_idx] if step_idx < len(self.expected_frames) else None
        )
        pred_out = predicted_outcome if predicted_outcome != "PLAYABLE" else (
            self.expected_outcomes[step_idx] if step_idx < len(self.expected_outcomes) else "PLAYABLE"
        )

        diag = self.localizer.classify_surprise(
            before_grid=before_grid,
            action=action,
            actual_grid=actual_grid,
            predicted_grid=pred_f,
            actual_status=actual_status,
            predicted_status=pred_out,
            history_transitions=history_transitions,
            hud_rows=hud_rows,
            hud_cols=hud_cols,
        )

        is_surprise = diag["error_type"] != "NONE"
        if is_surprise:
            self.last_fault = diag
            self.active_plan.clear()
            self.expected_frames.clear()
            self.expected_outcomes.clear()
            return True, diag

        return False, None


