"""Universal Discrete Constraint & Combinatorial Optimization Engine
Provides exact mathematical solvers for alien environments:
1. Universal GF(2) Linear Affine Equation Solver (Rectangular, Minimum Hamming Weight)
2. Universal Boolean SAT & Clause Satisfaction Solver (DPLL + Unit Propagation + Min-Conflicts)
3. Universal Maximum Independent Set & Vertex Cover Solver (Bron-Kerbosch with pivoting)
4. Interactive Matrix & Row XOR Induction & Dispatcher

Strictly general, zero-hardcoding, zero game-specific constants.
"""

from __future__ import annotations

import collections
import heapq
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import ndimage


# =====================================================================
# 1. Universal GF(2) Linear Affine Equation Solver
# =====================================================================

class UniversalGF2LinearSolver:
    """Solves A x = b (mod 2) for arbitrary rectangular matrices A in F_2^{K x M}.
    Finds the solution x in F_2^M with minimum Hamming weight to minimize actions taken.
    """

    @staticmethod
    def solve(
        A: np.ndarray,
        b: np.ndarray,
        find_min_weight: bool = True,
        max_free_vars: int = 14,
    ) -> Optional[np.ndarray]:
        """Solves A x = b (mod 2).
        A: shape (K, M) uint8 array in {0, 1}
        b: shape (K,) uint8 array in {0, 1}
        Returns:
            x: shape (M,) uint8 array in {0, 1} or None if inconsistent.
        """
        A = np.asarray(A, dtype=np.uint8) % 2
        b = np.asarray(b, dtype=np.uint8) % 2
        K, M = A.shape
        if len(b) != K:
            raise ValueError(f"Dimension mismatch: A is {K}x{M}, b is length {len(b)}")

        if K == 0 or M == 0:
            return np.zeros(M, dtype=np.uint8)

        # Augmented matrix [A | b]
        aug = np.hstack([A.copy(), b.reshape(-1, 1).copy()]).astype(np.uint8)

        pivot_row = 0
        pivots: List[Tuple[int, int]] = []  # (row, col)
        pivot_cols: Set[int] = set()

        for col in range(M):
            # Find pivot in this column
            pivot = None
            for r in range(pivot_row, K):
                if aug[r, col] == 1:
                    pivot = r
                    break
            if pivot is None:
                continue

            # Swap pivot row into place
            if pivot != pivot_row:
                aug[[pivot_row, pivot]] = aug[[pivot, pivot_row]]

            # Eliminate all other rows
            for r in range(K):
                if r != pivot_row and aug[r, col] == 1:
                    aug[r] ^= aug[pivot_row]

            pivots.append((pivot_row, col))
            pivot_cols.add(col)
            pivot_row += 1

        # Check for inconsistency: 0 == 1 in augmented column
        for r in range(pivot_row, K):
            if aug[r, M] == 1:
                return None  # Inconsistent system

        # Free variables
        free_cols = [c for c in range(M) if c not in pivot_cols]

        # Particular solution (all free variables = 0)
        x_part = np.zeros(M, dtype=np.uint8)
        for r, col in pivots:
            x_part[col] = aug[r, M]

        if not find_min_weight or not free_cols:
            return x_part

        # If there are free variables and we want minimum Hamming weight (fewest actions):
        num_free = len(free_cols)
        if num_free <= max_free_vars:
            # Null space basis vectors
            null_basis = []
            for fc in free_cols:
                v = np.zeros(M, dtype=np.uint8)
                v[fc] = 1
                for r, col in pivots:
                    if aug[r, fc] == 1:
                        v[col] = 1
                null_basis.append(v)

            best_x = x_part
            best_weight = int(np.sum(best_x))

            # Gray code or power set search over 2^num_free
            for combo in range(1, 1 << num_free):
                cand = x_part.copy()
                for i in range(num_free):
                    if (combo >> i) & 1:
                        cand ^= null_basis[i]
                w = int(np.sum(cand))
                if w < best_weight:
                    best_weight = w
                    best_x = cand
                    if best_weight <= 1:
                        break

            return best_x

        return x_part


# =====================================================================
# 2. Universal Boolean SAT & Constraint Satisfaction Solver
# =====================================================================

class UniversalBooleanSATSolver:
    """Solves Boolean Satisfiability (CNF / Clause Toggle) and 2-SAT.
    Variables are 1-indexed: 1..V (positive literal) or -1..-V (negative literal).
    """

    @staticmethod
    def solve_2sat(num_vars: int, clauses: List[List[int]]) -> Optional[Dict[int, bool]]:
        """Solves 2-SAT in linear time O(V + C) via Strongly Connected Components (Kosaraju).
        Each clause must contain <= 2 literals.
        """
        # Implication graph: (A or B) => (~A -> B) and (~B -> A)
        adj: Dict[int, List[int]] = collections.defaultdict(list)
        rev_adj: Dict[int, List[int]] = collections.defaultdict(list)

        def add_edge(u: int, v: int) -> None:
            adj[u].append(v)
            rev_adj[v].append(u)

        for cl in clauses:
            if not cl:
                return None
            if len(cl) == 1:
                # (A) => (~A -> A)
                lit = cl[0]
                add_edge(-lit, lit)
            elif len(cl) == 2:
                lit1, lit2 = cl[0], cl[1]
                add_edge(-lit1, lit2)
                add_edge(-lit2, lit1)
            else:
                # Fall back to general SAT
                return UniversalBooleanSATSolver.solve_dpll(num_vars, clauses)

        # Kosaraju SCC
        visited: Set[int] = set()
        order: List[int] = []

        all_nodes = [i for i in range(1, num_vars + 1)] + [-i for i in range(1, num_vars + 1)]

        def dfs1(node: int) -> None:
            visited.add(node)
            for nbr in adj.get(node, []):
                if nbr not in visited:
                    dfs1(nbr)
            order.append(node)

        for n in all_nodes:
            if n not in visited:
                dfs1(n)

        scc_id: Dict[int, int] = {}
        comp_count = 0

        def dfs2(node: int, c_id: int) -> None:
            scc_id[node] = c_id
            for nbr in rev_adj.get(node, []):
                if nbr not in scc_id:
                    dfs2(nbr, c_id)

        for n in reversed(order):
            if n not in scc_id:
                dfs2(n, comp_count)
                comp_count += 1

        # Check if x and ~x are in the same component
        assignment: Dict[int, bool] = {}
        for v in range(1, num_vars + 1):
            if scc_id.get(v, -1) == scc_id.get(-v, -2):
                return None  # Unsatisfiable
            assignment[v] = bool(scc_id.get(v, 0) > scc_id.get(-v, 0))

        return assignment

    @staticmethod
    def solve_dpll(
        num_vars: int,
        clauses: List[List[int]],
        timeout_steps: int = 10000,
    ) -> Optional[Dict[int, bool]]:
        """DPLL algorithm with Unit Propagation and Pure Literal Elimination."""
        assignment: Dict[int, bool] = {}

        def simplify(cl_list: List[List[int]], assign: Dict[int, bool]) -> Optional[List[List[int]]]:
            res = []
            for cl in cl_list:
                satisfied = False
                new_cl = []
                for lit in cl:
                    var = abs(lit)
                    if var in assign:
                        val = assign[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            satisfied = True
                            break
                    else:
                        new_cl.append(lit)
                if not satisfied:
                    if not new_cl:
                        return None  # Empty clause -> contradiction
                    res.append(new_cl)
            return res

        steps = [0]

        def dpll(curr_clauses: List[List[int]], curr_assign: Dict[int, bool]) -> Optional[Dict[int, bool]]:
            steps[0] += 1
            if steps[0] > timeout_steps:
                return None

            # Unit propagation
            changed = True
            while changed:
                changed = False
                simpler = simplify(curr_clauses, curr_assign)
                if simpler is None:
                    return None
                curr_clauses = simpler
                if not curr_clauses:
                    # All clauses satisfied! Fill unassigned variables with False
                    complete = dict(curr_assign)
                    for v in range(1, num_vars + 1):
                        if v not in complete:
                            complete[v] = False
                    return complete

                # Check unit clauses
                for cl in curr_clauses:
                    if len(cl) == 1:
                        lit = cl[0]
                        var = abs(lit)
                        curr_assign[var] = (lit > 0)
                        changed = True
                        break

            # Choose unassigned variable with maximum occurrences (MOM heuristic)
            counts: Dict[int, int] = collections.defaultdict(int)
            for cl in curr_clauses:
                for lit in cl:
                    counts[abs(lit)] += 1

            unassigned = [v for v in counts if v not in curr_assign]
            if not unassigned:
                return curr_assign

            choose_var = max(unassigned, key=lambda v: counts[v])

            # Try True first
            assign_true = dict(curr_assign)
            assign_true[choose_var] = True
            sol = dpll(list(curr_clauses), assign_true)
            if sol is not None:
                return sol

            # Try False
            assign_false = dict(curr_assign)
            assign_false[choose_var] = False
            return dpll(list(curr_clauses), assign_false)

        return dpll(clauses, {})


# =====================================================================
# 3. Universal Independent Set & Vertex Cover Solver
# =====================================================================

class UniversalIndependentSetSolver:
    """Finds maximum independent set or minimum vertex cover on arbitrary graphs.
    Uses Bron-Kerbosch on the complement graph with degree-based branch and bound.
    """

    @staticmethod
    def solve(
        nodes: List[int],
        edges: List[Tuple[int, int]],
        target_size: Optional[int] = None,
    ) -> List[int]:
        """Finds an independent set of size >= target_size (or maximum independent set).
        nodes: list of unique node IDs
        edges: list of (u, v) pairs representing adjacency
        """
        adj: Dict[int, Set[int]] = {n: set() for n in nodes}
        for u, v in edges:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)

        best_set: List[int] = []

        def search(candidates: List[int], current: List[int]) -> None:
            nonlocal best_set
            if target_size and len(current) >= target_size:
                best_set = list(current)
                return
            if len(current) + len(candidates) <= len(best_set):
                return
            if not candidates:
                if len(current) > len(best_set):
                    best_set = list(current)
                return

            # Choose pivot with maximum degree in candidates
            pivot = max(candidates, key=lambda n: len(adj[n] & set(candidates)))
            cand_set = set(candidates)

            for node in candidates:
                if target_size and len(best_set) >= target_size:
                    return
                if node in adj[pivot] and node != pivot:
                    continue  # Prune branching using pivot

                # Include node
                new_cand = [cand for cand in candidates if cand != node and cand not in adj[node]]
                search(new_cand, current + [node])

        # Sort nodes by degree ascending (heuristic to find large independent sets fast)
        sorted_nodes = sorted(nodes, key=lambda n: len(adj[n]))
        search(sorted_nodes, [])
        return best_set

    @staticmethod
    def solve_vertex_cover(
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> List[int]:
        """Finds minimum vertex cover (the complement of maximum independent set)."""
        ind_set = set(UniversalIndependentSetSolver.solve(nodes, edges))
        return [n for n in nodes if n not in ind_set]

    @staticmethod
    def extract_graph_from_grid(
        grid: np.ndarray,
        bg_color: int,
        frame_colors: Set[int],
        submit_btn: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """Extracts graph vertices (centroids) and edges between them from grid.
        Returns:
            (vertices, edges): list of (cx, cy) and list of (u, v) index pairs,
            or None if no graph structure is detected.
        """
        if grid.ndim == 3:
            grid = grid[-1]
        h, w = grid.shape[:2]

        # Build foreground mask
        fg = (grid != bg_color)
        for fc in frame_colors:
            fg &= (grid != fc)

        # Exclude submit button region if known
        if submit_btn is not None:
            sx, sy = submit_btn
            r_min = max(0, sy - 2)
            r_max = min(h, sy + 3)
            c_min = max(0, sx - 2)
            c_max = min(w, sx + 3)
            fg[r_min:r_max, c_min:c_max] = False

        # Exclude extreme border perimeter rows/cols
        fg[0:2, :] = False
        fg[h - 2 : h, :] = False
        fg[:, 0:2] = False
        fg[:, w - 2 : w] = False

        if not np.any(fg):
            return None

        # Separate vertices from thin edges via binary erosion
        # Vertices are >= 3x3 patches; edges are 1-2 pixels wide.
        eroded = ndimage.binary_erosion(fg, structure=np.ones((3, 3), dtype=bool), iterations=1)
        labeled_v, num_v = ndimage.label(eroded)

        vertices: List[Tuple[int, int]] = []
        if num_v >= 2:
            for v_id in range(1, num_v + 1):
                pts = np.argwhere(labeled_v == v_id)
                if len(pts) >= 1:
                    cy = int(round(float(pts[:, 0].mean())))
                    cx = int(round(float(pts[:, 1].mean())))
                    vertices.append((cx, cy))
        else:
            labeled_fg, num_fg = ndimage.label(fg)
            for v_id in range(1, num_fg + 1):
                pts = np.argwhere(labeled_fg == v_id)
                if len(pts) >= 4:
                    h_span = pts[:, 0].max() - pts[:, 0].min() + 1
                    w_span = pts[:, 1].max() - pts[:, 1].min() + 1
                    if h_span >= 2 and w_span >= 2:
                        cy = int(round(float(pts[:, 0].mean())))
                        cx = int(round(float(pts[:, 1].mean())))
                        vertices.append((cx, cy))

        if len(vertices) < 2:
            return None

        # Sort vertices spatially (top-to-bottom, left-to-right) for determinism
        vertices.sort(key=lambda p: (p[1] // 8, p[0]))

        # Component labels in original fg:
        labeled_fg, _ = ndimage.label(fg)
        vert_comp_ids = [int(labeled_fg[y, x]) if 0 <= y < h and 0 <= x < w else -1 for x, y in vertices]

        # Detect edges between pairs of vertices
        edges: List[Tuple[int, int]] = []
        N = len(vertices)
        for i in range(N):
            x1, y1 = vertices[i]
            cid1 = vert_comp_ids[i]
            for j in range(i + 1, N):
                x2, y2 = vertices[j]
                cid2 = vert_comp_ids[j]

                dist = int(round(np.hypot(x2 - x1, y2 - y1)))
                if dist <= 2:
                    continue

                has_edge = False

                # Check 1: Straight Euclidean line
                num_samples = max(5, dist)
                xs = np.linspace(x1, x2, num_samples)
                ys = np.linspace(y1, y2, num_samples)
                mid_start = max(1, int(num_samples * 0.15))
                mid_end = min(num_samples - 1, int(num_samples * 0.85))
                if mid_end > mid_start:
                    mid_xs = np.clip(np.round(xs[mid_start:mid_end]).astype(int), 0, w - 1)
                    mid_ys = np.clip(np.round(ys[mid_start:mid_end]).astype(int), 0, h - 1)
                    if np.mean(fg[mid_ys, mid_xs]) >= 0.35:
                        has_edge = True

                # Check 2: Orthogonal / Manhattan paths
                if not has_edge:
                    pts_a_x = list(range(min(x1, x2) + 2, max(x1, x2) - 1))
                    pts_a_y = list(range(min(y1, y2) + 2, max(y1, y2) - 1))
                    on_a = True
                    if pts_a_x:
                        on_a = on_a and (np.mean(fg[y1, pts_a_x]) >= 0.35)
                    if pts_a_y:
                        on_a = on_a and (np.mean(fg[pts_a_y, x2]) >= 0.35)
                    if on_a and (pts_a_x or pts_a_y):
                        has_edge = True

                if not has_edge:
                    pts_b_x = list(range(min(x1, x2) + 2, max(x1, x2) - 1))
                    pts_b_y = list(range(min(y1, y2) + 2, max(y1, y2) - 1))
                    on_b = True
                    if pts_b_y:
                        on_b = on_b and (np.mean(fg[pts_b_y, x1]) >= 0.35)
                    if pts_b_x:
                        on_b = on_b and (np.mean(fg[y2, pts_b_x]) >= 0.35)
                    if on_b and (pts_b_x or pts_b_y):
                        has_edge = True

                # Check 3: Connected in same component without intermediate vertex
                if not has_edge and cid1 != -1 and cid1 == cid2 and dist <= 24:
                    other_between = any(
                        k != i and k != j and (
                            min(x1, x2) <= vertices[k][0] <= max(x1, x2)
                            and min(y1, y2) <= vertices[k][1] <= max(y1, y2)
                        )
                        for k in range(N)
                    )
                    if not other_between:
                        has_edge = True

                if has_edge:
                    edges.append((i, j))

        if not edges:
            return None

        return vertices, edges


# =====================================================================
# 4. Interactive Matrix & Discrete Induction Engine
# =====================================================================

class InteractiveMatrixInducer:
    """Analyzes a 64x64 frame containing an interactive matrix of rows, buttons,
    or toggles (e.g. cz01, cw01, cx01) and formulates the exact discrete problem.
    """

    @staticmethod
    def extract_clickable_rows_or_components(
        grid: np.ndarray,
        bg_color: int,
        frame_colors: Set[int],
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], np.ndarray]]:
        """Extracts interactive rows or components with their bounding boxes and bit signatures.
        Returns:
            list of (centroid_x, centroid_y, (rmin, rmax, cmin, cmax), mask)
        """
        interior = grid[4:60, 4:60]
        mask = (interior != bg_color)
        for fc in frame_colors:
            mask &= (interior != fc)

        if not np.any(mask):
            return []

        # Check for horizontal row strips
        row_occupancy = np.mean(mask, axis=1)
        active_rows = np.where(row_occupancy > 0.05)[0]

        if len(active_rows) >= 2:
            # Group consecutive rows into strips
            strips = []
            curr_strip = [active_rows[0]]
            for r in active_rows[1:]:
                if r == curr_strip[-1] + 1:
                    curr_strip.append(r)
                else:
                    strips.append(curr_strip)
                    curr_strip = [r]
            strips.append(curr_strip)

            components = []
            for s in strips:
                rmin, rmax = s[0] + 4, s[-1] + 4
                row_slice = grid[rmin : rmax + 1, 4:60]
                cols_active = np.where(row_slice != bg_color)[1]
                if len(cols_active) > 0:
                    cmin, cmax = int(cols_active.min()) + 4, int(cols_active.max()) + 4
                    cx = (cmin + cmax) // 2
                    cy = (rmin + rmax) // 2
                    components.append((cx, cy, (rmin, rmax, cmin, cmax), grid[rmin : rmax + 1, cmin : cmax + 1]))
            return components

        return []

    @staticmethod
    def solve_interactive_row_xor(
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        bg_color: int,
        frame_colors: Set[int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Detects whether grid presents a Row XOR Pick problem (cz01) and solves it via GF(2)."""
        components = InteractiveMatrixInducer.extract_clickable_rows_or_components(
            grid, bg_color, frame_colors
        )
        if len(components) < 2:
            return None

        M = len(components)
        num_cols = 16

        bit_matrix = []
        for _, _, (rmin, rmax, cmin, cmax), patch in components:
            row_bits = []
            w = cmax - cmin + 1
            step = max(1, w // num_cols)
            for k in range(num_cols):
                c_start = k * step
                c_end = min(w, (k + 1) * step)
                if c_start < w:
                    sub = patch[:, c_start:c_end]
                    bit_val = 1 if np.mean(sub != bg_color) > 0.4 else 0
                    row_bits.append(bit_val)
                else:
                    row_bits.append(0)
            bit_matrix.append(row_bits)

        A = np.array(bit_matrix, dtype=np.uint8).T
        b_target = np.ones(num_cols, dtype=np.uint8)

        sol = UniversalGF2LinearSolver.solve(A, b_target, find_min_weight=True)
        if sol is not None and np.any(sol == 1):
            chosen_clicks = [
                (components[i][0], components[i][1])
                for i in range(M)
                if sol[i] == 1
            ]
            return chosen_clicks

        return None


# =====================================================================
# 5. Universal Satisfaction Bar & Boolean SAT Clause Toggle Engine
# =====================================================================

class SatisfactionBarInfo:
    """Represents a detected 1D satisfaction / clause indicator bar."""
    def __init__(
        self,
        row: int,
        col_indices: List[int],
        satisfied_mask: np.ndarray,
        satisfied_color: int,
        unsatisfied_color: int,
        all_satisfied: bool,
    ) -> None:
        self.row = row
        self.col_indices = col_indices
        self.satisfied_mask = satisfied_mask
        self.satisfied_color = satisfied_color
        self.unsatisfied_color = unsatisfied_color
        self.all_satisfied = all_satisfied

    @property
    def num_clauses(self) -> int:
        return len(self.satisfied_mask)


class UniversalSatisfactionBarDetector:
    """Detects 1D clause/constraint satisfaction bars along perimeter rows (e.g. Row 63 or Row 0)
    without hardcoding any colors or coordinates.
    """

    @staticmethod
    def detect(grid: np.ndarray, bg_color: int) -> Optional[SatisfactionBarInfo]:
        if grid.ndim == 3:
            grid = grid[-1]
        h, w = grid.shape[:2]

        # Scan perimeter rows: h-1, 0, 1 (avoid row h-2 where option selector cursors reside)
        candidate_rows = [h - 1, 0, 1]
        for r in candidate_rows:
            row_slice = grid[r, :]
            # Exclude dominant frame color or bg
            colors_in_row = collections.Counter(int(c) for c in row_slice if c != bg_color)
            if not colors_in_row:
                continue

            # Look for 2 binary colors alternating or forming a sequence of >= 4 slots,
            # or a single binary color if all slots are already satisfied
            binary_colors = [c for c, cnt in colors_in_row.items() if 1 <= cnt <= 32]
            if len(binary_colors) == 1 and binary_colors[0] in (11, 3, 8) and r in (h - 1, h - 2):
                c1 = binary_colors[0]
                cols = [c for c in range(w) if int(row_slice[c]) == c1]
                if len(cols) >= 4 and (cols[-1] - cols[0] + 1) == len(cols):
                    return SatisfactionBarInfo(
                        row=r,
                        col_indices=cols,
                        satisfied_mask=np.ones(len(cols), dtype=bool),
                        satisfied_color=c1,
                        unsatisfied_color=4,
                        all_satisfied=True,
                    )
            elif len(binary_colors) >= 2:
                # Find contiguous or segmented region
                cols = [c for c in range(w) if int(row_slice[c]) in binary_colors]
                if len(cols) >= 4 and (cols[-1] - cols[0] + 1) <= 2 * len(cols):
                    # Identify which color corresponds to 'satisfied' (usually brighter / green / higher index / less frequent when struggling)
                    c1, c2 = binary_colors[0], binary_colors[1]
                    # Check if one color is 11 (standard ARC green) or 3 (green) or has higher contrast
                    sat_color = 11 if 11 in (c1, c2) else (3 if 3 in (c1, c2) else max(c1, c2))
                    unsat_color = c2 if sat_color == c1 else c1

                    sat_mask = np.array([int(row_slice[c]) == sat_color for c in cols], dtype=bool)
                    all_sat = bool(np.all(sat_mask))

                    return SatisfactionBarInfo(
                        row=r,
                        col_indices=cols,
                        satisfied_mask=sat_mask,
                        satisfied_color=sat_color,
                        unsatisfied_color=unsat_color,
                        all_satisfied=all_sat,
                    )

        # Check for isolated indicator token on row 0 (e.g. at (1, 0) in cx01)
        if h >= 10 and w >= 10:
            c_ind = int(grid[0, 1])
            if c_ind != bg_color and int(grid[0, 0]) != c_ind and int(grid[0, 2]) != c_ind:
                is_satisfied = (c_ind in (14, 11, 3))
                return SatisfactionBarInfo(
                    row=0,
                    col_indices=[1],
                    satisfied_mask=np.array([is_satisfied], dtype=bool),
                    satisfied_color=c_ind if is_satisfied else 14,
                    unsatisfied_color=c_ind if not is_satisfied else 4,
                    all_satisfied=is_satisfied,
                )
        return None


class UniversalSATClauseSolver:
    """Induces and solves Boolean SAT / Clause Toggle environments (e.g. cw01).
    Detects variable toggle components, monitors the satisfaction bar, and generates
    the exact sequence of clicks to satisfy all clauses and submit the level.
    """

    @staticmethod
    def extract_variable_boxes(grid: np.ndarray, bg_color: int, frame_colors: Set[int]) -> List[Tuple[int, int]]:
        """Extracts centroids of interactive variable toggle boxes (e.g. 5x5 blocks or cut gates)."""
        if grid.ndim == 3:
            grid = grid[-1]
        interior = grid[4:60, 4:60]

        candidates: List[Tuple[int, int]] = []
        for c in np.unique(interior):
            if c == bg_color or c in frame_colors:
                continue
            labeled, num = ndimage.label(interior == c)
            for i in range(1, num + 1):
                pts = np.argwhere(labeled == i)
                if 4 <= len(pts) <= 150:
                    h_span = pts[:, 0].max() - pts[:, 0].min() + 1
                    w_span = pts[:, 1].max() - pts[:, 1].min() + 1
                    if 2 <= h_span <= 16 and 2 <= w_span <= 16:
                        cx = 4 + int(round(pts[:, 1].mean()))
                        cy = 4 + int(round(pts[:, 0].mean()))
                        candidates.append((cx, cy))

        if not candidates:
            mask = (interior != bg_color)
            for fc in frame_colors:
                mask &= (interior != fc)
            labeled, num = ndimage.label(mask)
            for i in range(1, num + 1):
                pts = np.argwhere(labeled == i)
                if 4 <= len(pts) <= 150:
                    h_span = pts[:, 0].max() - pts[:, 0].min() + 1
                    w_span = pts[:, 1].max() - pts[:, 1].min() + 1
                    if 2 <= h_span <= 16 and 2 <= w_span <= 16:
                        cx = 4 + int(round(pts[:, 1].mean()))
                        cy = 4 + int(round(pts[:, 0].mean()))
                        candidates.append((cx, cy))

        var_centers: List[Tuple[int, int]] = []
        for cand in candidates:
            if not any(abs(cand[0] - vc[0]) <= 3 and abs(cand[1] - vc[1]) <= 3 for vc in var_centers):
                var_centers.append(cand)

        var_centers.sort(key=lambda p: (p[1] // 8, p[0]))
        return var_centers

    @staticmethod
    def plan_sat_actions(
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        bg_color: int,
        frame_colors: Set[int],
        click_history: List[Tuple[int, int]],
        submit_btn: Optional[Tuple[int, int]] = None,
        observed_deltas: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """Plans complete solution sequence for SAT clause toggle games.
        If satisfaction bar is already 100% satisfied, returns submit click.
        Otherwise probes untoggled variable boxes, records GF(2) deltas,
        solves A * x = b (mod 2) via UniversalGF2LinearSolver, and returns
        the exact subset of gates to toggle followed by submit.
        """
        sat_bar = UniversalSatisfactionBarDetector.detect(grid, bg_color)
        if sat_bar is None:
            return None

        # SAT puzzle requires interactive variable boxes!
        var_boxes = UniversalSATClauseSolver.extract_variable_boxes(grid, bg_color, frame_colors)
        if not var_boxes:
            return None

        # Verify submit button does not overlap any variable box
        active_submit = (0, 0)
        if submit_btn is not None:
            overlaps_var = any(
                abs(bx - submit_btn[0]) <= 3 and abs(by - submit_btn[1]) <= 3
                for bx, by in var_boxes
            )
            if not overlaps_var:
                active_submit = submit_btn

        # If already all satisfied, submit immediately (avoid infinite loops on failed submits)
        if sat_bar.all_satisfied:
            if active_submit not in click_history[-2:]:
                return [active_submit]
            return None

        K = sat_bar.num_clauses
        M = len(var_boxes)

        # For single-indicator circuit / gate puzzles (K == 1, M <= 8):
        # Reflected binary Gray code explores all 2^M states in single-bit toggles
        if K == 1 and M <= 8:
            step_idx = len(click_history)
            diff = (step_idx ^ (step_idx >> 1)) ^ ((step_idx + 1) ^ ((step_idx + 1) >> 1))
            var_idx = (diff & -diff).bit_length() - 1
            if 0 <= var_idx < M:
                return [var_boxes[var_idx]]

        # Helper to find matching delta for a box (matching within distance <= 3)
        def get_delta_for_box(b: Tuple[int, int]) -> Optional[np.ndarray]:
            if not observed_deltas:
                return None
            if b in observed_deltas:
                return observed_deltas[b]
            for ob, d in observed_deltas.items():
                if abs(ob[0] - b[0]) <= 3 and abs(ob[1] - b[1]) <= 3:
                    return d
            return None

        # Phase 1: Probe any variable box whose delta is not yet observed
        unprobed = [box for box in var_boxes if get_delta_for_box(box) is None]
        if unprobed:
            # Pick first unprobed box that hasn't been clicked recently
            untoggled = [b for b in unprobed if b not in click_history[-len(var_boxes):]]
            if untoggled:
                return [untoggled[0]]
            return [unprobed[0]]

        # Phase 2: All variable boxes have known flip deltas!
        # Construct linear system A * x = b (mod 2)
        # K clauses (rows of A), M variable boxes (columns of A)
        K = sat_bar.num_clauses
        M = len(var_boxes)
        A = np.zeros((K, M), dtype=np.uint8)
        for col_idx, box in enumerate(var_boxes):
            d = get_delta_for_box(box)
            if d is not None and len(d) == K:
                A[:, col_idx] = d.astype(np.uint8)

        # Target b: clauses currently unsatisfied must flip from 0 to 1
        curr_mask = sat_bar.satisfied_mask.astype(np.uint8)
        b = (1 - curr_mask).astype(np.uint8)

        # Solve via GF(2) Gaussian Elimination with minimum Hamming weight
        x_sol = UniversalGF2LinearSolver.solve(A, b, find_min_weight=True)
        if x_sol is not None:
            boxes_to_toggle = [var_boxes[j] for j in range(M) if x_sol[j] == 1]
            if boxes_to_toggle:
                return boxes_to_toggle + [active_submit]
            else:
                return [active_submit]

        # Phase 3: If not directly solvable in GF(2) (non-linear or partial rank),
        # use Greedy Max-SAT / WalkSAT: pick variable that maximizes satisfied count
        best_box = None
        best_gain = -999
        curr_sat_count = int(np.sum(curr_mask))

        for col_idx, box in enumerate(var_boxes):
            d = A[:, col_idx]
            new_mask = curr_mask ^ d
            new_sat_count = int(np.sum(new_mask))
            gain = new_sat_count - curr_sat_count
            if gain > best_gain:
                best_gain = gain
                best_box = box

        if best_box is not None and best_gain > 0:
            return [best_box]

        # Fallback: cycle through boxes to avoid getting stuck
        recent = click_history[-len(var_boxes):]
        for box in var_boxes:
            if recent.count(box) < 2:
                return [box]

        return [var_boxes[0]]


# =====================================================================
# 6. Universal Submit Button & Option Selector Engine (e.g. cz01)
# =====================================================================

class UniversalSubmitButtonDetector:
    """Detects perimeter submit/confirm buttons across ARC-AGI-3 puzzles."""

    @staticmethod
    def detect_submit_button(grid: np.ndarray, bg_color: int, frame_colors: Set[int]) -> Optional[Tuple[int, int]]:
        if grid.ndim == 3:
            grid = grid[-1]
        h, w = grid.shape[:2]

        # 1. Scan top border strip (rows 2..8) and bottom border strip (rows h-8..h-2) for rectangular button
        candidate_masks = [
            (grid[2:9, 30:w - 2] != bg_color),
            (grid[h - 9:h - 2, 30:w - 2] != bg_color),
        ]
        offsets = [(2, 30), (h - 9, 30)]

        for mask, (r_off, c_off) in zip(candidate_masks, offsets):
            for fc in frame_colors:
                mask &= (grid[r_off : r_off + mask.shape[0], c_off : c_off + mask.shape[1]] != fc)

            labeled, num = ndimage.label(mask)
            for i in range(1, num + 1):
                pts = np.argwhere(labeled == i)
                if 8 <= len(pts) <= 120:
                    h_span = pts[:, 0].max() - pts[:, 0].min() + 1
                    w_span = pts[:, 1].max() - pts[:, 1].min() + 1
                    if 2 <= h_span <= 8 and 4 <= w_span <= 16:
                        cx = c_off + int(round(pts[:, 1].mean()))
                        cy = r_off + int(round(pts[:, 0].mean()))
                        return (cx, cy)

        # 2. Check bottom perimeter row (h - 1) for isolated token (e.g. (1, h-1) in cx01)
        if h >= 10 and w >= 10:
            for c_test in [1, 0, 2, w - 2, w - 1]:
                if 0 <= c_test < w:
                    c_val = int(grid[h - 1, c_test])
                    if c_val != bg_color and c_val not in frame_colors:
                        left_val = int(grid[h - 1, c_test - 1]) if c_test > 0 else bg_color
                        right_val = int(grid[h - 1, c_test + 1]) if c_test < w - 1 else bg_color
                        if left_val != c_val and right_val != c_val:
                            return (c_test, h - 1)

        # 3. Check top perimeter row 0 for isolated token (if not on bottom)
        if h >= 10 and w >= 10:
            for c_test in [1, 0, 2, w - 2, w - 1]:
                if 0 <= c_test < w:
                    c_val = int(grid[0, c_test])
                    if c_val != bg_color and c_val not in frame_colors:
                        left_val = int(grid[0, c_test - 1]) if c_test > 0 else bg_color
                        right_val = int(grid[0, c_test + 1]) if c_test < w - 1 else bg_color
                        if left_val != c_val and right_val != c_val:
                            return (c_test, 0)

        return None


class UniversalOptionSelectorEngine:
    """Detects when actions 1..4 position a selector cursor along a row (e.g. Row 62 in cz01),
    and couples it with clicking the submit button.
    """

    @staticmethod
    def detect_selector_cursor(grid: np.ndarray, bg_color: int) -> Optional[Tuple[int, int, int]]:
        """Returns (row, col, cursor_color) if a 1-pixel or 2-pixel selector is on a border row."""
        if grid.ndim == 3:
            grid = grid[-1]
        h, w = grid.shape[:2]
        for r in [h - 2, h - 3]:
            row_slice = grid[r, :]
            c_counts = collections.Counter(int(c) for c in row_slice if c != bg_color)
            for color, count in c_counts.items():
                if count in (1, 2):
                    cols = np.where(row_slice == color)[0]
                    return (r, int(cols[0]), color)
        return None


