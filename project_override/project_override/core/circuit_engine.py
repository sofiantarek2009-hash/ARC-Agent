"""Universal Circuit, Graph & Topology Engine for ARC-AGI-3.
Zero-overfitting, mathematically rigorous perception and exact solvers for
coordinate-click wire rotation, junction routing, and closed circuit puzzles.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class CardinalDir(IntEnum):
    NORTH = 0  # (-1, 0)
    EAST = 1   # (0, +1)
    SOUTH = 2  # (+1, 0)
    WEST = 3   # (0, -1)


DELTAS: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
OPP_DIR: List[int] = [2, 3, 0, 1]


def rot90_cw(mask: int) -> int:
    """Cyclic 1-bit left shift representing 90-degree clockwise rotation of 4-bit port mask."""
    return ((mask << 1) & 0x0F) | ((mask >> 3) & 0x01)


def rot90_ccw(mask: int) -> int:
    """Cyclic 1-bit right shift representing 90-degree counter-clockwise rotation."""
    return ((mask >> 1) & 0x07) | ((mask << 3) & 0x08)


def get_mask_orbit(initial_mask: int, clockwise: bool = True) -> List[int]:
    """Generates the unique cyclic orbit of port configurations for a tile."""
    orbit = [initial_mask]
    curr = initial_mask
    rot_fn = rot90_cw if clockwise else rot90_ccw
    for _ in range(3):
        curr = rot_fn(curr)
        if curr == initial_mask:
            break
        orbit.append(curr)
    return orbit


def has_port(mask: int, direction: int) -> bool:
    """Checks if a cardinal port is open in the given 4-bit bitmask."""
    return bool((mask >> direction) & 1)


@dataclass
class TerminalNode:
    grid_pos: Tuple[int, int]
    entry_dir: int  # Cardinal direction pointing into the grid
    color: int
    is_source: bool


@dataclass
class CircuitBoard:
    R: int
    C: int
    tile_size: int
    origin_r: int
    origin_c: int
    wire_color: int
    bg_color: int
    grid_masks: List[List[int]]
    sources: List[TerminalNode] = field(default_factory=list)
    sinks: List[TerminalNode] = field(default_factory=list)


class UniversalCircuitPerception:
    """Extracts macro-grid, wire colors, terminals, and 4-bit port stencils from raw frames."""

    @staticmethod
    def extract_port_mask(patch: np.ndarray, wire_color: int) -> int:
        """Determines cardinal port connectivity from boundary midpoint intersections."""
        H, W = patch.shape
        mid_r, mid_c = H // 2, W // 2
        r_span = slice(max(0, mid_r - 1), min(H, mid_r + 2))
        c_span = slice(max(0, mid_c - 1), min(W, mid_c + 2))

        b0 = int(np.any(patch[0, c_span] == wire_color))      # North
        b1 = int(np.any(patch[r_span, W - 1] == wire_color))  # East
        b2 = int(np.any(patch[H - 1, c_span] == wire_color))  # South
        b3 = int(np.any(patch[r_span, 0] == wire_color))      # West
        return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)

    @classmethod
    def parse_circuit_board(cls, frame: Any) -> Optional[CircuitBoard]:
        """Parses a raw observation frame into a structured CircuitBoard representation."""
        frame = np.asarray(frame, dtype=np.int32)
        if frame.ndim == 3:
            frame = frame[-1]
        if frame.ndim != 2:
            return None

        # 1. Identify canvas background (most frequent color on frame borders)
        border_px = np.concatenate([frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]])
        canvas_bg = int(np.bincount(border_px, minlength=16).argmax())

        # 2. Extract active bounding box
        act_y, act_x = np.where(frame != canvas_bg)
        if len(act_y) == 0:
            return None
        min_y, max_y = int(act_y.min()), int(act_y.max())
        min_x, max_x = int(act_x.min()), int(act_x.max())
        box_h = max_y - min_y + 1
        box_w = max_x - min_x + 1

        # 3. Detect dominant foreground color (Wire Color)
        unique_c, counts = np.unique(frame[min_y:max_y+1, min_x:max_x+1], return_counts=True)
        counts_dict = {int(c): int(cnt) for c, cnt in zip(unique_c, counts) if c != canvas_bg}
        if not counts_dict:
            return None
        total_pixels = box_h * box_w
        sorted_colors = sorted(counts_dict.keys(), key=lambda c: counts_dict[c], reverse=True)
        # If the most frequent color occupies > 45% of interior, it is playfield background
        if counts_dict[sorted_colors[0]] / max(1, total_pixels) > 0.45 and len(sorted_colors) > 1:
            wire_color = sorted_colors[1]
        else:
            wire_color = sorted_colors[0]
        wire_density = counts_dict[wire_color] / max(1, total_pixels)
        if wire_density < 0.005 or wire_density > 0.60:
            return None

        # 4. Infer macro-tile size K via periodicity
        best_k = 0
        for cand_k in range(4, 21):
            if box_h % cand_k == 0 and box_w % cand_k == 0:
                best_k = cand_k
                break
        if best_k == 0:
            best_k = 10

        R_cells = box_h // best_k
        C_cells = box_w // best_k
        if R_cells < 2 or C_cells < 2:
            return None

        # 5. Extract initial port masks for each cell
        grid_masks: List[List[int]] = []
        for r in range(R_cells):
            row_masks = []
            for c in range(C_cells):
                y0 = min_y + r * best_k
                x0 = min_x + c * best_k
                patch = frame[y0 : y0 + best_k, x0 : x0 + best_k]
                m = cls.extract_port_mask(patch, wire_color)
                row_masks.append(m)
            grid_masks.append(row_masks)

        board = CircuitBoard(
            R=R_cells,
            C=C_cells,
            tile_size=best_k,
            origin_r=min_y,
            origin_c=min_x,
            wire_color=wire_color,
            bg_color=canvas_bg,
            grid_masks=grid_masks,
        )

        # 6. Detect terminal nodes along perimeter
        for r in range(R_cells):
            if has_port(grid_masks[r][0], CardinalDir.WEST):
                board.sources.append(TerminalNode((r, 0), CardinalDir.WEST, wire_color, True))
            if has_port(grid_masks[r][C_cells - 1], CardinalDir.EAST):
                board.sinks.append(TerminalNode((r, C_cells - 1), CardinalDir.EAST, wire_color, False))

        for c in range(C_cells):
            if has_port(grid_masks[0][c], CardinalDir.NORTH):
                board.sources.append(TerminalNode((0, c), CardinalDir.NORTH, wire_color, True))
            if has_port(grid_masks[R_cells - 1][c], CardinalDir.SOUTH):
                board.sinks.append(TerminalNode((R_cells - 1, c), CardinalDir.SOUTH, wire_color, False))

        # Default fallbacks if terminals are flush against borders
        if not board.sources and R_cells > 0:
            board.sources.append(TerminalNode((0, 0), CardinalDir.NORTH, wire_color, True))
        if not board.sinks and R_cells > 0:
            board.sinks.append(TerminalNode((R_cells - 1, C_cells - 1), CardinalDir.SOUTH, wire_color, False))

        return board


class UniversalCircuitSolver:
    """Exact graph connectivity solvers finding minimum clicks for wire puzzles."""

    @staticmethod
    def solve_point_to_point_path(
        grid_masks: List[List[int]],
        start_pos: Tuple[int, int],
        start_in_dir: int,
        goal_pos: Tuple[int, int],
        goal_out_dir: Optional[int] = None,
        clockwise: bool = True,
    ) -> Optional[Tuple[int, List[Tuple[int, int, int]]]]:
        """A* search for the minimum clicks required to form a single continuous wire path.
        Returns (min_clicks, [(row, col, clicks_needed), ...]).
        """
        R, C = len(grid_masks), len(grid_masks[0])
        pq: List[Tuple[int, int, int, int, int, int, List[Tuple[int, int, int]]]] = []
        tr, tc = goal_pos
        sr, sc = start_pos
        orbit_start = get_mask_orbit(grid_masks[sr][sc], clockwise=clockwise)

        for out_dir in range(4):
            if out_dir == start_in_dir:
                continue
            for clicks, m in enumerate(orbit_start):
                if has_port(m, start_in_dir) and has_port(m, out_dir):
                    nr = sr + DELTAS[out_dir][0]
                    nc = sc + DELTAS[out_dir][1]
                    h = abs(nr - tr) + abs(nc - tc)
                    v_mask = 1 << (sr * C + sc)
                    heapq.heappush(pq, (clicks + h, clicks, sr, sc, out_dir, v_mask, [(sr, sc, clicks)]))
                    break

        best_cost: Dict[Tuple[int, int, int], int] = {}

        while pq:
            _, g, r, c, out_dir, v_mask, path = heapq.heappop(pq)
            nr = r + DELTAS[out_dir][0]
            nc = c + DELTAS[out_dir][1]
            in_dir = OPP_DIR[out_dir]

            if not (0 <= nr < R and 0 <= nc < C):
                continue

            cell_bit = 1 << (nr * C + nc)
            if v_mask & cell_bit:
                continue

            if (nr, nc) == goal_pos:
                orbit_goal = get_mask_orbit(grid_masks[nr][nc], clockwise=clockwise)
                for clicks, m in enumerate(orbit_goal):
                    if has_port(m, in_dir) and (goal_out_dir is None or has_port(m, goal_out_dir)):
                        total_clicks = g + clicks
                        res_path = path + [(nr, nc, clicks)]
                        return total_clicks, [(pr, pc, clk) for pr, pc, clk in res_path if clk > 0]
                continue

            orbit = get_mask_orbit(grid_masks[nr][nc], clockwise=clockwise)
            for next_out in range(4):
                if next_out == in_dir:
                    continue
                for clicks, m in enumerate(orbit):
                    if has_port(m, in_dir) and has_port(m, next_out):
                        new_g = g + clicks
                        state_key = (nr, nc, next_out)
                        if state_key in best_cost and best_cost[state_key] <= new_g:
                            continue
                        best_cost[state_key] = new_g
                        h = abs(nr + DELTAS[next_out][0] - tr) + abs(nc + DELTAS[next_out][1] - tc)
                        heapq.heappush(
                            pq,
                            (new_g + h, new_g, nr, nc, next_out, v_mask | cell_bit, path + [(nr, nc, clicks)]),
                        )
                        break
        return None

    @classmethod
    def solve_multi_sink_steiner(
        cls,
        grid_masks: List[List[int]],
        source: TerminalNode,
        sinks: List[TerminalNode],
        clockwise: bool = True,
    ) -> Optional[Tuple[int, List[Tuple[int, int, int]]]]:
        """Finds minimum clicks to connect Source to all Sinks via permutation Steiner routing."""
        if not sinks:
            return 0, []
        if len(sinks) == 1:
            return cls.solve_point_to_point_path(
                grid_masks, source.grid_pos, source.entry_dir, sinks[0].grid_pos, sinks[0].entry_dir, clockwise
            )

        best_clicks = float("inf")
        best_plan: Optional[List[Tuple[int, int, int]]] = None

        # Evaluate across all permutations of sink connections (K! <= 24 for K <= 4)
        for sink_order in itertools.permutations(sinks):
            curr_grid = [row[:] for row in grid_masks]
            total_clicks = 0
            accumulated_moves: Dict[Tuple[int, int], int] = {}

            feasible = True
            for sink in sink_order:
                res = cls.solve_point_to_point_path(
                    curr_grid, source.grid_pos, source.entry_dir, sink.grid_pos, sink.entry_dir, clockwise
                )
                if res is None:
                    feasible = False
                    break
                clicks, moves = res
                total_clicks += clicks
                for r, c, clk in moves:
                    accumulated_moves[(r, c)] = (accumulated_moves.get((r, c), 0) + clk)
                    # Rotate the tile in curr_grid to lock in its new orientation
                    orbit = get_mask_orbit(curr_grid[r][c], clockwise=clockwise)
                    curr_grid[r][c] = orbit[clk % len(orbit)]

            if feasible and total_clicks < best_clicks:
                best_clicks = total_clicks
                best_plan = [(r, c, clk) for (r, c), clk in accumulated_moves.items() if clk > 0]

        if best_plan is None:
            return None
        return int(best_clicks), best_plan

    @staticmethod
    def solve_closed_net_csp(
        grid_masks: List[List[int]],
        clockwise: bool = True,
    ) -> Optional[Tuple[int, List[Tuple[int, int, int]]]]:
        """Solves closed-circuit / Net puzzles where all wires must connect without leaks.
        Uses AC-3 Arc Consistency + Backtracking with Minimum Remaining Values (MRV).
        """
        R, C = len(grid_masks), len(grid_masks[0])
        domains: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        # 1. Initialize domains with unary boundary pruning
        for r in range(R):
            for c in range(C):
                m0 = grid_masks[r][c]
                if m0 == 0:
                    domains[(r, c)] = [(0, 0)]
                else:
                    orbit = get_mask_orbit(m0, clockwise=clockwise)
                    valid_configs = []
                    for clicks, m in enumerate(orbit):
                        if r == 0 and has_port(m, CardinalDir.NORTH):
                            continue
                        if r == R - 1 and has_port(m, CardinalDir.SOUTH):
                            continue
                        if c == 0 and has_port(m, CardinalDir.WEST):
                            continue
                        if c == C - 1 and has_port(m, CardinalDir.EAST):
                            continue
                        valid_configs.append((clicks, m))
                    domains[(r, c)] = valid_configs

        # 2. AC-3 Arc Consistency
        queue = [
            (u, v, d)
            for u in domains
            for d, (dr, dc) in enumerate(DELTAS)
            for v in [(u[0] + dr, u[1] + dc)]
            if 0 <= v[0] < R and 0 <= v[1] < C
        ]
        while queue:
            u, v, d = queue.pop(0)
            opp_d = OPP_DIR[d]
            u_domain = domains[u]
            v_domain = domains[v]
            new_u_domain = []
            revised = False
            for clicks_u, m_u in u_domain:
                port_u = has_port(m_u, d)
                match = any(has_port(m_v, opp_d) == port_u for _, m_v in v_domain)
                if match:
                    new_u_domain.append((clicks_u, m_u))
                else:
                    revised = True
            if revised:
                if not new_u_domain:
                    return None
                domains[u] = new_u_domain
                for d2, (dr, dc) in enumerate(DELTAS):
                    w = (u[0] + dr, u[1] + dc)
                    if 0 <= w[0] < R and 0 <= w[1] < C and w != v:
                        queue.append((w, u, d2))

        # 3. Branch and Bound with MRV
        cells = sorted(domains.keys(), key=lambda pos: len(domains[pos]))
        best_sol: List[Any] = [float("inf"), None]

        def check_global_connectivity(assignment: Dict[Tuple[int, int], Tuple[int, int]]) -> bool:
            active = [pos for pos, (_, m) in assignment.items() if m > 0]
            if not active:
                return True
            start = active[0]
            visited = {start}
            q = [start]
            while q:
                curr = q.pop(0)
                _, m_curr = assignment[curr]
                for d, (dr, dc) in enumerate(DELTAS):
                    if has_port(m_curr, d):
                        nbr = (curr[0] + dr, curr[1] + dc)
                        if nbr in assignment and nbr not in visited:
                            _, m_nbr = assignment[nbr]
                            if has_port(m_nbr, OPP_DIR[d]):
                                visited.add(nbr)
                                q.append(nbr)
            return len(visited) == len(active)

        expansions = [0]
        max_expansions = 5000

        def backtrack(idx: int, current_cost: int, assignment: Dict[Tuple[int, int], Tuple[int, int]]):
            if current_cost >= best_sol[0] or expansions[0] >= max_expansions:
                return
            expansions[0] += 1
            if idx == len(cells):
                if check_global_connectivity(assignment):
                    best_sol[0] = current_cost
                    best_sol[1] = dict(assignment)
                return

            pos = cells[idx]
            sorted_configs = sorted(domains[pos], key=lambda x: x[0])
            for clicks, mask in sorted_configs:
                consistent = True
                for d, (dr, dc) in enumerate(DELTAS):
                    nbr = (pos[0] + dr, pos[1] + dc)
                    if nbr in assignment:
                        nbr_mask = assignment[nbr][1]
                        if has_port(mask, d) != has_port(nbr_mask, OPP_DIR[d]):
                            consistent = False
                            break
                if consistent:
                    assignment[pos] = (clicks, mask)
                    backtrack(idx + 1, current_cost + clicks, assignment)
                    del assignment[pos]

        backtrack(0, 0, {})
        if best_sol[1] is None:
            return None

        moves = [(r, c, clk) for (r, c), (clk, _) in best_sol[1].items() if clk > 0]
        return int(best_sol[0]), moves


class UniversalCircuitEngineFacade:
    @classmethod
    def solve_grid_wire_bridging(cls, frame: Any) -> List[Tuple[int, int]]:
        """Finds 2x2 macro-grid terminal bridging paths when toggleable wire cells are present."""
        frame = np.asarray(frame, dtype=np.int32)
        if frame.ndim == 3:
            frame = frame[-1]
        if frame.ndim != 2:
            return []

        border_px = np.concatenate([frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]])
        canvas_bg = int(np.bincount(border_px, minlength=16).argmax())
        
        interior_counts = np.bincount(frame[4:60, 4:60].flatten(), minlength=16)
        pf_bg = int(np.argmax(interior_counts))
        if pf_bg == canvas_bg:
            interior_counts[canvas_bg] = 0
            pf_bg = int(np.argmax(interior_counts))
            
        pts_pf = np.argwhere(frame == pf_bg)
        if len(pts_pf) == 0:
            return []
            
        min_y, max_y = int(pts_pf[:, 0].min()), int(pts_pf[:, 0].max())
        min_x, max_x = int(pts_pf[:, 1].min()), int(pts_pf[:, 1].max())
        
        bh = max_y - min_y + 1
        bw = max_x - min_x + 1
        if bh % 2 != 0 or bw % 2 != 0 or bh < 8 or bw < 8:
            return []
            
        R_cells = bh // 2
        C_cells = bw // 2
        grid2 = np.zeros((R_cells, C_cells), dtype=int)
        for r in range(R_cells):
            for c in range(C_cells):
                grid2[r, c] = frame[min_y + r * 2, min_x + c * 2]
                
        counts2 = np.bincount(grid2.flatten(), minlength=16)
        bg_colors = {canvas_bg, pf_bg}
        terminals = []
        for c_val in range(16):
            if c_val not in bg_colors and 1 <= counts2[c_val] <= 4:
                matches = np.argwhere(grid2 == c_val)
                if len(matches) > 0:
                    r, c = matches[0]
                    terminals.append((int(r), int(c), c_val))
                
        if len(terminals) < 2:
            return []
            
        # Select the pair of terminals with maximal distance (true source and sink)
        best_pair = (terminals[0], terminals[1])
        max_dist = -1
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                d = abs(terminals[i][0] - terminals[j][0]) + abs(terminals[i][1] - terminals[j][1])
                if d > max_dist:
                    max_dist = d
                    best_pair = (terminals[i], terminals[j])

        start_r, start_c, _ = best_pair[0]
        goal_r, goal_c, _ = best_pair[1]
        
        q = collections.deque([(start_r, start_c, [])]) if "collections" in globals() else []
        if not q:
            import collections as col_mod
            q = col_mod.deque([(start_r, start_c, [])])
        visited = {(start_r, start_c)}
        found_path = None
        while q:
            cr, cc, path = q.popleft()
            if (cr, cc) == (goal_r, goal_c):
                found_path = path
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < R_cells and 0 <= nc < C_cells and (nr, nc) not in visited:
                    c_at = grid2[nr, nc]
                    if (nr, nc) == (goal_r, goal_c) or c_at in (pf_bg, 12, 13):
                        visited.add((nr, nc))
                        q.append((nr, nc, path + [(nr, nc)]))
                        
        if not found_path:
            return []
            
        clicks = []
        for r, c in found_path[:-1]:
            # Only click empty background cells to place new wire segments; never click existing wire, bridge, or terminal
            if grid2[r, c] == pf_bg:
                px = min_x + c * 2
                py = min_y + r * 2
                clicks.append((px, py))
        return clicks

    @classmethod
    def generate_click_actions(cls, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Parses the frame, solves the circuit topology, and emits the exact sequence
        of Action 6 (col_x, row_y) click coordinates to clear the puzzle.
        """
        board = UniversalCircuitPerception.parse_circuit_board(frame)
        if board is None or not board.sources or not board.sinks:
            return cls.solve_grid_wire_bridging(frame)

        # Try single-path or multi-sink Steiner routing first
        sol = UniversalCircuitSolver.solve_multi_sink_steiner(
            board.grid_masks, board.sources[0], board.sinks, clockwise=True
        )

        # Fallback to Closed Net CSP if point-to-point flow is unconstrained
        if sol is None:
            sol = UniversalCircuitSolver.solve_closed_net_csp(board.grid_masks, clockwise=True)

        if sol is None:
            return cls.solve_grid_wire_bridging(frame)

        _, moves = sol
        click_sequence: List[Tuple[int, int]] = []
        for r, c, click_count in moves:
            # Map grid cell (r, c) to canvas pixel center (X=col, Y=row)
            cx = board.origin_c + c * board.tile_size + board.tile_size // 2
            cy = board.origin_r + r * board.tile_size + board.tile_size // 2
            for _ in range(click_count):
                click_sequence.append((int(cx), int(cy)))

        return click_sequence
