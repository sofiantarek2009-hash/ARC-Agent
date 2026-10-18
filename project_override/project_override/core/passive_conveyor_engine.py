"""Universal Passive Vector Field & Conveyor Velocity Engine
Handles dynamic environments with passive surface currents:
- Conveyor belts
- Ice sliding & momentum glides
- River currents & wind fields

Calculates the residual velocity:
    v_passive = Delta_p - v_action
and integrates forward simulation into A* pathfinding.

Strictly general, zero-hardcoding, zero game-specific constants.
"""

from __future__ import annotations

import collections
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class PassiveVelocityField:
    """Maintains an empirical vector field of passive surface velocities."""
    tile_color_velocity: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    cell_velocity: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    has_conveyors: bool = False

    def has_conveyors_active(self) -> bool:
        return self.has_conveyors or len(self.cell_velocity) > 0

    def observe_transition(
        self,
        p_before: Tuple[int, int],
        p_after: Tuple[int, int],
        action_vector: Tuple[int, int],
        board: Optional[np.ndarray],
        walls: Set[Tuple[int, int]],
        gh: int = 10,
        gw: int = 10,
    ) -> None:
        """Induces passive velocity from observed displacement residual."""
        dr_act, dc_act = action_vector
        expected_p = (p_before[0] + dr_act, p_before[1] + dc_act)

        # Actual displacement
        dr_act_real = p_after[0] - p_before[0]
        dc_act_real = p_after[1] - p_before[1]

        # If agent bumped a wall and didn't move at all
        if (dr_act_real, dc_act_real) == (0, 0):
            return

        # Residual drift not explained by player action
        dr_surf = dr_act_real - dr_act
        dc_surf = dc_act_real - dc_act

        if abs(dr_surf) > 0 or abs(dc_surf) > 0:
            self.has_conveyors = True
            self.cell_velocity[expected_p] = (dr_surf, dc_surf)

            if board is not None and 0 <= expected_p[0] < board.shape[0] and 0 <= expected_p[1] < board.shape[1]:
                tile_c = int(board[expected_p[0], expected_p[1]])
                self.tile_color_velocity[tile_c] = (dr_surf, dc_surf)

    def get_velocity_at(self, pos: Tuple[int, int], board: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """Returns passive velocity vector (dr, dc) at given position."""
        if pos in self.cell_velocity:
            return self.cell_velocity[pos]
        if board is not None and 0 <= pos[0] < board.shape[0] and 0 <= pos[1] < board.shape[1]:
            tile_c = int(board[pos[0], pos[1]])
            if tile_c in self.tile_color_velocity:
                return self.tile_color_velocity[tile_c]
        return (0, 0)

    def simulate_step(
        self,
        pos: Tuple[int, int],
        action_vector: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        gh: int,
        gw: int,
        board: Optional[np.ndarray] = None,
        max_glide: int = 10,
        goals: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """Simulates one action step with passive velocity field gliding."""
        dr_act, dc_act = action_vector
        np_pos = (pos[0] + dr_act, pos[1] + dc_act)

        # Check boundary & wall collision for agent step
        if np_pos in walls or not (0 <= np_pos[0] < gh and 0 <= np_pos[1] < gw):
            np_pos = pos  # Player didn't move

        # Now apply passive surface velocity drift
        curr = np_pos
        for _ in range(max_glide):
            if goals and curr in goals:
                break
            v_surf = self.get_velocity_at(curr, board)
            if v_surf == (0, 0):
                break
            nxt = (curr[0] + v_surf[0], curr[1] + v_surf[1])
            if nxt in walls or not (0 <= nxt[0] < gh and 0 <= nxt[1] < gw):
                break
            curr = nxt

        return curr


def solve_conveyor_astar(
    p_start: Tuple[int, int],
    goals: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    gh: int,
    gw: int,
    field: PassiveVelocityField,
    displacement_vectors: Optional[Dict[int, Tuple[int, int]]] = None,
    board: Optional[np.ndarray] = None,
    max_nodes: int = 25000,
) -> Optional[List[int]]:
    """Coupled A* search navigating passive conveyor velocity fields."""
    v = displacement_vectors or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def heuristic(pos: Tuple[int, int]) -> float:
        if not goals:
            return 0.0
        return float(min(abs(pos[0] - gr) + abs(pos[1] - gc) for gr, gc in goals))

    init_state = p_start
    q = [(heuristic(init_state), 0.0, init_state, [])]
    visited: Dict[Tuple[int, int], float] = {init_state: 0.0}
    nodes_expanded = 0

    while q and nodes_expanded < max_nodes:
        _, cost, cur_p, path = heapq.heappop(q)
        nodes_expanded += 1

        if cur_p in goals:
            return path

        if cost > visited.get(cur_p, 1e9):
            continue

        for act, vec in v.items():
            next_p = field.simulate_step(cur_p, vec, walls, gh, gw, board=board, goals=goals)
            ncost = cost + 1.0

            if ncost < visited.get(next_p, 1e9):
                visited[next_p] = ncost
                h = heuristic(next_p)
                heapq.heappush(q, (ncost + h, ncost, next_p, path + [act]))

    return None
