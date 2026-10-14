"""Project Override: High-Performance Obstacle-Aware A* Navigation Engine
Handles discrete grid pathfinding, collision hulls, multi-goal routing,
and Sokoban push kinematics for ARC-AGI-3.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


class NavAction(IntEnum):
    RESET = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5
    CLICK = 6
    UNDO = 7


ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    NavAction.UP: (-1, 0),
    NavAction.DOWN: (1, 0),
    NavAction.LEFT: (0, -1),
    NavAction.RIGHT: (0, 1),
}

DELTA_TO_ACTION: Dict[Tuple[int, int], int] = {
    (-1, 0): NavAction.UP,
    (1, 0): NavAction.DOWN,
    (0, -1): NavAction.LEFT,
    (0, 1): NavAction.RIGHT,
}


@dataclass
class CollisionHull:
    """Represents the multi-cell spatial footprint of an entity."""
    offsets: List[Tuple[int, int]]  # Relative (dr, dc) from anchor (min_r, min_c)
    height: int
    width: int
    mass: int

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> CollisionHull:
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return cls(offsets=[(0, 0)], height=1, width=1, mass=1)
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        offsets = [(int(r - min_r), int(c - min_c)) for r, c in coords]
        return cls(
            offsets=offsets,
            height=int(max_r - min_r + 1),
            width=int(max_c - min_c + 1),
            mass=len(offsets),
        )

    @classmethod
    def single_cell(cls) -> CollisionHull:
        return cls(offsets=[(0, 0)], height=1, width=1, mass=1)


@dataclass
class PathResult:
    """Outcome of A* pathfinding query."""
    success: bool
    actions: List[int]
    path_coords: List[Tuple[int, int]]
    cost: float
    reached_goal: bool
    closest_distance: float


class AStarNavigator:
    """Industrial-grade, zero-allocation A* pathfinder for 2D discrete grids."""

    def __init__(self, max_dim: int = 64) -> None:
        self.max_dim = max_dim
        # Pre-allocated arrays for fast state management
        self._g_score = np.full((max_dim, max_dim), np.inf, dtype=np.float32)
        self._visited = np.zeros((max_dim, max_dim), dtype=bool)
        self._parent_r = np.full((max_dim, max_dim), -1, dtype=np.int16)
        self._parent_c = np.full((max_dim, max_dim), -1, dtype=np.int16)
        self._parent_act = np.zeros((max_dim, max_dim), dtype=np.int8)

    def _reset_buffers(self, h: int, w: int) -> None:
        self._g_score[:h, :w].fill(np.inf)
        self._visited[:h, :w].fill(False)
        self._parent_r[:h, :w].fill(-1)
        self._parent_c[:h, :w].fill(-1)
        self._parent_act[:h, :w].fill(0)

    @staticmethod
    def detect_passability(
        grid: np.ndarray,
        player_coords: Optional[List[Tuple[int, int]]] = None,
        goal_coords: Optional[List[Tuple[int, int]]] = None,
        floor_color: Optional[int] = None,
        wall_colors: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """Constructs a boolean 2D passability grid where True = passable."""
        h, w = grid.shape
        passable = np.zeros((h, w), dtype=bool)

        if floor_color is not None:
            passable = (grid == floor_color)
        else:
            # Auto-detect background/floor color: most frequent color in grid
            colors, counts = np.unique(grid, return_counts=True)
            # Exclude border cells if needed
            bg_color = colors[np.argmax(counts)]
            passable = (grid == bg_color)

        if wall_colors:
            for wc in wall_colors:
                passable[grid == wc] = False

        # Ensure start and goals are passable
        if player_coords:
            for pr, pc in player_coords:
                if 0 <= pr < h and 0 <= pc < w:
                    passable[pr, pc] = True
        if goal_coords:
            for gr, gc in goal_coords:
                if 0 <= gr < h and 0 <= gc < w:
                    passable[gr, gc] = True

        return passable

    def find_path(
        self,
        passable_mask: np.ndarray,
        start: Tuple[int, int],
        goals: List[Tuple[int, int]],
        hull: Optional[CollisionHull] = None,
        turn_penalty: float = 0.05,
    ) -> PathResult:
        """Finds shortest obstacle-avoiding path from start to nearest goal."""
        h, w = passable_mask.shape
        sr, sc = start
        if not (0 <= sr < h and 0 <= sc < w):
            return PathResult(False, [], [], 0.0, False, float("inf"))

        goal_set = {(int(gr), int(gc)) for gr, gc in goals if 0 <= gr < h and 0 <= gc < w}
        if not goal_set:
            return PathResult(False, [], [], 0.0, False, float("inf"))

        if (sr, sc) in goal_set:
            return PathResult(True, [], [(sr, sc)], 0.0, True, 0.0)

        hull = hull or CollisionHull.single_cell()
        self._reset_buffers(h, w)

        # Min-heap: (f_score, g_score, r, c, last_action)
        heap: List[Tuple[float, float, int, int, int]] = []

        def heuristic(r: int, c: int) -> float:
            return min(abs(r - gr) + abs(c - gc) for gr, gc in goal_set)

        def is_hull_passable(r: int, c: int) -> bool:
            for dr, dc in hull.offsets:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not passable_mask[nr, nc]:
                    return False
            return True

        h_start = heuristic(sr, sc)
        self._g_score[sr, sc] = 0.0
        heapq.heappush(heap, (h_start, 0.0, sr, sc, 0))

        closest_r, closest_c = sr, sc
        min_dist_to_goal = h_start
        reached_target: Optional[Tuple[int, int]] = None

        while heap:
            f, g, cr, cc, last_act = heapq.heappop(heap)
            if self._visited[cr, cc]:
                continue
            self._visited[cr, cc] = True

            dist = heuristic(cr, cc)
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                closest_r, closest_c = cr, cc

            if (cr, cc) in goal_set:
                reached_target = (cr, cc)
                break

            for act, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue

                if not is_hull_passable(nr, nc):
                    # Allow stepping on goal even if hull isn't fully inside
                    if (nr, nc) not in goal_set:
                        continue

                move_cost = 1.0 + (turn_penalty if (last_act != 0 and last_act != act) else 0.0)
                new_g = g + move_cost

                if new_g < self._g_score[nr, nc]:
                    self._g_score[nr, nc] = new_g
                    self._parent_r[nr, nc] = cr
                    self._parent_c[nr, nc] = cc
                    self._parent_act[nr, nc] = act
                    new_f = new_g + heuristic(nr, nc)
                    heapq.heappush(heap, (new_f, new_g, nr, nc, act))

        # Reconstruct path
        target_r, target_c = reached_target if reached_target else (closest_r, closest_c)
        path_coords: List[Tuple[int, int]] = []
        actions: List[int] = []

        curr_r, curr_c = target_r, target_c
        while not (curr_r == sr and curr_c == sc):
            path_coords.append((curr_r, curr_c))
            act = int(self._parent_act[curr_r, curr_c])
            if act != 0:
                actions.append(act)
            pr = int(self._parent_r[curr_r, curr_c])
            pc = int(self._parent_c[curr_r, curr_c])
            if pr == -1 or pc == -1:
                break
            curr_r, curr_c = pr, pc

        path_coords.append((sr, sc))
        path_coords.reverse()
        actions.reverse()

        success = reached_target is not None
        cost = float(self._g_score[target_r, target_c]) if success else float("inf")

        return PathResult(
            success=success,
            actions=actions,
            path_coords=path_coords,
            cost=cost,
            reached_goal=success,
            closest_distance=min_dist_to_goal,
        )

    def plan_multi_target_tour(
        self,
        passable_mask: np.ndarray,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
        hull: Optional[CollisionHull] = None,
    ) -> List[int]:
        """Plans greedy nearest-neighbor collection tour across multiple targets."""
        remaining = list(targets)
        current_pos = start
        total_actions: List[int] = []

        while remaining:
            res = self.find_path(passable_mask, current_pos, remaining, hull)
            if not res.success or not res.actions:
                break
            total_actions.extend(res.actions)
            # Reached goal is end of path
            reached = res.path_coords[-1]
            if reached in remaining:
                remaining.remove(reached)
            else:
                # Find closest target to reached position
                closest = min(remaining, key=lambda t: abs(t[0] - reached[0]) + abs(t[1] - reached[1]))
                remaining.remove(closest)
            current_pos = reached

        return total_actions

    def plan_sokoban_push(
        self,
        passable_mask: np.ndarray,
        player_pos: Tuple[int, int],
        box_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ) -> Optional[List[int]]:
        """Plans player movement to get behind the box and push it toward goal."""
        h, w = passable_mask.shape
        pr, pc = player_pos
        br, bc = box_pos
        gr, gc = goal_pos

        # Desired push direction from box to goal
        dr_to_goal = gr - br
        dc_to_goal = gc - bc

        # Determine dominant push vector
        if abs(dc_to_goal) > 0:
            push_dir = (0, 1) if dc_to_goal > 0 else (0, -1)
        elif abs(dr_to_goal) > 0:
            push_dir = (1, 0) if dr_to_goal > 0 else (-1, 0)
        else:
            return []

        # Player must stand opposite of push direction
        stance_r = br - push_dir[0]
        stance_c = bc - push_dir[1]

        # Verify box destination is passable
        box_dest_r = br + push_dir[0]
        box_dest_c = bc + push_dir[1]
        if not (0 <= box_dest_r < h and 0 <= box_dest_c < w) or not passable_mask[box_dest_r, box_dest_c]:
            return None

        # Verify stance is within grid
        if not (0 <= stance_r < h and 0 <= stance_c < w):
            return None

        # Treat box as obstacle while navigating player to stance
        mask_with_box = passable_mask.copy()
        mask_with_box[br, bc] = False

        # Route player to stance
        nav_res = self.find_path(mask_with_box, (pr, pc), [(stance_r, stance_c)])
        if not nav_res.success and (pr, pc) != (stance_r, stance_c):
            return None

        push_action = DELTA_TO_ACTION.get(push_dir)
        if push_action is None:
            return None

        return nav_res.actions + [push_action]
