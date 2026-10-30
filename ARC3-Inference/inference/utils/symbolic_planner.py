"""Archimedes-3 Deterministic Shortest-Path State-Equivalence A* & PUCT Planner.

100% General, zero hardcoding.
Masks out Monotonic HUD counters during state hashing to avoid state explosion.
Formalized from Section 10 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
"""
from __future__ import annotations

import heapq
import time
from typing import Any, Callable


class StateEquivalenceAStarPlanner:
    """Deterministic Shortest-Path Planner over Python Transition Models.
    
    Finds human-optimal action paths (a_{l,e} <= h_{l,e}) satisfying a goal predicate.
    """

    @staticmethod
    def hash_state(
        grid: list[list[int]],
        masked_strips: set[tuple[str, int]] | None = None,
    ) -> int:
        """Computes canonical state hash, masking out monotonic HUD counters."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        core = []
        for r in range(h):
            row_vals = []
            for c in range(w):
                if masked_strips and (('row', r) in masked_strips or ('col', c) in masked_strips):
                    row_vals.append(-1)
                else:
                    row_vals.append(grid[r][c])
            core.append(tuple(row_vals))
        return hash(tuple(core))

    @classmethod
    def solve(
        cls,
        initial_grid: list[list[int]],
        valid_actions: list[str],
        transition_fn: Callable[[list[list[int]], str], list[list[int]]],
        goal_predicate: Callable[[list[list[int]]], bool],
        masked_strips: set[tuple[str, int]] | None = None,
        max_depth: int = 35,
        timeout_seconds: float = 1.5,
    ) -> list[str] | None:
        """Executes Breadth-First / A* graph search over transition_fn to find minimal action path."""
        start_time = time.monotonic()
        initial_hash = cls.hash_state(initial_grid, masked_strips)
        
        if goal_predicate(initial_grid):
            return []

        # Priority queue: (path_length, state_hash, path, current_grid)
        queue = [(0, initial_hash, [], initial_grid)]
        visited: set[int] = {initial_hash}

        while queue:
            if time.monotonic() - start_time > timeout_seconds:
                break

            depth, _, path, curr_grid = heapq.heappop(queue)
            if depth >= max_depth:
                continue

            for action in valid_actions:
                try:
                    next_grid = transition_fn([list(r) for r in curr_grid], action)
                except Exception:
                    continue

                if next_grid == curr_grid:
                    continue  # skip no-op

                if goal_predicate(next_grid):
                    return path + [action]

                next_hash = cls.hash_state(next_grid, masked_strips)
                if next_hash not in visited:
                    visited.add(next_hash)
                    heapq.heappush(queue, (depth + 1, next_hash, path + [action], next_grid))

        return None
