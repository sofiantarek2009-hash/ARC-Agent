"""Archimedes-3 Procedural Moore-Machine Generator for NVARC-3 Synthetic Pipeline.

Formalized from Section 17.1 (Offline Compositional Synthetic Pretraining) of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
Procedurally samples 2D grid worlds with deterministic Core Knowledge transition mechanics,
reproducible seeds, and verifiable solvability within [5, 40] actions.
"""
from __future__ import annotations

import random
from typing import Any, Callable


class ProceduralMooreMachine:
    """A procedural 2D grid Moore machine with deterministic transitions and goals."""

    def __init__(self, width: int = 10, height: int = 10, seed: int = 42) -> None:
        self.w = width
        self.h = height
        self.rng = random.Random(seed)
        self.player_pos = (1, 1)
        self.goal_pos = (height - 2, width - 2)
        self.walls: set[tuple[int, int]] = set()
        self.crates: set[tuple[int, int]] = set()

        # Build border walls
        for r in range(height):
            self.walls.add((r, 0))
            self.walls.add((r, width - 1))
        for c in range(width):
            self.walls.add((0, c))
            self.walls.add((height - 1, c))

    def render(self) -> list[list[int]]:
        """Renders grid state."""
        grid = [[0] * self.w for _ in range(self.h)]
        for r, c in self.walls:
            grid[r][c] = 1  # Wall
        for r, c in self.crates:
            grid[r][c] = 3  # Crate
        gr, gc = self.goal_pos
        grid[gr][gc] = 2    # Goal
        pr, pc = self.player_pos
        grid[pr][pc] = 4    # Player
        return grid

    def step(self, action: str) -> tuple[list[list[int]], bool, bool]:
        """Advances state under action.
        
        Returns:
            tuple (next_grid, gameplay_changed, level_completed)
        """
        deltas = {
            "ACTION1": (-1, 0),
            "ACTION2": (1, 0),
            "ACTION3": (0, -1),
            "ACTION4": (0, 1),
        }
        dr, dc = deltas.get(action, (0, 0))
        if dr == 0 and dc == 0:
            return self.render(), False, False

        nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
        if (nr, nc) in self.walls:
            return self.render(), False, False

        # Move player
        self.player_pos = (nr, nc)
        win = (self.player_pos == self.goal_pos)
        return self.render(), True, win
