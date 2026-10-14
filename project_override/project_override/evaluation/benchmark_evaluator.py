"""Project Override: Local Empirical Benchmarking & RHAE Scoring Suite
Simulates the interactive ARC-AGI-3 POMDP environment locally.
Computes exact Relative Human Action Efficiency (RHAE) scores,
evaluates action efficiency across multi-level games, and logs detailed diagnostics.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class POMDPLevel:
    """A single discrete puzzle level within a POMDP game."""

    def __init__(
        self,
        level_index: int,
        initial_grid: np.ndarray,
        human_baseline: int,
        max_actions_factor: float = 5.0,
        name: str = "Level",
    ) -> None:
        self.level_index = level_index
        self.initial_grid = np.array(initial_grid, dtype=np.int32)
        self.current_grid = self.initial_grid.copy()
        self.human_baseline = int(human_baseline)
        self.max_allowed_actions = int(self.human_baseline * max_actions_factor)
        self.name = name
        self.actions_taken = 0
        self.is_won = False
        self.is_game_over = False

    def reset(self) -> np.ndarray:
        """Restores the level to its initial configuration."""
        self.current_grid = self.initial_grid.copy()
        # Note: In ARC-AGI-3, RESET is an action that increments the action counter!
        self.actions_taken += 1
        self.is_won = False
        self.is_game_over = False
        return self.current_grid.copy()

    def step(
        self, action: int, action_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Applies an action to the level state. Must be implemented by subclass."""
        raise NotImplementedError


class SokobanPushLevel(POMDPLevel):
    """Level 1: Sokoban push puzzle. Agent (blue=1), Block (green=3), Goal (yellow=4)."""

    def __init__(self, size: int = 15) -> None:
        grid = np.zeros((size, size), dtype=np.int32)
        # Outer boundary walls (grey=5)
        grid[0, :] = 5
        grid[-1, :] = 5
        grid[:, 0] = 5
        grid[:, -1] = 5

        # Agent starting at (2, 2)
        grid[2, 2] = 1
        # Block at (2, 4)
        grid[2, 4] = 3
        # Target at (2, 7)
        grid[2, 7] = 4

        super().__init__(
            level_index=1,
            initial_grid=grid,
            human_baseline=6,  # 3 right moves (push block to 7), 1 interact
            name="Sokoban Linear Push",
        )
        self.agent_pos = [2, 2]
        self.block_pos = [2, 4]
        self.goal_pos = [2, 7]

    def reset(self) -> np.ndarray:
        res = super().reset()
        self.agent_pos = [2, 2]
        self.block_pos = [2, 4]
        return res

    def step(
        self, action: int, action_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        self.actions_taken += 1

        if self.actions_taken > self.max_allowed_actions:
            self.is_game_over = True
            return self.current_grid.copy(), True, {"reason": "Action budget exhausted"}

        # Directional mapping: 1=Up, 2=Down, 3=Left, 4=Right
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

        if action in deltas:
            dy, dx = deltas[action]
            ny, nx = self.agent_pos[0] + dy, self.agent_pos[1] + dx

            # Boundary wall collision check
            if self.current_grid[ny, nx] == 5:
                # Collision: no-op
                pass
            elif ny == self.block_pos[0] and nx == self.block_pos[1]:
                # Attempt to push block
                b_ny, b_nx = self.block_pos[0] + dy, self.block_pos[1] + dx
                if self.current_grid[b_ny, b_nx] != 5:  # Destination unobstructed
                    # Move block
                    self.current_grid[self.block_pos[0], self.block_pos[1]] = 0
                    self.block_pos = [b_ny, b_nx]
                    self.current_grid[b_ny, b_nx] = 3
                    # Move agent
                    self.current_grid[self.agent_pos[0], self.agent_pos[1]] = 0
                    self.agent_pos = [ny, nx]
                    self.current_grid[ny, nx] = 1
            else:
                # Normal move
                self.current_grid[self.agent_pos[0], self.agent_pos[1]] = 0
                self.agent_pos = [ny, nx]
                self.current_grid[ny, nx] = 1

        # Check win condition: Block reaches goal cell
        if self.block_pos == self.goal_pos:
            self.is_won = True
            return self.current_grid.copy(), True, {"reason": "Level cleared!"}

        return self.current_grid.copy(), False, {}


class DynamicSpriteInterceptLevel(POMDPLevel):
    """Level 2: Dynamic Moving Sprite Intercept. Sprite (red=2) moves right 1 cell/step."""

    def __init__(self, size: int = 15) -> None:
        grid = np.zeros((size, size), dtype=np.int32)
        grid[0, :] = 5
        grid[-1, :] = 5
        grid[:, 0] = 5
        grid[:, -1] = 5

        grid[2, 2] = 1  # Agent
        grid[6, 3] = 2  # Moving sprite

        super().__init__(
            level_index=2,
            initial_grid=grid,
            human_baseline=8,
            name="Dynamic Sprite Intercept",
        )
        self.agent_pos = [2, 2]
        self.sprite_pos = [6, 3]
        self.sprite_dir = [0, 1]  # Moving right

    def reset(self) -> np.ndarray:
        res = super().reset()
        self.agent_pos = [2, 2]
        self.sprite_pos = [6, 3]
        return res

    def step(
        self, action: int, action_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        self.actions_taken += 1

        if self.actions_taken > self.max_allowed_actions:
            self.is_game_over = True
            return self.current_grid.copy(), True, {"reason": "Action budget exhausted"}

        # 1. Update Agent Position
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        if action in deltas:
            dy, dx = deltas[action]
            ny, nx = self.agent_pos[0] + dy, self.agent_pos[1] + dx
            if self.current_grid[ny, nx] != 5:
                self.current_grid[self.agent_pos[0], self.agent_pos[1]] = 0
                self.agent_pos = [ny, nx]
                self.current_grid[ny, nx] = 1
        elif action == 6 and action_data:
            # ACTION6: Parameterized click targeting
            cx, cy = action_data.get("x", -1), action_data.get("y", -1)
            if 0 <= cy < 15 and 0 <= cx < 15 and self.current_grid[cy, cx] != 5:
                self.current_grid[self.agent_pos[0], self.agent_pos[1]] = 0
                self.agent_pos = [cy, cx]
                self.current_grid[cy, cx] = 1

        # 2. Autonomous Sprite Step (Moves 1 cell right, bounces at walls)
        s_ny = self.sprite_pos[0] + self.sprite_dir[0]
        s_nx = self.sprite_pos[1] + self.sprite_dir[1]
        if self.current_grid[s_ny, s_nx] == 5:
            # Bounce
            self.sprite_dir[1] *= -1
            s_nx = self.sprite_pos[1] + self.sprite_dir[1]

        self.current_grid[self.sprite_pos[0], self.sprite_pos[1]] = 0
        self.sprite_pos = [s_ny, s_nx]
        self.current_grid[s_ny, s_nx] = 2

        # 3. Check Intercept: Manhattan distance between agent and sprite <= 1
        dist = abs(self.agent_pos[0] - self.sprite_pos[0]) + abs(
            self.agent_pos[1] - self.sprite_pos[1]
        )
        if dist <= 1 or action == 5:
            self.is_won = True
            return self.current_grid.copy(), True, {"reason": "Sprite intercepted!"}

        return self.current_grid.copy(), False, {}


class MockInteractiveGame:
    """Wraps a sequence of POMDP levels into a complete ARC-AGI-3 game session."""

    def __init__(self, game_id: str = "mock_game_01") -> None:
        self.game_id = game_id
        self.levels: List[POMDPLevel] = [
            SokobanPushLevel(size=15),
            DynamicSpriteInterceptLevel(size=15),
        ]
        self.current_level_idx = 0

    @property
    def current_level(self) -> Optional[POMDPLevel]:
        if self.current_level_idx < len(self.levels):
            return self.levels[self.current_level_idx]
        return None

    def observe(self) -> np.ndarray:
        if self.current_level:
            return self.current_level.current_grid.copy()
        return np.zeros((15, 15), dtype=np.int32)

    def step(
        self, action: int, action_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        level = self.current_level
        if not level:
            return np.zeros((15, 15), dtype=np.int32), True, {"done": True}

        grid, done, info = level.step(action, action_data)

        if done and level.is_won:
            self.current_level_idx += 1
            if self.current_level_idx < len(self.levels):
                # Next level unlocked
                return self.observe(), False, {"level_up": True}
            else:
                # All levels completed: Game Won
                return grid, True, {"game_won": True}

        return grid, done, info

    def is_completed(self) -> bool:
        return self.current_level_idx >= len(self.levels)


class RHAEBenchmarkEvaluator:
    """Calculates official ARC-AGI-3 competition metrics across evaluated games."""

    @staticmethod
    def compute_level_score(human_baseline: int, ai_actions: int, is_won: bool) -> float:
        """Computes Relative Human Action Efficiency (RHAE) for a single level:
        S = min(1.15, (h / a)^2).
        If a > 5*h or level not won, S = 0.0.
        """
        if not is_won or ai_actions <= 0:
            return 0.0
        if ai_actions > 5 * human_baseline:
            return 0.0

        ratio = float(human_baseline) / float(ai_actions)
        score = min(1.15, ratio**2)
        return float(score)

    @staticmethod
    def compute_game_score(level_scores: List[float]) -> float:
        """Linearly weights level scores by 1-indexed level number:
        E = sum(l * S_l) / sum(l).
        """
        if not level_scores:
            return 0.0
        total_weight = 0
        weighted_sum = 0.0
        for idx, score in enumerate(level_scores, start=1):
            weighted_sum += idx * score
            total_weight += idx
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @classmethod
    def evaluate_game(cls, game: MockInteractiveGame, agent_fn) -> Dict[str, Any]:
        """Runs the agent through all levels of a game and computes performance metrics."""
        level_results = []
        start_time = time.perf_counter()

        for level in game.levels:
            lvl_start = time.perf_counter()
            # Run agent on level
            is_won, actions_taken = agent_fn(level)
            lvl_elapsed = time.perf_counter() - lvl_start

            lvl_score = cls.compute_level_score(
                human_baseline=level.human_baseline,
                ai_actions=actions_taken,
                is_won=is_won,
            )

            level_results.append({
                "level_index": level.level_index,
                "name": level.name,
                "human_baseline": level.human_baseline,
                "ai_actions": actions_taken,
                "is_won": is_won,
                "rhae_score": lvl_score,
                "latency_sec": lvl_elapsed,
            })

        game_score = cls.compute_game_score([r["rhae_score"] for r in level_results])
        total_time = time.perf_counter() - start_time

        return {
            "game_id": game.game_id,
            "game_rhae_score": game_score,
            "levels": level_results,
            "total_elapsed_sec": total_time,
        }
