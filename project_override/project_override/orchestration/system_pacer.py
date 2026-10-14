"""Project Override: System Pacer & Compute Budget Watchdog
Manages the strict 9-hour (32,400s) Kaggle container execution limit across all games.
Implements dynamic rollout scaling, tail-fade prioritization, and hard per-level timeout gates.
"""

from __future__ import annotations

import gc
import time
from typing import Optional


class ComputeMode:
    LLM_HEAVY = "LLM_HEAVY"           # Full code synthesis + deep reflective rulebook
    FAST_PLANNER = "FAST_PLANNER"     # Lightweight MCTS / fast macro action plans
    REACTIVE_FASTPATH = "REACTIVE"    # Pure deterministic heuristic / 0-cost reactive


class SystemPacer:
    """Adaptive resource and time budget manager for Kaggle ARC-AGI-3 execution."""

    TOTAL_BUDGET_SECONDS = 32400.0  # 9 Hours global wall-clock limit
    TOTAL_GAMES = 135               # Kaggle hidden evaluation game count
    DEFAULT_LEVEL_TIMEOUT = 35.0    # Hard ceiling per level before forced fallback

    def __init__(
        self,
        total_budget: float = TOTAL_BUDGET_SECONDS,
        total_games: int = TOTAL_GAMES,
        max_level_timeout: float = DEFAULT_LEVEL_TIMEOUT,
    ) -> None:
        self.total_budget = float(total_budget)
        self.total_games = int(total_games)
        self.max_level_timeout = float(max_level_timeout)
        self.start_time = time.perf_counter()
        self.completed_games = 0
        self.current_level_start = time.perf_counter()
        self.mode = ComputeMode.LLM_HEAVY
        self.level_count_in_game = 0

    def start_level(self) -> None:
        """Mark the beginning of a new level interaction."""
        self.current_level_start = time.perf_counter()
        self.level_count_in_game += 1

    def end_game(self) -> None:
        """Mark the completion of a game."""
        self.completed_games += 1
        self.level_count_in_game = 0
        # Regular garbage collection between games to prevent OOM
        gc.collect()

    @property
    def elapsed_total(self) -> float:
        """Total seconds elapsed since agent launch."""
        return time.perf_counter() - self.start_time

    @property
    def remaining_total(self) -> float:
        """Remaining seconds in global 9-hour container budget."""
        return max(0.0, self.total_budget - self.elapsed_total)

    @property
    def remaining_games(self) -> int:
        """Number of games left to process."""
        return max(1, self.total_games - self.completed_games)

    @property
    def seconds_per_game(self) -> float:
        """Amortized seconds available per remaining game."""
        return self.remaining_total / self.remaining_games

    @property
    def is_level_timed_out(self) -> bool:
        """Check if current level has exceeded safe compute boundaries."""
        level_elapsed = time.perf_counter() - self.current_level_start
        # Scale level timeout if global budget is depleted
        allowed = min(self.max_level_timeout, max(5.0, self.seconds_per_game / 4.0))
        return level_elapsed >= allowed

    def get_level_time_budget(self) -> float:
        """Returns the maximum execution seconds allowed for the current level."""
        sec_game = self.seconds_per_game
        # Allocate roughly 1/5th to 1/7th of game budget per level
        return min(self.max_level_timeout, max(4.0, sec_game / 5.0))

    def update_pacing(self) -> str:
        """Evaluates remaining budget and modulates compute engine mode.
        Returns the active ComputeMode string.
        """
        sec_game = self.seconds_per_game
        rem_fraction = self.remaining_total / self.total_budget

        # Tail-fade logic (Franzen M2): If under 20% total budget, throttle heavy synthesis
        if rem_fraction < 0.20 or sec_game < 60.0:
            self.mode = ComputeMode.REACTIVE_FASTPATH
        elif sec_game < 120.0:
            self.mode = ComputeMode.FAST_PLANNER
        else:
            self.mode = ComputeMode.LLM_HEAVY

        return self.mode

    def get_mcts_rollout_limit(self) -> int:
        """Dynamically scales MCTS rollout limit based on compute mode."""
        if self.mode == ComputeMode.LLM_HEAVY:
            return 8000
        elif self.mode == ComputeMode.FAST_PLANNER:
            return 3000
        else:
            return 500
