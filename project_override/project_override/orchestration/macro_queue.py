"""Project Override: Macro-Action Queue & Sahu Verification Delta Controller
Batches action plans into 1-4 step execution bursts while continuously verifying
predicted state hashes against live environment observations.
Halts immediately upon divergence, protecting against the quadratic RHAE penalty.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class MacroPlanStep:
    """A single step within a planned macro-action burst."""

    def __init__(
        self,
        action: int,
        action_data: Optional[Dict[str, Any]] = None,
        expected_player_pos: Optional[Tuple[int, int]] = None,
        expected_state_hash: Optional[int] = None,
        expected_grid: Optional[np.ndarray] = None,
        description: str = "",
    ) -> None:
        self.action = int(action)
        self.action_data = action_data or {}
        self.expected_player_pos = expected_player_pos
        self.expected_state_hash = expected_state_hash
        self.expected_grid = expected_grid
        self.description = description


class VerificationResult:
    """Outcome of a macro-action burst execution."""

    def __init__(
        self,
        completed_all: bool,
        actions_executed: int,
        diverged_at_step: Optional[int] = None,
        is_terminal: bool = False,
        is_noop: bool = False,
        last_grid: Optional[np.ndarray] = None,
    ) -> None:
        self.completed_all = completed_all
        self.actions_executed = actions_executed
        self.diverged_at_step = diverged_at_step
        self.is_terminal = is_terminal
        self.is_noop = is_noop
        self.last_grid = last_grid


class MacroActionQueue:
    """Manages burst execution with real-time verification delta checks."""

    def __init__(self, max_burst_size: int = 4) -> None:
        self.max_burst_size = max_burst_size
        self.plan: List[MacroPlanStep] = []

    def load_plan(self, steps: List[MacroPlanStep]) -> None:
        """Loads a sequence of planned actions (capped at max_burst_size)."""
        self.plan = list(steps[: self.max_burst_size])

    def add_step(
        self,
        action: int,
        action_data: Optional[Dict[str, Any]] = None,
        expected_player_pos: Optional[Tuple[int, int]] = None,
        expected_state_hash: Optional[int] = None,
        expected_grid: Optional[np.ndarray] = None,
        description: str = "",
    ) -> None:
        """Appends a single planned step to the queue."""
        self.plan.append(
            MacroPlanStep(
                action=action,
                action_data=action_data,
                expected_player_pos=expected_player_pos,
                expected_state_hash=expected_state_hash,
                expected_grid=expected_grid,
                description=description,
            )
        )

    def clear(self) -> None:
        """Wipes the queue."""
        self.plan.clear()

    @property
    def is_empty(self) -> bool:
        return len(self.plan) == 0

    def execute_burst(
        self,
        env_step_fn: Callable[[int, Optional[Dict[str, Any]]], Tuple[np.ndarray, bool, Dict[str, Any]]],
        hash_fn: Callable[[np.ndarray], int],
        on_dead_action_cb: Optional[Callable[[int, int], None]] = None,
        interior_mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> VerificationResult:
        """Executes queued macro actions one by one, checking live state against predicted state.

        Args:
            env_step_fn: Function taking (action, action_data) -> (next_grid, is_done, info).
            hash_fn: Function computing topological hash from grid.
            on_dead_action_cb: Callback (action, state_hash) called if action produced a no-op.
            interior_mask_fn: Optional mask function to exclude outer border timer bars.
        """
        executed_count = 0
        current_grid: Optional[np.ndarray] = None

        while self.plan:
            step = self.plan.pop(0)
            executed_count += 1

            prev_grid = current_grid

            # 1. Execute live environment step
            next_grid, is_done, info = env_step_fn(step.action, step.action_data)
            current_grid = next_grid

            # 2. Check for null state transition (No-op detection)
            if prev_grid is not None:
                # Compare interior gameplay cells if mask provided
                g1 = interior_mask_fn(prev_grid) if interior_mask_fn else prev_grid
                g2 = interior_mask_fn(next_grid) if interior_mask_fn else next_grid

                if np.array_equal(g1, g2):
                    # Null transition detected: action did not alter the board
                    if on_dead_action_cb and step.expected_state_hash is not None:
                        on_dead_action_cb(step.action, step.expected_state_hash)
                    self.clear()
                    return VerificationResult(
                        completed_all=False,
                        actions_executed=executed_count,
                        diverged_at_step=executed_count,
                        is_terminal=is_done,
                        is_noop=True,
                        last_grid=current_grid,
                    )

            # 3. Check for terminal state (Level cleared or game over)
            if is_done:
                self.clear()
                return VerificationResult(
                    completed_all=True,
                    actions_executed=executed_count,
                    is_terminal=True,
                    last_grid=current_grid,
                )

            # 4. Sahu Verification Delta: Verify live state against simulated expectation
            if step.expected_state_hash is not None:
                live_hash = hash_fn(next_grid)
                if live_hash != step.expected_state_hash:
                    # Simulation and live reality diverged!
                    # Halt burst immediately and purge remaining queued actions
                    self.clear()
                    return VerificationResult(
                        completed_all=False,
                        actions_executed=executed_count,
                        diverged_at_step=executed_count,
                        is_terminal=False,
                        last_grid=current_grid,
                    )

        return VerificationResult(
            completed_all=True,
            actions_executed=executed_count,
            is_terminal=False,
            last_grid=current_grid,
        )
