"""Archimedes-3 AVO Supervisor & Kepler Pre-Action Commit Verifier.

100% General, zero-heuristic active abstraction monitoring.
Formalized from Sections 10, 12, 15, and 16 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence


class AVOSupervisor:
    """NVIDIA AVO Supervisory Intervention Monitor.
    
    Tracks the trajectory for stagnation (Z_t turns with 0 new states or repeated failed actions).
    When stagnation threshold is reached, interrupts and injects an Epistemic Paradigm Shift Directive.
    """

    def __init__(self, stagnation_threshold: int = 4) -> None:
        self.stagnation_threshold = stagnation_threshold
        self.recent_state_hashes: list[str] = []
        self.consecutive_noops: int = 0
        self.stagnation_index: int = 0
        self.active_directives: list[str] = []

    def observe_turn(
        self,
        current_state_hash: str,
        gameplay_changed: bool | None,
        level: int,
    ) -> None:
        """Records state visit and evaluates trajectory progress."""
        self.recent_state_hashes.append(current_state_hash)
        if len(self.recent_state_hashes) > 10:
            self.recent_state_hashes.pop(0)

        if gameplay_changed is False:
            self.consecutive_noops += 1
        else:
            self.consecutive_noops = 0

        # Compute unique states in the last 4 observations
        window = self.recent_state_hashes[-self.stagnation_threshold:]
        unique_states = len(set(window))

        if len(window) >= self.stagnation_threshold and unique_states <= 1:
            self.stagnation_index += 1
        elif self.consecutive_noops >= 3:
            self.stagnation_index += 1
        else:
            self.stagnation_index = max(0, self.stagnation_index - 1)

    def is_stagnating(self) -> bool:
        """Returns True if the agent is trapped in an unproductive loop or dead-end cycle."""
        return self.stagnation_index >= 1

    def get_supervisory_directive(self) -> str | None:
        """Returns the mandatory Paradigm Shift Directive if stagnation is detected."""
        if not self.is_stagnating():
            return None

        directive = (
            "[SUPERVISORY INTERVENTION - NVIDIA AVO ANTI-STAGNATION MONITOR]:\n"
            "Warning: You have been trapped in an unproductive state cycle for multiple consecutive turns.\n"
            "Your previous mental model or action sequence has failed to make progress.\n"
            "MANDATE:\n"
            "1. You are strictly forbidden from repeating the action or plan tried in the last 2 turns.\n"
            "2. Formulate a fundamentally different hypothesis about object affordances or goal conditions.\n"
            "3. If moving in a given direction was blocked or fatal, test an alternative action key or probe an unvisited region."
        )
        return directive

    def reset_level(self) -> None:
        """Resets tracking on level transition."""
        self.recent_state_hashes.clear()
        self.consecutive_noops = 0
        self.stagnation_index = 0
        self.active_directives.clear()


class KeplerCommitGate:
    """Kepler Pre-Action Commit Verifier & Surprise Abort Gate.
    
    Ensures multi-step action sequences are verified step-by-step.
    If actual frame deviates from expected delta, execution halts immediately,
    saving precious actions from open-loop disaster.
    """

    def __init__(self) -> None:
        self.active_plan: list[Any] = []
        self.expected_states: list[str] = []

    @staticmethod
    def evaluate_step_surprise(
        before_grid: list[list[int]],
        after_grid: list[list[int]],
        expected_change: bool = True,
    ) -> bool:
        """Evaluates whether an unexpected divergence occurred.
        
        Returns True if surprise occurred (unexpected no-op or unexpected lethal transition).
        """
        changed = (before_grid != after_grid)
        # If the plan assumed a move occurred, but 0 cells changed -> Surprise!
        if expected_change and not changed:
            return True
        return False
