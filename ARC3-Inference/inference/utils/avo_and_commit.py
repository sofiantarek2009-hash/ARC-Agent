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


from inference.utils.abductive_fault_localizer import AbductiveFaultLocalizer, FaultDiagnosis


class KeplerCommitGate:
    """Kepler Pre-Action Commit Verifier & Surprise Abort Gate (§10.3 & §16.1).
    
    Ensures multi-step action sequences are verified step-by-step.
    If actual frame deviates from expected delta or predicted render, execution
    halts immediately, saving precious actions from open-loop disaster, and passes
    the counterexample to the Abductive Fault Localizer.
    """

    def __init__(self, abstain_val: int = -1) -> None:
        self.abstain_val = abstain_val
        self.localizer = AbductiveFaultLocalizer(abstain_val=abstain_val)
        self.active_plan: list[Any] = []
        self.expected_frames: list[Any] = []
        self.expected_outcomes: list[str] = []
        self.last_fault: FaultDiagnosis | None = None

    def commit_plan(
        self,
        action_seq: Sequence[Any],
        expected_frames: Sequence[Any] | None = None,
        expected_outcomes: Sequence[str] | None = None,
    ) -> None:
        """Registers a candidate plan with its step-by-step predicted outcomes."""
        self.active_plan = list(action_seq)
        self.expected_frames = list(expected_frames or [])
        self.expected_outcomes = list(expected_outcomes or [])
        self.last_fault = None

    def abort_plan(self) -> None:
        """Clears remaining queued plan actions upon surprise divergence."""
        self.active_plan.clear()
        self.expected_frames.clear()
        self.expected_outcomes.clear()

    def evaluate_step(
        self,
        before_grid: Sequence[Sequence[int]],
        action: str,
        actual_grid: Sequence[Sequence[int]],
        step_idx: int = 0,
        actual_status: str = "PLAYABLE",
        history_transitions: Sequence[dict[str, Any]] | None = None,
        hud_rows: Sequence[int] | None = None,
        hud_cols: Sequence[int] | None = None,
    ) -> tuple[bool, FaultDiagnosis | None]:
        """Evaluates whether surprise occurred at step_idx of committed plan.
        
        Returns:
            (is_surprise, fault_diagnosis)
        """
        predicted_frame = None
        if step_idx < len(self.expected_frames):
            predicted_frame = self.expected_frames[step_idx]

        predicted_outcome = "PLAYABLE"
        if step_idx < len(self.expected_outcomes):
            predicted_outcome = self.expected_outcomes[step_idx]

        # Surprise check:
        # 1. Terminal outcome mismatch
        if actual_status != predicted_outcome:
            diag = self.localizer.classify_surprise(
                before_grid=before_grid,
                action=action,
                actual_grid=actual_grid,
                predicted_grid=predicted_frame,
                actual_status=actual_status,
                predicted_status=predicted_outcome,
                history_transitions=history_transitions,
                hud_rows=hud_rows,
                hud_cols=hud_cols,
            )
            self.last_fault = diag
            self.abort_plan()
            return True, diag

        # 2. Predicted frame cell-level mismatch (on non-abstained interior cells)
        if predicted_frame is not None:
            hud_r_set = set(hud_rows or [])
            hud_c_set = set(hud_cols or [])
            h = len(actual_grid)
            w = len(actual_grid[0]) if h > 0 else 0
            mismatch_detected = False

            for r in range(h):
                if r in hud_r_set:
                    continue
                for c in range(w):
                    if c in hud_c_set:
                        continue
                    p_val = predicted_frame[r][c]
                    if p_val != self.abstain_val and p_val != actual_grid[r][c]:
                        mismatch_detected = True
                        break
                if mismatch_detected:
                    break

            if mismatch_detected:
                diag = self.localizer.classify_surprise(
                    before_grid=before_grid,
                    action=action,
                    actual_grid=actual_grid,
                    predicted_grid=predicted_frame,
                    actual_status=actual_status,
                    predicted_status=predicted_outcome,
                    history_transitions=history_transitions,
                    hud_rows=hud_rows,
                    hud_cols=hud_cols,
                )
                self.last_fault = diag
                self.abort_plan()
                return True, diag

        # 3. Simple fallback check if no predicted frame: check if action was a no-op
        if predicted_frame is None and before_grid == actual_grid:
            diag = FaultDiagnosis(
                error_type="TYPE_A",
                root_cause="UNEXPECTED_NO_OP",
                diagnostic_message=f"Action '{action}' produced zero grid change.",
                suggested_action="Avoid repeating action in this state.",
            )
            self.last_fault = diag
            self.abort_plan()
            return True, diag

        return False, None

    @staticmethod
    def evaluate_step_surprise(
        before_grid: list[list[int]],
        after_grid: list[list[int]],
        expected_change: bool = True,
    ) -> bool:
        """Evaluates whether an unexpected divergence occurred (legacy signature)."""
        changed = (before_grid != after_grid)
        if expected_change and not changed:
            return True
        return False
