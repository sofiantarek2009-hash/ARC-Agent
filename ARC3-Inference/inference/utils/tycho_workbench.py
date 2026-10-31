"""Archimedes-3 Tycho Programmatic World Model & Complete-History Retrodiction Workbench.

100% General, zero-heuristic transition testing.
Formalized from Sections 7, 10, and 13 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np


@dataclass
class TransitionRecord:
    """Lossless record of one interaction turn."""
    turn_idx: int
    before_grid: list[list[int]]
    action: str
    action_data: dict[str, Any]
    after_grid: list[list[int]]
    gameplay_changed: bool
    game_over: bool = False
    level_completed: bool = False


class ProgrammaticWorldModelWorkbench:
    """Auditable Programmatic World-Model Workbench with Complete-History Retrodiction.
    
    Allows testing candidate transition hypotheses against all recorded history
    turns in Python for 0 environment actions.
    """

    def __init__(self) -> None:
        self.transition_tape: list[TransitionRecord] = []
        self.registered_models: dict[str, Any] = {}

    def record_transition(
        self,
        turn_idx: int,
        before_grid: list[list[int]],
        action: str,
        action_data: dict[str, Any],
        after_grid: list[list[int]],
        gameplay_changed: bool,
        game_over: bool = False,
        level_completed: bool = False,
    ) -> None:
        """Appends a transition to the lossless episodic tape."""
        rec = TransitionRecord(
            turn_idx=turn_idx,
            before_grid=[list(r) for r in before_grid],
            action=str(action),
            action_data=dict(action_data or {}),
            after_grid=[list(r) for r in after_grid],
            gameplay_changed=bool(gameplay_changed),
            game_over=bool(game_over),
            level_completed=bool(level_completed),
        )
        self.transition_tape.append(rec)

    def retrodict_hypothesis(
        self,
        transition_fn: Callable[[list[list[int]], str, dict[str, Any]], list[list[int]]],
        timeout_seconds: float = 1.0,
    ) -> dict[str, Any]:
        """Tests a proposed Python transition function against the entire recorded history tape.
        
        Returns exact metrics:
        - total_tested: number of historical turns evaluated
        - matched_count: number of turns where predicted grid == actual grid
        - dynamic_recall: accuracy specifically on turns where cells changed
        - first_failure: step index and diff of first mismatch (if any)
        """
        if not self.transition_tape:
            return {
                "total_tested": 0,
                "matched_count": 0,
                "accuracy": 1.0,
                "dynamic_recall": 1.0,
                "first_failure": None,
            }

        start_time = time.monotonic()
        total_tested = 0
        matched_count = 0
        dynamic_total = 0
        dynamic_matched = 0
        first_failure = None

        for rec in self.transition_tape:
            if time.monotonic() - start_time > timeout_seconds:
                break

            total_tested += 1
            is_dynamic = (rec.before_grid != rec.after_grid)
            if is_dynamic:
                dynamic_total += 1

            try:
                pred_grid = transition_fn(
                    [list(r) for r in rec.before_grid],
                    rec.action,
                    rec.action_data,
                )
                if pred_grid == rec.after_grid:
                    matched_count += 1
                    if is_dynamic:
                        dynamic_matched += 1
                elif first_failure is None:
                    # Capture first counterexample
                    diff_coords = []
                    h = len(rec.after_grid)
                    w = len(rec.after_grid[0]) if h > 0 else 0
                    for r in range(h):
                        for c in range(w):
                            if r < len(pred_grid) and c < len(pred_grid[r]):
                                if pred_grid[r][c] != rec.after_grid[r][c]:
                                    diff_coords.append((r, c, pred_grid[r][c], rec.after_grid[r][c]))
                    first_failure = {
                        "turn": rec.turn_idx,
                        "action": rec.action,
                        "mismatches": diff_coords[:10],
                    }
            except Exception as e:
                if first_failure is None:
                    first_failure = {
                        "turn": rec.turn_idx,
                        "action": rec.action,
                        "exception": f"{type(e).__name__}: {e}",
                    }

        accuracy = matched_count / max(1, total_tested)
        dynamic_recall = dynamic_matched / max(1, dynamic_total)

        return {
            "total_tested": total_tested,
            "matched_count": matched_count,
            "accuracy": accuracy,
            "dynamic_recall": dynamic_recall,
            "first_failure": first_failure,
        }

    def reset_level(self) -> None:
        """Clears tape on level transition."""
        self.transition_tape.clear()


class EpistemicStatus:
    """5-valued epistemic output (§31.2 Innovation 1)."""
    DETERMINISTIC = "DETERMINISTIC"
    PROBABLE = "PROBABLE"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    ABSTAIN = "ABSTAIN"


def predict_epistemic_transition(
    workbench: ProgrammaticWorldModelWorkbench,
    current_grid: list[list[int]],
    action: str,
    action_data: dict[str, Any] | None = None,
    candidate_models: list[Callable[..., list[list[int]]]] | None = None,
    abstain_val: int = -1,
) -> tuple[list[list[int]] | None, str]:
    """Universal Transition Oracle evaluating candidate models (§31.2 Innovation 1).
    
    Returns:
        (predicted_grid, EpistemicStatus)
    """
    if not candidate_models:
        for rec in reversed(workbench.transition_tape):
            if rec.before_grid == current_grid and rec.action == action:
                return [list(r) for r in rec.after_grid], EpistemicStatus.DETERMINISTIC
        return None, EpistemicStatus.UNKNOWN

    predictions = []
    abstained = False
    for model in candidate_models:
        try:
            pred = model([list(r) for r in current_grid], action, action_data or {})
            has_abstain = any(any(c == abstain_val or c == 16 for c in row) for row in pred)
            if has_abstain:
                abstained = True
            predictions.append(pred)
        except Exception:
            continue

    if not predictions:
        return None, EpistemicStatus.UNKNOWN
    if abstained:
        return predictions[0], EpistemicStatus.ABSTAIN

    first = predictions[0]
    if all(p == first for p in predictions):
        return first, EpistemicStatus.DETERMINISTIC
    elif len(predictions) >= 3:
        for candidate in predictions:
            if predictions.count(candidate) > len(predictions) / 2:
                return candidate, EpistemicStatus.PROBABLE
        return predictions[0], EpistemicStatus.AMBIGUOUS
    else:
        return predictions[0], EpistemicStatus.AMBIGUOUS
