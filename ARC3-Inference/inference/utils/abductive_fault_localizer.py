"""Archimedes-3 Abductive Fault Localizer (§16.1 & §18.1).

Five-Type Grounded Empirical Falsification and Root-Cause Diagnostic Engine:
- Type A (Unexpected No-Op): F_{t+1, interior} == F_{t, interior} when F_pred != F_t.
  Diagnoses unmodeled solid obstruction or boundary limit.
- Type B (Localized Entity Kinematics Error): Background/static match, but entity moved
  to different position (sliding k cells, bouncing, diagonal).
- Type C (Unmodeled Contact or Non-Local Side Effect): Entity moved as predicted, but
  distant cells mutated (switches opening doors, portals, flood-fill).
- Type D (Latent State / History-Dependent Aliasing): (F_t, a_t) identical to past turn,
  but produced divergent F_{t+1}. Proves hidden state h_t in Moore machine.
- Type E (Unpredicted Fatal Termination): Death occurred when playable predicted.
  Separates interior hazard collision from HUD timer budget exhaustion.

100% General, zero hardcoded game IDs, zero color assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class FaultDiagnosis:
    error_type: str  # "NONE", "TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D", "TYPE_E"
    root_cause: str
    diagnostic_message: str
    counterexample: dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""


class AbductiveFaultLocalizer:
    """Classifies prediction divergences and execution surprises into 5 root causes (§16.1)."""

    def __init__(self, abstain_val: int = -1) -> None:
        self.abstain_val = abstain_val

    def classify_surprise(
        self,
        before_grid: Sequence[Sequence[int]],
        action: str,
        actual_grid: Sequence[Sequence[int]],
        predicted_grid: Optional[Sequence[Sequence[int]]] = None,
        actual_status: str = "PLAYABLE",
        predicted_status: str = "PLAYABLE",
        history_transitions: Optional[Sequence[dict[str, Any]]] = None,
        hud_rows: Optional[Sequence[int]] = None,
        hud_cols: Optional[Sequence[int]] = None,
    ) -> FaultDiagnosis:
        """Evaluates prediction divergence and classifies root cause into Type A..E."""
        # 1. Type E Check: Fatal Termination
        if actual_status == "DEATH" and predicted_status != "DEATH":
            return self._diagnose_type_e_fatal(
                before_grid, action, actual_grid, hud_rows, hud_cols
            )

        # Compute cell-wise changes between before and actual (interior only)
        h = len(actual_grid)
        w = len(actual_grid[0]) if h > 0 else 0
        hud_r_set = set(hud_rows or [])
        hud_c_set = set(hud_cols or [])

        interior_diffs_before_actual: list[tuple[int, int, int, int]] = []
        for r in range(h):
            if r in hud_r_set:
                continue
            for c in range(w):
                if c in hud_c_set:
                    continue
                v_before = before_grid[r][c]
                v_actual = actual_grid[r][c]
                if v_before != v_actual:
                    interior_diffs_before_actual.append((r, c, v_before, v_actual))

        has_interior_change = len(interior_diffs_before_actual) > 0

        # Check if prediction was provided
        has_prediction = predicted_grid is not None
        predicts_change = False
        discrepancies: list[tuple[int, int, int, int]] = []

        if has_prediction and predicted_grid:
            for r in range(h):
                if r in hud_r_set:
                    continue
                for c in range(w):
                    if c in hud_c_set:
                        continue
                    v_pred = predicted_grid[r][c]
                    v_before = before_grid[r][c]
                    if v_pred != self.abstain_val and v_pred != v_before:
                        predicts_change = True
                    v_actual = actual_grid[r][c]
                    if v_pred != self.abstain_val and v_pred != v_actual:
                        discrepancies.append((r, c, v_pred, v_actual))

        # 2. Type A Check: Unexpected No-Op
        # If predicted change or non-fatal action attempted, but 0 interior cells mutated
        if not has_interior_change and (predicts_change or (has_prediction and len(discrepancies) > 0)):
            return FaultDiagnosis(
                error_type="TYPE_A",
                root_cause="UNEXPECTED_NO_OP",
                diagnostic_message=(
                    f"Action '{action}' produced 0 interior grid mutations despite predicting state change. "
                    f"Root cause: Unmodeled solid obstacle, boundary collision, or unmet movement precondition."
                ),
                counterexample={
                    "action": action,
                    "interior_cells_changed": 0,
                    "predicted_diff_count": len(discrepancies),
                },
                suggested_action="Record (state_hash, action) in NoOpLedger and prune from plan.",
            )

        # 3. Type D Check: Latent State / History-Dependent Aliasing
        # If (before_grid, action) previously appeared with different outcome
        if history_transitions:
            before_hash = self._grid_hash(before_grid, hud_r_set, hud_c_set)
            actual_hash = self._grid_hash(actual_grid, hud_r_set, hud_c_set)
            for prev in history_transitions:
                p_before = prev.get("before_grid")
                p_action = prev.get("action")
                p_after = prev.get("after_grid")
                if p_before and p_action == action:
                    p_before_h = self._grid_hash(p_before, hud_r_set, hud_c_set)
                    if p_before_h == before_hash:
                        p_after_h = self._grid_hash(p_after, hud_r_set, hud_c_set)
                        if p_after_h != actual_hash:
                            return FaultDiagnosis(
                                error_type="TYPE_D",
                                root_cause="LATENT_STATE_ALIASING",
                                diagnostic_message=(
                                    f"Observed non-injective transition for action '{action}'. "
                                    f"Identical visible frame at turn {prev.get('turn_idx', '?')} produced different outcome. "
                                    f"Root cause: Hidden internal state register h_t in Moore machine (toggle, charge, counter)."
                                ),
                                counterexample={
                                    "action": action,
                                    "divergent_turn": prev.get("turn_idx"),
                                    "divergent_action": p_action,
                                },
                                suggested_action="Augment world_model.py state with a latent register updated along path.",
                            )

        # If no discrepancies with prediction, return NONE
        if not discrepancies:
            return FaultDiagnosis(
                error_type="NONE",
                root_cause="CONCORDANT",
                diagnostic_message="Actual transition matches prediction on all non-abstained cells.",
            )

        # 4. Distinguish Type B vs Type C:
        # Check if the discrepancy is localized entity displacement (Type B)
        # vs distant / non-local mutation (Type C)
        diag = self._diagnose_type_b_or_c(
            before_grid, action, actual_grid, predicted_grid, discrepancies, hud_r_set, hud_c_set
        )
        return diag

    def _diagnose_type_e_fatal(
        self,
        before_grid: Sequence[Sequence[int]],
        action: str,
        actual_grid: Sequence[Sequence[int]],
        hud_rows: Optional[Sequence[int]],
        hud_cols: Optional[Sequence[int]],
    ) -> FaultDiagnosis:
        """Decomposes fatal reset into interior collision vs HUD timeout."""
        hud_r_set = set(hud_rows or [])
        hud_c_set = set(hud_cols or [])
        h = len(actual_grid)
        w = len(actual_grid[0]) if h > 0 else 0

        # Check if interior changes indicate collision
        colliding_cells = []
        for r in range(h):
            if r in hud_r_set:
                continue
            for c in range(w):
                if c in hud_c_set:
                    continue
                if before_grid[r][c] != actual_grid[r][c]:
                    colliding_cells.append((r, c, before_grid[r][c], actual_grid[r][c]))

        is_hud_exhaustion = len(colliding_cells) == 0

        if is_hud_exhaustion:
            return FaultDiagnosis(
                error_type="TYPE_E",
                root_cause="BUDGET_EXHAUSTION",
                diagnostic_message=(
                    f"Action '{action}' triggered GAME_OVER via HUD budget/timer depletion. "
                    f"No fatal interior spatial collision detected."
                ),
                counterexample={"action": action, "mechanism": "timer_depletion"},
                suggested_action="Optimize path length via A* to solve within remaining budget.",
            )
        else:
            return FaultDiagnosis(
                error_type="TYPE_E",
                root_cause="LETHAL_HAZARD_COLLISION",
                diagnostic_message=(
                    f"Action '{action}' triggered GAME_OVER due to lethal contact/hazard collision at interior cells. "
                    f"Recorded {len(colliding_cells)} mutated cells prior to reset."
                ),
                counterexample={
                    "action": action,
                    "colliding_cells": colliding_cells[:8],
                    "mechanism": "spatial_hazard",
                },
                suggested_action="Log state and lethal action into FatalLedger and add avoidance constraint to planner.",
            )

    def _diagnose_type_b_or_c(
        self,
        before_grid: Sequence[Sequence[int]],
        action: str,
        actual_grid: Sequence[Sequence[int]],
        predicted_grid: Sequence[Sequence[int]],
        discrepancies: Sequence[Tuple[int, int, int, int]],
        hud_r_set: set[int],
        hud_c_set: set[int],
    ) -> FaultDiagnosis:
        """Distinguishes localized kinematic error (Type B) from non-local contact side-effect (Type C)."""
        # Find where moving entity is in before_grid vs actual_grid
        # If discrepancies are clustered tightly around the moving entity -> Type B (Kinematics)
        # If discrepancies are split into mover + distant unconnected mutation -> Type C (Non-local)
        rows = [r for (r, c, _, _) in discrepancies]
        cols = [c for (r, c, _, _) in discrepancies]
        r_span = max(rows) - min(rows)
        c_span = max(cols) - min(cols)

        # Compute bounding diameter of discrepancy
        max_dist = max(r_span, c_span) if rows else 0

        # Also check distance from where before_grid mutated (action contact/origin)
        action_origin_coords = []
        for r in range(len(before_grid)):
            if r in hud_r_set:
                continue
            for c in range(len(before_grid[0])):
                if c in hud_c_set:
                    continue
                if before_grid[r][c] != actual_grid[r][c] and (r, c) not in [(dr, dc) for dr, dc, _, _ in discrepancies]:
                    action_origin_coords.append((r, c))

        dist_from_origin = 0
        if action_origin_coords:
            for dr, dc, _, _ in discrepancies:
                min_d = min(abs(dr - or_r) + abs(dc - or_c) for or_r, or_c in action_origin_coords)
                dist_from_origin = max(dist_from_origin, min_d)

        if max_dist <= 4 and dist_from_origin <= 3:
            # Localized discrepancy: Entity moved differently than modeled (e.g. multi-step slide, bounce)
            return FaultDiagnosis(
                error_type="TYPE_B",
                root_cause="LOCALIZED_KINEMATICS_ERROR",
                diagnostic_message=(
                    f"Entity kinematic prediction divergence for action '{action}'. "
                    f"Background matches, but entity displacement span ({r_span}x{c_span}) differs from prediction. "
                    f"Likely cause: sliding stride, inertia, reflection, or diagonal translation."
                ),
                counterexample={
                    "action": action,
                    "discrepancy_count": len(discrepancies),
                    "bbox_span": (r_span, c_span),
                    "sample_mismatches": discrepancies[:5],
                },
                suggested_action="Update world_model.py entity kinematic stride / raycast loop.",
            )
        else:
            # Discrepancies span distant regions or far from action origin: Non-local trigger (e.g., switch toggled gate at opposite end)
            return FaultDiagnosis(
                error_type="TYPE_C",
                root_cause="NON_LOCAL_CONTACT_EFFECT",
                diagnostic_message=(
                    f"Non-local side effect detected for action '{action}'. "
                    f"Discrepancies span {r_span} rows and {c_span} cols across the board (distance from origin={dist_from_origin}). "
                    f"Likely cause: contact at one location triggered mutation, door toggle, or teleporter at another."
                ),
                counterexample={
                    "action": action,
                    "discrepancy_count": len(discrepancies),
                    "span": (r_span, c_span),
                    "sample_mismatches": discrepancies[:6],
                },
                suggested_action="Model relational dependency between contact event and distant cell mutation.",
            )

    @staticmethod
    def _grid_hash(grid: Sequence[Sequence[int]], hud_r_set: set[int], hud_c_set: set[int]) -> int:
        """Hashes only interior grid cells, masking HUD."""
        vals = []
        for r, row in enumerate(grid):
            if r in hud_r_set:
                continue
            for c, val in enumerate(row):
                if c in hud_c_set:
                    continue
                vals.append(val)
        return hash(tuple(vals))
