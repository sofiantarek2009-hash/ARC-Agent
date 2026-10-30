"""Archimedes-3 Speed-Depth Bayesian Optimal Experiment Design (BOED) Acquisition & Frame-Change Classifier.

100% General, zero handcoded game heuristics.
Formalized from Section 9 (Speed-Depth Pareto Frontier) and Section 17.2 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence


class OnlineFrameChangeClassifier:
    """Fast online predictor P(Delta F_interior != 0 | s_t, a).
    
    Inspired by StochasticGoose and Franzen's no-op ledger.
    Tracks state-action transition changes to filter zero-information no-ops in O(1).
    """

    def __init__(self) -> None:
        # Map: (canonical_state_hash, action_str) -> bool (changed)
        self._action_effect_history: dict[tuple[int, str], bool] = {}
        # Prior empirical probability that an action changes gameplay
        self._action_change_counts: dict[str, int] = {}
        self._action_total_counts: dict[str, int] = {}

    def observe(self, state_hash: int, action: str, gameplay_changed: bool) -> None:
        """Records the outcome of an action in a state."""
        self._action_effect_history[(state_hash, action)] = bool(gameplay_changed)
        self._action_total_counts[action] = self._action_total_counts.get(action, 0) + 1
        if gameplay_changed:
            self._action_change_counts[action] = self._action_change_counts.get(action, 0) + 1

    def predict_change_probability(self, state_hash: int, action: str) -> float:
        """Returns empirical probability P(Delta F != 0 | s, a)."""
        key = (state_hash, action)
        if key in self._action_effect_history:
            return 1.0 if self._action_effect_history[key] else 0.0

        total = self._action_total_counts.get(action, 0)
        if total == 0:
            return 0.5  # Neutral uninformative prior
        changes = self._action_change_counts.get(action, 0)
        # Laplace-smoothed probability
        return (changes + 1.0) / (total + 2.0)

    def is_known_noop(self, state_hash: int, action: str) -> bool:
        """Returns True if this exact (state, action) is proven inert."""
        return self._action_effect_history.get((state_hash, action)) is False


class SpeedDepthBOEDAcquisition:
    """Speed-Depth Bayesian Optimal Experiment Design (BOED) Prober.
    
    Computes acquisition value A(a | b_t) = lambda_1 * Progress + lambda_2 * IG - lambda_3 * Risk - lambda_4 * ActionCost.
    Governs transition between Phase I/II active probing and Phase III shortest-path planning.
    """

    def __init__(self) -> None:
        self.frame_change_classifier = OnlineFrameChangeClassifier()

    def get_epistemic_phase(
        self,
        turn_idx: int,
        dynamic_recall: float,
        hypothesis_entropy: float = 0.0,
        expended_actions: int = 0,
        human_baseline: int = 20,
    ) -> str:
        """Determines current Epistemic Phase according to Section 9.2 table."""
        if turn_idx <= 3 or dynamic_recall < 0.30:
            return "PHASE_I_COLD_START"
        if expended_actions >= int(3.5 * human_baseline):
            return "PHASE_IV_BUDGET_CRITICAL"
        if dynamic_recall >= 0.90 and hypothesis_entropy <= 0.15:
            return "PHASE_III_EXPLOITATION_SHORTEST_PATH"
        return "PHASE_II_TARGETED_DISAMBIGUATION"

    def compute_acquisition(
        self,
        action: str,
        state_hash: int,
        epistemic_phase: str,
        untested_affordance: bool = False,
        is_known_deadly: bool = False,
        expended_actions: int = 0,
        human_baseline: int = 20,
    ) -> float:
        """Evaluates A(a | b_t) for candidate action."""
        p_change = self.frame_change_classifier.predict_change_probability(state_hash, action)
        
        # Hard penalty for known death or known inert no-op
        if is_known_deadly:
            return -1000.0
        if self.frame_change_classifier.is_known_noop(state_hash, action):
            return -500.0

        # Weights per phase
        if epistemic_phase == "PHASE_I_COLD_START":
            # High weight on novel information gain, zero on goal progress
            l1, l2, l3, l4 = 0.1, 10.0, 5.0, 1.0
        elif epistemic_phase == "PHASE_II_TARGETED_DISAMBIGUATION":
            l1, l2, l3, l4 = 3.0, 5.0, 5.0, 1.5
        elif epistemic_phase == "PHASE_III_EXPLOITATION_SHORTEST_PATH":
            l1, l2, l3, l4 = 10.0, 0.5, 10.0, 3.0
        else:  # PHASE_IV_BUDGET_CRITICAL
            l1, l2, l3, l4 = 20.0, 0.0, 2.0, 10.0

        # Information Gain term
        ig = p_change * (2.0 if untested_affordance else 0.5)
        # Goal Progress estimate
        goal_prog = 1.0 if epistemic_phase in ("PHASE_III_EXPLOITATION_SHORTEST_PATH", "PHASE_IV_BUDGET_CRITICAL") else 0.2
        # Action Cost based on quadratic RHAE sensitivity
        action_cost = (expended_actions / max(1, human_baseline))

        score = (l1 * goal_prog) + (l2 * ig) - (l4 * action_cost)
        return float(score)
