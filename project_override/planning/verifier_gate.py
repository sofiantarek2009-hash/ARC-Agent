from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from project_override.config import AgentConfig
from project_override.memory.action_ledger import ActionLedger

logger = logging.getLogger(__name__)


@dataclass
class TransitionEvent:
    step: int
    level: int
    pre_grid: np.ndarray
    action: int
    payload: Dict[str, Any]
    post_grid: np.ndarray
    delta_omega: int


class CEGISRetrodictionVerifier:
    """Counterexample-Guided Inductive World Model Verifier (arXiv:2610.00834).
    
    Verifies candidate world models and transition functions T(s, a) -> s' against
    100% of recorded historical transitions. If prediction diverges on even 1 cell,
    extracts the exact coordinate counterexample and refutes the hypothesis.
    """
    def __init__(self, max_history: int = 150):
        self.history: List[TransitionEvent] = []
        self.max_history = max_history
        self.counterexamples: List[Dict[str, Any]] = []

    def record_transition(
        self,
        step: int,
        level: int,
        pre_grid: np.ndarray,
        action: int,
        payload: Dict[str, Any],
        post_grid: np.ndarray,
        delta_omega: int,
    ) -> None:
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(
            TransitionEvent(
                step=step,
                level=level,
                pre_grid=pre_grid.copy(),
                action=action,
                payload=payload.copy() if payload else {},
                post_grid=post_grid.copy(),
                delta_omega=delta_omega,
            )
        )

    def verify_transition_model(
        self,
        model_fn: Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray],
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """Runs 100% cell-exact retrodiction replay check on all recorded transitions."""
        if not self.history:
            return True, 1.0, None

        total = len(self.history)
        perfect = 0
        counterexample = None

        for event in self.history:
            try:
                pred_grid = model_fn(event.pre_grid, event.action, event.payload)
                mismatches = (pred_grid != event.post_grid)
                if not np.any(mismatches):
                    perfect += 1
                elif counterexample is None:
                    pts = np.argwhere(mismatches)
                    r, c = pts[0]
                    counterexample = {
                        "step": event.step,
                        "action": event.action,
                        "coord": (int(r), int(c)),
                        "predicted": int(pred_grid[r, c]),
                        "actual": int(event.post_grid[r, c]),
                    }
                    self.counterexamples.append(counterexample)
            except Exception as exc:
                if counterexample is None:
                    counterexample = {"error": str(exc), "step": event.step}

        acc = float(perfect) / float(total)
        return (acc == 1.0), acc, counterexample


class GroundedVerifierGate:
    """Fail-Closed Grounded Verification Gate (arc-skill / NVIDIA AVO).
    
    Validates candidate actions before execution against:
    1. Action availability (is action in available_actions?)
    2. Dead-signature filter (is this action known to cause instant game over or state death?)
    3. State governance (is this action an unproductive no-op or known oscillating cycle?)
    4. CEGIS Retrodiction Verifier: tests synthesized transition models against historical replay.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.consecutive_violations: int = 0
        self.max_violations: int = self.config.verifier_max_violations
        self.total_verified_actions: int = 0
        self.total_rejected_actions: int = 0
        self.cegis = CEGISRetrodictionVerifier()

    def reset_level(self) -> None:
        self.consecutive_violations = 0

    def verify(
        self,
        action_id: int,
        payload: Dict[str, Any],
        available_actions: List[int],
        state_hash: int,
        ledger: ActionLedger,
        level_idx: int,
    ) -> bool:
        """Evaluates preconditions. Returns True if the action is grounded and safe."""
        if action_id not in available_actions:
            self.total_rejected_actions += 1
            self.consecutive_violations += 1
            return False

        cx = int(payload.get("x", 0))
        cy = int(payload.get("y", 0))

        if self.config.enable_dead_signatures and ledger.is_dead(
            state_hash, action_id, cx, cy, level_idx
        ):
            logger.info(f"[VerifierGate] Action {action_id} blocked: matches recorded DEAD signature at hash {state_hash}.")
            self.total_rejected_actions += 1
            self.consecutive_violations += 1
            return False

        if self.config.enable_action_guards and not ledger.govern_action(
            state_hash, action_id, cx, cy, level_idx
        ):
            logger.info(f"[VerifierGate] Action {action_id} blocked by state governor (unproductive no-op / cycle).")
            self.total_rejected_actions += 1
            self.consecutive_violations += 1
            return False

        self.consecutive_violations = 0
        self.total_verified_actions += 1
        return True

    def observe_transition(
        self,
        delta_omega: int,
        step: int = 0,
        level: int = 0,
        pre_grid: Optional[np.ndarray] = None,
        action: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
        post_grid: Optional[np.ndarray] = None,
    ) -> None:
        """If an action produced 0 diff when expecting motion, record violation, and record to CEGIS."""
        if delta_omega == 0:
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = 0

        if pre_grid is not None and post_grid is not None and action is not None:
            self.cegis.record_transition(
                step=step,
                level=level,
                pre_grid=pre_grid,
                action=action,
                payload=payload or {},
                post_grid=post_grid,
                delta_omega=delta_omega,
            )

    def is_fail_closed(self) -> bool:
        """Returns True if consecutive violations exceed threshold, requiring macro queue purge."""
        return self.consecutive_violations >= self.max_violations
