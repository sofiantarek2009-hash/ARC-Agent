import logging
from typing import Any, Dict, List, Optional, Tuple
from project_override.config import AgentConfig
from project_override.memory.action_ledger import ActionLedger

logger = logging.getLogger(__name__)


class GroundedVerifierGate:
    """Fail-Closed Grounded Verification Gate (arc-skill / NVIDIA AVO).
    
    Validates candidate actions before execution against:
    1. Action availability (is action in available_actions?)
    2. Dead-signature filter (is this action known to cause instant game over or state death?)
    3. State governance (is this action an unproductive no-op or known oscillating cycle?)
    4. Plan divergence monitoring: if LLM-predicted outcomes repeatedly diverge from reality,
       the gate triggers fail-closed state, discarding hallucinated macro-queues.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.consecutive_violations: int = 0
        self.max_violations: int = self.config.verifier_max_violations
        self.total_verified_actions: int = 0
        self.total_rejected_actions: int = 0

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
        # 1. Action availability
        if action_id not in available_actions:
            self.total_rejected_actions += 1
            self.consecutive_violations += 1
            return False

        cx = int(payload.get("x", 0))
        cy = int(payload.get("y", 0))

        # 2. Dead signature check
        if self.config.enable_dead_signatures and ledger.is_dead(
            state_hash, action_id, cx, cy, level_idx
        ):
            logger.info(f"[VerifierGate] Action {action_id} blocked: matches recorded DEAD signature at hash {state_hash}.")
            self.total_rejected_actions += 1
            self.consecutive_violations += 1
            return False

        # 3. Action governance (no-ops & cycle suppression)
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

    def observe_transition(self, delta_omega: int) -> None:
        """If an action produced 0 diff when expecting motion, record violation."""
        if delta_omega == 0:
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = 0

    def is_fail_closed(self) -> bool:
        """Returns True if consecutive violations exceed threshold, requiring macro queue purge."""
        return self.consecutive_violations >= self.max_violations
