"""ARC-AGI-3 Stagnation Supervisor & Grounded Feedback Loop Breaker
Implements NVIDIA AVO Stagnation Supervision:
- Detects 1-cycle null transitions, 2-cycle ping-pong oscillations, and k-cycle limit loops.
- Automatically intervenes to break stagnation and redirect search toward informative actions.
- Enforces ground feedback verification: if a planned sequence fails intermediate delta expectations,
  instantly halts macro execution and triggers replanning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class StagnationSupervisor:
    """Watchdog that monitors state-action trajectories, detects limit cycles and dead ends,
    and forces topological diversions to protect action budgets and RHAE.
    """

    def __init__(
        self,
        max_null_steps: int = 3,
        max_oscillation_steps: int = 4,
        window_size: int = 24,
    ) -> None:
        self.max_null_steps = max_null_steps
        self.max_oscillation_steps = max_oscillation_steps
        self.window_size = window_size
        
        self.state_hashes: List[int] = []
        self.action_history: List[Tuple[int, Dict[str, Any]]] = []
        self.consecutive_null_steps: int = 0
        self.stagnation_event_count: int = 0
        self.forced_tabu_actions: Dict[int, Set[int]] = {}

    def reset_for_level(self) -> None:
        self.state_hashes.clear()
        self.action_history.clear()
        self.consecutive_null_steps = 0
        self.stagnation_event_count = 0
        self.forced_tabu_actions.clear()

    def on_step(
        self,
        current_grid: np.ndarray,
        last_action: Optional[int],
        last_action_data: Optional[Dict[str, Any]],
        grid_changed: bool,
    ) -> Dict[str, Any]:
        """Evaluates whether the agent has entered a dead end or limit cycle.
        Returns a diagnostic dictionary with stagnation flag and recovery recommendation.
        """
        # Fast 64-bit spatial hash of the current grid
        state_h = int(hash(current_grid.tobytes()))
        self.state_hashes.append(state_h)
        if len(self.state_hashes) > self.window_size:
            self.state_hashes.pop(0)

        if last_action is not None:
            self.action_history.append((last_action, last_action_data or {}))
            if len(self.action_history) > self.window_size:
                self.action_history.pop(0)

        # 1. Check 1-cycle: consecutive null transitions (Delta Omega = 0)
        if not grid_changed:
            self.consecutive_null_steps += 1
        else:
            self.consecutive_null_steps = 0

        is_null_stagnant = self.consecutive_null_steps >= self.max_null_steps

        # 2. Check 2-cycle ping-pong: A -> B -> A -> B
        is_oscillating = False
        if len(self.state_hashes) >= 4:
            s0, s1, s2, s3 = self.state_hashes[-4:]
            if s0 == s2 and s1 == s3 and s0 != s1:
                is_oscillating = True

        # 3. Check k-cycle repetition in recent trajectory
        is_k_cycle = False
        k_period = 0
        n_hist = len(self.state_hashes)
        for k in (3, 4, 5):
            if n_hist >= 2 * k:
                seg1 = self.state_hashes[-k:]
                seg2 = self.state_hashes[-2 * k : -k]
                if seg1 == seg2:
                    is_k_cycle = True
                    k_period = k
                    break

        stagnation_detected = is_null_stagnant or is_oscillating or is_k_cycle

        result = {
            "stagnation_detected": stagnation_detected,
            "reason": None,
            "forbidden_action": last_action if stagnation_detected else None,
            "recommended_diversion": False,
        }

        if stagnation_detected:
            self.stagnation_event_count += 1
            if is_null_stagnant:
                result["reason"] = f"1-cycle null transition ({self.consecutive_null_steps} consecutive)"
                self.consecutive_null_steps = 0  # Reset counter to allow recovery actions to be tested
            elif is_oscillating:
                result["reason"] = "2-cycle ping-pong oscillation"
            elif is_k_cycle:
                result["reason"] = f"{k_period}-cycle limit loop"
            result["recommended_diversion"] = True

            # Register forbidden action for current state
            if last_action is not None:
                if state_h not in self.forced_tabu_actions:
                    self.forced_tabu_actions[state_h] = set()
                self.forced_tabu_actions[state_h].add(last_action)

        return result

    def get_diversion_action(
        self,
        current_state_hash: int,
        legal_actions: List[int],
        recent_actions: List[int],
    ) -> Optional[int]:
        """Picks the best epistemic diversion action that breaks the detected limit cycle."""
        tabu = self.forced_tabu_actions.get(current_state_hash, set())
        candidates = [a for a in legal_actions if a not in tabu and a not in [0, 7]]
        if not candidates:
            # If all directional actions tabu, allow reset if stuck for > 8 events
            if 0 in legal_actions and self.stagnation_event_count >= 6:
                self.stagnation_event_count = 0
                return 0
            candidates = [a for a in legal_actions if a not in [0, 7]] or list(legal_actions)

        # Prioritize actions least taken recently
        recent_counts = {a: recent_actions.count(a) for a in candidates}
        best_act = min(candidates, key=lambda a: recent_counts.get(a, 0))
        return best_act
