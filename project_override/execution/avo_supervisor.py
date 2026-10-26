from collections import deque
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from project_override.config import AgentConfig

logger = logging.getLogger(__name__)


class AVOSupervisor:
    """NVIDIA Agentic Variation Operators (AVO) Trajectory Supervisor (arXiv:2603.24517).
    
    Monitors trajectory health and stagnation index:
        sigma_t = |unique(H_t)| / |H_t|
    over a sliding window of recent state hashes.
    
    Escalation Tiers:
      Tier 0: Healthy exploration / progress.
      Tier 1: Stagnation (sigma_t < threshold) or >25% level budget used -> Strategy Audit Prompt.
      Tier 2: Deadlock / loop trap / consecutive stagnant windows -> Multi-step ACTION7 (Undo) chain.
      Tier 3: Near level action limit -> Emergency exit / reset discipline.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.window_size = self.config.stagnation_window
        self.hash_window: deque = deque(maxlen=self.window_size)
        self.action_window: deque = deque(maxlen=self.window_size)
        self.escalation_tier: int = 0
        self.consecutive_stagnant: int = 0
        self.undo_chain_remaining: int = 0
        self.strategy_audit_active: bool = False
        self.last_reset_step: int = -100

    def reset_level(self) -> None:
        self.hash_window.clear()
        self.action_window.clear()
        self.escalation_tier = 0
        self.consecutive_stagnant = 0
        self.undo_chain_remaining = 0
        self.strategy_audit_active = False

    def observe_state(self, state_hash: int, action_id: Optional[int], step_in_level: int) -> None:
        if not self.config.enable_avo_supervisor:
            return
        
        self.hash_window.append(state_hash)
        if action_id is not None:
            self.action_window.append(action_id)

        # Budget-based strategy audit trigger
        budget_pct = float(step_in_level) / max(1.0, float(self.config.max_actions_per_level))
        if budget_pct >= self.config.strategy_audit_pct:
            self.strategy_audit_active = True

        if len(self.hash_window) >= self.window_size:
            unique_states = len(set(self.hash_window))
            sigma_t = float(unique_states) / float(self.window_size)

            if sigma_t < self.config.stagnation_threshold:
                self.consecutive_stagnant += 1
                if self.consecutive_stagnant >= 3:
                    self.escalation_tier = 2
                    if self.undo_chain_remaining <= 0 and self.config.enable_undo_backtrack:
                        self.undo_chain_remaining = self.config.max_undo_chain
                        logger.info(f"[AVOSupervisor] Stagnation sigma_t={sigma_t:.2f}. Triggering {self.undo_chain_remaining}-step ACTION7 undo chain.")
                else:
                    self.escalation_tier = 1
                    self.strategy_audit_active = True
            else:
                self.consecutive_stagnant = 0
                if self.undo_chain_remaining <= 0:
                    self.escalation_tier = 0
                    if budget_pct < self.config.strategy_audit_pct:
                        self.strategy_audit_active = False

    def should_force_undo(self, available_actions: List[int]) -> bool:
        """Checks if supervisor mandates an ACTION7 (Undo) rollback."""
        if not self.config.enable_undo_backtrack or 7 not in available_actions:
            self.undo_chain_remaining = 0
            return False
        if self.undo_chain_remaining > 0:
            self.undo_chain_remaining -= 1
            return True
        return False

    def can_issue_reset(self, current_step: int) -> bool:
        """Enforces reset discipline: never issue two resets in a row."""
        if current_step - self.last_reset_step <= 1:
            return False
        self.last_reset_step = current_step
        return True


class DPrimeComputeScheduler:
    """Paces compute time across 110 games to strictly respect the 9-hour limit."""
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.start_wall_time = time.time()
        self.games_played: int = 0
        self.games_completed: int = 0
        self.total_levels_cleared: int = 0

    def get_remaining_budget_s(self) -> float:
        elapsed = time.time() - self.start_wall_time
        return max(0.0, self.config.total_kaggle_budget_s - elapsed)

    def allocate_game_budget(self, levels_cleared: int = 0) -> Tuple[int, float]:
        remaining_s = self.get_remaining_budget_s()
        remaining_games = max(1, self.config.expected_num_games - self.games_completed)
        base_time = remaining_s / float(remaining_games)
        # Give more time if levels are being cleared successfully
        time_budget = min(base_time * (1.0 + 0.2 * levels_cleared), 450.0)
        return self.config.max_actions_per_game, time_budget


class SafeGameExecutor:
    """Wraps agent execution with fail-safe recovery so crashes never terminate the game."""
    def __init__(self, scheduler: Optional[DPrimeComputeScheduler] = None):
        self.scheduler = scheduler or DPrimeComputeScheduler()

    def safe_step(
        self,
        step_fn: Callable[[], Tuple[int, Dict[str, Any]]],
        default_action: int = 1,
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            return step_fn()
        except Exception as exc:
            logger.error(f"[SafeGameExecutor] Unhandled exception in agent step: {exc}", exc_info=True)
            return default_action, {}
