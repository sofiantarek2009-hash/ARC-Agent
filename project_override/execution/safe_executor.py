import logging
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple
from project_override.config import AgentConfig

logger = logging.getLogger(__name__)

class TrajectorySupervisor:
    """
    NVIDIA AVO Trajectory Supervisor (arXiv:2603.24517) + Retrodict/THEA Tiered Escalation:
    Monitors the agent's search trajectory for stagnation or unproductive loops,
    redirects exploration across escalation tiers, triggers strategy-audit directives
    when 25% of level action budget is spent, and enforces strict RESET discipline
    (never issuing two RESETs in a row so the game never resets back to Level 1).
    """
    def __init__(self, enabled: bool = True, max_actions_per_level: int = 120):
        self.enabled = enabled
        self.max_actions_per_level = max_actions_per_level
        self.seen_hashes_in_level: Set[int] = set()
        self.steps_since_novel_state: int = 0
        self.consecutive_resets: int = 0
        self.escalation_tier: int = 0
        self.interventions: int = 0
        self.strategy_audit_active: bool = False

    def reset_level(self) -> None:
        self.seen_hashes_in_level.clear()
        self.steps_since_novel_state = 0
        self.escalation_tier = 0
        self.strategy_audit_active = False

    def observe_state(
        self, state_hash: int, action_id: Optional[int], step_in_level: int = 0
    ) -> None:
        if action_id == 0:
            self.consecutive_resets += 1
        elif action_id is not None:
            self.consecutive_resets = 0
        if step_in_level >= max(8, int(0.25 * self.max_actions_per_level)):
            self.strategy_audit_active = True
        if state_hash not in self.seen_hashes_in_level:
            self.seen_hashes_in_level.add(state_hash)
            self.steps_since_novel_state = 0
            if self.escalation_tier > 0:
                self.escalation_tier = max(0, self.escalation_tier - 1)
        else:
            self.steps_since_novel_state += 1
            if self.enabled:
                if self.steps_since_novel_state >= 6:
                    self.escalation_tier = 2
                    self.interventions += 1
                elif self.steps_since_novel_state >= 3:
                    self.escalation_tier = max(self.escalation_tier, 1)
                    self.interventions += 1

    def can_issue_reset(self) -> bool:
        """Enforces Retrodict RESET discipline: never issue two RESETs in a row."""
        return self.consecutive_resets == 0

class DPrimeComputeScheduler:
    """
    Daniel Franzen Milestone 2 / D-Prime & UCB style dynamic compute scheduler:
    Tracks wall-clock budget across all games, allocates extra steps/time
    to games where levels are actively being cleared per action, and applies tail fade.
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.start_time = time.perf_counter()
        self.games_started = 0
        self.games_completed = 0
        self.total_levels_cleared = 0

    def elapsed_total_s(self) -> float:
        return time.perf_counter() - self.start_time

    def remaining_total_s(self) -> float:
        return max(10.0, self.config.total_kaggle_budget_s - self.elapsed_total_s())

    def allocate_game_budget(
        self, levels_cleared_in_game: int, current_novelty_rate: float = 1.0
    ) -> Tuple[int, float]:
        rem_games = max(1, self.config.expected_num_games - self.games_completed)
        base_time_s = self.remaining_total_s() / float(rem_games)
        base_steps = self.config.max_actions_per_game
        if levels_cleared_in_game >= 2:
            multiplier = 1.75
        elif levels_cleared_in_game == 1 or current_novelty_rate > 0.5:
            multiplier = 1.35
        else:
            multiplier = 1.0
        max_steps = int(min(700, base_steps * multiplier))
        max_time_s = float(min(900.0, base_time_s * multiplier))
        return max_steps, max_time_s

class SafeGameExecutor:
    """Isolates per-action and per-game exceptions so no single error can halt the run."""
    def __init__(self, scheduler: Optional[DPrimeComputeScheduler] = None):
        self.scheduler = scheduler or DPrimeComputeScheduler()

    def safe_step(
        self,
        step_fn: Callable[[], Tuple[int, Dict[str, Any]]],
        default_action: int = 1,
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            act_id, payload = step_fn()
            return int(act_id), dict(payload or {})
        except Exception as exc:
            logger.warning(f"SafeGameExecutor caught action exception: {exc}")
            return int(default_action), {}
