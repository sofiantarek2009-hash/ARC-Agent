"""Project Override: NVIDIA AVO (Agentic Variation Operators) Trajectory Supervisor.

Implements the complete dual-agent supervisory layer from arXiv:2603.24517:
- Continuous Stagnation Index sigma_t monitoring progress deltas, hash diversity, and budget burn.
- Tier 1: Programmatic Action Banning (bans top-2 repeated actions in window).
- Tier 2: Multi-step Undo (Action 7) rollback chain.
- Tier 3: Programmatic Checkpoint Reset (Action 0) when deadlocked, preserving learned memory.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from project_override.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class KeyframeNode:
    node_id: int
    parent_id: Optional[int]
    state_hash: int
    action_taken: Optional[int]
    depth: int
    step: int
    children: List[int] = field(default_factory=list)
    banned_actions: Set[int] = field(default_factory=set)


class KeyframeSearchTree:
    """MCTS-style interactive keyframe search tree for multi-step Action 7 backtracking."""
    def __init__(self):
        self.nodes: Dict[int, KeyframeNode] = {}
        self.current_node_id: int = 0
        self.next_node_id: int = 1
        self.nodes[0] = KeyframeNode(node_id=0, parent_id=None, state_hash=0, action_taken=None, depth=0, step=0)
        self.history_stack: List[int] = [0]

    def reset(self) -> None:
        self.nodes.clear()
        self.current_node_id = 0
        self.next_node_id = 1
        self.nodes[0] = KeyframeNode(node_id=0, parent_id=None, state_hash=0, action_taken=None, depth=0, step=0)
        self.history_stack = [0]

    def record_step(self, action: int, post_hash: int, step: int) -> int:
        curr = self.nodes.get(self.current_node_id)
        depth = (curr.depth + 1) if curr else 1
        nid = self.next_node_id
        self.next_node_id += 1
        node = KeyframeNode(
            node_id=nid,
            parent_id=self.current_node_id,
            state_hash=post_hash,
            action_taken=action,
            depth=depth,
            step=step,
        )
        self.nodes[nid] = node
        if curr:
            curr.children.append(nid)
        self.current_node_id = nid
        self.history_stack.append(nid)
        return nid

    def record_undo(self) -> None:
        if len(self.history_stack) > 1:
            self.history_stack.pop()
            self.current_node_id = self.history_stack[-1]

    def compute_backtrack_to_fork(self) -> int:
        """Finds nearest unexhausted ancestor and returns steps to undo."""
        if len(self.history_stack) <= 1:
            return 0
        curr = self.nodes.get(self.current_node_id)
        if not curr:
            return 0

        if curr.parent_id is not None and curr.action_taken is not None:
            parent = self.nodes.get(curr.parent_id)
            if parent:
                parent.banned_actions.add(curr.action_taken)

        steps = 1
        target_ancestor = curr.parent_id
        while target_ancestor is not None and target_ancestor != 0:
            anc_node = self.nodes.get(target_ancestor)
            if anc_node and len(anc_node.banned_actions) < 4:
                break
            if anc_node and anc_node.parent_id is not None:
                target_ancestor = anc_node.parent_id
                steps += 1
            else:
                break
        return min(steps, 6)


class AVOSupervisor:
    """NVIDIA Agentic Variation Operators (AVO) Trajectory Supervisor (arXiv:2603.24517).
    
    Monitors trajectory health via continuous Stagnation Index:
        sigma_t = w1 * (delta_progress <= 0) + w2 * (1 - |unique(H_t)| / |H_t|) + w3 * (A_spent / A_budget)
    
    Escalation Tiers:
      Tier 0: Healthy exploration / normal progress (sigma_t < 0.50).
      Tier 1: Stagnation (sigma_t >= 0.50) -> Programmatically ban repeated actions & Strategy Audit Prompt.
      Tier 2: Deadlock / loop trap (sigma_t >= 0.70) -> Multi-step Action 7 (Undo) rollback chain.
      Tier 3: Irreversible trap / severe stagnation (sigma_t >= 0.85) -> Programmatic Action 0 (Reset) discipline.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.window_size = self.config.stagnation_window
        self.hash_window: deque = deque(maxlen=self.window_size)
        self.action_window: deque = deque(maxlen=self.window_size)
        self.progress_window: deque = deque(maxlen=self.window_size)
        self.escalation_tier: int = 0
        self.consecutive_stagnant: int = 0
        self.undo_chain_remaining: int = 0
        self.strategy_audit_active: bool = False
        self.last_reset_step: int = -100
        self.last_sigma_t: float = 0.0
        self.keyframe_tree = KeyframeSearchTree()

    def reset_level(self) -> None:
        """Resets sliding trajectory windows upon level entry or reset."""
        self.hash_window.clear()
        self.action_window.clear()
        self.progress_window.clear()
        self.escalation_tier = 0
        self.consecutive_stagnant = 0
        self.undo_chain_remaining = 0
        self.strategy_audit_active = False
        self.last_sigma_t = 0.0
        self.keyframe_tree.reset()

    def observe_state(
        self,
        state_hash: int,
        action_id: Optional[int],
        step_in_level: int,
        progress_val: Optional[float] = None,
    ) -> float:
        """Observes transition and updates continuous Stagnation Index sigma_t."""
        if not self.config.enable_avo_supervisor:
            return 0.0

        self.hash_window.append(state_hash)
        if action_id is not None:
            self.action_window.append(action_id)
            self.keyframe_tree.record_step(action_id, state_hash, step_in_level)
        if progress_val is not None:
            self.progress_window.append(progress_val)

        # Budget exhaustion ratio
        budget_ratio = min(1.0, float(step_in_level) / max(1.0, float(self.config.max_actions_per_level)))
        if budget_ratio >= self.config.strategy_audit_pct:
            self.strategy_audit_active = True

        if len(self.hash_window) < self.window_size:
            self.last_sigma_t = 0.2 * budget_ratio
            return self.last_sigma_t

        # 1. State hash diversity penalty: w2 * (1 - |unique| / N)
        unique_states = len(set(self.hash_window))
        diversity_ratio = float(unique_states) / float(self.window_size)
        w2_penalty = 0.40 * (1.0 - diversity_ratio)

        # 2. Progress stall penalty: w1 * I(delta_progress <= 0)
        w1_penalty = 0.0
        if len(self.progress_window) >= 4:
            prog_delta = self.progress_window[-1] - self.progress_window[0]
            if prog_delta <= 0:
                w1_penalty = 0.40
        else:
            # If no explicit progress metric provided, penalize low hash diversity as proxy
            w1_penalty = 0.40 if diversity_ratio < 0.60 else 0.0

        # 3. Budget burn penalty: w3 * (A_spent / A_budget)
        w3_penalty = 0.20 * budget_ratio

        # Continuous Stagnation Index sigma_t in [0.0, 1.0]
        sigma_t = w1_penalty + w2_penalty + w3_penalty
        self.last_sigma_t = sigma_t

        # Escalation Tier Evaluation
        if sigma_t >= 0.85:
            self.escalation_tier = 3
            self.consecutive_stagnant += 1
            self.strategy_audit_active = True
        elif sigma_t >= 0.70:
            self.escalation_tier = 2
            self.consecutive_stagnant += 1
            self.strategy_audit_active = True
            if self.undo_chain_remaining <= 0 and self.config.enable_undo_backtrack:
                self.undo_chain_remaining = self.config.max_undo_chain
                logger.info(f"[AVOSupervisor] Stagnation sigma_t={sigma_t:.2f}. Triggering {self.undo_chain_remaining}-step Action 7 undo chain.")
        elif sigma_t >= 0.50:
            self.escalation_tier = 1
            self.strategy_audit_active = True
        else:
            self.escalation_tier = 0
            self.consecutive_stagnant = 0
            if self.undo_chain_remaining <= 0 and budget_ratio < self.config.strategy_audit_pct:
                self.strategy_audit_active = False

        return sigma_t

    def should_force_undo(self, available_actions: List[int]) -> bool:
        """Operator 2: Multi-step Action 7 (Undo) Rollback Chain via Keyframe Tree."""
        if not self.config.enable_undo_backtrack or 7 not in available_actions:
            self.undo_chain_remaining = 0
            return False
        if self.undo_chain_remaining <= 0 and self.escalation_tier >= 2:
            backtrack = self.keyframe_tree.compute_backtrack_to_fork()
            if backtrack > 0:
                self.undo_chain_remaining = backtrack
                logger.info(f"[AVOSupervisor] Keyframe Search Tree computed {backtrack}-step backtrack to fork.")
        if self.undo_chain_remaining > 0:
            self.undo_chain_remaining -= 1
            self.keyframe_tree.record_undo()
            return True
        return False

    def should_force_reset(self, current_step: int, available_actions: List[int]) -> bool:
        """Operator 3: Checkpoint Reset Discipline.
        
        When in a severe deadlock/loop trap (Tier 3 or consecutive stagnant windows >= 2)
        and Action 7 (Undo) is not available, force Action 0 (Reset)
        to escape the irreversible trap.
        """
        if (self.escalation_tier >= 3 or self.consecutive_stagnant >= 2) and 7 not in available_actions:
            if self.can_issue_reset(current_step):
                logger.info(f"[AVOSupervisor] Irreversible stagnation trap (sigma_t={self.last_sigma_t:.2f}). Enforcing Action 0 (Reset).")
                return True
        return False

    def filter_banned_actions(self, available_actions: List[int]) -> List[int]:
        """Operator 1: Programmatic Action Banning.
        
        When stagnation (sigma_t >= 0.50) is detected, programmatically bans the
        top-2 most frequent actions in the window and failed branches from Keyframe Tree.
        """
        if not self.strategy_audit_active or len(self.action_window) < 4:
            return available_actions

        counts = Counter(self.action_window)
        most_common = set(act for act, _ in counts.most_common(2))

        curr_node = self.keyframe_tree.nodes.get(self.keyframe_tree.current_node_id)
        tree_banned = curr_node.banned_actions if curr_node else set()
        all_banned = most_common | tree_banned

        filtered = [a for a in available_actions if a not in all_banned]
        if filtered:
            logger.info(f"[AVOSupervisor] Stagnation active (sigma_t={self.last_sigma_t:.2f}): Banning {all_banned}. Allowed: {filtered}")
            return filtered

        least_common = [act for act, _ in counts.most_common()][-1]
        return [least_common] if least_common in available_actions else available_actions

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

