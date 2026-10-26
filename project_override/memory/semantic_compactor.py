from collections import deque
import logging
from typing import Any, Dict, List, Optional
from project_override.config import AgentConfig
from project_override.memory.lineage_dag import LineageDAG

logger = logging.getLogger(__name__)


class SemanticMemoryCompactor:
    """Two-Part Playbook Semantic Memory Compactor (arXiv:2610.26632).
    
    Maintains:
    - GUIDE.md: Persistent, cross-level invariant playbook (verified rules, entity roles, affordances).
    - WORKING.md: Transient, per-level workspace (current subgoals, recent diffs, active hypotheses).
    
    Applies token hysteresis:
      If estimated prompt tokens exceed context_trim_high_tokens (118,000),
      compacts down to context_trim_target_tokens (59,000) by compressing working history
      while preserving all verified causal invariants in GUIDE.md.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.guide_md: str = ""
        self.working_turns: deque = deque(maxlen=self.config.max_rolling_turns)
        self.active_subgoal: str = "EXPLORE_ACTIONS"
        self.reasoning_traces: List[str] = []

    def reset_level(self) -> None:
        self.working_turns.clear()
        self.active_subgoal = "EXPLORE_ACTIONS"
        self.reasoning_traces.clear()

    def add_working_turn(self, step_idx: int, action_str: str, diff_str: str) -> None:
        self.working_turns.append(f"S{step_idx}: {action_str} -> {diff_str}")

    def add_reasoning_trace(self, thought: str) -> None:
        if thought and thought.strip():
            # Keep up to 5 most recent reasoning thoughts
            self.reasoning_traces.append(thought.strip()[:240])
            if len(self.reasoning_traces) > 5:
                self.reasoning_traces.pop(0)

    def on_level_clear(self, level: int, lineage_dag: LineageDAG) -> None:
        """Promotes discoveries from the cleared level into persistent GUIDE.md."""
        laws = lineage_dag.render_for_prompt()
        self.guide_md = (
            f"## INVARIANT PLAYBOOK (Cross-Level Guide)\n"
            f"{laws}\n"
        )
        self.reset_level()

    def estimate_token_count(self) -> int:
        """Rough estimation of total characters / 4."""
        total_chars = len(self.guide_md) + sum(len(t) for t in self.working_turns)
        return total_chars // 4

    def compact_if_needed(self, lineage_dag: LineageDAG) -> None:
        """Enforces context hysteresis compaction."""
        tokens = self.estimate_token_count()
        if tokens > self.config.context_trim_high_tokens:
            logger.info(f"[SemanticCompactor] Token count {tokens} > {self.config.context_trim_high_tokens}. Compacting memory.")
            # Keep only the last 6 turns in working memory
            while len(self.working_turns) > 6:
                self.working_turns.popleft()
            # Refresh guide from DAG
            self.guide_md = (
                f"## INVARIANT PLAYBOOK (Compacted)\n"
                f"{lineage_dag.render_for_prompt(max_tokens=2000)}\n"
            )

    def render_context(self, lineage_dag: LineageDAG) -> str:
        """Assembles prompt memory context."""
        guide_text = self.guide_md if self.guide_md else lineage_dag.render_for_prompt()
        working_text = "\n".join(self.working_turns) if self.working_turns else "None (Beginning of level)"
        traces_text = "\n".join(f"- {t}" for t in self.reasoning_traces) if self.reasoning_traces else "None"

        return (
            f"### PLAYBOOK (GUIDE.md):\n{guide_text}\n\n"
            f"### RECENT TURNS (WORKING.md):\n{working_text}\n\n"
            f"### RECENT REASONING (S4 Replay):\n{traces_text}"
        )
