from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

@dataclass
class TelemetryVector:
    """Structured telemetry vector capturing one state transition."""
    step: int
    level: int
    action_id: int
    x: int
    y: int
    pre_hash: int
    post_hash: int
    delta_omega: int
    avatar_pos: Optional[Tuple[int, int]] = None
    diff_summary: str = ""
    inferred_effect: Optional[str] = None
    entities_changed: int = 0


@dataclass
class ReplayTransition:
    """Historical transition for offline counterexample verification."""
    step: int
    level: int
    pre_grid: np.ndarray
    action_id: int
    x: int
    y: int
    post_grid: np.ndarray
    delta_omega: int
    diff_summary: str


class GroundedReplayVerifier:
    """Replay verifier: validates causal hypotheses against historical transitions."""
    def __init__(self):
        self.transitions: List[ReplayTransition] = []

    def add(self, trans: ReplayTransition) -> None:
        self.transitions.append(trans)

    def verify_action_effect(self, action_id: int, expected_vector: Tuple[int, int]) -> Tuple[bool, int]:
        """Checks what percentage of historical actions matched this displacement vector."""
        matches = 0
        total = 0
        for t in self.transitions:
            if t.action_id == action_id and "MOVED" in t.diff_summary:
                total += 1
                dr, dc = expected_vector
                tag = f"MOVED({dr:+d},{dc:+d})"
                if tag in t.diff_summary:
                    matches += 1
        if total == 0:
            return True, 0
        return (matches == total), total


class LineageDAG:
    """Persistent Lineage Directed Acyclic Graph & Cross-Level Invariant Memory (NVIDIA AVO).
    
    Preserves discovered game laws across level boundaries by abstracting away level-specific
    coordinates and retaining:
    1. Action semantics (e.g., ACTION1 = move North, ACTION6 = toggle card)
    2. Object affordances (walls, collectibles, switches, hazards)
    3. Falsified hypotheses (preventing repeated dead-end strategies)
    """

    def __init__(self):
        self.telemetry: List[TelemetryVector] = []
        self.replay_verifier = GroundedReplayVerifier()
        self.verified_laws: Dict[str, str] = {}
        self.disproven_hypotheses: List[Dict[str, Any]] = []
        self.level_summaries: Dict[int, str] = {}
        self.persistent_notes: Dict[str, Any] = {}

    def record_step(
        self,
        step: int,
        level: int,
        action_id: int,
        x: int,
        y: int,
        pre_hash: int,
        post_hash: int,
        delta_omega: int,
        pre_grid: Optional[np.ndarray] = None,
        post_grid: Optional[np.ndarray] = None,
        avatar_pos: Optional[Tuple[int, int]] = None,
        diff_summary: str = "",
    ) -> None:
        tv = TelemetryVector(
            step=step,
            level=level,
            action_id=action_id,
            x=x,
            y=y,
            pre_hash=pre_hash,
            post_hash=post_hash,
            delta_omega=delta_omega,
            avatar_pos=avatar_pos,
            diff_summary=diff_summary,
        )
        self.telemetry.append(tv)
        if pre_grid is not None and post_grid is not None:
            self.replay_verifier.add(
                ReplayTransition(
                    step=step,
                    level=level,
                    pre_grid=pre_grid.copy(),
                    action_id=action_id,
                    x=x,
                    y=y,
                    post_grid=post_grid.copy(),
                    delta_omega=delta_omega,
                    diff_summary=diff_summary,
                )
            )

    def record_falsification(
        self, hypothesis: str, step: int, level: int, reason: str
    ) -> None:
        self.disproven_hypotheses.append({
            "hypothesis": hypothesis,
            "step": step,
            "level": level,
            "reason": reason,
        })

    def record_verified_law(self, key: str, value: str) -> None:
        self.verified_laws[key] = value

    def on_level_clear(self, level: int, actions_used: int) -> None:
        """Strips level coordinates and extracts invariant game laws upon level clearance."""
        level_tvs = [t for t in self.telemetry if t.level == level]
        # Infer consistent action displacements
        action_displacements: Dict[int, List[str]] = defaultdict(list)
        for t in level_tvs:
            if "MOVED" in t.diff_summary:
                action_displacements[t.action_id].append(t.diff_summary)

        for aid, summaries in action_displacements.items():
            cnt = Counter(summaries)
            most_common, freq = cnt.most_common(1)[0]
            if freq >= max(2, len(summaries) // 2):
                self.verified_laws[f"ACTION_{aid}_SEMANTICS"] = most_common

        summary = f"Level {level} cleared in {actions_used} actions. Consistent actions: {list(action_displacements.keys())}"
        self.level_summaries[level] = summary

    def render_for_prompt(self, max_tokens: int = 500) -> str:
        """Renders persistent memory for LLM context injection."""
        sections = []
        if self.verified_laws:
            sections.append("### VERIFIED GAME LAWS (Cross-Level Invariants):")
            for k, v in self.verified_laws.items():
                sections.append(f"- {k}: {v}")
        if self.persistent_notes:
            sections.append("### SCIENTIFIC NOTES:")
            for k, v in list(self.persistent_notes.items())[-8:]:
                sections.append(f"- {k}: {v}")
        if self.disproven_hypotheses:
            sections.append("### DISPROVEN HYPOTHESES (Do not repeat):")
            for dh in self.disproven_hypotheses[-4:]:
                sections.append(f"- [FAILED at L{dh['level']} S{dh['step']}] {dh['hypothesis']}: {dh['reason']}")
        if self.level_summaries:
            sections.append("### CLEARED LEVELS:")
            for lvl, summ in self.level_summaries.items():
                sections.append(f"- {summ}")

        return "\n".join(sections) if sections else "No verified laws established yet."
