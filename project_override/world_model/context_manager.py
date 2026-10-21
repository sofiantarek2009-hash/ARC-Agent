from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class ContextManager:
    """
    4-Tier Memory & Franzen Hysteresis Context Compaction (HARNESS_DESIGN.md).
    - Immutable System Prefix (KV-cache anchored)
    - Tier 1: Episodic Ring Buffer
    - Tier 2: Compacted Causal Summary (Hysteresis band: High=1900 tokens -> Low=1100 tokens)
    - Tier 3: Confirmed Physical Laws, Remote Triggers & Hazards
    - Tier 4: Cross-Level Schema & Reflexion Post-Mortems
    """

    SYSTEM_PREFIX = (
        "You are the Project Override Fluid-Intelligence Architect for ARC-AGI-3.\n"
        "Grid is 64x64 int8 (0..15). Actions: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=INTERACT, "
        "6=CLICK(x=col,y=row), 7=UNDO/CYCLE.\n"
        "Respond ONLY with valid JSON: {"subgoal": "NAVIGATE"|"CLICK"|"ACTION", "
        ""target_rc": [r, c], "click_xy": [x, y], "action_id": int, "hypothesis": str}"
    )

    def __init__(self, high_water_tokens: int = 1900, low_water_tokens: int = 1100):
        self.high_water_tokens = high_water_tokens
        self.low_water_tokens = low_water_tokens
        self.tier1_episodic: deque = deque(maxlen=24)
        self.tier2_summaries: List[str] = []
        self.tier4_reflexions: List[str] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def record_step(self, step: int, action_id: int, delta_omega: int, note: str = "") -> None:
        entry = f"t={step}: act={action_id} dOmega={delta_omega} {note}".strip()
        self.tier1_episodic.append(entry)
        self._enforce_hysteresis()

    def record_reflexion(self, failure_reason: str) -> None:
        if failure_reason and failure_reason not in self.tier4_reflexions:
            self.tier4_reflexions.append(failure_reason[:160])
            if len(self.tier4_reflexions) > 6:
                self.tier4_reflexions.pop(0)

    def _enforce_hysteresis(self) -> None:
        raw_text = "\n".join(self.tier2_summaries + list(self.tier1_episodic))
        if self.estimate_tokens(self.SYSTEM_PREFIX + raw_text) <= self.high_water_tokens:
            return
        compacted_chunk = []
        while len(self.tier1_episodic) > 6 and self.estimate_tokens(
            self.SYSTEM_PREFIX + "\n".join(self.tier2_summaries + list(self.tier1_episodic))
        ) > self.low_water_tokens:
            compacted_chunk.append(self.tier1_episodic.popleft())
        if compacted_chunk:
            digest = f"[Compacted {len(compacted_chunk)} steps: {compacted_chunk[0]} .. {compacted_chunk[-1]}]"
            self.tier2_summaries.append(digest)
        while len(self.tier2_summaries) > 4:
            self.tier2_summaries.pop(0)

    def build_prompt(
        self,
        level_idx: int,
        av_pos: Optional[Tuple[int, int]],
        vectors: Dict[int, Tuple[int, int]],
        walls: Set[int],
        hazards: Set[int],
        entities_summary: str,
        available_actions: List[int],
    ) -> str:
        tier3 = (
            f"Level={level_idx} | AvatarPos={av_pos} | Vectors={vectors} | "
            f"Walls={sorted(walls)} | Hazards={sorted(hazards)} | Avail={available_actions}"
        )
        t2 = "\n".join(self.tier2_summaries[-3:])
        t1 = "\n".join(list(self.tier1_episodic)[-8:])
        t4 = "\n".join(self.tier4_reflexions[-3:])
        return (
            f"{self.SYSTEM_PREFIX}\n"
            f"[TIER 3 LAWS] {tier3}\n"
            f"[TIER 4 REFLEXION] {t4}\n"
            f"[TIER 2 COMPACTED] {t2}\n"
            f"[TIER 1 EPISODIC]\n{t1}\n"
            f"[ENTITIES] {entities_summary}\n"
            f"Emit next JSON subgoal:"
        )
