from collections import deque
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ProgrammaticMemoryEngine:
    def __init__(self, game_id: str = "arc_game"):
        self.game_id = game_id
        self.records: List[Dict[str, Any]] = []
        self.log_path = Path(f"/tmp/arc_prolong_{game_id}.jsonl")

    def reset_level(self) -> None:
        self.records.clear()

    def append(
        self,
        step: int,
        level_idx: int,
        pre_hash: int,
        action_id: int,
        coord_xy: Tuple[int, int],
        post_hash: int,
        delta_omega: int,
        diff_summary: str,
        av_pos: Optional[Tuple[int, int]] = None,
        remote_triggers: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        rec = {
            "step": int(step),
            "level": int(level_idx),
            "pre_hash": int(pre_hash),
            "action": int(action_id),
            "xy": [int(coord_xy[0]), int(coord_xy[1])],
            "post_hash": int(post_hash),
            "dOmega": int(delta_omega),
            "av_pos": list(av_pos) if av_pos else None,
            "remote_triggers": [list(rt) for rt in (remote_triggers or [])],
            "diff": diff_summary,
        }
        self.records.append(rec)
        if len(self.records) > 400:
            self.records.pop(0)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def grep(self, pattern: str, max_results: int = 8) -> List[Dict[str, Any]]:
        pat = pattern.lower()
        matches = [r for r in reversed(self.records) if pat in json.dumps(r).lower()]
        return list(reversed(matches[:max_results]))

    def find_causal_triggers(self) -> List[Dict[str, Any]]:
        return [
            r for r in self.records
            if r.get("remote_triggers") or (r.get("action") in (5, 6, 7) and r.get("dOmega", 0) > 0)
        ]

    def query_state_action(self, state_hash: int) -> Dict[int, int]:
        return {int(r["action"]): int(r["post_hash"]) for r in self.records if r["pre_hash"] == state_hash}

    def format_recent_table(self, n: int = 8) -> str:
        return "\n".join(
            f"t={r['step']} a={r['action']} xy={r['xy']} dOmega={r['dOmega']} av={r['av_pos']} trig={r['remote_triggers']} {r['diff']}"
            for r in self.records[-n:]
        )


StateHistoryLog = ProgrammaticMemoryEngine


class ContextManager:
    _RAW_PREFIX = (
        "You are the Project Override Neurosymbolic + CodeAct Architect for ARC-AGI-3.\n"
        "Grid is 64x64 int8 (0..15). Actions: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=INTERACT, "
        "6=CLICK(x=col,y=row), 7=UNDO/CYCLE.\n"
        'Respond ONLY with valid JSON: {"subgoal": "NAVIGATE"|"CLICK"|"ACTION"|"CODEACT", '
        '"target_rc": [r, c], "click_xy": [x, y], "action_id": int, "code": str, "hypothesis": str}\n'
    )
    SYSTEM_PREFIX = _RAW_PREFIX + (" " * ((64 - (len(_RAW_PREFIX) % 64)) % 64))

    def __init__(self, high_water_tokens: int = 1900, low_water_tokens: int = 1100, game_id: str = "arc"):
        self.high_water_tokens = high_water_tokens
        self.low_water_tokens = low_water_tokens
        self.tier1_episodic: deque = deque(maxlen=24)
        self.tier2_summaries: List[str] = []
        self.tier4_reflexions: List[str] = []
        self.history_log = ProgrammaticMemoryEngine(game_id=game_id)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def record_step(
        self,
        step: int,
        action_id: int,
        delta_omega: int,
        note: str = "",
        level_idx: int = 0,
        pre_hash: int = 0,
        post_hash: int = 0,
        coord_xy: Tuple[int, int] = (0, 0),
        av_pos: Optional[Tuple[int, int]] = None,
        remote_triggers: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        entry = f"t={step}: act={action_id} dOmega={delta_omega} {note}".strip()
        self.tier1_episodic.append(entry)
        self.history_log.append(
            step=step,
            level_idx=level_idx,
            pre_hash=pre_hash,
            action_id=action_id,
            coord_xy=coord_xy,
            post_hash=post_hash,
            delta_omega=delta_omega,
            diff_summary=note,
            av_pos=av_pos,
            remote_triggers=remote_triggers,
        )
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
            causal_events = self.history_log.find_causal_triggers()[-2:]
            trig_note = f" causal_events={len(causal_events)}" if causal_events else ""
            self.tier2_summaries.append(
                f"[Compacted {len(compacted_chunk)} steps: {compacted_chunk[0]} .. {compacted_chunk[-1]}{trig_note}]"
            )
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
        tier3 = f"Level={level_idx} | AvatarPos={av_pos} | Vectors={vectors} | Walls={sorted(walls)} | Hazards={sorted(hazards)} | Avail={available_actions}"
        t2 = "\n".join(self.tier2_summaries[-3:])
        t1 = self.history_log.format_recent_table(n=8)
        t4 = "\n".join(self.tier4_reflexions[-3:])
        causal_hits = self.history_log.find_causal_triggers()[-3:]
        causal_str = "; ".join(f"t={c['step']}:a={c['action']}->trig={c['remote_triggers']}" for c in causal_hits)
        return (
            f"{self.SYSTEM_PREFIX}"
            f"[TIER 3 LAWS] {tier3}\n"
            f"[CAUSAL TRIGGERS] {causal_str}\n"
            f"[TIER 4 REFLEXION] {t4}\n"
            f"[TIER 2 COMPACTED] {t2}\n"
            f"[TIER 1 PRO-LONG LOG]\n{t1}\n"
            f"[ENTITIES] {entities_summary}\n"
            f"Emit next JSON subgoal:"
        )
