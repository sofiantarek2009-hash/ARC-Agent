#!/usr/bin/env python3
import ast
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PO_DIR = ROOT / "project_override"

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_json = kaggle_dir / "kaggle.json"
kaggle_json.write_text(
    json.dumps({"username": "sofianali188201", "key": "KGAT_35fa12ec577db15861b1ed74c79b9e93"}),
    encoding="utf-8",
)
os.chmod(kaggle_json, stat.S_IRUSR | stat.S_IWUSR)

for sub in ["core", "perception", "planning", "world_model", "orchestration"]:
    sub_path = PO_DIR / sub
    sub_path.mkdir(parents=True, exist_ok=True)
    init_f = sub_path / "__init__.py"
    if not init_f.exists():
        init_f.write_text("", encoding="utf-8")
if not (PO_DIR / "__init__.py").exists():
    (PO_DIR / "__init__.py").write_text("", encoding="utf-8")

# =========================================================================
# Synchronize V15 Core & World-Model Modules (1, 8, 9, 10, 11)
# =========================================================================
(PO_DIR / "core" / "dead_signature.py").write_text(r'''import hashlib
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class DynamicHUDMasker:
    def __init__(self, shape: Tuple[int, int] = (64, 64), default_bezel: int = 4):
        self.shape = shape
        self.default_bezel = default_bezel
        self.hud_mask = np.zeros(shape, dtype=bool)
        self.change_counts = np.zeros(shape, dtype=np.int32)
        self.total_transitions = 0
        self._init_bezel()

    def _init_bezel(self) -> None:
        h, w = self.shape
        b = self.default_bezel
        if h > 2 * b and w > 2 * b:
            self.hud_mask[:b, :] = True
            self.hud_mask[h - b:, :] = True
            self.hud_mask[:, :b] = True
            self.hud_mask[:, w - b:] = True

    def reset_level(self) -> None:
        self.change_counts.fill(0)
        self.total_transitions = 0

    def observe_transition(self, pre_grid: np.ndarray, post_grid: np.ndarray) -> None:
        if pre_grid.shape != self.shape or post_grid.shape != self.shape:
            return
        diff = pre_grid != post_grid
        changed_px = int(np.sum(diff))
        if 0 < changed_px <= 120:
            self.total_transitions += 1
            self.change_counts[diff] += 1
            h, w = self.shape
            border_band = np.zeros((h, w), dtype=bool)
            border_band[:8, :] = True
            border_band[h - 8:, :] = True
            border_band[:, :8] = True
            border_band[:, w - 8:] = True
            self.hud_mask |= (self.change_counts >= 2) & border_band
            for r in range(h):
                if (r < 8 or r >= h - 8) and int(np.sum(self.change_counts[r, :])) >= 3:
                    self.hud_mask[r, self.change_counts[r, :] >= 1] = True
            for c in range(w):
                if (c < 8 or c >= w - 8) and int(np.sum(self.change_counts[:, c])) >= 3:
                    self.hud_mask[self.change_counts[:, c] >= 1, c] = True

    def mask_grid(self, grid: np.ndarray) -> np.ndarray:
        arr = np.asarray(grid, dtype=np.int8).copy()
        if arr.shape == self.hud_mask.shape:
            arr[self.hud_mask] = 0
        return arr

    def coverage_ratio(self) -> float:
        return float(np.mean(~self.hud_mask))


def compute_arena_hash(
    grid: np.ndarray,
    bezel_px: int = 4,
    hud_masker: Optional[DynamicHUDMasker] = None,
) -> int:
    arr = np.asarray(grid, dtype=np.int8)
    if hud_masker is not None and arr.shape == hud_masker.shape:
        masked = hud_masker.mask_grid(arr)
        digest = hashlib.blake2b(np.ascontiguousarray(masked).tobytes(), digest_size=8).digest()
        return int.from_bytes(digest, "little") & ((1 << 44) - 1)
    h, w = arr.shape
    core = np.ascontiguousarray(arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px]) if (h > 2 * bezel_px and w > 2 * bezel_px) else np.ascontiguousarray(arr)
    digest = hashlib.blake2b(core.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 44) - 1)


def pack_signature_64(
    arena_hash: int,
    action_id: int,
    coord_x: int = 0,
    coord_y: int = 0,
    level_idx: int = 0,
) -> int:
    h44 = arena_hash & ((1 << 44) - 1)
    lvl4 = level_idx & 0xF
    cy6 = coord_y & 0x3F
    cx6 = coord_x & 0x3F
    act4 = action_id & 0xF
    return (h44 << 20) | (lvl4 << 16) | (cy6 << 10) | (cx6 << 4) | act4


class DeadSignatureTable:
    INVERSE_MAP = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self, ring_size: int = 32):
        self.dead_set: Set[int] = set()
        self.death_set: Set[int] = set()
        self.ring_buffer: deque = deque(maxlen=ring_size)
        self.transitions: Dict[int, Dict[Tuple[int, int, int], int]] = {}
        self.state_grids: Dict[int, np.ndarray] = {}
        self.state_novelty: Dict[int, int] = {}
        self.state_visit_counts: Dict[int, int] = {}
        self.action_fail_counts: Dict[Tuple[int, int], int] = {}
        self.seen_atoms: Set[Tuple[int, int, int]] = set()
        self.seen_pairs: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()

    def reset_level(self, clear_deaths: bool = False) -> None:
        self.ring_buffer.clear()
        self.transitions.clear()
        self.state_grids.clear()
        self.state_novelty.clear()
        self.state_visit_counts.clear()
        self.action_fail_counts.clear()
        self.seen_atoms.clear()
        self.seen_pairs.clear()
        if clear_deaths:
            self.death_set.clear()
            self.dead_set.clear()

    def record_death(self, pre_hash: int, action_id: int, coord_x: int = 0, coord_y: int = 0, level_idx: int = 0) -> None:
        sig = pack_signature_64(pre_hash, action_id, coord_x, coord_y, level_idx)
        self.death_set.add(sig)
        self.dead_set.add(sig)
        self.action_fail_counts[(pre_hash, int(action_id))] = 999

    def record_transition(
        self,
        pre_hash: int,
        action_id: int,
        post_hash: int,
        delta_omega: int,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
        post_grid: Optional[np.ndarray] = None,
    ) -> None:
        self.state_visit_counts[post_hash] = self.state_visit_counts.get(post_hash, 0) + 1
        if delta_omega == 0 or pre_hash == post_hash:
            sig = pack_signature_64(pre_hash, action_id, coord_x, coord_y, level_idx)
            self.dead_set.add(sig)
            key = (pre_hash, int(action_id))
            self.action_fail_counts[key] = self.action_fail_counts.get(key, 0) + 1
        else:
            act_key = (int(action_id), int(coord_x), int(coord_y))
            if pre_hash not in self.transitions:
                self.transitions[pre_hash] = {}
            self.transitions[pre_hash][act_key] = post_hash
            if post_grid is not None and post_hash not in self.state_grids:
                self.state_grids[post_hash] = post_grid.copy()
                self.state_novelty[post_hash] = self.update_novelty(post_grid)
        self.ring_buffer.append(post_hash)

    def update_novelty(self, grid: np.ndarray) -> int:
        arr = np.asarray(grid, dtype=np.int8)
        new_count = 0
        active_atoms = []
        for r in range(4, arr.shape[0] - 4, 4):
            for c in range(4, arr.shape[1] - 4, 4):
                atom = (r, c, int(arr[r, c]))
                active_atoms.append(atom)
                if atom not in self.seen_atoms:
                    self.seen_atoms.add(atom)
                    new_count += 2
        if new_count == 0 and len(active_atoms) <= 64 and len(self.seen_pairs) < 8000:
            for i in range(0, len(active_atoms), 2):
                for j in range(i + 1, min(i + 6, len(active_atoms))):
                    pair = (active_atoms[i], active_atoms[j])
                    if pair not in self.seen_pairs:
                        self.seen_pairs.add(pair)
                        new_count += 1
        return new_count

    def is_dead(self, arena_hash: int, action_id: int, coord_x: int = 0, coord_y: int = 0, level_idx: int = 0) -> bool:
        sig = pack_signature_64(arena_hash, action_id, coord_x, coord_y, level_idx)
        return (sig in self.dead_set) or (sig in self.death_set)

    def get_untried_actions(self, arena_hash: int, available_actions: List[int], level_idx: int = 0) -> List[int]:
        tried = {k[0] for k in self.transitions.get(arena_hash, {}).keys()}
        return [
            int(a) for a in available_actions
            if int(a) in (1, 2, 3, 4, 5, 7) and int(a) not in tried and not self.is_dead(arena_hash, int(a), 0, 0, level_idx)
        ]

    def compute_action_penalty(
        self,
        arena_hash: int,
        action_id: int,
        last_action_id: Optional[int] = None,
        coord_x: int = 0,
        coord_y: int = 0,
        level_idx: int = 0,
    ) -> float:
        if self.is_dead(arena_hash, action_id, coord_x, coord_y, level_idx):
            return 1e6
        penalty = 0.0
        act_key = (int(action_id), int(coord_x), int(coord_y))
        known_out = self.transitions.get(arena_hash, {})
        if act_key in known_out:
            dest_hash = known_out[act_key]
            penalty += self.state_visit_counts.get(dest_hash, 0) * 4.5
            if len(self.ring_buffer) >= 2 and dest_hash == self.ring_buffer[-2]:
                penalty += 25.0
        if last_action_id is not None and self.INVERSE_MAP.get(last_action_id) == action_id:
            penalty += 8.0
        return penalty

    def detect_loop(self) -> bool:
        n = len(self.ring_buffer)
        if n < 6:
            return False
        recent = list(self.ring_buffer)
        if len(set(recent[-4:])) == 1:
            return True
        if recent[-1] == recent[-3] == recent[-5] and recent[-2] == recent[-4]:
            return True
        if n >= 8 and recent[-4:] == recent[-8:-4]:
            return True
        if n >= 6 and recent[-3:] == recent[-6:-3]:
            return True
        return False

    def bfs_to_unexplored_action(
        self,
        current_hash: int,
        candidate_actions: List[int],
        level_idx: int = 0,
        last_action_id: Optional[int] = None,
    ) -> Optional[List[int]]:
        if not candidate_actions:
            return None
        untried_here = self.get_untried_actions(current_hash, candidate_actions, level_idx)
        if untried_here:
            untried_here.sort(key=lambda a: self.compute_action_penalty(current_hash, a, last_action_id, 0, 0, level_idx))
            return [untried_here[0]]

        queue = deque([(current_hash, [])])
        visited = {current_hash}
        best_path: Optional[List[int]] = None
        best_score = -1e9

        while queue and len(visited) < 450:
            sh, path = queue.popleft()
            if len(path) >= 24:
                continue
            tried = {k[0] for k in self.transitions.get(sh, {}).keys()}
            for a in candidate_actions:
                if not self.is_dead(sh, a, 0, 0, level_idx) and a not in tried:
                    cand_path = path + [a]
                    score = (self.state_novelty.get(sh, 0) * 3.0) - (len(cand_path) * 1.2) - (self.state_visit_counts.get(sh, 1) * 0.8)
                    if score > best_score:
                        best_score = score
                        best_path = cand_path
                    if len(cand_path) <= 2:
                        return cand_path
            for (act_id, cx, cy), nxt_h in self.transitions.get(sh, {}).items():
                if act_id in candidate_actions and nxt_h not in visited and not self.is_dead(sh, act_id, cx, cy, level_idx):
                    visited.add(nxt_h)
                    queue.append((nxt_h, path + [act_id]))
        return best_path
''', encoding="utf-8")

(PO_DIR / "world_model" / "metagame_schema.py").write_text(r'''from typing import Callable, Dict, Optional, Set, Tuple


class MetagameSchema:
    INVERSE_PAIRS = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self):
        self.displacement_vectors: Dict[int, Tuple[int, int]] = {}
        self.frozen: bool = False
        self.verified_world_model: Optional[Callable] = None
        self.verified_model_code: Optional[str] = None
        self.key_to_door_colors: Dict[int, Set[int]] = {}

    def record_vector(self, action_id: int, vec: Tuple[int, int]) -> None:
        if vec == (0, 0):
            return
        self.displacement_vectors[int(action_id)] = (int(vec[0]), int(vec[1]))
        inv = self.INVERSE_PAIRS.get(int(action_id))
        if inv is not None and inv not in self.displacement_vectors:
            self.displacement_vectors[inv] = (-int(vec[0]), -int(vec[1]))

    def record_key_unlock(self, key_color: int, unlocked_door_color: int) -> None:
        if key_color not in self.key_to_door_colors:
            self.key_to_door_colors[int(key_color)] = set()
        self.key_to_door_colors[int(key_color)].add(int(unlocked_door_color))

    def freeze_on_level_clear(self) -> None:
        if self.displacement_vectors:
            self.frozen = True
''', encoding="utf-8")

(PO_DIR / "world_model" / "hypothesis_manager.py").write_text(r'''from typing import Dict, List, Optional, Set, Tuple


class HypothesisManager:
    def __init__(self):
        self.probe_order: List[int] = [1, 3, 2, 4]
        self.tested_actions: Dict[int, int] = {}
        self.color_semantics: Dict[int, str] = {}
        self.portal_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.interacted_positions: Set[Tuple[int, int]] = set()
        self.collected_keys: Set[int] = set()

    def reset_level(self, keep_semantics: bool = True) -> None:
        self.tested_actions.clear()
        self.portal_map.clear()
        self.interacted_positions.clear()
        self.collected_keys.clear()
        if not keep_semantics:
            self.color_semantics.clear()

    def record_color_role(self, color: int, role: str) -> None:
        self.color_semantics[int(color)] = role

    def register_portal_jump(self, entry_rc: Tuple[int, int], exit_rc: Tuple[int, int]) -> None:
        self.portal_map[entry_rc] = exit_rc
        self.portal_map[exit_rc] = entry_rc

    def next_probe(
        self,
        available_moves: List[int],
        known_vectors: Dict[int, Tuple[int, int]],
        schema_frozen: bool = False,
    ) -> Optional[int]:
        if schema_frozen and len(known_vectors) >= 2:
            return None
        for act in self.probe_order:
            if act in available_moves and act not in known_vectors:
                if self.tested_actions.get(act, 0) < 1:
                    self.tested_actions[act] = self.tested_actions.get(act, 0) + 1
                    return act
        return None
''', encoding="utf-8")

(PO_DIR / "world_model" / "context_manager.py").write_text(r'''from collections import deque
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
''', encoding="utf-8")

(PO_DIR / "world_model" / "ast_sandbox.py").write_text(r'''import ast
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

FORBIDDEN_AST_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Delete)
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals"}
SAFE_BUILTINS: Dict[str, Any] = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "int": int, "float": float, "bool": bool, "str": str, "tuple": tuple,
    "list": list, "set": set, "dict": dict, "enumerate": enumerate,
    "zip": zip, "sum": sum, "any": any, "all": all, "round": round, "sorted": sorted,
}


class ASTSandbox:
    @staticmethod
    def validate_and_compile(code_str: str, fn_name: str = "predict_next_grid") -> Optional[Callable]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_AST_NODES):
                    return None
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    return None
            safe_globals: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "np": np}
            local_ns: Dict[str, Any] = {}
            exec(compile(tree, filename="<cegis_world_model>", mode="exec"), safe_globals, local_ns)
            fn = local_ns.get(fn_name)
            return fn if callable(fn) else None
        except Exception:
            return None

    @staticmethod
    def evaluate_retrodiction(
        fn: Callable,
        transitions: List[Tuple[np.ndarray, int, Dict[str, Any], np.ndarray]],
        hud_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[str]]:
        if not transitions:
            return 0.0, None
        passed = 0
        counterexample = None
        window = transitions[-14:]
        for pre_g, act_id, data, post_g in window:
            try:
                pred = fn(pre_g.copy(), int(act_id), data or {})
                if isinstance(pred, np.ndarray) and pred.shape == post_g.shape:
                    non_abstained = pred != -1
                    if np.mean(non_abstained) < 0.70:
                        counterexample = f"Action {act_id}: coverage too low"
                        continue
                    diff = (pred != post_g) & non_abstained
                    if hud_mask is not None and hud_mask.shape == diff.shape:
                        diff = diff & (~hud_mask)
                    else:
                        diff[:4, :] = False; diff[-4:, :] = False; diff[:, :4] = False; diff[:, -4:] = False
                    n_err = int(np.sum(diff))
                    if n_err == 0:
                        passed += 1
                    elif counterexample is None:
                        rs, cs = np.where(diff)
                        r0, c0 = int(rs[0]), int(cs[0])
                        counterexample = f"Action {act_id}: {n_err} px mismatch at ({r0},{c0})"
                else:
                    counterexample = f"Action {act_id}: invalid return shape"
            except Exception as e:
                counterexample = f"Action {act_id} raised {type(e).__name__}: {e}"
        return passed / float(len(window)), counterexample


class CodeActREPL:
    def __init__(self):
        self.queued_plan: List[Any] = []
        self.workspace_vars: Dict[str, Any] = {}

    def reset_level(self) -> None:
        self.queued_plan.clear()

    def execute_codeact(
        self,
        code_str: str,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        av_pos: Optional[Tuple[int, int]],
        bg_color: int,
        entities: List[Any],
        cavities: List[Tuple[int, int]],
        vectors: Dict[int, Tuple[int, int]],
        avail_actions: List[int],
        history_records: List[Dict[str, Any]],
        portals: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Optional[List[Any]]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_AST_NODES):
                    return None
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    return None
        except Exception:
            return None

        self.queued_plan = []

        def _queue_actions(acts: List[Any]) -> None:
            for a in acts[:16]:
                self.queued_plan.append(a)

        def _click_xy(x: int, y: int) -> None:
            self.queued_plan.append((6, {"x": int(x), "y": int(y)}))

        def _find_color_centroids(color: int) -> List[Tuple[int, int]]:
            pts = np.argwhere(grid == int(color))
            return [(int(pts[:, 0].mean()), int(pts[:, 1].mean()))] if len(pts) > 0 else []

        def _grep_history(pattern: str) -> List[Dict[str, Any]]:
            p = str(pattern).lower()
            return [r for r in history_records if p in str(r).lower()][-8:]

        def _find_causal_triggers() -> List[Dict[str, Any]]:
            return [r for r in history_records if r.get("remote_triggers")]

        env_vars: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "np": np,
            "grid": grid.copy(),
            "prev_grid": prev_grid.copy() if prev_grid is not None else grid.copy(),
            "av_pos": av_pos,
            "bg_color": int(bg_color),
            "entities": [{"id": e.id, "color": e.color, "mass": e.mass, "rc": (int(round(e.centroid[0])), int(round(e.centroid[1])))} for e in entities],
            "cavities": list(cavities),
            "vectors": dict(vectors),
            "portals": dict(portals or {}),
            "avail_actions": list(avail_actions),
            "history": list(history_records[-16:]),
            "workspace": self.workspace_vars,
            "queue_actions": _queue_actions,
            "click_xy": _click_xy,
            "find_color_centroids": _find_color_centroids,
            "grep_history": _grep_history,
            "find_causal_triggers": _find_causal_triggers,
        }
        local_ns: Dict[str, Any] = {}
        try:
            exec(compile(tree, filename="<duck_codeact_repl>", mode="exec"), env_vars, local_ns)
            if "solve_turn" in local_ns and callable(local_ns["solve_turn"]):
                res = local_ns["solve_turn"](env_vars)
                if isinstance(res, list) and res:
                    return res[:16]
                if isinstance(res, int):
                    return [res]
            if self.queued_plan:
                return list(self.queued_plan)
            if "plan" in local_ns and isinstance(local_ns["plan"], list) and local_ns["plan"]:
                return local_ns["plan"][:16]
        except Exception:
            return None
        return None
''', encoding="utf-8")

(PO_DIR / "world_model" / "retrodict_synthesizer.py").write_text(r'''import heapq
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.core.dead_signature import DeadSignatureTable, DynamicHUDMasker, compute_arena_hash
from project_override.world_model.ast_sandbox import ASTSandbox

logger = logging.getLogger(__name__)


def _install_pure_pytorch_awq_fallback() -> None:
    try:
        import awq  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import torch
        import torch.nn as nn
        from transformers.quantizers import AutoHfQuantizer, HfQuantizer

        class TorchAWQLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 128):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.group_size = group_size
                self.register_buffer("qweight", torch.zeros((in_features, out_features // 8), dtype=torch.int32))
                self.register_buffer("qzeros", torch.zeros((in_features // group_size, out_features // 8), dtype=torch.int32))
                self.register_buffer("scales", torch.zeros((in_features // group_size, out_features), dtype=torch.float16))
                self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float16)) if bias else setattr(self, "bias", None)
                self.register_buffer("_shifts", torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32).view(1, 1, 8), persistent=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = torch.bitwise_and(torch.bitwise_right_shift(self.qweight.unsqueeze(-1), self._shifts), 0x0F).to(torch.uint8).reshape(self.in_features, self.out_features)
                z = torch.bitwise_and(torch.bitwise_right_shift(self.qzeros.unsqueeze(-1), self._shifts), 0x0F).to(torch.uint8).reshape(self.in_features // self.group_size, self.out_features)
                w_fp16 = (w.to(torch.float16) - z.repeat_interleave(self.group_size, dim=0).to(torch.float16)) * self.scales.repeat_interleave(self.group_size, dim=0)
                out = torch.matmul(x.to(torch.float16), w_fp16)
                return (out + self.bias if self.bias is not None else out).to(x.dtype)

        class PureTorchAwqQuantizer(HfQuantizer):
            requires_calibration = False
            required_packages = []

            def __init__(self, quantization_config: Any, **kwargs: Any):
                super().__init__(quantization_config, **kwargs)
                self.group_size = getattr(quantization_config, "group_size", 128)

            def validate_environment(self, *args: Any, **kwargs: Any) -> None:
                return

            def update_torch_dtype(self, torch_dtype: Any) -> Any:
                return torch.float16

            def _process_model_before_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
                def _replace(module: nn.Module, prefix: str = "") -> None:
                    for name, child in list(module.named_children()):
                        full = f"{prefix}.{name}" if prefix else name
                        if isinstance(child, nn.Linear) and "lm_head" not in full and child.in_features % self.group_size == 0 and child.out_features % 8 == 0:
                            setattr(module, name, TorchAWQLinear(child.in_features, child.out_features, bias=child.bias is not None, group_size=self.group_size))
                        else:
                            _replace(child, full)
                _replace(model)

            def _process_model_after_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
                return

            @property
            def is_trainable(self) -> bool:
                return False

            @property
            def is_serializable(self) -> bool:
                return True

        AutoHfQuantizer.register("awq", PureTorchAwqQuantizer, exist_ok=True)
    except Exception:
        pass


class OfflineLLMRuntime:
    _INSTANCE: Optional["OfflineLLMRuntime"] = None
    _LOAD_LOCK = threading.Lock()
    _INFERENCE_LOCK = threading.Lock()

    CANDIDATE_PATHS = [
        "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen-2.5-coder-32b-instruct-awq",
    ]

    def __init__(self):
        self.model_path: Optional[str] = self._resolve_model_path()
        self.model: Any = None
        self.tokenizer: Any = None
        self.is_loaded: bool = False
        self.load_attempted: bool = False

    @classmethod
    def get_instance(cls) -> "OfflineLLMRuntime":
        if cls._INSTANCE is None:
            with cls._LOAD_LOCK:
                if cls._INSTANCE is None:
                    cls._INSTANCE = cls()
        return cls._INSTANCE

    def _resolve_model_path(self) -> Optional[str]:
        env_p = os.environ.get("ARC_LOCAL_MODEL_PATH")
        if env_p and Path(env_p).exists():
            return env_p
        for p in self.CANDIDATE_PATHS:
            if Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                if "qwen" in str(cfg).lower() or "coder" in str(cfg).lower():
                    return str(cfg.parent)
        return None

    def ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True
        if self.load_attempted or self.model_path is None:
            return False
        with self._LOAD_LOCK:
            if self.is_loaded:
                return True
            self.load_attempted = True
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                _install_pure_pytorch_awq_fallback()
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
                max_mem = {i: "14GiB" for i in range(torch.cuda.device_count())}
                max_mem["cpu"] = "24GiB"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, device_map="auto", max_memory=max_mem, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True
                )
                self.model.eval()
                self.is_loaded = True
                return True
            except Exception as e:
                logger.warning(f"OfflineLLMRuntime fallback to symbolic CEGIS ({e})")
                self.is_loaded = False
                return False

    def generate_text(self, prompt: str, max_new_tokens: int = 210, timeout_s: float = 14.0) -> Optional[str]:
        if not self.ensure_loaded() or not self._INFERENCE_LOCK.acquire(timeout=2.0):
            return None
        try:
            import torch
            t0 = time.perf_counter()
            if (time.perf_counter() - t0) > timeout_s:
                return None
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(self.tokenizer, "apply_chat_template") else prompt
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception:
            return None
        finally:
            self._INFERENCE_LOCK.release()

    def infer_subgoal(self, prompt: str) -> Optional[Dict[str, Any]]:
        resp = self.generate_text(prompt, max_new_tokens=160)
        if not resp:
            return None
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


class RetrodictWorldModelSynthesizer:
    def __init__(self):
        self.transitions: List[Tuple[np.ndarray, int, Dict[str, Any], np.ndarray]] = []
        self.verified_fn: Optional[Callable] = None
        self.verified_code: Optional[str] = None
        self.verified_accuracy: float = 0.0
        self.needs_repair: bool = False

    def reset_level(self, keep_verified: bool = True) -> None:
        self.transitions.clear()
        self.needs_repair = False
        if not keep_verified:
            self.verified_fn = None
            self.verified_code = None
            self.verified_accuracy = 0.0

    def record_transition(
        self,
        pre_grid: np.ndarray,
        action_id: int,
        data: Dict[str, Any],
        post_grid: np.ndarray,
        hud_mask: Optional[np.ndarray] = None,
    ) -> None:
        if not np.array_equal(pre_grid, post_grid):
            self.transitions.append((pre_grid.copy(), int(action_id), dict(data or {}), post_grid.copy()))
            if len(self.transitions) > 24:
                self.transitions.pop(0)
            if self.verified_fn is not None:
                acc, _ = ASTSandbox.evaluate_retrodiction(self.verified_fn, [self.transitions[-1]], hud_mask=hud_mask)
                if acc < 1.0:
                    self.needs_repair = True

    def _build_archetype_candidates(
        self,
        floor_color: int,
        avatar_colors: Set[int],
        wall_colors: Set[int],
        box_colors: Set[int],
        vectors: Dict[int, Tuple[int, int]],
        portals: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        key_to_door_colors: Optional[Dict[int, Set[int]]] = None,
    ) -> List[str]:
        fl = int(floor_color)
        av_repr = repr(set(int(c) for c in avatar_colors))
        wl_repr = repr(set(int(c) for c in wall_colors))
        bx_repr = repr(set(int(c) for c in box_colors))
        vc_repr = repr(vectors)
        pt_repr = repr(dict(portals or {}))
        kd_repr = repr({int(k): list(v) for k, v in (key_to_door_colors or {}).items()})

        c1_cardinal = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    floor_c = {fl}; av_cols = {av_repr}; wall_cols = {wl_repr}; box_cols = {bx_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0: return grid
    if action == 5:
        nxt = grid.copy()
        r0, c0 = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r0 + dr, c0 + dc
                if 4 <= nr < h - 4 and 4 <= nc < w - 4 and int(nxt[nr, nc]) == floor_c:
                    nxt[nr, nc] = min(av_cols)
        return nxt
    if action not in vecs: return grid
    dr, dc = vecs[action]
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (4 <= nr < h - 4 and 4 <= nc < w - 4) or int(grid[nr, nc]) in wall_cols:
            return grid
    nxt = grid.copy()
    for r, c in pts: nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if int(grid[nr, nc]) in box_cols:
            br, bc = int(nr + dr), int(nc + dc)
            if 4 <= br < h - 4 and 4 <= bc < w - 4 and int(grid[br, bc]) not in wall_cols:
                nxt[br, bc] = grid[nr, nc]
        nxt[nr, nc] = grid[int(r), int(c)]
    return nxt
"""
        c2_toroidal = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    if action not in vecs: return grid
    dr, dc = vecs[action]
    floor_c = {fl}; av_cols = {av_repr}; wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0: return grid
    nxt = grid.copy()
    for r, c in pts:
        nr = 4 + ((int(r + dr) - 4) % (h - 8))
        nc = 4 + ((int(c + dc) - 4) % (w - 8))
        if int(grid[nr, nc]) in wall_cols: return grid
    for r, c in pts: nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nr = 4 + ((int(r + dr) - 4) % (h - 8))
        nc = 4 + ((int(c + dc) - 4) % (w - 8))
        nxt[nr, nc] = grid[int(r), int(c)]
    return nxt
"""
        c3_slide = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    if action not in vecs: return grid
    dr, dc = vecs[action]
    ur = 0 if dr == 0 else (1 if dr > 0 else -1)
    uc = 0 if dc == 0 else (1 if dc > 0 else -1)
    floor_c = {fl}; av_cols = {av_repr}; wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0: return grid
    k = 0
    while k < 56:
        if any(not (4 <= int(r + ur * (k + 1)) < h - 4 and 4 <= int(c + uc * (k + 1)) < w - 4) or int(grid[int(r + ur * (k + 1)), int(c + uc * (k + 1))]) in wall_cols for r, c in pts):
            break
        k += 1
    if k == 0: return grid
    nxt = grid.copy()
    for r, c in pts: nxt[int(r), int(c)] = floor_c
    for r, c in pts: nxt[int(r + ur * k), int(c + uc * k)] = grid[int(r), int(c)]
    return nxt
"""
        c4_gravity = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    if action not in vecs: return grid
    dr, dc = vecs[action]
    floor_c = {fl}; av_cols = {av_repr}; wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0: return grid
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (4 <= nr < h - 4 and 4 <= nc < w - 4) or int(grid[nr, nc]) in wall_cols: return grid
    nxt = grid.copy()
    for r, c in pts: nxt[int(r), int(c)] = floor_c
    for r, c in pts: nxt[int(r + dr), int(c + dc)] = grid[int(r), int(c)]
    for _ in range(4):
        pts2 = np.argwhere(np.isin(nxt, list(av_cols)))
        if len(pts2) == 0 or not all(int(r + 1) < h - 4 and (int(nxt[int(r + 1), int(c)]) == floor_c or int(nxt[int(r + 1), int(c)]) in av_cols) for r, c in pts2):
            break
        prev_nxt = nxt.copy()
        for r, c in pts2: nxt[int(r), int(c)] = floor_c
        for r, c in pts2: nxt[int(r + 1), int(c)] = prev_nxt[int(r), int(c)]
    return nxt
"""
        c5_portal = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}; portals = {pt_repr}
    if action not in vecs: return grid
    dr, dc = vecs[action]
    floor_c = {fl}; av_cols = {av_repr}; wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0: return grid
    r0, c0 = int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))
    nr, nc = r0 + dr, c0 + dc
    for p_in, p_out in portals.items():
        if abs(nr - p_in[0]) <= 2 and abs(nc - p_in[1]) <= 2:
            dr = p_out[0] - r0; dc = p_out[1] - c0; break
    nxt = grid.copy()
    for r, c in pts:
        tr, tc = int(r + dr), int(c + dc)
        if not (4 <= tr < h - 4 and 4 <= tc < w - 4) or int(grid[tr, tc]) in wall_cols: return grid
    for r, c in pts: nxt[int(r), int(c)] = floor_c
    for r, c in pts: nxt[int(r + dr), int(c + dc)] = grid[int(r), int(c)]
    return nxt
"""
        c6_key_door = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}; kd_map = {kd_repr}
    if action not in vecs: return grid
    dr, dc = vecs[action]
    floor_c = {fl}; av_cols = {av_repr}; wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0: return grid
    stepped_colors = set()
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (4 <= nr < h - 4 and 4 <= nc < w - 4) or int(grid[nr, nc]) in wall_cols: return grid
        stepped_colors.add(int(grid[nr, nc]))
    nxt = grid.copy()
    for sc in stepped_colors:
        if sc in kd_map:
            for door_c in kd_map[sc]: nxt[nxt == door_c] = floor_c
    for r, c in pts: nxt[int(r), int(c)] = floor_c
    for r, c in pts: nxt[int(r + dr), int(c + dc)] = grid[int(r), int(c)]
    return nxt
"""
        return [c1_cardinal, c2_toroidal, c3_slide, c4_gravity, c5_portal, c6_key_door]

    def synthesize_world_model(
        self,
        floor_color: Optional[int],
        avatar_colors: Set[int],
        wall_colors: Set[int],
        box_colors: Set[int],
        vectors: Dict[int, Tuple[int, int]],
        hud_mask: Optional[np.ndarray] = None,
        portals: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        key_to_door_colors: Optional[Dict[int, Set[int]]] = None,
    ) -> Optional[Callable]:
        if floor_color is None or not avatar_colors or not vectors:
            return None
        incumbent_acc = -1.0
        if self.verified_fn is not None and len(self.transitions) >= 2:
            incumbent_acc, _ = ASTSandbox.evaluate_retrodiction(self.verified_fn, self.transitions, hud_mask=hud_mask)
        best_fn = self.verified_fn
        best_acc = incumbent_acc
        best_ce = None
        best_code = self.verified_code

        for code_cand in self._build_archetype_candidates(
            floor_color, avatar_colors, wall_colors, box_colors, vectors, portals=portals, key_to_door_colors=key_to_door_colors
        ):
            fn = ASTSandbox.validate_and_compile(code_cand)
            if fn is not None:
                if len(self.transitions) >= 2:
                    acc, ce = ASTSandbox.evaluate_retrodiction(fn, self.transitions, hud_mask=hud_mask)
                    if acc > best_acc:
                        best_acc = acc; best_fn = fn; best_ce = ce; best_code = code_cand
                    if acc >= 0.92:
                        self.verified_fn = fn; self.verified_code = code_cand; self.verified_accuracy = acc; self.needs_repair = False
                        return fn
                elif best_fn is None:
                    best_fn = fn; best_code = code_cand; best_acc = 0.5

        llm = OfflineLLMRuntime.get_instance()
        if llm.ensure_loaded() and len(self.transitions) >= 2:
            curr_ce = best_ce
            for _ in range(2):
                cegis_prompt = (
                    "Write ONLY a pure Python function `def predict_next_grid(grid, action, data=None):` "
                    "using `np` (no imports) that predicts the next 64x64 int8 grid (may set uncertain HUD cells to -1).\n"
                    f"Induced parameters: floor={floor_color}, avatar={avatar_colors}, walls={wall_colors}, "
                    f"boxes={box_colors}, vectors={vectors}, portals={portals}.\n"
                    f"Counterexample on current model: {curr_ce}\n"
                    "Return ONLY valid Python code inside ```python ... ```."
                )
                resp = llm.generate_text(cegis_prompt, max_new_tokens=240)
                if not resp:
                    break
                m = re.search(r"```(?:python)?\s*(def predict_next_grid.*?)```", resp, re.DOTALL)
                cand_code = m.group(1).strip() if m else resp.strip()
                llm_fn = ASTSandbox.validate_and_compile(cand_code)
                if llm_fn is not None:
                    acc2, ce2 = ASTSandbox.evaluate_retrodiction(llm_fn, self.transitions, hud_mask=hud_mask)
                    if acc2 > best_acc:
                        best_acc = acc2; best_fn = llm_fn; best_code = cand_code
                    if acc2 >= 0.85 and acc2 >= incumbent_acc:
                        self.verified_fn = llm_fn; self.verified_code = cand_code; self.verified_accuracy = acc2; self.needs_repair = False
                        return llm_fn
                    curr_ce = ce2

        if best_fn is not None:
            self.verified_fn = best_fn; self.verified_code = best_code; self.verified_accuracy = max(0.0, best_acc); self.needs_repair = False
        return best_fn

    def plan_iw_lookahead(
        self,
        start_grid: np.ndarray,
        available_actions: List[int],
        target_pos: Optional[Tuple[int, int]],
        avatar_colors: Set[int],
        dead_table: Optional[DeadSignatureTable] = None,
        hud_masker: Optional[DynamicHUDMasker] = None,
        level_idx: int = 0,
        max_expansions: int = 160,
    ) -> Optional[List[int]]:
        if self.verified_fn is None or not avatar_colors:
            return None
        moves = [a for a in available_actions if a in (1, 2, 3, 4, 5)]
        if not moves:
            return None
        seen_atoms: Set[Tuple[int, int, int]] = set()
        seen_pairs: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()

        def _evaluate_novelty(g: np.ndarray) -> int:
            nov = 0
            fg_atoms = []
            for r in range(4, g.shape[0] - 4, 3):
                for c in range(4, g.shape[1] - 4, 3):
                    val = int(g[r, c])
                    if val < 0:
                        continue
                    atom = (r, c, val)
                    fg_atoms.append(atom)
                    if atom not in seen_atoms:
                        seen_atoms.add(atom); nov += 2
            if nov == 0 and len(fg_atoms) <= 48:
                for i in range(0, len(fg_atoms), 2):
                    for j in range(i + 1, min(i + 4, len(fg_atoms))):
                        p = (fg_atoms[i], fg_atoms[j])
                        if p not in seen_pairs:
                            seen_pairs.add(p); nov += 1
            return nov

        _evaluate_novelty(start_grid)
        tie_breaker = 0
        open_heap: List[Tuple[float, int, np.ndarray, List[int]]] = [(0.0, tie_breaker, start_grid.copy(), [])]
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, _, g, path = heapq.heappop(open_heap)
            expansions += 1
            if len(path) >= 16:
                continue
            g_hash = compute_arena_hash(g, bezel_px=4, hud_masker=hud_masker) if dead_table is not None else 0
            for a in moves:
                if dead_table is not None and dead_table.is_dead(g_hash, a, 0, 0, level_idx):
                    continue
                try:
                    nxt_g = self.verified_fn(g, a, {})
                except Exception:
                    continue
                if np.array_equal(nxt_g, g):
                    continue
                new_path = path + [a]
                if target_pos is not None:
                    tr, tc = target_pos
                    if 0 <= tr < nxt_g.shape[0] and 0 <= tc < nxt_g.shape[1] and int(nxt_g[tr, tc]) in avatar_colors:
                        return new_path
                nov = _evaluate_novelty(nxt_g)
                if nov > 0:
                    dist_pen = 0.0
                    if target_pos is not None:
                        pts = np.argwhere(np.isin(nxt_g, list(avatar_colors)))
                        if len(pts) > 0:
                            dist_pen = (abs(float(pts[:, 0].mean()) - target_pos[0]) + abs(float(pts[:, 1].mean()) - target_pos[1])) * 0.15
                    tie_breaker += 1
                    heapq.heappush(open_heap, (len(new_path) + dist_pen - (nov * 1.5), tie_breaker, nxt_g, new_path))
        return None
''', encoding="utf-8")

# =========================================================================
# 12. project_override/orchestration/general_agent.py (Complete V15)
# =========================================================================
(PO_DIR / "orchestration" / "general_agent.py").write_text(r'''from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.core.circuit_engine import AC3CircuitSolver
from project_override.core.combinatorial_engine import GF2ToggleSolver, SplitPanelSymmetrySolver
from project_override.core.dead_signature import DeadSignatureTable, DynamicHUDMasker, compute_arena_hash
from project_override.core.universal_engine import SystemPacer
from project_override.core.visual_goal_induction import JordanCurveCavityInducer, LatentMemoryMatcher
from project_override.perception.observation_processor import ObservationProcessor
from project_override.planning.symbolic_planners import (
    MacroActionQueue,
    astar_grid,
    bfs_frontier_explore,
    ice_slide_bfs,
    sokoban_bfs,
)
from project_override.world_model.ast_sandbox import CodeActREPL
from project_override.world_model.context_manager import ContextManager
from project_override.world_model.hypothesis_manager import HypothesisManager
from project_override.world_model.metagame_schema import MetagameSchema
from project_override.world_model.retrodict_synthesizer import (
    OfflineLLMRuntime,
    RetrodictWorldModelSynthesizer,
)


class WorkspaceCritic:
    @staticmethod
    def audit_plan(
        plan: List[Any],
        curr_hash: int,
        dead_table: DeadSignatureTable,
        level_idx: int = 0,
    ) -> Optional[List[Any]]:
        if not plan:
            return None
        first = plan[0]
        if isinstance(first, tuple) and len(first) == 2:
            act_id, data = int(first[0]), dict(first[1])
            cx, cy = int(data.get("x", 0)), int(data.get("y", 0))
        else:
            act_id, cx, cy = int(first), 0, 0
        if dead_table.is_dead(curr_hash, act_id, cx, cy, level_idx):
            return None
        return plan


class RecursiveSubAgentController:
    BUDGETS = {
        "PROBE": 2,
        "INTERACT": 3,
        "MOORE_PLANNER": 16,
        "CODEACT_EXPLORER": 5,
    }

    def __init__(self):
        self.active_role: str = "PROBE"
        self.remaining_budget: int = 2
        self.role_failures: Dict[str, int] = {}

    def reset_level(self, schema_frozen: bool = False) -> None:
        self.role_failures.clear()
        if schema_frozen:
            self.active_role = "MOORE_PLANNER"
            self.remaining_budget = self.BUDGETS["MOORE_PLANNER"]
        else:
            self.active_role = "PROBE"
            self.remaining_budget = self.BUDGETS["PROBE"]

    def delegate(self, role: str) -> None:
        self.active_role = role
        self.remaining_budget = self.BUDGETS.get(role, 6)

    def on_step_outcome(self, delta_omega: int) -> None:
        self.remaining_budget -= 1
        if delta_omega == 0:
            self.role_failures[self.active_role] = self.role_failures.get(self.active_role, 0) + 1
            self.remaining_budget = 0


class GeneralAgent:
    def __init__(self, game_id: str = "unknown"):
        self.game_id = game_id
        self.processor = ObservationProcessor(bezel_px=4)
        self.hud_masker = DynamicHUDMasker(shape=(64, 64), default_bezel=4)
        self.dead_table = DeadSignatureTable()
        self.macro_queue = MacroActionQueue()
        self.subagent_ctrl = RecursiveSubAgentController()
        self.critic = WorkspaceCritic()
        self.schema = MetagameSchema()
        self.hypotheses = HypothesisManager()
        self.context = ContextManager(high_water_tokens=1900, low_water_tokens=1100, game_id=game_id)
        self.synthesizer = RetrodictWorldModelSynthesizer()
        self.codeact_repl = CodeActREPL()
        self.memory_matcher = LatentMemoryMatcher()
        self.gf2_solver = GF2ToggleSolver()
        self.pacer = SystemPacer(max_level_compute_s=45.0)

        self.current_level: int = 0
        self.step_in_level: int = 0
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_hash: Optional[int] = None
        self.prev_av_pos: Optional[Tuple[int, int]] = None
        self.last_known_av_pos: Optional[Tuple[int, int]] = None
        self.last_action_id: Optional[int] = None
        self.last_click_xy: Tuple[int, int] = (0, 0)
        self.last_delta_omega: Optional[int] = None

        self.learned_floor_color: Optional[int] = None
        self.learned_avatar_colors: Set[int] = set()
        self.learned_avatar_color: Optional[int] = None
        self.learned_game_modality: str = "NAV"
        self.confirmed_wall_colors: Set[int] = set()
        self.confirmed_hazard_colors: Set[int] = set()
        self.confirmed_box_colors: Set[int] = set()
        self.priority_remote_subgoals: List[Tuple[int, int]] = []
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.visited_goals: Set[Tuple[int, int]] = set()
        self.transition_history: List[Tuple[np.ndarray, int, np.ndarray, int]] = []

    def on_game_over(self, final_grid: Optional[np.ndarray] = None) -> None:
        if self.prev_hash is not None and self.last_action_id is not None:
            cx, cy = self.last_click_xy if self.last_action_id == 6 else (0, 0)
            self.dead_table.record_death(self.prev_hash, self.last_action_id, cx, cy, self.current_level)
            if (
                self.prev_grid is not None
                and self.prev_av_pos is not None
                and self.last_action_id in self.schema.displacement_vectors
            ):
                dr, dc = self.schema.displacement_vectors[self.last_action_id]
                nr, nc = self.prev_av_pos[0] + dr, self.prev_av_pos[1] + dc
                h, w = self.prev_grid.shape
                if 4 <= nr < h - 4 and 4 <= nc < w - 4:
                    lethal_color = int(self.prev_grid[nr, nc])
                    if lethal_color != self.learned_floor_color and lethal_color not in self.learned_avatar_colors:
                        self.confirmed_hazard_colors.add(lethal_color)
                        self.hypotheses.record_color_role(lethal_color, "HAZARD")
                        self.context.record_reflexion(
                            f"GAME_OVER stepping onto hazard color {lethal_color} via action {self.last_action_id}"
                        )
        self.macro_queue.clear()
        self.prev_grid = None
        self.prev_hash = None
        self.last_known_av_pos = None
        self.last_action_id = None
        self.last_delta_omega = None
        self.visited_goals.clear()

    def _on_level_advance(self, new_level: int) -> None:
        self.current_level = new_level
        self.step_in_level = 0
        self.prev_grid = None
        self.prev_hash = None
        self.prev_av_pos = None
        self.last_known_av_pos = None
        self.last_action_id = None
        self.last_delta_omega = None
        self.macro_queue.clear()
        self.hud_masker.reset_level()
        self.dead_table.reset_level(clear_deaths=True)
        self.synthesizer.reset_level(keep_verified=True)
        self.codeact_repl.reset_level()
        self.hypotheses.reset_level(keep_semantics=True)
        self.context.history_log.reset_level()
        self.memory_matcher.reset()
        self.gf2_solver.reset()
        self.pacer.reset_level()
        self.priority_remote_subgoals.clear()
        self.visited_cells.clear()
        self.visited_goals.clear()
        self.transition_history.clear()
        self.schema.freeze_on_level_clear()
        self.subagent_ctrl.reset_level(schema_frozen=self.schema.frozen)

    def _locate_avatar_centroid(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        if not self.learned_avatar_colors:
            return None
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        mask = np.zeros((h, w), dtype=bool)
        for c in self.learned_avatar_colors:
            mask |= (arr == c)
        if arr.shape == self.hud_masker.shape:
            mask &= (~self.hud_masker.hud_mask)
        pts = np.argwhere(mask)
        if len(pts) == 0:
            return None
        if self.last_known_av_pos is not None:
            anchor = np.array(self.last_known_av_pos, dtype=float)
            dists = np.linalg.norm(pts - anchor, axis=1)
            local_pts = pts[dists <= 14.0]
            if len(local_pts) > 0:
                pts = local_pts
        return (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))

    def _induce_blocking_wall(self, grid: np.ndarray, failed_action: int) -> None:
        if failed_action not in self.schema.displacement_vectors:
            return
        av_pos = self._locate_avatar_centroid(grid)
        if av_pos is None:
            return
        dr, dc = self.schema.displacement_vectors[failed_action]
        nr, nc = av_pos[0] + dr, av_pos[1] + dc
        h, w = grid.shape
        if 4 <= nr < h - 4 and 4 <= nc < w - 4:
            c = int(grid[nr, nc])
            if (
                c != self.learned_floor_color
                and c not in self.learned_avatar_colors
                and c not in self.confirmed_box_colors
            ):
                self.confirmed_wall_colors.add(c)
                self.hypotheses.record_color_role(c, "WALL")

    def _retroactive_wall_scan(self) -> None:
        for pre_g, act_id, _, delta in self.transition_history:
            if delta == 0 and act_id in self.schema.displacement_vectors:
                self._induce_blocking_wall(pre_g, act_id)

    def _build_obstacle_set(self, grid: np.ndarray) -> Set[Tuple[int, int]]:
        obstacles: Set[Tuple[int, int]] = set()
        blocked_colors = self.confirmed_wall_colors | self.confirmed_hazard_colors
        if not blocked_colors:
            return obstacles
        arr = np.asarray(grid, dtype=np.int8)
        for c in blocked_colors:
            for r, col in np.argwhere(arr == c):
                obstacles.add((int(r), int(col)))
        return obstacles

    def _find_candidate_goals(self, grid: np.ndarray, av_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        bg = self.processor.induce_background_color(grid, self.learned_floor_color)
        entities = self.processor.extract_entities(grid, bg_color=bg, ignore_bezel=True)
        strides = [max(abs(v[0]), abs(v[1])) for v in self.schema.displacement_vectors.values() if v != (0, 0)]
        s = max(1, min(strides)) if strides else 1

        ordered_goals: List[Tuple[int, int]] = []
        for rem_rc in self.priority_remote_subgoals:
            if (rem_rc[0] // s, rem_rc[1] // s) not in self.visited_goals:
                ordered_goals.append(rem_rc)

        candidates: List[Tuple[float, Tuple[int, int]]] = []
        blocked_colors = self.learned_avatar_colors | self.confirmed_wall_colors | self.confirmed_hazard_colors
        for ent in entities:
            if ent.color in blocked_colors or ent.mass > 400:
                continue
            gr, gc = int(round(ent.centroid[0])), int(round(ent.centroid[1]))
            norm_g = (gr // s, gc // s)
            if norm_g in self.visited_goals:
                continue
            dist = abs(gr - av_pos[0]) + abs(gc - av_pos[1])
            if dist >= s:
                role_bonus = -5.0 if self.hypotheses.color_semantics.get(ent.color) in ("KEY", "SWITCH", "GOAL") else 0.0
                score = ent.mass * 0.4 + dist * 0.25 + role_bonus
                candidates.append((score, (gr, gc)))

        for cr, cc in JordanCurveCavityInducer.find_cavities(grid, bg_color=bg, bezel_px=4)[:4]:
            norm_g = (cr // s, cc // s)
            if norm_g not in self.visited_goals:
                dist = abs(cr - av_pos[0]) + abs(cc - av_pos[1])
                candidates.append((dist * 0.2, (cr, cc)))

        candidates.sort(key=lambda x: x[0])
        ordered_goals.extend([pos for _, pos in candidates])
        return ordered_goals

    def _select_smart_click(self, grid: np.ndarray, arena_hash: int) -> Tuple[int, Dict[str, int]]:
        bg = self.processor.induce_background_color(grid, self.learned_floor_color)

        gf2_clicks = self.gf2_solver.try_solve_current_board(grid, bg_color=bg)
        if gf2_clicks:
            valid_gf2 = [
                (cx, cy) for cx, cy in gf2_clicks
                if not self.dead_table.is_dead(arena_hash, 6, cx, cy, self.current_level)
            ]
            if valid_gf2:
                first_xy = valid_gf2[0]
                if len(valid_gf2) > 1:
                    self.macro_queue.push_plan(
                        [(6, {"x": int(x), "y": int(y)}) for x, y in valid_gf2[1:]],
                        subagent_tag="GF2_SOLVER",
                    )
                self.last_click_xy = (int(first_xy[0]), int(first_xy[1]))
                return 6, {"x": int(first_xy[0]), "y": int(first_xy[1])}

        for r, c in SplitPanelSymmetrySolver.find_asymmetric_clicks(grid, bg_color=bg, bezel_px=4):
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        for r, c in JordanCurveCavityInducer.find_cavities(grid, bg_color=bg, bezel_px=4):
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        circuit_clicks = AC3CircuitSolver.detect_periodic_circuit_clicks(grid, bg_color=bg, bezel_px=4)
        if circuit_clicks:
            for r, c in circuit_clicks:
                x, y = int(c), int(r)
                if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                    self.last_click_xy = (x, y)
                    return 6, {"x": x, "y": y}

        entities = self.processor.extract_entities(grid, bg_color=bg, ignore_bezel=True)
        color_counts: Dict[int, int] = {}
        for e in entities:
            color_counts[e.color] = color_counts.get(e.color, 0) + 1

        ranked = sorted(entities, key=lambda e: (color_counts.get(e.color, 99), abs(e.mass - 9), e.id))
        for ent in ranked:
            r = int(round(ent.centroid[0]))
            c = int(round(ent.centroid[1]))
            if (r, c) not in ent.pixels and ent.pixels:
                r, c = ent.pixels[len(ent.pixels) // 2]
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        h, w = grid.shape
        for stride in (6, 4, 3):
            for r in range(6, h - 6, stride):
                for c in range(6, w - 6, stride):
                    x, y = int(c), int(r)
                    if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                        self.last_click_xy = (x, y)
                        return 6, {"x": x, "y": y}

        return 6, {"x": 32, "y": 32}

    def step(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int = 0,
        frame_sequence: Optional[List[np.ndarray]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        self.pacer.begin_step()
        try:
            return self._step_impl(grid, available_actions, levels_completed, frame_sequence)
        finally:
            self.pacer.end_step()

    def _step_impl(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int = 0,
        frame_sequence: Optional[List[np.ndarray]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        arr = np.asarray(grid, dtype=np.int8)
        if levels_completed > self.current_level:
            self._on_level_advance(levels_completed)

        if frame_sequence and len(frame_sequence) > 1:
            anim_info = self.processor.analyze_animation_frames(frame_sequence)
            if anim_info.get("is_continuous_slide") and self.learned_game_modality == "NAV":
                self.learned_game_modality = "SLIDE"

        detected_remote_triggers: List[Tuple[int, int]] = []

        if self.prev_grid is not None and self.last_action_id is not None and self.prev_hash is not None:
            self.hud_masker.observe_transition(self.prev_grid, arr)
            curr_hash = compute_arena_hash(arr, bezel_px=4, hud_masker=self.hud_masker)
            delta = self.processor.compute_delta_omega(self.prev_grid, arr, hud_mask=self.hud_masker.hud_mask)
            self.last_delta_omega = delta
            self.subagent_ctrl.on_step_outcome(delta)

            cx, cy = self.last_click_xy if self.last_action_id == 6 else (0, 0)
            self.dead_table.record_transition(
                self.prev_hash,
                self.last_action_id,
                curr_hash,
                delta,
                coord_x=cx,
                coord_y=cy,
                level_idx=self.current_level,
                post_grid=arr,
            )
            self.transition_history.append((self.prev_grid.copy(), self.last_action_id, arr.copy(), delta))
            if len(self.transition_history) > 24:
                self.transition_history.pop(0)
            self.synthesizer.record_transition(
                self.prev_grid,
                self.last_action_id,
                {"x": cx, "y": cy} if self.last_action_id == 6 else {},
                arr,
                hud_mask=self.hud_masker.hud_mask,
            )
            diff_note = self.processor.summarize_transition_diff(self.prev_grid, arr) if delta > 0 else "BLOCKED"

            if self.last_action_id == 6 and delta > 0:
                bg = self.processor.induce_background_color(arr, self.learned_floor_color)
                self.gf2_solver.record_click_effect(self.prev_grid, arr, self.last_click_xy)
                match_xy = self.memory_matcher.observe_reveal(self.prev_grid, arr, self.last_click_xy, bg)
                if match_xy is not None:
                    self.macro_queue.push_plan(
                        [(6, {"x": int(match_xy[0]), "y": int(match_xy[1])})],
                        subagent_tag="MEMORY_MATCH",
                    )

            if self.last_action_id in (1, 2, 3, 4):
                if delta > 0:
                    rigid = self.processor.infer_rigid_translation(
                        self.prev_grid,
                        arr,
                        self.learned_avatar_colors if self.learned_avatar_colors else None,
                        hud_mask=self.hud_masker.hud_mask,
                    )
                    if rigid is not None:
                        fl_c, av_colors, (dr, dc), post_cent, pushed_boxes, remote_triggers = rigid
                        detected_remote_triggers = list(remote_triggers)
                        self.learned_floor_color = fl_c
                        self.hypotheses.record_color_role(fl_c, "FLOOR")
                        self.learned_avatar_colors |= av_colors
                        for ac in av_colors:
                            self.hypotheses.record_color_role(ac, "AVATAR")
                        self.learned_avatar_color = min(self.learned_avatar_colors)
                        new_av_rc = (int(round(post_cent[0])), int(round(post_cent[1])))

                        expected_vec = self.schema.displacement_vectors.get(self.last_action_id)
                        if expected_vec is not None and self.prev_av_pos is not None:
                            exp_stride = max(abs(expected_vec[0]), abs(expected_vec[1]))
                            actual_jump = max(abs(dr), abs(dc))
                            if actual_jump > max(12, exp_stride * 2):
                                entry_rc = (self.prev_av_pos[0] + expected_vec[0], self.prev_av_pos[1] + expected_vec[1])
                                self.hypotheses.register_portal_jump(entry_rc, new_av_rc)
                            else:
                                self.schema.record_vector(self.last_action_id, (dr, dc))
                        else:
                            self.schema.record_vector(self.last_action_id, (dr, dc))

                        self.last_known_av_pos = new_av_rc
                        if pushed_boxes:
                            self.confirmed_box_colors |= pushed_boxes
                            for bc in pushed_boxes:
                                self.hypotheses.record_color_role(bc, "BOX")
                            self.learned_game_modality = "SOKOBAN"
                        if remote_triggers:
                            stepped_c = int(self.prev_grid[new_av_rc[0], new_av_rc[1]])
                            self.hypotheses.record_color_role(stepped_c, "KEY")
                            self.hypotheses.collected_keys.add(stepped_c)
                            for r_trig in remote_triggers:
                                door_c = int(self.prev_grid[r_trig[0], r_trig[1]])
                                if door_c in self.confirmed_wall_colors:
                                    self.confirmed_wall_colors.discard(door_c)
                                self.schema.record_key_unlock(stepped_c, door_c)
                                if r_trig not in self.priority_remote_subgoals:
                                    self.priority_remote_subgoals.insert(0, r_trig)
                            self.visited_goals.clear()
                        self._retroactive_wall_scan()
                        if max(abs(dr), abs(dc)) > 8 and not self.hypotheses.portal_map:
                            self.learned_game_modality = "SLIDE"
                        elif abs(dr) > 0 and abs(dc) > 0:
                            self.learned_game_modality = "KNIGHT"
                else:
                    self._induce_blocking_wall(arr, self.last_action_id)

            self.context.record_step(
                step=self.step_in_level,
                action_id=self.last_action_id,
                delta_omega=delta,
                note=diff_note,
                level_idx=self.current_level,
                pre_hash=self.prev_hash,
                post_hash=curr_hash,
                coord_xy=(cx, cy),
                av_pos=self.last_known_av_pos,
                remote_triggers=detected_remote_triggers,
            )
        else:
            curr_hash = compute_arena_hash(arr, bezel_px=4, hud_masker=self.hud_masker)

        avail = [int(a) for a in available_actions if int(a) != 0] if available_actions else [1, 2, 3, 4]
        self.prev_grid = arr.copy()
        self.prev_hash = curr_hash
        self.step_in_level += 1

        av_pos = self._locate_avatar_centroid(arr)
        if av_pos is not None:
            self.last_known_av_pos = av_pos
        self.prev_av_pos = av_pos
        strides = [max(abs(v[0]), abs(v[1])) for v in self.schema.displacement_vectors.values() if v != (0, 0)]
        s = max(1, min(strides)) if strides else 1

        if av_pos is not None:
            norm_av = (av_pos[0] // s, av_pos[1] // s)
            self.visited_cells.add(norm_av)
            self.visited_goals.add(norm_av)

        move_actions = [a for a in avail if a in (1, 2, 3, 4)]
        if not move_actions and 6 in avail:
            queued = self.macro_queue.pop_next(self.last_delta_omega)
            if queued is not None and queued[0] == 6:
                qx, qy = int(queued[1].get("x", 32)), int(queued[1].get("y", 32))
                if not self.dead_table.is_dead(curr_hash, 6, qx, qy, self.current_level):
                    self.last_action_id = 6
                    self.last_click_xy = (qx, qy)
                    return 6, {"x": qx, "y": qy}
            act_id, data = self._select_smart_click(arr, curr_hash)
            self.last_action_id = act_id
            return act_id, data

        if self.dead_table.detect_loop():
            self.macro_queue.clear()
            all_disc = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
            escape_plan = self.dead_table.bfs_to_unexplored_action(
                curr_hash, all_disc, self.current_level, self.last_action_id
            )
            if escape_plan:
                first_act = escape_plan[0]
                self.macro_queue.push_plan(escape_plan[1:], subagent_tag="LOOP_ESCAPE")
                self.last_action_id = first_act
                return first_act, {}

        queued = self.macro_queue.pop_next(self.last_delta_omega)
        if queued is not None:
            q_act, q_data = queued
            qx = int(q_data.get("x", 0)) if q_act == 6 else 0
            qy = int(q_data.get("y", 0)) if q_act == 6 else 0
            if not self.dead_table.is_dead(curr_hash, q_act, qx, qy, self.current_level):
                self.last_action_id = q_act
                if q_act == 6:
                    self.last_click_xy = (qx, qy)
                return q_act, q_data
            else:
                self.macro_queue.clear()

        probe_act = self.hypotheses.next_probe(
            move_actions, self.schema.displacement_vectors, schema_frozen=self.schema.frozen
        )
        if probe_act is not None and not self.dead_table.is_dead(curr_hash, probe_act, 0, 0, self.current_level):
            self.subagent_ctrl.delegate("PROBE")
            self.last_action_id = probe_act
            return probe_act, {}

        if av_pos is not None and self.schema.displacement_vectors and self.pacer.has_compute_budget():
            obstacles = self._build_obstacle_set(arr)
            goals = self._find_candidate_goals(arr, av_pos)

            if self.confirmed_box_colors and goals:
                bg = self.processor.induce_background_color(arr, self.learned_floor_color)
                ents = self.processor.extract_entities(arr, bg_color=bg, ignore_bezel=True)
                box_pts: Set[Tuple[int, int]] = {
                    (int(round(e.centroid[0])), int(round(e.centroid[1])))
                    for e in ents if e.color in self.confirmed_box_colors and e.mass <= 120
                }
                if box_pts:
                    soko_plan = sokoban_bfs(
                        av_pos, box_pts, set(goals[:4]), obstacles, self.schema.displacement_vectors, arr.shape
                    )
                    audited = self.critic.audit_plan(soko_plan or [], curr_hash, self.dead_table, self.current_level)
                    if audited:
                        first_act = int(audited[0])
                        self.subagent_ctrl.delegate("MOORE_PLANNER")
                        self.macro_queue.push_plan(audited[1:], subagent_tag="SOKOBAN")
                        self.last_action_id = first_act
                        return first_act, {}

            for goal in goals[:5]:
                if self.learned_game_modality == "SLIDE":
                    plan = ice_slide_bfs(av_pos, goal, obstacles, self.schema.displacement_vectors, arr.shape)
                else:
                    plan = astar_grid(
                        av_pos,
                        goal,
                        obstacles,
                        self.schema.displacement_vectors,
                        arr.shape,
                        portals=self.hypotheses.portal_map,
                    )
                audited = self.critic.audit_plan(plan or [], curr_hash, self.dead_table, self.current_level)
                if audited:
                    first_act = int(audited[0])
                    full_plan: List[Any] = list(audited[1:])
                    if 5 in avail:
                        full_plan.append(5)
                    self.subagent_ctrl.delegate("MOORE_PLANNER")
                    self.macro_queue.push_plan(full_plan, subagent_tag="GOAL_ASTAR")
                    self.last_action_id = first_act
                    return first_act, {}

            if self.step_in_level >= 3 and (
                self.synthesizer.verified_fn is None
                or self.synthesizer.needs_repair
                or self.step_in_level % 4 == 0
            ):
                self.synthesizer.synthesize_world_model(
                    self.learned_floor_color,
                    self.learned_avatar_colors,
                    self.confirmed_wall_colors | self.confirmed_hazard_colors,
                    self.confirmed_box_colors,
                    self.schema.displacement_vectors,
                    hud_mask=self.hud_masker.hud_mask,
                    portals=self.hypotheses.portal_map,
                    key_to_door_colors=self.schema.key_to_door_colors,
                )
            if self.synthesizer.verified_fn is not None:
                iw_plan = self.synthesizer.plan_iw_lookahead(
                    arr,
                    avail,
                    goals[0] if goals else None,
                    self.learned_avatar_colors,
                    dead_table=self.dead_table,
                    hud_masker=self.hud_masker,
                    level_idx=self.current_level,
                )
                audited = self.critic.audit_plan(iw_plan or [], curr_hash, self.dead_table, self.current_level)
                if audited:
                    first_act = int(audited[0])
                    self.subagent_ctrl.delegate("MOORE_PLANNER")
                    self.macro_queue.push_plan(audited[1:], subagent_tag="BFWS_MOORE")
                    self.last_action_id = first_act
                    return first_act, {}

            frontier_plan = bfs_frontier_explore(
                av_pos, self.visited_cells, obstacles, self.schema.displacement_vectors, arr.shape
            )
            audited = self.critic.audit_plan(frontier_plan or [], curr_hash, self.dead_table, self.current_level)
            if audited:
                first_act = int(audited[0])
                self.subagent_ctrl.delegate("MOORE_PLANNER")
                self.macro_queue.push_plan(audited[1:], subagent_tag="FRONTIER")
                self.last_action_id = first_act
                return first_act, {}

            if self.step_in_level >= 5 and self.step_in_level % 6 == 0:
                self.subagent_ctrl.delegate("CODEACT_EXPLORER")
                llm = OfflineLLMRuntime.get_instance()
                bg = self.processor.induce_background_color(arr, self.learned_floor_color)
                ent_graph = self.processor.encode_ascii_entity_graph(arr, bg_color=bg)
                prompt = self.context.build_prompt(
                    level_idx=self.current_level,
                    av_pos=av_pos,
                    vectors=self.schema.displacement_vectors,
                    walls=self.confirmed_wall_colors,
                    hazards=self.confirmed_hazard_colors,
                    entities_summary=f"goals={goals[:4]} | roles={self.hypotheses.color_semantics} | graph={ent_graph}",
                    available_actions=avail,
                )
                sub = llm.infer_subgoal(prompt)
                if sub:
                    if sub.get("subgoal") == "CODEACT" and isinstance(sub.get("code"), str):
                        ents = self.processor.extract_entities(arr, bg_color=bg, ignore_bezel=True)
                        cavs = JordanCurveCavityInducer.find_cavities(arr, bg_color=bg, bezel_px=4)
                        repl_plan = self.codeact_repl.execute_codeact(
                            code_str=sub["code"],
                            grid=arr,
                            prev_grid=self.prev_grid,
                            av_pos=av_pos,
                            bg_color=bg,
                            entities=ents,
                            cavities=cavs,
                            vectors=self.schema.displacement_vectors,
                            avail_actions=avail,
                            history_records=self.context.history_log.records,
                            portals=self.hypotheses.portal_map,
                        )
                        audited_repl = self.critic.audit_plan(
                            repl_plan or [], curr_hash, self.dead_table, self.current_level
                        )
                        if audited_repl:
                            self.macro_queue.push_plan(audited_repl, subagent_tag="CODEACT_REPL")
                            nxt_item = self.macro_queue.pop_next(1)
                            if nxt_item is not None:
                                self.last_action_id = nxt_item[0]
                                return nxt_item
                    elif sub.get("subgoal") == "NAVIGATE" and isinstance(sub.get("target_rc"), list):
                        tr, tc = int(sub["target_rc"][0]), int(sub["target_rc"][1])
                        plan = astar_grid(
                            av_pos,
                            (tr, tc),
                            obstacles,
                            self.schema.displacement_vectors,
                            arr.shape,
                            portals=self.hypotheses.portal_map,
                        )
                        audited_nav = self.critic.audit_plan(
                            plan or [], curr_hash, self.dead_table, self.current_level
                        )
                        if audited_nav:
                            first_act = int(audited_nav[0])
                            self.macro_queue.push_plan(audited_nav[1:], subagent_tag="LLM_NAV")
                            self.last_action_id = first_act
                            return first_act, {}
                    elif sub.get("subgoal") == "CLICK" and 6 in avail and isinstance(sub.get("click_xy"), list):
                        cx, cy = int(sub["click_xy"][0]), int(sub["click_xy"][1])
                        if not self.dead_table.is_dead(curr_hash, 6, cx, cy, self.current_level):
                            self.last_action_id = 6
                            self.last_click_xy = (cx, cy)
                            return 6, {"x": cx, "y": cy}
                    elif sub.get("subgoal") == "ACTION" and isinstance(sub.get("action_id"), int):
                        aid = int(sub["action_id"])
                        if aid in avail and not self.dead_table.is_dead(curr_hash, aid, 0, 0, self.current_level):
                            self.last_action_id = aid
                            return aid, {}

        all_discrete = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
        graph_plan = self.dead_table.bfs_to_unexplored_action(
            curr_hash, all_discrete, self.current_level, self.last_action_id
        )
        if graph_plan:
            first_act = graph_plan[0]
            self.macro_queue.push_plan(graph_plan[1:], subagent_tag="GRAPH_BFS")
            self.last_action_id = first_act
            return first_act, {}

        if 6 in avail:
            act_id, data = self._select_smart_click(arr, curr_hash)
            self.last_action_id = act_id
            return act_id, data

        if all_discrete:
            ranked_disc = sorted(
                all_discrete,
                key=lambda a: self.dead_table.compute_action_penalty(
                    curr_hash, a, self.last_action_id, 0, 0, self.current_level
                ),
            )
            chosen = ranked_disc[0]
            self.last_action_id = chosen
            return chosen, {}

        self.last_action_id = 1
        return 1, {}
''', encoding="utf-8")

# =========================================================================
# 13. Entrypoint: my_agent.py (Multi-Frame Animation Sequence Support)
# =========================================================================
MY_AGENT_CODE = r'''import threading
from typing import Any, List
import numpy as np

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from project_override.orchestration.general_agent import GeneralAgent

_COMPUTE_SEMAPHORE = threading.Semaphore(2)


class MyAgent(Agent):
    """Competition Entrypoint for ARC-AGI-3 (Swarm / Multi-Thread Safe)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        gid = getattr(self, "game_id", "arc_game")
        self.orchestration_agent = GeneralAgent(game_id=str(gid))

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state == GameState.GAME_OVER:
                self.orchestration_agent.on_game_over()
            return GameAction.RESET

        raw_frame = latest_frame.frame
        frame_seq: List[np.ndarray] = []
        if isinstance(raw_frame, list) and len(raw_frame) > 0:
            arr = np.asarray(raw_frame, dtype=np.int8)
            if arr.ndim == 3:
                frame_seq = [arr[i] for i in range(arr.shape[0])]
                grid = arr[-1]
            else:
                grid = arr
        else:
            grid = np.zeros((64, 64), dtype=np.int8)

        avail_raw = getattr(latest_frame, "available_actions", None) or [1, 2, 3, 4]
        avail_ids = [int(getattr(a, "value", a)) for a in avail_raw]
        lvls = int(getattr(latest_frame, "levels_completed", 0) or 0)

        with _COMPUTE_SEMAPHORE:
            act_id, data = self.orchestration_agent.step(
                grid=grid,
                available_actions=avail_ids,
                levels_completed=lvls,
                frame_sequence=frame_seq,
            )

        try:
            action = GameAction.from_id(int(act_id))
        except Exception:
            action = GameAction.ACTION1

        if action == GameAction.ACTION6 and data:
            action.set_data({"x": int(data.get("x", 32)), "y": int(data.get("y", 32))})

        return action
'''

for target in [
    ROOT / "agent" / "my_agent.py",
    ROOT / "ARC-AGI-3-Kaggle-Starter" / "agent" / "my_agent.py",
    ROOT / "ARC-AGI-3-Agents" / "agents" / "templates" / "my_agent.py",
]:
    if target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(MY_AGENT_CODE, encoding="utf-8")

# =========================================================================
# 14. Full AST Validation + 14-Part End-to-End Verification Suite
# =========================================================================
for py_f in sorted(PO_DIR.rglob("*.py")):
    src = py_f.read_text(encoding="utf-8")
    ast.parse(src, filename=str(py_f))
ast.parse(MY_AGENT_CODE, filename="my_agent.py")
print("[AST-GATE] Every Python file in project_override/ and my_agent.py passed ast.parse() with zero syntax errors.")

sys.path.insert(0, str(ROOT))
import numpy as np
from project_override.orchestration.general_agent import GeneralAgent
from project_override.core.combinatorial_engine import GF2ToggleSolver
from project_override.core.visual_goal_induction import BlueprintMatcher
from project_override.core.dead_signature import DynamicHUDMasker
from project_override.planning.symbolic_planners import astar_grid

# Test 1-3: Composite 2-Color Avatar + Single-Probe Kinematic Basis + Inverse Induction
ag = GeneralAgent(game_id="self_test_v15")
g0 = np.zeros((64, 64), dtype=np.int8)
g0[30:35, 30:35] = 12
g0[30:32, 30:32] = 9
g0[10:15, 30:35] = 14
a1, _ = ag.step(g0, [1, 2, 3, 4, 5], 0)
g1 = np.zeros((64, 64), dtype=np.int8)
g1[25:30, 30:35] = 12
g1[25:27, 30:32] = 9
g1[10:15, 30:35] = 14
a2, _ = ag.step(g1, [1, 2, 3, 4, 5], 0)
assert ag.learned_avatar_colors == {9, 12}, f"Composite sprite test failed: {ag.learned_avatar_colors}"
assert ag.schema.displacement_vectors.get(1) == (-5, 0), "Vector inference failed"
assert ag.schema.displacement_vectors.get(2) == (5, 0), "Inverse vector failed"

# Test 4: Tycho Dynamic HUD Masker (Depleting Move-Budget Bar Detection)
masker = DynamicHUDMasker(shape=(64, 64), default_bezel=4)
f_a = np.zeros((64, 64), dtype=np.int8)
for step_i in range(4):
    f_b = f_a.copy()
    f_b[5, 10 + step_i] = 11
    masker.observe_transition(f_a, f_b)
assert bool(masker.hud_mask[5, 10]), "DynamicHUDMasker failed to detect depleting HUD bar"

# Test 5: Duck CodeAct Python REPL + Persistent Workspace
repl_plan = ag.codeact_repl.execute_codeact(
    code_str="def solve_turn(env):\n    env['workspace']['visited_flag'] = True\n    queue_actions([1, 1, 5])\n",
    grid=g1,
    prev_grid=g0,
    av_pos=(27, 32),
    bg_color=0,
    entities=[],
    cavities=[],
    vectors=ag.schema.displacement_vectors,
    avail_actions=[1, 2, 3, 4, 5],
    history_records=ag.context.history_log.records,
)
assert repl_plan == [1, 1, 5] and ag.codeact_repl.workspace_vars.get("visited_flag") is True, "CodeAct REPL failed"

# Test 6: PRO-LONG Programmatic Memory Grep & Causal Trigger Search
grep_hits = ag.context.history_log.grep("n_px")
assert len(grep_hits) >= 1, "PRO-LONG grep failed"

# Test 7: 16-Token Block-Aligned Static Prefix Verification
assert len(ag.context.SYSTEM_PREFIX) % 64 == 0, "SYSTEM_PREFIX not 64-char block-aligned"

# Test 8: RHAE Action Penalty Scorer
rev_pen = ag.dead_table.compute_action_penalty(ag.prev_hash, 2, last_action_id=1)
fwd_pen = ag.dead_table.compute_action_penalty(ag.prev_hash, 3, last_action_id=1)
assert rev_pen > fwd_pen, "Action penalty failed to downvote immediate reversal"

# Test 9: Tycho 6-Archetype Moore Machine + Non-Regression Gate + BFWS Lookahead
fn = ag.synthesizer.synthesize_world_model(
    ag.learned_floor_color,
    ag.learned_avatar_colors,
    ag.confirmed_wall_colors,
    ag.confirmed_box_colors,
    ag.schema.displacement_vectors,
    hud_mask=ag.hud_masker.hud_mask,
    portals=ag.hypotheses.portal_map,
    key_to_door_colors=ag.schema.key_to_door_colors,
)
assert fn is not None, "Moore world model synthesis failed"
bfws_plan = ag.synthesizer.plan_iw_lookahead(
    g1, [1, 2, 3, 4], (12, 32), ag.learned_avatar_colors, dead_table=ag.dead_table, hud_masker=ag.hud_masker
)
assert bfws_plan == [1, 1, 1], f"BFWS lookahead failed: {bfws_plan}"

# Test 10: Portal Teleportation A* Planning
ag.hypotheses.register_portal_jump((20, 32), (12, 32))
portal_plan = astar_grid((25, 32), (12, 32), set(), {1: (-5, 0), 2: (5, 0)}, portals=ag.hypotheses.portal_map)
assert portal_plan == [1], f"Portal-aware A* failed: {portal_plan}"

# Test 11: WorkspaceCritic Plan Auditing
ag.dead_table.record_death(ag.prev_hash, 3, 0, 0, 0)
assert ag.critic.audit_plan([3, 1], ag.prev_hash, ag.dead_table, 0) is None, "WorkspaceCritic failed to veto dead plan"

# Test 12: GAME_OVER Hazard Immunity Across Resets
ag.on_game_over(g1)
assert len(ag.dead_table.death_set) >= 1, "Death signature recording failed"

# Test 13: GF(2) Linear Toggle Solver
A_mat = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int8)
b_vec = np.array([1, 1, 0], dtype=np.int8)
sol = GF2ToggleSolver.solve_toggle_matrix(A_mat, b_vec)
assert sol is not None and np.array_equal((A_mat @ sol) % 2, b_vec), "GF(2) solver failed"

# Test 14: Multi-Scale Blueprint Matcher
bp_grid = np.zeros((64, 64), dtype=np.int8)
bp_grid[6:9, 6:9] = np.array([[5, 5, 0], [5, 0, 5], [0, 5, 5]], dtype=np.int8)
scaled_3x = np.repeat(np.repeat(bp_grid[6:9, 6:9], 3, axis=0), 3, axis=1)
bp_grid[30:39, 30:39] = scaled_3x
bp_grid[36:39, 36:39] = 0
bp_clicks = BlueprintMatcher.find_blueprint_clicks(bp_grid, bg_color=0, bezel_px=4)
assert len(bp_clicks) >= 1, f"Blueprint matcher failed: {bp_clicks}"

print("[SELF-TEST] All 14 V15 Verification Suites (PRO-LONG Memory, 6-Archetype Tycho Moore Machine, DreamTeam Critic, Duck CodeAct REPL, HUD Bar Masker, Portal A*, BFWS, GF(2), Blueprint) PASSED 100%!")
