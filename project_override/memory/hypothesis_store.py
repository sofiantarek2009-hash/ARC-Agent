from collections import deque
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from project_override.perception.observation_processor import Entity

class BayesianExplorationTracker:
    """
    OPINE-World Lifted Bayesian Belief State (arXiv:2607.01531):
    Tracks object-action-context transition counts C_t(j, e) with symmetric Dirichlet
    prior alpha_0 = 1.0, computes normalized posterior predictive entropy eta_j,
    and ranks unresolved object interactions by actor-facing exploration priority:
        priority(j) = 1.0 - max_e C_t(j, e) / (1.0 + n_j).
    """
    def __init__(self, alpha_0: float = 1.0, effect_alphabet_size: int = 6):
        self.alpha_0 = alpha_0
        self.m = max(2, effect_alphabet_size)
        self.outcome_counts: Dict[Tuple[Tuple[int, int, int, int], int], Dict[str, int]] = {}

    def record_effect(
        self, obj_sig: Tuple[int, int, int, int], action_id: int, effect_tag: str
    ) -> None:
        key = (obj_sig, int(action_id))
        if key not in self.outcome_counts:
            self.outcome_counts[key] = {}
        bucket = self.outcome_counts[key]
        bucket[effect_tag] = bucket.get(effect_tag, 0) + 1

    def posterior_entropy(
        self, obj_sig: Tuple[int, int, int, int], action_id: int
    ) -> float:
        key = (obj_sig, int(action_id))
        counts = self.outcome_counts.get(key, {})
        n_j = sum(counts.values())
        denom = self.m * self.alpha_0 + float(n_j)
        probs: List[float] = []
        observed_types = list(counts.values())
        for c_val in observed_types:
            probs.append((self.alpha_0 + float(c_val)) / denom)
        unobserved_slots = max(0, self.m - len(observed_types))
        for _ in range(unobserved_slots):
            probs.append(self.alpha_0 / denom)
        norm_sum = sum(probs)
        if norm_sum <= 0:
            return 1.0
        probs = [p / norm_sum for p in probs]
        ent = -sum(p * math.log(max(1e-12, p)) for p in probs)
        return float(ent / math.log(float(self.m)))

    def exploration_priority(
        self, obj_sig: Tuple[int, int, int, int], action_id: int
    ) -> float:
        key = (obj_sig, int(action_id))
        counts = self.outcome_counts.get(key, {})
        n_j = sum(counts.values())
        if n_j == 0:
            return 1.0
        max_c = max(counts.values())
        base_prio = 1.0 - (float(max_c) / float(1 + n_j))
        if len(counts) == 1 and "NO_OP" in counts:
            return base_prio * (0.25 / float(n_j))
        return base_prio + 0.15 * self.posterior_entropy(obj_sig, action_id)

class HypothesisStore:
    """
    S4 Epistemic Persistence Store (arXiv:2610.26632) + Retrodict Playbook Memory:
    Separates WorkingModel (checked_rules vs assumed_hypotheses across levels)
    from WorkingMemory (current level attempt state) and retains reasoning replay.
    """
    def __init__(self, max_reasoning_items: int = 14):
        self.displacement_vectors: Dict[int, Tuple[int, int]] = {}
        self.avatar_colors: Set[int] = set()
        self.wall_colors: Set[int] = set()
        self.hazard_colors: Set[int] = set()
        self.goal_colors: Set[int] = set()
        self.pushable_colors: Set[int] = set()
        self.switch_colors: Set[int] = set()
        self.click_color_transitions: Dict[int, int] = {}
        self.portal_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.checked_rules: Dict[str, str] = {}
        self.assumed_hypotheses: Dict[str, str] = {}
        self.persistent_notes: Dict[str, str] = {}
        self.reasoning_replay: deque = deque(maxlen=max_reasoning_items)
        self.level_clear_summaries: List[str] = []
        self.bayesian_tracker = BayesianExplorationTracker()
        self.interacted_object_centers: Set[Tuple[int, int]] = set()

    def reset_for_new_level(self, cleared_previous: bool = False, prev_level: int = 0) -> None:
        if cleared_previous:
            vec_str = ", ".join(f"A{k}:{v}" for k, v in sorted(self.displacement_vectors.items()))
            summary = (
                f"Cleared Level {prev_level}: avatar={sorted(self.avatar_colors)}, "
                f"vectors={{{vec_str}}}, pushable={sorted(self.pushable_colors)}, "
                f"click_map={self.click_color_transitions}"
            )
            self.level_clear_summaries.append(summary)
            self.checked_rules[f"L{prev_level}_verified"] = summary
        self.portal_map.clear()
        self.interacted_object_centers.clear()

    def observe_entity_motions(
        self,
        action_id: int,
        tracked: List[Tuple[Entity, Entity, Tuple[float, float]]],
        pre_grid: Optional[np.ndarray] = None,
        post_grid: Optional[np.ndarray] = None,
        click_xy: Tuple[int, int] = (0, 0),
    ) -> None:
        if action_id == 6 and pre_grid is not None and post_grid is not None:
            cx, cy = int(click_xy[0]), int(click_xy[1])
            if 0 <= cy < pre_grid.shape[0] and 0 <= cx < pre_grid.shape[1]:
                c_before = int(pre_grid[cy, cx])
                c_after = int(post_grid[cy, cx])
                if c_before != c_after:
                    self.click_color_transitions[c_before] = c_after
                    self.checked_rules[f"click_c{c_before}"] = f"c{c_before}->c{c_after}"
            return
        if action_id not in (1, 2, 3, 4) or not tracked:
            return

        if len(tracked) == 1:
            pe, ce, (dr, dc) = tracked[0]
            ir, ic = int(round(dr)), int(round(dc))
            if (ir != 0 or ic != 0) and abs(ir) <= 16 and abs(ic) <= 16:
                known_strides = [
                    max(abs(v[0]), abs(v[1])) for v in self.displacement_vectors.values()
                ]
                min_stride = min(known_strides) if known_strides else max(abs(ir), abs(ic))
                if known_strides and max(abs(ir), abs(ic)) > max(3, min_stride * 2):
                    p_in = (
                        int(round(pe.centroid[0]))
                        + self.displacement_vectors.get(action_id, (0, 0))[0],
                        int(round(pe.centroid[1]))
                        + self.displacement_vectors.get(action_id, (0, 0))[1],
                    )
                    p_out = (int(round(ce.centroid[0])), int(round(ce.centroid[1])))
                    self.portal_map[p_in] = p_out
                else:
                    self.displacement_vectors[int(action_id)] = (ir, ic)
                    if pe.color not in self.pushable_colors:
                        self.avatar_colors.add(pe.color)
                    self.checked_rules[f"move_A{action_id}"] = f"delta=({ir},{ic})"
        else:
            # Multi-entity movement: isolate controllable avatar from displaced pushable crates
            av_item = None
            for item in tracked:
                if self.avatar_colors and item[0].color in self.avatar_colors:
                    av_item = item
                    break
            if av_item is None:
                non_push = [item for item in tracked if item[0].color not in self.pushable_colors]
                av_item = non_push[0] if non_push else tracked[0]

            if av_item is not None and av_item[0].color not in self.pushable_colors:
                self.avatar_colors.add(av_item[0].color)

            for item in tracked:
                if item is not av_item:
                    pe_push, ce_push, (dr_p, dc_p) = item
                    if abs(dr_p) > 0.1 or abs(dc_p) > 0.1:
                        if pe_push.color not in self.avatar_colors:
                            self.pushable_colors.add(pe_push.color)
                            self.checked_rules[f"pushable_c{pe_push.color}"] = "movable_block"

    def locate_avatar(self, entities: List[Entity]) -> Optional[Tuple[int, int]]:
        if not entities:
            return None
        if self.avatar_colors:
            candidates = [e for e in entities if e.color in self.avatar_colors]
            if candidates:
                best = min(candidates, key=lambda e: e.mass)
                return (int(round(best.centroid[0])), int(round(best.centroid[1])))
        smallest = min(entities, key=lambda e: (e.mass, e.id))
        return (int(round(smallest.centroid[0])), int(round(smallest.centroid[1])))

    def add_reasoning_trace(self, step_idx: int, level_idx: int, thought: str) -> None:
        clean = thought.strip().replace("\n", " ")[:220]
        if clean:
            self.reasoning_replay.append(f"[L{level_idx}:S{step_idx}] {clean}")

    def format_epistemic_memory(self) -> str:
        vecs = ", ".join(f"A{k}:{v}" for k, v in sorted(self.displacement_vectors.items()))
        checked_str = "; ".join(f"{k}={v}" for k, v in list(self.checked_rules.items())[-6:])
        notes_str = "; ".join(f"{k}={v}" for k, v in list(self.persistent_notes.items())[-6:])
        clears = " | ".join(self.level_clear_summaries[-4:])
        replay = "\n  ".join(self.reasoning_replay)
        return (
            f"PLAYBOOK_WORKING_MODEL (CHECKED): {{{checked_str or vecs}}}\n"
            f"AVATAR_COLORS: {sorted(self.avatar_colors)} | PUSHABLE: {sorted(self.pushable_colors)} | CLICK_MAP: {self.click_color_transitions}\n"
            f"LEVEL_INVARIANTS: {clears or 'None yet'}\n"
            f"WORKING_MEMORY_NOTES: {notes_str or 'Empty'}\n"
            f"S4_REASONING_REPLAY:\n  {replay or 'No prior reasoning recorded.'}"
        )
