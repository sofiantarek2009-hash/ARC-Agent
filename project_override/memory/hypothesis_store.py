from collections import deque
import math
from typing import Any, Dict, List, Optional, Set, Tuple
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
    S4 Epistemic Persistence Store (arXiv:2610.26632) + Retrodict Playbook Memory + TELL Role Transfer:
    Separates WorkingModel (checked_rules vs assumed_hypotheses across levels)
    from WorkingMemory (current level attempt state), supports Active Orthogonal Basis Completion
    to eliminate the Level 1 probing tax, and retains native reasoning replay.
    """
    OPPOSITE_ACTION = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self, max_reasoning_items: int = 14):
        self.displacement_vectors: Dict[int, Tuple[int, int]] = {}
        self.provisional_vectors: Dict[int, Tuple[int, int]] = {}
        self.avatar_colors: Set[int] = set()
        self.wall_colors: Set[int] = set()
        self.hazard_colors: Set[int] = set()
        self.goal_colors: Set[int] = set()
        self.pushable_colors: Set[int] = set()
        self.switch_colors: Set[int] = set()
        self.canvas_bg: Optional[int] = None
        self.floor_color: Optional[int] = None
        self.avatar_half_w: int = 0
        self.avatar_stride: int = 1
        self.avatar_mass_by_color: Dict[int, int] = {}
        self.current_av_pos: Optional[Tuple[int, int]] = None
        self.click_color_transitions: Dict[int, int] = {}
        self.portal_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.checked_rules: Dict[str, str] = {}
        self.assumed_hypotheses: Dict[str, str] = {}
        self.persistent_notes: Dict[str, str] = {}
        self.reasoning_replay: deque = deque(maxlen=max_reasoning_items)
        self.level_clear_summaries: List[str] = []
        self.bayesian_tracker = BayesianExplorationTracker()
        self.interacted_object_centers: Set[Tuple[int, int]] = set()

    def update_from_kinematics(
        self, kin: Dict[str, Any], action_id: int, enable_orthogonal: bool = True
    ) -> None:
        if not kin:
            return
        if kin.get("floor_color") is not None:
            self.floor_color = int(kin["floor_color"])
        if kin.get("avatar_colors"):
            self.avatar_colors.update(kin["avatar_colors"])
        if kin.get("avatar_color_counts"):
            self.avatar_mass_by_color.update(kin["avatar_color_counts"])
        if kin.get("half_w"):
            self.avatar_half_w = max(self.avatar_half_w, int(kin["half_w"]))

        # Check for teleport / portal discontinuity
        if kin.get("vector") and kin["vector"] != (0, 0):
            dr, dc = kin["vector"]
            stride = max(abs(dr), abs(dc))
            known_strides = [
                max(abs(v[0]), abs(v[1])) for v in self.displacement_vectors.values()
            ]
            if known_strides and stride > max(3, min(known_strides) * 2):
                old_pos = self.current_av_pos
                new_pos = (
                    int(round(kin["avatar_centroid"][0])),
                    int(round(kin["avatar_centroid"][1])),
                )
                if old_pos is not None:
                    self.portal_map[old_pos] = new_pos
                self.current_av_pos = new_pos
                return

            if kin.get("stride"):
                self.avatar_stride = int(kin["stride"])
            self.displacement_vectors[int(action_id)] = (int(dr), int(dc))
            self.checked_rules[f"move_A{action_id}"] = f"delta=({dr},{dc})"
            opp = self.OPPOSITE_ACTION.get(int(action_id))
            if enable_orthogonal and opp is not None and opp not in self.displacement_vectors:
                self.provisional_vectors[opp] = (-int(dr), -int(dc))
                self.assumed_hypotheses[f"ortho_A{opp}"] = f"assumed_delta=({-dr},{-dc})"

        if kin.get("avatar_centroid"):
            self.current_av_pos = (
                int(round(kin["avatar_centroid"][0])),
                int(round(kin["avatar_centroid"][1])),
            )

    def get_effective_vectors(self, enable_orthogonal: bool = True) -> Dict[int, Tuple[int, int]]:
        if not enable_orthogonal:
            return dict(self.displacement_vectors)
        merged = dict(self.provisional_vectors)
        merged.update(self.displacement_vectors)
        return merged

    def has_2d_movement_basis(self, enable_orthogonal: bool = True) -> bool:
        vecs = self.get_effective_vectors(enable_orthogonal=enable_orthogonal)
        has_vert = any(dr != 0 and dc == 0 for dr, dc in vecs.values())
        has_horiz = any(dr == 0 and dc != 0 for dr, dc in vecs.values())
        return has_vert and has_horiz and len(vecs) >= 4

    def reset_for_new_level(self, cleared_previous: bool = False, prev_level: int = 0) -> None:
        if cleared_previous:
            vec_str = ", ".join(f"A{k}:{v}" for k, v in sorted(self.get_effective_vectors().items()))
            summary = (
                f"Cleared Level {prev_level}: avatar={sorted(self.avatar_colors)}, "
                f"vectors={{{vec_str}}}, pushable={sorted(self.pushable_colors)}, "
                f"click_map={self.click_color_transitions}"
            )
            self.level_clear_summaries.append(summary)
            self.checked_rules[f"L{prev_level}_verified"] = summary
        self.portal_map.clear()
        self.interacted_object_centers.clear()
        self.current_av_pos = None

    def observe_entity_motions(
        self,
        action_id: int,
        tracked: List[Tuple[Entity, Entity, Tuple[float, float]]],
        pre_grid: Optional[np.ndarray] = None,
        post_grid: Optional[np.ndarray] = None,
        click_xy: Tuple[int, int] = (0, 0),
        enable_orthogonal: bool = True,
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
        tracked_sorted = sorted(tracked, key=lambda item: item[0].mass)
        pe, ce, (dr, dc) = tracked_sorted[0]
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
                self.avatar_colors.add(pe.color)
                self.checked_rules[f"move_A{action_id}"] = f"delta=({ir},{ic})"
                opp = self.OPPOSITE_ACTION.get(int(action_id))
                if enable_orthogonal and opp is not None and opp not in self.displacement_vectors:
                    self.provisional_vectors[opp] = (-ir, -ic)
                    self.assumed_hypotheses[f"ortho_A{opp}"] = f"assumed_delta=({-ir},{-ic})"
        if len(tracked_sorted) >= 2:
            pe2, ce2, (dr2, dc2) = tracked_sorted[1]
            if abs(dr2) > 0.1 or abs(dc2) > 0.1:
                # If pe2 translated synchronously with the avatar, it is part of a composite avatar!
                if abs(dr2 - dr) < 0.5 and abs(dc2 - dc) < 0.5:
                    self.avatar_colors.add(pe2.color)
                    self.pushable_colors.discard(pe2.color)
                elif pe2.color not in self.avatar_colors:
                    self.pushable_colors.add(pe2.color)
                    self.checked_rules[f"pushable_c{pe2.color}"] = "movable_block"

    def locate_avatar(
        self, entities: List[Entity], grid: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int]]:
        if self.current_av_pos is not None:
            return self.current_av_pos
        if not entities or not self.avatar_colors:
            return None
        candidates = [e for e in entities if e.color in self.avatar_colors]
        if candidates:
            if self.avatar_mass_by_color:
                best = min(
                    candidates,
                    key=lambda e: abs(e.mass - self.avatar_mass_by_color.get(e.color, 0)),
                )
            else:
                best = min(candidates, key=lambda e: e.mass)
            return (int(round(best.centroid[0])), int(round(best.centroid[1])))
        return None

    def add_reasoning_trace(self, step_idx: int, level_idx: int, thought: str) -> None:
        clean = thought.strip().replace("\n", " ")[:220]
        if clean:
            self.reasoning_replay.append(f"[L{level_idx}:S{step_idx}] {clean}")

    def format_epistemic_memory(self) -> str:
        vecs = ", ".join(f"A{k}:{v}" for k, v in sorted(self.get_effective_vectors().items()))
        checked_str = "; ".join(f"{k}={v}" for k, v in list(self.checked_rules.items())[-6:])
        assumed_str = "; ".join(f"{k}={v}" for k, v in list(self.assumed_hypotheses.items())[-4:])
        notes_str = "; ".join(f"{k}={v}" for k, v in list(self.persistent_notes.items())[-6:])
        clears = " | ".join(self.level_clear_summaries[-4:])
        replay = "\n  ".join(self.reasoning_replay)
        return (
            f"PLAYBOOK_WORKING_MODEL (CHECKED): {{{checked_str or vecs}}}\n"
            f"ASSUMED_HYPOTHESES: {{{assumed_str or 'None'}}}\n"
            f"AVATAR_COLORS: {sorted(self.avatar_colors)} | PUSHABLE: {sorted(self.pushable_colors)} | CLICK_MAP: {self.click_color_transitions}\n"
            f"LEVEL_INVARIANTS: {clears or 'None yet'}\n"
            f"WORKING_MEMORY_NOTES: {notes_str or 'Empty'}\n"
            f"S4_REASONING_REPLAY:\n  {replay or 'No prior reasoning recorded.'}"
        )
