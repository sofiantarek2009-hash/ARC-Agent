from typing import Any, Dict, List, Optional, Set, Tuple
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


class RecursiveSubAgentController:
    """
    Symbolica ARCgentica Recursive Sub-Agent Orchestrator.
    Enforces strict per-subagent action budgets so no sub-agent can loop or exhaust RHAE.
    """

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
    """
    Unified Tycho Moore Machine + Duck CodeAct REPL + RGB History + ARCgentica Recursive Harness.
    """

    def __init__(self, game_id: str = "unknown"):
        self.game_id = game_id
        self.processor = ObservationProcessor(bezel_px=4)
        self.hud_masker = DynamicHUDMasker(shape=(64, 64), default_bezel=4)
        self.dead_table = DeadSignatureTable()
        self.macro_queue = MacroActionQueue()
        self.subagent_ctrl = RecursiveSubAgentController()
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
            self.dead_table.record_death(
                self.prev_hash, self.last_action_id, cx, cy, self.current_level
            )
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
                    if (
                        lethal_color != self.learned_floor_color
                        and lethal_color not in self.learned_avatar_colors
                    ):
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
                score = ent.mass * 0.4 + dist * 0.25
                candidates.append((score, (gr, gc)))

        for cr, cc in JordanCurveCavityInducer.find_cavities(grid, bg_color=bg, bezel_px=4)[:4]:
            norm_g = (cr // s, cc // s)
            if norm_g not in self.visited_goals:
                dist = abs(cr - av_pos[0]) + abs(cc - av_pos[1])
                candidates.append((dist * 0.2, (cr, cc)))

        candidates.sort(key=lambda x: x[0])
        ordered_goals.extend([pos for _, pos in candidates])
        return ordered_goals

    def _select_smart_click(
        self, grid: np.ndarray, arena_hash: int
    ) -> Tuple[int, Dict[str, int]]:
        bg = self.processor.induce_background_color(grid, self.learned_floor_color)

        # 0. GF(2) Linear Toggle / Lights-Out Auto-Solver
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

        # 1. D4 Split-Panel Symmetry & Multi-Scale Blueprint Completion
        for r, c in SplitPanelSymmetrySolver.find_asymmetric_clicks(grid, bg_color=bg, bezel_px=4):
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        # 2. Jordan Curve Topological Cavity Clicks
        for r, c in JordanCurveCavityInducer.find_cavities(grid, bg_color=bg, bezel_px=4):
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        # 3. AC-3 Periodic Circuit / Pipe Tile Clicks
        circuit_clicks = AC3CircuitSolver.detect_periodic_circuit_clicks(grid, bg_color=bg, bezel_px=4)
        if circuit_clicks:
            for r, c in circuit_clicks:
                x, y = int(c), int(r)
                if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                    self.last_click_xy = (x, y)
                    return 6, {"x": x, "y": y}

        # 4. Component-Priority Foreground Clicks
        entities = self.processor.extract_entities(grid, bg_color=bg, ignore_bezel=True)
        color_counts: Dict[int, int] = {}
        for e in entities:
            color_counts[e.color] = color_counts.get(e.color, 0) + 1

        ranked = sorted(
            entities,
            key=lambda e: (color_counts.get(e.color, 99), abs(e.mass - 9), e.id),
        )
        for ent in ranked:
            r = int(round(ent.centroid[0]))
            c = int(round(ent.centroid[1]))
            if (r, c) not in ent.pixels and ent.pixels:
                r, c = ent.pixels[len(ent.pixels) // 2]
            x, y = int(c), int(r)
            if not self.dead_table.is_dead(arena_hash, 6, x, y, self.current_level):
                self.last_click_xy = (x, y)
                return 6, {"x": x, "y": y}

        # 5. Multi-resolution lattice sweep
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

        # 1. Process transition from previous step (with Tycho Dynamic HUD Masking)
        if self.prev_grid is not None and self.last_action_id is not None and self.prev_hash is not None:
            self.hud_masker.observe_transition(self.prev_grid, arr)
            curr_hash = compute_arena_hash(arr, bezel_px=4, hud_masker=self.hud_masker)
            delta = self.processor.compute_delta_omega(
                self.prev_grid, arr, hud_mask=self.hud_masker.hud_mask
            )
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
                        self.learned_floor_color = fl_c
                        self.hypotheses.record_color_role(fl_c, "FLOOR")
                        self.learned_avatar_colors |= av_colors
                        for ac in av_colors:
                            self.hypotheses.record_color_role(ac, "AVATAR")
                        self.learned_avatar_color = min(self.learned_avatar_colors)
                        new_av_rc = (int(round(post_cent[0])), int(round(post_cent[1])))

                        # Detect portal teleportation jump vs normal stride
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
                            for r_trig in remote_triggers:
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

        # 2. Pure Click-Only Games
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

        # 3. Loop Detection -> Empirical State-Graph BFS Escape with Penalty Scoring
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

        # 4. Pop Queued Macro-Plan (Budget-checked by RecursiveSubAgentController)
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

        # 5. Sub-Agent 1: PROBE_SUBAGENT (Max 2 probes on Level 1, 0 probes on Levels 2..K)
        probe_act = self.hypotheses.next_probe(
            move_actions, self.schema.displacement_vectors, schema_frozen=self.schema.frozen
        )
        if probe_act is not None and not self.dead_table.is_dead(curr_hash, probe_act, 0, 0, self.current_level):
            self.subagent_ctrl.delegate("PROBE")
            self.last_action_id = probe_act
            return probe_act, {}

        # 6. Sub-Agent 2 & 3: MOORE_PLANNER_SUBAGENT + CODEACT_EXPLORER_SUBAGENT
        if av_pos is not None and self.schema.displacement_vectors and self.pacer.has_compute_budget():
            obstacles = self._build_obstacle_set(arr)
            goals = self._find_candidate_goals(arr, av_pos)

            # A. Sokoban Box Push Planner (Multi-Box Hungarian A*)
            if self.confirmed_box_colors and goals:
                bg = self.processor.induce_background_color(arr, self.learned_floor_color)
                ents = self.processor.extract_entities(arr, bg_color=bg, ignore_bezel=True)
                box_pts: Set[Tuple[int, int]] = set()
                for e in ents:
                    if e.color in self.confirmed_box_colors and e.mass <= 120:
                        box_pts.add((int(round(e.centroid[0])), int(round(e.centroid[1]))))
                if box_pts:
                    soko_plan = sokoban_bfs(
                        av_pos, box_pts, set(goals[:4]), obstacles, self.schema.displacement_vectors, arr.shape
                    )
                    if soko_plan:
                        first_act = soko_plan[0]
                        if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                            self.subagent_ctrl.delegate("MOORE_PLANNER")
                            self.macro_queue.push_plan(soko_plan[1:], subagent_tag="SOKOBAN")
                            self.last_action_id = first_act
                            return first_act, {}

            # B. Navigate to candidate goals via Portal-Aware A* / Ice-Slide + Action-5 Interact
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
                if plan:
                    first_act = plan[0]
                    if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                        full_plan: List[Any] = list(plan[1:])
                        if 5 in avail:
                            full_plan.append(5)
                        self.subagent_ctrl.delegate("MOORE_PLANNER")
                        self.macro_queue.push_plan(full_plan, subagent_tag="GOAL_ASTAR")
                        self.last_action_id = first_act
                        return first_act, {}

            # C. Tycho Auto-Triggered Moore World-Model Repair + Replay-Audited BFWS/IW(2) Lookahead
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
                if iw_plan:
                    first_act = iw_plan[0]
                    if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                        self.subagent_ctrl.delegate("MOORE_PLANNER")
                        self.macro_queue.push_plan(iw_plan[1:], subagent_tag="BFWS_MOORE")
                        self.last_action_id = first_act
                        return first_act, {}

            # D. Frontier Curiosity BFS to nearest unvisited stride-lattice cell
            frontier_plan = bfs_frontier_explore(
                av_pos, self.visited_cells, obstacles, self.schema.displacement_vectors, arr.shape
            )
            if frontier_plan:
                first_act = frontier_plan[0]
                if not self.dead_table.is_dead(curr_hash, first_act, 0, 0, self.current_level):
                    self.subagent_ctrl.delegate("MOORE_PLANNER")
                    self.macro_queue.push_plan(frontier_plan[1:], subagent_tag="FRONTIER")
                    self.last_action_id = first_act
                    return first_act, {}

            # E. Duck CodeAct REPL + Qwen 32B AWQ Subgoal Architect
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
                        )
                        if repl_plan:
                            self.macro_queue.push_plan(repl_plan, subagent_tag="CODEACT_REPL")
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
                        if plan and not self.dead_table.is_dead(curr_hash, plan[0], 0, 0, self.current_level):
                            self.macro_queue.push_plan(plan[1:], subagent_tag="LLM_NAV")
                            self.last_action_id = plan[0]
                            return plan[0], {}
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

        # 7. Empirical State-Graph BFS to nearest untried action (with Action Penalty Ranking)
        all_discrete = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
        graph_plan = self.dead_table.bfs_to_unexplored_action(
            curr_hash, all_discrete, self.current_level, self.last_action_id
        )
        if graph_plan:
            first_act = graph_plan[0]
            self.macro_queue.push_plan(graph_plan[1:], subagent_tag="GRAPH_BFS")
            self.last_action_id = first_act
            return first_act, {}

        # 8. Hybrid Game Action-6 Fallback
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
