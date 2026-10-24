from collections import deque
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.config import AgentConfig
from project_override.execution.safe_executor import (
    DPrimeComputeScheduler,
    SafeGameExecutor,
    TrajectorySupervisor,
)
from project_override.llm.local_runtime import LocalLLMRuntime
from project_override.llm.repl import SandboxedREPL
from project_override.memory.action_ledger import ActionLedger, compute_arena_hash
from project_override.memory.hypothesis_store import HypothesisStore
from project_override.perception.observation_processor import ObservationProcessor
from project_override.planning.explorer import SmartClickExplorer, bfs_frontier_explore
from project_override.planning.planner import EmpiricalWorldModel, MacroActionQueue, astar_grid

logger = logging.getLogger(__name__)

class GeneralAgent:
    """
    General, observation-driven ARC-AGI-3 agent combining:
    - Live Sandboxed Python REPL + PRO-LONG log query (Tufa Duck / PRO-LONG arXiv:2607.20064)
    - Executable World-Model Induction, Retrodiction & Goal-Directed BFS (Tycho / Schema / Kepler)
    - OPINE-World Lifted Bayesian Exploration Guidance S_B (arXiv:2607.01531)
    - NVIDIA AVO Trajectory Supervisor & Tiered Escalation (arXiv:2603.24517 / Retrodict)
    - Interior-board no-op guards & ACTION7 Undo backtracking (Franzen M2)
    - Dead-signature & object-class click pruning with No-Total-Coverage safety (Reki M1 / Kepler)
    - S4 Reasoning Replay & Two-Part Playbook Memory (arXiv:2610.26632 / Retrodict)
    """
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.processor = ObservationProcessor(enable_hud_masking=self.config.enable_hud_masking)
        self.ledger = ActionLedger(enable_dead_signatures=self.config.enable_dead_signatures)
        self.hypotheses = HypothesisStore()
        self.world_model = EmpiricalWorldModel(self.processor, self.hypotheses)
        self.macro_queue = MacroActionQueue()
        self.click_explorer = SmartClickExplorer(self.processor)
        self.repl = SandboxedREPL(max_plan_len=self.config.max_repl_plan_len)
        self.scheduler = DPrimeComputeScheduler(self.config)
        self.supervisor = TrajectorySupervisor(enabled=self.config.enable_avo_supervisor)
        self.executor = SafeGameExecutor(self.scheduler)
        self.current_level: int = 0
        self.step_in_game: int = 0
        self.step_in_level: int = 0
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_hash: Optional[int] = None
        self.last_action_id: Optional[int] = None
        self.last_click_xy: Tuple[int, int] = (0, 0)
        self.last_clicked_sig: Optional[Tuple[int, int, int, int]] = None
        self.last_delta_omega: Optional[int] = None
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.wall_cells: Set[Tuple[int, int]] = set()
        self.rolling_turns: deque = deque(maxlen=self.config.max_rolling_turns)
        self.last_llm_step: int = -999

    def reset_game(self) -> None:
        self.current_level = 0
        self.step_in_game = 0
        self.step_in_level = 0
        self.last_llm_step = -999
        self.prev_grid = None
        self.prev_hash = None
        self.last_action_id = None
        self.last_click_xy = (0, 0)
        self.last_clicked_sig = None
        self.last_delta_omega = None
        self.visited_cells.clear()
        self.wall_cells.clear()
        self.rolling_turns.clear()
        self.ledger.reset_level(clear_deaths=True)
        self.hypotheses = HypothesisStore()
        self.world_model = EmpiricalWorldModel(self.processor, self.hypotheses)
        self.macro_queue.clear()
        self.supervisor.reset_level()
        self.processor.hud_masker.reset_level()

    def _on_level_advance(self, new_level: int) -> None:
        self.hypotheses.reset_for_new_level(cleared_previous=True, prev_level=self.current_level)
        self.current_level = new_level
        self.step_in_level = 0
        self.last_llm_step = -999
        self.prev_grid = None
        self.prev_hash = None
        self.last_action_id = None
        self.last_delta_omega = None
        self.visited_cells.clear()
        self.wall_cells.clear()
        self.ledger.reset_level(clear_deaths=False)
        self.macro_queue.clear()
        self.supervisor.reset_level()
        self.processor.hud_masker.reset_level()

    def build_repl_prompt(
        self,
        grid: np.ndarray,
        avail_actions: List[int],
        av_pos: Optional[Tuple[int, int]],
        diff_str: str,
    ) -> str:
        obs_str = self.processor.encode_compact_observation(grid)
        epistemic_str = self.hypotheses.format_epistemic_memory()
        recent_str = "\n".join(self.rolling_turns) if self.rolling_turns else "None"
        untried = self.ledger.get_untried_discrete_actions(
            compute_arena_hash(grid, self.processor.hud_masker),
            avail_actions,
            self.current_level,
        )
        return (
            "ROLE: Autonomous ARC-AGI-3 scientist & coding agent (Duck REPL + Retrodict + Schema).\n"
            "DIRECTIVES:\n"
            "1. RETRODICTION: Check hypotheses against recorded history via query_log() before acting.\n"
            "2. EXPLORE THEN COMMIT: Probe uncertain mechanics with single moves; batch predictable paths.\n"
            "3. HUD GUIDANCE: Ignore border strips that tick on every step (timers/step counters).\n"
            "4. RESET DISCIPLINE: Use ACTION7 (Undo) when stuck; never issue two RESETs in a row.\n"
            f"LEVEL: {self.current_level} | STEP_IN_LEVEL: {self.step_in_level} | ESCALATION_TIER: {self.supervisor.escalation_tier}\n"
            f"AVAIL_ACTIONS: {avail_actions} | UNTRIED_DISCRETE: {untried}\n"
            f"AVATAR_RC: {av_pos} | LAST_DIFF: {diff_str} | WM_BACKTEST_ACC: {self.world_model.last_backtest_score:.2f}\n"
            f"{epistemic_str}\n"
            f"RECENT_TURNS:\n{recent_str}\n"
            f"CURRENT_OBSERVATION:\n{obs_str}\n\n"
            "INSTRUCTIONS:\n"
            "Respond with either a ```python ... ``` block calling act(action_id, x=0, y=0), note(k, v),\n"
            "or register_world_model(step_fn, goal_fn), OR a single JSON object:\n"
            '{"thought": "...", "subgoal": "ACTION", "action_id": 1} or {"thought": "...", "subgoal": "CLICK", "click_xy": [x, y]}.'
        )

    def act(
        self,
        raw_obs: Any,
        available_actions: Optional[List[int]] = None,
        levels_completed: int = 0,
        state_str: str = "PLAYING",
    ) -> Tuple[int, Dict[str, Any]]:
        return self.executor.safe_step(
            lambda: self._act_internal(raw_obs, available_actions, levels_completed, state_str),
            default_action=1,
        )

    def _act_internal(
        self,
        raw_obs: Any,
        available_actions: Optional[List[int]] = None,
        levels_completed: int = 0,
        state_str: str = "PLAYING",
    ) -> Tuple[int, Dict[str, Any]]:
        if state_str in ("NOT_PLAYED", "GAME_OVER"):
            if (
                state_str == "GAME_OVER"
                and self.prev_hash is not None
                and self.last_action_id is not None
            ):
                self.ledger.record_death(
                    self.prev_hash,
                    self.last_action_id,
                    self.last_click_xy[0],
                    self.last_click_xy[1],
                    self.current_level,
                )
            self.macro_queue.clear()
            self.prev_grid = None
            self.prev_hash = None
            if self.supervisor.can_issue_reset():
                self.supervisor.observe_state(0, 0)
                return 0, {}
            return 1, {}

        if levels_completed > self.current_level:
            self._on_level_advance(levels_completed)

        anim_frames = self.processor.extract_animation_sequence(raw_obs)
        arr = anim_frames[-1]
        avail = sorted({int(a) for a in (available_actions or [1, 2, 3, 4, 5, 6]) if int(a) != 0})
        if not avail:
            avail = [1, 2, 3, 4]

        self.step_in_game += 1
        self.step_in_level += 1

        bg_color = self.processor.induce_background_color(arr)
        curr_entities = self.processor.extract_entities(arr, bg_color=bg_color)
        diff_str = "INITIAL_FRAME"

        if (
            self.prev_grid is not None
            and self.prev_hash is not None
            and self.last_action_id is not None
        ):
            self.processor.hud_masker.observe_transition(self.prev_grid, arr)
            curr_hash = compute_arena_hash(arr, self.processor.hud_masker)
            delta_omega = self.processor.compute_delta_omega(self.prev_grid, arr)
            self.last_delta_omega = delta_omega
            diff_str = self.processor.summarize_transition(self.prev_grid, arr, anim_frames)
            prev_entities = self.processor.extract_entities(self.prev_grid, bg_color=bg_color)
            tracked = self.processor.track_entities(prev_entities, curr_entities)
            self.hypotheses.observe_entity_motions(
                self.last_action_id,
                tracked,
                pre_grid=self.prev_grid,
                post_grid=arr,
                click_xy=self.last_click_xy,
            )
            if self.last_clicked_sig is not None:
                eff_tag = "NO_OP" if delta_omega == 0 else f"DELTA_{min(20, delta_omega)}"
                self.hypotheses.bayesian_tracker.record_effect(
                    self.last_clicked_sig, self.last_action_id, eff_tag
                )
            if self.last_action_id in (1, 2, 3, 4) and delta_omega >= 28:
                self.wall_cells.clear()
                self.macro_queue.clear()
            self.ledger.record_transition(
                pre_hash=self.prev_hash,
                action_id=self.last_action_id,
                post_hash=curr_hash,
                delta_omega=delta_omega,
                coord_x=self.last_click_xy[0],
                coord_y=self.last_click_xy[1],
                level_idx=self.current_level,
                pre_grid=self.prev_grid,
                post_grid=arr,
                diff_summary=diff_str,
                clicked_obj_sig=self.last_clicked_sig,
                step_idx=self.step_in_level,
            )
            self.world_model.run_backtest(self.ledger.history)
            self.rolling_turns.append(
                f"S{self.step_in_level - 1}: A{self.last_action_id}{self.last_click_xy if self.last_action_id == 6 else ''} -> {diff_str}"
            )
        else:
            curr_hash = compute_arena_hash(arr, self.processor.hud_masker)
            self.ledger.state_visit_counts[curr_hash] = (
                self.ledger.state_visit_counts.get(curr_hash, 0) + 1
            )
            self.last_delta_omega = None

        self.supervisor.observe_state(curr_hash, self.last_action_id)

        av_pos = self.hypotheses.locate_avatar(curr_entities)
        if av_pos is not None:
            self.visited_cells.add(av_pos)
            self.hypotheses.interacted_object_centers.add(av_pos)
            if (
                self.last_delta_omega == 0
                and self.last_action_id in self.hypotheses.displacement_vectors
            ):
                dr, dc = self.hypotheses.displacement_vectors[self.last_action_id]
                blocked_rc = (av_pos[0] + dr, av_pos[1] + dc)
                self.wall_cells.add(blocked_rc)
                if 0 <= blocked_rc[0] < arr.shape[0] and 0 <= blocked_rc[1] < arr.shape[1]:
                    w_color = int(arr[blocked_rc[0], blocked_rc[1]])
                    if w_color != bg_color and w_color not in self.hypotheses.pushable_colors:
                        self.hypotheses.wall_colors.add(w_color)

        self.prev_grid = arr.copy()
        self.prev_hash = curr_hash
        self.last_clicked_sig = None

        # 1. Check guarded macro queue (Franzen M2 interior-board change guard + expectation check)
        queued = self.macro_queue.pop_next(
            last_delta_omega=self.last_delta_omega,
            guard_enabled=self.config.enable_action_guards,
            current_hash=curr_hash,
        )
        if queued is not None:
            q_act, q_data = queued
            if q_act in avail and not self.ledger.is_dead(
                curr_hash, q_act, q_data.get("x", 0), q_data.get("y", 0), self.current_level
            ):
                self.last_action_id = q_act
                self.last_click_xy = (q_data.get("x", 0), q_data.get("y", 0))
                return q_act, q_data

        # 2. AVO Supervisor + ACTION7 (UNDO) loop & dead-end trap recovery
        move_disc_here = [a for a in avail if a in (1, 2, 3, 4)]
        all_moves_dead = bool(move_disc_here) and all(
            self.ledger.is_dead(curr_hash, a, 0, 0, self.current_level) for a in move_disc_here
        )
        if (
            self.config.enable_undo_backtrack
            and 7 in avail
            and (
                all_moves_dead
                or self.ledger.is_in_loop()
                or (self.config.enable_avo_supervisor and self.supervisor.escalation_tier >= 1)
            )
            and self.last_action_id != 7
            and not self.ledger.is_dead(curr_hash, 7, 0, 0, self.current_level)
        ):
            if av_pos is not None:
                self.wall_cells.add(av_pos)
            self.macro_queue.clear()
            self.last_action_id = 7
            self.last_click_xy = (0, 0)
            return 7, {}

        # 3. Pattern-Matching Click Solver when ACTION6 is the primary interaction or mismatches exist
        if self.config.enable_pattern_click_solver and 6 in avail:
            has_move_actions = any(a in avail for a in (1, 2, 3, 4))
            if not has_move_actions:
                pattern_click = self.world_model.plan_click_pattern_match(
                    arr, curr_entities, curr_hash, self.ledger, self.current_level
                )
                if pattern_click is not None:
                    act_id, data, obj_sig = pattern_click
                    self.last_action_id = act_id
                    self.last_click_xy = (data["x"], data["y"])
                    self.last_clicked_sig = obj_sig
                    return act_id, data

        # 4. Probe untried movement actions (1..4) at initial state to rapidly induce movement vectors
        move_discrete = [a for a in avail if a in (1, 2, 3, 4)]
        if len(self.hypotheses.displacement_vectors) < len(move_discrete) and self.step_in_level <= 6:
            untried_moves = [
                a
                for a in self.ledger.get_untried_discrete_actions(
                    curr_hash, move_discrete, self.current_level
                )
                if a not in self.hypotheses.displacement_vectors
            ]
            if untried_moves:
                chosen = untried_moves[0]
                self.last_action_id = chosen
                self.last_click_xy = (0, 0)
                return chosen, {}

        # 5. Executable World-Model Goal/Switch/Box-Push Route Planner (Schema / Tycho / OPINE-World)
        # When a verified route to an interactive object is known, execute with zero latency
        if (
            self.config.enable_world_model_bfs
            and av_pos is not None
            and self.hypotheses.displacement_vectors
            and self.supervisor.escalation_tier == 0
        ):
            wm_route = self.world_model.plan_interactive_route(
                arr,
                curr_entities,
                av_pos,
                self.wall_cells,
                self.hypotheses.interacted_object_centers,
                enable_sokoban=self.config.enable_sokoban_planner,
            )
            if wm_route:
                first_act = int(wm_route[0])
                if first_act in avail and not self.ledger.is_dead(
                    curr_hash, first_act, 0, 0, self.current_level
                ):
                    if len(wm_route) > 1:
                        self.macro_queue.push_plan(
                            wm_route[1:],
                            subagent_tag="WORLD_MODEL_BFS",
                            max_len=self.config.max_repl_plan_len,
                        )
                    self.last_action_id = first_act
                    self.last_click_xy = (0, 0)
                    return first_act, {}

        # 6. Invoke Local Open-Weight LLM + Live Python REPL (Tufa Duck / Franzen M2 / Schema style)
        # Queried when no direct route exists, on level entry, or upon stagnation/escalation
        steps_since_llm = self.step_in_level - self.last_llm_step
        should_call_llm = (
            self.config.enable_llm
            and (steps_since_llm >= self.config.llm_call_interval)
            and (
                self.step_in_level == 1
                or self.supervisor.escalation_tier >= 1
                or (self.step_in_level % self.config.llm_call_interval == 0)
            )
        )
        if should_call_llm:
            self.last_llm_step = self.step_in_level
            llm = LocalLLMRuntime.get_instance(self.config)
            if llm.ensure_loaded():
                prompt = self.build_repl_prompt(arr, avail, av_pos, diff_str)
                decision = llm.infer_decision(prompt)
                if decision:
                    if isinstance(decision.get("thought"), str):
                        self.hypotheses.add_reasoning_trace(
                            self.step_in_level, self.current_level, decision["thought"]
                        )
                    subgoal = decision.get("subgoal", "")
                    if subgoal == "CODEACT" and isinstance(decision.get("code"), str):
                        repl_plan = self.repl.execute_codeact(
                            code_str=decision["code"],
                            grid=arr,
                            prev_grid=self.prev_grid,
                            av_pos=av_pos,
                            bg_color=bg_color,
                            entities=curr_entities,
                            vectors=self.hypotheses.displacement_vectors,
                            avail_actions=avail,
                            notes=self.hypotheses.persistent_notes,
                            query_log_fn=self.ledger.query_history,
                            register_wm_fn=self.world_model.register_custom_simulator,
                        )
                        valid_plan = [
                            (aid, pdata)
                            for aid, pdata in repl_plan
                            if aid in avail
                            and not self.ledger.is_dead(
                                curr_hash,
                                aid,
                                pdata.get("x", 0),
                                pdata.get("y", 0),
                                self.current_level,
                            )
                        ]
                        if valid_plan:
                            first_aid, first_data = valid_plan[0]
                            if len(valid_plan) > 1:
                                self.macro_queue.push_plan(
                                    valid_plan[1:],
                                    subagent_tag="DUCK_REPL",
                                    max_len=self.config.max_repl_plan_len,
                                )
                            self.last_action_id = first_aid
                            self.last_click_xy = (
                                first_data.get("x", 0),
                                first_data.get("y", 0),
                            )
                            return first_aid, first_data
                    elif (
                        subgoal == "NAVIGATE"
                        and isinstance(decision.get("target_rc"), list)
                        and av_pos
                    ):
                        tr, tc = int(decision["target_rc"][0]), int(decision["target_rc"][1])
                        nav_plan = astar_grid(
                            av_pos,
                            (tr, tc),
                            self.wall_cells,
                            self.hypotheses.displacement_vectors,
                            arr.shape,
                            portals=self.hypotheses.portal_map,
                        )
                        if nav_plan:
                            first_aid = int(nav_plan[0])
                            if not self.ledger.is_dead(
                                curr_hash, first_aid, 0, 0, self.current_level
                            ):
                                self.macro_queue.push_plan(nav_plan[1:], subagent_tag="LLM_NAV")
                                self.last_action_id = first_aid
                                self.last_click_xy = (0, 0)
                                return first_aid, {}
                    elif (
                        subgoal == "CLICK"
                        and 6 in avail
                        and isinstance(decision.get("click_xy"), list)
                    ):
                        cx, cy = int(decision["click_xy"][0]), int(decision["click_xy"][1])
                        if not self.ledger.is_dead(curr_hash, 6, cx, cy, self.current_level):
                            self.last_action_id = 6
                            self.last_click_xy = (cx, cy)
                            return 6, {"x": cx, "y": cy}
                    elif subgoal == "ACTION" and isinstance(decision.get("action_id"), int):
                        aid = int(decision["action_id"])
                        if aid in avail and not self.ledger.is_dead(
                            curr_hash, aid, 0, 0, self.current_level
                        ):
                            self.last_action_id = aid
                            self.last_click_xy = (0, 0)
                            return aid, {}

        # 7. Spatial frontier exploration if movement vectors are known
        if av_pos is not None and self.hypotheses.displacement_vectors:
            effective_walls = self.world_model.infer_obstacle_cells(
                arr, curr_entities, self.wall_cells
            )
            frontier_plan = bfs_frontier_explore(
                av_pos,
                self.visited_cells,
                effective_walls,
                self.hypotheses.displacement_vectors,
                arr.shape,
            )
            if frontier_plan:
                first_act = int(frontier_plan[0])
                if first_act in avail and not self.ledger.is_dead(
                    curr_hash, first_act, 0, 0, self.current_level
                ):
                    if len(frontier_plan) > 1:
                        self.macro_queue.push_plan(frontier_plan[1:], subagent_tag="FRONTIER_BFS")
                    self.last_action_id = first_act
                    self.last_click_xy = (0, 0)
                    return first_act, {}

        # 8. Graph-BFS to nearest state with unexplored discrete actions
        all_discrete = [a for a in avail if a in (1, 2, 3, 4, 5)]
        graph_plan = self.ledger.bfs_to_unexplored_action(
            curr_hash, all_discrete, self.current_level, self.last_action_id
        )
        if graph_plan:
            first_act = int(graph_plan[0])
            if len(graph_plan) > 1:
                self.macro_queue.push_plan(graph_plan[1:], subagent_tag="GRAPH_BFS")
            self.last_action_id = first_act
            self.last_click_xy = (0, 0)
            return first_act, {}

        # 9. Smart Bayesian object-segmented click exploration when ACTION6 is available
        if 6 in avail:
            if self.config.enable_pattern_click_solver:
                pattern_click = self.world_model.plan_click_pattern_match(
                    arr, curr_entities, curr_hash, self.ledger, self.current_level
                )
                if pattern_click is not None:
                    act_id, data, obj_sig = pattern_click
                    self.last_action_id = act_id
                    self.last_click_xy = (data["x"], data["y"])
                    self.last_clicked_sig = obj_sig
                    return act_id, data
            act_id, data, obj_sig = self.click_explorer.select_click(
                arr,
                curr_hash,
                self.ledger,
                self.current_level,
                bayesian_tracker=self.hypotheses.bayesian_tracker,
                enable_bayesian=self.config.enable_bayesian_exploration,
            )
            if not self.ledger.is_dead(
                curr_hash, 6, data["x"], data["y"], self.current_level, obj_sig=obj_sig
            ):
                self.last_action_id = act_id
                self.last_click_xy = (data["x"], data["y"])
                self.last_clicked_sig = obj_sig
                return act_id, data

        # 10. Kepler No-Total-Coverage Fallback: novelty-biased discrete choice that NEVER deadlocks
        candidate_disc = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
        if candidate_disc:
            ranked_disc = sorted(
                candidate_disc,
                key=lambda a: self.ledger.compute_action_penalty(
                    curr_hash, a, self.last_action_id, 0, 0, self.current_level
                ),
            )
            chosen = ranked_disc[0]
            self.last_action_id = chosen
            self.last_click_xy = (0, 0)
            return chosen, {}

        fallback_act = avail[0] if avail else 1
        self.last_action_id = fallback_act
        self.last_click_xy = (0, 0)
        return fallback_act, {}
