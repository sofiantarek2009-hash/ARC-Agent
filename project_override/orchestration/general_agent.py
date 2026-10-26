from collections import deque
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.config import AgentConfig
from project_override.execution.avo_supervisor import (
    AVOSupervisor,
    DPrimeComputeScheduler,
    SafeGameExecutor,
)
from project_override.llm.codeact_engine import (
    OfflineServerManager,
    SandboxedCodeActREPL,
)
from project_override.memory.action_ledger import ActionLedger, compute_arena_hash
from project_override.memory.lineage_dag import LineageDAG
from project_override.memory.semantic_compactor import SemanticMemoryCompactor
from project_override.perception.observation_processor import (
    Entity,
    ObservationProcessor,
)
from project_override.planning.general_explorer import (
    GeneralInteractionExplorer,
    MacroActionQueue,
    astar_grid,
)
from project_override.planning.verifier_gate import GroundedVerifierGate
from project_override.prompts.system_prompt import build_agent_prompt

logger = logging.getLogger(__name__)


class GeneralAgent:
    """GeneralAgent v5.0: ARC-AGI-3 NVIDIA AVO + CodeAct REPL Architecture.
    
    Implements a strict, game-agnostic, zero-handcoded 9-stage pipeline:
    1. State Management (Level advance / death handling / reset discipline)
    2. Dynamic Perception (scipy CCL segmentation, Hungarian tracking, compass vector fix)
    3. Structured Transition Recording (LineageDAG, ActionLedger, SemanticCompactor)
    4. NVIDIA AVO Trajectory Supervisor (sigma_t stagnation detection, multi-step ACTION7 Undo)
    5. Grounded Verifier Gate (fail-closed precondition checks against dead signatures and cycles)
    6. Guarded Macro Action Queue (buffered sequential execution with NO-OP drop guards)
    7. Sandboxed Python CodeAct REPL (OpenAI-compatible vLLM/SGLang offline inference)
    8. General Interaction Explorer (card-memory fix, deliberate ACTION5 probing, smart entity clicks)
    9. Novelty-Biased Discrete Fallback (Kepler graph exploration that never deadlocks)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Modules
        self.processor = ObservationProcessor(enable_hud_masking=self.config.enable_hud_masking)
        self.ledger = ActionLedger(enable_dead_signatures=self.config.enable_dead_signatures)
        self.lineage_dag = LineageDAG()
        self.compactor = SemanticMemoryCompactor(self.config)
        self.supervisor = AVOSupervisor(self.config)
        self.verifier_gate = GroundedVerifierGate(self.config)
        self.macro_queue = MacroActionQueue()
        self.explorer = GeneralInteractionExplorer(self.processor)
        self.repl = SandboxedCodeActREPL(max_plan_len=self.config.max_repl_plan_len)
        self.scheduler = DPrimeComputeScheduler(self.config)
        self.executor = SafeGameExecutor(self.scheduler)

        # State tracking
        self.current_level: int = 0
        self.step_in_game: int = 0
        self.step_in_level: int = 0
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_hash: Optional[int] = None
        self.last_action_id: Optional[int] = None
        self.last_click_xy: Tuple[int, int] = (0, 0)
        self.last_delta_omega: Optional[int] = None
        self.avatar_pos: Optional[Tuple[int, int]] = None
        self.wall_cells: Set[Tuple[int, int]] = set()
        self.detected_stride: Optional[int] = None
        self.action_map: Dict[Tuple[int, int], int] = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
        self.probed_directions: Set[int] = set()

    def reset_game(self) -> None:
        self.current_level = 0
        self.step_in_game = 0
        self.step_in_level = 0
        self.prev_grid = None
        self.prev_hash = None
        self.last_action_id = None
        self.last_click_xy = (0, 0)
        self.last_delta_omega = None
        self.avatar_pos = None
        self.wall_cells.clear()
        self.detected_stride = None
        self.action_map = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
        self.probed_directions.clear()
        self.ledger.reset_level(clear_deaths=True)
        self.lineage_dag = LineageDAG()
        self.compactor.reset_level()
        self.supervisor.reset_level()
        self.verifier_gate.reset_level()
        self.macro_queue.clear()
        self.explorer.reset_level()
        self.processor.hud_masker.reset_level()

    def _on_level_advance(self, new_level: int) -> None:
        logger.info(f"[GeneralAgent] Advanced to Level {new_level} (cleared Level {self.current_level} in {self.step_in_level} actions).")
        self.lineage_dag.on_level_clear(self.current_level, self.step_in_level)
        self.compactor.on_level_clear(self.current_level, self.lineage_dag)
        
        self.current_level = new_level
        self.step_in_level = 0
        self.prev_grid = None
        self.prev_hash = None
        self.last_action_id = None
        self.last_delta_omega = None
        self.avatar_pos = None
        self.wall_cells.clear()
        self.ledger.reset_level(clear_deaths=False)
        self.macro_queue.clear()
        self.supervisor.reset_level()
        self.verifier_gate.reset_level()
        self.explorer.reset_level()

    def act(
        self,
        raw_obs: Any,
        available_actions: Optional[List[int]] = None,
        levels_completed: int = 0,
        state_str: str = "PLAYING",
    ) -> Tuple[int, Dict[str, Any]]:
        """Public ABI step entry point, wrapped in SafeGameExecutor."""
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
        
        # ---------------------------------------------------------
        # STAGE 1: State Management (Game Over / Level Advance / Resets)
        # ---------------------------------------------------------
        if state_str in ("NOT_PLAYED", "GAME_OVER"):
            if state_str == "GAME_OVER" and self.prev_hash is not None and self.last_action_id is not None:
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
            if self.supervisor.can_issue_reset(self.step_in_level):
                return 0, {}
            return 1, {}

        if levels_completed > self.current_level:
            self._on_level_advance(levels_completed)

        # ---------------------------------------------------------
        # STAGE 2: Perception & Kinematics
        # ---------------------------------------------------------
        anim_frames = self.processor.extract_animation_sequence(raw_obs)
        arr = anim_frames[-1]
        avail = sorted({int(a) for a in (available_actions or [1, 2, 3, 4, 5, 6]) if int(a) != 0})
        if not avail:
            avail = [1, 2, 3, 4]

        self.step_in_game += 1
        self.step_in_level += 1

        canvas_bg, floor_color = self.processor.induce_canvas_and_floor(arr)
        curr_entities = self.processor.extract_entities(arr, bg_color=canvas_bg, floor_color=floor_color)
        diff_str = "INITIAL_FRAME"

        if self.prev_grid is not None and self.prev_hash is not None and self.last_action_id is not None:
            self.processor.hud_masker.observe_transition(self.prev_grid, arr)
            curr_hash = compute_arena_hash(arr, self.processor.hud_masker)
            delta_omega = self.processor.compute_delta_omega(self.prev_grid, arr)
            self.last_delta_omega = delta_omega
            diff_str = self.processor.summarize_transition(self.prev_grid, arr, anim_frames)

            # Update avatar position & movement kinematics
            kin = self.processor.extract_transition_kinematics(self.prev_grid, arr)
            if kin:
                cent = kin.get("avatar_centroid")
                if cent:
                    self.avatar_pos = (int(round(cent[0])), int(round(cent[1])))
                vec = kin.get("vector")
                if vec and vec != (0, 0):
                    dr, dc = vec
                    stride = kin.get("stride", max(abs(dr), abs(dc)))
                    if stride >= 2:
                        self.detected_stride = stride
                        dir_key = (dr // stride, dc // stride)
                        if self.last_action_id in (1, 2, 3, 4):
                            self.action_map[dir_key] = self.last_action_id
                            self.lineage_dag.persistent_notes["stride"] = stride
                            self.lineage_dag.persistent_notes["action_map"] = self.action_map

            # ---------------------------------------------------------
            # STAGE 3: Structured Transition Recording
            # ---------------------------------------------------------
            self.lineage_dag.record_step(
                step=self.step_in_level,
                level=self.current_level,
                action_id=self.last_action_id,
                x=self.last_click_xy[0],
                y=self.last_click_xy[1],
                pre_hash=self.prev_hash,
                post_hash=curr_hash,
                delta_omega=delta_omega,
                pre_grid=self.prev_grid,
                post_grid=arr,
                avatar_pos=self.avatar_pos,
                diff_summary=diff_str,
            )
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
                step_idx=self.step_in_level,
            )
            self.compactor.add_working_turn(
                step_idx=self.step_in_level - 1,
                action_str=f"A{self.last_action_id}{self.last_click_xy if self.last_action_id == 6 else ''}",
                diff_str=diff_str,
            )
            self.compactor.compact_if_needed(self.lineage_dag)
            self.verifier_gate.observe_transition(delta_omega)
        else:
            curr_hash = compute_arena_hash(arr, self.processor.hud_masker)
            self.ledger.state_visit_counts[curr_hash] = self.ledger.state_visit_counts.get(curr_hash, 0) + 1
            self.last_delta_omega = None

        # ---------------------------------------------------------
        # STAGE 4: NVIDIA AVO Trajectory Supervisor
        # ---------------------------------------------------------
        self.supervisor.observe_state(curr_hash, self.last_action_id, self.step_in_level)
        if self.supervisor.should_force_undo(avail):
            logger.info("[GeneralAgent] Supervisor enforced ACTION7 (Undo) rollback.")
            self.macro_queue.clear()
            self.last_action_id = 7
            self.last_click_xy = (0, 0)
            self.prev_grid = arr.copy()
            self.prev_hash = curr_hash
            return 7, {}

        # ---------------------------------------------------------
        # STAGE 5: Grounded Verifier Gate Check
        # ---------------------------------------------------------
        if self.verifier_gate.is_fail_closed():
            logger.info("[GeneralAgent] Verifier Gate is FAIL-CLOSED; purging macro queue.")
            self.macro_queue.clear()

        # ---------------------------------------------------------
        # STAGE 5.5: Level 1 Epistemic Cardinal Probe (Turns 1..4)
        # ---------------------------------------------------------
        if self.current_level <= 1 and self.step_in_level <= 4 and self.macro_queue.is_empty():
            dir_avail = [a for a in avail if a in (1, 2, 3, 4)]
            untried = [a for a in dir_avail if a not in self.probed_directions]
            if untried:
                probe_act = untried[0]
                self.probed_directions.add(probe_act)
                if self.verifier_gate.verify(
                    action_id=probe_act,
                    payload={},
                    available_actions=avail,
                    state_hash=curr_hash,
                    ledger=self.ledger,
                    level_idx=self.current_level,
                ):
                    logger.info(f"[GeneralAgent] Epistemic Cardinal Probe step {self.step_in_level}: action {probe_act}")
                    self.last_action_id = probe_act
                    self.last_click_xy = (0, 0)
                    self.prev_grid = arr.copy()
                    self.prev_hash = curr_hash
                    return probe_act, {}

        # ---------------------------------------------------------
        # STAGE 6: Guarded Macro Queue Execution
        # ---------------------------------------------------------
        queued = self.macro_queue.pop_next(
            last_delta_omega=self.last_delta_omega,
            guard_enabled=self.config.enable_action_guards,
        )
        if queued is not None:
            q_act, q_data = queued
            if self.verifier_gate.verify(
                action_id=q_act,
                payload=q_data,
                available_actions=avail,
                state_hash=curr_hash,
                ledger=self.ledger,
                level_idx=self.current_level,
            ):
                self.last_action_id = q_act
                self.last_click_xy = (q_data.get("x", 0), q_data.get("y", 0))
                self.prev_grid = arr.copy()
                self.prev_hash = curr_hash
                return q_act, q_data

        # ---------------------------------------------------------
        # STAGE 7: LLM CodeAct REPL Program Synthesis
        # ---------------------------------------------------------
        if self.config.enable_llm and (self.step_in_level % self.config.llm_call_interval == 0):
            server_mgr = OfflineServerManager.get_instance(self.config)
            if server_mgr.ensure_server_running():
                stride = self.detected_stride or self.processor.induce_stride(arr, canvas_bg, floor_color)
                av_cell = (self.avatar_pos[0] // stride, self.avatar_pos[1] // stride) if self.avatar_pos else None
                _, _, detected_items = self.processor.render_ascii_stride_grid(
                    arr, canvas_bg, floor_color, avatar_pos=self.avatar_pos, stride=stride
                )
                targets = [(r, c) for r, c, tag in detected_items]

                obs_summary = self.processor.encode_compact_observation(arr, avatar_pos=self.avatar_pos, stride=stride)
                mem_context = self.compactor.render_context(self.lineage_dag)
                prompt = build_agent_prompt(
                    level_idx=self.current_level,
                    step_in_level=self.step_in_level,
                    available_actions=avail,
                    obs_summary=obs_summary,
                    memory_context=mem_context,
                    supervisor_active=self.supervisor.strategy_audit_active,
                    last_diff=diff_str,
                )
                decision = server_mgr.infer_decision(prompt)
                if decision:
                    thought = decision.get("thought", "")
                    if thought:
                        self.compactor.add_reasoning_trace(thought)

                    subgoal = decision.get("subgoal", "")
                    if subgoal == "CODEACT" and isinstance(decision.get("code"), str):
                        env_ctx = {
                            "grid": arr,
                            "prev_grid": self.prev_grid,
                            "avatar_pos": av_cell or self.avatar_pos,
                            "targets": targets,
                            "stride": stride,
                            "bg_color": canvas_bg,
                            "floor_color": floor_color,
                            "action_map": self.action_map,
                            "available_actions": avail,
                            "entities": curr_entities,
                            "notes": self.lineage_dag.persistent_notes,
                            "query_log_fn": self.ledger.query_history,
                        }
                        plan = self.repl.execute_codeact(decision["code"], env_ctx)
                        # Filter synthesized plan through Verifier Gate
                        valid_plan = [
                            (aid, pdata)
                            for aid, pdata in plan
                            if self.verifier_gate.verify(
                                action_id=aid,
                                payload=pdata,
                                available_actions=avail,
                                state_hash=curr_hash,
                                ledger=self.ledger,
                                level_idx=self.current_level,
                            )
                        ]
                        if valid_plan:
                            first_act, first_data = valid_plan[0]
                            if len(valid_plan) > 1:
                                self.macro_queue.push_plan(
                                    valid_plan[1:],
                                    subagent_tag="CODEACT_REPL",
                                    max_len=self.config.max_repl_plan_len,
                                )
                            self.last_action_id = first_act
                            self.last_click_xy = (first_data.get("x", 0), first_data.get("y", 0))
                            self.prev_grid = arr.copy()
                            self.prev_hash = curr_hash
                            return first_act, first_data

        # ---------------------------------------------------------
        # STAGE 8: General Interaction Explorer
        # ---------------------------------------------------------
        exp_act, exp_data = self.explorer.select_exploration_action(
            grid=arr,
            entities=curr_entities,
            avatar_pos=self.avatar_pos,
            available_actions=avail,
            step_in_level=self.step_in_level,
            state_hash=curr_hash,
            ledger=self.ledger,
            level_idx=self.current_level,
        )
        if self.verifier_gate.verify(
            action_id=exp_act,
            payload=exp_data,
            available_actions=avail,
            state_hash=curr_hash,
            ledger=self.ledger,
            level_idx=self.current_level,
        ):
            self.last_action_id = exp_act
            self.last_click_xy = (exp_data.get("x", 0), exp_data.get("y", 0))
            self.prev_grid = arr.copy()
            self.prev_hash = curr_hash
            return exp_act, exp_data

        # ---------------------------------------------------------
        # STAGE 9: Novelty-Biased Discrete Fallback
        # ---------------------------------------------------------
        candidate_disc = [a for a in avail if a in (1, 2, 3, 4, 5, 7)]
        if candidate_disc:
            ranked = sorted(
                candidate_disc,
                key=lambda a: self.ledger.compute_action_penalty(
                    curr_hash, a, self.last_action_id, 0, 0, self.current_level
                ),
            )
            fallback_act = ranked[0]
        else:
            fallback_act = avail[0] if avail else 1

        self.last_action_id = fallback_act
        self.last_click_xy = (0, 0)
        self.prev_grid = arr.copy()
        self.prev_hash = curr_hash
        return fallback_act, {}
