"""Project Override: General Zero-Overfitting Agent (ARC-AGI-3)
A strictly general reasoning and planning agent:
- ZERO hardcoded game IDs, pixel-coordinate fingerprints, or bespoke game branches.
- Dynamic Kinematic Probing on Level 1 (reversible micro-probe for avatar & vector discovery).
- Universal Background & Foreground Lattice Separation.
- Visual Goal & Invariant Induction (Split-Screen/Symmetry, Topological Enclosure, HUD Sequence).
- Universal Discrete Board Extractor with dynamic scale and empirical + structural object-role induction.
- Universal GF(2) Impulse-Response Solver over GF(2).
- Universal Lattice & K-Tuple Memory Matcher.
- Universal Reachability Sokoban BFS (1-step & sliding crate).
- Universal Inertial Slide Raycast BFS.
- Universal Multi-Vector Jump / Knight BFS.
- Universal Multi-Waypoint Cardinal A* Navigator.
- 64-Bit Bitwise Dead-Signature Culling eliminating null transitions.
- Macro-Action Queue with Verification Delta rollback.
"""

from __future__ import annotations

import collections
import itertools
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage

from project_override.core.universal_engine import (
    DynamicKinematicProber,
    separate_background_and_lattice,
    extract_discrete_board,
    LatticeGrid,
    UniversalMemoryMatcher,
    detect_lo_N,
    parse_lo_board,
    induce_toggle_mode,
    solve_lights_out_universal,
    solve_sokoban_universal,
    solve_inertial_slide_universal,
    solve_knight_universal,
    solve_cardinal_astar,
    solve_coupled_joint_astar,
    solve_escort_follower_astar,
    solve_delayed_kinematics_astar,
    solve_echo_kinematics_bfs,
    predict_discrete_hazard_trajectory,
    pack_signature_64,
    fast_grid_hash,
)
from project_override.core.world_model_engine import (
    UniversalSemanticDeltaEngine,
    WorldModelTransitionGraph,
    FunctionalRole,
)
from project_override.core.goal_induction_engine import (
    DeltaGoalInductionEngine,
    GoalCategory,
    GoalGuidance,
)
from project_override.core.simulated_planner import (
    SimulatedModelPlanner,
    HierarchicalSubgoalPlanner,
    KinematicTransitionModel,
    SimulatedState,
    DeadlockDetector,
    EpistemicProber,
    VerificationDeltaController,
)
from project_override.core.dead_signature import (
    is_perimeter_timer_tick,
    mask_perimeter_timer_pixels,
)
from project_override.core.subframe_perception import (
    SubFrameDecomposer,
    LatentEpistemicGrid,
)
from project_override.core.empirical_planner import (
    EMPAObjectCuriosityPlanner,
    SplitPanelD4Solver,
    ExactCoverDLX,
)
from project_override.core.visual_goal_induction import VisualGoalInductionEngine
from project_override.core.circuit_engine import (
    UniversalCircuitEngineFacade,
    UniversalCircuitPerception,
    UniversalCircuitSolver,
)
from project_override.core.delay_hold_engine import (
    ActionPipelineTracker,
    solve_delayed_pipeline_astar,
    HoldMechanicInducer,
    synthesize_holding_action,
    solve_inertial_slide_to_hold,
)
from project_override.core.triangulation_engine import (
    ManhattanTriangulator,
    PerceptualBeaconSignalExtractor,
    UniversalBeaconPlanner,
)
from project_override.core.combinatorial_engine import (
    UniversalGF2LinearSolver,
    UniversalBooleanSATSolver,
    UniversalIndependentSetSolver,
    InteractiveMatrixInducer,
    SatisfactionBarInfo,
    UniversalSatisfactionBarDetector,
    UniversalSATClauseSolver,
    UniversalSubmitButtonDetector,
    UniversalOptionSelectorEngine,
)
from project_override.core.passive_conveyor_engine import (
    PassiveVelocityField,
    solve_conveyor_astar,
)
from project_override.core.online_world_model_synthesizer import (
    OnlineWorldModelSynthesizer,
    MultiStrideKinematicsSimulator,
)
from project_override.orchestration.macro_queue import (
    MacroActionQueue,
)
from project_override.orchestration.system_pacer import SystemPacer
from project_override.orchestration.memory_ledger import GameMemoryLedger
from project_override.orchestration.stagnation_supervisor import StagnationSupervisor
from project_override.world_model.retrodict_synthesizer import (
    HistoricalTransition,
    OfflineLLMRuntime,
    RetrodictWorldModelSynthesizer,
)


class GeneralAgent:
    """Universal Zero-Overfitting ARC-AGI-3 Agent."""

    def __init__(
        self,
        total_budget: float = 32400.0,
        total_games: int = 135,
        max_level_timeout: float = 35.0,
    ) -> None:
        self.pacer = SystemPacer(
            total_budget=total_budget,
            total_games=total_games,
            max_level_timeout=max_level_timeout,
        )
        self.macro_queue = MacroActionQueue(max_burst_size=50)
        self.prober = DynamicKinematicProber()
        self.memory_matcher = UniversalMemoryMatcher()
        self.goal_inducer = VisualGoalInductionEngine()
        self.memory_ledger = GameMemoryLedger()
        self.stagnation_supervisor = StagnationSupervisor()
        self.semantic_delta_engine = UniversalSemanticDeltaEngine()
        self.world_model_graph = WorldModelTransitionGraph(self.semantic_delta_engine)
        self.delta_goal_engine = DeltaGoalInductionEngine()
        self.simulated_planner = SimulatedModelPlanner(border_margin=4)
        self.hierarchical_planner = HierarchicalSubgoalPlanner(base_planner=self.simulated_planner)
        self.deployed_beacons: Set[Tuple[int, int]] = set()
        self.flagged_coords: Set[Tuple[int, int]] = set()

        # Dynamic state per game
        self.current_game_id: str = "default"
        self.current_level: int = 0
        self.action_counter: int = 0

        # Memory & Dead Signatures
        self.dead_signatures: Set[int] = set()
        self.dead_walls: Set[Tuple[int, int]] = set()
        self.dead_clicks: Set[Tuple[int, int]] = set()

        # Observations
        self.prev_grid: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_action_data: Dict[str, Any] = {}
        self.waiting_ticks: int = 0

        # Invariant transfer across levels of the same game
        self.learned_avatar_color: Optional[int] = None
        self.learned_displacement_vectors: Dict[int, Tuple[int, int]] = {}
        self.learned_lattice: Optional[LatticeGrid] = None
        self.learned_game_modality: Optional[str] = None
        self.learned_lo_mode: Optional[str] = None
        self.learned_lo_off_color: Optional[int] = None
        self.lo_lvl_N: Optional[int] = None
        self.learned_slide_push: bool = False
        self.learned_wall_colors: Set[int] = set()
        self.learned_box_colors: Set[int] = set()
        self.confirmed_box_colors: Set[int] = set()
        self.learned_goal_colors: Set[int] = set()
        self.learned_trail_colors: Set[int] = set()
        self.learned_co_avatar_colors: Set[int] = set()
        self.learned_command_latency: int = 0
        self.current_player_velocity: Tuple[int, int] = (0, 0)
        self.winning_action_tails: Dict[int, List[int]] = {}
        self.prev_world: Optional[Dict[str, Any]] = None
        self.level_first_world: Optional[Dict[str, Any]] = None
        self.visited_waypoints: Set[Tuple[int, int]] = set()
        self.level_known_plates: Set[Tuple[int, int]] = set()
        self.level_clicked_dots: Set[Tuple[int, int]] = set()
        self.inactive_tagged_colors: Set[int] = set()
        self.level_tried_options: Set[int] = set()
        self.level_tried_subsets: Set[Any] = set()
        self.option_selector_state: Set[int] = set()
        self.learned_submit_btn: Optional[Tuple[int, int]] = None
        self.learned_graph_rule: Optional[str] = None
        self.last_tried_graph_rule: Optional[str] = None
        self.sat_observed_deltas: Dict[Tuple[int, int], np.ndarray] = {}
        self.option_cursor_confirmed: bool = False

        # Tabu state & periodic orbit cycle avoidance
        self.recent_state_hashes: List[int] = []
        self.recent_actions: List[Tuple[int, Dict[str, Any]]] = []
        self.state_tabu_actions: Dict[int, Set[Any]] = collections.defaultdict(set)
        self.level_successful_clicks: Set[Tuple[int, int]] = set()
        # Dynamic entities & velocity tracking
        self.dynamic_entities: Dict[int, Dict[str, Any]] = {}
        # Subframe perception, EMPA epistemic curiosity, and Split-D4 solvers
        self.subframe_decomposer = SubFrameDecomposer(border_margin=4)
        self.empa_planner = EMPAObjectCuriosityPlanner(border_margin=4)
        self.split_d4_solver = SplitPanelD4Solver()
        # Tier-2 Retrodiction Synthesizer & Transition History
        self.retrodict_synthesizer = RetrodictWorldModelSynthesizer()
        self.verified_world_model: Optional[Any] = None
        self.transition_history: List[HistoricalTransition] = []
        # Breakthrough Kinematic & Topological Engines
        self.pipeline_tracker = ActionPipelineTracker()
        self.hold_inducer = HoldMechanicInducer()
        self.beacon_planner = UniversalBeaconPlanner()
        self.conveyor_field = PassiveVelocityField()
        self.online_synthesizer = OnlineWorldModelSynthesizer()

    def start_new_game(self, game_id: str) -> None:
        self.current_game_id = game_id
        self.current_level = 0
        self.action_counter = 0
        self.dead_signatures.clear()
        self.dead_walls.clear()
        self.dead_clicks.clear()
        self.dynamic_entities.clear()
        self.subframe_decomposer.epistemic_grid.reset_for_level()
        self.empa_planner.reset_for_game()
        self.verified_world_model = None
        self.transition_history.clear()
        self.macro_queue.clear()
        self.prober.reset_for_new_game()
        self.memory_matcher = UniversalMemoryMatcher()
        self.pipeline_tracker = ActionPipelineTracker()
        self.hold_inducer = HoldMechanicInducer()
        self.beacon_planner.reset_level()
        self.conveyor_field = PassiveVelocityField()
        self.online_synthesizer = OnlineWorldModelSynthesizer()
        self.learned_avatar_color = None
        self.learned_displacement_vectors.clear()
        self.learned_lattice = None
        self.learned_game_modality = None
        self.learned_lo_mode = None
        self.learned_lo_off_color = None
        self.lo_lvl_N = None
        self.learned_slide_push = False
        self.learned_wall_colors.clear()
        self.learned_box_colors.clear()
        self.confirmed_box_colors.clear()
        self.learned_goal_colors.clear()
        self.learned_trail_colors.clear()
        self.learned_co_avatar_colors.clear()
        self.learned_command_latency = 0
        self.current_player_velocity = (0, 0)
        self.winning_action_tails.clear()
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = {}
        self.waiting_ticks = 0
        self.prev_world = None
        self.level_first_world = None
        self.visited_waypoints.clear()
        self.level_known_plates.clear()
        self.recent_state_hashes.clear()
        self.recent_actions.clear()
        self.state_tabu_actions.clear()
        self.level_successful_clicks.clear()
        self.memory_ledger.start_new_game(game_id)
        self.delta_goal_engine.reset_for_game(game_id)
        self.deployed_beacons.clear()
        self.flagged_coords.clear()
        self.inactive_tagged_colors.clear()
        self.level_tried_options.clear()
        self.level_tried_subsets.clear()
        self.option_selector_state.clear()
        self.learned_submit_btn = None
        self.learned_graph_rule = None
        self.last_tried_graph_rule = None
        self.sat_observed_deltas.clear()
        self.option_cursor_confirmed = False
        self.start_new_level(0)

    def start_new_level(self, level_num: int) -> None:
        if self.learned_game_modality == "GRAPH":
            if self.last_tried_graph_rule is not None:
                self.learned_graph_rule = self.last_tried_graph_rule
            elif "VERT_COVER" in self.level_tried_subsets:
                self.learned_graph_rule = "VERT_COVER"
            elif "IND_SET" in self.level_tried_subsets:
                self.learned_graph_rule = "IND_SET"
        self.sat_observed_deltas.clear()
        self.current_level = level_num
        self.action_counter = 0
        self.macro_queue.clear()
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = {}
        self.waiting_ticks = 0
        self.dead_clicks.clear()
        self.dead_walls.clear()
        self.dynamic_entities.clear()
        self.memory_ledger.start_new_level(level_num)
        self.delta_goal_engine.reset_for_level(level_num)
        self.deployed_beacons.clear()
        self.flagged_coords.clear()
        self.level_clicked_dots.clear()
        self.level_tried_options.clear()
        self.level_tried_subsets.clear()
        self.option_selector_state.clear()
        self.stagnation_supervisor.reset_for_level()
        self.subframe_decomposer.epistemic_grid.reset_for_level()
        self.empa_planner.reset_level() if hasattr(self.empa_planner, "reset_level") else self.empa_planner.reset_for_level()
        self.pipeline_tracker.reset_for_level()
        self.hold_inducer.reset_for_level()
        self.beacon_planner.reset_level()
        self.learned_lattice = None
        self.lo_lvl_N = None
        self.prev_world = None
        self.level_first_world = None
        self.current_player_velocity = (0, 0)
        self.visited_waypoints.clear()
        self.level_known_plates.clear()
        self.level_pulsed = False
        if hasattr(self, "visited_pulse_waypoints"):
            self.visited_pulse_waypoints.clear()
        self.recent_state_hashes.clear()
        self.recent_actions.clear()
        self.state_tabu_actions.clear()
        self.level_successful_clicks.clear()
        self.visited_pulse_sectors = set()
        if self.learned_game_modality == "SOKOBAN" and not self.confirmed_box_colors:
            self.learned_game_modality = None
        if self.learned_game_modality in ("ECHO", "DELAY") and not self.pipeline_tracker.confirmed:
            self.learned_game_modality = None
            self.learned_command_latency = 0
        self.memory_matcher.reset_for_level()
        self.pacer.start_level()

    def record_level_won(self, winning_actions: List[int]) -> None:
        self.winning_action_tails[self.current_level] = list(winning_actions)
        if self.prev_grid is not None:
            try:
                self.delta_goal_engine.observe_transition(
                    prev_grid=self.prev_grid,
                    curr_grid=self.prev_grid,
                    action=self.last_action or 0,
                    action_data=self.last_action_data,
                    mover_prev=None,
                    mover_curr=None,
                    level_cleared=True,
                    current_level=self.current_level,
                )
            except Exception:
                pass
        if self.learned_game_modality in ("ECHO", "DELAY"):
            self.pipeline_tracker.confirmed = True
        if self.hold_inducer.dwell_ticks > 1:
            self.hold_inducer.hold_confirmed = True
            self.hold_inducer.required_hold_ticks = self.hold_inducer.dwell_ticks
        if self.transition_history:
            self.transition_history[-1].level_cleared = True
        source_w = self.level_first_world if self.level_first_world is not None else self.prev_world
        if source_w is not None:
            pw = source_w
            # Universal physical invariant: if goal was reached across distance D in fewer than D steps,
            # avatar motion is inertial sliding or stride > 1
            p_start = pw.get("player")
            goals = pw.get("all_goals")
            if p_start and goals and winning_actions:
                dir_actions = [a for a in winning_actions if a in (1, 2, 3, 4)]
                has_clicks_or_interact = any(a in (5, 6) for a in winning_actions)
                if not has_clicks_or_interact and dir_actions:
                    min_dist_to_goal = min(
                        max(abs(g[0] - p_start[0]), abs(g[1] - p_start[1]))
                        for g in goals
                    )
                    if min_dist_to_goal > len(dir_actions):
                        self.learned_game_modality = "SLIDE"
                        self.prober.profile.is_sliding = True

            bg_c = pw.get("bg_color", 0)
            frame_cs = set(pw.get("frame_colors", []))
            excluded_cs = {bg_c, self.learned_avatar_color} | frame_cs | self.learned_co_avatar_colors
            if self.confirmed_box_colors:
                # Confirmed Sokoban puzzle: preserve confirmed box colors
                self.learned_box_colors = set(self.confirmed_box_colors)
                self.learned_game_modality = "SOKOBAN"
                if pw.get("plates"):
                    for pr, pc in pw["plates"]:
                        c = int(pw["board"][pr, pc])
                        if (
                            c not in excluded_cs
                            and c not in self.confirmed_box_colors
                            and c not in self.learned_trail_colors
                        ):
                            self.learned_goal_colors.add(c)
                if pw.get("walls"):
                    for wr, wc in pw["walls"]:
                        c = int(pw["board"][wr, wc])
                        if (
                            c not in excluded_cs
                            and c not in self.confirmed_box_colors
                            and c not in self.learned_goal_colors
                        ):
                            self.learned_wall_colors.add(c)
            else:
                # No crates were ever displaced by intervention: environment is NAV / Waypoints
                self.learned_box_colors.clear()
                if self.learned_game_modality == "SOKOBAN":
                    self.learned_game_modality = "NAV"
                for ent_key in ("plates", "exit_doors"):
                    if pw.get(ent_key):
                        for r, c in pw[ent_key]:
                            cv = int(pw["board"][r, c])
                            if (
                                cv not in excluded_cs
                                and cv not in self.learned_trail_colors
                                and cv not in self.learned_wall_colors
                            ):
                                self.learned_goal_colors.add(cv)
                if pw.get("walls"):
                    for wr, wc in pw["walls"]:
                        c = int(pw["board"][wr, wc])
                        if (
                            c not in excluded_cs
                            and c not in self.learned_goal_colors
                        ):
                            self.learned_wall_colors.add(c)

    def choose_action(
        self,
        current_grid: np.ndarray,
        available_actions: Optional[List[int]] = None,
        game_state: Optional[str] = None,
        raw_frame: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        self.action_counter += 1
        legal_actions = available_actions or [1, 2, 3, 4, 5, 6, 7]

        # Ingest sub-frame animation sequences and extract transient flashes
        if raw_frame is not None:
            subframe_analysis = self.subframe_decomposer.analyze(
                raw_frame,
                current_step=self.action_counter,
            )
            if subframe_analysis.has_animation:
                current_grid = subframe_analysis.settled_frame

        # Compute true frame background and lattice separation immediately
        sep_bg = separate_background_and_lattice(current_grid)
        bg_color = sep_bg.bg_color

        # Record empirical transition into history for Tier-2 Retrodiction
        if self.prev_grid is not None and self.last_action is not None:
            H_g, W_g = current_grid.shape[:2]
            t_mask = np.zeros((H_g, W_g), dtype=bool)
            t_mask[:4, :] = True
            t_mask[H_g - 4 :, :] = True
            t_mask[:, :4] = True
            t_mask[:, W_g - 4 :] = True

            click_coord = None
            if self.last_action == 6 and self.last_action_data:
                click_coord = (
                    int(self.last_action_data.get("x", 0)),
                    int(self.last_action_data.get("y", 0)),
                )

            self.transition_history.append(
                HistoricalTransition(
                    step_idx=self.action_counter - 1,
                    pre_grid=self.prev_grid.copy(),
                    pre_hud={},
                    pre_latent=self.subframe_decomposer.epistemic_grid.grid.copy(),
                    action_id=self.last_action,
                    click_xy=click_coord,
                    post_grid=current_grid.copy(),
                    post_hud={},
                    timer_mask=t_mask,
                    level_cleared=False,
                    game_over=False,
                )
            )

        # 0. Check Null Transition on previous step (Dead-Signature Culling & Rollback)
        if (
            self.prev_grid is not None
            and self.last_action is not None
            and self.waiting_ticks == 0
            and not getattr(self.memory_matcher, "last_was_dummy", False)
            and self.learned_game_modality not in ("TOGGLE", "MEMORY", "OPTION")
        ):
            interior_prev = self.prev_grid[4:60, 4:60]
            interior_curr = current_grid[4:60, 4:60]
            delta_omega = np.count_nonzero(interior_prev != interior_curr)
            full_delta = np.count_nonzero(self.prev_grid != current_grid)
            is_timer_tick = is_perimeter_timer_tick(self.prev_grid, current_grid, border_margin=4)

            has_exempt_steps = any("DLX" in s.description or "submit" in s.description.lower() for s in self.macro_queue.plan)
            if delta_omega == 0 and (full_delta == 0 or is_timer_tick) and not has_exempt_steps:
                ctx_h = fast_grid_hash(interior_prev)
                sig = pack_signature_64(
                    action_byte=self.last_action,
                    coords_12=0,
                    mass_color_12=0,
                    ctx_hash_32=ctx_h,
                )
                self.dead_signatures.add(sig)

                if self.last_action == 6 and self.last_action_data:
                    clk_pos = (
                        int(self.last_action_data.get("x", 0)),
                        int(self.last_action_data.get("y", 0)),
                    )
                    is_submit = (
                        (self.learned_submit_btn is not None and abs(clk_pos[0] - self.learned_submit_btn[0]) <= 2 and abs(clk_pos[1] - self.learned_submit_btn[1]) <= 2)
                        or (clk_pos[1] in (0, 1, current_grid.shape[0] - 2, current_grid.shape[0] - 1) and clk_pos[0] in (0, 1, 2))
                        or (clk_pos[1] <= 8 and clk_pos[0] >= 30)
                        or (clk_pos[1] >= current_grid.shape[0] - 9 and clk_pos[0] >= 30)
                    )
                    if not is_submit:
                        self.dead_clicks.add(clk_pos)
                    elif self.learned_game_modality == "OPTION":
                        h_g = current_grid.shape[0]
                        has_active_ind = False
                        for r_check in [h_g - 2, h_g - 3, 1, 2]:
                            r_slice = current_grid[r_check, :]
                            if np.any((r_slice != bg_color) & (r_slice != 4) & (r_slice != 0)):
                                has_active_ind = True
                                break
                        if not has_active_ind:
                            self.option_selector_state.clear()
                elif self.last_action == 5:
                    self.level_successful_clicks.clear()

                if not self.macro_queue.is_empty:
                    self.macro_queue.clear()
            elif delta_omega > 0:
                is_reset_or_over = game_state is not None and ("RESET" in str(game_state).upper() or "GAME_OVER" in str(game_state).upper())
                if is_reset_or_over:
                    if self.last_action == 6 and self.last_action_data:
                        bad_clk = (int(self.last_action_data.get("x", 0)), int(self.last_action_data.get("y", 0)))
                        is_submit = (
                            (self.learned_submit_btn is not None and abs(bad_clk[0] - self.learned_submit_btn[0]) <= 2 and abs(bad_clk[1] - self.learned_submit_btn[1]) <= 2)
                            or (bad_clk[1] in (0, 1, current_grid.shape[0] - 2, current_grid.shape[0] - 1) and bad_clk[0] in (0, 1, 2))
                            or (bad_clk[1] <= 8 and bad_clk[0] >= 30)
                            or (bad_clk[1] >= current_grid.shape[0] - 9 and bad_clk[0] >= 30)
                        )
                        if not is_submit:
                            self.dead_clicks.add(bad_clk)
                            self.flagged_coords.add(bad_clk)
                    self.macro_queue.clear()
                else:
                    if self.last_action == 5:
                        self.dead_walls.clear()
                    if (
                        self.last_action == 6
                        and self.last_action_data
                        and self.learned_game_modality not in ("TOGGLE", "MEMORY")
                    ):
                        clk_x = int(self.last_action_data.get("x", 0))
                        clk_y = int(self.last_action_data.get("y", 0))
                        self.level_successful_clicks.add((clk_x, clk_y))
                        if 0 <= clk_y < current_grid.shape[0] and 0 <= clk_x < current_grid.shape[1] and self.prev_grid is not None:
                            c_new = int(current_grid[clk_y, clk_x])
                            c_old = int(self.prev_grid[clk_y, clk_x])
                            if c_new != c_old and c_new != int(current_grid[0, 0]):
                                self.inactive_tagged_colors.add(c_new)
                                self.learned_goal_colors.discard(c_new)

                        if self.prev_grid is not None:
                            p_bar = UniversalSatisfactionBarDetector.detect(self.prev_grid, bg_color)
                            c_bar = UniversalSatisfactionBarDetector.detect(current_grid, bg_color)
                            if p_bar is not None and c_bar is not None and p_bar.num_clauses == c_bar.num_clauses:
                                delta = (c_bar.satisfied_mask != p_bar.satisfied_mask).astype(np.uint8)
                                self.sat_observed_deltas[(clk_x, clk_y)] = delta

                    if self.last_action in (1, 2, 3, 4) and self.prev_grid is not None:
                        prev_cursor = UniversalOptionSelectorEngine.detect_selector_cursor(self.prev_grid, bg_color)
                        curr_cursor = UniversalOptionSelectorEngine.detect_selector_cursor(current_grid, bg_color)
                        if prev_cursor is not None and curr_cursor is not None:
                            if prev_cursor[1] != curr_cursor[1] and prev_cursor[2] == curr_cursor[2]:
                                self.option_cursor_confirmed = True
                                self.learned_game_modality = "OPTION"
                                self.learned_avatar_color = None

        # 0.1 Extract current world state if avatar is known
        world: Optional[Dict[str, Any]] = None
        if (
            self.learned_avatar_color is not None
            and self.learned_game_modality in [None, "NAV", "SLIDE", "SOKOBAN", "KNIGHT", "DELAY", "ECHO", "CONVEYOR"]
        ):
            world = extract_discrete_board(
                current_grid,
                avatar_color=self.learned_avatar_color,
                prev_world=self.prev_world,
                last_action=self.last_action,
                learned_wall_colors=self.learned_wall_colors,
                learned_box_colors=self.learned_box_colors,
                learned_goal_colors=self.learned_goal_colors,
                learned_co_avatar_colors=self.learned_co_avatar_colors,
                learned_trail_colors=self.learned_trail_colors,
                conveyor_colors=set(self.conveyor_field.tile_color_velocity.keys()),
            )

        # Grounded Feedback & Stagnation Supervisor (AVO limit-cycle & dead-end detector)
        if self.prev_grid is not None and self.last_action is not None:
            p_prev = self.prev_world.get("player") if self.prev_world else None
            p_curr = world.get("player") if world else None
            self.memory_ledger.record_transition(
                action=self.last_action,
                action_data=self.last_action_data,
                prev_frame=self.prev_grid,
                curr_frame=current_grid,
                mover_prev=p_prev,
                mover_curr=p_curr,
            )
            try:
                frame_delta = self.semantic_delta_engine.process_transition(
                    pre_grid=self.prev_grid,
                    post_grid=current_grid,
                    action=self.last_action,
                    action_data=self.last_action_data,
                    step_idx=self.action_counter,
                    level_cleared=False,
                )
                self.world_model_graph.record_transition(frame_delta)
                self.delta_goal_engine.observe_transition(
                    prev_grid=self.prev_grid,
                    curr_grid=current_grid,
                    action=self.last_action,
                    action_data=self.last_action_data,
                    mover_prev=p_prev,
                    mover_curr=p_curr,
                    level_cleared=False,
                    current_level=self.current_level,
                )
                if p_prev and p_curr and self.last_action in [1, 2, 3, 4]:
                    act_vec = self.learned_displacement_vectors.get(self.last_action, {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(self.last_action, (0, 0)))
                    self.conveyor_field.observe_transition(
                        p_before=p_prev,
                        p_after=p_curr,
                        action_vector=act_vec,
                        board=world.get("board") if world else None,
                        walls=world.get("walls", set()) if world else set(),
                        gh=world.get("gh", 10) if world else 10,
                        gw=world.get("gw", 10) if world else 10,
                    )
                    if self.conveyor_field.has_conveyors_active():
                        self.learned_game_modality = "CONVEYOR"
                        self.learned_wall_colors -= set(self.conveyor_field.tile_color_velocity.keys())
                if p_prev and p_curr:
                    st_prev = (p_prev, fast_grid_hash(self.prev_grid[4:60, 4:60]))
                    st_curr = (p_curr, fast_grid_hash(current_grid[4:60, 4:60]))
                    self.online_synthesizer.record_transition(
                        state_sig=st_prev,
                        action=self.last_action,
                        next_state_sig=st_curr,
                    )
            except Exception:
                pass

        if self.learned_game_modality not in ("TOGGLE", "MEMORY", "OPTION"):
            grid_changed = (self.prev_grid is None) or bool(np.any(current_grid != self.prev_grid))
            stag_check = self.stagnation_supervisor.on_step(
                current_grid=current_grid,
                last_action=self.last_action,
                last_action_data=self.last_action_data,
                grid_changed=grid_changed,
            )
            if stag_check["stagnation_detected"]:
                self.macro_queue.clear()
                if self.last_action == 6 and self.last_action_data:
                    bad_c = (int(self.last_action_data.get("x", 0)), int(self.last_action_data.get("y", 0)))
                    is_submit = (
                        (self.learned_submit_btn is not None and abs(bad_c[0] - self.learned_submit_btn[0]) <= 2 and abs(bad_c[1] - self.learned_submit_btn[1]) <= 2)
                        or (bad_c[1] in (0, 1, current_grid.shape[0] - 2, current_grid.shape[0] - 1) and bad_c[0] in (0, 1, 2))
                        or (bad_c[1] <= 8 and bad_c[0] >= 30)
                        or (bad_c[1] >= current_grid.shape[0] - 9 and bad_c[0] >= 30)
                    )
                    if not is_submit:
                        self.dead_clicks.add(bad_c)
                        self.flagged_coords.add(bad_c)
                elif self.last_action in [1, 2, 3, 4] and p_prev:
                    self.dead_walls.add(p_prev)

                # Only force emergency diversion if stagnation persists across multiple planning cycles
                if getattr(self.stagnation_supervisor, "stagnation_event_count", 0) >= 3:
                    diversion_act = self.stagnation_supervisor.get_diversion_action(
                        current_state_hash=self.stagnation_supervisor.state_hashes[-1],
                        legal_actions=legal_actions,
                        recent_actions=[a for a, _ in self.recent_actions[-6:]],
                    )
                    if diversion_act is not None and diversion_act in legal_actions:
                        self.last_action = diversion_act
                        self.prev_grid = current_grid.copy()
                        self.recent_actions.append((diversion_act, {}))
                        if len(self.recent_actions) > 16:
                            self.recent_actions.pop(0)
                        return diversion_act, {}

            curr_interior = current_grid[4:60, 4:60]
            ctx_curr_h = fast_grid_hash(curr_interior)
            if len(self.recent_state_hashes) >= 2:
                for cycle_len in range(2, min(5, len(self.recent_state_hashes) + 1)):
                    if ctx_curr_h == self.recent_state_hashes[-cycle_len]:
                        act_taken, data_taken = self.recent_actions[-cycle_len]
                        if act_taken == 6 and data_taken:
                            self.state_tabu_actions[ctx_curr_h].add((
                                int(data_taken.get("x", 0)),
                                int(data_taken.get("y", 0)),
                            ))
                        else:
                            self.state_tabu_actions[ctx_curr_h].add(act_taken)
                        break
            self.recent_state_hashes.append(ctx_curr_h)
            if len(self.recent_state_hashes) > 16:
                self.recent_state_hashes.pop(0)

        # 1. System budget timeout watchdog
        if self.pacer.is_level_timed_out:
            fallback = 1 if 1 in legal_actions else legal_actions[0]
            self.last_action = fallback
            self.prev_grid = current_grid.copy()
            self.recent_actions.append((fallback, {}))
            if len(self.recent_actions) > 16:
                self.recent_actions.pop(0)
            return fallback, {}

        # 2. Update board world observation & empirical object-role induction
        sep_bg = separate_background_and_lattice(current_grid)
        bg_color = sep_bg.bg_color
        if (
            self.learned_avatar_color is not None
            and self.learned_game_modality in [None, "NAV", "SLIDE", "SOKOBAN", "KNIGHT", "DELAY", "ECHO", "CONVEYOR"]
        ):
            prev_w = self.prev_world
            if world is None:
                world = extract_discrete_board(
                    current_grid,
                    avatar_color=self.learned_avatar_color,
                    prev_world=prev_w,
                    last_action=self.last_action,
                    learned_wall_colors=self.learned_wall_colors,
                    learned_box_colors=self.learned_box_colors,
                    learned_goal_colors=self.learned_goal_colors,
                    learned_co_avatar_colors=self.learned_co_avatar_colors,
                    learned_trail_colors=self.learned_trail_colors,
                    conveyor_colors=set(self.conveyor_field.tile_color_velocity.keys()),
                )
            if world is not None:
                if self.level_first_world is None:
                    self.level_first_world = dict(world)
                if prev_w is not None and self.last_action in (1, 2, 3, 4, 5):
                    if self.learned_displacement_vectors:
                        deltas = dict(self.learned_displacement_vectors)
                        scale = max(1, world.get("scale", 1))
                        if scale > 1:
                            deltas = {
                                a: (int(round(v[0] / scale)), int(round(v[1] / scale)))
                                if (abs(v[0]) > 1 or abs(v[1]) > 1) else v
                                for a, v in deltas.items()
                            }
                    else:
                        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 5: (0, 0)}
                    if 5 not in deltas:
                        deltas[5] = (0, 0)

                    dr, dc = deltas.get(self.last_action, (0, 0))
                    expected_p = (prev_w["player"][0] + dr, prev_w["player"][1] + dc)
                    p_prev = prev_w["player"]
                    p_curr = world["player"]
                    dp = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])

                    # Empirical sliding kinematics induction
                    if abs(dp[0]) >= 2 or abs(dp[1]) >= 2:
                        self.learned_game_modality = "SLIDE"
                        self.prober.profile.is_sliding = True
                        self.macro_queue.clear()

                    # Sync displacement kinematics into pipeline tracker
                    self.pipeline_tracker.set_deltas(deltas)

                    # Pipeline delay tracking
                    k_inferred = self.pipeline_tracker.record_transition(
                        action=self.last_action,
                        p_before=prev_w["player"],
                        p_after=world["player"],
                    )
                    if k_inferred > 1 and self.pipeline_tracker.confirmed:
                        self.learned_command_latency = k_inferred
                        self.learned_game_modality = "DELAY"
                        self.macro_queue.clear()
                    elif k_inferred >= 1 and self.pipeline_tracker.confirmed:
                        self.learned_command_latency = k_inferred
                        if 5 in legal_actions or self.learned_game_modality == "DELAY":
                            self.learned_game_modality = "DELAY"
                        else:
                            self.learned_game_modality = "ECHO"
                        self.macro_queue.clear()
                    elif len(self.recent_actions) >= 2:
                        prev_act, _ = self.recent_actions[-2]
                        if prev_act in deltas and self.last_action in deltas and self.last_action != prev_act:
                            if dp == deltas[prev_act] and dp != deltas[self.last_action]:
                                self.learned_command_latency = 1
                                self.learned_game_modality = "DELAY" if 5 in legal_actions else "ECHO"
                                self.pipeline_tracker.inferred_k = 1
                                self.pipeline_tracker.confirmed = True
                                self.macro_queue.clear()

                    if self.last_action in deltas:
                        self.current_player_velocity = deltas[self.last_action]

                    if self.last_action in (1, 2, 3, 4):
                        # Empirical crate push & slide-push induction
                        moved_crate_detected = False
                        for ent_key in ("boxes", "plates"):
                            if prev_w[ent_key] != world[ent_key]:
                                moved_old = prev_w[ent_key] - world[ent_key]
                                moved_new = world[ent_key] - prev_w[ent_key]
                                if len(moved_old) == 1 and len(moved_new) == 1:
                                    ob = next(iter(moved_old))
                                    nb = next(iter(moved_new))
                                    is_causally_aligned = (
                                        ob == expected_p
                                        and world["player"] == expected_p
                                        and (
                                            (dr != 0 and (nb[0] - ob[0]) * dr > 0 and nb[1] == ob[1])
                                            or (dc != 0 and (nb[1] - ob[1]) * dc > 0 and nb[0] == ob[0])
                                        )
                                    )
                                    if is_causally_aligned:
                                        box_c = int(prev_w["board"][ob[0], ob[1]])
                                        frame_colors = set(sep_bg.frame_colors)
                                        excluded_fg = {bg_color, self.learned_avatar_color} | frame_colors | self.learned_co_avatar_colors
                                        if box_c not in excluded_fg:
                                            self.confirmed_box_colors.add(box_c)
                                            self.learned_box_colors.add(box_c)
                                            self.learned_goal_colors.discard(box_c)
                                            self.learned_game_modality = "SOKOBAN"
                                            moved_crate_detected = True
                                        if abs(ob[0] - nb[0]) + abs(ob[1] - nb[1]) >= 2 and not self.learned_slide_push:
                                            self.learned_slide_push = True
                                            self.macro_queue.clear()
                                        break
                        if not moved_crate_detected:
                            frame_colors = set(sep_bg.frame_colors)
                            excluded_fg = {bg_color, self.learned_avatar_color} | frame_colors | self.learned_co_avatar_colors
                            if world["player"] == prev_w["player"]:
                                # Player did not move into expected cell (Collision/Wall)
                                if expected_p in prev_w["boxes"]:
                                    if not self.learned_slide_push:
                                        self.learned_slide_push = True
                                        self.macro_queue.clear()
                                    else:
                                        blocked_c = int(prev_w["board"][expected_p[0], expected_p[1]])
                                        cur_p = world["player"]
                                        p_cur_c = int(world["board"][cur_p[0], cur_p[1]])
                                        if (
                                            blocked_c not in (excluded_fg | {p_cur_c})
                                            and blocked_c not in self.learned_goal_colors
                                            and not self.conveyor_field.has_conveyors_active()
                                        ):
                                            all_candidates = {int(c) for c in np.unique(world["board"])} - (excluded_fg | {p_cur_c})
                                            if len(all_candidates) > 1 and blocked_c not in self.confirmed_box_colors:
                                                self.learned_wall_colors.add(blocked_c)
                                                self.learned_box_colors.discard(blocked_c)
                                        self.learned_game_modality = "NAV"
                                        self.dead_walls.add(expected_p)
                                        self.macro_queue.clear()
                                elif self.learned_game_modality not in ("SLIDE", "KNIGHT") and self.learned_command_latency == 0:
                                    if 0 <= expected_p[0] < world["gh"] and 0 <= expected_p[1] < world["gw"]:
                                        self.dead_walls.add(expected_p)
                                        blocked_c = int(prev_w["board"][expected_p[0], expected_p[1]])
                                        cur_p = world["player"]
                                        p_cur_c = int(world["board"][cur_p[0], cur_p[1]])
                                        if (
                                            blocked_c not in (excluded_fg | {p_cur_c})
                                            and blocked_c not in self.learned_goal_colors
                                            and not self.conveyor_field.has_conveyors_active()
                                        ):
                                            all_candidates = {int(c) for c in np.unique(world["board"])} - (excluded_fg | {p_cur_c})
                                            if len(all_candidates) > 1:
                                                self.learned_wall_colors.add(blocked_c)
                                        self.macro_queue.clear()
                            elif world["player"] == expected_p:
                                # Player successfully moved into expected cell without displacing a crate!
                                cell_c = int(prev_w["board"][expected_p[0], expected_p[1]])
                                if expected_p in prev_w["boxes"] or cell_c in self.confirmed_box_colors:
                                    # The avatar stepped directly onto a cell hypothesized to be a crate without displacing it!
                                    # In physical mechanics, an impassable/pushable crate cannot be stepped on directly.
                                    # Therefore, this entity is a passable waypoint / plate / collectible, NOT a crate!
                                    if cell_c not in excluded_fg:
                                        self.confirmed_box_colors.discard(cell_c)
                                        self.learned_box_colors.discard(cell_c)
                                        self.learned_goal_colors.add(cell_c)
                                        self.visited_waypoints.add(expected_p)
                                        if self.learned_game_modality == "SOKOBAN" and not self.confirmed_box_colors:
                                            self.learned_game_modality = "NAV"
                                        self.macro_queue.clear()
                            elif self.learned_command_latency == 1:
                                if world["player"] == prev_w["player"] and self.current_player_velocity != (0, 0):
                                    hit_pos = (prev_w["player"][0] + self.current_player_velocity[0], prev_w["player"][1] + self.current_player_velocity[1])
                                    if 0 <= hit_pos[0] < world["gh"] and 0 <= hit_pos[1] < world["gw"] and hit_pos not in world["all_goals"]:
                                        self.dead_walls.add(hit_pos)
                                    self.macro_queue.clear()
                            elif world["player"] != expected_p and self.learned_game_modality not in ("SLIDE", "KNIGHT") and self.learned_command_latency == 0:
                                if expected_p not in world["all_goals"]:
                                    self.dead_walls.add(expected_p)
                                self.macro_queue.clear()

                    # Causal Trail / Footprint Induction
                    if (
                        prev_w is not None
                        and world["player"] != prev_w["player"]
                        and prev_w.get("gh") == world.get("gh")
                        and prev_w.get("gw") == world.get("gw")
                    ):
                        p_prev = prev_w["player"]
                        if 0 <= p_prev[0] < world["gh"] and 0 <= p_prev[1] < world["gw"]:
                            c_trail = int(world["board"][p_prev[0], p_prev[1]])
                            frame_colors = set(sep_bg.frame_colors)
                            excluded_fg = {bg_color, self.learned_avatar_color} | frame_colors | self.learned_co_avatar_colors
                            if (
                                c_trail not in excluded_fg
                                and c_trail not in self.learned_goal_colors
                            ):
                                self.learned_trail_colors.add(c_trail)
                                self.learned_wall_colors.add(c_trail)
                                self.learned_goal_colors.discard(c_trail)
                                self.confirmed_box_colors.discard(c_trail)
                                self.learned_box_colors.discard(c_trail)
                                world["plates"] = {p for p in world["plates"] if int(world["board"][p[0], p[1]]) != c_trail}
                                world["boxes"] = {b for b in world["boxes"] if int(world["board"][b[0], b[1]]) != c_trail}
                                world["walls"].add(p_prev)

                # Dynamic entity velocity induction via causal non-intervention
                if prev_w is not None and prev_w.get("gh") == world.get("gh") and prev_w.get("gw") == world.get("gw"):
                    gh_d, gw_d = world["gh"], world["gw"]
                    p_prev = prev_w["player"]
                    p_curr = world["player"]
                    excluded_cells = {p_prev, p_curr}
                    if "ob" in locals() and "nb" in locals() and moved_crate_detected:
                        excluded_cells.add(ob)
                        excluded_cells.add(nb)

                    frame_colors = set(sep_bg.frame_colors)
                    excluded_fg = {bg_color, self.learned_avatar_color} | frame_colors | self.learned_co_avatar_colors
                    for r in range(gh_d):
                        for c in range(gw_d):
                            if (r, c) in excluded_cells:
                                continue
                            c_prev = int(prev_w["board"][r, c])
                            c_curr = int(world["board"][r, c])
                            if c_prev not in excluded_fg and c_curr == bg_color:
                                for nr in range(max(0, r - 2), min(gh_d, r + 3)):
                                    for nc in range(max(0, c - 2), min(gw_d, c + 3)):
                                        if (nr, nc) in excluded_cells or (nr == r and nc == c):
                                            continue
                                        if int(world["board"][nr, nc]) == c_prev and int(prev_w["board"][nr, nc]) != c_prev:
                                            vel = (nr - r, nc - c)
                                            self.dynamic_entities[c_prev] = {
                                                "pos": (nr, nc),
                                                "vel": vel,
                                                "color": c_prev,
                                            }
                                            self.learned_wall_colors.discard(c_prev)
                                            self.learned_box_colors.discard(c_prev)
                                            self.confirmed_box_colors.discard(c_prev)
                                            world["walls"] = {w for w in world["walls"] if int(world["board"][w[0], w[1]]) != c_prev}
                                            world["boxes"] = {b for b in world["boxes"] if int(world["board"][b[0], b[1]]) != c_prev}
                                            world["plates"] = {p for p in world["plates"] if int(world["board"][p[0], p[1]]) != c_prev}
                                            break

                    for ent_c, ent_info in list(self.dynamic_entities.items()):
                        curr_matches = np.argwhere(world["board"] == ent_c)
                        if len(curr_matches) > 0:
                            ent_info["pos"] = (int(curr_matches[0][0]), int(curr_matches[0][1]))

                self.prev_world = world
                p_pos = world["player"]
                world["walls"].discard(p_pos)
                if 0 <= p_pos[0] < world["gh"] and 0 <= p_pos[1] < world["gw"]:
                    cur_tile_c = int(world["board"][p_pos[0], p_pos[1]])
                    if cur_tile_c in self.learned_wall_colors and cur_tile_c != world["bg_color"]:
                        self.learned_wall_colors.discard(cur_tile_c)
                for conv_c in self.conveyor_field.tile_color_velocity.keys():
                    self.learned_wall_colors.discard(conv_c)
                    world["walls"] = {w for w in world["walls"] if int(world["board"][w[0], w[1]]) != conv_c}
                if not world["boxes"] and p_pos in world["plates"]:
                    self.visited_waypoints.add(p_pos)
                    unvisited = [p for p in world["plates"] if p not in self.visited_waypoints and p != p_pos]
                    if not unvisited and world["exit_doors"]:
                        self.macro_queue.clear()

        # 3. Macro Queue Execution (Bursts with Verification Delta & Fast Rollback)
        if not self.macro_queue.is_empty:
            step = self.macro_queue.plan[0]
            # Sahu Verification Delta: Verify world state against expectation
            if step.expected_player_pos is not None and world is not None:
                if world["player"] != step.expected_player_pos:
                    # Divergence: live state diverged from expectation! Halt and purge macro queue.
                    self.macro_queue.clear()

            if not self.macro_queue.is_empty:
                step = self.macro_queue.plan.pop(0)
                if step.action in legal_actions:
                    self.last_action = step.action
                    self.last_action_data = step.action_data
                    self.prev_grid = current_grid.copy()
                    self.recent_actions.append((step.action, dict(step.action_data)))
                    if len(self.recent_actions) > 16:
                        self.recent_actions.pop(0)
                    return step.action, step.action_data

        # 4. Core Policy: Plan Next Action
        action, action_data = self._plan_universal_step(current_grid, legal_actions, world=world)

        self.last_action = action
        self.last_action_data = action_data
        self.prev_grid = current_grid.copy()
        self.recent_actions.append((action, dict(action_data)))
        if len(self.recent_actions) > 16:
            self.recent_actions.pop(0)
        return action, action_data

    def _plan_universal_step(
        self,
        grid: np.ndarray,
        legal_actions: List[int],
        world: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Universal, zero-overfitting planner."""
        sep = separate_background_and_lattice(grid)
        bg_color = sep.bg_color
        interior = grid[sep.y_min:sep.y_max, sep.x_min:sep.x_max]

        # -------------------------------------------------------------
        # Branch A: Dynamic Kinematic Probing (Level 1, Steps 0–3)
        # -------------------------------------------------------------
        if self.current_level == 0 and not self.prober.profile.probed and 1 in legal_actions:
            probe_act = self.prober.get_probe_action(grid, legal_actions)
            if getattr(self.prober.profile, "trail_colors", None):
                self.learned_trail_colors.update(self.prober.profile.trail_colors)
            if self.prober.profile.avatar_color is not None:
                self.learned_avatar_color = self.prober.profile.avatar_color
                if getattr(self.prober.profile, "co_avatar_colors", None):
                    self.learned_co_avatar_colors.update(self.prober.profile.co_avatar_colors)
                self.learned_displacement_vectors = dict(self.prober.profile.displacement_vectors)
                if self.prober.profile.modality:
                    self.learned_game_modality = self.prober.profile.modality
            elif self.prober.profile.probed:
                self.learned_avatar_color = None
                self.learned_game_modality = self.prober.profile.modality or "CLICK"

            if probe_act is not None and probe_act in legal_actions:
                return probe_act, {}

        if self.prober.profile.probed and self.learned_game_modality is None:
            if getattr(self.prober.profile, "trail_colors", None):
                self.learned_trail_colors.update(self.prober.profile.trail_colors)
            if self.prober.profile.avatar_color is not None:
                self.learned_avatar_color = self.prober.profile.avatar_color
                if getattr(self.prober.profile, "co_avatar_colors", None):
                    self.learned_co_avatar_colors.update(self.prober.profile.co_avatar_colors)
                self.learned_displacement_vectors = dict(self.prober.profile.displacement_vectors)
                self.learned_game_modality = self.prober.profile.modality
            else:
                self.learned_game_modality = "CLICK"

        # -------------------------------------------------------------
        # Branch B: Click / Interactive Matrix Environments (Action 6)
        # -------------------------------------------------------------
        submit_btn = UniversalSubmitButtonDetector.detect_submit_button(grid, bg_color, set(sep.frame_colors))
        if submit_btn is not None:
            self.learned_submit_btn = submit_btn
        elif self.learned_submit_btn is not None:
            submit_btn = self.learned_submit_btn

        if self.action_counter <= 10:
            print(f"[DEBUG {self.current_game_id}] act_cnt={self.action_counter} legal={legal_actions} submit_btn={submit_btn} modality={self.learned_game_modality} sep_bg={bg_color} frame={sep.frame_colors}")

        selector_cursor = UniversalOptionSelectorEngine.detect_selector_cursor(grid, bg_color)

        has_interactive_matrix = False
        if 6 in legal_actions:
            try:
                has_interactive_matrix = (
                    sep.lattice is not None
                    or bool(UniversalCircuitEngineFacade.generate_click_actions(grid))
                    or bool(InteractiveMatrixInducer.extract_clickable_rows_or_components(grid, bg_color, set(sep.frame_colors)))
                    or UniversalSatisfactionBarDetector.detect(grid, bg_color) is not None
                )
            except Exception:
                has_interactive_matrix = False

        is_click_mode = (
            self.learned_game_modality in ["TOGGLE", "MEMORY", "CLICK", "SAT", "OPTION", "GRAPH"]
            or (self.learned_avatar_color is None and not any(a in legal_actions for a in [1, 2, 3, 4]))
            or (self.prober.profile.probed and self.learned_avatar_color is None)
            or (6 in legal_actions and has_interactive_matrix and not self.confirmed_box_colors)
            or (6 in legal_actions and submit_btn is not None)
        )

        if 6 in legal_actions and is_click_mode:
            active_submit = submit_btn or self.learned_submit_btn

            # 0. SAT Satisfaction Bar & Clause Toggle Fast-Path (e.g. cw01, ct01, cx01)
            sat_bar = UniversalSatisfactionBarDetector.detect(grid, bg_color)
            if sat_bar is not None and self.learned_game_modality not in ("OPTION", "GRAPH"):
                past_clicks = [
                    (int(a_data.get("x", 0)), int(a_data.get("y", 0)))
                    for _, a_data in self.recent_actions
                    if a_data and "x" in a_data
                ]
                sat_clicks = UniversalSATClauseSolver.plan_sat_actions(
                    grid,
                    self.prev_grid,
                    bg_color,
                    set(sep.frame_colors),
                    past_clicks,
                    submit_btn=active_submit,
                    observed_deltas=self.sat_observed_deltas,
                )
                if sat_clicks:
                    self.learned_game_modality = "SAT"
                    for clk in sat_clicks[1:]:
                        self.macro_queue.add_step(
                            action=6,
                            action_data={"x": int(clk[0]), "y": int(clk[1])},
                            description="SAT variable toggle click",
                        )
                    return 6, {"x": int(sat_clicks[0][0]), "y": int(sat_clicks[0][1])}

            # 0.5 Submit Button & Discrete Option Selector Fast-Path (e.g. cz01)
            if active_submit is not None and self.learned_game_modality not in ("SAT", "GRAPH"):
                if (selector_cursor is not None or self.learned_game_modality == "OPTION") and any(a in legal_actions for a in [1, 2, 3, 4]):
                    self.learned_avatar_color = None
                    self.learned_game_modality = "OPTION"
                    avail_options = [a for a in [1, 2, 3, 4] if a in legal_actions]

                    # Generate candidate option subsets in increasing cardinality:
                    # Size 1 (singletons), Size 2 (pairs), Size 3 (triples), Size 4
                    all_subsets: List[Tuple[int, ...]] = []
                    for k in range(1, len(avail_options) + 1):
                        for combo in itertools.combinations(avail_options, k):
                            all_subsets.append(tuple(sorted(combo)))

                    untried = [s for s in all_subsets if s not in self.level_tried_subsets]
                    if not untried:
                        self.level_tried_subsets.clear()
                        untried = all_subsets
                    target_subset = untried[0]
                    self.level_tried_subsets.add(target_subset)

                    first_act = target_subset[0]
                    for act in target_subset[1:]:
                        self.macro_queue.add_step(
                            action=act,
                            action_data={},
                            description=f"Option selector action {act}",
                        )

                    self.macro_queue.add_step(
                        action=6,
                        action_data={"x": int(active_submit[0]), "y": int(active_submit[1])},
                        description="Click submit button after option selection",
                    )
                    return first_act, {}
                elif self.current_level == 0 and selector_cursor is not None and () not in self.level_tried_subsets and (active_submit not in self.dead_clicks):
                    self.level_tried_subsets.add(())
                    return 6, {"x": int(active_submit[0]), "y": int(active_submit[1])}

            # 0.7 Universal Graph Independent Set & Vertex Cover Solver
            if active_submit is not None and self.learned_game_modality not in ("OPTION", "SAT"):
                graph_data = UniversalIndependentSetSolver.extract_graph_from_grid(
                    grid, bg_color, set(sep.frame_colors), submit_btn=active_submit
                )
                if graph_data is not None:
                    self.learned_game_modality = "GRAPH"
                    vertices, edges = graph_data
                    nodes = list(range(len(vertices)))
                    first_rule = self.learned_graph_rule or "IND_SET"
                    second_rule = "VERT_COVER" if first_rule == "IND_SET" else "IND_SET"

                    target_rule = None
                    if first_rule not in self.level_tried_subsets:
                        target_rule = first_rule
                    elif second_rule not in self.level_tried_subsets:
                        target_rule = second_rule

                    if target_rule is not None:
                        self.level_tried_subsets.add(target_rule)
                        self.last_tried_graph_rule = target_rule
                        if target_rule == "IND_SET":
                            sol_nodes = UniversalIndependentSetSolver.solve(nodes, edges)
                            desc = "Graph Independent Set vertex click"
                        else:
                            sol_nodes = UniversalIndependentSetSolver.solve_vertex_cover(nodes, edges)
                            desc = "Graph Vertex Cover vertex click"

                        clicks = [vertices[i] for i in sol_nodes]
                        if clicks:
                            for clk in clicks[1:]:
                                self.macro_queue.add_step(
                                    action=6,
                                    action_data={"x": int(clk[0]), "y": int(clk[1])},
                                    description=desc,
                                )
                            self.macro_queue.add_step(
                                action=6,
                                action_data={"x": int(active_submit[0]), "y": int(active_submit[1])},
                                description=f"Click submit button after {target_rule}",
                            )
                            return 6, {"x": int(clicks[0][0]), "y": int(clicks[0][1])}

            # 1. Dedicated Modality Fast-Path (Once learned, strictly execute dedicated solver)
            if self.learned_game_modality == "MEMORY":
                if self.memory_matcher.lattice is None and sep.lattice is not None:
                    self.memory_matcher.lattice = sep.lattice
                    self.memory_matcher.hidden_color = sep.tile_back_color or int(grid[sep.lattice.centers[0][1], sep.lattice.centers[0][0]])
                    if (sep.lattice.rows * sep.lattice.cols) % 2 != 0 and (sep.lattice.rows * sep.lattice.cols) % 3 == 0:
                        self.memory_matcher.match_size = 3
                if self.prev_grid is not None and self.last_action == 6:
                    last_click = (
                        int(self.last_action_data.get("x", 0)),
                        int(self.last_action_data.get("y", 0)),
                    ) if self.last_action_data else None
                    self.memory_matcher.on_observation(self.prev_grid, grid, bg_color, last_click_pos=last_click)
                act_id, act_dict = self.memory_matcher.plan_action(grid, bg_color)
                return act_id, act_dict

            if self.learned_game_modality == "TOGGLE":
                if self.learned_lo_off_color is None:
                    self.learned_lo_off_color = bg_color
                if self.learned_lo_mode is None and self.prev_grid is not None and self.last_action == 6:
                    last_click = (
                        int(self.last_action_data.get("x", 0)),
                        int(self.last_action_data.get("y", 0)),
                    ) if self.last_action_data else None
                    if last_click:
                        induced = induce_toggle_mode(self.prev_grid, grid, last_click, N=self.lo_lvl_N)
                        if induced is not None:
                            self.learned_lo_mode = induced

                if self.learned_lo_mode is not None:
                    lo_clicks = solve_lights_out_universal(
                        grid, self.learned_lo_mode, N=self.lo_lvl_N, off_color=self.learned_lo_off_color
                    )
                    if lo_clicks:
                        for c in lo_clicks[1:]:
                            self.macro_queue.add_step(
                                action=6,
                                action_data={"x": int(c[0]), "y": int(c[1])},
                                description="GF(2) Toggle Click",
                            )
                        return 6, {"x": int(lo_clicks[0][0]), "y": int(lo_clicks[0][1])}
                    else:
                        pad_lo = (64 - self.lo_lvl_N * (64 // self.lo_lvl_N)) // 2
                        scale_lo = 64 // self.lo_lvl_N
                        return 6, {"x": int(pad_lo + scale_lo // 2), "y": int(pad_lo + scale_lo // 2)}
                else:
                    N_lo, scale_lo, pad_lo, _, walls_lo = parse_lo_board(grid, N=self.lo_lvl_N)
                    probe_cell = (
                        (1, 0)
                        if (1, 0) not in walls_lo
                        else next(
                            ((gx, gy) for gy in range(N_lo) for gx in range(N_lo) if (gx, gy) not in walls_lo),
                            (0, 0),
                        )
                    )
                    return 6, {
                        "x": int(pad_lo + probe_cell[0] * scale_lo + scale_lo // 2),
                        "y": int(pad_lo + probe_cell[1] * scale_lo + scale_lo // 2),
                    }

            # 2. Universal Lattice & K-Tuple Memory Card Matcher Induction
            if (
                sep.lattice is not None
                and sep.tile_back_color is not None
                and (5 not in legal_actions)
                and self.learned_game_modality is None
            ):
                self.learned_game_modality = "MEMORY"
                self.memory_matcher.lattice = sep.lattice
                self.memory_matcher.hidden_color = sep.tile_back_color
                if (sep.lattice.rows * sep.lattice.cols) % 2 != 0 and (sep.lattice.rows * sep.lattice.cols) % 3 == 0:
                    self.memory_matcher.match_size = 3
                if self.prev_grid is not None and self.last_action == 6:
                    last_click = (
                        int(self.last_action_data.get("x", 0)),
                        int(self.last_action_data.get("y", 0)),
                    ) if self.last_action_data else None
                    self.memory_matcher.on_observation(self.prev_grid, grid, bg_color, last_click_pos=last_click)
                act_id, act_dict = self.memory_matcher.plan_action(grid, bg_color)
                return act_id, act_dict

            # 2.5 Circuit, Directed Wire & Junction Puzzles (UniversalCircuitEngine)
            if self.learned_game_modality in (None, "CIRCUIT"):
                circuit_clicks = UniversalCircuitEngineFacade.generate_click_actions(grid)
                if circuit_clicks:
                    self.learned_game_modality = "CIRCUIT"
                    for clk in circuit_clicks[1:]:
                        self.macro_queue.add_step(
                            action=6,
                            action_data={"x": int(clk[0]), "y": int(clk[1])},
                            description="Circuit wire rotation click",
                        )
                    if 5 in legal_actions:
                        self.macro_queue.add_step(
                            action=5,
                            action_data={},
                            description="Circuit power closing switch",
                        )
                    return 6, {"x": int(circuit_clicks[0][0]), "y": int(circuit_clicks[0][1])}
                elif self.learned_game_modality == "CIRCUIT" and 5 in legal_actions:
                    if self.last_action == 5:
                        self.learned_game_modality = None
                    else:
                        return 5, {}

            # 2.6 Polyomino Exact Cover Tiling (Dominoes k=2, Trominoes k=3)
            fg_colors_dlx = [c for c in np.unique(interior) if c != bg_color and c not in sep.frame_colors]
            for c_target in fg_colors_dlx:
                pts_dlx = np.argwhere(interior == c_target)
                if len(pts_dlx) >= 16:
                    rmin, rmax = int(pts_dlx[:, 0].min()), int(pts_dlx[:, 0].max())
                    cmin, cmax = int(pts_dlx[:, 1].min()), int(pts_dlx[:, 1].max())
                    h_span = rmax - rmin + 1
                    w_span = cmax - cmin + 1
                    for s in [8, 6, 12, 10, 4]:
                        if h_span % s == 0 and w_span % s == 0:
                            rows = h_span // s
                            cols = w_span // s
                            active_cells = []
                            cell_centers = {}
                            for r in range(rows):
                                for c in range(cols):
                                    patch = interior[rmin + r * s : rmin + (r + 1) * s, cmin + c * s : cmin + (c + 1) * s]
                                    if np.mean(patch == c_target) > 0.5:
                                        active_cells.append((r, c))
                                        cell_centers[(r, c)] = (sep.x_min + cmin + c * s + s // 2, sep.y_min + rmin + r * s + s // 2)
                            N_cells = len(active_cells)
                            if N_cells >= 2:
                                cell_to_id = {pos: i for i, pos in enumerate(active_cells)}
                                candidate_ks = [2, 3] if N_cells % 2 == 0 and N_cells % 3 == 0 else ([2] if N_cells % 2 == 0 else ([3] if N_cells % 3 == 0 else []))
                                for k_tile in candidate_ks:
                                    matrix = {}
                                    row_polys = {}
                                    row_id = 0
                                    if k_tile == 2:
                                        for r, c in active_cells:
                                            for dr, dc in [(0, 1), (1, 0)]:
                                                nbr = (r + dr, c + dc)
                                                if nbr in cell_to_id:
                                                    matrix[row_id] = {cell_to_id[(r, c)], cell_to_id[nbr]}
                                                    row_polys[row_id] = [(r, c), nbr]
                                                    row_id += 1
                                    elif k_tile == 3:
                                        for r, c in active_cells:
                                            for dr1, dc1 in [(-1, 0), (1, 0)]:
                                                for dr2, dc2 in [(0, -1), (0, 1)]:
                                                    p1, p2 = (r + dr1, c + dc1), (r + dr2, c + dc2)
                                                    if p1 in cell_to_id and p2 in cell_to_id:
                                                        matrix[row_id] = {cell_to_id[(r, c)], cell_to_id[p1], cell_to_id[p2]}
                                                        row_polys[row_id] = [(r, c), p1, p2]
                                                        row_id += 1
                                            for dr, dc in [(0, 1), (1, 0)]:
                                                p1, p2 = (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)
                                                if p1 in cell_to_id and p2 in cell_to_id:
                                                    matrix[row_id] = {cell_to_id[(r, c)], cell_to_id[p1], cell_to_id[p2]}
                                                    row_polys[row_id] = [(r, c), p1, p2]
                                                    row_id += 1
                                    sol = ExactCoverDLX.solve(matrix, primary_columns=set(range(N_cells)))
                                    if sol:
                                        self.learned_game_modality = "CLICK"
                                        all_clicks = [cell_centers[cell] for r_id in sol for cell in row_polys[r_id]]
                                        for clk in all_clicks[1:]:
                                            self.macro_queue.add_step(
                                                action=6,
                                                action_data={"x": int(clk[0]), "y": int(clk[1])},
                                                description="DLX Polyomino tile click",
                                            )
                                        return 6, {"x": int(all_clicks[0][0]), "y": int(all_clicks[0][1])}

            # 2.7 Active Sonar / Beacon Pulse & Interrogate (Action 5 pulse, Action 6 click)
            if 5 in legal_actions and not any(a in legal_actions for a in [1, 2, 3, 4]):
                frame_cols = set(sep.frame_colors) | {bg_color, 0}
                int_mask = np.zeros_like(grid, dtype=bool)
                int_mask[sep.y_min:sep.y_max, sep.x_min:sep.x_max] = True
                sal_y, sal_x = np.where(int_mask & (~np.isin(grid, list(frame_cols))))
                unclicked_beacons = []
                for sy, sx in zip(sal_y, sal_x):
                    pt = (int(sx), int(sy))
                    if pt not in self.dead_clicks and pt not in self.level_successful_clicks:
                        unclicked_beacons.append(pt)
                if unclicked_beacons:
                    chosen_beacon = unclicked_beacons[len(unclicked_beacons) // 2]
                    return 6, {"x": int(chosen_beacon[0]), "y": int(chosen_beacon[1])}
                else:
                    return 5, {}

            # 3. Visual Goal & Invariant Induction (Cavity Jordan Loops / Split-Screen / HUD Sequence)
            goal_state = self.goal_inducer.induce(
                grid=grid,
                prev_grid=self.prev_grid,
                avatar_color=self.learned_avatar_color,
            )
            prioritized_clicks = goal_state.get_prioritized_click_targets()
            ctx_h = fast_grid_hash(interior)
            tabu_for_state = self.state_tabu_actions.get(ctx_h, set())

            fresh_clicks = []
            revisit_clicks = []
            for cx, cy, _ in prioritized_clicks:
                pt = (int(cx), int(cy))
                if pt not in self.dead_clicks and pt not in tabu_for_state and 0 <= cx < 64 and 0 <= cy < 64:
                    if pt not in self.level_successful_clicks:
                        fresh_clicks.append(pt)
                    else:
                        revisit_clicks.append(pt)

            candidate_clicks = fresh_clicks or revisit_clicks
            if candidate_clicks:
                chosen_pt = candidate_clicks[0]
                return 6, {"x": int(chosen_pt[0]), "y": int(chosen_pt[1])}

            # 3.1 Split-Panel D4 Symmetry Solver (Direct Exact Matching)
            split_match = self.split_d4_solver.analyze_symmetry(grid)
            if split_match and split_match.mismatch_coords:
                unclicked = [
                    (c[1], c[0]) for c in split_match.mismatch_coords
                    if (c[1], c[0]) not in self.dead_clicks
                ]
                if unclicked:
                    target_click = unclicked[0]
                    for other_click in unclicked[1:]:
                        self.macro_queue.add_step(
                            action=6,
                            action_data={"x": int(other_click[0]), "y": int(other_click[1])},
                            description="Split-Panel D4 Mismatch Click",
                        )
                    return 6, {"x": int(target_click[0]), "y": int(target_click[1])}

            # 4. GF(2) Impulse-Response Toggle Solver Induction
            lo_detected_N = (
                detect_lo_N(grid)
                if (self.learned_avatar_color is None)
                else None
            )
            is_lo_board = (
                lo_detected_N is not None
                and (5 not in legal_actions)
                and len(parse_lo_board(grid, N=lo_detected_N)[3]) > 0
                and len(parse_lo_board(grid, N=lo_detected_N)[4]) == 0
                and (goal_state.enclosure is None or len(goal_state.enclosure.cavities) == 0)
            )
            if is_lo_board:
                self.learned_game_modality = "TOGGLE"
                self.lo_lvl_N = lo_detected_N or 4
                if self.learned_lo_off_color is None:
                    self.learned_lo_off_color = bg_color
                N_lo, scale_lo, pad_lo, _, walls_lo = parse_lo_board(grid, N=self.lo_lvl_N)
                probe_cell = (
                    (1, 0)
                    if (1, 0) not in walls_lo
                    else next(
                        ((gx, gy) for gy in range(N_lo) for gx in range(N_lo) if (gx, gy) not in walls_lo),
                        (0, 0),
                    )
                )
                return 6, {
                    "x": int(pad_lo + probe_cell[0] * scale_lo + scale_lo // 2),
                    "y": int(pad_lo + probe_cell[1] * scale_lo + scale_lo // 2),
                }

            # 4. Dynamic Salience: Centroid of newly appeared foreground entities
            if self.prev_grid is not None:
                int_curr = grid[sep.y_min:sep.y_max, sep.x_min:sep.x_max]
                int_prev = self.prev_grid[sep.y_min:sep.y_max, sep.x_min:sep.x_max]
                spawn_mask = (int_curr != bg_color) & (~np.isin(int_curr, list(sep.frame_colors))) & (int_prev == bg_color)
                if np.any(spawn_mask):
                    labeled_sp, num_sp = ndimage.label(spawn_mask)
                    ctx_h_sp = fast_grid_hash(interior)
                    tabu_sp = self.state_tabu_actions.get(ctx_h_sp, set())
                    for sp_idx in range(1, num_sp + 1):
                        coords = np.argwhere(labeled_sp == sp_idx)
                        if len(coords) >= 4:
                            cr = sep.y_min + int(round(coords[:, 0].mean()))
                            cc = sep.x_min + int(round(coords[:, 1].mean()))
                            pt = (cc, cr)
                            if pt not in self.dead_clicks and pt not in tabu_sp and 4 <= cc < 60 and 4 <= cr < 60:
                                return 6, {"x": int(cc), "y": int(cr)}

            # 4.5 Universal Interactive Matrix & Combinatorial Constraint Solver (GF(2) & Boolean SAT)
            row_xor_clicks = InteractiveMatrixInducer.solve_interactive_row_xor(
                grid, self.prev_grid, bg_color, set(sep.frame_colors)
            )
            if row_xor_clicks:
                self.learned_game_modality = "CLICK"
                for c in row_xor_clicks[1:]:
                    self.macro_queue.add_step(
                        action=6,
                        action_data={"x": int(c[0]), "y": int(c[1])},
                        description="GF(2) Matrix Row XOR Click",
                    )
                return 6, {"x": int(row_xor_clicks[0][0]), "y": int(row_xor_clicks[0][1])}

            # 5. Fallback for point-and-click: salient unvisited component centroids in interior [4:60, 4:60]
            fg_mask_click = (grid != bg_color) & (~np.isin(grid, list(sep.frame_colors)))
            labeled_fg, num_fg = ndimage.label(fg_mask_click)
            ctx_h_fb = fast_grid_hash(interior)
            tabu_fb = self.state_tabu_actions.get(ctx_h_fb, set())

            # Track colors of previously successful clicks
            succ_colors: Set[int] = set()
            for sx, sy in self.level_successful_clicks:
                if 0 <= sx < 64 and 0 <= sy < 64:
                    sc = int(grid[sy, sx])
                    if sc != bg_color and sc not in sep.frame_colors:
                        succ_colors.add(sc)

            fg_components = []
            for comp_i in range(1, num_fg + 1):
                pts = np.argwhere(labeled_fg == comp_i)
                mass = len(pts)
                r_min, c_min = pts.min(axis=0)
                r_max, c_max = pts.max(axis=0)
                h_c = r_max - r_min + 1
                w_c = c_max - c_min + 1
                cr = int(round(float(pts[:, 0].mean())))
                cc = int(round(float(pts[:, 1].mean())))
                dom_c = int(np.bincount(grid[pts[:, 0], pts[:, 1]]).argmax())

                # Compactness score: blobs (e.g. vertices, buttons, tiles) vs thin 1-pixel lines
                is_compact = (h_c >= 2 and w_c >= 2 and mass >= 4)
                has_succ_color = (dom_c in succ_colors)
                fg_components.append((has_succ_color, is_compact, mass, cc, cr, dom_c, pts))

            # Rank components: successful colors first, then compact blobs, then larger mass
            fg_components.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

            # Find unvisited candidate centroids
            candidate_click = None
            for _, _, mass, cc, cr, _, pts in fg_components:
                if mass < 4:
                    continue  # Ignore 1-3 pixel HUD markers/indicators
                pt = (int(cc), int(cr))
                if not (0 <= pt[1] < 64 and 0 <= pt[0] < 64 and grid[pt[1], pt[0]] != bg_color):
                    pt = (int(pts[0, 1]), int(pts[0, 0]))
                if 0 <= pt[0] < 64 and 0 <= pt[1] < 64 and pt not in self.dead_clicks and pt not in tabu_fb:
                    if submit_btn is not None and abs(pt[0] - submit_btn[0]) <= 2 and abs(pt[1] - submit_btn[1]) <= 2:
                        continue
                    comp_pixels = {(int(p[1]), int(p[0])) for p in pts}
                    if not (comp_pixels & self.level_successful_clicks):
                        candidate_click = pt
                        break

            if candidate_click is not None:
                unvisited_fg_count = sum(
                    1 for _, _, mass, cc, cr, _, pts in fg_components
                    if mass >= 4
                    and not ({(int(p[1]), int(p[0])) for p in pts} & self.level_successful_clicks)
                    and not (submit_btn is not None and abs(cc - submit_btn[0]) <= 2 and abs(cr - submit_btn[1]) <= 2)
                )
                if submit_btn is not None and unvisited_fg_count <= 1:
                    self.macro_queue.add_step(
                        action=6,
                        action_data={"x": int(submit_btn[0]), "y": int(submit_btn[1])},
                        description="Submit button click after interactive component selection",
                    )
                return 6, {"x": int(candidate_click[0]), "y": int(candidate_click[1])}

            if 5 in legal_actions:
                return 5, {}

            # If all centroids visited, try individual component points
            for _, _, mass, _, _, _, pts in fg_components:
                if mass < 4:
                    continue
                for pr, pc in pts:
                    pt = (int(pc), int(pr))
                    if 0 <= pc < 64 and 0 <= pr < 64 and pt not in self.dead_clicks and pt not in tabu_fb:
                        if submit_btn is not None and abs(pt[0] - submit_btn[0]) <= 2 and abs(pt[1] - submit_btn[1]) <= 2:
                            continue
                        return 6, {"x": int(pt[0]), "y": int(pt[1])}

            if submit_btn is not None:
                return 6, {"x": int(submit_btn[0]), "y": int(submit_btn[1])}

            return 6, {"x": 32, "y": 32}

        # -------------------------------------------------------------
        # Branch C: Directional Discrete Board Extraction & Planning
        # -------------------------------------------------------------
        prev_step_world = self.prev_world
        if world is None:
            world = extract_discrete_board(
                grid,
                avatar_color=self.learned_avatar_color,
                prev_world=self.prev_world,
                last_action=self.last_action,
                learned_wall_colors=self.learned_wall_colors,
                learned_box_colors=self.learned_box_colors,
                learned_goal_colors=self.learned_goal_colors,
                learned_co_avatar_colors=self.learned_co_avatar_colors,
                learned_trail_colors=self.learned_trail_colors,
                conveyor_colors=set(self.conveyor_field.tile_color_velocity.keys()),
            )
            if world is not None:
                self.prev_world = world

        if world is not None:
            if self.learned_avatar_color is None:
                self.learned_avatar_color = world["avatar_color"]
            p_pos = world["player"]
            all_goals = world["all_goals"]
            walls = (world["walls"] - all_goals) | self.dead_walls
            boxes = world["boxes"]
            plates = world["plates"]
            exit_doors = world["exit_doors"]
            gh, gw = world["gh"], world["gw"]
            scale = max(1, world.get("scale", 1))

            for g in (plates | exit_doors | all_goals):
                if g in self.level_clicked_dots:
                    continue
                self.level_known_plates.add(g)
                c_val = int(world["board"][g[0], g[1]])
                if c_val not in (world["bg_color"], self.learned_avatar_color) and c_val not in self.inactive_tagged_colors:
                    self.learned_goal_colors.add(c_val)

            norm_displacement_vectors = {}
            if self.learned_displacement_vectors:
                for a, (dr, dc) in self.learned_displacement_vectors.items():
                    if scale > 1 and (abs(dr) > 1 or abs(dc) > 1):
                        norm_displacement_vectors[a] = (int(round(dr / scale)), int(round(dc / scale)))
                    else:
                        norm_displacement_vectors[a] = (dr, dc)
            else:
                norm_displacement_vectors = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

            # Check if HUD Sequence highlights a specific active color on the board
            hud_inv = self.goal_inducer.hud_detector.detect(grid)
            if hud_inv is not None and not boxes:
                active_c = hud_inv.active_token_color
                hud_cells = {
                    (r, c)
                    for r in range(gh)
                    for c in range(gw)
                    if int(world["board"][r, c]) == active_c and (r, c) != p_pos
                }
                if hud_cells:
                    plates = hud_cells
                    all_goals = hud_cells

            # -3. Hold mechanic verification if player is on goal plate
            candidate_goals = set(plates | exit_doors | all_goals | self.level_known_plates)
            if prev_step_world and p_pos in prev_step_world.get("all_goals", set()):
                candidate_goals.add(p_pos)
            if self.learned_goal_colors and prev_step_world:
                if 0 <= p_pos[0] < prev_step_world.get("gh", gh) and 0 <= p_pos[1] < prev_step_world.get("gw", gw):
                    if int(prev_step_world["board"][p_pos[0], p_pos[1]]) in self.learned_goal_colors:
                        candidate_goals.add(p_pos)
            if self.hold_inducer.active_plate == p_pos and self.hold_inducer.hold_confirmed:
                candidate_goals.add(p_pos)

            remaining_unvisited = (candidate_goals - set(self.visited_waypoints)) - {p_pos}
            if p_pos in candidate_goals and (len(remaining_unvisited) == 0 or self.hold_inducer.hold_confirmed):
                hold_status = self.hold_inducer.on_step(
                    player_pos=p_pos,
                    candidate_goals=candidate_goals,
                    prev_frame=self.prev_grid,
                    curr_frame=grid,
                    level_won=False,
                )
                if hold_status.get("action_needed") and hold_status.get("probe_hold"):
                    hold_act = synthesize_holding_action(
                        player_pos=p_pos,
                        walls=walls,
                        gh=gh,
                        gw=gw,
                        legal_actions=legal_actions,
                        deltas=norm_displacement_vectors,
                    )
                    if hold_act in legal_actions:
                        return hold_act, {}

            # -2. Beacon signal check & Manhattan triangulation / Active Radar Probing
            # -2. Active Coordinate-Click on Clickable Targets / Radar Pulse Discoveries (Action 6 Priority)
            if 6 in legal_actions:
                click_targets = []
                
                # 1. Sparse plates/waypoints (interior only)
                if plates and len(plates) <= 8:
                    for pt in plates:
                        if 4 <= pt[0] < gh - 4 and 4 <= pt[1] < gw - 4:
                            if pt not in self.level_clicked_dots and pt not in self.flagged_coords and pt not in self.dead_clicks and pt != p_pos:
                                click_targets.append(pt)
                
                # 2. Known goal colors (persisted across levels, interior only)
                if self.learned_goal_colors:
                    for r in range(4, gh - 4):
                        for c in range(4, gw - 4):
                            if (r, c) == p_pos or (r, c) in click_targets or (r, c) in self.level_clicked_dots:
                                continue
                            cv = int(world["board"][r, c])
                            if cv in self.learned_goal_colors and cv not in self.inactive_tagged_colors and (r, c) not in self.flagged_coords and (r, c) not in self.dead_clicks:
                                if (r, c) not in walls and (r, c) not in self.dead_walls:
                                    click_targets.append((r, c))
                
                # 3. Direct Diff & Sparse components of unclassified foreground (revealed target dots, interior only)
                if not click_targets and self.prev_grid is not None:
                    diff_mask = (grid != self.prev_grid) & (grid != world["bg_color"]) & (grid != self.learned_avatar_color)
                    for r, c in np.argwhere(diff_mask):
                        if 4 <= r < gh - 4 and 4 <= c < gw - 4:
                            if (r, c) not in self.level_clicked_dots and (r, c) not in self.flagged_coords and (r, c) not in self.dead_clicks:
                                cv = int(world["board"][r, c])
                                if cv not in world.get("frame_colors", set()) and cv not in self.learned_wall_colors and cv not in self.inactive_tagged_colors:
                                    click_targets.append((int(r), int(c)))

                if not click_targets and self.prev_grid is not None:
                    for r in range(4, gh - 4):
                        for c in range(4, gw - 4):
                            if (r, c) == p_pos or (r, c) in click_targets or (r, c) in walls or (r, c) in self.dead_walls or (r, c) in self.level_clicked_dots:
                                continue
                            cv = int(world["board"][r, c])
                            if cv not in (world["bg_color"], self.learned_avatar_color) and cv not in world.get("frame_colors", set()) and cv not in self.learned_wall_colors and cv not in self.inactive_tagged_colors:
                                if (r, c) not in self.flagged_coords and (r, c) not in self.dead_clicks:
                                    comp_count = int(np.sum(world["board"] == cv))
                                    if comp_count <= 8:
                                        click_targets.append((r, c))

                if click_targets:
                    # Sort by distance and enqueue subsequent dots into macro_queue
                    click_targets.sort(key=lambda pt: abs(pt[0] - p_pos[0]) + abs(pt[1] - p_pos[1]))
                    target_cell = click_targets[0]
                    self.level_clicked_dots.add(target_cell)
                    self.flagged_coords.add(target_cell)
                    c_clicked = int(world["board"][target_cell[0], target_cell[1]])
                    if c_clicked not in (world["bg_color"], self.learned_avatar_color) and c_clicked not in self.inactive_tagged_colors:
                        self.learned_goal_colors.add(c_clicked)
                    px = int(world["x0"] + target_cell[1] * scale + scale // 2) if "x0" in world else int(target_cell[1])
                    py = int(world["y0"] + target_cell[0] * scale + scale // 2) if "y0" in world else int(target_cell[0])
                    for dot in click_targets[1:]:
                        self.level_clicked_dots.add(dot)
                        self.flagged_coords.add(dot)
                        dpx = int(world["x0"] + dot[1] * scale + scale // 2) if "x0" in world else int(dot[1])
                        dpy = int(world["y0"] + dot[0] * scale + scale // 2) if "y0" in world else int(dot[0])
                        self.macro_queue.add_step(
                            action=6,
                            action_data={"x": dpx, "y": dpy},
                            description="Click revealed dot",
                        )
                    return 6, {"x": px, "y": py}
                elif 5 in legal_actions:
                    # Active Frontier Epistemic Exploration
                    if not hasattr(self, "visited_pulse_waypoints"):
                        self.visited_pulse_waypoints = set()
                    
                    if not getattr(self, "level_pulsed", False):
                        self.level_pulsed = True
                        self.visited_pulse_waypoints.add(p_pos)
                        self.flagged_coords.clear()
                        return 5, {}
                    
                    r_q1 = max(1, int(gh * 0.25))
                    r_q2 = min(gh - 2, int(gh * 0.75))
                    c_q1 = max(1, int(gw * 0.25))
                    c_q2 = min(gw - 2, int(gw * 0.75))
                    quadrants = [
                        (r_q1, c_q1),
                        (r_q1, c_q2),
                        (r_q2, c_q2),
                        (r_q2, c_q1),
                    ]
                    unvisited_quads = [q for q in quadrants if q not in self.visited_pulse_waypoints]
                    if not unvisited_quads:
                        self.visited_pulse_waypoints.clear()
                        unvisited_quads = list(quadrants)
                    
                    target_q = min(unvisited_quads, key=lambda q: abs(q[0] - p_pos[0]) + abs(q[1] - p_pos[1]))
                    dist_to_q = abs(target_q[0] - p_pos[0]) + abs(target_q[1] - p_pos[1])
                    if dist_to_q <= 1:
                        self.visited_pulse_waypoints.add(target_q)
                        self.flagged_coords.clear()
                        return 5, {}
                    else:
                        cand_target = target_q
                        if cand_target in (walls | self.dead_walls):
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                                cand = (cand_target[0] + dr, cand_target[1] + dc)
                                if 0 <= cand[0] < gh and 0 <= cand[1] < gw and cand not in (walls | self.dead_walls):
                                    cand_target = cand
                                    break
                        nav_path = solve_cardinal_astar(
                            p_pos, cand_target, walls=walls | self.dead_walls, gh=gh, gw=gw, displacement_vectors=norm_displacement_vectors
                        )
                        if nav_path and nav_path[0] in legal_actions:
                            for a in nav_path[1:]:
                                self.macro_queue.add_step(action=a, description="Quadrant sweep")
                            return nav_path[0], {}
                        else:
                            self.visited_pulse_waypoints.add(target_q)
                            self.flagged_coords.clear()
                            return 5, {}

            # STRICT GUARD: Only activate beacon triangulation when there are NO visible targets (hidden goal modality)
            has_visible_targets = bool(plates or exit_doors or all_goals or boxes)
            if not has_visible_targets:

                # C. Perimeter HUD / Ring Distance Triangulation Planner Fallback
                dist_signal = PerceptualBeaconSignalExtractor.detect_perimeter_hud_metric(
                    prev_frame=self.prev_grid if self.prev_grid is not None else grid,
                    curr_frame=grid,
                    avatar_delta=dp if 'dp' in locals() else (0, 0),
                )
                if dist_signal is None and self.prev_grid is not None:
                    dist_signal = PerceptualBeaconSignalExtractor.detect_transient_ring_radius(
                        prev_frame=self.prev_grid,
                        curr_frame=grid,
                        avatar_pos=p_pos,
                        background_color=bg_color,
                    )
                if dist_signal is not None or self.beacon_planner.observations:
                    b_act, b_data = self.beacon_planner.plan_next_action(
                        avatar_pos=p_pos,
                        walls=walls,
                        current_distance_reading=dist_signal,
                        legal_actions=legal_actions,
                        has_beacon_action=(5 in legal_actions),
                    )
                    if b_act in legal_actions:
                        return b_act, b_data

            # -1. DELAY PIPELINE MODALITY (K-Step Action Pipeline, dl01)
            if self.learned_game_modality == "DELAY" or (self.learned_command_latency >= 1 and self.learned_game_modality != "ECHO"):
                self.learned_game_modality = "DELAY"
                targets = plates or exit_doors or all_goals
                if targets:
                    pipe = self.pipeline_tracker.get_current_pipeline()
                    delay_plan = solve_delayed_pipeline_astar(
                        p_pos, pipe, set(targets), walls, gh, gw, deltas=norm_displacement_vectors, legal_actions=legal_actions
                    )
                    if delay_plan:
                        for a in delay_plan[1:]:
                            self.macro_queue.add_step(
                                action=a,
                                description="Delayed pipeline step",
                            )
                        return delay_plan[0], {}

            # 0. ECHO MOVE MODALITY (2-Step Echo Kinematics)
            if self.learned_game_modality == "ECHO":
                targets = plates or exit_doors or all_goals
                if targets:
                    echo_plan = solve_echo_kinematics_bfs(
                        p_pos, targets, walls, gh, gw, deltas=norm_displacement_vectors
                    )
                    if echo_plan:
                        for a in echo_plan[1:]:
                            self.macro_queue.add_step(
                                action=a,
                                description="Echo kinematics step",
                            )
                        return echo_plan[0], {}

            # 1. KNIGHT / JUMP MODALITY
            if self.learned_game_modality == "KNIGHT" or getattr(self.prober.profile, "modality", "") == "KNIGHT":
                self.learned_game_modality = "KNIGHT"
                targets = plates or exit_doors or all_goals
                if targets:
                    knight_plan = solve_knight_universal(
                        p_pos, targets, walls, gh, gw, has_action_5=(5 in legal_actions)
                    )
                    if knight_plan:
                        for a in knight_plan[1:]:
                            self.macro_queue.add_step(action=a, description="Jump step")
                        return knight_plan[0], {}

            # 2. SLIDE MODALITY (Inertial Slide Raycast BFS)
            if self.learned_game_modality == "SLIDE" or getattr(self.prober.profile, "is_sliding", False):
                self.learned_game_modality = "SLIDE"
                targets = plates or exit_doors or all_goals
                if targets:
                    deltas = norm_displacement_vectors
                    slide_plan = solve_inertial_slide_universal(
                        p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=False, stop_on_goal=True
                    )
                    if not slide_plan:
                        slide_plan = solve_inertial_slide_universal(
                            p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=False, stop_on_goal=False
                        )
                    if not slide_plan:
                        slide_plan = solve_inertial_slide_universal(
                            p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=True, stop_on_goal=True
                        )
                    if not slide_plan:
                        slide_plan = solve_inertial_slide_universal(
                            p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=True, stop_on_goal=False
                        )
                    if slide_plan:
                        for a in slide_plan[1:]:
                            self.macro_queue.add_step(action=a, description="Inertial slide")
                        return slide_plan[0], {}

            # 3. SOKOBAN / CRATE MODALITY (1-Step & Sliding Crate)
            can_be_sokoban = (
                (self.learned_game_modality is None or self.learned_game_modality == "SOKOBAN")
                and ((self.current_level == 0 and len(boxes) >= 2) or bool(self.confirmed_box_colors))
            )
            boxes_not_on_plates = boxes - plates
            sokoban_plan = None
            if can_be_sokoban and len(boxes) >= 1 and len(plates) >= len(boxes) and boxes_not_on_plates and gh <= 32 and gw <= 32:
                sokoban_plan = solve_sokoban_universal(
                    p_pos, boxes, plates, walls, gh, gw, slide_push=self.learned_slide_push
                )
                if not sokoban_plan and not self.learned_slide_push:
                    sokoban_plan = solve_sokoban_universal(
                        p_pos, boxes, plates, walls, gh, gw, slide_push=True
                    )
                    if sokoban_plan:
                        self.learned_slide_push = True

                # If equal cardinality, test opposite box/plate orientation only if box colors unconfirmed
                if len(boxes) == len(plates) and not self.confirmed_box_colors:
                    alt_plan = solve_sokoban_universal(
                        p_pos, plates, boxes, walls, gh, gw, slide_push=self.learned_slide_push
                    )
                    if not alt_plan and not self.learned_slide_push:
                        alt_plan = solve_sokoban_universal(
                            p_pos, plates, boxes, walls, gh, gw, slide_push=True
                        )
                    if alt_plan and (not sokoban_plan or len(alt_plan) < len(sokoban_plan)):
                        sokoban_plan = alt_plan
                        boxes, plates = plates, boxes

                if sokoban_plan:
                    cand_sig = pack_signature_64(sokoban_plan[0], 0, 0, fast_grid_hash(interior))
                    if cand_sig not in self.dead_signatures:
                        if self.learned_box_colors:
                            self.learned_game_modality = "SOKOBAN"
                        for a in sokoban_plan[1:]:
                            self.macro_queue.add_step(action=a, description="Sokoban box push")
                        return sokoban_plan[0], {}
                    else:
                        sokoban_plan = None

            # Dynamic hazard map for time-expanded collision avoidance
            hazard_map = None
            if self.dynamic_entities:
                hazard_map = collections.defaultdict(set)
                for ent in self.dynamic_entities.values():
                    traj = predict_discrete_hazard_trajectory(
                        start_pos=ent["pos"],
                        velocity=ent["vel"],
                        walls=walls,
                        gh=gh,
                        gw=gw,
                        steps=40,
                    )
                    for t_step, (hr, hc) in enumerate(traj):
                        hazard_map[t_step].add((hr, hc))

            if self.learned_game_modality == "SOKOBAN" and exit_doors:
                impassable = walls | plates
                door = min(exit_doors, key=lambda d: abs(d[0] - p_pos[0]) + abs(d[1] - p_pos[1]))
                exit_plan = solve_cardinal_astar(
                    p_pos, door, impassable, gh, gw,
                    dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                    displacement_vectors=norm_displacement_vectors,
                )
                if exit_plan:
                    deltas = norm_displacement_vectors
                    curr_p = p_pos
                    dr, dc = deltas.get(exit_plan[0], (0, 0))
                    curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                    for a in exit_plan[1:]:
                        dr, dc = deltas.get(a, (0, 0))
                        next_p = (curr_p[0] + dr, curr_p[1] + dc)
                        self.macro_queue.add_step(
                            action=a,
                            expected_player_pos=curr_p,
                            description="A* exit door",
                        )
                        curr_p = next_p
                    return exit_plan[0], {}

            # Escort / Companion Multi-Agent Coupled Navigation
            escort_goals = candidate_goals | all_goals | plates
            if world.get("co_avatars") and not self.confirmed_box_colors:
                escort_goals |= boxes

            if world.get("co_avatars") and len(escort_goals) >= 2:
                e_pos = list(world["co_avatars"])[0]
                escort_plan = solve_escort_follower_astar(
                    p_start=p_pos,
                    e_start=e_pos,
                    goals=escort_goals,
                    walls=walls - escort_goals,
                    gh=gh,
                    gw=gw,
                    vectors=norm_displacement_vectors,
                    max_nodes=40000,
                )
                if escort_plan:
                    deltas = norm_displacement_vectors
                    curr_p = p_pos
                    dr, dc = deltas.get(escort_plan[0], (0, 0))
                    curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                    for a in escort_plan[1:]:
                        dr, dc = deltas.get(a, (0, 0))
                        next_p = (curr_p[0] + dr, curr_p[1] + dc)
                        self.macro_queue.add_step(
                            action=a,
                            expected_player_pos=curr_p,
                            description="Escort Coupled Navigation",
                        )
                        curr_p = next_p
                    return escort_plan[0], {}

            # 3.4 PASSIVE VECTOR FIELD & CONVEYOR BELT NAVIGATION
            if self.conveyor_field.has_conveyors_active() or self.learned_game_modality == "CONVEYOR":
                conveyor_targets = set(plates | exit_doors | all_goals) - self.visited_waypoints
                if not conveyor_targets:
                    conveyor_targets = set(plates | exit_doors | all_goals)
                if conveyor_targets:
                    conveyor_plan = solve_conveyor_astar(
                        p_start=p_pos,
                        goals=conveyor_targets,
                        walls=walls,
                        gh=gh,
                        gw=gw,
                        field=self.conveyor_field,
                        displacement_vectors=norm_displacement_vectors,
                        board=world.get("board") if world else None,
                    )
                    if conveyor_plan and conveyor_plan[0] in legal_actions:
                        for a in conveyor_plan[1:]:
                            self.macro_queue.add_step(action=a, description="Conveyor navigation step")
                        return conveyor_plan[0], {}

            # 3.5 UNIFIED SIMULATED MODEL PLANNER (Multi-Physics Offline Forward Rollout)
            if world is not None and (plates or exit_doors or all_goals or boxes):
                try:
                    sim_model = KinematicTransitionModel.from_world_dict(world)
                    sim_model.is_ice_sliding = bool(self.learned_game_modality == "SLIDE" or getattr(self.prober.profile, "is_sliding", False))
                    sim_model.displacement_vectors = norm_displacement_vectors

                    sim_init_st = SimulatedState(
                        player_pos=p_pos,
                        boxes=frozenset(boxes),
                        active_switches=frozenset(),
                        open_doors=frozenset(),
                        collected_items=frozenset(self.visited_waypoints),
                    )

                    sim_plan = self.hierarchical_planner.plan_hierarchical(
                        init_state=sim_init_st,
                        model=sim_model,
                        max_expansions_per_milestone=3000,
                        time_limit_per_milestone=0.8,
                    )
                    if sim_plan and len(sim_plan) > 0:
                        first_act = sim_plan[0]
                        if first_act in legal_actions:
                            self.simulated_planner.populate_macro_queue(
                                actions=sim_plan[1:],
                                init_state=sim_init_st,
                                model=sim_model,
                                macro_queue=self.macro_queue,
                            )
                            return first_act, {}
                except Exception:
                    pass

            # 4. STANDARD CARDINAL NAVIGATION / A* (Waypoints & Exit Doors)
            if not sokoban_plan:
                if p_pos in plates:
                    self.visited_waypoints.add(p_pos)
                unvisited_plates = {p for p in plates if p not in self.visited_waypoints and p != p_pos}
                if unvisited_plates:
                    target = min(unvisited_plates, key=lambda g: abs(g[0] - p_pos[0]) + abs(g[1] - p_pos[1]))
                    astar_plan = solve_cardinal_astar(
                        p_pos, target, walls, gh, gw,
                        dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                        displacement_vectors=norm_displacement_vectors,
                    )
                    if astar_plan:
                        deltas = norm_displacement_vectors
                        curr_p = p_pos
                        dr, dc = deltas.get(astar_plan[0], (0, 0))
                        curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                        for a in astar_plan[1:]:
                            dr, dc = deltas.get(a, (0, 0))
                            next_p = (curr_p[0] + dr, curr_p[1] + dc)
                            self.macro_queue.add_step(
                                action=a,
                                expected_player_pos=curr_p,
                                description="A* waypoint",
                            )
                            curr_p = next_p
                        return astar_plan[0], {}

            if exit_doors:
                door = min(exit_doors, key=lambda d: abs(d[0] - p_pos[0]) + abs(d[1] - p_pos[1]))
                exit_plan = solve_cardinal_astar(
                    p_pos, door, walls, gh, gw,
                    dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                    displacement_vectors=norm_displacement_vectors,
                )
                if exit_plan:
                    deltas = norm_displacement_vectors
                    curr_p = p_pos
                    dr, dc = deltas.get(exit_plan[0], (0, 0))
                    curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                    for a in exit_plan[1:]:
                        dr, dc = deltas.get(a, (0, 0))
                        next_p = (curr_p[0] + dr, curr_p[1] + dc)
                        self.macro_queue.add_step(
                            action=a,
                            expected_player_pos=curr_p,
                            description="A* exit door",
                        )
                        curr_p = next_p
                    return exit_plan[0], {}

            # 5. Epistemic Latent Targets (Revealed by Sub-Frame Flashes)
            if self.subframe_decomposer.epistemic_grid.has_active_targets:
                unvisited_epistemic = self.subframe_decomposer.epistemic_grid.get_unvisited_targets()
                ep_target = min(unvisited_epistemic, key=lambda g: abs(g[0] - p_pos[0]) + abs(g[1] - p_pos[1]))
                ep_plan = solve_cardinal_astar(
                    p_pos, ep_target, walls, gh, gw,
                    dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                    displacement_vectors=norm_displacement_vectors,
                )
                if ep_plan:
                    self.subframe_decomposer.epistemic_grid.mark_visited(ep_target[0], ep_target[1])
                    deltas = norm_displacement_vectors
                    curr_p = p_pos
                    dr, dc = deltas.get(ep_plan[0], (0, 0))
                    curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                    for a in ep_plan[1:]:
                        dr, dc = deltas.get(a, (0, 0))
                        next_p = (curr_p[0] + dr, curr_p[1] + dc)
                        self.macro_queue.add_step(
                            action=a,
                            expected_player_pos=curr_p,
                            description="A* transient flash target",
                        )
                        curr_p = next_p
                    return ep_plan[0], {}
            # 4.5 DELTA-DRIVEN INVARIANT GOAL GUIDANCE
            if self.macro_queue.is_empty:
                try:
                    goal_guidance = self.delta_goal_engine.plan_next_subgoal(
                        current_grid=current_grid,
                        mover_pos=p_pos,
                        current_level=self.current_level,
                    )
                    if goal_guidance and goal_guidance.target_coordinates:
                        guidance_targets = [
                            (int(r), int(c)) for r, c in goal_guidance.target_coordinates if (r, c) not in walls
                        ]
                        if guidance_targets:
                            sorted_gt = sorted(
                                guidance_targets,
                                key=lambda g: abs(g[0] - p_pos[0]) + abs(g[1] - p_pos[1])
                            )
                            for gt in sorted_gt:
                                g_plan = solve_cardinal_astar(
                                    p_pos, gt, walls, gh, gw,
                                    dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                                    displacement_vectors=norm_displacement_vectors,
                                )
                                if g_plan:
                                    deltas = norm_displacement_vectors
                                    curr_p = p_pos
                                    dr, dc = deltas.get(g_plan[0], (0, 0))
                                    curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                                    for a in g_plan[1:]:
                                        dr, dc = deltas.get(a, (0, 0))
                                        next_p = (curr_p[0] + dr, curr_p[1] + dc)
                                        self.macro_queue.add_step(
                                            action=a,
                                            expected_player_pos=curr_p,
                                            description="Goal Invariant Guidance",
                                        )
                                        curr_p = next_p
                                    return g_plan[0], {}
                except Exception:
                    pass

            targets = plates or all_goals
            if targets:
                sorted_targets = sorted(
                    [g for g in targets if g not in walls],
                    key=lambda g: abs(g[0] - p_pos[0]) + abs(g[1] - p_pos[1])
                )
                astar_plan = None
                for target in sorted_targets:
                    plan = solve_cardinal_astar(
                        p_pos, target, walls, gh, gw,
                        dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                        displacement_vectors=norm_displacement_vectors,
                    )
                    if plan:
                        astar_plan = plan
                        break

                if astar_plan:
                    deltas = norm_displacement_vectors
                    curr_p = p_pos
                    dr, dc = deltas.get(astar_plan[0], (0, 0))
                    curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                    for a in astar_plan[1:]:
                        dr, dc = deltas.get(a, (0, 0))
                        next_p = (curr_p[0] + dr, curr_p[1] + dc)
                        self.macro_queue.add_step(
                            action=a,
                            expected_player_pos=curr_p,
                            description="A* navigation",
                        )
                        curr_p = next_p
                    return astar_plan[0], {}
                elif 5 in legal_actions and self.dead_walls:
                    self.dead_walls.clear()
                    return 5, {}

            # 6. EMPA Object-Pair Epistemic Curiosity: Route to untested candidate object classes
            empa_targets = self.empa_planner.extract_candidate_targets(
                grid=grid,
                avatar_pos=p_pos,
                avatar_color=self.learned_avatar_color,
                bg_color=bg_color,
                wall_colors=self.learned_wall_colors,
            )
            for cand in empa_targets:
                if not cand.contacted:
                    cand_pts = sorted(cand.coords, key=lambda g: abs(g[0] - p_pos[0]) + abs(g[1] - p_pos[1]))
                    for cand_pt in cand_pts[:3]:
                        cand_plan = solve_cardinal_astar(
                            p_pos, cand_pt, walls - cand.coords, gh, gw,
                            dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                            displacement_vectors=norm_displacement_vectors,
                        )
                        if cand_plan:
                            self.empa_planner.record_contact(color=cand.color, coord=cand_pt, delta_observed=1)
                            deltas = norm_displacement_vectors
                            curr_p = p_pos
                            dr, dc = deltas.get(cand_plan[0], (0, 0))
                            curr_p = (curr_p[0] + dr, curr_p[1] + dc)
                            for a in cand_plan[1:]:
                                dr, dc = deltas.get(a, (0, 0))
                                next_p = (curr_p[0] + dr, curr_p[1] + dc)
                                self.macro_queue.add_step(
                                    action=a,
                                    expected_player_pos=curr_p,
                                    description="EMPA Curiosity Contact",
                                )
                                curr_p = next_p
                            return cand_plan[0], {}

        # -------------------------------------------------------------
        # Branch D: Tier-2 Retrodiction-Gated World Model Planning & Dead-Signature Fallback
        # -------------------------------------------------------------
        # 1. Online world-model synthesis if not yet verified and transitions observed
        if (
            self.verified_world_model is None
            and len(self.transition_history) >= 2
            and not self.pacer.is_level_timed_out
        ):
            try:
                candidate = self.retrodict_synthesizer.synthesize_and_verify(
                    history=self.transition_history, max_cegis_iters=3, max_total_seconds=2.0
                )
                if candidate is not None:
                    self.verified_world_model = candidate
            except Exception:
                pass

        # 2. Plan forward through verified world model
        if self.verified_world_model is not None:
            wm_plan = self._plan_with_verified_world_model(grid, legal_actions)
            if wm_plan and len(wm_plan) > 0:
                first_act = wm_plan[0]
                for a in wm_plan[1:]:
                    self.macro_queue.add_step(
                        action=a,
                        expected_player_pos=None,
                        description="Verified World Model Plan",
                    )
                return first_act, {}

        # 3. Dead-Signature Culling & Tabu Cycle Avoidance
        ctx_h = fast_grid_hash(interior)
        for cand_act in legal_actions:
            if cand_act in [0, 7]:
                continue
            sig = pack_signature_64(cand_act, 0, 0, ctx_h)
            if sig not in self.dead_signatures and cand_act not in self.state_tabu_actions.get(ctx_h, set()):
                return cand_act, {}

        for cand_act in legal_actions:
            if cand_act in [0, 7]:
                continue
            sig = pack_signature_64(cand_act, 0, 0, ctx_h)
            if sig not in self.dead_signatures:
                return cand_act, {}

        fallback = 1 if 1 in legal_actions else legal_actions[0]
        return fallback, {}

    def _plan_with_verified_world_model(
        self,
        current_grid: np.ndarray,
        legal_actions: List[int],
        max_depth: int = 25,
        max_expansions: int = 400,
    ) -> Optional[List[int]]:
        model = self.verified_world_model
        if model is None:
            return None
        try:
            init_st = model.init_state(
                current_grid,
                {},
                self.subframe_decomposer.epistemic_grid.grid,
            )
            if model.outcome(init_st) == "LEVEL_CLEAR":
                return []

            queue: collections.deque = collections.deque([(init_st, [])])
            visited: Set[int] = set()
            rendered_init = model.render(init_st)
            visited.add(fast_grid_hash(rendered_init[4:60, 4:60]))

            expansions = 0
            while queue and expansions < max_expansions:
                curr_st, path = queue.popleft()
                expansions += 1

                if len(path) >= max_depth:
                    continue

                for a in legal_actions:
                    if a in (0, 7):
                        continue
                    nxt_st = model.transition(curr_st, a)
                    out = model.outcome(nxt_st)
                    if out == "LEVEL_CLEAR":
                        return path + [a]
                    if out == "GAME_OVER":
                        continue

                    rendered = model.render(nxt_st)
                    h = fast_grid_hash(rendered[4:60, 4:60])
                    if h not in visited:
                        visited.add(h)
                        queue.append((nxt_st, path + [a]))
        except Exception:
            return None
        return None
