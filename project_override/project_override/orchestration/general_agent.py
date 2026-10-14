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
    predict_discrete_hazard_trajectory,
    pack_signature_64,
    fast_grid_hash,
)
from project_override.core.visual_goal_induction import VisualGoalInductionEngine
from project_override.orchestration.macro_queue import (
    MacroActionQueue,
)
from project_override.orchestration.system_pacer import SystemPacer


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
        self.winning_action_tails: Dict[int, List[int]] = {}
        self.prev_world: Optional[Dict[str, Any]] = None
        self.visited_waypoints: Set[Tuple[int, int]] = set()

        # Tabu state & periodic orbit cycle avoidance
        self.recent_state_hashes: List[int] = []
        self.recent_actions: List[Tuple[int, Dict[str, Any]]] = []
        self.state_tabu_actions: Dict[int, Set[Any]] = collections.defaultdict(set)
        self.level_successful_clicks: Set[Tuple[int, int]] = set()
        # Dynamic entities & velocity tracking
        self.dynamic_entities: Dict[int, Dict[str, Any]] = {}

    def start_new_game(self, game_id: str) -> None:
        self.current_game_id = game_id
        self.current_level = 0
        self.action_counter = 0
        self.dead_signatures.clear()
        self.dead_walls.clear()
        self.dead_clicks.clear()
        self.dynamic_entities.clear()
        self.macro_queue.clear()
        self.prober.reset_for_new_game()
        self.memory_matcher = UniversalMemoryMatcher()
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
        self.winning_action_tails.clear()
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = {}
        self.waiting_ticks = 0
        self.prev_world = None
        self.visited_waypoints.clear()
        self.recent_state_hashes.clear()
        self.recent_actions.clear()
        self.state_tabu_actions.clear()
        self.level_successful_clicks.clear()

    def start_new_level(self, level_num: int) -> None:
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
        self.learned_lattice = None
        self.lo_lvl_N = None
        self.prev_world = None
        self.visited_waypoints.clear()
        self.recent_state_hashes.clear()
        self.recent_actions.clear()
        self.state_tabu_actions.clear()
        self.level_successful_clicks.clear()
        if self.learned_game_modality == "SOKOBAN" and not self.confirmed_box_colors:
            self.learned_game_modality = None
        self.memory_matcher.reset_for_level()
        self.pacer.start_level()

    def record_level_won(self, winning_actions: List[int]) -> None:
        self.winning_action_tails[self.current_level] = list(winning_actions)
        if self.prev_world is not None:
            pw = self.prev_world
            if self.confirmed_box_colors:
                # Confirmed Sokoban puzzle: preserve confirmed box colors
                self.learned_box_colors = set(self.confirmed_box_colors)
                self.learned_game_modality = "SOKOBAN"
                if pw.get("plates"):
                    for pr, pc in pw["plates"]:
                        c = int(pw["board"][pr, pc])
                        if c not in (0, 4, self.learned_avatar_color) and c not in self.confirmed_box_colors:
                            self.learned_goal_colors.add(c)
                if pw.get("walls"):
                    for wr, wc in pw["walls"]:
                        c = int(pw["board"][wr, wc])
                        if c not in (0, 4, self.learned_avatar_color) and c not in self.confirmed_box_colors:
                            self.learned_wall_colors.add(c)
            else:
                # No crates were ever displaced by intervention: environment is NAV / Waypoints
                self.learned_box_colors.clear()
                if self.learned_game_modality == "SOKOBAN":
                    self.learned_game_modality = "NAV"
                for ent_key in ("plates", "boxes", "exit_doors"):
                    if pw.get(ent_key):
                        for r, c in pw[ent_key]:
                            cv = int(pw["board"][r, c])
                            if cv not in (0, 4, self.learned_avatar_color):
                                self.learned_goal_colors.add(cv)

    def choose_action(
        self,
        current_grid: np.ndarray,
        available_actions: Optional[List[int]] = None,
        game_state: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        self.action_counter += 1
        legal_actions = available_actions or [1, 2, 3, 4, 5, 6, 7]

        # 0. Check Null Transition on previous step (Dead-Signature Culling & Rollback)
        if (
            self.prev_grid is not None
            and self.last_action is not None
            and self.waiting_ticks == 0
            and not getattr(self.memory_matcher, "last_was_dummy", False)
            and self.learned_game_modality not in ("TOGGLE", "MEMORY")
        ):
            interior_prev = self.prev_grid[4:60, 4:60]
            interior_curr = current_grid[4:60, 4:60]
            delta_omega = np.count_nonzero(interior_prev != interior_curr)
            full_delta = np.count_nonzero(self.prev_grid != current_grid)

            if delta_omega == 0 and full_delta == 0:
                ctx_h = fast_grid_hash(interior_prev)
                sig = pack_signature_64(
                    action_byte=self.last_action,
                    coords_12=0,
                    mass_color_12=0,
                    ctx_hash_32=ctx_h,
                )
                self.dead_signatures.add(sig)

                if self.last_action == 6 and self.last_action_data:
                    self.dead_clicks.add((
                        int(self.last_action_data.get("x", 0)),
                        int(self.last_action_data.get("y", 0)),
                    ))

                if not self.macro_queue.is_empty:
                    self.macro_queue.clear()
            elif delta_omega > 0:
                if (
                    self.last_action == 6
                    and self.last_action_data
                    and self.learned_game_modality not in ("TOGGLE", "MEMORY")
                ):
                    self.level_successful_clicks.add((
                        int(self.last_action_data.get("x", 0)),
                        int(self.last_action_data.get("y", 0)),
                    ))

        # 0.1 Periodic orbit / limit-cycle detection (Tabu list)
        if self.learned_game_modality not in ("TOGGLE", "MEMORY"):
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
        world: Optional[Dict[str, Any]] = None
        if (
            self.learned_avatar_color is not None
            and self.learned_game_modality in [None, "NAV", "SLIDE", "SOKOBAN", "KNIGHT"]
        ):
            prev_w = self.prev_world
            world = extract_discrete_board(
                current_grid,
                avatar_color=self.learned_avatar_color,
                prev_world=prev_w,
                last_action=self.last_action,
                learned_wall_colors=self.learned_wall_colors,
                learned_box_colors=self.learned_box_colors,
                learned_goal_colors=self.learned_goal_colors,
            )
            if world is not None:
                if prev_w is not None and self.last_action in (1, 2, 3, 4):
                    deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                    dr, dc = deltas[self.last_action]
                    expected_p = (prev_w["player"][0] + dr, prev_w["player"][1] + dc)

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
                                    and (
                                        (dr != 0 and (nb[0] - ob[0]) * dr > 0 and nb[1] == ob[1])
                                        or (dc != 0 and (nb[1] - ob[1]) * dc > 0 and nb[0] == ob[0])
                                    )
                                )
                                if is_causally_aligned:
                                    box_c = int(prev_w["board"][ob[0], ob[1]])
                                    if box_c not in (0, 4, self.learned_avatar_color):
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
                        if world["player"] == prev_w["player"]:
                            # Player did not move into expected cell (Collision/Wall)
                            if expected_p in prev_w["boxes"]:
                                if not self.learned_slide_push:
                                    self.learned_slide_push = True
                                    self.macro_queue.clear()
                                else:
                                    blocked_c = int(prev_w["board"][expected_p[0], expected_p[1]])
                                    if blocked_c not in (0, 4, self.learned_avatar_color):
                                        if blocked_c not in self.confirmed_box_colors:
                                            self.learned_wall_colors.add(blocked_c)
                                            self.learned_box_colors.discard(blocked_c)
                                    self.learned_game_modality = "NAV"
                                    self.dead_walls.add(expected_p)
                                    self.macro_queue.clear()
                            elif self.learned_game_modality not in ("SLIDE", "KNIGHT"):
                                if 0 <= expected_p[0] < world["gh"] and 0 <= expected_p[1] < world["gw"]:
                                    self.dead_walls.add(expected_p)
                                    blocked_c = int(prev_w["board"][expected_p[0], expected_p[1]])
                                    cur_p = world["player"]
                                    p_cur_c = int(world["board"][cur_p[0], cur_p[1]])
                                    if blocked_c not in (0, 4, self.learned_avatar_color, p_cur_c):
                                        self.learned_wall_colors.add(blocked_c)
                                    self.macro_queue.clear()
                        elif world["player"] == expected_p:
                            # Player successfully moved into expected cell without displacing a crate!
                            if expected_p in prev_w["boxes"]:
                                # The avatar stepped directly onto a cell hypothesized to be a crate without displacing it!
                                # In physical mechanics, an impassable/pushable crate cannot be stepped on directly.
                                # Therefore, this entity is a passable waypoint / plate / collectible, NOT a crate!
                                cell_c = int(prev_w["board"][expected_p[0], expected_p[1]])
                                if cell_c not in (0, 4, self.learned_avatar_color):
                                    if cell_c not in self.confirmed_box_colors:
                                        self.learned_box_colors.discard(cell_c)
                                        self.learned_goal_colors.add(cell_c)
                                        self.visited_waypoints.add(expected_p)
                                        if self.learned_game_modality == "SOKOBAN" and not self.learned_box_colors:
                                            self.learned_game_modality = "NAV"
                                        self.macro_queue.clear()
                        elif world["player"] != expected_p and self.learned_game_modality not in ("SLIDE", "KNIGHT"):
                            self.dead_walls.add(expected_p)
                            self.macro_queue.clear()

                # Dynamic entity velocity induction via causal non-intervention
                if prev_w is not None and prev_w.get("gh") == world.get("gh") and prev_w.get("gw") == world.get("gw"):
                    gh_d, gw_d = world["gh"], world["gw"]
                    p_prev = prev_w["player"]
                    p_curr = world["player"]
                    excluded_cells = {p_prev, p_curr}
                    if "ob" in locals() and "nb" in locals() and moved_crate_detected:
                        excluded_cells.add(ob)
                        excluded_cells.add(nb)

                    for r in range(gh_d):
                        for c in range(gw_d):
                            if (r, c) in excluded_cells:
                                continue
                            c_prev = int(prev_w["board"][r, c])
                            c_curr = int(world["board"][r, c])
                            if c_prev not in (0, 4, self.learned_avatar_color) and c_curr in (0, 4):
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
            if self.prober.profile.avatar_color is not None:
                self.learned_avatar_color = self.prober.profile.avatar_color
                self.learned_displacement_vectors = dict(self.prober.profile.displacement_vectors)
                if self.prober.profile.modality:
                    self.learned_game_modality = self.prober.profile.modality
            elif self.prober.profile.probed:
                self.learned_avatar_color = None
                self.learned_game_modality = self.prober.profile.modality or "CLICK"

            if probe_act is not None and probe_act in legal_actions:
                return probe_act, {}

        if self.prober.profile.probed and self.learned_game_modality is None:
            if self.prober.profile.avatar_color is not None:
                self.learned_avatar_color = self.prober.profile.avatar_color
                self.learned_displacement_vectors = dict(self.prober.profile.displacement_vectors)
                self.learned_game_modality = self.prober.profile.modality
            else:
                self.learned_game_modality = "CLICK"

        # -------------------------------------------------------------
        # Branch B: Click / Interactive Matrix Environments (Action 6)
        # -------------------------------------------------------------
        is_click_mode = (
            self.learned_game_modality in ["TOGGLE", "MEMORY", "CLICK"]
            or (self.learned_avatar_color is None and not any(a in legal_actions for a in [1, 2, 3, 4]))
            or (self.prober.profile.probed and self.learned_avatar_color is None)
        )

        if 6 in legal_actions and is_click_mode:
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
            if sep.lattice is not None and sep.tile_back_color is not None:
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

            # 4. GF(2) Impulse-Response Toggle Solver Induction
            lo_detected_N = (
                detect_lo_N(grid)
                if (self.learned_avatar_color is None)
                else None
            )
            is_lo_board = (
                lo_detected_N is not None
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
                spawn_mask = (int_curr != bg_color) & (int_curr != 4) & (int_curr != 0) & (int_prev == bg_color)
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

            # 5. Fallback for point-and-click: salient unvisited component centroids in interior [4:60, 4:60]
            fg_mask_click = (grid != bg_color) & (grid != 4) & (grid != 0)
            labeled_fg, num_fg = ndimage.label(fg_mask_click)
            ctx_h_fb = fast_grid_hash(interior)
            tabu_fb = self.state_tabu_actions.get(ctx_h_fb, set())

            # Track colors of previously successful clicks
            succ_colors: Set[int] = set()
            for sx, sy in self.level_successful_clicks:
                if 0 <= sx < 64 and 0 <= sy < 64:
                    sc = int(grid[sy, sx])
                    if sc not in (bg_color, 4, 0):
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
            for _, _, _, cc, cr, _, pts in fg_components:
                pt = (cc, cr)
                if 4 <= cc < 60 and 4 <= cr < 60 and pt not in self.dead_clicks and pt not in tabu_fb:
                    if pt not in self.level_successful_clicks:
                        candidate_click = pt
                        break
                    elif candidate_click is None:
                        candidate_click = pt

            if candidate_click is not None:
                return 6, {"x": int(candidate_click[0]), "y": int(candidate_click[1])}

            # If all centroids visited, try individual component points
            for _, _, _, _, _, _, pts in fg_components:
                for pr, pc in pts:
                    pt = (int(pc), int(pr))
                    if 4 <= pc < 60 and 4 <= pr < 60 and pt not in self.dead_clicks and pt not in tabu_fb:
                        return 6, {"x": int(pt[0]), "y": int(pt[1])}

            return 6, {"x": 32, "y": 32}

        # -------------------------------------------------------------
        # Branch C: Directional Discrete Board Extraction & Planning
        # -------------------------------------------------------------
        if world is None:
            world = extract_discrete_board(
                grid,
                avatar_color=self.learned_avatar_color,
                prev_world=self.prev_world,
                last_action=self.last_action,
                learned_wall_colors=self.learned_wall_colors,
                learned_box_colors=self.learned_box_colors,
                learned_goal_colors=self.learned_goal_colors,
            )
            if world is not None:
                self.prev_world = world

        if world is not None:
            if self.learned_avatar_color is None:
                self.learned_avatar_color = world["avatar_color"]
            p_pos = world["player"]
            walls = world["walls"] | self.dead_walls
            boxes = world["boxes"]
            plates = world["plates"]
            exit_doors = world["exit_doors"]
            all_goals = world["all_goals"]
            gh, gw = world["gh"], world["gw"]

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
                    deltas = self.learned_displacement_vectors or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                    slide_plan = solve_inertial_slide_universal(
                        p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=False, stop_on_goal=False
                    )
                    if not slide_plan:
                        slide_plan = solve_inertial_slide_universal(
                            p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=True, stop_on_goal=False
                        )
                    if not slide_plan:
                        slide_plan = solve_inertial_slide_universal(
                            p_pos, targets, walls, gh, gw, deltas=deltas, max_steps=None, allow_wrap=False, stop_on_goal=True
                        )
                    if slide_plan:
                        for a in slide_plan[1:]:
                            self.macro_queue.add_step(action=a, description="Inertial slide")
                        return slide_plan[0], {}

            # 3. SOKOBAN / CRATE MODALITY (1-Step & Sliding Crate)
            boxes_not_on_plates = boxes - plates
            sokoban_plan = None
            if len(boxes) >= 1 and len(plates) >= len(boxes) and boxes_not_on_plates and gh <= 32 and gw <= 32:
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
                    if self.learned_box_colors:
                        self.learned_game_modality = "SOKOBAN"
                    for a in sokoban_plan[1:]:
                        self.macro_queue.add_step(action=a, description="Sokoban box push")
                    return sokoban_plan[0], {}

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
                )
                if exit_plan:
                    deltas = self.learned_displacement_vectors or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
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
                    )
                    if astar_plan:
                        deltas = self.learned_displacement_vectors or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
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
                )
                if exit_plan:
                    deltas = self.learned_displacement_vectors or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
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

            targets = plates or all_goals
            if targets:
                target = min(targets, key=lambda g: abs(g[0] - p_pos[0]) + abs(g[1] - p_pos[1]))
                astar_plan = solve_cardinal_astar(
                    p_pos, target, walls, gh, gw,
                    dynamic_hazards=hazard_map, allow_wait=(5 in legal_actions),
                )
                if astar_plan:
                    deltas = self.learned_displacement_vectors or {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
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

        # -------------------------------------------------------------
        # Branch D: Non-Dead Signature Fallback
        # -------------------------------------------------------------
        ctx_h = fast_grid_hash(interior)
        for cand_act in legal_actions:
            if cand_act in [0, 7]:
                continue
            sig = pack_signature_64(cand_act, 0, 0, ctx_h)
            if sig not in self.dead_signatures:
                return cand_act, {}

        fallback = 1 if 1 in legal_actions else legal_actions[0]
        return fallback, {}
