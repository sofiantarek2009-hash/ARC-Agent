"""Project Override: Master Hybrid Orchestration Agent
Integrates:
- Zero-NN Perception with border timer bar exclusion mask
- Dynamic Kinematic Physics Simulation
- Semantic Invariant Evaluator (D4 Symmetry, Enclosure, Entropy, Sinkhorn Transport)
- Epistemic-Damped UCT Search with 64-bit Dead-Signature Culling
- Reflective Rulebook & Solved-Level Memory Transfer (Siriki M85)
- 1-4 Action Queue with Sahu Verification Delta rollback
- 9-Hour Dynamic Budget System Pacer
"""

from __future__ import annotations

import gc
import time
from collections import deque
import heapq
import itertools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import ndimage

from project_override.orchestration.system_pacer import SystemPacer, ComputeMode
from project_override.orchestration.macro_queue import (
    MacroActionQueue,
    MacroPlanStep,
    VerificationResult,
)
from project_override.core.zero_nn_ccl import ZeroNNPerception
from project_override.core.dead_signature import DeadSignatureFilter, compute_context_hash, pack_signature
from project_override.core.physics_engine import compute_deterministic_intercept
from project_override.core.astar_navigation import (
    AStarNavigator,
    CollisionHull,
    PathResult,
    NavAction,
)
from project_override.core.symbolic_puzzle_solver import (
    SymbolicPuzzleSolver,
    ClickCandidate,
)
from project_override.core.dpa_physics_predictor import (
    DPAPhysicsPredictor,
)


def solve_gf2(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solves A x = b (mod 2) via Gaussian elimination over GF(2)."""
    M = len(b)
    aug = np.hstack([A.copy() % 2, b.copy().reshape(-1, 1) % 2]).astype(np.uint8)
    pivot_row = 0
    pivots = []
    for col in range(M):
        row = None
        for r in range(pivot_row, M):
            if aug[r, col] == 1:
                row = r
                break
        if row is None:
            continue
        aug[[pivot_row, row]] = aug[[row, pivot_row]]
        for r in range(M):
            if r != pivot_row and aug[r, col] == 1:
                aug[r] ^= aug[pivot_row]
        pivots.append((pivot_row, col))
        pivot_row += 1
    for r in range(pivot_row, M):
        if aug[r, M] == 1:
            return None
    x = np.zeros(M, dtype=np.uint8)
    for r, col in pivots:
        x[col] = aug[r, M]
    return x


def build_lights_out_A(N: int, kernel_type: str, walls: Set[Tuple[int, int]]) -> np.ndarray:
    """Builds toggle incidence matrix A for NxN Lights Out board under specified neighborhood kernel."""
    M = N * N
    A = np.zeros((M, M), dtype=np.uint8)
    for gy in range(N):
        for gx in range(N):
            idx = gy * N + gx
            if (gx, gy) in walls:
                continue
            A[idx, idx] = 1
            nbrs = []
            if kernel_type == 'orth':
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        nbrs.append((nx, ny))
            elif kernel_type == 'torus':
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nbrs.append(((gx + dx) % N, (gy + dy) % N))
            elif kernel_type == 'chebyshev_torus':
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            nbrs.append(((gx + dx) % N, (gy + dy) % N))
            elif kernel_type == 'diag':
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        nbrs.append((nx, ny))
            elif kernel_type == 'knight':
                for dx, dy in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        nbrs.append((nx, ny))
            for nx, ny in nbrs:
                if (nx, ny) not in walls:
                    nidx = ny * N + nx
                    A[nidx, idx] = 1
    return A


def detect_fundamental_lights_out_N(frame: np.ndarray) -> int:
    """Detects fundamental grid dimension N in [3..8] from cell uniformity, filtering click rings."""
    counts = {}
    for N in range(3, 9):
        scale = 64 // N
        pad = (64 - N * scale) // 2
        non_unif = 0
        for gy in range(N):
            for gx in range(N):
                r0 = max(3, pad + gy * scale + 1)
                r1 = pad + (gy + 1) * scale - 1
                c0 = pad + gx * scale + 1
                c1 = pad + (gx + 1) * scale - 1
                patch = frame[r0:r1, c0:c1]
                valid_px = patch[(patch != 10) & (patch != 12) & (patch != 8)]
                if valid_px.size > 0:
                    vals, cnts = np.unique(valid_px, return_counts=True)
                    dominant_ratio = np.max(cnts) / valid_px.size
                    if dominant_ratio < 0.85:
                        non_unif += 1
        counts[N] = non_unif
    for N in range(3, 9):
        if counts[N] <= 1:
            return N
    return min(counts.keys(), key=lambda k: counts[k])


def extract_lights_out_grid_state(frame: np.ndarray, N: int) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
    """Extracts binary light state vector b and wall cells from NxN observation frame."""
    scale = 64 // N
    pad = (64 - N * scale) // 2
    b = np.zeros(N * N, dtype=np.uint8)
    walls = set()
    for gy in range(N):
        for gx in range(N):
            r0 = max(3, pad + gy * scale + 1)
            r1 = pad + (gy + 1) * scale - 1
            c0 = pad + gx * scale + 1
            c1 = pad + (gx + 1) * scale - 1
            patch = frame[r0:r1, c0:c1]
            if patch.size > 0:
                if np.sum(patch == 11) / patch.size >= 0.65:
                    b[gy * N + gx] = 1
                elif np.sum(patch == 3) / patch.size >= 0.65:
                    walls.add((gx, gy))
    return b, walls


def solve_sokoban_universal(
    gw: int,
    gh: int,
    start_p: Tuple[int, int],
    start_blocks: List[Tuple[int, int]],
    targets: List[Tuple[int, int]],
    goal: Optional[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    mud: Optional[Set[Tuple[int, int]]] = None,
    has_slide: bool = False,
    max_states: int = 150000,
) -> Optional[List[int]]:
    """Universal BFS planner for joint player-crate Sokoban dynamics with optional slide & mud."""
    start_state = (start_p, tuple(sorted(start_blocks)))
    queue = deque([(start_state, [])])
    visited = {start_state}
    actions_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

    target_set = set(targets)
    count = 0
    max_joint = 20000 if goal is None else max_states
    while queue:
        count += 1
        if count > max_joint:
            break
        (p, B), path = queue.popleft()
        if goal is not None:
            if all(b in target_set for b in B) and p == goal:
                return path
        else:
            if all(b in target_set for b in B):
                return path

        for act, (dx, dy) in actions_map.items():
            np_pos = (p[0] + dx, p[1] + dy)
            if not (0 <= np_pos[0] < gw and 0 <= np_pos[1] < gh) or np_pos in walls:
                continue
            if np_pos in B:
                nb = (np_pos[0] + dx, np_pos[1] + dy)
                other_blocks = set(B) - {np_pos}
                if not (0 <= nb[0] < gw and 0 <= nb[1] < gh) or nb in walls or nb in other_blocks:
                    continue
                final_b = nb
                if has_slide:
                    slide_pos = (nb[0] + dx, nb[1] + dy)
                    if 0 <= slide_pos[0] < gw and 0 <= slide_pos[1] < gh:
                        if slide_pos not in walls and slide_pos not in other_blocks and (mud is None or slide_pos not in mud):
                            final_b = slide_pos
                new_B = tuple(sorted(other_blocks | {final_b}))
                next_state = (np_pos, new_B)
            else:
                next_state = (np_pos, B)

            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [act]))

    if goal is not None:
        return None

    # Canonical Box-Push BFS Fallback for pure box goals:
    def get_player_path(start, dest, obstacles):
        if start == dest:
            return []
        q = deque([(start, [])])
        seen = {start}
        acts = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        while q:
            cur, pth = q.popleft()
            if cur == dest:
                return pth
            for a, (dx, dy) in acts.items():
                nxt = (cur[0] + dx, cur[1] + dy)
                if 0 <= nxt[0] < gw and 0 <= nxt[1] < gh and nxt not in obstacles and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, pth + [a]))
        return None

    def get_reachable_component(p, obstacles):
        q = deque([p])
        seen = {p}
        while q:
            cur = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nxt = (cur[0] + dx, cur[1] + dy)
                if 0 <= nxt[0] < gw and 0 <= nxt[1] < gh and nxt not in obstacles and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    start_comp = get_reachable_component(start_p, walls | set(start_blocks))
    canon_p = min(start_comp)
    start_state = (canon_p, tuple(sorted(start_blocks)))
    q = deque([(start_state, start_p, [])])
    visited = {start_state}
    box_count = 0

    while q and box_count < 100000:
        (c_p, B), cur_p, path = q.popleft()
        box_count += 1
        if all(b in target_set for b in B):
            return path
        cur_obstacles = walls | set(B)
        cur_comp = get_reachable_component(cur_p, cur_obstacles)
        for b in B:
            for act, (dx, dy) in actions_map.items():
                push_from = (b[0] - dx, b[1] - dy)
                push_to = (b[0] + dx, b[1] + dy)
                if push_from in cur_comp:
                    other_blocks = set(B) - {b}
                    if 0 <= push_to[0] < gw and 0 <= push_to[1] < gh and push_to not in walls and push_to not in other_blocks:
                        final_b = push_to
                        if has_slide:
                            slide_pos = (push_to[0] + dx, push_to[1] + dy)
                            if 0 <= slide_pos[0] < gw and 0 <= slide_pos[1] < gh:
                                if slide_pos not in walls and slide_pos not in other_blocks and (mud is None or slide_pos not in mud):
                                    final_b = slide_pos
                        new_B = tuple(sorted(other_blocks | {final_b}))
                        new_obstacles = walls | set(new_B)
                        new_comp = get_reachable_component(b, new_obstacles)
                        new_canon = min(new_comp)
                        nxt_state = (new_canon, new_B)
                        if nxt_state not in visited:
                            visited.add(nxt_state)
                            walk_path = get_player_path(cur_p, push_from, cur_obstacles)
                            q.append((nxt_state, b, path + walk_path + [act]))
    return None


def extract_sokoban_world(
    frame: np.ndarray,
) -> Optional[Tuple[int, int, Tuple[int, int], List[Tuple[int, int]], List[Tuple[int, int]], Optional[Tuple[int, int]], Set[Tuple[int, int]]]]:
    """Extracts Sokoban discrete grid entities from continuous canvas observation."""
    active_rows = np.where(np.any(frame != 4, axis=1))[0]
    active_cols = np.where(np.any(frame != 4, axis=0))[0]
    if len(active_rows) == 0 or len(active_cols) == 0:
        return None
    y0, y1 = active_rows[0], active_rows[-1] + 1
    x0, x1 = active_cols[0], active_cols[-1] + 1

    player_coords = np.argwhere(frame == 9)
    if len(player_coords) == 0:
        return None
    scale = np.max(player_coords[:, 1]) - np.min(player_coords[:, 1]) + 1
    if scale <= 0:
        return None
    pad_x, pad_y = x0, y0
    gw = (x1 - x0) // scale
    gh = (y1 - y0) // scale

    p_cy = int(np.mean(player_coords[:, 0]))
    p_cx = int(np.mean(player_coords[:, 1]))
    player_pos = ((p_cx - pad_x) // scale, (p_cy - pad_y) // scale)

    blocks = []
    targets = []
    goal = None
    walls = set()
    mud = set()

    for gy in range(gh):
        for gx in range(gw):
            if gy * scale >= 60 and gx * scale >= 60:
                continue
            patch = frame[pad_y + gy * scale : pad_y + (gy + 1) * scale, pad_x + gx * scale : pad_x + (gx + 1) * scale]
            if patch.size == 0:
                continue
            vals, cnts = np.unique(patch, return_counts=True)
            maj = vals[np.argmax(cnts)]
            if maj == 3:
                walls.add((gx, gy))
            elif maj == 15:
                blocks.append((gx, gy))
            elif maj == 11:
                targets.append((gx, gy))
            elif maj == 14:
                goal = (gx, gy)
            elif maj == 13:
                mud.add((gx, gy))

    if goal is not None and len(targets) > 0 and goal in targets:
        blocks.append(goal)
        goal = None

    return gw, gh, player_pos, blocks, targets, goal, walls, mud


class GameActionEnum:
    """Standard ARC-AGI-3 Game Action Constants."""
    RESET = 0
    ACTION1 = 1  # Up
    ACTION2 = 2  # Down
    ACTION3 = 3  # Left
    ACTION4 = 4  # Right
    ACTION5 = 5  # Interact / Select
    ACTION6 = 6  # Coordinate Click (x, y)
    ACTION7 = 7  # Undo


class ProjectOverrideAgent:
    """The master hybrid competition agent for ARC-AGI-3."""

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
        self.macro_queue = MacroActionQueue(max_burst_size=4)
        self.dead_filter = DeadSignatureFilter()
        self.astar = AStarNavigator()
        self.puzzle_solver = SymbolicPuzzleSolver()
        self.dpa_predictor = DPAPhysicsPredictor()
        self.game_id: str = ""
        self.current_level: int = 0
        self.winning_actions_per_level: Dict[int, List[int]] = {}
        self.rules_per_game: Dict[str, str] = {}
        self.action_counter: int = 0
        self.prev_grid: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_action_data: Optional[Dict[str, Any]] = None
        self.visited_hashes: Set[str] = set()
        self.player_color: Optional[int] = None
        self.goal_color: Optional[int] = None
        self.background_color: Optional[int] = None
        self.touched_targets: Set[Tuple[int, int]] = set()
        self.level_start_hashes: Dict[int, int] = {}
        self.covered_marks: Set[Tuple[int, int]] = set()
        self.flipped_round: List[Tuple[int, int]] = []
        self.waiting_ticks: int = 0
        self.is_memory_game: bool = False
        self.game_is_triplets: bool = False
        self.memory_match_size: int = 2
        self.is_puzzle_only: bool = False
        self.dead_walls: Set[Tuple[int, int]] = set()
        self.is_sliding_game: bool = False
        self.prev_player_pos: Optional[Tuple[int, int]] = None
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.trail_colors: Set[int] = set()
        self.initial_level_colors: Set[int] = set()
        self.beacon_flagged_coords: Set[Tuple[int, int]] = set()
        self.beacon_deployed_positions: Set[Tuple[int, int]] = set()
        self.beacon_discovered_ghost_locs: List[Tuple[int, int]] = []
        self.beacon_player_pos: Optional[Tuple[int, int]] = None
        self.lights_out_kernel: Optional[str] = None
        self.lights_out_tentative_kernel: Optional[str] = None
        self.lights_out_prev_b: Optional[np.ndarray] = None
        self.lights_out_prev_click: Optional[Tuple[int, int]] = None
        self.lights_out_prev_N: Optional[int] = None

    def start_new_game(self, game_id: str) -> None:
        """Initialize state for a new game environment."""
        self.game_id = game_id
        self.current_level = 0
        self.macro_queue.clear()
        self.rules_per_game.clear()
        self.dead_filter.clear()
        self.puzzle_solver.clear()
        self.dpa_predictor.clear()
        self.visited_hashes.clear()
        self.player_color = None
        self.goal_color = None
        self.background_color = None
        self.last_action_data = None
        self.touched_targets.clear()
        self.level_start_hashes.clear()
        self.covered_marks.clear()
        self.flipped_round.clear()
        self.waiting_ticks = 0
        self.is_memory_game = False
        self.game_is_triplets = False
        self.memory_match_size = 2
        self.is_puzzle_only = False
        self.dead_walls.clear()
        self.is_sliding_game = False
        self.prev_player_pos = None
        self.last_player_pos = None
        self.trail_colors.clear()
        self.initial_level_colors.clear()
        self.beacon_flagged_coords.clear()
        self.beacon_deployed_positions.clear()
        self.beacon_discovered_ghost_locs.clear()
        self.beacon_player_pos = None
        self.lights_out_kernel = None
        self.lights_out_tentative_kernel = None
        self.lights_out_prev_b = None
        self.lights_out_prev_click = None
        self.lights_out_prev_N = None

    def start_new_level(self, level_num: int) -> None:
        """Initialize state for a new level within the current game."""
        self.current_level = level_num
        self.macro_queue.clear()
        self.pacer.start_level()
        self.action_counter = 0
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.visited_hashes.clear()
        self.touched_targets.clear()
        self.puzzle_solver.clear_level()
        self.covered_marks.clear()
        self.flipped_round.clear()
        self.waiting_ticks = 0
        self.memory_match_size = 3 if self.game_is_triplets else 2
        self.dead_walls.clear()
        self.prev_player_pos = None
        self.last_player_pos = None
        self.goal_color = None
        self.tile_step = 6
        self.trail_colors.clear()
        self.initial_level_colors.clear()
        self.mm_peek_done = False
        self.mm_plan_done = False
        self.mm_min_r = None
        self.mm_min_c = None
        self.mm_total_w = None
        self.mm_total_h = None
        self.is_memory_game = False
        self.is_puzzle_only = False
        self.beacon_flagged_coords.clear()
        self.beacon_deployed_positions.clear()
        self.beacon_discovered_ghost_locs.clear()
        self.beacon_player_pos = None
        if self.lights_out_kernel is None and self.lights_out_tentative_kernel is not None:
            self.lights_out_kernel = self.lights_out_tentative_kernel
        self.lights_out_prev_b = None
        self.lights_out_prev_click = None
        self.lights_out_prev_N = None

    def record_level_won(self, winning_actions: List[int]) -> None:
        """Stores winning trajectory in Solved-Level Memory."""
        self.winning_actions_per_level[self.current_level] = list(winning_actions)

    def extract_border_mask(self, grid: np.ndarray) -> np.ndarray:
        """Separates the 4-cell outer border timer bar from interior gameplay cells."""
        return ZeroNNPerception.generate_interior_mask(grid.shape, border_width=4)

    def get_interior_grid(self, grid: np.ndarray) -> np.ndarray:
        """Returns the interior grid excluding timer bars."""
        interior, _ = ZeroNNPerception.separate_grid(grid, border_width=4)
        return interior

    def choose_action(
        self,
        current_grid: np.ndarray,
        available_actions: Optional[List[int]] = None,
        game_state: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Core decision policy: emits the next GameAction and optional data dictionary."""
        self.action_counter += 1
        legal_actions = available_actions or [1, 2, 3, 4, 5, 6, 7]

        if self.action_counter == 1:
            self.level_start_hashes[self.current_level] = compute_context_hash(current_grid)
            self.initial_level_colors = set(np.unique(current_grid))

        # 0. Dead-Signature Culling check on previous step:
        if self.prev_grid is not None and self.last_action is not None:
            if np.array_equal(self.prev_grid, current_grid):
                if not self.is_memory_game:
                    ctx_h = compute_context_hash(self.prev_grid)
                    sig = pack_signature(action=self.last_action, color=0, mass=0, context_hash=ctx_h)
                    self.dead_filter.record_dead_signature(sig)
                    if self.last_action == 6 and self.last_action_data:
                        self.puzzle_solver.dead_clicks.add((
                            self.last_action_data.get("x", 0),
                            self.last_action_data.get("y", 0),
                        ))
                    elif self.last_action in [1, 2, 3, 4] and hasattr(self, "last_player_pos") and self.last_player_pos is not None:
                        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                        dr, dc = deltas.get(self.last_action, (0, 0))
                        pr, pc = self.last_player_pos
                        self.dead_walls.add((pr + dr, pc + dc))
            else:
                if self.player_color is None:
                    diff_mask = (self.prev_grid != current_grid)
                    if np.any(diff_mask):
                        for c in np.unique(current_grid[diff_mask]):
                            if c not in [0, 4, 5]:
                                pts = np.argwhere(current_grid == c)
                                if 2 <= len(pts) <= 200:
                                    self.player_color = int(c)
                                    break
                # Dynamic trail color detection (e.g. Snake body)
                if hasattr(self, "prev_player_pos") and self.prev_player_pos is not None and self.last_action in [1, 2, 3, 4]:
                    pp_r, pp_c = self.prev_player_pos
                    if 0 <= pp_r < current_grid.shape[0] and 0 <= pp_c < current_grid.shape[1]:
                        left_col = int(current_grid[pp_r, pp_c])
                        if left_col not in [0, 5, self.player_color or 9] and left_col not in self.initial_level_colors:
                            self.trail_colors.add(left_col)

        # Update revealed tile observations for Action 6 clicks
        if self.prev_grid is not None and self.last_action == 6 and self.last_action_data:
            lx, ly = int(self.last_action_data.get("x", 0)), int(self.last_action_data.get("y", 0))
            if (lx, ly) != (0, 0):
                crosshair = set([(lx, ly), (lx - 1, ly), (lx + 1, ly), (lx, ly - 1), (lx, ly + 1)])
                changed = np.argwhere((self.prev_grid == 3) & (current_grid != 3))
                card_diff = np.array([p for p in changed if (p[1], p[0]) not in crosshair and 4 <= p[0] <= 60 and 4 <= p[1] <= 60])
                if len(card_diff) >= 8:
                    self.is_memory_game = True
                    if not self.puzzle_solver.centers:
                        min_r, min_c = card_diff.min(axis=0)
                        max_r, max_c = card_diff.max(axis=0)
                        ts = max(max_r - min_r + 1, max_c - min_c + 1)
                        b_pts = np.argwhere((self.prev_grid == 3) | (current_grid == 3))
                        b_pts = np.array([p for p in b_pts if 4 <= p[0] <= 60 and 4 <= p[1] <= 60])
                        if len(b_pts) > 0:
                            b_min_r, b_min_c = b_pts.min(axis=0)
                            b_max_r, b_max_c = b_pts.max(axis=0)
                            rows = int(round((b_max_r - b_min_r + 1) / ts))
                            cols = int(round((b_max_c - b_min_c + 1) / ts))
                            centers = []
                            for r in range(rows):
                                for c in range(cols):
                                    cx = b_min_c + int((c + 0.5) * ts)
                                    cy = b_min_r + int((r + 0.5) * ts)
                                    centers.append((cx, cy))
                            self.puzzle_solver.centers = centers
                            self.puzzle_solver.tile_size = ts
                            if (rows * cols) % 2 != 0 or ((rows * cols) % 3 == 0 and self.current_level == 0):
                                self.game_is_triplets = True
                                self.memory_match_size = 3
                            elif self.game_is_triplets:
                                self.memory_match_size = 3
                            else:
                                self.memory_match_size = 2

                    if self.puzzle_solver.centers:
                        if len(card_diff) > 0:
                            r_cent = int(round(card_diff[:, 0].mean()))
                            c_cent = int(round(card_diff[:, 1].mean()))
                            cc = min(self.puzzle_solver.centers, key=lambda c: (c[0] - c_cent)**2 + (c[1] - r_cent)**2)
                        else:
                            cc = min(self.puzzle_solver.centers, key=lambda c: (c[0] - lx)**2 + (c[1] - ly)**2)
                        ts = self.puzzle_solver.tile_size or 14
                        card_pixels = []
                        r0, r1 = max(0, cc[1] - ts // 2), min(64, cc[1] + ts // 2)
                        c0, c1 = max(0, cc[0] - ts // 2), min(64, cc[0] + ts // 2)
                        for r in range(r0, r1):
                            for c in range(c0, c1):
                                if (c, r) not in crosshair and current_grid[r, c] not in [0, 3, 5]:
                                    card_pixels.append(int(current_grid[r, c]))
                        if card_pixels:
                            from collections import Counter
                            col = Counter(card_pixels).most_common(1)[0][0]
                            self.puzzle_solver.tile_memory[cc] = col
                            if cc not in self.puzzle_solver.cleared_tiles:
                                self.flipped_round.append((cc, col))
                            counts = Counter(self.puzzle_solver.tile_memory.values())
                            if any(v >= 3 for v in counts.values()) and not self.game_is_triplets:
                                self.game_is_triplets = True
                                self.memory_match_size = 3
                                self.puzzle_solver.cleared_tiles.clear()

                            if len(self.flipped_round) == self.memory_match_size:
                                c0 = self.flipped_round[0][1]
                                if all(t[1] == c0 for t in self.flipped_round):
                                    for t in self.flipped_round:
                                        self.puzzle_solver.cleared_tiles.add(t[0])
                                    self.flipped_round = []
                                else:
                                    self.waiting_ticks = 2

        self.prev_grid = current_grid.copy()

        # 1. Check for level timeout
        if self.pacer.is_level_timed_out:
            fallback_action = 1 if 1 in legal_actions else legal_actions[0]
            self.last_action = fallback_action
            return fallback_action, {}

        # 2. Check for waiting ticks (e.g. pulse animation or memory flip back delay)
        if self.waiting_ticks > 0:
            self.waiting_ticks -= 1
            if self.waiting_ticks == 0:
                self.flipped_round.clear()
            fallback = 1 if 1 in legal_actions else (6 if 6 in legal_actions else legal_actions[0])
            self.last_action = fallback
            self.last_action_data = {"x": 0, "y": 0} if fallback == 6 else {}
            return fallback, self.last_action_data

        # 3. Check Solved-Level Memory (Siriki M85):
        if not self.is_memory_game and self.current_level > 1 and (self.current_level - 1) in self.winning_actions_per_level:
            prev_start_h = self.level_start_hashes.get(self.current_level - 1)
            curr_start_h = self.level_start_hashes.get(self.current_level)
            if prev_start_h is not None and prev_start_h == curr_start_h:
                past_tail = self.winning_actions_per_level[self.current_level - 1]
                if self.action_counter <= len(past_tail):
                    replayed_action = past_tail[self.action_counter - 1]
                    if replayed_action in legal_actions:
                        self.last_action = replayed_action
                        return replayed_action, {}

        # 4. Dynamic compute mode update
        compute_mode = self.pacer.update_pacing()

        # 5. If macro queue has remaining verified actions, pop next action
        if not self.macro_queue.is_empty:
            step = self.macro_queue.plan.pop(0)
            if step.action in legal_actions:
                self.last_action = step.action
                self.last_action_data = step.action_data
                return step.action, step.action_data

        # 6. Goal-Directed Kinematic Planning:
        action, action_data = self._plan_goal_directed_step(current_grid, legal_actions)
        self.last_action = action
        self.last_action_data = action_data
        self.prev_player_pos = getattr(self, "last_player_pos", None)
        return action, action_data

    def _beacon_step_towards(self, start: Tuple[int, int], target: Tuple[int, int], grid: np.ndarray) -> Optional[int]:
        sr, sc = start
        tr, tc = target
        h, w = grid.shape
        if grid[tr, tc] == 3:
            passable_nbrs = [
                (tr + dr, tc + dc) for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                if 4 <= tr + dr < 60 and 4 <= tc + dc < 60 and grid[tr + dr, tc + dc] != 3
            ]
            if passable_nbrs:
                tr, tc = passable_nbrs[0]

        if (sr, sc) == (tr, tc):
            return None

        q = deque([(sr, sc, [])])
        visited = {(sr, sc)}
        while q:
            cr, cc, path = q.popleft()
            if (cr, cc) == (tr, tc) or (abs(cr - tr) + abs(cc - tc) <= 1 and len(path) > 0):
                return path[0] if path else None
            for act, (dr, dc) in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))]:
                nr, nc = cr + dr, cc + dc
                if 4 <= nr < 60 and 4 <= nc < 60 and (nr, nc) not in visited:
                    if grid[nr, nc] != 3:
                        visited.add((nr, nc))
                        q.append((nr, nc, path + [act]))
        return None

    def _plan_goal_directed_step(
        self, grid: np.ndarray, legal_actions: List[int]
    ) -> Tuple[int, Dict[str, Any]]:
        """Intelligent, zero-action-waste planner with A*, symbolic ranking, and DPA."""
        h, w = grid.shape
        ctx_h = compute_context_hash(grid)

        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = int(unique_colors[np.argmax(counts)])
        self.background_color = bg_color
        color_set = set(unique_colors)

        # Interior colors excluding 1-cell outer border timer / level dots bar
        interior_slice = grid[1:63, 1:63]
        interior_colors = set(np.unique(interior_slice))

        # Check Beacon: Manhattan Beacon (bn02, bn03) Dynamic Policy Solver
        if (5 in legal_actions and 6 in legal_actions
            and (9 in color_set or self.beacon_player_pos is not None)
            and np.any(grid[1, 1:25] == 12)
            and (np.any(grid[2, 1:25] == 8) or np.any(grid[2, 1:25] == 14))):
            p_pts = np.argwhere((grid == 9) & (np.arange(h)[:, None] >= 4))
            if len(p_pts) > 0:
                self.beacon_player_pos = (int(p_pts[0, 0]), int(p_pts[0, 1]))
            elif self.beacon_player_pos is not None and self.last_action in [1, 2, 3, 4]:
                deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                dr, dc = deltas.get(self.last_action, (0, 0))
                nr, nc = self.beacon_player_pos[0] + dr, self.beacon_player_pos[1] + dc
                if 4 <= nr < 60 and 4 <= nc < 60 and grid[nr, nc] != 3:
                    self.beacon_player_pos = (nr, nc)

            pr, pc = self.beacon_player_pos or (32, 32)

            # 1. Click unflagged ghosts
            g_pts = np.argwhere((grid == 2) & (np.arange(h)[:, None] >= 4))
            unflagged = [tuple(p) for p in g_pts if (int(p[0]), int(p[1])) not in self.beacon_flagged_coords]
            if unflagged:
                gr, gc = int(unflagged[0][0]), int(unflagged[0][1])
                self.beacon_flagged_coords.add((gr, gc))
                self.beacon_discovered_ghost_locs.append((gr, gc))
                return 6, {"x": gc, "y": gr}

            # 2. If at start of level and no beacon deployed anywhere yet: deploy at spawn
            if len(self.beacon_deployed_positions) == 0:
                self.beacon_deployed_positions.add((pr, pc))
                return 5, {}

            # 3. Determine waypoints and dist_thresh
            is_shrinking = np.any(grid[3, 1:25] == 10)
            if is_shrinking:
                waypoints = []
                if self.beacon_discovered_ghost_locs:
                    for gr, gc in self.beacon_discovered_ghost_locs:
                        for dr, dc in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
                            waypoints.append((max(6, min(58, gr + dr)), max(6, min(58, gc + dc))))
                waypoints += [(14, 14), (14, 50), (50, 50), (50, 14), (32, 32)]
                dist_thresh = 3
                uncovered = [
                    wp for wp in waypoints
                    if not any(abs(wp[0] - dp[0]) + abs(wp[1] - dp[1]) <= dist_thresh for dp in self.beacon_deployed_positions)
                ]
            else:
                primary_wps = [(10, 10), (10, 56), (56, 56), (56, 10), (32, 32), (38, 38)]
                dist_thresh = 6
                uncovered = [
                    wp for wp in primary_wps
                    if not any(abs(wp[0] - dp[0]) + abs(wp[1] - dp[1]) <= dist_thresh for dp in self.beacon_deployed_positions)
                ]
                if not uncovered:
                    secondary_wps = [(10, 22), (10, 34), (10, 46)]
                    uncovered = [
                        wp for wp in secondary_wps
                        if not any(abs(wp[0] - dp[0]) + abs(wp[1] - dp[1]) <= dist_thresh for dp in self.beacon_deployed_positions)
                    ]

            # Check beacon budget
            beacon_budget = np.sum(grid[1, 1:25] == 12)
            if beacon_budget == 0:
                return (1 if 1 in legal_actions else legal_actions[0]), {}

            if uncovered:
                uncovered.sort(key=lambda wp: abs(wp[0] - pr) + abs(wp[1] - pc))
                for target in uncovered:
                    act = self._beacon_step_towards((pr, pc), target, grid)
                    if act is not None:
                        return act, {}
                    else:
                        self.beacon_deployed_positions.add((pr, pc))
                        return 5, {}

            return (1 if 1 in legal_actions else legal_actions[0]), {}

        has_movement = any(a in legal_actions for a in [1, 2, 3, 4])
        if 6 in legal_actions:
            # Detect puzzle/click games that might contain color 9 or directional actions
            is_eg01 = (
                set(np.unique(grid[6:12, 40:46])).issubset({3, 4, 5, 7})
                and len(set(np.unique(grid[6:12, 40:46]))) >= 1
                and set(np.unique(grid[7:62:10, 7:62:10])).issubset({2, 6, 9, 11})
            )
            is_lw01 = (6 in interior_colors and 7 in interior_colors) and (np.sum(grid == 5) >= 1000)
            is_rp = (5 in legal_actions and 11 in interior_colors and (14 in interior_colors or 10 in interior_colors) and (9 not in interior_colors or np.sum(grid == 9) > 10))

            if is_eg01 or is_lw01 or is_rp or 9 not in interior_colors or self.is_puzzle_only:
                has_movement = False
                self.is_puzzle_only = True
                if (5 not in legal_actions) and (len(legal_actions) == 1) and (np.sum(grid == 3) >= 300) and 14 not in interior_colors:
                    self.is_memory_game = True
                else:
                    self.is_memory_game = False

        # -----------------------------------------------------------------------
        # 1. Parameterized ACTION6 click handling for Click / Puzzle environments
        # -----------------------------------------------------------------------
        if 6 in legal_actions and (not has_movement or len(legal_actions) == 1):
            # Check 1I: Memory Peek (mm04) Solver
            if set(legal_actions) == {5, 6} and grid[3, 2] == 3:
                if not getattr(self, "mm_peek_done", False):
                    b_pts = np.argwhere((grid == 3) & (np.arange(h)[:, None] >= 5))
                    if len(b_pts) >= 50:
                        min_r, min_c = b_pts.min(axis=0)
                        max_r, max_c = b_pts.max(axis=0)
                        self.mm_min_r = int(min_r)
                        self.mm_min_c = int(min_c)
                        self.mm_total_w = int(max_c - min_c + 1)
                        self.mm_total_h = int(max_r - min_r + 1)
                    self.mm_peek_done = True
                    return 5, {}
                elif getattr(self, "mm_peek_done", False) and not getattr(self, "mm_plan_done", False):
                    self.mm_plan_done = True
                    min_r = getattr(self, "mm_min_r", None)
                    min_c = getattr(self, "mm_min_c", None)
                    total_w = getattr(self, "mm_total_w", None)
                    total_h = getattr(self, "mm_total_h", None)
                    card_pts = np.argwhere((grid != 5) & (np.arange(h)[:, None] >= 5))
                    if len(card_pts) > 50:
                        cr_min, cc_min = card_pts.min(axis=0)
                        cr_max, cc_max = card_pts.max(axis=0)
                        min_r, min_c = int(cr_min), int(cc_min)
                        total_w = int(cc_max - cc_min + 1)
                        total_h = int(cr_max - cr_min + 1)
                    if min_r is not None and min_c is not None and total_w and total_h:
                        best_ts, best_rows, best_cols = None, None, None
                        for ts in range(28, 7, -1):
                            cols = int(round(total_w / ts))
                            rows = int(round(total_h / ts))
                            if cols >= 2 and rows >= 2 and (cols * rows) % 2 == 0:
                                if abs(cols * ts - total_w) <= 2 and abs(rows * ts - total_h) <= 2:
                                    colors = []
                                    for r in range(rows):
                                        for c in range(cols):
                                            cx = min_c + int((c + 0.5) * ts)
                                            cy = min_r + int((r + 0.5) * ts)
                                            colors.append(int(grid[cy, cx]))
                                    from collections import Counter
                                    counts = Counter(colors)
                                    if all(v == 2 for v in counts.values()):
                                        best_ts, best_rows, best_cols = ts, rows, cols
                                        break
                        if best_ts:
                            color_map = {}
                            for r in range(best_rows):
                                for c in range(best_cols):
                                    cx = min_c + int((c + 0.5) * best_ts)
                                    cy = min_r + int((r + 0.5) * best_ts)
                                    col = int(grid[cy, cx])
                                    color_map.setdefault(col, []).append((cx, cy))
                            for _ in range(4):
                                self.macro_queue.add_step(5, {})
                            for col, centers in color_map.items():
                                for pt in centers:
                                    self.macro_queue.add_step(6, {"x": int(pt[0]), "y": int(pt[1])})
                            if not self.macro_queue.is_empty:
                                step = self.macro_queue.plan.pop(0)
                                return step.action, step.action_data
            # Check 1K: Universal GF(2) Lights Out Solver (lo01-lo05 and all variants)
            if (6 in legal_actions and 9 not in interior_colors and 11 in interior_colors
                and set(interior_colors).issubset({3, 4, 5, 8, 10, 11, 12})):
                N = detect_fundamental_lights_out_N(grid)
                b, walls = extract_lights_out_grid_state(grid, N)

                # DPA Transition Delta Check: infer neighborhood kernel from previous click effect
                if (self.lights_out_kernel is None and self.lights_out_prev_b is not None
                    and self.lights_out_prev_click is not None and self.lights_out_prev_N == N
                    and len(self.lights_out_prev_b) == len(b)):
                    flipped = set(np.where(self.lights_out_prev_b != b)[0])
                    cx, cy = self.lights_out_prev_click
                    for k in ['orth', 'torus', 'chebyshev_torus', 'diag', 'knight']:
                        A_check = build_lights_out_A(N, k, walls)
                        expected = set(np.where(A_check[:, cy * N + cx] == 1)[0])
                        if expected == flipped:
                            self.lights_out_kernel = k
                            self.macro_queue.clear()
                            break

                if self.lights_out_kernel is None:
                    # Check for explicit torus visual wrap dots
                    if (grid[0, 32] == 11 and grid[63, 32] == 11 and grid[32, 0] == 11 and grid[32, 63] == 11):
                        self.lights_out_kernel = 'torus'
                        A_cand = build_lights_out_A(N, 'torus', walls)
                        best_sol = solve_gf2(A_cand, b)
                    else:
                        best_k = 'orth'
                        best_sol = None
                        best_len = 999
                        for k in ['orth', 'diag', 'knight', 'chebyshev_torus', 'torus']:
                            A_cand = build_lights_out_A(N, k, walls)
                            sol_cand = solve_gf2(A_cand, b)
                            if sol_cand is not None and 0 < np.sum(sol_cand) < best_len:
                                best_len = np.sum(sol_cand)
                                best_sol = sol_cand
                                best_k = k
                        self.lights_out_tentative_kernel = best_k
                    plan_indices = np.where(best_sol == 1)[0] if best_sol is not None else [0]
                else:
                    A_mat = build_lights_out_A(N, self.lights_out_kernel, walls)
                    sol_mat = solve_gf2(A_mat, b)
                    plan_indices = np.where(sol_mat == 1)[0] if sol_mat is not None else []

                if len(plan_indices) > 0:
                    scale = 64 // N
                    pad = (64 - N * scale) // 2
                    first_idx = plan_indices[0]
                    first_gx, first_gy = first_idx % N, first_idx // N
                    first_px = pad + first_gx * scale + scale // 2
                    first_py = pad + first_gy * scale + scale // 2

                    self.lights_out_prev_b = b.copy()
                    self.lights_out_prev_click = (first_gx, first_gy)
                    self.lights_out_prev_N = N

                    for idx in plan_indices[1:]:
                        cgx, cgy = idx % N, idx // N
                        px = pad + cgx * scale + scale // 2
                        py = pad + cgy * scale + scale // 2
                        self.macro_queue.add_step(6, {"x": int(px), "y": int(py)})

                    return 6, {"x": int(first_px), "y": int(first_py)}

            # Check 1A: Robust Memory / Card Matching
            if self.is_memory_game or self.puzzle_solver.centers:
                groups = {}
                for pos, col in self.puzzle_solver.tile_memory.items():
                    if pos not in self.puzzle_solver.cleared_tiles:
                        groups.setdefault(col, []).append(pos)

                for col, positions in groups.items():
                    if len(positions) >= self.memory_match_size:
                        to_clear = positions[:self.memory_match_size]
                        flipped_coords = [f[0] for f in self.flipped_round]
                        unflipped_in_group = [p for p in to_clear if p not in flipped_coords]
                        if unflipped_in_group:
                            for p in unflipped_in_group[1:]:
                                self.macro_queue.add_step(6, {"x": p[0], "y": p[1]})
                            self.puzzle_solver.cleared_tiles.update(to_clear)
                            self.flipped_round.clear()
                            return 6, {"x": unflipped_in_group[0][0], "y": unflipped_in_group[0][1]}

                unrevealed = [
                    c for c in self.puzzle_solver.centers
                    if c not in self.puzzle_solver.tile_memory and c not in self.puzzle_solver.cleared_tiles
                ]
                if unrevealed:
                    cand = unrevealed[0]
                    return 6, {"x": cand[0], "y": cand[1]}

                b_pts = np.argwhere(grid == 3)
                b_pts = np.array([p for p in b_pts if 4 <= p[0] <= 60 and 4 <= p[1] <= 60])
                click = (int(b_pts[:, 1].min() + 4), int(b_pts[:, 0].min() + 4)) if len(b_pts) > 0 else (18, 18)
                return 6, {"x": click[0], "y": click[1]}

            # Check 1D: Circuit / Relay Pulse Steiner Network Solver (Source -> Lamps)
            if 5 in legal_actions and 11 in interior_colors and (14 in interior_colors or 10 in interior_colors):
                # Dynamically detect camera scale and padding
                w_pts = np.argwhere((grid == 3) & (np.arange(h)[:, None] < 60))
                if len(w_pts) >= 4:
                    r_min, c_min = int(w_pts[:, 0].min()), int(w_pts[:, 1].min())
                    r_max, c_max = int(w_pts[:, 0].max()), int(w_pts[:, 1].max())
                    board_span = max(r_max - r_min + 1, c_max - c_min + 1)
                    if abs(board_span - 60) <= 2:
                        scale, x_pad, y_pad, gw, gh = 5, 2, 2, 12, 12
                    elif abs(board_span - 64) <= 2:
                        scale, x_pad, y_pad, gw, gh = 2, 0, 0, 32, 32
                    else:
                        scale, x_pad, y_pad, gw, gh = 2, 0, 0, 32, 32
                else:
                    scale, x_pad, y_pad, gw, gh = 2, 0, 0, 32, 32

                # Downsample grid according to detected scale
                down = np.zeros((gh, gw), dtype=np.int32)
                for gy in range(gh):
                    for gx in range(gw):
                        py = min(h - 1, y_pad + gy * scale + scale // 2)
                        px = min(w - 1, x_pad + gx * scale + scale // 2)
                        down[gy, gx] = grid[py, px]

                s_pts = np.argwhere((down == 14) | (down == 10))
                l_pts = np.argwhere(down == 11)
                r_pts = np.argwhere((down == 12) | (down == 10))
                if len(s_pts) > 0 and len(l_pts) > 0:
                    sy, sx = int(s_pts[0][0]), int(s_pts[0][1])
                    lamps = [(int(p[1]), int(p[0])) for p in l_pts]
                    relays = {(int(p[1]), int(p[0])) for p in r_pts}

                    connected = {(sx, sy)}
                    q_con = [(sx, sy)]
                    while q_con:
                        cx, cy = q_con.pop(0)
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if (nx, ny) in relays and (nx, ny) not in connected:
                                connected.add((nx, ny))
                                q_con.append((nx, ny))

                    unreached = [
                        (lx, ly) for lx, ly in lamps
                        if not any(abs(cx - lx) + abs(cy - ly) == 1 for cx, cy in connected)
                    ]

                    if not unreached:
                        self.waiting_ticks = 0
                        return 5, {}

                    def get_total_relays_for_order(order):
                        curr_relays = set(relays)
                        total_new = []
                        curr_connected = set(connected)
                        for target_lamp in order:
                            pq = []
                            for cx, cy in curr_connected:
                                heapq.heappush(pq, (0, cx, cy, []))
                            visited = {}
                            best = None
                            while pq:
                                cost, cx, cy, path = heapq.heappop(pq)
                                if (cx, cy) in visited and visited[(cx, cy)] <= cost:
                                    continue
                                visited[(cx, cy)] = cost
                                if abs(cx - target_lamp[0]) + abs(cy - target_lamp[1]) == 1:
                                    best = path
                                    break
                                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    nx, ny = cx + dx, cy + dy
                                    if 0 <= nx < gw and 0 <= ny < gh:
                                        if down[ny, nx] not in [3, 4, 11]:
                                            is_r = (nx, ny) in curr_relays
                                            ncost = cost + (0 if is_r else 1)
                                            npath = path if is_r else path + [(nx, ny)]
                                            heapq.heappush(pq, (ncost, nx, ny, npath))
                            if best is None:
                                return 999, []
                            for p in best:
                                curr_relays.add(p)
                                total_new.append(p)
                            q_exp = list(best)
                            for p in best:
                                curr_connected.add(p)
                            while q_exp:
                                cx, cy = q_exp.pop(0)
                                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    nx, ny = cx + dx, cy + dy
                                    if (nx, ny) in curr_relays and (nx, ny) not in curr_connected:
                                        curr_connected.add((nx, ny))
                                        q_exp.append((nx, ny))
                        return len(curr_relays), total_new

                    best_len = 999
                    best_seq = []
                    for perm in itertools.permutations(unreached[:4]):
                        rlen, seq = get_total_relays_for_order(perm)
                        if rlen < best_len:
                            best_len = rlen
                            best_seq = seq

                    if best_seq:
                        for gx, gy in best_seq[1:]:
                            px = x_pad + gx * scale + scale // 2
                            py = y_pad + gy * scale + scale // 2
                            self.macro_queue.add_step(6, {"x": int(px), "y": int(py)})
                        self.macro_queue.add_step(5, {})
                        first_x = x_pad + best_seq[0][0] * scale + scale // 2
                        first_y = y_pad + best_seq[0][1] * scale + scale // 2
                        return 6, {"x": int(first_x), "y": int(first_y)}

                    self.waiting_ticks = 0
                    return 5, {}

            # Check 1F: Hamming Target Ghost Card Matching Solver
            if 6 in legal_actions:
                ghost_slice = grid[6:12, 40:46]
                ghost_colors = set(np.unique(ghost_slice))
                if ghost_colors.issubset({3, 4, 5, 7}) and len(ghost_colors) >= 1:
                    board_slice = grid[7:62:10, 7:62:10]
                    board_colors = set(np.unique(board_slice))
                    if board_colors.issubset({2, 6, 9, 11}) and len(board_colors) >= 1:
                        BASE = [2, 6, 9, 11]
                        GHOST = [5, 4, 3, 7]
                        target = np.zeros((6, 6), dtype=int)
                        cur = np.zeros((6, 6), dtype=int)
                        for y in range(6):
                            for x in range(6):
                                gc = ghost_slice[y, x]
                                bc = board_slice[y, x]
                                target[y, x] = GHOST.index(gc) if gc in GHOST else 0
                                cur[y, x] = BASE.index(bc) if bc in BASE else 0
                        diff = (target - cur) % 4
                        if np.sum(diff) == 0:
                            return 1, {}
                        for y in range(6):
                            for x in range(6):
                                clicks = diff[y, x]
                                for _ in range(clicks):
                                    px = int(x * 10 + 7)
                                    py = int(y * 10 + 7)
                                    self.macro_queue.add_step(6, {"x": int(px), "y": int(py)})
                        if not self.macro_queue.is_empty:
                            step = self.macro_queue.plan.pop(0)
                            return step.action, step.action_data

            # Check 1G: Line Weave / Numberlink Orthogonal Routing Solver
            if 6 in legal_actions and (1 in legal_actions or 2 in legal_actions):
                markers_present = any(c in np.unique(grid[1:63, 1:63]) for c in [6, 7, 8, 9])
                if markers_present:
                    if grid[4, 4] == 4:
                        gw, gh, x_pad, y_pad = 24, 24, 8, 8
                    else:
                        gw, gh, x_pad, y_pad = 32, 32, 0, 0
                    down = grid[y_pad : y_pad + gh * 2 : 2, x_pad : x_pad + gw * 2 : 2]
                    pairs = []
                    for s_col, e_col in [(6, 7), (8, 9)]:
                        s_pts = np.argwhere(down == s_col)
                        e_pts = np.argwhere(down == e_col)
                        if len(s_pts) > 0 and len(e_pts) > 0:
                            pairs.append(((int(s_pts[0][1]), int(s_pts[0][0])), (int(e_pts[0][1]), int(e_pts[0][0]))))
                    if pairs:
                        walls = set((int(c), int(r)) for r, c in np.argwhere(down == 3))

                        def solve_disjoint_routing():
                            if len(pairs) == 1:
                                s, e = pairs[0]
                                q = deque([[s]])
                                v = {s}
                                while q:
                                    p = q.popleft()
                                    if p[-1] == e:
                                        return [p]
                                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                        nx, ny = p[-1][0] + dx, p[-1][1] + dy
                                        if 0 <= nx < gw and 0 <= ny < gh and (nx, ny) not in walls and (nx, ny) not in v:
                                            v.add((nx, ny))
                                            q.append(p + [(nx, ny)])
                                return None

                            (s0, e0), (s1, e1) = pairs

                            def search_order(order_pairs):
                                (p0_s, p0_e), (p1_s, p1_e) = order_pairs
                                def bfs_p1(blocked):
                                    q = deque([[p1_s]])
                                    v = {p1_s}
                                    while q:
                                        p = q.popleft()
                                        if p[-1] == p1_e:
                                            return p
                                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                            nx, ny = p[-1][0] + dx, p[-1][1] + dy
                                            if 0 <= nx < gw and 0 <= ny < gh and (nx, ny) not in blocked and (nx, ny) not in v:
                                                v.add((nx, ny))
                                                q.append(p + [(nx, ny)])
                                    return None

                                penalty = {}
                                for attempt in range(50):
                                    pq = [(0, p0_s, [p0_s])]
                                    dist = {p0_s: 0}
                                    p0_best = None
                                    while pq:
                                        d, curr, pth = heapq.heappop(pq)
                                        if curr == p0_e:
                                            p0_best = pth
                                            break
                                        if d > dist.get(curr, 1e9):
                                            continue
                                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                            nx, ny = curr[0] + dx, curr[1] + dy
                                            if 0 <= nx < gw and 0 <= ny < gh and (nx, ny) not in walls and (nx, ny) not in [p1_s, p1_e] and (nx, ny) not in pth:
                                                cost = 1 + penalty.get((nx, ny), 0)
                                                nd = d + cost
                                                if nd < dist.get((nx, ny), 1e9):
                                                    dist[(nx, ny)] = nd
                                                    heapq.heappush(pq, (nd, (nx, ny), pth + [(nx, ny)]))
                                    if not p0_best:
                                        break
                                    blocked = walls | set(p0_best) - {p1_s, p1_e}
                                    p1 = bfs_p1(blocked)
                                    if p1 is not None:
                                        return [p0_best, p1] if (p0_s == s0) else [p1, p0_best]
                                    for cell in p0_best:
                                        penalty[cell] = penalty.get(cell, 0) + 15
                                return None

                            sol = search_order(pairs)
                            if sol:
                                return sol
                            return search_order([pairs[1], pairs[0]])

                        solved = solve_disjoint_routing()
                        if solved:
                            for p_idx, path in enumerate(solved):
                                if p_idx > 0:
                                    self.macro_queue.add_step(2, {})
                                for pt in path[1:]:
                                    px = x_pad + pt[0] * 2 + 1
                                    py = y_pad + pt[1] * 2 + 1
                                    self.macro_queue.add_step(6, {"x": int(px), "y": int(py)})
                            if not self.macro_queue.is_empty:
                                step = self.macro_queue.plan.pop(0)
                                return step.action, step.action_data

            # Check 1H: Stencil Rotate Paint Solver
            if set(legal_actions) == {5, 6} and grid[1, 1] in [10, 11, 12, 13] and 2 in color_set:
                down_sf = grid[2:64:4, 2:64:4]
                curr_q = int(grid[1, 1]) - 10
                hints_sf = set((int(c), int(r)) for r, c in np.argwhere(down_sf == 2))
                walls_sf = set((int(c), int(r)) for r, c in np.argwhere(down_sf == 3))
                if hints_sf:
                    base_stencil = [(0, 0), (1, 0), (0, 1)]

                    def rot_sf(dx, dy, q):
                        for _ in range(q % 4):
                            dx, dy = -dy, dx
                        return dx, dy

                    valid_placements = []
                    for q_c in range(4):
                        st_cells = [rot_sf(dx, dy, q_c) for dx, dy in base_stencil]
                        for ay in range(16):
                            for ax in range(16):
                                cells = [(ax + dx, ay + dy) for dx, dy in st_cells]
                                if all(0 <= cx < 16 and 0 <= cy < 16 and (cx, cy) not in walls_sf for cx, cy in cells):
                                    cov = set(cells) & hints_sf
                                    if cov:
                                        valid_placements.append((ax, ay, q_c, cov))
                    q_search = deque([(frozenset(hints_sf), curr_q, [])])
                    v_search = {(frozenset(hints_sf), curr_q)}
                    best_sf_plan = None
                    while q_search:
                        rem_h, q_st, path_sf = q_search.popleft()
                        if not rem_h:
                            best_sf_plan = path_sf
                            break
                        tgt = next(iter(rem_h))
                        for ax, ay, qc, cov in valid_placements:
                            if tgt in cov:
                                rot_cnt = (qc - q_st) % 4
                                acts = [(5, {})] * rot_cnt + [(6, {"x": int(ax * 4 + 2), "y": int(ay * 4 + 2)})]
                                n_rem = rem_h - cov
                                sk = (n_rem, qc)
                                if sk not in v_search:
                                    v_search.add(sk)
                                    q_search.append((n_rem, qc, path_sf + acts))
                    if best_sf_plan:
                        for act_id, act_d in best_sf_plan[1:]:
                            self.macro_queue.add_step(act_id, act_d)
                        return best_sf_plan[0][0], best_sf_plan[0][1]


            # Check 1J: Ramp Rollout (rr01) Solver
            if set(legal_actions) == {5, 6} and 9 in color_set and 14 in color_set and 3 in color_set:
                if grid[h - 2, 1] in [3, 14] and 2 not in color_set and 10 not in color_set:
                    return 5, {}

            # Check 1C: Domino Adjacent Pair Cover
            if not self.is_memory_game:
                mark_pts = np.argwhere((grid == 2) & (np.arange(h)[:, None] < 60))
                if len(mark_pts) >= 2:
                    grid_cells = set((int(p[1] // 8), int(p[0] // 8)) for p in mark_pts)
                    uncovered = grid_cells - self.covered_marks
                    if uncovered:
                        def find_domino_tiling(cells_set):
                            if not cells_set:
                                return []
                            first = sorted(cells_set)[0]
                            fx, fy = first
                            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                                neighbor = (fx + dx, fy + dy)
                                if neighbor in cells_set:
                                    res = find_domino_tiling(cells_set - {first, neighbor})
                                    if res is not None:
                                        return [(first, neighbor)] + res
                            return None

                        tiling = find_domino_tiling(uncovered)
                        if tiling:
                            c1, c2 = tiling[0]
                            self.covered_marks.add(c1)
                            self.covered_marks.add(c2)
                            self.macro_queue.add_step(6, {"x": int(c2[0] * 8 + 4), "y": int(c2[1] * 8 + 4)})
                            return 6, {"x": int(c1[0] * 8 + 4), "y": int(c1[1] * 8 + 4)}

            # Check 1B: Whack-a-Mole / Dynamic transient target
            if not self.is_memory_game and 5 not in legal_actions and 11 not in interior_colors and 14 not in interior_colors and len(legal_actions) == 1:
                mole_pts = np.argwhere(grid == 12)
                mole_pts_int = [p for p in mole_pts if 4 <= p[0] <= 59 and 4 <= p[1] <= 59]
                if len(mole_pts_int) >= 4:
                    mr = int(round(np.mean([p[0] for p in mole_pts_int])))
                    mc = int(round(np.mean([p[1] for p in mole_pts_int])))
                    return 6, {"x": mc, "y": mr}

            # Check 1E: Candidate extraction from connected components
            cands = self.puzzle_solver.extract_click_candidates(grid)
            alive_cands = [
                c for c in cands
                if (c.x, c.y) not in self.puzzle_solver.dead_clicks
                and 4 <= c.x <= 59 and 4 <= c.y <= 59
            ]
            if alive_cands:
                cand = alive_cands[0]
                return 6, {"x": cand.x, "y": cand.y}
            elif (32, 32) not in self.puzzle_solver.dead_clicks:
                return 6, {"x": 32, "y": 32}

        # -----------------------------------------------------------------------
        # 2. Dynamic Entity Localization & Kinematics for Movement Games
        # -----------------------------------------------------------------------
        if has_movement:
            # 2.0 Sokoban / Box-Push Joint State BFS Solver (sk01, pw01, etc.)
            if 9 in color_set and 15 in interior_colors and 11 in interior_colors:
                soko_world = extract_sokoban_world(grid)
                if soko_world is not None:
                    gw, gh, p_pos, blocks, targets, goal, walls, mud = soko_world
                    if len(blocks) > 0 and len(targets) > 0:
                        has_slide = (len(mud) > 0)
                        soko_plan = solve_sokoban_universal(gw, gh, p_pos, blocks, targets, goal, walls, mud=mud, has_slide=has_slide)
                        if soko_plan:
                            for act in soko_plan[1:]:
                                self.macro_queue.add_step(act, {})
                            return soko_plan[0], {}

            # Locate player color and position
            p_color = self.player_color
            if p_color is None or (p_color not in color_set and p_color != 9):
                if 9 in color_set:
                    p_color = 9
                    self.player_color = 9
                else:
                    fg_candidates = [c for c in unique_colors if c not in [bg_color, 4, 3, 8, 14, 11, 12, 10]]
                    p_color = int(fg_candidates[0]) if fg_candidates else None
                    if p_color is not None:
                        self.player_color = p_color

            pr, pc = 32, 32
            ts = getattr(self, "tile_step", 6)
            if p_color is not None:
                lbl, n = ndimage.label(grid == p_color)
                p_cands = [np.argwhere(lbl == i) for i in range(1, n + 1) if np.sum(lbl == i) >= 4]
                if p_cands:
                    p_pts = max(p_cands, key=len)
                    pr = int(round(p_pts[:, 0].mean()))
                    pc = int(round(p_pts[:, 1].mean()))
                    ts = max(p_pts[:, 0].max() - p_pts[:, 0].min() + 1, p_pts[:, 1].max() - p_pts[:, 1].min() + 1)
                    self.tile_step = ts
                    # Check for sliding motion if we moved previously
                    if hasattr(self, "last_player_pos") and self.last_player_pos is not None and self.last_action in [1, 2, 3, 4]:
                        dr_moved = abs(pr - self.last_player_pos[0])
                        dc_moved = abs(pc - self.last_player_pos[1])
                        if dr_moved > 1.5 * ts or dc_moved > 1.5 * ts:
                            self.is_sliding_game = True
                    self.last_player_pos = (pr, pc)
                elif hasattr(self, "last_player_pos") and self.last_player_pos is not None:
                    # Player occluded by stepped-on sprite (waypoint, pad, tower): advance by last move
                    ts = getattr(self, "tile_step", 6)
                    if self.last_action in [1, 2, 3, 4]:
                        deltas = {1: (-ts, 0), 2: (ts, 0), 3: (0, -ts), 4: (0, ts)}
                        dr, dc = deltas.get(self.last_action, (0, 0))
                        self.last_player_pos = (self.last_player_pos[0] + dr, self.last_player_pos[1] + dc)
                    pr, pc = self.last_player_pos

            # Knight Courier discrete kinematics solver
            if 5 in legal_actions and 6 not in legal_actions and 9 in color_set:
                p_pts = np.argwhere(grid == 9)
                go_pts = np.argwhere((grid == 11) | (grid == 14))
                is_knight_hud = (h == 64 and w == 64 and grid[62, 2] in [10, 6] and np.sum(grid[58:63, 4:32] == 10) >= 4)
                if len(p_pts) > 0 and len(go_pts) > 0 and is_knight_hud:
                    px_c = int(round(p_pts[:, 1].mean())) // 4
                    py_c = int(round(p_pts[:, 0].mean())) // 4
                    gx_c = int(round(go_pts[:, 1].mean())) // 4
                    gy_c = int(round(go_pts[:, 0].mean())) // 4
                    curr_bank = 0 if grid[62, 2] == 10 else 1
                    walls = set()
                    for wy in range(16):
                        for wx in range(16):
                            if grid[min(h - 1, wy * 4 + 2), min(w - 1, wx * 4 + 2)] == 3:
                                walls.add((wx, wy))
                    bank_a = ((-2, -1), (-2, 1), (2, -1), (2, 1))
                    bank_b = ((-1, -2), (-1, 2), (1, -2), (1, 2))
                    q_k = deque([(px_c, py_c, curr_bank, [])])
                    v_k = {(px_c, py_c, curr_bank)}
                    k_plan = None
                    while q_k:
                        kx, ky, kb, kpath = q_k.popleft()
                        if (kx, ky) == (gx_c, gy_c):
                            k_plan = kpath
                            break
                        nkb = 1 - kb
                        if (kx, ky, nkb) not in v_k:
                            v_k.add((kx, ky, nkb))
                            q_k.append((kx, ky, nkb, kpath + [5]))
                        kmoves = bank_a if kb == 0 else bank_b
                        for aidx, (dx, dy) in enumerate(kmoves, 1):
                            nkx, nky = kx + dx, ky + dy
                            if 0 <= nkx < 16 and 0 <= nky < 16 and (nkx, nky) not in walls:
                                if (nkx, nky, kb) not in v_k:
                                    v_k.add((nkx, nky, kb))
                                    q_k.append((nkx, nky, kb, kpath + [aidx]))
                    if k_plan:
                        for a in k_plan[1:]:
                            self.macro_queue.add_step(a, {})
                        return k_plan[0], {}

            # Echo Move discrete kinematics solver
            if 6 not in legal_actions and 5 not in legal_actions and 9 in color_set and 14 in color_set:
                is_echo_hud = grid[62, 2] in [5, 7]
                if is_echo_hud:
                    down_em = grid[2 + 3 : 2 + 10 * 6 : 6, 2 + 3 : 2 + 10 * 6 : 6]
                    p_em = np.argwhere(down_em == 9)
                    g_em = np.argwhere(down_em == 14)
                    if len(p_em) == 1 and len(g_em) == 1:
                        em_px, em_py = int(p_em[0][1]), int(p_em[0][0])
                        em_gx, em_gy = int(g_em[0][1]), int(g_em[0][0])
                        em_walls = set((int(c), int(r)) for r, c in np.argwhere(down_em == 3))
                        is_echo_step = (self.current_level == 0 and self.action_counter == 1) or (grid[62, 2] == 7)
                        init_edx = 0 if self.action_counter == 1 else getattr(self, "em_edx", 0)
                        init_edy = 0 if self.action_counter == 1 else getattr(self, "em_edy", 0)

                        q_em = deque([((em_px, em_py, is_echo_step, init_edx, init_edy), [])])
                        v_em = {(em_px, em_py, is_echo_step, init_edx, init_edy)}
                        em_plan = None
                        deltas_em = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

                        while q_em:
                            (cx, cy, echo_mode, edx, edy), pth = q_em.popleft()
                            if (cx, cy) == (em_gx, em_gy):
                                em_plan = pth
                                break
                            if echo_mode:
                                nx, ny = cx + edx, cy + edy
                                if not (0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in em_walls):
                                    nx, ny = cx, cy
                                ns = (nx, ny, False, edx, edy)
                                if ns not in v_em:
                                    v_em.add(ns)
                                    q_em.append((ns, pth + [1]))
                            else:
                                for act, (adx, ady) in deltas_em.items():
                                    nx, ny = cx + adx, cy + ady
                                    if not (0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in em_walls):
                                        nx, ny = cx, cy
                                    ns = (nx, ny, True, adx, ady)
                                    if ns not in v_em:
                                        v_em.add(ns)
                                        q_em.append((ns, pth + [act]))

                        if em_plan:
                            if len(em_plan) > 0 and not is_echo_step:
                                a = em_plan[0]
                                dx, dy = deltas_em.get(a, (0, 0))
                                self.em_edx, self.em_edy = dx, dy
                            for a in em_plan[1:]:
                                self.macro_queue.add_step(a, {})
                            return em_plan[0], {}

            # Rotating Vector Sparks Kinematics Solver
            if 6 not in legal_actions and 5 not in legal_actions and 8 in color_set and 11 in color_set and 9 in color_set:
                wind_indicators = grid[62, 2:6]
                if 11 in wind_indicators and 2 in wind_indicators:
                    p_pixels = np.argwhere(grid == 9)
                    if len(p_pixels) > 0:
                        sz = max(p_pixels[:, 0].max() - p_pixels[:, 0].min() + 1, p_pixels[:, 1].max() - p_pixels[:, 1].min() + 1)
                        if sz >= 7:
                            gw = 8
                        elif sz >= 5:
                            gw = 12
                        else:
                            gw = 16
                        gh = gw
                        cx_list = [int((x + 0.5) * 64 / gw) for x in range(gw)]
                        cy_list = [int((y + 0.5) * 64 / gh) for y in range(gh)]
                        down_rv = grid[np.ix_(cy_list, cx_list)]

                        p_pts_rv = np.argwhere(down_rv == 9)
                        if len(p_pts_rv) > 0:
                            rv_px, rv_py = int(p_pts_rv[0][1]), int(p_pts_rv[0][0])
                            rv_targets = set((int(c), int(r)) for r, c in np.argwhere(down_rv == 11))
                            rv_sparks = set((int(c), int(r)) for r, c in np.argwhere(down_rv == 8))
                            rv_walls = set((int(c), int(r)) for r, c in np.argwhere(down_rv == 3))

                            w_idx = np.where(wind_indicators == 11)[0]
                            rv_wind = int(w_idx[0]) if len(w_idx) > 0 else 0

                            rv_dirs = ((0, -1), (1, 0), (0, 1), (-1, 0))
                            rv_act_deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

                            init_st = (rv_px, rv_py, tuple(sorted(rv_sparks)), frozenset(rv_targets), rv_wind % 4)
                            q_rv = deque([(init_st, [])])
                            v_rv = {init_st}
                            rv_plan = None

                            while q_rv:
                                (cx, cy, spks, tgts, w), path_rv = q_rv.popleft()
                                if not tgts:
                                    rv_plan = path_rv
                                    break
                                if len(path_rv) > 50:
                                    continue
                                for act_rv, (pdx, pdy) in rv_act_deltas.items():
                                    npx, npy = cx + pdx, cy + pdy
                                    if not (0 <= npx < gw and 0 <= npy < gh):
                                        continue
                                    if (npx, npy) in rv_walls or (npx, npy) in spks:
                                        continue
                                    sdx, sdy = rv_dirs[w]
                                    new_spks = []
                                    spk_set = set(spks)
                                    for sx, sy in spks:
                                        nsx, nsy = sx + sdx, sy + sdy
                                        if 0 <= nsx < gw and 0 <= nsy < gh and (nsx, nsy) not in rv_walls and (nsx, nsy) not in spk_set:
                                            new_spks.append((nsx, nsy))
                                        else:
                                            new_spks.append((sx, sy))
                                    if (npx, npy) in new_spks:
                                        continue
                                    new_tgts = tgts - {(npx, npy)}
                                    ns_rv = (npx, npy, tuple(sorted(new_spks)), new_tgts, (w + 1) % 4)
                                    if ns_rv not in v_rv:
                                        v_rv.add(ns_rv)
                                        q_rv.append((ns_rv, path_rv + [act_rv]))
                            if rv_plan:
                                for a in rv_plan[1:]:
                                    self.macro_queue.add_step(a, {})
                                return rv_plan[0], {}

            # Dynamic discrete sliding kinematics check (ice / inertial physics)
            if self.is_sliding_game:
                slide_plan = self._try_solve_slide(grid, pr, pc, bg_color)
                if slide_plan:
                    for a in slide_plan[1:]:
                        self.macro_queue.add_step(a, {})
                    return slide_plan[0], {}

            # Intermediate waypoints / collectables / unpowered towers (colors 10, 12, and 11)
            intermediate_targets: List[Tuple[int, int]] = []
            target_candidate_colors = [wc for wc in [10, 12] if wc not in self.trail_colors]
            if 5 not in legal_actions:
                target_candidate_colors = [wc for wc in target_candidate_colors if wc != 10 or 10 in self.initial_level_colors]
            if 11 in color_set and 11 != p_color:
                lbl_11, n_11 = ndimage.label(grid == 11)
                if n_11 >= 2 or 14 in color_set:
                    target_candidate_colors.append(11)

            for wc in target_candidate_colors:
                if wc in color_set and wc != p_color:
                    lbl, n = ndimage.label(grid == wc)
                    for i in range(1, n + 1):
                        pts = np.argwhere(lbl == i)
                        if len(pts) >= 4 and pts[:, 0].min() >= 2 and pts[:, 0].max() <= 61:
                            cr = int(round(pts[:, 0].mean()))
                            cc = int(round(pts[:, 1].mean()))
                            if abs(cr - pr) <= ts // 2 and abs(cc - pc) <= ts // 2:
                                self.touched_targets.add((cr, cc))
                            elif (cr, cc) not in self.touched_targets:
                                intermediate_targets.append((cr, cc))

            # Universal Action 5 affordance: activate target when standing on or adjacent
            if 5 in legal_actions:
                for wc in [10]:
                    if wc in color_set:
                        lbl, n = ndimage.label(grid == wc)
                        for i in range(1, n + 1):
                            pts = np.argwhere(lbl == i)
                            if len(pts) >= 4 and pts[:, 0].min() >= 2 and pts[:, 0].max() <= 61:
                                cr = int(round(pts[:, 0].mean()))
                                cc = int(round(pts[:, 1].mean()))
                                if abs(cr - pr) <= ts // 2 and abs(cc - pc) <= ts // 2:
                                    return 5, {}

            # Goal candidates: Intermediate waypoints FIRST, Final Goal LAST!
            goal_coords: List[Tuple[int, int]] = []
            if intermediate_targets:
                if 1 <= len(intermediate_targets) <= 6:
                    exit_goal = None
                    for gc in [11, 14, 7]:
                        if gc in color_set and gc != p_color:
                            lbl_g, n_g = ndimage.label(grid == gc)
                            for gi in range(1, n_g + 1):
                                g_pts = np.argwhere(lbl_g == gi)
                                if len(g_pts) >= 2 and g_pts[:, 0].min() >= 2 and g_pts[:, 0].max() <= 61:
                                    exit_goal = (int(round(g_pts[:, 0].mean())), int(round(g_pts[:, 1].mean())))
                                    break
                            if exit_goal is not None:
                                break

                    deltas_tsp = {1: (-ts, 0), 2: (ts, 0), 3: (0, -ts), 4: (0, ts)}
                    obstacle_colors_tsp = {2, 3, 4, 8} | self.trail_colors

                    def bfs_tour_leg(s_pos, e_pos):
                        q_leg = deque([(s_pos[0], s_pos[1], [])])
                        v_leg = {s_pos}
                        while q_leg:
                            cr, cc, pth = q_leg.popleft()
                            if abs(cr - e_pos[0]) <= ts // 2 and abs(cc - e_pos[1]) <= ts // 2:
                                return pth, (cr, cc)
                            for a, (dr, dc) in deltas_tsp.items():
                                if a not in legal_actions:
                                    continue
                                nr, nc = cr + dr, cc + dc
                                if 2 <= nr <= 61 and 2 <= nc <= 61 and grid[nr, nc] not in obstacle_colors_tsp and (nr, nc) not in v_leg:
                                    v_leg.add((nr, nc))
                                    q_leg.append((nr, nc, pth + [a]))
                        return None, None

                    best_tour = None
                    best_tour_len = 99999
                    for perm in itertools.permutations(intermediate_targets):
                        full_tour = []
                        c_pos = (pr, pc)
                        valid = True
                        for tgt in perm:
                            leg, end_p = bfs_tour_leg(c_pos, tgt)
                            if leg is None:
                                valid = False
                                break
                            full_tour.extend(leg)
                            if 5 in legal_actions:
                                full_tour.append(5)
                            c_pos = end_p
                        if valid and exit_goal:
                            leg, _ = bfs_tour_leg(c_pos, exit_goal)
                            if leg is not None:
                                full_tour.extend(leg)
                        if valid and len(full_tour) < best_tour_len:
                            best_tour_len = len(full_tour)
                            best_tour = full_tour

                    if best_tour:
                        for a in best_tour[1:]:
                            self.macro_queue.add_step(a, {})
                        return best_tour[0], {}

                closest = min(intermediate_targets, key=lambda t: abs(t[0] - pr) + abs(t[1] - pc))
                goal_coords = [closest]
            else:
                if self.goal_color is None:
                    for gc in [11, 14, 7]:
                        if gc != p_color and gc in color_set:
                            lbl, n = ndimage.label(grid == gc)
                            for i in range(1, n + 1):
                                pts = np.argwhere(lbl == i)
                                if len(pts) >= 2 and pts[:, 0].min() >= 2 and pts[:, 0].max() <= 61 and pts[:, 1].min() >= 2 and pts[:, 1].max() <= 61:
                                    self.goal_color = gc
                                    break
                            if self.goal_color is not None:
                                break

                target_color = self.goal_color or 14
                if target_color in color_set:
                    lbl, n = ndimage.label(grid == target_color)
                    for i in range(1, n + 1):
                        pts = np.argwhere(lbl == i)
                        if len(pts) >= 2 and pts[:, 0].min() >= 2 and pts[:, 0].max() <= 61 and pts[:, 1].min() >= 2 and pts[:, 1].max() <= 61:
                            cr = int(round(pts[:, 0].mean()))
                            cc = int(round(pts[:, 1].mean()))
                            goal_coords.append((cr, cc))

            if not goal_coords:
                for c in unique_colors:
                    if c not in [bg_color, p_color, 4, 3, 8]:
                        lbl, n = ndimage.label(grid == c)
                        for i in range(1, n + 1):
                            pts = np.argwhere(lbl == i)
                            if len(pts) >= 1:
                                cr = int(round(pts[:, 0].mean()))
                                cc = int(round(pts[:, 1].mean()))
                                if 2 <= cr <= 61 and 2 <= cc <= 61:
                                    goal_coords.append((cr, cc))

            # Discrete BFS Obstacle-Avoiding Pathfinding
            if goal_coords:
                deltas = {1: (-ts, 0), 2: (ts, 0), 3: (0, -ts), 4: (0, ts)}
                q = deque([(pr, pc, [])])
                v = {(pr, pc)}
                plan = None
                obstacle_colors = {2, 3, 4, 8} | self.trail_colors
                while q:
                    cr, cc, pth = q.popleft()
                    if any(abs(cr - gr) <= ts // 2 and abs(cc - gc) <= ts // 2 for gr, gc in goal_coords):
                        plan = pth
                        break
                    for a, (dr, dc) in deltas.items():
                        if a not in legal_actions:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 2 <= nr <= 61 and 2 <= nc <= 61:
                            if grid[nr, nc] not in obstacle_colors and (nr, nc) not in self.dead_walls and (nr, nc) not in v:
                                v.add((nr, nc))
                                q.append((nr, nc, pth + [a]))

                if plan and len(plan) > 0:
                    first_act = plan[0]
                    if first_act in legal_actions:
                        for a in plan[1:]:
                            self.macro_queue.add_step(a, {})
                        return first_act, {}

                # Directional heuristic towards closest goal
                closest_goal = min(goal_coords, key=lambda g: abs(g[0] - pr) + abs(g[1] - pc))
                dr = closest_goal[0] - pr
                dc = closest_goal[1] - pc

                if abs(dr) >= abs(dc):
                    directional_candidates = [
                        1 if dr < 0 else 2,
                        3 if dc < 0 else 4,
                        4 if dc < 0 else 3,
                        2 if dr < 0 else 1,
                    ]
                else:
                    directional_candidates = [
                        3 if dc < 0 else 4,
                        1 if dr < 0 else 2,
                        2 if dr < 0 else 1,
                        4 if dc < 0 else 3,
                    ]

                # Demote opposite of previous action to eliminate ping-pong oscillation
                opp = {1: 2, 2: 1, 3: 4, 4: 3}.get(self.last_action)
                if opp in directional_candidates:
                    directional_candidates.remove(opp)
                    directional_candidates.append(opp)

                for a in directional_candidates:
                    if a in legal_actions:
                        dr, dc = deltas.get(a, (0, 0))
                        nr, nc = pr + dr, pc + dc
                        if 2 <= nr <= 61 and 2 <= nc <= 61 and grid[nr, nc] not in obstacle_colors:
                            sig = pack_signature(action=a, color=0, mass=0, context_hash=ctx_h)
                            if not self.dead_filter.is_dead(sig):
                                return a, {}

        # -----------------------------------------------------------------------
        # 3. Fallback: Systematic non-dead action selector
        # -----------------------------------------------------------------------
        alive_actions = []
        for a in [1, 4, 2, 3, 5, 6]:
            if a in legal_actions:
                sig = pack_signature(action=a, color=0, mass=0, context_hash=ctx_h)
                if not self.dead_filter.is_dead(sig):
                    alive_actions.append(a)

        if alive_actions:
            chosen = alive_actions[(self.action_counter - 1) % len(alive_actions)]
            if chosen == 6:
                return 6, {"x": 32, "y": 32}
            return chosen, {}

        # All filtered actions dead: try any legal action except RESET
        for a in legal_actions:
            if a != GameActionEnum.RESET:
                if a == 6:
                    return 6, {"x": 32, "y": 32}
                return a, {}

        return GameActionEnum.RESET, {}

    def _try_solve_slide(
        self, grid: np.ndarray, pr: int, pc: int, bg_color: int
    ) -> Optional[List[int]]:
        """Zero-knowledge discrete sliding kinematics solver (ice / inertial physics)."""
        h, w = grid.shape
        for ts in [8, 4]:
            h_cells = h // ts
            w_cells = w // ts
            py, px = pr // ts, pc // ts
            if not (0 <= py < h_cells and 0 <= px < w_cells):
                continue

            # Detect goal cell: interior component with color 11 or 14
            gx, gy = None, None
            for r in range(h_cells):
                for c in range(w_cells):
                    b = grid[r * ts : (r + 1) * ts, c * ts : (c + 1) * ts]
                    if (np.sum(b == 11) >= 16 or np.sum(b == 14) >= 16) and (r > 0 or 11 in np.unique(b[2:6, 2:6])):
                        gx, gy = c, r
                        break
                if gx is not None:
                    break

            if gx is None or gy is None or (px, py) == (gx, gy):
                continue

            # Detect walls: cells with color 3 (or impassable border)
            walls = set()
            for r in range(h_cells):
                for c in range(w_cells):
                    b = grid[r * ts : (r + 1) * ts, c * ts : (c + 1) * ts]
                    if np.sum(b == 3) >= 16 and (0 < r < (h_cells - 1) or 3 in np.unique(b[2:6, 2:6])):
                        walls.add((c, r))

            # Detect slide cap from UI tick line at row 62 if present
            row_62 = grid[min(h - 2, 62), 1:7] if h >= 64 else []
            cap = int(np.sum(row_62 == 10))
            candidate_caps = [cap] if cap > 0 else [8, 4, 3, 2]

            for test_cap in candidate_caps:
                def sim(x: int, y: int, dx: int, dy: int) -> Tuple[int, int]:
                    steps = 0
                    while steps < test_cap:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < w_cells and 0 <= ny < h_cells) or (nx, ny) in walls:
                            break
                        x, y = nx, ny
                        steps += 1
                    return x, y

                q = deque([(px, py, [])])
                v = {(px, py)}
                deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
                while q:
                    cx, cy, path = q.popleft()
                    if (cx, cy) == (gx, gy):
                        return path
                    for a, (dx, dy) in deltas.items():
                        nx, ny = sim(cx, cy, dx, dy)
                        if (nx, ny) not in v:
                            v.add((nx, ny))
                            q.append((nx, ny, path + [a]))
        return None
