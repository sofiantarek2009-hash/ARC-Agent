
"""Project Override: Simulated Model Planner & Rollback Search Engine (ARC-AGI-3)
Mathematical, 100% General Offline Simulation, Deadlock Pruning, Epistemic Probing,
Verification Delta Rollback, and RHAE-Optimal MacroQueue Execution.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass, field
from enum import IntEnum
import heapq
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np


# =====================================================================
# 1. Action Space and Geometric Constants
# =====================================================================

class SimulatedAction(IntEnum):
    RESET = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5
    CLICK = 6
    UNDO = 7


ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    SimulatedAction.UP: (-1, 0),
    SimulatedAction.DOWN: (1, 0),
    SimulatedAction.LEFT: (0, -1),
    SimulatedAction.RIGHT: (0, 1),
}

DELTA_TO_ACTION: Dict[Tuple[int, int], int] = {
    (-1, 0): SimulatedAction.UP,
    (1, 0): SimulatedAction.DOWN,
    (0, -1): SimulatedAction.LEFT,
    (0, 1): SimulatedAction.RIGHT,
}

OPPOSITE_ACTIONS: Dict[int, int] = {
    SimulatedAction.UP: SimulatedAction.DOWN,
    SimulatedAction.DOWN: SimulatedAction.UP,
    SimulatedAction.LEFT: SimulatedAction.RIGHT,
    SimulatedAction.RIGHT: SimulatedAction.LEFT,
}


# =====================================================================
# 2. Immutable Discrete Simulated State
# =====================================================================

@dataclass(frozen=True)
class SimulatedState:
    """Hashable, immutable representation of the discrete world state."""
    player_pos: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]] = frozenset()
    active_switches: FrozenSet[Tuple[int, int]] = frozenset()
    open_doors: FrozenSet[Tuple[int, int]] = frozenset()
    collected_items: FrozenSet[Tuple[int, int]] = frozenset()
    hazard_step: int = 0
    step_count: int = 0

    @property
    def key(self) -> Tuple[Any, ...]:
        return (
            self.player_pos,
            self.boxes,
            self.active_switches,
            self.open_doors,
            self.collected_items,
            self.hazard_step,
        )


# =====================================================================
# 3. Irreversible Mistake Prevention & Deadlock Detector
# =====================================================================

class DeadlockDetector:
    """Rigorous mathematical detectors for irreversible dead-end states."""

    @staticmethod
    def is_concave_corner(
        pos: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        goal_plates: Set[Tuple[int, int]],
        gh: int,
        gw: int,
    ) -> bool:
        """Concave Corner Deadlock Theorem:
        If a pushable box b not in goal_plates has walls/boundaries along two
        perpendicular directions, it can never be pulled out. Deadlock!
        """
        if pos in goal_plates:
            return False
        r, c = pos

        def is_blocked(nr: int, nc: int) -> bool:
            return nr < 0 or nr >= gh or nc < 0 or nc >= gw or (nr, nc) in walls

        up = is_blocked(r - 1, c)
        down = is_blocked(r + 1, c)
        left = is_blocked(r, c - 1)
        right = is_blocked(r, c + 1)

        return (up and left) or (up and right) or (down and left) or (down and right)

    @staticmethod
    def is_continuous_wall_deadlock(
        pos: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        goal_plates: Set[Tuple[int, int]],
        gh: int,
        gw: int,
    ) -> bool:
        """Continuous Wall Deadlock Theorem:
        If a box b not in goal_plates is flush against a continuous wall,
        and no goal plates exist along the wall corridor between corners,
        and no escape gap exists, the box is permanently unscoreable.
        """
        if pos in goal_plates:
            return False
        r, c = pos

        def is_blocked(nr: int, nc: int) -> bool:
            return nr < 0 or nr >= gh or nc < 0 or nc >= gw or (nr, nc) in walls

        # Check horizontal wall line above or below
        for dr in (-1, 1):
            if is_blocked(r + dr, c):
                # Scan left
                c_left = c
                while c_left >= 0 and not is_blocked(r, c_left) and is_blocked(r + dr, c_left):
                    if (r, c_left) in goal_plates:
                        break
                    c_left -= 1
                has_left_goal = (r, c_left) in goal_plates if c_left >= 0 else False
                left_open = c_left >= 0 and not is_blocked(r, c_left) and not is_blocked(r + dr, c_left)

                # Scan right
                c_right = c
                while c_right < gw and not is_blocked(r, c_right) and is_blocked(r + dr, c_right):
                    if (r, c_right) in goal_plates:
                        break
                    c_right += 1
                has_right_goal = (r, c_right) in goal_plates if c_right < gw else False
                right_open = c_right < gw and not is_blocked(r, c_right) and not is_blocked(r + dr, c_right)

                if not has_left_goal and not has_right_goal and not left_open and not right_open:
                    return True

        # Check vertical wall line left or right
        for dc in (-1, 1):
            if is_blocked(r, c + dc):
                # Scan up
                r_up = r
                while r_up >= 0 and not is_blocked(r_up, c) and is_blocked(r_up, c + dc):
                    if (r_up, c) in goal_plates:
                        break
                    r_up -= 1
                has_up_goal = (r_up, c) in goal_plates if r_up >= 0 else False
                up_open = r_up >= 0 and not is_blocked(r_up, c) and not is_blocked(r_up, c + dc)

                # Scan down
                r_down = r
                while r_down < gh and not is_blocked(r_down, c) and is_blocked(r_down, c + dc):
                    if (r_down, c) in goal_plates:
                        break
                    r_down += 1
                has_down_goal = (r_down, c) in goal_plates if r_down < gh else False
                down_open = r_down < gh and not is_blocked(r_down, c) and not is_blocked(r_down, c + dc)

                if not has_up_goal and not has_down_goal and not up_open and not down_open:
                    return True

        return False

    @staticmethod
    def is_2x2_deadlock(
        boxes: FrozenSet[Tuple[int, int]],
        walls: Set[Tuple[int, int]],
        goal_plates: Set[Tuple[int, int]],
        gh: int,
        gw: int,
    ) -> bool:
        """2x2 Invariant Block Deadlock Theorem:
        If any 2x2 square consists entirely of walls and boxes, with at least
        one box, and not all boxes in the 2x2 are on goal plates, none of those
        boxes can ever be pushed. Deadlock!
        """
        blocked = set(boxes) | set(walls)
        for b in boxes:
            if b in goal_plates:
                continue
            r, c = b
            for dr, dc in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                r0, c0 = r + dr, c + dc
                if 0 <= r0 < gh - 1 and 0 <= c0 < gw - 1:
                    quad = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
                    if all(p in blocked for p in quad):
                        quad_boxes = [p for p in quad if p in boxes]
                        if not all(p in goal_plates for p in quad_boxes):
                            return True
        return False

    @staticmethod
    def is_bipartite_matching_deadlock(
        boxes: FrozenSet[Tuple[int, int]],
        goal_plates: Set[Tuple[int, int]],
        walls: Set[Tuple[int, int]],
        gh: int,
        gw: int,
    ) -> bool:
        """Reachability Deadlock:
        If the number of boxes that can reach at least one goal plate is strictly
        less than the number of goal plates, the puzzle is unachievable.
        """
        if not goal_plates:
            return False
        if len(boxes) < len(goal_plates):
            return True

        open_cells = {(r, c) for r in range(gh) for c in range(gw) if (r, c) not in walls}
        reachable_boxes: Set[Tuple[int, int]] = set()

        for g in goal_plates:
            if g in open_cells:
                visited = {g}
                queue = [g]
                while queue:
                    curr = queue.pop(0)
                    if curr in boxes:
                        reachable_boxes.add(curr)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr[0] + dr, curr[1] + dc
                        if (nr, nc) in open_cells and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

        return len(reachable_boxes) < len(goal_plates)


# =====================================================================
# 4. Kinematic Transition Model
# =====================================================================

@dataclass
class KinematicTransitionModel:
    """General offline discrete simulator for ARC-AGI-3 grid physics."""
    walls: Set[Tuple[int, int]] = field(default_factory=set)
    goal_plates: Set[Tuple[int, int]] = field(default_factory=set)
    exit_doors: Set[Tuple[int, int]] = field(default_factory=set)
    collectibles: Set[Tuple[int, int]] = field(default_factory=set)
    switches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = field(default_factory=dict)
    doors: Set[Tuple[int, int]] = field(default_factory=set)
    static_hazards: Set[Tuple[int, int]] = field(default_factory=set)
    dynamic_hazard_trajectories: List[List[Tuple[int, int]]] = field(default_factory=list)
    grid_h: int = 64
    grid_w: int = 64
    scale: int = 1
    max_step_budget: int = 500
    is_ice_sliding: bool = False
    displacement_vectors: Dict[int, Tuple[int, int]] = field(default_factory=lambda: dict(ACTION_DELTAS))
    affordances: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_world_dict(cls, world: Dict[str, Any], max_step_budget: int = 500) -> KinematicTransitionModel:
        """Constructs a KinematicTransitionModel from extract_discrete_board dictionary."""
        return cls(
            walls=set(world.get("walls", set())),
            goal_plates=set(world.get("plates", set())),
            exit_doors=set(world.get("exit_doors", set())),
            collectibles=set(world.get("collectibles", set())),
            switches=dict(world.get("switches", {})),
            doors=set(world.get("doors", set())),
            static_hazards=set(world.get("hazards", set())),
            grid_h=int(world.get("gh", 64)),
            grid_w=int(world.get("gw", 64)),
            scale=int(world.get("scale", 1)),
            max_step_budget=max_step_budget,
            is_ice_sliding=bool(world.get("is_sliding", False)),
            displacement_vectors=dict(world.get("displacement_vectors", ACTION_DELTAS)),
        )

    def is_goal_satisfied(self, state: SimulatedState) -> bool:
        """Evaluates invariant goal satisfaction."""
        if self.goal_plates and not self.goal_plates.issubset(state.boxes):
            return False
        if self.collectibles and not self.collectibles.issubset(state.collected_items):
            return False
        if self.exit_doors and state.player_pos not in self.exit_doors:
            return False
        return True

    def simulate_step(
        self,
        state: SimulatedState,
        action: int,
    ) -> Tuple[Optional[SimulatedState], str, bool]:
        """Performs forward transition rollout s_{t+1} = T(s_t, a_t).
        Returns (next_state, outcome_status, is_goal_satisfied).
        """
        if state.step_count >= self.max_step_budget:
            return None, "BUDGET_EXHAUSTED", False

        if action not in self.displacement_vectors:
            return None, "UNKNOWN_ACTION", False

        dr, dc = self.displacement_vectors[action]
        curr_r, curr_c = state.player_pos
        next_r, next_c = curr_r + dr, curr_c + dc

        # Check arena boundaries
        if next_r < 0 or next_r >= self.grid_h or next_c < 0 or next_c >= self.grid_w:
            return None, "BLOCKED_BOUNDARY", False

        # Check static walls
        if (next_r, next_c) in self.walls:
            return None, "BLOCKED_WALL", False

        # Check closed doors
        if (next_r, next_c) in self.doors and (next_r, next_c) not in state.open_doors:
            return None, "BLOCKED_DOOR", False

        boxes = set(state.boxes)
        open_doors = set(state.open_doors)
        active_switches = set(state.active_switches)
        collected_items = set(state.collected_items)

        # Sokoban push dynamics
        if (next_r, next_c) in boxes:
            box_dest_r = next_r + dr
            box_dest_c = next_c + dc

            if self.is_ice_sliding:
                # Frictionless ice slide until collision
                while (
                    0 <= box_dest_r + dr < self.grid_h
                    and 0 <= box_dest_c + dc < self.grid_w
                    and (box_dest_r + dr, box_dest_c + dc) not in self.walls
                    and (box_dest_r + dr, box_dest_c + dc) not in boxes
                    and not ((box_dest_r + dr, box_dest_c + dc) in self.doors and (box_dest_r + dr, box_dest_c + dc) not in open_doors)
                ):
                    box_dest_r += dr
                    box_dest_c += dc

            if (
                box_dest_r < 0
                or box_dest_r >= self.grid_h
                or box_dest_c < 0
                or box_dest_c >= self.grid_w
                or (box_dest_r, box_dest_c) in self.walls
                or (box_dest_r, box_dest_c) in boxes
                or ((box_dest_r, box_dest_c) in self.doors and (box_dest_r, box_dest_c) not in open_doors)
            ):
                return None, "BLOCKED_BOX_PUSH", False

            boxes.remove((next_r, next_c))
            boxes.add((box_dest_r, box_dest_c))

            # Irreversible deadlock verification
            f_boxes = frozenset(boxes)
            if DeadlockDetector.is_concave_corner((box_dest_r, box_dest_c), self.walls, self.goal_plates, self.grid_h, self.grid_w):
                return None, "SOKOBAN_CORNER_DEADLOCK", False
            if DeadlockDetector.is_continuous_wall_deadlock((box_dest_r, box_dest_c), self.walls, self.goal_plates, self.grid_h, self.grid_w):
                return None, "SOKOBAN_WALL_DEADLOCK", False
            if DeadlockDetector.is_2x2_deadlock(f_boxes, self.walls, self.goal_plates, self.grid_h, self.grid_w):
                return None, "SOKOBAN_2X2_DEADLOCK", False

            # Switch triggered by pushed box
            if (box_dest_r, box_dest_c) in self.switches:
                active_switches.add((box_dest_r, box_dest_c))
                for linked_door in self.switches[(box_dest_r, box_dest_c)]:
                    open_doors.add(linked_door)

        player_next_pos = (next_r, next_c)

        # Check static hazards
        if player_next_pos in self.static_hazards:
            return None, "LETHAL_STATIC_HAZARD", False

        # Check dynamic hazards
        next_hazard_step = state.hazard_step + 1
        for traj in self.dynamic_hazard_trajectories:
            if not traj:
                continue
            curr_h = traj[state.hazard_step % len(traj)]
            next_h = traj[next_hazard_step % len(traj)]
            # Direct collision or swap collision
            if player_next_pos == next_h or (player_next_pos == curr_h and state.player_pos == next_h):
                return None, "LETHAL_DYNAMIC_HAZARD", False

        # Collectible acquisition
        if player_next_pos in self.collectibles:
            collected_items.add(player_next_pos)

        # Switch triggered by player
        if player_next_pos in self.switches:
            active_switches.add(player_next_pos)
            for linked_door in self.switches[player_next_pos]:
                open_doors.add(linked_door)

        next_state = SimulatedState(
            player_pos=player_next_pos,
            boxes=frozenset(boxes),
            active_switches=frozenset(active_switches),
            open_doors=frozenset(open_doors),
            collected_items=frozenset(collected_items),
            hazard_step=next_hazard_step,
            step_count=state.step_count + 1,
        )
        goal_reached = self.is_goal_satisfied(next_state)
        return next_state, "SUCCESS", goal_reached


# =====================================================================
# 5. Cheap Hypothesis Testing & Epistemic Prober
# =====================================================================

@dataclass
class EpistemicProbePlan:
    """Action plan for a minimal-cost (<=2 actions) reversible epistemic probe."""
    approach_actions: List[int]
    probe_action: int
    reverse_action: int
    target_pos: Tuple[int, int]
    vantage_pos: Tuple[int, int]


@dataclass
class ProbeOutcome:
    """Empirical affordance classification resulting from epistemic probing."""
    affordance: str  # 'WALL', 'PUSHABLE_BOX', 'SWITCH', 'COLLECTIBLE', 'HAZARD', 'PASSABLE_FLOOR'
    is_reversible: bool
    is_terminal: bool
    delta_pixel_count: int
    details: Dict[str, Any] = field(default_factory=dict)


class EpistemicProber:
    """Constructs shortest reversible probes to induce affordances at minimal action cost."""

    @staticmethod
    def plan_probe(
        player_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        gh: int,
        gw: int,
    ) -> Optional[EpistemicProbePlan]:
        """Finds closest safe vantage adjacent to target and plans 1-step probe + reverse."""
        target_r, target_c = target_pos

        best_vantage: Optional[Tuple[int, int]] = None
        best_probe_action: Optional[int] = None
        min_dist = 1000000

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            vr, vc = target_r + dr, target_c + dc
            if 0 <= vr < gh and 0 <= vc < gw and (vr, vc) not in walls:
                dist = abs(vr - player_pos[0]) + abs(vc - player_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_vantage = (vr, vc)
                    probe_dr, probe_dc = target_r - vr, target_c - vc
                    best_probe_action = DELTA_TO_ACTION.get((probe_dr, probe_dc))

        if best_vantage is None or best_probe_action is None:
            return None

        # Shortest path from player_pos to best_vantage
        approach_actions: List[int] = []
        if player_pos != best_vantage:
            queue = [(player_pos, [])]
            visited = {player_pos}
            found = False
            while queue:
                curr, path = queue.pop(0)
                if curr == best_vantage:
                    approach_actions = path
                    found = True
                    break
                for act, (adr, adc) in ACTION_DELTAS.items():
                    nr, nc = curr[0] + adr, curr[1] + adc
                    if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in walls and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))
            if not found:
                return None

        reverse_act = OPPOSITE_ACTIONS.get(best_probe_action, SimulatedAction.UNDO)

        return EpistemicProbePlan(
            approach_actions=approach_actions,
            probe_action=best_probe_action,
            reverse_action=reverse_act,
            target_pos=target_pos,
            vantage_pos=best_vantage,
        )

    @staticmethod
    def evaluate_probe_feedback(
        pre_grid: np.ndarray,
        probe_grid: np.ndarray,
        post_grid: Optional[np.ndarray],
        player_pos_pre: Tuple[int, int],
        player_pos_probe: Optional[Tuple[int, int]],
        target_pos: Tuple[int, int],
        probe_action: int,
        is_terminal: bool = False,
    ) -> ProbeOutcome:
        """Classifies affordance from live frame deltas."""
        if is_terminal:
            return ProbeOutcome(
                affordance="HAZARD",
                is_reversible=False,
                is_terminal=True,
                delta_pixel_count=int(np.count_nonzero(probe_grid != pre_grid)),
            )

        diff = int(np.count_nonzero(probe_grid != pre_grid))
        player_moved = player_pos_probe is not None and player_pos_probe != player_pos_pre

        if not player_moved and diff == 0:
            return ProbeOutcome(
                affordance="WALL",
                is_reversible=True,
                is_terminal=False,
                delta_pixel_count=0,
            )

        # Check Sokoban push displacement
        dr, dc = ACTION_DELTAS.get(probe_action, (0, 0))
        box_dest = (target_pos[0] + dr, target_pos[1] + dc)
        if player_moved and player_pos_probe == target_pos:
            target_color_pre = pre_grid[target_pos[0], target_pos[1]]
            if (
                0 <= box_dest[0] < probe_grid.shape[0]
                and 0 <= box_dest[1] < probe_grid.shape[1]
                and probe_grid[box_dest[0], box_dest[1]] == target_color_pre
            ):
                return ProbeOutcome(
                    affordance="PUSHABLE_BOX",
                    is_reversible=True,
                    is_terminal=False,
                    delta_pixel_count=diff,
                    details={"pushed_to": box_dest},
                )

            return ProbeOutcome(
                affordance="COLLECTIBLE_OR_SWITCH",
                is_reversible=True,
                is_terminal=False,
                delta_pixel_count=diff,
            )

        return ProbeOutcome(
            affordance="PASSABLE_FLOOR",
            is_reversible=True,
            is_terminal=False,
            delta_pixel_count=diff,
        )


# =====================================================================
# 6. Verification Delta & Fast Rollback Controller
# =====================================================================

@dataclass
class VerificationDeltaReport:
    """Outcome of live observation vs simulated expectation comparison."""
    is_diverged: bool
    divergence_reason: str = ""
    expected_player_pos: Optional[Tuple[int, int]] = None
    actual_player_pos: Optional[Tuple[int, int]] = None
    expected_hash: Optional[int] = None
    actual_hash: Optional[int] = None
    suggested_patch: Dict[str, Any] = field(default_factory=dict)


class VerificationDeltaController:
    """Protects against quadratic RHAE penalties by catching divergences early."""

    @staticmethod
    def compute_context_hash(grid: np.ndarray, border_margin: int = 4) -> int:
        """Computes fast CRC32 hash of interior cells (excluding border timer bar)."""
        h, w = grid.shape[:2]
        bm = border_margin
        interior = grid[bm : h - bm, bm : w - bm] if h > 2 * bm and w > 2 * bm else grid
        return binascii.crc32(interior.tobytes()) & 0xFFFFFFFF

    @classmethod
    def verify_step(
        cls,
        actual_grid: np.ndarray,
        actual_player_pos: Optional[Tuple[int, int]],
        expected_player_pos: Optional[Tuple[int, int]],
        expected_hash: Optional[int],
        border_margin: int = 4,
    ) -> VerificationDeltaReport:
        """Compares live observation against expected state."""
        # 1. Player position check
        if expected_player_pos is not None and actual_player_pos is not None:
            if actual_player_pos != expected_player_pos:
                return VerificationDeltaReport(
                    is_diverged=True,
                    divergence_reason=f"Player pos diverged: expected {expected_player_pos}, actual {actual_player_pos}",
                    expected_player_pos=expected_player_pos,
                    actual_player_pos=actual_player_pos,
                    suggested_patch={"blocked_pos": expected_player_pos},
                )

        # 2. State hash check (timer excluded)
        if expected_hash is not None:
            actual_h = cls.compute_context_hash(actual_grid, border_margin=border_margin)
            if actual_h != expected_hash:
                return VerificationDeltaReport(
                    is_diverged=True,
                    divergence_reason=f"Topological hash diverged: expected {expected_hash}, actual {actual_h}",
                    expected_hash=expected_hash,
                    actual_hash=actual_h,
                )

        return VerificationDeltaReport(is_diverged=False)


# =====================================================================
# 7. Simulated Model Planner (Master Orchestrator)
# =====================================================================

class SimulatedModelPlanner:
    """Master offline simulator and A* / Macro-Queue Planner for ARC-AGI-3."""

    def __init__(self, border_margin: int = 4) -> None:
        self.border_margin = border_margin
        self.prober = EpistemicProber()
        self.verifier = VerificationDeltaController()

    @staticmethod
    def compute_admissible_heuristic(state: SimulatedState, model: KinematicTransitionModel) -> float:
        """Admissible and consistent lower-bound heuristic for A* search."""
        h = 0.0
        # 1. Sokoban boxes to goal plates
        if model.goal_plates:
            unplaced = [b for b in state.boxes if b not in model.goal_plates]
            unfilled = [g for g in model.goal_plates if g not in state.boxes]
            if unplaced and unfilled:
                box_dist = 0.0
                for b in unplaced:
                    min_d = min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in unfilled)
                    box_dist += min_d
                h += box_dist
                # Player to closest unplaced box
                p_to_box = min(abs(state.player_pos[0] - b[0]) + abs(state.player_pos[1] - b[1]) for b in unplaced)
                h += max(0, p_to_box - 1)

        # 2. Collectibles
        if model.collectibles:
            rem_coll = [c for c in model.collectibles if c not in state.collected_items]
            if rem_coll:
                h += min(abs(state.player_pos[0] - c[0]) + abs(state.player_pos[1] - c[1]) for c in rem_coll)

        # 3. Exit doors
        if model.exit_doors:
            h += min(abs(state.player_pos[0] - d[0]) + abs(state.player_pos[1] - d[1]) for d in model.exit_doors)

        return float(h)

    def plan_offline(
        self,
        init_state: SimulatedState,
        model: KinematicTransitionModel,
        max_expansions: int = 8000,
        time_limit: float = 3.0,
    ) -> Optional[List[int]]:
        """Simulates candidate action sequences offline and finds geodesic path to goal."""
        if model.is_goal_satisfied(init_state):
            return []

        start_time = time.perf_counter()
        pq: List[Tuple[float, int, int, SimulatedState, List[int]]] = []
        init_h = self.compute_admissible_heuristic(init_state, model)
        heapq.heappush(pq, (init_h, 0, id(init_state), init_state, []))

        visited: Dict[Tuple[Any, ...], int] = {init_state.key: 0}
        expansions = 0

        while pq and expansions < max_expansions:
            if (expansions % 500 == 0) and (time.perf_counter() - start_time > time_limit):
                break

            f, g, _, state, path = heapq.heappop(pq)
            expansions += 1

            for act in sorted(model.displacement_vectors.keys()):
                nxt_state, status, is_goal = model.simulate_step(state, act)
                if nxt_state is None:
                    continue

                if is_goal:
                    return path + [act]

                nxt_g = g + 1
                state_key = nxt_state.key
                if state_key in visited and visited[state_key] <= nxt_g:
                    continue

                visited[state_key] = nxt_g
                nxt_f = nxt_g + self.compute_admissible_heuristic(nxt_state, model)
                heapq.heappush(pq, (nxt_f, nxt_g, id(nxt_state), nxt_state, path + [act]))

        return None

    def populate_macro_queue(
        self,
        actions: List[int],
        init_state: SimulatedState,
        model: KinematicTransitionModel,
        macro_queue: Any,
    ) -> None:
        """Rolls out planned actions forward and batches them directly into MacroActionQueue."""
        curr = init_state
        for act in actions:
            nxt, status, is_goal = model.simulate_step(curr, act)
            if nxt is None:
                break
            macro_queue.add_step(
                action=act,
                expected_player_pos=nxt.player_pos,
                description=f"Simulated step to {nxt.player_pos}",
            )
            curr = nxt
