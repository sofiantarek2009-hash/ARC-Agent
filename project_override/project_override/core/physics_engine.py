"""
physics_engine.py - Dynamic Action Physics Engine for ARC-AGI-3 (Project Override)

Provides discrete grid kinematics and physics simulation:
- 64x64 canvas updates with integer grid representations.
- Directional action shifts (ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right)
  with static boundary collision detection and mass-proportional pushing (Sokoban dynamics).
- ACTION5 (Interact/Toggle/Collect) for keys, items, levers, and doors.
- ACTION6 (Parameterized coordinate targeting): Deterministic intercept geometry
  for static and dynamic moving target entities.
- ACTION7 (Undo): Fully reversible mechanics and history state unwinding stack.
- RESET: Root restoration to initial checkpoint.
- Dual execution backend: High-performance C++ core kinematics binding (via ctypes)
  with seamless pure Python/NumPy fallback.
"""

from __future__ import annotations

import copy
import ctypes
import math
import os
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


# =============================================================================
# Action and Entity Enums
# =============================================================================

class Action(IntEnum):
    NONE = 0
    ACTION1_UP = 1
    ACTION2_DOWN = 2
    ACTION3_LEFT = 3
    ACTION4_RIGHT = 4
    ACTION5_INTERACT = 5
    ACTION6_TARGET = 6
    ACTION7_UNDO = 7
    RESET = 8

    @classmethod
    def from_value(cls, val: Union[Action, int]) -> Action:
        if isinstance(val, Action):
            return val
        try:
            return cls(val)
        except ValueError:
            raise ValueError(f"Unknown action code: {val}")


class EntityType(IntEnum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    MOVABLE = 3  # Pushable Sokoban box/block
    BOX = 3      # Alias for MOVABLE
    TARGET = 4   # Goal or dynamic target entity
    COLLECTIBLE = 5  # Item, key, gem
    DOOR = 6     # Barrier opened/closed by toggles
    TOGGLE = 7   # Switch, lever, pressure plate


# Action delta lookup: (row_delta, col_delta)
DIRECTION_VECTORS: Dict[Action, Tuple[int, int]] = {
    Action.ACTION1_UP: (-1, 0),
    Action.ACTION2_DOWN: (1, 0),
    Action.ACTION3_LEFT: (0, -1),
    Action.ACTION4_RIGHT: (0, 1),
}


# =============================================================================
# Event & Snapshot Models
# =============================================================================

@dataclass
class PhysicsEvent:
    """Event triggered during a physics simulation step."""
    event_type: str  # 'move', 'push', 'collision', 'collect', 'toggle', 'intercept', 'undo', 'reset'
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Outcome of a single physics step."""
    action: Action
    success: bool
    canvas: np.ndarray  # 2D grid array (e.g. 64x64)
    player_pos: Tuple[int, int]
    player_facing: Tuple[int, int]
    score: int
    inventory: List[int]
    events: List[PhysicsEvent]
    history_depth: int
    message: str = ""


@dataclass
class InterceptResult:
    """Outcome of deterministic intercept geometry computation."""
    target_pos: Tuple[int, int]
    intercept_pos: Tuple[int, int]
    steps: int = 0
    success: bool = True


@dataclass
class StateSnapshot:
    """Immutable snapshot for history unwinding (Undo/Reset)."""
    step_count: int
    canvas: np.ndarray
    player_id: int
    player_pos: Tuple[int, int]
    player_facing: Tuple[int, int]
    player_push_strength: float
    score: int
    inventory: List[int]
    entities: Dict[int, Any]  # Deep-copied entity state dictionary


# =============================================================================
# Entity Class
# =============================================================================

@dataclass
class Entity:
    id: int
    type: EntityType
    pos: Tuple[int, int]  # (row, col)
    mass: float = 1.0
    color: int = 1
    is_passable: bool = False
    push_strength: float = 1.0  # Used if entity is an active pusher
    velocity: Tuple[int, int] = (0, 0)  # Dynamic target kinematics
    linked_doors: List[int] = field(default_factory=list)  # For switches/toggles
    is_active: bool = True  # For toggles/doors
    is_collected: bool = False

    def clone(self) -> Entity:
        return copy.deepcopy(self)


# =============================================================================
# Deterministic Intercept Geometry
# =============================================================================

def compute_deterministic_intercept(
    source_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    target_vel: Tuple[int, int] = (0, 0),
    speed: int = 1,
    max_steps: int = 128,
    metric: str = "manhattan",
    grid_shape: Tuple[int, int] = (64, 64),
    obstacles: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, int], int, bool]:
    """
    Compute deterministic intercept coordinates for a target entity.

    Solves for the minimal discrete time t >= 0 such that:
        dist(source_pos, Target(t)) <= speed * t
    Where Target(t) is projected kinematically along target_vel, bounded by grid_shape.

    Parameters:
    - source_pos: (r, c) of pursuer / targeting beam origin.
    - target_pos: (r, c) of target at t=0.
    - target_vel: (vr, vc) discrete velocity per step.
    - speed: cells per step pursuer can traverse.
    - max_steps: maximum time horizon.
    - metric: 'manhattan' (L1), 'chebyshev' (L_inf), 'euclidean' (L2), or 'bfs' (obstacle-aware).
    - grid_shape: (height, width).
    - obstacles: Optional 2D boolean mask where True = impassable barrier (for BFS metric).

    Returns:
    - (intercept_pos, intercept_time, success_flag)
    """
    if speed <= 0:
        speed = 1
    h, w = grid_shape
    sr, sc = source_pos
    tr0, tc0 = target_pos
    vr, vc = target_vel

    # Precompute BFS distances if obstacles are provided and BFS is selected
    bfs_dist_map: Optional[np.ndarray] = None
    if metric == "bfs" and obstacles is not None:
        bfs_dist_map = np.full((h, w), -1, dtype=np.int32)
        queue = deque([(sr, sc, 0)])
        bfs_dist_map[sr, sc] = 0
        while queue:
            cr, cc, d = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not obstacles[nr, nc] and bfs_dist_map[nr, nc] == -1:
                    bfs_dist_map[nr, nc] = d + 1
                    queue.append((nr, nc, d + 1))

    # Kinematic trajectory simulation
    cur_tr, cur_tc = tr0, tc0
    for t in range(max_steps + 1):
        # Bound target position inside grid (clamped/reflective)
        if t > 0:
            next_tr = cur_tr + vr
            next_tc = cur_tc + vc
            # Reflective bounce against boundaries
            if next_tr < 0:
                next_tr = -next_tr
                vr = -vr
            elif next_tr >= h:
                next_tr = 2 * (h - 1) - next_tr
                vr = -vr

            if next_tc < 0:
                next_tc = -next_tc
                vc = -vc
            elif next_tc >= w:
                next_tc = 2 * (w - 1) - next_tc
                vc = -vc

            cur_tr = max(0, min(h - 1, next_tr))
            cur_tc = max(0, min(w - 1, next_tc))

        # Calculate distance from source
        dr = abs(cur_tr - sr)
        dc = abs(cur_tc - sc)

        if metric == "bfs" and bfs_dist_map is not None:
            dist = bfs_dist_map[cur_tr, cur_tc]
            if dist == -1:
                dist = 999999  # Unreachable due to obstacles
        elif metric == "chebyshev":
            dist = max(dr, dc)
        elif metric == "euclidean":
            dist = math.ceil(math.sqrt(dr * dr + dc * dc))
        else:  # default 'manhattan'
            dist = dr + dc

        if dist <= speed * t:
            return (cur_tr, cur_tc), t, True

    # Fallback to last known position
    return (cur_tr, cur_tc), max_steps, False


# =============================================================================
# C++ Core Kinematics ctypes Bridge
# =============================================================================

class CPPKinematicsCore:
    """Optional high-performance C++ backend bridge."""

    _lib: Optional[ctypes.CDLL] = None
    _loaded: bool = False

    @classmethod
    def get_lib(cls) -> Optional[ctypes.CDLL]:
        if not cls._loaded:
            cls.load_library()
        return cls._lib

    @classmethod
    def is_available(cls) -> bool:
        return cls.get_lib() is not None

    @classmethod
    def load_library(cls) -> bool:
        if cls._loaded:
            return cls._lib is not None
        cls._loaded = True

        candidates = [
            Path(__file__).parent / "libkinematics.dylib",
            Path(__file__).parent / "libkinematics.so",
            Path("/Applications/Dad Company/project_override/project_override/core/libkinematics.dylib"),
        ]

        for p in candidates:
            if p.exists():
                try:
                    lib = ctypes.CDLL(str(p))
                    # Setup function signatures
                    lib.create_kinematics_engine.argtypes = [ctypes.c_int, ctypes.c_int]
                    lib.create_kinematics_engine.restype = ctypes.c_void_p

                    lib.destroy_kinematics_engine.argtypes = [ctypes.c_void_p]
                    lib.destroy_kinematics_engine.restype = None

                    lib.set_grid_cell.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                    lib.set_grid_cell.restype = None

                    lib.get_grid_cell.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                    lib.get_grid_cell.restype = ctypes.c_int

                    lib.get_cell_mass.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                    lib.get_cell_mass.restype = ctypes.c_float

                    lib.get_grid_canvas.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
                    lib.get_grid_canvas.restype = None

                    lib.set_player_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                    lib.set_player_state.restype = None

                    lib.step_action.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
                    lib.step_action.restype = ctypes.c_int

                    lib.undo_step.argtypes = [ctypes.c_void_p]
                    lib.undo_step.restype = ctypes.c_int

                    lib.reset_to_root.argtypes = [ctypes.c_void_p]
                    lib.reset_to_root.restype = None

                    lib.commit_root_checkpoint.argtypes = [ctypes.c_void_p]
                    lib.commit_root_checkpoint.restype = None

                    lib.get_history_depth.argtypes = [ctypes.c_void_p]
                    lib.get_history_depth.restype = ctypes.c_int

                    lib.compute_deterministic_intercept_c.argtypes = [
                        ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(ctypes.c_int),
                    ]
                    lib.compute_deterministic_intercept_c.restype = ctypes.c_int

                    cls._lib = lib
                    return True
                except Exception:
                    pass
        return False


# =============================================================================
# Discrete Grid Physics Engine (Python Core + Kinematics)
# =============================================================================

class DiscretePhysicsEngine:
    """
    Complete Discrete Grid Simulation Engine for ARC-AGI-3.
    """

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        use_cpp_backend: bool = True,
    ):
        self.width = width
        self.height = height
        self.step_count = 0
        self.score = 0
        self.inventory: List[int] = []

        # Entities table
        self._next_entity_id = 1
        self.entities: Dict[int, Entity] = {}
        self.player_id: Optional[int] = None
        self.player_facing: Tuple[int, int] = (-1, 0)  # Default facing UP

        # Grid canvas: 2D numpy array
        self.canvas: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # History unwinding stack for ACTION7 (Undo) and RESET
        self._history_stack: List[StateSnapshot] = []
        self._root_snapshot: Optional[StateSnapshot] = None

        # C++ acceleration handle
        self._cpp_available = False
        self._cpp_handle = None
        if use_cpp_backend and CPPKinematicsCore.load_library():
            self._cpp_available = True
            self._cpp_handle = CPPKinematicsCore._lib.create_kinematics_engine(self.width, self.height)

        # Initialize default player at center
        self.spawn_player(row=self.height // 2, col=self.width // 2, push_strength=1.0)
        self.checkpoint_root()

    def __del__(self):
        if self._cpp_available and self._cpp_handle is not None and CPPKinematicsCore._lib is not None:
            try:
                CPPKinematicsCore._lib.destroy_kinematics_engine(self._cpp_handle)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Initialization & Spawning
    # -------------------------------------------------------------------------

    def spawn_entity(
        self,
        entity_type: EntityType,
        pos: Tuple[int, int],
        mass: float = 1.0,
        color: int = 1,
        is_passable: bool = False,
        push_strength: float = 1.0,
        velocity: Tuple[int, int] = (0, 0),
        linked_doors: Optional[List[int]] = None,
        is_active: Optional[bool] = None,
    ) -> Entity:
        """Spawn an entity into the grid simulation."""
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"Spawn coordinate ({r}, {c}) is outside bounds {self.height}x{self.width}")

        eid = self._next_entity_id
        self._next_entity_id += 1

        if is_active is None:
            is_active = (entity_type != EntityType.TOGGLE)

        entity = Entity(
            id=eid,
            type=entity_type,
            pos=(r, c),
            mass=mass,
            color=color,
            is_passable=is_passable,
            push_strength=push_strength,
            velocity=velocity,
            linked_doors=linked_doors or [],
            is_active=is_active,
        )
        self.entities[eid] = entity
        self._sync_canvas_cell(r, c)
        return entity

    def spawn_player(self, row: int, col: int, push_strength: float = 1.0) -> Entity:
        """Spawn or reposition the controllable player."""
        if self.player_id is not None and self.player_id in self.entities:
            old_r, old_c = self.entities[self.player_id].pos
            del self.entities[self.player_id]
            self._sync_canvas_cell(old_r, old_c)

        player = self.spawn_entity(
            entity_type=EntityType.PLAYER,
            pos=(row, col),
            mass=1.0,
            color=int(EntityType.PLAYER),
            push_strength=push_strength,
        )
        self.player_id = player.id
        return player

    @property
    def player_pos(self) -> Optional[Tuple[int, int]]:
        """Returns the current (row, col) position of the player."""
        if self.player_id is not None and self.player_id in self.entities:
            return self.entities[self.player_id].pos
        return None

    def from_grid(
        self,
        grid: np.ndarray,
        player_push_strength: float = 1.0,
        wall_color: int = 1,
        player_color: int = 2,
        box_color: int = 3,
        target_color: int = 4,
        collectible_color: int = 5,
        door_color: int = 6,
        toggle_color: int = 7,
    ) -> DiscretePhysicsEngine:
        """Initialize engine entities directly from a 2D numpy grid."""
        grid = np.asarray(grid, dtype=np.int32)
        h, w = grid.shape
        self.height = h
        self.width = w
        self.canvas = np.zeros((h, w), dtype=np.int32)
        self.entities.clear()
        self._history_stack.clear()
        self._next_entity_id = 1
        self.player_id = None

        color_map = {
            wall_color: EntityType.WALL,
            player_color: EntityType.PLAYER,
            box_color: EntityType.MOVABLE,
            target_color: EntityType.TARGET,
            collectible_color: EntityType.COLLECTIBLE,
            door_color: EntityType.DOOR,
            toggle_color: EntityType.TOGGLE,
        }

        for r in range(h):
            for c in range(w):
                val = int(grid[r, c])
                if val == 0:
                    continue
                etype = color_map.get(val, EntityType.WALL)
                if etype == EntityType.PLAYER:
                    self.spawn_player(r, c, push_strength=player_push_strength)
                elif etype == EntityType.WALL:
                    self.spawn_entity(EntityType.WALL, (r, c), mass=float("inf"), color=val, is_passable=False)
                elif etype == EntityType.MOVABLE:
                    self.spawn_entity(EntityType.MOVABLE, (r, c), mass=1.0, color=val, is_passable=False)
                elif etype == EntityType.TARGET:
                    self.spawn_entity(EntityType.TARGET, (r, c), mass=0.0, color=val, is_passable=True)
                elif etype == EntityType.COLLECTIBLE:
                    self.spawn_entity(EntityType.COLLECTIBLE, (r, c), mass=0.0, color=val, is_passable=True)
                elif etype == EntityType.DOOR:
                    self.spawn_entity(EntityType.DOOR, (r, c), mass=float("inf"), color=val, is_passable=False)
                elif etype == EntityType.TOGGLE:
                    self.spawn_entity(EntityType.TOGGLE, (r, c), mass=0.0, color=val, is_passable=True)

        self.render()
        self.checkpoint_root()
        return self

    # -------------------------------------------------------------------------
    # Canvas Synchronization & Rendering
    # -------------------------------------------------------------------------

    def _sync_canvas_cell(self, r: int, c: int) -> None:
        """Update single cell in the canvas array based on occupying entities."""
        occupants = [e for e in self.entities.values() if e.pos == (r, c) and not e.is_collected]
        if not occupants:
            self.canvas[r, c] = 0
            if self._cpp_available and self._cpp_handle:
                CPPKinematicsCore._lib.set_grid_cell(self._cpp_handle, r, c, 0, 0.0)
            return

        # Prioritize solid/player over background interactables
        occupants.sort(key=lambda e: (
            0 if e.type == EntityType.PLAYER else
            1 if e.type == EntityType.MOVABLE else
            2 if e.type == EntityType.DOOR and not e.is_passable else
            3 if e.type == EntityType.WALL else
            4
        ))
        top = occupants[0]
        self.canvas[r, c] = top.color
        if self._cpp_available and self._cpp_handle:
            CPPKinematicsCore._lib.set_grid_cell(self._cpp_handle, r, c, int(top.type), float(top.mass))

    def render(self) -> np.ndarray:
        """Render the current 2D canvas of size (H, W)."""
        self.canvas.fill(0)
        for e in self.entities.values():
            if not e.is_collected:
                r, c = e.pos
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.canvas[r, c] = e.color
        return self.canvas.copy()

    def get_player(self) -> Optional[Entity]:
        if self.player_id is not None:
            return self.entities.get(self.player_id)
        return None

    def get_entity_at(self, pos: Tuple[int, int]) -> Optional[Entity]:
        for e in self.entities.values():
            if e.pos == pos and not e.is_collected:
                return e
        return None

    def get_all_entities_at(self, pos: Tuple[int, int]) -> List[Entity]:
        return [e for e in self.entities.values() if e.pos == pos and not e.is_collected]

    # -------------------------------------------------------------------------
    # Reversible Mechanics & History (Undo / Reset)
    # -------------------------------------------------------------------------

    def _create_snapshot(self) -> StateSnapshot:
        """Create deep snapshot of engine state."""
        player = self.get_player()
        pos = player.pos if player else (0, 0)
        strength = player.push_strength if player else 1.0

        entities_copy = {eid: e.clone() for eid, e in self.entities.items()}
        return StateSnapshot(
            step_count=self.step_count,
            canvas=self.canvas.copy(),
            player_id=self.player_id if self.player_id is not None else 0,
            player_pos=pos,
            player_facing=self.player_facing,
            player_push_strength=strength,
            score=self.score,
            inventory=list(self.inventory),
            entities=entities_copy,
        )

    def _restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore complete state from snapshot."""
        self.step_count = snapshot.step_count
        self.canvas = snapshot.canvas.copy()
        self.player_id = snapshot.player_id
        self.player_facing = snapshot.player_facing
        self.score = snapshot.score
        self.inventory = list(snapshot.inventory)

        # Remove entities that were not present in snapshot
        current_keys = set(self.entities.keys())
        snapshot_keys = set(snapshot.entities.keys())
        for removed_id in (current_keys - snapshot_keys):
            del self.entities[removed_id]

        # Update in-place to preserve object references
        for eid, snap_e in snapshot.entities.items():
            if eid in self.entities:
                target = self.entities[eid]
                target.type = snap_e.type
                target.pos = snap_e.pos
                target.mass = snap_e.mass
                target.color = snap_e.color
                target.is_passable = snap_e.is_passable
                target.push_strength = snap_e.push_strength
                target.velocity = snap_e.velocity
                target.linked_doors = list(snap_e.linked_doors)
                target.is_active = snap_e.is_active
                target.is_collected = snap_e.is_collected
            else:
                self.entities[eid] = snap_e.clone()
        self.render()

    def checkpoint_root(self) -> None:
        """Establish root baseline state for RESET actions."""
        self._root_snapshot = self._create_snapshot()
        if self._cpp_available and self._cpp_handle:
            CPPKinematicsCore._lib.commit_root_checkpoint(self._cpp_handle)

    def undo(self) -> StepResult:
        """
        ACTION7: Unwind one step in history stack.
        Returns StepResult with success=True if unwound, False if at root.
        """
        player = self.get_player()
        cur_pos = player.pos if player else (0, 0)

        if not self._history_stack:
            return StepResult(
                action=Action.ACTION7_UNDO,
                success=False,
                canvas=self.render(),
                player_pos=cur_pos,
                player_facing=self.player_facing,
                score=self.score,
                inventory=list(self.inventory),
                events=[],
                history_depth=0,
                message="Cannot undo: Already at root state.",
            )

        prev_state = self._history_stack.pop()
        self._restore_snapshot(prev_state)
        if self._cpp_available and self._cpp_handle:
            CPPKinematicsCore._lib.undo_step(self._cpp_handle)

        player = self.get_player()
        new_pos = player.pos if player else (0, 0)
        return StepResult(
            action=Action.ACTION7_UNDO,
            success=True,
            canvas=self.render(),
            player_pos=new_pos,
            player_facing=self.player_facing,
            score=self.score,
            inventory=list(self.inventory),
            events=[PhysicsEvent("undo", {"restored_step": self.step_count})],
            history_depth=len(self._history_stack),
            message="Successfully unwound state to previous step.",
        )

    def reset(self) -> StepResult:
        """
        RESET: Restore root baseline checkpoint.
        """
        if self._root_snapshot is None:
            self.checkpoint_root()

        assert self._root_snapshot is not None
        self._restore_snapshot(self._root_snapshot)
        self._history_stack.clear()
        if self._cpp_available and self._cpp_handle:
            CPPKinematicsCore._lib.reset_to_root(self._cpp_handle)

        player = self.get_player()
        pos = player.pos if player else (0, 0)
        return StepResult(
            action=Action.RESET,
            success=True,
            canvas=self.render(),
            player_pos=pos,
            player_facing=self.player_facing,
            score=self.score,
            inventory=list(self.inventory),
            events=[PhysicsEvent("reset", {})],
            history_depth=0,
            message="Engine restored to root baseline state.",
        )

    # -------------------------------------------------------------------------
    # Physics Action Dispatcher
    # -------------------------------------------------------------------------

    def step(
        self,
        action: Union[Action, int],
        target_coord: Optional[Tuple[int, int]] = None,
        target_entity_id: Optional[int] = None,
    ) -> StepResult:
        """
        Execute an action in the discrete simulation.

        Actions:
        - ACTION1=Up, 2=Down, 3=Left, 4=Right: Directional moves with Sokoban pushing.
        - ACTION5: Interact / Toggle / Collect.
        - ACTION6: Parameterized coordinate targeting with deterministic intercept geometry.
        - ACTION7: Undo.
        - RESET: Root restoration.
        """
        action = Action.from_value(action)

        if action == Action.ACTION7_UNDO:
            return self.undo()
        if action == Action.RESET:
            return self.reset()

        player = self.get_player()
        if player is None:
            return StepResult(
                action=action,
                success=False,
                canvas=self.render(),
                player_pos=(0, 0),
                player_facing=self.player_facing,
                score=self.score,
                inventory=list(self.inventory),
                events=[],
                history_depth=len(self._history_stack),
                message="No player entity found in simulation.",
            )

        # Record pre-action snapshot onto history stack
        snapshot = self._create_snapshot()

        events: List[PhysicsEvent] = []
        success = False

        if action in DIRECTION_VECTORS:
            dr, dc = DIRECTION_VECTORS[action]
            self.player_facing = (dr, dc)
            success = self._step_move(player, dr, dc, events)
        elif action == Action.ACTION5_INTERACT:
            success = self._step_interact(player, events)
        elif action == Action.ACTION6_TARGET:
            success = self._step_target(player, target_coord, target_entity_id, events)

        if success:
            self._history_stack.append(snapshot)
            self.step_count += 1
            self._step_dynamic_entities(events)
        else:
            # Action failed due to collision / invalid target - don't pollute undo stack
            pass

        self.render()
        return StepResult(
            action=action,
            success=success,
            canvas=self.canvas.copy(),
            player_pos=player.pos,
            player_facing=self.player_facing,
            score=self.score,
            inventory=list(self.inventory),
            events=events,
            history_depth=len(self._history_stack),
            message="Step completed." if success else "Action blocked by collision or invalid target.",
        )

    # -------------------------------------------------------------------------
    # Kinematic Movement & Mass-Proportional Pushing (Sokoban)
    # -------------------------------------------------------------------------

    def _step_move(
        self,
        player: Entity,
        dr: int,
        dc: int,
        events: List[PhysicsEvent],
    ) -> bool:
        """
        Execute directional movement with collision detection against static boundaries
        and mass-proportional Sokoban pushing.
        """
        pr, pc = player.pos
        nr, nc = pr + dr, pc + dc

        # 1. Boundary check
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            events.append(PhysicsEvent("collision", {"reason": "out_of_bounds", "target": (nr, nc)}))
            return False

        # 2. Check entities at target cell
        occupants = self.get_all_entities_at((nr, nc))

        # Check for static impassable obstacles (Walls, Closed Doors)
        for occ in occupants:
            if occ.type == EntityType.WALL or (occ.type == EntityType.DOOR and not occ.is_passable):
                events.append(PhysicsEvent("collision", {"reason": "static_obstacle", "entity_id": occ.id, "type": occ.type.name}))
                return False

        # Check for movable entities (Sokoban chain push)
        movable_chain: List[Entity] = []
        cur_r, cur_c = nr, nc
        while True:
            cell_entities = self.get_all_entities_at((cur_r, cur_c))
            movables = [e for e in cell_entities if e.type == EntityType.MOVABLE]
            if not movables:
                break
            movable_chain.append(movables[0])
            cur_r += dr
            cur_c += dc

        if movable_chain:
            # Check cell immediately after the chain
            dest_r, dest_c = cur_r, cur_c
            if not (0 <= dest_r < self.height and 0 <= dest_c < self.width):
                events.append(PhysicsEvent("collision", {"reason": "push_out_of_bounds", "chain_length": len(movable_chain)}))
                return False

            dest_occupants = self.get_all_entities_at((dest_r, dest_c))
            for occ in dest_occupants:
                if occ.type in (EntityType.WALL, EntityType.MOVABLE) or (occ.type == EntityType.DOOR and not occ.is_passable):
                    events.append(PhysicsEvent("collision", {"reason": "push_blocked_by_obstacle", "blocked_by": occ.id}))
                    return False

            # Mass calculation: check if total resistance exceeds pusher strength
            total_mass = sum(e.mass for e in movable_chain)
            if total_mass > player.push_strength:
                events.append(PhysicsEvent("collision", {
                    "reason": "insufficient_push_strength",
                    "total_mass": total_mass,
                    "pusher_strength": player.push_strength,
                }))
                return False

            # Execute chain push from back to front
            for b in reversed(movable_chain):
                old_pos = b.pos
                new_pos = (b.pos[0] + dr, b.pos[1] + dc)
                b.pos = new_pos
                self._sync_canvas_cell(old_pos[0], old_pos[1])
                self._sync_canvas_cell(new_pos[0], new_pos[1])

            events.append(PhysicsEvent("push", {
                "chain_length": len(movable_chain),
                "total_mass": total_mass,
                "direction": (dr, dc),
            }))

        # Collectible / Passable target interaction on walk-through
        for occ in occupants:
            if occ.type == EntityType.COLLECTIBLE and not occ.is_collected:
                occ.is_collected = True
                self.inventory.append(occ.color)
                self.score += 10
                events.append(PhysicsEvent("collect", {"entity_id": occ.id, "color": occ.color}))
            elif occ.type == EntityType.TARGET:
                self.score += 50
                events.append(PhysicsEvent("target_reached", {"entity_id": occ.id}))

        # Advance player
        old_player_pos = player.pos
        player.pos = (nr, nc)
        self._sync_canvas_cell(old_player_pos[0], old_player_pos[1])
        self._sync_canvas_cell(nr, nc)
        events.append(PhysicsEvent("move", {"from": old_player_pos, "to": (nr, nc)}))
        return True

    # -------------------------------------------------------------------------
    # ACTION5: Interact / Toggle / Collect
    # -------------------------------------------------------------------------

    def _step_interact(self, player: Entity, events: List[PhysicsEvent]) -> bool:
        """
        ACTION5: Interact with entities at player's current cell or facing cell.
        """
        interact_cells = [
            player.pos,  # Current cell (e.g. pressure plate or collectible)
            (player.pos[0] + self.player_facing[0], player.pos[1] + self.player_facing[1]),  # Facing cell
        ]

        interacted = False
        for pos in interact_cells:
            r, c = pos
            if not (0 <= r < self.height and 0 <= c < self.width):
                continue

            for entity in self.get_all_entities_at(pos):
                if entity.is_collected:
                    continue

                if entity.type == EntityType.COLLECTIBLE:
                    entity.is_collected = True
                    self.inventory.append(entity.color)
                    self.score += 10
                    self._sync_canvas_cell(r, c)
                    events.append(PhysicsEvent("collect", {"entity_id": entity.id, "color": entity.color}))
                    interacted = True

                elif entity.type == EntityType.TOGGLE:
                    # Flip switch state
                    entity.is_active = not entity.is_active
                    events.append(PhysicsEvent("toggle", {"entity_id": entity.id, "active": entity.is_active}))

                    # Open/close linked doors (or all doors if no explicit link)
                    target_doors = [
                        e for e in self.entities.values()
                        if e.type == EntityType.DOOR and (not entity.linked_doors or e.id in entity.linked_doors)
                    ]
                    for door in target_doors:
                        door.is_passable = entity.is_active
                        door.is_active = entity.is_active
                        door.color = 0 if door.is_passable else int(EntityType.DOOR)
                        self._sync_canvas_cell(door.pos[0], door.pos[1])
                        events.append(PhysicsEvent("door_state_change", {"door_id": door.id, "is_passable": door.is_passable}))
                    self.score += 5
                    interacted = True

                elif entity.type == EntityType.TARGET:
                    self.score += 50
                    events.append(PhysicsEvent("interact_target", {"entity_id": entity.id}))
                    interacted = True

        return interacted

    # -------------------------------------------------------------------------
    # ACTION6: Parameterized Coordinate Targeting & Intercept Geometry
    # -------------------------------------------------------------------------

    def _step_target(
        self,
        player: Entity,
        target_coord: Optional[Tuple[int, int]],
        target_entity_id: Optional[int],
        events: List[PhysicsEvent],
    ) -> bool:
        """
        ACTION6: Deterministic coordinate targeting.
        If coordinates are given, targets (r, c).
        If not given, computes deterministic intercept geometry for target entity!
        """
        coord = target_coord

        if coord is None:
            # Deterministic intercept calculation
            # Find candidate target entity
            target_entity = None
            if target_entity_id is not None:
                target_entity = self.entities.get(target_entity_id)
            else:
                targets = [e for e in self.entities.values() if e.type in (EntityType.TARGET, EntityType.MOVABLE) and not e.is_collected]
                if targets:
                    target_entity = targets[0]

            if target_entity is None:
                events.append(PhysicsEvent("targeting_failed", {"reason": "no_valid_target_found"}))
                return False

            # Compute obstacle mask (walls)
            obstacle_mask = (self.canvas == int(EntityType.WALL))
            intercept_pos, t_steps, ok = compute_deterministic_intercept(
                source_pos=player.pos,
                target_pos=target_entity.pos,
                target_vel=target_entity.velocity,
                speed=1,
                grid_shape=(self.height, self.width),
                obstacles=obstacle_mask,
            )
            coord = intercept_pos
            events.append(PhysicsEvent("intercept_computed", {
                "target_entity_id": target_entity.id,
                "intercept_coord": coord,
                "projected_steps": t_steps,
            }))

        tr, tc = coord
        if not (0 <= tr < self.height and 0 <= tc < self.width):
            events.append(PhysicsEvent("targeting_failed", {"reason": "out_of_bounds", "coord": coord}))
            return False

        # Execute targeted interaction at coord
        target_entities = self.get_all_entities_at((tr, tc))
        hit = False
        for e in target_entities:
            if e.type == EntityType.TARGET:
                self.score += 25
                events.append(PhysicsEvent("target_hit", {"entity_id": e.id, "coord": coord}))
                hit = True
            elif e.type == EntityType.COLLECTIBLE and not e.is_collected:
                e.is_collected = True
                self.inventory.append(e.color)
                self.score += 15
                self._sync_canvas_cell(tr, tc)
                events.append(PhysicsEvent("collect_targeted", {"entity_id": e.id, "coord": coord}))
                hit = True

        if not hit:
            events.append(PhysicsEvent("target_dispatched", {"coord": coord, "hit": False}))

        return True

    # -------------------------------------------------------------------------
    # Kinematic Velocity Update for Dynamic Targets
    # -------------------------------------------------------------------------

    def _step_dynamic_entities(self, events: List[PhysicsEvent]) -> None:
        """Update positions of dynamic entities with non-zero velocities."""
        for e in list(self.entities.values()):
            if e.type == EntityType.PLAYER or e.is_collected:
                continue
            vr, vc = e.velocity
            if vr == 0 and vc == 0:
                continue

            old_r, old_c = e.pos
            new_r = old_r + vr
            new_c = old_c + vc

            # Reflective bounce against boundaries
            if new_r < 0 or new_r >= self.height:
                e.velocity = (-vr, vc)
                new_r = max(0, min(self.height - 1, old_r - vr))
            if new_c < 0 or new_c >= self.width:
                e.velocity = (e.velocity[0], -vc)
                new_c = max(0, min(self.width - 1, old_c - vc))

            # Move entity
            e.pos = (new_r, new_c)
            self._sync_canvas_cell(old_r, old_c)
            self._sync_canvas_cell(new_r, new_c)
            events.append(PhysicsEvent("dynamic_entity_moved", {"entity_id": e.id, "new_pos": (new_r, new_c)}))

    # -------------------------------------------------------------------------
    # Convenience Properties & Direct Grid Cell Accessors
    # -------------------------------------------------------------------------

    @property
    def player_pos(self) -> Tuple[int, int]:
        """Current (row, col) position of the controllable player."""
        p = self.get_player()
        return p.pos if p else (0, 0)

    @player_pos.setter
    def player_pos(self, pos: Tuple[int, int]) -> None:
        p = self.get_player()
        if p:
            old_r, old_c = p.pos
            p.pos = pos
            self._sync_canvas_cell(old_r, old_c)
            self._sync_canvas_cell(pos[0], pos[1])
        else:
            self.spawn_player(pos[0], pos[1])

    @property
    def player_push_strength(self) -> float:
        """Pushing capacity / strength of the player."""
        p = self.get_player()
        return p.push_strength if p else 1.0

    @player_push_strength.setter
    def player_push_strength(self, val: float) -> None:
        p = self.get_player()
        if p:
            p.push_strength = float(val)

    def set_cell(self, r: int, c: int, val: Union[EntityType, int], mass: float = 1.0) -> None:
        """Set a cell in the grid simulation, syncing entities and canvas."""
        val = int(val)
        existing = [e for e in self.entities.values() if e.pos == (r, c)]
        for e in existing:
            if e.id == self.player_id and val != int(EntityType.PLAYER):
                continue
            del self.entities[e.id]

        if val == int(EntityType.PLAYER):
            self.spawn_player(r, c)
        elif val == int(EntityType.WALL):
            self.spawn_entity(EntityType.WALL, (r, c), mass=float("inf"), color=val, is_passable=False)
        elif val in (int(EntityType.MOVABLE), int(EntityType.BOX)):
            self.spawn_entity(EntityType.MOVABLE, (r, c), mass=mass, color=val, is_passable=False)
        elif val == int(EntityType.TARGET):
            self.spawn_entity(EntityType.TARGET, (r, c), mass=0.0, color=val, is_passable=True)
        elif val == int(EntityType.COLLECTIBLE):
            self.spawn_entity(EntityType.COLLECTIBLE, (r, c), mass=0.0, color=val, is_passable=True)
        elif val == int(EntityType.DOOR):
            self.spawn_entity(EntityType.DOOR, (r, c), mass=float("inf"), color=val, is_passable=False)
        elif val == int(EntityType.TOGGLE):
            self.spawn_entity(EntityType.TOGGLE, (r, c), mass=0.0, color=val, is_passable=True)
        elif val == int(EntityType.EMPTY):
            pass

        self._sync_canvas_cell(r, c)

    def get_cell(self, r: int, c: int) -> int:
        """Get cell value at (r, c)."""
        if 0 <= r < self.height and 0 <= c < self.width:
            return int(self.canvas[r, c])
        return int(EntityType.WALL)

    def commit_root(self) -> None:
        """Alias for checkpoint_root."""
        self.checkpoint_root()

    def compute_intercept(
        self,
        src: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_vel: Tuple[int, int] = (0, 0),
        speed: float = 1.0,
        metric: str = "manhattan",
    ) -> InterceptResult:
        """Compute deterministic intercept geometry and return InterceptResult."""
        intercept_pos, steps, ok = compute_deterministic_intercept(
            source_pos=src,
            target_pos=target_pos,
            target_vel=target_vel,
            speed=int(round(speed)),
            metric=metric,
            grid_shape=(self.height, self.width),
        )
        return InterceptResult(
            target_pos=target_pos,
            intercept_pos=intercept_pos,
            steps=steps,
            success=ok,
        )


# Alias for convenience and backward compatibility
PhysicsEngine = DiscretePhysicsEngine
