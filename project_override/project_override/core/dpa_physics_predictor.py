"""Project Override: Discrete Programmatic Automata (DPA) Physics Predictor
Models autonomous entities (bouncers, falling blocks, moving hazards) via discrete
cellular automata rules, predicts future trajectories, and computes intercept geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class AutomataClass(IntEnum):
    STATIC = 0
    LINEAR_BOUNCER = 1
    GRAVITY_FALL = 2
    CYCLIC_PATROL = 3
    CHASER = 4


@dataclass
class SpriteEntity:
    entity_id: int
    color: int
    coords: List[Tuple[int, int]]
    centroid: Tuple[float, float]
    velocity: Tuple[int, int]  # (vr, vc)
    automata_class: AutomataClass
    is_hazard: bool
    mass: int


@dataclass
class InterceptSolution:
    target_pos: Tuple[int, int]
    intercept_time: int
    is_feasible: bool


class DPAPhysicsPredictor:
    """Discrete cellular automata simulation and trajectory predictor for ARC-AGI-3."""

    def __init__(self) -> None:
        self.tracked_sprites: Dict[int, SpriteEntity] = {}

    def clear(self) -> None:
        self.tracked_sprites.clear()

    @staticmethod
    def detect_sprite_velocity(
        prev_coords: List[Tuple[int, int]],
        curr_coords: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Calculates discrete integer displacement vector (vr, vc)."""
        if not prev_coords or not curr_coords:
            return 0, 0
        p_r = sum(r for r, c in prev_coords) / len(prev_coords)
        p_c = sum(c for r, c in prev_coords) / len(prev_coords)
        c_r = sum(r for r, c in curr_coords) / len(curr_coords)
        c_c = sum(c for r, c in curr_coords) / len(curr_coords)
        vr = int(round(c_r - p_r))
        vc = int(round(c_c - p_c))
        return vr, vc

    @staticmethod
    def classify_automata(
        velocity: Tuple[int, int],
        is_gravity: bool = False,
    ) -> AutomataClass:
        vr, vc = velocity
        if vr == 0 and vc == 0:
            return AutomataClass.STATIC
        if is_gravity and vr > 0 and vc == 0:
            return AutomataClass.GRAVITY_FALL
        return AutomataClass.LINEAR_BOUNCER

    @staticmethod
    def predict_bouncer_trajectory(
        start_pos: Tuple[int, int],
        velocity: Tuple[int, int],
        steps: int,
        grid_shape: Tuple[int, int],
        solid_mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        """Simulates linear bouncer forward with discrete boundary reflection."""
        h, w = grid_shape
        r, c = start_pos
        vr, vc = velocity
        trajectory: List[Tuple[int, int]] = [(r, c)]

        for _ in range(steps):
            next_r = r + vr
            next_c = c + vc

            # Boundary reflection
            if next_r < 0 or next_r >= h or (solid_mask is not None and 0 <= next_r < h and 0 <= next_c < w and solid_mask[next_r, c]):
                vr = -vr
                next_r = r + vr

            if next_c < 0 or next_c >= w or (solid_mask is not None and 0 <= next_r < h and 0 <= next_c < w and solid_mask[r, next_c]):
                vc = -vc
                next_c = c + vc

            r, c = max(0, min(h - 1, next_r)), max(0, min(w - 1, next_c))
            trajectory.append((r, c))

        return trajectory

    @staticmethod
    def compute_deterministic_intercept(
        avatar_pos: Tuple[int, int],
        avatar_speed: float,
        target_start_pos: Tuple[int, int],
        target_velocity: Tuple[int, int],
        max_horizon: int = 40,
        grid_shape: Tuple[int, int] = (64, 64),
        solid_mask: Optional[np.ndarray] = None,
    ) -> InterceptSolution:
        """Finds minimum-time intercept coordinates for a moving dynamic target."""
        trajectory = DPAPhysicsPredictor.predict_bouncer_trajectory(
            start_pos=target_start_pos,
            velocity=target_velocity,
            steps=max_horizon,
            grid_shape=grid_shape,
            solid_mask=solid_mask,
        )

        ar, ac = avatar_pos
        for t, (tr, tc) in enumerate(trajectory):
            manhattan_dist = abs(ar - tr) + abs(ac - tc)
            # If avatar can reach (tr, tc) in <= t steps:
            time_needed = manhattan_dist / max(1e-5, avatar_speed)
            if time_needed <= t:
                return InterceptSolution(
                    target_pos=(tr, tc),
                    intercept_time=t,
                    is_feasible=True,
                )

        return InterceptSolution(
            target_pos=trajectory[-1],
            intercept_time=max_horizon,
            is_feasible=False,
        )

    @staticmethod
    def compute_forbidden_hazard_cells(
        hazards: List[Tuple[Tuple[int, int], Tuple[int, int]]],  # List of ((r, c), (vr, vc))
        steps: int,
        grid_shape: Tuple[int, int],
        solid_mask: Optional[np.ndarray] = None,
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """Returns map of time_step -> set of (r, c) cells occupied by moving hazards."""
        forbidden_per_step: Dict[int, Set[Tuple[int, int]]] = {t: set() for t in range(steps + 1)}

        for pos, vel in hazards:
            traj = DPAPhysicsPredictor.predict_bouncer_trajectory(
                start_pos=pos,
                velocity=vel,
                steps=steps,
                grid_shape=grid_shape,
                solid_mask=solid_mask,
            )
            for t, (hr, hc) in enumerate(traj):
                forbidden_per_step[t].add((hr, hc))
                # Hazard expansion buffer (adjacent cells for safety)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    forbidden_per_step[t].add((hr + dr, hc + dc))

        return forbidden_per_step
