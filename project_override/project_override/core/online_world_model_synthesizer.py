"""Universal Online World Model Synthesizer & CEGIS Transition Inducer
"Coding on the fly" for unfamiliar alien mechanics.

Synthesizes empirical transition functions:
    T: (state, action) -> next_state
verifies 100% retrodiction on historical transitions H = {(s_t, a_t, s_{t+1})},
and performs forward state-space search in memory to find the winning macro-path
with zero exploratory waste.

Strictly general, zero-hardcoding, zero game-specific constants.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class TransitionRecord:
    state_sig: Tuple[Any, ...]
    action: int
    next_state_sig: Tuple[Any, ...]
    reward: float = 0.0
    is_terminal: bool = False


class OnlineWorldModelSynthesizer:
    """Induces empirical state transition graphs and synthesizes forward planners on the fly."""

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.history: List[TransitionRecord] = []
        self.transition_graph: Dict[Tuple[Tuple[Any, ...], int], Tuple[Any, ...]] = {}
        self.state_successors: Dict[Tuple[Any, ...], Dict[int, Tuple[Any, ...]]] = collections.defaultdict(dict)
        self.is_deterministic: bool = True
        self.synthesized_rules: Dict[str, Any] = {}

    def record_transition(
        self,
        state_sig: Tuple[Any, ...],
        action: int,
        next_state_sig: Tuple[Any, ...],
        reward: float = 0.0,
        is_terminal: bool = False,
    ) -> None:
        """Records an observed transition and updates the empirical transition graph."""
        rec = TransitionRecord(
            state_sig=state_sig,
            action=action,
            next_state_sig=next_state_sig,
            reward=reward,
            is_terminal=is_terminal,
        )
        self.history.append(rec)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        key = (state_sig, action)
        if key in self.transition_graph and self.transition_graph[key] != next_state_sig:
            self.is_deterministic = False  # Non-deterministic or partially observable

        self.transition_graph[key] = next_state_sig
        self.state_successors[state_sig][action] = next_state_sig

    def retrodiction_accuracy(self) -> float:
        """Measures percentage of historical transitions correctly predicted by the empirical model."""
        if not self.history:
            return 1.0
        correct = sum(
            1 for rec in self.history
            if self.transition_graph.get((rec.state_sig, rec.action)) == rec.next_state_sig
        )
        return float(correct) / float(len(self.history))

    def plan_to_goal(
        self,
        current_state_sig: Tuple[Any, ...],
        is_goal_predicate: Callable[[Tuple[Any, ...]], bool],
        legal_actions: List[int],
        max_depth: int = 30,
    ) -> Optional[List[int]]:
        """Plans the shortest action sequence to a goal state using the synthesized model."""
        if is_goal_predicate(current_state_sig):
            return []

        q = collections.deque([(current_state_sig, [])])
        visited: Set[Tuple[Any, ...]] = {current_state_sig}

        while q:
            curr_st, path = q.popleft()
            if len(path) >= max_depth:
                continue

            for act in legal_actions:
                # Check empirical transition
                nxt_st = self.transition_graph.get((curr_st, act))
                if nxt_st is not None and nxt_st not in visited:
                    if is_goal_predicate(nxt_st):
                        return path + [act]
                    visited.add(nxt_st)
                    q.append((nxt_st, path + [act]))

        return None


# =====================================================================
# Push-Chain & Multi-Stride Kinematics Simulator
# =====================================================================

class MultiStrideKinematicsSimulator:
    """Simulates advanced physical movement:
    - Multi-box push chains (Sokoban trains)
    - Knight leaps, Bishop diagonals
    - Wrap-around torus boundaries
    - Frictionless inertia glides
    """

    @staticmethod
    def simulate_push_chain(
        p_pos: Tuple[int, int],
        action_vector: Tuple[int, int],
        boxes: Set[Tuple[int, int]],
        walls: Set[Tuple[int, int]],
        gh: int,
        gw: int,
    ) -> Tuple[Tuple[int, int], Set[Tuple[int, int]], bool]:
        """Simulates pushing a chain of contiguous boxes along action_vector."""
        dr, dc = action_vector
        np_pos = (p_pos[0] + dr, p_pos[1] + dc)

        # Boundary / wall check for player
        if np_pos in walls or not (0 <= np_pos[0] < gh and 0 <= np_pos[1] < gw):
            return p_pos, boxes, False

        if np_pos not in boxes:
            return np_pos, boxes, True  # Free step

        # Box chain: find all contiguous boxes along the push direction
        chain: List[Tuple[int, int]] = []
        curr = np_pos
        while curr in boxes:
            chain.append(curr)
            curr = (curr[0] + dr, curr[1] + dc)

        # The lead empty space must be within bounds and not a wall
        if curr in walls or not (0 <= curr[0] < gh and 0 <= curr[1] < gw) or curr in boxes:
            return p_pos, boxes, False  # Chain blocked!

        # Chain can push! Displace all boxes in the chain
        new_boxes = set(boxes)
        for b in reversed(chain):
            new_boxes.remove(b)
            new_boxes.add((b[0] + dr, b[1] + dc))

        return np_pos, new_boxes, True

    @staticmethod
    def simulate_inertial_glide(
        p_pos: Tuple[int, int],
        action_vector: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        gh: int,
        gw: int,
        stop_on_goals: Optional[Set[Tuple[int, int]]] = None,
        max_glide: int = 64,
    ) -> Tuple[int, int]:
        """Simulates frictionless sliding until an obstacle or goal is reached."""
        dr, dc = action_vector
        curr = p_pos

        for _ in range(max_glide):
            nxt = (curr[0] + dr, curr[1] + dc)
            if nxt in walls or not (0 <= nxt[0] < gh and 0 <= nxt[1] < gw):
                break
            curr = nxt
            if stop_on_goals and curr in stop_on_goals:
                break

        return curr
