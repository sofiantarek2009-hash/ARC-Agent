"""Project Override: Epistemic-Damped UCT MCTS Planner (ARC-AGI-3)
Implements:
1. Epistemic-Damped UCT search:
   UCT(s, a) = Q(s, a) + C_p * exp(-lambda * V(s)) * sqrt(ln(N) / (1 + n))
   where V(s) is epistemic uncertainty / state variance.
2. O(1) Dead-Signature Culling:
   Prunes dead actions instantaneously prior to simulation or expansion.
3. Semantic Evaluator Integration:
   Evaluates states via D4 symmetry, boundary enclosure, compaction,
   Shannon entropy transitions (Delta H <= -epsilon), and Sinkhorn W2 distance.
4. Macro-Burst Output:
   Extracts high-confidence action chains for MacroActionQueue.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from project_override.core.dead_signature import (
    ActionSignature,
    DeadSignatureFilter,
    compute_context_hash,
    pack_signature,
)
from project_override.core.semantic_evaluator import (
    SemanticEvaluator,
    StructuralEntropyScorer,
    SinkhornKnoppOptimalTransport,
)


# =====================================================================
# Discrete Grid Action Model for ARC-AGI-3
# =====================================================================

class DiscreteActionType:
    """Discrete primitive actions for ARC grid navigation and manipulation."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    ROTATE_90 = 4
    FLIP_HORIZONTAL = 5
    FLIP_VERTICAL = 6
    FILL_FLOOD = 7
    COLOR_TRANSFORM = 8
    NOOP = 9

    NAMES = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "ROTATE_90",
        5: "FLIP_HORIZONTAL",
        6: "FLIP_VERTICAL",
        7: "FILL_FLOOD",
        8: "COLOR_TRANSFORM",
        9: "NOOP",
    }


def simulate_grid_action(
    grid: np.ndarray,
    action: int,
    color: int = 1,
    target_pos: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Deterministic forward model for primitive discrete actions on ARC grids.

    Can be replaced or wrapped by arbitrary external environment simulators.
    """
    next_grid = np.copy(grid)
    h, w = next_grid.shape

    if action == DiscreteActionType.MOVE_UP:
        next_grid = np.roll(next_grid, -1, axis=0)
        next_grid[-1, :] = 0
    elif action == DiscreteActionType.MOVE_DOWN:
        next_grid = np.roll(next_grid, 1, axis=0)
        next_grid[0, :] = 0
    elif action == DiscreteActionType.MOVE_LEFT:
        next_grid = np.roll(next_grid, -1, axis=1)
        next_grid[:, -1] = 0
    elif action == DiscreteActionType.MOVE_RIGHT:
        next_grid = np.roll(next_grid, 1, axis=1)
        next_grid[:, 0] = 0
    elif action == DiscreteActionType.ROTATE_90:
        if h == w:
            next_grid = np.rot90(next_grid, k=3)
    elif action == DiscreteActionType.FLIP_HORIZONTAL:
        next_grid = np.fliplr(next_grid)
    elif action == DiscreteActionType.FLIP_VERTICAL:
        next_grid = np.flipud(next_grid)
    elif action == DiscreteActionType.COLOR_TRANSFORM:
        # Replaces dominant foreground color with specified color
        fg = [int(c) for c in np.unique(next_grid) if c != 0]
        if fg:
            dom = fg[0]
            next_grid[next_grid == dom] = color
    elif action == DiscreteActionType.FILL_FLOOD:
        if target_pos is not None:
            tr, tc = target_pos
            if 0 <= tr < h and 0 <= tc < w:
                old_color = next_grid[tr, tc]
                if old_color != color:
                    queue = [(tr, tc)]
                    visited = {(tr, tc)}
                    while queue:
                        r, c = queue.pop(0)
                        next_grid[r, c] = color
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if (nr, nc) not in visited and next_grid[nr, nc] == old_color:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
    elif action == DiscreteActionType.NOOP:
        pass

    return next_grid


# =====================================================================
# Epistemic MCTS Search Node
# =====================================================================

class EpistemicMCTSNode:
    """Search tree node representing a state s in Epistemic-Damped UCT."""

    def __init__(
        self,
        state: np.ndarray,
        parent: Optional[EpistemicMCTSNode] = None,
        action_from_parent: Optional[int] = None,
        action_color: int = 0,
        action_signature: Optional[int] = None,
        available_actions: Optional[List[int]] = None,
        prior_variance: float = 1.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.action_color = action_color
        self.action_signature = action_signature

        self.children: Dict[int, EpistemicMCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0

        # Action-specific statistics: a -> n(s, a), Q(s, a)
        self.action_visits: Dict[int, int] = {}
        self.action_values: Dict[int, float] = {}

        # Untried actions for progressive expansion
        self.untried_actions: List[int] = (
            list(available_actions) if available_actions is not None else []
        )

        # Epistemic uncertainty V(s)
        self.prior_variance: float = prior_variance
        self.epistemic_variance: float = prior_variance
        self.is_terminal: bool = False
        self.dead_actions: Set[int] = set()

    @property
    def mean_value(self) -> float:
        """Q(s) mean state value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def update_epistemic_variance(self) -> None:
        """Re-evaluates epistemic variance V(s).

        V(s) = Var_{a in children}(Q(s, a)) + prior_variance / (1 + visit_count).
        As visit count increases and child values converge, epistemic uncertainty drops.
        """
        decay_term = self.prior_variance / (1.0 + float(self.visit_count))
        if len(self.action_values) > 1:
            vals = list(self.action_values.values())
            empirical_var = float(np.var(vals))
            self.epistemic_variance = empirical_var + decay_term
        else:
            self.epistemic_variance = decay_term

    def compute_uct(
        self,
        action: int,
        c_p: float = 1.4142,
        lambd: float = 1.0,
        epsilon_n: float = 1e-4,
    ) -> float:
        """Epistemic-Damped UCT formula:

        UCT(s, a) = Q(s, a) + C_p * exp(-lambda * V(s)) * sqrt(ln(N) / (1 + n))
        """
        n_sa = self.action_visits.get(action, 0)
        q_sa = self.action_values.get(action, 0.0)
        n_total = max(1, self.visit_count)

        # Epistemic damping factor: exp(-lambda * V(s))
        # High epistemic uncertainty (chaos) -> dampens exploration
        # Grounded certainty (low V) -> full UCT exploration
        damping = math.exp(-max(0.0, lambd * self.epistemic_variance))

        # Exploration bonus
        exploration = c_p * damping * math.sqrt(math.log(n_total + 1.0) / (1.0 + n_sa))

        # If unvisited, provide an optimistic prior exploration incentive
        if n_sa == 0:
            return q_sa + exploration + 2.0 * damping

        return q_sa + exploration

    def select_best_action(
        self,
        c_p: float = 1.4142,
        lambd: float = 1.0,
    ) -> int:
        """Selects action maximizing Epistemic-Damped UCT."""
        viable_actions = [
            a for a in (list(self.children.keys()) + self.untried_actions)
            if a not in self.dead_actions
        ]
        if not viable_actions:
            return DiscreteActionType.NOOP

        best_score = -float("inf")
        best_action = viable_actions[0]

        for action in viable_actions:
            score = self.compute_uct(action, c_p=c_p, lambd=lambd)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action


# =====================================================================
# Epistemic MCTS Planner
# =====================================================================

@dataclass
class PlanResult:
    """Summary of MCTS search planning results."""
    best_action: int
    best_action_name: str
    macro_action_sequence: List[int]
    total_simulations: int
    culled_actions_count: int
    root_value: float
    root_epistemic_variance: float
    elapsed_time: float
    completed_iterations: int


class EpistemicMCTSPlanner:
    """Epistemic-Damped UCT Search Planner with O(1) Dead-Signature Culling

    and integrated Semantic Evaluation.
    """

    def __init__(
        self,
        c_p: float = 1.4142,
        lambd: float = 1.0,
        max_depth: int = 10,
        default_actions: Optional[List[int]] = None,
        transition_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
        dead_filter: Optional[DeadSignatureFilter] = None,
        evaluator: Optional[SemanticEvaluator] = None,
    ) -> None:
        self.c_p = c_p
        self.lambd = lambd
        self.max_depth = max_depth
        self.actions = (
            default_actions
            if default_actions is not None
            else [
                DiscreteActionType.MOVE_UP,
                DiscreteActionType.MOVE_DOWN,
                DiscreteActionType.MOVE_LEFT,
                DiscreteActionType.MOVE_RIGHT,
                DiscreteActionType.ROTATE_90,
                DiscreteActionType.FLIP_HORIZONTAL,
                DiscreteActionType.FLIP_VERTICAL,
                DiscreteActionType.COLOR_TRANSFORM,
            ]
        )
        self.transition_fn = transition_fn or simulate_grid_action
        self.dead_filter = dead_filter or DeadSignatureFilter()
        self.evaluator = evaluator or SemanticEvaluator()

    def evaluate_state_reward(
        self,
        prior_state: np.ndarray,
        curr_state: np.ndarray,
        target_state: Optional[np.ndarray] = None,
        goal_color: Optional[int] = None,
    ) -> float:
        """Composite semantic reward function:

        R = w_H * (Delta H <= -epsilon) + w_sym * Symmetry + w_enc * Enclosure
            + w_comp * Compaction - w_W2 * W2(curr, target)
        """
        reward = 0.0

        # 1. Structural Shannon Entropy Transition
        trans_res = self.evaluator.evaluate_transition(
            prior_state, curr_state, target_state=target_state
        )
        entropy_res = trans_res["entropy_transition"]
        if entropy_res.is_rewarded:
            reward += 1.0 + min(5.0, entropy_res.reward * 2.0)
        else:
            reward += max(-1.0, -entropy_res.delta_h)

        # 2. Compaction & Density
        comp_res = self.evaluator.compaction_scorer.evaluate(curr_state)
        reward += 0.5 * min(2.0, comp_res.compactness_score)

        # 3. Boundary Enclosure
        enc_res = self.evaluator.enclosure_scorer.compute_enclosure(
            curr_state, goal_color=goal_color
        )
        if enc_res.has_sealed_loop:
            reward += 1.0
        if enc_res.is_goal_enclosed:
            reward += 3.0

        # 4. Symmetry Bonus
        sym_res = self.evaluator.symmetry_scorer.evaluate(curr_state)
        reward += 0.5 * sym_res.composite_score

        # 5. Discrete Optimal Transport Proximity to Target Goal State
        if target_state is not None:
            w2_res = trans_res["target_w2"]
            if w2_res is not None:
                # Closer to target -> higher reward
                reward += max(-5.0, 5.0 - float(w2_res.w2_distance))

        return float(reward)

    def search(
        self,
        root_state: np.ndarray,
        target_state: Optional[np.ndarray] = None,
        goal_color: Optional[int] = None,
        num_simulations: int = 100,
        timeout_seconds: Optional[float] = 10.0,
    ) -> PlanResult:
        """Executes Epistemic-Damped UCT search from root_state."""
        start_time = time.perf_counter()
        root_state = np.asarray(root_state, dtype=np.int32)
        culled_count = 0

        root_node = EpistemicMCTSNode(
            state=root_state,
            available_actions=self.actions,
            prior_variance=1.5,
        )

        completed_simulations = 0

        for sim_idx in range(num_simulations):
            if timeout_seconds is not None:
                if (time.perf_counter() - start_time) >= timeout_seconds:
                    break

            # -------------------------------------------------------------
            # Phase 1: Selection
            # -------------------------------------------------------------
            node = root_node
            depth = 0
            path: List[Tuple[EpistemicMCTSNode, int]] = []

            while not node.untried_actions and node.children and depth < self.max_depth:
                action = node.select_best_action(c_p=self.c_p, lambd=self.lambd)
                if action in node.children:
                    path.append((node, action))
                    node = node.children[action]
                    depth += 1
                else:
                    break

            # -------------------------------------------------------------
            # Phase 2: Expansion with O(1) Dead-Signature Culling
            # -------------------------------------------------------------
            if node.untried_actions and depth < self.max_depth:
                action_to_expand: Optional[int] = None

                while node.untried_actions:
                    cand_action = node.untried_actions.pop(0)

                    # Calculate signature components
                    mass = int(np.sum(node.state != 0))
                    color = int(np.max(node.state)) if mass > 0 else 0
                    ctx_hash = compute_context_hash(node.state)
                    sig = pack_signature(
                        action=cand_action,
                        color=color,
                        mass=mass,
                        context_hash=ctx_hash,
                    )

                    # O(1) Dead Signature Check
                    if self.dead_filter.is_dead(sig):
                        node.dead_actions.add(cand_action)
                        culled_count += 1
                        continue

                    # Static null transition check
                    if self.dead_filter.predict_static_null(
                        cand_action, color, mass, node.state
                    ):
                        self.dead_filter.record_transition(sig, 0)
                        node.dead_actions.add(cand_action)
                        culled_count += 1
                        continue

                    # Action is viable for expansion
                    action_to_expand = cand_action
                    break

                if action_to_expand is not None:
                    # Execute transition
                    next_state = self.transition_fn(node.state, action_to_expand, color)

                    # State delta Delta Omega = ||s' - s||_0
                    delta_omega = int(np.sum(next_state != node.state))

                    if delta_omega == 0:
                        # Register dead signature in O(1)
                        self.dead_filter.record_transition(sig, 0)
                        node.dead_actions.add(action_to_expand)
                        culled_count += 1
                        # Backpropagate penalty for dead null-move
                        reward = -2.0
                        self._backpropagate(path, reward)
                        completed_simulations += 1
                        continue

                    # Create child node
                    child_node = EpistemicMCTSNode(
                        state=next_state,
                        parent=node,
                        action_from_parent=action_to_expand,
                        action_color=color,
                        action_signature=sig,
                        available_actions=self.actions,
                        prior_variance=max(0.5, node.epistemic_variance * 0.9),
                    )
                    node.children[action_to_expand] = child_node
                    path.append((node, action_to_expand))
                    node = child_node

            # -------------------------------------------------------------
            # Phase 3: Semantic Evaluation & Simulation
            # -------------------------------------------------------------
            prior_st = node.parent.state if node.parent is not None else root_state
            reward = self.evaluate_state_reward(
                prior_state=prior_st,
                curr_state=node.state,
                target_state=target_state,
                goal_color=goal_color,
            )

            # -------------------------------------------------------------
            # Phase 4: Backpropagation with Epistemic Uncertainty Updating
            # -------------------------------------------------------------
            self._backpropagate(path, reward)
            completed_simulations += 1

        # Extract best action from root
        best_action, macro_seq = self._extract_best_actions(root_node)
        action_name = DiscreteActionType.NAMES.get(best_action, f"ACTION_{best_action}")
        elapsed = time.perf_counter() - start_time

        return PlanResult(
            best_action=best_action,
            best_action_name=action_name,
            macro_action_sequence=macro_seq,
            total_simulations=completed_simulations,
            culled_actions_count=culled_count,
            root_value=root_node.mean_value,
            root_epistemic_variance=root_node.epistemic_variance,
            elapsed_time=elapsed,
            completed_iterations=completed_simulations,
        )

    def _backpropagate(
        self,
        path: List[Tuple[EpistemicMCTSNode, int]],
        reward: float,
    ) -> None:
        """Backpropagates reward and updates visit counts, values, and epistemic variance."""
        for parent_node, action in reversed(path):
            parent_node.visit_count += 1
            parent_node.total_value += reward

            # Update action-specific statistics
            curr_visits = parent_node.action_visits.get(action, 0) + 1
            parent_node.action_visits[action] = curr_visits

            curr_q = parent_node.action_values.get(action, 0.0)
            # Incremental mean update: Q <- Q + (r - Q) / n
            parent_node.action_values[action] = curr_q + (reward - curr_q) / curr_visits

            # Update epistemic variance V(s)
            parent_node.update_epistemic_variance()

    def _extract_best_actions(
        self,
        root: EpistemicMCTSNode,
        max_macro_steps: int = 4,
    ) -> Tuple[int, List[int]]:
        """Extracts optimal action and macro action sequence (up to max_macro_steps)."""
        if not root.action_visits:
            return DiscreteActionType.NOOP, []

        # Best action by highest visit count (robust choice)
        best_a = max(
            root.action_visits.keys(),
            key=lambda a: (root.action_visits[a], root.action_values.get(a, -float("inf"))),
        )

        sequence: List[int] = [best_a]
        curr = root.children.get(best_a)

        while curr is not None and len(sequence) < max_macro_steps:
            if not curr.action_visits:
                break
            next_a = max(
                curr.action_visits.keys(),
                key=lambda a: (curr.action_visits[a], curr.action_values.get(a, -float("inf"))),
            )
            sequence.append(next_a)
            curr = curr.children.get(next_a)

        return best_a, sequence

    def to_macro_plan_steps(
        self,
        plan_result: PlanResult,
        current_grid: np.ndarray,
        color: int = 1,
    ) -> List[Any]:
        """Converts macro action sequence into MacroPlanStep objects with
        predicted grids and expected context hashes for MacroActionQueue.
        """
        try:
            from project_override.orchestration.macro_queue import MacroPlanStep
        except ImportError:
            MacroPlanStep = None

        steps: List[Any] = []
        curr = np.copy(current_grid)

        for act in plan_result.macro_action_sequence:
            expected_grid = self.transition_fn(curr, act, color)
            expected_hash = compute_context_hash(expected_grid)
            act_name = DiscreteActionType.NAMES.get(act, f"ACTION_{act}")

            if MacroPlanStep is not None:
                step = MacroPlanStep(
                    action=act,
                    action_data={"color": color},
                    expected_state_hash=expected_hash,
                    expected_grid=expected_grid,
                    description=f"EpistemicMCTS:{act_name}",
                )
            else:
                step = {
                    "action": act,
                    "action_data": {"color": color},
                    "expected_state_hash": expected_hash,
                    "expected_grid": expected_grid,
                    "description": f"EpistemicMCTS:{act_name}",
                }
            steps.append(step)
            curr = expected_grid

        return steps
