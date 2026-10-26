from collections import deque
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from project_override.memory.action_ledger import ActionLedger
from project_override.perception.observation_processor import Entity, ObservationProcessor

logger = logging.getLogger(__name__)


def astar_grid(
    start_rc: Tuple[int, int],
    target_rc: Tuple[int, int],
    obstacle_cells: Set[Tuple[int, int]],
    effective_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
) -> List[int]:
    """Lightweight 2D A* pathfinder navigating avatar across grid."""
    import heapq

    if start_rc == target_rc:
        return []
    
    h, w = grid_shape
    heap = [(0, 0, start_rc, [])]
    visited = {start_rc: 0}

    while heap:
        f, g, (r, c), path = heapq.heappop(heap)
        if (r, c) == target_rc:
            return path

        for act_id, (dr, dc) in effective_vectors.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_cells:
                ng = g + 1
                if (nr, nc) not in visited or ng < visited[(nr, nc)]:
                    visited[(nr, nc)] = ng
                    manhattan = abs(nr - target_rc[0]) + abs(nc - target_rc[1])
                    heapq.heappush(heap, (ng + manhattan, ng, (nr, nc), path + [act_id]))

    return []


class MacroActionQueue:
    """Buffers sequential actions with Franzen M2 interior-change guards."""
    def __init__(self):
        self.queue: deque = deque()
        self.active_subagent: str = "DIRECT"

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self) -> int:
        return len(self.queue)

    def clear(self) -> None:
        self.queue.clear()
        self.active_subagent = "DIRECT"

    def push_plan(
        self,
        plan: List[Tuple[int, Dict[str, Any]]],
        subagent_tag: str = "MACRO",
        max_len: int = 18,
    ) -> None:
        self.queue.clear()
        self.active_subagent = subagent_tag
        for item in plan[:max_len]:
            if isinstance(item, int):
                self.queue.append((item, {}))
            elif isinstance(item, tuple) and len(item) == 2:
                self.queue.append(item)

    def pop_next(
        self,
        last_delta_omega: Optional[int] = None,
        guard_enabled: bool = True,
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        if not self.queue:
            return None
        # Guard: if the preceding macro action was an unexpected NO-OP, purge the hallucinated queue
        if guard_enabled and last_delta_omega == 0:
            logger.info(f"[MacroActionQueue] Unexpected NO-OP detected. Purging remaining {len(self.queue)} queued macro actions.")
            self.clear()
            return None
        return self.queue.popleft()


class GeneralInteractionExplorer:
    """Hypothesis-driven, sample-efficient interaction explorer for unfamiliar ARC-AGI-3 games.
    
    Cures:
    1. Card-Memory Bug: Ranks entity centroids by rarity / least-clicked (ascending visit counts)
       so card pairs are systematically discovered without clicking background pixels.
    2. Action 5 Blindspot: Deliberately probes ACTION 5 (Interact/Pulse) during early Level 1 turns.
    3. Graph BFS: Systematically chooses untried actions from the current state hash.
    """

    def __init__(self, processor: Optional[ObservationProcessor] = None):
        self.processor = processor or ObservationProcessor()
        self.action5_probed: bool = False
        self.click_history: Dict[Tuple[int, int], int] = {}

    def reset_level(self) -> None:
        self.action5_probed = False
        self.click_history.clear()

    def select_exploration_action(
        self,
        grid: np.ndarray,
        entities: List[Entity],
        avatar_pos: Optional[Tuple[int, int]],
        available_actions: List[int],
        step_in_level: int,
        state_hash: int,
        ledger: ActionLedger,
        level_idx: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """Selects the most informative exploratory action."""
        
        # Priority 1: Deliberately probe ACTION5 on early Level 1 steps if available
        if not self.action5_probed and 5 in available_actions and step_in_level <= 5:
            self.action5_probed = True
            logger.info("[GeneralInteractionExplorer] Probing ACTION5 (Interact/Pulse).")
            return 5, {}

        directional = [a for a in available_actions if a in (1, 2, 3, 4)]

        # Priority 2: If controllable avatar is present, explore via directional movement
        if avatar_pos is not None and directional:
            untried = ledger.get_untried_discrete_actions(state_hash, directional, level_idx)
            if untried:
                return untried[0], {}
            ranked = sorted(
                directional,
                key=lambda a: ledger.compute_action_penalty(state_hash, a, None, 0, 0, level_idx),
            )
            return ranked[0], {}

        # Priority 3: Coordinate Clicks (ACTION6) - explore salient entity centroids
        if 6 in available_actions:
            click_choice = self._select_smart_click(grid, entities, state_hash, ledger, level_idx)
            if click_choice is not None:
                return 6, click_choice

        # Priority 4: Fallback directional actions
        if directional:
            untried = ledger.get_untried_discrete_actions(state_hash, directional, level_idx)
            if untried:
                return untried[0], {}
            ranked = sorted(
                directional,
                key=lambda a: ledger.compute_action_penalty(state_hash, a, None, 0, 0, level_idx),
            )
            return ranked[0], {}

        # Priority 5: Fallback to first available action
        fallback = available_actions[0] if available_actions else 1
        return fallback, {}

    def _select_smart_click(
        self,
        grid: np.ndarray,
        entities: List[Entity],
        state_hash: int,
        ledger: ActionLedger,
        level_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Selects an unclicked or least-clicked entity centroid, avoiding background."""
        if not entities:
            h, w = grid.shape
            return {"x": w // 2, "y": h // 2}

        # Filter entities: ignore giant background-like objects
        salient = [e for e in entities if 1 <= e.mass <= 100]
        if not salient:
            salient = entities

        # Sort by click frequency (least-clicked first) to fix the card-memory bug
        candidates: List[Tuple[int, int, int]] = []
        for e in salient:
            cx, cy = int(round(e.centroid[1])), int(round(e.centroid[0]))
            count = self.click_history.get((cx, cy), 0)
            candidates.append((count, cx, cy))

        candidates.sort(key=lambda item: item[0])

        for count, cx, cy in candidates:
            if ledger.govern_action(state_hash, 6, cx, cy, level_idx):
                self.click_history[(cx, cy)] = count + 1
                return {"x": cx, "y": cy}

        # If all centroids governed out, pick first candidate
        if candidates:
            _, cx, cy = candidates[0]
            self.click_history[(cx, cy)] = self.click_history.get((cx, cy), 0) + 1
            return {"x": cx, "y": cy}

        return None
