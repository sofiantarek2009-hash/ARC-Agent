from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from project_override.memory.action_ledger import ActionLedger
from project_override.memory.hypothesis_store import BayesianExplorationTracker
from project_override.perception.observation_processor import Entity, ObservationProcessor

def bfs_frontier_explore(
    start: Tuple[int, int],
    visited_cells: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    action_vectors: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int] = (64, 64),
) -> Optional[List[int]]:
    if not action_vectors:
        return None
    h, w = grid_shape
    strides = [
        max(abs(dr), abs(dc))
        for dr, dc in action_vectors.values()
        if (dr, dc) != (0, 0)
    ]
    s = max(1, min(strides)) if strides else 1
    queue: deque = deque([(start, [])])
    seen: Set[Tuple[int, int]] = {start}
    while queue and len(seen) < 1200:
        curr, path = queue.popleft()
        norm_cell = (curr[0] // s, curr[1] // s)
        if path and norm_cell not in visited_cells:
            return path
        if len(path) >= 22:
            continue
        for act_id, (dr, dc) in action_vectors.items():
            if dr == 0 and dc == 0:
                continue
            nr, nc = curr[0] + dr, curr[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            steps_n = max(1, max(abs(dr), abs(dc)))
            blocked = any(
                (
                    int(round(curr[0] + (dr * k) / steps_n)),
                    int(round(curr[1] + (dc * k) / steps_n)),
                )
                in walls
                for k in range(1, steps_n + 1)
            )
            if not blocked and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), path + [int(act_id)]))
    return None

class SmartClickExplorer:
    """
    Selects high-information-gain click coordinates using object segmentation,
    OPINE-World Bayesian exploration priority S_B (arXiv:2607.01531),
    and Reki dead-signature pruning.
    """
    def __init__(self, processor: ObservationProcessor):
        self.processor = processor

    def select_click(
        self,
        grid: np.ndarray,
        arena_hash: int,
        ledger: ActionLedger,
        level_idx: int = 0,
        bayesian_tracker: Optional[BayesianExplorationTracker] = None,
        enable_bayesian: bool = True,
    ) -> Tuple[int, Dict[str, int], Optional[Tuple[int, int, int, int]]]:
        bg = self.processor.induce_background_color(grid)
        entities = self.processor.extract_entities(grid, bg_color=bg)
        color_counts: Dict[int, int] = {}
        for e in entities:
            color_counts[e.color] = color_counts.get(e.color, 0) + 1

        def _score_entity(ent: Entity) -> Tuple[float, int, int, int]:
            prio = 1.0
            if enable_bayesian and bayesian_tracker is not None:
                prio = bayesian_tracker.exploration_priority(ent.shape_signature, 6)
            return (-prio, color_counts.get(ent.color, 1), ent.mass, ent.id)

        ranked = sorted(entities, key=_score_entity)
        for ent in ranked:
            cy = int(round(ent.centroid[0]))
            cx = int(round(ent.centroid[1]))
            if (cy, cx) not in ent.pixels and ent.pixels:
                cy, cx = ent.pixels[len(ent.pixels) // 2]
            sig = ent.shape_signature
            if not ledger.is_dead(arena_hash, 6, cx, cy, level_idx, obj_sig=sig):
                return 6, {"x": int(cx), "y": int(cy)}, sig
        h, w = grid.shape
        for step in (8, 5, 3):
            for r in range(step, h - step, step):
                for c in range(step, w - step, step):
                    if not ledger.is_dead(arena_hash, 6, c, r, level_idx):
                        return 6, {"x": int(c), "y": int(r)}, None
        return 6, {"x": 32, "y": 32}, None
