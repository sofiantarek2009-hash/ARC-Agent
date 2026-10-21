from typing import Dict, List, Optional, Tuple


class HypothesisManager:
    """Manages active kinematic & interaction hypotheses during Level 1 exploration."""

    def __init__(self):
        self.probe_order: List[int] = [1, 3, 2, 4]
        self.tested_actions: Dict[int, int] = {}

    def next_probe(self, available_moves: List[int], known_vectors: Dict[int, Tuple[int, int]]) -> Optional[int]:
        for act in self.probe_order:
            if act in available_moves and act not in known_vectors:
                if self.tested_actions.get(act, 0) < 2:
                    self.tested_actions[act] = self.tested_actions.get(act, 0) + 1
                    return act
        return None
