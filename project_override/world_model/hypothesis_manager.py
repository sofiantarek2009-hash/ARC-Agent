from typing import Dict, List, Optional, Set, Tuple


class HypothesisManager:
    """
    Active Scientific Experimentation & Causal Entity Semantic Ledger.
    Tracks falsifiable hypotheses on directional kinematics and entity color roles:
    - FLOOR, AVATAR, WALL, HAZARD, BOX, PORTAL, SWITCH, GOAL
    """

    def __init__(self):
        self.probe_order: List[int] = [1, 3, 2, 4]
        self.tested_actions: Dict[int, int] = {}
        self.color_semantics: Dict[int, str] = {}
        self.portal_pairs: Dict[int, List[Tuple[int, int]]] = {}

    def reset_level(self, keep_semantics: bool = True) -> None:
        self.tested_actions.clear()
        if not keep_semantics:
            self.color_semantics.clear()
            self.portal_pairs.clear()

    def record_color_role(self, color: int, role: str) -> None:
        self.color_semantics[int(color)] = role

    def next_probe(
        self,
        available_moves: List[int],
        known_vectors: Dict[int, Tuple[int, int]],
        schema_frozen: bool = False,
    ) -> Optional[int]:
        """If schema_frozen is True (Levels 2..K), zero probing is needed (saves quadratic RHAE!)."""
        if schema_frozen and len(known_vectors) >= 2:
            return None
        for act in self.probe_order:
            if act in available_moves and act not in known_vectors:
                if self.tested_actions.get(act, 0) < 2:
                    self.tested_actions[act] = self.tested_actions.get(act, 0) + 1
                    return act
        return None
