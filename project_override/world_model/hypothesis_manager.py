from typing import Dict, List, Optional, Set, Tuple


class HypothesisManager:
    """
    Active Scientific Experimentation & Causal Entity Semantic Ledger.
    Tracks falsifiable hypotheses on directional kinematics, paired portals,
    and entity color roles: FLOOR, AVATAR, WALL, HAZARD, BOX, PORTAL, SWITCH, GOAL.
    """

    def __init__(self):
        self.probe_order: List[int] = [1, 3, 2, 4]
        self.tested_actions: Dict[int, int] = {}
        self.color_semantics: Dict[int, str] = {}
        self.portal_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.interacted_positions: Set[Tuple[int, int]] = set()

    def reset_level(self, keep_semantics: bool = True) -> None:
        self.tested_actions.clear()
        self.portal_map.clear()
        self.interacted_positions.clear()
        if not keep_semantics:
            self.color_semantics.clear()

    def record_color_role(self, color: int, role: str) -> None:
        self.color_semantics[int(color)] = role

    def register_portal_jump(self, entry_rc: Tuple[int, int], exit_rc: Tuple[int, int]) -> None:
        self.portal_map[entry_rc] = exit_rc
        self.portal_map[exit_rc] = entry_rc

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
                if self.tested_actions.get(act, 0) < 1:
                    self.tested_actions[act] = self.tested_actions.get(act, 0) + 1
                    return act
        return None
