from typing import Callable, Dict, Optional, Set, Tuple


class MetagameSchema:
    INVERSE_PAIRS = {1: 2, 2: 1, 3: 4, 4: 3}

    def __init__(self):
        self.displacement_vectors: Dict[int, Tuple[int, int]] = {}
        self.frozen: bool = False
        self.verified_world_model: Optional[Callable] = None
        self.verified_model_code: Optional[str] = None
        self.key_to_door_colors: Dict[int, Set[int]] = {}

    def record_vector(self, action_id: int, vec: Tuple[int, int]) -> None:
        if vec == (0, 0):
            return
        self.displacement_vectors[int(action_id)] = (int(vec[0]), int(vec[1]))
        inv = self.INVERSE_PAIRS.get(int(action_id))
        if inv is not None and inv not in self.displacement_vectors:
            self.displacement_vectors[inv] = (-int(vec[0]), -int(vec[1]))

    def record_key_unlock(self, key_color: int, unlocked_door_color: int) -> None:
        if key_color not in self.key_to_door_colors:
            self.key_to_door_colors[int(key_color)] = set()
        self.key_to_door_colors[int(key_color)].add(int(unlocked_door_color))

    def freeze_on_level_clear(self) -> None:
        if self.displacement_vectors:
            self.frozen = True
