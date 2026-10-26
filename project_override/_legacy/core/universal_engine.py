import time
from typing import Optional


class SystemPacer:
    GLOBAL_START_TIME = time.time()
    GLOBAL_HARD_LIMIT_S = 31800.0

    def __init__(self, max_level_compute_s: float = 45.0):
        self.max_level_compute_s = max_level_compute_s
        self.level_compute_s: float = 0.0
        self._step_t0: Optional[float] = None

    def reset_level(self) -> None:
        self.level_compute_s = 0.0
        self._step_t0 = None

    def begin_step(self) -> None:
        self._step_t0 = time.perf_counter()

    def end_step(self) -> None:
        if self._step_t0 is not None:
            self.level_compute_s += time.perf_counter() - self._step_t0
            self._step_t0 = None

    def has_compute_budget(self) -> bool:
        if (time.time() - self.GLOBAL_START_TIME) > self.GLOBAL_HARD_LIMIT_S:
            return False
        return self.level_compute_s < self.max_level_compute_s
