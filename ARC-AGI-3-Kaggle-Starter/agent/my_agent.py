import os
import sys
import threading
from pathlib import Path
from typing import Any, List
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR.parent, CURRENT_DIR.parent.parent, Path("/tmp")]:
    if (p / "project_override").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from project_override.orchestration.general_agent import GeneralAgent

# Throttle 110 concurrent Swarm threads to 8 active compute workers on Kaggle 4-vCPU VM
_COMPUTE_SEMAPHORE = threading.BoundedSemaphore(8)


class MyAgent(Agent):
    """Official ARC-AGI-3 Harness Adapter for Project Override GeneralAgent."""

    MAX_ACTIONS: int = 400

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.orchestration_agent = GeneralAgent(game_id=getattr(self, "game_id", "unknown"))

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state is GameState.NOT_PLAYED:
            return GameAction.RESET

        raw_frames = getattr(latest_frame, "frame", [])
        grid = None
        if raw_frames:
            grid = np.asarray(raw_frames[-1], dtype=np.int8)
            if grid.ndim == 3:
                grid = grid[-1]

        if latest_frame.state is GameState.GAME_OVER:
            with _COMPUTE_SEMAPHORE:
                self.orchestration_agent.on_game_over(grid)
            return GameAction.RESET

        if grid is None:
            return GameAction.ACTION1

        avail_raw = getattr(latest_frame, "available_actions", None) or [1, 2, 3, 4]
        avail_ids = [int(getattr(a, "value", a)) for a in avail_raw]
        lvls = int(getattr(latest_frame, "levels_completed", 0) or 0)

        with _COMPUTE_SEMAPHORE:
            act_id, data = self.orchestration_agent.step(
                grid=grid,
                available_actions=avail_ids,
                levels_completed=lvls,
            )

        try:
            action = GameAction.from_id(int(act_id))
        except Exception:
            action = GameAction.ACTION1

        if action == GameAction.ACTION6 and data:
            action.set_data({"x": int(data.get("x", 32)), "y": int(data.get("y", 32))})

        return action
