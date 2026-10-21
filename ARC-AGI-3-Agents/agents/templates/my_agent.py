import threading
from typing import Any, List
import numpy as np

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from project_override.orchestration.general_agent import GeneralAgent

_COMPUTE_SEMAPHORE = threading.Semaphore(2)


class MyAgent(Agent):
    """
    Competition Entrypoint for ARC-AGI-3 (Swarm / Multi-Thread Safe).
    Passes both settled frames and intermediate ARCEngine animation sequences to GeneralAgent.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        gid = getattr(self, "game_id", "arc_game")
        self.orchestration_agent = GeneralAgent(game_id=str(gid))

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state == GameState.GAME_OVER:
                self.orchestration_agent.on_game_over()
            return GameAction.RESET

        raw_frame = latest_frame.frame
        frame_seq: List[np.ndarray] = []
        if isinstance(raw_frame, list) and len(raw_frame) > 0:
            arr = np.asarray(raw_frame, dtype=np.int8)
            if arr.ndim == 3:
                frame_seq = [arr[i] for i in range(arr.shape[0])]
                grid = arr[-1]
            else:
                grid = arr
        else:
            grid = np.zeros((64, 64), dtype=np.int8)

        avail_raw = getattr(latest_frame, "available_actions", None) or [1, 2, 3, 4]
        avail_ids = [int(getattr(a, "value", a)) for a in avail_raw]
        lvls = int(getattr(latest_frame, "levels_completed", 0) or 0)

        with _COMPUTE_SEMAPHORE:
            act_id, data = self.orchestration_agent.step(
                grid=grid,
                available_actions=avail_ids,
                levels_completed=lvls,
                frame_sequence=frame_seq,
            )

        try:
            action = GameAction.from_id(int(act_id))
        except Exception:
            action = GameAction.ACTION1

        if action == GameAction.ACTION6 and data:
            action.set_data({"x": int(data.get("x", 32)), "y": int(data.get("y", 32))})

        return action
