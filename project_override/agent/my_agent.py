from typing import Any, List
from project_override.orchestration.general_agent import GeneralAgent

try:
    from agents.agent import Agent
    from arcengine import FrameData, GameAction, GameState
except ImportError:
    class Agent:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.game_id = "local"
            self.frames = []
            self.action_counter = 0

        def main(self) -> None:
            pass
    class GameState:  # type: ignore
        NOT_PLAYED = "NOT_PLAYED"
        PLAYING = "PLAYING"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"
    class GameAction:  # type: ignore
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

class MyAgent(Agent):
    """Official ARC-AGI-3-Agents / Kaggle-Starter ABI wrapper around GeneralAgent."""
    MAX_ACTIONS = 450

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._brain = GeneralAgent()

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        state = getattr(latest_frame, "state", None)
        return state is GameState.WIN or str(getattr(state, "name", state)).upper() == "WIN"

    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        state = getattr(latest_frame, "state", None)
        state_name = str(getattr(state, "name", state or "PLAYING")).upper()
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or state_name in (
            "NOT_PLAYED",
            "GAME_OVER",
        ):
            return GameAction.RESET
        avail_raw = getattr(latest_frame, "available_actions", None) or [1, 2, 3, 4, 5, 6]
        avail_ids: List[int] = []
        for a in avail_raw:
            if isinstance(a, int):
                avail_ids.append(a)
            elif hasattr(a, "value") and isinstance(a.value, int):
                avail_ids.append(int(a.value))
            elif hasattr(a, "name") and str(a.name).upper().startswith("ACTION"):
                suffix = str(a.name).upper()[6:]
                if suffix.isdigit():
                    avail_ids.append(int(suffix))
        levels_done = int(getattr(latest_frame, "levels_completed", 0) or 0)
        act_id, payload = self._brain.act(
            raw_obs=latest_frame,
            available_actions=avail_ids or [1, 2, 3, 4, 5, 6],
            levels_completed=levels_done,
            state_str=state_name,
        )
        if act_id == 0:
            return GameAction.RESET
        action_obj = getattr(GameAction, f"ACTION{act_id}", GameAction.ACTION1)
        if hasattr(action_obj, "is_complex") and action_obj.is_complex():
            action_obj.set_data({"x": int(payload.get("x", 32)), "y": int(payload.get("y", 32))})
            action_obj.reasoning = {
                "agent": "project_override_v4",
                "x": payload.get("x", 32),
                "y": payload.get("y", 32),
            }
        elif hasattr(action_obj, "is_simple") and action_obj.is_simple():
            action_obj.reasoning = f"project_override_v4 selected ACTION{act_id}"
        return action_obj
