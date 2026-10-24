from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Auto-resolve project_override and framework paths across local, starter, and Kaggle environments
_HERE = Path(__file__).resolve()
for candidate_dir in (
    Path.cwd(),
    Path("/kaggle/working"),
    Path("/kaggle/working/ARC-AGI-3-Agents"),
    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"),
    Path("/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"),
    _HERE.parent,
    _HERE.parent.parent,
    _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent,
    _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent,
):
    if candidate_dir.exists() and str(candidate_dir) not in sys.path:
        sys.path.insert(0, str(candidate_dir))

from project_override.orchestration.general_agent import GeneralAgent

try:
    from agents.agent import Agent
    from arcengine import FrameData, GameAction, GameState
    HAS_OFFICIAL_FRAMEWORK = True
except ImportError:
    HAS_OFFICIAL_FRAMEWORK = False

    class GameState:  # type: ignore
        NOT_PLAYED = "NOT_PLAYED"
        PLAYING = "PLAYING"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"
        NOT_FINISHED = "NOT_FINISHED"

    class GameAction:  # type: ignore
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class Agent:  # type: ignore
        MAX_ACTIONS: int = 450

        def __init__(
            self,
            card_id: str = "local",
            game_id: str = "local",
            agent_name: str = "MyAgent",
            ROOT_URL: str = "http://localhost",
            record: bool = False,
            arc_env: Any = None,
            tags: Optional[List[str]] = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            self.card_id = card_id
            self.game_id = game_id
            self.agent_name = agent_name
            self.ROOT_URL = ROOT_URL
            self.record = record
            self.arc_env = arc_env
            self.tags = tags or []
            self.action_counter = 0
            self.frames: List[Any] = []
            self.timer = 0.0

        def main(self) -> None:
            if self.arc_env is None:
                return
            obs = getattr(self.arc_env, "observation_space", None)
            if obs is not None:
                self.frames.append(obs)
            while (
                self.frames
                and not self.is_done(self.frames, self.frames[-1])
                and self.action_counter < self.MAX_ACTIONS
            ):
                action = self.choose_action(self.frames, self.frames[-1])
                next_obs = self.arc_env.step(action)
                if next_obs is not None:
                    self.frames.append(next_obs)
                self.action_counter += 1

        def cleanup(self, scorecard: Any = None) -> None:
            pass

        @property
        def state(self) -> Any:
            if not self.frames:
                return GameState.NOT_PLAYED
            return getattr(self.frames[-1], "state", GameState.NOT_PLAYED)

        @property
        def levels_completed(self) -> int:
            if not self.frames:
                return 0
            return int(getattr(self.frames[-1], "levels_completed", 0) or 0)

        def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
            raise NotImplementedError

        def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
            raise NotImplementedError


class MyAgent(Agent):
    """Official ARC-AGI-3-Agents / Kaggle-Starter ABI wrapper around GeneralAgent."""
    MAX_ACTIONS = 450

    def __init__(
        self,
        card_id: str = "local",
        game_id: str = "local",
        agent_name: str = "MyAgent",
        ROOT_URL: str = "http://localhost",
        record: bool = False,
        arc_env: Any = None,
        tags: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            card_id,
            game_id,
            agent_name,
            ROOT_URL,
            record,
            arc_env,
            tags,
            *args,
            **kwargs,
        )
        self._brain = GeneralAgent()
        # Expose alias for compatibility with code expecting orchestration_agent
        self.orchestration_agent = self._brain

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        state = getattr(latest_frame, "state", None)
        return (
            state is getattr(GameState, "WIN", "WIN")
            or str(getattr(state, "name", state)).upper() == "WIN"
        )

    def choose_action(self, frames: List[Any], latest_frame: Any) -> Any:
        state = getattr(latest_frame, "state", None)
        state_name = str(getattr(state, "name", state or "PLAYING")).upper()
        if (
            state in (getattr(GameState, "NOT_PLAYED", "NOT_PLAYED"), getattr(GameState, "GAME_OVER", "GAME_OVER"))
            or state_name in ("NOT_PLAYED", "GAME_OVER")
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
                "agent": "project_override_v3",
                "x": payload.get("x", 32),
                "y": payload.get("y", 32),
            }
        elif hasattr(action_obj, "is_simple") and action_obj.is_simple():
            action_obj.reasoning = f"project_override_v3 selected ACTION{act_id}"

        return action_obj
