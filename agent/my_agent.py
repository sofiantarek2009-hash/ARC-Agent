"""ARC-AGI-3 Official Starter Kit Compatible Agent: MyAgent
Drop-in replacement for `agent/my_agent.py` in ARC-AGI-3-Kaggle-Starter.

Connects the competition runtime to Project Override:
- Zero-NN Perception with 4-cell border timer mask
- Dead-Signature Bitwise Culling (O(1) null transition avoidance)
- Discrete Kinematic Physics & Sokoban Intercept Geometry
- Epistemic-Damped UCT Macro Planning
- Solved-Level Memory (Siriki M85)
- Dynamic System Pacing watchdog for 9-hour Kaggle offline budget
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Add project_override directory to path if needed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = CURRENT_DIR
PROJECT_DIR = os.path.join(CURRENT_DIR, "project_override")
for p in [CURRENT_DIR, PARENT_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Framework Imports & Fallback Definitions
# ---------------------------------------------------------------------------
try:
    from arcengine import GameAction, GameState, FrameData
except ImportError:
    # Standard ARC-AGI-3 Engine Enums & Structures Fallback
    class GameAction:
        RESET = 0
        ACTION1 = 1  # Up
        ACTION2 = 2  # Down
        ACTION3 = 3  # Left
        ACTION4 = 4  # Right
        ACTION5 = 5  # Interact / Select
        ACTION6 = 6  # Coordinate Click (x, y)
        ACTION7 = 7  # Undo

        def __init__(self, action_id: int, data: Optional[Dict[str, Any]] = None) -> None:
            self.id = action_id
            self.data = data or {}

        def set_data(self, data: Dict[str, Any]) -> GameAction:
            self.data = data
            return self

        def __repr__(self) -> str:
            return f"GameAction(id={self.id}, data={self.data})"

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, GameAction):
                return self.id == other.id and self.data == other.data
            return self.id == other

    class GameState:
        NOT_PLAYED = "NOT_PLAYED"
        PLAYING = "PLAYING"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class FrameData:
        def __init__(
            self,
            frame: Any,
            state: str = GameState.PLAYING,
            levels_completed: int = 0,
            win_levels: int = 1,
            available_actions: Optional[List[int]] = None,
            game_id: str = "default_game",
        ) -> None:
            self.frame = frame
            self.state = state
            self.levels_completed = levels_completed
            self.win_levels = win_levels
            self.available_actions = available_actions or [1, 2, 3, 4, 5, 6, 7]
            self.game_id = game_id

try:
    from agents.agent import Agent
except ImportError:
    class Agent:
        """Abstract base class matching ARC-AGI-3 Agents framework."""
        def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
            raise NotImplementedError

        def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
            raise NotImplementedError

# Import the core GeneralAgent (Zero-Overfitting Universal Engine)
from project_override.orchestration.general_agent import GeneralAgent as ProjectOverrideAgent


# ---------------------------------------------------------------------------
# Official Starter Kit Agent Implementation
# ---------------------------------------------------------------------------
class MyAgent(Agent):
    """Project Override Competition Agent for ARC-AGI-3.
    Subclasses official `Agent` to integrate into ARC-AGI-3-Kaggle-Starter.
    """

    def __init__(
        self,
        total_budget: float = 32400.0,
        total_games: int = 135,
        max_level_timeout: float = 35.0,
    ) -> None:
        super().__init__()
        self.orchestration_agent = ProjectOverrideAgent(
            total_budget=total_budget,
            total_games=total_games,
            max_level_timeout=max_level_timeout,
        )
        self.current_game_id: Optional[str] = None
        self.last_level_index: int = 0
        self.actions_in_current_level: int = 0
        self.level_actions_history: List[int] = []

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Determines when the agent has completed the current game/level."""
        if latest_frame is None:
            return True
        # Check if the environment has reached WIN state
        state_str = str(getattr(latest_frame, "state", "")).upper()
        if "WIN" in state_str or latest_frame.state is getattr(GameState, "WIN", "WIN"):
            return True

        # Check if levels completed reached target win levels
        completed = getattr(latest_frame, "levels_completed", 0)
        target = getattr(latest_frame, "win_levels", 1)
        if target is not None and completed >= target:
            return True

        # System budget watchdog check
        if self.orchestration_agent.pacer.is_level_timed_out:
            return True

        return False

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Inspects current frame state and returns the optimal GameAction."""
        state = getattr(latest_frame, "state", None)

        # 1. Handle Game Initialization / Restart
        if state in [getattr(GameState, "NOT_PLAYED", "NOT_PLAYED"), getattr(GameState, "GAME_OVER", "GAME_OVER")] or str(state) in ["GameState.GAME_OVER", "GAME_OVER"]:
            if state == getattr(GameState, "GAME_OVER", "GAME_OVER") or str(state) in ["GameState.GAME_OVER", "GAME_OVER"]:
                self.orchestration_agent.start_new_level(self.last_level_index)
                self.actions_in_current_level = 0
                self.level_actions_history = []
            return getattr(GameAction, "RESET", GameAction.from_id(0) if hasattr(GameAction, "from_id") else 0)

        # 2. Track Game and Level Transitions
        game_id = getattr(latest_frame, "game_id", "default_game")
        if self.current_game_id != game_id:
            self.current_game_id = game_id
            self.orchestration_agent.start_new_game(game_id)
            self.last_level_index = 0
            self.level_actions_history = []

        current_level = getattr(latest_frame, "levels_completed", 0)
        if current_level > self.last_level_index:
            self.orchestration_agent.record_level_won(self.level_actions_history)
            self.last_level_index = current_level
            self.orchestration_agent.start_new_level(current_level)
            self.actions_in_current_level = 0
            self.level_actions_history = []
        elif current_level < self.last_level_index:
            self.last_level_index = current_level
            self.orchestration_agent.start_new_level(current_level)
            self.actions_in_current_level = 0
            self.level_actions_history = []

        self.actions_in_current_level += 1

        # 3. Extract Grid Observation (settled 2D frame)
        raw_frame = getattr(latest_frame, "frame", None)
        if raw_frame is None:
            return getattr(GameAction, "ACTION1", GameAction.from_id(1) if hasattr(GameAction, "from_id") else 1)

        if isinstance(raw_frame, list) and len(raw_frame) > 0:
            grid = np.array(raw_frame[-1], dtype=np.int32)
        elif isinstance(raw_frame, np.ndarray):
            grid = raw_frame[-1] if raw_frame.ndim == 3 else raw_frame
        else:
            grid = np.array(raw_frame, dtype=np.int32)
            if grid.ndim == 3:
                grid = grid[-1]

        # 4. Extract Legal Actions
        available = getattr(latest_frame, "available_actions", None)
        if available is None or len(available) == 0:
            legal_actions = [1, 2, 3, 4, 5, 6, 7]
        else:
            # Normalize legal actions to integers
            legal_actions = []
            for a in available:
                if isinstance(a, int):
                    legal_actions.append(a)
                elif hasattr(a, "id"):
                    legal_actions.append(a.id)
                elif hasattr(a, "value"):
                    legal_actions.append(a.value)
                else:
                    try:
                        legal_actions.append(int(a))
                    except (ValueError, TypeError):
                        pass

        # 5. Core Policy Query
        action_id, action_data = self.orchestration_agent.choose_action(
            current_grid=grid,
            available_actions=legal_actions,
            game_state=str(state),
            raw_frame=raw_frame,
        )

        # 6. Map to Target GameAction Structure
        if hasattr(GameAction, "from_id"):
            try:
                selected_action = GameAction.from_id(action_id)
            except Exception:
                selected_action = getattr(GameAction, "ACTION1", None)
        else:
            action_map = {
                0: getattr(GameAction, "RESET", 0),
                1: getattr(GameAction, "ACTION1", 1),
                2: getattr(GameAction, "ACTION2", 2),
                3: getattr(GameAction, "ACTION3", 3),
                4: getattr(GameAction, "ACTION4", 4),
                5: getattr(GameAction, "ACTION5", 5),
                6: getattr(GameAction, "ACTION6", 6),
                7: getattr(GameAction, "ACTION7", 7),
            }
            selected_action = action_map.get(action_id, getattr(GameAction, "ACTION1", 1))

        # For ACTION6, attach coordinate targeting data if present
        if action_id == 6 and action_data:
            if hasattr(selected_action, "action_data") and selected_action.action_data is not None:
                selected_action.action_data.x = int(action_data.get("x", 0))
                selected_action.action_data.y = int(action_data.get("y", 0))
            if hasattr(selected_action, "set_data"):
                selected_action.set_data(action_data)
            selected_action.data = dict(action_data)
        else:
            selected_action.data = None

        self.level_actions_history.append(action_id)
        return selected_action
