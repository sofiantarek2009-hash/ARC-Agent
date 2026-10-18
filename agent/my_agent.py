"""ARC-AGI-3 Official Starter Kit Compatible Agent: MyAgent
Drop-in replacement for `agent/my_agent.py` in ARC-AGI-3-Kaggle-Starter.

Connects the competition runtime to Project Override:
- Zero-NN Perception with 4-cell border timer mask
- Dead-Signature Bitwise Culling (O(1) null transition avoidance)
- Discrete Kinematic Physics & Sokoban Intercept Geometry
- Epistemic-Damped UCT Macro Planning
- Solved-Level Memory (Siriki M85)
- Dynamic System Pacing watchdog for 9-hour Kaggle offline budget
- Defensive Exception & Coordinate Gating: Guarantees 0-crash thread execution
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Add project_override directory to path if needed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = CURRENT_DIR
PROJECT_DIR = os.path.join(CURRENT_DIR, "project_override")
for p in [CURRENT_DIR, PARENT_DIR, PROJECT_DIR, "/kaggle/working", "/kaggle/working/ARC-AGI-3-Agents", "/tmp"]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Framework Imports & Comprehensive Fallback Definitions
# ---------------------------------------------------------------------------
try:
    from arcengine import GameAction, GameState, FrameData
except ImportError:
    # Full standalone ARC-AGI-3 Engine Enums & Structures Fallback
    class ActionData:
        def __init__(self, x: int = 0, y: int = 0, game_id: str = "") -> None:
            self.x = int(x)
            self.y = int(y)
            self.game_id = str(game_id)

        def model_dump(self) -> Dict[str, Any]:
            return {"game_id": self.game_id, "x": self.x, "y": self.y}

    class GameAction:
        def __init__(self, action_id: int, name: str, is_complex: bool = False) -> None:
            self.value = action_id
            self.id = action_id
            self.name = name
            self._is_complex = is_complex
            self.action_data = ActionData()
            self.reasoning: Optional[Dict[str, Any]] = None
            self.data: Optional[Dict[str, Any]] = None

        def is_simple(self) -> bool:
            return not self._is_complex

        def is_complex(self) -> bool:
            return self._is_complex

        def set_data(self, data: Dict[str, Any]) -> Any:
            cx = max(0, min(63, int(data.get("x", 0)))) if data else 0
            cy = max(0, min(63, int(data.get("y", 0)))) if data else 0
            self.action_data = ActionData(x=cx, y=cy, game_id=data.get("game_id", ""))
            self.data = dict(data) if data else {}
            return self.action_data

        def __repr__(self) -> str:
            return f"GameAction.{self.name}"

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, GameAction):
                return self.value == other.value
            return self.value == other

    GameAction.RESET = GameAction(0, "RESET", is_complex=False)
    GameAction.ACTION1 = GameAction(1, "ACTION1", is_complex=False)
    GameAction.ACTION2 = GameAction(2, "ACTION2", is_complex=False)
    GameAction.ACTION3 = GameAction(3, "ACTION3", is_complex=False)
    GameAction.ACTION4 = GameAction(4, "ACTION4", is_complex=False)
    GameAction.ACTION5 = GameAction(5, "ACTION5", is_complex=False)
    GameAction.ACTION6 = GameAction(6, "ACTION6", is_complex=True)
    GameAction.ACTION7 = GameAction(7, "ACTION7", is_complex=False)

    _ALL_ACTIONS = [
        GameAction.RESET,
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
        GameAction.ACTION6,
        GameAction.ACTION7,
    ]

    @classmethod
    def _from_id(cls, action_id: int) -> GameAction:
        for a in _ALL_ACTIONS:
            if a.value == action_id:
                return a
        return GameAction.ACTION1

    GameAction.from_id = _from_id  # type: ignore[assignment]

    class GameState:
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class FrameData:
        def __init__(
            self,
            frame: Any = None,
            state: str = GameState.NOT_FINISHED,
            levels_completed: int = 0,
            win_levels: int = 1,
            available_actions: Optional[List[int]] = None,
            game_id: str = "default_game",
        ) -> None:
            self.frame = frame if frame is not None else [np.zeros((64, 64), dtype=int).tolist()]
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
        MAX_ACTIONS: int = 200

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.game_id = kwargs.get("game_id", "default_game")
            self.card_id = kwargs.get("card_id", "test_card")
            self.action_counter = 0
            self.frames = []

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

    MAX_ACTIONS: int = 200

    def __init__(
        self,
        *args: Any,
        total_budget: float = 32400.0,
        total_games: int = 135,
        max_level_timeout: float = 35.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.orchestration_agent = ProjectOverrideAgent(
            total_budget=total_budget,
            total_games=total_games,
            max_level_timeout=max_level_timeout,
        )
        self.current_game_id: Optional[str] = getattr(self, "game_id", None)
        self.last_level_index: int = 0
        self.actions_in_current_level: int = 0
        self.level_actions_history: List[int] = []
        if self.current_game_id:
            try:
                self.orchestration_agent.start_new_game(self.current_game_id)
            except Exception as e:
                logger.warning(f"Error starting new game {self.current_game_id}: {e}")

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Determines when the agent has completed the current game."""
        try:
            if latest_frame is None:
                return True

            state = getattr(latest_frame, "state", None)
            state_str = str(getattr(state, "name", state)).upper()

            # Check if environment reached WIN state
            if state is getattr(GameState, "WIN", None) or "WIN" in state_str:
                return True

            # Check if levels completed reached target win levels (when target > 0)
            completed = getattr(latest_frame, "levels_completed", 0)
            target = getattr(latest_frame, "win_levels", 1)
            if target is not None and isinstance(target, (int, float)) and target > 0 and completed >= target:
                return True

            return False
        except Exception as e:
            logger.warning(f"Error in is_done check: {e}")
            return False

    def _safe_coord(self, val: Any, default: int = 0) -> int:
        """Safely parses and clamps coordinate to [0, 63]."""
        try:
            if val is None:
                return default
            return max(0, min(63, int(val)))
        except (ValueError, TypeError):
            return default

    def _make_game_action(
        self,
        action_id: int,
        action_data: Optional[Dict[str, Any]] = None,
    ) -> GameAction:
        """Constructs an authentic GameAction instance guaranteed to pass engine validation."""
        action_data = action_data or {}
        selected_action: Optional[GameAction] = None

        if hasattr(GameAction, "from_id"):
            try:
                selected_action = GameAction.from_id(action_id)
            except Exception:
                selected_action = getattr(GameAction, "ACTION1", None)
        else:
            action_map = {
                0: getattr(GameAction, "RESET", None),
                1: getattr(GameAction, "ACTION1", None),
                2: getattr(GameAction, "ACTION2", None),
                3: getattr(GameAction, "ACTION3", None),
                4: getattr(GameAction, "ACTION4", None),
                5: getattr(GameAction, "ACTION5", None),
                6: getattr(GameAction, "ACTION6", None),
                7: getattr(GameAction, "ACTION7", None),
            }
            selected_action = action_map.get(action_id, getattr(GameAction, "ACTION1", None))

        if selected_action is None:
            selected_action = getattr(GameAction, "ACTION1", None)

        # Coordinate click action handling
        if action_id == 6:
            cx = self._safe_coord(action_data.get("x", 0))
            cy = self._safe_coord(action_data.get("y", 0))
            clean_data = {"x": cx, "y": cy}
            try:
                if hasattr(selected_action, "set_data"):
                    selected_action.set_data(clean_data)
                elif hasattr(selected_action, "action_data") and selected_action.action_data is not None:
                    selected_action.action_data.x = cx
                    selected_action.action_data.y = cy
            except Exception as e:
                logger.warning(f"Could not set action_data on ACTION6: {e}")
            selected_action.data = clean_data
        else:
            try:
                if hasattr(selected_action, "set_data"):
                    selected_action.set_data({})
            except Exception:
                pass
            selected_action.data = None

        # Attach standard reasoning metadata
        action_name = getattr(selected_action, "name", str(action_id))
        selected_action.reasoning = {
            "action": action_name,
            "step": self.actions_in_current_level,
        }

        # Defensive fallback: guarantee .action_data exists and has model_dump()
        if not hasattr(selected_action, "action_data") or selected_action.action_data is None:
            class _FallbackData:
                def __init__(self, data: Dict[str, Any]) -> None:
                    self._d = data
                def model_dump(self) -> Dict[str, Any]:
                    return self._d
            selected_action.action_data = _FallbackData(selected_action.data or {})

        return selected_action

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Inspects current frame state and returns the optimal GameAction.
        Defensive outer wrapper guarantees the thread never crashes under any exception.
        """
        try:
            return self._choose_action_impl(frames, latest_frame)
        except Exception as e:
            logger.error(f"Unhandled exception in MyAgent.choose_action for {self.current_game_id}: {e}", exc_info=True)
            # Safe legal fallback
            return self._make_game_action(1, {})

    def _choose_action_impl(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        state = getattr(latest_frame, "state", None)
        state_str = str(getattr(state, "name", state)).upper()

        # 1. Handle Game Initialization / Restart
        is_not_played = (
            state is getattr(GameState, "NOT_PLAYED", None)
            or state_str in ["NOT_PLAYED", "GAMESTATE.NOT_PLAYED"]
        )
        is_game_over = (
            state is getattr(GameState, "GAME_OVER", None)
            or state_str in ["GAME_OVER", "GAMESTATE.GAME_OVER"]
        )

        if is_not_played or is_game_over:
            if is_game_over:
                try:
                    self.orchestration_agent.start_new_level(self.last_level_index)
                except Exception as e:
                    logger.warning(f"Error resetting level after GAME_OVER: {e}")
                self.actions_in_current_level = 0
                self.level_actions_history = []
            return self._make_game_action(0, {})

        # 2. Track Game and Level Transitions
        game_id = getattr(latest_frame, "game_id", None) or getattr(self, "game_id", "default_game")
        if self.current_game_id != game_id and game_id:
            self.current_game_id = game_id
            try:
                self.orchestration_agent.start_new_game(game_id)
            except Exception as e:
                logger.warning(f"Error starting new game {game_id}: {e}")
            self.last_level_index = 0
            self.level_actions_history = []

        current_level = getattr(latest_frame, "levels_completed", 0)
        if current_level > self.last_level_index:
            try:
                self.orchestration_agent.record_level_won(self.level_actions_history)
            except Exception:
                pass
            self.last_level_index = current_level
            try:
                self.orchestration_agent.start_new_level(current_level)
            except Exception:
                pass
            self.actions_in_current_level = 0
            self.level_actions_history = []
        elif current_level < self.last_level_index:
            self.last_level_index = current_level
            try:
                self.orchestration_agent.start_new_level(current_level)
            except Exception:
                pass
            self.actions_in_current_level = 0
            self.level_actions_history = []

        self.actions_in_current_level += 1

        # 3. Extract Grid Observation (settled 2D frame with 64x64 guarantee)
        raw_frame = getattr(latest_frame, "frame", None)
        grid: Optional[np.ndarray] = None
        if raw_frame is not None:
            try:
                if isinstance(raw_frame, list) and len(raw_frame) > 0:
                    last_sub = raw_frame[-1]
                    grid = np.array(last_sub, dtype=np.int32)
                elif isinstance(raw_frame, np.ndarray):
                    grid = raw_frame[-1] if raw_frame.ndim == 3 else raw_frame
                else:
                    grid = np.array(raw_frame, dtype=np.int32)
                    if grid.ndim == 3:
                        grid = grid[-1]
            except Exception:
                grid = None

        # Guarantee a valid 2D 64x64 numpy array
        if grid is None or not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.shape != (64, 64):
            if isinstance(grid, np.ndarray) and grid.ndim == 2:
                padded = np.zeros((64, 64), dtype=np.int32)
                h = min(64, grid.shape[0])
                w = min(64, grid.shape[1])
                padded[:h, :w] = grid[:h, :w]
                grid = padded
            else:
                grid = np.zeros((64, 64), dtype=np.int32)

        # 4. Extract Legal Actions
        available = getattr(latest_frame, "available_actions", None)
        legal_actions: List[int] = []
        if available:
            for a in available:
                if isinstance(a, int):
                    legal_actions.append(a)
                elif hasattr(a, "id"):
                    legal_actions.append(int(a.id))
                elif hasattr(a, "value"):
                    legal_actions.append(int(a.value))
                else:
                    try:
                        legal_actions.append(int(a))
                    except (ValueError, TypeError):
                        pass

        # Exclude RESET from normal play options unless specifically requested
        legal_in_game = [a for a in legal_actions if a != 0]
        if not legal_in_game:
            legal_in_game = [1, 2, 3, 4, 5, 6, 7]

        # 5. Core Policy Query with Defensive Isolation
        try:
            action_id, action_data = self.orchestration_agent.choose_action(
                current_grid=grid,
                available_actions=legal_in_game,
                game_state=str(state),
                raw_frame=raw_frame,
            )
        except Exception as e:
            logger.warning(f"Exception inside orchestration_agent.choose_action: {e}", exc_info=True)
            action_id = legal_in_game[0] if legal_in_game else 1
            action_data = {}

        # 6. Ensure action_id is legal
        if action_id not in legal_in_game and legal_in_game:
            action_id = legal_in_game[0]

        # 7. Construct and return verified GameAction
        selected_action = self._make_game_action(action_id, action_data)
        self.level_actions_history.append(action_id)
        return selected_action
