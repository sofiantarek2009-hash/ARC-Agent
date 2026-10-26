import numpy as np
import pytest
from project_override.planning.general_explorer import MacroActionQueue
from project_override.llm.codeact_engine import SandboxedCodeActREPL
from project_override.core.goal_induction_engine import (
    DeltaGoalInductionEngine,
    GoalCategory,
    SolvedLevelMemory,
)
from project_override.core.simulated_planner import DeadlockDetector


def test_upgrade1_space_time_astar_hazard_dodge():
    repl = SandboxedCodeActREPL()
    # 5x5 grid (stride=1) where bg=0 (wall is 0, floor is 1)
    grid = np.ones((5, 5), dtype=int)
    grid[0, :] = 0
    grid[4, :] = 0
    grid[:, 0] = 0
    grid[:, 4] = 0

    # Dynamic hazard falling down column 2: at t=0 it is at (1, 2), at t=1 at (2, 2)
    hazards = [{"pos": (1, 2), "velocity": (1, 0)}]

    ctx = {
        "grid": grid,
        "stride": 1,
        "bg_color": 0,
        "action_map": {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4},
        "available_actions": [1, 2, 3, 4, 5],
        "moving_hazards": hazards,
    }

    # Agent at (2, 1) wants to reach (2, 3) crossing column 2
    code = """
plan = find_path((2, 1), (2, 3))
for a in plan:
    act(a)
"""
    actions = repl.execute_codeact(code, ctx)
    assert len(actions) == 3
    # Step 1: Wait (Action 5) for hazard to pass
    assert actions[0][0] == 5
    # Step 2: East (Action 4) to (2, 2)
    assert actions[1][0] == 4
    # Step 3: East (Action 4) to (2, 3)
    assert actions[2][0] == 4


def test_upgrade2_delta_goal_induction():
    inducer = DeltaGoalInductionEngine()
    s_0 = np.zeros((10, 10), dtype=int)
    s_0[2, 2] = 5
    s_0[3, 3] = 5
    s_0[1, 1] = 1 # mover

    # Avatar moves onto (2, 2), item 5 disappears
    s_1 = s_0.copy()
    s_1[1, 1] = 0
    s_1[2, 2] = 1 # avatar at (2, 2), color 5 disappeared!

    inducer.observe_transition(prev_grid=s_0, curr_grid=s_1, action=2, mover_curr=(2, 2), level_cleared=False)
    guidance = inducer.get_goal_guidance(s_1, mover_pos=(2, 2))
    assert guidance.category == GoalCategory.CONSUMPTION
    assert len(guidance.primary_waypoints) > 0


def test_upgrade3_closed_loop_sahu_rollback():
    q = MacroActionQueue()
    q.push_plan([(1, {}), (2, {}), (3, {})], expected_hashes=[100, 200, 300])
    assert len(q) == 3

    # First pop succeeds
    act = q.pop_next(last_delta_omega=1, curr_state_hash=100)
    assert act == (1, {})
    assert len(q) == 2

    # Second pop: unexpected NO-OP (delta_omega == 0) triggers instant Sahu rollback!
    act2 = q.pop_next(last_delta_omega=0, curr_state_hash=200)
    assert act2 is None
    assert len(q) == 0 # Queue is completely purged!


def test_upgrade4_pre_push_deadlock_detection():
    # 5x5 grid with concave corner at (1, 1) formed by walls at (0, 1) and (1, 0)
    walls = {(0, 1), (1, 0), (0, 0), (0, 2), (2, 0)}
    goal_plates = {(3, 3)}
    assert DeadlockDetector.is_concave_corner((1, 1), walls, goal_plates, 5, 5) is True
    # Non-corner cell (2, 2)
    assert DeadlockDetector.is_concave_corner((2, 2), walls, goal_plates, 5, 5) is False


def test_upgrade5_solved_level_zero_token_replay():
    from project_override.orchestration.general_agent import GeneralAgent
    from project_override.config import AgentConfig

    cfg = AgentConfig()
    cfg.enable_llm = False # LLM is offline, replay must work with 0 tokens!
    agent = GeneralAgent(config=cfg)
    agent.current_level = 2 # Level 2!
    agent.avatar_pos = (5, 5)
    agent.detected_stride = 1

    # Fake 10x10 observation
    grid = np.ones((10, 10), dtype=int)
    grid[0, :] = 0
    grid[9, :] = 0
    grid[:, 0] = 0
    grid[:, 9] = 0
    grid[5, 5] = 2 # player
    grid[5, 8] = 3 # target

    class FakeObs:
        frame = [grid]
        levels_completed = 2

    act, data = agent.act(FakeObs(), available_actions=[1, 2, 3, 4, 5])
    assert act in (1, 2, 3, 4)
    # The macro queue should have buffered the remaining path steps!
    assert len(agent.macro_queue) > 0
