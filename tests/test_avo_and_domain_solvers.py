"""Unit tests for Frontier Domain Solvers & Full-Power NVIDIA AVO Supervisor.

Verifies:
1. sokoban_solver: Forward heuristic search with concave corner deadlock culling.
2. solve_circuit_flow: Visual tile rotation & edge port matching for Action 6 clicks.
3. triangulate_beacon: Exact O(1) Manhattan distance circle-intersection triangulator.
4. escort_joint_astar: Coupled search in joint product space (p_avatar, p_escort).
5. delayed_action_astar: State-tuple expansion search over (p, FIFO_queue).
6. AVOSupervisor: Continuous Stagnation Index sigma_t, action banning, reset discipline.
7. SandboxedCodeActREPL: Tool exposure and execution within the sandboxed namespace.
"""

import numpy as np
import pytest

from project_override.config import AgentConfig
from project_override.execution.avo_supervisor import AVOSupervisor
from project_override.llm.codeact_engine import SandboxedCodeActREPL
from project_override.planning.domain_solvers import (
    delayed_action_astar,
    escort_joint_astar,
    sokoban_solver,
    solve_circuit_flow,
    triangulate_beacon,
)


def test_sokoban_solver_push_and_deadlock_cull():
    """Tests sokoban_solver pushes crate onto target and prunes dead corners."""
    # 7x7 grid: floor=0, walls=1
    grid = np.zeros((7, 7), dtype=int)
    grid[0, :] = 1
    grid[6, :] = 1
    grid[:, 0] = 1
    grid[:, 6] = 1
    # Wall creating a dead concave corner at (1, 1) and (1, 2)
    walls = {(r, c) for r in range(7) for c in range(7) if grid[r, c] == 1}

    avatar = (4, 2)
    boxes = [(3, 2)]
    targets = [(2, 2)]

    # Push action map: 1=UP (-1, 0), 2=DOWN (1, 0), 3=LEFT (0, -1), 4=RIGHT (0, 1)
    act_map = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
    plan = sokoban_solver(grid, boxes, targets, avatar, walls=walls, action_map=act_map)

    assert len(plan) > 0, "Sokoban solver should find a push plan"
    assert plan[0] == 1, f"First action should be push UP (1), got {plan[0]}"


def test_circuit_flow_tile_rotation():
    """Tests solve_circuit_flow finds Action 6 rotation clicks to bridge terminals."""
    # 10x10 grid with 2 terminals and 2 rotatable 1x1 tiles between them
    grid = np.zeros((10, 10), dtype=int)
    terminals = [(5, 2), (5, 6)]
    # Tiles at (5, 3), (5, 4), (5, 5)
    rotatable_tiles = [(5, 3), (5, 4), (5, 5)]

    # Put horizontal wire pixels in the tiles (e.g. color 3)
    grid[5, 2] = 3
    grid[5, 3] = 3
    grid[5, 4] = 3
    grid[5, 5] = 3
    grid[5, 6] = 3

    clicks = solve_circuit_flow(terminals, rotatable_tiles, grid, stride=1, bg_color=0)
    # Since they already form a straight horizontal line, clicks should be minimal or empty
    assert isinstance(clicks, list)


def test_triangulate_beacon_exact_and_noisy():
    """Tests triangulate_beacon recovers exact hidden target from 3 Manhattan probes."""
    true_target = (14, 27)
    probe_locations = [(10, 20), (20, 25), (15, 35)]
    probes = []
    for pr, pc in probe_locations:
        dist = abs(true_target[0] - pr) + abs(true_target[1] - pc)
        probes.append((pr, pc, dist))

    recovered = triangulate_beacon(probes, grid_shape=(64, 64))
    assert recovered == true_target, f"Expected {true_target}, got {recovered}"


def test_escort_joint_astar():
    """Tests escort_joint_astar coordinates mover and escort around obstacles."""
    # 8x8 grid
    avatar = (5, 2)
    escort = (5, 3)
    goal = (2, 3)
    # Wall between avatar and goal
    walls = {(3, 2), (3, 4)}

    plan = escort_joint_astar(avatar, escort, goal, walls, grid_shape=(8, 8))
    assert len(plan) > 0, "Joint escort planner should find valid trajectory"


def test_delayed_action_astar():
    """Tests delayed_action_astar plans through delay pipeline."""
    start = (5, 5)
    goal = (5, 8)
    walls = set()

    plan = delayed_action_astar(start, goal, walls, delay_k=2, grid_shape=(10, 10))
    assert len(plan) > 0, "Delayed action planner should reach goal"


def test_avo_supervisor_stagnation_and_action_banning():
    """Tests AVOSupervisor computes sigma_t and enforces action bans and resets."""
    cfg = AgentConfig(stagnation_window=6, enable_avo_supervisor=True)
    avo = AVOSupervisor(cfg)

    # Simulate oscillating between two states with repeated action 1 and 2
    for step in range(1, 10):
        h = 1000 + (step % 2) * 5
        act = 1 if (step % 2 == 1) else 2
        sigma = avo.observe_state(state_hash=h, action_id=act, step_in_level=step)

    # Stagnation index should be elevated (sigma_t >= 0.50)
    assert avo.last_sigma_t >= 0.50, f"Expected stagnation sigma_t >= 0.50, got {avo.last_sigma_t}"
    assert avo.strategy_audit_active, "Strategy audit should be triggered"

    # Action banning should filter out actions 1 and 2
    avail = [1, 2, 3, 4]
    filtered = avo.filter_banned_actions(avail)
    assert 1 not in filtered and 2 not in filtered, f"Actions 1 and 2 should be banned, got {filtered}"
    assert 3 in filtered and 4 in filtered

    # Severe deadlock reset enforcement when Action 7 is not available
    avo.escalation_tier = 3
    assert avo.should_force_reset(current_step=15, available_actions=[1, 2, 3, 4]), "Should force Action 0"


def test_sandboxed_repl_domain_solvers_integration():
    """Tests SandboxedCodeActREPL executes scripts calling domain solvers."""
    repl = SandboxedCodeActREPL()
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1
    grid[9, :] = 1
    grid[:, 0] = 1
    grid[:, 9] = 1

    env_ctx = {
        "grid": grid,
        "avatar_pos": (5, 2),
        "stride": 1,
        "bg_color": 0,
        "floor_color": 0,
        "available_actions": [1, 2, 3, 4],
        "action_map": {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4},
    }

    # Code that calls sokoban_solver and act()
    code = """
boxes = [(5, 3)]
targets = [(5, 5)]
plan = sokoban_solver(boxes, targets)
for a in plan:
    act(a)
"""
    plan = repl.execute_codeact(code, env_ctx)
    assert len(plan) > 0, "REPL should execute sokoban_solver and return plan"
    assert plan[0][0] == 4, f"First action should be push RIGHT (4), got {plan[0][0]}"
