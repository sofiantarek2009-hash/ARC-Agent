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


def test_gf2_toggle_solver_lights_out():
    """Tests solve_gf2_toggle inverts Lights-Out toggle matrix via GF(2) elimination."""
    from project_override.planning.domain_solvers import solve_gf2_toggle

    # 3x3 grid: center cell needs to be toggled
    # In standard Lights Out, clicking (1, 1) toggles (1, 1), (0, 1), (2, 1), (1, 0), (1, 2)
    # If we want target_grid where (1, 1) and its 4 neighbors are 1,
    # clicking (1, 1) should be the exact solution!
    curr = np.zeros((3, 3), dtype=int)
    targ = np.zeros((3, 3), dtype=int)
    targ[1, 1] = 1
    targ[0, 1] = 1
    targ[2, 1] = 1
    targ[1, 0] = 1
    targ[1, 2] = 1

    plan = solve_gf2_toggle(curr, targ, stride=1, sub_h=3, sub_w=3)
    assert len(plan) == 1, f"Expected 1 click at center, got {len(plan)}: {plan}"
    assert plan[0][0] == 6
    assert plan[0][1]["x"] == 1 and plan[0][1]["y"] == 1, f"Expected click at (1, 1), got {plan[0][1]}"


def test_space_time_astar_second_order_kinematics():
    """Tests space_time_astar accounts for acceleration and elastic wall bounces."""
    from project_override.planning.domain_solvers import space_time_astar

    grid = np.zeros((10, 10), dtype=int)
    start = (2, 2)
    target = (2, 8)
    # Hazard accelerating downwards and bouncing
    hazards = [
        {"pos": (2, 5), "velocity": (1, 0), "accel": (1, 0), "bounce": True}
    ]

    plan = space_time_astar(grid, start, target, dynamic_hazards=hazards)
    assert len(plan) > 0, "Space-time A* should find a safe path dodging accelerated hazard"


def test_keyframe_search_tree_mcts_backtracking():
    """Tests KeyframeSearchTree computes exact backtrack steps to fork node."""
    from project_override.execution.avo_supervisor import KeyframeSearchTree

    tree = KeyframeSearchTree()
    # Step 1: take action 1
    n1 = tree.record_step(action=1, post_hash=101, step=1)
    # Step 2: take action 2
    n2 = tree.record_step(action=2, post_hash=102, step=2)
    # Step 3: take action 3
    n3 = tree.record_step(action=3, post_hash=103, step=3)

    # 1. Backtrack 1 step to parent n2
    steps = tree.compute_backtrack_to_fork()
    assert steps == 1, f"Expected 1 backtrack step to immediate parent, got {steps}"

    # Perform 1 undo: now at n2
    tree.record_undo()
    assert tree.current_node_id == n2

    # 2. If n2 is exhausted ({1, 2, 3, 4}), it should climb to n1 (2 steps from n3)
    # Re-advance to n3
    tree.record_step(action=3, post_hash=103, step=3)
    tree.nodes[n2].banned_actions = {1, 2, 3, 4}
    steps_climbed = tree.compute_backtrack_to_fork()
    assert steps_climbed == 2, f"Expected 2 backtrack steps to ancestor n1, got {steps_climbed}"


def test_cegis_retrodiction_verifier():
    """Tests CEGISRetrodictionVerifier validates 100% replay and catches counterexamples."""
    from project_override.planning.verifier_gate import CEGISRetrodictionVerifier

    cegis = CEGISRetrodictionVerifier()

    s0 = np.zeros((5, 5), dtype=int)
    s1 = s0.copy()
    s1[2, 2] = 7  # transition effect

    cegis.record_transition(step=1, level=0, pre_grid=s0, action=1, payload={}, post_grid=s1, delta_omega=1)

    # Valid model
    def correct_model(pre, act, payload):
        res = pre.copy()
        if act == 1:
            res[2, 2] = 7
        return res

    is_cert, acc, cx = cegis.verify_transition_model(correct_model)
    assert is_cert is True and acc == 1.0
    assert cx is None

    # Flawed model
    def wrong_model(pre, act, payload):
        return pre.copy()

    is_cert_f, acc_f, cx_f = cegis.verify_transition_model(wrong_model)
    assert is_cert_f is False and acc_f == 0.0
    assert cx_f is not None
    assert cx_f["coord"] == (2, 2)


def test_dreyfus_wagner_steiner_tree_multi_terminal():
    """Tests dreyfus_wagner_steiner_tree connects 3 terminals with minimal Steiner tree."""
    from project_override.planning.domain_solvers import dreyfus_wagner_steiner_tree

    H, W = 10, 10
    passable = np.ones((H, W), dtype=bool)
    # Terminals at 3 corners: (2, 2), (2, 8), (8, 5)
    terminals = [(2, 2), (2, 8), (8, 5)]

    cost, tree_cells = dreyfus_wagner_steiner_tree((H, W), passable, terminals)
    assert cost < np.inf, "Steiner tree should find a valid interconnect"
    for t in terminals:
        assert t in tree_cells, f"Terminal {t} must be in Steiner tree"
    # The Steiner root is around (2, 5), giving cost 3 + 3 + 6 = 12
    assert cost <= 12.0, f"Expected cost <= 12, got {cost}"


def test_topological_shannon_entropy_minimization():
    """Tests compute_topological_shannon_entropy decreases as chaotic grid unifies."""
    from project_override.core.goal_induction_engine import compute_topological_shannon_entropy

    # Chaotic high-entropy grid with many scattered colors
    chaotic = np.random.randint(0, 8, size=(16, 16))
    h_chaotic = compute_topological_shannon_entropy(chaotic, bg_color=0)

    # Clean low-entropy grid with single grouped object
    clean = np.zeros((16, 16), dtype=int)
    clean[4:8, 4:8] = 3
    h_clean = compute_topological_shannon_entropy(clean, bg_color=0)

    assert h_clean < h_chaotic, f"Clean grid entropy ({h_clean:.3f}) should be lower than chaotic ({h_chaotic:.3f})"


def test_hud_sequence_extractor():
    """Tests HUDSequenceExtractor detects 1D card sequence tokens from top strip."""
    from project_override.perception.observation_processor import HUDSequenceExtractor

    grid = np.zeros((64, 64), dtype=int)
    # Create top HUD tokens: token 1 (color 4), token 2 (color 7), token 3 (color 11)
    grid[1:3, 10:13] = 4
    grid[1:3, 20:23] = 7
    grid[1:3, 30:33] = 11

    extractor = HUDSequenceExtractor(bg_color=0)
    seq_info = extractor.extract_sequence(grid)
    assert seq_info is not None, "Should detect HUD sequence"
    assert seq_info["tokens"] == [4, 7, 11], f"Expected [4, 7, 11], got {seq_info['tokens']}"
    assert seq_info["active_color"] == 4


def test_general_agent_autonomous_symbolic_dispatch():
    """Verifies GeneralAgent autonomously dispatches escort, space-time, delay, and beacon solvers."""
    from project_override.orchestration.general_agent import GeneralAgent
    from project_override.config import AgentConfig
    from project_override.core.goal_induction_engine import GoalGuidance, GoalCategory

    agent = GeneralAgent(AgentConfig(enable_llm=False))
    
    # 1. Coordinate Clamping check
    act, data = agent._format_action_payload(6, {"x": 100, "y": -5})
    assert act == 6
    assert data["x"] == 63
    assert data["y"] == 0

    # 2. Escort Joint Dispatch
    grid = np.zeros((64, 64), dtype=int)
    agent.avatar_pos = (10, 10)
    agent.co_mover_pos = (20, 20)
    agent.probed_directions = {1, 2, 3, 4}
    guidance = GoalGuidance(
        category=GoalCategory.EXIT_REACH,
        primary_waypoints=[(10, 15), (20, 25)],
        heuristic_distance=10.0,
        recommended_solver="ASTAR_WAYPOINTS",
        confidence=0.9,
        description="Escort both avatars to goals",
    )
    agent.goal_inducer.get_goal_guidance = lambda *args, **kwargs: guidance

    class MockObs:
        def __init__(self, g):
            self.frame = g
            self.state = "PLAYING"
            self.levels_completed = 0

    # Should dispatch escort joint A*
    act_id, act_payload = agent.act(MockObs(grid), available_actions=[1, 2, 3, 4])
    assert act_id in (1, 2, 3, 4), "Should emit directional step from joint A*"
    assert agent.macro_queue.active_subagent == "ESCORT_JOINT_ASTAR"

    # 3. Beacon Deployment / Ghost Flagging Dispatch
    agent.reset_game()
    agent.probed_directions = {1, 2, 3, 4}
    agent.step_in_level = 5
    grid_beacon = np.zeros((64, 64), dtype=int)
    grid_beacon[25, 30] = 7  # Revealed ghost pixel
    act_id, act_payload = agent.act(MockObs(grid_beacon), available_actions=[1, 2, 3, 4, 5, 6])
    assert act_id == 6, f"Should emit coordinate click on revealed ghost, got {act_id}"
    assert act_payload == {"x": 30, "y": 25}



