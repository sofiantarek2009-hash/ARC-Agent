import pytest
import numpy as np
import collections
import sys
import os

sys.path.insert(0, os.path.abspath("ARC3-Inference"))

from inference.utils.sim_helpers import (
    find_shortest_path,
    solve_pushable_crates,
    solve_bipartite_assignment,
    verify_retrodiction,
)

# ==============================================================================
# TEST SUITE 1: ADVERSARIAL MAZES & SLIDING CORRIDORS (Like tr87, ka59, tu93)
# ==============================================================================
def test_hard_spiral_labyrinth():
    """64x64 labyrinth with narrow spiral corridors and dead ends."""
    grid_shape = (64, 64)
    passable = set()
    r, c = 2, 2
    dr, dc = 0, 1
    visited = set()
    for _ in range(800):
        passable.add((r, c))
        visited.add((r, c))
        nr, nc = r + dr, c + dc
        if 2 <= nr < 62 and 2 <= nc < 62 and (nr, nc) not in visited:
            r, c = nr, nc
        else:
            dr, dc = dc, -dr
            r, c = r + dr, c + dc
            visited.add((r, c))
            passable.add((r, c))
    
    start = (2, 2)
    goals = {(r, c)}
    path = find_shortest_path(start, goals, passable, grid_shape)
    assert path is not None
    assert len(path) > 20

def test_hard_choke_point_corridor():
    """Navigation through a sequence of 1-cell bottleneck doors."""
    grid_shape = (64, 64)
    passable = set()
    for r in range(5, 55):
        for c in range(5, 55):
            passable.add((r, c))
    for r in range(5, 55):
        if r != 25: passable.discard((r, 20))
        if r != 40: passable.discard((r, 35))
        if r != 10: passable.discard((r, 45))
    
    start = (10, 10)
    goals = {(50, 50)}
    path = find_shortest_path(start, goals, passable, grid_shape)
    assert path is not None
    assert len(path) >= 80

# ==============================================================================
# TEST SUITE 2: MULTI-BOX COMPLEX SOKOBAN (Like lp85, ar25, m0r0)
# ==============================================================================
def test_hard_sokoban_multi_box():
    """Pushable crates in a constrained warehouse."""
    walls = set()
    for r in range(10):
        walls.add((r, 0))
        walls.add((r, 9))
    for c in range(10):
        walls.add((0, c))
        walls.add((9, c))
    
    player = (2, 2)
    boxes = [(3, 4), (5, 5)]
    targets = [(3, 7), (5, 7)]
    
    plan = solve_pushable_crates(player, boxes, targets, walls, grid_shape=(10, 10))
    assert plan is not None
    assert len(plan) > 0

def test_hard_sokoban_deadlock_rejection():
    """Should instantly detect unresolvable corner deadlocks and return empty list."""
    walls = {(0, 0), (0, 1), (1, 0)} # Concave corner at (1, 1)
    player = (3, 3)
    boxes = [(1, 1)] # In corner
    targets = [(5, 5)] # Goal elsewhere
    plan = solve_pushable_crates(player, boxes, targets, walls, grid_shape=(10, 10))
    assert plan == []

# ==============================================================================
# TEST SUITE 3: HIGH-DIMENSIONAL BIPARTITE MATCHING (Like vc33, cd82, sc25)
# ==============================================================================
def test_hard_hungarian_dense_mesh():
    """10 items to 10 targets optimal global assignment."""
    np.random.seed(42)
    sources = [tuple(x) for x in np.random.randint(5, 55, size=(10, 2))]
    targets = [tuple(x) for x in np.random.randint(5, 55, size=(10, 2))]
    
    mapping = solve_bipartite_assignment(sources, targets)
    assert len(mapping) == 10
    assigned_targets = set(mapping.values())
    assert len(assigned_targets) == 10
    
    total_cost = sum(abs(s[0]-t[0]) + abs(s[1]-t[1]) for s, t in mapping.items())
    assert total_cost > 0

# ==============================================================================
# TEST SUITE 4: MODEL RETRODICTION & CAUSAL VERIFICATION (Like g50t, tn36)
# ==============================================================================
def test_hard_retrodiction_sliding_mechanic():
    """Verifies a candidate world model on simulated ice/sliding transitions."""
    history = []
    pos = (10, 5)
    for _ in range(3):
        next_pos = (10, 20)
        history.append({
            "action": 4,
            "state": pos,
            "next_state": next_pos
        })
        pos = next_pos
    
    def sim_one_step(state, action):
        r, c = state
        if action == 4: return (r, c + 1)
        return state
    
    is_valid, _ = verify_retrodiction(sim_one_step, history)
    assert is_valid is False
    
    def sim_slider(state, action):
        r, c = state
        if action == 4: return (r, 20)
        return state
        
    is_valid_slider, _ = verify_retrodiction(sim_slider, history)
    assert is_valid_slider is True

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_hard_benchmark_sim.py"])

# ==============================================================================
# TEST SUITE 5: INVARIANT GEOMETRIC & KINEMATIC RAYCASTS (ALIEN GENERAL MECHANICS)
# ==============================================================================
def test_hard_kinematic_raycast_gravity():
    """Simulates gravity drop down until hitting obstacles or floor."""
    from inference.utils.sim_helpers import simulate_ray_trajectory
    start = (5, 10)
    gravity = (1, 0)
    obstacles = {(20, 10), (20, 11)} # Floor at row 20
    landing, traj = simulate_ray_trajectory(start, gravity, obstacles, grid_shape=(64, 64))
    assert landing == (19, 10), f"Expected landing at (19, 10), got {landing}"
    assert len(traj) == 14 # 19 - 5 = 14 steps taken

def test_hard_kinematic_raycast_boundary_floor():
    """Simulates raycast hitting outer boundary at row 63."""
    from inference.utils.sim_helpers import simulate_ray_trajectory
    start = (50, 5)
    gravity = (1, 0)
    obstacles = set()
    landing, traj = simulate_ray_trajectory(start, gravity, obstacles, grid_shape=(64, 64))
    assert landing == (63, 5), f"Expected landing at floor (63, 5), got {landing}"

def test_hard_manhattan_beacon_triangulation():
    """Triangulates hidden coordinate from 3 discrete distance readings."""
    from inference.utils.sim_helpers import triangulate_manhattan_beacon
    hidden_goal = (25, 40)
    # Reading 1 from (10, 10): dist = |25-10| + |40-10| = 15 + 30 = 45
    # Reading 2 from (50, 10): dist = |25-50| + |40-10| = 25 + 30 = 55
    # Reading 3 from (25, 60): dist = |25-25| + |40-60| = 0 + 20 = 20
    readings = [
        ((10, 10), 45),
        ((50, 10), 55),
        ((25, 60), 20),
    ]
    candidates = triangulate_manhattan_beacon(readings, grid_shape=(64, 64))
    assert hidden_goal in candidates
    assert len(candidates) == 1, f"Expected unique intersection point, got {candidates}"
