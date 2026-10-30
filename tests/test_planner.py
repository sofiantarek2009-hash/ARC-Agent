import unittest
import numpy as np
from project_override.llm.codeact_engine import parse_codeact_response, SandboxedCodeActREPL
from project_override.planning.general_explorer import MacroActionQueue, astar_grid
from project_override.planning.domain_solvers import sokoban_solver

def test_astar_and_2d_sokoban_planner() -> None:
    vectors = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    walls = {(1, 0), (1, 1)}
    path = astar_grid((0, 0), (2, 0), walls, vectors, grid_shape=(8, 8))
    assert path is not None and len(path) >= 2
    
    grid = np.zeros((16, 16), dtype=int)
    soko_path = sokoban_solver(
        grid=grid,
        boxes=[(6, 7)],
        targets=[(8, 9)],
        avatar_pos=(5, 5),
        walls=set(),
        action_map={(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4},
    )
    assert soko_path is not None and len(soko_path) >= 4
    
    mq = MacroActionQueue()
    mq.push_plan([(1, {}), (2, {})], expected_hashes=[12345, 99999], subagent_tag="REPL")
    first = mq.pop_next(last_delta_omega=None, guard_enabled=True, curr_state_hash=12345)
    assert first == (1, {})
    second = mq.pop_next(last_delta_omega=0, guard_enabled=True, curr_state_hash=11111)
    assert second is None

def test_duck_repl_codeact() -> None:
    raw_resp = (
        "We should move right twice and record a note.\n"
        "```python\nnote('rule', 'push_box')\nact(4)\nact(4)\n```"
    )
    parsed = parse_codeact_response(raw_resp)
    assert parsed is not None
    assert parsed["subgoal"] == "CODEACT"
    repl = SandboxedCodeActREPL()
    notes = {}
    plan = repl.execute_codeact(
        code_str=parsed["code"],
        environment_context={"notes": notes},
    )
    assert plan == [(4, {"x": 0, "y": 0}), (4, {"x": 0, "y": 0})]
    assert notes.get("rule") == "push_box"

class TestPlanner(unittest.TestCase):
    def test_planner_suite(self) -> None:
        test_astar_and_2d_sokoban_planner()
        test_duck_repl_codeact()

    def test_symbolic_puct_planner(self) -> None:
        from inference.utils.symbolic_planner import StateEquivalenceAStarPlanner
        from inference.utils.sim_helpers import PurePythonAStarPlanner

        initial_grid = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 2],
        ]
        valid_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        # Transition model: moves 1 right on ACTION4, down on ACTION2
        def model_1(grid, action):
            g = [list(r) for r in grid]
            r, c = None, None
            for i in range(len(g)):
                for j in range(len(g[0])):
                    if g[i][j] == 1:
                        r, c = i, j
            if r is None:
                return g
            dr, dc = 0, 0
            if action == "ACTION4":
                dc = 1
            elif action == "ACTION2":
                dr = 1
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(g) and 0 <= nc < len(g[0]):
                g[r][c] = 0
                g[nr][nc] = 1
            return g

        def goal_fn(grid):
            # Goal is reached when 1 lands on (2, 2)
            return grid[2][2] == 1

        # Test StateEquivalenceAStarPlanner.solve_puct
        plan = StateEquivalenceAStarPlanner.solve_puct(
            initial_grid,
            valid_actions,
            [model_1],
            goal_fn,
            max_depth=10,
            num_simulations=50,
        )
        assert plan is not None
        assert len(plan) >= 2

        # Test PurePythonAStarPlanner.solve_puct
        pure_plan = PurePythonAStarPlanner.solve_puct(
            initial_grid,
            valid_actions,
            [model_1],
            goal_fn,
            max_depth=10,
            num_simulations=50,
        )
        assert pure_plan is not None
        assert len(pure_plan) >= 2

    def test_hierarchical_macro_options(self) -> None:
        from inference.utils.symbolic_planner import HierarchicalMacroOptions
        from inference.utils.sim_helpers import PurePythonMacroOptions

        passable = {(r, c) for r in range(5) for c in range(5)}
        fatal = {(1, 0), (1, 1)}

        # Test option_nav avoiding fatal cells
        nav_path = HierarchicalMacroOptions.option_nav(
            start_pos=(0, 0),
            target_pos=(2, 0),
            passable_cells=passable,
            fatal_cells=fatal,
            grid_shape=(5, 5),
        )
        assert nav_path is not None
        assert len(nav_path) >= 2

        pure_nav = PurePythonMacroOptions.option_nav(
            start_pos=(0, 0),
            target_pos=(2, 0),
            passable_cells=passable,
            fatal_cells=fatal,
            grid_shape=(5, 5),
        )
        assert pure_nav is not None
        assert len(pure_nav) >= 2

        # Test option_contact
        contact_path = HierarchicalMacroOptions.option_contact(
            start_pos=(0, 0),
            target_obj_cells={(2, 2)},
            approach_dir="LEFT",  # stand to right at (2, 3), push LEFT
            passable_cells=passable,
            grid_shape=(5, 5),
        )
        assert contact_path is not None
        assert len(contact_path) >= 2
        assert contact_path[-1] == "ACTION3"


