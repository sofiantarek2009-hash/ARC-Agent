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
