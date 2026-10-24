import unittest
import numpy as np
from project_override.llm.local_runtime import parse_llm_response
from project_override.llm.repl import SandboxedREPL
from project_override.memory.hypothesis_store import HypothesisStore
from project_override.perception.observation_processor import ObservationProcessor
from project_override.planning.planner import EmpiricalWorldModel, MacroActionQueue, astar_grid

def test_astar_and_franzen_action_guard() -> None:
    vectors = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    walls = {(1, 0), (1, 1)}
    path = astar_grid((0, 0), (2, 0), walls, vectors, grid_shape=(8, 8))
    assert path is not None and len(path) >= 4
    mq = MacroActionQueue()
    mq.push_plan([(1, {}, 12345), (2, {}, 99999)], subagent_tag="REPL")
    first = mq.pop_next(last_delta_omega=5, guard_enabled=True, current_hash=12345)
    assert first == (1, {})
    second = mq.pop_next(last_delta_omega=5, guard_enabled=True, current_hash=11111)
    assert second is None

def test_duck_repl_and_world_model_backtest() -> None:
    raw_resp = (
        "We should move right twice and record a note.\n"
        "```python\nnote('rule', 'push_box')\nact(4)\nact(4)\n```"
    )
    parsed = parse_llm_response(raw_resp)
    assert parsed is not None
    assert parsed["subgoal"] == "CODEACT"
    repl = SandboxedREPL()
    notes = {}
    plan = repl.execute_codeact(
        code_str=parsed["code"],
        grid=np.zeros((64, 64), dtype=np.int8),
        prev_grid=None,
        av_pos=(10, 10),
        bg_color=0,
        entities=[],
        vectors={4: (0, 1)},
        avail_actions=[1, 2, 3, 4],
        notes=notes,
    )
    assert plan == [(4, {}), (4, {})]
    assert notes.get("rule") == "push_box"
    proc = ObservationProcessor()
    store = HypothesisStore()
    store.displacement_vectors[4] = (0, 1)
    store.avatar_colors.add(2)
    wm = EmpiricalWorldModel(proc, store)
    g0 = np.zeros((64, 64), dtype=np.int8)
    g0[10, 10] = 2
    g1 = wm.simulate_step(g0, 4)
    assert int(g1[10, 11]) == 2 and int(g1[10, 10]) == 0

class TestPlanner(unittest.TestCase):
    def test_planner_suite(self) -> None:
        test_astar_and_franzen_action_guard()
        test_duck_repl_and_world_model_backtest()
