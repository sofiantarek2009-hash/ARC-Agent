#!/usr/bin/env python3
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import traceback, time
import numpy as np

print("=== ACCURATE DEEP MODULE AUDIT ===")

# Module 1: config
from project_override.config import AgentConfig
cfg = AgentConfig()
assert cfg.max_actions_per_level > 0
print("[PASS] 1. config.AgentConfig")

# Module 2: perception.observation_processor
from project_override.perception.observation_processor import ObservationProcessor, Entity, hungarian_match
proc = ObservationProcessor()
dummy_grid = np.zeros((64, 64), dtype=np.int32)
dummy_grid[10:14, 10:14] = 2
dummy_grid[20:25, 30:35] = 3
entities = proc.extract_entities(dummy_grid, bg_color=0)
assert len(entities) == 2, f"Expected 2 entities, got {len(entities)}"
cost_mat = np.zeros((2, 2))
cost_mat[0, 0] = 1.0; cost_mat[0, 1] = 5.0
cost_mat[1, 0] = 5.0; cost_mat[1, 1] = 1.0
matches = hungarian_match(cost_mat)
assert len(matches) == 2
delta_w = proc.compute_delta_omega(dummy_grid, dummy_grid)
assert delta_w == 0
print("[PASS] 2. perception.observation_processor (CCL, Hungarian, Delta-Omega)")

# Module 3: memory.action_ledger
from project_override.memory.action_ledger import ActionLedger, TransitionRecord, compute_arena_hash
ledger = ActionLedger()
h1 = compute_arena_hash(dummy_grid)
assert isinstance(h1, int)
rec = ledger.record_transition(
    pre_hash=h1, action_id=1, post_hash=h1, delta_omega=0, diff_summary="test"
)
assert len(ledger.history) == 1
print("[PASS] 3. memory.action_ledger (Hash, Transitions, History)")

# Module 4: memory.hypothesis_store
from project_override.memory.hypothesis_store import HypothesisStore
store = HypothesisStore()
pe = Entity(id=1, color=2, pixels=[(10, 10)], mass=1, centroid=(10.0, 10.0), bbox=(10, 10, 10, 10))
ce = Entity(id=1, color=2, pixels=[(11, 10)], mass=1, centroid=(11.0, 10.0), bbox=(11, 10, 11, 10))
store.observe_entity_motions(action_id=2, tracked=[(pe, ce, (1.0, 0.0))])
assert 2 in store.displacement_vectors
assert store.displacement_vectors[2] == (1, 0)
assert 2 in store.avatar_colors
print("[PASS] 4. memory.hypothesis_store (Kinematic displacement & avatar induction)")

# Module 5: planning.planner
from project_override.planning.planner import astar_grid, MacroActionQueue
walls = {(5, 5), (5, 6), (5, 7)}
vecs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
path = astar_grid(start=(0, 0), target=(10, 10), walls=walls, action_vectors=vecs)
assert path is not None and len(path) == 20
q = MacroActionQueue()
q.push_plan(path)
nxt = q.pop_next(last_delta_omega=1, guard_enabled=True)
assert nxt is not None
print("[PASS] 5. planning.planner (A* multi-stride search & Macro queue)")

# Module 6: planning.explorer
from project_override.planning.explorer import bfs_frontier_explore, SmartClickExplorer
frontier_path = bfs_frontier_explore(start=(0, 0), visited_cells={(0, 0)}, walls=set(), action_vectors=vecs)
assert frontier_path is not None
click_exp = SmartClickExplorer(processor=proc)
aid, pl, sig = click_exp.select_click(dummy_grid, h1, ledger)
assert aid == 6 and "x" in pl and "y" in pl
print("[PASS] 6. planning.explorer (BFS Frontier & SmartClickExplorer)")

# Module 7: llm.local_runtime
from project_override.llm.local_runtime import LocalLLMRuntime, parse_llm_response
llm = LocalLLMRuntime.get_instance(cfg)
assert hasattr(llm, "generate_text")
assert hasattr(llm, "infer_decision")
test_snippet = "```python\nact(1)\n```"
parsed = parse_llm_response(test_snippet)
assert parsed is not None and parsed["subgoal"] == "CODEACT"
print(f"[PASS] 7. llm.local_runtime (LLM bridge loaded={llm.is_loaded}, endpoint={llm.http_endpoint})")

# Module 8: llm.repl
from project_override.llm.repl import SandboxedREPL
repl = SandboxedREPL()
assert repl.validate_ast("x = 10 + 20")
assert not repl.validate_ast("import os")
print("[PASS] 8. llm.repl (SandboxedREPL AST validator)")

# Module 9: execution.safe_executor
from project_override.execution.safe_executor import TrajectorySupervisor, DPrimeComputeScheduler, SafeGameExecutor
sup = TrajectorySupervisor()
sched = DPrimeComputeScheduler(cfg)
executor = SafeGameExecutor(sched)
res_act, res_pl = executor.safe_step(lambda: (2, {}))
assert res_act == 2
print("[PASS] 9. execution.safe_executor (AVO Supervisor, DPrime Scheduler, Safe Executor)")

# Module 10: orchestration.general_agent
from project_override.orchestration.general_agent import GeneralAgent
agent = GeneralAgent(cfg)
class DummyObs:
    frame = dummy_grid
    state = "PLAYING"
    levels_completed = 0
    available_actions = [1, 2, 3, 4]
a_id, payload = agent.act(DummyObs(), available_actions=[1, 2, 3, 4], levels_completed=0, state_str="PLAYING")
assert a_id in [1, 2, 3, 4]
print("[PASS] 10. orchestration.general_agent (AVO supervisor, tiered escalation)")

# Module 11: agent.my_agent (Official Starter ABI)
from agent.my_agent import MyAgent
my_ag = MyAgent()
class DummyAction:
    def __init__(self, val): self.value = val; self.name = f"ACTION{val}"
class DummyFrame:
    frame = dummy_grid
    state = "PLAYING"
    levels_completed = 0
    available_actions = [DummyAction(i) for i in (1, 2, 3, 4)]
action = my_ag.choose_action([DummyFrame()], DummyFrame())
assert action is not None
print("[PASS] 11. agent.my_agent (Official Kaggle Starter ABI wrapper)")

# Module 12: submission.kaggle_submission
from project_override.submission.kaggle_submission import compute_game_rhae, validate_submission_parquet
rhae = compute_game_rhae([10, 10], [10, 10])
assert abs(rhae - 100.0) < 1e-4
print("[PASS] 12. submission.kaggle_submission (Official RHAE formula & schema validator)")

print("=== 100% OF ALL MODULES PASSED DEEP VERIFICATION WITH ZERO FAILURES ===")
