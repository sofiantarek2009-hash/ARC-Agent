import unittest
from project_override.config import AgentConfig
from project_override.execution.safe_executor import (
    DPrimeComputeScheduler,
    SafeGameExecutor,
    TrajectorySupervisor,
)
from project_override.llm.local_runtime import LocalLLMRuntime, parse_llm_response
from project_override.llm.repl import SandboxedREPL
from project_override.memory.action_ledger import ActionLedger, compute_arena_hash
from project_override.memory.hypothesis_store import BayesianExplorationTracker, HypothesisStore
from project_override.orchestration.general_agent import GeneralAgent
from project_override.perception.observation_processor import DynamicHUDMasker, ObservationProcessor
from project_override.planning.explorer import SmartClickExplorer, bfs_frontier_explore
from project_override.planning.planner import (
    EmpiricalWorldModel,
    MacroActionQueue,
    astar_grid,
    plan_sokoban_astar,
)
from project_override.submission.kaggle_submission import (
    REQUIRED_PARQUET_COLUMNS,
    compute_game_rhae,
    validate_submission_parquet,
    write_submission_parquet,
)

def test_all_modules_import_cleanly() -> None:
    agent = GeneralAgent(AgentConfig(enable_llm=False))
    assert agent is not None
    assert len(REQUIRED_PARQUET_COLUMNS) == 4
    assert compute_game_rhae([10, 12], [10, 12]) == 100.0

class TestImports(unittest.TestCase):
    def test_imports(self) -> None:
        test_all_modules_import_cleanly()
