import unittest
from project_override.config import AgentConfig
from project_override.execution.avo_supervisor import (
    AVOSupervisor,
    KeyframeSearchTree,
    SafeGameExecutor,
)
from project_override.llm.codeact_engine import SandboxedCodeActREPL, OfflineServerManager, parse_codeact_response
from project_override.memory.action_ledger import ActionLedger, compute_arena_hash
from project_override.memory.hypothesis_store import HypothesisStore
from project_override.orchestration.general_agent import GeneralAgent
from project_override.perception.observation_processor import ObservationProcessor, HUDSequenceExtractor
from project_override.planning.general_explorer import MacroActionQueue, astar_grid
from project_override.planning.domain_solvers import (
    sokoban_solver,
    solve_circuit_flow,
    triangulate_beacon,
    escort_joint_astar,
    delayed_action_astar,
    solve_gf2_toggle,
    space_time_astar,
    dreyfus_wagner_steiner_tree,
)
from project_override.core.goal_induction_engine import compute_topological_shannon_entropy
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
