"""Project Override: Test Suite for World Model, AST Sandbox, and Context Manager."""

from __future__ import annotations

import unittest
import numpy as np

from project_override.world_model.ast_sandbox import (
    ASTSecurityGate,
    SafeCompiler,
    SandboxExecutor,
    SecurityViolationError,
)
from project_override.world_model.world_model_engine import (
    Rulebook,
    EntityDef,
    RuleDef,
    WorldModelCompiler,
    ReplayVerifier,
    RecordedTransition,
    SolvedLevelMemory,
)
from project_override.world_model.context_manager import (
    DuckContextManager,
    TokenEstimator,
    HelperFunctionRegistry,
    BlockType,
)


class TestASTSandbox(unittest.TestCase):
    """Tests for AST Security Gate and Sandbox Execution."""

    def setUp(self):
        self.compiler = SafeCompiler()
        self.executor = SandboxExecutor()

    def test_safe_numpy_execution(self):
        code = """
import numpy as np
def solve_grid(grid):
    return np.rot90(grid)
result = solve_grid(np.array([[1, 2], [3, 4]]))
"""
        res = self.executor.execute_code(code)
        self.assertTrue(res.success, f"Execution failed: {res.error}")

    def test_banned_while_loop(self):
        code = """
while True:
    pass
"""
        with self.assertRaises(SecurityViolationError):
            self.compiler.compile(code)

    def test_banned_os_import(self):
        code = """
import os
os.system('ls')
"""
        with self.assertRaises(SecurityViolationError):
            self.compiler.compile(code)

    def test_banned_file_io(self):
        code = """
with open('test.txt', 'w') as f:
    f.write('hello')
"""
        with self.assertRaises(SecurityViolationError):
            self.compiler.compile(code)


class TestWorldModelEngine(unittest.TestCase):
    """Tests for Reflective Rulebook, Compiler, ReplayVerifier, and SolvedLevelMemory."""

    def test_rulebook_serialization_and_compilation(self):
        rulebook = Rulebook(
            domain_name="game_01",
            level_id="lvl_01",
            entities={
                "agent": EntityDef(name="agent", color_id=1, role="agent"),
                "goal": EntityDef(name="goal", color_id=4, role="goal"),
            },
            rules=[
                RuleDef(
                    rule_id="r1",
                    name="shift_up",
                    trigger_action=1,
                    condition="action == 1",
                    effect="Agent shifts up by 1 cell",
                )
            ],
            transition_code="""
def simulate_step(grid, action, state):
    new_grid = grid.copy()
    if action == 1:
        new_grid = np.roll(new_grid, -1, axis=0)
    return new_grid, state
""",
        )

        md = rulebook.to_markdown()
        self.assertIn("## Entities", md)
        self.assertIn("## Executable Transition Function", md)

        compiler = WorldModelCompiler()
        compiled_model = compiler.compile(rulebook)
        self.assertIsNotNone(compiled_model)

        # Test simulation
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 2] = 1
        next_grid, state = compiled_model.simulate_step(grid, 1, {})
        self.assertEqual(next_grid[1, 2], 1)

    def test_replay_verification(self):
        rulebook = Rulebook(
            domain_name="game_01",
            level_id="lvl_01",
            transition_code="""
def simulate_step(grid, action, state):
    new_grid = grid.copy()
    if action == 4:
        new_grid[0, 1] = 1
        new_grid[0, 0] = 0
    return new_grid, state
""",
        )

        g0 = np.zeros((3, 3), dtype=int)
        g0[0, 0] = 1
        g1 = np.zeros((3, 3), dtype=int)
        g1[0, 1] = 1

        history = [RecordedTransition(0, g0, 4, g1, {}, {})]
        compiler = WorldModelCompiler()
        compiled = compiler.compile(rulebook)
        verifier = ReplayVerifier()
        report = verifier.verify(compiled, history)
        self.assertTrue(report.approved)
        self.assertEqual(report.transition_accuracy, 1.0)

    def test_solved_level_memory(self):
        memory = SolvedLevelMemory()
        rulebook = Rulebook(domain_name="g1", level_id="1")
        memory.register_solved_level(
            level_id="1",
            rulebook=rulebook,
            transition_code="# code",
            winning_actions=[1, 4, 4, 2, 5],
            distilled_mechanics=["Push block right into goal"],
        )
        record = memory.get_level("1")
        self.assertIsNotNone(record)
        self.assertEqual(record.winning_action_sequence, [1, 4, 4, 2, 5])


class TestContextManager(unittest.TestCase):
    """Tests for Franzen Hysteresis and Context Manager."""

    def test_context_hysteresis_drain(self):
        ctx = DuckContextManager(high_watermark_tokens=200, low_watermark_tokens=100)
        ctx.set_system_prefix("System prompt instructions.")

        # Add interactions until high watermark is breached
        drain_happened = False
        for i in range(15):
            drained, event = ctx.add_interaction(
                role="user",
                content=f"Observation {i}: Board grid " + "0 " * 20,
            )
            if drained:
                drain_happened = True

        self.assertTrue(drain_happened)
        self.assertLessEqual(ctx.get_total_tokens(), 200)
        self.assertGreater(len(ctx.drain_history), 0)


if __name__ == "__main__":
    unittest.main()
