"""Comprehensive Unit Tests for Project Override World Model Suite:
- ast_sandbox.py (AST Security Gate, Safe Compiler & Executor, Strict Timeout)
- world_model_engine.py (Rulebook Markdown Roundtrip, Replay Verification, Solved-Level Memory)
- context_manager.py (Franzen Hysteresis, Helper Function Registry, Token Estimator)
"""

import os
import sys
import time
import unittest
import numpy as np

# Ensure project_override package is in python path
sys.path.insert(0, "/Applications/Dad Company/project_override")

from project_override.world_model.ast_sandbox import (
    ASTSecurityGate,
    ExecutionResult,
    ExecutionTimeoutError,
    SafeCompiler,
    SandboxCompilationError,
    SandboxExecutor,
    SandboxSecurityError,
    SecurityViolationError,
    build_safe_builtins,
    build_safe_globals,
)
from project_override.world_model.world_model_engine import (
    CellMismatch,
    CompiledWorldModel,
    EntityDef,
    PriorKnowledge,
    RecordedTransition,
    ReplayVerifier,
    RuleDef,
    Rulebook,
    SolvedLevelMemory,
    SolvedLevelRecord,
    VerificationReport,
    WorldModelCompiler,
)
from project_override.world_model.context_manager import (
    BlockType,
    ContextBlock,
    DrainEvent,
    DuckContextManager,
    HelperFunction,
    HelperFunctionRegistry,
    TokenEstimator,
)


class TestASTSecurityGate(unittest.TestCase):
    """Tests the AST Security Gate whitelist and ban enforcement."""

    def setUp(self):
        self.gate = ASTSecurityGate()

    def test_whitelisted_imports_pass(self):
        valid_code = """
import numpy as np
import math
import itertools
from scipy.ndimage import label
from math import sqrt, floor

def compute(grid):
    return np.zeros_like(grid)
"""
        # Should not raise
        self.gate.validate(valid_code)

    def test_while_loop_strictly_banned(self):
        while_code = """
def loop(x):
    while x < 10:
        x += 1
    return x
"""
        with self.assertRaises(SecurityViolationError) as ctx:
            self.gate.validate(while_code)
        self.assertIn("While-loops are strictly prohibited", str(ctx.exception))
        self.assertIn("Halting Problem", str(ctx.exception))

    def test_bounded_for_loops_allowed(self):
        for_code = """
def bounded_search(grid):
    h, w = grid.shape
    acc = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 1:
                acc += 1
    return acc
"""
        self.gate.validate(for_code)

    def test_banned_modules_rejected(self):
        banned_modules = [
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "import urllib.request",
            "import shutil",
            "import pickle",
            "from pathlib import Path",
        ]
        for bm in banned_modules:
            with self.assertRaises(SecurityViolationError, msg=f"Failed to ban: {bm}"):
                self.gate.validate(bm)

    def test_banned_builtins_rejected(self):
        banned_calls = [
            "x = open('secret.txt', 'r')",
            "eval('2 + 2')",
            "exec('x = 1')",
            "compile('pass', '', 'exec')",
            "__import__('os')",
            "globals()['x'] = 1",
            "locals()['y'] = 2",
        ]
        for bc in banned_calls:
            with self.assertRaises(SecurityViolationError, msg=f"Failed to ban call: {bc}"):
                self.gate.validate(bc)

    def test_dunder_attribute_introspection_rejected(self):
        dunder_attacks = [
            "x = ().__class__.__bases__[0].__subclasses__()",
            "f = func.__globals__",
            "c = obj.__code__",
            "d = obj.__dict__",
        ]
        for da in dunder_attacks:
            with self.assertRaises(SecurityViolationError, msg=f"Failed to ban dunder: {da}"):
                self.gate.validate(da)

    def test_dangerous_methods_rejected(self):
        dangerous_methods = [
            "os_mock.system('ls')",
            "p.popen('echo')",
            "f.read()",
            "f.write('data')",
        ]
        for dm in dangerous_methods:
            with self.assertRaises(SecurityViolationError, msg=f"Failed to ban method: {dm}"):
                self.gate.validate(dm)


class TestSafeCompilerAndExecutor(unittest.TestCase):
    """Tests SafeCompiler and SandboxExecutor execution & timeouts."""

    def setUp(self):
        self.compiler = SafeCompiler()
        self.executor = SandboxExecutor(compiler=self.compiler, default_timeout=1.0)

    def test_successful_execution(self):
        code = """
import numpy as np

def simulate_step(grid, action, state):
    next_grid = grid.copy()
    if action == 1: # UP
        next_grid = np.roll(next_grid, -1, axis=0)
    return next_grid, {"action_taken": action}
"""
        grid = np.array([[1, 2], [3, 4]])
        res = self.executor.call_function(
            source_code_or_obj=code,
            func_name="simulate_step",
            args=(grid, 1, {}),
            timeout=2.0,
        )
        self.assertTrue(res.success)
        self.assertIsNone(res.error)
        next_grid, state = res.return_value
        np.testing.assert_array_equal(next_grid, np.array([[3, 4], [1, 2]]))
        self.assertEqual(state["action_taken"], 1)

    def test_runtime_error_handled_cleanly(self):
        bad_code = """
def crash(grid):
    return grid[100, 100]
"""
        grid = np.zeros((3, 3))
        res = self.executor.call_function(bad_code, "crash", args=(grid,), timeout=1.0)
        self.assertFalse(res.success)
        self.assertIn("IndexError", res.error)

    def test_strict_timeout_interruption(self):
        # Long bounded loop that exceeds a short 0.1s timeout quota
        timeout_code = """
def hang_loop():
    acc = 0
    for i in range(100_000_000):
        acc = (acc + i) % 10007
    return acc
"""
        res = self.executor.call_function(
            source_code_or_obj=timeout_code,
            func_name="hang_loop",
            timeout=0.1,
        )
        self.assertFalse(res.success)
        self.assertTrue(res.timed_out)

    def test_process_isolated_timeout(self):
        # Test process isolation mode with strict termination
        timeout_code = """
def hang_loop():
    acc = 0
    for i in range(100_000_000):
        acc = (acc + i) % 10007
    return acc
"""
        res = self.executor.call_function(
            source_code_or_obj=timeout_code,
            func_name="hang_loop",
            timeout=0.15,
            isolate_process=True,
        )
        self.assertFalse(res.success)
        self.assertTrue(res.timed_out)


class TestWorldModelEngine(unittest.TestCase):
    """Tests Rulebook Markdown parsing, compilation, replay verification, and solved memory."""

    def test_rulebook_markdown_roundtrip(self):
        rb = Rulebook(
            domain_name="Dungeon Grid",
            level_id="lvl_04",
            version=2,
            description="Agent collecting keys and unlocking doors.",
        )
        rb.add_entity(EntityDef(name="Agent", color_id=1, role="agent", properties={"movable": True}))
        rb.add_entity(EntityDef(name="Wall", color_id=2, role="wall", properties={"solid": True}))
        rb.add_invariant("Walls cannot be passed through.")
        rb.add_invariant("Grid boundaries are impassable.")
        rb.action_space = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
        rb.add_rule(RuleDef(
            rule_id="R1",
            name="Move Up",
            trigger_action=1,
            condition="Free space above",
            effect="Agent moves up 1 cell",
            priority=10,
        ))
        rb.hypotheses.append("Door color 3 turns into color 0 when key color 4 is collected.")
        rb.transition_code = """def simulate_step(grid, action, state):
    next_grid = grid.copy()
    return next_grid, state
"""

        # Serialize to Markdown
        md = rb.to_markdown()
        self.assertIn("# World Model: Dungeon Grid", md)
        self.assertIn("lvl_04", md)
        self.assertIn("Walls cannot be passed through.", md)
        self.assertIn("R1", md)

        # Parse back from Markdown
        parsed_rb = Rulebook.from_markdown(md)
        self.assertEqual(parsed_rb.domain_name, "Dungeon Grid")
        self.assertEqual(parsed_rb.level_id, "lvl_04")
        self.assertEqual(len(parsed_rb.entities), 2)
        self.assertIn("Agent", parsed_rb.entities)
        self.assertEqual(parsed_rb.entities["Agent"].color_id, 1)
        self.assertEqual(len(parsed_rb.invariants), 2)
        self.assertEqual(len(parsed_rb.rules), 1)
        self.assertEqual(parsed_rb.rules[0].rule_id, "R1")
        self.assertEqual(len(parsed_rb.hypotheses), 1)
        self.assertIn("def simulate_step", parsed_rb.transition_code)

    def test_cell_exact_replay_verification_approval(self):
        # A simple agent moving right (action 4)
        t_code = """
import numpy as np

def simulate_step(grid, action, state):
    next_grid = grid.copy()
    agent_pos = np.argwhere(grid == 1)
    if len(agent_pos) > 0:
        r, c = agent_pos[0]
        if action == 4 and c + 1 < grid.shape[1]:  # RIGHT
            next_grid[r, c] = 0
            next_grid[r, c + 1] = 1
    return next_grid, {}
"""
        compiler = WorldModelCompiler()
        compiled_model = compiler.compile(t_code)

        # Create 3 recorded transitions
        g0 = np.array([[1, 0, 0, 0]])
        g1 = np.array([[0, 1, 0, 0]])
        g2 = np.array([[0, 0, 1, 0]])
        g3 = np.array([[0, 0, 0, 1]])

        transitions = [
            RecordedTransition(step_index=0, grid=g0, action=4, state={}, next_grid=g1),
            RecordedTransition(step_index=1, grid=g1, action=4, state={}, next_grid=g2),
            RecordedTransition(step_index=2, grid=g2, action=4, state={}, next_grid=g3),
        ]

        verifier = ReplayVerifier()
        report = verifier.verify(compiled_model, transitions)

        self.assertTrue(report.approved)
        self.assertEqual(report.matched_transitions, 3)
        self.assertEqual(report.total_transitions, 3)
        self.assertEqual(report.transition_accuracy, 1.0)
        self.assertEqual(report.cell_accuracy, 1.0)
        self.assertEqual(len(report.mismatches), 0)

    def test_cell_exact_replay_verification_rejection_on_mismatch(self):
        # Flawed model: teleports 2 steps instead of 1
        flawed_code = """
import numpy as np

def simulate_step(grid, action, state):
    next_grid = grid.copy()
    agent_pos = np.argwhere(grid == 1)
    if len(agent_pos) > 0:
        r, c = agent_pos[0]
        if action == 4 and c + 2 < grid.shape[1]:
            next_grid[r, c] = 0
            next_grid[r, c + 2] = 1
    return next_grid, {}
"""
        compiler = WorldModelCompiler()
        compiled_model = compiler.compile(flawed_code)

        g0 = np.array([[1, 0, 0, 0]])
        g1 = np.array([[0, 1, 0, 0]])  # Expected step 1 move

        transitions = [
            RecordedTransition(step_index=0, grid=g0, action=4, state={}, next_grid=g1),
        ]

        verifier = ReplayVerifier()
        report = verifier.verify(compiled_model, transitions)

        self.assertFalse(report.approved)
        self.assertEqual(report.matched_transitions, 0)
        self.assertGreater(len(report.mismatches), 0)
        self.assertEqual(report.first_failure_step, 0)

    def test_solved_level_memory_siriki_m85(self):
        memory = SolvedLevelMemory(default_tail_length=3)

        rb1 = Rulebook(domain_name="Level 1")
        rb1.add_invariant("Wall color 2 is solid")
        rb1.add_invariant("Gravity pulls down")
        rb1.action_space = {1: "UP", 2: "DOWN"}

        actions1 = [1, 2, 1, 2, 4, 4, 5]  # Tail of 3: [4, 4, 5]
        memory.register_solved_level(
            level_id="lvl_01",
            rulebook=rb1,
            transition_code="def simulate_step(grid, a, s): return grid, s",
            winning_actions=actions1,
            distilled_mechanics=["Wall color 2 is solid", "Gravity pulls down"],
        )

        rb2 = Rulebook(domain_name="Level 2")
        rb2.add_invariant("Wall color 2 is solid")
        rb2.action_space = {1: "UP", 2: "DOWN", 3: "LEFT"}
        actions2 = [3, 3, 1, 4]  # Tail of 3: [3, 1, 4]
        memory.register_solved_level(
            level_id="lvl_02",
            rulebook=rb2,
            transition_code="def simulate_step(grid, a, s): return grid, s",
            winning_actions=actions2,
            distilled_mechanics=["Wall color 2 is solid"],
        )

        # Distill cross-level invariants
        invariants = memory.distill_cross_level_invariants()
        self.assertEqual(invariants[0], "Wall color 2 is solid")

        # Get prior for Level 3
        priors = memory.get_prior_for_level("lvl_03")
        self.assertIn("Wall color 2 is solid", priors.cross_level_invariants)
        self.assertEqual(priors.winning_tails["lvl_01"], [4, 4, 5])
        self.assertEqual(priors.winning_tails["lvl_02"], [3, 1, 4])


class TestContextManagerAndHysteresis(unittest.TestCase):
    """Tests TokenEstimator, HelperFunctionRegistry, and Franzen Hysteresis."""

    def test_token_estimator(self):
        estimator = TokenEstimator()
        tokens = estimator.count_tokens("The quick brown fox jumps over the lazy dog.")
        self.assertGreater(tokens, 5)
        self.assertLess(tokens, 20)

    def test_helper_function_registry_and_ast_gate(self):
        registry = HelperFunctionRegistry()

        # Valid helper function
        valid_helper = """
def flood_fill(grid, r, c, target_color, fill_color):
    \"\"\"Performs 4-way flood fill on grid.\"\"\"
    import numpy as np
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    stack = [(r, c)]
    res = grid.copy()
    for _ in range(h * w):
        if len(stack) == 0:
            break
        cr, cc = stack.pop()
        if cr < 0 or cr >= h or cc < 0 or cc >= w or visited[cr, cc]:
            continue
        visited[cr, cc] = True
        if res[cr, cc] == target_color:
            res[cr, cc] = fill_color
            stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])
    return res
"""
        fn = registry.register("flood_fill", valid_helper, tags=["geometry", "color"])
        self.assertEqual(fn.name, "flood_fill")
        self.assertIn("flood_fill", registry.export_code_block())

        # Unsafe helper function with while loop should be rejected by ASTSecurityGate
        unsafe_helper = """
def bad_helper():
    while True:
        pass
"""
        with self.assertRaises(SecurityViolationError):
            registry.register("bad_helper", unsafe_helper)

    def test_franzen_hysteresis_draining(self):
        # Configure thresholds to test hysteresis behavior accurately
        cm = DuckContextManager(
            high_watermark_tokens=400,
            low_watermark_tokens=200,
            min_active_turns=2,
        )

        cm.set_system_prefix("System root prompt: Solve ARC-AGI-3 puzzles.")

        # Register a persistent helper
        cm.registry.register(
            "dummy_tool",
            "def dummy_tool(x):\n    '''Tool doc.'''\n    return x + 1\n",
        )

        # 1. Growth phase: Context grows freely without pruning while <= 400
        for i in range(10):
            drain_occurred, event = cm.add_interaction(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Turn {i}: Action simulation test with moderate tokens.",
            )
            self.assertFalse(drain_occurred, f"Premature drain occurred at turn {i}")
            self.assertIsNone(event)

        # Verify KV prefix hotness: helper registry and system prefix are intact
        msgs = cm.render_messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("dummy_tool", msgs[1]["content"])

        # 2. Add interactions that push total tokens beyond high watermark of 400
        drain_occurred, event = cm.add_interaction(
            role="user",
            content="Additional turn with exploration details that pushes tokens over 400. " * 5,
        )
        self.assertTrue(drain_occurred)
        self.assertIsNotNone(event)
        self.assertLessEqual(event.tokens_after, cm.low_watermark_tokens + 50)
        self.assertGreater(event.tokens_reclaimed, 0)

        # 3. Verify persistent helper registry survived the drain!
        msgs_after = cm.render_messages()
        self.assertEqual(msgs_after[0]["role"], "system")
        self.assertIn("dummy_tool", msgs_after[1]["content"])

        # 4. Check telemetry
        status = cm.get_status()
        self.assertEqual(status["total_drains_executed"], 1)
        self.assertGreater(status["registered_helper_functions"], 0)


if __name__ == "__main__":
    unittest.main()
