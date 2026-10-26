"""Unit tests for Tier-2 Retrodict World-Model Synthesizer & AST Sandbox.
Verifies Tycho Moore Machine compilation, Entity-Centric Jaccard Retrodiction,
and GeneralAgent integration.
"""

import unittest
import numpy as np

from project_override.world_model.retrodict_synthesizer import (
    ASTWorldModelSandbox,
    HistoricalTransition,
    OfflineLLMRuntime,
    RetrodictionReport,
    RetrodictWorldModelSynthesizer,
)
from project_override.orchestration.general_agent import GeneralAgent


class TestRetrodictSynthesizer(unittest.TestCase):
    def test_ast_sandbox_whitelisting(self):
        # Valid Moore machine class
        valid_code = """
class SynthesizedWorldModel:
    def __init__(self):
        self.avatar_col = 3
    def init_state(self, grid, hud, latent):
        return {"grid": grid.copy(), "hud": dict(hud), "latent": latent.copy(), "cleared": False}
    def transition(self, state, action, click_xy=None):
        return state
    def render(self, state):
        return state["grid"]
    def outcome(self, state):
        return "ONGOING"
"""
        model = ASTWorldModelSandbox.validate_and_compile(valid_code)
        self.assertIsNotNone(model)
        self.assertEqual(model.avatar_col, 3)

        # Forbidden import
        forbidden_import = "import os\nclass SynthesizedWorldModel: pass"
        self.assertIsNone(ASTWorldModelSandbox.validate_and_compile(forbidden_import))

        # Forbidden function call
        forbidden_eval = "class SynthesizedWorldModel:\n    def __init__(self): eval('1+1')"
        self.assertIsNone(ASTWorldModelSandbox.validate_and_compile(forbidden_eval))

    def test_retrodiction_verification_matching(self):
        grid0 = np.zeros((64, 64), dtype=np.int32)
        grid0[10, 10] = 3  # Avatar at (10, 10)
        grid1 = np.zeros((64, 64), dtype=np.int32)
        grid1[10, 11] = 3  # Avatar moved right to (10, 11) by action 4

        t_mask = np.zeros((64, 64), dtype=bool)

        hist = [
            HistoricalTransition(
                step_idx=0,
                pre_grid=grid0,
                pre_hud={},
                pre_latent=np.zeros((64, 64)),
                action_id=4,
                click_xy=None,
                post_grid=grid1,
                post_hud={},
                timer_mask=t_mask,
                level_cleared=False,
                game_over=False,
            )
        ]

        class PerfectModel:
            def init_state(self, grid, hud, latent):
                return {"grid": grid.copy()}
            def transition(self, state, action, click_xy=None):
                g = state["grid"].copy()
                if action == 4:
                    pts = np.argwhere(g == 3)
                    for r, c in pts:
                        g[r, c] = 0
                        g[r, c + 1] = 3
                return {"grid": g}
            def render(self, state):
                return state["grid"]
            def outcome(self, state):
                return "ONGOING"

        report = RetrodictWorldModelSynthesizer.verify_retrodiction(PerfectModel(), hist)
        self.assertTrue(report.passed_all)
        self.assertEqual(report.mean_jaccard, 1.0)

    def test_retrodiction_verification_counterexample(self):
        grid0 = np.zeros((64, 64), dtype=np.int32)
        grid0[10, 10] = 3
        grid1 = np.zeros((64, 64), dtype=np.int32)
        grid1[10, 11] = 3

        t_mask = np.zeros((64, 64), dtype=bool)

        hist = [
            HistoricalTransition(
                step_idx=0,
                pre_grid=grid0,
                pre_hud={},
                pre_latent=np.zeros((64, 64)),
                action_id=4,
                click_xy=None,
                post_grid=grid1,
                post_hud={},
                timer_mask=t_mask,
                level_cleared=False,
                game_over=False,
            )
        ]

        class IncorrectModel:
            def init_state(self, grid, hud, latent):
                return {"grid": grid.copy()}
            def transition(self, state, action, click_xy=None):
                # Does nothing
                return {"grid": state["grid"].copy()}
            def render(self, state):
                return state["grid"]
            def outcome(self, state):
                return "ONGOING"

        report = RetrodictWorldModelSynthesizer.verify_retrodiction(IncorrectModel(), hist)
        self.assertFalse(report.passed_all)
        self.assertLess(report.mean_jaccard, 1.0)
        self.assertIn("Missing actual changes", report.counterexample_diff)

    def test_inductive_programmatic_synthesis(self):
        grid0 = np.zeros((64, 64), dtype=np.int32)
        grid0[10, 10] = 3
        grid1 = np.zeros((64, 64), dtype=np.int32)
        grid1[10, 11] = 3
        grid2 = np.zeros((64, 64), dtype=np.int32)
        grid2[9, 11] = 3

        t_mask = np.zeros((64, 64), dtype=bool)

        hist = [
            HistoricalTransition(
                step_idx=0,
                pre_grid=grid0,
                pre_hud={},
                pre_latent=np.zeros((64, 64)),
                action_id=4,
                click_xy=None,
                post_grid=grid1,
                post_hud={},
                timer_mask=t_mask,
                level_cleared=False,
                game_over=False,
            ),
            HistoricalTransition(
                step_idx=1,
                pre_grid=grid1,
                pre_hud={},
                pre_latent=np.zeros((64, 64)),
                action_id=1,
                click_xy=None,
                post_grid=grid2,
                post_hud={},
                timer_mask=t_mask,
                level_cleared=False,
                game_over=False,
            ),
        ]

        synth = RetrodictWorldModelSynthesizer()
        model = synth.synthesize_and_verify(hist, max_cegis_iters=2)
        self.assertIsNotNone(model)
        self.assertEqual(model.avatar_col, 3)
        self.assertIn(4, model.deltas)
        self.assertEqual(model.deltas[4], (0, 1))
        self.assertIn(1, model.deltas)
        self.assertEqual(model.deltas[1], (-1, 0))

    def test_general_agent_history_tracking(self):
        agent = GeneralAgent()
        agent.start_new_game("test_game")
        agent.start_new_level(0)

        grid0 = np.zeros((64, 64), dtype=np.int32)
        grid0[15, 15] = 2

        act, _ = agent.choose_action(grid0, available_actions=[1, 2, 3, 4])
        self.assertIn(act, [1, 2, 3, 4])

        grid1 = np.zeros((64, 64), dtype=np.int32)
        grid1[14, 15] = 2  # Moved UP

        act2, _ = agent.choose_action(grid1, available_actions=[1, 2, 3, 4])
        self.assertEqual(len(agent.transition_history), 1)
        self.assertEqual(agent.transition_history[0].action_id, act)

    def test_qwen_chatml_prompt_format(self):
        runtime = OfflineLLMRuntime()
        grid0 = np.zeros((64, 64), dtype=np.int32)
        grid1 = np.zeros((64, 64), dtype=np.int32)
        grid1[0, 1] = 1
        hist = [
            HistoricalTransition(
                step_idx=0,
                pre_grid=grid0,
                pre_hud={},
                pre_latent=np.zeros((64, 64)),
                action_id=4,
                click_xy=None,
                post_grid=grid1,
                post_hud={},
                timer_mask=np.zeros((64, 64), dtype=bool),
                level_cleared=False,
                game_over=False,
            )
        ]
        prompt = runtime._build_compact_prompt(hist, counterexample_feedback="None")
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant\n```python\n", prompt)
        self.assertIn("Qwen 2.5 Coder 32B Instruct", prompt)

    def test_qwen_candidate_paths(self):
        runtime = OfflineLLMRuntime()
        self.assertTrue(any("qwen" in p.lower() for p in runtime.CANDIDATE_MODEL_PATHS))

    def test_7b_local_default(self):
        runtime = OfflineLLMRuntime()
        self.assertEqual(runtime.model_size, "7B")
        self.assertEqual(runtime.model_name, "Qwen/Qwen2.5-Coder-7B-Instruct")

    def test_32b_kaggle_mode(self):
        runtime = OfflineLLMRuntime(model_size="32B")
        self.assertEqual(runtime.model_size, "32B")
        self.assertEqual(runtime.model_name, "Qwen/Qwen2.5-Coder-32B-Instruct")

    def test_local_http_generation(self):
        runtime = OfflineLLMRuntime()
        runtime.backend_type = "LOCAL_HTTP_SERVER"
        # Mock _call_local_http_api to verify parsing
        runtime._call_local_http_api = lambda prompt, attempt_idx=0: "```python\nclass SynthesizedWorldModel:\n    pass\n```"
        hist = []
        code = runtime.synthesize_world_model_code(hist)
        self.assertIn("class SynthesizedWorldModel", code)


if __name__ == "__main__":
    unittest.main()
