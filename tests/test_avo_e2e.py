"""End-to-end verification test for Project Override v5.0 NVIDIA AVO + CodeAct REPL."""

import os
import unittest
from project_override.config import AgentConfig
from project_override.orchestration.general_agent import GeneralAgent
from project_override.submission.kaggle_submission import run_offline_evaluation


class TestAVOArchitectureE2E(unittest.TestCase):
    def test_general_agent_initialization(self):
        config = AgentConfig()
        agent = GeneralAgent(config=config)
        self.assertIsNotNone(agent.supervisor)
        self.assertIsNotNone(agent.lineage_dag)
        self.assertIsNotNone(agent.verifier_gate)
        self.assertIsNotNone(agent.compactor)

    def test_e2e_real_game_execution(self):
        """Runs the agent on a real game from environment_files with zero mocks."""
        config = AgentConfig(enable_llm=False)
        res = run_offline_evaluation(
            game_ids=["ls20"],
            max_steps_override=10,
            config_override=config,
        )
        self.assertEqual(res["games_attempted"], 1)
        self.assertEqual(res["games_completed_without_crash"], 1)
        self.assertFalse(res["per_game"][0]["crashed"])
        self.assertEqual(res["per_game"][0]["actions"], 10)


if __name__ == "__main__":
    unittest.main()
