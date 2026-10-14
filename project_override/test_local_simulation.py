"""Project Override: Local Empirical Simulation & RHAE Benchmark Runner
Tests the integrated ProjectOverrideAgent against the POMDP environment suite.
Measures Relative Human Action Efficiency (RHAE) and prints comprehensive diagnostics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

# Ensure project_override is in python path
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

from project_override.orchestration.agent import ProjectOverrideAgent
from project_override.evaluation.benchmark_evaluator import (
    MockInteractiveGame,
    RHAEBenchmarkEvaluator,
    SokobanPushLevel,
    DynamicSpriteInterceptLevel,
)
from project_override.core.zero_nn_ccl import ZeroNNPerception
from project_override.core.semantic_evaluator import SemanticEvaluator
from project_override.core.dead_signature import DeadSignatureFilter, compute_context_hash


def run_local_benchmark() -> None:
    print("=" * 78)
    print(" PROJECT OVERRIDE: AUTONOMOUS SINGULARITY CLUSTER (ARC-AGI-3)")
    print(" LOCAL EMPIRICAL SIMULATION & RHAE MATHEMATICAL BENCHMARK")
    print("=" * 78)

    agent = ProjectOverrideAgent(
        total_budget=32400.0,
        total_games=10,
        max_level_timeout=30.0,
    )

    evaluator = RHAEBenchmarkEvaluator()
    game = MockInteractiveGame(game_id="arc3_eval_challenge_01")

    print(f"\n[INIT] Launching Game '{game.game_id}' ({len(game.levels)} Sequential Levels)...")
    agent.start_new_game(game.game_id)

    total_game_actions = 0
    total_human_baseline = 0
    level_scores = []

    for lvl in game.levels:
        print(f"\n----------------------------------------------------------------------")
        print(f" LEVEL {lvl.level_index}: {lvl.name}")
        print(f" Human Upper-Median Baseline (h): {lvl.human_baseline} actions")
        print(f" Max Allowed Actions (5x cutoff): {lvl.max_allowed_actions} actions")
        print(f"----------------------------------------------------------------------")

        agent.start_new_level(lvl.level_index)
        lvl_start_time = time.perf_counter()
        actions_taken_on_level = []

        obs = lvl.current_grid.copy()
        is_level_done = False

        # Execute turn-by-turn agentic loop
        while not is_level_done and lvl.actions_taken < lvl.max_allowed_actions:
            # Agent perception
            interior_grid = agent.get_interior_grid(obs)
            comps = ZeroNNPerception.extract_components(interior_grid)
            scene_hash, _ = ZeroNNPerception.hash_scene(comps)

            # Available legal actions for standard grid games
            legal_actions = [1, 2, 3, 4, 5, 6, 7]

            # Choose optimal action
            action, action_data = agent.choose_action(
                current_grid=obs,
                available_actions=legal_actions,
            )

            # Step the environment
            next_obs, done, info = lvl.step(action, action_data)
            obs = next_obs
            actions_taken_on_level.append(action)

            if done:
                is_level_done = True
                break

        lvl_elapsed = time.perf_counter() - lvl_start_time
        ai_actions = len(actions_taken_on_level)
        total_game_actions += ai_actions
        total_human_baseline += lvl.human_baseline

        # Compute level RHAE score
        lvl_score = evaluator.compute_level_score(
            human_baseline=lvl.human_baseline,
            ai_actions=ai_actions,
            is_won=lvl.is_won,
        )
        level_scores.append(lvl_score)

        if lvl.is_won:
            agent.record_level_won(actions_taken_on_level)
            status_str = "CLEARED"
        else:
            status_str = "FAILED / TIMED OUT"

        print(f" Result: [{status_str}] in {ai_actions} AI actions ({lvl_elapsed:.3f}s)")
        print(f" Action Efficiency Ratio (h/a): {lvl.human_baseline / max(1, ai_actions):.2f}")
        print(f" Level RHAE Score (S_l): {lvl_score * 100:.2f}%")

    # Compute overall game score
    game_score = evaluator.compute_game_score(level_scores)
    print("\n" + "=" * 78)
    print(" EMPIRICAL EVALUATION RESULTS SUMMARY")
    print("=" * 78)
    print(f" Total Levels Evaluated: {len(game.levels)}")
    print(f" Total Human Baseline Actions: {total_human_baseline}")
    print(f" Total AI Actions Expended: {total_game_actions}")
    print(f" Overall Relative Human Action Efficiency (RHAE): {game_score * 100:.2f}%")
    print(f" Top-3 Leaderboard Threshold Exceeded (>66.0%): {'YES' if game_score >= 0.66 else 'NO'}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_local_benchmark()
