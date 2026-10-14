"""Comprehensive Evaluation Benchmark on Public ARC-AGI-3 Dataset
Runs ProjectOverrideAgent / MyAgent across the real ARC-AGI-3 environments in arc-interactive.
Measures win rate, levels cleared, actions taken, and Relative Human Action Efficiency (RHAE).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Add repo directories to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "project_override"))

from arc_agi import Arcade, OperationMode
from arc_agi.scorecard import EnvironmentScoreCalculator
from arcengine import GameAction, GameState
from agent.my_agent import MyAgent


def compute_rhae(human_baseline: int, ai_actions: int, is_won: bool) -> float:
    """Computes exact Relative Human Action Efficiency (RHAE) score."""
    if not is_won or ai_actions <= 0:
        return 0.0
    ratio = float(human_baseline) / float(ai_actions)
    return min(1.15, ratio * ratio)


def evaluate_benchmark(
    max_games: Optional[int] = None,
    max_steps_per_game: int = 180,
    game_filter: Optional[List[str]] = None,
    output_json: str = "benchmark_results.json",
    output_csv: str = "benchmark_results.csv",
) -> Dict[str, Any]:
    print("=" * 80)
    print(" ARC-AGI-3 PUBLIC DATASET BENCHMARK EVALUATOR")
    print(f" Environment Suite: /Applications/Dad Company/arc-interactive/environment_files")
    print(f" Max Steps Per Game: {max_steps_per_game}")
    print("=" * 80)

    environments_dir = "/Applications/Dad Company/arc-interactive/environment_files"
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=environments_dir)
    all_envs = arcade.get_environments()
    print(f"[INIT] Discovered {len(all_envs)} environments in offline repository.")

    # Filter games if specified
    if game_filter:
        wanted = {g.strip().split("-")[0] for g in game_filter}
        envs_to_run = [e for e in all_envs if e.game_id.split("-")[0] in wanted]
    else:
        envs_to_run = all_envs

    if max_games is not None:
        envs_to_run = envs_to_run[:max_games]

    print(f"[EXEC] Running evaluation on {len(envs_to_run)} target environments...\n")

    results = []
    total_wins = 0
    total_levels_cleared = 0
    total_levels_possible = 0
    total_actions_spent = 0
    total_human_baseline = 0
    game_rhae_scores = []
    game_official_scores = []

    start_bench_time = time.time()

    for idx, env_info in enumerate(envs_to_run, 1):
        short_id = env_info.game_id.split("-")[0]
        title = getattr(env_info, "title", short_id) or short_id
        raw_b = getattr(env_info, "baseline_actions", None)
        win_levels = getattr(env_info, "win_levels", 5) or 5
        if raw_b and all(1 <= x <= 7 for x in raw_b) and len(raw_b) <= 7:
            baseline_actions_list = [15] * win_levels
        elif raw_b:
            baseline_actions_list = list(raw_b)
            if len(baseline_actions_list) < win_levels:
                baseline_actions_list = baseline_actions_list + [15] * (win_levels - len(baseline_actions_list))
        else:
            baseline_actions_list = [15] * win_levels

        print(f"[{idx:3d}/{len(envs_to_run):3d}] Game {short_id:6s} ({title[:25]:25s}) | Baseline: {baseline_actions_list} actions")

        try:
            env = arcade.make(short_id)
            if env is None:
                print(f"  -> Could not instantiate environment for {short_id}")
                continue

            agent = MyAgent(total_budget=32400.0, total_games=len(envs_to_run))
            obs = env.reset()
            frames_history = [obs]
            latest_frame = obs
            action_count = 0
            actions_in_current_level = 0
            last_levels_completed = 0

            score_calc = EnvironmentScoreCalculator(id=short_id)

            game_start = time.time()
            max_game_steps = max(max_steps_per_game, len(baseline_actions_list) * 40)

            while not agent.is_done(frames_history, latest_frame) and action_count < max_game_steps:
                action = agent.choose_action(frames_history, latest_frame)
                action_data = getattr(action, "data", None)
                if not action_data and hasattr(action, "action_data") and action.action_data:
                    action_data = action.action_data.model_dump()
                raw = env.step(action, data=action_data) if action_data else env.step(action)
                frames_history.append(raw)
                latest_frame = raw
                action_count += 1
                actions_in_current_level += 1

                curr_levels = getattr(latest_frame, "levels_completed", 0)
                if curr_levels > last_levels_completed:
                    lvl_idx = last_levels_completed
                    base_act = baseline_actions_list[lvl_idx] if lvl_idx < len(baseline_actions_list) else 15
                    score_calc.add_level(
                        level_index=lvl_idx + 1,
                        completed=True,
                        actions_taken=actions_in_current_level,
                        baseline_actions=base_act,
                        game_id=short_id,
                    )
                    last_levels_completed = curr_levels
                    actions_in_current_level = 0

                if str(getattr(latest_frame, "state", "")).upper().find("WIN") != -1:
                    break

            # Handle trailing incomplete level if game ended
            if last_levels_completed < len(baseline_actions_list):
                lvl_idx = last_levels_completed
                base_act = baseline_actions_list[lvl_idx] if lvl_idx < len(baseline_actions_list) else 15
                is_last_completed = str(getattr(latest_frame, "state", "")).upper().find("WIN") != -1
                if not is_last_completed and actions_in_current_level > 0:
                    score_calc.add_level(
                        level_index=lvl_idx + 1,
                        completed=False,
                        actions_taken=actions_in_current_level,
                        baseline_actions=base_act,
                        game_id=short_id,
                    )

            game_elapsed = time.time() - game_start

            state_str = str(getattr(latest_frame, "state", "NOT_FINISHED"))
            if hasattr(latest_frame.state, "name"):
                state_str = latest_frame.state.name
            levels_completed = getattr(latest_frame, "levels_completed", 0)
            win_levels = getattr(latest_frame, "win_levels", len(baseline_actions_list)) or 1
            is_won = ("WIN" in state_str.upper()) or (levels_completed >= win_levels and win_levels > 0)
            if is_won:
                state_str = "WIN"

            total_h = sum(baseline_actions_list) if baseline_actions_list else 15
            official_score_obj = score_calc.to_score()
            official_score = official_score_obj.score  # 0 to 115% scale
            rhae = compute_rhae(total_h, action_count, is_won)

            if is_won:
                total_wins += 1
            total_levels_cleared += levels_completed
            total_levels_possible += win_levels
            total_actions_spent += action_count
            total_human_baseline += total_h
            game_rhae_scores.append(rhae)
            game_official_scores.append(official_score)

            result_entry = {
                "game_id": short_id,
                "full_id": env_info.game_id,
                "title": title,
                "tags": getattr(env_info, "tags", []) or [],
                "state": state_str,
                "is_won": is_won,
                "levels_completed": levels_completed,
                "win_levels": win_levels,
                "actions": action_count,
                "human_baseline": total_h,
                "rhae_score": rhae,
                "official_score": official_score,
                "time_sec": round(game_elapsed, 3),
            }
            results.append(result_entry)

            status_flag = "[WIN]" if is_won else "[IN_PROGRESS]"
            print(f"       -> {status_flag:13s} Levels: {levels_completed:2d}/{win_levels:2d} | Actions: {action_count:3d} | Official Score: {official_score:5.1f}% | RHAE: {rhae*100:5.1f}% | Time: {game_elapsed:.2f}s")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"       -> [ERROR] Failed evaluating {short_id}: {e}")
            results.append({
                "game_id": short_id,
                "full_id": env_info.game_id,
                "title": title,
                "tags": getattr(env_info, "tags", []) or [],
                "state": "ERROR",
                "is_won": False,
                "levels_completed": 0,
                "win_levels": 1,
                "actions": 0,
                "human_baseline": 15,
                "rhae_score": 0.0,
                "time_sec": 0.0,
                "error": str(e),
            })
            game_rhae_scores.append(0.0)
            game_official_scores.append(0.0)

    total_bench_time = time.time() - start_bench_time
    total_games_run = len(results)
    mean_rhae = float(np.mean(game_rhae_scores)) if game_rhae_scores else 0.0
    mean_official = float(np.mean(game_official_scores)) if game_official_scores else 0.0
    win_rate = (total_wins / max(1, total_games_run)) * 100.0

    print("\n" + "=" * 80)
    print(" BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f" Total Games Evaluated:           {total_games_run}")
    print(f" Total Games Won (100% win):      {total_wins} ({win_rate:.1f}%)")
    print(f" Total Levels Cleared:            {total_levels_cleared} / {total_levels_possible}")
    print(f" Total AI Actions Expended:       {total_actions_spent}")
    print(f" Total Human Baseline Actions:    {total_human_baseline}")
    print(f" Mean Official Competition Score: {mean_official:.2f}%")
    print(f" Mean Benchmark RHAE Score:       {mean_rhae * 100:.2f}%")
    print(f" Total Benchmark Wallclock:       {total_bench_time:.2f} seconds ({total_bench_time/max(1, total_games_run):.3f}s/game)")
    print("=" * 80 + "\n")

    # Save results to JSON and CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[SAVE] CSV results saved to {output_csv}")

    summary = {
        "total_games": total_games_run,
        "total_wins": total_wins,
        "win_rate_pct": win_rate,
        "total_levels_cleared": total_levels_cleared,
        "total_levels_possible": total_levels_possible,
        "mean_official_score": mean_official,
        "mean_rhae": mean_rhae,
        "total_bench_time_sec": total_bench_time,
        "games": results,
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] JSON summary saved to {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent on ARC-AGI-3 public dataset")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to evaluate")
    parser.add_argument("--steps", type=int, default=80, help="Max steps per game")
    parser.add_argument("--games", type=str, default=None, help="Comma-separated list of game IDs")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="Path to output JSON")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="Path to output CSV")
    args = parser.parse_args()

    game_list = [g.strip() for g in args.games.split(",")] if args.games else None
    evaluate_benchmark(
        max_games=args.max_games,
        max_steps_per_game=args.steps,
        game_filter=game_list,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
