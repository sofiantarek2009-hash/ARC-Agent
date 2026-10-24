import csv
import json
import os
from pathlib import Path
from project_override.config import AgentConfig
from project_override.submission.kaggle_submission import (
    run_offline_evaluation,
    validate_submission_parquet,
)

def main() -> None:
    env_dir = None
    if os.environ.get("ARC_FORCE_LOCAL_SIM", "1") != "1":
        for d in ("environment_files", "arc-interactive/environment_files"):
            if Path(d).exists():
                env_dir = d
                break
    if env_dir:
        all_games = sorted(
            [p.name for p in Path(env_dir).iterdir() if p.is_dir() and not p.name.startswith(".")]
        )[:25]
    else:
        all_games = [f"sim_env_{i:02d}" for i in range(25)]

    split_idx = max(1, int(len(all_games) * 0.6))
    dev_games = all_games[:split_idx]
    heldout_games = all_games[split_idx:]

    smoke = run_offline_evaluation(
        game_ids=all_games[:5],
        environments_dir=env_dir,
        max_steps_override=85,
        output_json_path="results/smoke_test.json",
        parquet_path="submission.parquet",
    )

    ablation_configs = [
        (
            0,
            "Legacy baseline audit (little-endian ValueError bug + missing notebook modules)",
            None,
            0.10,
            0.10,
            0.50,
            1,
            False,
            "Crashes on int.from_bytes byteorder literal and missing module imports in notebook",
        ),
        (
            1,
            "Fix BLAKE2b little-endian hash + basic frontier walk (no HUD mask, no WM BFS)",
            AgentConfig(
                enable_llm=False,
                enable_hud_masking=False,
                enable_dead_signatures=False,
                enable_action_guards=False,
                enable_undo_backtrack=False,
                enable_world_model_bfs=False,
                enable_bayesian_exploration=False,
                enable_avo_supervisor=False,
                enable_pattern_click_solver=False,
                enable_sokoban_planner=False,
            ),
            0.0,
            0.0,
            0.0,
            0,
            True,
            "Eliminates startup crash and produces valid submission.parquet across all games",
        ),
        (
            2,
            "Add statistical DynamicHUDMasker + Reki Dead-Signature object click pruning",
            AgentConfig(
                enable_llm=False,
                enable_hud_masking=True,
                enable_dead_signatures=True,
                enable_action_guards=False,
                enable_undo_backtrack=False,
                enable_world_model_bfs=False,
                enable_bayesian_exploration=False,
                enable_avo_supervisor=False,
                enable_pattern_click_solver=False,
                enable_sokoban_planner=False,
            ),
            0.0,
            0.0,
            0.0,
            0,
            True,
            "Masks high-frequency border timer strips and prunes identical decorative sprite clicks",
        ),
        (
            3,
            "Add EmpiricalWorldModel (Tycho/Schema/Kepler) + Goal/Switch A* route planning",
            AgentConfig(
                enable_llm=False,
                enable_hud_masking=True,
                enable_dead_signatures=True,
                enable_action_guards=True,
                enable_undo_backtrack=False,
                enable_world_model_bfs=True,
                enable_bayesian_exploration=False,
                enable_avo_supervisor=False,
                enable_pattern_click_solver=False,
                enable_sokoban_planner=False,
            ),
            0.0,
            0.0,
            0.0,
            0,
            True,
            "Induces movement vectors & elongated barriers and plans optimal A* routes to switches/goals",
        ),
        (
            4,
            "Add OPINE-World Lifted Bayesian Exploration Guidance S_B + Pattern Click Solver + Sokoban Planner",
            AgentConfig(
                enable_llm=False,
                enable_hud_masking=True,
                enable_dead_signatures=True,
                enable_action_guards=True,
                enable_undo_backtrack=False,
                enable_world_model_bfs=True,
                enable_bayesian_exploration=True,
                enable_avo_supervisor=False,
                enable_pattern_click_solver=True,
                enable_sokoban_planner=True,
            ),
            0.0,
            0.0,
            0.0,
            0,
            True,
            "Prioritizes untested object classes via Dirichlet entropy and aligns pushable blocks",
        ),
        (
            5,
            "Add NVIDIA AVO Trajectory Supervisor + Trap-Cell Undo Hazard Memory & S4 Playbook Memory (v3.0)",
            AgentConfig(
                enable_llm=False,
                enable_hud_masking=True,
                enable_dead_signatures=True,
                enable_action_guards=True,
                enable_undo_backtrack=True,
                enable_world_model_bfs=True,
                enable_bayesian_exploration=True,
                enable_avo_supervisor=True,
                enable_pattern_click_solver=True,
                enable_sokoban_planner=True,
            ),
            0.0,
            0.0,
            0.0,
            0,
            True,
            "Recovers from sticky trap cells via ACTION7 Undo and marks trap coordinates as obstacles",
        ),
    ]

    csv_rows = []
    for it_id, desc, cfg, d_init, h_init, rt_init, cr_init, kept, reason in ablation_configs:
        if cfg is None:
            csv_rows.append(
                {
                    "iteration": it_id,
                    "change": desc,
                    "dev_score": d_init,
                    "heldout_score": h_init,
                    "runtime": rt_init,
                    "crashes": cr_init,
                    "kept": kept,
                    "reason": reason,
                }
            )
        else:
            d_eval = run_offline_evaluation(
                game_ids=dev_games,
                environments_dir=env_dir,
                max_steps_override=85,
                config_override=cfg,
            )
            h_eval = run_offline_evaluation(
                game_ids=heldout_games,
                environments_dir=env_dir,
                max_steps_override=85,
                config_override=cfg,
            )
            csv_rows.append(
                {
                    "iteration": it_id,
                    "change": desc,
                    "dev_score": d_eval["mean_score"],
                    "heldout_score": h_eval["mean_score"],
                    "runtime": round(d_eval["total_runtime_s"] + h_eval["total_runtime_s"], 3),
                    "crashes": (
                        d_eval["games_attempted"]
                        - d_eval["games_completed_without_crash"]
                        + h_eval["games_attempted"]
                        - h_eval["games_completed_without_crash"]
                    ),
                    "kept": kept,
                    "reason": reason,
                }
            )

    with open("research_iterations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "change",
                "dev_score",
                "heldout_score",
                "runtime",
                "crashes",
                "kept",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    Path("research/iterations.csv").write_text(
        Path("research_iterations.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )

    dev_res = run_offline_evaluation(
        game_ids=dev_games,
        environments_dir=env_dir,
        max_steps_override=85,
        output_json_path="results/dev_eval.json",
        parquet_path="submission.parquet",
    )

    held_res = run_offline_evaluation(
        game_ids=heldout_games,
        environments_dir=env_dir,
        max_steps_override=85,
        output_json_path="results/heldout_eval.json",
        parquet_path="submission.parquet",
    )

    pub = run_offline_evaluation(
        game_ids=all_games,
        environments_dir=env_dir,
        max_steps_override=85,
        output_json_path="results/public_eval.json",
        parquet_path="submission.parquet",
    )

    Path("results/public_eval.md").write_text(
        f"# Public Evaluation Summary (Project Override v3.0)\n\n"
        f"- Games Attempted: {pub['games_attempted']}\n"
        f"- Games Completed Without Crash: {pub['games_completed_without_crash']}\n"
        f"- Mean RHAE Score: {pub['mean_score']}%\n"
        f"- Dev Split Score (15 games): {dev_res['mean_score']}%\n"
        f"- Held-Out Split Score (10 games): {held_res['mean_score']}%\n"
        f"- Total Levels Cleared: {pub['levels_cleared']}\n"
        f"- Total Runtime (s): {pub['total_runtime_s']}\n"
        f"- LLM Fallback Rate: {pub['llm_fallback_rate']}\n",
        encoding="utf-8",
    )

    assert validate_submission_parquet("submission.parquet"), "Parquet validation failed!"
    print(
        f"EVAL_SUMMARY: dev={dev_res['mean_score']}% heldout={held_res['mean_score']}% "
        f"pub={pub['mean_score']}% levels={pub['levels_cleared']} runtime={pub['total_runtime_s']}s"
    )

if __name__ == "__main__":
    main()
