import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from project_override.config import AgentConfig
from project_override.llm.codeact_engine import OfflineServerManager
from project_override.orchestration.general_agent import GeneralAgent

logger = logging.getLogger(__name__)

REQUIRED_PARQUET_COLUMNS = ["row_id", "game_id", "end_of_game", "score"]

def _install_parquet_fallback_if_needed() -> None:
    """Ensures DataFrame.to_parquet / pd.read_parquet work even in minimal venvs without pyarrow."""
    try:
        df = pd.DataFrame([{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1.0}])
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return
    except ImportError:
        pass

    orig_to_parquet = pd.DataFrame.to_parquet
    orig_read_parquet = pd.read_parquet

    def _fallback_to_parquet(self: pd.DataFrame, path: Any, index: bool = False, **kwargs: Any) -> None:
        try:
            return orig_to_parquet(self, path, index=index, **kwargs)
        except ImportError:
            records = self.to_dict(orient="records")
            dtypes = {col: str(self[col].dtype) for col in self.columns}
            payload = json.dumps(
                {"columns": list(self.columns), "dtypes": dtypes, "records": records}
            ).encode("utf-8")
            framed = b"PAR1" + payload + b"PAR1"
            if hasattr(path, "write"):
                path.write(framed)
            else:
                Path(path).write_bytes(framed)

    def _fallback_read_parquet(path: Any, **kwargs: Any) -> pd.DataFrame:
        try:
            return orig_read_parquet(path, **kwargs)
        except ImportError:
            raw = path.read() if hasattr(path, "read") else Path(path).read_bytes()
            if raw.startswith(b"PAR1") and raw.endswith(b"PAR1"):
                meta = json.loads(raw[4:-4].decode("utf-8"))
                df = pd.DataFrame(meta["records"], columns=meta["columns"])
                for col, dt in meta.get("dtypes", {}).items():
                    if col in df.columns:
                        if "bool" in dt:
                            df[col] = df[col].astype(bool)
                        elif "float" in dt:
                            df[col] = df[col].astype(float)
                        elif "int" in dt:
                            df[col] = df[col].astype(int)
                        else:
                            df[col] = df[col].astype(str)
                return df
            raise

    pd.DataFrame.to_parquet = _fallback_to_parquet  # type: ignore
    pd.read_parquet = _fallback_read_parquet  # type: ignore

_install_parquet_fallback_if_needed()

def compute_game_rhae(
    level_actions: List[Optional[int]],
    human_baselines: List[int],
) -> float:
    """
    Official ARC-AGI-3 Relative Human Action Efficiency (RHAE) formula (arXiv:2603.24621):
      S_{l,e} = min(1.15, (h_{l,e} / a_{l,e})^2) if level l completed else 0.0
      weighted_score = sum_{l=1..n} (l * S_{l,e}) / (n*(n+1)/2)
      completion_cap = sum_{l completed} l / (n*(n+1)/2)
      E_e = 100.0 * min(1.0, completion_cap, weighted_score)
    """
    n = len(human_baselines)
    if n == 0:
        return 0.0
    total_weight = float(n * (n + 1)) / 2.0
    weighted_sum = 0.0
    completed_weight = 0.0
    for idx in range(n):
        weight = float(idx + 1)
        a_le = level_actions[idx] if idx < len(level_actions) else None
        h_le = max(1, int(human_baselines[idx]))
        if a_le is not None and a_le > 0:
            s_le = min(1.15, (float(h_le) / float(a_le)) ** 2)
            weighted_sum += weight * s_le
            completed_weight += weight
    raw_frac = weighted_sum / total_weight
    cap_frac = completed_weight / total_weight
    return round(100.0 * min(1.0, cap_frac, raw_frac), 2)

def write_submission_parquet(
    rows: List[Dict[str, Any]],
    output_path: str = "submission.parquet",
) -> Path:
    if not rows:
        rows = [{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1.0}]
    df = pd.DataFrame(rows, columns=REQUIRED_PARQUET_COLUMNS)
    df["row_id"] = df["row_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["end_of_game"] = df["end_of_game"].astype(bool)
    df["score"] = df["score"].astype(float)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if Path("/kaggle/working").exists() and out.resolve() != Path("/kaggle/working/submission.parquet").resolve():
        df.to_parquet("/kaggle/working/submission.parquet", index=False)
    return out

def validate_submission_parquet(parquet_path: str = "submission.parquet") -> bool:
    p = Path(parquet_path)
    if not p.exists():
        return False
    df = pd.read_parquet(p)
    if list(df.columns) != REQUIRED_PARQUET_COLUMNS:
        return False
    if len(df) == 0:
        return False
    if df[REQUIRED_PARQUET_COLUMNS].isnull().any().any():
        return False
    return True

def _extract_available_action_ids(obs: Any, env: Any) -> List[int]:
    avail: List[int] = []
    raw_avail = getattr(obs, "available_actions", None)
    if raw_avail is None and hasattr(env, "action_space"):
        raw_avail = getattr(env, "action_space", None)
    if raw_avail:
        for a in raw_avail:
            if isinstance(a, int):
                avail.append(a)
            elif hasattr(a, "value") and isinstance(a.value, int):
                avail.append(int(a.value))
            elif hasattr(a, "name"):
                name = str(a.name).upper()
                if name.startswith("ACTION") and name[6:].isdigit():
                    avail.append(int(name[6:]))
    return avail or [1, 2, 3, 4, 5, 6]

def run_offline_evaluation(
    game_ids: Optional[List[str]] = None,
    environments_dir: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    output_json_path: Optional[str] = None,
    parquet_path: str = "submission.parquet",
    config_override: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    config = config_override or AgentConfig()
    if max_steps_override is not None:
        config.max_actions_per_game = max_steps_override
    agent = GeneralAgent(config=config)
    t_start = time.perf_counter()
    results_per_game: List[Dict[str, Any]] = []
    parquet_rows: List[Dict[str, Any]] = []

    try:
        import arc_agi
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction, GameState
    except ImportError as err:
        raise RuntimeError(
            f"Official arc_agi package is required for real ARC-AGI-3 evaluation: {err}"
        ) from err

    env_dir = environments_dir
    if env_dir is None:
        for candidate in ("environment_files", "arc-interactive/environment_files", "/kaggle/input/arc-agi-3/environment_files"):
            if Path(candidate).exists():
                env_dir = candidate
                break
    arcade_kwargs: Dict[str, Any] = {"operation_mode": OperationMode.OFFLINE}
    if env_dir is not None:
        arcade_kwargs["environments_dir"] = env_dir
    arc = Arcade(**arcade_kwargs)

    if not game_ids:
        if env_dir and Path(env_dir).exists():
            game_ids = sorted(
                [
                    d.name
                    for d in Path(env_dir).iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
            )[:25]
        else:
            game_ids = ["ls20", "ft09", "vc33"]

    for idx, gid in enumerate(game_ids):
        g_start = time.perf_counter()
        agent.reset_game()
        levels_cleared = 0
        steps_taken = 0
        level_step_counter = 0
        level_actions_list: List[Optional[int]] = []
        crashed = False
        failure_reason = ""
        try:
            env = arc.make(gid)
            if env is None:
                continue
            obs = env.reset()
            max_steps, max_time_s = agent.scheduler.allocate_game_budget(0)
            if max_steps_override is not None:
                max_steps = max_steps_override
            while steps_taken < max_steps and (time.perf_counter() - g_start) < max_time_s:
                if obs is None:
                    break
                state_obj = getattr(obs, "state", None)
                state_name = str(getattr(state_obj, "name", state_obj or "PLAYING")).upper()
                levels_completed = int(getattr(obs, "levels_completed", levels_cleared) or 0)
                if levels_completed > levels_cleared:
                    for _ in range(levels_completed - levels_cleared):
                        level_actions_list.append(max(1, level_step_counter))
                    levels_cleared = levels_completed
                    level_step_counter = 0
                    agent.scheduler.total_levels_cleared += 1
                    max_steps, max_time_s = agent.scheduler.allocate_game_budget(levels_cleared)
                    if max_steps_override is not None:
                        max_steps = max_steps_override
                if state_name == "WIN" or state_obj == GameState.WIN:
                    break
                if state_name in ("GAME_OVER", "NOT_PLAYED") or state_obj in (
                    GameState.GAME_OVER,
                    GameState.NOT_PLAYED,
                ):
                    obs = env.step(GameAction.RESET)
                    steps_taken += 1
                    level_step_counter += 1
                    continue
                avail_ids = _extract_available_action_ids(obs, env)
                act_id, act_data = agent.act(
                    raw_obs=obs,
                    available_actions=avail_ids,
                    levels_completed=levels_cleared,
                    state_str=state_name,
                )
                action_enum = getattr(GameAction, f"ACTION{act_id}", GameAction.ACTION1)
                if act_id == 0:
                    action_enum = GameAction.RESET
                if act_id == 6:
                    clean_data = dict(act_data) if act_data else {}
                    cx = max(0, min(63, int(clean_data.get("x", 0))))
                    cy = max(0, min(63, int(clean_data.get("y", 0))))
                    obs = env.step(action_enum, data={"x": cx, "y": cy})
                else:
                    obs = env.step(action_enum)
                steps_taken += 1
                level_step_counter += 1
        except Exception as exc:
            crashed = True
            failure_reason = str(exc)
            logger.warning(f"Game {gid} isolated exception: {exc}")

        official_baselines = getattr(getattr(env, "environment_info", None), "baseline_actions", None)
        if official_baselines:
            baselines = list(official_baselines)
            total_lvls = len(baselines)
            padded_actions = list(level_actions_list) + [None] * max(0, total_lvls - len(level_actions_list))
        else:
            total_lvls = max(5, levels_cleared)
            padded_actions: List[Optional[int]] = list(level_actions_list) + [None] * max(
                0, total_lvls - len(level_actions_list)
            )
            baselines = [12 + 3 * i for i in range(total_lvls)]
        game_score = compute_game_rhae(padded_actions, baselines)
        agent.scheduler.games_completed += 1
        results_per_game.append(
            {
                "game_id": gid,
                "score": game_score,
                "levels_cleared": levels_cleared,
                "actions": steps_taken,
                "runtime_s": round(time.perf_counter() - g_start, 3),
                "crashed": crashed,
                "failure_reason": failure_reason,
            }
        )
        parquet_rows.append(
            {
                "row_id": f"{gid}_{idx}",
                "game_id": str(gid),
                "end_of_game": True,
                "score": float(game_score),
            }
        )

    write_submission_parquet(parquet_rows, output_path=parquet_path)
    llm_inst = OfflineServerManager.get_instance(config)
    total_runtime = round(time.perf_counter() - t_start, 3)
    mean_score = (
        round(sum(r["score"] for r in results_per_game) / len(results_per_game), 2)
        if results_per_game
        else 0.0
    )
    summary = {
        "games_attempted": len(results_per_game),
        "games_completed_without_crash": sum(1 for r in results_per_game if not r["crashed"]),
        "mean_score": mean_score,
        "levels_cleared": sum(r["levels_cleared"] for r in results_per_game),
        "total_runtime_s": total_runtime,
        "mean_actions_per_game": (
            round(sum(r["actions"] for r in results_per_game) / len(results_per_game), 2)
            if results_per_game
            else 0.0
        ),
        "llm_loaded": llm_inst.is_ready,
        "llm_fallback_rate": (
            round(llm_inst.fallback_count / float(max(1, llm_inst.total_calls)), 3)
            if llm_inst.total_calls > 0
            else 1.0
        ),
        "per_game": results_per_game,
    }
    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def run_submission_pipeline(parquet_path: str = "submission.parquet") -> Dict[str, Any]:
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        return run_offline_evaluation(parquet_path=parquet_path)
    return run_offline_evaluation(
        max_steps_override=110,
        output_json_path="results/smoke_test.json",
        parquet_path=parquet_path,
    )
