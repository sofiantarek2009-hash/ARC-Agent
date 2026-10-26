#!/usr/bin/env python3
"""Builds self-contained single-cell Kaggle submission notebook/script for ARC-AGI-3 v5.0."""

import json
from pathlib import Path

ACTIVE_FILES = [
    "project_override/__init__.py",
    "project_override/config.py",
    "project_override/perception/__init__.py",
    "project_override/perception/observation_processor.py",
    "project_override/execution/__init__.py",
    "project_override/execution/avo_supervisor.py",
    "project_override/memory/__init__.py",
    "project_override/memory/action_ledger.py",
    "project_override/memory/hypothesis_store.py",
    "project_override/memory/lineage_dag.py",
    "project_override/memory/semantic_compactor.py",
    "project_override/planning/__init__.py",
    "project_override/planning/domain_solvers.py",
    "project_override/planning/verifier_gate.py",
    "project_override/planning/general_explorer.py",
    "project_override/core/__init__.py",
    "project_override/core/goal_induction_engine.py",
    "project_override/core/simulated_planner.py",
    "project_override/core/world_model_engine.py",
    "project_override/llm/__init__.py",
    "project_override/llm/codeact_engine.py",
    "project_override/prompts/__init__.py",
    "project_override/prompts/system_prompt.py",
    "project_override/orchestration/__init__.py",
    "project_override/orchestration/general_agent.py",
    "project_override/submission/__init__.py",
    "project_override/submission/kaggle_submission.py",
    "project_override/agent/my_agent.py",
    "agent/my_agent.py",
]

def build_single_cell() -> str:
    embedded = {}
    for rel_path in ACTIVE_FILES:
        p = Path(rel_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required active file: {rel_path}")
        embedded[rel_path] = p.read_text(encoding="utf-8")

    embedded_json = json.dumps(embedded, indent=2)

    script = f'''# ==============================================================================
# ARC PRIZE 2026 (ARC-AGI-3) SUBMISSION NOTEBOOK
# PROJECT OVERRIDE v5.0: NVIDIA AVO + CODEACT REPL ARCHITECTURE
# ==============================================================================

# --- CELL 0: INSTALL OFFICIAL OFFLINE WHEELS ---
import os, sys, subprocess
from pathlib import Path

for wheels_path in (
    Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'),
    Path('/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'),
):
    if wheels_path.exists():
        print(f'[WHEELS] Installing offline wheels from {{wheels_path}}...')
        cmd = [
            sys.executable, '-m', 'pip', 'install', '--no-index',
            '--find-links', str(wheels_path),
            'arc-agi', 'python-dotenv'
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print('[OK] Offline dependencies successfully installed.')
        else:
            print(f'[WARNING] Pip install returned code {{res.returncode}}: {{res.stderr[:200]}}')
        break
else:
    print('[INFO] Local development environment (wheels directory not present).')


# --- CELL 1: UNPACK PROJECT OVERRIDE V5.0 MODULES ---
import json, os, sys
from pathlib import Path

EMBEDDED_FILES = {embedded_json}

for rel_path, content in EMBEDDED_FILES.items():
    p = Path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')

for extra_dir in ['.', 'agent', 'project_override']:
    if extra_dir not in sys.path:
        sys.path.insert(0, extra_dir)

print(f'[OK] Unpacked {{len(EMBEDDED_FILES)}} v5.0 module files into environment.')


# --- CELL 2: OFFLINE vLLM SERVER INITIALIZATION ---
from project_override.config import AgentConfig
from project_override.llm.codeact_engine import OfflineServerManager

config = AgentConfig()
server_mgr = OfflineServerManager.get_instance(config)
server_online = server_mgr.ensure_server_running()
if server_online:
    print(f'[OK] LLM inference engine online: {{server_mgr.http_model_name}} at {{server_mgr.http_endpoint}}')
else:
    print('[INFO] Running in autonomous exploration mode (no local LLM server active).')


# --- CELL 3: RUN OFFICIAL COMPETITION EVALUATION & WRITE PARQUET ---
from project_override.submission.kaggle_submission import run_offline_evaluation

is_competition_rerun = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
print(f'=== Starting ARC-AGI-3 Evaluation: KAGGLE_IS_COMPETITION_RERUN={{is_competition_rerun}} ===')

max_steps = None if is_competition_rerun else 120
summary = run_offline_evaluation(
    max_steps_override=max_steps,
    output_json_path='results/evaluation_summary.json',
    parquet_path='submission.parquet',
    config_override=config,
)

print('=== Evaluation Finished ===')
print(f"Games Attempted: {{summary['games_attempted']}}")
print(f"Completed Without Crash: {{summary['games_completed_without_crash']}}")
print(f"Mean RHAE Score: {{summary['mean_score']}}%")
print(f"Levels Cleared: {{summary['levels_cleared']}}")
print(f"Total Runtime: {{summary['total_runtime_s']}}s")
print(f"LLM Loaded: {{summary['llm_loaded']}} (Fallback rate: {{summary['llm_fallback_rate']}})")

# Verify submission.parquet exists and has required schema
import pandas as pd
df = pd.read_parquet('submission.parquet')
print(f'[OK] Verified submission.parquet with {{len(df)}} rows. Columns: {{list(df.columns)}}')
'''
    return script

if __name__ == "__main__":
    out_path = Path("dist/kaggle_single_cell_submission.py")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_single_cell()
    out_path.write_text(content, encoding="utf-8")
    print(f"Successfully generated {out_path} ({len(content)} bytes).")
