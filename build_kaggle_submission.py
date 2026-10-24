#!/usr/bin/env python3
"""
Builds self-contained Kaggle submission artifacts for ARC-AGI-3:
- notebooks/submission.ipynb
- ARC-AGI-3-Kaggle-Starter/notebooks/submission.ipynb
- dist/kaggle_single_cell_submission.py
- Synchronizes kernel-metadata.json across notebook directories
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

MODULE_FILES = [
    "project_override/__init__.py",
    "project_override/config.py",
    "project_override/perception/__init__.py",
    "project_override/perception/observation_processor.py",
    "project_override/memory/__init__.py",
    "project_override/memory/action_ledger.py",
    "project_override/memory/hypothesis_store.py",
    "project_override/planning/__init__.py",
    "project_override/planning/explorer.py",
    "project_override/planning/planner.py",
    "project_override/llm/__init__.py",
    "project_override/llm/local_runtime.py",
    "project_override/llm/repl.py",
    "project_override/execution/__init__.py",
    "project_override/execution/safe_executor.py",
    "project_override/orchestration/__init__.py",
    "project_override/orchestration/general_agent.py",
    "project_override/submission/__init__.py",
    "project_override/submission/kaggle_submission.py",
    "agent/my_agent.py",
]

def build_all() -> None:
    print("[BUILD] Packaging self-contained ARC-AGI-3 agent modules...")
    bundle_dict = {}
    for rel_path in MODULE_FILES:
        fp = ROOT / rel_path
        if not fp.exists():
            raise FileNotFoundError(f"Required module file missing: {fp}")
        bundle_dict[rel_path] = fp.read_text(encoding="utf-8")
    print(f"[PACK] Bundled {len(bundle_dict)} module files ({sum(len(v) for v in bundle_dict.values()):,} chars).")

    cell_1_unpack = (
        "# --- CELL 1: UNPACK SELF-CONTAINED PROJECT OVERRIDE V3.0 MODULES ---\n"
        "import json, os, sys\n"
        "from pathlib import Path\n\n"
        f"EMBEDDED_FILES = {json.dumps(bundle_dict, indent=2)}\n\n"
        "for rel_path, content in EMBEDDED_FILES.items():\n"
        "    p = Path(rel_path)\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_text(content, encoding='utf-8')\n\n"
        "for extra_dir in ['.', 'agent', 'project_override']:\n"
        "    if extra_dir not in sys.path:\n"
        "        sys.path.insert(0, extra_dir)\n\n"
        "print(f'[OK] Unpacked {len(EMBEDDED_FILES)} module files into environment.')\n"
    )

    cell_2_execute = (
        "# --- CELL 2: EXECUTE ARC-AGI-3 SUBMISSION PIPELINE ---\n"
        "import os, sys\n"
        "from pathlib import Path\n\n"
        "from project_override.submission.kaggle_submission import (\n"
        "    run_submission_pipeline,\n"
        "    validate_submission_parquet,\n"
        ")\n\n"
        "print('=== Executing ARC-AGI-3 Submission Pipeline ===')\n"
        "summary = run_submission_pipeline(parquet_path='submission.parquet')\n"
        "assert validate_submission_parquet('submission.parquet'), 'Invalid submission.parquet schema!'\n\n"
        "if Path('/kaggle/working').exists() and Path('submission.parquet').exists():\n"
        "    import shutil\n"
        "    shutil.copyfile('submission.parquet', '/kaggle/working/submission.parquet')\n"
        "    print('[OK] Copied submission.parquet to /kaggle/working/submission.parquet')\n\n"
        "print('Submission pipeline finished successfully:', summary)\n"
    )

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in cell_1_unpack.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in cell_2_execute.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Write root notebooks/submission.ipynb
    nb_dir1 = ROOT / "notebooks"
    nb_dir1.mkdir(parents=True, exist_ok=True)
    nb_path1 = nb_dir1 / "submission.ipynb"
    nb_path1.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[NOTEBOOK] Generated {nb_path1}")

    # Write ARC-AGI-3-Kaggle-Starter/notebooks/submission.ipynb
    nb_dir2 = ROOT / "ARC-AGI-3-Kaggle-Starter" / "notebooks"
    nb_dir2.mkdir(parents=True, exist_ok=True)
    nb_path2 = nb_dir2 / "submission.ipynb"
    nb_path2.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[NOTEBOOK] Generated {nb_path2}")

    # Write dist/kaggle_single_cell_submission.py
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    dist_file = dist_dir / "kaggle_single_cell_submission.py"
    single_cell_code = f"{cell_1_unpack}\n\n{cell_2_execute}\n"
    dist_file.write_text(single_cell_code, encoding="utf-8")
    print(f"[STANDALONE] Generated {dist_file}")

    # Synchronize kernel-metadata.json
    meta = {
        "id": "sofianali188201/arc-prize-2026-arc-agi-3-starter",
        "title": "ARC Prize 2026 - ARC-AGI-3 Starter V3.0",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [
            "intel-qwen3-8-flash-next-w4a16-autoround",
            "qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1",
        ],
    }
    for m_dir in (nb_dir1, nb_dir2):
        m_path = m_dir / "kernel-metadata.json"
        m_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[METADATA] Generated {m_path}")

    print("[SUCCESS] All Kaggle submission artifacts built cleanly.")

if __name__ == "__main__":
    build_all()
