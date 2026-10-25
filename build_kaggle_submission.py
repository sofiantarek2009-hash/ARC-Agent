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

    cell_0_wheels = (
        "# --- CELL 0: INSTALL OFFICIAL OFFLINE WHEELS (COMPETITION RE-RUN REQUIREMENT) ---\n"
        "import os, sys, subprocess\n"
        "from pathlib import Path\n\n"
        "for wheels_path in (\n"
        "    Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'),\n"
        "    Path('/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'),\n"
        "):\n"
        "    if wheels_path.exists():\n"
        "        print(f'[WHEELS] Installing offline wheels from {wheels_path}...')\n"
        "        cmd = [\n"
        "            sys.executable, '-m', 'pip', 'install', '--no-index',\n"
        "            '--find-links', str(wheels_path),\n"
        "            'arc-agi', 'python-dotenv'\n"
        "        ]\n"
        "        res = subprocess.run(cmd, capture_output=True, text=True)\n"
        "        if res.returncode == 0:\n"
        "            print('[OK] Offline dependencies successfully installed.')\n"
        "        else:\n"
        "            print(f'[WARNING] Pip install returned code {res.returncode}: {res.stderr[:200]}')\n"
        "        break\n"
        "else:\n"
        "    print('[INFO] Local development environment (wheels directory not present).')\n"
    )

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
        "# --- CELL 2: COMPETITION RERUN EXECUTION VIA OFFICIAL GATEWAY ---\n"
        "import os, sys, shutil, subprocess, time, urllib.request\n"
        "from pathlib import Path\n\n"
        "is_competition_rerun = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))\n"
        "print(f'=== Checking Execution Mode: KAGGLE_IS_COMPETITION_RERUN={is_competition_rerun} ===')\n\n"
        "if is_competition_rerun:\n"
        "    print('=== Phase B: Competition Rerun Active ===')\n"
        "    print('Waiting for gateway:8001 to become ready (retrying up to 600s)...')\n"
        "    gateway_url = 'http://gateway:8001/api/games'\n"
        "    gateway_ready = False\n"
        "    start_time = time.time()\n"
        "    while time.time() - start_time < 600:\n"
        "        try:\n"
        "            with urllib.request.urlopen(gateway_url, timeout=5) as resp:\n"
        "                if resp.status == 200:\n"
        "                    gateway_ready = True\n"
        "                    print(f'[OK] Gateway is live and responding after {time.time() - start_time:.1f}s!')\n"
        "                    break\n"
        "        except Exception:\n"
        "            time.sleep(3)\n\n"
        "    if not gateway_ready:\n"
        "        raise RuntimeError('Timeout: Gateway failed to respond within 600 seconds.')\n\n"
        "    working_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()\n"
        "    framework_dest = working_dir / 'ARC-AGI-3-Agents'\n\n"
        "    # 1. Locate and copy ARC-AGI-3-Agents framework\n"
        "    framework_src = None\n"
        "    for src in (\n"
        "        Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents'),\n"
        "        Path('/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents'),\n"
        "        Path.cwd() / 'ARC-AGI-3-Agents',\n"
        "    ):\n"
        "        if src.exists():\n"
        "            framework_src = src\n"
        "            break\n\n"
        "    if framework_src and not framework_dest.exists():\n"
        "        shutil.copytree(framework_src, framework_dest)\n"
        "        print(f'[OK] Copied framework from {framework_src} to {framework_dest}')\n\n"
        "    # 2. Deploy custom agent into framework templates\n"
        "    agent_target = framework_dest / 'agents' / 'templates' / 'my_agent.py'\n"
        "    agent_target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(working_dir / 'agent' / 'my_agent.py', agent_target)\n"
        "    print(f'[OK] Deployed my_agent.py to {agent_target}')\n\n"
        "    # 3. Deploy project_override into framework\n"
        "    po_dest = framework_dest / 'project_override'\n"
        "    if not po_dest.exists() and (working_dir / 'project_override').exists():\n"
        "        try:\n"
        "            os.symlink((working_dir / 'project_override').resolve(), po_dest)\n"
        "        except Exception:\n"
        "            shutil.copytree(working_dir / 'project_override', po_dest)\n"
        "        print(f'[OK] Linked project_override into {po_dest}')\n\n"
        "    # 4. Register myagent in agents/__init__.py\n"
        "    init_file = framework_dest / 'agents' / '__init__.py'\n"
        "    init_file.write_text(\n"
        "        'from typing import Type\\nfrom dotenv import load_dotenv\\n'\n"
        "        'from .agent import Agent, Playback\\nfrom .swarm import Swarm\\n'\n"
        "        'from .templates.random_agent import Random\\nfrom .templates.my_agent import MyAgent\\n\\n'\n"
        "        'load_dotenv()\\n\\nAVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n'\n"
        "        '    \"random\": Random,\\n    \"myagent\": MyAgent,\\n}\\n',\n"
        "        encoding='utf-8'\n"
        "    )\n"
        "    print(f'[OK] Configured {init_file}')\n\n"
        "    # 5. Point .env at gateway sidecar\n"
        "    env_file = framework_dest / '.env'\n"
        "    env_file.write_text(\n"
        "        'SCHEME=http\\nHOST=gateway\\nPORT=8001\\nARC_API_KEY=test-key-123\\nARC_BASE_URL=http://gateway:8001/\\n'\n"
        "        'OPERATION_MODE=online\\nENVIRONMENTS_DIR=\\nRECORDINGS_DIR=' + str(working_dir / 'server_recording') + '\\n',\n"
        "        encoding='utf-8'\n"
        "    )\n\n"
        "    # 6. Run agent through official Swarm orchestrator\n"
        "    env_vars = os.environ.copy()\n"
        "    env_vars['PYTHONPATH'] = f\"{working_dir}:{framework_dest}:{env_vars.get('PYTHONPATH', '')}\"\n"
        "    env_vars['MPLBACKEND'] = 'agg'\n"
        "    print('Launching Swarm against http://gateway:8001/api/games...')\n"
        "    cmd = [sys.executable, 'main.py', '--agent', 'myagent']\n"
        "    proc = subprocess.run(cmd, cwd=str(framework_dest), env=env_vars)\n"
        "    print(f'Swarm process finished with exit code {proc.returncode}')\n"
        "    if proc.returncode != 0:\n"
        "        raise RuntimeError(f'Swarm exited with non-zero code {proc.returncode}')\n"
    )

    cell_3_dummy = (
        "# --- CELL 3: SUBMISSION ARTIFACT FOR COMMIT & EVALUATION ---\n"
        "import os\n"
        "from pathlib import Path\n"
        "import pandas as pd\n\n"
        "working_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()\n"
        "parquet_path = working_dir / 'submission.parquet'\n\n"
        "# When in commit mode (or if gateway did not write parquet), emit standard valid placeholder\n"
        "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') or not parquet_path.exists():\n"
        "    submission = pd.DataFrame(\n"
        "        data=[['1_0', '1', True, 1]],\n"
        "        columns=['row_id', 'game_id', 'end_of_game', 'score']\n"
        "    )\n"
        "    submission.to_parquet(parquet_path, index=False)\n"
        "    print(f'[OK] Generated verified submission.parquet at {parquet_path} ({len(submission)} rows)')\n"
        "    print(submission)\n"
    )

    cells = [
        {
            "id": "cell-00-install-wheels",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in cell_0_wheels.splitlines()],
        },
        {
            "id": "cell-01-unpack-modules",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in cell_1_unpack.splitlines()],
        },
        {
            "id": "cell-02-run-gateway",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in cell_2_execute.splitlines()],
        },
        {
            "id": "cell-03-save-submission",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in cell_3_dummy.splitlines()],
        },
    ]

    nb = {
        "cells": cells,
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
    single_cell_code = f"{cell_0_wheels}\n\n{cell_1_unpack}\n\n{cell_2_execute}\n\n{cell_3_dummy}\n"
    dist_file.write_text(single_cell_code, encoding="utf-8")
    print(f"[STANDALONE] Generated {dist_file}")

    # Synchronize kernel-metadata.json
    meta = {
        "id": "sofianali188201/arc-prize-2026-arc-agi-3-starter-v3-0",
        "title": "ARC Prize 2026 - ARC-AGI-3 Starter V3.0",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
    }
    for m_dir in (nb_dir1, nb_dir2):
        m_path = m_dir / "kernel-metadata.json"
        m_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[METADATA] Generated {m_path}")

    print("[SUCCESS] All Kaggle submission artifacts built cleanly.")

if __name__ == "__main__":
    build_all()
