#!/usr/bin/env python3
"""ARC-AGI-3 Automated Kaggle Submission Packager & Bundler
Packages the complete Project Override agent and architecture into:
1. ARC-AGI-3-Kaggle-Starter/notebooks/submission.ipynb (Self-contained Kaggle notebook)
2. dist/kaggle_single_cell_submission.py (Single-cell copy-paste script)
3. Synchronizes kernel-metadata.json with proper GPU / competition settings

Zero manual notebook editing required.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
PROJECT_OVERRIDE_DIR = ROOT / "project_override" / "project_override"
DEAD_SIG_SRC = ROOT / "project_override" / "dead_signature.py"
AGENT_SRC = ROOT / "agent" / "my_agent.py"
STARTER_DIR = ROOT / "ARC-AGI-3-Kaggle-Starter"
NOTEBOOK_DIR = STARTER_DIR / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "submission.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"
DIST_DIR = ROOT / "dist"

ACCELERATOR = "t4"  # Options: "t4" (dual T4), "p100", "rtx6000", "cpu"
_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}


def create_codebase_payload() -> str:
    """Packs project_override/ and dead_signature.py into a gzipped, base64-encoded string."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add project_override package
        if PROJECT_OVERRIDE_DIR.exists():
            for p in PROJECT_OVERRIDE_DIR.rglob("*.py"):
                if "__pycache__" in p.parts:
                    continue
                rel_path = p.relative_to(ROOT / "project_override")
                tar.add(p, arcname=str(rel_path))

        # Add dead_signature.py
        if DEAD_SIG_SRC.exists():
            tar.add(DEAD_SIG_SRC, arcname="dead_signature.py")

    buf.seek(0)
    compressed_data = buf.read()
    b64_str = base64.b64encode(compressed_data).decode("ascii")
    print(f"[PACK] Packaged codebase payload: {len(compressed_data):,} bytes gzip -> {len(b64_str):,} chars base64")
    return b64_str


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build_notebook() -> dict:
    payload_b64 = create_codebase_payload()
    agent_body = AGENT_SRC.read_text(encoding="utf-8")

    # Cell 1: Install offline wheel dependencies
    c1 = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # Cell 2: Unpack project_override payload
    unpack_code = dedent(f'''\
        # Unpack self-contained project_override package into local environment
        import base64
        import io
        import os
        import sys
        import tarfile

        PAYLOAD = b"""{payload_b64}"""

        for target_dir in ["/kaggle/working", "/tmp", "."]:
            try:
                os.makedirs(target_dir, exist_ok=True)
                buf = io.BytesIO(base64.b64decode(PAYLOAD))
                with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                    tar.extractall(target_dir)
                if target_dir not in sys.path:
                    sys.path.insert(0, target_dir)
            except Exception as e:
                pass

        print("[OK] Project Override architecture extracted and registered in sys.path")
    ''')
    c2 = code_cell(unpack_code)

    # Cell 3: Write my_agent.py
    c3 = code_cell(f"%%writefile /tmp/my_agent.py\n{agent_body}")

    # Cell 4: Competition rerun vs commit execution
    run_code = dedent("""\
        import os
        import sys

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("=" * 70)
            print(" ARC-AGI-3 OFFICIAL COMPETITION RERUN INITIALIZED")
            print("=" * 70)

            # 1. Wait for gateway sidecar service
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # 2. Copy framework into writable location
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # 3. Inject our unpacked architecture into framework directory
            !cp -r /kaggle/working/project_override /kaggle/working/ARC-AGI-3-Agents/
            !cp /kaggle/working/dead_signature.py /kaggle/working/ARC-AGI-3-Agents/ 2>/dev/null || true

            # 4. Inject MyAgent as framework template
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # 5. Register MyAgent in agents/__init__.py
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write('''from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        ''')

            # 6. Configure sidecar connection
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write('''SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        ''')

            # 7. Execute official evaluation loop
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent

            print("[COMPLETE] Evaluation finished. Generated submission.parquet:")
            !ls -lh /kaggle/working/submission.parquet 2>/dev/null || ls -lh submission.parquet 2>/dev/null || true
    """)
    c4 = code_cell(run_code)

    # Cell 5: Phase A dummy submission for instant commit validation (<0.05s)
    c5 = code_cell(dedent("""\
        import os
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Save & Run All (Commit) mode: emit valid dummy parquet immediately
            import pandas as pd
            submission = pd.DataFrame(
                data=[['1_0', '1', True, 1.0]],
                columns=['row_id', 'game_id', 'end_of_game', 'score']
            )
            out_dirs = ['/kaggle/working/submission.parquet', 'submission.parquet']
            for out in out_dirs:
                try:
                    submission.to_parquet(out, index=False)
                    print(f"[COMMIT MODE] Emitted validation submission at {out}")
                    break
                except Exception:
                    pass
    """))

    accel = _ACCELERATORS[ACCELERATOR]
    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Kaggle Submission\n\n"
                "Autonomous Singularly-Gated Fluid Intelligence Agent (Project Override).\n"
                "- Built automatically via `build_kaggle_submission.py`.\n"
                "- Dual-Phase runtime: Phase A commit validation (<0.1s) and Phase B hidden competition evaluation."
            ),
            c1,
            c2,
            c3,
            c4,
            c5,
        ],
    }
    return notebook


def sync_kernel_metadata() -> None:
    if not METADATA_PATH.exists():
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": "sofiantarek2009/arc-prize-2026-arc-agi-3-agent",
            "title": "ARC Prize 2026 — ARC-AGI-3 Agent",
            "code_file": "submission.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-3"],
            "model_sources": [],
        }
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[METADATA] Initialized {METADATA_PATH.relative_to(ROOT)}")
    else:
        meta = json.loads(METADATA_PATH.read_text())
        # Replace default placeholder if present
        if "REPLACE_WITH_YOUR_USERNAME" in meta.get("id", ""):
            meta["id"] = meta["id"].replace("REPLACE_WITH_YOUR_USERNAME", "sofiantarek2009")
        meta["enable_gpu"] = _ACCELERATORS[ACCELERATOR]["gpu"]
        meta["code_file"] = "submission.ipynb"
        meta["kernel_type"] = "notebook"
        meta["enable_internet"] = False
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[METADATA] Synchronized {METADATA_PATH.relative_to(ROOT)} with ID: {meta['id']}")


def generate_single_cell_script(payload_b64: str) -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DIST_DIR / "kaggle_single_cell_submission.py"
    agent_body = AGENT_SRC.read_text(encoding="utf-8")

    agents_init_str = (
        "from typing import Type\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import MyAgent\n\n"
        "load_dotenv()\n\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n"
        "    'random': Random,\n"
        "    'myagent': MyAgent,\n"
        "}\n"
    )

    env_file_str = (
        "SCHEME=http\n"
        "HOST=gateway\n"
        "PORT=8001\n"
        "ARC_API_KEY=test-key-123\n"
        "ARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    )

    header = "# ARC Prize 2026 — ARC-AGI-3 Single-Cell Standalone Submission Script\n"
    header += "# Copy-paste this entire script into a single cell of any Kaggle notebook and run.\n"

    script_parts = [
        header,
        "import os\nimport sys\nimport io\nimport base64\nimport tarfile\n\n",
        "print('[KAGGLE CELL] Unpacking Project Override Agent Architecture...')\n",
        f"PAYLOAD = b'''{payload_b64}'''\n",
        "for target_dir in ['/kaggle/working', '/tmp', '.']:\n"
        "    try:\n"
        "        os.makedirs(target_dir, exist_ok=True)\n"
        "        buf = io.BytesIO(base64.b64decode(PAYLOAD))\n"
        "        with tarfile.open(fileobj=buf, mode='r:gz') as tar:\n"
        "            tar.extractall(target_dir)\n"
        "        if target_dir not in sys.path:\n"
        "            sys.path.insert(0, target_dir)\n"
        "    except Exception:\n"
        "        pass\n\n",
        "with open('/tmp/my_agent.py', 'w') as f:\n",
        f"    f.write({repr(agent_body)})\n\n",
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
        "    print('[KAGGLE CELL] Starting Phase B Competition Evaluation...')\n",
        "    os.system('curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games')\n",
        "    os.system('cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents')\n",
        "    os.system('cp -r /kaggle/working/project_override /kaggle/working/ARC-AGI-3-Agents/')\n",
        "    os.system('cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py')\n",
        f"    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:\n        f.write({repr(agents_init_str)})\n",
        f"    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:\n        f.write({repr(env_file_str)})\n",
        "    os.system('cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent')\n",
        "else:\n",
        "    print('[KAGGLE CELL] Commit Mode: Generating valid submission.parquet...')\n",
        "    import pandas as pd\n",
        "    df = pd.DataFrame(data=[['1_0', '1', True, 1.0]], columns=['row_id', 'game_id', 'end_of_game', 'score'])\n",
        "    df.to_parquet('/kaggle/working/submission.parquet', index=False)\n",
        "    print('[KAGGLE CELL] Done! submission.parquet generated successfully.')\n",
    ]

    out_file.write_text("".join(script_parts), encoding="utf-8")
    print(f"[STANDALONE] Wrote single-cell script to {out_file.relative_to(ROOT)}")


def main() -> None:
    print("[BUILD] Generating self-contained Kaggle submission notebook...")
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[NOTEBOOK] Successfully built {NOTEBOOK_PATH.relative_to(ROOT)}")

    # Also keep root copy in notebooks/
    root_nb_dir = ROOT / "notebooks"
    root_nb_dir.mkdir(parents=True, exist_ok=True)
    (root_nb_dir / "submission.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (root_nb_dir / "kernel-metadata.json").write_text(METADATA_PATH.read_text() if METADATA_PATH.exists() else "", encoding="utf-8")

    sync_kernel_metadata()
    payload_b64 = create_codebase_payload()
    generate_single_cell_script(payload_b64)
    print("\n[SUCCESS] Kaggle submission artifacts are 100% prepared and ready to submit!")


if __name__ == "__main__":
    main()
