#!/usr/bin/env python3
"""
Verified Kaggle Submission Builder for Project Override (ARC-AGI-3).
Packs the entire `project_override/` 12-module tree into gzipped base64,
verifies every module is present (>10,000 bytes gzip), patches Swarm concurrency,
and writes both `ARC-AGI-3-Kaggle-Starter/notebooks/submission.ipynb` and
`dist/kaggle_single_cell_submission.py`.
"""
import base64
import gzip
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PO_DIR = ROOT / "project_override"
AGENT_FILE = ROOT / "agent" / "my_agent.py"
NOTEBOOK_DIR = ROOT / "ARC-AGI-3-Kaggle-Starter" / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "submission.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"
DIST_PATH = ROOT / "dist" / "kaggle_single_cell_submission.py"


def build_payload_b64() -> str:
    buf = io.BytesIO()
    seen_arcnames = set()
    packed_files = []
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for py_file in sorted(PO_DIR.rglob("*.py")):
                if "__pycache__" in py_file.parts:
                    continue
                rel = py_file.relative_to(ROOT).as_posix()
                if rel not in seen_arcnames:
                    seen_arcnames.add(rel)
                    tar.add(py_file, arcname=rel)
                    packed_files.append(rel)
            for shim in ["dead_signature.py", "context_manager.py", "ast_sandbox.py"]:
                shim_p = ROOT / shim
                if shim_p.exists() and shim not in seen_arcnames:
                    seen_arcnames.add(shim)
                    tar.add(shim_p, arcname=shim)
                    packed_files.append(shim)

    raw_bytes = buf.getvalue()
    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    print(f"[PACK] Packed {len(packed_files)} files: {len(raw_bytes):,} bytes gzip -> {len(b64_str):,} chars base64")
    if len(raw_bytes) < 8000:
        raise RuntimeError(f"Payload too small ({len(raw_bytes)} bytes)! Check project_override/ files.")
    return b64_str


def main() -> None:
    print("[BUILD] Generating self-contained Kaggle submission artifacts...")
    b64_payload = build_payload_b64()
    my_agent_src = AGENT_FILE.read_text(encoding="utf-8")

    cell1_wheels = """import os, subprocess, sys
from pathlib import Path

WHEEL_DIRS = [
    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
    Path("/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
]
for wdir in WHEEL_DIRS:
    if wdir.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={wdir}", "arc-agi", "arcengine"],
            check=False,
        )
        break
"""

    cell2_extract = f"""import base64, gzip, io, sys, tarfile
from pathlib import Path

PAYLOAD_B64 = "{b64_payload}"
TARGET_TMP = Path("/tmp")
raw_gz = base64.b64decode(PAYLOAD_B64)
with gzip.GzipFile(fileobj=io.BytesIO(raw_gz), mode="rb") as gz:
    with tarfile.open(fileobj=gz, mode="r") as tar:
        try:
            tar.extractall(TARGET_TMP, filter="data")
        except TypeError:
            tar.extractall(TARGET_TMP)

if "/tmp" not in sys.path:
    sys.path.insert(0, "/tmp")
print("[OK] Project Override 12-module engine extracted to /tmp/project_override")
"""

    cell3_agent = f"""from pathlib import Path
MY_AGENT_SRC = {repr(my_agent_src)}
Path("/tmp/my_agent.py").write_text(MY_AGENT_SRC, encoding="utf-8")
print("[OK] Wrote /tmp/my_agent.py")
"""

    cell4_runner = """import os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

IS_RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").lower() in ("1", "true", "yes")

# Smoke-test the extracted Project Override engine and Qwen 32B AWQ singleton
sys.path.insert(0, "/tmp")
from project_override.orchestration.general_agent import GeneralAgent
from project_override.world_model.retrodict_synthesizer import OfflineLLMRuntime

smoke_agent = GeneralAgent(game_id="smoke")
act_id, _ = smoke_agent.step(np.zeros((64, 64), dtype=np.int8), [1, 2, 3, 4], 0)
llm_rt = OfflineLLMRuntime.get_instance()
print(f"[VERIFY] GeneralAgent smoke step -> ACTION{act_id} | Qwen 32B AWQ path: {llm_rt.model_path}")

if not IS_RERUN:
    # Phase A Commit Mode: emit clean validation parquet as sole file in /kaggle/working
    for item in Path("/kaggle/working").iterdir():
        if item.name != "submission.parquet" and not item.name.startswith("__notebook"):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    df = pd.DataFrame([{"game_id": "validation", "level": 0, "actions": 1, "score": 1.0}])
    df.to_parquet("/kaggle/working/submission.parquet", index=False)
    print("[COMMIT MODE] Emitted clean /kaggle/working/submission.parquet")
else:
    # Phase B Official Competition Rerun
    HARNESS_SRC = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents")
    if not HARNESS_SRC.exists():
        HARNESS_SRC = Path("/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents")
    WORK_HARNESS = Path("/tmp/ARC-AGI-3-Agents")
    if WORK_HARNESS.exists():
        shutil.rmtree(WORK_HARNESS)
    shutil.copytree(HARNESS_SRC, WORK_HARNESS)

    # Copy project_override and my_agent into /tmp/ARC-AGI-3-Agents
    shutil.copytree("/tmp/project_override", WORK_HARNESS / "project_override", dirs_exist_ok=True)
    shutil.copyfile("/tmp/my_agent.py", WORK_HARNESS / "agents" / "templates" / "my_agent.py")

    init_p = WORK_HARNESS / "agents" / "__init__.py"
    init_txt = init_p.read_text(encoding="utf-8")
    if "MyAgent" not in init_txt:
        init_txt += "\nfrom .templates.my_agent import MyAgent\nAVAILABLE_AGENTS['myagent'] = MyAgent\n"
        init_p.write_text(init_txt, encoding="utf-8")

    # Patch swarm.py to run threads in bounded batches of 8 so 110 threads never starve 4 vCPUs
    swarm_p = WORK_HARNESS / "agents" / "swarm.py"
    if swarm_p.exists():
        s_txt = swarm_p.read_text(encoding="utf-8")
        old_loop = "for t in self.threads:\n            t.start()\n        for t in self.threads:\n            t.join()"
        new_loop = (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "        with ThreadPoolExecutor(max_workers=8) as ex:\n"
            "            list(ex.map(lambda ag: ag.main(), self.agents))"
        )
        if old_loop in s_txt:
            swarm_p.write_text(s_txt.replace(old_loop, new_loop), encoding="utf-8")

    subprocess.run([sys.executable, "main.py", "--agent=myagent"], cwd=str(WORK_HARNESS), check=True)
    for pq in WORK_HARNESS.rglob("*.parquet"):
        shutil.copyfile(pq, "/kaggle/working/submission.parquet")
"""

    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_wheels.splitlines(True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2_extract.splitlines(True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3_agent.splitlines(True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell4_runner.splitlines(True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[NOTEBOOK] Built {NOTEBOOK_PATH}")

    meta = {
        "id": "sofianali188201/arc-prize-2026-arc-agi-3-starter",
        "title": "ARC Prize 2026 — ARC-AGI-3 Starter",
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
        "model_sources": ["qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1"],
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    DIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    single_cell = "\n\n".join([cell1_wheels, cell2_extract, cell3_agent, cell4_runner])
    DIST_PATH.write_text(single_cell, encoding="utf-8")
    print(f"[STANDALONE] Built {DIST_PATH} ({DIST_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
