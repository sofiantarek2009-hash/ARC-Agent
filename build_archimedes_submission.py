"""Builds the complete Archimedes-3 submission notebook for Kaggle.

Packages the entire ARC3-Inference tree (with Belief-Space PUCT planner,
Hierarchical Macro-Options, Kepler Commit Gate, 5-Type Abductive Fault Localizer,
NVIDIA AVO Supervisor, Speed-Depth BOED Acquisition, Five-Tier Memory Ledger,
and SGLang Issue #36537 fix) into runs/dfranzen_exact_submission/arc-agi-3-milestone-2-solution.ipynb.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATHS = [
    ROOT / "runs" / "dfranzen_exact_submission" / "arc-agi-3-milestone-2-solution.ipynb",
    ROOT / "runs" / "blitz" / "arc-agi-3-v3-blitz-70.ipynb",
]


def package_arc3_inference() -> str:
    """Creates a compressed, base64-encoded tar.gz of ARC3-Inference."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(ROOT / "ARC3-Inference"):
            if "__pycache__" in root or ".git" in root:
                continue
            for f in files:
                if f.endswith(".pyc") or f.startswith("."):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, ROOT)
                tar.add(full_path, arcname=rel_path)

    gz_bytes = buf.getvalue()
    b64_str = base64.b64encode(gz_bytes).decode("ascii")
    print(f"[ARCHIVE] Packaged ARC3-Inference: {len(gz_bytes):,} bytes gzip ({len(b64_str):,} b64 chars).")
    return b64_str


def update_notebook(b64_bundle: str, notebook_path: Path) -> None:
    """Updates the notebook cells to deploy Archimedes-3."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 2: Write out the Archimedes-3 bundle
    cell_2_code = (
        '%%writefile /kaggle/archimedes3_bundle.tar.gz.b64\n'
        + b64_bundle
        + '\n'
    )
    nb["cells"][2]["source"] = [cell_2_code]
    print(f"[NOTEBOOK] Updated Cell 2 in {notebook_path.name} with self-contained Archimedes-3 bundle.")

    # Cell 4: Extract the bundle over BUNDLE_DIR/src/ARC3-Inference and adjust concurrency
    cell_4_src = "".join(nb["cells"][4]["source"])
    
    # Replace git apply with direct tar extraction
    old_patch_call = (
        'subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\n'
        "print('harness patch applied successfully')"
    )
    new_extract_call = (
        '# Deploy Archimedes-3 full architecture directly over BUNDLE_DIR/src\n'
        'import base64, io, tarfile\n'
        'with open("/kaggle/archimedes3_bundle.tar.gz.b64", "rb") as _bf:\n'
        '    _tar_bytes = base64.b64decode(_bf.read())\n'
        'with tarfile.open(fileobj=io.BytesIO(_tar_bytes), mode="r:gz") as _tar:\n'
        '    _tar.extractall(path=f"{BUNDLE_DIR}/src")\n'
        'print("Archimedes-3 codebase deployed successfully to", f"{BUNDLE_DIR}/src/ARC3-Inference")'
    )
    if old_patch_call in cell_4_src:
        cell_4_src = cell_4_src.replace(old_patch_call, new_extract_call)
    elif "archimedes3_bundle.tar.gz.b64" not in cell_4_src:
        import re
        cell_4_src = re.sub(
            r'subprocess\.run\(\["git",\s*"apply"[^)]+\)\s*\n\s*print\([^\)]+\)',
            new_extract_call,
            cell_4_src,
        )

    # Remove any old _UPGRADE_PAYLOAD in Cell 4
    if "_UPGRADE_PAYLOAD" in cell_4_src:
        import re
        cell_4_src = re.sub(r'# --- INJECT FRONTIER 70% UPGRADES INTO BUNDLE ---.*?(?=TRUE_SUBMISSION =)', '', cell_4_src, flags=re.DOTALL)
        print(f"[NOTEBOOK] Stripped legacy _UPGRADE_PAYLOAD from {notebook_path.name}.")

    # Concurrency Headroom Governor (§14.2): Cap ARC3_MAX_ACTIVE_STREAMS at 6 to prevent MTP wedging
    if "'ARC3_MAX_ACTIVE_STREAMS': 10" in cell_4_src:
        cell_4_src = cell_4_src.replace("'ARC3_MAX_ACTIVE_STREAMS': 10", "'ARC3_MAX_ACTIVE_STREAMS': 6")
        print(f"[NOTEBOOK] Concurrency Headroom Governor: Adjusted ARC3_MAX_ACTIVE_STREAMS to 6 in {notebook_path.name} (§14.2).")

    nb["cells"][4]["source"] = [cell_4_src]
    print(f"[NOTEBOOK] Updated Cell 4 in {notebook_path.name} with Archimedes-3 extractor.")

    # Cell 12: Ensure --tool-call-parser qwen3_coder is present for OpenAI-format tool calling
    cell_12_src = "".join(nb["cells"][12]["source"])
    if '"--tool-call-parser", "qwen3_coder"' not in cell_12_src:
        cell_12_src = cell_12_src.replace(
            '"--reasoning-parser", "qwen3",',
            '"--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",',
        )
        print(f"[NOTEBOOK] Restored --tool-call-parser qwen3_coder in Cell 12 for {notebook_path.name}.")
    nb["cells"][12]["source"] = [cell_12_src]

    # Cell 16: Fast-track offline dry-run while keeping 100% full time & full games on real competition rerun
    cell_16_src = "".join(nb["cells"][16]["source"])
    old_c16_block = (
        "else:\n"
        "    if USE_PRIORITY_SCHEDULING:\n"
        "        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"
        "        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110\n"
        "    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:\n"
    )
    new_c16_block = (
        "else:\n"
        "    # FAST-TRACK offline dry-run: 1 pass, 15 min limit to emit submission.parquet quickly\n"
        "    bm.n_passes = 1\n"
        "    if USE_PRIORITY_SCHEDULING:\n"
        "        bm.solver.concurrency = 2\n"
        "        bm.solver.max_runtime_s_per_game = 15*60\n"
        "    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:\n"
    )
    if old_c16_block in cell_16_src:
        cell_16_src = cell_16_src.replace(old_c16_block, new_c16_block)
        print(f"[NOTEBOOK] Fast-tracked Cell 16 offline dry-run in {notebook_path.name}.")
    nb["cells"][16]["source"] = [cell_16_src]

    # Cell 20: Fast-track offline dry-run game selection and soft_end
    cell_20_src = "".join(nb["cells"][20]["source"])
    old_c20_block = (
        "if TRUE_SUBMISSION:\n"
        "    # Real submission: play the live competition Arcade served by the Kaggle gateway.\n"
        "    os.environ.setdefault(\"ARC_API_KEY\", \"test-key-123\")\n"
        "    os.environ.setdefault(\"ARC_BASE_URL\", \"http://gateway:8001/\")\n"
        "    # The gateway boots asynchronously; wait before swapping in its game list.\n"
        "    _wait_for_gateway(os.environ[\"ARC_BASE_URL\"])\n"
        "    bm.games = _competition_games()\n"
        "else:\n"
        "    # Interactive run: play the bundled competition environments offline (no gateway).\n"
        "    # The competition's environment files ship alongside the wheelhouse in the competition dataset.\n"
        "    competition_env_files = str(Path(\"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels\").parent / \"environment_files\")\n"
        "    bm.games = _offline_games(competition_env_files)\n"
        "\n"
        "# Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.\n"
        "soft_end = None\n"
        "if not TRUE_SUBMISSION:\n"
        "    budget = float(getattr(target, \"max_runtime_s\", 0.0) or 0.0)\n"
        "    if budget > 0:\n"
        "        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))"
    )
    new_c20_block = (
        "if TRUE_SUBMISSION:\n"
        "    # Real submission: play ALL 110 live competition games served by the Kaggle gateway.\n"
        "    os.environ.setdefault(\"ARC_API_KEY\", \"test-key-123\")\n"
        "    os.environ.setdefault(\"ARC_BASE_URL\", \"http://gateway:8001/\")\n"
        "    # The gateway boots asynchronously; wait before swapping in its game list.\n"
        "    _wait_for_gateway(os.environ[\"ARC_BASE_URL\"])\n"
        "    bm.games = _competition_games()\n"
        "    soft_end = None  # Full 9-hour competition runtime budget\n"
        "else:\n"
        "    # Fast-track dry run: 1 offline validation game to verify full stack and emit submission.parquet\n"
        "    competition_env_files = str(Path(\"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels\").parent / \"environment_files\")\n"
        "    bm.games = _offline_games(competition_env_files)[:1]\n"
        "    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(minutes=18)"
    )
    if old_c20_block in cell_20_src:
        cell_20_src = cell_20_src.replace(old_c20_block, new_c20_block)
        print(f"[NOTEBOOK] Fast-tracked Cell 20 offline dry-run in {notebook_path.name}.")
    nb["cells"][20]["source"] = [cell_20_src]

    # Enforce GPU metadata for Kaggle competition runner
    if "metadata" in nb:
        if "kaggle" not in nb["metadata"]:
            nb["metadata"]["kaggle"] = {}
        nb["metadata"]["kaggle"]["isGpuEnabled"] = True
        nb["metadata"]["kaggle"]["accelerator"] = "nvidiaRtxPro6000"
        print(f"[NOTEBOOK] Enforced isGpuEnabled=True and accelerator=nvidiaRtxPro6000 in {notebook_path.name}.")

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[NOTEBOOK] Successfully wrote updated notebook: {notebook_path}")


def main() -> None:
    print("=" * 70)
    print(" 🚀 ARCHIMEDES-3 KAGGLE DEPLOYMENT BUNDLER")
    print("=" * 70)
    b64 = package_arc3_inference()
    for np in NOTEBOOK_PATHS:
        if np.exists():
            update_notebook(b64, np)
    print("=" * 70)
    print("✅ Build complete. Ready to push directly to Kaggle via kaggle CLI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
