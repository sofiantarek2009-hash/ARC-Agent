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
NOTEBOOK_PATH = ROOT / "runs" / "dfranzen_exact_submission" / "arc-agi-3-milestone-2-solution.ipynb"


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


def update_notebook(b64_bundle: str) -> None:
    """Updates the notebook cells to deploy Archimedes-3."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 2: Write out the Archimedes-3 bundle
    cell_2_code = (
        '%%writefile /kaggle/archimedes3_bundle.tar.gz.b64\n'
        + b64_bundle
        + '\n'
    )
    nb["cells"][2]["source"] = [cell_2_code]
    print("[NOTEBOOK] Updated Cell 2 with self-contained Archimedes-3 codebase bundle.")

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
        # Fallback replacement if whitespace differs
        import re
        cell_4_src = re.sub(
            r'subprocess\.run\(\["git",\s*"apply"[^)]+\)\s*\n\s*print\([^\)]+\)',
            new_extract_call,
            cell_4_src,
        )

    # Concurrency Headroom Governor (§14.2): Cap ARC3_MAX_ACTIVE_STREAMS at 6 to prevent MTP wedging
    if "'ARC3_MAX_ACTIVE_STREAMS': 10" in cell_4_src:
        cell_4_src = cell_4_src.replace("'ARC3_MAX_ACTIVE_STREAMS': 10", "'ARC3_MAX_ACTIVE_STREAMS': 6")
        print("[NOTEBOOK] Concurrency Headroom Governor: Adjusted ARC3_MAX_ACTIVE_STREAMS to 6 (§14.2).")

    nb["cells"][4]["source"] = [cell_4_src]
    print("[NOTEBOOK] Updated Cell 4 with Archimedes-3 extractor.")

    # Cell 12: Remove --tool-call-parser qwen3_coder (Mitigating SGLang Issue #36537 token-0 loop)
    cell_12_src = "".join(nb["cells"][12]["source"])
    old_parser_arg = '"--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",'
    new_parser_arg = '"--reasoning-parser", "qwen3",'
    if old_parser_arg in cell_12_src:
        cell_12_src = cell_12_src.replace(old_parser_arg, new_parser_arg)
        print("[NOTEBOOK] Mitigated SGLang Issue #36537 in Cell 12: Removed --tool-call-parser qwen3_coder.")
    nb["cells"][12]["source"] = [cell_12_src]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[NOTEBOOK] Successfully wrote updated notebook: {NOTEBOOK_PATH}")


def main() -> None:
    print("=" * 70)
    print(" 🚀 ARCHIMEDES-3 KAGGLE DEPLOYMENT BUNDLER")
    print("=" * 70)
    b64 = package_arc3_inference()
    update_notebook(b64)
    print("=" * 70)
    print("✅ Build complete. Ready to push directly to Kaggle via kaggle CLI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
