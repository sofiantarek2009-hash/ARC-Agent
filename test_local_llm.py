"""Interactive Test Utility for Local 7B LLM Setup.
Verifies your local Qwen 2.5 Coder 7B configuration (Ollama, LM Studio, or Transformers MPS).
"""

import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "project_override"))

from project_override.world_model.retrodict_synthesizer import (
    OfflineLLMRuntime,
    HistoricalTransition,
    RetrodictWorldModelSynthesizer,
    ASTWorldModelSandbox,
)

def main():
    print("=" * 70)
    print(" ARC-AGI-3 LOCAL 7B LLM RUNTIME DIAGNOSTIC")
    print("=" * 70)

    # 1. Environment & Hardware Detection
    try:
        import torch
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        print(f"[PYTORCH] Version: {torch.__version__}")
        print(f"[ACCELERATION] Apple Silicon GPU (MPS): {has_mps} | NVIDIA CUDA: {has_cuda}")
    except ImportError:
        print("[PYTORCH] PyTorch not installed.")

    # 2. Initialize Runtime
    runtime = OfflineLLMRuntime()
    print(f"\n[RUNTIME CONFIG]")
    print(f" Target Model Size:  {runtime.model_size}")
    print(f" Target Model Name:  {runtime.model_name}")
    print(f" Active Backend:     {runtime.backend_type}")
    if runtime.local_server_url:
        print(f" Connected Server:   {runtime.local_server_url}")

    # 3. Test Synthesis on Sample Interaction
    print(f"\n[TESTING WORLD-MODEL SYNTHESIS]")
    g0 = np.zeros((64, 64), dtype=np.int32)
    g0[10, 10] = 3  # Player at (10, 10)
    g1 = np.zeros((64, 64), dtype=np.int32)
    g1[10, 11] = 3  # Moved Right (action 4) to (10, 11)

    hist = [
        HistoricalTransition(
            step_idx=0,
            pre_grid=g0,
            pre_hud={},
            pre_latent=np.zeros((64, 64)),
            action_id=4,
            click_xy=None,
            post_grid=g1,
            post_hud={},
            timer_mask=np.zeros((64, 64), dtype=bool),
            level_cleared=False,
            game_over=False,
        )
    ]

    code = runtime.synthesize_world_model_code(hist)
    print(f"\n[SYNTHESIZED CODE PREVIEW]:\n{'-' * 40}")
    print(code[:350] + ("\n..." if len(code) > 350 else ""))
    print(f"{'-' * 40}")

    model = ASTWorldModelSandbox.validate_and_compile(code)
    if model is not None:
        print("[AST SANDBOX] Model compiled and instantiated successfully!")
        synth = RetrodictWorldModelSynthesizer(runtime)
        report = synth.verify_retrodiction(model, hist)
        print(f"[RETRODICTION VERIFICATION] Passed: {report.passed_all} | Jaccard: {report.mean_jaccard:.2f}")
    else:
        print("[AST SANDBOX] Failed to compile code.")

    print("\n" + "=" * 70)
    print(" QUICK START GUIDE FOR LOCAL 7B TESTING:")
    print(" 1. Ollama (Fastest, zero setup):")
    print("    In a new terminal, run: ollama run qwen2.5-coder:7b")
    print("    OfflineLLMRuntime will automatically detect http://localhost:11434.")
    print(" 2. LM Studio (GUI):")
    print("    Load Qwen 2.5 Coder 7B and start the local server on port 1234.")
    print(" 3. HuggingFace / PyTorch MPS (In-process on Mac GPU):")
    print("    Run: export ARC_LOAD_LOCAL_WEIGHTS=true")
    print("=" * 70)

if __name__ == "__main__":
    main()
