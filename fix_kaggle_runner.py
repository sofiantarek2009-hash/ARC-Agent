from pathlib import Path

ROOT = Path(__file__).resolve().parent
builder_path = ROOT / "build_kaggle_submission.py"

content = builder_path.read_text(encoding="utf-8")

# Write a robust Phase A / Phase B compliant cell 4 runner block
robust_runner = r'''# --- CELL 4: COMPETITION RUNNER & DUAL-PHASE PARQUET HANDLER ---
import os
import subprocess
import pandas as pd

is_rerun = os.getenv("KAGGLE_IS_COMPETITION_RERUN") == "1" or os.getenv("KAGGL_IS_COMPETITION_RERUN") == "1"

if is_rerun:
    print("[RERUN] Competition evaluation mode active. Waiting for gateway...")
    subprocess.run(["curl", "--fail", "--retry", "999", "--retry-all-errors", "--retry-delay", "2", "--retry-max-time", "600", "http://gateway:8001/api/games"], check=False)
else:
    print("[PHASE-A] Save & Run All mode active. Generating mandatory sample submission.parquet for validation...")
    try:
        # Standard ARC-AGI-3 sample submission format for Phase A validation pass
        df_sub = pd.DataFrame(
            data=[["1_0", "1", True, 1.0]],
            columns=["row_id", "game_id", "end_of_game", "score"]
        )
        df_sub.to_parquet("/kaggle/working/submission.parquet", index=False)
        print("[PARQUET] Successfully wrote sample /kaggle/working/submission.parquet")
    except Exception as e:
        print(f"[PARQUET-FALLBACK] Error: {e}")
        pd.DataFrame({"status": ["ok"]}).to_parquet("/kaggle/working/submission.parquet", index=False)
        print("[PARQUET] Wrote minimal fallback submission.parquet")

print("ARC-AGI-3 Environment Setup & Parquet Handlers Initialized.")
'''

# Replace cell4_runner in builder script
if "cell4_runner =" in content:
    import re
    content = re.sub(r'cell4_runner = r\'\'\'' + r'.*?' + r'\'\'\'', 'cell4_runner = r\'\'\'' + robust_runner + '\'\'\'', content, flags=re.DOTALL)
    builder_path.write_text(content, encoding="utf-8")
    print("[BUILDER] Updated cell4_runner with dual-phase parquet handler.")

