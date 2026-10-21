from pathlib import Path

path = Path("build_kaggle_submission.py")
content = path.read_text()

# Inject or update the dual-phase check in cell4_runner
dual_phase_code = '''
import os, pandas as pd
is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').lower() == 'true'
if not is_rerun:
    print("[PHASE-A] Fast commit mode: writing dummy parquet.")
    os.makedirs("/kaggle/working", exist_ok=True)
    df = pd.DataFrame({"row_id": ["1_0"], "game_id": ["1"], "end_of_game": [True], "score": [1.0]})
    df.to_parquet("/kaggle/working/submission.parquet", index=False)
    df.to_parquet("submission.parquet", index=False)
else:
    print("[PHASE-B] Real rerun mode: running agent.")
    # --- Agent execution loop goes here ---
'''

print("Ready to integrate dual-phase runner.")
