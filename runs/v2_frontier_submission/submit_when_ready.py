import subprocess
import time
import sys

KERNEL = "sofianali188201/arc-agi-3-v2-frontier-70"
COMPETITION = "arc-prize-2026-arc-agi-3"

sys.stdout.reconfigure(line_buffering=True)
print(f"Monitoring {KERNEL} for completion...", flush=True)

while True:
    res = subprocess.run(["venv/bin/kaggle", "kernels", "status", KERNEL], capture_output=True, text=True)
    status_line = res.stdout.strip()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status_line}", flush=True)
    
    if "KernelWorkerStatus.COMPLETE" in status_line or "has status \"complete\"" in status_line.lower():
        print("\n>>> Kernel completed execution! Submitting to competition...", flush=True)
        submit_cmd = [
            "venv/bin/kaggle", "competitions", "submit",
            "-k", KERNEL,
            "-f", "submission.parquet",
            "-m", "Frontier V2: 70% Tier (In-REPL BFS + Retrodiction + Sokoban + Hungarian + AVO Loop Supervisor)",
            COMPETITION
        ]
        submit_res = subprocess.run(submit_cmd, capture_output=True, text=True)
        print("Submit STDOUT:\n", submit_res.stdout, flush=True)
        print("Submit STDERR:\n", submit_res.stderr, flush=True)
        break
        
    if "KernelWorkerStatus.ERROR" in status_line or "has status \"error\"" in status_line.lower() or "KernelWorkerStatus.CANCELLED" in status_line:
        print("\n>>> Kernel ended in error or cancelled. Check logs.", flush=True)
        break
        
    time.sleep(30)
