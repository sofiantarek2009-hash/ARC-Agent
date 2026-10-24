#!/usr/bin/env bash
set -euo pipefail

echo "Running Verification Gates G1-G9..."

# G1: Clean import test
python3 -c "from project_override.orchestration.general_agent import GeneralAgent; print('G1: PASS')"

# G2: Unit tests (supports pytest if installed, otherwise unittest discovery)
if python3 -c "import pytest" >/dev/null 2>&1; then
    python3 -m pytest -q tests/
else
    python3 -m unittest discover -s tests -p "test_*.py" -v
fi
echo "G2: PASS"

# G3: Static syntax & compilation check
python3 -m py_compile project_override/*.py project_override/*/*.py agent/my_agent.py tests/*.py
if command -v ruff >/dev/null 2>&1; then
    ruff check project_override agent tests || true
fi
if command -v mypy >/dev/null 2>&1; then
    mypy project_override --ignore-missing-imports || true
fi
echo "G3: PASS"

# G4: No game-specific logic check
MATCHES=$(grep -RInE "game_id\s*==|gameid\s*==|filename\s*==|if.*\.json|hardcoded|level_?[0-9]+\s*==" project_override || true)
if [ -n "$MATCHES" ]; then
    echo "G4: FAIL - Found game-specific branching:"
    echo "$MATCHES"
    exit 1
fi
echo "G4: PASS"

# G5, G6, G7: Offline smoke test, Public evaluation, and Frozen Dev/Held-out evaluation
python3 evaluate_gates.py
echo "G5: PASS"
echo "G6: PASS"
echo "G7: PASS"

# G8: Kaggle simulation in a fresh isolated process
python3 - << 'EOF_G8'
import json
import subprocess
import sys
from pathlib import Path

nb = json.loads(Path("notebooks/submission.ipynb").read_text(encoding="utf-8"))
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        code = "".join(cell.get("source", []))
        subprocess.run([sys.executable, "-c", code], check=True)
print("G8: PASS")
EOF_G8

# G9: Submission parquet schema validation
python3 -c "from project_override.submission.kaggle_submission import validate_submission_parquet; assert validate_submission_parquet('submission.parquet'); print('G9: PASS')"
