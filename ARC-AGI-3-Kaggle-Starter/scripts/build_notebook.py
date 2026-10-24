#!/usr/bin/env python3
"""Builds Kaggle submission notebook from project root builder."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_kaggle_submission import build_all

if __name__ == "__main__":
    build_all()
