#!/usr/bin/env bash
# ==============================================================================
# ARC-AGI-3 One-Command Kaggle Submission Pipeline
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "======================================================================"
echo " 🚀 ARC-AGI-3 KAGGLE DIRECT SUBMISSION PIPELINE"
echo "======================================================================"

# 1. Select Python binary
if [ -x "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
    KAGGLE_BIN="./venv/bin/kaggle"
elif [ -x "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
    KAGGLE_BIN="./.venv/bin/kaggle"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    KAGGLE_BIN="kaggle"
else
    echo "❌ Error: Python 3 not found."
    exit 1
fi

# 2. Package the latest codebase into submission notebook and single-cell script
echo "📦 Packaging agent architecture and building Kaggle notebook..."
$PYTHON_BIN build_kaggle_submission.py

# 3. Locate Kaggle authentication credentials
AUTH_FOUND=0

# Priority A: Check for .kaggle/access_token
if [ -s ".kaggle/access_token" ]; then
    export KAGGLE_API_TOKEN="$(cat .kaggle/access_token | tr -d '\r\n')"
    AUTH_FOUND=1
    echo "🔑 Found API token in .kaggle/access_token"
elif [ -s "ARC-AGI-3-Kaggle-Starter/.kaggle/access_token" ]; then
    export KAGGLE_API_TOKEN="$(cat ARC-AGI-3-Kaggle-Starter/.kaggle/access_token | tr -d '\r\n')"
    AUTH_FOUND=1
    echo "🔑 Found API token in ARC-AGI-3-Kaggle-Starter/.kaggle/access_token"
elif [ -s "$HOME/.kaggle/access_token" ]; then
    export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token" | tr -d '\r\n')"
    AUTH_FOUND=1
    echo "🔑 Found API token in ~/.kaggle/access_token"
elif [ -n "$KAGGLE_API_TOKEN" ]; then
    AUTH_FOUND=1
    echo "🔑 Using KAGGLE_API_TOKEN from environment"
elif [ -s "$HOME/.kaggle/kaggle.json" ]; then
    AUTH_FOUND=1
    echo "🔑 Found credentials in ~/.kaggle/kaggle.json"
elif [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    AUTH_FOUND=1
    echo "🔑 Using KAGGLE_USERNAME and KAGGLE_KEY from environment"
fi

# Ensure kaggle CLI is installed
if ! command -v "$KAGGLE_BIN" >/dev/null 2>&1 && [ ! -x "$KAGGLE_BIN" ]; then
    echo "⚙️ Installing kaggle CLI..."
    $PYTHON_BIN -m pip install -q kaggle
fi

# 4. Handle authentication & push
KERNEL_DIR="ARC-AGI-3-Kaggle-Starter/notebooks"
KERNEL_ID="$($PYTHON_BIN -c "import json; print(json.load(open('$KERNEL_DIR/kernel-metadata.json'))['id'])")"

if [ "$AUTH_FOUND" -eq 1 ]; then
    echo ""
    echo "📡 Pushing kernel '$KERNEL_ID' directly to Kaggle..."
    $KAGGLE_BIN kernels push -p "$KERNEL_DIR"
    echo ""
    echo "✅ Kernel successfully pushed to Kaggle!"
    echo "🌐 Kernel URL: https://www.kaggle.com/code/$KERNEL_ID"
    echo ""
    echo "⏳ Checking initial kernel status..."
    $KAGGLE_BIN kernels status "$KERNEL_ID" || true
    echo ""
    echo "======================================================================"
    echo "🎯 NEXT STEP TO SUBMIT:"
    echo "1. Once status shows 'complete', open:"
    echo "   https://www.kaggle.com/code/$KERNEL_ID"
    echo "2. Click 'Submit to Competition' in the top right."
    echo "3. Select 'submission.parquet'."
    echo "======================================================================"
else
    echo ""
    echo "⚠️  No Kaggle credentials detected on your machine."
    echo "   To enable 1-click push, add your Kaggle API token with:"
    echo ""
    echo "   mkdir -p .kaggle"
    echo "   echo \"YOUR_KAGGLE_API_TOKEN\" > .kaggle/access_token"
    echo "   chmod 600 .kaggle/access_token"
    echo ""
    echo "   (Get your token from: https://www.kaggle.com/settings -> 'Create New Token')"
    echo ""
    echo "----------------------------------------------------------------------"
    echo "💡 INSTANT ZERO-SETUP ALTERNATIVE (COPY-PASTE):"
    echo "   The complete, self-contained single-cell agent code is ready at:"
    echo "   📁 $(pwd)/dist/kaggle_single_cell_submission.py"
    echo ""
    echo "   Just create a new Kaggle notebook (or open any existing notebook),"
    echo "   paste the contents of that file into Cell 1, and click 'Run All'!"
    echo "----------------------------------------------------------------------"
fi
