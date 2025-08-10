#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

# Launch the app
exec streamlit run "$PROJECT_DIR/script.py"
