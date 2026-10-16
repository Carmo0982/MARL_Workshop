#!/usr/bin/env bash
# macOS / Linux equivalent of SETUP_ONCE.bat. Run once:  bash setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11+ not found. Install it (e.g. 'brew install python@3.13') and retry."; exit 1
fi
echo "Using $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements_fixed.txt
echo
echo "Setup complete. Next:  bash run.sh"
