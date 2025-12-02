#!/usr/bin/env bash
# Reset and recreate the uv-managed virtual environment for this project.
# Usage:
#   chmod +x scripts/reset_env.sh
#   ./scripts/reset_env.sh
#
# What it does:
# - Verifies we're in the project root (pyproject.toml exists)
# - Removes a potentially corrupted .venv
# - Runs `uv sync` to recreate the environment according to pyproject.toml/uv.lock
# - Prints the python version from the new venv
#
# Notes:
# - This project requires Python >=3.12 (see pyproject.toml)
# - If you have multiple Pythons installed, you can pass one explicitly:
#     UV_PYTHON=/usr/bin/python3.12 ./scripts/reset_env.sh

set -euo pipefail

# cd to repo root (script may be invoked from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
  echo "[reset_env] Error: pyproject.toml not found. Run this script from within the repo."
  exit 1
fi

# Detect uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[reset_env] Error: 'uv' is not installed. See https://docs.astral.sh/uv/ for installation instructions."
  exit 2
fi

# Remove possibly broken venv
if [[ -d .venv ]]; then
  echo "[reset_env] Removing existing .venv (might be corrupted) ..."
  rm -rf .venv
fi

# Select python
PY_CMD="${UV_PYTHON:-}"
if [[ -n "${PY_CMD}" ]]; then
  if ! "$PY_CMD" -c 'import sys; assert sys.version_info >= (3,12)'; then
    echo "[reset_env] Provided UV_PYTHON ($PY_CMD) is not Python >=3.12"
    exit 3
  fi
  echo "[reset_env] Using explicit Python: $PY_CMD"
  uv venv --python "$PY_CMD"
  # ensure uv uses local venv
  source .venv/bin/activate
  uv sync
else
  echo "[reset_env] Creating venv and syncing deps using default Python (must be >=3.12) ..."
  uv sync
fi

# Basic verification
if [[ ! -x .venv/bin/python3 ]]; then
  echo "[reset_env] Error: .venv/bin/python3 missing after sync."
  exit 4
fi

echo "[reset_env] Interpreter info:"
.venv/bin/python3 -c 'import sys; print(sys.executable); print(sys.version)'

# Quick uv check – query interpreter via uv (similar to what failed previously)
echo "[reset_env] uv run -- python -c 'print(42)'"
uv run -- python -c 'print(42)'

echo "[reset_env] Done. Activate with: source .venv/bin/activate"
