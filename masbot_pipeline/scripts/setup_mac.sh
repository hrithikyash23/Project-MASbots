#!/bin/zsh
set -euo pipefail

# macOS setup: create venv and install requirements
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR="${SCRIPT_DIR%/scripts}"

PY=${PYTHON:-python3}
VENV_DIR="${PROJECT_DIR}/.venv"

echo "[setup] Creating virtual environment at ${VENV_DIR}"
${PY} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Ensure Homebrew and cmake exist if apriltag fallback is needed
if ! command -v brew >/dev/null 2>&1; then
  echo "[setup] Homebrew not found. If 'apriltag' build fails, install cmake manually."
else
  if ! command -v cmake >/dev/null 2>&1; then
    echo "[setup] Installing cmake via Homebrew (needed only for 'apriltag' build)"
    brew install cmake || true
  fi
fi

pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[setup] Done. Activate with: source ${VENV_DIR}/bin/activate"


