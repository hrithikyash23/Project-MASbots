#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR="${SCRIPT_DIR%/scripts}"

source "${PROJECT_DIR}/.venv/bin/activate" || {
  echo "Activate venv first: source ${PROJECT_DIR}/.venv/bin/activate" >&2
  exit 1
}

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/*_tracks.csv [--save /path/to/out.png]" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
python -m src.visualize "$@"


