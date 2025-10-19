#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR="${SCRIPT_DIR%/scripts}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 /absolute/path/to/video.(mov|MOV) [config_path]" >&2
  exit 1;
fi

VIDEO_PATH="$1"
CONFIG_PATH="${2:-${PROJECT_DIR}/config.yaml}"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "Video not found: $VIDEO_PATH" >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

source "${PROJECT_DIR}/.venv/bin/activate" || {
  echo "Activate venv first: source ${PROJECT_DIR}/.venv/bin/activate" >&2
  exit 1
}

cd "${PROJECT_DIR}"
python -m src.main --video "$VIDEO_PATH" --config "$CONFIG_PATH"


