#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR="${SCRIPT_DIR%/scripts}"

source "${PROJECT_DIR}/.venv/bin/activate" || {
  echo "Activate venv first: source ${PROJECT_DIR}/.venv/bin/activate" >&2
  exit 1
}

cd "${PROJECT_DIR}"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"

# Load video_dir from config.yaml with python (robust to spaces)
VIDEO_DIR=$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os, yaml
cfg_path = os.environ['CONFIG_PATH']
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
print(os.path.expanduser(os.path.expandvars(cfg['video_dir'])))
PY
)

echo "[run_all] Processing *.mov in: ${VIDEO_DIR}"

# In zsh, enable null globbing so unmatched patterns vanish
setopt NULL_GLOB

for f in "$VIDEO_DIR"/*.mov "$VIDEO_DIR"/*.MOV; do
  if [ -e "$f" ]; then
    echo "[run_all] Processing: $f"
    python -m src.main --video "$f" --config "$CONFIG_PATH"
  fi
done

echo "[run_all] Completed. Outputs in data/processed and reports/figures."


