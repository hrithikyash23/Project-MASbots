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

# Load dirs from config via python (each var separately; preserves spaces)
VIDEO_DIR=$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os, yaml
cfg_path = os.environ['CONFIG_PATH']
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
def norm(p):
    p = os.path.expanduser(os.path.expandvars(p))
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(os.path.dirname(cfg_path), p))
    return p
print(norm(cfg['video_dir']))
PY
)

OUTPUT_DIR=$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os, yaml
cfg_path = os.environ['CONFIG_PATH']
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
def norm(p):
    p = os.path.expanduser(os.path.expandvars(p))
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(os.path.dirname(cfg_path), p))
    return p
print(norm(cfg['output_dir']))
PY
)

echo "[next5] Video dir: $VIDEO_DIR"
echo "[next5] Output dir: $OUTPUT_DIR"

processed=0
setopt NULL_GLOB
for f in "$VIDEO_DIR"/*.mov "$VIDEO_DIR"/*.MOV; do
  [ -e "$f" ] || continue
  stem_no_ext="${${f##*/}%.*}"
  out_csv="$OUTPUT_DIR/tracks_${stem_no_ext}.csv"
  if [ -f "$out_csv" ]; then
    continue
  fi
  echo "[next5] Processing: $f"
  python -m src.main --video "$f" --config "$CONFIG_PATH" || break
  processed=$((processed+1))
  if [ $processed -ge 5 ]; then
    break
  fi
done

echo "[next5] Done. Newly processed: $processed"


