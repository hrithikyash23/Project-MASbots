from pathlib import Path
import json, time, subprocess
from typing import Dict, Any


def current_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def save_calibrated_params(path: Path, params: Dict[str, Any], meta: Dict[str, Any]) -> None:
    payload = {
        "params": params,
        "meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "git": current_commit_hash(), **meta},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_calibrated_params(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


