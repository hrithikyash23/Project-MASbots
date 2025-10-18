import os
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional

import cv2
import yaml
import pandas as pd


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load YAML config. Expands ~ and env vars for paths.

    Order of precedence:
    - Explicit path argument
    - MASBOT_CONFIG env var
    - Project-local ./config.yaml
    """
    if config_path is None:
        config_path = os.environ.get("MASBOT_CONFIG")
    if config_path is None:
        config_path = str(Path(__file__).resolve().parents[1] / "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Normalize paths
    for key in ["video_dir", "output_dir", "figures_dir"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expanduser(os.path.expandvars(cfg[key]))

    return cfg


def ensure_dirs(cfg: Dict) -> None:
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["figures_dir"]).mkdir(parents=True, exist_ok=True)


def enumerate_videos(video_dir: str) -> Iterator[Path]:
    """Yield .mov files (case-insensitive) from directory, sorted by name."""
    p = Path(video_dir)
    if not p.exists():
        return iter(())
    movs = sorted(list(p.glob("*.mov")) + list(p.glob("*.MOV")))
    for m in movs:
        yield m


def open_video_capture(video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # Fallback to 30.0 if FPS is 0 or NaN
    if not fps or fps != fps:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta = {
        "fps": float(fps),
        "width_px": int(width),
        "height_px": int(height),
        "n_frames": int(n_frames),
    }
    return cap, meta


def make_output_paths(video_path: str, cfg: Dict) -> Dict[str, Path]:
    stem = Path(video_path).stem
    outputs = {
        "tracks_csv": Path(cfg["output_dir"]) / f"{stem}_tracks.csv",
        "meta_yaml": Path(cfg["output_dir"]) / f"{stem}_meta.yaml",
        "metrics_csv": Path(cfg["output_dir"]) / f"{stem}_metrics.csv",
        "overview_png": Path(cfg["figures_dir"]) / f"{stem}_overview.png",
        # New calibration-ready state CSV: tracks_<video_stem>.csv
        "state_csv": Path(cfg["output_dir"]) / f"tracks_{stem}.csv",
    }
    return outputs


def write_tracks_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_metadata_yaml(meta: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def write_metrics_csv(metrics: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def write_state_csv(df: pd.DataFrame, path: Path) -> None:
    """Write calibration-ready per-frame state CSV.

    Expected columns:
      frame_idx, t_sec, tag_id, x_px, y_px, heading_deg, omega_deg_per_sec
    Optionally: x_m, y_m if pixels_per_meter configured
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


