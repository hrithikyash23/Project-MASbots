import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .io_utils import (
    load_config,
    ensure_dirs,
    open_video_capture,
    make_output_paths,
    write_metadata_yaml,
    write_metrics_csv,
    write_state_csv,
)
from .detect_tags import iterate_detections_full
from .track_pipeline import TagTracker
from .metrics import compute_com_and_radius, detect_coalescence
from .plotting import plot_com_only_gradient, plot_tracks_only


def process_video(video_path: str, cfg_path: Optional[str] = None) -> None:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)

    cap, vid_meta = open_video_capture(video_path)
    fps = float(vid_meta["fps"])
    outputs = make_output_paths(video_path, cfg)

    # Detection and tracking: use full detections including corners for heading/omega
    detections_stream_full = iterate_detections_full(cap, fps)

    # Accumulate state for calibration-ready CSV
    ppm = None
    if "pixels_per_meter" in cfg and cfg["pixels_per_meter"]:
        ppm = float(cfg["pixels_per_meter"])
    tag_tracker = TagTracker(pixels_per_meter=ppm)

    for frame_idx, t_sec, dets in detections_stream_full:
        tag_tracker.update_frame(frame_idx, t_sec, dets)
    cap.release()

    # Build state dataframe and ensure time column is present
    state_df = tag_tracker.to_dataframe()
    if not state_df.empty:
        state_df["t_sec"] = state_df["frame_idx"].astype(float) / (fps if fps else 1.0)

    video_stem = Path(video_path).name

    # Write new calibration-ready state CSV
    write_state_csv(state_df, outputs["state_csv"])

    # Metadata YAML
    meta = {
        **vid_meta,
        "fps": float(fps),
        # timezone-aware UTC
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pixels_per_meter": None,
        "calibrated": False,
    }
    write_metadata_yaml(meta, outputs["meta_yaml"])

    # Metrics and plot computed from state_df (x_px,y_px per tag)
    com_df = compute_com_and_radius(state_df)
    metrics = detect_coalescence(
        com_df,
        fps=fps,
        radius_threshold_px=float(cfg["radius_threshold_px"]),
        dwell_threshold_sec=float(cfg["dwell_threshold_sec"]),
    )
    metrics_row = {"video": video_stem, **metrics}
    write_metrics_csv(metrics_row, outputs["metrics_csv"])

    # Plots (overview removed per request)
    plot_com_only_gradient(com_df, outputs["com_only_png"])
    plot_tracks_only(state_df, outputs["tracks_only_png"])


def main():
    parser = argparse.ArgumentParser(description="Process MASBot videos for AprilTag trajectories.")
    parser.add_argument("--video", type=str, required=True, help="Path to a video file or stem name in configured video_dir")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vid_arg = args.video
    # If user passed a stem, resolve within video_dir for .mov or .MOV
    vp = Path(vid_arg)
    if not vp.exists():
        candidate1 = Path(cfg["video_dir"]) / f"{vp.name}.mov"
        candidate2 = Path(cfg["video_dir"]) / f"{vp.name}.MOV"
        if candidate1.exists():
            vp = candidate1
        elif candidate2.exists():
            vp = candidate2
        else:
            raise FileNotFoundError(f"Video not found: {vid_arg}")

    process_video(str(vp), args.config)


if __name__ == "__main__":
    main()


