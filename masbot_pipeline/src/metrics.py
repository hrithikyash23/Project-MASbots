from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_com_and_radius(df: pd.DataFrame) -> pd.DataFrame:
    """Compute center-of-mass (COM) and max radius per frame.

    Returns a DataFrame with columns: frame_idx, t_sec, com_x_px, com_y_px, radius_px
    Only frames with at least one detection are included.
    """
    if df.empty:
        return pd.DataFrame(columns=["frame_idx", "t_sec", "com_x_px", "com_y_px", "radius_px"])

    groups = df.groupby("frame_idx", as_index=False)
    records = []
    for frame_idx, g in groups:
        if isinstance(frame_idx, tuple):  # pandas quirk depending on version
            frame_idx = frame_idx[1]
        gx = g["x_px"].to_numpy(dtype=float)
        gy = g["y_px"].to_numpy(dtype=float)
        if gx.size == 0:
            continue
        com_x = float(np.mean(gx))
        com_y = float(np.mean(gy))
        radius = float(np.max(np.sqrt((gx - com_x) ** 2 + (gy - com_y) ** 2)))
        t_sec = float(g["t_sec"].iloc[0]) if "t_sec" in g.columns else np.nan
        records.append({
            "frame_idx": int(frame_idx),
            "t_sec": t_sec,
            "com_x_px": com_x,
            "com_y_px": com_y,
            "radius_px": radius,
        })
    out = pd.DataFrame.from_records(records)
    out.sort_values("frame_idx", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def detect_coalescence(
    com_df: pd.DataFrame,
    fps: float,
    radius_threshold_px: float,
    dwell_threshold_sec: float,
) -> Dict:
    """Detect first coalescence and duration based on radius threshold and dwell time.

    Returns metrics dict with keys:
      - t_first_coalesce_sec
      - stayed_sec
      - radius_threshold_px
      - dwell_threshold_sec
    """
    result = {
        "t_first_coalesce_sec": np.nan,
        "stayed_sec": 0.0,
        "radius_threshold_px": float(radius_threshold_px),
        "dwell_threshold_sec": float(dwell_threshold_sec),
    }
    if com_df.empty:
        return result

    below = com_df["radius_px"] <= float(radius_threshold_px)
    if not below.any():
        return result

    dwell_frames = int(np.ceil(dwell_threshold_sec * float(fps))) if fps else 0
    if dwell_frames <= 0:
        dwell_frames = 1

    # Find first index where a run of 'below' of length >= dwell_frames starts
    idx = below.to_numpy().astype(int)
    n = len(idx)
    start_idx: Optional[int] = None
    run = 0
    for i in range(n):
        if idx[i] == 1:
            run += 1
            if run == dwell_frames:
                start_idx = i - dwell_frames + 1
                break
        else:
            run = 0

    if start_idx is None:
        return result

    # Determine how long it stayed below threshold from start_idx
    j = start_idx
    while j < n and idx[j] == 1:
        j += 1

    t_first = float(com_df["t_sec"].iloc[start_idx]) if "t_sec" in com_df.columns else (start_idx / float(fps) if fps else 0.0)
    stayed = (j - start_idx) / float(fps) if fps else 0.0

    result["t_first_coalesce_sec"] = t_first
    result["stayed_sec"] = float(stayed)
    return result


