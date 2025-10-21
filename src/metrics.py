import numpy as np
import pandas as pd
from typing import Optional


def cluster_radius_max(pts: np.ndarray) -> float:
    com = pts.mean(axis=0)
    return float(np.sqrt(((pts - com) ** 2).sum(axis=1)).max())


def time_to_coalescence_from_df(
    df: pd.DataFrame,
    radius_threshold_m: float,
    dwell_time_s: float,
    fps: float,
) -> Optional[float]:
    g = df.groupby("frame")[
        ["x", "y"]
    ]
    frames = sorted(g.groups.keys())
    R = []
    for fr in frames:
        pts = g.get_group(fr).values  # Nx2 (meters)
        R.append(cluster_radius_max(pts))
    R = np.array(R)
    dwell_frames = int(np.ceil(dwell_time_s * fps))
    for i in range(0, len(R) - dwell_frames):
        if np.all(R[i : i + dwell_frames] <= radius_threshold_m):
            return i / fps
    return None


