from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class Detection:
    frame_idx: int
    t_sec: float
    tag_id: int
    x_px: float
    y_px: float


def build_tracks(
    detections_stream: Iterable[Tuple[int, float, List[Tuple[int, float, float]]]]
) -> pd.DataFrame:
    """Flatten detections into a long DataFrame: one row per detection per frame.

    Columns: frame_idx, t_sec, bot_id, tag_id, x_px, y_px
    """
    rows: List[Dict] = []
    for frame_idx, t_sec, dets in detections_stream:
        for tag_id, x_px, y_px in dets:
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "t_sec": float(t_sec),
                    "bot_id": int(tag_id),
                    "tag_id": int(tag_id),
                    "x_px": float(x_px),
                    "y_px": float(y_px),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["frame_idx", "t_sec", "bot_id", "tag_id", "x_px", "y_px"])

    df = pd.DataFrame(rows)
    df.sort_values(["frame_idx", "tag_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def interpolate_short_gaps(df: pd.DataFrame, gap_max_frames: int) -> pd.DataFrame:
    """Interpolate missing frames per tag_id up to gap_max_frames.

    Approach: for each tag, reindex to full frame range, then linear interpolate
    numeric columns and drop any remaining NaNs (gaps longer than allowed).
    """
    if df.empty:
        return df.copy()

    all_frames = np.arange(df["frame_idx"].min(), df["frame_idx"].max() + 1)
    out_parts: List[pd.DataFrame] = []
    for tag_id, g in df.groupby("tag_id"):
        g = g.set_index("frame_idx").reindex(all_frames)
        g["tag_id"] = tag_id
        g["bot_id"] = tag_id
        # forward fill t_sec if present, otherwise recompute later
        # we prefer to recompute t_sec later using fps if needed
        g["x_px"] = g["x_px"].astype(float)
        g["y_px"] = g["y_px"].astype(float)

        # Identify consecutive NaN spans
        is_nan = g["x_px"].isna() | g["y_px"].isna()
        if is_nan.any():
            # only interpolate spans up to gap_max_frames
            # We will perform interpolation then remove spans that exceed the limit
            g[["x_px", "y_px"]] = g[["x_px", "y_px"]].interpolate(
                method="linear", limit=gap_max_frames, limit_direction="both"
            )

            # Now remove points that remain NaN after limited interpolation
            g = g[~(g["x_px"].isna() | g["y_px"].isna())]
        out_parts.append(g.reset_index().rename(columns={"index": "frame_idx"}))

    out = pd.concat(out_parts, ignore_index=True) if out_parts else df.copy()
    out.sort_values(["frame_idx", "tag_id"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    # Recompute t_sec by group using min observed t_sec/frame slope if missing
    if "t_sec" in out.columns and out["t_sec"].isna().any():
        out.drop(columns=["t_sec"], inplace=True)
    return out


class TagTracker:
    """Accumulates per-frame tag states: position, heading, and angular velocity.

    Uses corner 0->1 vector to define heading in radians.
    Computes omega using wrapped angle difference and dt from frame times.
    """

    def __init__(self, pixels_per_meter: Optional[float] = None) -> None:
        self.tag_state: Dict[int, Dict[str, Any]] = {}
        self.records: List[Dict[str, Any]] = []
        self.pixels_per_meter: Optional[float] = float(pixels_per_meter) if pixels_per_meter else None

    @staticmethod
    def _compute_heading_deg(corners: Optional[List[Tuple[float, float]]]) -> Optional[float]:
        if corners is None or len(corners) < 2:
            return None
        (x0, y0), (x1, y1) = corners[0], corners[1]
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        heading_rad = float(np.arctan2(dy, dx))
        heading_deg = float(np.degrees(heading_rad))
        return heading_deg

    def update_frame(self, frame_idx: int, t_sec: float, dets_full: List[Dict[str, Any]]) -> None:
        for d in dets_full:
            tag_id = int(d["tag_id"]) if d.get("tag_id") is not None else -1
            center = d.get("center")
            corners = d.get("corners")
            if center is None:
                continue
            cx, cy = float(center[0]), float(center[1])
            heading_deg = self._compute_heading_deg(corners)

            # omega computation in deg/s with angle wrapping
            omega_deg_per_sec = 0.0
            prev = self.tag_state.get(tag_id, {})
            prev_heading_deg = prev.get("prev_heading_deg")
            prev_time = prev.get("prev_time")
            if prev_heading_deg is not None and prev_time is not None:
                dt = float(t_sec) - float(prev_time)
                if dt > 0:
                    # wrap difference into [-180, 180)
                    dtheta = (float(heading_deg) - float(prev_heading_deg) + 180.0) % 360.0 - 180.0 if heading_deg is not None else 0.0
                    omega_deg_per_sec = float(dtheta) / float(dt)

            # persist state
            self.tag_state[tag_id] = {
                "prev_heading_deg": heading_deg,
                "prev_time": float(t_sec),
            }

            rec: Dict[str, Any] = {
                "frame_idx": int(frame_idx),
                "t_sec": float(t_sec),
                "tag_id": tag_id,
                "x_px": float(cx),
                "y_px": float(cy),
                "heading_deg": float(heading_deg) if heading_deg is not None else np.nan,
                "omega_deg_per_sec": float(omega_deg_per_sec),
            }
            if self.pixels_per_meter:
                ppm = float(self.pixels_per_meter)
                rec["x_m"] = float(cx) / ppm
                rec["y_m"] = float(cy) / ppm
            self.records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            cols = [
                "frame_idx",
                "t_sec",
                "tag_id",
                "x_px",
                "y_px",
                "heading_deg",
                "omega_deg_per_sec",
            ]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self.records)
        # Column order as requested; include meters if present
        cols = ["frame_idx", "t_sec", "tag_id", "x_px", "y_px", "heading_deg", "omega_deg_per_sec"]
        if "x_m" in df.columns:
            cols += ["x_m", "y_m"]
        df = df[cols]
        df.sort_values(["frame_idx", "tag_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


