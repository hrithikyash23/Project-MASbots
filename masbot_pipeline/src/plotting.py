from typing import Optional

import matplotlib

# Use non-interactive backend to avoid hanging windows during batch runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def plot_overview(df: pd.DataFrame, com_df: pd.DataFrame, out_path) -> None:
    """Create a 2D overview of trajectories and COM path in pixels.

    - Plots each bot_id trajectory as a line
    - Overlays COM path
    - Adds label 'Uncalibrated (pixels)'
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if not df.empty:
        for tag_id, g in df.groupby("tag_id"):
            ax.plot(g["x_px"], g["y_px"], lw=1.0, alpha=0.7, label=f"tag {tag_id}")

    if not com_df.empty:
        ax.plot(com_df["com_x_px"], com_df["com_y_px"], color="black", lw=2.0, label="COM")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("MASBots Trajectories — Uncalibrated (pixels)")
    # Limit legend size for many tags
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 10:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_com_only_gradient(com_df: pd.DataFrame, out_path) -> None:
    """Plot only the COM path using a red→blue gradient over time.

    - Start marked in red, end marked in blue
    - Uses pixel coordinates
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if not com_df.empty:
        x = com_df["com_x_px"].to_numpy()
        y = com_df["com_y_px"].to_numpy()
        if x.size >= 2:
            # Build line segments between consecutive points
            points = np.column_stack([x, y])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            # Normalize color along the path (0=start, 1=end)
            norm = Normalize(vmin=0.0, vmax=1.0)
            t = np.linspace(0.0, 1.0, len(points) - 1)
            lc = LineCollection(segments, cmap="RdBu", norm=norm)
            lc.set_array(t)
            lc.set_linewidth(2.0)
            ax.add_collection(lc)
            # Markers at start/end
            ax.scatter([x[0]], [y[0]], c=["red"], s=40, zorder=3)
            ax.scatter([x[-1]], [y[-1]], c=["blue"], s=40, zorder=3)
        else:
            ax.plot(x, y, color="red", lw=2.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Center of Mass — red→blue (start→end)")
    # Colorbar indicating time progression (start→end)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time (start → end)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tracks_only(df: pd.DataFrame, out_path) -> None:
    """Plot only the individual bot trajectories (no COM), with red→blue time gradient
    and start/end markers for each trajectory.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if not df.empty:
        # Build a global time normalization for consistent colors across tags
        if "t_sec" in df.columns and df["t_sec"].notna().any():
            t_all = df["t_sec"].to_numpy(dtype=float)
        else:
            t_all = df["frame_idx"].to_numpy(dtype=float)
        t_min = float(np.nanmin(t_all)) if t_all.size else 0.0
        t_max = float(np.nanmax(t_all)) if t_all.size else 1.0
        if t_max == t_min:
            t_max = t_min + 1.0
        norm = Normalize(vmin=t_min, vmax=t_max)

        for tag_id, g in df.groupby("tag_id"):
            g = g.sort_values("frame_idx")
            x = g["x_px"].to_numpy(dtype=float)
            y = g["y_px"].to_numpy(dtype=float)
            if "t_sec" in g.columns and g["t_sec"].notna().any():
                t = g["t_sec"].to_numpy(dtype=float)
            else:
                t = g["frame_idx"].to_numpy(dtype=float)

            if x.size >= 2:
                points = np.column_stack([x, y])
                segments = np.stack([points[:-1], points[1:]], axis=1)
                t_seg = 0.5 * (t[:-1] + t[1:])
                lc = LineCollection(segments, cmap="RdBu", norm=norm)
                lc.set_array(t_seg)
                lc.set_linewidth(1.8)
                lc.set_alpha(0.95)
                ax.add_collection(lc)
                # Start/end markers per tag
                ax.scatter([x[0]], [y[0]], c=["red"], s=25, zorder=3)
                ax.scatter([x[-1]], [y[-1]], c=["blue"], s=25, zorder=3)
            else:
                ax.plot(x, y, color="red", lw=1.8, alpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Bot Trajectories — red→blue (start→end)")
    # Colorbar shared across all trajectories
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time (start → end)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


