from typing import Optional

import matplotlib

# Use non-interactive backend to avoid hanging windows during batch runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


