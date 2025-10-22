from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_tracks_with_final_circles(
    tracks_csv: Path,
    canvas_height_px: int = 1080,
    canvas_width_px: int = 1920,
    bot_diameter_px: int = 150,
    out_png: Path | None = None,
    ellipse_center: tuple[int, int] | None = None,
    ellipse_width_px: int | None = None,
    ellipse_height_px: int | None = None,
):
    df = pd.read_csv(tracks_csv)
    # Expect columns: frame_idx, t_sec, tag_id, x_px, y_px
    req = {"frame_idx", "t_sec", "tag_id", "x_px", "y_px"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prepare figure (pixel-perfect size)
    dpi = 100
    fig_w = canvas_width_px / dpi
    fig_h = canvas_height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, canvas_width_px)
    ax.set_ylim(canvas_height_px, 0)  # invert y-axis to match pixel coords
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tracks_csv.stem)

    # Plot trajectories per tag_id
    for tag_id, g in df.sort_values(["frame_idx"]).groupby("tag_id"):
        ax.plot(g["x_px"].values, g["y_px"].values, lw=2, label=f"{tag_id}")

    # Draw final circles
    radius = bot_diameter_px / 2.0
    final_points = (
        df.sort_values(["frame_idx"]).groupby("tag_id").tail(1)[["tag_id", "x_px", "y_px"]]
    )
    for _, row in final_points.iterrows():
        circ = plt.Circle((row["x_px"], row["y_px"]), radius=radius, color="C0", alpha=0.25)
        ax.add_patch(circ)
        ax.scatter([row["x_px"]], [row["y_px"]], color="C0", s=20)

    # Optional ellipse overlay
    if (
        ellipse_center is not None
        and ellipse_width_px is not None
        and ellipse_height_px is not None
    ):
        ex, ey = ellipse_center
        ell = Ellipse(
            (ex, ey),
            width=ellipse_width_px,
            height=ellipse_height_px,
            edgecolor="red",
            facecolor="none",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.add_patch(ell)

    ax.legend(loc="upper right", ncols=2, fontsize=8, frameon=False)

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_csv", type=str)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--diameter", type=int, default=150)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--ellipse-center-x", type=int, default=None)
    parser.add_argument("--ellipse-center-y", type=int, default=None)
    parser.add_argument("--ellipse-width", type=int, default=None)
    parser.add_argument("--ellipse-height", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    ecenter = None
    if (
        args.ellipse_center_x is not None
        and args.ellipse_center_y is not None
        and args.ellipse_width is not None
        and args.ellipse_height is not None
    ):
        ecenter = (args.ellipse_center_x, args.ellipse_center_y)
    plot_tracks_with_final_circles(
        tracks_csv=Path(args.tracks_csv),
        canvas_height_px=args.height,
        canvas_width_px=args.width,
        bot_diameter_px=args.diameter,
        out_png=out,
        ellipse_center=ecenter,
        ellipse_width_px=args.ellipse_width,
        ellipse_height_px=args.ellipse_height,
    )

