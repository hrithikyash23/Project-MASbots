import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import compute_com_and_radius
from .plotting import plot_com_only_gradient, plot_tracks_only


def _default_png_path(csv_path: str) -> Path:
    p = Path(csv_path)
    # Normalize both legacy and new naming to <stem>_quick.png
    stem = p.stem
    if stem.startswith("tracks_"):
        base = stem[len("tracks_"):]
    else:
        base = stem.replace("_tracks", "")
    return p.with_name(f"{base}_quick.png")


def visualize_tracks(csv_path: str, save_path: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    com_df = compute_com_and_radius(df)

    fig, ax = plt.subplots(figsize=(6, 6))
    if not df.empty:
        for tag_id, g in df.groupby("tag_id"):
            ax.plot(g["x_px"], g["y_px"], lw=1.0, alpha=0.7, label=f"tag {tag_id}")
    if not com_df.empty:
        ax.plot(com_df["com_x_px"], com_df["com_y_px"], color="black", lw=2.0, label="COM")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Tracks overview — Uncalibrated (pixels)")
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 12:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = Path(save_path) if save_path else _default_png_path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_extras(csv_path: str, com_only_path: str | None = None, tracks_only_path: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    com_df = compute_com_and_radius(df)

    if com_only_path:
        plot_com_only_gradient(com_df, com_only_path)
    if tracks_only_path:
        plot_tracks_only(df, tracks_only_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize tracks CSV; optionally save COM-only and tracks-only plots")
    parser.add_argument("csv", type=str, help="Path to *_tracks.csv")
    parser.add_argument("--save", type=str, default=None, help="Overview PNG path (default: <csv>_quick.png)")
    parser.add_argument("--com-only", dest="com_only", type=str, default=None, help="Optional path to save COM-only gradient plot")
    parser.add_argument("--tracks-only", dest="tracks_only", type=str, default=None, help="Optional path to save tracks-only plot")
    args = parser.parse_args()
    visualize_tracks(args.csv, args.save)
    if args.com_only or args.tracks_only:
        visualize_extras(args.csv, args.com_only, args.tracks_only)


if __name__ == "__main__":
    main()


