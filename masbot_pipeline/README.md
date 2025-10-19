## MASBots Data Processing Pipeline

### Overview

This pipeline processes MASBots experiment videos to detect AprilTags, build per-bot trajectories, compute basic group metrics (center of mass and cluster radius), optionally detect coalescence events, and generate figures/CSVs for analysis.

Key modules in `src/`:
- `detect_tags.py`: AprilTag detection wrapper (prefers `pupil-apriltags`).
- `track_pipeline.py`: Accumulates per-frame tag states and trajectories.
- `metrics.py`: Computes per-frame COM/radius and coalescence detection.
- `plotting.py`: Generates overview and per-metric plots.
- `main.py`: CLI entry for end-to-end processing.

Artifacts are written under `data/processed/` and `reports/figures/`.

### Setup

```bash
pip install -r requirements.txt
# optional (for local testing)
pip install pytest
```

### Usage

- CLI:
```bash
python -m src.main --video <path-or-stem> --config config.yaml
```

- Convenience scripts (see `scripts/`):
  - `scripts/run_one.sh "<path-or-stem>"`
  - `scripts/run_next5.sh`
  - `scripts/run_all.sh`
  - `scripts/visualize.sh <state_csv>`

### Configuration

Edit `config.yaml` for defaults such as:
- `video_dir`: base directory to resolve stems
- `pixels_per_meter`: if available for metric unit conversion
- `radius_threshold_px`, `dwell_threshold_sec`: parameters for coalescence detection

### Outputs

For each video:
- State CSV: per-frame tag positions (and derived headings/omega)
- Metrics CSV: coalescence timing and dwell metrics
- Figures: COM heatmap (`*_com_only.png`) and trajectories (`*_tracks_only.png`)

### Development and tests

```bash
pytest -q
```

## Metrics and evaluation utilities

Reusable NumPy-based metrics are provided for downstream experiments in `src/metrics.py` and are re-exported at the package level.

### Available functions

- `center_of_mass(points, mask=None, axis=0)`
- `cluster_radius(points, mask=None, center=None, sample_axis=0)`
- `rmse_series(pred, true, mask=None, normalizer=None)`
- `rmse_traj(pred, true, mask=None, normalizer=None)`
- `objective_multi(pred_dict, true_dict, masks=None, normalizers=None, weights=None, reduce='mean')`

All functions operate on NumPy arrays and support optional masking. `rmse_*` functions optionally divide errors by a scalar/array `normalizer` before squaring to normalize scales.

### Example usage

```python
import numpy as np
from src import center_of_mass, cluster_radius, rmse_series, rmse_traj, objective_multi

pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
com = center_of_mass(pts)
rad = cluster_radius(pts, center=com)

y_true = np.array([0.0, 1.0, 2.0])
y_pred = np.array([0.0, 1.5, 1.5])
e1 = rmse_series(y_pred, y_true)
e2 = rmse_traj(np.stack([y_pred, y_pred], axis=1), np.stack([y_true, y_true], axis=1))

obj = objective_multi(
    pred={"a": y_pred, "b": np.stack([y_pred, y_pred], axis=1)},
    true={"a": y_true, "b": np.stack([y_true, y_true], axis=1)},
    masks={"a": np.array([True, True, False])},
    normalizers={"a": 2.0},
    weights={"a": 1.0, "b": 0.5},
    reduce="mean",
)
```

### Dependencies

These utilities require NumPy (already listed in `requirements.txt`).


