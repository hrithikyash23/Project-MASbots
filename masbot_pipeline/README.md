## Metrics and evaluation utilities

This repository exposes reusable NumPy-based metrics to evaluate swarm behavior and model outputs. They live in `src/metrics.py` and are re-exported at the package level for convenience.

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
com = center_of_mass(pts)             # -> array([0.666..., 0.666...])
rad = cluster_radius(pts, center=com) # -> max distance to COM

# RMSE examples
y_true = np.array([0.0, 1.0, 2.0])
y_pred = np.array([0.0, 1.5, 1.5])
e1 = rmse_series(y_pred, y_true)
e2 = rmse_traj(np.stack([y_pred, y_pred], axis=1), np.stack([y_true, y_true], axis=1))

# Aggregate objective across multiple signals
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


