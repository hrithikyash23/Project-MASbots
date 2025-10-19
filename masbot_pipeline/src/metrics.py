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



# -----------------------------
# NumPy evaluation utilities
# -----------------------------

def center_of_mass(
    points: np.ndarray,
    mask: Optional[np.ndarray] = None,
    axis: int = 0,
) -> np.ndarray:
    """Compute the center of mass (arithmetic mean) along a sample axis.

    Parameters
    ----------
    points
        Array of point coordinates. Expected shape is ``(N, D)`` by default
        where ``N`` indexes samples and ``D`` indexes spatial dimensions.
        The feature/spatial dimension is assumed to be the last axis.
    mask
        Optional boolean mask selecting which samples along ``axis`` to include.
        If provided with shape matching the reduced axis (e.g., ``(N,)``), it
        will be broadcast along feature dimensions. Masked-out samples are
        ignored in the mean. If all samples are masked out, returns ``NaN``s.
    axis
        Axis over which to compute the mean of the points (default: 0).

    Returns
    -------
    np.ndarray
        The center of mass with the same dtype as float(points). Shape equals
        ``points`` with ``axis`` removed (e.g., ``(D,)`` for the default case).
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        # Return an empty float array matching the reduced dimensionality
        # (consistent with NumPy reductions)
        return np.mean(pts, axis=axis)

    if mask is None:
        return np.mean(pts, axis=axis)

    m = np.asarray(mask, dtype=bool)
    # Align mask to sample axis and broadcast to points
    if m.ndim == 1 and pts.ndim > 1:
        # Insert singleton dims to match pts for broadcasting
        expand_axes = [slice(None)] * pts.ndim
        expand_axes[axis] = ...
        m = np.expand_dims(m, axis=tuple(i for i in range(pts.ndim) if i != axis))

    w = m.astype(float)
    weighted_sum = np.sum(pts * w, axis=axis)
    weight_total = np.sum(w, axis=axis)

    # Avoid divide-by-zero: where total weight is 0, return NaNs
    with np.errstate(invalid="ignore", divide="ignore"):
        com = weighted_sum / weight_total
    # If weight_total is 0 at any position, set result to NaN
    if np.isscalar(weight_total):
        if float(weight_total) == 0.0:
            com = np.full_like(weighted_sum, np.nan, dtype=float)
    else:
        com = np.where(weight_total == 0.0, np.nan, com)
    return com


def cluster_radius(
    points: np.ndarray,
    mask: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    sample_axis: int = 0,
) -> float:
    """Compute the maximum Euclidean distance from the center within a cluster.

    Parameters
    ----------
    points
        Array of point coordinates, shape ``(N, D)`` by default.
    mask
        Optional boolean mask selecting which samples to include.
    center
        Optional precomputed center-of-mass. If ``None``, it is computed with
        :func:`center_of_mass` using the provided ``mask`` and ``sample_axis``.
    sample_axis
        Axis indexing samples (default: 0). The spatial/features axis is
        assumed to be the last axis.

    Returns
    -------
    float
        The maximum distance of any included sample to the center. If no
        samples are included after masking, returns 0.0.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0

    ctr = center
    if ctr is None:
        ctr = center_of_mass(pts, mask=mask, axis=sample_axis)
    if ctr is None or (isinstance(ctr, np.ndarray) and np.any(np.isnan(ctr))):
        return 0.0

    # Broadcast center to points along sample axis
    expand_shape = [1] * pts.ndim
    expand_shape[sample_axis] = pts.shape[sample_axis]
    ctr_b = np.reshape(ctr, [s for i, s in enumerate(pts.shape) if i != sample_axis] or [])
    # In case ctr is 1D like (D,), reshape to broadcast over samples
    ctr_b = np.expand_dims(ctr_b, axis=sample_axis)
    diff = pts - ctr_b

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 1 and pts.ndim > 1:
            m = np.expand_dims(m, axis=tuple(i for i in range(pts.ndim) if i != sample_axis))
        diff = np.where(m, diff, 0.0)

    # Euclidean distances across feature axis (assumed last)
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    if mask is not None:
        # Exclude masked-out distances from max by setting them to 0.0
        m_simple = np.asarray(mask, dtype=bool)
        if m_simple.ndim == dists.ndim:
            dists = np.where(m_simple, dists, 0.0)
        elif m_simple.ndim == 1 and dists.ndim >= 1:
            dists = np.where(np.expand_dims(m_simple, axis=tuple(range(dists.ndim - 1))), dists, 0.0)

    return float(np.max(dists)) if dists.size else 0.0


def _apply_mask_and_normalizer(
    diff: np.ndarray,
    mask: Optional[np.ndarray],
    normalizer: Optional[np.ndarray | float],
    sample_axes: tuple[int, ...],
) -> np.ndarray:
    """Utility: apply mask and normalizer to a difference array.

    - Mask zeros out excluded samples along the given ``sample_axes``.
    - Normalizer divides the difference elementwise by a scalar or array.
    """
    out = np.asarray(diff, dtype=float)

    if normalizer is not None:
        norm = np.asarray(normalizer, dtype=float)
        out = out / norm

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        # Expand mask along non-sample axes for broadcasting
        if m.ndim < out.ndim:
            expand_dims = [slice(None)] * out.ndim
            for ax in range(out.ndim):
                if ax not in sample_axes:
                    m = np.expand_dims(m, axis=ax)
        out = np.where(m, out, 0.0)

    return out


def rmse_series(
    pred: np.ndarray,
    true: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalizer: Optional[np.ndarray | float] = None,
) -> float:
    """Root-mean-squared error between 1D series with optional mask/normalization.

    The RMSE is computed as ``sqrt(mean((pred - true)^2))`` over unmasked
    elements. If ``normalizer`` is provided, the differences are divided by
    it elementwise before squaring. If ``mask`` excludes all samples, returns
    ``0.0``.
    """
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.shape != t.shape:
        raise ValueError("pred and true must have the same shape for rmse_series")

    diff = p - t
    diff = _apply_mask_and_normalizer(diff, mask=mask, normalizer=normalizer, sample_axes=(0,))

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        denom = float(np.sum(m))
    else:
        denom = float(diff.size)
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum(diff**2) / denom))


def rmse_traj(
    pred: np.ndarray,
    true: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalizer: Optional[np.ndarray | float] = None,
) -> float:
    """Root-mean-squared error between trajectory arrays (e.g., ``(T, D)``).

    Parameters
    ----------
    pred, true
        Arrays of equal shape, typically ``(T, D)``.
    mask
        Optional boolean mask over the time/sample axis; shape ``(T,)`` or
        broadcastable to the shape of ``pred``/``true``.
    normalizer
        Optional scalar or array to divide the differences elementwise prior to
        squaring. Useful to normalize units or scales.
    """
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.shape != t.shape:
        raise ValueError("pred and true must have the same shape for rmse_traj")
    if p.ndim < 2:
        raise ValueError("rmse_traj expects at least 2D arrays, e.g., (T, D)")

    diff = p - t
    # Assume the first axis indexes time/samples; count all elements in denom
    diff = _apply_mask_and_normalizer(diff, mask=mask, normalizer=normalizer, sample_axes=(0,))

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        # Each time step contributes D elements
        elems_per_sample = int(np.prod(p.shape[1:]))
        denom = float(np.sum(m) * elems_per_sample)
    else:
        denom = float(diff.size)
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum(diff**2) / denom))


def objective_multi(
    pred: Dict[str, np.ndarray],
    true: Dict[str, np.ndarray],
    *,
    masks: Optional[Dict[str, np.ndarray]] = None,
    normalizers: Optional[Dict[str, np.ndarray | float] | float] = None,
    weights: Optional[Dict[str, float]] = None,
    reduce: str = "mean",
) -> float:
    """Aggregate objective over multiple named metrics using RMSE.

    For each key in ``true`` (and present in ``pred``), computes an RMSE:
    - 1D arrays use :func:`rmse_series`
    - 2D+ arrays use :func:`rmse_traj`

    Each metric may have an optional mask and normalizer. ``normalizers`` may be
    a single scalar applied to all metrics or a dict keyed by metric name.

    Parameters
    ----------
    pred, true
        Dicts mapping metric name to arrays of equal shapes.
    masks
        Optional dict of boolean masks per metric name.
    normalizers
        Optional scalar or dict of scalars/arrays for elementwise normalization
        of differences prior to squaring.
    weights
        Optional dict of weights per metric name. Defaults to 1.0.
    reduce
        Either ``"mean"`` (default) or ``"sum"`` to combine metric losses.

    Returns
    -------
    float
        Scalar objective value.
    """
    if reduce not in {"mean", "sum"}:
        raise ValueError("reduce must be 'mean' or 'sum'")

    keys = [k for k in true.keys() if k in pred]
    if not keys:
        return float("nan")

    loss_values = []
    for k in keys:
        y_p = pred[k]
        y_t = true[k]
        m_k = masks.get(k) if masks is not None else None
        if isinstance(normalizers, dict):
            n_k = normalizers.get(k)
        else:
            n_k = normalizers

        if np.asarray(y_t).ndim == 1:
            v = rmse_series(y_p, y_t, mask=m_k, normalizer=n_k)
        else:
            v = rmse_traj(y_p, y_t, mask=m_k, normalizer=n_k)

        w = 1.0 if weights is None else float(weights.get(k, 1.0))
        loss_values.append(w * v)

    if reduce == "sum":
        return float(np.sum(loss_values))
    return float(np.mean(loss_values))

