import numpy as np

from src.metrics import (
    center_of_mass,
    cluster_radius,
    rmse_series,
    rmse_traj,
    objective_multi,
)


def test_center_of_mass_and_cluster_radius_simple():
    # Three points in 2D: (0,0), (2,0), (0,2)
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    com = center_of_mass(pts)
    # Mean of x: (0 + 2 + 0) / 3 = 2/3; Mean of y: (0 + 0 + 2) / 3 = 2/3
    np.testing.assert_allclose(com, np.array([2.0 / 3.0, 2.0 / 3.0]), rtol=0, atol=1e-8)

    rad = cluster_radius(pts, center=com)
    # Distances to COM: sqrt((0-2/3)^2 + (0-2/3)^2) etc.; max should be same for all corners
    # Just ensure it's consistent with direct computation
    dists = np.sqrt(np.sum((pts - com) ** 2, axis=1))
    np.testing.assert_allclose(rad, np.max(dists), rtol=0, atol=1e-8)


def test_rmse_series_with_and_without_mask():
    true = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.0, 1.0, 1.0, 5.0])
    # Differences: [0,0,-1,2] -> squared: [0,0,1,4] mean=1.25 -> sqrt
    expected = np.sqrt((0 + 0 + 1 + 4) / 4.0)
    np.testing.assert_allclose(rmse_series(pred, true), expected, rtol=0, atol=1e-12)

    mask = np.array([True, True, False, True])
    # Consider indices 0,1,3 only: diffs [0,0,2] -> squared [0,0,4]; mean over 3 -> 4/3
    expected_masked = np.sqrt(4.0 / 3.0)
    np.testing.assert_allclose(rmse_series(pred, true, mask=mask), expected_masked, rtol=0, atol=1e-12)

    # Normalizer: divide diffs by 2
    expected_norm = np.sqrt(((0/2)**2 + (0/2)**2 + (-1/2)**2 + (2/2)**2) / 4.0)
    np.testing.assert_allclose(rmse_series(pred, true, normalizer=2.0), expected_norm, rtol=0, atol=1e-12)


def test_rmse_traj_and_mask():
    # Two timesteps, 2D
    true = np.array([[0.0, 0.0], [1.0, 1.0]])
    pred = np.array([[0.0, 1.0], [2.0, 1.0]])
    # diffs: [[0,1],[1,0]]; squares: [[0,1],[1,0]]; mean over 4 -> 0.5; sqrt -> sqrt(0.5)
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(rmse_traj(pred, true), expected, rtol=0, atol=1e-12)

    mask = np.array([True, False])
    # Only first timestep counts: diffs [[0,1]] -> squares [0,1]; mean over 2 -> 0.5
    expected_masked = np.sqrt(0.5)
    np.testing.assert_allclose(rmse_traj(pred, true, mask=mask), expected_masked, rtol=0, atol=1e-12)


def test_objective_multi_with_and_without_normalization():
    true = {
        "x": np.array([0.0, 1.0, 2.0]),
        "traj": np.array([[0.0, 0.0], [1.0, 1.0]]),
    }
    pred = {
        "x": np.array([0.0, 2.0, 2.0]),
        "traj": np.array([[1.0, 0.0], [1.0, 2.0]]),
    }
    masks = {
        "x": np.array([True, True, False]),
        "traj": np.array([True, True]),
    }

    # Without normalization
    rmse_x = rmse_series(pred["x"], true["x"], mask=masks["x"])  # over 2 elements
    rmse_tr = rmse_traj(pred["traj"], true["traj"], mask=masks["traj"])  # both steps
    expected_mean = (rmse_x + rmse_tr) / 2.0
    np.testing.assert_allclose(
        objective_multi(pred, true, masks=masks, reduce="mean"), expected_mean, rtol=0, atol=1e-12
    )

    # With normalization: scale x errors by 2.0, traj by scalar 0.5
    norms = {"x": 2.0, "traj": 0.5}
    rmse_x_n = rmse_series(pred["x"], true["x"], mask=masks["x"], normalizer=2.0)
    rmse_tr_n = rmse_traj(pred["traj"], true["traj"], mask=masks["traj"], normalizer=0.5)
    expected_sum = rmse_x_n + rmse_tr_n
    np.testing.assert_allclose(
        objective_multi(pred, true, masks=masks, normalizers=norms, reduce="sum"), expected_sum, rtol=0, atol=1e-12
    )


