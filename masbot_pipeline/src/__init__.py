"""MASBot pipeline package.

This file marks the directory as a Python package so that
`python -m src.main` works reliably when run from the project root.
"""


from .metrics import (
    compute_com_and_radius,
    detect_coalescence,
    center_of_mass,
    cluster_radius,
    rmse_series,
    rmse_traj,
    objective_multi,
)

__all__ = [
    "compute_com_and_radius",
    "detect_coalescence",
    "center_of_mass",
    "cluster_radius",
    "rmse_series",
    "rmse_traj",
    "objective_multi",
]

