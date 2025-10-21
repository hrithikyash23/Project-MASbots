from typing import Dict, List
import numpy as np
import pandas as pd

from .metrics import time_to_coalescence_from_df
from .sim import simulate_swarm_time_to_coalescence


def huber(e: float, delta: float = 1.0) -> float:
    return 0.5 * e * e if abs(e) <= delta else delta * (abs(e) - 0.5 * delta)


def objective_time_error(
    params: Dict[str, float],
    runs: List[Dict],
    fps: float,
    dt: float,
    T: float,
    radius_threshold_m: float,
    dwell_time_s: float,
    initial_state_builder,
    input_builder,
) -> float:
    """
    For each run:
      - observed_t = coalescence time from CSV (meters already)
      - sim_t = simulate with same IC + inputs under 'params'
      - accumulate robust error (Huber)
    Return sum over runs.
    """
    total = 0.0
    for run in runs:
        df = run["df"]
        obs_t = time_to_coalescence_from_df(df, radius_threshold_m, dwell_time_s, fps)
        # If no observed coalescence, penalize miss (treat as T + margin)
        obs_t = obs_t if obs_t is not None else T + 2.0

        X0 = initial_state_builder(df)  # np.ndarray state vector
        U = input_builder(df, dt, T)  # np.ndarray inputs over time

        sim_t = simulate_swarm_time_to_coalescence(
            params=params,
            initial_state=X0,
            inputs=U,
            dt=dt,
            T=T,
            radius_threshold_m=radius_threshold_m,
            dwell_time_s=dwell_time_s,
        )
        sim_t = sim_t if sim_t is not None else T + 2.0
        total += huber(sim_t - obs_t, delta=1.0)
    return float(total)


