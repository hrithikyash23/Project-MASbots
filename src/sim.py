from typing import Dict, Any, Optional
import numpy as np


class SimNotWiredError(RuntimeError):
    pass


def simulate_swarm_time_to_coalescence(
    params: Dict[str, float],
    initial_state: np.ndarray,
    inputs: np.ndarray,
    dt: float,
    T: float,
    radius_threshold_m: float,
    dwell_time_s: float,
) -> Optional[float]:
    """
    Return the simulated time-to-coalescence (seconds) under 'params'.
    Must call the existing simulator (RK4 etc.) and compute cluster radius R(t),
    then detect first time R(t) <= threshold for at least dwell_time_s.
    Return None if never coalesces in horizon T.
    """
    # TODO: Wire this to your existing simulation code (import or refactor).
    # Raise for now so the wiring step is explicit.
    raise SimNotWiredError("Wire to existing simulation notebook/code.")


