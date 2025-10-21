from typing import Dict, Tuple
import numpy as np
from scipy.optimize import differential_evolution


def run_global_optimization(
    loss_fn,
    bounds: Dict[str, Tuple[float, float]],
    max_iter: int = 60,
    seed: int = 42,
) -> Dict[str, float]:
    keys = list(bounds.keys())
    rng = np.random.default_rng(seed)

    def _wrapped(x):
        params = {k: float(v) for k, v in zip(keys, x)}
        return loss_fn(params)

    result = differential_evolution(
        _wrapped,
        bounds=[bounds[k] for k in keys],
        maxiter=max_iter,
        polish=True,
        updating="deferred",
        seed=seed,
    )
    best = {k: float(v) for k, v in zip(keys, result.x)}
    best["loss"] = float(result.fun)
    return best


