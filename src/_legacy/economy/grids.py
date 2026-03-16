"""
src/economy/grids.py

Utilities for generating discretized state grids.
Used for:
1. Evaluation plotting (DNN)
2. Dynamic Programming (DDP) value iteration
"""

import numpy as np
from typing import Tuple, Optional, Dict


def generate_state_grids(
    bounds: Dict[str, Tuple[float, float]],
    *,
    delta: float,
    n_z: int = 10,
    n_b: int = 50
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate discretized state grids based on sampling bounds and economic parameters.

    The capital grid uses a multiplicative scheme dictated by the depreciation rate:
    k_{i} = (1 - delta) * k_{i+1}
    This ensures that inaction (I=0) lands exactly on grid points.
    The number of capital grid points is determined automatically by (k_min, k_max, delta).

    Args:
        bounds: Dictionary containing {'k', 'log_z'} intervals as tuples.
            Optional: 'b' for debt bounds (defaults to (0, 0) if absent).
            Legacy key 'logz' is also accepted for 'log_z'.
        delta: Depreciation rate used by the multiplicative capital grid.
        n_z: Number of productivity points
        n_b: Number of debt points (if b_bounds explicitly defined)

    Returns:
        (k_grid, z_grid, b_grid)
        - k_grid: array in levels (size determined by delta)
        - z_grid: (n_z,) array in levels
        - b_grid: (n_b,) array or None if b_bounds is (0,0) or similar
    """
    if "k" not in bounds:
        raise ValueError("bounds must include key 'k'.")

    logz_key = "log_z" if "log_z" in bounds else "logz"
    if logz_key not in bounds:
        raise ValueError("bounds must include key 'log_z' (or legacy 'logz').")
    if "b" not in bounds:
        bounds = {**bounds, "b": (0.0, 0.0)}

    k_min, k_max = bounds["k"]
    log_z_min, log_z_max = bounds[logz_key]
    b_min, b_max = bounds["b"]

    # --- Capital Grid (Multiplicative) ---
    # k_grid[i] = (1-δ) * k_grid[i+1]
    # step ratio r = 1 / (1-δ)
    g = 1.0 - float(delta)
    if g <= 0 or g >= 1:
        # Fallback for invalid delta: linear grid
        k_grid = np.linspace(k_min, k_max, 50)
    else:
        r = 1.0 / g

        # Auto-compute grid size to span [k_min, k_max]
        # k_max = k_min * r^(n-1)
        n_k = int(np.ceil(1 + np.log(k_max / k_min) / np.log(r)))
        n_k = max(n_k, 2)

        exponents = np.arange(n_k)
        k_grid = k_min * (r ** exponents)

        # Clip to exact max and ensure sorted
        k_grid = np.clip(k_grid, k_min, k_max)

        # Fix potential duplicates from clipping or precision
        for i in range(1, len(k_grid)):
            if k_grid[i] <= k_grid[i-1]:
                k_grid[i] = k_grid[i-1] + 1e-8

    # --- Productivity Grid (Log-Linear) ---
    z_grid = np.exp(np.linspace(log_z_min, log_z_max, n_z))

    # --- Debt Grid (Linear) ---
    if b_max > b_min + 1e-6:
        b_grid = np.linspace(b_min, b_max, n_b)
    else:
        b_grid = None

    return k_grid, z_grid, b_grid
