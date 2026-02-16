"""
Offline DDP configuration and transition estimation.

The DDP stack in this module is dataset-driven:
- bounds are extracted from dataset metadata
- grids are generated from canonical grid utilities
- Markov transitions are estimated from observed (z_t, z_{t+1}) pairs

No Tauchen-based shock simulation is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple
import warnings

import numpy as np
import tensorflow as tf

from src.economy.data import canonicalize_bounds
from src.economy.grids import generate_state_grids
from src.economy.parameters import EconomicParams, ShockParams


BoundsDict = Dict[str, Tuple[float, float]]


@dataclass
class DDPGridConfig:
    """
    Numerical grid settings for offline DDP.

    Attributes:
        z_size: number of productivity states.
        k_size: target number of capital points.
        b_size: number of debt points.
    """

    z_size: int = 15
    k_size: int = 100
    b_size: int = 80
    # Backward-compatible fields for legacy constructor usage.
    k_min_multiplier: float = 0.2
    k_max_multiplier: float = 3.0
    grid_type: str = "multiplicative"

    def __post_init__(self) -> None:
        valid_grid_types = {"multiplicative", "delta_rule", "log_linear", "power_grid"}
        if self.grid_type not in valid_grid_types:
            raise ValueError(
                f"Unknown grid_type: {self.grid_type}. Valid options: {sorted(valid_grid_types)}"
            )
        if self.z_size < 2:
            raise ValueError(f"z_size must be >= 2. Got {self.z_size}")
        if self.k_size < 2:
            raise ValueError(f"k_size must be >= 2. Got {self.k_size}")
        if self.b_size < 2:
            raise ValueError(f"b_size must be >= 2. Got {self.b_size}")
        if self.k_min_multiplier <= 0.0:
            raise ValueError(f"k_min_multiplier must be > 0. Got {self.k_min_multiplier}")
        if self.k_max_multiplier <= self.k_min_multiplier:
            raise ValueError(
                f"k_max_multiplier must be > k_min_multiplier. "
                f"Got {self.k_max_multiplier} <= {self.k_min_multiplier}"
            )

    def generate_grids(self, bounds: BoundsDict, *, delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build state grids from canonical bounds.
        """
        if self.grid_type not in {"multiplicative", "delta_rule"}:
            warnings.warn(
                f"Offline grid generation enforces multiplicative capital grids; "
                f"grid_type='{self.grid_type}' is ignored in generate_grids().",
                UserWarning,
                stacklevel=2,
            )
        k_grid, z_grid, b_grid = generate_state_grids(
            bounds,
            delta=delta,
            n_z=self.z_size,
            n_b=self.b_size,
        )
        if b_grid is None:
            # Keep a numeric 1-point debt grid for basic model convenience.
            b_grid = np.array([0.0], dtype=np.float64)
        return np.asarray(k_grid), np.asarray(z_grid), np.asarray(b_grid)

    # ------------------------------------------------------------------
    # Backward-compatible helpers (legacy API)
    # ------------------------------------------------------------------
    def generate_capital_grid(self, source: object, delta: float | None = None) -> np.ndarray:
        """
        Legacy-compatible capital grid interface.

        Preferred new interface: generate_grids(bounds, delta=...).
        """
        if isinstance(source, dict):
            bounds = source
            delta_val = float(delta if delta is not None else 0.15)
        elif hasattr(source, "steady_state_k"):
            params = source  # EconomicParams
            k_ss = float(params.steady_state_k())
            bounds = {
                "k": (self.k_min_multiplier * k_ss, self.k_max_multiplier * k_ss),
                "log_z": (-0.5, 0.5),
                "b": (0.0, 1.0),
            }
            delta_val = float(getattr(params, "delta", 0.15) if delta is None else delta)
        else:
            raise TypeError("generate_capital_grid expects bounds dict or EconomicParams-like object.")

        k_min, k_max = bounds["k"]
        if self.grid_type == "delta_rule":
            density = 4.0
            multiplier = (1.0 / max(1.0 - delta_val, 1e-8)) ** (1.0 / density)
            n_steps = np.log(k_max / k_min) / np.log(multiplier)
            n_points = int(np.ceil(n_steps)) + 1
            return np.asarray(k_min * (multiplier ** np.arange(n_points)))
        if self.grid_type == "multiplicative":
            k_grid, _, _ = generate_state_grids(
                bounds,
                delta=delta_val,
                n_z=self.z_size,
                n_b=self.b_size,
            )
            return np.asarray(k_grid)
        if self.grid_type == "power_grid":
            gamma = 2.0
            linear_steps = np.linspace(0.0, 1.0, self.k_size)
            return np.asarray(k_min + (k_max - k_min) * (linear_steps ** gamma))
        if self.grid_type == "log_linear":
            return np.asarray(np.linspace(k_min, k_max, self.k_size))
        raise ValueError(f"Unsupported grid_type: {self.grid_type}")

    def generate_bond_grid(self, source: object, *, k_max: float | None = None, z_min: float | None = None) -> np.ndarray:
        """
        Legacy-compatible debt grid interface.
        """
        if isinstance(source, dict):
            b_min, b_max = source["b"]
            return np.linspace(float(b_min), float(b_max), self.b_size)

        if k_max is None or z_min is None:
            raise ValueError("Legacy generate_bond_grid requires (params, k_max, z_min).")

        params = source  # EconomicParams
        y_worst = float(z_min) * (float(k_max) ** float(params.theta))
        collateral_value = float(params.frac_liquid) * float(k_max)
        tax_shield = float(params.tax) * float(params.delta) * float(k_max)
        after_tax_profit = (1.0 - float(params.tax)) * y_worst
        b_max_val = after_tax_profit + tax_shield + collateral_value
        return np.linspace(0.0, b_max_val, self.b_size)


def extract_bounds_from_metadata(metadata: Dict[str, object]) -> BoundsDict:
    """
    Extract canonical bounds {'k', 'log_z', 'b'} from dataset metadata.
    """
    return canonicalize_bounds(metadata)


def estimate_transition_matrix_from_flat_data(
    z_curr: np.ndarray,
    z_next_main: np.ndarray,
    log_z_bounds: Tuple[float, float],
    *,
    z_size: int,
    alpha: float = 1.0,
    prior: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate P(z'|z) from observed transitions in flat dataset.

    Estimator:
        C_ij = transition counts from bin i to bin j
        P_ij = (C_ij + alpha * prior_j) / (sum_j C_ij + alpha)

    Args:
        z_curr: observed current productivity values (levels).
        z_next_main: observed next productivity values (levels, main shock only).
        log_z_bounds: canonical log-z bounds from metadata.
        z_size: number of productivity bins.
        alpha: Dirichlet smoothing mass.
        prior: currently supports "uniform".
    """
    if z_size < 2:
        raise ValueError(f"z_size must be >= 2. Got {z_size}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0. Got {alpha}")
    if prior != "uniform":
        raise ValueError(f"Unsupported prior '{prior}'. Only 'uniform' is supported.")

    z_curr = np.asarray(z_curr, dtype=np.float64).reshape(-1)
    z_next_main = np.asarray(z_next_main, dtype=np.float64).reshape(-1)
    if z_curr.shape != z_next_main.shape:
        raise ValueError(
            f"z and z_next_main must have same shape. Got {z_curr.shape} vs {z_next_main.shape}"
        )

    log_z_min, log_z_max = log_z_bounds
    log_grid = np.linspace(log_z_min, log_z_max, z_size, dtype=np.float64)
    z_grid = np.exp(log_grid)

    def to_bin_index(z_vals: np.ndarray) -> np.ndarray:
        log_vals = np.log(np.maximum(z_vals, 1e-12))
        clipped = np.clip(log_vals, log_z_min, log_z_max)
        # Map into [0, z_size-1]
        scaled = (clipped - log_z_min) / max(log_z_max - log_z_min, 1e-12)
        idx = np.floor(scaled * (z_size - 1) + 1e-12).astype(np.int64)
        return np.clip(idx, 0, z_size - 1)

    i_idx = to_bin_index(z_curr)
    j_idx = to_bin_index(z_next_main)

    counts = np.zeros((z_size, z_size), dtype=np.float64)
    np.add.at(counts, (i_idx, j_idx), 1.0)

    if prior == "uniform":
        prior_vec = np.full((z_size,), 1.0 / z_size, dtype=np.float64)
    else:
        prior_vec = np.full((z_size,), 1.0 / z_size, dtype=np.float64)

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = (counts + alpha * prior_vec[None, :]) / np.maximum(row_sums + alpha, 1e-12)

    # Safety normalization.
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return z_grid, probs


def estimate_transition_matrix_from_dataset(
    dataset: Dict[str, tf.Tensor],
    bounds: BoundsDict,
    *,
    z_size: int,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to estimate transition matrix from tensor dataset.
    """
    required = {"z", "z_next_main"}
    missing = required - set(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset missing required keys for transition estimation: {sorted(missing)}"
        )
    return estimate_transition_matrix_from_flat_data(
        z_curr=np.asarray(dataset["z"].numpy()),
        z_next_main=np.asarray(dataset["z_next_main"].numpy()),
        log_z_bounds=bounds["log_z"],
        z_size=z_size,
        alpha=alpha,
    )


# ----------------------------------------------------------------------
# Legacy compatibility wrappers (deprecated)
# ----------------------------------------------------------------------
def initialize_markov_process(
    shock_params: ShockParams,
    z_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deprecated Tauchen wrapper kept for backward compatibility.

    Offline workflows should call estimate_transition_matrix_from_dataset(...)
    using observed transitions from a fixed dataset.
    """
    warnings.warn(
        "initialize_markov_process is deprecated for offline workflows. "
        "Use estimate_transition_matrix_from_dataset instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rho, sigma, mu = shock_params.rho, shock_params.sigma, shock_params.mu
    if z_size < 2:
        raise ValueError(f"z_size must be >= 2. Got {z_size}")
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1). Got {rho}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0. Got {sigma}")

    n_std = 3.0
    sigma_x = sigma / np.sqrt(1.0 - rho**2)
    x_max = mu + n_std * sigma_x
    x_min = mu - n_std * sigma_x
    x_grid = np.linspace(x_min, x_max, z_size)
    step = x_grid[1] - x_grid[0]

    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(x_arr / np.sqrt(2.0)))

    prob_matrix = np.zeros((z_size, z_size), dtype=np.float64)
    for i in range(z_size):
        mean_i = (1.0 - rho) * mu + rho * x_grid[i]
        for j in range(z_size):
            if j == 0:
                z_upper = (x_grid[j] - mean_i + step / 2.0) / sigma
                prob_matrix[i, j] = _norm_cdf(z_upper)
            elif j == z_size - 1:
                z_lower = (x_grid[j] - mean_i - step / 2.0) / sigma
                prob_matrix[i, j] = 1.0 - _norm_cdf(z_lower)
            else:
                z_upper = (x_grid[j] - mean_i + step / 2.0) / sigma
                z_lower = (x_grid[j] - mean_i - step / 2.0) / sigma
                prob_matrix[i, j] = _norm_cdf(z_upper) - _norm_cdf(z_lower)

    z_grid = np.exp(x_grid).astype(np.float64)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(prob_matrix, row_sums, out=np.zeros_like(prob_matrix), where=row_sums > 0)
    return z_grid, prob_matrix


def get_sampling_bounds(
    params: EconomicParams,
    shock_params: ShockParams,
    grid_config: DDPGridConfig | None = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Deprecated legacy helper kept for backward compatibility.
    """
    warnings.warn(
        "get_sampling_bounds is deprecated for offline workflows.",
        DeprecationWarning,
        stacklevel=2,
    )
    grid_cfg = grid_config or DDPGridConfig()
    k_grid = grid_cfg.generate_capital_grid(params)
    k_min, k_max = float(k_grid[0]), float(k_grid[-1])
    z_grid, _ = initialize_markov_process(shock_params, grid_cfg.z_size)
    z_min = float(z_grid[0])
    b_grid = grid_cfg.generate_bond_grid(params, k_max=k_max, z_min=z_min)
    return (k_min, k_max), (float(b_grid[0]), float(b_grid[-1]))
