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
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from src.economy.data import canonicalize_bounds
from src.economy.grids import generate_state_grids


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
    capital_grid_type: str = "multiplicative"

    def __post_init__(self) -> None:
        valid_grid_types = {"multiplicative", "linear"}
        if self.capital_grid_type not in valid_grid_types:
            raise ValueError(
                f"Unknown capital_grid_type: {self.capital_grid_type}. "
                f"Valid options: {sorted(valid_grid_types)}"
            )
        if self.z_size < 2:
            raise ValueError(f"z_size must be >= 2. Got {self.z_size}")
        if self.k_size < 2:
            raise ValueError(f"k_size must be >= 2. Got {self.k_size}")
        if self.b_size < 2:
            raise ValueError(f"b_size must be >= 2. Got {self.b_size}")

    def generate_grids(self, bounds: BoundsDict, *, delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build state grids from canonical bounds.

        Grid behavior:
        - Capital k:
            - capital_grid_type='multiplicative': geometric grid implied by depreciation (delta)
            - capital_grid_type='linear': fixed-size np.linspace(k_min, k_max, k_size)
        - Productivity z: always log-linear in levels via exp(linspace(log_z_min, log_z_max, z_size))
        - Debt b: always linear in levels via linspace(b_min, b_max, b_size), or [0.0] when degenerate
        """
        if "k" not in bounds:
            raise ValueError("bounds must include key 'k'.")
        if "log_z" not in bounds:
            raise ValueError("bounds must include key 'log_z'.")

        k_min, k_max = bounds["k"]
        log_z_min, log_z_max = bounds["log_z"]
        b_min, b_max = bounds.get("b", (0.0, 0.0))

        if self.capital_grid_type == "multiplicative":
            # Canonical DDP grid: multiplicative spacing tied to depreciation.
            k_grid, _, _ = generate_state_grids(
                {"k": (k_min, k_max), "log_z": (log_z_min, log_z_max), "b": (0.0, 0.0)},
                delta=delta,
                n_z=self.z_size,
                n_b=self.b_size,
            )
            k_grid = np.asarray(k_grid, dtype=np.float64)
        elif self.capital_grid_type == "linear":
            k_grid = np.asarray(np.linspace(k_min, k_max, self.k_size), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported capital_grid_type: {self.capital_grid_type}")

        z_grid = np.exp(np.linspace(log_z_min, log_z_max, self.z_size)).astype(np.float64)
        if b_max > b_min + 1e-6:
            b_grid = np.linspace(float(b_min), float(b_max), self.b_size).astype(np.float64)
        else:
            # Keep a numeric 1-point debt grid for basic model convenience.
            b_grid = np.array([0.0], dtype=np.float64)

        return np.asarray(k_grid), np.asarray(z_grid), np.asarray(b_grid)


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
        idx = np.round(scaled * (z_size - 1)).astype(np.int64)
        return np.clip(idx, 0, z_size - 1)

    i_idx = to_bin_index(z_curr)
    j_idx = to_bin_index(z_next_main)

    counts = np.zeros((z_size, z_size), dtype=np.float64)
    np.add.at(counts, (i_idx, j_idx), 1.0)

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
