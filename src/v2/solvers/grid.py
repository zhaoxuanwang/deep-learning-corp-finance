"""
src/v2/solvers/grid.py

Grid construction and Markov transition estimation for discrete solvers.

Grid specification
------------------
Each variable (endo, exo, action) can have its own spacing rule via GridAxis.
Environments provide grid_spec() to communicate domain-specific grid structure
(e.g., log-spaced capital, log-spaced productivity).
The solver reads the spec and builds grids mechanically — no domain knowledge
leaks into the solver itself.

Transition estimation
---------------------
P(z'|z) is estimated from observed (z, z_next_main) pairs in the flattened
dataset. The estimator bins observations into the exogenous grid, counts
transitions, and applies Dirichlet smoothing. Migrated from
src/ddp/ddp_config.py with generalization to multi-dimensional exogenous state.

Multi-dimensional support
-------------------------
For problems with endo_dim > 1 or action_dim > 1, grids are formed as
Cartesian products of per-variable 1D grids. The flat index into the product
grid is used throughout the Bellman iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


# =============================================================================
# Grid axis specification
# =============================================================================

@dataclass(frozen=True)
class GridAxis:
    """Specification for discretizing a single continuous variable.

    Attributes:
        low:     Lower bound (in levels, not log-space).
        high:    Upper bound (in levels, not log-space).
        spacing: Grid point placement rule:
                 - "linear": np.linspace(low, high, n).
                   Uniform spacing. Use for actions or variables with
                   no strong prior on where density is needed.
                 - "log":    np.exp(np.linspace(log(low), log(high), n)).
                   Denser at low values, sparser at high values.
                   Use for capital (curvature highest at low k) and
                   productivity (log-AR(1) natural coordinate).
                   Grid values are always in levels, not log-space.
        power:   Power exponent for non-linear spacing modes such as
                 "zero_power". Larger values concentrate more points near
                 zero while keeping the user-specified grid size fixed.
    """
    low: float
    high: float
    spacing: str = "linear"
    power: float = 2.0

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(
                f"GridAxis requires low < high. Got low={self.low}, high={self.high}")
        valid = {"linear", "log", "zero_power"}
        if self.spacing not in valid:
            raise ValueError(
                f"Unknown spacing '{self.spacing}'. Valid: {sorted(valid)}")
        if self.spacing == "log" and self.low <= 0:
            raise ValueError(
                f"Log spacing requires low > 0. Got low={self.low}")
        if self.spacing == "zero_power" and not (self.low < 0.0 < self.high):
            raise ValueError(
                "zero_power spacing requires bounds that straddle zero. "
                f"Got low={self.low}, high={self.high}")
        if self.power <= 0:
            raise ValueError(f"GridAxis.power must be > 0. Got {self.power}")


# =============================================================================
# Grid construction
# =============================================================================

def build_1d_grid(axis: GridAxis, n: int) -> np.ndarray:
    """Build a 1-D grid from a GridAxis specification.

    Args:
        axis: GridAxis with bounds and spacing rule.
        n:    Number of grid points (always respected exactly).

    Returns:
        Sorted numpy array of shape (n,) in levels.
    """
    if n < 2:
        raise ValueError(f"Grid size must be >= 2. Got {n}")
    if axis.spacing == "zero_power" and n < 3:
        raise ValueError(
            f"zero_power spacing requires n >= 3 so 0 can be included. Got {n}")

    if axis.spacing == "linear":
        return np.linspace(axis.low, axis.high, n, dtype=np.float64)

    elif axis.spacing == "log":
        log_low  = np.log(axis.low)
        log_high = np.log(axis.high)
        return np.exp(np.linspace(log_low, log_high, n)).astype(np.float64)

    elif axis.spacing == "zero_power":
        n_remaining = n - 1
        neg_span = abs(axis.low)
        pos_span = axis.high

        n_neg = int(round(n_remaining * neg_span / (neg_span + pos_span)))
        n_neg = min(max(n_neg, 1), n_remaining - 1)
        n_pos = n_remaining - n_neg

        neg_u = np.linspace(1.0, 0.0, n_neg + 1, dtype=np.float64)[:-1]
        pos_u = np.linspace(0.0, 1.0, n_pos + 1, dtype=np.float64)[1:]
        neg = -neg_span * np.power(neg_u, axis.power)
        pos = pos_span * np.power(pos_u, axis.power)
        return np.concatenate([neg, np.array([0.0]), pos]).astype(np.float64)

    raise ValueError(f"Unsupported spacing: {axis.spacing}")


def build_product_grid(grids_1d: List[np.ndarray]) -> np.ndarray:
    """Cartesian product of 1-D grids.

    Args:
        grids_1d: List of K 1-D arrays, each shape (n_i,).

    Returns:
        Array of shape (n_1 * n_2 * ... * n_K, K) — every combination.
    """
    meshes = np.meshgrid(*grids_1d, indexing="ij")
    return np.column_stack([m.ravel() for m in meshes]).astype(np.float64)


def _default_grid_axes(low: np.ndarray, high: np.ndarray, dim: int) -> List[GridAxis]:
    """Create default linear GridAxis specs from bounds arrays."""
    axes = []
    for i in range(dim):
        lo = float(low[i]) if hasattr(low, '__getitem__') else float(low)
        hi = float(high[i]) if hasattr(high, '__getitem__') else float(high)
        axes.append(GridAxis(lo, hi, spacing="linear"))
    return axes


# =============================================================================
# Full grid construction from environment
# =============================================================================

def build_grids(
    env,
    exo_sizes: List[int],
    endo_sizes: List[int],
    action_sizes: List[int],
) -> Dict:
    """Build all grids for a discrete solver from an MDPEnvironment.

    Reads env.grid_spec() for per-variable spacing rules.
    Falls back to linspace when grid_spec() returns None or is not implemented.

    Args:
        env:          MDPEnvironment instance.
        exo_sizes:    Number of grid points per exogenous variable.
        endo_sizes:   Number of grid points per endogenous variable.
        action_sizes: Number of grid points per action variable.

    Returns:
        Dict with keys:
            exo_grids_1d:    list of 1-D arrays (one per exo variable)
            endo_grids_1d:   list of 1-D arrays (one per endo variable)
            action_grids_1d: list of 1-D arrays (one per action variable)
            exo_product:     (n_exo_flat, exo_dim)  Cartesian product
            endo_product:    (n_endo_flat, endo_dim) Cartesian product
            action_product:  (n_action_flat, action_dim) Cartesian product
    """
    spec = None
    if hasattr(env, 'grid_spec'):
        spec = env.grid_spec()

    # --- Exogenous ---
    if spec and "exo" in spec:
        exo_axes = spec["exo"]
    else:
        s_exo_sample = env.sample_initial_exogenous(
            1000, seed=tf.constant([999, 999], dtype=tf.int32))
        lo = s_exo_sample.numpy().min(axis=0)
        hi = s_exo_sample.numpy().max(axis=0)
        exo_axes = _default_grid_axes(lo, hi, env.exo_dim())

    if len(exo_axes) != env.exo_dim():
        raise ValueError(
            f"grid_spec exo has {len(exo_axes)} axes but env.exo_dim()={env.exo_dim()}")
    if len(exo_sizes) != env.exo_dim():
        raise ValueError(
            f"exo_sizes has {len(exo_sizes)} entries but env.exo_dim()={env.exo_dim()}")

    exo_grids_1d = [build_1d_grid(ax, n) for ax, n in zip(exo_axes, exo_sizes)]

    # --- Endogenous ---
    if spec and "endo" in spec:
        endo_axes = spec["endo"]
    else:
        s_endo_sample = env.sample_initial_endogenous(
            1000, seed=tf.constant([998, 998], dtype=tf.int32))
        lo = s_endo_sample.numpy().min(axis=0)
        hi = s_endo_sample.numpy().max(axis=0)
        endo_axes = _default_grid_axes(lo, hi, env.endo_dim())

    if len(endo_axes) != env.endo_dim():
        raise ValueError(
            f"grid_spec endo has {len(endo_axes)} axes but env.endo_dim()={env.endo_dim()}")
    if len(endo_sizes) != env.endo_dim():
        raise ValueError(
            f"endo_sizes has {len(endo_sizes)} entries but env.endo_dim()={env.endo_dim()}")

    endo_grids_1d = [build_1d_grid(ax, n) for ax, n in zip(endo_axes, endo_sizes)]

    # --- Actions ---
    if spec and "action" in spec:
        action_axes = spec["action"]
    else:
        low_a, high_a = env.action_bounds()
        action_axes = _default_grid_axes(
            low_a.numpy(), high_a.numpy(), env.action_dim())

    if len(action_axes) != env.action_dim():
        raise ValueError(
            f"grid_spec action has {len(action_axes)} axes but env.action_dim()={env.action_dim()}")
    if len(action_sizes) != env.action_dim():
        raise ValueError(
            f"action_sizes has {len(action_sizes)} entries but env.action_dim()={env.action_dim()}")

    action_grids_1d = [build_1d_grid(ax, n) for ax, n in zip(action_axes, action_sizes)]

    return {
        "exo_grids_1d":    exo_grids_1d,
        "endo_grids_1d":   endo_grids_1d,
        "action_grids_1d": action_grids_1d,
        "exo_product":     build_product_grid(exo_grids_1d),
        "endo_product":    build_product_grid(endo_grids_1d),
        "action_product":  build_product_grid(action_grids_1d),
    }


# =============================================================================
# Transition matrix estimation
# =============================================================================

def estimate_exo_transition_matrix(
    z_curr: np.ndarray,
    z_next: np.ndarray,
    exo_grids_1d: List[np.ndarray],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate Markov transition P(z'|z) from observed (z, z_next) pairs.

    Observations are binned into the exogenous product grid via nearest-neighbor.
    Transition counts are smoothed with a Dirichlet prior.

    Supports multi-dimensional exogenous state: each observation is assigned
    to its nearest product-grid point, and transitions are counted on the
    flat product-grid index.

    Args:
        z_curr:       Observed current exogenous values, shape (N, exo_dim).
        z_next:       Observed next exogenous values, shape (N, exo_dim).
        exo_grids_1d: List of 1-D grids (one per exo variable), from build_grids.
        alpha:        Dirichlet smoothing mass (default 1.0).

    Returns:
        Transition matrix, shape (n_exo_flat, n_exo_flat), rows sum to 1.
    """
    z_curr = np.asarray(z_curr, dtype=np.float64)
    z_next = np.asarray(z_next, dtype=np.float64)

    if z_curr.ndim == 1:
        z_curr = z_curr.reshape(-1, 1)
    if z_next.ndim == 1:
        z_next = z_next.reshape(-1, 1)

    exo_dim = len(exo_grids_1d)
    assert z_curr.shape[1] == exo_dim, (
        f"z_curr has {z_curr.shape[1]} columns but exo_dim={exo_dim}")

    # Per-variable bin assignment via nearest neighbor
    def assign_bins(z_vals: np.ndarray) -> np.ndarray:
        """Assign each observation to its nearest product-grid flat index."""
        per_var_idx = []
        for d in range(exo_dim):
            grid_d = exo_grids_1d[d]
            # Nearest-neighbor: find index of closest grid point
            idx = np.searchsorted(grid_d, z_vals[:, d], side='left')
            idx = np.clip(idx, 0, len(grid_d) - 1)
            # Check if left neighbor is closer
            left = np.clip(idx - 1, 0, len(grid_d) - 1)
            dist_right = np.abs(z_vals[:, d] - grid_d[idx])
            dist_left  = np.abs(z_vals[:, d] - grid_d[left])
            idx = np.where(dist_left < dist_right, left, idx)
            per_var_idx.append(idx)

        # Convert per-variable indices to flat product index
        sizes = [len(g) for g in exo_grids_1d]
        flat_idx = per_var_idx[0]
        for d in range(1, exo_dim):
            flat_idx = flat_idx * sizes[d] + per_var_idx[d]
        return flat_idx

    i_idx = assign_bins(z_curr)
    j_idx = assign_bins(z_next)

    n_flat = int(np.prod([len(g) for g in exo_grids_1d]))
    counts = np.zeros((n_flat, n_flat), dtype=np.float64)
    np.add.at(counts, (i_idx, j_idx), 1.0)

    # Dirichlet smoothing
    prior_vec = np.full((n_flat,), 1.0 / n_flat, dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = (counts + alpha * prior_vec[None, :]) / np.maximum(row_sums + alpha, 1e-12)

    # Safety normalization
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs


def snap_to_grid(values: tf.Tensor, grid_product: tf.Tensor) -> tf.Tensor:
    """Find the nearest product-grid point index for each row of values.

    Args:
        values:       (M, dim) tensor of continuous values.
        grid_product: (G, dim) tensor of grid points.

    Returns:
        Integer tensor of shape (M,) — flat indices into grid_product.
    """
    # Broadcast difference: (M, 1, dim) - (1, G, dim) -> (M, G, dim)
    diff = tf.expand_dims(values, 1) - tf.expand_dims(grid_product, 0)
    dist_sq = tf.reduce_sum(diff ** 2, axis=-1)  # (M, G)
    return tf.argmin(dist_sq, axis=1, output_type=tf.int32)
