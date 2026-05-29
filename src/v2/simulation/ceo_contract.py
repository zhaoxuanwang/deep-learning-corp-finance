"""Monte-Carlo SDE simulation of the CEO short-termism contract model.

Simulates a cross-section of CEO contracts forward over ``[0, T]`` by
Euler-Maruyama discretisation of the state law of motion

    dz = [(r+kappa) z + a r gamma (sigma sigma_z - 1)] dt + sigma_z dB,

using the solved policy functions ``a(z, t)``, ``sigma_z(z, t)``, ``m(z, t)``
(interpolated on the solver's grid).  The same Brownian increment drives the
observable cash flow

    dX = (a + m - theta M) dt + sigma dB,

and the manipulation stock accumulates as ``M_{t+dt} = e^{-kappa dt} M_t + m_t dt``.
Initial duration ``z_0`` is drawn from a cross-sectional distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.v2.environments.ceo_contract import CEOContractParams, drift_z


@dataclass(frozen=True)
class CEOContractSimulationConfig:
    """Runtime settings for a CEO-contract panel simulation."""

    n_paths: int = 2000
    seed: int = 20260528
    z0_low: float | None = None   # default: 25th pct of the z grid
    z0_high: float | None = None  # default: 75th pct of the z grid

    def __post_init__(self) -> None:
        if self.n_paths < 1:
            raise ValueError(f"n_paths must be >= 1. Got {self.n_paths}")
        if (self.z0_low is None) != (self.z0_high is None):
            raise ValueError("z0_low and z0_high must be set together or both None.")
        if self.z0_low is not None and not (self.z0_low < self.z0_high):
            raise ValueError(
                f"z0_low must be < z0_high. Got ({self.z0_low}, {self.z0_high})"
            )


def simulate_ceo_contract_panel(
    result: dict[str, Any],
    params: CEOContractParams,
    config: CEOContractSimulationConfig | None = None,
) -> pd.DataFrame:
    """Simulate a CEO panel from a solved contract via Euler-Maruyama.

    Returns a long DataFrame with one row per (path, time) and columns
    ``path_id, z0, t, z, a, m, sigma_z, M, X``.  Deterministic given the seed.
    """

    config = config or CEOContractSimulationConfig()
    grids = result["grids"]
    policy = result["policy"]
    rng = np.random.default_rng(config.seed)

    z_grid = np.asarray(grids["z"], dtype=np.float64)
    t_grid = np.asarray(grids["t"], dtype=np.float64)
    dt = float(grids["dt"])
    z_min, z_max = float(z_grid[0]), float(z_grid[-1])
    n_t = t_grid.size
    n_paths = config.n_paths

    pol_a = np.asarray(policy["a"], dtype=np.float64)
    pol_m = np.asarray(policy["m"], dtype=np.float64)
    pol_s = np.asarray(policy["sigma_z"], dtype=np.float64)

    z0_low = config.z0_low if config.z0_low is not None else float(
        np.quantile(z_grid, 0.25)
    )
    z0_high = config.z0_high if config.z0_high is not None else float(
        np.quantile(z_grid, 0.75)
    )
    z0 = rng.uniform(z0_low, z0_high, size=n_paths)

    z_hist = np.empty((n_t, n_paths), dtype=np.float64)
    a_hist = np.empty((n_t, n_paths), dtype=np.float64)
    m_hist = np.empty((n_t, n_paths), dtype=np.float64)
    s_hist = np.empty((n_t, n_paths), dtype=np.float64)
    M_hist = np.empty((n_t, n_paths), dtype=np.float64)
    X_hist = np.empty((n_t, n_paths), dtype=np.float64)

    z = np.clip(z0.copy(), z_min, z_max)
    M = np.zeros(n_paths, dtype=np.float64)
    X = np.zeros(n_paths, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for n in range(n_t):
        a = np.interp(z, z_grid, pol_a[n])
        m = np.interp(z, z_grid, pol_m[n])
        sigma_z = np.interp(z, z_grid, pol_s[n])

        z_hist[n] = z
        a_hist[n], m_hist[n], s_hist[n] = a, m, sigma_z
        M_hist[n], X_hist[n] = M, X

        if n == n_t - 1:
            break

        xi = rng.standard_normal(n_paths)
        mu_z = drift_z(z, a, sigma_z, params)
        z = np.clip(z + mu_z * dt + sigma_z * sqrt_dt * xi, z_min, z_max)
        M = np.exp(-params.kappa * dt) * M + m * dt
        X = X + (a + m - params.theta * M_hist[n]) * dt + params.sigma * sqrt_dt * xi

    path_ids = np.repeat(np.arange(n_paths, dtype=np.int64), n_t)
    return pd.DataFrame(
        {
            "path_id": path_ids,
            "z0": np.repeat(z0, n_t),
            "t": np.tile(t_grid, n_paths),
            "z": z_hist.T.reshape(-1),
            "a": a_hist.T.reshape(-1),
            "m": m_hist.T.reshape(-1),
            "sigma_z": s_hist.T.reshape(-1),
            "M": M_hist.T.reshape(-1),
            "X": X_hist.T.reshape(-1),
        }
    )
