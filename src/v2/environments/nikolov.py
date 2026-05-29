"""Shared primitives and grids for Nikolov-Schmid-Steri LP baselines.

This module is intentionally not an ``MDPEnvironment``.  The TO, LE, and MH
baselines are finite-grid linear programs with model-specific contract
constraints, so the generic reward/transition interface would add ceremony
without simplifying the numerical work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm as _norm


@dataclass(frozen=True)
class NikolovParams:
    """Economic parameters shared by the Nikolov LP model variants."""

    tau: float = 0.30
    alpha: float = 0.70
    f: float = 0.0
    delta: float = 0.15
    psi: float = 0.0
    r: float = 0.04
    kappa: float = 0.50
    eta_bar: float = 0.0
    xi: float = 0.50
    theta: float = 1.0
    lambda_diversion: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau < 1.0):
            raise ValueError(f"tau must be in [0, 1). Got {self.tau}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1). Got {self.alpha}")
        if self.f < 0.0:
            raise ValueError(f"f must be >= 0. Got {self.f}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1). Got {self.delta}")
        if self.psi < 0.0:
            raise ValueError(f"psi must be >= 0. Got {self.psi}")
        if self.r <= 0.0:
            raise ValueError(f"r must be > 0 for a discounted LP. Got {self.r}")
        if not (0.0 <= self.kappa <= 1.0):
            raise ValueError(f"kappa must be in [0, 1]. Got {self.kappa}")
        if self.eta_bar < 0.0:
            raise ValueError(f"eta_bar must be >= 0. Got {self.eta_bar}")
        if not (0.0 <= self.xi <= 1.0):
            raise ValueError(f"xi must be in [0, 1]. Got {self.xi}")
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(f"theta must be in [0, 1]. Got {self.theta}")
        if not (0.0 <= self.lambda_diversion <= 1.0):
            raise ValueError(
                "lambda_diversion must be in [0, 1]. "
                f"Got {self.lambda_diversion}"
            )


@dataclass(frozen=True)
class NikolovGridConfig:
    """User-imposed finite-grid configuration for the Nikolov LP baselines."""

    k_bounds: tuple[float, float] = (0.5, 1.5)
    b_bounds: tuple[float, float] = (0.0, 0.5)
    v_bounds: tuple[float, float] = (0.0, 0.5)
    z_bounds: tuple[float, float] = (0.8, 1.2)
    p_bounds: tuple[float, float] = (0.0, 0.05)
    d_bounds: tuple[float, float] = (0.0, 0.05)

    n_k: int = 3
    n_b: int = 3
    n_v: int = 3
    n_z: int = 3
    n_p: int = 3
    n_d: int = 3

    k_spacing: str = "linear"
    b_spacing: str = "linear"
    v_spacing: str = "linear"
    z_spacing: str = "linear"
    p_spacing: str = "linear"
    d_spacing: str = "linear"

    z_rho: float = 0.70
    z_sigma: float = 0.15
    z_mu: float = 0.0
    z_transition: Any | None = None

    def __post_init__(self) -> None:
        for name, size in [
            ("n_k", self.n_k),
            ("n_b", self.n_b),
            ("n_v", self.n_v),
            ("n_z", self.n_z),
            ("n_p", self.n_p),
            ("n_d", self.n_d),
        ]:
            if size < 2:
                raise ValueError(f"{name} must be >= 2. Got {size}")

        _validate_positive_bounds("k_bounds", self.k_bounds)
        _validate_positive_bounds("z_bounds", self.z_bounds)
        for name, bounds in [
            ("b_bounds", self.b_bounds),
            ("p_bounds", self.p_bounds),
        ]:
            _validate_zero_based_nonnegative_bounds(name, bounds)
        _validate_nonnegative_bounds("v_bounds", self.v_bounds)
        _validate_nonnegative_bounds("d_bounds", self.d_bounds)

        valid_spacings = {"linear", "log", "geometric"}
        for name, spacing in [
            ("k_spacing", self.k_spacing),
            ("b_spacing", self.b_spacing),
            ("v_spacing", self.v_spacing),
            ("z_spacing", self.z_spacing),
            ("p_spacing", self.p_spacing),
            ("d_spacing", self.d_spacing),
        ]:
            if spacing not in valid_spacings:
                raise ValueError(
                    f"{name} must be one of {sorted(valid_spacings)}. "
                    f"Got {spacing!r}"
                )
        if not (-1.0 < self.z_rho < 1.0):
            raise ValueError(f"z_rho must be in (-1, 1). Got {self.z_rho}")
        if self.z_sigma <= 0.0:
            raise ValueError(f"z_sigma must be > 0. Got {self.z_sigma}")


@dataclass(frozen=True)
class _TauchenShockParams:
    rho: float
    sigma: float
    mu: float


def profit_pre_tax(k, z, eta, params: NikolovParams):
    """Pre-tax operating profit."""

    return (z + eta) * np.power(k, params.alpha) - params.f


def adjustment_cost(k_next, k_current, params: NikolovParams):
    """Quadratic adjustment cost from the paper's capital convention."""

    safe_k = np.maximum(k_current, 1e-12)
    investment = k_next - (1.0 - params.delta) * k_current
    return 0.5 * params.psi * np.square(investment / safe_k) * safe_k


def eta_grid(params: NikolovParams) -> tuple[np.ndarray, np.ndarray]:
    """Return eta values and probabilities in fixed (+eta, -eta) order."""

    values = np.array([params.eta_bar, -params.eta_bar], dtype=np.float64)
    probs = np.array([params.kappa, 1.0 - params.kappa], dtype=np.float64)
    return values, probs


def ar1_z_bounds(
    *,
    rho: float,
    sigma: float,
    mu: float = 0.0,
    z_sd_mult: float = 2.5,
) -> tuple[float, float]:
    """Level-space bounds for log z from a stationary log-AR(1)."""

    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1). Got {rho}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0. Got {sigma}")
    if z_sd_mult <= 0.0:
        raise ValueError(f"z_sd_mult must be > 0. Got {z_sd_mult}")

    sigma_stationary = sigma / np.sqrt(1.0 - rho ** 2)
    low = float(np.exp(mu - z_sd_mult * sigma_stationary))
    high = float(np.exp(mu + z_sd_mult * sigma_stationary))
    return low, high


def reference_capital(
    params: NikolovParams,
    *,
    z_bar: float = 1.0,
) -> float:
    """Rough frictionless capital scale used for finite-grid calibration."""

    if z_bar <= 0.0:
        raise ValueError(f"z_bar must be > 0. Got {z_bar}")
    numerator = (1.0 - params.tau) * params.alpha * z_bar
    denominator = params.r + (1.0 - params.tau) * params.delta
    if denominator <= 0.0:
        raise ValueError("Capital reference denominator must be positive.")
    return float((numerator / denominator) ** (1.0 / (1.0 - params.alpha)))


def calibrated_k_bounds(
    params: NikolovParams,
    *,
    low_mult: float = 0.35,
    high_mult: float = 1.65,
    z_bar: float = 1.0,
) -> tuple[float, float]:
    """Capital bounds as multipliers of ``reference_capital``."""

    if not (0.0 < low_mult < high_mult):
        raise ValueError(
            "Require 0 < low_mult < high_mult. "
            f"Got {low_mult}, {high_mult}."
        )
    k_ref = reference_capital(params, z_bar=z_bar)
    return float(low_mult * k_ref), float(high_mult * k_ref)


def conservative_debt_bound(
    params: NikolovParams,
    *,
    k_low: float,
    pledge_fraction: float = 0.35,
) -> float:
    """Conservative debt upper bound from pledgeable depreciated capital."""

    if k_low <= 0.0:
        raise ValueError(f"k_low must be > 0. Got {k_low}")
    if not (0.0 < pledge_fraction <= 1.0):
        raise ValueError(
            f"pledge_fraction must be in (0, 1]. Got {pledge_fraction}"
        )
    return float(
        pledge_fraction
        * params.theta
        * (1.0 - params.delta)
        * k_low
        / (1.0 + params.r)
    )


def build_nikolov_grids(config: NikolovGridConfig) -> dict[str, np.ndarray]:
    """Build K, B, V, Z, payment/dividend grids and the Z transition matrix."""

    k_grid = _build_axis(config.k_bounds, config.n_k, config.k_spacing)
    b_grid = _build_axis(config.b_bounds, config.n_b, config.b_spacing)
    v_grid = _build_axis(config.v_bounds, config.n_v, config.v_spacing)
    z_grid = _build_axis(config.z_bounds, config.n_z, config.z_spacing)
    p_grid = _build_axis(config.p_bounds, config.n_p, config.p_spacing)
    d_grid = _build_axis(config.d_bounds, config.n_d, config.d_spacing)

    q = _resolve_transition_matrix(config, z_grid)

    return {
        "k": k_grid,
        "b": b_grid,
        "v": v_grid,
        "z": z_grid,
        "p": p_grid,
        "d": d_grid,
        "Q": q,
    }


def _validate_positive_bounds(name: str, bounds: tuple[float, float]) -> None:
    low, high = map(float, bounds)
    if not (0.0 < low < high):
        raise ValueError(f"{name} must satisfy 0 < low < high. Got {bounds}")


def _validate_zero_based_nonnegative_bounds(
    name: str,
    bounds: tuple[float, float],
) -> None:
    low, high = map(float, bounds)
    if low != 0.0 or high <= 0.0:
        raise ValueError(
            f"{name} must be zero-based and nonnegative: low=0 < high. "
            f"Got {bounds}"
        )


def _validate_nonnegative_bounds(name: str, bounds: tuple[float, float]) -> None:
    low, high = map(float, bounds)
    if not (0.0 <= low < high):
        raise ValueError(f"{name} must satisfy 0 <= low < high. Got {bounds}")


def _build_axis(
    bounds: tuple[float, float],
    size: int,
    spacing: str,
) -> np.ndarray:
    low, high = map(float, bounds)
    if spacing == "linear":
        grid = np.linspace(low, high, size, dtype=np.float64)
    elif spacing in {"log", "geometric"}:
        if low <= 0.0:
            raise ValueError("Log/geometric spacing requires a positive lower bound.")
        grid = np.exp(np.linspace(np.log(low), np.log(high), size)).astype(np.float64)
    else:  # pragma: no cover - validated in config
        raise ValueError(f"Unknown spacing {spacing!r}")
    if bounds[0] == 0.0:
        grid[0] = 0.0
    return grid


def _resolve_transition_matrix(
    config: NikolovGridConfig,
    z_grid: np.ndarray,
) -> np.ndarray:
    if config.z_transition is None:
        q = _tauchen_transition_matrix(
            _TauchenShockParams(config.z_rho, config.z_sigma, config.z_mu),
            z_grid,
        )
    else:
        q = np.asarray(config.z_transition, dtype=np.float64)

    expected_shape = (len(z_grid), len(z_grid))
    if q.shape != expected_shape:
        raise ValueError(
            "z_transition shape does not match z grid. "
            f"Expected {expected_shape}, got {q.shape}."
        )
    if np.any(q < -1e-12):
        raise ValueError("z_transition cannot contain negative probabilities.")
    row_sums = q.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("Rows of z_transition must sum to one.")
    return q / row_sums[:, None]


def _tauchen_transition_matrix(
    shock_params: _TauchenShockParams,
    z_grid: np.ndarray,
) -> np.ndarray:
    log_z = np.log(np.maximum(np.asarray(z_grid, dtype=np.float64), 1e-12))
    n = len(log_z)
    cond_mean = (
        (1.0 - shock_params.rho) * shock_params.mu
        + shock_params.rho * log_z
    )
    midpoints = (log_z[:-1] + log_z[1:]) / 2.0
    standardized = (midpoints[None, :] - cond_mean[:, None]) / shock_params.sigma
    cdf_vals = _norm.cdf(standardized)

    q = np.zeros((n, n), dtype=np.float64)
    q[:, 0] = cdf_vals[:, 0]
    q[:, 1:-1] = np.diff(cdf_vals, axis=1)
    q[:, -1] = 1.0 - cdf_vals[:, -1]
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
