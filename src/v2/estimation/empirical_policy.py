"""Empirical policy-function regressions for simulated Nikolov panels."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class EmpiricalPolicyConfig:
    """Configuration for the auxiliary empirical policy regressions."""

    degree: int = 2
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1. Got {self.degree}")


@dataclass(frozen=True)
class EmpiricalPolicyOutcome:
    """One fitted auxiliary regression for a single action variable."""

    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    r2: float
    sigma2: float
    coef_se: np.ndarray
    dof: int


@dataclass(frozen=True)
class EmpiricalPolicyFit:
    """Collection of fitted auxiliary policy regressions."""

    config: EmpiricalPolicyConfig
    x_cols: tuple[str, ...]
    y_cols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_powers: tuple[tuple[int, ...], ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    xtx_inv: np.ndarray
    outcomes: dict[str, EmpiricalPolicyOutcome]
    n_observations: int


def fit_empirical_policy(
    df: pd.DataFrame,
    x_cols: list[str] | tuple[str, ...],
    y_cols: list[str] | tuple[str, ...],
    config: EmpiricalPolicyConfig | None = None,
) -> EmpiricalPolicyFit:
    """Fit separate polynomial OLS policy functions for each action."""

    config = config or EmpiricalPolicyConfig()
    x_cols = tuple(x_cols)
    y_cols = tuple(y_cols)
    sample = _finite_policy_sample(df, x_cols, y_cols)
    if sample.empty:
        raise ValueError("No finite non-default observations available for fitting.")

    x = sample.loc[:, x_cols].to_numpy(dtype=np.float64)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale <= 1e-12, 1.0, x_scale)
    x_model = (x - x_mean) / x_scale if config.standardize else x

    feature_powers = _polynomial_feature_powers(len(x_cols), config.degree)
    feature_names = _polynomial_feature_names(x_cols, feature_powers)
    design = _polynomial_design(x_model, feature_powers)

    n_obs = design.shape[0]
    n_features = design.shape[1]
    # Shared Gram inverse: the design X is identical for every y_col, so the
    # coefficient covariance scales with it (only sigma2 differs per outcome).
    xtx_inv = np.linalg.pinv(design.T @ design)

    outcomes: dict[str, EmpiricalPolicyOutcome] = {}
    for y_col in y_cols:
        y = sample[y_col].to_numpy(dtype=np.float64)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coefficients
        residuals = y - fitted
        ss_res = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(y - np.mean(y))))
        if ss_tot <= 1e-12:
            r2 = 1.0 if ss_res <= 1e-12 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        dof = n_obs - n_features
        if dof > 0:
            sigma2 = ss_res / dof
            coef_se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
        else:
            sigma2 = float("nan")
            coef_se = np.full(n_features, np.nan, dtype=np.float64)
        outcomes[y_col] = EmpiricalPolicyOutcome(
            coefficients=np.asarray(coefficients, dtype=np.float64),
            fitted_values=np.asarray(fitted, dtype=np.float64),
            residuals=np.asarray(residuals, dtype=np.float64),
            r2=float(r2),
            sigma2=float(sigma2),
            coef_se=np.asarray(coef_se, dtype=np.float64),
            dof=int(dof),
        )

    return EmpiricalPolicyFit(
        config=config,
        x_cols=x_cols,
        y_cols=y_cols,
        feature_names=feature_names,
        feature_powers=feature_powers,
        x_mean=x_mean,
        x_scale=x_scale,
        xtx_inv=np.asarray(xtx_inv, dtype=np.float64),
        outcomes=outcomes,
        n_observations=len(sample),
    )


def predict_policy_slices(
    fit: EmpiricalPolicyFit,
    df: pd.DataFrame,
    x_cols: list[str] | tuple[str, ...],
    grid_size: int = 100,
    quantile_range: tuple[float, float] = (0.05, 0.95),
) -> dict[str, pd.DataFrame]:
    """Predict fitted policies over one-dimensional state slices."""

    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2. Got {grid_size}")
    q_low, q_high = quantile_range
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            "quantile_range must satisfy 0 <= low < high <= 1. "
            f"Got {quantile_range}"
        )

    x_cols = tuple(x_cols)
    sample = _finite_policy_sample(df, x_cols, fit.y_cols)
    if sample.empty:
        raise ValueError("No finite non-default observations available for slicing.")

    held = sample.loc[:, x_cols].median()
    slices: dict[str, pd.DataFrame] = {}
    for x_col in x_cols:
        lo, hi = sample[x_col].quantile([q_low, q_high]).to_numpy(dtype=np.float64)
        grid = np.linspace(lo, hi, grid_size, dtype=np.float64)
        x_slice = pd.DataFrame(
            np.tile(held.to_numpy(dtype=np.float64), (grid_size, 1)),
            columns=x_cols,
        )
        x_slice[x_col] = grid
        pred = _predict_from_fit(fit, x_slice)
        pred.insert(0, x_col, grid)
        slices[x_col] = pred
    return slices


def partial_dependence_slices(
    fit: EmpiricalPolicyFit,
    df: pd.DataFrame,
    x_cols: list[str] | tuple[str, ...],
    grid_size: int = 100,
    ci_level: float = 0.95,
    quantile_range: tuple[float, float] = (0.02, 0.98),
    sample_cap: int = 4000,
    random_state: int | None = 0,
) -> dict[str, pd.DataFrame]:
    """Partial-dependence policy slices with pointwise confidence bands.

    For each state in ``x_cols`` the varied state is swept over its
    ``quantile_range`` window while the other states keep their *observed*
    values; the fitted prediction is averaged over the sample at each grid
    point. This integrates over the empirical joint distribution of the other
    states rather than pinning them at a single point. Because the partial
    dependence is linear in the OLS coefficients, the pointwise variance has
    the closed form ``sigma2 * Xbar @ xtx_inv @ Xbar``.
    """

    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2. Got {grid_size}")
    q_low, q_high = quantile_range
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            "quantile_range must satisfy 0 <= low < high <= 1. "
            f"Got {quantile_range}"
        )
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must satisfy 0 < ci_level < 1. Got {ci_level}")

    x_cols = tuple(x_cols)
    sample = _finite_policy_sample(df, x_cols, fit.y_cols)
    if sample.empty:
        raise ValueError("No finite non-default observations available for slicing.")
    if sample_cap is not None and len(sample) > sample_cap:
        sample = sample.sample(n=sample_cap, random_state=random_state)

    z_score = float(stats.norm.ppf(0.5 + 0.5 * ci_level))
    slices: dict[str, pd.DataFrame] = {}
    for x_col in x_cols:
        lo, hi = sample[x_col].quantile([q_low, q_high]).to_numpy(dtype=np.float64)
        grid = np.linspace(lo, hi, grid_size, dtype=np.float64)
        base = sample.loc[:, x_cols].to_numpy(dtype=np.float64)
        col_idx = x_cols.index(x_col)

        out = {x_col: grid}
        mean_design = np.empty((grid_size, len(fit.feature_names)), dtype=np.float64)
        for g_idx, value in enumerate(grid):
            x_eval = base.copy()
            x_eval[:, col_idx] = value
            design = _slice_design(fit, x_eval)
            mean_design[g_idx] = design.mean(axis=0)

        for y_col, outcome in fit.outcomes.items():
            pd_mean = mean_design @ outcome.coefficients
            var = np.einsum("gi,ij,gj->g", mean_design, fit.xtx_inv, mean_design)
            se = np.sqrt(np.maximum(outcome.sigma2 * var, 0.0))
            out[y_col] = pd_mean
            out[f"{y_col}_se"] = se
            out[f"{y_col}_lower"] = pd_mean - z_score * se
            out[f"{y_col}_upper"] = pd_mean + z_score * se
        slices[x_col] = pd.DataFrame(out)
    return slices


def empirical_policy_coefficients(fit: EmpiricalPolicyFit) -> pd.DataFrame:
    """Return fitted coefficients in long tabular form."""

    rows = []
    for y_col, outcome in fit.outcomes.items():
        for name, coefficient, se in zip(
            fit.feature_names, outcome.coefficients, outcome.coef_se
        ):
            t_stat = float(coefficient / se) if se > 0 else float("nan")
            rows.append(
                {
                    "outcome": y_col,
                    "feature": name,
                    "coefficient": float(coefficient),
                    "std_error": float(se),
                    "t_stat": t_stat,
                    "r2": outcome.r2,
                    "n_observations": fit.n_observations,
                }
            )
    return pd.DataFrame(rows)


def _slice_design(fit: EmpiricalPolicyFit, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_model = (x - fit.x_mean) / fit.x_scale if fit.config.standardize else x
    return _polynomial_design(x_model, fit.feature_powers)


def _predict_from_fit(
    fit: EmpiricalPolicyFit,
    x_df: pd.DataFrame,
) -> pd.DataFrame:
    design = _slice_design(fit, x_df.loc[:, fit.x_cols].to_numpy(dtype=np.float64))
    predictions = {
        y_col: design @ outcome.coefficients
        for y_col, outcome in fit.outcomes.items()
    }
    return pd.DataFrame(predictions)


def _finite_policy_sample(
    df: pd.DataFrame,
    x_cols: tuple[str, ...],
    y_cols: tuple[str, ...],
) -> pd.DataFrame:
    required = set(x_cols).union(y_cols)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    sample = df.copy()
    if "default" in sample.columns:
        sample = sample.loc[~sample["default"].astype(bool)]

    cols = list(x_cols + y_cols)
    finite = np.isfinite(sample.loc[:, cols].to_numpy(dtype=np.float64)).all(axis=1)
    return sample.loc[finite, cols]


def _polynomial_feature_powers(
    n_features: int,
    degree: int,
) -> tuple[tuple[int, ...], ...]:
    powers: list[tuple[int, ...]] = [tuple([0] * n_features)]
    for total_degree in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), total_degree):
            power = [0] * n_features
            for idx in combo:
                power[idx] += 1
            powers.append(tuple(power))
    return tuple(powers)


def _polynomial_feature_names(
    x_cols: tuple[str, ...],
    powers: tuple[tuple[int, ...], ...],
) -> tuple[str, ...]:
    names = []
    for power in powers:
        parts = []
        for col, exponent in zip(x_cols, power):
            if exponent == 1:
                parts.append(col)
            elif exponent > 1:
                parts.append(f"{col}^{exponent}")
        names.append(" ".join(parts) if parts else "1")
    return tuple(names)


def _polynomial_design(
    x: np.ndarray,
    powers: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    design = np.ones((x.shape[0], len(powers)), dtype=np.float64)
    for j, power in enumerate(powers):
        for i, exponent in enumerate(power):
            if exponent:
                design[:, j] *= np.power(x[:, i], exponent)
    return design
