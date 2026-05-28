"""Indirect inference (II) for the Nikolov-Schmid-Steri Trade-Off (TO) model.

The auxiliary model is the empirical policy-function regression of the doc:
each action ``y in {investment_rate, future_leverage, payout_rate}`` is regressed
on a degree-2 polynomial in the observable states
``x in {log_k, profitability, leverage}``.  The stacked regression coefficients
form the binding function ``h(D)`` for a panel ``D``.  Indirect inference matches
``h_data = h(real panel)`` against ``h_sim(beta) = mean_r h(sim panel r)``.

This module owns the ONLY ``beta -> NikolovParams`` and ``beta -> NikolovGridConfig``
builders, so the displayed anchor solve in a notebook and the II loop cannot drift
apart: evaluating the II callback at ``beta_anchor`` reproduces the anchor solve
exactly.  The generic two-step SMM core (``solve_smm``) is reused unchanged; the
moments are simply the auxiliary coefficients rather than data moments.

Design decisions baked in (see the project plan):
  * TO only.
  * ``standardize=False`` so coefficient vectors are comparable across panels:
    data and simulation are fit on RAW observables in a common reference frame.
  * ``drop_intercept_from_moments=True`` so only slopes/curvature are matched.
    The rate-variable levels differ between model and data by large factors
    (mainly the deflator gap: the model deflates by capital k, the data by total
    assets AT), which no structural parameter can reconcile, so the intercepts
    are not matched and the level offset is absorbed by the dropped intercept.
  * Identity-W headline (use ``two_step=False`` for a one-step point estimate).
  * Real-data observables use the winsorized columns; no firm-fixed-effect
    reconciliation at this baseline stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.v2.environments.nikolov import (
    NikolovGridConfig,
    NikolovParams,
    ar1_z_bounds,
    calibrated_k_bounds,
    reference_capital,
)
from src.v2.estimation.empirical_policy import (
    EmpiricalPolicyConfig,
    EmpiricalPolicyFit,
    fit_empirical_policy,
)
from src.v2.estimation.smm import (
    SMMPanelMoments,
    SMMRunConfig,
    SMMSolveResult,
    SMMSpec,
    SMMTargetMoments,
    solve_smm,
)
from src.v2.simulation.nikolov_to import (
    NikolovTOSimulationConfig,
    construct_to_observables,
    simulate_nikolov_to_panel,
)
from src.v2.solvers.nikolov_to import solve_nikolov_to
from src.v2.utils.seeding import fold_in_seed, make_seed_int

# Parameters that map onto NikolovParams fields (the rest feed grid construction).
_PARAM_FIELDS = ("alpha", "f", "delta", "psi", "eta_bar", "xi")
# Sentinel added to h_data when a candidate beta cannot be solved, so the
# objective is huge and the optimizer steps away.
_PENALTY = 1e3


@dataclass(frozen=True)
class NikolovIIConfig:
    """Configuration for Trade-Off indirect inference.

    The same instance drives ``compute_h_data`` and every simulated fit, so the
    auxiliary model is identical on both sides of ``g(beta) = h_data - h_sim``.
    """

    param_names: tuple[str, ...] = (
        "alpha", "f", "z_rho", "z_sigma", "delta", "psi", "eta_bar", "xi",
    )
    bounds: tuple[tuple[float, float], ...] = (
        # alpha capped at 0.85: reference_capital ~ base^(1/(1-alpha)), so the
        # grid scale explodes as alpha -> 1.  The ratio observables are
        # scale-free, but keeping alpha <= 0.85 avoids extreme capital grids.
        (0.50, 0.85),   # alpha
        (0.0, 1.50),    # f
        (0.30, 0.95),   # z_rho
        (0.05, 0.45),   # z_sigma
        (0.05, 0.25),   # delta
        (0.0, 2.0),     # psi
        (0.0, 0.50),    # eta_bar
        (0.20, 0.90),   # xi
    )
    initial_guess: tuple[float, ...] | None = None  # default: bounds midpoints
    fixed_params: Mapping[str, float] = field(
        default_factory=lambda: dict(
            tau=0.30, r=0.04, kappa=0.50,
            theta=1.0, lambda_diversion=0.0,
        )
    )
    z_mu: float = 0.0

    # auxiliary-model config (shared by the diagnostic plot and II)
    x_cols: tuple[str, ...] = ("log_k", "profitability", "leverage")
    y_cols: tuple[str, ...] = ("investment_rate", "future_leverage", "payout_rate")
    degree: int = 2
    standardize: bool = False
    drop_intercept_from_moments: bool = True

    # deterministic grid-construction knobs (single source of truth).
    # The bound recipe mirrors notebook 10's known-feasible BASE setup: a
    # generous debt ceiling (multiple of reference capital) is required so
    # firms can roll over inherited debt and keep limited-liability feasible --
    # the tight pledgeable-collateral bound leaves states with no feasible
    # action.  The default 15x12x5 size solves in ~8s; cost grows steeply
    # (O(n_k^2 n_b^2 n_z)), e.g. 25x20x7 is ~2 min/solve, so larger grids make
    # a full multi-parameter II run take many hours.
    n_k: int = 15
    n_b: int = 12
    n_z: int = 5
    k_low_mult: float = 0.10
    k_high_mult: float = 2.00
    z_sd_mult: float = 3.0
    debt_fraction_of_k_ref: float = 1.50
    k_spacing: str = "log"
    b_spacing: str = "linear"
    z_spacing: str = "log"

    def __post_init__(self) -> None:
        if len(self.param_names) != len(self.bounds):
            raise ValueError(
                "param_names and bounds must have matching lengths. "
                f"Got {len(self.param_names)} and {len(self.bounds)}."
            )
        if self.initial_guess is not None and len(self.initial_guess) != len(self.param_names):
            raise ValueError(
                "initial_guess length must match param_names. "
                f"Got {len(self.initial_guess)} and {len(self.param_names)}."
            )

    def resolved_initial_guess(self) -> np.ndarray:
        if self.initial_guess is not None:
            return np.asarray(self.initial_guess, dtype=np.float64)
        return np.array(
            [0.5 * (lo + hi) for lo, hi in self.bounds], dtype=np.float64
        )


def _beta_dict(beta: Sequence[float], cfg: NikolovIIConfig) -> dict[str, float]:
    """Map a beta vector to a name->value dict, clipped to bounds."""
    beta = np.asarray(beta, dtype=np.float64)
    if beta.size != len(cfg.param_names):
        raise ValueError(
            f"beta has {beta.size} entries but param_names has "
            f"{len(cfg.param_names)}."
        )
    clipped = {}
    for name, value, (lo, hi) in zip(cfg.param_names, beta, cfg.bounds):
        clipped[name] = float(np.clip(value, lo, hi))
    return clipped


def build_nikolov_params(beta: Sequence[float], cfg: NikolovIIConfig) -> NikolovParams:
    """Deterministic ``beta -> NikolovParams`` (clipped to bounds)."""
    values = _beta_dict(beta, cfg)
    kwargs = dict(cfg.fixed_params)
    for name in _PARAM_FIELDS:
        if name in values:
            kwargs[name] = values[name]
    return NikolovParams(**kwargs)


def build_nikolov_grid_config(
    beta: Sequence[float], cfg: NikolovIIConfig
) -> NikolovGridConfig:
    """Deterministic ``beta -> NikolovGridConfig``.

    Capital bounds track ``reference_capital`` (which depends on alpha/delta/tau/r),
    z bounds track the stationary AR(1) spread, and the debt ceiling tracks
    pledgeable depreciated capital, so identical beta always yields an identical
    grid and the moment surface is well-defined.
    """
    values = _beta_dict(beta, cfg)
    params = build_nikolov_params(beta, cfg)
    z_rho = values.get("z_rho", 0.70)
    z_sigma = values.get("z_sigma", 0.15)

    z_bounds = ar1_z_bounds(
        rho=z_rho, sigma=z_sigma, mu=cfg.z_mu, z_sd_mult=cfg.z_sd_mult
    )
    z_bar = float(np.exp(cfg.z_mu))
    k_bounds = calibrated_k_bounds(
        params, low_mult=cfg.k_low_mult, high_mult=cfg.k_high_mult, z_bar=z_bar
    )
    k_ref = reference_capital(params, z_bar=z_bar)
    b_high = float(cfg.debt_fraction_of_k_ref * k_ref)
    return NikolovGridConfig(
        k_bounds=k_bounds,
        b_bounds=(0.0, b_high),
        z_bounds=z_bounds,
        n_k=cfg.n_k,
        n_b=cfg.n_b,
        n_z=cfg.n_z,
        k_spacing=cfg.k_spacing,
        b_spacing=cfg.b_spacing,
        z_spacing=cfg.z_spacing,
        z_rho=z_rho,
        z_sigma=z_sigma,
        z_mu=cfg.z_mu,
        z_transition=None,
    )


def pack_beta(
    param_kwargs: Mapping[str, float],
    ar1_kwargs: Mapping[str, float],
    cfg: NikolovIIConfig,
) -> np.ndarray:
    """Assemble ``beta_anchor`` from notebook 10's PARAM_KWARGS / AR1_KWARGS.

    ``z_rho``/``z_sigma`` are read from ``ar1_kwargs`` (keys ``rho``/``sigma``);
    every other parameter is read from ``param_kwargs``.
    """
    ar1_lookup = {"z_rho": ar1_kwargs.get("rho"), "z_sigma": ar1_kwargs.get("sigma")}
    beta = []
    for name in cfg.param_names:
        if name in ar1_lookup and ar1_lookup[name] is not None:
            beta.append(float(ar1_lookup[name]))
        elif name in param_kwargs:
            beta.append(float(param_kwargs[name]))
        else:
            raise KeyError(
                f"Cannot resolve anchor value for parameter {name!r} from the "
                "supplied param_kwargs / ar1_kwargs."
            )
    return np.asarray(beta, dtype=np.float64)


def _aux_config(cfg: NikolovIIConfig) -> EmpiricalPolicyConfig:
    return EmpiricalPolicyConfig(degree=cfg.degree, standardize=cfg.standardize)


def _moment_mask(fit: EmpiricalPolicyFit, cfg: NikolovIIConfig) -> np.ndarray:
    """Boolean mask over feature_names selecting non-intercept terms if requested."""
    if not cfg.drop_intercept_from_moments:
        return np.ones(len(fit.feature_names), dtype=bool)
    return np.array([name != "1" for name in fit.feature_names], dtype=bool)


def _stack_coefficients(fit: EmpiricalPolicyFit, cfg: NikolovIIConfig) -> np.ndarray:
    """Stack coefficients in fixed (y_cols outer, feature inner) order."""
    mask = _moment_mask(fit, cfg)
    return np.concatenate(
        [fit.outcomes[y].coefficients[mask] for y in cfg.y_cols]
    )


def _moment_names(fit: EmpiricalPolicyFit, cfg: NikolovIIConfig) -> list[str]:
    mask = _moment_mask(fit, cfg)
    kept = [name for name, keep in zip(fit.feature_names, mask) if keep]
    return [f"{y}::{name}" for y in cfg.y_cols for name in kept]


def compute_h_data(
    real_panel_df: pd.DataFrame, cfg: NikolovIIConfig
) -> tuple[np.ndarray, list[str], int]:
    """Fit the auxiliary regression on the real panel and stack coefficients."""
    fit = fit_empirical_policy(real_panel_df, cfg.x_cols, cfg.y_cols, _aux_config(cfg))
    return _stack_coefficients(fit, cfg), _moment_names(fit, cfg), fit.n_observations


def make_nikolov_ii_target(
    real_panel_df: pd.DataFrame, cfg: NikolovIIConfig
) -> SMMTargetMoments:
    """Compute ``h_data`` once and wrap it as fixed SMM target moments."""
    h_data, moment_names, n_obs = compute_h_data(real_panel_df, cfg)
    return SMMTargetMoments(
        values=h_data,
        n_observations=n_obs,
        metadata={"moment_names": tuple(moment_names)},
    )


def make_nikolov_ii_spec(
    cfg: NikolovIIConfig,
    real_panel_df: pd.DataFrame,
    *,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> SMMSpec:
    """Build the SMM spec whose moments are the auxiliary policy coefficients."""
    h_data, moment_names, _ = compute_h_data(real_panel_df, cfg)
    n_moments = h_data.size
    solver_kwargs = dict(solver_kwargs or {})
    rng_jitter = np.random.default_rng(0)

    def _sentinel(n_panels: int) -> SMMPanelMoments:
        jitter = 1e-6 * rng_jitter.standard_normal((n_panels, n_moments))
        return SMMPanelMoments(
            panel_moments=h_data[None, :] + _PENALTY + jitter,
            n_observations=1,
        )

    def _simulate_panel_moments(beta, run_config, seed) -> SMMPanelMoments:
        S = run_config.n_sim_panels
        try:
            params = build_nikolov_params(beta, cfg)
            grid_cfg = build_nikolov_grid_config(beta, cfg)
            to_result = solve_nikolov_to(grid_cfg, params, **solver_kwargs)
        except (RuntimeError, ValueError):
            return _sentinel(S)

        rows: list[np.ndarray] = []
        total_obs = 0
        for r in range(S):
            sim_seed = make_seed_int(fold_in_seed(seed, "nikolov_ii", "panel", r))
            sim_cfg = NikolovTOSimulationConfig(
                n_firms=run_config.n_firms,
                horizon=run_config.horizon,
                burn_in=run_config.burn_in,
                seed=sim_seed,
                default_restart=True,
            )
            panel = simulate_nikolov_to_panel(to_result, params, sim_cfg)
            obs = construct_to_observables(panel)
            try:
                fit = fit_empirical_policy(
                    obs, cfg.x_cols, cfg.y_cols, _aux_config(cfg)
                )
                coef = _stack_coefficients(fit, cfg)
                if not np.all(np.isfinite(coef)):
                    raise ValueError("non-finite coefficients")
                rows.append(coef)
                total_obs += int(fit.n_observations)
            except ValueError:
                rows.append(h_data + _PENALTY)
        return SMMPanelMoments(
            panel_moments=np.vstack(rows),
            n_observations=max(total_obs, 1),
        )

    return SMMSpec(
        parameter_names=tuple(cfg.param_names),
        moment_names=tuple(moment_names),
        bounds=cfg.bounds,
        initial_guess=cfg.resolved_initial_guess(),
        simulate_panel_moments=_simulate_panel_moments,
    )


def run_nikolov_ii(
    real_panel_df: pd.DataFrame,
    cfg: NikolovIIConfig,
    run_config: SMMRunConfig,
    *,
    simulation_seed: tuple[int, int] | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    two_step: bool = True,
    compute_standard_errors: bool = True,
) -> SMMSolveResult:
    """Run Trade-Off indirect inference end to end.

    Set ``two_step=False, compute_standard_errors=False`` for a fast one-step
    point estimate (no Omega, no 2K Jacobian solves).
    """
    spec = make_nikolov_ii_spec(cfg, real_panel_df, solver_kwargs=solver_kwargs)
    target = make_nikolov_ii_target(real_panel_df, cfg)
    return solve_smm(
        spec,
        target,
        run_config,
        simulation_seed=simulation_seed,
        two_step=two_step,
        compute_standard_errors=compute_standard_errors,
    )
