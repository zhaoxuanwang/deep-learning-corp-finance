"""Tests for the generic GMM core and basic-investment GMM integration."""

from __future__ import annotations

import numpy as np
import pytest

from src.v2.estimation import (
    GMMMonteCarloConfig,
    GMMRunConfig,
    GMMSpec,
    solve_gmm,
    validate_gmm,
)
from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    BasicInvestmentSMMSolverConfig,
    EconomicParams,
    ShockParams,
)
from src.v2.solvers.config import GridConfig, PFIConfig
from src.v2.utils.seeding import make_seed_int


_TOY_A = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=np.float64,
)


def _make_linear_gmm_spec(
    *,
    beta_true: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    seed: tuple[int, int] = (20, 26),
) -> GMMSpec:
    beta_true = (
        np.array([0.40, -0.20], dtype=np.float64)
        if beta_true is None
        else np.asarray(beta_true, dtype=np.float64)
    )
    guess = (
        np.array([0.0, 0.0], dtype=np.float64)
        if initial_guess is None
        else np.asarray(initial_guess, dtype=np.float64)
    )
    rng = np.random.default_rng(make_seed_int(seed, "linear_gmm"))
    noise = rng.normal(scale=0.05, size=(24, _TOY_A.shape[0]))
    noise = noise - np.mean(noise, axis=0, keepdims=True)
    observed_moments = _TOY_A @ beta_true + noise

    def compute_moment_contributions(beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=np.float64)
        return (_TOY_A @ beta)[None, :] - observed_moments

    def compute_moments(beta: np.ndarray) -> np.ndarray:
        return np.mean(compute_moment_contributions(beta), axis=0)

    def resample_spec(beta: np.ndarray, rep_seed: tuple[int, int]) -> GMMSpec:
        return _make_linear_gmm_spec(beta_true=beta, seed=rep_seed)

    return GMMSpec(
        parameter_names=("beta_0", "beta_1"),
        moment_names=("m_0", "m_1", "m_2"),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        initial_guess=guess,
        n_observations=24,
        n_firms=6,
        n_periods=4,
        compute_moments=compute_moments,
        compute_moment_contributions=compute_moment_contributions,
        resample_spec=resample_spec,
    )


def _make_basic_gmm_env() -> BasicInvestmentEnv:
    return BasicInvestmentEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.70,
            cost_convex=0.20,
            cost_fixed=0.0,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.70, sigma=0.10),
    )


def _make_tiny_pfi_solver_config() -> BasicInvestmentSMMSolverConfig:
    return BasicInvestmentSMMSolverConfig(
        method="PFI",
        pfi_config=PFIConfig(
            grid=GridConfig(
                exo_sizes=[3],
                endo_sizes=[4],
                action_sizes=[4],
            ),
            max_iter=3,
            eval_steps=4,
        ),
    )


def test_gmm_spec_validates_dimensions():
    with pytest.raises(ValueError, match="R >= K"):
        GMMSpec(
            parameter_names=("a", "b"),
            moment_names=("m",),
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            initial_guess=np.zeros(2),
            n_observations=4,
            n_firms=2,
            n_periods=2,
            compute_moments=lambda beta: np.asarray(beta[:1], dtype=np.float64),
            compute_moment_contributions=lambda beta: np.zeros((4, 1)),
        )

    with pytest.raises(ValueError, match="initial_guess shape"):
        GMMSpec(
            parameter_names=("a",),
            moment_names=("m",),
            bounds=((-1.0, 1.0),),
            initial_guess=np.zeros(2),
            n_observations=4,
            n_firms=2,
            n_periods=2,
            compute_moments=lambda beta: np.asarray(beta[:1], dtype=np.float64),
            compute_moment_contributions=lambda beta: np.zeros((4, 1)),
        )

    with pytest.raises(ValueError, match="n_observations"):
        GMMSpec(
            parameter_names=("a",),
            moment_names=("m",),
            bounds=((-1.0, 1.0),),
            initial_guess=np.zeros(1),
            n_observations=0,
            n_firms=2,
            n_periods=2,
            compute_moments=lambda beta: np.asarray(beta[:1], dtype=np.float64),
            compute_moment_contributions=lambda beta: np.zeros((4, 1)),
        )


def test_solve_gmm_recovers_linear_toy_problem():
    beta_true = np.array([0.40, -0.20], dtype=np.float64)
    spec = _make_linear_gmm_spec(beta_true=beta_true)
    config = GMMRunConfig(optimizer_name="L-BFGS-B", optimizer_maxiter=5)

    result = solve_gmm(spec, config=config)

    np.testing.assert_allclose(result.beta_hat, beta_true, atol=1e-5)
    assert result.parameter_names == spec.parameter_names
    assert result.moment_names == spec.moment_names
    assert result.stage1.moment_vector.shape == (3,)
    assert result.stage2.weighting_matrix.shape == (3, 3)
    assert result.j_df == 1
    assert np.all(np.isfinite(result.standard_errors))


def test_solve_gmm_is_deterministic_under_fixed_seed():
    spec = _make_linear_gmm_spec()
    config = GMMRunConfig(
        master_seed=(11, 17),
        optimizer_name="L-BFGS-B",
        optimizer_maxiter=3,
    )

    result_a = solve_gmm(spec, config=config)
    result_b = solve_gmm(spec, config=config)

    np.testing.assert_allclose(result_a.beta_hat, result_b.beta_hat, atol=1e-12)
    np.testing.assert_allclose(
        result_a.stage2.moment_vector,
        result_b.stage2.moment_vector,
        atol=1e-12,
    )


def test_validate_gmm_returns_monte_carlo_summary():
    beta_true = np.array([0.40, -0.20], dtype=np.float64)
    spec = _make_linear_gmm_spec(beta_true=beta_true)

    result = validate_gmm(
        spec,
        beta_true,
        run_config=GMMRunConfig(optimizer_name="L-BFGS-B", optimizer_maxiter=3),
        mc_config=GMMMonteCarloConfig(n_replications=2),
    )

    assert len(result.replications) == 2
    np.testing.assert_allclose(result.summary.mean_beta_hat, beta_true, atol=1e-5)
    assert np.all(np.isfinite(result.summary.bias))
    assert np.all(np.isfinite(result.summary.rmse))
    assert result.summary.coverage_95.shape == (2,)


def test_basic_investment_gmm_smoke_uses_env_owned_spec():
    env = _make_basic_gmm_env()
    solver_config = _make_tiny_pfi_solver_config()
    panel = env.simulate_gmm_panel(
        seed=(20, 26),
        n_firms=4,
        horizon=5,
        burn_in=1,
        solver_config=solver_config,
    )
    spec = env.make_gmm_spec(
        panel,
        initial_guess=env.gmm_true_beta(),
        solver_config=solver_config,
    )
    config = GMMRunConfig(optimizer_name="L-BFGS-B", optimizer_maxiter=1)

    result_a = solve_gmm(spec, config=config)
    result_b = solve_gmm(spec, config=config)

    assert spec.parameter_names == env.gmm_parameter_names()
    assert spec.moment_names == env.gmm_moment_names()
    assert spec.n_observations == panel.n_observations
    assert result_a.beta_hat.shape == (4,)
    assert result_a.stage2.moment_vector.shape == (6,)
    assert result_a.omega_hat.shape == (6, 6)
    assert result_a.j_df == 2
    assert np.all(np.isfinite(result_a.beta_hat))
    np.testing.assert_allclose(result_a.beta_hat, result_b.beta_hat, atol=1e-12)
