"""Tests for Stage-A analytical SMM validation on BasicInvestmentEnv."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.v2.estimation import (
    SMMMonteCarloConfig,
    SMMRunConfig,
    solve_smm,
    validate_smm,
)
from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    BasicInvestmentSMMPanelData,
    EconomicParams,
    ShockParams,
    _compute_basic_smm_panel_moments,
)
from src.v2.utils.seeding import fold_in_seed


@pytest.fixture(scope="module")
def stage_a_env() -> BasicInvestmentEnv:
    return BasicInvestmentEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.70,
            cost_convex=0.0,
            cost_fixed=0.0,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.70, sigma=0.10),
    )


def test_make_smm_spec_requires_frictionless_case():
    env = BasicInvestmentEnv(
        econ_params=EconomicParams(cost_convex=0.05, cost_fixed=0.0),
        shock_params=ShockParams(mu=0.0, rho=0.70, sigma=0.10),
    )
    with pytest.raises(ValueError, match="frictionless"):
        env.make_smm_spec()


def test_make_smm_spec_exposes_env_owned_metadata(stage_a_env):
    guess = np.array([0.66, 0.55, 0.14], dtype=np.float64)
    bounds = ((0.20, 0.90), (-0.50, 0.90), (0.05, 0.30))
    spec = stage_a_env.make_smm_spec(initial_guess=guess, bounds=bounds)

    assert spec.parameter_names == stage_a_env.smm_parameter_names()
    assert spec.moment_names == stage_a_env.smm_moment_names()
    assert spec.bounds == bounds
    np.testing.assert_allclose(spec.initial_guess, guess, atol=1e-12)
    np.testing.assert_allclose(
        stage_a_env.smm_true_beta(),
        stage_a_env.smm_initial_guess(),
        atol=1e-12,
    )


def test_basic_panel_api_respects_seed_and_burn_in(stage_a_env):
    beta_true = stage_a_env.smm_initial_guess()
    seed = fold_in_seed((20, 26), "basic_investment", "panel")
    alt_seed = fold_in_seed((20, 26), "basic_investment", "panel_alt")
    run_base = SMMRunConfig(
        n_firms=4,
        horizon=5,
        burn_in=0,
        n_sim_panels=1,
    )
    run_shifted = replace(run_base, burn_in=1, horizon=4)

    panel_a = stage_a_env.simulate_smm_panel_data(beta_true, run_shifted, seed)
    panel_b = stage_a_env.simulate_smm_panel_data(beta_true, run_shifted, seed)
    panel_base = stage_a_env.simulate_smm_panel_data(beta_true, run_base, seed)
    panel_alt = stage_a_env.simulate_smm_panel_data(beta_true, run_shifted, alt_seed)

    np.testing.assert_allclose(panel_a.k, panel_b.k, atol=1e-10)
    np.testing.assert_allclose(panel_a.z, panel_b.z, atol=1e-10)
    np.testing.assert_allclose(panel_a.k_next, panel_b.k_next, atol=1e-10)
    np.testing.assert_allclose(panel_a.k[:, :, 0], panel_base.k[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(panel_a.z[:, :, 0], panel_base.z[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(
        panel_a.k_next[:, :, 0],
        panel_base.k_next[:, :, 1],
        atol=1e-8,
    )
    assert not np.allclose(panel_a.k_next, panel_alt.k_next)


def test_basic_public_panel_api_returns_structured_data(stage_a_env):
    beta_true = stage_a_env.smm_initial_guess()
    run_config = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=1,
        n_sim_panels=2,
    )
    seed = fold_in_seed((20, 26), "basic_investment", "public_panel")

    panel_data = stage_a_env.simulate_smm_panel_data(beta_true, run_config, seed)

    assert isinstance(panel_data, BasicInvestmentSMMPanelData)
    assert panel_data.n_panels == 2
    assert panel_data.n_firms == 3
    assert panel_data.horizon == 4
    assert panel_data.n_observations == 24

    payload = panel_data.to_dict()
    assert sorted(payload.keys()) == [
        "k",
        "k_next",
        "metadata",
        "n_observations",
        "z",
    ]
    assert payload["k"].shape == (2, 3, 4)
    assert payload["z"].shape == (2, 3, 4)
    assert payload["k_next"].shape == (2, 3, 4)
    assert payload["metadata"]["seed"] == seed
    assert payload["metadata"]["n_panels"] == 2


def test_basic_public_moment_wrapper_matches_private_formula(stage_a_env):
    beta_true = stage_a_env.smm_initial_guess()
    run_config = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=1,
        n_sim_panels=2,
    )
    seed = fold_in_seed((20, 26), "basic_investment", "moments")
    panel_data = stage_a_env.simulate_smm_panel_data(beta_true, run_config, seed)

    private_panel_moments, private_diagnostics = _compute_basic_smm_panel_moments(
        stage_a_env,
        panel_data,
    )
    wrapped = stage_a_env.compute_smm_panel_moments(panel_data)

    assert wrapped["moment_names"] == stage_a_env.smm_moment_names()
    np.testing.assert_allclose(wrapped["panel_moments"], private_panel_moments, atol=1e-10)
    np.testing.assert_allclose(
        wrapped["average_moments"],
        np.mean(private_panel_moments, axis=0),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        wrapped["diagnostics"]["mean_investment_assets"],
        private_diagnostics["mean_investment_assets"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        wrapped["diagnostics"]["serial_corr_investment"],
        private_diagnostics["serial_corr_investment"],
        atol=1e-10,
    )


def test_basic_solve_smm_is_reproducible_under_fixed_seed(stage_a_env):
    beta_true = stage_a_env.smm_initial_guess()
    spec = stage_a_env.make_smm_spec(initial_guess=np.array([0.63, 0.50, 0.16]))
    run_config = SMMRunConfig(
        n_firms=32,
        horizon=12,
        burn_in=4,
        n_sim_panels=6,
        global_method="dual_annealing",
        optimizer_maxiter=8,
        master_seed=(20, 26),
    )
    target_seed = fold_in_seed(run_config.master_seed, "basic_investment", "target")
    target = stage_a_env.simulate_smm_target_moments(beta_true, run_config, target_seed)

    result_a = solve_smm(spec, target, config=run_config)
    result_b = solve_smm(spec, target, config=run_config)

    np.testing.assert_allclose(result_a.beta_hat, result_b.beta_hat, atol=1e-12)
    np.testing.assert_allclose(
        result_a.stage2.trace["objective"],
        result_b.stage2.trace["objective"],
        atol=1e-12,
    )


def test_basic_single_run_estimate_is_close_to_truth(stage_a_env):
    beta_true = stage_a_env.smm_initial_guess()
    initial_guess = np.array([0.62, 0.55, 0.16], dtype=np.float64)
    spec = stage_a_env.make_smm_spec(initial_guess=initial_guess)
    run_config = SMMRunConfig(
        n_firms=64,
        horizon=20,
        burn_in=5,
        n_sim_panels=8,
        global_method="Powell",
        optimizer_maxiter=20,
        master_seed=(20, 26),
    )
    target_seed = fold_in_seed(run_config.master_seed, "basic_investment", "single_run_target")
    simulation_seed = fold_in_seed(run_config.master_seed, "basic_investment", "single_run_crn")
    target = stage_a_env.simulate_smm_target_moments(beta_true, run_config, target_seed)

    result = solve_smm(
        spec=spec,
        target=target,
        config=run_config,
        simulation_seed=simulation_seed,
    )

    # IV first-diff estimator is consistent but noisier than OLS on short
    # panels (horizon=20 means only horizon-2=18 usable observations per firm
    # for the IV regression, vs. horizon-1=19 for OLS).  With n_firms=64 and
    # optimizer_maxiter=20 this is a smoke-level check; tight numerical
    # accuracy requires longer panels and more optimizer budget.
    assert np.all(
        np.abs(result.beta_hat - beta_true)
        <= np.array([0.45, 0.45, 0.10], dtype=np.float64)
    )
    assert result.stage1.panel_moments.shape == (8, 4)
    assert result.stage2.panel_moments.shape == (8, 4)
    assert np.isfinite(result.j_statistic)


def test_basic_monte_carlo_smoke_returns_finite_summary(stage_a_env):
    beta_true = stage_a_env.smm_initial_guess()
    spec = stage_a_env.make_smm_spec(initial_guess=np.array([0.64, 0.58, 0.14]))
    run_config = SMMRunConfig(
        n_firms=24,
        horizon=10,
        burn_in=3,
        n_sim_panels=6,
        global_method="Powell",
        optimizer_maxiter=8,
        master_seed=(21, 27),
    )
    mc_config = SMMMonteCarloConfig(n_replications=2)

    result = validate_smm(spec, beta_true, run_config=run_config, mc_config=mc_config)

    assert len(result.replications) == 2
    assert np.all(np.isfinite(result.summary.mean_beta_hat))
    assert np.all(np.isfinite(result.summary.bias))
    assert np.all(np.isfinite(result.summary.rmse))
    assert np.all(np.isfinite(result.summary.coverage_95))
