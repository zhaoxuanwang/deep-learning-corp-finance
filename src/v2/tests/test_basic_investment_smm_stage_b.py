"""Tests for frictional-numerical solve-in-loop SMM on BasicInvestmentEnv."""

from __future__ import annotations

import numpy as np
import pytest

from src.v2.data.generator import DataGeneratorConfig
from src.v2.estimation import SMMRunConfig, solve_smm
from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    BasicInvestmentSMMSolverBundle,
    BasicInvestmentSMMSolverConfig,
    EconomicParams,
    ShockParams,
)
from src.v2.solvers.config import GridConfig, PFIConfig
from src.v2.trainers.config import ERConfig, NetworkConfig
from src.v2.utils.seeding import fold_in_seed


def _make_frictional_numerical_env(cost_fixed: float = 0.0) -> BasicInvestmentEnv:
    return BasicInvestmentEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.70,
            cost_convex=0.20,
            cost_fixed=cost_fixed,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.70, sigma=0.10),
    )


def _make_er_solver_config() -> BasicInvestmentSMMSolverConfig:
    return BasicInvestmentSMMSolverConfig(
        method="ER",
        dataset_config=DataGeneratorConfig(
            n_paths=8,
            horizon=4,
        ),
        er_config=ERConfig(
            n_steps=2,
            batch_size=4,
            eval_interval=1,
            monitor="euler_residual_val",
            mode="min",
            threshold_patience=1,
        ),
        er_network_config=NetworkConfig(
            n_layers=1,
            n_neurons=8,
        ),
    )


def _make_pfi_solver_config() -> BasicInvestmentSMMSolverConfig:
    return BasicInvestmentSMMSolverConfig(
        method="PFI",
        dataset_config=DataGeneratorConfig(
            n_paths=8,
            horizon=4,
        ),
        pfi_config=PFIConfig(
            grid=GridConfig(
                exo_sizes=[3],
                endo_sizes=[5],
                action_sizes=[5],
            ),
            max_iter=3,
            eval_steps=5,
        ),
    )


@pytest.fixture(scope="module")
def frictional_numerical_env() -> BasicInvestmentEnv:
    return _make_frictional_numerical_env()


# ── Fast tests (no model solve) ──────────────────────────────────────


def test_frictional_numerical_metadata_switches_to_four_parameters_and_five_moments(frictional_numerical_env):
    spec = frictional_numerical_env.make_smm_spec(
        mode="frictional_numerical",
        initial_guess=np.array([0.66, 0.18, 0.60, 0.12], dtype=np.float64),
        solver_config=_make_er_solver_config(),
    )

    assert spec.parameter_names == ("alpha", "psi1", "rho", "sigma_epsilon")
    assert spec.moment_names == (
        "mean_investment_assets",
        "var_investment_assets",
        "serial_corr_investment",
        "income_ar1_beta",
        "income_ar1_resid_std",
    )
    assert len(spec.bounds) == 4
    np.testing.assert_allclose(spec.initial_guess, [0.66, 0.18, 0.60, 0.12])


def test_frictional_numerical_rejects_fixed_cost_models():
    env = _make_frictional_numerical_env(cost_fixed=0.05)
    with pytest.raises(ValueError, match="cost_fixed == 0"):
        env.make_smm_spec(mode="frictional_numerical", solver_config=_make_er_solver_config())


# ── Slow tests (involve model solves) ────────────────────────────────


@pytest.mark.slow
def test_frictional_numerical_pfi_panel_api_is_seeded_and_bundle_reusable(frictional_numerical_env):
    beta_true = frictional_numerical_env.smm_initial_guess(mode="frictional_numerical")
    run_config = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=1,
        n_sim_panels=2,
    )
    solver_config = _make_pfi_solver_config()
    seed = fold_in_seed((20, 26), "frictional_numerical", "panel")
    alt_seed = fold_in_seed((20, 26), "frictional_numerical", "panel_alt")

    # Solve once, reuse bundle for all panels to avoid redundant PFI solves.
    bundle = frictional_numerical_env.solve_smm_policy_bundle(
        beta_true,
        solver_config,
        seed=fold_in_seed(seed, "solver"),
    )

    panel_a = frictional_numerical_env.simulate_smm_panel_data(
        beta_true,
        run_config,
        seed,
        mode="frictional_numerical",
        solver_config=solver_config,
        policy_bundle=bundle,
    )
    panel_b = frictional_numerical_env.simulate_smm_panel_data(
        beta_true,
        run_config,
        seed,
        mode="frictional_numerical",
        solver_config=solver_config,
        policy_bundle=bundle,
    )
    panel_alt = frictional_numerical_env.simulate_smm_panel_data(
        beta_true,
        run_config,
        alt_seed,
        mode="frictional_numerical",
        solver_config=solver_config,
        policy_bundle=bundle,
    )

    np.testing.assert_allclose(panel_a.k, panel_b.k, atol=1e-10)
    np.testing.assert_allclose(panel_a.z, panel_b.z, atol=1e-10)
    np.testing.assert_allclose(panel_a.k_next, panel_b.k_next, atol=1e-10)
    # A different seed must change the simulated shock path. Checked on z (the
    # stochastic driver), not k_next: on this coarse smoke grid the PFI policy
    # can be near-constant, so k_next is state-independent and matches across seeds.
    assert not np.allclose(panel_a.z, panel_alt.z)

    wrapped = frictional_numerical_env.compute_smm_panel_moments(panel_a)
    assert wrapped["mode"] == "frictional_numerical"
    assert wrapped["moment_names"] == frictional_numerical_env.smm_moment_names(mode="frictional_numerical")
    assert wrapped["panel_moments"].shape == (2, 5)


@pytest.mark.slow
def test_frictional_numerical_er_bundle_is_deterministic_under_fixed_seed(frictional_numerical_env):
    beta_true = frictional_numerical_env.smm_initial_guess(mode="frictional_numerical")
    solver_config = _make_er_solver_config()
    seed = fold_in_seed((20, 26), "frictional_numerical", "er_bundle")

    bundle_a = frictional_numerical_env.solve_smm_policy_bundle(beta_true, solver_config, seed)
    bundle_b = frictional_numerical_env.solve_smm_policy_bundle(beta_true, solver_config, seed)

    assert isinstance(bundle_a, BasicInvestmentSMMSolverBundle)
    assert bundle_a.method == "ER"
    assert bundle_a.selected_step == bundle_b.selected_step
    assert bundle_a.stop_reason == bundle_b.stop_reason
    assert np.isfinite(bundle_a.best_metric_value)
    assert np.isfinite(bundle_a.wall_time_sec)

    run_config = SMMRunConfig(
        n_firms=2,
        horizon=3,
        burn_in=1,
        n_sim_panels=1,
    )
    panel_a = frictional_numerical_env.simulate_smm_panel_data(
        beta_true,
        run_config,
        seed=(11, 17),
        mode="frictional_numerical",
        solver_config=solver_config,
        policy_bundle=bundle_a,
    )
    panel_b = frictional_numerical_env.simulate_smm_panel_data(
        beta_true,
        run_config,
        seed=(11, 17),
        mode="frictional_numerical",
        solver_config=solver_config,
        policy_bundle=bundle_b,
    )
    np.testing.assert_allclose(panel_a.k_next, panel_b.k_next, atol=1e-10)


@pytest.mark.slow
def test_frictional_numerical_pfi_bundle_returns_finite_panel_moments(frictional_numerical_env):
    beta_true = frictional_numerical_env.smm_initial_guess(mode="frictional_numerical")
    solver_config = _make_pfi_solver_config()
    bundle = frictional_numerical_env.solve_smm_policy_bundle(
        beta_true,
        solver_config,
        seed=fold_in_seed((20, 26), "frictional_numerical", "pfi_bundle"),
    )

    assert bundle.method == "PFI"
    assert bundle.converged
    assert bundle.stop_reason == "converged"

    panel = frictional_numerical_env.simulate_smm_panel_data(
        beta_true,
        SMMRunConfig(n_firms=3, horizon=4, burn_in=1, n_sim_panels=1),
        seed=(3, 5),
        mode="frictional_numerical",
        solver_config=solver_config,
        policy_bundle=bundle,
    )
    wrapped = frictional_numerical_env.compute_smm_panel_moments(panel)
    assert wrapped["panel_moments"].shape == (1, 5)
    assert np.all(np.isfinite(wrapped["panel_moments"]))


@pytest.mark.slow
def test_frictional_numerical_er_spec_simulates_finite_panel_moments(frictional_numerical_env):
    beta_true = frictional_numerical_env.smm_initial_guess(mode="frictional_numerical")
    solver_config = _make_er_solver_config()
    spec = frictional_numerical_env.make_smm_spec(
        mode="frictional_numerical",
        initial_guess=beta_true,
        solver_config=solver_config,
    )
    run_config = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=1,
        n_sim_panels=6,
        master_seed=(30, 40),
    )

    panel_result = spec.simulate_panel_moments(
        beta_true,
        run_config,
        fold_in_seed(run_config.master_seed, "frictional_numerical", "ER", "panel"),
    )

    assert panel_result.panel_moments.shape == (6, 5)
    assert panel_result.average_moments.shape == (5,)
    assert np.all(np.isfinite(panel_result.panel_moments))
    assert np.all(np.isfinite(panel_result.average_moments))


@pytest.mark.slow
def test_frictional_numerical_pfi_solve_smm_smoke_returns_finite_results(frictional_numerical_env):
    beta_true = frictional_numerical_env.smm_initial_guess(mode="frictional_numerical")
    initial_guess = np.array([0.66, 0.16, 0.60, 0.12], dtype=np.float64)
    solver_config = _make_pfi_solver_config()
    spec = frictional_numerical_env.make_smm_spec(
        mode="frictional_numerical",
        initial_guess=initial_guess,
        solver_config=solver_config,
    )
    run_config = SMMRunConfig(
        n_firms=4,
        horizon=6,
        burn_in=2,
        n_sim_panels=6,
        global_method="Powell",
        local_method="least_squares",
        optimizer_maxiter=1,
        master_seed=(30, 40),
    )
    target = frictional_numerical_env.simulate_smm_target_moments(
        beta_true,
        run_config,
        seed=fold_in_seed(run_config.master_seed, "frictional_numerical", "PFI", "target"),
        mode="frictional_numerical",
        solver_config=solver_config,
    )

    result = solve_smm(
        spec=spec,
        target=target,
        config=run_config,
        simulation_seed=fold_in_seed(
            run_config.master_seed, "frictional_numerical", "PFI", "crn",
        ),
    )

    assert result.beta_hat.shape == (4,)
    assert np.all(np.isfinite(result.beta_hat))
    if result.j_test_valid:
        assert np.isfinite(result.j_statistic)
    else:
        assert np.isnan(result.j_statistic)
    assert result.stage2.panel_moments.shape[1] == 5
