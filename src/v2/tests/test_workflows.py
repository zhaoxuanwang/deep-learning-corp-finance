"""Tiny integration tests for notebook-level public workflows."""

from __future__ import annotations

import numpy as np

from src.v2.estimation import (
    GMMRunConfig,
    SMMMonteCarloConfig,
    SMMRunConfig,
    solve_gmm,
    validate_smm,
)
from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    BasicInvestmentSMMSolverConfig,
    EconomicParams as BasicEconomicParams,
    ShockParams as BasicShockParams,
)
from src.v2.environments.risky_debt import (
    EconomicParams as DebtEconomicParams,
    RiskyDebtEnv,
    ShockParams as DebtShockParams,
)
from src.v2.solvers import (
    GridConfig,
    PFIConfig,
    RiskyDebtSolverConfig,
    solve_risky_debt,
)


def _make_basic_env(*, cost_convex: float) -> BasicInvestmentEnv:
    return BasicInvestmentEnv(
        econ_params=BasicEconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.70,
            cost_convex=cost_convex,
            cost_fixed=0.0,
        ),
        shock_params=BasicShockParams(mu=0.0, rho=0.70, sigma=0.10),
    )


def _make_basic_pfi_solver_config() -> BasicInvestmentSMMSolverConfig:
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


def test_gmm_notebook_path_smoke_returns_finite_schema():
    env = _make_basic_env(cost_convex=0.20)
    solver_config = _make_basic_pfi_solver_config()
    panel = env.simulate_gmm_panel(
        seed=(31, 41),
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

    result = solve_gmm(
        spec,
        config=GMMRunConfig(optimizer_name="L-BFGS-B", optimizer_maxiter=1),
    )

    assert result.parameter_names == env.gmm_parameter_names()
    assert result.moment_names == env.gmm_moment_names()
    assert result.n_observations == panel.n_observations
    assert result.beta_hat.shape == (4,)
    assert result.stage2.moment_vector.shape == (6,)
    assert result.j_df == 2
    assert np.all(np.isfinite(result.beta_hat))


def test_stage_a_smm_notebook_validation_smoke_returns_summary():
    env = _make_basic_env(cost_convex=0.0)
    beta_true = env.smm_true_beta(mode="stage_a")
    spec = env.make_smm_spec(
        mode="stage_a",
        initial_guess=np.array([0.64, 0.58, 0.14], dtype=np.float64),
    )
    run_config = SMMRunConfig(
        n_firms=8,
        horizon=5,
        burn_in=1,
        n_sim_panels=5,
        global_method="Powell",
        optimizer_maxiter=1,
        master_seed=(21, 27),
    )

    result = validate_smm(
        spec,
        beta_true,
        run_config=run_config,
        mc_config=SMMMonteCarloConfig(n_replications=1),
    )

    assert len(result.replications) == 1
    assert result.summary.parameter_names == spec.parameter_names
    assert result.moment_names == spec.moment_names
    assert result.summary.mean_beta_hat.shape == beta_true.shape
    assert np.all(np.isfinite(result.summary.mean_beta_hat))
    assert np.all(np.isfinite(result.summary.rmse))


def test_risky_debt_public_solver_workflow_smoke_returns_contract():
    env = RiskyDebtEnv(
        econ_params=DebtEconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.70,
            cost_convex=0.05,
            tax=0.30,
            default_haircut=0.45,
            cost_inject_fixed=0.50,
            cost_inject_linear=0.10,
        ),
        shock_params=DebtShockParams(mu=0.0, rho=0.70, sigma=0.15),
        k_min_mult=0.10,
        k_max_mult=1.20,
        b_max_mult=1.50,
        b_min_mult=0.40,
    )
    config = RiskyDebtSolverConfig(
        n_k=3,
        n_b=3,
        n_z=3,
        n_z_solve=2,
        adaptive=False,
    )

    result = solve_risky_debt(env, config=config)

    expected_keys = {
        "value",
        "policy_action",
        "policy_endo",
        "r_tilde_grid",
        "default_mask",
        "funded_mask",
        "zero_profit_residual",
        "grids",
    }
    assert expected_keys.issubset(result)
    assert result["value"].shape == (3, 9)
    assert result["policy_action"].shape == (3, 9, 2)
    assert result["r_tilde_grid"].shape == (3, 3, 3)
    assert result["default_mask"].shape == (3, 3, 3)
    assert result["funded_mask"].shape == (3, 3, 3)
    assert result["zero_profit_residual"].shape == (3, 3, 3)
    assert np.all(np.isfinite(result["value"]))
    if np.any(result["funded_mask"]):
        residual = result["zero_profit_residual"][result["funded_mask"]]
        assert np.all(np.isfinite(residual))
