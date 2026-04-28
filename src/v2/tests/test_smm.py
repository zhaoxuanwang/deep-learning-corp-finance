"""Tests for the generic SMM core and risky-debt SMM integration."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.v2.data.rng import SeedSchedule, SeedScheduleConfig, VariableID
from src.v2.estimation import (
    SMMMonteCarloConfig,
    SMMPanelMoments,
    SMMRunConfig,
    SMMSpec,
    SMMSolveResult,
    SMMTargetMoments,
    solve_smm,
    validate_smm,
)
from src.v2.environments.risky_debt import (
    EconomicParams,
    RiskyDebtEnv,
    RiskyDebtSMMPanelData,
    ShockParams,
    _compute_smm_panel_moments,
    _simulate_smm_panel_data,
    _uniforms_to_standard_normal,
)
from src.v2.estimation.smm import (
    _panel_covariance,
    _panel_iv_first_diff_ar1,
    _panel_serial_correlation,
)
from src.v2.solvers import NestedVFIConfig, NestedVFIGridConfig, solve_nested_vfi
from src.v2.utils.seeding import fold_in_seed, make_seed_int


def _make_linear_toy_spec(
    initial_guess: np.ndarray | None = None,
) -> tuple[SMMSpec, callable]:
    parameter_names = ("beta_0", "beta_1")
    moment_names = ("m_1", "m_2", "m_3")
    bounds = ((-2.0, 2.0), (-2.0, 2.0))
    guess = (
        np.array([0.0, 0.0], dtype=np.float64)
        if initial_guess is None
        else np.asarray(initial_guess, dtype=np.float64)
    )

    def moment_map(beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=np.float64)
        return np.array(
            [
                beta[0] + 2.0 * beta[1],
                beta[0] - beta[1],
                0.5 * beta[0] + 1.5 * beta[1],
            ],
            dtype=np.float64,
        )

    def panel_noise(n_panels: int) -> np.ndarray:
        offsets = np.linspace(-0.03, 0.03, n_panels, dtype=np.float64)[:, None]
        directions = np.array([[1.0, -0.5, 0.25]], dtype=np.float64)
        return offsets * directions

    def simulate_panel_moments(
        beta,
        run_config: SMMRunConfig,
        seed: tuple[int, int],
    ) -> SMMPanelMoments:
        del seed
        panel_moments = moment_map(beta)[None, :] + panel_noise(run_config.n_sim_panels)
        return SMMPanelMoments(
            panel_moments=panel_moments,
            n_observations=(
                run_config.n_sim_panels * run_config.n_firms * run_config.horizon
            ),
        )

    def simulate_target_moments(
        beta,
        run_config: SMMRunConfig,
        seed: tuple[int, int],
    ) -> SMMTargetMoments:
        rng = np.random.default_rng(make_seed_int(seed, "toy_problem", "target_noise"))
        target_noise = rng.normal(scale=0.01, size=len(moment_names))
        return SMMTargetMoments(
            values=moment_map(beta) + target_noise,
            n_observations=run_config.n_firms * run_config.horizon,
        )

    return (
        SMMSpec(
            parameter_names=parameter_names,
            moment_names=moment_names,
            bounds=bounds,
            initial_guess=guess,
            simulate_panel_moments=simulate_panel_moments,
            simulate_target_moments=simulate_target_moments,
        ),
        moment_map,
    )


@pytest.fixture(scope="module")
def small_risky_debt_setup():
    env = RiskyDebtEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.7,
            cost_convex=0.05,
            tax=0.30,
            default_haircut=0.45,
            cost_inject_fixed=0.50,
            cost_inject_linear=0.10,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.7, sigma=0.15),
        k_min_mult=0.10,
        k_max_mult=1.20,
        b_max_mult=1.50,
        b_min_mult=0.40,
    )
    solver_config = NestedVFIConfig(
        grid=NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[3, 3]),
        max_iter_inner=200,
        tol_inner=1e-6,
        max_iter_outer=8,
        tol_outer_value=1e-3,
    )
    solved = solve_nested_vfi(env, config=solver_config)
    return env, solver_config, solved


def test_solve_smm_recovers_linear_toy_problem_with_dual_annealing():
    spec, moment_map = _make_linear_toy_spec()
    run_config = SMMRunConfig(
        n_firms=8,
        horizon=4,
        burn_in=0,
        n_sim_panels=4,
        global_method="dual_annealing",
        optimizer_maxiter=30,
    )
    beta_true = np.array([0.8, -0.3], dtype=np.float64)
    target = SMMTargetMoments(
        values=moment_map(beta_true),
        n_observations=run_config.n_firms * run_config.horizon,
    )
    simulation_seed = fold_in_seed((20, 26), "toy", "shared")

    result = solve_smm(
        spec=spec,
        target=target,
        config=run_config,
        simulation_seed=simulation_seed,
    )

    np.testing.assert_allclose(result.beta_hat, beta_true, atol=1e-3, rtol=1e-3)
    assert result.stage1.panel_moments.shape == (4, 3)
    assert result.stage2.panel_moments.shape == (4, 3)
    assert result.j_df == 1
    assert result.standard_errors.shape == (2,)


def test_solve_smm_is_reproducible_under_fixed_seed():
    spec, moment_map = _make_linear_toy_spec()
    run_config = SMMRunConfig(
        n_firms=8,
        horizon=4,
        burn_in=0,
        n_sim_panels=4,
        global_method="dual_annealing",
        optimizer_maxiter=20,
        master_seed=(11, 29),
    )
    beta_true = np.array([0.6, 0.2], dtype=np.float64)
    target = SMMTargetMoments(
        values=moment_map(beta_true),
        n_observations=run_config.n_firms * run_config.horizon,
    )

    result_a = solve_smm(spec=spec, target=target, config=run_config)
    result_b = solve_smm(spec=spec, target=target, config=run_config)

    np.testing.assert_allclose(result_a.beta_hat, result_b.beta_hat, atol=1e-10)
    np.testing.assert_allclose(
        result_a.stage2.trace["objective"],
        result_b.stage2.trace["objective"],
        atol=1e-12,
    )


def test_solve_smm_supports_local_optimizer_dispatch():
    spec, moment_map = _make_linear_toy_spec()
    run_config = SMMRunConfig(
        n_firms=8,
        horizon=4,
        burn_in=0,
        n_sim_panels=4,
        global_method="Powell",
        optimizer_maxiter=50,
    )
    beta_true = np.array([0.4, -0.1], dtype=np.float64)
    target = SMMTargetMoments(
        values=moment_map(beta_true),
        n_observations=run_config.n_firms * run_config.horizon,
    )

    result = solve_smm(spec=spec, target=target, config=run_config)

    np.testing.assert_allclose(result.beta_hat, beta_true, atol=2e-2, rtol=2e-2)


def test_validate_smm_returns_monte_carlo_summary():
    spec, _ = _make_linear_toy_spec(initial_guess=np.array([0.2, 0.1], dtype=np.float64))
    run_config = SMMRunConfig(
        n_firms=8,
        horizon=4,
        burn_in=0,
        n_sim_panels=4,
        global_method="Powell",
        optimizer_maxiter=10,
    )
    mc_config = SMMMonteCarloConfig(n_replications=2)
    beta_true = np.array([0.6, 0.2], dtype=np.float64)

    result = validate_smm(spec, beta_true, run_config=run_config, mc_config=mc_config)

    assert len(result.replications) == 2
    assert result.summary.bias.shape == (2,)
    assert result.summary.rmse.shape == (2,)
    assert result.summary.coverage_95.shape == (2,)
    # j_test_size is nan when all replications have j_test_valid=False (e.g.,
    # small n_sim_panels relative to n_moments triggers the guardrail).
    assert np.isnan(result.summary.j_test_size) or result.summary.j_test_size >= 0.0


def test_risky_debt_make_smm_spec_exposes_env_authority(small_risky_debt_setup):
    env, solver_config, _ = small_risky_debt_setup
    custom_guess = env.smm_initial_guess().copy()
    custom_guess[0] = 0.68

    spec = env.make_smm_spec(
        solver_config=solver_config,
        initial_guess=custom_guess,
    )

    assert spec.parameter_names == env.smm_parameter_names()
    assert spec.moment_names == env.smm_moment_names()
    assert spec.bounds == env.smm_default_bounds()
    np.testing.assert_allclose(spec.initial_guess, custom_guess, atol=1e-12)


def test_risky_debt_make_smm_spec_validates_disabled_moments(small_risky_debt_setup):
    env, solver_config, _ = small_risky_debt_setup

    spec = env.make_smm_spec(
        solver_config=solver_config,
        estimated_params=("eta0",),
        disabled_moments=("conditional_issuance_size", "autocorr_equity_issuance"),
    )

    assert "conditional_issuance_size" not in spec.moment_names
    assert "autocorr_equity_issuance" not in spec.moment_names
    assert "frequency_equity_issuance" in spec.moment_names

    with pytest.raises(ValueError, match="Unknown moment name"):
        env.make_smm_spec(
            solver_config=solver_config,
            disabled_moments=("not_a_moment",),
        )

    with pytest.raises(ValueError, match="Underidentified"):
        env.make_smm_spec(
            solver_config=solver_config,
            estimated_params=("alpha",),
            disabled_moments=("mean_investment_assets",),
        )


def test_risky_debt_panel_simulation_respects_seed_and_burn_in(small_risky_debt_setup):
    env, _, solved = small_risky_debt_setup
    seed = fold_in_seed((20, 26), "risky_debt", "panel")
    run_zero = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=0,
        n_sim_panels=1,
    )
    run_one = SMMRunConfig(
        n_firms=3,
        horizon=3,
        burn_in=1,
        n_sim_panels=1,
    )

    panel_zero = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_zero,
        n_panels=1,
        seed=seed,
    )
    panel_one = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_one,
        n_panels=1,
        seed=seed,
    )
    panel_one_repeat = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_one,
        n_panels=1,
        seed=seed,
    )
    other_seed = fold_in_seed((20, 26), "risky_debt", "panel_other")
    panel_other = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_one,
        n_panels=1,
        seed=other_seed,
    )

    np.testing.assert_allclose(panel_one.k[:, :, 0], panel_zero.k[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(panel_one.b[:, :, 0], panel_zero.b[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(panel_one.z[:, :, 0], panel_zero.z[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(panel_one.cash_flow, panel_one_repeat.cash_flow, atol=1e-8)
    assert not np.allclose(panel_one.cash_flow, panel_other.cash_flow)


def test_public_risky_debt_panel_api_returns_structured_data(small_risky_debt_setup):
    env, solver_config, solved = small_risky_debt_setup
    beta_true = env.smm_initial_guess()
    run_config = SMMRunConfig(
        n_firms=2,
        horizon=3,
        burn_in=1,
        n_sim_panels=2,
    )
    seed = fold_in_seed((20, 26), "risky_debt", "public_panel")

    panel_data = env.simulate_smm_panel_data(
        beta=beta_true,
        run_config=run_config,
        seed=seed,
        solver_config=solver_config,
        solved_result=solved,
    )

    assert isinstance(panel_data, RiskyDebtSMMPanelData)
    assert panel_data.n_panels == 2
    assert panel_data.n_firms == 2
    assert panel_data.horizon == 3
    assert panel_data.n_observations == 12

    payload = panel_data.to_dict()
    assert sorted(payload.keys()) == [
        "b",
        "b_next",
        "cash_flow",
        "debt_discount",
        "k",
        "k_next",
        "metadata",
        "n_observations",
        "value",
        "z",
    ]
    assert payload["k"].shape == (2, 2, 3)
    assert payload["cash_flow"].shape == (2, 2, 3)
    assert payload["metadata"]["seed"] == seed
    assert payload["metadata"]["solver_reused"] is True
    assert payload["metadata"]["n_panels"] == 2


def test_public_risky_debt_panel_api_respects_seed_and_burn_in(small_risky_debt_setup):
    env, solver_config, solved = small_risky_debt_setup
    beta_true = env.smm_initial_guess()
    seed = fold_in_seed((20, 26), "risky_debt", "public_panel_alignment")
    alt_seed = fold_in_seed((20, 26), "risky_debt", "public_panel_alignment_alt")
    run_base = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=0,
        n_sim_panels=1,
    )
    run_shifted = SMMRunConfig(
        n_firms=3,
        horizon=3,
        burn_in=1,
        n_sim_panels=1,
    )

    panel_a = env.simulate_smm_panel_data(
        beta=beta_true,
        run_config=run_shifted,
        seed=seed,
        solver_config=solver_config,
        solved_result=solved,
    )
    panel_b = env.simulate_smm_panel_data(
        beta=beta_true,
        run_config=run_shifted,
        seed=seed,
        solver_config=solver_config,
        solved_result=solved,
    )
    panel_base = env.simulate_smm_panel_data(
        beta=beta_true,
        run_config=run_base,
        seed=seed,
        solver_config=solver_config,
        solved_result=solved,
    )
    panel_alt = env.simulate_smm_panel_data(
        beta=beta_true,
        run_config=run_shifted,
        seed=alt_seed,
        solver_config=solver_config,
        solved_result=solved,
    )

    np.testing.assert_allclose(panel_a.cash_flow, panel_b.cash_flow, atol=1e-8)
    np.testing.assert_allclose(panel_a.k[:, :, 0], panel_base.k[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(panel_a.b[:, :, 0], panel_base.b[:, :, 1], atol=1e-8)
    np.testing.assert_allclose(panel_a.z[:, :, 0], panel_base.z[:, :, 1], atol=1e-8)
    assert not np.allclose(panel_a.cash_flow, panel_alt.cash_flow)


def test_risky_debt_on_grid_simulation_stays_on_grid(small_risky_debt_setup):
    """Verify that the discrete-Markov simulation keeps all states on-grid."""
    env, _, solved = small_risky_debt_setup
    run_config = SMMRunConfig(
        n_firms=3,
        horizon=3,
        burn_in=0,
        n_sim_panels=1,
    )
    seed = fold_in_seed((20, 26), "risky_debt", "on_grid")
    panel_data = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_config,
        n_panels=1,
        seed=seed,
    )

    z_grid = set(solved["grids"]["exo_grids_1d"][0].tolist())
    k_grid = set(solved["grids"]["endo_grids_1d"][0].tolist())
    b_grid = set(solved["grids"]["endo_grids_1d"][1].tolist())

    for val in panel_data.z.ravel():
        assert float(val) in z_grid, f"z={val} not on grid"
    for val in panel_data.k_next.ravel():
        assert float(val) in k_grid, f"k_next={val} not on grid"
    for val in panel_data.b_next.ravel():
        assert float(val) in b_grid, f"b_next={val} not on grid"


def test_risky_debt_moments_use_cash_flow_and_ratio_construction(small_risky_debt_setup):
    env, _, solved = small_risky_debt_setup
    seed = fold_in_seed((20, 26), "risky_debt", "moments")
    run_config = SMMRunConfig(
        n_firms=3,
        horizon=4,
        burn_in=1,
        n_sim_panels=1,
    )
    panel_data = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_config,
        n_panels=1,
        seed=seed,
    )
    panel_moments, diagnostics = _compute_smm_panel_moments(env, panel_data)

    safe_k = np.maximum(panel_data.k[0], 1e-12)
    investment_ratio = (
        panel_data.k_next[0] - (1.0 - env.econ.depreciation_rate) * panel_data.k[0]
    ) / safe_k
    issuance_ratio = np.maximum(0.0, -panel_data.cash_flow[0]) / safe_k
    leverage = (
        panel_data.b_next[0] * panel_data.debt_discount[0]
        / np.maximum(
            panel_data.value[0] + panel_data.b_next[0] * panel_data.debt_discount[0],
            1e-8,
        )
    )
    log_income_ratio = np.log(np.maximum(
        panel_data.z[0] * np.power(safe_k, env.econ.production_elasticity - 1.0),
        1e-12,
    ))
    manual_beta, manual_sigma = _panel_iv_first_diff_ar1(log_income_ratio)

    first_step_discount = panel_data.debt_discount[0, :, 0]
    r_tilde = np.where(first_step_discount > 1e-12, 1.0 / first_step_discount - 1.0, np.inf)
    recomputed_cash_flow = env.cash_flow_from_choice(
        panel_data.k[0, :, 0].astype(np.float32),
        panel_data.b[0, :, 0].astype(np.float32),
        panel_data.z[0, :, 0].astype(np.float32),
        panel_data.k_next[0, :, 0].astype(np.float32),
        panel_data.b_next[0, :, 0].astype(np.float32),
        r_tilde.astype(np.float32),
    ).numpy()
    np.testing.assert_allclose(recomputed_cash_flow, panel_data.cash_flow[0, :, 0], atol=1e-4)

    # Conditional issuance size
    issuing_mask = panel_data.cash_flow[0] < 0.0
    cond_iss = float(np.mean(issuance_ratio[issuing_mask])) if np.any(issuing_mask) else 0.0

    # Autocorr of issuance
    autocorr_iss = _panel_serial_correlation(issuance_ratio)

    # Cross-corr: leverage_t -> issuance_{t+1}
    lev_lead = leverage[:, :-1].reshape(-1)
    iss_lag = issuance_ratio[:, 1:].reshape(-1)
    lev_c = lev_lead - np.mean(lev_lead)
    iss_c = iss_lag - np.mean(iss_lag)
    denom_cc = np.sqrt(np.mean(lev_c ** 2) * np.mean(iss_c ** 2))
    crosscorr = float(np.mean(lev_c * iss_c) / denom_cc) if denom_cc > 1e-12 else 0.0

    # Default frequency: fraction of firm-periods with V <= 0
    default_freq = float(np.mean(panel_data.value[0] <= 0.0))

    # H&W 2007 leverage moments (book leverage = b'/k; pecking order = Cov)
    book_lev_panel = panel_data.b_next[0] / safe_k
    book_lev = float(np.mean(book_lev_panel))
    cov_lev_inv = _panel_covariance(book_lev_panel, investment_ratio)

    # H&W 2007 issuance moments (correlation, not covariance — avoids
    # pathological Omega conditioning from the tiny Cov population variance)
    freq_eq_iss = float(np.mean(panel_data.cash_flow[0] < 0.0))
    iss_c = issuance_ratio - np.mean(issuance_ratio)
    inv_c = investment_ratio - np.mean(investment_ratio)
    denom_iss_inv = np.sqrt(np.mean(iss_c ** 2) * np.mean(inv_c ** 2))
    corr_iss_inv = (
        float(np.mean(iss_c * inv_c) / denom_iss_inv)
        if denom_iss_inv > 1e-12 else 0.0
    )

    manual_moments = np.array(
        [
            np.mean(issuance_ratio),
            cond_iss,
            autocorr_iss,
            crosscorr,
            book_lev,
            cov_lev_inv,
            np.mean(investment_ratio),
            _panel_serial_correlation(investment_ratio),
            np.var(investment_ratio),
            manual_beta,
            manual_sigma,
            default_freq,
            freq_eq_iss,
            corr_iss_inv,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(panel_moments[0], manual_moments, atol=1e-6, rtol=1e-6)
    assert diagnostics["payout_mean"].shape == (1,)
    assert diagnostics["payout_var"].shape == (1,)


def test_public_moment_wrapper_matches_private_formula(small_risky_debt_setup):
    env, _, solved = small_risky_debt_setup
    seed = fold_in_seed((20, 26), "risky_debt", "public_moments")
    run_config = SMMRunConfig(
        n_firms=2,
        horizon=3,
        burn_in=1,
        n_sim_panels=2,
    )
    panel_data = _simulate_smm_panel_data(
        env,
        solved,
        run_config=run_config,
        n_panels=2,
        seed=seed,
    )
    private_panel_moments, private_diagnostics = _compute_smm_panel_moments(env, panel_data)
    wrapped = env.compute_smm_panel_moments(panel_data)

    assert wrapped["moment_names"] == env.smm_moment_names()
    np.testing.assert_allclose(wrapped["panel_moments"], private_panel_moments, atol=1e-8)
    np.testing.assert_allclose(
        wrapped["average_moments"],
        np.mean(private_panel_moments, axis=0),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        wrapped["diagnostics"]["payout_mean"],
        private_diagnostics["payout_mean"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        wrapped["diagnostics"]["payout_var"],
        private_diagnostics["payout_var"],
        atol=1e-8,
    )


def test_risky_debt_smm_smoke_validation_runs(small_risky_debt_setup):
    env, solver_config, _ = small_risky_debt_setup
    beta_true = env.smm_initial_guess()
    spec = env.make_smm_spec(solver_config=solver_config, initial_guess=beta_true)
    spec = SMMSpec(
        parameter_names=spec.parameter_names,
        moment_names=spec.moment_names,
        bounds=tuple(
            (
                max(lower, value - 1e-3),
                min(upper, value + 1e-3),
            )
            for (lower, upper), value in zip(spec.bounds, beta_true)
        ),
        initial_guess=beta_true.copy(),
        simulate_panel_moments=spec.simulate_panel_moments,
        simulate_target_moments=spec.simulate_target_moments,
    )

    run_config = SMMRunConfig(
        n_firms=2,
        horizon=3,
        burn_in=1,
        n_sim_panels=15,
        global_method="Powell",
        optimizer_maxiter=1,
    )
    mc_config = SMMMonteCarloConfig(n_replications=1)

    result = validate_smm(spec, beta_true, run_config=run_config, mc_config=mc_config)

    assert len(result.replications) == 1
    assert result.summary.bias.shape == (7,)
    solve_result = result.replications[0]
    assert isinstance(solve_result, SMMSolveResult)
    assert solve_result.beta_hat.shape == (7,)
