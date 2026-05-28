from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v2.environments.nikolov import NikolovGridConfig, NikolovParams
from src.v2.estimation import nikolov_ii as nii
from src.v2.estimation.nikolov_ii import (
    NikolovIIConfig,
    build_nikolov_grid_config,
    build_nikolov_params,
    compute_h_data,
    make_nikolov_ii_spec,
    pack_beta,
    run_nikolov_ii,
)
from src.v2.estimation.smm import (
    SMMPanelMoments,
    SMMRunConfig,
    SMMSpec,
    SMMTargetMoments,
    solve_smm,
)


def _fake_panel(n_firms: int = 60, horizon: int = 8, seed: int = 0) -> pd.DataFrame:
    """A synthetic observable panel with the columns the auxiliary fit needs."""
    rng = np.random.default_rng(seed)
    n = n_firms * horizon
    log_k = rng.normal(8.0, 1.0, n)
    profitability = rng.normal(0.04, 0.10, n)
    leverage = rng.normal(0.0, 0.30, n)
    return pd.DataFrame(
        {
            "log_k": log_k,
            "profitability": profitability,
            "leverage": leverage,
            "investment_rate": 0.03 + 0.01 * log_k - 0.05 * leverage
            + rng.normal(0, 0.02, n),
            "future_leverage": 0.9 * leverage + rng.normal(0, 0.05, n),
            "payout_rate": 0.4 * profitability + rng.normal(0, 0.01, n),
            "default": False,
        }
    )


@pytest.fixture(scope="module")
def cfg() -> NikolovIIConfig:
    return NikolovIIConfig()


@pytest.fixture(scope="module")
def anchor(cfg) -> np.ndarray:
    return pack_beta(
        dict(alpha=0.75, f=0.70, delta=0.15, psi=0.15, eta_bar=0.30, xi=0.60),
        dict(rho=0.80, sigma=0.30),
        cfg,
    )


def test_beta_to_params_and_grid_are_deterministic(cfg, anchor):
    p1 = build_nikolov_params(anchor, cfg)
    p2 = build_nikolov_params(anchor, cfg)
    g1 = build_nikolov_grid_config(anchor, cfg)
    g2 = build_nikolov_grid_config(anchor, cfg)
    assert p1 == p2
    assert g1 == g2
    assert isinstance(g1, NikolovGridConfig) and isinstance(p1, NikolovParams)
    # grid bounds are zero-based debt and positive capital/z, as the solver needs
    assert g1.b_bounds[0] == 0.0 and g1.b_bounds[1] > 0.0
    assert g1.k_bounds[0] > 0.0 and g1.k_bounds[1] > g1.k_bounds[0]


def test_pack_beta_reads_ar1_and_param_kwargs(cfg, anchor):
    names = cfg.param_names
    assert anchor[names.index("z_rho")] == pytest.approx(0.80)
    assert anchor[names.index("z_sigma")] == pytest.approx(0.30)
    assert anchor[names.index("alpha")] == pytest.approx(0.75)
    assert anchor[names.index("xi")] == pytest.approx(0.60)


def test_beta_components_are_clipped_to_bounds(cfg):
    names = cfg.param_names
    out_of_bounds = np.array([b[1] + 10.0 for b in cfg.bounds])
    params = build_nikolov_params(out_of_bounds, cfg)
    # alpha clipped to its upper bound, not the absurd input
    assert params.alpha == pytest.approx(cfg.bounds[names.index("alpha")][1])


def test_h_data_moment_count_and_ordering(cfg):
    panel = _fake_panel()
    h_data, names, n_obs = compute_h_data(panel, cfg)
    # 3 outcomes x (10 features - 1 intercept dropped) = 27
    assert h_data.size == 27
    assert len(names) == 27
    assert n_obs == len(panel)
    # outcome-outer ordering: first block is investment_rate, no intercept term
    assert names[0].startswith("investment_rate::")
    assert all("::1" not in nm for nm in names)
    assert names[9].startswith("future_leverage::")
    assert names[18].startswith("payout_rate::")


def test_sim_stacking_matches_h_data_ordering(cfg):
    """The simulated coefficient stack must use the same order as h_data."""
    panel = _fake_panel(seed=1)
    from src.v2.estimation.empirical_policy import fit_empirical_policy

    fit = fit_empirical_policy(panel, cfg.x_cols, cfg.y_cols, nii._aux_config(cfg))
    stacked = nii._stack_coefficients(fit, cfg)
    h_data, _, _ = compute_h_data(panel, cfg)
    np.testing.assert_allclose(stacked, h_data)


def test_sentinel_returned_on_solve_failure(cfg, monkeypatch):
    panel = _fake_panel(seed=3)
    spec = make_nikolov_ii_spec(cfg, panel)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced LP failure")

    monkeypatch.setattr(nii, "solve_nikolov_to", _boom)
    run = SMMRunConfig(n_firms=40, horizon=6, burn_in=3, n_sim_panels=30)
    seed = (1, 2)
    panel_moments = spec.simulate_panel_moments(spec.initial_guess, run, seed)
    assert isinstance(panel_moments, SMMPanelMoments)
    assert panel_moments.panel_moments.shape == (30, 27)
    # every row is penalised far from the target
    assert np.all(panel_moments.panel_moments > 100.0)


def test_solve_smm_point_estimate_skips_inference():
    """two_step=False + compute_standard_errors=False -> stage-1 only, no SEs."""
    target = SMMTargetMoments(values=np.array([1.0, 2.0]), n_observations=100)

    def simulate(beta, run_config, seed):
        beta = np.asarray(beta, dtype=np.float64)
        # deterministic moments = beta, replicated across panels with jitter
        rng = np.random.default_rng(0)
        rows = beta[None, :] + 1e-3 * rng.standard_normal((run_config.n_sim_panels, 2))
        return SMMPanelMoments(panel_moments=rows, n_observations=100)

    spec = SMMSpec(
        parameter_names=("a", "b"),
        moment_names=("m0", "m1"),
        bounds=((0.0, 3.0), (0.0, 3.0)),
        initial_guess=np.array([1.5, 1.5]),
        simulate_panel_moments=simulate,
    )
    run = SMMRunConfig(n_firms=10, horizon=4, burn_in=2, n_sim_panels=2,
                       global_method="Nelder-Mead", optimizer_maxiter=20)
    res = solve_smm(spec, target, run, two_step=False, compute_standard_errors=False)
    assert res.standard_errors is None
    assert res.jacobian is None
    assert res.omega_hat is None
    assert res.stage2 is res.stage1
    # one-step recovers the target (moments are identity in beta)
    np.testing.assert_allclose(res.beta_hat, [1.0, 2.0], atol=5e-2)


def test_solve_smm_one_step_with_standard_errors_uses_identity_weight():
    """two_step=False + compute_standard_errors=True: one-step sandwich SE.

    The estimator uses W=I (stage 1); inference must use the SAME W=I in the
    bread, with Omega_hat only in the meat.  J-test is unavailable (needs the
    efficient weight).  Requires n_sim_panels > n_moments to form Omega_hat.
    """
    target = SMMTargetMoments(values=np.array([1.0, 2.0]), n_observations=100)

    def simulate(beta, run_config, seed):
        beta = np.asarray(beta, dtype=np.float64)
        rng = np.random.default_rng(int(abs(beta.sum()) * 1e6) % (2**31))
        rows = beta[None, :] + 1e-2 * rng.standard_normal((run_config.n_sim_panels, 2))
        return SMMPanelMoments(panel_moments=rows, n_observations=100)

    spec = SMMSpec(
        parameter_names=("a", "b"),
        moment_names=("m0", "m1"),
        bounds=((0.0, 3.0), (0.0, 3.0)),
        initial_guess=np.array([1.5, 1.5]),
        simulate_panel_moments=simulate,
    )
    run = SMMRunConfig(n_firms=10, horizon=4, burn_in=2, n_sim_panels=8,
                       global_method="Nelder-Mead", optimizer_maxiter=20)
    res = solve_smm(spec, target, run, two_step=False, compute_standard_errors=True)
    assert res.stage2 is res.stage1                      # one-step headline
    assert res.omega_hat is not None                     # built for the meat
    assert res.standard_errors is not None
    assert res.jacobian is not None
    assert not res.j_test_valid                          # no efficient weight
    np.testing.assert_allclose(res.beta_hat, [1.0, 2.0], atol=8e-2)


@pytest.mark.slow
def test_run_nikolov_ii_end_to_end_smoke():
    """Tiny feasible grid; confirm the full II loop returns a valid result."""
    cfg = NikolovIIConfig(
        param_names=("alpha", "eta_bar"),
        bounds=((0.65, 0.85), (0.15, 0.45)),
        n_k=8,
        n_b=6,
        n_z=3,
    )
    panel = _fake_panel(seed=4)
    run = SMMRunConfig(
        n_firms=120,
        horizon=10,
        burn_in=5,
        n_sim_panels=3,
        master_seed=(5, 9),
        global_method="Nelder-Mead",
        local_method="Powell",
        optimizer_maxiter=3,
    )
    res = run_nikolov_ii(
        panel, cfg, run, two_step=False, compute_standard_errors=False
    )
    assert len(res.moment_names) == 27
    assert res.beta_hat.shape == (2,)
    assert np.all(np.isfinite(res.beta_hat))
    lo = np.array([b[0] for b in cfg.bounds])
    hi = np.array([b[1] for b in cfg.bounds])
    assert np.all(res.beta_hat >= lo - 1e-9) and np.all(res.beta_hat <= hi + 1e-9)
    assert res.standard_errors is None
