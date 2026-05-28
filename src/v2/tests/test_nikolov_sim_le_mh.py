from __future__ import annotations

import numpy as np
import pytest

from src.v2.environments.nikolov import NikolovGridConfig, NikolovParams
from src.v2.simulation.nikolov_le import (
    NikolovLESimulationConfig,
    construct_le_observables,
    simulate_nikolov_le_panel,
)
from src.v2.simulation.nikolov_mh import (
    NikolovMHSimulationConfig,
    construct_mh_observables,
    simulate_nikolov_mh_panel,
)
from src.v2.solvers.nikolov_le import solve_nikolov_le
from src.v2.solvers.nikolov_mh import solve_nikolov_mh

OBS_COLS = ["log_k", "profitability", "leverage",
            "investment_rate", "future_leverage", "payout_rate"]


@pytest.fixture(scope="module")
def tiny_transition():
    return np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)


@pytest.fixture(scope="module")
def tiny_config(tiny_transition):
    return NikolovGridConfig(
        k_bounds=(1.0, 1.2),
        b_bounds=(0.0, 0.5),
        v_bounds=(0.0, 0.5),
        z_bounds=(1.0, 1.1),
        p_bounds=(0.0, 0.02),
        d_bounds=(0.0, 0.02),
        n_k=2, n_b=2, n_v=2, n_z=2, n_p=2, n_d=2,
        z_transition=tiny_transition,
    )


@pytest.fixture(scope="module")
def tiny_params():
    return NikolovParams(
        tau=0.2, alpha=0.5, f=0.0, delta=0.1, psi=0.0, r=0.04,
        kappa=0.5, eta_bar=0.0, xi=0.5, theta=1.0, lambda_diversion=0.0,
    )


@pytest.fixture(scope="module")
def le_result(tiny_config, tiny_params):
    return solve_nikolov_le(tiny_config, tiny_params)


@pytest.fixture(scope="module")
def mh_result(tiny_config, tiny_params):
    return solve_nikolov_mh(tiny_config, tiny_params)


# --------------------------------------------------------------------------- LE


def test_le_panel_shape_and_observables(le_result, tiny_params):
    cfg = NikolovLESimulationConfig(n_firms=40, horizon=6, burn_in=3, seed=11)
    panel = simulate_nikolov_le_panel(le_result, tiny_params, cfg)
    assert len(panel) == cfg.n_firms * cfg.horizon
    obs = construct_le_observables(panel)
    for col in OBS_COLS:
        assert col in obs.columns
        assert np.isfinite(obs[col].to_numpy(dtype=np.float64)).all()


def test_le_is_deterministic(le_result, tiny_params):
    cfg = NikolovLESimulationConfig(n_firms=30, horizon=5, burn_in=2, seed=7)
    a = simulate_nikolov_le_panel(le_result, tiny_params, cfg)
    b = simulate_nikolov_le_panel(le_result, tiny_params, cfg)
    assert a.equals(b)


def test_le_leverage_definitions(le_result, tiny_params):
    cfg = NikolovLESimulationConfig(n_firms=30, horizon=5, burn_in=2, seed=3)
    obs = construct_le_observables(
        simulate_nikolov_le_panel(le_result, tiny_params, cfg)
    )
    np.testing.assert_allclose(obs["leverage"], obs["b"] / obs["k"])
    np.testing.assert_allclose(obs["future_leverage"], obs["b_next"] / obs["k_next"])
    # limited liability => distribution to equity is non-negative on the grid
    assert (obs["dividend_next"] >= -1e-9).all()


def test_le_contingent_gather_matches_policy(le_result, tiny_params):
    cfg = NikolovLESimulationConfig(n_firms=50, horizon=6, burn_in=2, seed=5)
    panel = simulate_nikolov_le_panel(le_result, tiny_params, cfg)
    gathered = le_result["policy"]["b_next_idx"][
        panel["k_idx"], panel["b_idx"], panel["z_idx"],
        panel["z_next_idx"], panel["eta_next_idx"],
    ]
    np.testing.assert_array_equal(gathered, panel["b_next_idx"].to_numpy())


# --------------------------------------------------------------------------- MH


def test_mh_panel_shape_and_observables(mh_result, tiny_params):
    cfg = NikolovMHSimulationConfig(n_firms=40, horizon=6, burn_in=3, seed=11)
    panel = simulate_nikolov_mh_panel(mh_result, tiny_params, cfg)
    assert len(panel) == cfg.n_firms * cfg.horizon
    obs = construct_mh_observables(panel)
    for col in OBS_COLS:
        assert col in obs.columns
        assert np.isfinite(obs[col].to_numpy(dtype=np.float64)).all()


def test_mh_is_deterministic(mh_result, tiny_params):
    cfg = NikolovMHSimulationConfig(n_firms=30, horizon=5, burn_in=2, seed=7)
    a = simulate_nikolov_mh_panel(mh_result, tiny_params, cfg)
    b = simulate_nikolov_mh_panel(mh_result, tiny_params, cfg)
    assert a.equals(b)


def test_mh_implied_debt_leverage(mh_result, tiny_params):
    cfg = NikolovMHSimulationConfig(n_firms=30, horizon=5, burn_in=2, seed=3)
    obs = construct_mh_observables(
        simulate_nikolov_mh_panel(mh_result, tiny_params, cfg)
    )
    np.testing.assert_allclose(obs["leverage"], (obs["W"] - obs["V"]) / obs["k"])
    np.testing.assert_allclose(
        obs["future_leverage"], (obs["W_next"] - obs["V_next"]) / obs["k_next"]
    )
    np.testing.assert_allclose(obs["payout_rate"], obs["dividend_next"] / obs["k_next"])


def test_mh_contingent_gather_matches_policy(mh_result, tiny_params):
    cfg = NikolovMHSimulationConfig(n_firms=50, horizon=6, burn_in=2, seed=5)
    panel = simulate_nikolov_mh_panel(mh_result, tiny_params, cfg)
    gathered = mh_result["policy"]["V_next_idx"][
        panel["k_idx"], panel["v_idx"], panel["z_idx"],
        panel["z_next_idx"], panel["eta_next_idx"],
    ]
    np.testing.assert_array_equal(gathered, panel["v_next_idx"].to_numpy())
