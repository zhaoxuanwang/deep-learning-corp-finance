from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v2.environments.nikolov import NikolovGridConfig, NikolovParams
from src.v2.estimation.empirical_policy import (
    EmpiricalPolicyConfig,
    fit_empirical_policy,
    predict_policy_slices,
)
from src.v2.simulation.nikolov_to import (
    NikolovTOSimulationConfig,
    construct_to_observables,
    simulate_nikolov_to_panel,
)
from src.v2.solvers.nikolov_to import solve_nikolov_to


@pytest.fixture(scope="module")
def pipeline_params() -> NikolovParams:
    return NikolovParams(
        tau=0.2,
        alpha=0.5,
        f=0.0,
        delta=0.1,
        psi=0.02,
        r=0.04,
        kappa=0.5,
        eta_bar=0.005,
        xi=0.5,
        theta=1.0,
        lambda_diversion=0.02,
    )


@pytest.fixture(scope="module")
def pipeline_config() -> NikolovGridConfig:
    return NikolovGridConfig(
        k_bounds=(1.0, 1.4),
        b_bounds=(0.0, 0.4),
        v_bounds=(0.0, 0.5),
        z_bounds=(0.95, 1.10),
        p_bounds=(0.0, 0.02),
        d_bounds=(0.0, 0.02),
        n_k=4,
        n_b=3,
        n_v=2,
        n_z=2,
        n_p=2,
        n_d=2,
        z_transition=np.array([[0.85, 0.15], [0.25, 0.75]], dtype=np.float64),
    )


@pytest.fixture(scope="module")
def solved_to(pipeline_config, pipeline_params):
    return solve_nikolov_to(pipeline_config, pipeline_params)


def test_to_simulation_is_deterministic(solved_to, pipeline_params):
    config = NikolovTOSimulationConfig(
        n_firms=8,
        horizon=5,
        burn_in=3,
        seed=1234,
    )
    panel_a = simulate_nikolov_to_panel(solved_to, pipeline_params, config)
    panel_b = simulate_nikolov_to_panel(solved_to, pipeline_params, config)

    pd.testing.assert_frame_equal(panel_a, panel_b)


def test_to_simulation_indices_remain_on_grid(solved_to, pipeline_params):
    sim_config = NikolovTOSimulationConfig(
        n_firms=10,
        horizon=6,
        burn_in=2,
        seed=4321,
    )
    panel = simulate_nikolov_to_panel(solved_to, pipeline_params, sim_config)
    grids = solved_to["grids"]

    assert panel["k_idx"].between(0, len(grids["k"]) - 1).all()
    assert panel["b_idx"].between(0, len(grids["b"]) - 1).all()
    assert panel["z_idx"].between(0, len(grids["z"]) - 1).all()
    assert panel["k_next_idx"].between(0, len(grids["k"]) - 1).all()
    assert panel["b_next_idx"].between(0, len(grids["b"]) - 1).all()
    assert panel["z_next_idx"].between(0, len(grids["z"]) - 1).all()


def test_to_observables_are_finite_after_default_filter(solved_to, pipeline_params):
    sim_config = NikolovTOSimulationConfig(
        n_firms=12,
        horizon=6,
        burn_in=2,
        seed=2222,
    )
    panel = simulate_nikolov_to_panel(solved_to, pipeline_params, sim_config)
    obs = construct_to_observables(panel)
    columns = [
        "log_k",
        "profitability",
        "leverage",
        "investment_rate",
        "future_leverage",
        "payout_rate",
    ]

    assert set(columns).issubset(obs.columns)
    filtered = obs.loc[~obs["default"].astype(bool), columns]
    assert np.isfinite(filtered.to_numpy(dtype=np.float64)).all()


def test_empirical_policy_fit_excludes_defaults_and_predicts_slices(
    solved_to,
    pipeline_params,
):
    sim_config = NikolovTOSimulationConfig(
        n_firms=30,
        horizon=8,
        burn_in=2,
        seed=3333,
    )
    panel = simulate_nikolov_to_panel(solved_to, pipeline_params, sim_config)
    obs = construct_to_observables(panel)
    obs.loc[obs.index[0], "default"] = True
    obs.loc[obs.index[0], "investment_rate"] = np.nan

    x_cols = ["log_k", "profitability", "leverage"]
    y_cols = ["investment_rate", "future_leverage", "payout_rate"]
    fit = fit_empirical_policy(
        obs,
        x_cols,
        y_cols,
        EmpiricalPolicyConfig(degree=2),
    )
    finite_nondefault = (
        ~obs["default"].astype(bool)
        & np.isfinite(obs[x_cols + y_cols].to_numpy(dtype=np.float64)).all(axis=1)
    )

    assert fit.n_observations == int(finite_nondefault.sum())
    assert set(fit.outcomes) == set(y_cols)
    for outcome in fit.outcomes.values():
        assert outcome.coefficients.shape == (len(fit.feature_names),)
        assert np.isfinite(outcome.coefficients).all()
        assert np.isfinite(outcome.fitted_values).all()

    slices = predict_policy_slices(fit, obs, x_cols, grid_size=11)
    assert set(slices) == set(x_cols)
    for x_col, frame in slices.items():
        assert frame.shape == (11, 1 + len(y_cols))
        assert x_col in frame.columns
        assert np.isfinite(frame.to_numpy(dtype=np.float64)).all()
