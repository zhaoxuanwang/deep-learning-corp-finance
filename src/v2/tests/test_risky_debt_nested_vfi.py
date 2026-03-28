"""Tests for the canonical risky-debt nested VFI solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.v2.environments.risky_debt import EconomicParams, RiskyDebtEnv, ShockParams
from src.v2.solvers import NestedVFIConfig, NestedVFIGridConfig, solve_nested_vfi
from src.v2.solvers.nested_vfi_np import _build_nested_vfi_grids, _reward_tensor_np


@pytest.fixture
def env():
    return RiskyDebtEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.7,
            cost_convex=0.02,
            tax=0.35,
            default_haircut=0.6,
            cost_inject_fixed=0.01,
            cost_inject_linear=0.05,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.7, sigma=0.15),
        k_min_mult=0.01,
        k_max_mult=1.5,
        b_max_mult=5.0,
    )


@pytest.fixture(scope="module")
def solved_result():
    env = RiskyDebtEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.7,
            cost_convex=0.02,
            tax=0.35,
            default_haircut=0.6,
            cost_inject_fixed=0.01,
            cost_inject_linear=0.05,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.7, sigma=0.15),
        k_min_mult=0.01,
        k_max_mult=1.5,
        b_max_mult=5.0,
    )
    config = NestedVFIConfig(
        grid=NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[4, 4]),
        max_iter_inner=500,
        tol_inner=1e-6,
        max_iter_outer=20,
        tol_outer_value=1e-3,
    )
    result = solve_nested_vfi(env, config=config)
    return env, result


def test_nested_vfi_values_are_nonnegative(solved_result):
    _, result = solved_result
    assert result["backend"] == "numpy"
    assert result["dtype"] == "float64"
    value = result["value"]
    assert np.all(value >= -1e-8)


def test_default_mask_matches_zero_value_states(solved_result):
    _, result = solved_result
    value = result["value"].reshape(result["default_mask"].shape)
    default_mask = result["default_mask"]
    np.testing.assert_array_equal(default_mask, value <= 0.0)


def test_policy_action_matches_policy_endo(solved_result):
    env, result = solved_result
    policy_action = result["policy_action"]
    policy_endo = result["policy_endo"]
    current_k = result["grids"]["endo_product"][:, 0][None, :]
    implied_investment = (
        policy_endo[..., 0]
        - (1.0 - env.econ.depreciation_rate) * current_k
    )

    np.testing.assert_allclose(
        policy_action[..., 0], implied_investment, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        policy_action[..., 1], policy_endo[..., 1], rtol=1e-6, atol=1e-6
    )


def test_zero_profit_residual_is_small_on_funded_cells(solved_result):
    _, result = solved_result
    funded = result["funded_mask"]
    residual = result["zero_profit_residual"]
    assert funded.any()
    assert np.nanmax(np.abs(residual[funded])) < 1e-6


def test_all_default_cells_report_infinite_r_tilde(solved_result):
    _, result = solved_result
    r_tilde = result["r_tilde_grid"]
    funded = result["funded_mask"]
    inf_mask = np.isinf(r_tilde)
    assert inf_mask.any()
    assert np.all(~funded[inf_mask])


def test_outer_value_and_pricing_histories_are_reported(solved_result):
    _, result = solved_result
    assert len(result["outer_value_diff_history"]) == result["n_outer"]
    assert len(result["pricing_diff_history"]) == result["n_outer"]


def test_nonpositive_b_choices_use_risk_free_rate_and_are_not_funded(solved_result):
    env, result = solved_result
    b_grid = result["grids"]["endo_grids_1d"][1]
    nonpositive = b_grid <= 1e-12
    assert np.any(nonpositive)
    np.testing.assert_allclose(
        result["r_tilde_grid"][:, :, nonpositive],
        env.econ.interest_rate,
        atol=1e-10,
    )
    assert not np.any(result["funded_mask"][:, :, nonpositive])


def test_nested_vfi_grid_includes_negative_savings_points(env):
    grids = _build_nested_vfi_grids(
        env,
        NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[4, 5]),
    )
    b_grid = grids["endo_grids_1d"][1]
    assert b_grid[0] < 0.0
    assert b_grid[-1] == pytest.approx(env.b_max, rel=1e-6)


def test_returned_value_satisfies_bellman_under_returned_pricing(solved_result):
    env, result = solved_result
    value = result["value"]
    prob_matrix = result["prob_matrix"]
    reward = _reward_tensor_np(env, result["grids"], result["r_tilde_grid"])

    expected_continuation = prob_matrix @ value
    rhs = reward + env.discount() * expected_continuation[:, None, :]
    bellman_update = np.maximum(rhs.max(axis=2), 0.0)
    assert np.max(np.abs(bellman_update - value)) < 1e-5


def test_solver_stops_on_inner_nonconvergence(env):
    config = NestedVFIConfig(
        grid=NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[4, 4]),
        max_iter_inner=1,
        tol_inner=1e-12,
        max_iter_outer=5,
        tol_outer_value=1e-4,
    )
    result = solve_nested_vfi(env, config=config)
    assert result["stop_reason"] == "max_inner"
    assert not result["converged_outer"]
    assert not result["converged_inner"]


def test_nested_vfi_reward_tensor_has_expected_shape(env):
    grids = _build_nested_vfi_grids(
        env,
        NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[4, 4]),
    )
    r_tilde_grid = np.full((3, 4, 4), env.econ.interest_rate, dtype=np.float64)
    reward_tensor = _reward_tensor_np(env, grids, r_tilde_grid)
    assert reward_tensor.shape == (3, 16, 16)
    assert np.all(np.isfinite(reward_tensor))


def test_nested_vfi_grid_config_can_override_capital_spacing(env):
    default_grids = _build_nested_vfi_grids(
        env,
        NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[5, 4]),
    )
    linear_grids = _build_nested_vfi_grids(
        env,
        NestedVFIGridConfig(
            exo_sizes=[3],
            endo_sizes=[5, 4],
            exo_spacings=["log"],
            endo_spacings=["linear", "linear"],
        ),
    )

    default_k_grid = default_grids["endo_grids_1d"][0]
    linear_k_grid = linear_grids["endo_grids_1d"][0]

    assert not np.allclose(np.diff(default_k_grid), np.diff(default_k_grid)[0])
    np.testing.assert_allclose(
        np.diff(linear_k_grid),
        np.diff(linear_k_grid)[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_nested_vfi_grid_config_validates_spacing_lengths():
    with pytest.raises(ValueError, match="endo_spacings must have length 2"):
        NestedVFIGridConfig(
            exo_sizes=[3],
            endo_sizes=[4, 4],
            endo_spacings=["linear"],
        )
