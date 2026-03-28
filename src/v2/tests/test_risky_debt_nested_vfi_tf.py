"""Tests for the TF-native risky-debt nested VFI solver."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.v2.environments.risky_debt import EconomicParams, RiskyDebtEnv, ShockParams
from src.v2.solvers import (
    NestedVFIConfig,
    NestedVFIGridConfig,
    NestedVFITFRuntimeConfig,
    solve_nested_vfi,
    solve_nested_vfi_tf,
)
from src.v2.solvers.grid import tauchen_transition_matrix, tauchen_transition_matrix_tf
from src.v2.solvers.nested_vfi_np import _reward_tensor_np


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _make_env() -> RiskyDebtEnv:
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
def tf_solved_result():
    env = _make_env()
    config = NestedVFIConfig(
        grid=NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[4, 4]),
        max_iter_inner=500,
        tol_inner=1e-6,
        max_iter_outer=20,
        tol_outer_value=1e-3,
    )
    runtime_config = NestedVFITFRuntimeConfig(
        dtype="float32",
        choice_block_size=8,
        jit_compile=True,
        record_inner_history=False,
    )
    result = solve_nested_vfi_tf(env, config=config, runtime_config=runtime_config)
    return env, result


@pytest.fixture(scope="module")
def backend_results():
    config = NestedVFIConfig(
        grid=NestedVFIGridConfig(exo_sizes=[3], endo_sizes=[4, 4]),
        max_iter_inner=500,
        tol_inner=1e-6,
        max_iter_outer=20,
        tol_outer_value=1e-3,
    )
    env_np = _make_env()
    env_tf = _make_env()
    np_result = solve_nested_vfi(env_np, config=config)
    tf_result = solve_nested_vfi_tf(
        env_tf,
        config=config,
        runtime_config=NestedVFITFRuntimeConfig(
            dtype="float32",
            choice_block_size=8,
            jit_compile=True,
            record_inner_history=False,
        ),
    )
    return env_np, np_result, env_tf, tf_result


def test_tauchen_transition_matrix_tf_matches_numpy():
    env = _make_env()
    z_grid = [np.geomspace(env.z_min, env.z_max, 7, dtype=np.float64)]
    np_probs = tauchen_transition_matrix(env.shocks, z_grid)
    tf_probs = tauchen_transition_matrix_tf(
        env.shocks,
        z_grid,
        dtype=tf.float64,
    ).numpy()
    np.testing.assert_allclose(tf_probs, np_probs, rtol=1e-7, atol=1e-7)


def test_tf_nested_vfi_values_are_nonnegative(tf_solved_result):
    _, result = tf_solved_result
    assert result["backend"] == "tensorflow"
    assert result["dtype"] == "float32"
    assert result["device"]
    value = result["value"].numpy()
    assert np.all(value >= -1e-6)


def test_tf_default_mask_matches_zero_value_states(tf_solved_result):
    _, result = tf_solved_result
    value = result["value"].numpy().reshape(result["default_mask"].shape)
    default_mask = result["default_mask"].numpy()
    np.testing.assert_array_equal(default_mask, value <= 0.0)


def test_tf_policy_action_matches_policy_endo(tf_solved_result):
    env, result = tf_solved_result
    policy_action = result["policy_action"].numpy()
    policy_endo = result["policy_endo"].numpy()
    current_k = result["grids"]["endo_product"][:, 0][None, :]
    implied_investment = (
        policy_endo[..., 0]
        - (1.0 - env.econ.depreciation_rate) * current_k
    )
    np.testing.assert_allclose(
        policy_action[..., 0], implied_investment, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        policy_action[..., 1], policy_endo[..., 1], rtol=1e-5, atol=1e-5
    )


def test_tf_zero_profit_residual_is_small_on_funded_cells(tf_solved_result):
    _, result = tf_solved_result
    funded = result["funded_mask"].numpy()
    residual = result["zero_profit_residual"].numpy()
    assert funded.any()
    assert np.nanmax(np.abs(residual[funded])) < 5e-5


def test_tf_nonpositive_b_choices_use_risk_free_rate_and_are_not_funded(tf_solved_result):
    env, result = tf_solved_result
    b_grid = result["grids"]["endo_grids_1d"][1]
    nonpositive = b_grid <= 1e-12
    assert np.any(nonpositive)
    np.testing.assert_allclose(
        result["r_tilde_grid"].numpy()[:, :, nonpositive],
        env.econ.interest_rate,
        atol=1e-6,
    )
    assert not np.any(result["funded_mask"].numpy()[:, :, nonpositive])


def test_tf_outer_value_and_pricing_histories_are_reported(tf_solved_result):
    _, result = tf_solved_result
    assert len(result["outer_value_diff_history"]) == result["n_outer"]
    assert len(result["pricing_diff_history"]) == result["n_outer"]
    assert result["inner_diff_history"] == []


def test_tf_returned_value_satisfies_bellman_under_returned_pricing(tf_solved_result):
    env, result = tf_solved_result
    value = result["value"].numpy()
    prob_matrix = result["prob_matrix"].numpy()
    reward = _reward_tensor_np(env, result["grids"], result["r_tilde_grid"].numpy())

    expected_continuation = prob_matrix @ value
    rhs = reward + env.discount() * expected_continuation[:, None, :]
    bellman_update = np.maximum(rhs.max(axis=2), 0.0)
    assert np.max(np.abs(bellman_update - value)) < 1e-4


def test_tf_and_numpy_backends_agree_economically(backend_results):
    _, np_result, _, tf_result = backend_results

    value_gap = np.max(
        np.abs(_to_numpy(tf_result["value"]) - _to_numpy(np_result["value"]))
    )
    assert value_gap < 1e-2

    np_r = _to_numpy(np_result["r_tilde_grid"])
    tf_r = _to_numpy(tf_result["r_tilde_grid"])
    np_discount = np.where(np.isfinite(np_r), 1.0 / np.maximum(1.0 + np_r, 1e-12), 0.0)
    tf_discount = np.where(np.isfinite(tf_r), 1.0 / np.maximum(1.0 + tf_r, 1e-12), 0.0)
    pricing_gap = np.max(np.abs(tf_discount - np_discount))
    assert pricing_gap < 5e-3

    policy_gap = np.max(
        np.abs(_to_numpy(tf_result["policy_endo"]) - _to_numpy(np_result["policy_endo"]))
    )
    k_step = np.max(np.diff(np_result["grids"]["endo_grids_1d"][0]))
    b_step = np.max(np.diff(np_result["grids"]["endo_grids_1d"][1]))
    assert policy_gap <= max(k_step, b_step) + 1e-6

    default_agreement = np.mean(
        _to_numpy(tf_result["default_mask"]) == _to_numpy(np_result["default_mask"])
    )
    assert default_agreement >= 0.95
