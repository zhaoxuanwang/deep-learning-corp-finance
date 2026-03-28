"""Tests for the canonical risky-debt environment."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.v2.environments.risky_debt import EconomicParams, RiskyDebtEnv, ShockParams


def _manual_reward(env, k, b, z, k_next, b_next, r_tilde):
    investment = k_next - (1.0 - env.econ.depreciation_rate) * k
    after_tax_profit = (
        (1.0 - env.econ.tax) * z * (k ** env.econ.production_elasticity)
    )
    adjustment_cost = (
        0.5 * env.econ.cost_convex * investment ** 2 / max(k, 1e-12)
    )
    if np.isfinite(r_tilde):
        debt_discount = 1.0 / (1.0 + r_tilde)
    else:
        debt_discount = 0.0
    debt_proceeds = b_next * debt_discount
    tax_shield = (
        env.econ.tax
        * b_next
        * (1.0 - debt_discount)
        / (1.0 + env.econ.interest_rate)
    )
    cash_flow = (
        after_tax_profit
        - adjustment_cost
        - investment
        - b
        + debt_proceeds
        + tax_shield
    )
    shortfall = max(-cash_flow, 0.0)
    issuance = (
        env.econ.cost_inject_fixed * float(shortfall > 0.0)
        + env.econ.cost_inject_linear * shortfall
    )
    return cash_flow - issuance


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


def test_recovery_matches_c_d_formula(env):
    k_next = np.array([env.k_min, env.k_max], dtype=np.float32)
    z_next = np.array([env.z_min, env.z_max], dtype=np.float32)
    recovery = np.asarray(env.recovery_value(k_next, z_next))
    expected = (1.0 - env.econ.default_haircut) * (
        (1.0 - env.econ.tax)
        * z_next
        * np.power(k_next, env.econ.production_elasticity)
        + (1.0 - env.econ.depreciation_rate) * k_next
    )
    np.testing.assert_allclose(recovery, expected, rtol=1e-6, atol=1e-6)


def test_b_max_scales_with_k_max(env):
    assert env.b_max == pytest.approx(5.0 * env.k_max, rel=1e-6)
    assert env.b_min == pytest.approx(-0.2 * env.b_max, rel=1e-6)


def test_action_bounds_and_scale_reference_reflect_savings_limit(env):
    low, high = env.action_bounds()
    np.testing.assert_allclose(
        low.numpy(),
        np.array([env.I_min, env.b_min], dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        high.numpy(),
        np.array([env.I_max, env.b_max], dtype=np.float32),
        atol=1e-7,
    )

    center, half_range = env.action_scale_reference()
    np.testing.assert_allclose(
        center.numpy(),
        np.array([0.0, 0.5 * (env.b_min + env.b_max)], dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        half_range.numpy(),
        np.array(
            [max(abs(env.I_min), env.I_max), 0.5 * (env.b_max - env.b_min)],
            dtype=np.float32,
        ),
        atol=1e-7,
    )


def test_sample_initial_endogenous_honors_negative_savings_bound(env):
    samples = env.sample_initial_endogenous(
        256,
        seed=tf.constant([123, 456], dtype=tf.int32),
    ).numpy()
    assert np.all(samples[:, 0] >= env.k_min - 1e-6)
    assert np.all(samples[:, 0] <= env.k_max + 1e-6)
    assert np.all(samples[:, 1] >= env.b_min - 1e-6)
    assert np.all(samples[:, 1] <= env.b_max + 1e-6)
    assert np.any(samples[:, 1] < 0.0)


def test_resolve_r_tilde_defaults_to_risk_free_without_installed_grid(env):
    r_tilde = env.resolve_r_tilde(
        tf.constant([0.7 * env.k_max], dtype=tf.float32),
        tf.constant([0.3 * env.b_max], dtype=tf.float32),
        tf.constant([np.exp(env.shocks.mu)], dtype=tf.float32),
    ).numpy()
    np.testing.assert_allclose(r_tilde, [env.econ.interest_rate], atol=1e-7)


def test_reward_uses_installed_r_tilde_grid(env):
    grids = {
        "exo_grids_1d": [np.array([env.z_min, env.z_max], dtype=np.float32)],
        "endo_grids_1d": [
            np.array([env.k_min, env.k_max], dtype=np.float32),
            np.array([env.b_min, env.b_max], dtype=np.float32),
        ],
    }
    r_tilde_grid = np.array(
        [
            [[0.10, 0.20], [0.30, 0.40]],
            [[0.50, 0.60], [0.70, 0.80]],
        ],
        dtype=np.float32,
    )
    env.install_r_tilde_grid(grids, r_tilde_grid)

    k = 0.6 * env.k_max
    b = 0.4 * env.b_max
    z = 0.9 * env.z_max
    k_next = 0.92 * env.k_max
    b_next = 0.88 * env.b_max
    investment = k_next - (1.0 - env.econ.depreciation_rate) * k
    state = tf.constant([[k, b, z]], dtype=tf.float32)
    action = tf.constant([[investment, b_next]], dtype=tf.float32)

    reward = float(tf.squeeze(env.reward(state, action)))
    expected = _manual_reward(env, k, b, z, k_next, b_next, r_tilde=0.80)
    assert reward == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_reward_handles_infinite_r_tilde(env):
    grids = {
        "exo_grids_1d": [np.array([env.z_min, env.z_max], dtype=np.float32)],
        "endo_grids_1d": [
            np.array([env.k_min, env.k_max], dtype=np.float32),
            np.array([env.b_min, env.b_max], dtype=np.float32),
        ],
    }
    r_tilde_grid = np.array(
        [
            [[0.10, 0.20], [0.30, 0.40]],
            [[0.50, 0.60], [0.70, np.inf]],
        ],
        dtype=np.float32,
    )
    env.install_r_tilde_grid(grids, r_tilde_grid)

    k = 0.7 * env.k_max
    b = 0.2 * env.b_max
    z = env.z_max
    k_next = env.k_max
    b_next = env.b_max
    investment = k_next - (1.0 - env.econ.depreciation_rate) * k
    state = tf.constant([[k, b, z]], dtype=tf.float32)
    action = tf.constant([[investment, b_next]], dtype=tf.float32)

    reward = float(tf.squeeze(env.reward(state, action)))
    expected = _manual_reward(env, k, b, z, k_next, b_next, r_tilde=np.inf)
    assert reward == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_clear_r_tilde_grid_restores_risk_free_fallback(env):
    grids = {
        "exo_grids_1d": [np.array([env.z_min, env.z_max], dtype=np.float32)],
        "endo_grids_1d": [
            np.array([env.k_min, env.k_max], dtype=np.float32),
            np.array([env.b_min, env.b_max], dtype=np.float32),
        ],
    }
    env.install_r_tilde_grid(
        grids,
        np.full((2, 2, 2), 0.75, dtype=np.float32),
    )
    env.clear_r_tilde_grid()

    r_tilde = env.resolve_r_tilde(
        tf.constant([env.k_max], dtype=tf.float32),
        tf.constant([env.b_max], dtype=tf.float32),
        tf.constant([env.z_max], dtype=tf.float32),
    ).numpy()
    np.testing.assert_allclose(r_tilde, [env.econ.interest_rate], atol=1e-7)


def test_reward_matches_manual_formula_with_savings_choice(env):
    grids = {
        "exo_grids_1d": [np.array([env.z_min, env.z_max], dtype=np.float32)],
        "endo_grids_1d": [
            np.array([env.k_min, env.k_max], dtype=np.float32),
            np.array([env.b_min, env.b_max], dtype=np.float32),
        ],
    }
    env.install_r_tilde_grid(
        grids,
        np.full((2, 2, 2), env.econ.interest_rate, dtype=np.float32),
    )

    k = 0.65 * env.k_max
    b = -0.4 * abs(env.b_min)
    z = 0.85 * env.z_max
    k_next = 0.75 * env.k_max
    b_next = 0.6 * env.b_min
    investment = k_next - (1.0 - env.econ.depreciation_rate) * k
    state = tf.constant([[k, b, z]], dtype=tf.float32)
    action = tf.constant([[investment, b_next]], dtype=tf.float32)

    reward = float(tf.squeeze(env.reward(state, action)))
    expected = _manual_reward(env, k, b, z, k_next, b_next, r_tilde=env.econ.interest_rate)
    assert reward == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_continuation_transform_is_relu(env):
    values = tf.constant([-2.0, 0.0, 1.5], dtype=tf.float32)
    transformed = env.continuation_transform(values).numpy()
    np.testing.assert_allclose(transformed, [0.0, 0.0, 1.5], atol=1e-7)
