"""Tests for v2 environment modules: BasicInvestmentEnv."""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.environments.base import MDPEnvironment
from src.economy.parameters import EconomicParams, ShockParams


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def env():
    """BasicInvestmentEnv with default parameters."""
    return BasicInvestmentEnv()


@pytest.fixture
def custom_env():
    """BasicInvestmentEnv with custom parameters."""
    econ = EconomicParams(r_rate=0.04, delta=0.1, theta=0.7)
    shock = ShockParams(rho=0.7, sigma=0.1)
    return BasicInvestmentEnv(econ_params=econ, shock_params=shock)


# =============================================================================
# Interface conformance
# =============================================================================

class TestMDPInterface:
    """Test that BasicInvestmentEnv conforms to MDPEnvironment interface."""

    def test_is_mdp_environment(self, env):
        """BasicInvestmentEnv is a subclass of MDPEnvironment."""
        assert isinstance(env, MDPEnvironment)

    def test_state_dim(self, env):
        """State dimension is 2: (k, z)."""
        assert env.state_dim() == 2

    def test_action_dim(self, env):
        """Action dimension is 1: (I,)."""
        assert env.action_dim() == 1

    def test_action_bounds_shape(self, env):
        """action_bounds returns two 1-D tensors of shape (1,)."""
        low, high = env.action_bounds()
        assert low.shape == (1,)
        assert high.shape == (1,)
        assert float(low[0]) < float(high[0])

    def test_discount_is_beta(self, env):
        """Discount factor is 1/(1+r)."""
        expected_beta = 1.0 / (1.0 + env.econ.r_rate)
        assert env.discount() == pytest.approx(expected_beta, rel=1e-6)


# =============================================================================
# State/shock sampling
# =============================================================================

class TestSampling:
    """Tests for state and shock sampling."""

    def test_sample_initial_states_shape(self, env):
        """Sampled states have shape (n, 2)."""
        s = env.sample_initial_states(100, seed=(1, 2))
        assert s.shape == (100, 2)

    def test_sample_initial_states_in_bounds(self, env):
        """All sampled k in [k_min, k_max], z in [z_min, z_max]."""
        s = env.sample_initial_states(1000, seed=(3, 4))
        k = s[:, 0]
        z = s[:, 1]
        assert float(tf.reduce_min(k)) >= env.k_min - 1e-6
        assert float(tf.reduce_max(k)) <= env.k_max + 1e-6
        assert float(tf.reduce_min(z)) >= env.z_min - 1e-6
        assert float(tf.reduce_max(z)) <= env.z_max + 1e-6

    def test_sample_initial_states_deterministic(self, env):
        """Same seed produces same states."""
        s1 = env.sample_initial_states(50, seed=(10, 20))
        s2 = env.sample_initial_states(50, seed=(10, 20))
        np.testing.assert_allclose(s1.numpy(), s2.numpy())

    def test_sample_shocks_shape(self, env):
        """Sampled shocks have shape (n, 1)."""
        eps = env.sample_shocks(100, seed=(5, 6))
        assert eps.shape == (100, 1)

    def test_sample_shocks_deterministic(self, env):
        """Same seed produces same shocks."""
        e1 = env.sample_shocks(50, seed=(7, 8))
        e2 = env.sample_shocks(50, seed=(7, 8))
        np.testing.assert_allclose(e1.numpy(), e2.numpy())


# =============================================================================
# Reward and transition
# =============================================================================

class TestDynamics:
    """Tests for reward and transition functions."""

    def test_reward_shape(self, env):
        """Reward has shape (batch,)."""
        s = env.sample_initial_states(32, seed=(11, 12))
        a = tf.zeros((32, 1))  # zero investment
        r = env.reward(s, a)
        assert r.shape == (32,) or r.shape == (32, 1)

    def test_transition_shape(self, env):
        """Transition output has shape (batch, 2)."""
        s = env.sample_initial_states(32, seed=(13, 14))
        a = tf.zeros((32, 1))
        eps = env.sample_shocks(32, seed=(15, 16))
        s_next = env.transition(s, a, eps)
        assert s_next.shape == (32, 2)

    def test_capital_accumulation(self, env):
        """k' = (1-delta)*k + I when result is above k_min."""
        # Use k_val well within valid bounds.
        k_val = float(env.k_star)  # steady-state capital
        z_val = 1.0
        I_val = 0.5
        s = tf.constant([[k_val, z_val]])
        a = tf.constant([[I_val]])
        eps = tf.constant([[0.0]])  # no shock
        s_next = env.transition(s, a, eps)
        k_next = float(s_next[0, 0])
        expected = (1.0 - env.econ.delta) * k_val + I_val
        assert k_next == pytest.approx(expected, rel=1e-5)

    def test_capital_floor(self, env):
        """Capital cannot go below k_min even with large negative I."""
        s = tf.constant([[1.0, 1.0]])
        a = tf.constant([[-100.0]])  # extreme disinvestment
        eps = tf.constant([[0.0]])
        s_next = env.transition(s, a, eps)
        k_next = float(s_next[0, 0])
        assert k_next >= env.k_min - 1e-6

    def test_zero_shock_preserves_log_z(self, env):
        """With eps=0, log(z') = (1-rho)*mu + rho*log(z)."""
        z_val = 1.5
        s = tf.constant([[3.0, z_val]])
        a = tf.constant([[0.1]])
        eps = tf.constant([[0.0]])
        s_next = env.transition(s, a, eps)
        z_next = float(s_next[0, 1])

        rho = env.shocks.rho
        mu = env.shocks.mu
        log_z = np.log(z_val)
        expected_log_z_next = (1 - rho) * mu + rho * log_z
        expected_z_next = np.exp(expected_log_z_next)
        assert z_next == pytest.approx(expected_z_next, rel=1e-4)

    def test_reward_differentiable(self, env):
        """Reward is differentiable w.r.t. action."""
        s = tf.constant([[3.0, 1.0]])
        a = tf.Variable([[0.5]])
        with tf.GradientTape() as tape:
            r = env.reward(s, a)
            loss = tf.reduce_sum(r)
        grad = tape.gradient(loss, a)
        assert grad is not None

    def test_transition_differentiable(self, env):
        """Transition is differentiable w.r.t. action."""
        s = tf.constant([[3.0, 1.0]])
        a = tf.Variable([[0.5]])
        eps = tf.constant([[0.0]])
        with tf.GradientTape() as tape:
            s_next = env.transition(s, a, eps)
            loss = tf.reduce_sum(s_next)
        grad = tape.gradient(loss, a)
        assert grad is not None


# =============================================================================
# Euler residual
# =============================================================================

class TestEulerResidual:
    """Tests for the Euler residual computation."""

    def test_euler_residual_shape(self, env):
        """Euler residual has shape (batch,)."""
        s = env.sample_initial_states(16, seed=(20, 21))
        a = tf.zeros((16, 1))
        eps = env.sample_shocks(16, seed=(22, 23))
        s_next = env.transition(s, a, eps)
        a_next = tf.zeros((16, 1))
        er = env.euler_residual(s, a, s_next, a_next)
        assert er.shape[0] == 16

    def test_euler_residual_finite(self, env):
        """Euler residual should be finite for reasonable inputs."""
        s = tf.constant([[3.0, 1.0]])
        a = tf.constant([[0.3]])
        eps = tf.constant([[0.0]])
        s_next = env.transition(s, a, eps)
        a_next = tf.constant([[0.3]])
        er = env.euler_residual(s, a, s_next, a_next)
        assert np.all(np.isfinite(er.numpy()))


# =============================================================================
# Terminal value
# =============================================================================

class TestTerminalValue:
    """Tests for the terminal value approximation."""

    def test_terminal_value_shape(self, env):
        """Terminal value has shape (batch,)."""
        s_endo = env.sample_initial_endogenous(16, seed=(30, 31))
        v = env.terminal_value(s_endo)
        assert v.shape[0] == 16

    def test_terminal_value_steady_state(self, env):
        """At steady state: V = r(s̄, ā) / (1 - gamma)."""
        s_endo = tf.constant([[3.0]])
        v = env.terminal_value(s_endo)
        # Reconstruct what terminal_value computes internally
        z_bar = env.exo_stationary_mean()
        s_bar = env.merge_state(s_endo, tf.reshape(z_bar, [1, -1]))
        a_bar = env.stationary_action(s_endo)
        r = env.reward(s_bar, a_bar)
        r_scalar = float(tf.squeeze(r))
        expected = r_scalar / (1.0 - env.discount())
        assert float(tf.squeeze(v)) == pytest.approx(expected, rel=1e-5)


# =============================================================================
# Reward scale (λ-preprocessing)
# =============================================================================

class TestRewardScale:
    """Tests for compute_reward_scale (λ-preprocessing)."""

    def test_reward_scale_positive(self, env):
        """compute_reward_scale returns a positive scalar."""
        seed = tf.constant([42, 0], dtype=tf.int32)
        lam = env.compute_reward_scale(seed=seed)
        assert lam > 0

    def test_reward_scale_requires_seed_on_base(self, env):
        """Generic compute_reward_scale raises if seed not provided."""
        from src.v2.environments.base import MDPEnvironment
        with pytest.raises(ValueError, match="seed is required"):
            MDPEnvironment.compute_reward_scale(env)

    def test_reward_scale_analytical_inverse_of_v_star(self, env):
        """Analytical override equals 1 / |V*(k*, z_mean)|."""
        s_endo_ss = tf.constant([[env.k_star]])
        abs_v_star = float(tf.abs(env.terminal_value(s_endo_ss)[0]))
        expected = 1.0 / abs_v_star
        seed = tf.constant([42, 0], dtype=tf.int32)
        assert env.compute_reward_scale(seed=seed) == pytest.approx(
            expected, rel=1e-5)

    def test_reward_scale_makes_q_order_one(self, env):
        """λ · |V*| ≈ 1 — the whole point of the normalizer."""
        seed = tf.constant([42, 0], dtype=tf.int32)
        lam = env.compute_reward_scale(seed=seed)
        s_endo_ss = tf.constant([[env.k_star]])
        abs_v_star = float(tf.abs(env.terminal_value(s_endo_ss)[0]))
        assert lam * abs_v_star == pytest.approx(1.0, rel=0.01)

    def test_reward_scale_generic_vs_analytical(self, env):
        """Generic and analytical λ should be same order of magnitude."""
        from src.v2.environments.base import MDPEnvironment
        seed = tf.constant([42, 0], dtype=tf.int32)
        lam_analytical = env.compute_reward_scale(seed=seed)
        lam_generic = MDPEnvironment.compute_reward_scale(
            env, n_samples=2000, seed=seed)
        # Both positive, analytical is tighter (larger λ = smaller C).
        assert lam_analytical > lam_generic, (
            "Analytical λ should be larger (tighter) than generic")
        ratio = lam_generic / lam_analytical
        assert ratio > 0.01, (
            f"Ratio {ratio:.4f} too small — estimates diverge excessively")
