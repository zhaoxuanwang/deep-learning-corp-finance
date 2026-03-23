"""Tests for v2 environment modules: BasicInvestmentEnv."""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.environments.basic_investment import BasicInvestmentEnv, EconomicParams, ShockParams
from src.v2.environments.base import MDPEnvironment


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
    econ = EconomicParams(
        interest_rate=0.04,
        depreciation_rate=0.1,
        production_elasticity=0.7,
    )
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
        expected_beta = 1.0 / (1.0 + env.econ.interest_rate)
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
        """k' = (1-depreciation_rate)*k + I when result is above k_min."""
        # Use k_val well within valid bounds.
        k_val = float(env.k_star)  # steady-state capital
        z_val = 1.0
        I_val = 0.5
        s = tf.constant([[k_val, z_val]])
        a = tf.constant([[I_val]])
        eps = tf.constant([[0.0]])  # no shock
        s_next = env.transition(s, a, eps)
        k_next = float(s_next[0, 0])
        expected = (1.0 - env.econ.depreciation_rate) * k_val + I_val
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

    def test_adjustment_indicator_supports_hard_and_soft_modes(self):
        """Fixed-cost gate can be queried in either hard or soft mode."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_convex=0.2, cost_fixed=0.02))
        s = tf.constant([[env.k_star, 1.0]], dtype=tf.float32)

        no_adjust = tf.constant([[0.0]], dtype=tf.float32)
        invest = tf.constant([[0.5]], dtype=tf.float32)

        hard_no_adjust = env.adjustment_indicator(
            s, no_adjust, gate_mode="hard")
        soft_no_adjust = env.adjustment_indicator(
            s, no_adjust, temperature=1e-6, gate_mode="soft")
        hard_invest = env.adjustment_indicator(
            s, invest, gate_mode="hard")
        soft_invest = env.adjustment_indicator(
            s, invest, temperature=1e-6, gate_mode="soft")

        assert float(hard_no_adjust[0]) == pytest.approx(0.0)
        assert 0.45 < float(soft_no_adjust[0]) < 0.55
        assert float(hard_invest[0]) == pytest.approx(1.0)
        assert float(soft_invest[0]) > 1.0 - 1e-6

    def test_reward_accepts_hard_gate_mode(self):
        """Hard-gate reward evaluation is available for discrete solvers."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_convex=0.2, cost_fixed=0.02))
        s = tf.constant([[env.k_star, 1.0]], dtype=tf.float32)
        a = tf.constant([[0.25]], dtype=tf.float32)
        reward = env.reward(s, a, gate_mode="hard")
        assert np.isfinite(float(tf.squeeze(reward)))

    def test_economic_params_accept_legacy_aliases(self):
        """Legacy aliases still map onto the renamed public fields."""
        econ = EconomicParams(r_rate=0.03, delta=0.2, theta=0.65)
        assert econ.interest_rate == pytest.approx(0.03)
        assert econ.depreciation_rate == pytest.approx(0.2)
        assert econ.production_elasticity == pytest.approx(0.65)
        assert econ.r_rate == pytest.approx(0.03)
        assert econ.delta == pytest.approx(0.2)
        assert econ.theta == pytest.approx(0.65)

    def test_adjustment_epsilon_is_configurable(self):
        """The fixed-cost gate tolerance can be varied explicitly."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(
                cost_convex=0.2,
                cost_fixed=0.02,
                adjustment_epsilon=1e-3,
            )
        )
        s = tf.constant([[env.k_star, 1.0]], dtype=tf.float32)
        a_small = tf.constant([[5e-4 * env.k_star]], dtype=tf.float32)
        hard_gate = env.adjustment_indicator(s, a_small, gate_mode="hard")
        assert float(hard_gate[0]) == pytest.approx(0.0)

    def test_soft_adjustment_epsilon_overrides_soft_gate_only(self):
        """Soft/STE gates can use a training-only epsilon without changing hard mode."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(
                cost_convex=0.2,
                cost_fixed=0.02,
                adjustment_epsilon=1e-8,
            ),
            soft_adjustment_epsilon=5e-3,
        )
        s = tf.constant([[env.k_star, 1.0]], dtype=tf.float32)
        no_adjust = tf.constant([[0.0]], dtype=tf.float32)

        soft_gate = env.adjustment_indicator(
            s, no_adjust, temperature=1e-3, gate_mode="soft"
        )
        hard_gate = env.adjustment_indicator(
            s, no_adjust, gate_mode="hard"
        )
        assert float(hard_gate[0]) == pytest.approx(0.0)
        assert float(soft_gate[0]) < 0.01

    def test_default_gate_mode_applies_when_gate_mode_omitted(self):
        """Env-level gate mode lets trainers switch from soft to STE generically."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_convex=0.2, cost_fixed=0.02),
            soft_adjustment_epsilon=5e-3,
            default_gate_mode="ste",
        )
        s = tf.constant([[env.k_star, 1.0]], dtype=tf.float32)
        no_adjust = tf.constant([[0.0]], dtype=tf.float32)

        gate = env.adjustment_indicator(s, no_adjust, temperature=1e-3)
        reward = env.reward(s, no_adjust, temperature=1e-3)

        assert float(gate[0]) == pytest.approx(0.0)
        assert np.isfinite(float(tf.squeeze(reward)))

    def test_soft_adjustment_epsilon_factor_tracks_temperature(self):
        """Dynamic soft epsilon can stay proportional to the current temperature."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_convex=0.2, cost_fixed=0.02),
            soft_adjustment_epsilon_factor=5.0,
        )
        s = tf.constant([[env.k_star, 1.0]], dtype=tf.float32)
        no_adjust = tf.constant([[0.0]], dtype=tf.float32)

        gate_hi = env.adjustment_indicator(
            s, no_adjust, temperature=1e-2, gate_mode="soft"
        )
        gate_lo = env.adjustment_indicator(
            s, no_adjust, temperature=1e-3, gate_mode="soft"
        )

        target = float(tf.sigmoid(-5.0))
        assert float(gate_hi[0]) == pytest.approx(target)
        assert float(gate_lo[0]) == pytest.approx(target)


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
        z_bar = env.stationary_exo()
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
    """Tests for reward_scale (λ-preprocessing)."""

    def test_reward_scale_positive(self, env):
        """reward_scale returns a positive scalar."""
        lam = env.reward_scale()
        assert lam > 0

    def test_reward_scale_no_seed_required(self, env):
        """reward_scale works without seed (analytical override ignores it)."""
        lam = env.reward_scale()
        assert lam > 0

    def test_reward_scale_base_uses_fixed_seed_when_none(self, env):
        """Base class reward_scale uses fixed internal seed when called without seed."""
        from src.v2.environments.base import MDPEnvironment
        lam1 = MDPEnvironment.reward_scale(env)
        lam2 = MDPEnvironment.reward_scale(env)
        assert lam1 == pytest.approx(lam2, rel=1e-6)

    def test_reward_scale_analytical_inverse_of_v_star(self, env):
        """Analytical override equals 1 / |V*(k*, z_mean)|."""
        s_endo_ss = tf.constant([[env.k_star]])
        abs_v_star = float(tf.abs(env.terminal_value(s_endo_ss)[0]))
        expected = 1.0 / abs_v_star
        assert env.reward_scale() == pytest.approx(expected, rel=1e-5)

    def test_reward_scale_makes_q_order_one(self, env):
        """λ · |V*| ≈ 1 — the whole point of the normalizer."""
        lam = env.reward_scale()
        s_endo_ss = tf.constant([[env.k_star]])
        abs_v_star = float(tf.abs(env.terminal_value(s_endo_ss)[0]))
        assert lam * abs_v_star == pytest.approx(1.0, rel=0.01)

    def test_reward_scale_generic_vs_analytical(self, env):
        """Generic and analytical λ should be same order of magnitude."""
        from src.v2.environments.base import MDPEnvironment
        seed = tf.constant([42, 0], dtype=tf.int32)
        lam_analytical = env.reward_scale()
        lam_generic = MDPEnvironment.reward_scale(
            env, n_samples=2000, seed=seed)
        # Both positive, analytical is tighter (larger λ = smaller C).
        assert lam_analytical > lam_generic, (
            "Analytical λ should be larger (tighter) than generic")
        ratio = lam_generic / lam_analytical
        assert ratio > 0.01, (
            f"Ratio {ratio:.4f} too small — estimates diverge excessively")


# =============================================================================
# Annealing schedule
# =============================================================================

class TestAnnealingSchedule:
    """Tests for env.annealing_schedule() factory method."""

    def test_none_when_no_fixed_cost(self):
        """No annealing needed when cost_fixed == 0."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_fixed=0.0))
        assert env.annealing_schedule() is None

    def test_returns_schedule_with_fixed_cost(self):
        """Annealing schedule returned when cost_fixed > 0."""
        from src.v2.utils.annealing import AnnealingSchedule
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_fixed=0.02))
        sched = env.annealing_schedule()
        assert isinstance(sched, AnnealingSchedule)
        assert sched.init_temp == 1.0
        assert sched.min_temp == 1e-6

    def test_returns_fresh_instance(self):
        """Each call returns a new schedule (no shared state)."""
        env = BasicInvestmentEnv(
            econ_params=EconomicParams(cost_fixed=0.02))
        s1 = env.annealing_schedule()
        s2 = env.annealing_schedule()
        assert s1 is not s2
        # Mutating s1 doesn't affect s2
        s1.update()
        assert s1.step == 1
        assert s2.step == 0

    def test_base_class_returns_none(self):
        """MDPEnvironment base default is None."""
        env = BasicInvestmentEnv()  # cost_fixed=0 by default
        assert MDPEnvironment.annealing_schedule(env) is None


# =============================================================================
# Analytical policy
# =============================================================================

class TestAnalyticalPolicy:
    """Tests for env.analytical_policy()."""

    def test_output_shape(self, env):
        """analytical_policy returns shape (batch, 1)."""
        s = env.sample_initial_states(32, seed=tf.constant([50, 51]))
        a = env.analytical_policy(s)
        assert a.shape == (32, 1)

    def test_differentiable(self, env):
        """analytical_policy is differentiable w.r.t. state."""
        s = tf.Variable(env.sample_initial_states(4, seed=tf.constant([52, 53])))
        with tf.GradientTape() as tape:
            a = env.analytical_policy(s)
            loss = tf.reduce_sum(a)
        grad = tape.gradient(loss, s)
        assert grad is not None

    def test_base_class_raises(self, env):
        """Base class analytical_policy raises NotImplementedError."""
        from src.v2.environments.base import MDPEnvironment
        with pytest.raises(NotImplementedError):
            MDPEnvironment.analytical_policy(env, tf.zeros((1, 2)))

    def test_frictionless_policy_matches_compute_frictionless_policy(self, env):
        """analytical_policy matches the module-level compute_frictionless_policy."""
        from src.v2.environments.basic_investment import compute_frictionless_policy
        z_vals = np.array([0.8, 1.0, 1.2])
        k_star_val = env.k_star
        s = tf.constant(
            [[k_star_val, z] for z in z_vals], dtype=tf.float32)
        I_analytical = env.analytical_policy(s).numpy().reshape(-1)
        kprime_analytical = (1.0 - env.econ.depreciation_rate) * k_star_val + I_analytical
        kprime_reference = compute_frictionless_policy(z_vals, env.econ, env.shocks)
        kprime_reference = np.clip(kprime_reference, env.k_min, env.k_max)
        np.testing.assert_allclose(kprime_analytical, kprime_reference, rtol=1e-5)
