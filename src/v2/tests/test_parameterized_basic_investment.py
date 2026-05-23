"""Tests for ParameterizedBasicInvestmentEnv (5-D β, frictional).

What we want to be sure of:

  1. Dimensions, bounds, β shape expectations.
  2. At the frictionless slice (φ=0), the new env reproduces
     `BasicInvestmentEnv` numerics for reward, transitions, and Euler
     residual — i.e., the additional friction machinery does no harm.
  3. Euler residual ≈ 0 at the closed-form analytical policy for
     frictionless β draws (filtering to interior states).
  4. With non-zero frictions, the reward function actually subtracts the
     adjustment cost and the Euler residual deviates from the frictionless
     closed form by the expected amount.
  5. `analytical_policy` raises when frictions are non-zero — the foot-gun
     guard works.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    EconomicParams,
    ShockParams,
)
from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
    _friction_cost,
)


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def nominal_econ():
    return EconomicParams(
        interest_rate=0.04,
        depreciation_rate=0.10,
        production_elasticity=0.5,    # ≈ Beta(2,2) mean
    )


@pytest.fixture
def nominal_shocks():
    return ShockParams(rho=0.5, sigma=0.24, mu=0.0)


@pytest.fixture
def param_env(nominal_econ, nominal_shocks):
    return ParameterizedBasicInvestmentEnv(
        nominal_econ=nominal_econ, nominal_shocks=nominal_shocks,
        k_min_mult=0.1, k_max_mult=10.0, z_sd_mult=4.0)


@pytest.fixture
def nominal_env(nominal_econ, nominal_shocks):
    """Existing BasicInvestmentEnv with frictionless costs (matches param env
    at the φ=0 slice). z_sd_mult set to match param env's default."""
    return BasicInvestmentEnv(
        econ_params=nominal_econ, shock_params=nominal_shocks,
        k_min_mult=0.1, k_max_mult=10.0, z_sd_mult=4.0)


def _beta(n: int, alpha: float, rho: float, sigma: float,
          phi_quad: float = 0.0, phi_prop: float = 0.0) -> tf.Tensor:
    """Build a (n, 5) β tensor at the given anchor."""
    row = [alpha, rho, sigma, phi_quad, phi_prop]
    return tf.constant([row] * n, dtype=tf.float32)


def _nominal_beta(econ, shocks, n) -> tf.Tensor:
    return _beta(n, econ.production_elasticity, shocks.rho, shocks.sigma)


# --------------------------------------------------------------------- 1. interface

def test_dimensions(param_env):
    assert param_env.exo_dim() == 1
    assert param_env.endo_dim() == 1
    assert param_env.action_dim() == 1
    assert param_env.beta_dim() == 5
    assert param_env.state_dim() == 2


def test_bounds_match_nominal(param_env, nominal_env):
    assert param_env.k_min == nominal_env.k_min
    assert param_env.k_max == nominal_env.k_max
    assert param_env.z_min == nominal_env.z_min
    assert param_env.z_max == nominal_env.z_max
    assert param_env.k_star == nominal_env.k_star
    np.testing.assert_allclose(param_env.discount(), nominal_env.discount())


def test_nominal_with_cost_fields_strips_them(nominal_shocks):
    """If the nominal_econ accidentally has cost_convex > 0, the env still
    uses the frictionless k* for bounds (we don't want bounds to depend on
    a nominal friction setting)."""
    econ_with_frictions = EconomicParams(
        interest_rate=0.04, depreciation_rate=0.10,
        production_elasticity=0.5, cost_convex=0.5)
    env_frictional_nominal = ParameterizedBasicInvestmentEnv(
        nominal_econ=econ_with_frictions, nominal_shocks=nominal_shocks)
    env_frictionless_nominal = ParameterizedBasicInvestmentEnv(
        nominal_econ=EconomicParams(
            interest_rate=0.04, depreciation_rate=0.10,
            production_elasticity=0.5),
        nominal_shocks=nominal_shocks)
    np.testing.assert_allclose(
        env_frictional_nominal.k_star, env_frictionless_nominal.k_star)


# --------------------------------------------------------------------- 2. frictionless slice reproduces BasicInvestmentEnv

def _sample_states(env, n=8, seed=(1, 2)):
    s_endo = env.sample_initial_endogenous(n, seed=tf.constant(seed, dtype=tf.int32))
    s_exo  = env.sample_initial_exogenous(
        n, seed=tf.constant([seed[0] + 1, seed[1]], dtype=tf.int32))
    return s_endo, s_exo


def test_exogenous_transition_frictionless_matches_nominal(
    param_env, nominal_env, nominal_econ, nominal_shocks
):
    _, s_exo = _sample_states(nominal_env)
    eps  = tf.constant(
        [[0.1], [-0.4], [0.0], [1.2], [-1.0], [0.3], [0.7], [-0.2]])
    beta = _nominal_beta(nominal_econ, nominal_shocks, n=s_exo.shape[0])
    z_p = param_env.exogenous_transition(s_exo, eps, beta)
    z_n = nominal_env.exogenous_transition(s_exo, eps)
    np.testing.assert_allclose(z_p.numpy(), z_n.numpy(), rtol=1e-5)


def test_endogenous_transition_frictionless_matches_nominal(
    param_env, nominal_env, nominal_econ, nominal_shocks
):
    s_endo, s_exo = _sample_states(nominal_env)
    a = tf.constant([[0.5]] * s_endo.shape[0], dtype=tf.float32)
    beta = _nominal_beta(nominal_econ, nominal_shocks, n=s_endo.shape[0])
    k_p = param_env.endogenous_transition(s_endo, a, s_exo, beta)
    k_n = nominal_env.endogenous_transition(s_endo, a, s_exo)
    np.testing.assert_allclose(k_p.numpy(), k_n.numpy(), rtol=1e-5)


def test_reward_frictionless_matches_nominal(
    param_env, nominal_env, nominal_econ, nominal_shocks
):
    s_endo, s_exo = _sample_states(nominal_env)
    s = nominal_env.merge_state(s_endo, s_exo)
    a = tf.constant([[0.5]] * s.shape[0], dtype=tf.float32)
    beta = _nominal_beta(nominal_econ, nominal_shocks, n=s.shape[0])
    r_p = param_env.reward(s, a, beta)
    r_n = nominal_env.reward(s, a)
    np.testing.assert_allclose(r_p.numpy(), r_n.numpy(), rtol=1e-5)


def test_euler_residual_frictionless_matches_nominal(
    param_env, nominal_env, nominal_econ, nominal_shocks
):
    s_endo, s_exo = _sample_states(nominal_env)
    s = nominal_env.merge_state(s_endo, s_exo)
    a = tf.constant([[0.3]] * s.shape[0], dtype=tf.float32)
    s_endo_next = nominal_env.endogenous_transition(s_endo, a, s_exo)
    eps = tf.zeros_like(s_exo)
    s_exo_next = nominal_env.exogenous_transition(s_exo, eps)
    s_next = nominal_env.merge_state(s_endo_next, s_exo_next)
    a_next = tf.constant([[0.2]] * s.shape[0], dtype=tf.float32)

    beta = _nominal_beta(nominal_econ, nominal_shocks, n=s.shape[0])
    r_p = param_env.euler_residual(s, a, s_next, a_next, beta)
    r_n = nominal_env.euler_residual(s, a, s_next, a_next)
    np.testing.assert_allclose(r_p.numpy(), r_n.numpy(), rtol=1e-4)


# --------------------------------------------------------------------- 3. Euler residual ≈ 0 at the analytical policy (φ=0 slice)

def test_euler_residual_zero_at_analytical_policy_frictionless(
    nominal_econ, nominal_shocks
):
    """At φ_quad = φ_prop = 0 and z' = E[z'|z], plugging in the closed-form
    analytical policy should give ≈ 0 residual on the interior."""
    env = ParameterizedBasicInvestmentEnv(
        nominal_econ=nominal_econ, nominal_shocks=nominal_shocks,
        k_min_mult=0.05, k_max_mult=30.0)

    n = 128
    rng = np.random.default_rng(0)
    alphas = rng.uniform(0.30, 0.75, size=n).astype(np.float32)
    rhos   = rng.uniform(0.20, 0.80, size=n).astype(np.float32)
    sigmas = rng.uniform(0.05, 0.40, size=n).astype(np.float32)
    zeros  = np.zeros(n, dtype=np.float32)
    beta = tf.stack(
        [tf.constant(alphas), tf.constant(rhos), tf.constant(sigmas),
         tf.constant(zeros),  tf.constant(zeros)],
        axis=-1)

    k0 = rng.uniform(env.k_min * 5.0, env.k_max * 0.3, size=n).astype(np.float32)
    z0 = rng.uniform(env.z_min * 1.5, env.z_max * 0.7, size=n).astype(np.float32)
    s = tf.stack([tf.constant(k0), tf.constant(z0)], axis=-1)

    # Build the unclipped k' to filter interior samples
    alpha = beta[..., 0:1]
    rho   = beta[..., 1:2]
    sigma = beta[..., 2:3]
    z = s[..., 1:2]
    log_z = tf.math.log(tf.maximum(z, 1e-8))
    log_ez = (1.0 - rho) * env.mu + rho * log_z + 0.5 * sigma * sigma
    log_brk = (tf.math.log(alpha) + log_ez
               - tf.math.log(env.r_rate + env.delta_rate))
    kp_unclipped = tf.exp(log_brk / (1.0 - alpha))[..., 0].numpy()
    interior = (kp_unclipped > env.k_min * 1.01) & (kp_unclipped < env.k_max * 0.99)
    assert interior.sum() >= n // 2

    a = env.analytical_policy(s, beta)
    z_next = tf.exp(log_ez)
    k_next = env.endogenous_transition(s[..., 0:1], a, s[..., 1:2], beta)
    s_next = tf.concat([k_next, z_next], axis=-1)
    a_next = tf.zeros([n, 1], dtype=tf.float32)

    residual = env.euler_residual(s, a, s_next, a_next, beta).numpy()
    np.testing.assert_allclose(
        residual[interior], np.zeros(int(interior.sum())), atol=1e-4)


# --------------------------------------------------------------------- 4. Frictions actually move things

def test_friction_cost_helper_signs(param_env):
    """Sanity: positive φ → positive cost; zero φ → zero cost."""
    k = tf.constant([1.0, 1.0])
    I = tf.constant([0.3, -0.3])
    # All zero
    assert float(tf.reduce_max(_friction_cost(k, I,
        tf.constant([0.0, 0.0]), tf.constant([0.0, 0.0])))) == pytest.approx(0.0)
    # Pure quadratic
    pq = _friction_cost(k, I, tf.constant([1.0, 1.0]), tf.constant([0.0, 0.0]))
    np.testing.assert_allclose(pq.numpy(), [0.045, 0.045], rtol=1e-5)
    # Pure proportional — symmetric in |I|
    pp = _friction_cost(k, I, tf.constant([0.0, 0.0]), tf.constant([0.2, 0.2]))
    np.testing.assert_allclose(pp.numpy(), [0.06, 0.06], rtol=1e-5)


def test_reward_drops_with_frictions(param_env, nominal_econ, nominal_shocks):
    """At fixed (s, a), the reward with positive φ should be strictly lower
    than the frictionless reward by exactly the adjustment cost."""
    s_endo, s_exo = _sample_states(param_env)
    s = param_env.merge_state(s_endo, s_exo)
    a = tf.constant([[0.5]] * s.shape[0], dtype=tf.float32)

    beta_free = _nominal_beta(nominal_econ, nominal_shocks, n=s.shape[0])
    beta_fric = _beta(
        n=s.shape[0],
        alpha=nominal_econ.production_elasticity,
        rho=nominal_shocks.rho, sigma=nominal_shocks.sigma,
        phi_quad=0.3, phi_prop=0.05)

    r_free = param_env.reward(s, a, beta_free).numpy()
    r_fric = param_env.reward(s, a, beta_fric).numpy()
    diff = r_free - r_fric                # should equal the adjustment cost
    # Independently compute the expected adjustment cost
    k = s[..., 0:1]
    I_effective = a[..., 0:1] + tf.zeros_like(k)   # no clipping at small a
    expected_adj = _friction_cost(
        k, I_effective,
        tf.constant([[0.3]] * s.shape[0]),
        tf.constant([[0.05]] * s.shape[0]))
    np.testing.assert_allclose(diff, expected_adj.numpy()[..., 0], rtol=1e-4)


def test_euler_residual_changes_with_frictions(
    param_env, nominal_econ, nominal_shocks
):
    """Switching frictions on perturbs the Euler residual."""
    s_endo, s_exo = _sample_states(param_env)
    s = param_env.merge_state(s_endo, s_exo)
    a = tf.constant([[0.3]] * s.shape[0], dtype=tf.float32)
    s_endo_next = param_env.endogenous_transition(
        s_endo, a, s_exo, _nominal_beta(nominal_econ, nominal_shocks, n=s.shape[0]))
    eps = tf.zeros_like(s_exo)
    s_exo_next = param_env.exogenous_transition(
        s_exo, eps, _nominal_beta(nominal_econ, nominal_shocks, n=s.shape[0]))
    s_next = param_env.merge_state(s_endo_next, s_exo_next)
    a_next = tf.constant([[0.2]] * s.shape[0], dtype=tf.float32)

    beta_free = _nominal_beta(nominal_econ, nominal_shocks, n=s.shape[0])
    beta_fric = _beta(
        n=s.shape[0],
        alpha=nominal_econ.production_elasticity,
        rho=nominal_shocks.rho, sigma=nominal_shocks.sigma,
        phi_quad=0.5, phi_prop=0.1)

    r_free = param_env.euler_residual(s, a, s_next, a_next, beta_free).numpy()
    r_fric = param_env.euler_residual(s, a, s_next, a_next, beta_fric).numpy()
    # The frictional residual is strictly different from frictionless
    assert np.max(np.abs(r_fric - r_free)) > 1e-3


# --------------------------------------------------------------------- 5. analytical_policy guard

def test_analytical_policy_raises_on_frictions(param_env, nominal_econ, nominal_shocks):
    s_endo, s_exo = _sample_states(param_env)
    s = param_env.merge_state(s_endo, s_exo)
    beta = _beta(
        n=s.shape[0],
        alpha=nominal_econ.production_elasticity,
        rho=nominal_shocks.rho, sigma=nominal_shocks.sigma,
        phi_quad=0.1, phi_prop=0.0)
    with pytest.raises(ValueError, match="frictionless slice"):
        _ = param_env.analytical_policy(s, beta)


def test_analytical_policy_matches_compute_frictionless_policy(
    param_env, nominal_env, nominal_econ, nominal_shocks
):
    from src.v2.environments.basic_investment import compute_frictionless_policy

    n = 32
    rng = np.random.default_rng(42)
    z = rng.uniform(nominal_env.z_min * 1.2, nominal_env.z_max * 0.8, size=n).astype(np.float32)
    k = rng.uniform(nominal_env.k_min * 1.5, nominal_env.k_max * 0.5, size=n).astype(np.float32)
    s = tf.stack([tf.constant(k), tf.constant(z)], axis=-1)
    beta = _nominal_beta(nominal_econ, nominal_shocks, n=n)

    I_param = param_env.analytical_policy(s, beta).numpy().ravel()
    kp_ref  = compute_frictionless_policy(z.astype(np.float64), nominal_econ, nominal_shocks)
    kp_ref  = np.clip(kp_ref, param_env.k_min, param_env.k_max)
    I_ref   = (kp_ref - (1.0 - nominal_econ.depreciation_rate) * k).astype(np.float32)
    np.testing.assert_allclose(I_param, I_ref, rtol=1e-4, atol=1e-4)
