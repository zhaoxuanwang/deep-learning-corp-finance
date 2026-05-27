"""Tests for the neural-surrogate Bayesian factory (Phase 2 frictionless slice).

Covers:
  1. Spec metadata (parameter names, observation keys, filter/sampler tags).
  2. Truncated Sigmoid bijector range + Jacobian sanity.
  3. Synthetic panel determinism under fixed seed.
  4. EKF log-likelihood: shape contracts (scalar + batched β), finiteness,
     differentiability through the inner ForwardAccumulator JVP.
  5. Diagnostic helper returns the closed-form analytical slope / intercept.
  6. **Strong integration test**: with a *mock* policy network that implements
     the exact closed-form k'(z; β), the EKF reduces to a standard Kalman
     filter with zero linearization error.  A 100-step NUTS run on a small
     synthetic panel should put true β inside the 95% credible interval.
  7. Reproducibility: identical seeds → bit-identical posterior samples.

The mock-policy test is the critical correctness gate: it isolates the
EKF/Kalman recursion from any NN approximation error and proves the
machinery is unbiased before we plug in a real surrogate.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pytest
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    EconomicParams,
    ShockParams,
)
from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
)
from src.v2.estimation import (
    BayesianCoverageConfig,
    BayesianRunConfig,
    BayesianSpec,
    FilterKind,
    PosteriorPredictiveConfig,
    SamplerKind,
    run_coverage_check,
    run_mcmc,
    run_posterior_predictive_check,
    run_prior_sensitivity,
)
from src.v2.estimation.bayesian_basic_investment import (
    OBSERVATION_KEYS,
    PARAMETER_NAMES,
    _truncated_sigmoid_bijector,
    derive_env_bounds_from_panel,
    diagnose_nn_linear_slope,
    generate_frictionless_panel,
    load_policy_with_normalizer,
    make_closed_form_bayesian_spec,
    make_neural_bayesian_spec,
    neural_true_beta,
    save_policy_with_normalizer,
)
from src.v2.estimation.beta_sampler import DEFAULT_UNIFORM_BOUNDS
from src.v2.networks.policy import ParameterizedPolicyNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def param_env() -> ParameterizedBasicInvestmentEnv:
    return ParameterizedBasicInvestmentEnv(
        nominal_econ=EconomicParams(
            interest_rate=0.04, depreciation_rate=0.10, production_elasticity=0.5,
        ),
        nominal_shocks=ShockParams(rho=0.5, sigma=0.24, mu=0.0),
        k_min_mult=0.1, k_max_mult=8.0, z_sd_mult=3.0,
    )


@pytest.fixture(scope="module")
def untrained_nn(param_env) -> ParameterizedPolicyNetwork:
    nn = ParameterizedPolicyNetwork(
        state_dim=param_env.state_dim(),
        action_dim=param_env.action_dim(),
        beta_dim=5,
        **param_env.action_spec(),
        n_layers=2, n_neurons=32, seed=(0, 0),
    )
    # Build with a dummy call so the layers materialize.
    nn(tf.zeros((1, 2)), tf.zeros((1, 5)))
    nn.trainable = False
    return nn


# ---------------------------------------------------------------------------
# 1. Spec metadata
# ---------------------------------------------------------------------------

def test_spec_metadata(param_env, untrained_nn):
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    assert isinstance(spec, BayesianSpec)
    assert spec.parameter_names == PARAMETER_NAMES
    assert spec.observation_keys == OBSERVATION_KEYS
    assert spec.filter_kind == FilterKind.KALMAN  # EKF dispatches as Kalman
    assert spec.sampler_kind == SamplerKind.NUTS
    # Bijector keys match parameter names.
    assert set(spec.bijector.bijectors.keys()) == set(PARAMETER_NAMES)


def test_param_env_strips_frictions_from_nominal():
    """``ParameterizedBasicInvestmentEnv`` structurally guarantees a
    frictionless nominal (frictions enter via per-sample β only). This is
    why ``make_neural_bayesian_spec`` does not need to gate on env-baked
    cost_convex / cost_fixed — confirm the env contract directly so we
    don't need a runtime check in the factory.
    """
    env = ParameterizedBasicInvestmentEnv(
        nominal_econ=EconomicParams(
            interest_rate=0.04, depreciation_rate=0.10,
            production_elasticity=0.5,
            cost_convex=0.5,   # supplied but should be discarded
            cost_fixed=1.0,
        ),
        nominal_shocks=ShockParams(rho=0.5, sigma=0.24, mu=0.0),
        k_min_mult=0.1, k_max_mult=8.0, z_sd_mult=3.0,
    )
    assert float(env.nominal_econ.cost_convex) == 0.0
    assert float(env.nominal_econ.cost_fixed)  == 0.0


# ---------------------------------------------------------------------------
# 2. Truncated Sigmoid bijector
# ---------------------------------------------------------------------------

def test_truncated_sigmoid_bijector_range():
    bij = _truncated_sigmoid_bijector(0.2, 0.85)
    # At ℝ extremes, output approaches the box endpoints.
    very_negative = bij.forward(tf.constant(-10.0, tf.float32))
    very_positive = bij.forward(tf.constant(10.0, tf.float32))
    assert float(very_negative) > 0.2 and float(very_negative) < 0.21
    assert float(very_positive) > 0.84 and float(very_positive) < 0.85
    # At 0, output is the box midpoint.
    midpoint = bij.forward(tf.constant(0.0, tf.float32))
    assert abs(float(midpoint) - 0.5 * (0.2 + 0.85)) < 1e-5


def test_truncated_sigmoid_bijector_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="high > low"):
        _truncated_sigmoid_bijector(0.5, 0.5)
    with pytest.raises(ValueError, match="high > low"):
        _truncated_sigmoid_bijector(0.5, 0.3)


# ---------------------------------------------------------------------------
# 3. Panel synthesizer
# ---------------------------------------------------------------------------

def test_synthesize_panel_shapes_and_keys(param_env, untrained_nn):
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, n_firms=8, horizon=12,
                                     burn_in=5, seed=(42, 0))
    assert set(panel.keys()) == set(OBSERVATION_KEYS) | {"metadata"}
    for key in OBSERVATION_KEYS:
        assert panel[key].shape == (8, 12, 1)
        assert panel[key].dtype == np.float32
    # log_k_next[:, t-1] should equal log_k[:, t] for t in 1..T-1
    # (consecutive periods within the same trajectory).
    assert np.allclose(panel["log_k_next"][:, :-1, 0], panel["log_k"][:, 1:, 0])


def test_synthesize_panel_determinism(param_env, untrained_nn):
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    beta_true = neural_true_beta(param_env)
    p1 = spec.synthesize_panel_fn(beta_true, n_firms=4, horizon=6,
                                  burn_in=5, seed=(7, 19))
    p2 = spec.synthesize_panel_fn(beta_true, n_firms=4, horizon=6,
                                  burn_in=5, seed=(7, 19))
    for k in OBSERVATION_KEYS:
        np.testing.assert_array_equal(p1[k], p2[k])


# ---------------------------------------------------------------------------
# 4. EKF log-likelihood
# ---------------------------------------------------------------------------

def _data_tensors_from_panel(panel: dict, spec: BayesianSpec):
    return {k: tf.constant(panel[k], tf.float32) for k in spec.observation_keys}


def _make_beta_dict_scalar(beta_true: Mapping[str, float]):
    return {k: tf.constant(float(v), tf.float32) for k, v in beta_true.items()}


def _make_beta_dict_batched(beta_true: Mapping[str, float], n_chains: int = 2):
    return {k: tf.constant([float(v)] * n_chains, tf.float32)
            for k, v in beta_true.items()}


def test_log_likelihood_evaluates_scalar(param_env, untrained_nn):
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, 4, 6, 5, seed=(1, 1))
    data = _data_tensors_from_panel(panel, spec)
    ll = spec.log_likelihood_fn(_make_beta_dict_scalar(beta_true), data)
    assert ll.shape == ()
    assert tf.math.is_finite(ll)


def test_log_likelihood_evaluates_batched(param_env, untrained_nn):
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, 4, 6, 5, seed=(1, 1))
    data = _data_tensors_from_panel(panel, spec)
    ll = spec.log_likelihood_fn(_make_beta_dict_batched(beta_true, 3), data)
    assert ll.shape == (3,)
    assert tf.reduce_all(tf.math.is_finite(ll))


def test_log_likelihood_differentiable_in_beta(param_env, untrained_nn):
    """Outer reverse-mode tape must backprop through the inner forward-mode JVP."""
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, 4, 6, 5, seed=(1, 1))
    data = _data_tensors_from_panel(panel, spec)

    # Differentiate only over the sampler-side params (spec.parameter_names);
    # neural_true_beta also returns the derived investment_mean for caller
    # convenience but the likelihood does not consume it.
    beta_t = {k: tf.constant(float(beta_true[k]), tf.float32) for k in spec.parameter_names}
    with tf.GradientTape() as tape:
        for t in beta_t.values():
            tape.watch(t)
        ll = spec.log_likelihood_fn(beta_t, data)
    grads = tape.gradient(ll, list(beta_t.values()))
    assert all(g is not None for g in grads), \
        "EKF must be autodiff-compatible end-to-end for NUTS to work."
    assert all(tf.math.is_finite(g) for g in grads)


# ---------------------------------------------------------------------------
# 5. Diagnostic helper
# ---------------------------------------------------------------------------

def test_diagnose_nn_linear_slope_returns_analytical_reference(param_env, untrained_nn):
    beta_true = neural_true_beta(param_env)
    diag = diagnose_nn_linear_slope(param_env, untrained_nn, beta_true)
    # The analytical reference is independent of the NN, so it should match
    # ρ / (1 - α) regardless of NN training state.
    expected_slope = beta_true["rho"] / (1.0 - beta_true["alpha"])
    assert abs(diag["slope_analytical"] - expected_slope) < 1e-6
    # κ(β) = (log α + σ_ε²/2 - log(r + δ)) / (1 - α)
    a = beta_true["alpha"]; s = beta_true["sigma_epsilon"]
    expected_intercept = (
        np.log(a) + 0.5 * s ** 2
        - np.log(param_env.r_rate + param_env.delta_rate)
    ) / (1.0 - a)
    assert abs(diag["intercept_analytical"] - expected_intercept) < 1e-6


# ---------------------------------------------------------------------------
# 5b. Standalone panel generator + bounds derivation.
# ---------------------------------------------------------------------------

def _calibrated(env):
    return {
        "interest_rate":     env.r_rate,
        "depreciation_rate": env.delta_rate,
        "mu":                env.mu,
    }


def test_generate_frictionless_panel_shapes_and_keys(param_env):
    beta = neural_true_beta(param_env)
    panel = generate_frictionless_panel(
        beta,
        n_firms=64, horizon=20, burn_in=30,
        seed=(7, 19),
        **_calibrated(param_env),
    )
    # Observation-ready output; latent z is dropped to prevent leakage.
    assert set(panel.keys()) == {"y", "log_k", "log_k_next"}
    for key in ("y", "log_k", "log_k_next"):
        assert panel[key].shape == (64, 20)


def test_generate_frictionless_panel_determinism(param_env):
    """Same master seed → bit-identical panel."""
    beta = neural_true_beta(param_env)
    p1 = generate_frictionless_panel(
        beta, n_firms=16, horizon=8, burn_in=10, seed=(7, 19),
        **_calibrated(param_env),
    )
    p2 = generate_frictionless_panel(
        beta, n_firms=16, horizon=8, burn_in=10, seed=(7, 19),
        **_calibrated(param_env),
    )
    for key in ("y", "log_k", "log_k_next"):
        np.testing.assert_array_equal(p1[key], p2[key])


def test_generate_frictionless_panel_consecutive_timing(param_env):
    """log_k_next[:, t-1] equals log_k[:, t]: same trajectory across time."""
    beta = neural_true_beta(param_env)
    panel = generate_frictionless_panel(
        beta, n_firms=32, horizon=15, burn_in=30, seed=(101, 202),
        **_calibrated(param_env),
    )
    np.testing.assert_allclose(panel["log_k_next"][:, :-1],
                                panel["log_k"][:, 1:], rtol=0, atol=0)


def test_generate_frictionless_panel_residuals_match_dgp(param_env):
    """Empirical residual statistics match the DGP (production_mean, std,
    investment_mean, std) within Monte Carlo error.

    Tests the noise scheme added in the (μ, σ) refactor. With latent z
    dropped from output, we recover log z from the noiseless production
    relation by setting production_std=0 and reading log z = y - α log k.
    """
    # Noiseless production residual → log_z is exactly recoverable.
    beta = dict(neural_true_beta(param_env))
    beta.update(production_mean=0.0, production_std=0.0,
                investment_mean=-0.10, investment_std=0.05)
    panel = generate_frictionless_panel(
        beta, n_firms=512, horizon=60, burn_in=30, seed=(11, 13),
        **_calibrated(param_env),
    )
    alpha = beta["alpha"]; rho = beta["rho"]; sigma = beta["sigma_epsilon"]
    r = param_env.r_rate; delta = param_env.delta_rate
    kappa = (np.log(alpha) + 0.5 * sigma ** 2 - np.log(r + delta)) / (1.0 - alpha)
    slope = rho / (1.0 - alpha)

    # log z_t recovered from y_t = log z_t + α log k_t (production_std=0).
    log_z = panel["y"] - alpha * panel["log_k"]
    # Investment residual: log_k_next[t] - (slope * log_z[t] + κ) should be
    # iid N(investment_mean, investment_std²).
    inv_residual = panel["log_k_next"] - (slope * log_z + kappa)
    assert abs(float(inv_residual.mean()) - beta["investment_mean"]) < 0.01
    assert abs(float(inv_residual.std())  - beta["investment_std"])  < 0.01


def test_derive_env_bounds_from_panel_basic(param_env):
    beta = neural_true_beta(param_env)
    panel = generate_frictionless_panel(
        beta, n_firms=128, horizon=40, burn_in=30, seed=(1, 2),
        **_calibrated(param_env),
    )
    k = np.exp(panel["log_k"])
    bounds = derive_env_bounds_from_panel(k, k_margin=1.2, mu=0.0)
    assert set(bounds.keys()) == {"k_min", "k_max", "z_min", "z_max"}
    # k bounds include the panel with margin.
    assert bounds["k_min"] < float(k.min())
    assert bounds["k_max"] > float(k.max())
    # z bounds are exp(±half_width=1) around μ=0 by default.
    assert abs(np.log(bounds["z_max"]) - 1.0) < 1e-6
    assert abs(np.log(bounds["z_min"]) + 1.0) < 1e-6
    assert bounds["z_max"] / bounds["z_min"] < 10.0


def test_derive_env_bounds_z_independent_of_panel(param_env):
    """z bounds must NOT depend on panel content (z is latent at inference)."""
    beta = neural_true_beta(param_env)
    p1 = generate_frictionless_panel(
        beta, n_firms=32, horizon=20, burn_in=30, seed=(1, 2),
        **_calibrated(param_env),
    )
    p2 = generate_frictionless_panel(
        beta, n_firms=32, horizon=20, burn_in=30, seed=(99, 99),
        **_calibrated(param_env),
    )
    b1 = derive_env_bounds_from_panel(np.exp(p1["log_k"]), log_z_half_width=3.0)
    b2 = derive_env_bounds_from_panel(np.exp(p2["log_k"]), log_z_half_width=3.0)
    assert b1["z_min"] == b2["z_min"]
    assert b1["z_max"] == b2["z_max"]


def test_derive_env_bounds_rejects_invalid_inputs(param_env):
    beta = neural_true_beta(param_env)
    panel = generate_frictionless_panel(
        beta, n_firms=8, horizon=4, burn_in=5, seed=(3, 4),
        **_calibrated(param_env),
    )
    k = np.exp(panel["log_k"])
    with pytest.raises(ValueError, match="k_margin"):
        derive_env_bounds_from_panel(k, k_margin=1.0)
    with pytest.raises(ValueError, match="log_z_half_width"):
        derive_env_bounds_from_panel(k, log_z_half_width=0.0)


# ---------------------------------------------------------------------------
# 5c. ParameterizedBasicInvestmentEnv explicit-bounds path.
# ---------------------------------------------------------------------------

def test_env_explicit_bounds_overrides_multipliers():
    """When ``bounds=...`` is supplied, env exposes exactly those bounds
    (ignoring k_min_mult / k_max_mult / z_sd_mult), and I_min/I_max are
    derived consistently."""
    bounds = {"k_min": 2.0, "k_max": 50.0, "z_min": 0.5, "z_max": 1.8}
    env = ParameterizedBasicInvestmentEnv(
        nominal_econ=EconomicParams(
            interest_rate=0.04, depreciation_rate=0.10,
            production_elasticity=0.5,
        ),
        nominal_shocks=ShockParams(rho=0.5, sigma=0.24, mu=0.0),
        bounds=bounds,
    )
    assert env.k_min == pytest.approx(2.0)
    assert env.k_max == pytest.approx(50.0)
    assert env.z_min == pytest.approx(0.5)
    assert env.z_max == pytest.approx(1.8)
    one_minus_d = 1.0 - env.delta_rate
    assert env.I_min == pytest.approx(2.0 - one_minus_d * 50.0)
    assert env.I_max == pytest.approx(50.0 - one_minus_d * 2.0)
    # k_star is still computed from nominal β (used only for diagnostics).
    assert env.k_star > 0


def test_env_explicit_bounds_validates_inputs():
    base_econ = EconomicParams(interest_rate=0.04, depreciation_rate=0.10,
                                production_elasticity=0.5)
    base_shk  = ShockParams(rho=0.5, sigma=0.24, mu=0.0)
    with pytest.raises(ValueError, match="missing required keys"):
        ParameterizedBasicInvestmentEnv(
            nominal_econ=base_econ, nominal_shocks=base_shk,
            bounds={"k_min": 1.0, "k_max": 10.0},   # missing z_*
        )
    with pytest.raises(ValueError, match="k_min < k_max"):
        ParameterizedBasicInvestmentEnv(
            nominal_econ=base_econ, nominal_shocks=base_shk,
            bounds={"k_min": 10.0, "k_max": 5.0, "z_min": 0.5, "z_max": 2.0},
        )


def test_env_explicit_bounds_initial_state_samplers_use_overridden_box():
    bounds = {"k_min": 2.0, "k_max": 50.0, "z_min": 0.5, "z_max": 1.8}
    env = ParameterizedBasicInvestmentEnv(
        nominal_econ=EconomicParams(interest_rate=0.04, depreciation_rate=0.10,
                                     production_elasticity=0.5),
        nominal_shocks=ShockParams(rho=0.5, sigma=0.24, mu=0.0),
        bounds=bounds,
    )
    k0 = env.sample_initial_endogenous(2048, seed=tf.constant([0, 0], tf.int32)).numpy()
    z0 = env.sample_initial_exogenous(2048, seed=tf.constant([1, 0], tf.int32)).numpy()
    # Samples respect the explicit box.
    assert float(k0.min()) >= bounds["k_min"] - 1e-5
    assert float(k0.max()) <= bounds["k_max"] + 1e-5
    assert float(z0.min()) >= bounds["z_min"] - 1e-5
    assert float(z0.max()) <= bounds["z_max"] + 1e-5


# ---------------------------------------------------------------------------
# 6. Strong integration test: mock NN = exact closed-form.
# ---------------------------------------------------------------------------

class _ClosedFormMockPolicy(tf.Module):
    """Stand-in for ParameterizedPolicyNetwork that returns the analytical
    optimal investment ``I*(s; β) = k'(z; β) - (1 - δ) k`` exactly.

    Has the same call signature as ``ParameterizedPolicyNetwork``:
        ``policy(s, beta, training=False) -> action``
    with ``s = [k, z]``, ``beta = [α, ρ, σ_ε, φ_q, φ_p]``.

    Importantly: this mock has **no normalizer** — that's deliberate. The
    EKF and panel synthesizer only call ``policy(s, beta)``, so they
    don't care whether a normalizer exists. The save/load helpers are
    only relevant when a real ``ParameterizedPolicyNetwork`` is used.
    """

    def __init__(self, env: ParameterizedBasicInvestmentEnv):
        super().__init__()
        self._r = tf.constant(float(env.r_rate), tf.float32)
        self._delta = tf.constant(float(env.delta_rate), tf.float32)
        self._mu = tf.constant(float(env.mu), tf.float32)
        # Match Keras call signature for compatibility with the factory.
        self.trainable = False
        # The factory's EKF reads `policy_nn.normalizer.mean` etc. via the
        # diagnostic helper only — not in the EKF likelihood. To avoid
        # confusion, expose a tiny stub normalizer with frozen identity.
        class _IdentityNormalizer:
            mean = tf.constant([0.0] * 7, tf.float32)
            std  = tf.constant([1.0] * 7, tf.float32)
        self.normalizer = _IdentityNormalizer()

    def __call__(self, state: tf.Tensor, beta: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        del training
        k = state[..., 0]
        z = state[..., 1]
        alpha = beta[..., 0]
        rho   = beta[..., 1]
        sigma = beta[..., 2]
        # log E[z'|z] = (1-ρ) μ + ρ log z + 0.5 σ²
        log_z = tf.math.log(tf.maximum(z, 1e-12))
        log_ez = (1.0 - rho) * self._mu + rho * log_z + 0.5 * sigma * sigma
        # log k' = (log α + log E[z'|z] - log(r+δ)) / (1 - α)
        log_brk = tf.math.log(alpha) + log_ez - tf.math.log(self._r + self._delta)
        kp = tf.exp(log_brk / (1.0 - alpha))
        # action I* = k' - (1-δ) k
        a = kp - (1.0 - self._delta) * k
        return a[..., tf.newaxis]


def test_ekf_with_closed_form_mock_is_unbiased(param_env):
    """With an *exact* closed-form policy, EKF should be Kalman with no
    linearization error.  Verify the log-likelihood at β_true is higher
    than at perturbed β (a basic identification check)."""
    mock = _ClosedFormMockPolicy(param_env)
    spec = make_neural_bayesian_spec(param_env, mock)
    beta_true = neural_true_beta(param_env, production_std=0.05, investment_std=0.02)
    panel = spec.synthesize_panel_fn(beta_true, n_firms=30, horizon=15,
                                     burn_in=15, seed=(101, 202))
    data = _data_tensors_from_panel(panel, spec)

    ll_true = float(spec.log_likelihood_fn(
        _make_beta_dict_scalar(beta_true), data))

    # Perturb each parameter and check ll decreases (within numerical noise).
    for name in ("alpha", "rho", "sigma_epsilon"):
        bad = dict(beta_true)
        bad[name] = beta_true[name] + 0.10
        ll_bad = float(spec.log_likelihood_fn(
            _make_beta_dict_scalar(bad), data))
        assert ll_bad < ll_true + 1.0, (
            f"Perturbing {name} by +0.10 should decrease log-likelihood "
            f"(or at worst leave it unchanged within noise); got "
            f"ll_true={ll_true:.3f}, ll_bad={ll_bad:.3f}."
        )


@pytest.mark.slow
def test_ekf_with_closed_form_mock_recovers_beta_true(param_env):
    """With the exact closed-form policy, a short NUTS run should bracket
    every component of β_true inside its 95% credible interval.  This is
    the **only** test that exercises the full {EKF + NUTS + bijector +
    prior} stack; it serves as the correctness gate before plugging in a
    trained NN surrogate.
    """
    mock = _ClosedFormMockPolicy(param_env)
    spec = make_neural_bayesian_spec(param_env, mock)
    beta_true = neural_true_beta(param_env, production_std=0.05, investment_std=0.02)
    panel = spec.synthesize_panel_fn(beta_true, n_firms=40, horizon=15,
                                     burn_in=15, seed=(303, 404))
    observed = {k: panel[k] for k in spec.observation_keys}

    cfg = BayesianRunConfig(
        n_chains=2, n_warmup=200, n_samples=200,
        target_accept_prob=0.80, master_seed=(20, 26),
    )
    result = run_mcmc(spec, observed, cfg, seed=(20, 26))

    for name in PARAMETER_NAMES:
        samples = result.posterior_samples[name].reshape(-1)
        lo = float(np.quantile(samples, 0.025))
        hi = float(np.quantile(samples, 0.975))
        true_val = beta_true[name]
        assert lo <= true_val <= hi, (
            f"95% CI for {name} = [{lo:.4f}, {hi:.4f}] missed β_true = "
            f"{true_val:.4f}. (Closed-form mock should recover truth.)"
        )
        # Convergence diagnostics — loose at 200 samples × 2 chains.
        assert result.r_hat[name] < 1.30, (
            f"R-hat for {name} = {result.r_hat[name]:.3f} too high.")


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 8. Save/load roundtrip — guards against the StaticNormalizer serialization
#    bug (mean/std are tf.Module variables, not Keras-tracked).
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_nn_output(param_env, untrained_nn, tmp_path):
    """If only ``save_weights`` were used, the normalizer state would be
    lost on reload and the NN's forward pass would diverge.  The wrapper
    helpers ``save_policy_with_normalizer`` / ``load_policy_with_normalizer``
    persist both Keras weights and ``StaticNormalizer.mean/std`` together,
    so the reloaded NN must produce **bit-identical** outputs.
    """
    # First, give the normalizer non-trivial stats so loading them back
    # actually matters (default is mean=0, std=1).
    fake_data = tf.constant(np.random.RandomState(0).randn(256, 7), tf.float32)
    untrained_nn.normalizer.fit(fake_data)
    mean_before = untrained_nn.normalizer.mean.numpy().copy()
    std_before  = untrained_nn.normalizer.std.numpy().copy()

    s = tf.constant([[10.0, 1.0], [5.0, 0.5]], tf.float32)
    b = tf.constant([[0.5, 0.5, 0.24, 0.0, 0.0]] * 2, tf.float32)
    out_before = untrained_nn(s, b, training=False).numpy()

    weights_path = tmp_path / "nn.weights.h5"
    save_policy_with_normalizer(untrained_nn, weights_path)
    assert weights_path.exists()
    assert (tmp_path / "nn.weights.h5.normalizer.npz").exists()

    # Build a fresh NN with the same architecture (different seed → different
    # initial weights) so the roundtrip is meaningful.
    fresh = ParameterizedPolicyNetwork(
        state_dim=param_env.state_dim(),
        action_dim=param_env.action_dim(),
        beta_dim=5,
        **param_env.action_spec(),
        n_layers=2, n_neurons=32, seed=(99, 99),  # different seed
    )
    fresh(tf.zeros((1, 2)), tf.zeros((1, 5)))
    load_policy_with_normalizer(fresh, weights_path)

    np.testing.assert_array_equal(fresh.normalizer.mean.numpy(), mean_before)
    np.testing.assert_array_equal(fresh.normalizer.std.numpy(),  std_before)

    out_after = fresh(s, b, training=False).numpy()
    np.testing.assert_array_equal(out_before, out_after)


def test_load_without_normalizer_sidecar_raises(param_env, untrained_nn, tmp_path):
    """Loading raw ``save_weights`` output (no sidecar) must raise loudly.

    Silent failures here produce a degenerate NN that saturates at the
    action bounds — exactly the bug we are guarding against.
    """
    weights_path = tmp_path / "nn.weights.h5"
    untrained_nn.save_weights(weights_path)   # no sidecar -- the trap
    with pytest.raises(FileNotFoundError, match="Normalizer sidecar"):
        load_policy_with_normalizer(untrained_nn, weights_path)


# ---------------------------------------------------------------------------
# 9. Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility_same_master_seed_same_posterior(param_env, untrained_nn):
    """Same master seed → bit-identical posterior samples on CPU."""
    spec = make_neural_bayesian_spec(param_env, untrained_nn)
    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, n_firms=4, horizon=6,
                                     burn_in=5, seed=(42, 0))
    observed = {k: panel[k] for k in spec.observation_keys}
    cfg = BayesianRunConfig(
        n_chains=2, n_warmup=20, n_samples=20,
        target_accept_prob=0.80, master_seed=(20, 26),
    )
    r1 = run_mcmc(spec, observed, cfg, seed=(20, 26))
    r2 = run_mcmc(spec, observed, cfg, seed=(20, 26))
    for name in PARAMETER_NAMES:
        np.testing.assert_array_equal(
            r1.posterior_samples[name],
            r2.posterior_samples[name],
            err_msg=f"Posterior for {name} is not deterministic under fixed seed.",
        )


# ---------------------------------------------------------------------------
# 10. Closed-form spec factory + generic-runner smoke tests
# ---------------------------------------------------------------------------
#
# `make_closed_form_bayesian_spec` substitutes the analytical frictionless
# policy for the NN inside the same EKF likelihood. Because the closed-form
# policy is exactly linear in log z, the EKF reduces to a standard Kalman
# filter and runs ~10-30× faster than the NN path. The four tests below
# (a) gate the closed-form factory at the spec contract, then
# (b-d) re-house the three generic-runner smoke tests previously housed in
# the deleted Phase 1 test file, retargeted to this closed-form spec.

def test_make_closed_form_bayesian_spec_returns_valid_spec(param_env):
    spec = make_closed_form_bayesian_spec(param_env)
    assert isinstance(spec, BayesianSpec)
    assert spec.parameter_names == PARAMETER_NAMES
    assert spec.observation_keys == OBSERVATION_KEYS
    assert spec.filter_kind == FilterKind.KALMAN
    assert spec.sampler_kind == SamplerKind.NUTS
    assert set(spec.bijector.bijectors.keys()) == set(PARAMETER_NAMES)

    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, n_firms=4, horizon=6,
                                     burn_in=5, seed=(11, 22))
    observed = {k: panel[k] for k in spec.observation_keys}

    beta_tf = {k: tf.constant(float(v), tf.float32) for k, v in beta_true.items()}
    data_tf = {k: tf.constant(observed[k], tf.float32) for k in spec.observation_keys}
    ll = spec.log_likelihood_fn(beta_tf, data_tf)
    assert ll.shape == ()
    assert tf.math.is_finite(ll)


@pytest.mark.slow
def test_run_coverage_check_smoke_returns_summary(param_env):
    """Generic-runner smoke test (re-housed from the deleted Phase 1 suite)."""
    from src.v2.estimation import run_coverage_check

    spec = make_closed_form_bayesian_spec(param_env)
    run_cfg = BayesianRunConfig(
        n_chains=2, n_warmup=50, n_samples=50,
        target_accept_prob=0.80, master_seed=(20, 26),
    )
    cov_cfg = BayesianCoverageConfig(
        n_replicates=2, n_firms=4, horizon=6, burn_in=5,
        credible_level=0.95,
    )
    result = run_coverage_check(spec, run_cfg, cov_cfg, master_seed=(20, 26))
    assert set(result.coverage_per_parameter.keys()) == set(PARAMETER_NAMES)
    assert len(result.per_replicate_intervals) == 2
    assert len(result.per_replicate_beta0)     == 2
    for cov in result.coverage_per_parameter.values():
        assert 0.0 <= cov <= 1.0


@pytest.mark.slow
def test_run_posterior_predictive_check_smoke(param_env):
    """Generic-runner smoke test (re-housed from the deleted Phase 1 suite)."""
    spec = make_closed_form_bayesian_spec(param_env)
    beta_true = neural_true_beta(param_env)
    panel = spec.synthesize_panel_fn(beta_true, n_firms=6, horizon=8,
                                     burn_in=5, seed=(33, 44))
    observed = {k: panel[k] for k in spec.observation_keys}
    run_cfg = BayesianRunConfig(
        n_chains=2, n_warmup=50, n_samples=50,
        target_accept_prob=0.80, master_seed=(20, 26),
    )
    mcmc = run_mcmc(spec, observed, run_cfg, seed=(20, 26))
    ppc_cfg = PosteriorPredictiveConfig(n_draws=20, burn_in=5)
    result = run_posterior_predictive_check(
        spec, mcmc.posterior_samples, observed, ppc_cfg, seed=(77, 88),
    )
    assert set(result.observed_stats.keys()) == set(ppc_cfg.statistics)
    for stat in ppc_cfg.statistics:
        assert result.replicated_stats[stat].shape == (ppc_cfg.n_draws,)
        assert 0.0 <= result.bayes_p_values[stat] <= 1.0


@pytest.mark.slow
def test_run_prior_sensitivity_smoke(param_env):
    """Generic-runner smoke test (re-housed from the deleted Phase 1 suite)."""
    from functools import partial

    spec_factory = partial(make_closed_form_bayesian_spec, param_env)
    beta_true = neural_true_beta(param_env)
    panel = spec_factory().synthesize_panel_fn(
        beta_true, n_firms=4, horizon=6, burn_in=5, seed=(55, 66),
    )
    observed = {k: panel[k] for k in OBSERVATION_KEYS}
    run_cfg = BayesianRunConfig(
        n_chains=2, n_warmup=50, n_samples=50,
        target_accept_prob=0.80, master_seed=(20, 26),
    )
    variants = {
        "default": dict(production_std_prior_scale=0.5,
                         investment_std_prior_scale=0.5),
        "tight":   dict(production_std_prior_scale=0.1,
                         investment_std_prior_scale=0.1),
    }
    result = run_prior_sensitivity(
        spec_factory, variants, observed, run_cfg,
        seed=(99, 11), default_variant_name="default",
    )
    assert set(result.posterior_summaries.keys()) == set(variants.keys())
    for variant_summary in result.posterior_summaries.values():
        assert set(variant_summary.keys()) == set(PARAMETER_NAMES)
        for stats in variant_summary.values():
            assert stats["ci_lo"] <= stats["median"] <= stats["ci_hi"]
    assert set(result.movement_metrics.keys()) == set(variants.keys())
    for name in PARAMETER_NAMES:
        assert result.movement_metrics["default"][name] == pytest.approx(0.0, abs=1e-9)
