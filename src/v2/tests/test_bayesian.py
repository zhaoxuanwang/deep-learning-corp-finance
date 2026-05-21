"""Tests for the Bayesian estimation pipeline (Phase 1 frictionless basic).

Covers:
  1. Spec metadata (parameter names, observation keys, filter/sampler tags).
  2. Filter+sampler validation: (particle, nuts) is a hard error.
  3. Synthetic panel determinism under fixed seed.
  4. Log-likelihood evaluates to finite and is differentiable in β.
  5. Smoke MCMC run returns shapes + finite diagnostics.
  6. Coverage check smoke runs end-to-end and returns valid coverage rates.
  7. Reproducibility: identical seeds → bit-identical posterior samples.

Slow tests (involve actual MCMC) are marked ``slow``.  Fast tests run in
seconds; slow tests in tens of seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    EconomicParams,
    ShockParams,
)
from src.v2.estimation import (
    BayesianCoverageConfig,
    BayesianRunConfig,
    BayesianSpec,
    FilterKind,
    SamplerKind,
    run_coverage_check,
    run_mcmc,
)
from src.v2.estimation.bayesian import (
    _validate_filter_sampler_combo,
)
from src.v2.estimation.bayesian_basic_investment import (
    bayesian_true_beta,
    make_bayesian_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frictionless_env() -> BasicInvestmentEnv:
    return BasicInvestmentEnv(
        econ_params=EconomicParams(
            interest_rate=0.04,
            depreciation_rate=0.15,
            production_elasticity=0.60,
            cost_convex=0.0,
            cost_fixed=0.0,
        ),
        shock_params=ShockParams(mu=0.0, rho=0.70, sigma=0.15),
    )


@pytest.fixture(scope="module")
def spec(frictionless_env) -> BayesianSpec:
    return make_bayesian_spec(frictionless_env)


@pytest.fixture(scope="module")
def beta_true(frictionless_env) -> dict[str, float]:
    return bayesian_true_beta(frictionless_env, sigma_eta=0.05)


# ---------------------------------------------------------------------------
# Fast tests (no MCMC)
# ---------------------------------------------------------------------------

def test_spec_metadata_matches_expected_shape(spec):
    assert spec.parameter_names == ("alpha", "rho", "sigma_epsilon", "sigma_eta")
    assert spec.observation_keys == ("y", "log_k")
    assert spec.filter_kind is FilterKind.KALMAN
    assert spec.sampler_kind is SamplerKind.NUTS


def test_make_bayesian_spec_rejects_frictional_env():
    env = BasicInvestmentEnv(
        econ_params=EconomicParams(cost_convex=0.05, cost_fixed=0.0),
        shock_params=ShockParams(mu=0.0, rho=0.7, sigma=0.15),
    )
    with pytest.raises(ValueError, match="frictionless"):
        make_bayesian_spec(env)


def test_validate_filter_sampler_rejects_particle_nuts():
    # The one combination NUTS cannot consume.
    with pytest.raises(ValueError, match="Incompatible filter/sampler"):
        _validate_filter_sampler_combo(FilterKind.PARTICLE, SamplerKind.NUTS)

    # All three valid pairs must pass.
    _validate_filter_sampler_combo(FilterKind.KALMAN,   SamplerKind.NUTS)
    _validate_filter_sampler_combo(FilterKind.KALMAN,   SamplerKind.RW_MH)
    _validate_filter_sampler_combo(FilterKind.PARTICLE, SamplerKind.RW_MH)


def test_synthesize_panel_is_seeded_and_shapes_match_spec(spec, beta_true):
    panel_a = spec.synthesize_panel_fn(beta_true, n_firms=5, horizon=8,
                                       burn_in=4, seed=(1, 2))
    panel_b = spec.synthesize_panel_fn(beta_true, n_firms=5, horizon=8,
                                       burn_in=4, seed=(1, 2))
    panel_c = spec.synthesize_panel_fn(beta_true, n_firms=5, horizon=8,
                                       burn_in=4, seed=(3, 4))

    # Shapes per spec contract.
    for key in spec.observation_keys:
        assert panel_a[key].shape == (5, 8, 1)
        assert panel_a[key].dtype == np.float32

    # Reproducibility under fixed seed; divergence under alt seed.
    np.testing.assert_allclose(panel_a["y"],     panel_b["y"],     atol=1e-12)
    np.testing.assert_allclose(panel_a["log_k"], panel_b["log_k"], atol=1e-12)
    assert not np.allclose(panel_a["y"], panel_c["y"])


def test_log_likelihood_is_finite_and_differentiable(spec, beta_true):
    panel = spec.synthesize_panel_fn(beta_true, n_firms=8, horizon=10,
                                     burn_in=4, seed=(5, 7))
    data = {k: tf.constant(panel[k]) for k in spec.observation_keys}

    params = {k: tf.Variable(v, dtype=tf.float32) for k, v in beta_true.items()}
    with tf.GradientTape() as tape:
        ll = spec.log_likelihood_fn(params, data)

    assert np.isfinite(float(ll))
    grads = tape.gradient(ll, list(params.values()))
    assert all(g is not None and np.all(np.isfinite(g.numpy())) for g in grads)
    # Gradients should not be all zero (some signal in the data).
    assert any(float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads)


# ---------------------------------------------------------------------------
# Smoke tests (involve actual MCMC; marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_mcmc_smoke_returns_finite_diagnostics(spec, beta_true):
    panel = spec.synthesize_panel_fn(beta_true, n_firms=20, horizon=15,
                                     burn_in=10, seed=(11, 13))
    observed = {k: panel[k] for k in spec.observation_keys}
    cfg = BayesianRunConfig(n_chains=2, n_warmup=80, n_samples=80,
                            target_accept_prob=0.80,
                            master_seed=(20, 26))
    result = run_mcmc(spec, observed, cfg, seed=(20, 26))

    for k in spec.parameter_names:
        assert result.posterior_samples[k].shape == (80, 2)
        assert np.all(np.isfinite(result.posterior_samples[k]))
        assert np.isfinite(result.r_hat[k])
        assert result.ess[k] > 0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.wall_time_sec > 0
    assert result.metadata["filter_kind"] == "kalman"
    assert result.metadata["sampler_kind"] == "nuts"


@pytest.mark.slow
def test_run_mcmc_is_reproducible_under_fixed_seed(spec, beta_true):
    """§2.9 reproducibility contract: same master seed → identical samples."""
    panel = spec.synthesize_panel_fn(beta_true, n_firms=15, horizon=10,
                                     burn_in=4, seed=(31, 41))
    observed = {k: panel[k] for k in spec.observation_keys}
    cfg = BayesianRunConfig(n_chains=2, n_warmup=40, n_samples=40,
                            target_accept_prob=0.8)

    result_a = run_mcmc(spec, observed, cfg, seed=(99, 7))
    result_b = run_mcmc(spec, observed, cfg, seed=(99, 7))

    for k in spec.parameter_names:
        np.testing.assert_allclose(
            result_a.posterior_samples[k],
            result_b.posterior_samples[k],
            atol=1e-6,
            err_msg=f"Posterior samples for {k} should be bit-identical under "
                    f"fixed seed, but diverge.",
        )


@pytest.mark.slow
def test_run_coverage_check_smoke_returns_summary(spec):
    """Tiny coverage check: not statistically meaningful, just shape-checks."""
    cfg = BayesianRunConfig(n_chains=2, n_warmup=40, n_samples=40,
                            target_accept_prob=0.8)
    cov_cfg = BayesianCoverageConfig(n_replicates=2, n_firms=10, horizon=8,
                                      burn_in=4, credible_level=0.95)
    result = run_coverage_check(spec, cfg, cov_cfg, master_seed=(20, 26))

    assert set(result.coverage_per_parameter.keys()) == set(spec.parameter_names)
    for k, frac in result.coverage_per_parameter.items():
        assert 0.0 <= frac <= 1.0
    assert len(result.per_replicate_intervals) == 2
    assert len(result.per_replicate_beta0) == 2
    assert len(result.per_replicate_diagnostics) == 2
    assert result.credible_level == 0.95
    assert result.wall_time_sec > 0
