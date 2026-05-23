"""Generic Bayesian estimation for v2.

The core layer is model-agnostic.  It knows only about:
  - parameter vectors and priors
  - a filter that returns scalar ``log p(y | beta)``
  - an MCMC sampler that produces posterior draws from the joint log-target
  - a coverage-check loop over R synthetic replicates

All model-specific structure (state-space matrices, observation equation,
policy-function plug-ins) lives in per-env factory modules such as
``src/v2/estimation/bayesian_basic_investment.py``.  Those factories return
a ``BayesianSpec`` consumable by ``run_mcmc`` and ``run_coverage_check``.

Design (matches docs/paper/Bayesian.md Sections 1.2, 2.4, 2.6, 2.8, 2.9):

  - Filter and sampler are orthogonal:
      filter ∈ {kalman, particle}    — owns the likelihood
      sampler ∈ {nuts, rw_mh}        — owns the proposal mechanism
    The pairing (particle, nuts) is rejected: a particle-filter likelihood
    estimate is non-differentiable through the resampling step, so NUTS
    gradients are meaningless.  (Kalman, nuts), (Kalman, rw_mh), and
    (particle, rw_mh = PMMH) are valid.
  - Reproducibility (Bayesian.md §2.9): every RNG-consuming step takes a
    seed derived from one master pair via ``fold_in_seed``.  No global
    TF/NumPy state.
  - TFP is delegated to for the filter (LGSSM.log_prob) and the sampler
    (windowed_adaptive_nuts / RandomWalkMetropolis).  This module is glue,
    not a re-implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.v2.utils.seeding import fold_in_seed

tfd = tfp.distributions
tfb = tfp.bijectors


# ---------------------------------------------------------------------------
# Enums: filter / sampler choice
# ---------------------------------------------------------------------------

class FilterKind(str, Enum):
    """Which likelihood-evaluation method the spec uses."""
    KALMAN = "kalman"
    PARTICLE = "particle"


class SamplerKind(str, Enum):
    """Which MCMC kernel ``run_mcmc`` dispatches to."""
    NUTS = "nuts"
    RW_MH = "rw_mh"


_VALID_FILTER_SAMPLER_PAIRS: frozenset[tuple[FilterKind, SamplerKind]] = frozenset({
    (FilterKind.KALMAN,   SamplerKind.NUTS),
    (FilterKind.KALMAN,   SamplerKind.RW_MH),
    (FilterKind.PARTICLE, SamplerKind.RW_MH),  # PMMH
})


def _validate_filter_sampler_combo(filter_kind: FilterKind,
                                   sampler_kind: SamplerKind) -> None:
    """Hard-error on (particle, nuts).

    A particle-filter log-likelihood is unbiased but non-differentiable through
    the resampling step, so gradient-based samplers (NUTS) silently converge
    to garbage.  The valid pairings are documented in Bayesian.md §1.3:
        (kalman,   nuts)    — Phase 1
        (kalman,   rw_mh)   — gradient-free fallback
        (particle, rw_mh)   — Particle Marginal MH (PMMH)
    """
    if (filter_kind, sampler_kind) not in _VALID_FILTER_SAMPLER_PAIRS:
        raise ValueError(
            f"Incompatible filter/sampler combination: "
            f"filter={filter_kind.value}, sampler={sampler_kind.value}. "
            f"Particle-filter likelihoods are not differentiable, so they "
            f"cannot be paired with gradient-based samplers (NUTS). "
            f"Valid pairs: {sorted((f.value, s.value) for f, s in _VALID_FILTER_SAMPLER_PAIRS)}."
        )


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BayesianRunConfig:
    """Runtime knobs for one MCMC run (Bayesian.md §2.6).

    Attributes:
        n_chains:           Number of independent chains.
        n_warmup:           Warm-up draws per chain (step-size + mass-matrix
                            adaptation; discarded).
        n_samples:          Post-warmup draws per chain.
        target_accept_prob: NUTS dual-averaging target acceptance.  Ignored
                            by RW-MH (which uses ``rw_step_size`` instead).
        rw_step_size:       Initial step size for the random-walk proposal.
                            Ignored by NUTS.
        master_seed:        Single (m0, m1) pair from which every RNG step
                            in this run is derived via ``fold_in_seed``.
    """

    n_chains: int = 4
    n_warmup: int = 1000
    n_samples: int = 2000
    target_accept_prob: float = 0.80
    rw_step_size: float = 0.1
    master_seed: tuple[int, int] = (20, 26)

    def __post_init__(self):
        if self.n_chains < 1:
            raise ValueError(f"n_chains must be >= 1. Got {self.n_chains}.")
        if self.n_warmup < 1:
            raise ValueError(f"n_warmup must be >= 1. Got {self.n_warmup}.")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1. Got {self.n_samples}.")
        if not 0.0 < self.target_accept_prob < 1.0:
            raise ValueError(
                f"target_accept_prob must be in (0, 1). "
                f"Got {self.target_accept_prob}."
            )
        if self.rw_step_size <= 0.0:
            raise ValueError(f"rw_step_size must be > 0. Got {self.rw_step_size}.")
        if self.master_seed is None or len(self.master_seed) != 2:
            raise ValueError(
                f"master_seed must be a length-2 integer tuple. "
                f"Got {self.master_seed!r}."
            )


@dataclass(frozen=True)
class BayesianCoverageConfig:
    """Replicate-level configuration for the §2.8 coverage check.

    Attributes:
        n_replicates:    R — number of (β₀, Y) replicates.
        n_firms:         N — firms per synthetic panel.
        horizon:         T — recorded periods per firm (post burn-in).
        burn_in:         Periods discarded before recording (lets the
                         simulated panel reach its stationary distribution).
        credible_level:  Width of the reported credible interval.  0.95 is
                         the canonical SBC-style choice.
    """

    n_replicates: int = 10
    n_firms: int = 200
    horizon: int = 40
    burn_in: int = 50
    credible_level: float = 0.95

    def __post_init__(self):
        if self.n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1. Got {self.n_replicates}.")
        if self.n_firms < 1:
            raise ValueError(f"n_firms must be >= 1. Got {self.n_firms}.")
        if self.horizon < 2:
            raise ValueError(f"horizon must be >= 2. Got {self.horizon}.")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0. Got {self.burn_in}.")
        if not 0.0 < self.credible_level < 1.0:
            raise ValueError(
                f"credible_level must be in (0, 1). Got {self.credible_level}."
            )


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

PriorSampler   = Callable[[tuple[int, int]], dict[str, float]]
LogLikelihood  = Callable[[Mapping[str, tf.Tensor], Mapping[str, tf.Tensor]], tf.Tensor]
PanelSynthesizer = Callable[
    [Mapping[str, float], int, int, int, tuple[int, int]],
    dict[str, np.ndarray],
]


@dataclass(frozen=True)
class BayesianSpec:
    """Flat Bayesian problem specification consumed by ``run_mcmc``.

    Three orthogonal axes (Bayesian.md §1.2, §1.3):

      1. Prior + bijector  — what β is and how it maps to ℝᵈ.
      2. Filter            — how ``log p(y | β)`` is computed
                             (advertised via ``filter_kind``; the choice is
                             baked into ``log_likelihood_fn``).
      3. Sampler           — which MCMC kernel ``run_mcmc`` dispatches to.

    Per-env factories (e.g. ``bayesian_basic_investment.make_bayesian_spec``)
    build this once at the env's current state and hand it to the generic
    runners below.  No env-specific logic enters this module.

    Attributes:
        parameter_names:    Ordered parameter names.  Drives all downstream
                            dict keys (prior, posterior samples, intervals).
        prior_distribution: ``tfd.JointDistributionNamed`` over the constrained
                            parameter space.  Must use ``parameter_names`` as
                            keys.
        bijector:           ``tfb.JointMap`` mapping unconstrained ℝᵈ →
                            constrained support.  Keys match ``parameter_names``.
        filter_kind:        Advertised filter (Bayesian.md §1.3).  Validated
                            against ``sampler_kind`` in ``run_mcmc``.
        sampler_kind:       MCMC kernel; the actual dispatch key.
        log_likelihood_fn:  ``(beta_dict, observed_data_dict) -> scalar
                            tf.Tensor``.  Closure over solver state if any
                            (e.g. analytical policy coefficients, a trained
                            NN, or a PFI grid).  Must be differentiable in
                            β when ``sampler_kind == NUTS``.
        synthesize_panel_fn:``(beta_dict, n_firms, horizon, burn_in, seed)
                            -> dict of arrays with keys == observation_keys
                            ∪ {"metadata"}``.  Used by the coverage loop and
                            by notebook-level sanity checks.
        observation_keys:   Names of the data tensors that
                            ``log_likelihood_fn`` and ``synthesize_panel_fn``
                            agree on (e.g. ``("y", "log_k")``).
    """

    parameter_names:     tuple[str, ...]
    prior_distribution:  tfd.JointDistributionNamed
    bijector:            tfb.JointMap
    filter_kind:         FilterKind
    sampler_kind:        SamplerKind
    log_likelihood_fn:   LogLikelihood
    synthesize_panel_fn: PanelSynthesizer
    observation_keys:    tuple[str, ...]

    def __post_init__(self):
        if not self.parameter_names:
            raise ValueError("parameter_names must be non-empty.")
        if not self.observation_keys:
            raise ValueError("observation_keys must be non-empty.")
        _validate_filter_sampler_combo(self.filter_kind, self.sampler_kind)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BayesianMCMCResult:
    """One MCMC run's output (Bayesian.md §2.6, §2.7)."""

    posterior_samples:  dict[str, np.ndarray]   # name -> (n_samples, n_chains)
    r_hat:              dict[str, float]
    ess:                dict[str, float]
    acceptance_rate:    float
    wall_time_sec:      float
    metadata:           dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianCoverageResult:
    """Coverage-check output (Bayesian.md §2.8)."""

    coverage_per_parameter:    dict[str, float]
    per_replicate_intervals:   list[dict[str, tuple[float, float]]]
    per_replicate_beta0:       list[dict[str, float]]
    per_replicate_diagnostics: list[dict[str, Any]]
    credible_level:            float
    wall_time_sec:             float
    metadata:                  dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JointDistribution wrapping (NUTS path)
# ---------------------------------------------------------------------------
#
# windowed_adaptive_nuts requires a tfd.JointDistribution rather than a bare
# target_log_prob_fn. Spec factories produce (prior, log_likelihood_fn) — we
# pack them into a JointDistributionCoroutine here. The likelihood is wrapped
# in a pseudo-distribution (_LogProbStandin) whose ``log_prob`` returns the
# pre-computed scalar from log_likelihood_fn; the "value" passed to it is
# pinned to zero and ignored.

class _LogProbStandin(tfd.Distribution):
    """Pseudo-distribution whose ``log_prob`` returns a pre-computed scalar.

    Lets us embed an arbitrary ``log_likelihood_fn(beta, data) -> scalar``
    inside a ``tfd.JointDistributionCoroutine``. The "value" passed to
    ``_log_prob`` is ignored; the contribution is fully determined by the
    ``log_prob_value`` tensor captured at construction. Used only by the
    NUTS path (windowed_adaptive_nuts requires a JointDistribution).
    """

    def __init__(self, log_prob_value: tf.Tensor, name: str = "LogProbStandin"):
        """Note: ``name`` default must NOT match what callers pass explicitly.
        ``JointDistributionCoroutine`` treats the yielded distribution as
        "unnamed" (and auto-assigns ``varN``) when ``name`` equals the
        ``__init__`` default; only explicit overrides survive. Hence the
        nondescript default — callers always pass an explicit ``name``.
        """
        self._log_prob_value = log_prob_value
        super().__init__(
            dtype=log_prob_value.dtype,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True,
            # ``name`` must also appear in ``parameters`` so JointDistribution-
            # Coroutine picks it up; passing it to super().__init__ alone is
            # not enough.
            parameters=dict(log_prob_value=log_prob_value, name=name),
            name=name,
        )

    def _log_prob(self, value):
        return self._log_prob_value

    def _sample_n(self, n, seed=None):
        return tf.zeros([n], dtype=self.dtype)

    def _event_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _batch_shape(self):
        return self._log_prob_value.shape

    def _batch_shape_tensor(self):
        return tf.shape(self._log_prob_value)


def _build_joint_distribution(spec: BayesianSpec,
                              observed_data: Mapping[str, tf.Tensor]):
    """Pack (prior, likelihood) into a pinned ``tfd.JointDistribution``.

    The result is the form ``windowed_adaptive_nuts`` requires. Bijection to
    unconstrained ℝᵈ is derived automatically by TFP from each prior's
    support via ``pinned.experimental_default_event_space_bijector()``, so
    ``spec.bijector`` is unused on the NUTS path (the RW-MH path still
    consumes it).
    """
    param_names = spec.parameter_names
    prior_model = spec.prior_distribution.model
    log_lik_fn  = spec.log_likelihood_fn

    @tfd.JointDistributionCoroutineAutoBatched
    def model():
        param_dict: dict[str, tf.Tensor] = {}
        for name in param_names:
            # .copy(name=name) propagates the dict key as the distribution
            # name so JointDistributionCoroutine yields it under that name.
            param_dict[name] = yield prior_model[name].copy(name=name)
        log_lik = log_lik_fn(param_dict, observed_data)
        yield _LogProbStandin(log_lik, name="likelihood")

    # The standin's value is irrelevant; pinning with zeros removes it
    # from the sampled latent set without affecting log_prob.
    return model.experimental_pin(likelihood=tf.zeros([]))


# ---------------------------------------------------------------------------
# Target log-prob construction (RW-MH path)
# ---------------------------------------------------------------------------

def _build_target_log_prob(spec: BayesianSpec,
                           observed_data: Mapping[str, tf.Tensor]):
    """Return a callable ``target_log_prob(*unconstrained_args) -> scalar``.

    Pieces together log prior + log|det J| + log likelihood. Used only by
    the RW-MH path; the NUTS path goes through ``_build_joint_distribution``
    + ``windowed_adaptive_nuts``, which handle bijection internally.
    """

    param_names = spec.parameter_names
    prior = spec.prior_distribution
    bijector = spec.bijector

    bijector_keys = set(bijector.bijectors.keys()) if isinstance(bijector.bijectors, dict) else set()
    if bijector_keys and bijector_keys != set(param_names):
        raise ValueError(
            "spec.bijector keys must match spec.parameter_names. "
            f"Got bijector keys {sorted(bijector_keys)} "
            f"vs parameter_names {sorted(param_names)}."
        )

    def target_log_prob(*unconstrained):
        if len(unconstrained) != len(param_names):
            raise ValueError(
                f"target_log_prob expected {len(param_names)} positional args "
                f"(one per parameter). Got {len(unconstrained)}."
            )
        u = dict(zip(param_names, unconstrained))
        constrained = bijector.forward(u)
        log_det = bijector.forward_log_det_jacobian(u, event_ndims={k: 0 for k in param_names})
        log_prior = prior.log_prob(constrained)
        log_lik   = spec.log_likelihood_fn(constrained, observed_data)
        if isinstance(log_det, Mapping):
            log_det = tf.add_n([tf.cast(v, log_prior.dtype) for v in log_det.values()])
        return log_prior + log_det + log_lik

    return target_log_prob


def _draw_initial_state(spec: BayesianSpec, n_chains: int,
                        seed: tuple[int, int]) -> list[tf.Tensor]:
    """Sample initial unconstrained states for all chains from the prior (RW-MH only).

    The NUTS path goes through ``windowed_adaptive_nuts``, which samples
    its own initial state from the joint distribution internally.
    """
    samples = spec.prior_distribution.sample(n_chains, seed=tf.constant(seed, tf.int32))
    unconstrained = spec.bijector.inverse({k: samples[k] for k in spec.parameter_names})
    return [unconstrained[k] for k in spec.parameter_names]


# ---------------------------------------------------------------------------
# Sampler dispatch
# ---------------------------------------------------------------------------
#
# Both samplers share the contract:
#   (spec, observed_data, run_config, seed)
#       -> (posterior_dict, acceptance_float, sampler_metadata_dict)
# where ``posterior_dict[name]`` is a (n_samples, n_chains) ndarray on the
# constrained scale (already mapped back through any bijector).

def _run_nuts(spec: BayesianSpec,
              observed_data: Mapping[str, tf.Tensor],
              run_config: BayesianRunConfig,
              seed: tuple[int, int]):
    """NUTS via ``windowed_adaptive_nuts`` (Stan-style windowed warmup).

    Bundles step-size dual averaging and diagonal mass-matrix adaptation
    across expanding warmup windows — the documented choice in
    Bayesian.md §2.6. The previous in-house composition (NoUTurnSampler +
    DualAveragingStepSizeAdaptation alone) omitted mass-matrix adaptation
    and was the root cause of multi-hour stalls on heterogeneous-scale
    targets such as the 5-D NN-surrogate posterior in NB09.

    Returns:
        posterior:  {param_name: ndarray (n_samples, n_chains)} on constrained scale.
        acceptance: mean accept_ratio over post-warmup draws.
        metadata:   leapfrog and divergence diagnostics.
    """
    joint_dist = _build_joint_distribution(spec, observed_data)
    target_accept = float(run_config.target_accept_prob)

    @tf.function(jit_compile=False)
    def _sample():
        return tfp.experimental.mcmc.windowed_adaptive_nuts(
            n_draws=run_config.n_samples,
            joint_dist=joint_dist,
            n_chains=run_config.n_chains,
            num_adaptation_steps=run_config.n_warmup,
            seed=tf.constant(seed, tf.int32),
            dual_averaging_kwargs={"target_accept_prob": target_accept},
        )

    states, trace = _sample()
    posterior = {k: getattr(states, k).numpy() for k in spec.parameter_names}
    acceptance = float(tf.reduce_mean(trace["accept_ratio"]))
    metadata = {
        "mean_leapfrog_steps": float(tf.reduce_mean(tf.cast(trace["n_steps"], tf.float32))),
        "max_leapfrog_steps":  int(tf.reduce_max(trace["n_steps"]).numpy()),
        "divergence_count":    int(tf.reduce_sum(tf.cast(trace["diverging"], tf.int32)).numpy()),
    }
    return posterior, acceptance, metadata


def _run_rw_mh(spec: BayesianSpec,
               observed_data: Mapping[str, tf.Tensor],
               run_config: BayesianRunConfig,
               seed: tuple[int, int]):
    """Random-walk Metropolis-Hastings — for the (particle, RW-MH) PMMH path.

    Gradient-free; works with non-differentiable likelihoods such as
    particle-filter estimates (Andrieu, Doucet, Holenstein 2010,
    Bayesian.md §1.3). Same return contract as ``_run_nuts``.
    """
    target = _build_target_log_prob(spec, observed_data)
    init_seed   = fold_in_seed(seed, "init")
    sample_seed = fold_in_seed(seed, "sample")
    initial_state = _draw_initial_state(spec, run_config.n_chains, init_seed)

    kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(run_config.n_warmup * 0.8),
        target_accept_prob=tf.constant(0.234, dtype=initial_state[0].dtype),
        adaptation_rate=0.05,
        new_step_size_fn=lambda step, _: step,
    )

    @tf.function(jit_compile=False)
    def _sample():
        samples, log_accept = tfp.mcmc.sample_chain(
            num_results=run_config.n_samples,
            num_burnin_steps=run_config.n_warmup,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
            seed=tf.constant(sample_seed, tf.int32),
        )
        return samples, log_accept

    raw_samples, log_accept = _sample()
    acceptance = float(tf.reduce_mean(tf.exp(tf.minimum(0.0, log_accept))))

    unconstrained_dict = {k: raw_samples[i] for i, k in enumerate(spec.parameter_names)}
    constrained_dict = spec.bijector.forward(unconstrained_dict)
    posterior = {k: constrained_dict[k].numpy() for k in spec.parameter_names}
    return posterior, acceptance, {}


_SAMPLERS = {
    SamplerKind.NUTS:  _run_nuts,
    SamplerKind.RW_MH: _run_rw_mh,
}


# ---------------------------------------------------------------------------
# Public: single MCMC run
# ---------------------------------------------------------------------------

def run_mcmc(spec: BayesianSpec,
             observed_data: Mapping[str, np.ndarray],
             run_config: BayesianRunConfig,
             seed: tuple[int, int]) -> BayesianMCMCResult:
    """Run one MCMC chain set on ``observed_data`` at the given seed.

    Steps (Bayesian.md §2.6):
      1. Convert observed data to tf.Tensor.
      2. Dispatch by ``spec.sampler_kind``:
           NUTS  -> ``windowed_adaptive_nuts`` over a packed JointDistribution
                    (initial state, bijection, and adaptation all internal).
           RW-MH -> hand-composed RW-MH over the explicit unconstrained
                    target_log_prob; initial state sampled from the prior.
      3. Compute R̂ / ESS on the returned constrained-scale samples.

    Args:
        spec:           BayesianSpec from a per-env factory.
        observed_data:  Dict matching ``spec.observation_keys``. Arrays are
                        cast to float32 tensors before use.
        run_config:     Sampler runtime config.
        seed:           Length-2 int seed pair. Passed straight to NUTS; for
                        RW-MH, init/sample sub-seeds are derived via
                        ``fold_in_seed``. Same master seed reproduces the
                        run bit-identically on a given TFP version + CPU.

    Returns:
        BayesianMCMCResult with per-parameter posterior samples, R̂, ESS,
        average acceptance probability, wall time, and (for NUTS) leapfrog
        and divergence diagnostics in ``metadata``.
    """

    _validate_filter_sampler_combo(spec.filter_kind, spec.sampler_kind)
    missing = [k for k in spec.observation_keys if k not in observed_data]
    if missing:
        raise KeyError(
            f"observed_data is missing required keys: {missing}. "
            f"Required by spec.observation_keys: {spec.observation_keys}."
        )

    data_tensors = {
        k: tf.convert_to_tensor(observed_data[k], dtype=tf.float32)
        for k in spec.observation_keys
    }

    start = time.perf_counter()
    sampler_fn = _SAMPLERS[spec.sampler_kind]
    posterior, acceptance, sampler_metadata = sampler_fn(
        spec, data_tensors, run_config, seed)

    r_hat = {k: float(tfp.mcmc.potential_scale_reduction(posterior[k]).numpy())
             for k in spec.parameter_names}
    ess = {k: float(tf.reduce_sum(tfp.mcmc.effective_sample_size(posterior[k])).numpy())
           for k in spec.parameter_names}

    return BayesianMCMCResult(
        posterior_samples=posterior,
        r_hat=r_hat,
        ess=ess,
        acceptance_rate=acceptance,
        wall_time_sec=time.perf_counter() - start,
        metadata={
            "filter_kind":   spec.filter_kind.value,
            "sampler_kind":  spec.sampler_kind.value,
            "seed":          tuple(int(x) for x in seed),
            "n_chains":      run_config.n_chains,
            "n_warmup":      run_config.n_warmup,
            "n_samples":     run_config.n_samples,
            **sampler_metadata,
        },
    )


# ---------------------------------------------------------------------------
# Public: coverage check (Bayesian.md §2.8)
# ---------------------------------------------------------------------------

def _credible_interval(samples: np.ndarray,
                       credible_level: float) -> tuple[float, float]:
    """Empirical equal-tailed credible interval at the given level."""
    flat = samples.reshape(-1)
    alpha = 0.5 * (1.0 - credible_level)
    lo, hi = np.quantile(flat, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def run_coverage_check(spec: BayesianSpec,
                       run_config: BayesianRunConfig,
                       coverage_config: BayesianCoverageConfig,
                       master_seed: tuple[int, int]) -> BayesianCoverageResult:
    """R-replicate coverage check (Bayesian.md §2.8).

    Per replicate r ∈ [0, R):
      1. β₀_r  ~  spec.prior_distribution           (seed: master/"replicate"/r/"beta0")
      2. Y_r    = spec.synthesize_panel_fn(β₀_r)    (seed: master/"replicate"/r/"panel")
      3. post_r = run_mcmc(spec, Y_r, run_config)   (seed: master/"replicate"/r/"mcmc")
      4. CI_r   = quantile_{α/2, 1-α/2}(post_r) at credible_level

    Returns the per-parameter fraction of replicates whose CI contains the
    corresponding component of β₀, plus per-replicate detail for tables /
    plots.  See §2.9 for the seed-folding contract.
    """

    start = time.perf_counter()
    R = coverage_config.n_replicates
    N = coverage_config.n_firms
    T = coverage_config.horizon
    burn_in = coverage_config.burn_in
    cred = coverage_config.credible_level

    intervals: list[dict[str, tuple[float, float]]] = []
    beta0s:    list[dict[str, float]]               = []
    diags:     list[dict[str, Any]]                 = []

    for r in range(R):
        rep_seed = fold_in_seed(master_seed, "replicate", r)

        # 1. Draw β₀ from the prior.
        beta0_seed = fold_in_seed(rep_seed, "beta0")
        beta0_sample = spec.prior_distribution.sample(
            seed=tf.constant(beta0_seed, tf.int32))
        beta0 = {k: float(beta0_sample[k].numpy()) for k in spec.parameter_names}

        # 2. Synthesize panel at β₀.
        panel_seed = fold_in_seed(rep_seed, "panel")
        panel = spec.synthesize_panel_fn(beta0, N, T, burn_in, panel_seed)
        observed = {k: panel[k] for k in spec.observation_keys}

        # 3. Run MCMC.
        mcmc_seed = fold_in_seed(rep_seed, "mcmc")
        mcmc_result = run_mcmc(spec, observed, run_config, mcmc_seed)

        # 4. Empirical CIs per parameter.
        ci = {k: _credible_interval(mcmc_result.posterior_samples[k], cred)
              for k in spec.parameter_names}

        intervals.append(ci)
        beta0s.append(beta0)
        diags.append({
            "r_hat":           mcmc_result.r_hat,
            "ess":             mcmc_result.ess,
            "acceptance_rate": mcmc_result.acceptance_rate,
            "wall_time_sec":   mcmc_result.wall_time_sec,
        })

    coverage = {
        k: float(np.mean([intervals[r][k][0] <= beta0s[r][k] <= intervals[r][k][1]
                          for r in range(R)]))
        for k in spec.parameter_names
    }

    return BayesianCoverageResult(
        coverage_per_parameter=coverage,
        per_replicate_intervals=intervals,
        per_replicate_beta0=beta0s,
        per_replicate_diagnostics=diags,
        credible_level=cred,
        wall_time_sec=time.perf_counter() - start,
        metadata={
            "filter_kind":  spec.filter_kind.value,
            "sampler_kind": spec.sampler_kind.value,
            "master_seed":  tuple(int(x) for x in master_seed),
            "n_replicates": R,
            "n_firms":      N,
            "horizon":      T,
            "burn_in":      burn_in,
        },
    )
