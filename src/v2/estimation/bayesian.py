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

import collections
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.v2.utils.seeding import fold_in_seed, make_seed_int

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
    NUTS  = "nuts"
    RW_MH = "rw_mh"
    RAM   = "ram"   # Robust Adaptive Metropolis (Vihola 2012)


_VALID_FILTER_SAMPLER_PAIRS: frozenset[tuple[FilterKind, SamplerKind]] = frozenset({
    (FilterKind.KALMAN,   SamplerKind.NUTS),
    (FilterKind.KALMAN,   SamplerKind.RW_MH),
    (FilterKind.KALMAN,   SamplerKind.RAM),
    (FilterKind.PARTICLE, SamplerKind.RW_MH),  # PMMH
    (FilterKind.PARTICLE, SamplerKind.RAM),    # PMMH-RAM (Vihola-style PMMH)
})


def _validate_filter_sampler_combo(filter_kind: FilterKind,
                                   sampler_kind: SamplerKind) -> None:
    """Hard-error on (particle, nuts).

    A particle-filter log-likelihood is unbiased but non-differentiable through
    the resampling step, so gradient-based samplers (NUTS) silently converge
    to garbage.  The valid pairings are documented in Bayesian.md §1.3:
        (kalman,   nuts)    — Phase 1
        (kalman,   rw_mh)   — gradient-free fallback
        (kalman,   ram)     — adaptive gradient-free (Vihola 2012)
        (particle, rw_mh)   — Particle Marginal MH (PMMH)
        (particle, ram)     — PMMH with RAM proposal (PMMH-RAM)
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
        n_chains:             Number of independent chains.
        n_warmup:             Warm-up draws per chain (step-size + mass-matrix
                              adaptation; discarded).
        n_samples:            Post-warmup draws per chain.
        target_accept_prob:   NUTS dual-averaging target acceptance.  Ignored
                              by RW-MH and RAM.
        rw_step_size:         Proposal scale(s) for the **RW-MH** sampler in
                              unconstrained ℝᵈ space.  Either a positive scalar
                              or a length-d sequence of positives (one per
                              parameter in ``spec.parameter_names`` order).
                              Use ``default_rw_scales_from_prior`` for a
                              prior-based starting list.  Ignored by NUTS / RAM.
        ram_target_accept:    RAM coerced acceptance rate (default 0.234, the
                              d-large RGG optimum).  RAM adapts its Cholesky
                              factor so that empirical acceptance converges to
                              this value.  Ignored by NUTS / RW-MH.
        ram_initial_scale:    Initial diagonal of RAM's lower-triangular
                              proposal Cholesky.  Positive scalar (S ← scalar
                              × I) or length-d positive sequence (S ←
                              diag(sequence)).  Warm-start the chain by passing
                              prior-based or pilot-based posterior scales;
                              defaults to 0.1 (small isotropic).  Ignored by
                              NUTS / RW-MH.
        ram_adaptation_decay: RAM Robbins-Monro decay γ ∈ (0.5, 1.0].  η_t =
                              (t+1)^(-γ) is the per-step adaptation magnitude;
                              γ > 0.5 is required for ergodicity (Roberts-
                              Rosenthal 2007).  Default 0.7.  Ignored by NUTS /
                              RW-MH.
        master_seed:          Single (m0, m1) pair from which every RNG step
                              in this run is derived via ``fold_in_seed``.
    """

    n_chains: int = 4
    n_warmup: int = 1000
    n_samples: int = 2000
    target_accept_prob: float = 0.80
    rw_step_size: Union[float, Sequence[float]] = 0.1
    ram_target_accept: float = 0.234
    ram_initial_scale: Union[float, Sequence[float]] = 0.1
    ram_adaptation_decay: float = 0.7
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
        if isinstance(self.rw_step_size, (int, float)):
            if self.rw_step_size <= 0.0:
                raise ValueError(f"rw_step_size must be > 0. Got {self.rw_step_size}.")
        else:
            try:
                scales = tuple(float(s) for s in self.rw_step_size)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"rw_step_size must be a positive float or a sequence of "
                    f"positive floats (one per parameter). Got {self.rw_step_size!r}."
                ) from e
            if not scales or any(s <= 0.0 for s in scales):
                raise ValueError(
                    f"rw_step_size sequence must be non-empty and all entries "
                    f"strictly positive. Got {self.rw_step_size}."
                )
        if not 0.0 < self.ram_target_accept < 1.0:
            raise ValueError(
                f"ram_target_accept must be in (0, 1). "
                f"Got {self.ram_target_accept}."
            )
        if isinstance(self.ram_initial_scale, (int, float)):
            if self.ram_initial_scale <= 0.0:
                raise ValueError(
                    f"ram_initial_scale must be > 0. Got {self.ram_initial_scale}."
                )
        else:
            try:
                init_scales = tuple(float(s) for s in self.ram_initial_scale)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"ram_initial_scale must be a positive float or a sequence "
                    f"of positive floats (one per parameter). "
                    f"Got {self.ram_initial_scale!r}."
                ) from e
            if not init_scales or any(s <= 0.0 for s in init_scales):
                raise ValueError(
                    f"ram_initial_scale sequence must be non-empty and all "
                    f"entries strictly positive. Got {self.ram_initial_scale}."
                )
        if not 0.5 < self.ram_adaptation_decay <= 1.0:
            raise ValueError(
                f"ram_adaptation_decay must be in (0.5, 1.0] for ergodicity "
                f"(Roberts-Rosenthal 2007).  Got {self.ram_adaptation_decay}."
            )
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

    Per-env factories (e.g. ``bayesian_basic_investment.make_neural_bayesian_spec``)
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


def default_rw_scales_from_prior(
    spec: BayesianSpec,
    *,
    n_samples: int = 5000,
    rgg_factor: Optional[float] = None,
    seed: tuple[int, int] = (0, 1),
) -> tuple[float, ...]:
    """Per-parameter RW-MH proposal scales derived from prior samples on ℝᵈ.

    Draws ``n_samples`` β from the prior, transforms each parameter to its
    unconstrained scale via ``spec.bijector.inverse``, and returns the
    empirical standard deviation per parameter times ``rgg_factor`` (default
    ``2.38/√d`` per Roberts-Gelman-Gilks 1997).  Output order matches
    ``spec.parameter_names``; pass the result directly as
    ``BayesianRunConfig.rw_step_size``.

    Caveat: for parameters with Uniform priors on bounded support (mapped to
    ℝ through Sigmoid), the prior std on the unconstrained scale (~π/√3 ≈ 1.8
    for Uniform(0,1) → Logit) is much larger than the typical posterior std
    once data is observed.  The helper therefore tends to overshoot by a
    factor of ``σ_prior / σ_posterior`` for those parameters — typically
    10-30× — and acceptance will land well below the RGG optimum until
    rescaled.  For tighter calibration, run a short pilot RW-MH at these
    defaults and multiply each scale by ``σ_unconstrained_pilot / σ_prior_unconstrained``.
    """
    d = len(spec.parameter_names)
    factor = 2.38 / math.sqrt(d) if rgg_factor is None else float(rgg_factor)
    samples = spec.prior_distribution.sample(
        n_samples, seed=tf.constant(seed, tf.int32))
    unconstrained = spec.bijector.inverse(
        {k: samples[k] for k in spec.parameter_names})
    return tuple(
        float(tf.math.reduce_std(unconstrained[k]).numpy()) * factor
        for k in spec.parameter_names
    )


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
    """Random-walk Metropolis-Hastings — gradient-free baseline sampler.

    Pairs with the Kalman filter (KF + RW-MH baseline, used by 08b) or the
    particle filter (PMMH per Andrieu, Doucet, Holenstein 2010, Bayesian.md
    §1.3).  Same return contract as ``_run_nuts``.

    Uses a **fixed proposal scale** ``run_config.rw_step_size`` in the
    unconstrained ℝᵈ space (post-bijector).  Either a scalar (applied
    identically across parameters) or a length-d list (one per parameter
    in ``spec.parameter_names`` order); the sequence form is essential
    when parameters have heterogeneous posterior scales.  Step-size
    adaptation is not currently wired; ``tfp.mcmc.SimpleStepSizeAdaptation``
    is designed for HMC-style kernels with a ``step_size`` attribute, not
    for the ``RandomWalkMetropolis`` proposal-scale parameter.  A Robust
    Adaptive Metropolis (RAM) variant would be the natural upgrade;
    tracked as a follow-up.
    """
    target = _build_target_log_prob(spec, observed_data)
    init_seed   = fold_in_seed(seed, "init")
    sample_seed = fold_in_seed(seed, "sample")
    initial_state = _draw_initial_state(spec, run_config.n_chains, init_seed)

    state_dtype = initial_state[0].dtype
    n_params = len(spec.parameter_names)
    step = run_config.rw_step_size
    if isinstance(step, (int, float)):
        rw_scale = tf.constant(float(step), dtype=state_dtype)
    else:
        scales_list = [float(s) for s in step]
        if len(scales_list) != n_params:
            raise ValueError(
                f"rw_step_size sequence must have length {n_params} "
                f"(one per parameter in {spec.parameter_names}). "
                f"Got length {len(scales_list)}."
            )
        rw_scale = [tf.constant(s, dtype=state_dtype) for s in scales_list]

    new_state_fn = tfp.mcmc.random_walk_normal_fn(scale=rw_scale)
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target,
        new_state_fn=new_state_fn,
    )

    @tf.function(jit_compile=False)
    def _sample():
        samples, log_accept = tfp.mcmc.sample_chain(
            num_results=run_config.n_samples,
            num_burnin_steps=run_config.n_warmup,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _, pkr: pkr.log_accept_ratio,
            seed=tf.constant(sample_seed, tf.int32),
        )
        return samples, log_accept

    raw_samples, log_accept = _sample()
    acceptance = float(tf.reduce_mean(tf.exp(tf.minimum(0.0, log_accept))))

    unconstrained_dict = {k: raw_samples[i] for i, k in enumerate(spec.parameter_names)}
    constrained_dict = spec.bijector.forward(unconstrained_dict)
    posterior = {k: constrained_dict[k].numpy() for k in spec.parameter_names}
    return posterior, acceptance, {}


# Robust Adaptive Metropolis as a TFP TransitionKernel.  Carries the per-chain
# lower-triangular Cholesky factor ``S`` and a step counter through
# ``previous_kernel_results`` so the whole chain runs inside ``sample_chain``'s
# ``tf.function``-compiled loop.  Same Vihola (2012) Algorithm 1 as the
# previous NumPy outer-loop implementation; replaced for TF-native consistency
# with ``_run_nuts``/``_run_rw_mh`` and to remove Python<->TF boundary crossings
# (which dominated the wall on the negligible-likelihood Gaussian test).
_RAMKernelResults = collections.namedtuple(
    "_RAMKernelResults",
    ["target_log_prob", "cholesky", "log_accept_ratio", "step", "seed"],
)


class _RAMKernel(tfp.mcmc.TransitionKernel):
    """Vihola (2012) Robust Adaptive Metropolis kernel.

    State: list of d tensors, each shape ``(n_chains,)`` (one per parameter,
    same convention as TFP RW-MH).  Adaptation happens only while
    ``previous_kernel_results.step < n_warmup``; afterwards the proposal
    covariance is frozen and the kernel reduces to a fixed-scale RW-MH with
    full lower-triangular Cholesky factor ``S``.  ``S`` is stored per chain
    so each chain adapts independently — preserves cross-chain R-hat as a
    valid convergence diagnostic.

    Per step:
      1. ξ ~ N(0, I_d) (vectorized over chains).
      2. β' = β + S ξ.
      3. α = exp(min(0, log_target(β') - log_target(β))).
      4. Accept iff log(U) < log α; state/log-target updated where accepted.
      5. If warmup: S_{t+1} S_{t+1}ᵀ = S_t (I + η_t (α - α*) ξξᵀ/||ξ||²) S_tᵀ
         with η_t = (t+1)^(-γ); fall back to S_t if Cholesky of the
         symmetrized S M Sᵀ contains NaN (rare float32 fluke).
    """

    def __init__(self,
                 target_log_prob_fn: Callable,
                 initial_cholesky: tf.Tensor,
                 target_accept: float,
                 decay: float,
                 n_warmup: int,
                 name: Optional[str] = None):
        self._target_log_prob_fn = target_log_prob_fn
        self._initial_cholesky   = initial_cholesky          # (n_chains, d, d)
        self._target_accept      = float(target_accept)
        self._decay              = float(decay)
        self._n_warmup           = int(n_warmup)
        self._name               = name or "ram_kernel"

    @property
    def is_calibrated(self) -> bool:
        # Detailed balance holds at every step (symmetric Gaussian proposal);
        # ``True`` matches the TFP convention for adaptive RW kernels.
        return True

    @property
    def parameters(self) -> dict:
        return {
            "target_log_prob_fn": self._target_log_prob_fn,
            "initial_cholesky":  self._initial_cholesky,
            "target_accept":     self._target_accept,
            "decay":             self._decay,
            "n_warmup":          self._n_warmup,
            "name":              self._name,
        }

    def bootstrap_results(self, init_state) -> _RAMKernelResults:
        with tf.name_scope(f"{self._name}.bootstrap"):
            init_target = self._target_log_prob_fn(*init_state)
            return _RAMKernelResults(
                target_log_prob = init_target,
                cholesky        = tf.identity(self._initial_cholesky),
                log_accept_ratio= tf.zeros_like(init_target),
                step            = tf.constant(0, dtype=tf.int32),
                seed            = tfp.random.sanitize_seed([0, 0], salt="ram_init"),
            )

    def one_step(self, current_state, previous_kernel_results, seed=None):
        with tf.name_scope(f"{self._name}.one_step"):
            d_static = len(current_state)
            S        = previous_kernel_results.cholesky                 # (n_chains, d, d)
            log_pi   = previous_kernel_results.target_log_prob          # (n_chains,)
            step     = previous_kernel_results.step                     # scalar int32
            cur_concat = tf.stack(current_state, axis=-1)               # (n_chains, d)

            seed_proposal, seed_accept = tfp.random.split_seed(
                seed if seed is not None else previous_kernel_results.seed,
                n=2, salt="ram_step",
            )
            xi   = tf.random.stateless_normal(
                shape=tf.shape(cur_concat), seed=seed_proposal, dtype=cur_concat.dtype)
            prop_concat = cur_concat + tf.einsum("cij,cj->ci", S, xi)
            prop_parts  = tf.unstack(prop_concat, num=d_static, axis=-1)
            log_pi_prop = self._target_log_prob_fn(*prop_parts)

            log_alpha = tf.minimum(tf.zeros_like(log_pi), log_pi_prop - log_pi)
            log_u     = tf.math.log(tf.random.stateless_uniform(
                shape=tf.shape(log_pi), seed=seed_accept, dtype=log_pi.dtype))
            accept    = log_u < log_alpha

            new_concat = tf.where(accept[:, None], prop_concat, cur_concat)
            new_log_pi = tf.where(accept, log_pi_prop, log_pi)

            # Per-step adaptation only while step < n_warmup.
            is_warmup = step < self._n_warmup
            eta = tf.where(
                is_warmup,
                tf.pow(tf.cast(step + 1, S.dtype),
                       tf.constant(-self._decay, dtype=S.dtype)),
                tf.zeros([], dtype=S.dtype),
            )
            alphas        = tf.exp(log_alpha)                           # (n_chains,)
            norm_sq       = tf.einsum("ci,ci->c", xi, xi)               # (n_chains,)
            target_accept = tf.constant(self._target_accept, dtype=S.dtype)
            factor        = eta * (alphas - target_accept) / tf.maximum(
                norm_sq, tf.constant(1e-12, dtype=S.dtype))             # (n_chains,)

            # M = I + factor * ξξᵀ; S_new Cholesky-factors S M Sᵀ.
            outer = tf.einsum("ci,cj->cij", xi, xi)                     # (n_chains, d, d)
            M_eye = tf.eye(d_static, dtype=S.dtype, batch_shape=[tf.shape(S)[0]])
            M     = M_eye + factor[:, None, None] * outer
            new_cov = tf.einsum("cij,cjk->cik", S,
                                tf.einsum("cij,cjk->cik", M,
                                          tf.linalg.matrix_transpose(S)))
            new_cov = 0.5 * (new_cov + tf.linalg.matrix_transpose(new_cov))
            S_attempt = tf.linalg.cholesky(new_cov)                     # NaNs if non-PD

            # Fall back to old S if (a) post-warmup (eta == 0), (b) update would
            # make M near-singular, or (c) Cholesky produced any NaN.
            chol_ok = tf.reduce_all(tf.math.is_finite(S_attempt), axis=[1, 2])
            pd_ok   = (1.0 + factor * norm_sq) > tf.constant(1e-6, dtype=S.dtype)
            do_update = is_warmup & chol_ok & pd_ok
            new_S = tf.where(do_update[:, None, None], S_attempt, S)

            new_state = tf.unstack(new_concat, num=d_static, axis=-1)
            return new_state, _RAMKernelResults(
                target_log_prob = new_log_pi,
                cholesky        = new_S,
                log_accept_ratio= log_alpha,
                step            = step + 1,
                seed            = seed_accept,
            )


def _run_ram(spec: BayesianSpec,
             observed_data: Mapping[str, tf.Tensor],
             run_config: BayesianRunConfig,
             seed: tuple[int, int]):
    """Robust Adaptive Metropolis (Vihola, 2012) dispatch.

    Same algorithm as ``_RAMKernel.one_step`` above; this dispatch wires up
    the initial Cholesky, runs ``tfp.mcmc.sample_chain`` for
    ``n_warmup + n_samples`` total iterations with no built-in burn-in
    (we trace every iter and split warmup/post-warmup post-hoc so the
    warmup acceptance rate is exposed for diagnostics), then maps samples
    back through the bijector.

    Reproducibility: same master seed → bit-identical chain on a given TFP
    version + CPU.  Both ``stateless_normal`` (proposal) and
    ``stateless_uniform`` (accept) are seeded by sub-seeds split from the
    sample-chain seed.
    """
    target    = _build_target_log_prob(spec, observed_data)
    init_seed = fold_in_seed(seed, "init")
    initial_state = _draw_initial_state(spec, run_config.n_chains, init_seed)

    d        = len(spec.parameter_names)
    n_chains = int(run_config.n_chains)
    n_warmup = int(run_config.n_warmup)
    n_samples= int(run_config.n_samples)

    # Initial Cholesky from the (scalar or length-d) ram_initial_scale.
    init_scale = run_config.ram_initial_scale
    if isinstance(init_scale, (int, float)):
        init_diag = tf.fill([d], tf.constant(float(init_scale), tf.float32))
    else:
        init_diag = tf.constant(list(init_scale), dtype=tf.float32)
        if init_diag.shape != (d,):
            raise ValueError(
                f"ram_initial_scale sequence must have length {d} (one per "
                f"parameter in {spec.parameter_names}).  Got shape {init_diag.shape}."
            )
    initial_cholesky = tf.tile(
        tf.linalg.diag(init_diag)[None, :, :], [n_chains, 1, 1])    # (n_chains, d, d)

    kernel = _RAMKernel(
        target_log_prob_fn=target,
        initial_cholesky  =initial_cholesky,
        target_accept     =float(run_config.ram_target_accept),
        decay             =float(run_config.ram_adaptation_decay),
        n_warmup          =n_warmup,
    )

    sample_seed = fold_in_seed(seed, "sample")
    total_iters = n_warmup + n_samples

    # Trace acceptance at every iter (so we can compute warmup-vs-post-warmup
    # acceptance rates after the fact).  ``return_final_kernel_results=True``
    # gives us the final per-chain Cholesky for diagnostics.
    @tf.function(jit_compile=False)
    def _sample():
        return tfp.mcmc.sample_chain(
            num_results       = total_iters,
            num_burnin_steps  = 0,
            current_state     = initial_state,
            kernel            = kernel,
            trace_fn          = lambda _, pkr: pkr.log_accept_ratio,
            seed              = tf.constant(sample_seed, tf.int32),
            return_final_kernel_results=True,
        )
    raw_samples, log_accept_trace, final_kr = _sample()
    # raw_samples: list of d tensors, each shape (total_iters, n_chains).
    # log_accept_trace: shape (total_iters, n_chains).

    log_accept = log_accept_trace.numpy()
    warmup_acceptance = float(np.mean(np.exp(np.minimum(0.0, log_accept[:n_warmup]))))
    post_acceptance   = float(np.mean(np.exp(np.minimum(0.0, log_accept[n_warmup:]))))

    # Slice post-warmup samples and forward through the bijector.
    unconstrained = {
        name: raw_samples[i][n_warmup:, :]      # (n_samples, n_chains)
        for i, name in enumerate(spec.parameter_names)
    }
    constrained = spec.bijector.forward(unconstrained)
    posterior = {k: constrained[k].numpy() for k in spec.parameter_names}

    # Final-Cholesky diagnostic: per-chain log|S Sᵀ| = 2 sum log diag(S).
    final_S = final_kr.cholesky.numpy()                              # (n_chains, d, d)
    log_det_final = [
        float(2.0 * np.sum(np.log(np.maximum(np.diag(final_S[c]), 1e-30))))
        for c in range(n_chains)
    ]

    metadata = {
        "ram_target_accept":           float(run_config.ram_target_accept),
        "ram_adaptation_decay":        float(run_config.ram_adaptation_decay),
        "ram_initial_scale":           init_diag.numpy().tolist(),
        "ram_warmup_acceptance":       warmup_acceptance,
        "ram_post_acceptance":         post_acceptance,
        "ram_final_log_det_per_chain": log_det_final,
    }
    return posterior, post_acceptance, metadata


_SAMPLERS = {
    SamplerKind.NUTS:  _run_nuts,
    SamplerKind.RW_MH: _run_rw_mh,
    SamplerKind.RAM:   _run_ram,
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


# ---------------------------------------------------------------------------
# Public: posterior predictive check (Bayesian.md §1.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PosteriorPredictiveConfig:
    """Configuration for a posterior predictive check.

    Attributes:
        n_draws:    Number of posterior β draws used to generate replicated panels.
        burn_in:    Burn-in periods for each replicated panel synthesis.  Match
                    whatever burn-in produced the observed data so the comparison
                    is stationary-to-stationary.
        statistics: Names of summary statistics to compute on observed and on
                    every replicated panel.  Must be keys in the module-level
                    PPC statistics registry.
    """

    n_draws: int = 100
    burn_in: int = 20
    statistics: tuple[str, ...] = (
        "y_mean", "y_std",
        "log_k_mean", "log_k_std",
        "log_k_growth_mean",
        "y_log_k_slope",
    )

    def __post_init__(self):
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be >= 1. Got {self.n_draws}.")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0. Got {self.burn_in}.")
        if not self.statistics:
            raise ValueError("statistics must be non-empty.")


@dataclass
class PosteriorPredictiveResult:
    """Output of ``run_posterior_predictive_check``."""

    observed_stats:    dict[str, float]
    replicated_stats:  dict[str, np.ndarray]   # (n_draws,) per stat
    bayes_p_values:    dict[str, float]        # mean(rep > obs) under posterior
    wall_time_sec:     float
    metadata:          dict[str, Any] = field(default_factory=dict)


def _panel_squeeze(panel: Mapping[str, Any],
                   keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Drop a trailing singleton last dim if present, return float64 ndarrays.

    Observed panels arrive with shape (N, T, 1) (notebook expansion via
    ``np.newaxis``); synthesized panels can come back as (N, T) or (N, T, 1)
    depending on the synthesizer.  Squeezing only when the last dim is 1
    keeps both code paths shape-aligned for the statistics functions.
    """
    out: dict[str, np.ndarray] = {}
    for k in keys:
        arr = np.asarray(panel[k], dtype=np.float64)
        if arr.ndim >= 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        out[k] = arr
    return out


# Statistics registry: name -> function(panel_dict) -> float.  All stats are
# model-free: they consume the panel arrays only, not β.  This keeps PPC a
# pure data-replication test (Bayesian.md §1.4).

def _ppc_y_mean(panel):
    return float(np.mean(panel["y"]))


def _ppc_y_std(panel):
    return float(np.std(panel["y"]))


def _ppc_log_k_mean(panel):
    return float(np.mean(panel["log_k"]))


def _ppc_log_k_std(panel):
    return float(np.std(panel["log_k"]))


def _ppc_log_k_growth_mean(panel):
    return float(np.mean(panel["log_k_next"] - panel["log_k"]))


def _ppc_y_log_k_slope(panel):
    """OLS slope of pooled y on pooled log_k after demeaning."""
    y = panel["y"].reshape(-1)
    x = panel["log_k"].reshape(-1)
    x_dm = x - x.mean()
    y_dm = y - y.mean()
    denom = float(np.sum(x_dm * x_dm))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_dm * y_dm) / denom)


_PPC_STATISTICS_REGISTRY: dict[str, Callable[[Mapping[str, np.ndarray]], float]] = {
    "y_mean":            _ppc_y_mean,
    "y_std":             _ppc_y_std,
    "log_k_mean":        _ppc_log_k_mean,
    "log_k_std":         _ppc_log_k_std,
    "log_k_growth_mean": _ppc_log_k_growth_mean,
    "y_log_k_slope":     _ppc_y_log_k_slope,
}


def run_posterior_predictive_check(
    spec: BayesianSpec,
    posterior_samples: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    config: PosteriorPredictiveConfig,
    seed: tuple[int, int],
) -> PosteriorPredictiveResult:
    """Posterior predictive check (Bayesian.md §1.4).

    Draws ``n_draws`` β from the (flattened) posterior, synthesizes one
    replicated panel per draw at the observed shape, and compares observed
    vs replicated summary statistics via Bayesian p-values
    ``P(T_rep > T_obs | y)``.  Values in [0.05, 0.95] are the rule-of-thumb
    "no glaring misfit" range; extremes flag features the model fails to
    reproduce.

    Args:
        spec:              BayesianSpec; ``synthesize_panel_fn`` is the
                           per-β replicator.
        posterior_samples: ``{name: (n_samples, n_chains) ndarray}`` from a
                           prior ``run_mcmc`` result.
        observed:          The observed panel dict matching
                           ``spec.observation_keys``.  Trailing singleton
                           dims are stripped before statistic evaluation.
        config:            PPC configuration (n_draws, statistics, burn_in).
        seed:              Base seed; draw-selection RNG and per-panel
                           synthesis seeds are derived via ``fold_in_seed``.

    Returns:
        PosteriorPredictiveResult with observed and replicated statistics
        and per-statistic Bayesian p-values.
    """

    start = time.perf_counter()

    unknown = [s for s in config.statistics if s not in _PPC_STATISTICS_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown PPC statistics: {unknown}. "
            f"Available: {sorted(_PPC_STATISTICS_REGISTRY)}."
        )

    missing = [k for k in spec.observation_keys if k not in observed]
    if missing:
        raise KeyError(
            f"observed is missing required keys: {missing}. "
            f"Required by spec.observation_keys: {spec.observation_keys}."
        )

    obs_panel = _panel_squeeze(observed, spec.observation_keys)
    # Use the first observation key to infer (N, T) — all keys share the
    # same panel shape by the spec contract.
    n_firms_obs, horizon_obs = obs_panel[spec.observation_keys[0]].shape[:2]

    observed_stats = {
        s: _PPC_STATISTICS_REGISTRY[s](obs_panel) for s in config.statistics
    }

    # Flatten posterior to (n_samples * n_chains,) per param so we can
    # subsample with a single index array.
    flat_posterior = {
        k: np.asarray(posterior_samples[k]).reshape(-1)
        for k in spec.parameter_names
    }
    n_total = next(iter(flat_posterior.values())).shape[0]
    if n_total < config.n_draws:
        raise ValueError(
            f"Posterior has only {n_total} samples but n_draws={config.n_draws}. "
            f"Reduce n_draws or run a longer MCMC."
        )

    select_seed_int = make_seed_int(seed, "ppc", "select")
    rng = np.random.default_rng(select_seed_int)
    indices = rng.choice(n_total, size=config.n_draws, replace=False)

    replicated_stats: dict[str, list[float]] = {s: [] for s in config.statistics}
    for d, idx in enumerate(indices):
        beta_d = {k: float(flat_posterior[k][int(idx)]) for k in spec.parameter_names}
        panel_seed = fold_in_seed(seed, "ppc", "draw", int(d))
        replicated_panel = spec.synthesize_panel_fn(
            beta_d, int(n_firms_obs), int(horizon_obs), config.burn_in, panel_seed)
        rep_panel = _panel_squeeze(replicated_panel, spec.observation_keys)
        for s in config.statistics:
            replicated_stats[s].append(_PPC_STATISTICS_REGISTRY[s](rep_panel))

    replicated_stats_arr = {
        s: np.asarray(replicated_stats[s], dtype=np.float64)
        for s in config.statistics
    }
    bayes_p_values = {
        s: float(np.mean(replicated_stats_arr[s] > observed_stats[s]))
        for s in config.statistics
    }

    return PosteriorPredictiveResult(
        observed_stats=observed_stats,
        replicated_stats=replicated_stats_arr,
        bayes_p_values=bayes_p_values,
        wall_time_sec=time.perf_counter() - start,
        metadata={
            "n_draws":      config.n_draws,
            "burn_in":      config.burn_in,
            "statistics":   list(config.statistics),
            "n_firms_obs":  int(n_firms_obs),
            "horizon_obs":  int(horizon_obs),
            "seed":         tuple(int(x) for x in seed),
        },
    )


# ---------------------------------------------------------------------------
# Public: prior sensitivity analysis (Bayesian.md §1.4)
# ---------------------------------------------------------------------------

@dataclass
class PriorSensitivityResult:
    """Output of ``run_prior_sensitivity``.

    Attributes:
        posterior_summaries: ``{variant_name: {param: {median, ci_lo, ci_hi,
                             r_hat, ess}}}`` per-variant per-param summary.
        movement_metrics:    ``{variant_name: {param: float}}`` where each
                             float is ``|median_v - median_default| /
                             ci_width_default``.  Bounded movement (< 0.25
                             rule of thumb) means the posterior is robust to
                             prior tightening / loosening on that param.
        default_ci_widths:   ``{param: ci_97.5 - ci_2.5}`` from the default
                             variant; used as the normalizer for the
                             movement metric.
        wall_time_sec:       Total wall time across all variant runs.
        metadata:            Variant kwargs, default-variant name, parameter
                             names, base seed.
    """

    posterior_summaries: dict[str, dict[str, dict[str, float]]]
    movement_metrics:    dict[str, dict[str, float]]
    default_ci_widths:   dict[str, float]
    wall_time_sec:       float
    metadata:            dict[str, Any] = field(default_factory=dict)


def run_prior_sensitivity(
    spec_factory: Callable[..., BayesianSpec],
    variants: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, np.ndarray],
    run_config: BayesianRunConfig,
    seed: tuple[int, int],
    default_variant_name: str = "default",
) -> PriorSensitivityResult:
    """Prior sensitivity analysis (Bayesian.md §1.4).

    Re-runs MCMC under K prior variants.  Each variant builds a new spec via
    ``spec_factory(**variant_kwargs)`` (typically forwarding the kwargs into
    ``make_closed_form_bayesian_spec`` or its NN equivalent).  Movement is
    measured against the variant named ``default_variant_name``.

    Args:
        spec_factory:          Callable taking prior-scale kwargs, returning
                               a BayesianSpec.  E.g.
                               ``partial(make_closed_form_bayesian_spec, param_env)``.
        variants:              ``{name: {kwarg: value}}``.  Each variant's
                               kwargs are unpacked into ``spec_factory``.
        observed:              Same observed data passed to all variants.
        run_config:            MCMC run config (shared across variants).  Per-
                               variant seeds are derived from ``seed``, so the
                               same ``run_config.master_seed`` here is fine.
        seed:                  Base seed; per-variant seeds via
                               ``fold_in_seed(seed, "sensitivity", variant_name)``.
        default_variant_name:  Name of the reference variant for the movement
                               metric.  Must be a key in ``variants``.

    Returns:
        PriorSensitivityResult with per-variant posterior summaries and per-
        variant movement metrics normalized by the default variant's CI width.
    """

    start = time.perf_counter()

    if default_variant_name not in variants:
        raise KeyError(
            f"default_variant_name={default_variant_name!r} not in variants. "
            f"Available: {sorted(variants)}."
        )

    posterior_summaries: dict[str, dict[str, dict[str, float]]] = {}
    parameter_names: Optional[tuple[str, ...]] = None

    for variant_name, variant_kwargs in variants.items():
        spec = spec_factory(**dict(variant_kwargs))
        if parameter_names is None:
            parameter_names = spec.parameter_names
        elif parameter_names != spec.parameter_names:
            raise ValueError(
                f"All variants must produce the same parameter_names. "
                f"Got {spec.parameter_names} for {variant_name!r}, "
                f"expected {parameter_names}."
            )

        variant_seed = fold_in_seed(seed, "sensitivity", variant_name)
        mcmc_result = run_mcmc(spec, observed, run_config, variant_seed)

        per_param: dict[str, dict[str, float]] = {}
        for name in parameter_names:
            samples = np.asarray(mcmc_result.posterior_samples[name]).reshape(-1)
            lo = float(np.quantile(samples, 0.025))
            hi = float(np.quantile(samples, 0.975))
            per_param[name] = {
                "median": float(np.median(samples)),
                "ci_lo":  lo,
                "ci_hi":  hi,
                "r_hat":  float(mcmc_result.r_hat[name]),
                "ess":    float(mcmc_result.ess[name]),
            }
        posterior_summaries[variant_name] = per_param

    assert parameter_names is not None  # guaranteed by the loop above

    default_summary = posterior_summaries[default_variant_name]
    default_ci_widths = {
        name: default_summary[name]["ci_hi"] - default_summary[name]["ci_lo"]
        for name in parameter_names
    }
    movement_metrics: dict[str, dict[str, float]] = {}
    for variant_name, summary in posterior_summaries.items():
        movement_metrics[variant_name] = {}
        for name in parameter_names:
            width = default_ci_widths[name]
            if width <= 0.0:
                movement_metrics[variant_name][name] = float("nan")
            else:
                shift = abs(summary[name]["median"] - default_summary[name]["median"])
                movement_metrics[variant_name][name] = float(shift / width)

    return PriorSensitivityResult(
        posterior_summaries=posterior_summaries,
        movement_metrics=movement_metrics,
        default_ci_widths=default_ci_widths,
        wall_time_sec=time.perf_counter() - start,
        metadata={
            "variants":             {k: dict(v) for k, v in variants.items()},
            "default_variant_name": default_variant_name,
            "parameter_names":      list(parameter_names),
            "seed":                 tuple(int(x) for x in seed),
        },
    )
