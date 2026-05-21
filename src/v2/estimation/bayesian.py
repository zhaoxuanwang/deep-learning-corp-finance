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
# Target log-prob construction
# ---------------------------------------------------------------------------

def _build_target_log_prob(spec: BayesianSpec,
                           observed_data: Mapping[str, tf.Tensor]):
    """Return a callable ``target_log_prob(*unconstrained_args) -> scalar``.

    Pieces together:
      - log prior on constrained scale (from spec.prior_distribution),
      - log |det J| of the constrained→unconstrained map (from spec.bijector),
      - log likelihood at the constrained parameters (from spec.log_likelihood_fn).
    """

    param_names = spec.parameter_names
    prior = spec.prior_distribution
    bijector = spec.bijector

    # Sanity: the bijector and prior must use the same key set.
    # JointMap stores its dict on .bijectors (in 0.24).
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
        # log_det may be returned as a dict or summed scalar depending on
        # JointMap's behaviour. Reduce defensively.
        if isinstance(log_det, Mapping):
            log_det = tf.add_n([tf.cast(v, log_prior.dtype) for v in log_det.values()])
        return log_prior + log_det + log_lik

    return target_log_prob


def _draw_initial_state(spec: BayesianSpec, n_chains: int,
                        seed: tuple[int, int]) -> list[tf.Tensor]:
    """Sample initial unconstrained states for all chains from the prior."""

    samples = spec.prior_distribution.sample(n_chains, seed=tf.constant(seed, tf.int32))
    # `samples` is a dict {name: tensor of shape [n_chains]}.
    unconstrained = spec.bijector.inverse({k: samples[k] for k in spec.parameter_names})
    return [unconstrained[k] for k in spec.parameter_names]


# ---------------------------------------------------------------------------
# Sampler dispatch
# ---------------------------------------------------------------------------

def _run_nuts(target_log_prob,
              initial_state: list[tf.Tensor],
              run_config: BayesianRunConfig,
              seed: tuple[int, int]):
    """NUTS via DualAveragingStepSizeAdaptation.

    We use the explicit DualAveraging + NoUTurnSampler stack instead of
    ``windowed_adaptive_nuts`` because (a) it accepts a custom
    ``target_log_prob_fn`` directly (windowed_adaptive_nuts insists on a
    JointDistribution, which would force us to fold the prior, bijector,
    and likelihood into a single coroutine and complicates the shape
    semantics around the chain batch dim), and (b) the windowed routine's
    mass-matrix adaptation is only mildly better than dual-averaging step-
    size adaptation for the 4-D Phase-1 target — not worth the API friction.
    """

    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob,
        step_size=[tf.constant(0.1, dtype=s.dtype) * tf.ones_like(s)
                   for s in initial_state],
    )
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(run_config.n_warmup * 0.8),
        target_accept_prob=tf.constant(run_config.target_accept_prob, dtype=initial_state[0].dtype),
        step_size_setter_fn=lambda pkr, new_step: pkr._replace(step_size=new_step),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
    )

    @tf.function(jit_compile=False)
    def _sample():
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=run_config.n_samples,
            num_burnin_steps=run_config.n_warmup,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
            seed=tf.constant(seed, tf.int32),
        )
        return samples, kernel_results

    samples, log_accept_ratio = _sample()
    accept_rate = float(tf.reduce_mean(tf.exp(tf.minimum(0.0, log_accept_ratio))))
    return samples, accept_rate


def _run_rw_mh(target_log_prob,
               initial_state: list[tf.Tensor],
               run_config: BayesianRunConfig,
               seed: tuple[int, int]):
    """Random-walk Metropolis-Hastings with simple step-size adaptation.

    Used when (a) the user requests a gradient-free baseline, or
    (b) the likelihood is a particle-filter estimate (PMMH).  In both
    cases the kernel below is correct as long as the likelihood estimate
    is unbiased (Andrieu, Doucet, Holenstein 2010, Bayesian.md §1.3).
    """

    kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_log_prob)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(run_config.n_warmup * 0.8),
        target_accept_prob=tf.constant(0.234, dtype=initial_state[0].dtype),
        adaptation_rate=0.05,
        new_step_size_fn=lambda step, _: step,
    )

    @tf.function(jit_compile=False)
    def _sample():
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=run_config.n_samples,
            num_burnin_steps=run_config.n_warmup,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
            seed=tf.constant(seed, tf.int32),
        )
        return samples, kernel_results

    samples, log_accept_ratio = _sample()
    accept_rate = float(tf.reduce_mean(tf.exp(tf.minimum(0.0, log_accept_ratio))))
    return samples, accept_rate


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
      1. Convert observed data to tf.Tensor and build the joint target.
      2. Sample n_chains independent initial points from the prior.
      3. Dispatch to NUTS or RW-MH per ``spec.sampler_kind``.
      4. Map samples back to constrained scale, compute R̂ / ESS.

    Args:
        spec:           BayesianSpec from a per-env factory.
        observed_data:  Dict matching ``spec.observation_keys``.  Arrays are
                        cast to float32 tensors before use.
        run_config:     Sampler runtime config.
        seed:           Length-2 int seed pair.  All subsequent seeds (init,
                        sampler) are derived from this via ``fold_in_seed``
                        so a single value reproduces the entire run.

    Returns:
        BayesianMCMCResult with per-parameter posterior samples, R̂, ESS,
        average acceptance probability, and wall time.
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
    target = _build_target_log_prob(spec, data_tensors)

    init_seed   = fold_in_seed(seed, "init")
    sample_seed = fold_in_seed(seed, "sample")

    initial_state = _draw_initial_state(spec, run_config.n_chains, init_seed)

    sampler_fn = _SAMPLERS[spec.sampler_kind]
    raw_samples, acceptance = sampler_fn(target, initial_state, run_config, sample_seed)

    # Map back to constrained scale and pack as {name: array shape (n_samples, n_chains)}.
    unconstrained_dict = {k: raw_samples[i]
                          for i, k in enumerate(spec.parameter_names)}
    constrained_dict = spec.bijector.forward(unconstrained_dict)

    posterior = {k: constrained_dict[k].numpy() for k in spec.parameter_names}
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
