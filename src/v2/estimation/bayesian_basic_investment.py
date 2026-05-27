"""Bayesian spec factories for the basic investment model (EKF + NUTS).

Implements the EKF + NUTS pipeline of Bayesian.md §2.4 / §3.4 for the
frictionless basic model. Exposes two sibling factories that share the
same likelihood, priors, bijectors, and synthesizer; only the policy
callable differs:

* ``make_neural_bayesian_spec(env, policy_nn)`` — uses a pre-trained
  ParameterizedPolicyNetwork as the EKF observation function.
* ``make_closed_form_bayesian_spec(env)`` — substitutes the analytical
  frictionless policy. Diagnostic / regression-test variant.

Two observation equations with full (μ, σ) residual structure
(Bayesian.md §1.2 / §2.3):

* Eq 1 (production): ``y_t = α log k_t + x_t + production_mean + η_t``
                     where η ~ N(0, production_std²)
* Eq 2 (capital LoM): ``log k_{t+1} = log φ(k_t, exp(x_t); β)
                                    + investment_mean + ξ^k_t``
                     where ξ^k ~ N(0, investment_std²)

with latent state ``x_t = log z_t`` and AR(1) transition unchanged. The 7
estimable parameters are (alpha, rho, sigma_epsilon, production_mean,
production_std, investment_mean, investment_std).

Eq 2 is **potentially nonlinear in the latent state** through φ (the
optimal policy, either closed-form analytical or a pre-trained NN
surrogate). We use an Extended Kalman Filter: at each step, evaluate
φ and its Jacobian d log φ / d x at the *predicted* latent mean
``m_{t|t-1}`` (per chain × firm), then run the standard Kalman update
with a 2-D observation and 2×1 observation matrix ``H_t = [[1], [H_2(t)]]``.

For the closed-form policy, log k'(z; β) is exactly linear in log z, so
H_2 = ρ/(1-α) is constant and EKF reduces to standard Kalman with zero
linearization error. This is the validation benchmark in §2.4 / §3.4.

Reproducibility (Bayesian.md §2.9): same master seed → bit-identical
posterior samples on CPU. All RNG paths derive from one master pair via
``fold_in_seed``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
)
from src.v2.estimation.bayesian import (
    BayesianSpec,
    FilterKind,
    SamplerKind,
)
from src.v2.estimation.beta_sampler import (
    BETA_DIM_NAMES,
    DEFAULT_UNIFORM_BOUNDS,
)
from src.v2.networks.policy import ParameterizedPolicyNetwork
from src.v2.utils.seeding import fold_in_seed

tfd = tfp.distributions
tfb = tfp.bijectors


PARAMETER_NAMES: tuple[str, ...] = (
    "alpha", "rho", "sigma_epsilon",
    "production_mean", "production_std",
    "investment_mean", "investment_std",
)
OBSERVATION_KEYS: tuple[str, ...] = ("y", "log_k", "log_k_next")


# ---------------------------------------------------------------------------
# Bijector helpers — Sigmoid composed with affine rescale to a finite box.
# ---------------------------------------------------------------------------

def _truncated_sigmoid_bijector(low: float, high: float) -> tfb.Bijector:
    """Bijector mapping ℝ → (low, high). Wraps ``tfb.Sigmoid(low, high)``.

    Paired with a ``tfd.Uniform(low, high)`` prior so the RW-MH chain (the
    only sampler path still consuming ``spec.bijector``) never asks the NN
    surrogate to extrapolate outside its training box. The NUTS path derives
    the bijector automatically from the prior and does not consult this.
    """
    if not (high > low):
        raise ValueError(
            f"_truncated_sigmoid_bijector requires high > low; "
            f"got low={low}, high={high}."
        )
    return tfb.Sigmoid(
        low=tf.constant(float(low),  tf.float32),
        high=tf.constant(float(high), tf.float32),
    )


# ---------------------------------------------------------------------------
# EKF likelihood — core piece of the factory.
# ---------------------------------------------------------------------------

def _build_ekf_log_likelihood(
    policy_nn,
    delta_rate: float,
    V0: float,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
):
    """Return a ``log_likelihood_fn(beta, data)`` closure for the EKF.

    Closure captures the (frozen) policy, the env's depreciation rate, and the
    prior latent variance. The returned function is differentiable in
    ``beta`` end-to-end (forward-mode JVP for H_2 composes cleanly with the
    outer reverse-mode NUTS tape).

    The ``policy_nn`` argument is duck-typed: any callable matching the
    standard ``(state, beta, training=False) -> action`` signature works.

    The ``k_min`` / ``k_max`` parameters bound ``k_next`` before ``log`` so
    the likelihood matches the env's analytical-policy behaviour (see
    ``_build_analytical_policy_fn`` below, which clips ``kp`` to
    ``[k_min, k_max]``).  Without clipping, the NN can occasionally drive
    ``k_next`` very negative at extreme β corners, producing
    ``log(1e-12) = -27.6`` outliers in the likelihood.  When either bound is
    ``None``, falls back to the old behaviour (``tf.maximum(k_next, 1e-12)``).

    Shape contract:

      * ``beta``: dict of tensors of shape ``[B...]`` (chain batch).
      * ``data["y"]``, ``["log_k"]``, ``["log_k_next"]``: ``[N, T, 1]``.

    Returns scalar ``[B...]`` log-likelihood (sum over firms × t).
    """

    delta_const = tf.constant(float(delta_rate), tf.float32)
    V0_const    = tf.constant(float(V0),         tf.float32)
    log_2pi     = tf.constant(float(np.log(2.0 * np.pi)), tf.float32)
    use_knext_clipping = (k_min is not None) and (k_max is not None)
    k_min_const = tf.constant(float(k_min), tf.float32) if use_knext_clipping else None
    k_max_const = tf.constant(float(k_max), tf.float32) if use_knext_clipping else None

    # ``@tf.function(reduce_retracing=True, jit_compile=True)`` caches the
    # EKF graph at the first MCMC leapfrog call so subsequent calls re-use
    # the compiled graph instead of paying Python-level trace overhead.
    # ``reduce_retracing`` allows shape-compatible reuse so we don't
    # retrace on minor batch-shape variations between warmup and sampling.
    # XLA is enabled now that the inner Jacobian uses ``tf.GradientTape``
    # (reverse-mode) rather than ``tf.autodiff.ForwardAccumulator``; the
    # two are computationally equivalent at 1-D scalar latent state, but
    # only GradientTape is XLA-compatible.
    @tf.function(reduce_retracing=True, jit_compile=True)
    def log_likelihood_fn(beta: Mapping[str, tf.Tensor],
                          data:  Mapping[str, tf.Tensor]) -> tf.Tensor:
        alpha     = tf.cast(beta["alpha"],            tf.float32)   # [B...]
        rho       = tf.cast(beta["rho"],              tf.float32)
        sigma_eps = tf.cast(beta["sigma_epsilon"],    tf.float32)
        prod_mean = tf.cast(beta["production_mean"],  tf.float32)
        prod_std  = tf.cast(beta["production_std"],   tf.float32)
        inv_mean  = tf.cast(beta["investment_mean"],  tf.float32)
        inv_std   = tf.cast(beta["investment_std"],   tf.float32)

        # 5-D β for the NN: (α, ρ, σ_ε, φ_quad=0, φ_prop=0).
        # Frictionless slice — the NN was trained with BetaSampler(freeze_dims=(3,4)).
        zeros_B = tf.zeros_like(alpha)
        beta_5d = tf.stack([alpha, rho, sigma_eps, zeros_B, zeros_B], axis=-1)  # [B..., 5]

        # Strip the trailing event-dim from the observation tensors.
        y          = tf.cast(data["y"][..., 0],          tf.float32)   # [N, T]
        log_k      = tf.cast(data["log_k"][..., 0],      tf.float32)
        log_k_next = tf.cast(data["log_k_next"][..., 0], tf.float32)
        k          = tf.exp(log_k)                                      # [N, T]

        N_static = int(y.shape[0]) if y.shape[0] is not None else tf.shape(y)[0]
        T_static = int(y.shape[1]) if y.shape[1] is not None else tf.shape(y)[1]
        if not isinstance(T_static, int):
            raise ValueError(
                "EKF likelihood requires a static second dimension on `y` "
                "(unrolled Python loop). Got dynamic shape."
            )

        # Broadcast targets: m, V are [B..., N]; β scalars expand to [B..., 1].
        # `tf.shape(alpha)` may be rank 0 (single eval) or rank 1 ([n_chains]).
        BN_shape = tf.concat([tf.shape(alpha), [N_static]], axis=0)

        m_prev = tf.zeros(BN_shape, dtype=tf.float32)
        V_prev = tf.fill(BN_shape, V0_const)

        rho_e        = rho[..., tf.newaxis]               # [B..., 1]
        alpha_e      = alpha[..., tf.newaxis]
        sig_eps_2    = (sigma_eps ** 2)[..., tf.newaxis]
        prod_mean_e  = prod_mean[..., tf.newaxis]
        prod_var     = (prod_std ** 2)[..., tf.newaxis]
        inv_mean_e   = inv_mean[..., tf.newaxis]
        inv_var      = (inv_std  ** 2)[..., tf.newaxis]

        log_lik_per_firm = tf.zeros(BN_shape, dtype=tf.float32)

        for t_idx in range(T_static):
            k_t          = k[:, t_idx]            # [N]
            y_t          = y[:, t_idx]            # [N]
            log_k_t      = log_k[:, t_idx]        # [N]
            log_k_next_t = log_k_next[:, t_idx]   # [N]

            # 1. Predict latent state (AR(1)).
            m_pred = rho_e * m_prev                              # [B..., N]
            V_pred = (rho_e ** 2) * V_prev + sig_eps_2           # [B..., N]

            # 2. Compute g = log k_{t+1}^* and H_2 = ∂g/∂x at x = m_pred via
            #    reverse-mode autodiff (XLA-compatible at 1-D scalar latent state).
            k_t_b = tf.broadcast_to(k_t, tf.shape(m_pred))            # [B..., N]

            beta_5d_b = tf.broadcast_to(
                beta_5d[..., tf.newaxis, :],
                tf.concat([tf.shape(beta_5d)[:-1], [N_static, 5]], axis=0),
            )                                                       # [B..., N, 5]
            with tf.GradientTape() as tape:
                tape.watch(m_pred)
                z_pred = tf.exp(m_pred)                              # [B..., N]
                state = tf.stack([k_t_b, z_pred], axis=-1)           # [B..., N, 2]
                state_flat   = tf.reshape(state,     [-1, 2])
                beta_5d_flat = tf.reshape(beta_5d_b, [-1, 5])
                action_flat  = policy_nn(state_flat, beta_5d_flat,
                                          training=False)            # [..., 1]
                action = tf.reshape(action_flat[..., 0], tf.shape(m_pred))
                k_next = (1.0 - delta_const) * k_t_b + action
                # Clip k_next to physical bounds [k_min, k_max] when
                # available; matches the env's analytical-policy
                # behaviour and prevents log(near-zero) outliers from
                # NN extrapolation at unusual β corners.  Falls back to
                # the old `max(k_next, 1e-12)` safeguard otherwise.
                if use_knext_clipping:
                    k_next_safe = tf.clip_by_value(k_next, k_min_const, k_max_const)
                else:
                    k_next_safe = tf.maximum(k_next, 1e-12)
                g_pred = tf.math.log(k_next_safe)                    # [B..., N]
            H_2 = tape.gradient(g_pred, m_pred)                      # [B..., N]

            # 3. Innovation. Subtract residual mean shifts (Bayesian.md §2.4).
            nu_1 = y_t - alpha_e * log_k_t - m_pred - prod_mean_e   # [B..., N]
            nu_2 = log_k_next_t - g_pred - inv_mean_e               # [B..., N]

            # 4. Observation covariance S = H V_pred Hᵀ + R   (closed form, 2×2).
            #    H = [[1], [H_2]],  R = diag(production_std², investment_std²)
            s11 = V_pred + prod_var                                 # [B..., N]
            s12 = H_2 * V_pred                                      # [B..., N]
            s22 = (H_2 ** 2) * V_pred + inv_var                     # [B..., N]
            det_S = s11 * s22 - s12 ** 2                            # [B..., N]

            # 5. Quadratic form νᵀ S⁻¹ ν = (s22 ν₁² − 2 s12 ν₁ ν₂ + s11 ν₂²) / det_S
            quad = (s22 * nu_1 ** 2
                    - 2.0 * s12 * nu_1 * nu_2
                    + s11 * nu_2 ** 2) / det_S

            log_lik_per_firm = log_lik_per_firm - 0.5 * (
                quad + tf.math.log(det_S) + 2.0 * log_2pi
            )

            # 6. Kalman gain K = V_pred Hᵀ S⁻¹ (1×2):
            #      K_1 = V_pred · (s22 - H_2·s12) / det_S
            #      K_2 = V_pred · (-s12 + H_2·s11) / det_S
            K_1 = V_pred * (s22 - H_2 * s12) / det_S
            K_2 = V_pred * (-s12 + H_2 * s11) / det_S

            # 7. Update: m_post = m_pred + K ν;  V_post = V_pred (1 - K H).
            m_post = m_pred + K_1 * nu_1 + K_2 * nu_2
            V_post = V_pred * (1.0 - K_1 - K_2 * H_2)

            m_prev = m_post
            V_prev = V_post

        # Sum over firms → [B...].
        return tf.reduce_sum(log_lik_per_firm, axis=-1)

    return log_likelihood_fn


# ---------------------------------------------------------------------------
# Diagnostic helper — exposed for the notebook's "Sec 2 sanity check".
# ---------------------------------------------------------------------------

def diagnose_nn_linear_slope(
    env: ParameterizedBasicInvestmentEnv,
    policy_nn: ParameterizedPolicyNetwork,
    beta: Mapping[str, float],
    *,
    z_anchor: Optional[float] = None,
    k_anchor: Optional[float] = None,
) -> dict[str, float]:
    """Probe the NN's log-k' slope vs log z at a single (k_anchor, z_anchor) point.

    Returns a dict with the NN-implied slope, the analytical reference
    ``ρ / (1 - α)``, and the relative error. Used by NB09 Sec 2 to gate the
    EKF integration test on NN linearization quality.

    Args:
        env:       ParameterizedBasicInvestmentEnv (only ``delta_rate``,
                   ``mu``, ``k_star``, and ``r_rate`` are read).
        policy_nn: trained ParameterizedPolicyNetwork, frictionless slice.
        beta:      dict with keys (alpha, rho, sigma_epsilon, …); friction
                   dims default to 0 if absent.
        z_anchor:  scalar z (in levels) at which to probe; default ``exp(μ)``.
        k_anchor:  scalar k (in levels) at which to probe; default ``env.k_star``.
                   The frictionless closed-form k' is independent of k, so the
                   choice only stresses-tests whether the NN learned that
                   independence — match it to env.k_star for the standard
                   diagnostic.

    Returns:
        {"slope_nn", "slope_analytical", "rel_error",
         "intercept_nn", "intercept_analytical"}.
    """
    alpha = float(beta["alpha"])
    rho   = float(beta["rho"])
    sigma_eps = float(beta["sigma_epsilon"])
    z = float(z_anchor) if z_anchor is not None else float(np.exp(env.mu))
    k_star = float(k_anchor) if k_anchor is not None else float(env.k_star)
    log_z = float(np.log(max(z, 1e-12)))

    beta_5d = tf.constant([[alpha, rho, sigma_eps, 0.0, 0.0]], dtype=tf.float32)
    m_pred  = tf.constant([log_z], dtype=tf.float32)
    k_b     = tf.constant([k_star], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(m_pred)
        z_pred = tf.exp(m_pred)
        state_t = tf.stack([k_b, z_pred], axis=-1)        # [1, 2]
        action = policy_nn(state_t, beta_5d, training=False)
        k_next = (1.0 - env.delta_rate) * k_b + action[..., 0]
        g_pred = tf.math.log(tf.maximum(k_next, 1e-12))
    slope_nn = float(tape.gradient(g_pred, m_pred).numpy()[0])
    intercept_nn = float(g_pred.numpy()[0]) - slope_nn * log_z

    slope_analytical = rho / (1.0 - alpha)
    # κ(β) = (log α + σ_ε²/2 - log(r + δ)) / (1 - α)   (Bayesian.md §2.2)
    kappa = (np.log(alpha) + 0.5 * sigma_eps**2
             - np.log(env.r_rate + env.delta_rate)
             ) / (1.0 - alpha)
    intercept_analytical = float(kappa)
    denom = max(abs(slope_analytical), 1e-12)
    rel_error = abs(slope_nn - slope_analytical) / denom

    return {
        "slope_nn":             slope_nn,
        "slope_analytical":     slope_analytical,
        "rel_error":            float(rel_error),
        "intercept_nn":         intercept_nn,
        "intercept_analytical": intercept_analytical,
    }


# ---------------------------------------------------------------------------
# Save / load: persist the normalizer state alongside Keras weights.
# ---------------------------------------------------------------------------
#
# Why this helper exists: ``ParameterizedPolicyNetwork`` carries a
# ``StaticNormalizer`` z-score normalizer whose ``mean`` and ``std`` are
# ``tf.Module`` variables (not Keras-tracked layer weights). The trainer
# fits these from the training distribution; without them, the NN sees
# un-normalized inputs and silently saturates at the action bounds.
# Keras's ``Model.save_weights`` does **not** capture these — only the
# Dense layers' kernels and biases. Saving the H5 weights alone produces a
# file that loads to a degenerate model: the slope vs log z collapses to
# zero, posterior recovery is meaningless.
#
# Fix: write a sidecar ``.normalizer.npz`` next to the weights file with
# the normalizer's mean and std arrays, and reload both atomically.
# ``load_policy_with_normalizer`` raises if the sidecar is missing so the
# failure is loud, not silent.

def save_policy_with_normalizer(
    policy_nn: ParameterizedPolicyNetwork,
    weights_path,
) -> dict[str, "Path"]:
    """Save ``policy_nn.weights`` AND the normalizer mean/std.

    Args:
        policy_nn:    The ParameterizedPolicyNetwork to persist.
        weights_path: Path-like. The Keras weights go to this exact path;
                      the normalizer sidecar goes to ``<weights_path>.normalizer.npz``
                      (file's ``.h5`` extension preserved on the main file).

    Returns:
        ``{"weights": Path, "normalizer": Path}`` for downstream logging.
    """
    from pathlib import Path
    p = Path(weights_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    policy_nn.save_weights(p)
    sidecar = p.with_name(p.name + ".normalizer.npz")
    np.savez(
        sidecar,
        mean=policy_nn.normalizer.mean.numpy(),
        std=policy_nn.normalizer.std.numpy(),
    )
    return {"weights": p, "normalizer": sidecar}


def load_policy_with_normalizer(
    policy_nn: ParameterizedPolicyNetwork,
    weights_path,
) -> None:
    """Inverse of ``save_policy_with_normalizer``.

    Raises:
        FileNotFoundError: if the ``.normalizer.npz`` sidecar is missing.
        The Keras weights alone are useless — the normalizer would reset
        to (mean=0, std=1) and the NN saturates at action bounds.
    """
    from pathlib import Path
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Weights file not found: {p}")
    sidecar = p.with_name(p.name + ".normalizer.npz")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Normalizer sidecar not found: {sidecar}. "
            "Keras save_weights does not persist StaticNormalizer state "
            "(it is a tf.Module submodule, not a Keras layer). Saved "
            "weights alone produce a degenerate model that saturates "
            "at action bounds. Re-save using "
            "`save_policy_with_normalizer(...)`."
        )
    policy_nn.load_weights(p)
    data = np.load(sidecar)
    policy_nn.normalizer.mean.assign(tf.constant(data["mean"], tf.float32))
    policy_nn.normalizer.std.assign(tf.constant(data["std"], tf.float32))


# ---------------------------------------------------------------------------
# Public: factory.
# ---------------------------------------------------------------------------

def _make_ekf_bayesian_spec(
    env: ParameterizedBasicInvestmentEnv,
    policy_callable,
    *,
    production_mean_prior_scale: float,
    production_std_prior_scale:  float,
    investment_mean_prior_scale: float,
    investment_std_prior_scale:  float,
    initial_state_variance: float,
    uniform_bounds: Optional[Mapping[str, tuple[float, float]]],
) -> BayesianSpec:
    """Internal helper. Build an EKF BayesianSpec from any policy callable.

    Both ``make_neural_bayesian_spec`` (NN surrogate) and
    ``make_closed_form_bayesian_spec`` (analytical regression test) delegate
    here. The only thing they vary is ``policy_callable``: the NN model in
    the production path, ``env.analytical_policy`` in the diagnostic path.
    Priors, bijectors, observation equations, and synthesizer are identical.
    """
    V0 = float(initial_state_variance)
    if V0 <= 0:
        raise ValueError(f"initial_state_variance must be > 0. Got {V0}.")

    bounds = dict(DEFAULT_UNIFORM_BOUNDS)
    if uniform_bounds is not None:
        bounds.update({k: tuple(v) for k, v in uniform_bounds.items()})

    a_lo, a_hi = bounds["alpha"]
    r_lo, r_hi = bounds["rho"]
    s_lo, s_hi = bounds["sigma_epsilon"]

    # ---- Priors (constrained scale; see Bayesian.md §1.2 baseline table) ---
    prior = tfd.JointDistributionNamed({
        "alpha":            tfd.Uniform(low=tf.constant(a_lo, tf.float32),
                                        high=tf.constant(a_hi, tf.float32)),
        "rho":              tfd.Uniform(low=tf.constant(r_lo, tf.float32),
                                        high=tf.constant(r_hi, tf.float32)),
        "sigma_epsilon":    tfd.Uniform(low=tf.constant(s_lo, tf.float32),
                                        high=tf.constant(s_hi, tf.float32)),
        "production_mean":  tfd.Normal(loc=tf.constant(0.0, tf.float32),
                                       scale=tf.constant(production_mean_prior_scale, tf.float32)),
        "production_std":   tfd.HalfNormal(scale=tf.constant(production_std_prior_scale, tf.float32)),
        "investment_mean":  tfd.Normal(loc=tf.constant(0.0, tf.float32),
                                       scale=tf.constant(investment_mean_prior_scale, tf.float32)),
        "investment_std":   tfd.HalfNormal(scale=tf.constant(investment_std_prior_scale, tf.float32)),
    })

    # ---- Bijectors: unconstrained ℝ⁷ → constrained support -----------------
    bijector = tfb.JointMap({
        "alpha":            _truncated_sigmoid_bijector(a_lo, a_hi),
        "rho":              _truncated_sigmoid_bijector(r_lo, r_hi),
        "sigma_epsilon":    _truncated_sigmoid_bijector(s_lo, s_hi),
        "production_mean":  tfb.Identity(),
        "production_std":   tfb.Exp(),
        "investment_mean":  tfb.Identity(),
        "investment_std":   tfb.Exp(),
    })

    # ---- Likelihood (EKF) --------------------------------------------------
    # Pass env's k bounds so the raw-NN path clips k_next to [k_min, k_max]
    # before log, matching the analytical policy's tf.clip_by_value. The
    # CF path is unaffected (its k_next is already in range by construction).
    log_likelihood_fn = _build_ekf_log_likelihood(
        policy_nn=policy_callable,
        delta_rate=float(env.delta_rate),
        V0=V0,
        k_min=float(env.k_min),
        k_max=float(env.k_max),
    )

    # ---- Synthetic panel ---------------------------------------------------
    # Closed-form analytical policy generates the panel — we know ground-truth β
    # and use that as the recovery target. Output keys (y, log_k, log_k_next)
    # are aligned with the 2-D observation that the EKF likelihood expects.
    #
    # The simulator is `env.simulate_panel(...)` from the parameterized env,
    # which uses the same `sample_initial_*` + `exogenous_transition` +
    # `_apply_action` that the NN was trained on. Single source of truth ⇒
    # no state-space drift between training and inference.

    def synthesize_panel_fn(beta: Mapping[str, float],
                            n_firms: int,
                            horizon: int,
                            burn_in: int,
                            seed: tuple[int, int]) -> dict[str, np.ndarray]:
        # Simulate ``horizon + 1`` raw periods, then slice to ``horizon``
        # model-time observations (we lose one tail period because Eq 2
        # observes log k_{t+1}, which requires a one-step-ahead capital).
        raw_horizon = int(horizon) + 1
        beta_5d = tf.constant([
            float(beta["alpha"]),
            float(beta["rho"]),
            float(beta["sigma_epsilon"]),
            0.0,  # phi_quad — frictionless slice
            0.0,  # phi_prop — frictionless slice
        ], dtype=tf.float32)
        alpha_f         = float(beta["alpha"])
        production_mean = float(beta["production_mean"])
        production_std  = float(beta["production_std"])
        investment_mean = float(beta["investment_mean"])
        investment_std  = float(beta["investment_std"])

        panel = env.simulate_panel(
            beta=beta_5d,
            policy_fn=env.analytical_policy,
            n_firms=int(n_firms),
            horizon=raw_horizon,
            burn_in=int(burn_in),
            seed=tuple(int(x) for x in seed),
        )
        k = panel["k"]   # (N, raw_horizon)
        z = panel["z"]   # (N, raw_horizon)

        log_k_optimal = np.log(np.maximum(k, 1e-12))
        log_z_raw     = np.log(np.maximum(z, 1e-12))

        # ξ noise on log k: log k_observed = log k_optimal + investment_mean + xi.
        xi_seed = fold_in_seed(seed, "obs_noise_xi")
        xi = tf.random.stateless_normal(
            shape=[int(n_firms), raw_horizon],
            seed=tf.constant(xi_seed, tf.int32),
            stddev=tf.constant(investment_std, tf.float32),
        ).numpy().astype(np.float64)
        log_k_observed = log_k_optimal + investment_mean + xi

        # η noise on y: y_t = log z_t + α log k_observed_t + production_mean + η_t.
        eta_seed = fold_in_seed(seed, "obs_noise_eta")
        eta = tf.random.stateless_normal(
            shape=[int(n_firms), raw_horizon],
            seed=tf.constant(eta_seed, tf.int32),
            stddev=tf.constant(production_std, tf.float32),
        ).numpy().astype(np.float64)
        y_raw = log_z_raw + alpha_f * log_k_observed + production_mean + eta

        # Slice to T = horizon model-time observations:
        #   y_t, log_k_t at original t = 0 .. T-1
        #   log_k_{t+1} at original t = 1 .. T
        y_out          = y_raw[:, :int(horizon)]
        log_k_out      = log_k_observed[:, :int(horizon)]
        log_k_next_out = log_k_observed[:, 1:int(horizon) + 1]

        return {
            "y":          y_out[..., np.newaxis].astype(np.float32),
            "log_k":      log_k_out[..., np.newaxis].astype(np.float32),
            "log_k_next": log_k_next_out[..., np.newaxis].astype(np.float32),
            "metadata": {
                "true_beta": dict(beta),
                "seed":      tuple(int(x) for x in seed),
                "n_firms":   int(n_firms),
                "horizon":   int(horizon),
                "burn_in":   int(burn_in),
            },
        }

    return BayesianSpec(
        parameter_names=PARAMETER_NAMES,
        prior_distribution=prior,
        bijector=bijector,
        filter_kind=FilterKind.KALMAN,
        sampler_kind=SamplerKind.NUTS,
        log_likelihood_fn=log_likelihood_fn,
        synthesize_panel_fn=synthesize_panel_fn,
        observation_keys=OBSERVATION_KEYS,
    )


def make_neural_bayesian_spec(
    env: ParameterizedBasicInvestmentEnv,
    policy_nn: ParameterizedPolicyNetwork,
    *,
    production_mean_prior_scale: float = 0.1,
    production_std_prior_scale:  float = 0.5,
    investment_mean_prior_scale: float = 0.1,
    investment_std_prior_scale:  float = 0.5,
    initial_state_variance: float = 10.0,
    uniform_bounds: Optional[Mapping[str, tuple[float, float]]] = None,
) -> BayesianSpec:
    """Build the §3.4 neural-surrogate ``BayesianSpec``.

    Priors follow Bayesian.md §1.2 baseline table: α, ρ, σ_ε get Uniform
    priors aligned with the NN training box; the four residual parameters
    (production_mean, production_std, investment_mean, investment_std)
    get Normal(0, 0.1) / HalfNormal(0.5) priors centered/peaked at zero.
    Filter / sampler: EKF (labeled as KALMAN in the spec, since the
    generic dispatch doesn't distinguish) + NUTS.

    **State-space alignment contract.** Both the NN training data and the
    synthetic panel come from this same ``ParameterizedBasicInvestmentEnv``:
    training data via ``sample_initial_*`` + ER/SHAC's internal rollout,
    synthetic panel via ``env.simulate_panel``. Same bounds, same initial
    distribution, same transitions.

    Args:
        env:                            ParameterizedBasicInvestmentEnv in the
                                        frictionless case (nominal_econ.cost_convex
                                        == 0 and cost_fixed == 0; asserted).
        policy_nn:                      Pre-trained ParameterizedPolicyNetwork.
                                        Set ``policy_nn.trainable = False`` first.
        production_mean_prior_scale:    Normal scale for μ_η.
        production_std_prior_scale:     HalfNormal scale for σ_η.
        investment_mean_prior_scale:    Normal scale for μ_ξ.
        investment_std_prior_scale:     HalfNormal scale for σ_ξ.
        initial_state_variance:         V_0 (Bayesian.md §2.3) — diffuse prior on x_1.
        uniform_bounds:                 Override per-coordinate (low, high) box for
                                        α, ρ, σ_ε. ``None`` → ``DEFAULT_UNIFORM_BOUNDS``.

    Returns:
        BayesianSpec with filter_kind=KALMAN, sampler_kind=NUTS, and a
        synthesize_panel_fn closing over ``env``.
    """
    return _make_ekf_bayesian_spec(
        env=env,
        policy_callable=policy_nn,
        production_mean_prior_scale=production_mean_prior_scale,
        production_std_prior_scale=production_std_prior_scale,
        investment_mean_prior_scale=investment_mean_prior_scale,
        investment_std_prior_scale=investment_std_prior_scale,
        initial_state_variance=initial_state_variance,
        uniform_bounds=uniform_bounds,
    )


def _build_analytical_policy_fn(env: ParameterizedBasicInvestmentEnv):
    """Closed-form frictionless policy with the policy_nn callable signature.

    ``env.analytical_policy`` is graph-incompatible inside ``@tf.function``
    because its inner ``_assert_frictionless`` helper formats symbolic
    tensors in an f-string. We bypass it here: the EKF likelihood pins
    phi_quad = phi_prop = 0 in ``beta_5d`` (see ``_build_ekf_log_likelihood``),
    so the assertion is trivially satisfied and can be skipped.

    Returns a function with signature ``(state, beta, training=False) -> action``
    matching the ParameterizedPolicyNetwork ``__call__``.
    """
    mu              = tf.constant(float(env.mu),                tf.float32)
    delta           = tf.constant(float(env.delta_rate),        tf.float32)
    k_min           = tf.constant(float(env.k_min),             tf.float32)
    k_max           = tf.constant(float(env.k_max),             tf.float32)
    log_r_plus_d    = tf.constant(float(np.log(env.r_rate + env.delta_rate)),
                                  tf.float32)

    def analytical_policy_fn(state, beta, training=False):
        del training
        k     = state[..., 0:1]           # [..., 1]
        z     = state[..., 1:2]           # [..., 1]
        alpha = beta[...,  0:1]
        rho   = beta[...,  1:2]
        sigma = beta[...,  2:3]
        log_z   = tf.math.log(tf.maximum(z, 1e-8))
        log_ez  = (1.0 - rho) * mu + rho * log_z + 0.5 * sigma * sigma
        log_brk = tf.math.log(alpha) + log_ez - log_r_plus_d
        kp      = tf.exp(log_brk / (1.0 - alpha))
        kp      = tf.clip_by_value(kp, k_min, k_max)
        return kp - (1.0 - delta) * k

    return analytical_policy_fn


def make_closed_form_bayesian_spec(
    env: ParameterizedBasicInvestmentEnv,
    *,
    production_mean_prior_scale: float = 0.1,
    production_std_prior_scale:  float = 0.5,
    investment_mean_prior_scale: float = 0.1,
    investment_std_prior_scale:  float = 0.5,
    initial_state_variance: float = 10.0,
    uniform_bounds: Optional[Mapping[str, tuple[float, float]]] = None,
) -> BayesianSpec:
    """Closed-form regression-test variant of ``make_neural_bayesian_spec``.

    Diagnostic factory: substitutes ``env.analytical_policy`` for the NN
    inside the same EKF likelihood. Priors, bijectors, observation
    equations, and synthesizer are identical to the NN spec; only the
    policy callable differs.

    Use case. When the neural-surrogate spec produces a biased posterior,
    swap to this spec and rerun. On the frictionless slice the analytical
    policy is exactly linear in log z, so the EKF linearization is zero-
    error and the data exactly matches the model. If this spec recovers
    ground-truth β within its CIs but the NN spec does not, the bias in
    the NN spec comes from the NN approximation, not the inference machinery.
    """
    return _make_ekf_bayesian_spec(
        env=env,
        policy_callable=_build_analytical_policy_fn(env),
        production_mean_prior_scale=production_mean_prior_scale,
        production_std_prior_scale=production_std_prior_scale,
        investment_mean_prior_scale=investment_mean_prior_scale,
        investment_std_prior_scale=investment_std_prior_scale,
        initial_state_variance=initial_state_variance,
        uniform_bounds=uniform_bounds,
    )


# ---------------------------------------------------------------------------
# Standalone panel generation — pure math, no env required.
# ---------------------------------------------------------------------------
#
# The frictionless basic-investment model has a closed-form policy
# (Strebulaev-Whited 2012, §3.1):
#
#     log k_{t+1}(z; β) = ρ/(1-α) · log z_t + κ(β)
#     κ(β) = (log α + σ_ε²/2 - log(r + δ)) / (1 - α)
#
# and z is plain AR(1). Both are expressible in pure TF/numpy without any
# env instance. By generating the panel first (without an env), we can
# *derive* the env's k bounds from the panel's empirical range — which
# eliminates the need for the user-supplied k_min_mult / k_max_mult
# multipliers and aligns NN training distribution with where MCMC will
# actually evaluate the surrogate.
#
# All randomness flows through tf.random.stateless_* keyed by seeds
# derived from one master pair via fold_in_seed. Same MASTER_SEED →
# bit-identical panel on the same hardware.

def generate_frictionless_panel(
    beta: Mapping[str, float],
    *,
    interest_rate: float,
    depreciation_rate: float,
    mu: float = 0.0,
    n_firms: int,
    horizon: int,
    burn_in: int,
    seed: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Closed-form frictionless rollout, observation-ready (Bayesian.md §2.3).

    Generates an observable firm-year panel under the Strebulaev-Whited
    (2012, §3.1) analytical policy plus the residual structure prescribed
    in Bayesian.md §1.2 / §2.3:

      log z_{t+1}     = (1-ρ)μ + ρ log z_t + σ_ε ε_t       (latent AR(1))
      log k_optimal_t = ρ/(1-α) · log z_{t-1} + κ(β)        (closed-form policy)
      log k_observed_t = log k_optimal_t + μ_ξ + σ_ξ ζ_t    (firm under/over-invest)
      y_t              = log z_t + α log k_observed_t       (production)
                          + μ_η + σ_η η_t

    The latent z is used internally for the rollout and is NOT returned,
    eliminating any risk of latent-variable leakage into inference.

    Args:
        beta:               dict with the 7 keys required by the inference
                            framework (alpha, rho, sigma_epsilon,
                            production_mean, production_std,
                            investment_mean, investment_std).
        interest_rate:      r in the analytical formula.
        depreciation_rate:  δ in the analytical formula.
        mu:                 unconditional mean of log z; default 0.
        n_firms:            N — number of firms.
        horizon:            T — recorded model-time observations.
        burn_in:            periods discarded so the panel reaches its
                            stationary distribution. Recommend ≥ 30 for
                            ρ ≤ 0.9.
        seed:               (m0, m1) master seed for all randomness.

    Returns:
        ``{"y": (N, T), "log_k": (N, T), "log_k_next": (N, T)}`` — all
        observation-ready, with residual noise already applied. Latent
        ``z`` is not returned.

    Reproducibility: same ``seed`` → bit-identical arrays on the same
    hardware (CPU). Sub-seeds for initial state, AR(1) shocks, η, and ξ
    are derived via ``fold_in_seed(seed, ...)``.
    """
    alpha           = float(beta["alpha"])
    rho             = float(beta["rho"])
    sigma           = float(beta["sigma_epsilon"])
    production_mean = float(beta["production_mean"])
    production_std  = float(beta["production_std"])
    investment_mean = float(beta["investment_mean"])
    investment_std  = float(beta["investment_std"])
    r     = float(interest_rate)
    delta = float(depreciation_rate)
    mu_f  = float(mu)
    N     = int(n_firms)
    Tp    = int(horizon)
    Tb    = int(burn_in)
    # Need one extra raw period because Eq 2 observes log k_{t+1}.
    Ttot  = Tb + Tp + 1

    # κ(β) for the closed-form k'(z; β) = exp(ρ/(1-α) · log z + κ)
    kappa = (np.log(alpha) + 0.5 * sigma ** 2 - np.log(r + delta)) / (1.0 - alpha)
    slope = rho / (1.0 - alpha)

    # Initial conditions: draw log z_0 from the stationary distribution
    # N(μ, σ²/(1-ρ²)). Burn-in damps any residual transient.
    sigma_stat = sigma / np.sqrt(max(1.0 - rho ** 2, 1e-8))

    seed_z0  = tf.constant(fold_in_seed(seed, "panel", "z0"),  tf.int32)
    seed_eps = tf.constant(fold_in_seed(seed, "panel", "eps"), tf.int32)
    seed_eta = tf.constant(fold_in_seed(seed, "panel", "eta_noise"), tf.int32)
    seed_xi  = tf.constant(fold_in_seed(seed, "panel", "xi_noise"),  tf.int32)

    log_z0 = (mu_f + sigma_stat * tf.random.stateless_normal(
        [N], seed=seed_z0, dtype=tf.float32)).numpy().astype(np.float64)
    eps = tf.random.stateless_normal(
        [N, Ttot], seed=seed_eps, dtype=tf.float32).numpy().astype(np.float64)
    eta_draws = tf.random.stateless_normal(
        [N, Ttot], seed=seed_eta, dtype=tf.float32).numpy().astype(np.float64)
    xi_draws = tf.random.stateless_normal(
        [N, Ttot], seed=seed_xi, dtype=tf.float32).numpy().astype(np.float64)

    log_z         = np.empty((N, Ttot), dtype=np.float64)
    log_k_optimal = np.empty((N, Ttot), dtype=np.float64)

    log_z_prev = log_z0
    for t in range(Ttot):
        log_z_t = (1.0 - rho) * mu_f + rho * log_z_prev + sigma * eps[:, t]
        log_z[:, t] = log_z_t
        # Standard model timing: log k at panel index t was decided one step
        # earlier from z at panel index t-1, so policy(log_z_prev).
        log_k_optimal[:, t] = slope * log_z_prev + kappa
        log_z_prev = log_z_t

    # Residual structure: ξ on log k (investment policy deviation), η on y.
    log_k_observed = log_k_optimal + investment_mean + investment_std * xi_draws
    y_raw = log_z + alpha * log_k_observed + production_mean + production_std * eta_draws

    # Slice off burn-in, align timing: y_t and log_k_t at original t=Tb..Tb+T-1;
    # log_k_{t+1} at original t=Tb+1..Tb+T.
    y          = y_raw[:, Tb:Tb + Tp]
    log_k      = log_k_observed[:, Tb:Tb + Tp]
    log_k_next = log_k_observed[:, Tb + 1:Tb + Tp + 1]

    return {
        "y":          y,
        "log_k":      log_k,
        "log_k_next": log_k_next,
    }


# ---------------------------------------------------------------------------
# Derive env bounds from the observed panel + prior.
# ---------------------------------------------------------------------------

def derive_env_bounds_from_panel(
    panel_k: np.ndarray,
    *,
    k_margin: float = 1.2,
    log_z_half_width: float = 1.0,
    mu: float = 0.0,
) -> dict[str, float]:
    """Compute (k_min, k_max, z_min, z_max) for the env from data only.

    **k bounds** come from the observed panel: ``[k.min/k_margin,
    k.max·k_margin]``.

    **z bounds** are a **fixed reasonable range** in log-z space:
    ``log z ∈ [μ - log_z_half_width, μ + log_z_half_width]``. That is
    *deliberately independent of the panel's z and the prior*:

    * Inference treats z as latent — we cannot inspect ``panel_z`` even
      in the synthetic test, because that would cheat the realistic
      contract that the inference pipeline only sees ``k`` and ``y``.
    * Sizing z to the prior's worst-case ``σ_stat`` (``= σ_ε / sqrt(1-ρ²)``
      at the prior corner) blows up to 6 orders of magnitude under
      standard priors — the NN must then learn over a state space where
      the optimal action is structurally infeasible (the closed-form
      ``k'(z; β)`` at extreme z far exceeds any sensible action bound),
      gradients collapse, and the NN saturates at the action clip.

    Why ``log_z_half_width = 1.0`` is the right default:

    * Inside MCMC, the EKF's predicted latent mean ``m_{t|t-1}`` is
      pulled toward ``y_t - α log k_t`` by the Kalman gain. For typical
      finance AR(1) (``σ_stat ≤ 0.4``), this lives well within ``[-1, 1]``
      regardless of the chain's ``(ρ, σ_ε)`` — the data anchors it.
    * Setting ``log_z_half_width = 1.0`` puts the env's ``z_max``
      approximately at the level where the closed-form
      ``k'(z; β_center)`` reaches ``k_max``, so the NN training
      distribution has **near-zero structurally-clipped samples**.
      Wider z ranges put the NN's training into a regime where the
      optimal investment exceeds the action box, killing SHAC's
      gradient signal and biasing the surrogate.
    * For unusually wide priors (large ``σ_stat``), the chain *could*
      visit β values where the data's implied latent range exceeds
      ``±1``. In that case, raise ``log_z_half_width`` to ``1.2`` or
      ``1.5`` for safety. Track NN held-out MAE at the new bound to
      check the trade-off.

    Args:
        panel_k:          shape ``(n_firms, horizon)`` — observed k.
        k_margin:         multiplicative margin on observed k range.
                          Default 1.2.
        log_z_half_width: half-width of the log-z support around ``μ``.
                          Default 1.0 → ``z ∈ [exp(-1), exp(1)] ≈ [0.37, 2.72]``
                          at ``μ=0``. Mirrors 08a's default z box, which
                          achieves <2% held-out MAE of the k range.
        mu:               unconditional mean of log z. Default 0.

    Returns:
        ``{"k_min", "k_max", "z_min", "z_max"}`` as Python floats.
    """
    k = np.asarray(panel_k)
    if k.size == 0:
        raise ValueError("panel_k must be non-empty.")
    if not (k_margin > 1.0):
        raise ValueError(f"k_margin must be > 1; got {k_margin}.")
    if not (log_z_half_width > 0):
        raise ValueError(
            f"log_z_half_width must be > 0; got {log_z_half_width}."
        )

    k_min_obs = float(np.min(k))
    k_max_obs = float(np.max(k))
    if k_min_obs <= 0:
        raise ValueError(
            f"panel_k must be strictly positive; got min={k_min_obs}."
        )
    k_min = k_min_obs / float(k_margin)
    k_max = k_max_obs * float(k_margin)

    half = float(log_z_half_width)
    z_min = float(np.exp(float(mu) - half))
    z_max = float(np.exp(float(mu) + half))

    return {"k_min": k_min, "k_max": k_max, "z_min": z_min, "z_max": z_max}


def neural_true_beta(env: ParameterizedBasicInvestmentEnv,
                      *,
                      production_mean: float = 0.0,
                      production_std:  float = 0.05,
                      investment_mean: float = -0.10,
                      investment_std:  float = 0.05) -> dict[str, float]:
    """Convenience: env's nominal structural params packaged as a 7-tuple β dict.

    Pairs with ``make_neural_bayesian_spec`` for single-run NB09 recovery.
    The four residual parameters are exogenous to the env, so they are
    supplied here. Defaults match the Bayesian.md §2.3 ground-truth panel
    (no production misspec, mild measurement noise; 10% systematic
    investment under-shooting; both stds at 5%).
    """
    return {
        "alpha":           float(env.nominal_econ.production_elasticity),
        "rho":             float(env.nominal_shocks.rho),
        "sigma_epsilon":   float(env.nominal_shocks.sigma),
        "production_mean": float(production_mean),
        "production_std":  float(production_std),
        "investment_mean": float(investment_mean),
        "investment_std":  float(investment_std),
    }
