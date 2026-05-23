"""Bayesian spec factory for the **neural-surrogate** basic investment model.

Phase 2 sibling of :mod:`src.v2.estimation.bayesian_basic_investment`. Where
the Phase-1 module assumes the policy is the closed-form Strebulaev-Whited
(2012, §3.1) frictionless solution (capital enters as exogenous offset,
LGSSM exact), this module assumes the policy is a *pre-trained neural
surrogate* ``policy_nn(s, β; θ)`` and the candidate β is fed into the
likelihood through **two** observation equations (Bayesian.md §1.5):

* Eq 1 (production): ``y_t = α log k_t + x_t + η_t``,   η ~ N(0, σ_η²)
* Eq 2 (capital LoM): ``log k_{t+1} = log φ_NN(k_t, exp(x_t); β) + ξ^k_t``,
  ξ^k ~ N(0, σ_ξ²)

with latent state ``x_t = log z_t`` and AR(1) transition unchanged.

Eq 2 is **nonlinear in the latent state** through the NN, so a plain LGSSM
no longer applies. We use an Extended Kalman Filter: at each step, evaluate
the NN and its Jacobian d log φ_NN / d x at the *predicted* latent mean
``m_{t|t-1}`` (per chain × firm), then run the standard Kalman update with
a 2-D observation and 2×1 observation matrix ``H_t = [[1], [H_2(t)]]``.

For the closed-form ground truth, log k'(z; β) is exactly linear in log z
(Bayesian.md §2.2), so a perfectly-trained NN gives a constant H_2 ≈
ρ/(1-α) and EKF reduces to standard Kalman with zero linearization error.
This is the §3.4 integration test: if the EKF + NUTS recovers ground-truth
β through the NN, the same machinery is ready for the frictional / risky-debt
models where no closed-form policy exists.

Reproducibility (Bayesian.md §2.9): same master seed → bit-identical posterior
samples on CPU. All RNG paths derive from one master pair via ``fold_in_seed``.
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
    "alpha", "rho", "sigma_epsilon", "sigma_eta", "sigma_xi",
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
    policy_nn: ParameterizedPolicyNetwork,
    delta_rate: float,
    V0: float,
):
    """Return a ``log_likelihood_fn(beta, data)`` closure for the EKF.

    Closure captures the (frozen) NN, the env's depreciation rate, and the
    prior latent variance. The returned function is differentiable in
    ``beta`` end-to-end (forward-mode JVP for H_2 composes cleanly with the
    outer reverse-mode NUTS tape).

    Shape contract:

      * ``beta``: dict of tensors of shape ``[B...]`` (chain batch).
      * ``data["y"]``, ``["log_k"]``, ``["log_k_next"]``: ``[N, T, 1]``.

    Returns scalar ``[B...]`` log-likelihood (sum over firms × t).
    """

    delta_const = tf.constant(float(delta_rate), tf.float32)
    V0_const    = tf.constant(float(V0),         tf.float32)
    log_2pi     = tf.constant(float(np.log(2.0 * np.pi)), tf.float32)

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
        alpha     = tf.cast(beta["alpha"],          tf.float32)   # [B...]
        rho       = tf.cast(beta["rho"],            tf.float32)
        sigma_eps = tf.cast(beta["sigma_epsilon"],  tf.float32)
        sigma_eta = tf.cast(beta["sigma_eta"],      tf.float32)
        sigma_xi  = tf.cast(beta["sigma_xi"],       tf.float32)

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

        rho_e     = rho[..., tf.newaxis]               # [B..., 1]
        alpha_e   = alpha[..., tf.newaxis]
        sig_eps_2 = (sigma_eps ** 2)[..., tf.newaxis]
        sig_eta_2 = (sigma_eta ** 2)[..., tf.newaxis]
        sig_xi_2  = (sigma_xi  ** 2)[..., tf.newaxis]

        log_lik_per_firm = tf.zeros(BN_shape, dtype=tf.float32)

        for t_idx in range(T_static):
            k_t          = k[:, t_idx]            # [N]
            y_t          = y[:, t_idx]            # [N]
            log_k_t      = log_k[:, t_idx]        # [N]
            log_k_next_t = log_k_next[:, t_idx]   # [N]

            # 1. Predict latent state (AR(1)).
            m_pred = rho_e * m_prev                              # [B..., N]
            V_pred = (rho_e ** 2) * V_prev + sig_eps_2           # [B..., N]

            # 2. NN forward + reverse-mode gradient for H_2 = d g / d x at
            #    x = m_pred. At 1-D scalar latent state per (chain, firm),
            #    reverse-mode is computationally equivalent to forward-mode
            #    but XLA-compatible.
            k_t_b = tf.broadcast_to(k_t, tf.shape(m_pred))       # [B..., N]
            beta_5d_b = tf.broadcast_to(
                beta_5d[..., tf.newaxis, :],
                tf.concat([tf.shape(beta_5d)[:-1], [N_static, 5]], axis=0),
            )                                                     # [B..., N, 5]

            with tf.GradientTape() as tape:
                tape.watch(m_pred)
                z_pred = tf.exp(m_pred)                            # [B..., N]
                state = tf.stack([k_t_b, z_pred], axis=-1)         # [B..., N, 2]
                state_flat   = tf.reshape(state,     [-1, 2])
                beta_5d_flat = tf.reshape(beta_5d_b, [-1, 5])
                action_flat  = policy_nn(state_flat, beta_5d_flat,
                                          training=False)          # [..., 1]
                action = tf.reshape(action_flat[..., 0], tf.shape(m_pred))
                k_next = (1.0 - delta_const) * k_t_b + action
                g_pred = tf.math.log(tf.maximum(k_next, 1e-12))    # [B..., N]
            H_2 = tape.gradient(g_pred, m_pred)                    # [B..., N]

            # 3. Innovation.
            nu_1 = y_t - alpha_e * log_k_t - m_pred                # [B..., N]
            nu_2 = log_k_next_t - g_pred                            # [B..., N]

            # 4. Observation covariance S = H V_pred Hᵀ + R   (closed form, 2×2).
            #    H = [[1], [H_2]],  R = diag(σ_η², σ_ξ²)
            s11 = V_pred + sig_eta_2                                # [B..., N]
            s12 = H_2 * V_pred                                       # [B..., N]
            s22 = (H_2 ** 2) * V_pred + sig_xi_2                    # [B..., N]
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

def make_neural_bayesian_spec(
    env: ParameterizedBasicInvestmentEnv,
    policy_nn: ParameterizedPolicyNetwork,
    *,
    sigma_eta_prior_scale: float = 0.1,
    sigma_xi_prior_scale:  float = 0.1,
    initial_state_variance: float = 10.0,
    uniform_bounds: Optional[Mapping[str, tuple[float, float]]] = None,
) -> BayesianSpec:
    """Build the §3.4 neural-surrogate ``BayesianSpec``.

    Priors follow Bayesian.md §3.3 — the surrogate is only reliable on its
    training box, so α, ρ, σ_ε get Uniform priors aligned with that box;
    σ_η, σ_ξ get HalfNormal priors. Filter / sampler: EKF (labeled as
    KALMAN in the spec, since the generic dispatch doesn't distinguish) + NUTS.

    **State-space alignment contract.** Both the NN training data and the
    synthetic panel come from this same ``ParameterizedBasicInvestmentEnv``:
    training data via ``sample_initial_*`` + ER/SHAC's internal rollout,
    synthetic panel via ``env.simulate_panel``. Same bounds, same initial
    distribution, same transitions. This eliminates the divergence that
    used to exist when the panel was synthesized through
    ``BasicInvestmentEnv.simulate_smm_panel_data`` (different bound formula,
    bounds recomputed per candidate β).

    Args:
        env:                    ParameterizedBasicInvestmentEnv in the
                                frictionless case (nominal_econ.cost_convex
                                == 0 and cost_fixed == 0; asserted).
                                Provides ``r``, ``δ``, the panel simulator,
                                and the analytical policy.
        policy_nn:              Pre-trained ParameterizedPolicyNetwork, with
                                frictionless dims frozen at zero by training
                                under ``BetaSampler(freeze_dims=(3, 4))``.
                                Set ``policy_nn.trainable = False`` before
                                building the spec.
        sigma_eta_prior_scale:  HalfNormal scale for σ_η.
        sigma_xi_prior_scale:   HalfNormal scale for σ_ξ.
        initial_state_variance: V_0 (Bayesian.md §2.3) — diffuse prior on x_1.
        uniform_bounds:         Override per-coordinate (low, high) box for
                                α, ρ, σ_ε. ``None`` → ``DEFAULT_UNIFORM_BOUNDS``
                                (matches the surrogate's training box).

    Returns:
        BayesianSpec with filter_kind=KALMAN, sampler_kind=NUTS, and a
        synthesize_panel_fn closing over ``env`` (closed-form analytical
        policy ⇒ ground-truth β can be recovered).
    """
    # Note: ``ParameterizedBasicInvestmentEnv`` structurally enforces
    # ``cost_convex == cost_fixed == 0`` on its nominal — frictions enter
    # only via per-sample β (``φ_quad``, ``φ_prop``). For this factory we
    # additionally require the policy_nn to have been trained with
    # ``freeze_dims=(3, 4)`` so it always evaluates on the frictionless
    # slice; we cannot statically detect that, but the diagnostic in
    # NB09 Sec 3 will catch a mismatched NN via the slope test.

    V0 = float(initial_state_variance)
    if V0 <= 0:
        raise ValueError(f"initial_state_variance must be > 0. Got {V0}.")

    bounds = dict(DEFAULT_UNIFORM_BOUNDS)
    if uniform_bounds is not None:
        bounds.update({k: tuple(v) for k, v in uniform_bounds.items()})

    a_lo, a_hi = bounds["alpha"]
    r_lo, r_hi = bounds["rho"]
    s_lo, s_hi = bounds["sigma_epsilon"]

    # ---- Priors (constrained scale) ----------------------------------------
    prior = tfd.JointDistributionNamed({
        "alpha":         tfd.Uniform(low=tf.constant(a_lo, tf.float32),
                                     high=tf.constant(a_hi, tf.float32)),
        "rho":           tfd.Uniform(low=tf.constant(r_lo, tf.float32),
                                     high=tf.constant(r_hi, tf.float32)),
        "sigma_epsilon": tfd.Uniform(low=tf.constant(s_lo, tf.float32),
                                     high=tf.constant(s_hi, tf.float32)),
        "sigma_eta":     tfd.HalfNormal(scale=tf.constant(sigma_eta_prior_scale, tf.float32)),
        "sigma_xi":      tfd.HalfNormal(scale=tf.constant(sigma_xi_prior_scale,  tf.float32)),
    })

    # ---- Bijectors: unconstrained ℝ⁵ → constrained support -----------------
    bijector = tfb.JointMap({
        "alpha":         _truncated_sigmoid_bijector(a_lo, a_hi),
        "rho":           _truncated_sigmoid_bijector(r_lo, r_hi),
        "sigma_epsilon": _truncated_sigmoid_bijector(s_lo, s_hi),
        "sigma_eta":     tfb.Exp(),
        "sigma_xi":      tfb.Exp(),
    })

    # ---- Likelihood (EKF) --------------------------------------------------
    log_likelihood_fn = _build_ekf_log_likelihood(
        policy_nn=policy_nn,
        delta_rate=float(env.delta_rate),
        V0=V0,
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
        sigma_eta = float(beta["sigma_eta"])

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

        log_k_raw = np.log(np.maximum(k, 1e-12))
        log_z_raw = np.log(np.maximum(z, 1e-12))

        # y_t = log z_t + α log k_t + η_t  (raw periods).
        noise_seed = fold_in_seed(seed, "obs_noise_y")
        eta = tf.random.stateless_normal(
            shape=[int(n_firms), raw_horizon],
            seed=tf.constant(noise_seed, tf.int32),
            stddev=tf.constant(sigma_eta, tf.float32),
        ).numpy().astype(np.float64)
        y_raw = log_z_raw + float(beta["alpha"]) * log_k_raw + eta

        # Slice to T = horizon model-time observations:
        #   y_t, log_k_t at original t = 0 .. T-1
        #   log_k_{t+1} at original t = 1 .. T
        y_out          = y_raw[:, :int(horizon)]
        log_k_out      = log_k_raw[:, :int(horizon)]
        log_k_next_out = log_k_raw[:, 1:int(horizon) + 1]

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
    """Closed-form frictionless rollout — no env required.

    Generates a (k, z) panel under the Strebulaev-Whited (2012, §3.1)
    analytical policy: ``k_{t+1} = ((α · E[z'|z]) / (r+δ))^(1/(1-α))``,
    with ``z_{t+1} = exp((1-ρ)μ + ρ log z + σ_ε ε_t)``,
    ``ε_t ~ N(0, 1)`` i.i.d. across firms and time.

    Args:
        beta:               dict with keys (alpha, rho, sigma_epsilon).
                            Other keys (sigma_eta, sigma_xi, ...) are ignored.
        interest_rate:      r in the analytical formula.
        depreciation_rate:  δ in the analytical formula.
        mu:                 unconditional mean of log z; default 0.
        n_firms:            N — number of firms.
        horizon:            T — recorded periods (after burn-in).
        burn_in:            periods discarded so the panel reaches its
                            stationary distribution. Recommend ≥ 30 for
                            ρ ≤ 0.9.
        seed:               (m0, m1) master seed for all randomness.

    Returns:
        ``{"k": np.ndarray (N, T), "z": np.ndarray (N, T)}``.

    Reproducibility: same ``seed`` → bit-identical (k, z) arrays on
    the same hardware (CPU). Sub-seeds for initial state and AR(1)
    shocks are derived via ``fold_in_seed(seed, ...)`` so different
    notebooks / stages can re-derive independent draws by varying
    only the token chain.
    """
    alpha = float(beta["alpha"])
    rho   = float(beta["rho"])
    sigma = float(beta["sigma_epsilon"])
    r     = float(interest_rate)
    delta = float(depreciation_rate)
    mu_f  = float(mu)
    N     = int(n_firms)
    Tp    = int(horizon)
    Tb    = int(burn_in)
    Ttot  = Tb + Tp

    # κ(β) for the closed-form k'(z; β) = exp(ρ/(1-α) · log z + κ)
    kappa = (np.log(alpha) + 0.5 * sigma ** 2 - np.log(r + delta)) / (1.0 - alpha)
    slope = rho / (1.0 - alpha)

    # Initial conditions: draw log z_0 from the stationary distribution
    # N(μ, σ²/(1-ρ²)). Then k_0 = closed-form k'(z_0; β) so the trajectory
    # starts on the policy surface; burn-in further damps any residual
    # transient.
    sigma_stat = sigma / np.sqrt(max(1.0 - rho ** 2, 1e-8))

    seed_z0  = tf.constant(fold_in_seed(seed, "panel", "z0"),  tf.int32)
    seed_eps = tf.constant(fold_in_seed(seed, "panel", "eps"), tf.int32)

    log_z0 = (mu_f + sigma_stat * tf.random.stateless_normal(
        [N], seed=seed_z0, dtype=tf.float32)).numpy()
    eps = tf.random.stateless_normal(
        [N, Ttot], seed=seed_eps, dtype=tf.float32).numpy()

    log_z = np.empty((N, Ttot), dtype=np.float64)
    log_k = np.empty((N, Ttot), dtype=np.float64)

    log_z_prev = log_z0.astype(np.float64)
    for t in range(Ttot):
        # AR(1) on log z
        log_z_t = (1.0 - rho) * mu_f + rho * log_z_prev + sigma * eps[:, t]
        log_z[:, t] = log_z_t
        # Analytical k_{t+1} given z_t: linear in log z
        log_k[:, t] = slope * log_z_t + kappa
        log_z_prev = log_z_t

    # Slice off burn-in
    log_k = log_k[:, Tb:]
    log_z = log_z[:, Tb:]
    return {
        "k": np.exp(log_k).astype(np.float64),
        "z": np.exp(log_z).astype(np.float64),
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
                          at ``μ=0``. Mirrors NB08's default z box, which
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
                      sigma_eta: float = 0.05,
                      sigma_xi:  float = 0.01) -> dict[str, float]:
    """Convenience: env's nominal structural params packaged as a 5-tuple β dict.

    Pairs with ``make_neural_bayesian_spec`` for single-run NB09 recovery.
    ``σ_η`` and ``σ_ξ`` are exogenous to the env, so they are supplied here.
    Default ``σ_ξ`` is small (0.01) — under synthetic closed-form data, the
    NN approximation residual should be tiny, so the *posterior* of ``σ_ξ``
    is the diagnostic we care about, not this initial seed.
    """
    return {
        "alpha":         float(env.nominal_econ.production_elasticity),
        "rho":           float(env.nominal_shocks.rho),
        "sigma_epsilon": float(env.nominal_shocks.sigma),
        "sigma_eta":     float(sigma_eta),
        "sigma_xi":      float(sigma_xi),
    }
