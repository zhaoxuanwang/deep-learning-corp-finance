"""Bayesian spec factory for the frictionless-analytical basic investment model.

Builds a ``BayesianSpec`` (see ``src/v2/estimation/bayesian.py``) for Phase 1
of the project's Bayesian validation pipeline.  Spec contents follow
docs/paper/Bayesian.md Sections 2.1–2.5:

  - Parameters β = (α, ρ, σ_ε, σ_η)
  - Priors: α, ρ ~ Beta(2, 2);  σ_ε, σ_η ~ HalfNormal
  - Bijectors: Sigmoid for (0, 1), Exp for positive scales
  - Filter: Kalman (LGSSM via tfp.distributions.LinearGaussianStateSpaceModel)
  - Sampler: NUTS (the LGSSM likelihood is exact and differentiable in β)
  - Synthetic panels: reuse BasicInvestmentEnv.simulate_smm_panel_data
    (mode="frictionless_analytical", n_panels=1) and overlay η ~ N(0, σ_η²).

Phase 2 will add a sibling factory for the frictional / NN-solver case;
Phase 3 a separate file for risky-debt + particle filter.  The generic
``run_mcmc`` / ``run_coverage_check`` runners do not need to change.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.estimation.bayesian import (
    BayesianSpec,
    FilterKind,
    SamplerKind,
)
from src.v2.estimation.smm import SMMRunConfig
from src.v2.utils.seeding import fold_in_seed

tfd = tfp.distributions
tfb = tfp.bijectors


PARAMETER_NAMES: tuple[str, ...] = ("alpha", "rho", "sigma_epsilon", "sigma_eta")
OBSERVATION_KEYS: tuple[str, ...] = ("y", "log_k")


def make_bayesian_spec(
    env: BasicInvestmentEnv,
    *,
    initial_state_variance: float = 10.0,
    sigma_epsilon_prior_scale: float = 0.3,
    sigma_eta_prior_scale: float = 0.1,
    alpha_prior_concentrations: tuple[float, float] = (2.0, 2.0),
    rho_prior_concentrations: tuple[float, float] = (2.0, 2.0),
) -> BayesianSpec:
    """Build the Phase 1 ``BayesianSpec`` for ``env``.

    Args:
        env:                       BasicInvestmentEnv in the frictionless
                                   case (cost_convex == cost_fixed == 0;
                                   asserted before returning).  Only
                                   `interest_rate` and `depreciation_rate`
                                   are read off; structural β is freshly
                                   sampled by the prior.
        initial_state_variance:    V_0 in Bayesian.md §2.3.  Diffuse prior
                                   on x_{i,1} = log z_{i,1}.
        sigma_epsilon_prior_scale: HalfNormal scale for σ_ε.
        sigma_eta_prior_scale:     HalfNormal scale for σ_η.
        alpha_prior_concentrations:(c1, c0) of Beta(c1, c0) for α.
        rho_prior_concentrations:  (c1, c0) of Beta(c1, c0) for ρ.

    Returns:
        BayesianSpec with filter_kind=KALMAN, sampler_kind=NUTS, and
        synthesize_panel_fn closing over ``env`` (so the simulator follows
        the analytical policy at any β passed in).
    """

    if abs(float(env.econ.cost_convex)) > 1e-12 or abs(float(env.econ.cost_fixed)) > 1e-12:
        raise ValueError(
            "make_bayesian_spec requires the frictionless case "
            "(cost_convex == 0 and cost_fixed == 0). "
            f"Got cost_convex={env.econ.cost_convex}, "
            f"cost_fixed={env.econ.cost_fixed}."
        )

    V0 = float(initial_state_variance)
    if V0 <= 0:
        raise ValueError(f"initial_state_variance must be > 0. Got {V0}.")

    # ---- Prior on the constrained scale --------------------------------
    a1, a0 = alpha_prior_concentrations
    r1, r0 = rho_prior_concentrations
    prior = tfd.JointDistributionNamed({
        "alpha":          tfd.Beta(tf.constant(a1, tf.float32),
                                   tf.constant(a0, tf.float32)),
        "rho":            tfd.Beta(tf.constant(r1, tf.float32),
                                   tf.constant(r0, tf.float32)),
        "sigma_epsilon":  tfd.HalfNormal(scale=tf.constant(sigma_epsilon_prior_scale,
                                                           tf.float32)),
        "sigma_eta":      tfd.HalfNormal(scale=tf.constant(sigma_eta_prior_scale,
                                                           tf.float32)),
    })

    # ---- Bijector: unconstrained ℝ⁴ → constrained support --------------
    bijector = tfb.JointMap({
        "alpha":         tfb.Sigmoid(),
        "rho":           tfb.Sigmoid(),
        "sigma_epsilon": tfb.Exp(),
        "sigma_eta":     tfb.Exp(),
    })

    # ---- Likelihood: Kalman via TFP LGSSM ------------------------------
    # Per Bayesian.md §2.4: subtract the time-varying observation offset
    # (alpha * log_k) from y so the LGSSM sees a zero-offset model, then
    # take the per-firm log_prob and sum.  Avoids the need for TFP 0.24's
    # absent `observation_offset` kwarg.
    #
    # Shape contract:
    #   y, log_k:    [N, T, 1] (data, fixed across chains)
    #   alpha, rho, sigma_eps, sigma_eta: arbitrary leading-batch shape ``B``
    #     (scalar [] in single-eval, [n_chains] inside NUTS).
    # The LGSSM is given matrices of batch_shape [B..., N] by tiling β across
    # the firm dim.  TFP's LGSSM treats the trailing latent/obs dims as event;
    # ``.log_prob`` on y_centered of shape [B..., N, T, 1] then returns
    # [B..., N] which we sum over firms to a per-chain scalar [B...].
    def log_likelihood_fn(beta: Mapping[str, tf.Tensor],
                          data: Mapping[str, tf.Tensor]) -> tf.Tensor:
        alpha = beta["alpha"]
        rho = beta["rho"]
        sigma_eps = beta["sigma_epsilon"]
        sigma_eta = beta["sigma_eta"]

        y = data["y"]            # [N, T, 1], float32
        log_k = data["log_k"]    # [N, T, 1], float32
        T_steps = tf.shape(y)[-2]
        N_firms = tf.shape(y)[-3]

        # Add three trailing dims so alpha broadcasts against log_k [N, T, 1].
        alpha_b = alpha[..., tf.newaxis, tf.newaxis, tf.newaxis]
        y_centered = y - alpha_b * log_k  # → [B..., N, T, 1]

        # Tile each scalar (per chain) β-component across the N firm dim.
        # Result shape: [B..., N].  Then add latent/obs dims as needed.
        def _tile_to_firms(t: tf.Tensor) -> tf.Tensor:
            target = tf.concat([tf.shape(t), [N_firms]], axis=0)
            return tf.broadcast_to(t[..., tf.newaxis], target)

        rho_b       = _tile_to_firms(rho)         # [B..., N]
        sigma_eps_b = _tile_to_firms(sigma_eps)   # [B..., N]
        sigma_eta_b = _tile_to_firms(sigma_eta)   # [B..., N]

        rho_mat     = rho_b[..., tf.newaxis, tf.newaxis]   # [B..., N, 1, 1]
        sigma_eps_d = sigma_eps_b[..., tf.newaxis]         # [B..., N, 1]
        sigma_eta_d = sigma_eta_b[..., tf.newaxis]         # [B..., N, 1]
        init_loc    = tf.zeros_like(sigma_eps_d)           # [B..., N, 1]
        init_scale  = tf.fill(tf.shape(sigma_eps_d),
                              tf.cast(tf.sqrt(tf.constant(V0, tf.float32)),
                                      sigma_eps_d.dtype))  # [B..., N, 1]

        lgssm = tfd.LinearGaussianStateSpaceModel(
            num_timesteps=T_steps,
            transition_matrix=rho_mat,
            transition_noise=tfd.MultivariateNormalDiag(scale_diag=sigma_eps_d),
            observation_matrix=tf.constant([[1.0]], tf.float32),
            observation_noise=tfd.MultivariateNormalDiag(scale_diag=sigma_eta_d),
            initial_state_prior=tfd.MultivariateNormalDiag(
                loc=init_loc, scale_diag=init_scale),
        )
        # log_prob on [B..., N, T, 1] returns shape [B..., N]; sum firms → [B...].
        return tf.reduce_sum(lgssm.log_prob(y_centered), axis=-1)

    # ---- Synthetic panel: SMM Stage-A + measurement-noise overlay -------
    # Built as a closure over `env` so the analytical policy stays in
    # lockstep with the candidate β handed to the simulator.
    def synthesize_panel_fn(beta: Mapping[str, float],
                            n_firms: int,
                            horizon: int,
                            burn_in: int,
                            seed: tuple[int, int]) -> dict[str, np.ndarray]:
        beta_smm = np.array([
            float(beta["alpha"]),
            float(beta["rho"]),
            float(beta["sigma_epsilon"]),
        ], dtype=np.float64)
        sigma_eta = float(beta["sigma_eta"])

        run_config = SMMRunConfig(
            n_firms=int(n_firms),
            horizon=int(horizon),
            burn_in=int(burn_in),
            n_sim_panels=1,
            master_seed=tuple(int(x) for x in seed),
        )
        panel = env.simulate_smm_panel_data(
            beta_smm,
            run_config,
            tuple(int(x) for x in seed),
            mode="frictionless_analytical",
        )
        # Panel arrays have shape (n_panels=1, n_firms, horizon); collapse.
        k = panel.k[0]   # (N, T)
        z = panel.z[0]   # (N, T)

        # Log-revenue with measurement noise:  y = log z + α log k + η.
        log_k = np.log(np.maximum(k, 1e-12))
        log_z = np.log(np.maximum(z, 1e-12))

        noise_seed = fold_in_seed(seed, "obs_noise")
        eta = tf.random.stateless_normal(
            shape=[int(n_firms), int(horizon)],
            seed=tf.constant(noise_seed, tf.int32),
            stddev=tf.constant(sigma_eta, tf.float32),
        ).numpy().astype(np.float64)

        y = log_z + float(beta["alpha"]) * log_k + eta

        return {
            "y":     y[..., np.newaxis].astype(np.float32),       # [N, T, 1]
            "log_k": log_k[..., np.newaxis].astype(np.float32),    # [N, T, 1]
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


def bayesian_true_beta(env: BasicInvestmentEnv,
                        *,
                        sigma_eta: float = 0.05) -> dict[str, float]:
    """Convenience: env's structural params packaged as a β dict.

    Pairs cleanly with ``make_bayesian_spec`` for single-run sanity checks
    in the notebook (``synthesize_panel_fn(bayesian_true_beta(env), ...)``).
    σ_η is exogenous to the env, so it must be supplied here.
    """
    return {
        "alpha":         float(env.econ.production_elasticity),
        "rho":           float(env.shocks.rho),
        "sigma_epsilon": float(env.shocks.sigma),
        "sigma_eta":     float(sigma_eta),
    }
