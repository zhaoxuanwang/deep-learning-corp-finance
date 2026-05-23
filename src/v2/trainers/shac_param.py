"""β-conditional SHAC trainer (DDPG-style variant) for the neural surrogate.

Parallel to `train_shac` (`src/v2/trainers/shac.py`). Differences:

  * Env is `ParameterizedBasicInvestmentEnv` (β-aware methods).
  * Networks are `ParameterizedPolicyNetwork` and
    `ParameterizedStateValueNetwork`.
  * Each trajectory uses one β draw held constant across all windows
    (β is a structural parameter, not time-varying).
  * Data contract is flat (s_endo, z) — treated as initial (k_0, z_0).
    The full z-trajectory is simulated internally per minibatch under the
    sampled β, via `env.exogenous_transition`.
  * Reward scaling: only `reward_scale_override` is honored. Auto-scaling
    via `env.reward_scale()` is not supported here because reward depends
    on β. Default scale: 1.0.

The existing `train_shac` and the production env stay untouched.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import tensorflow as tf

from src.v2.data.pipeline import build_iterator, validate_dataset_keys
from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
)
from src.v2.estimation.beta_sampler import BetaSampler
from src.v2.networks.policy import ParameterizedPolicyNetwork
from src.v2.networks.state_value import ParameterizedStateValueNetwork
from src.v2.trainers.config import SHACConfig
from src.v2.trainers.core import (
    StopTracker,
    append_history_row,
    build_target_policy,
    build_target_value,
    capture_checkpoint,
    polyak_update,
)
from src.v2.trainers.er_param import fit_normalizer_param_flat
from src.v2.utils.seeding import fold_in_seed, make_seed_int, seed_runtime


_DATASET_KEYS = ["s_endo", "z"]


def train_shac_param(
    env: ParameterizedBasicInvestmentEnv,
    policy: ParameterizedPolicyNetwork,
    value_net: ParameterizedStateValueNetwork,
    beta_sampler: BetaSampler,
    train_dataset: Mapping[str, tf.Tensor],
    val_dataset: Optional[Mapping[str, tf.Tensor]] = None,
    config: Optional[SHACConfig] = None,
    eval_callback: Optional[Any] = None,
    normalizer_beta_sampler: Optional[BetaSampler] = None,
) -> dict:
    """Train policy and value networks via β-conditional SHAC.

    Actor (h-step BPTT, current π + current V bootstrap):
        Unroll h steps with current policy through differentiable dynamics
        under the per-trajectory β. Bootstrap with current V_φ(s_h, β).
        loss = -mean[ Σ_τ γ^τ r(s_τ, π(s_τ, β), β) + γ^h V(s_h, β) ]

    Critic (1-step Bellman, target π̄ + target V̄):
        For each visited s_t in the window (under β):
            y(s, β) = r(s, π̄(s, β), β) + γ · V̄(f(s, π̄(s, β), z'), β)
        loss_critic = mean[ (V(s, β) - stop_gradient(y))² ]

    Args:
        env:           ParameterizedBasicInvestmentEnv instance.
        policy:        ParameterizedPolicyNetwork instance (actor π_θ).
        value_net:     ParameterizedStateValueNetwork instance (critic V_φ).
        beta_sampler:  BetaSampler that draws β ∼ prior per minibatch.
        train_dataset: Flat dict {s_endo, z} — each row is an initial
                       (k_0, z_0) for a rollout.
        val_dataset:   Optional validation dict (passed to `eval_callback`).
        config:        SHACConfig with hyperparameters. Only
                       `reward_scale_override` is honored for scaling;
                       `normalize_rewards` is ignored.
        eval_callback: Optional `(step, env, policy, value_net, val_dataset)
                       -> dict` for custom validation metrics.

    Returns:
        dict with keys: policy, value_net, target_policy, target_value_net,
        history, config, wall_time_sec, plus StopTracker output.
    """
    config = config or SHACConfig()
    env.validate_nn_training_support("train_shac_param")
    seed_runtime(
        config.master_seed, "train_shac_param",
        strict_reproducibility=config.strict_reproducibility,
    )
    gamma       = env.discount()
    horizon     = config.horizon
    window_len  = config.short_horizon

    if horizon % window_len != 0:
        raise ValueError(
            f"SHACConfig.horizon={horizon} must be divisible by "
            f"SHACConfig.short_horizon={window_len}."
        )
    n_windows = horizon // window_len

    validate_dataset_keys(
        train_dataset, _DATASET_KEYS, "train_shac_param", "train_dataset")

    if config.reward_scale_override is not None:
        reward_scale = float(config.reward_scale_override)
    else:
        reward_scale = 1.0   # no auto-scaling for β-conditional case
    reward_scale_tf = tf.constant(reward_scale, dtype=tf.float32)
    discount_tf     = tf.constant(gamma, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    actor_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)
    critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.critic_optimizer.learning_rate,
        clipnorm=config.critic_optimizer.clipnorm)

    # ------------------------------------------------------------------
    # Fit normalizer on joint (s, β) distribution.
    # If `normalizer_beta_sampler` is provided, use it for the fit (and the
    # target-net re-fit below) so curriculum stages keep a stable baseline.
    # ------------------------------------------------------------------
    norm_sampler = normalizer_beta_sampler if normalizer_beta_sampler is not None else beta_sampler
    fit_seed = fold_in_seed(
        config.master_seed, "train_shac_param", "normalizer_beta")
    fit_normalizer_param_flat(
        env, train_dataset, norm_sampler,
        policy, value_net, seed=fit_seed)

    # ------------------------------------------------------------------
    # Target networks (vanilla nets with augmented input_dim)
    # ------------------------------------------------------------------
    target_value_net = build_target_value(value_net)
    target_policy    = build_target_policy(policy)
    # Targets share the same normalizer fit
    fit_normalizer_param_flat(
        env, train_dataset, norm_sampler,
        target_policy, target_value_net, seed=fit_seed)

    # Precompute discount powers: γ^0, ..., γ^h
    discount_powers = tf.constant(
        [gamma ** t for t in range(window_len + 1)], dtype=tf.float32)

    exo_dim    = env.exo_dim()
    batch_sz   = config.batch_size

    # ------------------------------------------------------------------
    # Compiled steps
    # ------------------------------------------------------------------

    @tf.function
    def actor_step(k_endo, z_window, beta):
        """h-step actor rollout under per-trajectory β.

        Args:
            k_endo:   (B, endo_dim)  initial endogenous state for this window.
            z_window: (B, h+1, exo_dim) z trajectory slice for this window.
            beta:     (B, beta_dim) per-trajectory β (constant in window).
        """
        with tf.GradientTape() as tape:
            k_current = k_endo
            total_reward = tf.zeros(tf.shape(k_endo)[0])

            collected_states = []
            collected_z_next = []

            for tau in range(window_len):
                z_t = z_window[:, tau, :]
                s_t = env.merge_state(k_current, z_t)
                a_t = policy(s_t, beta, training=False)

                r_t = env.reward(s_t, a_t, beta)
                r_t = tf.reshape(r_t, [-1]) * reward_scale_tf

                collected_states.append(tf.stop_gradient(s_t))
                collected_z_next.append(z_window[:, tau + 1, :])
                total_reward = total_reward + discount_powers[tau] * r_t

                k_current = env.endogenous_transition(
                    k_current, a_t, z_t, beta)

            # Terminal state and bootstrap with CURRENT V
            z_end = z_window[:, window_len, :]
            s_end = env.merge_state(k_current, z_end)
            collected_states.append(tf.stop_gradient(s_end))

            v_bootstrap = tf.squeeze(
                value_net(s_end, beta, training=False))
            total_reward = total_reward + discount_powers[window_len] * v_bootstrap

            loss_actor = -tf.reduce_mean(total_reward)

        grads_actor = tape.gradient(loss_actor, policy.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(grads_actor, policy.trainable_variables))

        all_states = tf.stack(collected_states, axis=1)    # (B, h+1, sd)
        all_z_next = tf.stack(collected_z_next, axis=1)    # (B, h, exo_dim)
        return loss_actor, tf.stop_gradient(k_current), all_states, all_z_next

    @tf.function
    def critic_step(s_batch, z_next_batch, beta_batch):
        """1-step Bellman target with target π̄ + target V̄, under per-sample β."""
        # Target policy via single-arg call (vanilla net): x_in = [s, β].
        x_in = tf.concat([s_batch, beta_batch], axis=-1)
        a_target = target_policy(x_in, training=False)
        r_target = env.reward(s_batch, a_target, beta_batch)
        r_target = tf.reshape(r_target, [-1]) * reward_scale_tf

        s_endo, s_exo = env.split_state(s_batch)
        k_next_target = env.endogenous_transition(
            s_endo, a_target, s_exo, beta_batch)
        s_next_target = env.merge_state(k_next_target, z_next_batch)

        x_next_in = tf.concat([s_next_target, beta_batch], axis=-1)
        v_next_target = tf.squeeze(
            target_value_net(x_next_in, training=False))
        bellman_target = tf.stop_gradient(
            r_target + discount_tf * v_next_target)

        with tf.GradientTape() as tape_c:
            v_pred = tf.squeeze(
                value_net(s_batch, beta_batch, training=True), axis=-1)
            loss_critic = tf.reduce_mean((v_pred - bellman_target) ** 2)
        grads_c = tape_c.gradient(loss_critic, value_net.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(grads_c, value_net.trainable_variables))
        return loss_critic

    @tf.function
    def simulate_z_path(z0, eps_path, beta):
        """Simulate z_{0..T} under per-trajectory β via env.exogenous_transition.

        Args:
            z0:       (B, exo_dim) initial exogenous state.
            eps_path: (B, T, shock_dim) standard normal shocks.
            beta:     (B, beta_dim).

        Returns:
            z_path: (B, T+1, exo_dim).
        """
        T = eps_path.shape[1]
        z = z0
        out = [z0]
        for t in range(T):
            z = env.exogenous_transition(z, eps_path[:, t, :], beta)
            out.append(z)
        return tf.stack(out, axis=1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(
        {"s_endo": train_dataset["s_endo"], "z": train_dataset["z"]},
        batch_sz,
        shuffle_seed=make_seed_int(
            config.master_seed, "train_shac_param", "batch_shuffle"),
    )
    history: dict = {}
    stop_tracker = StopTracker(
        monitor=config.monitor, mode=config.mode,
        threshold=config.threshold,
        threshold_patience=config.threshold_patience,
        plateau_patience=config.plateau_patience,
        plateau_min_delta=config.plateau_min_delta,
        plateau_rel_delta=config.plateau_rel_delta,
        min_steps_before_stop=config.min_steps_before_stop,
    )

    train_start = time.perf_counter()
    step = 0
    last_step = -1
    minibatch_idx = 0
    loss_c = tf.constant(0.0)

    for batch in train_iter:
        if step >= config.n_steps:
            break
        minibatch_idx += 1

        # Per-minibatch: sample β and ε-path; simulate z trajectory under β.
        beta_seed = fold_in_seed(
            config.master_seed, "train_shac_param", "mb", minibatch_idx, "beta")
        eps_seed  = tf.constant(fold_in_seed(
            config.master_seed, "train_shac_param", "mb", minibatch_idx, "eps"),
            dtype=tf.int32)

        beta_traj = beta_sampler.sample(batch_sz, seed=beta_seed)   # (B, 5)
        eps_path  = tf.random.stateless_normal(
            (batch_sz, horizon, env.shock_dim()),
            seed=eps_seed, dtype=tf.float32)
        z0    = batch["z"]
        z_full = simulate_z_path(z0, eps_path, beta_traj)            # (B, T+1, 1)

        k_endo = tf.stop_gradient(batch["s_endo"])

        for win_idx in range(n_windows):
            if step >= config.n_steps:
                break
            last_step = step

            t0 = win_idx * window_len
            z_window = z_full[:, t0:t0 + window_len + 1, :]

            # (a) Actor
            loss_actor, k_endo, all_states, all_z_next = actor_step(
                k_endo, z_window, beta_traj)

            # (b) Critic on collected transitions, broadcast β across h.
            state_dim = all_states.shape[-1]
            beta_dim_ = beta_traj.shape[-1]
            critic_s = tf.reshape(
                all_states[:, :-1, :], [-1, state_dim])            # (B*h, sd)
            critic_z = tf.reshape(
                all_z_next, [-1, exo_dim])                          # (B*h, 1)
            critic_b = tf.reshape(
                tf.broadcast_to(
                    beta_traj[:, tf.newaxis, :],
                    [batch_sz, window_len, beta_dim_]),
                [-1, beta_dim_])                                    # (B*h, 5)

            for _ in range(config.n_critic):
                loss_c = critic_step(critic_s, critic_z, critic_b)

            # (c) Polyak
            polyak_update(value_net, target_value_net, tau=config.polyak_rate)
            polyak_update(policy,    target_policy,    tau=config.polyak_rate)

            # Logging
            elapsed_sec = time.perf_counter() - train_start
            cap_reached = (
                config.max_wall_time_sec is not None
                and elapsed_sec >= config.max_wall_time_sec
            )
            is_eval = (
                step % config.eval_interval == 0
                or step == config.n_steps - 1
                or cap_reached
            )
            if is_eval:
                base_scalars = {
                    "loss_actor":  float(loss_actor),
                    "loss_critic": float(loss_c),
                }
                eval_metrics: dict = {}
                if eval_callback is not None:
                    from src.v2.trainers.core import run_eval_callback
                    eval_metrics = run_eval_callback(
                        step, env, policy, value_net, val_dataset,
                        eval_callback=eval_callback)

                stop_on_threshold = stop_tracker.record_eval(
                    step, elapsed_sec, {**base_scalars, **eval_metrics})
                if not stop_on_threshold and cap_reached:
                    stop_tracker.finalize("max_wall_time", step, elapsed_sec)
                elif not stop_on_threshold and step == config.n_steps - 1:
                    stop_tracker.finalize("max_steps", step, elapsed_sec)

                append_history_row(
                    history, step, elapsed_sec,
                    base_scalars=base_scalars,
                    eval_metrics=eval_metrics)
                print(
                    f"SHAC-param step {step:5d} | "
                    f"loss_actor={float(loss_actor):.4f} | "
                    f"loss_critic={float(loss_c):.6f} "
                    f"| elapsed={elapsed_sec:.1f}s"
                )
                capture_checkpoint(
                    step, config, policy=policy, value_net=value_net)
                if stop_on_threshold or cap_reached:
                    step += 1
                    break
            step += 1

        if stop_tracker.stop_reason in {"converged", "plateau", "max_wall_time"}:
            break

    wall_time_sec = time.perf_counter() - train_start
    if stop_tracker.stop_reason is None and last_step >= 0:
        stop_tracker.finalize("max_steps", last_step, wall_time_sec)

    return {
        "policy":           policy,
        "value_net":        value_net,
        "target_policy":    target_policy,
        "target_value_net": target_value_net,
        "history":          history,
        "config":           config,
        "wall_time_sec":    wall_time_sec,
        **stop_tracker.result_dict(),
    }
