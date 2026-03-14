"""Vanilla SHAC trainer — faithful to Xu et al. (2022).

Archived reference implementation.  This trainer uses TD-λ critic targets
with on-policy actor rollout data, which causes a positive feedback loop
between actor and critic leading to divergence in economic environments.

The production SHAC trainer (src/v2/trainers/shac.py) replaces the TD-λ
critic with a 1-step Bellman target using target π̄ + target V̄, which
breaks the feedback loop and converges reliably.

See src/v2/experimental/shac_rollout_ablation.py for ablation results
demonstrating the divergence.

Reference
---------
Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A.,
& Macklin, M. (2022). Accelerated Policy Learning with Parallel
Differentiable Simulation. ICLR 2022.
"""

import tensorflow as tf
import numpy as np
from src.v2.trainers.config import SHACVanillaConfig
from src.v2.trainers.core import (
    polyak_update, build_target_value, warm_start_value_net,
    evaluate_euler_residual, evaluate_bellman_residual_v,
)
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys, fit_normalizer_traj,
)


_TRAIN_KEYS = ["s_endo_0", "z_path"]
_VAL_KEYS   = ["s_endo", "z", "z_next_main", "z_next_fork"]


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_shac_vanilla(env, policy, value_net, train_dataset: dict,
                       val_dataset: dict = None,
                       config: SHACVanillaConfig = None):
    """Train policy and value networks using vanilla SHAC (Xu et al. 2022).

    WARNING: This implementation diverges in economic environments due to
    the on-policy critic feedback loop.  Use train_shac() from
    src/v2/trainers/shac.py instead.

    Args:
        env:           MDPEnvironment instance.
        policy:        PolicyNetwork instance (actor π_θ).
        value_net:     StateValueNetwork instance (critic V_φ).
        train_dataset: Trajectory-format dataset dict.
        val_dataset:   Flattened-format dataset for evaluation (optional).
        config:        SHACVanillaConfig with hyperparameters.

    Returns:
        dict with keys: policy, value_net, history, config.
    """
    config      = config or SHACVanillaConfig()
    gamma       = env.discount()
    temperature = config.temperature
    T           = config.horizon
    h           = config.short_horizon
    td_lambda   = config.td_lambda

    # ------------------------------------------------------------------
    # Reward normalization
    # ------------------------------------------------------------------
    if config.reward_scale is None:
        reward_scale = float(env.compute_reward_scale(
            seed=tf.constant(list(config.master_seed), dtype=tf.int32)))
        print(f"SHAC-vanilla: auto reward_scale = {reward_scale:.6f} "
              f"(1/|V*| ≈ {1.0/reward_scale:.1f})")
    else:
        reward_scale = float(config.reward_scale)
    rs = tf.constant(reward_scale, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    validate_dataset_keys(train_dataset, _TRAIN_KEYS,
                          "train_shac_vanilla", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _VAL_KEYS,
                              "train_shac_vanilla", "val_dataset")

    T_data = train_dataset["z_path"].shape[1] - 1
    if T > T_data:
        raise ValueError(
            f"SHACVanillaConfig.horizon={T} exceeds dataset horizon={T_data}. "
            f"Re-generate data with DataGeneratorConfig.horizon >= {T}."
        )
    if T % h != 0:
        raise ValueError(
            f"SHACVanillaConfig.horizon={T} must be divisible by "
            f"SHACVanillaConfig.short_horizon={h}. "
            f"Got {T} % {h} = {T % h}."
        )

    n_windows = T // h  # windows per mini-batch

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
    # Fit normalizer from full dataset (once, before gradient steps)
    # ------------------------------------------------------------------
    fit_normalizer_traj(env, train_dataset, policy, value_net)

    # ------------------------------------------------------------------
    # Warm-start critic (before building target network)
    # ------------------------------------------------------------------
    if config.warm_start_steps > 0:
        warm_start_value_net(env, value_net, train_dataset,
                             n_steps=config.warm_start_steps,
                             reward_scale=reward_scale)

    # ------------------------------------------------------------------
    # Target value network V̄_φ (copies warm-started weights)
    # ------------------------------------------------------------------
    target_value_net = build_target_value(value_net)

    # ------------------------------------------------------------------
    # Precompute discount powers: γ^0, γ^1, ..., γ^{h-1}, γ^h
    # ------------------------------------------------------------------
    discount_powers = tf.constant(
        [gamma ** t for t in range(h + 1)], dtype=tf.float32)  # (h+1,)

    # ------------------------------------------------------------------
    # Compiled functions (closures capture env, networks, optimizers)
    # ------------------------------------------------------------------

    @tf.function
    def actor_step(k, z_window):
        """Actor rollout + gradient + optimizer step."""
        with tf.GradientTape() as tape:
            k_current = k
            total_reward = tf.zeros(tf.shape(k)[0])

            collected_states = []
            collected_rewards = []

            for tau in range(h):
                z_t = z_window[:, tau, :]
                s_t = env.merge_state(k_current, z_t)
                a_t = policy(s_t, training=False)

                r_t = env.reward(s_t, a_t, temperature=temperature)
                r_t = tf.reshape(r_t, [-1]) * rs

                collected_states.append(tf.stop_gradient(s_t))
                collected_rewards.append(tf.stop_gradient(r_t))

                total_reward = total_reward + discount_powers[tau] * r_t

                k_current = env.endogenous_transition(k_current, a_t, z_t)

            # Terminal state and bootstrap
            z_h = z_window[:, h, :]
            s_h = env.merge_state(k_current, z_h)
            collected_states.append(tf.stop_gradient(s_h))

            v_bootstrap = tf.squeeze(
                target_value_net(s_h, training=False))
            total_reward = total_reward + discount_powers[h] * v_bootstrap

            loss_actor = -tf.reduce_mean(total_reward)

        grads_actor = tape.gradient(loss_actor, policy.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(grads_actor, policy.trainable_variables))

        all_states = tf.stack(collected_states, axis=1)
        all_rewards = tf.stack(collected_rewards, axis=1)

        return loss_actor, tf.stop_gradient(k_current), all_states, all_rewards

    @tf.function
    def compute_critic_labels(all_states, all_rewards):
        """Compute TD-λ regression labels from detached states and rewards."""
        B = tf.shape(all_states)[0]
        state_dim = all_states.shape[-1]

        flat = tf.reshape(all_states, [-1, state_dim])
        v_flat = tf.squeeze(
            target_value_net(flat, training=False), axis=-1)
        v_all = tf.reshape(v_flat, [B, h + 1])

        deltas = all_rewards + gamma * v_all[:, 1:] - v_all[:, :-1]

        adv_list = [None] * h
        a_next = tf.zeros(tf.shape(all_rewards)[0])
        for tau in range(h - 1, -1, -1):
            a_next = deltas[:, tau] + gamma * td_lambda * a_next
            adv_list[tau] = a_next

        advantages = tf.stack(adv_list, axis=1)
        return advantages + v_all[:, :-1]

    @tf.function
    def critic_step(mb_states, mb_labels):
        """Single critic mini-batch MSE update."""
        with tf.GradientTape() as tape_c:
            v_pred = tf.squeeze(
                value_net(mb_states, training=True), axis=-1)
            loss_critic = tf.reduce_mean(
                (v_pred - mb_labels) ** 2)

        grads_c = tape_c.gradient(
            loss_critic, value_net.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(grads_c, value_net.trainable_variables))
        return loss_critic

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(train_dataset, config.batch_size)
    history = {
        "step": [], "loss_actor": [], "loss_critic": [],
        "euler_residual": [], "bellman_residual": [],
    }

    step = 0

    for batch in train_iter:
        if step >= config.n_steps:
            break

        s_endo_0 = batch["s_endo_0"]
        z_path   = batch["z_path"]

        k = tf.stop_gradient(s_endo_0)

        for w in range(n_windows):
            if step >= config.n_steps:
                break

            t0 = w * h
            z_window = z_path[:, t0:t0 + h + 1, :]

            loss_actor, k, all_states, all_rewards = actor_step(k, z_window)

            td_labels = compute_critic_labels(all_states, all_rewards)

            state_dim = all_states.shape[-1]
            critic_states = tf.reshape(
                all_states[:, :-1, :], [-1, state_dim])
            critic_labels = tf.reshape(td_labels, [-1])

            n_pairs = tf.shape(critic_states)[0]
            mb_size = n_pairs // config.n_mb

            total_critic_loss = 0.0
            n_critic_steps = 0

            for epoch in range(config.n_critic):
                perm = tf.random.shuffle(tf.range(n_pairs))
                critic_states_shuffled = tf.gather(critic_states, perm)
                critic_labels_shuffled = tf.gather(critic_labels, perm)

                for mb in range(config.n_mb):
                    start = mb * mb_size
                    end = start + mb_size
                    mb_states = critic_states_shuffled[start:end]
                    mb_labels = critic_labels_shuffled[start:end]

                    loss_c = critic_step(mb_states, mb_labels)

                    total_critic_loss += float(loss_c)
                    n_critic_steps += 1

            avg_critic_loss = total_critic_loss / max(n_critic_steps, 1)

            polyak_update(value_net, target_value_net, tau=config.polyak_rate)

            if step % config.eval_interval == 0 or step == config.n_steps - 1:
                er = br = float("nan")
                if val_dataset is not None:
                    er = evaluate_euler_residual(
                        env, policy, val_dataset, temperature=temperature)
                    br = evaluate_bellman_residual_v(
                        env, policy, value_net, val_dataset,
                        temperature=temperature)
                history["step"].append(step)
                history["loss_actor"].append(float(loss_actor))
                history["loss_critic"].append(avg_critic_loss)
                history["euler_residual"].append(er)
                history["bellman_residual"].append(br)
                print(f"SHAC-vanilla step {step:5d} | "
                      f"L_actor={float(loss_actor):.4f} | "
                      f"L_critic={avg_critic_loss:.6f} | "
                      f"euler={er:.6f} | bellman={br:.6f}")

            step += 1

    return {
        "policy":    policy,
        "value_net": value_net,
        "history":   history,
        "config":    config,
    }
