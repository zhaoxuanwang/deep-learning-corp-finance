"""Short-Horizon Actor-Critic (SHAC) trainer — DDPG-style variant.

A variant of Xu et al. (2022) adapted for economic environments.  Retains
SHAC's core structure — h-step actor BPTT through differentiable dynamics
with windowed continuation across the trajectory — but replaces the critic
with a DDPG-style 1-step Bellman target.

Key modifications from vanilla SHAC
------------------------------------
1. **Actor bootstrap**: current V_φ (not target V̄_φ).  Gradients flow
   through V_φ(s_h) into the actor, providing a richer signal.

2. **Critic target**: 1-step Bellman with target π̄_θ + target V̄_φ:
       y(s) = r(s, π̄(s)) + γ · V̄(f(s, π̄(s), z'))
   This decouples the critic's regression target from the rapidly-changing
   current policy, breaking the positive feedback loop that causes
   divergence in vanilla SHAC on economic problems.

3. **Target networks**: both π̄ and V̄ (vanilla SHAC uses V̄ only).

4. **Cold start only**: the h-step actor rollout provides enough exact
   gradient signal through dynamics that the actor learns correctly even
   without an accurate initial critic.

Performance
-----------
The actor rollout (h-step BPTT + gradient) and critic 1-step Bellman
update are each compiled with @tf.function.  The h-step loop is statically
unrolled (no TensorArray) for best Metal/M1 performance.

Step counting
-------------
1 window = 1 step = 1 actor gradient update.
Each mini-batch of B trajectories yields horizon / short_horizon steps.
Total critic gradient steps per window = n_critic.

Data contract
-------------
train_dataset (trajectory format):
    s_endo_0:  (N, endo_dim)         initial endogenous state
    z_path:    (N, T+1, exo_dim)     pre-computed exogenous trajectory
    z_fork:    (N, T, exo_dim)       unused (present for pipeline compat)

val_dataset (flattened format, for evaluation):
    s_endo:      (M, endo_dim)
    z:           (M, exo_dim)
    z_next_main: (M, exo_dim)

Reference
---------
Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A.,
& Macklin, M. (2022). Accelerated Policy Learning with Parallel
Differentiable Simulation. ICLR 2022.
"""

import tensorflow as tf
from src.v2.trainers.config import SHACConfig
from src.v2.trainers.core import (
    polyak_update, build_target_value, build_target_policy,
    evaluate_euler_residual, evaluate_bellman_residual_v,
)
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys, fit_normalizer_traj,
)


_TRAIN_KEYS = ["s_endo_0", "z_path"]
_VAL_KEYS   = ["s_endo", "z", "z_next_main"]


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_shac(env, policy, value_net, train_dataset: dict,
               val_dataset: dict = None, config: SHACConfig = None):
    """Train policy and value networks using SHAC (DDPG-style variant).

    Actor (h-step BPTT, current π + current V):
      Unroll h steps with current policy through differentiable dynamics.
      Bootstrap with current V_φ(s_h).  Gradient flows through π and V
      into the dynamics chain.
      loss = -mean[ sum_{τ=0}^{h-1} γ^τ r(s_τ, π(s_τ)) + γ^h V(s_h) ]

    Critic (1-step Bellman, target π̄ + target V̄):
      For each visited state s_t in the window:
        y = r(s_t, π̄(s_t)) + γ · V̄(f(s_t, π̄(s_t), z_{t+1}))
      loss = mean[ (V(s_t) - stop_gradient(y))^2 ]

    Args:
        env:           MDPEnvironment instance.
        policy:        PolicyNetwork instance (actor π_θ).
        value_net:     StateValueNetwork instance (critic V_φ).
        train_dataset: Trajectory-format dataset dict (see module docstring).
        val_dataset:   Flattened-format dataset for evaluation (optional).
        config:        SHACConfig with hyperparameters.

    Returns:
        dict with keys: policy, value_net, history, config.
    """
    config      = config or SHACConfig()
    gamma       = env.discount()
    horizon     = config.horizon
    window_len  = config.short_horizon

    # Annealing: env decides; tf.Variable so @tf.function reads it live.
    schedule = env.annealing_schedule()
    temp_var = tf.Variable(
        schedule.value if schedule else config.temperature,
        dtype=tf.float32, trainable=False)

    # ------------------------------------------------------------------
    # Reward normalization
    # ------------------------------------------------------------------
    reward_scale = _resolve_reward_scale(env, config)
    reward_scale_tf = tf.constant(reward_scale, dtype=tf.float32)
    discount_tf     = tf.constant(gamma, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    validate_dataset_keys(train_dataset, _TRAIN_KEYS,
                          "train_shac", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _VAL_KEYS,
                              "train_shac", "val_dataset")

    horizon_data = train_dataset["z_path"].shape[1] - 1
    if horizon > horizon_data:
        raise ValueError(
            f"SHACConfig.horizon={horizon} exceeds dataset "
            f"horizon={horizon_data}. Re-generate data with "
            f"DataGeneratorConfig.horizon >= {horizon}."
        )
    if horizon % window_len != 0:
        raise ValueError(
            f"SHACConfig.horizon={horizon} must be divisible by "
            f"SHACConfig.short_horizon={window_len}. "
            f"Got {horizon} % {window_len} = {horizon % window_len}."
        )

    n_windows = horizon // window_len

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
    # Target networks: both π̄ and V̄
    # ------------------------------------------------------------------
    target_value_net = build_target_value(value_net)
    target_policy    = build_target_policy(policy)
    fit_normalizer_traj(env, train_dataset, target_policy, target_value_net)

    # ------------------------------------------------------------------
    # Precompute discount powers: γ^0, γ^1, ..., γ^{h-1}, γ^h
    # ------------------------------------------------------------------
    discount_powers = tf.constant(
        [gamma ** t for t in range(window_len + 1)], dtype=tf.float32)

    # ------------------------------------------------------------------
    # Compiled functions (closures capture env, networks, optimizers)
    # ------------------------------------------------------------------
    # The h-step loop is statically unrolled — no TensorArray or
    # tf.while_loop — for best performance on Metal/M1.

    @tf.function
    def actor_step(k_endo, z_window):
        """h-step actor rollout with current π + current V bootstrap.

        Args:
            k_endo:   (B, endo_dim) initial endogenous state for this window.
            z_window: (B, h+1, exo_dim) exogenous path slice for this window.

        Returns:
            loss_actor:  scalar loss.
            k_next:      (B, endo_dim) detached endogenous state at window end.
            all_states:  (B, h+1, state_dim) detached states for critic.
            all_z_next:  (B, h, exo_dim) next-step exogenous states for critic.
        """
        with tf.GradientTape() as tape:
            k_current = k_endo
            total_reward = tf.zeros(tf.shape(k_endo)[0])

            # Python lists → statically unrolled, no TensorArray
            collected_states = []
            collected_z_next = []

            for tau in range(window_len):
                z_t = z_window[:, tau, :]
                s_t = env.merge_state(k_current, z_t)
                a_t = policy(s_t, training=False)

                r_t = env.reward(s_t, a_t, temperature=temp_var)
                r_t = tf.reshape(r_t, [-1]) * reward_scale_tf

                collected_states.append(tf.stop_gradient(s_t))
                collected_z_next.append(z_window[:, tau + 1, :])

                total_reward = total_reward + discount_powers[tau] * r_t

                k_current = env.endogenous_transition(k_current, a_t, z_t)

            # Terminal state and bootstrap with CURRENT V (not target)
            z_end = z_window[:, window_len, :]
            s_end = env.merge_state(k_current, z_end)
            collected_states.append(tf.stop_gradient(s_end))

            v_bootstrap = tf.squeeze(
                value_net(s_end, training=False))
            total_reward = total_reward + discount_powers[window_len] * v_bootstrap

            loss_actor = -tf.reduce_mean(total_reward)

        grads_actor = tape.gradient(loss_actor, policy.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(grads_actor, policy.trainable_variables))

        all_states = tf.stack(collected_states, axis=1)    # (B, h+1, sd)
        all_z_next = tf.stack(collected_z_next, axis=1)    # (B, h, exo_dim)

        return loss_actor, tf.stop_gradient(k_current), all_states, all_z_next

    @tf.function
    def critic_step(s_batch, z_next_batch):
        """1-step Bellman target with target π̄ + target V̄.

        Args:
            s_batch:      (N, state_dim) states to fit.
            z_next_batch: (N, exo_dim) next-step exogenous states.
        """
        # Target policy action and reward
        a_target = target_policy(s_batch, training=False)
        r_target = env.reward(s_batch, a_target, temperature=temp_var)
        r_target = tf.reshape(r_target, [-1]) * reward_scale_tf

        # Target transition: s' under target policy
        s_endo, s_exo = env.split_state(s_batch)
        k_next_target = env.endogenous_transition(s_endo, a_target, s_exo)
        s_next_target = env.merge_state(k_next_target, z_next_batch)

        # Target value at s'
        v_next_target = tf.squeeze(
            target_value_net(s_next_target, training=False))
        bellman_target = tf.stop_gradient(
            r_target + discount_tf * v_next_target)

        # MSE regression on current value network
        with tf.GradientTape() as tape_c:
            v_pred = tf.squeeze(
                value_net(s_batch, training=True), axis=-1)
            loss_critic = tf.reduce_mean((v_pred - bellman_target) ** 2)

        grads_c = tape_c.gradient(loss_critic, value_net.trainable_variables)
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

        s_endo_0 = batch["s_endo_0"]   # (B, endo_dim)
        z_path   = batch["z_path"]     # (B, T+1, exo_dim)

        k_endo = tf.stop_gradient(s_endo_0)

        for win_idx in range(n_windows):
            if step >= config.n_steps:
                break

            t0 = win_idx * window_len
            z_window = z_path[:, t0:t0 + window_len + 1, :]

            # ==============================================================
            # (a) Actor: h-step rollout + gradient update
            # ==============================================================
            loss_actor, k_endo, all_states, all_z_next = \
                actor_step(k_endo, z_window)

            # ==============================================================
            # (b) Critic: n_critic gradient steps on collected transitions
            # ==============================================================
            state_dim = all_states.shape[-1]
            exo_dim = all_z_next.shape[-1]
            critic_s = tf.reshape(
                all_states[:, :-1, :], [-1, state_dim])     # (B*h, sd)
            critic_z = tf.reshape(
                all_z_next, [-1, exo_dim])                   # (B*h, exo_dim)

            for _ in range(config.n_critic):
                loss_c = critic_step(critic_s, critic_z)

            # ==============================================================
            # (c) Polyak update both target networks
            # ==============================================================
            polyak_update(value_net, target_value_net,
                          tau=config.polyak_rate)
            polyak_update(policy, target_policy,
                          tau=config.polyak_rate)

            if schedule:
                schedule.update()
                temp_var.assign(schedule.value)

            # Logging
            if step % config.eval_interval == 0 or step == config.n_steps - 1:
                temperature = float(temp_var)
                euler_res = bellman_res = float("nan")
                if val_dataset is not None:
                    euler_res = evaluate_euler_residual(
                        env, policy, val_dataset, temperature=temperature)
                    bellman_res = evaluate_bellman_residual_v(
                        env, policy, value_net, val_dataset,
                        temperature=temperature)
                history["step"].append(step)
                history["loss_actor"].append(float(loss_actor))
                history["loss_critic"].append(float(loss_c))
                history["euler_residual"].append(euler_res)
                history["bellman_residual"].append(bellman_res)
                print(f"SHAC step {step:5d} | "
                      f"L_actor={float(loss_actor):.4f} | "
                      f"L_critic={float(loss_c):.6f} | "
                      f"euler={euler_res:.6f} | bellman={bellman_res:.6f}")

            step += 1

    return {
        "policy":    policy,
        "value_net": value_net,
        "history":   history,
        "config":    config,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_reward_scale(env, config: SHACConfig) -> float:
    """Determine the reward scale from config settings.

    Priority: reward_scale_override > normalize_rewards > 1.0 (no scaling).
    """
    if config.reward_scale_override is not None:
        reward_scale = float(config.reward_scale_override)
        print(f"SHAC: manual reward_scale = {reward_scale:.6f}")
        return reward_scale

    if config.normalize_rewards:
        reward_scale = float(env.compute_reward_scale(
            seed=tf.constant(list(config.master_seed), dtype=tf.int32)))
        print(f"SHAC: auto reward_scale = {reward_scale:.6f} "
              f"(1/|V*| ≈ {1.0/reward_scale:.1f})")
        return reward_scale

    return 1.0
