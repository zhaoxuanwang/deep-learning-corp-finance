"""Core training utilities: target network ops and evaluation metrics.

Shared across all training methods (LR, ER, BRM, SHAC).

Removed in this refactor:
    - ReplayBuffer      (replaced by offline DataGenerator datasets)
    - SeedSchedule      (moved to src/v2/data/rng.py)
    - collect_transitions / generate_eval_dataset  (data is now external)

Evaluation functions use the flattened dataset format:
    {s_endo, z, z_next_main, z_next_fork}
which is produced by DataGenerator.get_flattened_dataset().
"""

import tensorflow as tf
from src.v2.data.pipeline import fit_normalizer_traj, fit_normalizer_flat


# ---------------------------------------------------------------------------
# Target Network Utilities
# ---------------------------------------------------------------------------

def polyak_update(source: tf.keras.Model, target: tf.keras.Model, tau: float):
    """Soft-update: w_target ← τ·w_target + (1-τ)·w_source."""
    for src_var, tgt_var in zip(source.trainable_variables,
                                target.trainable_variables):
        tgt_var.assign(tau * tgt_var + (1.0 - tau) * src_var)


def hard_update(source: tf.keras.Model, target: tf.keras.Model):
    """Copy source weights to target."""
    for src_var, tgt_var in zip(source.trainable_variables,
                                target.trainable_variables):
        tgt_var.assign(src_var)


def build_target_policy(policy):
    """Create a target policy network with identical architecture.

    Used by ER (stable next-action), BRM, and SHAC (DDPG-style critic
    Bellman targets with target π̄).
    """
    from src.v2.networks.policy import PolicyNetwork
    target = PolicyNetwork(
        state_dim=policy.input_dim,
        action_dim=policy.action_dim,
        action_low=policy.action_low,
        action_high=policy.action_high,
        action_center=policy.action_center,
        action_half_range=policy.action_half_range,
        n_layers=policy.hidden_stack.dense_layers.__len__(),
        n_neurons=policy.hidden_stack.dense_layers[0].units,
        name="policy_target",
    )
    dummy = tf.zeros((1, policy.input_dim))
    target(dummy)
    hard_update(policy, target)
    return target


def build_target_value(value_net):
    """Create a target value network with identical architecture.

    Used by SHAC for stable critic Bellman targets (target V̄_φ).
    """
    from src.v2.networks.state_value import StateValueNetwork
    target = StateValueNetwork(
        state_dim=value_net.input_dim,
        n_layers=value_net.hidden_stack.dense_layers.__len__(),
        n_neurons=value_net.hidden_stack.dense_layers[0].units,
        name="value_target",
    )
    dummy = tf.zeros((1, value_net.input_dim))
    target(dummy)
    hard_update(value_net, target)
    return target


# ---------------------------------------------------------------------------
# Warm-start utilities
# ---------------------------------------------------------------------------

def warm_start_value_net(env, value_net, train_dataset: dict,
                         n_steps: int = 200, learning_rate: float = 1e-3,
                         batch_size: int = 256, reward_scale: float = 1.0):
    """Pre-train value network on analytical terminal-value targets.

    Fits V_φ(s) ≈ λ · V^term(k) = λ · r(k, z̄, δk) / (1-γ), giving the
    critic a reasonable initialization where dV/dk > 0.  This avoids the
    cold-start degenerate equilibrium where an uninformative bootstrap
    causes the actor to learn zero investment.

    When reward_scale (λ) != 1.0, targets are scaled so V_φ is consistent
    with the scaled rewards used during training (e.g. SHAC with reward
    normalization).

    Accepts both dataset formats:
      - Trajectory (SHAC): keys s_endo_0, z_path
      - Flattened  (BRM):  keys s_endo, z

    The normalizer is fit internally (idempotent with later trainer fitting).

    Args:
        env:           MDPEnvironment with terminal_value() implemented.
        value_net:     StateValueNetwork to warm-start (modified in-place).
        train_dataset: Trajectory- or flattened-format dataset dict.
        n_steps:       Number of MSE regression steps.
        learning_rate: Adam learning rate for warm-start.
        batch_size:    Mini-batch size for warm-start.
        reward_scale:  Multiplier for V targets (default 1.0 = no scaling).
    """
    # Detect format and build (s_all, v_target) pairs
    if "z_path" in train_dataset:
        # Trajectory format (SHAC, LR)
        fit_normalizer_traj(env, train_dataset, value_net)

        z_path   = train_dataset["z_path"]      # (N, T+1, exo_dim)
        s_endo_0 = train_dataset["s_endo_0"]    # (N, endo_dim)
        T_plus_1 = tf.shape(z_path)[1]
        exo_dim  = env.exo_dim()

        z_flat  = tf.reshape(z_path, [-1, exo_dim])
        k_rep   = tf.repeat(s_endo_0, T_plus_1, axis=0)
        s_all   = env.merge_state(k_rep, z_flat)
        v_target = tf.reshape(env.terminal_value(k_rep), [-1])
    else:
        # Flattened format (BRM)
        fit_normalizer_flat(env, train_dataset, value_net)

        s_endo = train_dataset["s_endo"]        # (N, endo_dim)
        z      = train_dataset["z"]             # (N, exo_dim)
        s_all  = env.merge_state(s_endo, z)
        v_target = tf.reshape(env.terminal_value(s_endo), [-1])

    v_target = v_target * reward_scale

    ds = tf.data.Dataset.from_tensor_slices((s_all, v_target))
    ds = ds.shuffle(min(int(s_all.shape[0]), 50000)).batch(
        batch_size, drop_remainder=True).repeat()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def _step(s_batch, v_batch):
        with tf.GradientTape() as tape:
            v_pred = tf.squeeze(value_net(s_batch, training=True), axis=-1)
            loss = tf.reduce_mean((v_pred - v_batch) ** 2)
        grads = tape.gradient(loss, value_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, value_net.trainable_variables))
        return loss

    for i, (s_b, v_b) in enumerate(ds.take(n_steps)):
        loss = _step(s_b, v_b)
        if i % 50 == 0 or i == n_steps - 1:
            print(f"  warm-start step {i:4d} | MSE={float(loss):.2f}")


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
# All evaluation functions accept the flattened dataset format:
#     {s_endo: (N, endo_dim), z: (N, exo_dim),
#      z_next_main: (N, exo_dim), z_next_fork: (N, exo_dim)}
# ---------------------------------------------------------------------------

def evaluate_euler_residual(env, policy, val_dataset: dict,
                            temperature: float = 1e-6) -> float:
    """Mean absolute Euler residual on the validation dataset.

    Computes the one-step Euler condition using pre-computed z transitions.
    Returns NaN if the environment does not support ER.

    Args:
        env:         MDPEnvironment instance.
        policy:      Policy network (callable: s -> a).
        val_dataset: Flattened dataset dict with keys:
                     s_endo, z, z_next_main.
        temperature: Smooth-gate temperature.

    Returns:
        float: mean absolute residual, or nan if ER not supported.
    """
    try:
        s_endo = val_dataset["s_endo"]
        z      = val_dataset["z"]
        z_next = val_dataset["z_next_main"]

        s      = env.merge_state(s_endo, z)
        a      = policy(s, training=False)

        k_next = env.endogenous_transition(s_endo, a, z)
        s_next = env.merge_state(k_next, z_next)
        a_next = policy(s_next, training=False)

        residual = env.euler_residual(s, a, s_next, a_next,
                                      temperature=temperature)
        return float(tf.reduce_mean(tf.abs(residual)))
    except NotImplementedError:
        return float("nan")


def evaluate_bellman_residual_v(env, policy, value_net, val_dataset: dict,
                                temperature: float = 1e-6) -> float:
    """Mean absolute Bellman residual V(s) - r(s,a) - γ·V(s') on val data.

    Args:
        env:         MDPEnvironment instance.
        policy:      Policy network.
        value_net:   State-value network V(s).
        val_dataset: Flattened dataset dict with keys:
                     s_endo, z, z_next_main.
        temperature: Smooth-gate temperature.

    Returns:
        float: mean absolute residual.
    """
    s_endo = val_dataset["s_endo"]
    z      = val_dataset["z"]
    z_next = val_dataset["z_next_main"]

    s = env.merge_state(s_endo, z)
    a = policy(s, training=False)

    r = env.reward(s, a, temperature=temperature)
    r = tf.squeeze(r) if r.shape.rank > 1 else r

    k_next = env.endogenous_transition(s_endo, a, z)
    s_next = env.merge_state(k_next, z_next)

    v_s    = tf.squeeze(value_net(s,      training=False))
    v_next = tf.squeeze(value_net(s_next, training=False))

    residual = v_s - r - env.discount() * v_next
    return float(tf.reduce_mean(tf.abs(residual)))
