"""Core training utilities: target network ops and evaluation metrics.

Shared across all training methods (LR, ER, BRM, MVE).

Removed in this refactor:
    - ReplayBuffer      (replaced by offline DataGenerator datasets)
    - SeedSchedule      (moved to src/v2/data/rng.py)
    - collect_transitions / generate_eval_dataset  (data is now external)

Evaluation functions use the flattened dataset format:
    {s_endo, z, z_next_main, z_next_fork}
which is produced by DataGenerator.get_flattened_dataset().
"""

import tensorflow as tf


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

    Shared by ER and BRM trainers for stable next-action computation.
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
