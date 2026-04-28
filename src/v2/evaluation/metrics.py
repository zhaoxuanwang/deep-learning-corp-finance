"""Shared metric helpers for notebook-facing policy evaluation."""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf


def evaluate_lifetime_reward(
    env,
    policy,
    traj_dataset: Mapping[str, tf.Tensor],
    *,
    horizon: int,
    n_samples: int | None = None,
) -> float:
    """Average discounted reward along fixed exogenous-shock trajectories."""
    s_endo_0 = traj_dataset["s_endo_0"]
    z_path = traj_dataset["z_path"]
    if n_samples is not None:
        s_endo_0 = s_endo_0[:n_samples]
        z_path = z_path[:n_samples]

    max_steps = int(z_path.shape[1]) - 1
    n_steps = min(int(horizon), max_steps)
    k = s_endo_0
    total_reward = tf.zeros(tf.shape(k)[0], dtype=k.dtype)
    discount_t = tf.cast(1.0, k.dtype)
    gamma = tf.cast(env.discount(), k.dtype)

    for t in range(n_steps):
        z_t = z_path[:, t, :]
        s_t = env.merge_state(k, z_t)
        a_t = policy(s_t, training=False)
        r_t = tf.cast(tf.reshape(env.reward(s_t, a_t), [-1]), k.dtype)
        total_reward = total_reward + discount_t * r_t
        k = env.endogenous_transition(k, a_t, z_t)
        discount_t = discount_t * gamma

    return float(tf.reduce_mean(total_reward))


def evaluate_policy_mae(env, policy, flat_dataset: Mapping[str, tf.Tensor]) -> float:
    """Mean absolute error in next-period capital versus the analytical policy."""
    s_endo = flat_dataset["s_endo"]
    z = flat_dataset["z"]
    state = env.merge_state(s_endo, z)
    a_pred = policy(state, training=False)
    a_true = env.analytical_policy(state)
    k_next_pred = env.endogenous_transition(s_endo, a_pred, z)
    k_next_true = env.endogenous_transition(s_endo, a_true, z)
    return float(tf.reduce_mean(tf.abs(k_next_pred - k_next_true)))
