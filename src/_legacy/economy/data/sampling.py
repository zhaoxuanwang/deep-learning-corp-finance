"""
Sampling and trajectory rollout helpers for DataGenerator.
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf

from src.economy.parameters import ShockParams
from src.economy.rng import VariableID
from src.economy.shocks import step_ar1_tf


def generate_initial_states(
    batch_size: int,
    k_bounds: Tuple[float, float],
    logz_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    *,
    k_seed: tf.Tensor,
    z_seed: tf.Tensor,
    b_seed: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Sample initial state tuple (k0, z0, b0) from configured bounds.
    """
    k_min, k_max = k_bounds
    logz_min, logz_max = logz_bounds
    b_min, b_max = b_bounds

    k0 = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=k_seed,
        minval=k_min,
        maxval=k_max,
        dtype=tf.float32,
    )

    logz0 = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=z_seed,
        minval=logz_min,
        maxval=logz_max,
        dtype=tf.float32,
    )
    z0 = tf.exp(logz0)

    b0 = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=b_seed,
        minval=b_min,
        maxval=b_max,
        dtype=tf.float32,
    )

    return k0, z0, b0


def generate_shocks(
    batch_size: int,
    horizon: int,
    *,
    eps1_seed: tf.Tensor,
    eps2_seed: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Sample two independent N(0,1) shock matrices.
    """
    eps1 = tf.random.stateless_normal(
        shape=(batch_size, horizon),
        seed=eps1_seed,
        dtype=tf.float32,
    )
    eps2 = tf.random.stateless_normal(
        shape=(batch_size, horizon),
        seed=eps2_seed,
        dtype=tf.float32,
    )
    return eps1, eps2


def rollout_forked_path(
    z0: tf.Tensor,
    eps_main: tf.Tensor,
    eps_fork: tf.Tensor,
    shock_params: ShockParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Roll out main trajectory z_path and one-step forks z_fork.
    """
    horizon = tf.shape(eps_main)[1]

    z_main_list = [z0]
    z_fork_list = []
    z_current = z0

    for t in range(horizon):
        z_next_main = step_ar1_tf(
            z_current,
            shock_params.rho,
            shock_params.sigma,
            shock_params.mu,
            eps=eps_main[:, t],
        )
        z_main_list.append(z_next_main)

        z_next_fork = step_ar1_tf(
            z_current,
            shock_params.rho,
            shock_params.sigma,
            shock_params.mu,
            eps=eps_fork[:, t],
        )
        z_fork_list.append(tf.reshape(z_next_fork, [-1, 1]))

        z_current = z_next_main

    z_path = tf.stack(z_main_list, axis=1)  # (B, T+1)
    z_fork = tf.stack(z_fork_list, axis=1)  # (B, T, 1)
    return z_path, z_fork


def generate_batch(
    seeds_dict: Dict[VariableID, tf.Tensor],
    batch_size: int,
    horizon: int,
    k_bounds: Tuple[float, float],
    logz_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    shock_params: ShockParams,
) -> Dict[str, tf.Tensor]:
    """
    Generate one trajectory-format batch.
    """
    k0, z0, b0 = generate_initial_states(
        batch_size,
        k_bounds,
        logz_bounds,
        b_bounds,
        k_seed=seeds_dict[VariableID.K0],
        z_seed=seeds_dict[VariableID.Z0],
        b_seed=seeds_dict[VariableID.B0],
    )

    eps1, eps2 = generate_shocks(
        batch_size,
        horizon,
        eps1_seed=seeds_dict[VariableID.EPS1],
        eps2_seed=seeds_dict[VariableID.EPS2],
    )

    z_path, z_fork = rollout_forked_path(z0, eps1, eps2, shock_params)

    return {
        "k0": k0,
        "z0": z0,
        "b0": b0,
        "z_path": z_path,
        "z_fork": z_fork,
        "eps_path": eps1,
        "eps_fork": eps2,
    }

