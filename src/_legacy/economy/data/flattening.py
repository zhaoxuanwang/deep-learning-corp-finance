"""
Flatten trajectory datasets into i.i.d. transition datasets.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import tensorflow as tf

from src.economy.rng import SeedSchedule, VariableID


def _single_seed(
    seed_schedule: SeedSchedule,
    split: str,
    variable: VariableID,
    step: Optional[int] = None,
) -> tf.Tensor:
    if step is None:
        return seed_schedule.get_single_seed(split, variable)
    return seed_schedule.get_single_seed(split, variable, step=step)


def build_flattened_dataset(
    traj_data: Dict[str, tf.Tensor],
    seed_schedule: SeedSchedule,
    *,
    split: str,
    k_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    include_debt: bool = False,
    shuffle: bool = False,
    seed_step: Optional[int] = None,
) -> Dict[str, tf.Tensor]:
    """
    Build flattened transition dataset from trajectory-format data.
    """
    z_path = traj_data["z_path"]  # (N, T+1)
    z_fork = traj_data["z_fork"]  # (N, T, 1)

    n_paths = tf.shape(z_path)[0]
    horizon = tf.shape(z_path)[1] - 1
    n_total = n_paths * horizon

    z_curr = z_path[:, :-1]  # (N, T)
    z_next_main = z_path[:, 1:]  # (N, T)
    z_next_fork = tf.squeeze(z_fork, axis=-1)  # (N, T)

    z_flat = tf.reshape(z_curr, [-1])  # (N*T,)
    z_next_main_flat = tf.reshape(z_next_main, [-1])
    z_next_fork_flat = tf.reshape(z_next_fork, [-1])

    k_seed = _single_seed(seed_schedule, split, VariableID.K0, seed_step)
    k_min, k_max = k_bounds
    k_flat = tf.random.stateless_uniform(
        shape=[n_total],
        seed=k_seed,
        minval=k_min,
        maxval=k_max,
        dtype=tf.float32,
    )

    b_flat = None
    if include_debt:
        b_seed = _single_seed(seed_schedule, split, VariableID.B0, seed_step)
        b_min, b_max = b_bounds
        b_flat = tf.random.stateless_uniform(
            shape=[n_total],
            seed=b_seed,
            minval=b_min,
            maxval=b_max,
            dtype=tf.float32,
        )

    if shuffle:
        shuffle_seed = _single_seed(seed_schedule, split, VariableID.EPS1, seed_step)
        indices = tf.range(n_total, dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=shuffle_seed)

        k_flat = tf.gather(k_flat, shuffled_indices)
        z_flat = tf.gather(z_flat, shuffled_indices)
        z_next_main_flat = tf.gather(z_next_main_flat, shuffled_indices)
        z_next_fork_flat = tf.gather(z_next_fork_flat, shuffled_indices)
        if include_debt:
            b_flat = tf.gather(b_flat, shuffled_indices)

    result: Dict[str, tf.Tensor] = {
        "k": k_flat,
        "z": z_flat,
        "z_next_main": z_next_main_flat,
        "z_next_fork": z_next_fork_flat,
    }
    if include_debt and b_flat is not None:
        result["b"] = b_flat
    return result

