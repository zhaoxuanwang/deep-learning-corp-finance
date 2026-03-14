"""
src/v2/data/pipeline.py

Mini-batch iteration utilities and normalizer fitting for offline training.

The mini-batch ORDER within each epoch is intentionally non-deterministic
(standard SGD convention). Reproducibility is guaranteed at the dataset
level via the master seed in DataGenerator — the underlying data is fixed.

Normalizer fitting
------------------
fit_normalizer_traj  — for trajectory-format datasets (LR, SHAC).
fit_normalizer_flat  — for flattened-format datasets (ER, BRM).

Both functions accept one or more network instances and call
network.normalizer.fit(s_all) on each, so all networks in a training
run receive identical normalizer statistics.
"""

from __future__ import annotations

from typing import Dict, List

import tensorflow as tf


def validate_dataset_keys(
    dataset: Dict[str, tf.Tensor],
    required_keys: List[str],
    method_name: str = "",
    dataset_label: str = "dataset",
) -> None:
    """Raise ValueError if dataset is missing any required keys.

    Args:
        dataset:       Dict of tensor arrays.
        required_keys: Keys that must be present.
        method_name:   Name of calling method (for error messages).
        dataset_label: Human-readable label for the dataset (for errors).
    """
    missing = [k for k in required_keys if k not in dataset]
    if missing:
        prefix = f"{method_name}: " if method_name else ""
        raise ValueError(
            f"{prefix}{dataset_label} is missing required keys: {missing}. "
            f"Got: {list(dataset.keys())}"
        )


def build_iterator(
    dataset: Dict[str, tf.Tensor],
    batch_size: int,
    shuffle_buffer: int = 10000,
) -> tf.data.Dataset:
    """Wrap a dataset dict into a repeating, shuffled tf.data mini-batch iterator.

    The dataset is shuffled at the element level each epoch (non-deterministic
    mini-batch order) and repeated indefinitely. Call `.take(n_steps)` on the
    returned dataset to consume exactly n_steps batches.

    Args:
        dataset:        Dict mapping key -> tensor of shape (N, ...).
                        All tensors must have the same leading dimension N.
        batch_size:     Mini-batch size.
        shuffle_buffer: Buffer size for tf.data.Dataset.shuffle().
                        Set to >= N for a full shuffle each epoch.

    Returns:
        tf.data.Dataset that yields batched dicts indefinitely.

    Example::

        ds = build_iterator(train_flat, batch_size=256)
        for batch in ds.take(n_steps):
            s = env.merge_state(batch["s_endo"], batch["z"])
            ...
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = (
        tf_dataset
        .shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    return tf_dataset


# ---------------------------------------------------------------------------
# Normalizer fitting
# ---------------------------------------------------------------------------

def fit_normalizer_traj(env, dataset: Dict[str, tf.Tensor],
                        *networks) -> None:
    """Fit StaticNormalizer for trajectory-format datasets (LR, SHAC).

    Exogenous states: all N * T_data values from z_path (ergodic distribution).
    Endogenous states: the N initial states s_endo_0 (uniform or harvested).

    Each s_endo_0[i] is paired with all T_data exogenous states from
    trajectory i by repeating it T_data times.  Per-feature mean and std
    are computed from the resulting N * T_data array — this gives exact
    ergodic stats for z and full-range stats for k, in a single pass.

    All supplied networks receive identical normalizer statistics.

    Args:
        env:      MDPEnvironment (provides merge_state, exo_dim).
        dataset:  Trajectory-format dict with keys s_endo_0, z_path.
        *networks: One or more GenericNetwork instances to fit.
    """
    z_path  = dataset["z_path"]    # (N, T_data+1, exo_dim)
    s_endo0 = dataset["s_endo_0"]  # (N, endo_dim)

    T_plus_1 = tf.shape(z_path)[1]
    exo_dim  = env.exo_dim()

    z_flat  = tf.reshape(z_path, [-1, exo_dim])                    # (N*(T+1), exo_dim)
    k_rep   = tf.repeat(s_endo0, T_plus_1, axis=0)                 # (N*(T+1), endo_dim)
    s_all   = env.merge_state(k_rep, z_flat)                       # (N*(T+1), state_dim)

    for net in networks:
        net.normalizer.fit(s_all)


def fit_normalizer_flat(env, dataset: Dict[str, tf.Tensor],
                        *networks) -> None:
    """Fit StaticNormalizer for flattened-format datasets (ER, BRM).

    Uses all N * T_data paired (s_endo, z) observations directly.
    s_endo is i.i.d. uniform (covering the full endogenous range) and
    z is ergodic, so both components are well-represented.

    All supplied networks receive identical normalizer statistics.

    Args:
        env:      MDPEnvironment (provides merge_state).
        dataset:  Flattened-format dict with keys s_endo, z.
        *networks: One or more GenericNetwork instances to fit.
    """
    s_all = env.merge_state(dataset["s_endo"], dataset["z"])  # (N*T, state_dim)

    for net in networks:
        net.normalizer.fit(s_all)
