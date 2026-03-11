"""
src/v2/data/pipeline.py

Mini-batch iteration utilities for offline training.

The mini-batch ORDER within each epoch is intentionally non-deterministic
(standard SGD convention). Reproducibility is guaranteed at the dataset
level via the master seed in DataGenerator — the underlying data is fixed.
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
