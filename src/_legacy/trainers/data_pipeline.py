"""
Data pipeline helpers shared by trainer entrypoints.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import tensorflow as tf


def validate_dataset_keys(
    dataset: Dict[str, tf.Tensor],
    required_keys: Iterable[str],
    *,
    method_name: str,
    dataset_label: str = "dataset",
) -> None:
    required = set(required_keys)
    provided = set(dataset.keys())
    if not required.issubset(provided):
        raise ValueError(
            f"{method_name} requires {dataset_label} keys {required}. "
            f"Got keys: {provided}."
        )


def build_training_iterator(
    dataset: Dict[str, tf.Tensor],
    *,
    batch_size: int,
    batch_order: str = "as_is",
    permutation_seed: Optional[Tuple[int, int]] = None,
) -> tf.data.Iterator:
    """
    Build repeated TF dataset iterator with explicit ordering semantics.
    """
    valid_orders = {"as_is", "fixed_permutation"}
    if batch_order not in valid_orders:
        raise ValueError(
            f"batch_order must be one of {valid_orders}, got '{batch_order}'"
        )

    prepared = dataset
    if batch_order == "fixed_permutation":
        if permutation_seed is None:
            raise ValueError(
                "permutation_seed is required when batch_order='fixed_permutation'."
            )
        seed = tf.constant([int(permutation_seed[0]), int(permutation_seed[1])], dtype=tf.int32)
        first_key = next(iter(dataset.keys()))
        n_obs = tf.shape(dataset[first_key])[0]
        indices = tf.range(n_obs, dtype=tf.int32)
        perm = tf.random.experimental.stateless_shuffle(indices, seed=seed)
        prepared = {k: tf.gather(v, perm) for k, v in dataset.items()}

    tf_dataset = tf.data.Dataset.from_tensor_slices(prepared)
    tf_dataset = tf_dataset.batch(batch_size).repeat()
    return iter(tf_dataset)
