"""
Data pipeline helpers shared by trainer entrypoints.
"""

from __future__ import annotations

from typing import Dict, Iterable

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
    shuffle_key: str,
) -> tf.data.Iterator:
    """
    Build standard shuffled/repeated TF dataset iterator.
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset[shuffle_key].shape[0]).batch(batch_size).repeat()
    return iter(tf_dataset)

