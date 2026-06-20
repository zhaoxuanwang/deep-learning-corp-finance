"""Stateless keyed seeds (DF26 Sec 10).

Every random draw is a pure function of ``(master, purpose, *indices)`` via
``tf.random.stateless_*`` ops, so output is identical regardless of call order,
parallelism, or device. Never use the stateful global generator.
"""
from __future__ import annotations

from enum import IntEnum

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NET


class Purpose(IntEnum):
    """Stable per-substream labels (do not renumber; reproducibility depends on it)."""

    INIT = 1        # weight initialization
    TRAIN = 2       # Block-1 training state/parameter draws
    COLLECT = 3     # collector parameter draws
    SIM_INIT = 4    # simulation initial (z, k, b)
    SIM_STEP = 5    # simulation Tauchen transitions
    SIM_RESEED = 6  # simulation reseed after default
    SURROGATE = 7   # surrogate mini-batch shuffling
    FOLDS = 8       # cross-validation fold assignment
    LM = 9          # Levenberg-Marquardt restart starts


_MASK = 0x7FFFFFFF


def key(master, purpose, *indices) -> tf.Tensor:
    """Derive a stateless seed ``[2]`` (int32) from a master, a purpose, and indices."""
    seed = tf.constant([int(master) & _MASK, int(purpose) & _MASK], dtype=tf.int32)
    for idx in indices:
        seed = tf.random.experimental.stateless_fold_in(seed, tf.cast(idx, tf.int32))
    return seed


def uniform(shape, master, purpose, *indices, minval=0.0, maxval=1.0, dtype=TF_FLOAT_NET):
    """Stateless uniform draw keyed by ``(master, purpose, *indices)``."""
    return tf.random.stateless_uniform(
        shape, seed=key(master, purpose, *indices),
        minval=minval, maxval=maxval, dtype=dtype,
    )


def normal(shape, master, purpose, *indices, mean=0.0, stddev=1.0, dtype=TF_FLOAT_NET):
    """Stateless normal draw keyed by ``(master, purpose, *indices)``."""
    z = tf.random.stateless_normal(shape, seed=key(master, purpose, *indices), dtype=dtype)
    return tf.cast(mean, dtype) + tf.cast(stddev, dtype) * z


def randint(shape, master, purpose, *indices, minval, maxval):
    """Stateless integer draw in ``[minval, maxval)`` keyed by the indices."""
    return tf.random.stateless_uniform(
        shape, seed=key(master, purpose, *indices),
        minval=minval, maxval=maxval, dtype=tf.int32,
    )
