"""Network checkpointing via ``tf.train.Checkpoint`` (DF26 Sec 12.5; review NN-2).

Saving raw ``tf.Variable`` state with ``tf.train.Checkpoint`` is robust across Keras
versions (unlike Keras model serialization, ENV-1). A reloaded bundle must reproduce
the in-memory predictions (NN-2), which :func:`save_bundle`/:func:`load_bundle` and
the regression test verify.
"""
from __future__ import annotations

import tensorflow as tf


def save_bundle(bundle, path: str) -> str:
    """Write all bundle weights to ``path`` (a checkpoint prefix)."""
    ckpt = tf.train.Checkpoint(bundle=bundle)
    return ckpt.write(str(path))


def load_bundle(bundle, path: str) -> None:
    """Restore bundle weights from ``path`` in place."""
    ckpt = tf.train.Checkpoint(bundle=bundle)
    ckpt.restore(str(path)).expect_partial()
