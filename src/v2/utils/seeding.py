"""Stateless seed helpers for reproducible v2 experiments.

The project standard is to control all routine experiment randomness from a
single master seed pair `(m0, m1)`.  This module provides small utilities to
derive deterministic sub-seeds for distinct tasks such as:

- network initialization
- tf.data mini-batch shuffling
- BRM critic warm-start shuffling
- notebook/runtime seeding

Strict reproducibility beyond these routine controls (for example deterministic
TensorFlow kernels on a fixed device) remains optional.
"""

from __future__ import annotations

import hashlib
import random
from typing import Tuple

import numpy as np
import tensorflow as tf


MasterSeed = Tuple[int, int]


def _normalize_master_seed(master_seed) -> MasterSeed:
    """Validate and normalize a master seed pair."""
    if master_seed is None or len(master_seed) != 2:
        raise ValueError(
            "master_seed must be a length-2 pair of integers. "
            f"Got {master_seed!r}"
        )
    return int(master_seed[0]), int(master_seed[1])


def _token_to_int32(token) -> int:
    """Convert a namespace token into a stable signed int32."""
    if isinstance(token, (int, np.integer)):
        return int(np.int32(int(token)))
    if isinstance(token, str):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int(np.frombuffer(digest[:4], dtype=np.int32)[0])
    raise TypeError(
        "Seed tokens must be ints or strings. "
        f"Got {type(token).__name__}: {token!r}"
    )


def fold_in_seed(master_seed, *tokens) -> MasterSeed:
    """Derive a deterministic child seed pair from a master seed pair."""
    seed = tf.constant(_normalize_master_seed(master_seed), dtype=tf.int32)
    for token in tokens:
        seed = tf.random.experimental.stateless_fold_in(
            seed, _token_to_int32(token)
        )
    seed_np = seed.numpy()
    return int(seed_np[0]), int(seed_np[1])


def seed_pair_to_int(seed_pair) -> int:
    """Compress a seed pair into a stable positive int for Keras / tf.data."""
    s0, s1 = _normalize_master_seed(seed_pair)
    u0 = int(s0) & 0xFFFFFFFF
    u1 = int(s1) & 0xFFFFFFFF
    value = ((u0 << 1) ^ u1) % (2**31 - 1)
    return value if value != 0 else 1


def make_seed_int(master_seed, *tokens) -> int:
    """Derive a deterministic positive int seed from a master seed pair."""
    return seed_pair_to_int(fold_in_seed(master_seed, *tokens))


def seed_runtime(master_seed, *tokens, strict_reproducibility: bool = False) -> int:
    """Seed Python, NumPy, TensorFlow, and optionally strict TF determinism."""
    runtime_seed = make_seed_int(master_seed, "runtime", *tokens)
    random.seed(runtime_seed)
    np.random.seed(runtime_seed)
    tf.keras.utils.set_random_seed(runtime_seed)
    if strict_reproducibility:
        tf.config.experimental.enable_op_determinism()
    return runtime_seed
