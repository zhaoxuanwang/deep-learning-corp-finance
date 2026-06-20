"""Small statistical primitives, tfp-free.

v3 needs only two things from TensorFlow-Probability: the standard normal CDF and a
median. Implementing them here (via ``tf.math.erf`` and ``tf.sort``) drops the tfp
dependency, so v3 depends only on ``tensorflow`` + ``numpy``. That removes the pinned
old-stack requirement (tfp forced TF 2.16 / Keras 2 / numpy<2) and lets the same code
run on a modern CUDA / Colab image (numpy 2, recent TF) and on Apple Silicon.
"""
from __future__ import annotations

import math

import tensorflow as tf

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def normal_cdf(x):
    """Standard normal CDF Phi(x) = 0.5 (1 + erf(x / sqrt 2)). Exact; replaces ``tfp Normal.cdf``."""
    x = tf.convert_to_tensor(x)
    return 0.5 * (1.0 + tf.math.erf(x * tf.constant(_INV_SQRT2, x.dtype)))


def median(x, axis=0):
    """Median along ``axis`` (linear interpolation for even counts). Replaces
    ``tfp.stats.percentile(x, 50, axis)``."""
    x = tf.convert_to_tensor(x)
    n = tf.shape(x)[axis]
    xs = tf.sort(x, axis=axis)
    mid = n // 2
    hi = tf.gather(xs, mid, axis=axis)
    lo = tf.gather(xs, tf.maximum(mid - 1, 0), axis=axis)
    return tf.where(n % 2 == 0, 0.5 * (lo + hi), hi)
