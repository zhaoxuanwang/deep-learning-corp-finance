"""Smooth (differentiable) approximations of indicator functions.

These provide gradient-friendly replacements for hard 0/1 indicators
used in economic models — fixed cost gates, external financing triggers,
default probability, etc.  As temperature tau -> 0 they recover the
exact hard indicator.

Three modes are supported:

    "soft"  — sigmoid approximation (default, fully differentiable)
    "hard"  — exact indicator (no gradient through the gate)
    "ste"   — straight-through estimator (hard forward, soft backward)

Reference:
    v2_report.md, Section "Soft Gates": sigma((|x| - eps) / tau)
"""

from typing import Literal, Union, Optional, Any

import tensorflow as tf

Numeric = Union[tf.Tensor, Any]

# Default constants (self-contained — no external imports).
DEFAULT_THRESHOLD = 1e-4
DEFAULT_LOGIT_CLIP = 5.0


def indicator_abs_gt(
    x: Numeric,
    threshold: float = DEFAULT_THRESHOLD,
    temperature: float = 0.1,
    logit_clip: float = DEFAULT_LOGIT_CLIP,
    mode: Literal["hard", "ste", "soft"] = "soft",
) -> tf.Tensor:
    """Smooth gate for |x| > threshold.

    sigma( (|x| - eps) / tau )

    Args:
        x:           Input tensor (e.g. I/k).
        threshold:   Epsilon tolerance.
        temperature: Annealing temperature tau.
        logit_clip:  Clip range for the logit (prevents saturation).
        mode:        "hard", "soft", or "ste".
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    eps = tf.cast(threshold, dtype)
    temp = tf.maximum(tf.cast(temperature, dtype), 1e-6)

    abs_x = tf.abs(x)
    g_hard = tf.cast(abs_x > eps, dtype)

    if mode == "hard":
        return g_hard

    logit = (abs_x - eps) / temp
    logit = tf.clip_by_value(logit, -logit_clip, logit_clip)
    g_soft = tf.nn.sigmoid(logit)

    if mode == "soft":
        return g_soft

    # STE: hard forward, soft backward
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def indicator_lt(
    x: Numeric,
    threshold: float = DEFAULT_THRESHOLD,
    temperature: float = 0.1,
    logit_clip: float = DEFAULT_LOGIT_CLIP,
    mode: Literal["hard", "ste", "soft"] = "soft",
) -> tf.Tensor:
    """Smooth gate for x < -threshold.

    sigma( -(x + eps) / tau )

    Args:
        x:           Input tensor (e.g. e/k).
        threshold:   Epsilon tolerance.
        temperature: Annealing temperature tau.
        logit_clip:  Clip range for the logit.
        mode:        "hard", "soft", or "ste".
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    eps = tf.cast(threshold, dtype)
    temp = tf.maximum(tf.cast(temperature, dtype), 1e-6)

    g_hard = tf.cast(x < -eps, dtype)

    if mode == "hard":
        return g_hard

    logit = -(x + eps) / temp
    logit = tf.clip_by_value(logit, -logit_clip, logit_clip)
    g_soft = tf.nn.sigmoid(logit)

    if mode == "soft":
        return g_soft

    # STE
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def indicator_default(
    V_tilde_norm: Numeric,
    temperature: float = 0.1,
    logit_clip: float = DEFAULT_LOGIT_CLIP,
    noise: bool = True,
    noise_seed: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Smooth default probability via Gumbel-Sigmoid.

    With noise=True (training):
        sigma( (-V/k + log(u) - log(1-u)) / tau )

    With noise=False (evaluation):
        sigma( -V/k / tau )

    Args:
        V_tilde_norm: Normalized latent value (V_tilde / k).
        temperature:  Temperature tau.
        logit_clip:   Clip range for the normalized value.
        noise:        If True, add Gumbel noise for exploration.
        noise_seed:   Optional stateless RNG seed [2] for reproducibility.
    """
    x = tf.convert_to_tensor(V_tilde_norm)
    dtype = x.dtype
    temp = tf.maximum(tf.cast(temperature, dtype), 1e-6)

    x_clipped = tf.clip_by_value(x, -logit_clip, logit_clip)

    if noise:
        if noise_seed is not None:
            u = tf.random.stateless_uniform(
                tf.shape(x), seed=tf.cast(noise_seed, tf.int32),
                minval=1e-6, maxval=1.0 - 1e-6, dtype=dtype)
        else:
            u = tf.random.uniform(
                tf.shape(x), minval=1e-6, maxval=1.0 - 1e-6, dtype=dtype)
        gumbel_noise = tf.math.log(u) - tf.math.log(1.0 - u)
        logit = (-x_clipped + gumbel_noise) / temp
    else:
        logit = -x_clipped / temp

    return tf.nn.sigmoid(logit)
