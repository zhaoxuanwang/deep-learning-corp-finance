"""Sigmoid bound reparametrization for the estimator (DF26 Eq 42).

LM searches over an unconstrained x in R^8; the parameter is
beta = lo + (hi - lo) * sigmoid(x), keeping it strictly inside [lo, hi]. The
derivative d beta/d x = (hi - lo) sigmoid(x)(1 - sigmoid(x)) is carried by autodiff
in the LM Jacobian, so callers usually just use :func:`to_constrained`.
"""
from __future__ import annotations

import tensorflow as tf

_EPS = 1e-7


def to_constrained(x, lo, hi):
    """Unconstrained x -> beta in (lo, hi) (Eq 42)."""
    return lo + (hi - lo) * tf.sigmoid(x)


def to_unconstrained(beta, lo, hi):
    """Inverse of :func:`to_constrained` (clamped off the open-interval edges)."""
    u = tf.clip_by_value((beta - lo) / (hi - lo), _EPS, 1.0 - _EPS)
    return tf.math.log(u / (1.0 - u))


def dconstrained_dx(x, lo, hi):
    """d beta/d x (diagonal); used only for checks - LM gets it via autodiff."""
    s = tf.sigmoid(x)
    return (hi - lo) * s * (1.0 - s)
