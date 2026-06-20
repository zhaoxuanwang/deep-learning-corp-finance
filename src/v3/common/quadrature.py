"""Gauss-Hermite quadrature for the conditional expectation (DF26 Sec 3.6).

The nodes/weights are computed once with NumPy (the one place NumPy is the right
tool, Sec 11) and converted to ``tf.constant``. With the substitution
``eps = sqrt(2) x`` the expectation against the standard normal is

    E_{z'|z}[f(z')] = (1/sqrt(pi)) sum_q w_q f(exp(rho log z + sigma sqrt(2) x_q)),

exact for polynomial integrands up to degree ``2Q-1`` (Eq 30).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM, as_float

_SQRT2 = np.sqrt(2.0)
_INV_SQRTPI = 1.0 / np.sqrt(np.pi)


def gauss_hermite(Q: int, dtype=TF_FLOAT_NUM):
    """Return ``(x_nodes[Q], w_nodes[Q])`` for the weight ``exp(-x^2)``."""
    x, w = np.polynomial.hermite.hermgauss(int(Q))
    return tf.constant(x, dtype=dtype), tf.constant(w, dtype=dtype)


def zprime_nodes(logz, rho, sigma, x_nodes):
    """Productivity at each quadrature node: ``z'_q = exp(rho log z + sigma sqrt(2) x_q)``.

    Broadcasts: ``logz`` of shape ``[...]`` (and scalar or ``[...]`` rho, sigma)
    with ``x_nodes`` of shape ``[Q]`` gives ``[..., Q]``.
    """
    dtype = x_nodes.dtype
    logz = as_float(logz, dtype)
    rho = as_float(rho, dtype)
    sigma = as_float(sigma, dtype)
    mean_term = (rho * logz)[..., None]                 # [..., 1]
    shock_term = sigma[..., None] * (_SQRT2 * x_nodes)  # [..., Q] or [Q]
    return tf.exp(mean_term + shock_term)


def integrate(f_nodes, w_nodes):
    """Combine node values ``f_nodes[..., Q]`` with weights into ``E[f]`` ([...])."""
    return _INV_SQRTPI * tf.reduce_sum(w_nodes * f_nodes, axis=-1)


def expectation(logz, rho, sigma, f, x_nodes, w_nodes):
    """``E_{z'|z}[f(z')]`` where ``f`` maps a ``[..., Q]`` tensor of z' to values."""
    zp = zprime_nodes(logz, rho, sigma, x_nodes)
    return integrate(f(zp), w_nodes)
