"""Tauchen discretization of the AR(1) log-productivity process (DF26 Sec 4.1).

Builds an ``n_z``-state grid and transition matrix from ``(rho, sigma)`` with
``m``-SD coverage, batched over a leading dimension of parameter vectors (Sec 11).
The conditional mean of log z is 0 (the process has zero unconditional mean), so
the grid is symmetric about 0 and matches the Sec 3.2 log z bounds.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM, as_float
from src.v3.common.stats import normal_cdf as _normal_cdf


def tauchen(rho, sigma, n_z, m=2.5, dtype=TF_FLOAT_NUM):
    """Return ``(logz_grid, P)``.

    ``logz_grid`` has shape ``[..., n_z]`` and ``P`` shape ``[..., n_z, n_z]`` with
    rows summing to 1. Leading ``...`` follows the broadcast of ``rho``/``sigma``.
    """
    rho = as_float(rho, dtype)
    sigma = as_float(sigma, dtype)
    sd_z = sigma / tf.sqrt(1.0 - rho * rho)
    hi = m * sd_z
    lo = -hi
    ramp = tf.linspace(tf.constant(0.0, dtype), tf.constant(1.0, dtype), n_z)  # [n_z]
    grid = lo[..., None] + (hi - lo)[..., None] * ramp  # [..., n_z]
    step = (hi - lo) / tf.cast(n_z - 1, dtype)          # [...]

    zi = grid[..., :, None]                             # current  [..., n_z, 1]
    zj = grid[..., None, :]                             # next     [..., 1, n_z]
    half = (step / 2.0)[..., None, None]
    rho_b = rho[..., None, None]
    sig_b = sigma[..., None, None]

    upper = _normal_cdf((zj - rho_b * zi + half) / sig_b)
    lower = _normal_cdf((zj - rho_b * zi - half) / sig_b)

    first = upper[..., :, :1]                            # P[:, 0]   = Phi(upper edge)
    last = 1.0 - lower[..., :, -1:]                      # P[:, n-1] = 1 - Phi(lower edge)
    middle = (upper - lower)[..., :, 1:-1]
    P = tf.concat([first, middle, last], axis=-1)
    return grid, P


def stationary_distribution(P, max_iter=5000, tol=1e-14):
    """Stationary distribution pi (pi P = pi) by power iteration; shape ``[..., n_z]``."""
    n = tf.shape(P)[-1]
    pi0 = tf.ones_like(P[..., 0, :]) / tf.cast(n, P.dtype)
    Pt = tf.linalg.matrix_transpose(P)

    def cond(i, pi, diff):
        return tf.logical_and(i < max_iter, diff > tf.cast(tol, P.dtype))

    def body(i, pi, diff):
        nxt = tf.linalg.matvec(Pt, pi)
        nxt = nxt / tf.reduce_sum(nxt, axis=-1, keepdims=True)
        return i + 1, nxt, tf.reduce_max(tf.abs(nxt - pi))

    _, pi, _ = tf.while_loop(
        cond, body,
        [tf.constant(0), pi0, tf.constant(1.0, P.dtype)],
    )
    return pi
