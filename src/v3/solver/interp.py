"""Bilinear interpolation of grid value/policy functions over (log k, b).

Used by the VFI benchmark, grid refinement, and simulation to evaluate an on-grid
function at off-grid ``(k, b)`` points (DF26 Sec 4.2, Sec 11). ``k`` is log-spaced,
so interpolation is linear in ``log k``; ``b`` is uniform. Queries are clamped to
the grid box (the bounds guarantee they lie inside).
"""
from __future__ import annotations

import tensorflow as tf


def _axis_weights(query, lo, step, n):
    """Lower index and weight for a regular (uniform) axis of length ``n``."""
    frac = (query - lo) / step
    idx_lo = tf.clip_by_value(tf.cast(tf.floor(frac), tf.int32), 0, n - 2)
    w = tf.clip_by_value(frac - tf.cast(idx_lo, query.dtype), 0.0, 1.0)
    return idx_lo, w


def interp_grid(values, logk_grid, b_grid, k_query, b_query):
    """Bilinearly interpolate ``values[n_z, n_k, n_b]`` at ``(k_query, b_query)``.

    ``logk_grid`` is ``log`` of the (log-spaced, hence uniform-in-log) capital grid;
    ``b_grid`` is the uniform net-debt grid. ``k_query``/``b_query`` share shape
    ``[...]``; the result is ``[n_z, ...]`` (one interpolated surface per z row).
    """
    n_z, n_k, n_b = values.shape
    dtype = values.dtype

    lk0 = logk_grid[0]
    dlk = logk_grid[1] - logk_grid[0]
    lo_k, wk = _axis_weights(tf.math.log(k_query), lk0, dlk, n_k)

    b0 = b_grid[0]
    db = b_grid[1] - b_grid[0]
    lo_b, wb = _axis_weights(b_query, b0, db, n_b)

    q_shape = tf.shape(k_query)
    lo_k_f = tf.reshape(lo_k, [-1])
    lo_b_f = tf.reshape(lo_b, [-1])
    wk_f = tf.reshape(wk, [-1])[None, :]
    wb_f = tf.reshape(wb, [-1])[None, :]

    values_flat = tf.reshape(values, [n_z, n_k * n_b])

    def corner(ki, bi):
        return tf.gather(values_flat, ki * n_b + bi, axis=1)  # [n_z, Q]

    v00 = corner(lo_k_f, lo_b_f)
    v10 = corner(lo_k_f + 1, lo_b_f)
    v01 = corner(lo_k_f, lo_b_f + 1)
    v11 = corner(lo_k_f + 1, lo_b_f + 1)

    out = ((1.0 - wk_f) * (1.0 - wb_f) * v00
           + wk_f * (1.0 - wb_f) * v10
           + (1.0 - wk_f) * wb_f * v01
           + wk_f * wb_f * v11)  # [n_z, Q]
    return tf.reshape(out, tf.concat([[n_z], q_shape], axis=0))


def bilinear_corners(logk_grid, b_grid, n_k, n_b, k_query, b_query):
    """Flat-(k,b) corner indices and bilinear weights for flattened queries.

    For each query (k, b) returns the four surrounding (k, b) grid corners as flat
    indices ``k_idx * n_b + b_idx`` (shape ``[Q, 4]``, order
    [(lo,lo),(hi,lo),(lo,hi),(hi,hi)]) and their bilinear weights (shape ``[Q, 4]``,
    over log k and b). Used to assemble the policy-evaluation linear operator so the
    same interpolation as :func:`interp_grid` enters the matrix.
    """
    lo_k, wk = _axis_weights(tf.math.log(k_query), logk_grid[0],
                             logk_grid[1] - logk_grid[0], n_k)
    lo_b, wb = _axis_weights(b_query, b_grid[0], b_grid[1] - b_grid[0], n_b)
    lo_k = tf.reshape(lo_k, [-1])
    lo_b = tf.reshape(lo_b, [-1])
    wk = tf.reshape(wk, [-1])
    wb = tf.reshape(wb, [-1])
    corners = tf.stack([
        lo_k * n_b + lo_b,
        (lo_k + 1) * n_b + lo_b,
        lo_k * n_b + (lo_b + 1),
        (lo_k + 1) * n_b + (lo_b + 1),
    ], axis=-1)  # [Q, 4]
    weights = tf.stack([
        (1.0 - wk) * (1.0 - wb),
        wk * (1.0 - wb),
        (1.0 - wk) * wb,
        wk * wb,
    ], axis=-1)  # [Q, 4]
    return corners, weights
