"""Grid-refinement policy improvement (DF26 Sec 4.1 step 2a).

Local search per state node over a candidate set built from the current policy, the
policy at the six immediate (z, k, b) neighbors, and +/-1 grid-index perturbations:
9 candidate indices per control dimension, hence 9^3 = 729 combinations. The network
starting point makes this local search sufficient (Sec 4.1). Exact dividend, current V.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.solver.refine_evaluate import rhs_for_controls


def candidate_indices(idx, n_grid):
    """Return ``[S, 9]`` candidate indices for one control dimension (clamped)."""
    a = idx
    z_up = tf.concat([a[1:], a[-1:]], axis=0)
    z_dn = tf.concat([a[:1], a[:-1]], axis=0)
    k_up = tf.concat([a[:, 1:], a[:, -1:]], axis=1)
    k_dn = tf.concat([a[:, :1], a[:, :-1]], axis=1)
    b_up = tf.concat([a[:, :, 1:], a[:, :, -1:]], axis=2)
    b_dn = tf.concat([a[:, :, :1], a[:, :, :-1]], axis=2)
    plus = a + 1
    minus = a - 1
    stack = tf.stack([a, z_up, z_dn, k_up, k_dn, b_up, b_dn, plus, minus], axis=0)  # [9, n_z, n_k, n_b]
    cand = tf.transpose(tf.reshape(stack, [9, -1]), [1, 0])  # [S, 9]
    return tf.clip_by_value(cand, 0, n_grid - 1)


def policy_improve(V, a_idx, c_idx, d_idx, z, k, b, zi, grids, logk, params, ext):
    """Return new (a, c, d) index grids maximizing the Bellman RHS locally."""
    n_kp, n_bp, n_cp = grids.kp.shape[0], grids.bp.shape[0], grids.cp.shape[0]
    a_c = candidate_indices(a_idx, n_kp)   # [S, 9]
    c_c = candidate_indices(c_idx, n_bp)
    d_c = candidate_indices(d_idx, n_cp)
    s = tf.shape(a_c)[0]

    kp_c = tf.gather(grids.kp, a_c)        # [S, 9]
    bp_c = tf.gather(grids.bp, c_c)
    cp_c = tf.gather(grids.cp, d_c)
    full = [s, 9, 9, 9]
    kp = tf.reshape(tf.broadcast_to(kp_c[:, :, None, None], full), [s, 729])
    bp = tf.reshape(tf.broadcast_to(bp_c[:, None, :, None], full), [s, 729])
    cp = tf.reshape(tf.broadcast_to(cp_c[:, None, None, :], full), [s, 729])

    rhs, _ = rhs_for_controls(V, z, k, b, zi, kp, bp, cp, grids.P, logk, grids.b, params, ext)
    best = tf.argmax(rhs, axis=1, output_type=tf.int32)  # [S], index in [0, 729)
    ia = best // 81
    ic = (best % 81) // 9
    idd = best % 9

    shape = V.shape
    new_a = tf.reshape(tf.gather(a_c, ia, batch_dims=1), shape)
    new_c = tf.reshape(tf.gather(c_c, ic, batch_dims=1), shape)
    new_d = tf.reshape(tf.gather(d_c, idd, batch_dims=1), shape)
    return new_a, new_c, new_d
