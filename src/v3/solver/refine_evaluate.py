"""Grid-refinement policy evaluation (DF26 Sec 4.1 step 2b).

Given a fixed policy, solve the joint value/bond-price fixed point on the grid with
the EXACT (non-smoothed) dividend and the Tauchen expectation. Two methods share the
``method`` seam: ``"dense"`` (the spec's semismooth-Newton active-set + dense
``(I - beta A) V = D`` linear solve, Sec 4.1 step 2b / Sec 11) is the default used by
:func:`src.v3.solver.refine.refine`; ``"value_iteration"`` (bond price recomputed each
sweep, the same joint iteration as the VFI benchmark) is kept as the cross-check oracle.
float64.

``rhs_for_controls`` is the shared Bellman right-hand side used by both policy
evaluation (one control per state) and policy improvement (many candidates per
state).
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.economics import debt as _debt
from src.v3.economics import dividends as _div
from src.v3.solver.interp import bilinear_corners, interp_grid


def dividend_kwargs(params, ext):
    return dict(theta=params.theta, alpha=ext.alpha, delta=params.delta,
                gamma1=params.gamma1, gamma0=params.gamma0, cf=params.cf,
                lambda0=ext.lambda0, lambda1=ext.lambda1, iota_c=ext.iota_c)


def rhs_for_controls(V, z, k, b, zi, kp, bp, cp, P, logk, b_grid, params, ext):
    """Bellman RHS for per-state controls (DF26 Eqs 6-10, exact).

    States ``z, k, b, zi`` are ``[S]``; controls ``kp, bp, cp`` are ``[S, M]``
    (M candidate controls per state). Returns ``(rhs, q)`` of shape ``[S, M]``,
    recomputing the exact bond price from the current V.
    """
    dtype = V.dtype
    z_b, k_b, b_b = z[:, None], k[:, None], b[:, None]
    bpp = bp - cp
    vprime = interp_grid(V, logk, b_grid, kp, bpp)            # [n_z', S, M]
    w = tf.maximum(vprime, 0.0)
    wdef = tf.cast(vprime < 0.0, dtype)
    w_t = tf.transpose(w, [1, 2, 0])                          # [S, M, n_z']
    wdef_t = tf.transpose(wdef, [1, 2, 0])
    p_rows = tf.gather(P, zi)[:, None, :]                     # [S, 1, n_z']
    cont = tf.reduce_sum(p_rows * w_t, axis=-1)              # [S, M]
    pdef = tf.reduce_sum(p_rows * wdef_t, axis=-1)           # [S, M]

    q = _debt.bond_price(pdef, kp, cp, bp,
                         tf.constant(params.chi, dtype), tf.constant(params.delta, dtype),
                         tf.constant(ext.bond_discount, dtype))
    i_rate = kp / k_b - (1.0 - params.delta)
    D = _div.dividend(z_b, k_b, b_b, i_rate, bp, cp, q,
                      **dividend_kwargs(params, ext), smooth=False).D
    rhs = D + ext.discount * cont
    return rhs, q


def policy_evaluate(V_init, z, k, b, zi, kp, bp, cp, P, logk, b_grid, params, ext,
                    *, method="value_iteration", tol=1e-9, max_iter=3000):
    """Solve V under a fixed policy (kp, bp, cp per state). Returns (V, n_iter).

    ``method="value_iteration"`` (the cross-check oracle) iterates the Bellman
    operator with the bond price recomputed each sweep. ``method="dense"`` is the
    spec method (Sec 4.1 step 2b, Sec 11): a semismooth-Newton active-set loop with
    a dense ``(I - beta A) V = D`` linear solve, the bond price held fixed during the
    solve (recomputed between rounds by the caller).
    """
    if method == "dense":
        return _policy_evaluate_dense(V_init, z, k, b, zi, kp, bp, cp, P, logk, b_grid,
                                      params, ext, tol=tol, max_iter=max_iter)
    if method != "value_iteration":
        raise ValueError(f"unknown method {method!r}")
    shape = V_init.shape
    kp1, bp1, cp1 = kp[:, None], bp[:, None], cp[:, None]    # [S, 1]
    V = V_init
    it = 0
    for it in range(1, max_iter + 1):
        rhs, _ = rhs_for_controls(V, z, k, b, zi, kp1, bp1, cp1, P, logk, b_grid, params, ext)
        v_new = tf.reshape(tf.maximum(rhs[:, 0], 0.0), shape)
        change = float(tf.reduce_max(tf.abs(v_new - V)))
        V = v_new
        if change < tol:
            break
    return V, it


def _vtilde(V, corners, weights):
    """Interpolated next value V(z', k'(s), bpp(s)) for every z' and state: [n_z, S]."""
    n_z = V.shape[0]
    v_flat = tf.reshape(V, [n_z, -1])
    vg = tf.gather(v_flat, corners, axis=1)              # [n_z, S, 4]
    return tf.reduce_sum(vg * weights[None, :, :], axis=-1)  # [n_z, S]


def _build_M(active, corners, weights, P, zi, beta, n_kb):
    """Assemble M = I - beta A for the active-set-linearized Bellman (Sec 11)."""
    n_z = P.shape[0]
    S = active.shape[1]
    dtype = weights.dtype
    a_T = tf.transpose(active)                           # [S, n_z]
    p_rows = tf.gather(P, zi)                            # [S, n_z]
    val = p_rows[:, :, None] * a_T[:, :, None] * weights[:, None, :]   # [S, n_z, 4]
    zp = tf.reshape(tf.range(n_z), [1, n_z, 1])
    col = zp * n_kb + corners[:, None, :]               # [S, n_z, 4]
    s_idx = tf.broadcast_to(tf.reshape(tf.range(S), [S, 1, 1]), [S, n_z, 4])
    indices = tf.stack([tf.reshape(s_idx, [-1]), tf.reshape(col, [-1])], axis=-1)
    A = tf.scatter_nd(indices, tf.reshape(val, [-1]), [S, S])
    return tf.eye(S, dtype=dtype) - beta * A


def _policy_evaluate_dense(V_init, z, k, b, zi, kp, bp, cp, P, logk, b_grid, params, ext,
                           *, tol=1e-10, max_iter=50):
    dtype = V_init.dtype
    shape = V_init.shape
    n_kb = shape[1] * shape[2]
    beta = tf.constant(ext.discount, dtype)
    bpp = bp - cp
    corners, weights = bilinear_corners(logk, b_grid, shape[1], shape[2], kp, bpp)

    # Bond price (hence dividend) held fixed during the solve, from the entry V
    # (Sec 4.1: q recomputed between rounds, not inside step 2b -> keeps it linear).
    vt0 = _vtilde(V_init, corners, weights)             # [n_z, S]
    pdef0 = tf.reduce_sum(tf.gather(P, zi) * tf.transpose(tf.cast(vt0 < 0.0, dtype)), axis=-1)
    q = _debt.bond_price(pdef0, kp, cp, bp, tf.constant(params.chi, dtype),
                         tf.constant(params.delta, dtype), tf.constant(ext.bond_discount, dtype))
    D = _div.dividend(z, k, b, kp / k - (1.0 - params.delta), bp, cp, q,
                      **dividend_kwargs(params, ext), smooth=False).D   # [S]

    V = V_init
    active = tf.cast(_vtilde(V, corners, weights) > 0.0, dtype)         # [n_z, S]
    it = 0
    for it in range(1, max_iter + 1):
        M = _build_M(active, corners, weights, P, zi, beta, n_kb)
        v_flat = tf.linalg.solve(M, D[:, None])[:, 0]                   # (I - beta A) V = D
        v_new = tf.reshape(v_flat, shape)
        new_active = tf.cast(_vtilde(v_new, corners, weights) > 0.0, dtype)
        rel = float(tf.reduce_max(tf.abs(v_new - V)) / (1.0 + tf.reduce_max(tf.abs(V))))
        stable = bool(tf.reduce_all(new_active == active))
        V, active = v_new, new_active
        if stable or rel < tol:
            break
    return V, it
