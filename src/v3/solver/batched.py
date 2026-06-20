"""Batched-over-parameters refinement collection (DF26 Sec 11: "vectorize over the
batch of parameter vectors processed together").

Carries a leading batch dimension ``[B, ...]`` through the per-parameter grids, the
network evaluation, the bilinear interpolation, and the dense policy-evaluation
solve, so B parameter vectors are refined at once. Mathematically identical to
looping the single-parameter ``solver.refine`` (cross-checked in tests); on a CUDA
GPU the batched linear solve and einsums give the throughput the serial collector
lacks. Device-agnostic: runs on whatever device ``precision.configure_devices`` left
visible. float64.

State/control grids: only z (Tauchen), k, k' depend on the parameters; the net-debt
b, gross-debt b', and cash c' grids are shared (their bounds are parameter-free).
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common.normalization import ParamScaler, to_unit
from src.v3.common.precision import TF_FLOAT_NET, TF_FLOAT_NUM
from src.v3.economics import bounds as _bounds
from src.v3.economics import debt as _debt
from src.v3.economics import dividends as _div
from src.v3.economics import tauchen as _tauchen


class BatchedGrids(NamedTuple):
    logz: tf.Tensor   # [B, n_z]
    z: tf.Tensor      # [B, n_z]
    k: tf.Tensor      # [B, n_k]
    b: tf.Tensor      # [n_b]      (shared)
    kp: tf.Tensor     # [B, n_kp]
    bp: tf.Tensor     # [n_bp]     (shared)
    cp: tf.Tensor     # [n_cp]     (shared)
    P: tf.Tensor      # [B, n_z, n_z]
    stationary: tf.Tensor  # [B, n_z]
    k_lo: tf.Tensor   # [B]
    k_hi: tf.Tensor   # [B]


def _log_grid_batch(lo, hi, n, dtype):
    ramp = tf.linspace(tf.constant(0.0, dtype), tf.constant(1.0, dtype), n)   # [n]
    return tf.exp(tf.math.log(lo)[:, None] + (tf.math.log(hi) - tf.math.log(lo))[:, None] * ramp)


def _uniform_grid(lo, hi, n, dtype):
    return tf.linspace(tf.constant(lo, dtype), tf.constant(hi, dtype), n)


def build_grids_batch(beta_raw, ext, grid_cfg, dtype=TF_FLOAT_NUM) -> BatchedGrids:
    """Per-parameter state/control grids for a batch ``beta_raw [B, 8]``."""
    theta, rho, sigma, delta = (beta_raw[:, 0], beta_raw[:, 1], beta_raw[:, 2], beta_raw[:, 3])
    logz, P = _tauchen.tauchen(rho, sigma, grid_cfg.n_z, grid_cfg.tauchen_m, dtype)
    k_lo, k_hi = _bounds.capital_bounds(theta, ext.alpha, delta, ext.rf, rho, sigma, grid_cfg.tauchen_m)
    return BatchedGrids(
        logz=logz, z=tf.exp(logz),
        k=_log_grid_batch(k_lo, k_hi, grid_cfg.n_k, dtype),
        b=_uniform_grid(grid_cfg.b_lo, grid_cfg.b_hi, grid_cfg.n_b, dtype),
        kp=_log_grid_batch(k_lo, k_hi, grid_cfg.n_kp, dtype),
        bp=_uniform_grid(grid_cfg.bp_lo, grid_cfg.bp_hi, grid_cfg.n_bp, dtype),
        cp=_uniform_grid(grid_cfg.cp_lo, grid_cfg.cp_hi, grid_cfg.n_cp, dtype),
        P=P, stationary=_tauchen.stationary_distribution(P), k_lo=k_lo, k_hi=k_hi,
    )


def interp_batch(values, logk, b_grid, k_query, b_query):
    """Bilinear interp with PER-BATCH grids.

    ``values [B, n_z, n_k, n_b]``; ``logk [B, n_k]`` (log of the per-beta k grid);
    ``b_grid [n_b]`` (shared); queries ``k_query, b_query [B, Q]``; returns
    ``[B, n_z, Q]``.
    """
    B, n_z, n_k, n_b = values.shape
    dtype = values.dtype
    lk0 = logk[:, :1]                                  # [B, 1]
    dlk = (logk[:, 1:2] - logk[:, :1])                 # [B, 1]
    fk = (tf.math.log(k_query) - lk0) / dlk            # [B, Q]
    lo_k = tf.clip_by_value(tf.cast(tf.floor(fk), tf.int32), 0, n_k - 2)
    wk = tf.clip_by_value(fk - tf.cast(lo_k, dtype), 0.0, 1.0)
    b0, db = b_grid[0], b_grid[1] - b_grid[0]
    fb = (b_query - b0) / db
    lo_b = tf.clip_by_value(tf.cast(tf.floor(fb), tf.int32), 0, n_b - 2)
    wb = tf.clip_by_value(fb - tf.cast(lo_b, dtype), 0.0, 1.0)

    vflat = tf.reshape(values, [B, n_z, n_k * n_b])    # [B, n_z, n_k*n_b]

    def corner(ki, bi):
        idx = ki * n_b + bi                            # [B, Q]
        return tf.gather(vflat, idx, axis=2, batch_dims=1)  # [B, n_z, Q]

    wk_, wb_ = wk[:, None, :], wb[:, None, :]
    return ((1 - wk_) * (1 - wb_) * corner(lo_k, lo_b)
            + wk_ * (1 - wb_) * corner(lo_k + 1, lo_b)
            + (1 - wk_) * wb_ * corner(lo_k, lo_b + 1)
            + wk_ * wb_ * corner(lo_k + 1, lo_b + 1))


def _bilinear_corners_batch(logk, b_grid, n_k, n_b, k_query, b_query):
    """Per-batch corner flat-(k,b) indices and weights. Queries/grids carry [B]."""
    dtype = k_query.dtype
    lk0, dlk = logk[:, :1], logk[:, 1:2] - logk[:, :1]
    fk = (tf.math.log(k_query) - lk0) / dlk
    lo_k = tf.clip_by_value(tf.cast(tf.floor(fk), tf.int32), 0, n_k - 2)
    wk = tf.clip_by_value(fk - tf.cast(lo_k, dtype), 0.0, 1.0)
    b0, db = b_grid[0], b_grid[1] - b_grid[0]
    fb = (b_query - b0) / db
    lo_b = tf.clip_by_value(tf.cast(tf.floor(fb), tf.int32), 0, n_b - 2)
    wb = tf.clip_by_value(fb - tf.cast(lo_b, dtype), 0.0, 1.0)
    corners = tf.stack([lo_k * n_b + lo_b, (lo_k + 1) * n_b + lo_b,
                        lo_k * n_b + (lo_b + 1), (lo_k + 1) * n_b + (lo_b + 1)], axis=-1)
    weights = tf.stack([(1 - wk) * (1 - wb), wk * (1 - wb),
                        (1 - wk) * wb, wk * wb], axis=-1)
    return corners, weights                              # [B, Q, 4], [B, Q, 4]



# --- batched network evaluation on the per-parameter grids ---------------------

class BatchedRefineResult(NamedTuple):
    value: tf.Tensor       # [B, n_z, n_k, n_b] equity value max(V, 0)
    value_raw: tf.Tensor   # [B, n_z, n_k, n_b] unclamped V
    policy_i: tf.Tensor
    policy_kp: tf.Tensor
    policy_bp: tf.Tensor
    policy_cp: tf.Tensor
    grids: BatchedGrids


def _state_box_batch(beta, ext, grid_cfg):
    return _bounds.state_box(beta[:, 0], ext.alpha, beta[:, 3], ext.rf, beta[:, 1], beta[:, 2],
                             grid_cfg.tauchen_m, grid_cfg.b_lo, grid_cfg.b_hi)


def network_on_grid_batch(bundle, beta_raw, ext, grid_cfg, bounds, grids):
    """Evaluate value + policies on each parameter's grid: returns [B, n_z, n_k, n_b]."""
    B, n_z, n_k, n_b = beta_raw.shape[0], grid_cfg.n_z, grid_cfg.n_k, grid_cfg.n_b
    S = n_z * n_k * n_b
    state = tf.stack([
        tf.broadcast_to(grids.logz[:, :, None, None], [B, n_z, n_k, n_b]),
        tf.broadcast_to(grids.k[:, None, :, None], [B, n_z, n_k, n_b]),
        tf.broadcast_to(grids.b[None, None, None, :], [B, n_z, n_k, n_b]),
    ], axis=-1)                                          # [B, n_z, n_k, n_b, 3]
    box = _state_box_batch(beta_raw, ext, grid_cfg)
    lo = tf.stack([box.logz_lo, box.k_lo, box.b_lo], axis=-1)[:, None, None, None, :]
    hi = tf.stack([box.logz_hi, box.k_hi, box.b_hi], axis=-1)[:, None, None, None, :]
    state_norm = tf.cast(to_unit(state, lo, hi), TF_FLOAT_NET)
    sflat = tf.reshape(state_norm, [B * S, 3])

    scaler = ParamScaler(bounds, dtype=TF_FLOAT_NET)
    pnorm = scaler.normalize(beta_raw)                   # [B, 8] (float32)
    pflat = tf.repeat(pnorm, S, axis=0)                  # [B*S, 8]

    delta = tf.cast(beta_raw[:, 3], TF_FLOAT_NET)
    k32 = tf.cast(grids.k, TF_FLOAT_NET)
    i_lo, i_hi = _bounds.investment_rate_bounds(
        k32[:, None, :, None], tf.cast(box.k_lo, TF_FLOAT_NET)[:, None, None, None],
        tf.cast(box.k_hi, TF_FLOAT_NET)[:, None, None, None], delta[:, None, None, None])
    i_lo = tf.reshape(tf.broadcast_to(i_lo, [B, n_z, n_k, n_b]), [B * S])
    i_hi = tf.reshape(tf.broadcast_to(i_hi, [B, n_z, n_k, n_b]), [B * S])

    def grid_of(x):
        return tf.reshape(tf.cast(x, TF_FLOAT_NUM), [B, n_z, n_k, n_b])

    value = grid_of(bundle.value(sflat, pflat))
    pi = grid_of(bundle.policy_i(sflat, pflat, i_lo, i_hi))
    pbp = grid_of(bundle.policy_bp(sflat, pflat, grid_cfg.bp_lo, grid_cfg.bp_hi))
    pcp = grid_of(bundle.policy_cp(sflat, pflat, grid_cfg.cp_lo, grid_cfg.cp_hi))
    return value, pi, pbp, pcp


# --- batched Bellman RHS / policy evaluation / improvement ---------------------

def _pkw(beta, ext, nd):
    col = lambda c: tf.reshape(beta[:, c], [-1] + [1] * (nd - 1))
    return dict(theta=col(0), alpha=ext.alpha, delta=col(3), gamma1=col(4), gamma0=col(5),
                cf=col(7), lambda0=ext.lambda0, lambda1=ext.lambda1, iota_c=ext.iota_c)


def _vtilde_batch(V, corners, weights):
    """Interp V at the (per-state) corners: V [B,n_z,n_k,n_b], corners/weights [B,S,4] -> [B,n_z,S]."""
    B, n_z = V.shape[0], V.shape[1]
    vflat = tf.reshape(V, [B, n_z, -1])
    vg = tf.gather(vflat, corners, axis=2, batch_dims=1)        # [B, n_z, S, 4]
    return tf.reduce_sum(vg * weights[:, None, :, :], axis=-1)  # [B, n_z, S]


def rhs_for_controls_batch(V, z, k, b, zi, kp, bp, cp, grids, beta, ext):
    """Bellman RHS for per-state candidate controls; controls [B, S, M] -> rhs [B, S, M]."""
    B, n_z, n_k, n_b = V.shape
    S, M = kp.shape[1], kp.shape[2]
    bpp = bp - cp
    Vp = interp_batch(V, tf.math.log(grids.k), grids.b,
                      tf.reshape(kp, [B, S * M]), tf.reshape(bpp, [B, S * M]))   # [B, n_z, S*M]
    Vp = tf.reshape(Vp, [B, n_z, S, M])
    w = tf.maximum(Vp, 0.0)
    wdef = tf.cast(Vp < 0.0, V.dtype)
    p_rows = tf.gather(grids.P, zi, batch_dims=1)               # [B, S, n_z]
    cont = tf.einsum("bsz,bzsm->bsm", p_rows, w)
    pdef = tf.einsum("bsz,bzsm->bsm", p_rows, wdef)
    chi, delta = beta[:, 6][:, None, None], beta[:, 3][:, None, None]
    q = _debt.bond_price(pdef, kp, cp, bp, chi, delta, ext.bond_discount)
    z3, k3, b3 = z[:, :, None], k[:, :, None], b[:, :, None]
    i_rate = kp / k3 - (1.0 - delta)
    D = _div.dividend(z3, k3, b3, i_rate, bp, cp, q, **_pkw(beta, ext, 3), smooth=False).D
    return D + ext.discount * cont, q


def _build_M_batch(active, corners, weights, P, zi, beta_disc, n_kb):
    """M = I - beta A for each batch element: active [B,n_z,S], returns [B, S, S]."""
    B, n_z, S = active.shape
    dtype = weights.dtype
    a_T = tf.transpose(active, [0, 2, 1])                       # [B, S, n_z]
    p_rows = tf.gather(P, zi, batch_dims=1)                     # [B, S, n_z]
    val = p_rows[:, :, :, None] * a_T[:, :, :, None] * weights[:, :, None, :]   # [B, S, n_z, 4]
    col = tf.reshape(tf.range(n_z), [1, 1, n_z, 1]) * n_kb + corners[:, :, None, :]
    bidx = tf.broadcast_to(tf.reshape(tf.range(B), [B, 1, 1, 1]), [B, S, n_z, 4])
    sidx = tf.broadcast_to(tf.reshape(tf.range(S), [1, S, 1, 1]), [B, S, n_z, 4])
    idx = tf.stack([tf.reshape(bidx, [-1]), tf.reshape(sidx, [-1]), tf.reshape(col, [-1])], axis=-1)
    A = tf.scatter_nd(idx, tf.reshape(val, [-1]), [B, S, S])
    return tf.eye(S, dtype=dtype)[None] - beta_disc * A


def policy_evaluate_dense_batch(V_init, z, k, b, zi, kp, bp, cp, grids, beta, ext,
                                *, tol=1e-10, max_iter=50):
    B, n_z, n_k, n_b = V_init.shape
    n_kb = n_k * n_b
    disc = tf.constant(ext.discount, V_init.dtype)
    corners, weights = _bilinear_corners_batch(tf.math.log(grids.k), grids.b, n_k, n_b, kp, bp - cp)
    p_rows = tf.gather(grids.P, zi, batch_dims=1)              # [B, S, n_z]
    chi, delta = beta[:, 6][:, None], beta[:, 3][:, None]
    pdef0 = tf.einsum("bsz,bzs->bs", p_rows, tf.cast(_vtilde_batch(V_init, corners, weights) < 0.0, V_init.dtype))
    q = _debt.bond_price(pdef0, kp, cp, bp, chi, delta, ext.bond_discount)
    D = _div.dividend(z, k, b, kp / k - (1.0 - delta), bp, cp, q, **_pkw(beta, ext, 2), smooth=False).D

    V = V_init
    active = tf.cast(_vtilde_batch(V, corners, weights) > 0.0, V.dtype)
    for _ in range(max_iter):
        M = _build_M_batch(active, corners, weights, grids.P, zi, disc, n_kb)
        v_flat = tf.linalg.solve(M, D[..., None])[..., 0]      # [B, S]
        v_new = tf.reshape(v_flat, [B, n_z, n_k, n_b])
        new_active = tf.cast(_vtilde_batch(v_new, corners, weights) > 0.0, V.dtype)
        stable = bool(tf.reduce_all(new_active == active))
        V, active = v_new, new_active
        if stable:
            break
    return V


def _candidate_indices_batch(idx, n_grid):
    """[B, n_z, n_k, n_b] index grid -> [B, S, 9] candidate indices (clamped)."""
    a = idx
    cands = tf.stack([
        a,
        tf.concat([a[:, 1:], a[:, -1:]], axis=1), tf.concat([a[:, :1], a[:, :-1]], axis=1),
        tf.concat([a[:, :, 1:], a[:, :, -1:]], axis=2), tf.concat([a[:, :, :1], a[:, :, :-1]], axis=2),
        tf.concat([a[:, :, :, 1:], a[:, :, :, -1:]], axis=3), tf.concat([a[:, :, :, :1], a[:, :, :, :-1]], axis=3),
        a + 1, a - 1,
    ], axis=1)                                                # [B, 9, n_z, n_k, n_b]
    B = tf.shape(a)[0]
    cands = tf.reshape(cands, [B, 9, -1])
    return tf.clip_by_value(tf.transpose(cands, [0, 2, 1]), 0, n_grid - 1)   # [B, S, 9]


def policy_improve_batch(V, a_idx, c_idx, d_idx, z, k, b, zi, grids, beta, ext):
    B, n_z, n_k, n_b = V.shape
    S = n_z * n_k * n_b
    n_kp, n_bp, n_cp = grids.kp.shape[1], grids.bp.shape[0], grids.cp.shape[0]
    a_c = _candidate_indices_batch(a_idx, n_kp)               # [B, S, 9]
    c_c = _candidate_indices_batch(c_idx, n_bp)
    d_c = _candidate_indices_batch(d_idx, n_cp)
    kp_c = tf.gather(grids.kp, a_c, axis=1, batch_dims=1)     # [B, S, 9]
    bp_c = tf.gather(grids.bp, c_c)                           # [B, S, 9] (shared grid)
    cp_c = tf.gather(grids.cp, d_c)
    full = [B, S, 9, 9, 9]
    kp = tf.reshape(tf.broadcast_to(kp_c[:, :, :, None, None], full), [B, S, 729])
    bp = tf.reshape(tf.broadcast_to(bp_c[:, :, None, :, None], full), [B, S, 729])
    cp = tf.reshape(tf.broadcast_to(cp_c[:, :, None, None, :], full), [B, S, 729])
    rhs, _ = rhs_for_controls_batch(V, z, k, b, zi, kp, bp, cp, grids, beta, ext)
    best = tf.argmax(rhs, axis=2, output_type=tf.int32)       # [B, S]
    ia, ic, idd = best // 81, (best % 81) // 9, best % 9
    shape = [B, n_z, n_k, n_b]
    new_a = tf.reshape(tf.gather(a_c, ia, axis=2, batch_dims=2), shape)
    new_c = tf.reshape(tf.gather(c_c, ic, axis=2, batch_dims=2), shape)
    new_d = tf.reshape(tf.gather(d_c, idd, axis=2, batch_dims=2), shape)
    return new_a, new_c, new_d


def _nearest_index_batch(grid_1d, values):
    """grid_1d [B, n] (or [n]); values [B, S] -> nearest index [B, S]."""
    if len(grid_1d.shape) == 1:
        diff = tf.abs(values[:, :, None] - grid_1d)
    else:
        diff = tf.abs(values[:, :, None] - grid_1d[:, None, :])
    return tf.cast(tf.argmin(diff, axis=-1), tf.int32)


def refine_batch(bundle, beta_raw, ext, grid_cfg, bounds, *, n_rounds=6,
                 eval_tol=1e-10, eval_max_iter=50) -> BatchedRefineResult:
    """Batched grid refinement over a batch of parameter vectors ``beta_raw [B, 8]``."""
    dtype = TF_FLOAT_NUM
    beta = tf.cast(beta_raw, dtype)
    g = build_grids_batch(beta, ext, grid_cfg, dtype)
    B, n_z, n_k, n_b = beta.shape[0], grid_cfg.n_z, grid_cfg.n_k, grid_cfg.n_b
    shape = [B, n_z, n_k, n_b]
    value, pi, pbp, pcp = network_on_grid_batch(bundle, beta, ext, grid_cfg, bounds, g)

    zf = tf.reshape(tf.broadcast_to(g.z[:, :, None, None], shape), [B, -1])
    kf = tf.reshape(tf.broadcast_to(g.k[:, None, :, None], shape), [B, -1])
    bf = tf.reshape(tf.broadcast_to(g.b[None, None, None, :], shape), [B, -1])
    zi = tf.broadcast_to(tf.repeat(tf.range(n_z), n_k * n_b)[None, :], [B, n_z * n_k * n_b])

    delta = beta[:, 3][:, None, None, None]
    kp_net = (1.0 + pi - delta) * g.k[:, None, :, None]
    a = tf.reshape(_nearest_index_batch(g.kp, tf.reshape(kp_net, [B, -1])), shape)
    c = tf.reshape(_nearest_index_batch(g.bp, tf.reshape(pbp, [B, -1])), shape)
    d = tf.reshape(_nearest_index_batch(g.cp, tf.reshape(pcp, [B, -1])), shape)
    V = tf.maximum(value, 0.0)

    kp = bp = cp = None
    for _ in range(n_rounds):
        a, c, d = policy_improve_batch(V, a, c, d, zf, kf, bf, zi, g, beta, ext)
        kp = tf.gather(g.kp, tf.reshape(a, [B, -1]), axis=1, batch_dims=1)
        bp = tf.gather(g.bp, tf.reshape(c, [B, -1]))
        cp = tf.gather(g.cp, tf.reshape(d, [B, -1]))
        V = policy_evaluate_dense_batch(V, zf, kf, bf, zi, kp, bp, cp, g, beta, ext,
                                        tol=eval_tol, max_iter=eval_max_iter)

    i_ref = tf.reshape(kp / kf - (1.0 - beta[:, 3][:, None]), shape)
    return BatchedRefineResult(
        value=tf.maximum(V, 0.0), value_raw=V, policy_i=i_ref,
        policy_kp=tf.reshape(kp, shape), policy_bp=tf.reshape(bp, shape),
        policy_cp=tf.reshape(cp, shape), grids=g)
