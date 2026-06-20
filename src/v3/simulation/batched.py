"""Batched-over-parameters panel simulation + moments (DF26 Sec 4.2, 4.3).

The batched counterpart of :mod:`src.v3.simulation.panel` and
:mod:`src.v3.simulation.moments`: carries a leading ``[B, ...]`` parameter-batch
dimension so B firm panels (one per parameter vector) simulate together, then
computes the 11 moments per parameter as ``[B, 11]``. Mathematically identical to
looping the single-parameter functions (cross-checked in tests).

The good-observation mask varies in count across parameters, so the moments use
masked reductions (sum over good obs / count) rather than ragged ``boolean_mask``;
this is exactly the single-parameter formula with the mask carried explicitly.

Per-parameter RNG: each batch slot draws its own stateless stream keyed by its own
seed, so slot ``i`` reproduces the single simulation seeded with ``seeds[i]``.
float64.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.economics import production as _prod

_EPS = 1e-12
_P = seeding.Purpose


def _interp_firms_multi(grids, logk, b_grid, k, b, zi, n_z):
    """Interpolate several [B,n_z,n_k,n_b] grids at each firm's (k,b) and own z-node -> list of [B,n_f].

    The bilinear corner indices, weights, and the z-selection one-hot depend only on the firm
    state (k, b, zi), not on the grid values, so they are computed ONCE and reused across all
    grids (the four per-step policy/value interpolations share them). Bit-identical to calling
    the single-grid interpolation per grid, without the redundant index/weight/one-hot work."""
    B, _, n_k, n_b = grids[0].shape
    dtype = grids[0].dtype
    lk0, dlk = logk[:, :1], logk[:, 1:2] - logk[:, :1]
    fk = (tf.math.log(k) - lk0) / dlk
    lo_k = tf.clip_by_value(tf.cast(tf.floor(fk), tf.int32), 0, n_k - 2)
    wk = tf.clip_by_value(fk - tf.cast(lo_k, dtype), 0.0, 1.0)
    fb = (b - b_grid[0]) / (b_grid[1] - b_grid[0])
    lo_b = tf.clip_by_value(tf.cast(tf.floor(fb), tf.int32), 0, n_b - 2)
    wb = tf.clip_by_value(fb - tf.cast(lo_b, dtype), 0.0, 1.0)
    c00, c10 = lo_k * n_b + lo_b, (lo_k + 1) * n_b + lo_b
    c01, c11 = lo_k * n_b + (lo_b + 1), (lo_k + 1) * n_b + (lo_b + 1)
    w00 = ((1.0 - wk) * (1.0 - wb))[:, None, :]
    w10 = (wk * (1.0 - wb))[:, None, :]
    w01 = ((1.0 - wk) * wb)[:, None, :]
    w11 = (wk * wb)[:, None, :]
    oh = tf.one_hot(zi, n_z, dtype=dtype)                      # [B, n_f, n_z]
    out = []
    for vals in grids:
        vflat = tf.reshape(vals, [B, n_z, n_k * n_b])
        g = lambda c: tf.gather(vflat, c, axis=2, batch_dims=1)   # [B, n_z, n_f]
        allz = w00 * g(c00) + w10 * g(c10) + w01 * g(c01) + w11 * g(c11)
        out.append(tf.einsum("bfz,bzf->bf", oh, allz))
    return out


def _categorical(cdf, u):
    """Bin index of u along the last cdf axis. cdf [..., n], u [...] -> [...]."""
    return tf.reduce_sum(tf.cast(cdf < u[..., None], tf.int32), axis=-1)


def simulate_panel_batch(solution, beta_raw, ext, grid_cfg, seeds,
                         *, n_firms=2000, T=120, burn_in=60) -> dict:
    """Simulate B panels (one per ``beta_raw`` row) on the batched grid ``solution.grids``."""
    dtype = TF_FLOAT_NUM
    g = solution.grids
    B, n_z, n_k, n_b = beta_raw.shape[0], grid_cfg.n_z, grid_cfg.n_k, grid_cfg.n_b
    logk = tf.math.log(g.k)
    V_grid = tf.cast(solution.value_raw, dtype)
    i_grid = tf.cast(solution.policy_i, dtype)
    bp_grid = tf.cast(solution.policy_bp, dtype)
    cp_grid = tf.cast(solution.policy_cp, dtype)
    delta = tf.cast(beta_raw[:, 3], dtype)[:, None]

    seeds = [int(s) for s in seeds]

    def stack_b(fn):                                          # per-slot stream -> [B, ...]
        return tf.stack([fn(s) for s in seeds], axis=0)

    u0 = stack_b(lambda s: seeding.uniform([n_firms], s, _P.SIM_INIT, 0, dtype=dtype))
    ki0 = stack_b(lambda s: seeding.randint([n_firms], s, _P.SIM_INIT, 1, minval=0, maxval=n_k))
    bi0 = stack_b(lambda s: seeding.randint([n_firms], s, _P.SIM_INIT, 2, minval=0, maxval=n_b))
    u_step = stack_b(lambda s: tf.stack(
        [seeding.uniform([n_firms], s, _P.SIM_STEP, t, dtype=dtype) for t in range(T)], axis=1))
    ki_r = stack_b(lambda s: tf.stack(
        [seeding.randint([n_firms], s, _P.SIM_RESEED, t, 0, minval=0, maxval=n_k) for t in range(T)], axis=1))
    bi_r = stack_b(lambda s: tf.stack(
        [seeding.randint([n_firms], s, _P.SIM_RESEED, t, 1, minval=0, maxval=n_b) for t in range(T)], axis=1))

    stat_cdf = tf.cumsum(tf.cast(g.stationary, dtype), axis=-1)        # [B, n_z]
    zi = tf.clip_by_value(_categorical(stat_cdf[:, None, :], u0), 0, n_z - 1)   # [B, n_f]
    k = tf.gather(g.k, ki0, batch_dims=1)
    b_net = tf.gather(g.b, bi0)
    c = tf.zeros([B, n_firms], dtype)

    rec_z, rec_k, rec_b, rec_c, rec_V = [], [], [], [], []
    for t in range(T):
        z = tf.gather(g.z, zi, batch_dims=1)
        Vf, i_f, bp_f, cp_f = _interp_firms_multi(
            [V_grid, i_grid, bp_grid, cp_grid], logk, g.b, k, b_net, zi, n_z)

        rec_z.append(z); rec_k.append(k); rec_b.append(b_net); rec_c.append(c); rec_V.append(Vf)
        default = Vf < 0.0

        k_next = (1.0 + i_f - delta) * k
        bnet_next = bp_f - cp_f
        p_cdf = tf.cumsum(tf.gather(g.P, zi, batch_dims=1), axis=-1)   # [B, n_f, n_z]
        zi_step = tf.clip_by_value(_categorical(p_cdf, u_step[:, :, t]), 0, n_z - 1)

        k = tf.where(default, tf.gather(g.k, ki_r[:, :, t], batch_dims=1), k_next)
        b_net = tf.where(default, tf.gather(g.b, bi_r[:, :, t]), bnet_next)
        c = tf.where(default, tf.zeros_like(c), cp_f)
        zi = tf.where(default, zi, zi_step)

    st = lambda xs: tf.stack(xs, axis=2)                      # [B, n_f, T]
    theta, alpha = tf.cast(beta_raw[:, 0], dtype), ext.alpha
    return {
        "z": st(rec_z), "k": st(rec_k), "b_net": st(rec_b), "c": st(rec_c), "V": st(rec_V),
        "burn_in": burn_in, "T": T,
        "A_pi": _prod.a_pi(theta, alpha), "xi": _prod.xi(theta, alpha),
        "cf": tf.cast(beta_raw[:, 7], dtype), "delta": tf.cast(beta_raw[:, 3], dtype),
    }


# --- batched 11 moments (masked reductions over good observations) --------------

def _wmean(x, m, n):
    return tf.reduce_sum(x * m, axis=1) / n


def _wstd(x, m, n):
    mu = _wmean(x, m, n)
    var = _wmean(x * x, m, n) - mu * mu
    return tf.sqrt(tf.maximum(var, 0.0))


def _ols_slope(x, y, m, n):
    xc = x - _wmean(x, m, n)[:, None]
    yc = y - _wmean(y, m, n)[:, None]
    return tf.reduce_sum(m * xc * yc, axis=1) / (tf.reduce_sum(m * xc * xc, axis=1) + _EPS)


def _ols_multi(y, x1, x2, m, n):
    x1c = x1 - _wmean(x1, m, n)[:, None]
    x2c = x2 - _wmean(x2, m, n)[:, None]
    yc = y - _wmean(y, m, n)[:, None]
    s = lambda a, b: tf.reduce_sum(m * a * b, axis=1)
    s11, s22, s12 = s(x1c, x1c), s(x2c, x2c), s(x1c, x2c)
    s1y, s2y = s(x1c, yc), s(x2c, yc)
    det = s11 * s22 - s12 * s12 + _EPS
    return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det


def compute_moments_batch(panel) -> tf.Tensor:
    """The 11 moments per parameter -> ``[B, 11]`` (Sec 4.3 order)."""
    bi, T = panel["burn_in"], panel["T"]
    A_pi = panel["A_pi"][:, None, None]
    xi = panel["xi"][:, None, None]
    cf = panel["cf"][:, None, None]
    delta = panel["delta"][:, None, None]
    z, k, b_net, c, V = panel["z"], panel["k"], panel["b_net"], panel["c"], panel["V"]
    cur = lambda a: a[:, :, bi:T - 1]
    nxt = lambda a: a[:, :, bi + 1:T]

    K, C, Bn = cur(k), cur(c) * cur(k), cur(b_net) * cur(k)
    inc = (cur(z) * A_pi * K ** xi - cf) / (K + C + _EPS)
    net = Bn / (K + C + _EPS)
    Kp, Cp = nxt(k), nxt(c) * nxt(k)
    Bgp = (nxt(b_net) + nxt(c)) * nxt(k)
    inc_nxt = (nxt(z) * A_pi * Kp ** xi - cf) / (Kp + Cp + _EPS)
    invrate = (Kp - (1.0 - delta) * K) / (K + _EPS)
    debt = Bgp / (Kp + Cp + _EPS)
    cash = Cp / (Kp + Cp + _EPS)
    dcash = (Cp - C) / (Kp + Cp + _EPS)
    good = tf.cast((cur(V) > 0.0) & (nxt(V) > 0.0), V.dtype)

    B = K.shape[0]
    flat = lambda a: tf.reshape(a, [B, -1])
    invrate, inc, inc_nxt = flat(invrate), flat(inc), flat(inc_nxt)
    debt, cash, dcash, net, m = flat(debt), flat(cash), flat(dcash), flat(net), flat(good)
    n = tf.reduce_sum(m, axis=1) + _EPS
    b_net_slope, inc_slope = _ols_multi(dcash, net, inc, m, n)
    return tf.stack([
        _wmean(invrate, m, n), _wstd(invrate, m, n),
        _wmean(inc, m, n), _wstd(inc, m, n), _ols_slope(inc, inc_nxt, m, n),
        _wmean(debt, m, n), _wstd(debt, m, n),
        _wmean(cash, m, n), _wstd(cash, m, n),
        b_net_slope, inc_slope,
    ], axis=1)
