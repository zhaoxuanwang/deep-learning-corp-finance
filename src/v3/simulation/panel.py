"""Firm-panel simulation from a grid solution (DF26 Sec 4.2).

Simulates ``n_firms`` firms over ``T`` periods (discard ``burn_in``). Productivity
moves on the 11-state Tauchen chain (no z interpolation); the continuous ``(k, b)``
state interpolates the grid policies bilinearly. A firm defaults when its
(interpolated) value V < 0 (limited liability, Appendix A): it exits and is replaced
by a fresh ``(k, b)`` at random grid nodes, keeping the current z.

The net-debt state ``b`` does not pin down cash, but the moments need cash C; we
carry cash-per-capital ``c`` (= the previous period's c' control), with gross debt
per capital ``b_gross = b_net + c``. Initial/reseed cash is 0 (washed out by burn-in).

Stateless RNG keyed by period (Sec 10); reproducible for fixed ``n_firms``.
"""
from __future__ import annotations

from typing import Dict

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.economics import production as _prod
from src.v3.solver.interp import _axis_weights


def _categorical(probs_cdf, u):
    """Index of the bin u falls in, given a CDF along the last axis."""
    return tf.reduce_sum(tf.cast(probs_cdf < u[..., None], tf.int32), axis=-1)


def _interp_at_firms_multi(grids, logk, b_grid, k, b, firm_zi, n_firms):
    """Interpolate several [n_z, n_k, n_b] grids at each firm's (k, b) and own z-node -> list of
    [n_firms]. The bilinear indices/weights and the z-selection indices depend only on the firm
    state, so they are computed ONCE and reused across grids (the four per-step value/policy
    interpolations share them). Bit-identical to per-grid interp_grid + z-select."""
    n_z, n_k, n_b = grids[0].shape
    lo_k, wk = _axis_weights(tf.math.log(k), logk[0], logk[1] - logk[0], n_k)
    lo_b, wb = _axis_weights(b, b_grid[0], b_grid[1] - b_grid[0], n_b)
    wk_, wb_ = wk[None, :], wb[None, :]
    zsel = tf.stack([firm_zi, tf.range(n_firms)], axis=-1)
    out = []
    for vals in grids:
        vflat = tf.reshape(vals, [n_z, n_k * n_b])
        corner = lambda ki, bi: tf.gather(vflat, ki * n_b + bi, axis=1)   # [n_z, n_firms]
        allz = ((1.0 - wk_) * (1.0 - wb_) * corner(lo_k, lo_b)
                + wk_ * (1.0 - wb_) * corner(lo_k + 1, lo_b)
                + (1.0 - wk_) * wb_ * corner(lo_k, lo_b + 1)
                + wk_ * wb_ * corner(lo_k + 1, lo_b + 1))
        out.append(tf.gather_nd(allz, zsel))
    return out


def simulate_panel(solution, params, ext, grid_cfg, master_seed,
                   *, n_firms=5000, T=300, burn_in=200) -> Dict:
    dtype = TF_FLOAT_NUM
    g = solution.grids
    n_z, n_k, n_b = g.z.shape[0], g.k.shape[0], g.b.shape[0]
    logk = tf.math.log(g.k)
    V_grid = tf.cast(solution.value_raw, dtype)   # unclamped V; V < 0 => default (Sec 4.2)
    i_grid = tf.cast(solution.policy_i, dtype)
    bp_grid = tf.cast(solution.policy_bp, dtype)
    cp_grid = tf.cast(solution.policy_cp, dtype)
    delta = params.delta

    # Initial state: z ~ stationary; (k, b) at random grid nodes; cash 0.
    stat_cdf = tf.cumsum(tf.cast(g.stationary, dtype))
    u0 = seeding.uniform([n_firms], master_seed, seeding.Purpose.SIM_INIT, 0, dtype=dtype)
    zi = tf.clip_by_value(_categorical(stat_cdf, u0), 0, n_z - 1)
    ki = seeding.randint([n_firms], master_seed, seeding.Purpose.SIM_INIT, 1, minval=0, maxval=n_k)
    bi = seeding.randint([n_firms], master_seed, seeding.Purpose.SIM_INIT, 2, minval=0, maxval=n_b)
    k = tf.gather(g.k, ki)
    b_net = tf.gather(g.b, bi)
    c = tf.zeros([n_firms], dtype)

    rec_z, rec_k, rec_b, rec_c, rec_V = [], [], [], [], []
    for t in range(T):
        z = tf.gather(g.z, zi)
        Vf, i_f, bp_f, cp_f = _interp_at_firms_multi(
            [V_grid, i_grid, bp_grid, cp_grid], logk, g.b, k, b_net, zi, n_firms)

        rec_z.append(z); rec_k.append(k); rec_b.append(b_net); rec_c.append(c); rec_V.append(Vf)
        default = Vf < 0.0

        # Non-default transition.
        k_next = (1.0 + i_f - delta) * k
        bnet_next = bp_f - cp_f
        p_cdf = tf.cumsum(tf.gather(g.P, zi), axis=-1)
        u_step = seeding.uniform([n_firms], master_seed, seeding.Purpose.SIM_STEP, t, dtype=dtype)
        zi_step = tf.clip_by_value(_categorical(p_cdf, u_step), 0, n_z - 1)

        # Reseed on default: fresh (k, b) at random grid nodes, keep z.
        ki_r = seeding.randint([n_firms], master_seed, seeding.Purpose.SIM_RESEED, t, 0, minval=0, maxval=n_k)
        bi_r = seeding.randint([n_firms], master_seed, seeding.Purpose.SIM_RESEED, t, 1, minval=0, maxval=n_b)

        k = tf.where(default, tf.gather(g.k, ki_r), k_next)
        b_net = tf.where(default, tf.gather(g.b, bi_r), bnet_next)
        c = tf.where(default, tf.zeros_like(c), cp_f)
        zi = tf.where(default, zi, zi_step)

    stack = lambda xs: tf.stack(xs, axis=1)  # [n_firms, T]
    return {
        "z": stack(rec_z), "k": stack(rec_k), "b_net": stack(rec_b),
        "c": stack(rec_c), "V": stack(rec_V),
        "burn_in": burn_in, "T": T,
        "A_pi": float(_prod.a_pi(params.theta, ext.alpha)),
        "xi": float(_prod.xi(params.theta, ext.alpha)),
        "cf": float(params.cf), "delta": float(params.delta),
    }
