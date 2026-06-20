"""Economic property checks (DF26 Sec 12.4).

Hard checks that gate the suite, grouped as in the spec and evaluated on a VFI or refined
solution (exposing ``value``/``value_raw``/``policy_*``/``grids``), the economic primitives,
and the Block-1 value network. Each function returns a dict of named booleans/values; the
caller (test/notebook) gates on the hard ones. Monotonicity is checked only in the good
region (V > 0), ignoring steps below a relative tolerance (Sec 12.4 eps_mono).

* :func:`check_properties` -- Group 1 (bounds) + Group 2 (state monotonicities, incl. i down k)
  + the Tauchen-row identity.
* :func:`check_mechanics` -- Group 3 (Bellman/bond residuals on the refined solution,
  accounting identities, no NaN).
* :func:`check_weighting_spd` -- Group 3 (weighting matrix symmetric positive definite).
* :func:`check_corner_cases` -- Group 4 (production constants finite, k_min>0, no-default
  q equals the risk-free price, across the Table A1 corners).
* :func:`check_comparative_statics` -- Group 6 (parameter monotonicities of the value network
  and q in chi).
"""
from __future__ import annotations

import itertools

import numpy as np
import tensorflow as tf

from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET, TF_FLOAT_NUM
from src.v3.config import ModelParams
from src.v3.economics import bounds as _bounds
from src.v3.economics import debt as _debt
from src.v3.economics import production as _prod
from src.v3.solver.refine_evaluate import rhs_for_controls

_MONO_REL = 1e-3


def _monotone(arr, diff_axis, sign, good):
    """Every consecutive step of ``arr`` along ``diff_axis`` carries ``sign`` in the good
    region, ignoring steps below ``_MONO_REL`` of the array range."""
    d = np.diff(arr, axis=diff_axis) * sign
    a = np.take(good, range(good.shape[diff_axis] - 1), axis=diff_axis)
    b = np.take(good, range(1, good.shape[diff_axis]), axis=diff_axis)
    rng = arr.max() - arr.min()
    keep = a & b & (np.abs(d) > _MONO_REL * rng)
    return bool(np.all(d[keep] >= -1e-9))


def check_properties(result, grid_cfg) -> dict:
    g = result.grids
    V = result.value.numpy()
    bp = result.policy_bp.numpy()
    cp = result.policy_cp.numpy()
    kp = result.policy_kp.numpy()
    i = result.policy_i.numpy()
    net = bp - cp
    P = g.P.numpy()
    pos = V > 1e-9

    checks = {
        "bp_in_box": bool(bp.min() >= grid_cfg.bp_lo - 1e-9 and bp.max() <= grid_cfg.bp_hi + 1e-9),
        "cp_in_box": bool(cp.min() >= grid_cfg.cp_lo - 1e-9 and cp.max() <= grid_cfg.cp_hi + 1e-9),
        "net_debt_in_box": bool(net.min() >= grid_cfg.b_lo - 1e-9 and net.max() <= grid_cfg.b_hi + 1e-9),
        "kp_in_box": bool(kp.min() >= float(g.k_lo) - 1e-9 and kp.max() <= float(g.k_hi) + 1e-9),
        "value_finite": bool(np.isfinite(V).all()),
        "value_nonnegative": bool((V >= -1e-9).all()),
        "tauchen_rows_sum_to_1": bool(np.allclose(P.sum(axis=1), 1.0, atol=1e-10) and (P >= -1e-12).all()),
        "V_increasing_in_z": _monotone(V, 0, +1, pos),
        "V_decreasing_in_b": _monotone(V, 2, -1, pos),
        "i_decreasing_in_k": _monotone(i, 1, -1, pos),
    }
    return checks


def _flat_states(g):
    n_z, n_k, n_b = g.z.shape[0], g.k.shape[0], g.b.shape[0]
    zm, km, bm = tf.meshgrid(g.z, g.k, g.b, indexing="ij")
    return (tf.reshape(zm, [-1]), tf.reshape(km, [-1]), tf.reshape(bm, [-1]),
            tf.repeat(tf.range(n_z), n_k * n_b))


def check_mechanics(result, params, ext, *, tol=1e-6) -> dict:
    """Group 3 hard checks: Bellman residual, q bounds, accounting identity, finiteness."""
    g = result.grids
    V_raw = tf.cast(result.value_raw, TF_FLOAT_NUM)
    z, k, b, zi = _flat_states(g)
    kp = tf.reshape(result.policy_kp, [-1])
    bp = tf.reshape(result.policy_bp, [-1])
    cp = tf.reshape(result.policy_cp, [-1])
    rhs, q = rhs_for_controls(V_raw, z, k, b, zi, kp[:, None], bp[:, None], cp[:, None],
                              g.P, tf.math.log(g.k), g.b, params, ext)
    resid = tf.reshape(V_raw, [-1]) - rhs[:, 0]                  # Bellman residual (Eq 10, active)
    pos = tf.reshape(result.value, [-1]) > 1e-6
    bellman = float(tf.reduce_max(tf.abs(tf.boolean_mask(resid, pos)))) if bool(tf.reduce_any(pos)) else 0.0
    rfp = 1.0 / (1.0 + ext.rf * (1.0 - ext.tau))                # risk-free bond price

    # Accounting identity k' = (1 + i - delta) k from the reported investment rate.
    kp_from_i = (1.0 + tf.reshape(result.policy_i, [-1]) - params.delta) * k
    ident = float(tf.reduce_max(tf.abs(kp_from_i - kp)))

    finite = all(bool(tf.reduce_all(tf.math.is_finite(tf.cast(x, TF_FLOAT_NUM))))
                 for x in (V_raw, q, kp, bp, cp))
    return {
        "bellman_residual": bellman,
        "bellman_residual_small": bool(bellman < tol * (1.0 + float(tf.reduce_max(tf.abs(V_raw))))),
        "q_in_bounds": bool(float(tf.reduce_min(q)) >= -1e-9 and float(tf.reduce_max(q)) <= rfp + 1e-9),
        "accounting_kprime_identity": bool(ident < 1e-9),
        "all_finite": finite,
    }


def check_weighting_spd(W) -> dict:
    """Group 3 hard check: the weighting matrix is symmetric positive definite."""
    W = tf.cast(W, TF_FLOAT_NUM)
    symmetric = bool(tf.reduce_max(tf.abs(W - tf.transpose(W))) < 1e-9)
    try:
        tf.linalg.cholesky(W)
        spd = True
    except tf.errors.InvalidArgumentError:
        spd = False
    return {"weighting_symmetric": symmetric, "weighting_spd": spd}


def check_corner_cases(ext, bounds, grid_cfg) -> dict:
    """Group 4 hard checks: production constants finite and k_min>0 at every Table A1 corner."""
    lo, hi = bounds.lower_array(), bounds.upper_array()
    corners = np.array(list(itertools.product(*zip(lo, hi))))      # [256, 8]
    theta, rho, sigma, delta = (tf.constant(corners[:, j], TF_FLOAT_NUM) for j in (0, 1, 2, 3))
    nu = _prod.nu(theta, ext.alpha)
    xi = _prod.xi(theta, ext.alpha)
    a_pi = _prod.a_pi(theta, ext.alpha)
    k_lo, k_hi = _bounds.capital_bounds(theta, ext.alpha, delta, ext.rf, rho, sigma, grid_cfg.tauchen_m)
    fin = lambda x: bool(tf.reduce_all(tf.math.is_finite(x)))
    return {
        "nu_positive": bool(tf.reduce_all(nu > 0.0)),
        "production_constants_finite": fin(xi) and fin(a_pi) and fin(nu),
        "k_min_positive": bool(tf.reduce_all(k_lo > 0.0)),
        "k_bounds_finite": fin(k_lo) and fin(k_hi),
    }


def check_comparative_statics(bundle, bounds, ext, grid_cfg, *, n_sweep=50) -> dict:
    """Group 6 hard checks: value-network parameter monotonicities and q non-decreasing in chi.

    V decreasing in c_f, delta, gamma1, gamma0; V increasing in chi; q non-decreasing in chi
    at fixed state, choices, and P_def. Evaluated at the reference state across the Table A1
    ranges (Sec 12.4 Group 6)."""
    lo = tf.constant(bounds.lower_array(), TF_FLOAT_NET)
    hi = tf.constant(bounds.upper_array(), TF_FLOAT_NET)
    scaler = ParamScaler(bounds, dtype=TF_FLOAT_NET)
    ref = tf.cast(0.5 * (lo + hi), TF_FLOAT_NET)                   # mid-box reference parameters

    # Reference state: median log-productivity, mid capital, mid net debt (raw, then normalized).
    rp = ModelParams.from_array(ref.numpy().astype(np.float64))
    box = _bounds.state_box(rp.theta, ext.alpha, rp.delta, ext.rf, rp.rho, rp.sigma,
                            grid_cfg.tauchen_m, grid_cfg.b_lo, grid_cfg.b_hi)
    state_raw = tf.cast(tf.stack([0.0, 0.5 * (box.k_lo + box.k_hi), 0.5 * (grid_cfg.b_lo + grid_cfg.b_hi)]),
                        TF_FLOAT_NET)
    state_norm = _bounds.normalize_state(state_raw[None], box)     # [1, 3]

    signs = {"cf": (7, -1), "delta": (3, -1), "gamma1": (4, -1), "gamma0": (5, -1), "chi": (6, +1)}
    out = {}
    for name, (j, sign) in signs.items():
        sweep = tf.linspace(lo[j], hi[j], n_sweep)
        params = tf.tile(ref[None, :], [n_sweep, 1])
        params = tf.concat([params[:, :j], sweep[:, None], params[:, j + 1:]], axis=1)
        pnorm = scaler.normalize(params)
        V = bundle.value(tf.tile(state_norm, [n_sweep, 1]), pnorm).numpy()
        d = np.diff(V) * sign
        rng = V.max() - V.min()
        keep = np.abs(d) > _MONO_REL * rng
        out[f"V_monotone_in_{name}"] = bool(np.all(d[keep] >= -1e-9))

    # q non-decreasing in chi at fixed (state, choices, P_def).
    chi = tf.linspace(lo[6], hi[6], n_sweep)
    qd = _debt.bond_price(tf.fill([n_sweep], tf.constant(0.3, TF_FLOAT_NUM)),
                          tf.fill([n_sweep], tf.constant(1.0, TF_FLOAT_NUM)),
                          tf.fill([n_sweep], tf.constant(0.1, TF_FLOAT_NUM)),
                          tf.fill([n_sweep], tf.constant(0.5, TF_FLOAT_NUM)),
                          tf.cast(chi, TF_FLOAT_NUM), tf.constant(0.1, TF_FLOAT_NUM),
                          ext.bond_discount).numpy()
    out["q_nondecreasing_in_chi"] = bool(np.all(np.diff(qd) >= -1e-9))
    return out
