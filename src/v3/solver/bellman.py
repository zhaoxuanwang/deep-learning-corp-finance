"""Bellman residual and right-hand side for Block-1 training (DF26 Sec 3.4).

The continuation E_{z'|z}[max(V', 0)] uses Gauss-Hermite quadrature with the CURRENT
value network; the bond price q uses the TARGET network through the analytic
default probability (Sec 3.5). The dividend is the smoothed form (Sec 3.7). The
next-state V' is evaluated by the network directly (no interpolation) and normalized
with the same parameter vector's state box.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.v3.common import quadrature
from src.v3.common.normalization import to_unit
from src.v3.economics import bounds as _bounds
from src.v3.economics import debt as _debt
from src.v3.economics import dividends as _div
from src.v3.solver import default_prob as _dp

_SQRT2 = np.sqrt(2.0)


def _state_lo_hi(box: _bounds.StateBox):
    lo = tf.stack([box.logz_lo, box.k_lo, box.b_lo], axis=-1)   # [N, 3]
    hi = tf.stack([box.logz_hi, box.k_hi, box.b_hi], axis=-1)
    return lo, hi


def policy_controls(bundle, batch, ext, grid_cfg):
    """Investment rate, gross debt, and cash from the policy networks (Eq 26)."""
    k = batch.state_raw[:, 1]
    delta = batch.param_raw[:, 3]
    i_lo, i_hi = _bounds.investment_rate_bounds(k, batch.box.k_lo, batch.box.k_hi, delta)
    i = bundle.policy_i(batch.state_norm, batch.param_norm, i_lo, i_hi)
    bp = bundle.policy_bp(batch.state_norm, batch.param_norm, grid_cfg.bp_lo, grid_cfg.bp_hi)
    cp = bundle.policy_cp(batch.state_norm, batch.param_norm, grid_cfg.cp_lo, grid_cfg.cp_hi)
    return i, bp, cp


def continuation(value_net, logz, kprime, bpp, param_norm, rho, sigma, box, x_nodes, w_nodes):
    """E_{z'|z}[max(V(z', k', b'-c'), 0)] via Gauss-Hermite quadrature (Eq 30)."""
    n = tf.shape(logz)[0]
    q = x_nodes.shape[0]
    logz_p = (rho * logz)[:, None] + sigma[:, None] * (_SQRT2 * x_nodes)   # [N, Q]
    ones_q = tf.ones_like(logz_p)
    state_next = tf.stack([logz_p, kprime[:, None] * ones_q, bpp[:, None] * ones_q], axis=-1)  # [N, Q, 3]
    lo, hi = _state_lo_hi(box)
    sn = to_unit(state_next, lo[:, None, :], hi[:, None, :])              # [N, Q, 3]
    sn_flat = tf.reshape(sn, [n * q, 3])
    pn_flat = tf.repeat(param_norm, q, axis=0)
    vprime = tf.reshape(value_net(sn_flat, pn_flat), [n, q])
    return quadrature.integrate(tf.maximum(vprime, 0.0), w_nodes)         # [N]


def bond_price_training(target_net, logz, kprime, bpp, cp, bp, param_norm,
                        rho, sigma, box, chi, delta, bond_discount):
    """Bond price using the target network's Taylor default probability (Sec 3.5)."""
    lo, hi = _state_lo_hi(box)

    def normalize_next(lz, kp, bb):
        return to_unit(tf.stack([lz, kp, bb], axis=-1), lo, hi)

    v0, a, c = _dp.taylor_coeffs(target_net, logz, kprime, bpp, param_norm, rho, sigma, normalize_next)
    pdef = _dp.default_probability(v0, a, c)
    q = _debt.bond_price(pdef, kprime, cp, bp, chi, delta, bond_discount)
    return q, pdef


def residual_and_rhs(bundle, batch, ext, grid_cfg, x_nodes, w_nodes, tau):
    """Return (Bellman residual, RHS, V) for the batch (Eqs 27-29)."""
    logz = batch.state_raw[:, 0]
    k = batch.state_raw[:, 1]
    b = batch.state_raw[:, 2]
    z = tf.exp(logz)
    theta, rho, sigma, delta = (batch.param_raw[:, 0], batch.param_raw[:, 1],
                                batch.param_raw[:, 2], batch.param_raw[:, 3])
    gamma1, gamma0, chi, cf = (batch.param_raw[:, 4], batch.param_raw[:, 5],
                               batch.param_raw[:, 6], batch.param_raw[:, 7])

    i, bp, cp = policy_controls(bundle, batch, ext, grid_cfg)
    kprime = (1.0 + i - delta) * k
    bpp = bp - cp

    q, _ = bond_price_training(bundle.target, logz, kprime, bpp, cp, bp, batch.param_norm,
                               rho, sigma, batch.box, chi, delta, ext.bond_discount)

    D = _div.dividend(z, k, b, i, bp, cp, q,
                      theta=theta, alpha=ext.alpha, delta=delta, gamma1=gamma1,
                      gamma0=gamma0, cf=cf, lambda0=ext.lambda0, lambda1=ext.lambda1,
                      iota_c=ext.iota_c, smooth=True, tau=tau).D

    cont = continuation(bundle.value, logz, kprime, bpp, batch.param_norm,
                        rho, sigma, batch.box, x_nodes, w_nodes)
    rhs = D + ext.discount * cont
    V = bundle.value(batch.state_norm, batch.param_norm)
    return V - rhs, rhs, V
