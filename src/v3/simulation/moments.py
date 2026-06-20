"""The 11 targeted moments from a simulated panel (DF26 Sec 4.3).

Consecutive period pairs (t, t+1) over the post-burn-in window; an observation is
"good" only if V > 0 in both periods (Sec 4.3 / Appendix A). Quantities (Eqs 34-39):
debt/cash/cash-saving use next-period values over next-period total assets K'+C';
investment, operating income, and net debt use current-period values. Moment 5 is
the OLS slope of inc' on inc; moments 10/11 are the OLS slopes of cash saving on
(net debt, operating income).

Synthetic regime: no firm fixed-effect removal (model firms are ex-ante homogeneous;
target and fitted moments use this same function). ``remove_firm_fe`` is a flag for
the deferred real-data path.

Returns an ``[11]`` float64 vector in the Sec 4.3 order.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM

_EPS = 1e-12


def _wmean(x):
    return tf.reduce_mean(x)


def _wstd(x):
    return tf.math.reduce_std(x)


def _ols_slope(x, y):
    """Univariate OLS slope of y on x (both centered)."""
    xc = x - tf.reduce_mean(x)
    yc = y - tf.reduce_mean(y)
    return tf.reduce_sum(xc * yc) / (tf.reduce_sum(xc * xc) + _EPS)


def _ols_multi(y, x1, x2):
    """Multivariate OLS slopes of y on [x1, x2] (centered). Returns (b1, b2)."""
    x1c = x1 - tf.reduce_mean(x1)
    x2c = x2 - tf.reduce_mean(x2)
    yc = y - tf.reduce_mean(y)
    s11 = tf.reduce_sum(x1c * x1c)
    s22 = tf.reduce_sum(x2c * x2c)
    s12 = tf.reduce_sum(x1c * x2c)
    s1y = tf.reduce_sum(x1c * yc)
    s2y = tf.reduce_sum(x2c * yc)
    det = s11 * s22 - s12 * s12
    b1 = (s22 * s1y - s12 * s2y) / (det + _EPS)
    b2 = (s11 * s2y - s12 * s1y) / (det + _EPS)
    return b1, b2


def observations(panel) -> dict:
    """Per-good-observation arrays for the 11 moments, plus the firm-cluster index.

    A good observation is a consecutive pair with V > 0 in both periods. Returns 1D
    arrays (invrate, inc, inc_nxt, debt, cash, dcash, net) and ``firm`` (the firm-slot
    index, for clustering the weighting matrix). Shared by :func:`compute_moments`
    and the weighting matrix.
    """
    dtype = TF_FLOAT_NUM
    bi, T = panel["burn_in"], panel["T"]
    A_pi = tf.constant(panel["A_pi"], dtype)
    xi = tf.constant(panel["xi"], dtype)
    cf = tf.constant(panel["cf"], dtype)
    delta = tf.constant(panel["delta"], dtype)
    n_firms = panel["k"].shape[0]

    def cur(a):
        return a[:, bi:T - 1]

    def nxt(a):
        return a[:, bi + 1:T]

    z, k, b_net, c, V = panel["z"], panel["k"], panel["b_net"], panel["c"], panel["V"]
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
    firm = tf.broadcast_to(tf.range(n_firms)[:, None], tf.shape(invrate))

    good = tf.reshape((cur(V) > 0.0) & (nxt(V) > 0.0), [-1])

    def keep(a):
        return tf.boolean_mask(tf.reshape(a, [-1]), good)

    return {
        "invrate": keep(invrate), "inc": keep(inc), "inc_nxt": keep(inc_nxt),
        "debt": keep(debt), "cash": keep(cash), "dcash": keep(dcash), "net": keep(net),
        "firm": tf.boolean_mask(tf.reshape(firm, [-1]), good),
    }


def compute_moments(panel, remove_firm_fe=False) -> tf.Tensor:
    o = observations(panel)
    invrate, inc, inc_nxt = o["invrate"], o["inc"], o["inc_nxt"]
    debt, cash, dcash, net = o["debt"], o["cash"], o["dcash"], o["net"]
    b_net_slope, inc_slope = _ols_multi(dcash, net, inc)
    return tf.stack([
        _wmean(invrate),      # 1 mean investment rate
        _wstd(invrate),       # 2 sd investment rate
        _wmean(inc),          # 3 mean operating income
        _wstd(inc),           # 4 sd operating income
        _ols_slope(inc, inc_nxt),  # 5 autocorr (OLS slope of inc' on inc)
        _wmean(debt),         # 6 mean debt ratio
        _wstd(debt),          # 7 sd debt ratio
        _wmean(cash),         # 8 mean cash ratio
        _wstd(cash),          # 9 sd cash ratio
        b_net_slope,          # 10 cash-saving coef on net debt
        inc_slope,            # 11 cash-saving coef on operating income
    ])
