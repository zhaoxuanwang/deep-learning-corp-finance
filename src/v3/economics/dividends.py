"""Two-subperiod dividends and net payout (DF26 Sec 1.6, with Sec 3.7 smoothing).

One implementation serves both the exact (grid/VFI) and smoothed (training) paths,
switched by ``smooth``. The smoothing (Eqs 31-33) replaces the step indicators, the
``max{d1,0}`` kink, and the ``min{d1,0}`` term with temperature-``tau`` softplus/
sigmoid surrogates; ``|x| = max(x,0) - min(x,0)`` follows from the two softplus
forms. As ``tau -> 0`` the smoothed dividend converges to the exact one.
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.economics import production as _prod


def _zeros_like(x):
    return tf.zeros_like(x)


def _step_neg(x, smooth, tau):
    """1{x < 0} (smoothed: sigmoid(-x/tau), Eq 31)."""
    if smooth:
        return tf.sigmoid(-x / tau)
    return tf.cast(x < 0, x.dtype)


def _step_pos(x, smooth, tau):
    """1{x > 0} (smoothed: sigmoid(x/tau), the complement form for 1{I>0})."""
    if smooth:
        return tf.sigmoid(x / tau)
    return tf.cast(x > 0, x.dtype)


def _pos(x, smooth, tau):
    """max(x, 0) (smoothed: tau softplus(x/tau), Eq 32)."""
    if smooth:
        return tau * tf.math.softplus(x / tau)
    return tf.maximum(x, _zeros_like(x))


def _neg(x, smooth, tau):
    """min(x, 0) (smoothed: -tau softplus(-x/tau), Eq 33)."""
    if smooth:
        return -tau * tf.math.softplus(-x / tau)
    return tf.minimum(x, _zeros_like(x))


class DividendParts(NamedTuple):
    D: tf.Tensor       # net payout to shareholders (Eq 9)
    d1: tf.Tensor      # preproduction cash flow (Eq 7)
    d2: tf.Tensor      # postproduction cash flow (Eq 8)
    kprime: tf.Tensor  # next-period capital
    invest: tf.Tensor  # investment level I = i k
    issue: tf.Tensor   # total equity-issuance cost


def dividend(
    z, k, b, i_rate, bp, cp, q,
    *, theta, alpha, delta, gamma1, gamma0, cf,
    lambda0, lambda1, iota_c,
    smooth: bool = False, tau: float = 1e-3,
) -> DividendParts:
    """Compute net payout D and its components for one (state, control, q).

    All arguments may be scalars or broadcastable tensors. ``q`` is the bond price
    for this choice (computed in :mod:`economics.debt`).
    """
    invest = i_rate * k
    kprime = (1.0 + i_rate - delta) * k
    gross = _prod.gross_output(z, k, theta, alpha)

    # Preproduction (Eq 7).
    d1 = q * bp * kprime - b * k - cf
    pos_d1 = _pos(d1, smooth, tau)
    neg_d1 = _neg(d1, smooth, tau)
    abs_d1 = pos_d1 - neg_d1
    ind_d1_neg = _step_neg(d1, smooth, tau)
    ind_i_pos = _step_pos(invest, smooth, tau)

    conv = 0.5 * gamma1 * invest * invest / k
    fixed = gamma0 * k * ind_i_pos

    # Postproduction (Eq 8): carries the positive preproduction excess.
    d2 = gross + pos_d1 - invest - conv - cp * kprime * iota_c - fixed
    pos_d2 = _pos(d2, smooth, tau)
    neg_d2 = _neg(d2, smooth, tau)
    abs_d2 = pos_d2 - neg_d2
    ind_d2_neg = _step_neg(d2, smooth, tau)

    issue1 = (lambda0 * k + lambda1 * abs_d1) * ind_d1_neg
    issue2 = (lambda0 * k + lambda1 * abs_d2) * ind_d2_neg

    # Net payout (Eq 9): d1*1{d1<0} = min(d1,0) = neg_d1.
    D = d2 + neg_d1 - issue1 - issue2
    return DividendParts(D=D, d1=d1, d2=d2, kprime=kprime, invest=invest, issue=issue1 + issue2)
