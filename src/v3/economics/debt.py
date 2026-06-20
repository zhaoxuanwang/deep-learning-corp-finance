"""Debt, default recovery, and bond pricing (DF26 Sec 1.5).

The default GATE ``g = 1{R < b'k'}`` depends only on controls/parameters, not on
z' (Sec 1.5). Given the default probability ``P_def = Pr(V' < 0)`` (computed by the
training-time Taylor step or exactly on the Tauchen grid elsewhere), the bond price
collapses to the closed form in Eq 6.
"""
from __future__ import annotations

import tensorflow as tf

_EPS = 1e-12


def recovery(kprime, cp, chi, delta):
    """R(k', c') = c' k' + chi (1 - delta) k' (Eq 5)."""
    return cp * kprime + chi * (1.0 - delta) * kprime


def face_value(kprime, bp):
    """Outstanding gross debt b' k'."""
    return bp * kprime


def gate(kprime, cp, bp, chi, delta):
    """Deterministic default gate g = 1{R < b'k'} (Sec 1.5).

    With b' >= 0 and R >= 0, g = 0 whenever there is no debt (b' = 0).
    """
    R = recovery(kprime, cp, chi, delta)
    return tf.cast(R < face_value(kprime, bp), R.dtype)


def recovery_ratio(kprime, cp, bp, chi, delta, eps=_EPS):
    """R / (b' k'), guarded against a zero face value (the term is gated by g)."""
    R = recovery(kprime, cp, chi, delta)
    face = face_value(kprime, bp)
    return R / tf.maximum(face, tf.cast(eps, R.dtype))


def bond_price(pdef, kprime, cp, bp, chi, delta, bond_discount, eps=_EPS):
    """Bond price q (Eq 6, collapsed form):

    q = bond_discount * [(1 - g P_def) + g P_def R/(b'k')],

    with g the deterministic gate. q lies in [0, bond_discount]; it equals the
    risk-free price bond_discount when g = 0, and has recovery below 1 when g = 1.
    """
    g = gate(kprime, cp, bp, chi, delta)
    ratio = recovery_ratio(kprime, cp, bp, chi, delta, eps)
    return bond_discount * ((1.0 - g * pdef) + g * pdef * ratio)
