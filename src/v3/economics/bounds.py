"""Parameter-dependent state and control bounds (DF26 Sec 3.2).

State bounds depend on the parameter vector and are recomputed per vector:

* log z in [-m sigma/sqrt(1-rho^2), +m sigma/sqrt(1-rho^2)], m = 2.5;
* k in [k_ss(z_min), k_ss(z_max)] (permanent-z steady state, Sec 3.2);
* net debt b in [-1, 2].

Control bounds map each control to its feasible interval (Eq 26 targets):

* investment rate i in [k_min/k - (1-delta), k_max/k - (1-delta)] (per current k);
* gross debt b' in [0, 2]; cash c' in [0, 1].

Functions follow the dtype of their inputs and broadcast over a leading batch of
parameter vectors.
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common.normalization import from_unit, to_unit
from src.v3.economics import production as _prod


class StateBox(NamedTuple):
    logz_lo: tf.Tensor
    logz_hi: tf.Tensor
    k_lo: tf.Tensor
    k_hi: tf.Tensor
    b_lo: tf.Tensor
    b_hi: tf.Tensor


def logz_bounds(rho, sigma, m):
    """Unconditional +/- m SD bounds of the AR(1) in logs (Sec 3.2)."""
    sd = sigma / tf.sqrt(1.0 - rho * rho)
    hi = m * sd
    return -hi, hi


def capital_bounds(theta, alpha, delta, rf, rho, sigma, m):
    """k in [k_ss(z_min), k_ss(z_max)] with z_min/max from the log z bounds."""
    lz_lo, lz_hi = logz_bounds(rho, sigma, m)
    k_lo = _prod.k_steady_state(tf.exp(lz_lo), theta, alpha, delta, rf)
    k_hi = _prod.k_steady_state(tf.exp(lz_hi), theta, alpha, delta, rf)
    return k_lo, k_hi


def state_box(theta, alpha, delta, rf, rho, sigma, m, b_lo=-1.0, b_hi=2.0) -> StateBox:
    """Full (log z, k, b) box for one (or a batch of) parameter vectors."""
    lz_lo, lz_hi = logz_bounds(rho, sigma, m)
    k_lo, k_hi = capital_bounds(theta, alpha, delta, rf, rho, sigma, m)
    b_lo_t = tf.cast(tf.fill(tf.shape(k_lo), b_lo), k_lo.dtype) if tf.is_tensor(k_lo) else b_lo
    b_hi_t = tf.cast(tf.fill(tf.shape(k_hi), b_hi), k_hi.dtype) if tf.is_tensor(k_hi) else b_hi
    return StateBox(lz_lo, lz_hi, k_lo, k_hi, b_lo_t, b_hi_t)


def investment_rate_bounds(k, k_lo, k_hi, delta):
    """i bounds so that k' = (1+i-delta)k stays in [k_lo, k_hi] (Sec 3.2)."""
    return k_lo / k - (1.0 - delta), k_hi / k - (1.0 - delta)


def normalize_state(state, box: StateBox):
    """Normalize a ``[..., 3]`` (log z, k, b) tensor to ``[-1, 1]`` using ``box``."""
    lo = tf.stack([box.logz_lo, box.k_lo, box.b_lo], axis=-1)
    hi = tf.stack([box.logz_hi, box.k_hi, box.b_hi], axis=-1)
    return to_unit(state, lo, hi)


def denormalize_state(unit_state, box: StateBox):
    """Inverse of :func:`normalize_state`."""
    lo = tf.stack([box.logz_lo, box.k_lo, box.b_lo], axis=-1)
    hi = tf.stack([box.logz_hi, box.k_hi, box.b_hi], axis=-1)
    return from_unit(unit_state, lo, hi)
