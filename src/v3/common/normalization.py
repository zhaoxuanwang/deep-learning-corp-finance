"""Min-max normalization to [-1, 1] (DF26 Eq 24, Sec 3.2).

Generic ``to_unit`` / ``from_unit`` plus :class:`ParamScaler` for the 8-dim
parameter vector. Parameter-dependent STATE bounds and their scaling live in
``economics/bounds.py`` (they are economic and depend on the production block).
Always store raw parameters + current bounds, never normalized values, so a bound
shrink is a single scaler swap (Sec 6.2).
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM, as_float
from src.v3.config import ParamBounds


def to_unit(x, lo, hi):
    """Map ``x`` from ``[lo, hi]`` to ``[-1, 1]`` (Eq 24)."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def from_unit(u, lo, hi):
    """Inverse of :func:`to_unit`: map ``u`` from ``[-1, 1]`` back to ``[lo, hi]``."""
    return lo + (u + 1.0) * 0.5 * (hi - lo)


class ParamScaler:
    """Normalizes/denormalizes the 8-dim parameter vector against current bounds."""

    def __init__(self, bounds: ParamBounds, dtype=TF_FLOAT_NUM):
        self.dtype = dtype
        self.lower = tf.constant(bounds.lower_array(), dtype=dtype)
        self.upper = tf.constant(bounds.upper_array(), dtype=dtype)

    def normalize(self, beta):
        """``beta[..., 8]`` (raw) -> ``[..., 8]`` in ``[-1, 1]``."""
        return to_unit(as_float(beta, self.dtype), self.lower, self.upper)

    def denormalize(self, u):
        """``u[..., 8]`` in ``[-1, 1]`` -> raw parameters ``[..., 8]``."""
        return from_unit(as_float(u, self.dtype), self.lower, self.upper)
