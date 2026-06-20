"""State and control grids for the discrete solvers (DF26 Sec 3.2, 4.1).

The capital grids (state ``k`` and control ``k'``) are log-spaced, denser at low
capital; net debt ``b`` and the controls ``b'``, ``c'`` are uniform; productivity
``z`` uses the Tauchen nodes (Appendix A). Grids span the parameter-dependent
state/control bounds of Sec 3.2.
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM, as_float  # noqa: F401
from src.v3.config import ExternalParams, GridConfig, ModelParams
from src.v3.economics import bounds as _bounds
from src.v3.economics import tauchen as _tauchen


def log_grid(lo, hi, n, dtype=TF_FLOAT_NUM):
    """Log-spaced (geometric) grid over ``[lo, hi]`` with ``lo > 0``."""
    return tf.exp(tf.linspace(tf.math.log(as_float(lo, dtype)),
                              tf.math.log(as_float(hi, dtype)), n))


def uniform_grid(lo, hi, n, dtype=TF_FLOAT_NUM):
    """Uniform grid over ``[lo, hi]``."""
    return tf.linspace(as_float(lo, dtype), as_float(hi, dtype), n)


class Grids(NamedTuple):
    logz: tf.Tensor   # [n_z] log-productivity nodes
    z: tf.Tensor      # [n_z] productivity nodes
    k: tf.Tensor      # [n_k] capital state grid (log-spaced)
    b: tf.Tensor      # [n_b] net-debt state grid (uniform)
    kp: tf.Tensor     # [n_kp] next-capital control grid (log-spaced)
    bp: tf.Tensor     # [n_bp] gross-debt control grid (uniform)
    cp: tf.Tensor     # [n_cp] cash control grid (uniform)
    P: tf.Tensor      # [n_z, n_z] Tauchen transition
    stationary: tf.Tensor  # [n_z]
    k_lo: tf.Tensor   # capital lower bound k_ss(z_min)
    k_hi: tf.Tensor   # capital upper bound k_ss(z_max)


def build_grids(params: ModelParams, ext: ExternalParams, g: GridConfig,
                dtype=TF_FLOAT_NUM) -> Grids:
    """Build all state/control grids and the Tauchen chain for one parameter vector."""
    # Cast parameters to the numeric dtype so the bounds avoid the tf.sqrt/tf.exp
    # float32 detour that a Python-float argument would trigger (matches the batched path).
    theta = as_float(params.theta, dtype)
    rho = as_float(params.rho, dtype)
    sigma = as_float(params.sigma, dtype)
    delta = as_float(params.delta, dtype)
    logz, P = _tauchen.tauchen(rho, sigma, g.n_z, g.tauchen_m, dtype)
    z = tf.exp(logz)
    k_lo, k_hi = _bounds.capital_bounds(theta, ext.alpha, delta, ext.rf, rho, sigma, g.tauchen_m)
    return Grids(
        logz=logz, z=z,
        k=log_grid(k_lo, k_hi, g.n_k, dtype),
        b=uniform_grid(g.b_lo, g.b_hi, g.n_b, dtype),
        kp=log_grid(k_lo, k_hi, g.n_kp, dtype),
        bp=uniform_grid(g.bp_lo, g.bp_hi, g.n_bp, dtype),
        cp=uniform_grid(g.cp_lo, g.cp_hi, g.n_cp, dtype),
        P=P, stationary=_tauchen.stationary_distribution(P),
        k_lo=k_lo, k_hi=k_hi,
    )
