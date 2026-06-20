"""Minimum loss functions (DF26 Sec 6.1, Eqs 44/45).

For each parameter beta^j, the minimum loss function is the best achievable weighted-GMM
fit when beta^j is pinned at a value and all other parameters adjust optimally:

    L(beta^j) = min_{beta^{-j}} (target - g(beta^j, beta^{-j}))' W (target - g(...)).   (44)

Evaluated at ``n_loss_points`` evenly spaced beta^j across its current bounds; at each
point the inner min over beta^{-j} runs the surrogate LM (``n_restarts`` restarts) for
each of the K folds; we report the median across folds and the across-fold SD (after
recentering each fold by its own minimum, so level differences do not bias the SD).

The same routine serves the global identification diagnostic (Eq 45): pass the simulated
moments at beta-hat as ``target`` instead of the data moments.

The inner LM optimizes the 7-vector beta^{-j} in the same unconstrained sigmoid space as
:mod:`src.v3.estimation.estimate`, with beta^j inserted at its grid value per row, so the
analytic Jacobian flows through surrogate -> normalize -> reparam exactly as in estimation.
float64.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.estimation import reparam
from src.v3.estimation.lm import levenberg_marquardt


class LossProfile(NamedTuple):
    grid: np.ndarray      # [P, n_points] beta^j grid values per parameter
    L: np.ndarray         # [P, n_points] median-across-folds minimum loss
    L_sd: np.ndarray      # [P, n_points] across-fold SD (each fold recentered by its own min)
    L_folds: np.ndarray   # [P, F, n_points] per-fold minimum loss
    argmin: np.ndarray    # [P] beta^{j*} minimizing the median curve


def _insert_column(beta_other, value, j):
    """Insert ``value`` [B] at column ``j`` of ``beta_other`` [B, P-1] -> [B, P]."""
    cols = tf.unstack(beta_other, axis=1)
    cols.insert(j, value)
    return tf.stack(cols, axis=1)


def _profile_param(surrogate, scaler, target, W, lo, hi, j, grid_j, fold,
                   master_seed, n_restarts, max_iter):
    """Minimum loss over beta^{-j} at each grid value of beta^j, for one fold -> [n_points]."""
    dtype = TF_FLOAT_NUM
    P = lo.shape[0]
    other = [i for i in range(P) if i != j]
    lo_o = tf.gather(lo, other)
    hi_o = tf.gather(hi, other)
    n_points = grid_j.shape[0]
    v = tf.repeat(grid_j, n_restarts)                              # [n_points * n_restarts]
    x0 = seeding.normal([n_points * n_restarts, P - 1], master_seed,
                        seeding.Purpose.LM, j, fold, dtype=dtype)

    def predict(x):
        beta = _insert_column(reparam.to_constrained(x, lo_o, hi_o), v, j)   # [B, P]
        return surrogate.forward_fold(scaler.normalize(beta), fold)

    _, f = levenberg_marquardt(predict, x0, target, W, max_iter=max_iter)
    return tf.reduce_min(tf.reshape(f, [n_points, n_restarts]), axis=1)      # [n_points]


def minimum_loss_profile(surrogate, target_m, W, bounds, master_seed, ctrl,
                         *, target_label="data") -> LossProfile:
    """Compute the minimum loss function for every parameter (Eq 44, or Eq 45 if ``target_m``
    is the simulated moments at beta-hat). ``ctrl`` is a :class:`ControllerConfig`."""
    dtype = TF_FLOAT_NUM
    lo = tf.constant(bounds.lower_array(), dtype)
    hi = tf.constant(bounds.upper_array(), dtype)
    scaler = ParamScaler(bounds, dtype=dtype)
    target = tf.cast(target_m, dtype)
    W = tf.cast(W, dtype)
    P = lo.shape[0]
    F = min(ctrl.n_folds, surrogate.F)
    npts = ctrl.n_loss_points

    grids = np.zeros([P, npts])
    L_folds = np.zeros([P, F, npts])
    for j in range(P):
        grid_j = tf.linspace(lo[j], hi[j], npts)
        grids[j] = grid_j.numpy()
        for k in range(F):
            L_folds[j, k] = _profile_param(surrogate, scaler, target, W, lo, hi, j, grid_j, k,
                                           master_seed, ctrl.n_restarts, max_iter=20).numpy()

    L = np.median(L_folds, axis=1)                                # [P, npts] median over folds
    centered = L_folds - L_folds.min(axis=2, keepdims=True)        # recenter each fold by its min
    L_sd = centered.std(axis=1)                                    # [P, npts]
    argmin = grids[np.arange(P), L.argmin(axis=1)]                 # [P] beta^{j*}
    return LossProfile(grid=grids, L=L, L_sd=L_sd, L_folds=L_folds, argmin=argmin)
