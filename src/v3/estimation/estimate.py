"""Estimation driver (DF26 Sec 5.4-5.6).

For each CV fold, run batched LM from ``n_restarts`` standard-normal starts in the
unconstrained x-space, keep the lowest-objective restart's beta; the reported
beta_hat is the median across the 10 folds (Sec 6.1 convention). Returns the per-fold
estimates and all restart estimates (for the controller's containment guard). float64.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.common.stats import median
from src.v3.estimation import reparam
from src.v3.estimation.lm import levenberg_marquardt


def estimate(surrogate, target_m, W, bounds, master_seed, *, n_restarts=30, max_iter=20):
    dtype = TF_FLOAT_NUM
    lo = tf.constant(bounds.lower_array(), dtype)
    hi = tf.constant(bounds.upper_array(), dtype)
    scaler = ParamScaler(bounds, dtype=dtype)
    target = tf.cast(target_m, dtype)
    W = tf.cast(W, dtype)

    def make_predict(fold):
        def predict(x):
            return surrogate.forward_fold(scaler.normalize(reparam.to_constrained(x, lo, hi)), fold)
        return predict

    fold_best, all_betas = [], []
    for k in range(surrogate.F):
        x0 = seeding.normal([n_restarts, 8], master_seed, seeding.Purpose.LM, k, dtype=dtype)
        x_k, f_k = levenberg_marquardt(make_predict(k), x0, target, W, max_iter=max_iter)
        betas_k = reparam.to_constrained(x_k, lo, hi)          # [n_restarts, 8]
        fold_best.append(betas_k[int(tf.argmin(f_k))])
        all_betas.append(betas_k)

    fold_best = tf.stack(fold_best)                            # [F, 8]
    return {
        "beta_hat": median(fold_best, axis=0),                       # median across folds
        "fold_betas": fold_best,
        "all_betas": tf.stack(all_betas),                     # [F, n_restarts, 8]
    }
