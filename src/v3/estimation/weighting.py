"""Erickson-Whited influence-function weighting matrix (DF26 Sec 5.7).

W is the inverse of the moment variance-covariance matrix, estimated from the panel
with per-moment influence functions, clustered at the firm level (Eq 43). In the
synthetic regime the simulated panel IS the data, so the influence functions are
computed on it. All 11 moments use the same good-pair observations, so the effective
sample size N is common (Eq 43 uses 1/(N_j N_l) = 1/N^2). Returns (W, W_half) with
W_half the Cholesky factor normalized by its median column norm (Sec 5.7); the
normalization rescales the objective but not the argmin. float64.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.stats import median
from src.v3.simulation.moments import observations

_EPS = 1e-12


def _mean_psi(x):
    return x - tf.reduce_mean(x)


def _std_psi(x):
    mu = tf.reduce_mean(x)
    sd = tf.math.reduce_std(x)
    return ((x - mu) ** 2 - sd ** 2) / (2.0 * sd + _EPS)


def _autocorr_psi(x, y):
    """Influence function for the OLS slope of y on x (Sec 5.7 autocorrelation)."""
    xc = x - tf.reduce_mean(x)
    yc = y - tf.reduce_mean(y)
    rho = tf.reduce_sum(xc * yc) / (tf.reduce_sum(xc * xc) + _EPS)
    eps = yc - rho * xc
    Q = tf.reduce_mean(xc * xc)
    return xc * eps / (Q + _EPS)


def _reg_psi(y, x1, x2):
    """Influence functions for the two cash-saving regression coefficients: [N, 2].

    Ridged so a degenerate regressor (e.g. a near-constant debt/net measure on a
    coarse grid) does not make X'X singular.
    """
    X = tf.stack([x1 - tf.reduce_mean(x1), x2 - tf.reduce_mean(x2)], axis=1)  # [N, 2]
    yc = y - tf.reduce_mean(y)
    n = tf.cast(tf.shape(X)[0], X.dtype)
    XtX = tf.matmul(X, X, transpose_a=True)                    # [2, 2]
    ridge = (1e-10 + 1e-8 * tf.reduce_mean(tf.linalg.diag_part(XtX))) * tf.eye(2, dtype=X.dtype)
    XtX = XtX + ridge
    beta = tf.linalg.solve(XtX, tf.matmul(X, yc[:, None], transpose_a=True))[:, 0]
    eps = yc - tf.linalg.matvec(X, beta)
    return tf.einsum("jk,nk->nj", tf.linalg.inv(XtX / n), X) * eps[:, None]    # [N, 2]


def _moment_covariance(panel, jitter=1e-10):
    """Firm-clustered moment variance-covariance Sigma (Eq 43, common N). Returns [11, 11]."""
    o = observations(panel)
    inc = o["inc"]
    reg = _reg_psi(o["dcash"], o["net"], inc)
    Psi = tf.stack([
        _mean_psi(o["invrate"]), _std_psi(o["invrate"]),
        _mean_psi(inc), _std_psi(inc),
        _autocorr_psi(inc, o["inc_nxt"]),
        _mean_psi(o["debt"]), _std_psi(o["debt"]),
        _mean_psi(o["cash"]), _std_psi(o["cash"]),
        reg[:, 0], reg[:, 1],
    ], axis=1)                                                 # [N, 11]
    Psi = tf.where(tf.math.is_finite(Psi), Psi, tf.zeros_like(Psi))  # robust to degeneracy
    dtype = Psi.dtype
    n = tf.cast(tf.shape(Psi)[0], dtype)
    n_clusters = tf.reduce_max(o["firm"]) + 1
    cluster = tf.math.unsorted_segment_sum(Psi, o["firm"], n_clusters)  # [G, 11]
    Sigma = tf.matmul(cluster, cluster, transpose_a=True) / (n * n)     # Eq 43 (common N)
    return 0.5 * (Sigma + tf.transpose(Sigma)) + jitter * tf.eye(11, dtype=dtype)


def weighting_matrix(panel, jitter=1e-10):
    Sigma = _moment_covariance(panel, jitter)
    W = tf.linalg.inv(Sigma)
    W = 0.5 * (W + tf.transpose(W))
    w_half = tf.transpose(tf.linalg.cholesky(W))              # W = w_half' w_half
    med = median(tf.norm(w_half, axis=0))
    w_half = w_half / med
    return tf.matmul(w_half, w_half, transpose_a=True), w_half


def moment_se(panel):
    """Per-moment firm-clustered standard errors sqrt(diag(Sigma)) [11] (for confidence bands)."""
    return tf.sqrt(tf.maximum(tf.linalg.diag_part(_moment_covariance(panel)), 0.0))
