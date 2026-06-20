"""Native batched Levenberg-Marquardt (DF26 Sec 5.5, Sec 11).

TFP has no LM, so this implements it directly. Minimizes, for each problem b in a
batch, the weighted-GMM objective r' W r with r = target - predict(x_b), using the
analytic Jacobian J = d predict/d x from ``GradientTape.batch_jacobian``:

    (J' W J + mu I) delta = J' W r,   x <- x + delta

with a trust-region update on mu (accept and shrink mu if the objective drops; else
reject and grow mu). Batched over the leading dimension (e.g. 30 restarts), so the
caller runs one LM per fold over its restarts. float64.
"""
from __future__ import annotations

import tensorflow as tf


def _objective(r, W):
    """r' W r per problem: r [B, M], W [M, M] -> [B]."""
    return tf.reduce_sum(r * tf.matmul(r, W, transpose_b=True), axis=1)


def levenberg_marquardt(predict, x0, target, W, *, max_iter=20, tol=1e-10,
                        mu0=1e-3, mu_up=2.0, mu_down=0.5, mu_min=1e-8):
    """Run batched LM. ``predict``: [B,P]->[B,M]; returns (x[B,P], objective[B]).

    ``mu`` is floored at ``mu_min`` so the damped normal-equation matrix
    (J'WJ + mu I) stays positive definite even when J'WJ is rank-deficient in a
    weakly-identified direction (e.g. the recovery rate chi).
    """
    dtype = x0.dtype
    target = tf.cast(target, dtype)
    W = tf.cast(W, dtype)
    P = x0.shape[1]
    eye = tf.eye(P, dtype=dtype)
    x = x0
    mu = tf.fill([tf.shape(x0)[0]], tf.constant(mu0, dtype))

    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(x)
            m = predict(x)
        J = tape.batch_jacobian(m, x)                       # [B, M, P]
        r = target[None, :] - m                             # [B, M]
        f = _objective(r, W)
        Wr = tf.matmul(r, W, transpose_b=True)              # [B, M]
        WJ = tf.einsum("ij,bjp->bip", W, J)                 # [B, M, P]
        JtWJ = tf.einsum("bmp,bmq->bpq", J, WJ)             # [B, P, P]
        JtWr = tf.einsum("bmp,bm->bp", J, Wr)               # [B, P]
        damp = tf.maximum(mu, mu_min)[:, None, None]
        delta = tf.linalg.solve(JtWJ + damp * eye, JtWr[..., None])[..., 0]

        x_new = x + delta
        f_new = _objective(target[None, :] - predict(x_new), W)
        accept = f_new < f
        x = tf.where(accept[:, None], x_new, x)
        mu = tf.maximum(tf.where(accept, mu * mu_down, mu * mu_up), mu_min)
        if bool(tf.reduce_all(tf.reduce_max(tf.abs(delta), axis=1) < tol)):
            break

    return x, _objective(target[None, :] - predict(x), W)
