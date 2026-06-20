"""Batched Levenberg-Marquardt known-answer tests (DF26 Sec 5.5)."""
import numpy as np
import tensorflow as tf

from src.v3.estimation import reparam
from src.v3.estimation.lm import levenberg_marquardt

DT = tf.float64


def test_recovers_known_nonlinear_least_squares():
    # predict(x) = [x0, x1, x0*x1]; injective near generic points -> unique optimum.
    def predict(x):
        return tf.stack([x[:, 0], x[:, 1], x[:, 0] * x[:, 1]], axis=1)

    x_star = np.array([0.3, -0.5])
    target = tf.constant([x_star[0], x_star[1], x_star[0] * x_star[1]], DT)
    W = tf.eye(3, dtype=DT)
    x0 = tf.constant(np.random.default_rng(0).normal(size=(12, 2)), DT)
    x, f = levenberg_marquardt(predict, x0, target, W, max_iter=40)
    best = int(tf.argmin(f))
    np.testing.assert_allclose(x.numpy()[best], x_star, atol=1e-6)
    assert float(f[best]) < 1e-12


def test_weighting_matrix_used():
    # Overdetermined linear map; weighted solution differs from unweighted, and LM
    # matches the closed-form weighted least squares.
    A = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], DT)

    def predict(x):
        return tf.einsum("mp,bp->bm", A, x)

    target = tf.constant([1.0, 1.0, 0.0], DT)        # inconsistent -> nontrivial WLS
    W = tf.constant(np.diag([1.0, 1.0, 10.0]), DT)
    x0 = tf.constant(np.random.default_rng(1).normal(size=(8, 2)), DT)
    x, f = levenberg_marquardt(predict, x0, target, W, max_iter=60)
    best = x.numpy()[int(tf.argmin(f))]
    # Closed form: x = (A'WA)^-1 A'W target.
    An, Wn, tn = A.numpy(), W.numpy(), target.numpy()
    xref = np.linalg.solve(An.T @ Wn @ An, An.T @ Wn @ tn)
    np.testing.assert_allclose(best, xref, atol=1e-6)


def test_with_sigmoid_reparam_stays_in_bounds():
    # Optimizing over x with beta = reparam(x) recovers a target beta inside bounds.
    lo = tf.constant([0.0, -1.0], DT)
    hi = tf.constant([1.0, 2.0], DT)
    beta_star = tf.constant([0.7, 0.4], DT)

    def predict(x):
        beta = reparam.to_constrained(x, lo, hi)
        return tf.stack([beta[:, 0], beta[:, 1], beta[:, 0] * beta[:, 1]], axis=1)

    target = tf.stack([beta_star[0], beta_star[1], beta_star[0] * beta_star[1]])[None, :][0]
    W = tf.eye(3, dtype=DT)
    x0 = tf.constant(np.random.default_rng(2).normal(size=(10, 2)), DT)
    x, f = levenberg_marquardt(predict, x0, target, W, max_iter=80)
    beta = reparam.to_constrained(x, lo, hi).numpy()
    best = beta[int(tf.argmin(f))]
    assert (best >= lo.numpy() - 1e-9).all() and (best <= hi.numpy() + 1e-9).all()
    np.testing.assert_allclose(best, beta_star.numpy(), atol=1e-4)
