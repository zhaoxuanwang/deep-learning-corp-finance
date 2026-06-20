"""Erickson-Whited weighting matrix tests (DF26 Sec 5.7; review NUM-1 Cholesky)."""
import numpy as np
import tensorflow as tf

from src.v3.estimation.weighting import weighting_matrix

DT = tf.float64


def _panel(n_firms=60, T=15, bi=5, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        z=tf.constant(rng.uniform(0.6, 1.4, (n_firms, T)), DT),
        k=tf.constant(rng.uniform(0.5, 3.0, (n_firms, T)), DT),
        b_net=tf.constant(rng.uniform(-0.5, 1.5, (n_firms, T)), DT),
        c=tf.constant(rng.uniform(0.0, 0.4, (n_firms, T)), DT),
        V=tf.constant(rng.uniform(0.5, 2.0, (n_firms, T)), DT),  # all good
        burn_in=bi, T=T, A_pi=0.3, xi=0.45, cf=0.05, delta=0.1)


def test_weighting_symmetric_pd_and_normalized():
    W, W_half = weighting_matrix(_panel())
    Wn = W.numpy()
    assert np.isfinite(Wn).all()
    np.testing.assert_allclose(Wn, Wn.T, atol=1e-8)         # symmetric
    np.linalg.cholesky(Wn)                                   # PD (raises otherwise)
    assert (np.linalg.eigvalsh(Wn) > 0).all()
    colnorm = np.linalg.norm(W_half.numpy(), axis=0)
    np.testing.assert_allclose(np.median(colnorm), 1.0, atol=1e-6)  # median-norm normalization


def test_weighting_consistent_factor():
    # W = W_half' W_half by construction.
    W, W_half = weighting_matrix(_panel(seed=1))
    np.testing.assert_allclose(W.numpy(), (tf.transpose(W_half) @ W_half).numpy(), atol=1e-9)
