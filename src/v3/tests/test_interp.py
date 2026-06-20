"""Bilinear (log k, b) interpolation tests (DF26 Sec 4.2)."""
import numpy as np
import tensorflow as tf

from src.v3.solver import interp

DT = tf.float64


def _axes(n_k, n_b, k_lo=0.5, k_hi=2.0, b_lo=-1.0, b_hi=2.0):
    k = tf.exp(tf.linspace(tf.math.log(tf.constant(k_lo, DT)),
                           tf.math.log(tf.constant(k_hi, DT)), n_k))
    b = tf.linspace(tf.constant(b_lo, DT), tf.constant(b_hi, DT), n_b)
    return k, b, tf.math.log(k)


def test_recovers_grid_nodes():
    n_z, n_k, n_b = 3, 5, 4
    k, b, logk = _axes(n_k, n_b)
    V = tf.reshape(tf.range(n_z * n_k * n_b, dtype=DT), [n_z, n_k, n_b])
    kq, bq = tf.meshgrid(k, b, indexing="ij")
    out = interp.interp_grid(V, logk, b, kq, bq)
    np.testing.assert_allclose(out.numpy(), V.numpy(), atol=1e-9)


def test_linear_in_b():
    k, b, logk = _axes(2, 3, k_lo=1.0, k_hi=2.0, b_lo=0.0, b_hi=2.0)  # b = [0,1,2]
    V = tf.constant([[[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]], DT)
    out = interp.interp_grid(V, logk, b, tf.constant([1.0], DT), tf.constant([0.5], DT))
    np.testing.assert_allclose(out.numpy().ravel(), [5.0], atol=1e-9)


def test_linear_in_logk():
    k, b, logk = _axes(3, 2, k_lo=1.0, k_hi=4.0, b_lo=0.0, b_hi=1.0)  # k = [1,2,4]
    V = tf.constant([[[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]], DT)
    # k = 2 is the middle node (uniform in log) -> value 10.
    out = interp.interp_grid(V, logk, b, tf.constant([2.0], DT), tf.constant([0.0], DT))
    np.testing.assert_allclose(out.numpy().ravel(), [10.0], atol=1e-9)


def test_bilinear_corners_matches_interp_grid():
    # The corner indices/weights used to assemble the policy-eval operator must
    # reproduce interp_grid exactly.
    from src.v3.solver.interp import bilinear_corners
    n_z, n_k, n_b = 4, 6, 5
    k, b, logk = _axes(n_k, n_b)
    V = tf.random.stateless_normal([n_z, n_k, n_b], seed=[7, 8], dtype=DT)
    kq = tf.constant([1.0, 1.3, 0.7], DT)
    bq = tf.constant([0.0, 0.5, -0.5], DT)
    corners, w = bilinear_corners(logk, b, n_k, n_b, kq, bq)
    v_flat = tf.reshape(V, [n_z, n_k * n_b])
    vt = tf.reduce_sum(tf.gather(v_flat, corners, axis=1) * w[None, :, :], axis=-1)
    np.testing.assert_allclose(vt.numpy(), interp.interp_grid(V, logk, b, kq, bq).numpy(), atol=1e-12)


def test_shape_and_batch():
    n_z, n_k, n_b = 4, 6, 5
    k, b, logk = _axes(n_k, n_b)
    V = tf.random.stateless_normal([n_z, n_k, n_b], seed=[1, 2], dtype=DT)
    kq = tf.constant([[1.0, 1.5], [0.8, 1.2]], DT)
    bq = tf.constant([[0.0, 0.5], [-0.5, 1.0]], DT)
    out = interp.interp_grid(V, logk, b, kq, bq)
    assert tuple(out.shape) == (n_z, 2, 2)
