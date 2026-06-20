"""Gauss-Hermite quadrature known-answer tests (DF26 Sec 3.6)."""
import numpy as np
import tensorflow as tf

from src.v3.common import quadrature as Q


def _moment(k, x, w):
    eps = np.sqrt(2.0) * x  # eps ~ N(0,1) after substitution
    return float(Q.integrate(eps ** k, w))


def test_standard_normal_moments_exact_to_degree_9():
    x, w = Q.gauss_hermite(5)  # exact for polynomials up to degree 2Q-1 = 9
    np.testing.assert_allclose(_moment(0, x, w), 1.0, atol=1e-12)
    np.testing.assert_allclose(_moment(2, x, w), 1.0, atol=1e-12)
    np.testing.assert_allclose(_moment(4, x, w), 3.0, atol=1e-12)
    np.testing.assert_allclose(_moment(6, x, w), 15.0, atol=1e-10)
    np.testing.assert_allclose(_moment(8, x, w), 105.0, atol=1e-9)


def test_odd_moments_vanish():
    x, w = Q.gauss_hermite(5)
    for k in (1, 3, 5):
        np.testing.assert_allclose(_moment(k, x, w), 0.0, atol=1e-12)


def test_lognormal_conditional_mean():
    # E[z'|z] = exp(rho log z + sigma^2 / 2) for log z' = rho log z + sigma eps.
    x, w = Q.gauss_hermite(5)
    logz = tf.constant(0.3, tf.float64)
    rho = tf.constant(0.6, tf.float64)
    sigma = tf.constant(0.2, tf.float64)
    val = Q.expectation(logz, rho, sigma, lambda zp: zp, x, w)
    expected = np.exp(0.6 * 0.3 + 0.5 * 0.2 ** 2)
    np.testing.assert_allclose(float(val), expected, rtol=1e-4)


def test_expectation_batched_over_logz():
    x, w = Q.gauss_hermite(5)
    logz = tf.constant([0.0, 0.3, -0.2], tf.float64)
    val = Q.expectation(logz, 0.6, 0.2, lambda zp: zp, x, w)
    expected = np.exp(0.6 * np.array([0.0, 0.3, -0.2]) + 0.5 * 0.2 ** 2)
    np.testing.assert_allclose(val.numpy(), expected, rtol=1e-4)
