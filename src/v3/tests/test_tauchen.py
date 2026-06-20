"""Tauchen discretization tests (DF26 Sec 4.1; review OPT-3/property Group 1)."""
import numpy as np
import tensorflow as tf

from src.v3.economics import tauchen as T


def test_rows_sum_to_one_and_nonnegative():
    _, P = T.tauchen(0.6, 0.2, 11, 2.5)
    P = P.numpy()
    assert (P >= 0).all()
    np.testing.assert_allclose(P.sum(axis=-1), np.ones(11), atol=1e-12)


def test_grid_symmetric_and_bounds():
    grid, _ = T.tauchen(0.6, 0.2, 11, 2.5)
    g = grid.numpy()
    np.testing.assert_allclose(g, -g[::-1], atol=1e-12)
    expected_hi = 2.5 * 0.2 / np.sqrt(1.0 - 0.6 ** 2)
    np.testing.assert_allclose(g[-1], expected_hi, atol=1e-12)


def test_batched_matches_scalar():
    g1, P1 = T.tauchen(0.6, 0.2, 11, 2.5)
    g2, P2 = T.tauchen(0.7, 0.15, 11, 2.5)
    rho = tf.constant([0.6, 0.7], tf.float64)
    sigma = tf.constant([0.2, 0.15], tf.float64)
    gb, Pb = T.tauchen(rho, sigma, 11, 2.5)
    np.testing.assert_allclose(gb.numpy()[0], g1.numpy(), atol=1e-12)
    np.testing.assert_allclose(gb.numpy()[1], g2.numpy(), atol=1e-12)
    np.testing.assert_allclose(Pb.numpy()[0], P1.numpy(), atol=1e-12)
    np.testing.assert_allclose(Pb.numpy()[1], P2.numpy(), atol=1e-12)


def test_stationary_distribution():
    _, P = T.tauchen(0.6, 0.2, 11, 2.5)
    pi = T.stationary_distribution(P).numpy()
    assert (pi >= 0).all()
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(pi @ P.numpy(), pi, atol=1e-8)


def test_stationary_distribution_batched():
    rho = tf.constant([0.6, 0.8], tf.float64)
    sigma = tf.constant([0.2, 0.1], tf.float64)
    _, P = T.tauchen(rho, sigma, 11, 2.5)
    pi = T.stationary_distribution(P).numpy()
    np.testing.assert_allclose(pi.sum(axis=-1), np.ones(2), atol=1e-10)
    for b in range(2):
        np.testing.assert_allclose(pi[b] @ P.numpy()[b], pi[b], atol=1e-8)
