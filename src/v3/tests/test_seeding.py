"""Stateless keyed-seed reproducibility (DF26 Sec 10; review REPRO-2)."""
import numpy as np
import tensorflow as tf

from src.v3.common import seeding as S


def test_same_key_same_draw():
    a = S.uniform([100], 123, S.Purpose.TRAIN, 0, 0)
    b = S.uniform([100], 123, S.Purpose.TRAIN, 0, 0)
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_different_index_differs():
    a = S.uniform([100], 123, S.Purpose.TRAIN, 0, 0)
    b = S.uniform([100], 123, S.Purpose.TRAIN, 0, 1)
    assert not np.allclose(a.numpy(), b.numpy())


def test_different_purpose_differs():
    a = S.uniform([100], 123, S.Purpose.TRAIN, 5)
    b = S.uniform([100], 123, S.Purpose.COLLECT, 5)
    assert not np.allclose(a.numpy(), b.numpy())


def test_different_master_differs():
    a = S.normal([50], 1, S.Purpose.INIT, 0)
    b = S.normal([50], 2, S.Purpose.INIT, 0)
    assert not np.allclose(a.numpy(), b.numpy())


def test_uniform_range_respected():
    u = S.uniform([2000], 7, S.Purpose.TRAIN, 1, minval=-2.0, maxval=3.0)
    assert float(tf.reduce_min(u)) >= -2.0
    assert float(tf.reduce_max(u)) <= 3.0


def test_normal_moments_reasonable():
    z = S.normal([200000], 99, S.Purpose.INIT, 3)
    assert abs(float(tf.reduce_mean(z))) < 0.02
    assert abs(float(tf.math.reduce_std(z)) - 1.0) < 0.02
