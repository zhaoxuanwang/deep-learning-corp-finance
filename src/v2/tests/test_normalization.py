"""Tests for v2 normalization module: StaticNormalizer."""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.normalization import StaticNormalizer


class TestStaticNormalizer:
    """Tests for StaticNormalizer."""

    def test_initial_state(self):
        """Before fit(), mean=0 and std=1 (identity transform)."""
        sn = StaticNormalizer(dim=3)
        np.testing.assert_allclose(sn.mean.numpy(), [0, 0, 0])
        np.testing.assert_allclose(sn.std.numpy(), [1, 1, 1])

    def test_normalize_before_fit_is_identity(self):
        """With default mean=0, std=1, normalize(x) == x."""
        sn = StaticNormalizer(dim=2)
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = sn.normalize(x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)

    def test_fit_sets_correct_mean(self):
        """fit() stores exact per-feature mean of the data."""
        sn = StaticNormalizer(dim=2)
        data = tf.constant([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        sn.fit(data)
        np.testing.assert_allclose(sn.mean.numpy(), [3.0, 20.0], atol=1e-5)

    def test_fit_sets_correct_std(self):
        """fit() stores exact per-feature std of the data."""
        sn = StaticNormalizer(dim=1)
        data = tf.constant([[0.0], [2.0], [4.0]])  # std = sqrt(8/3)
        sn.fit(data)
        expected_std = float(np.std([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(sn.std.numpy(), [expected_std], atol=1e-5)

    def test_normalize_after_fit(self):
        """After fit(), normalize(mean) ≈ 0 and normalize(mean+std) ≈ 1."""
        sn = StaticNormalizer(dim=2)
        data = tf.constant(
            [[float(i), float(i) * 10] for i in range(100)],
            dtype=tf.float32)
        sn.fit(data)

        mean_val = tf.reduce_mean(data, axis=0)
        std_val  = tf.math.reduce_std(data, axis=0)

        # Normalizing the mean should give ~0
        result_mean = sn.normalize(mean_val[tf.newaxis])
        np.testing.assert_allclose(result_mean.numpy(), [[0.0, 0.0]], atol=1e-4)

        # Normalizing mean + std should give ~1
        result_one = sn.normalize((mean_val + std_val)[tf.newaxis])
        np.testing.assert_allclose(result_one.numpy(), [[1.0, 1.0]], atol=1e-4)

    def test_normalize_output_shape(self):
        """Output shape matches input shape."""
        sn = StaticNormalizer(dim=4)
        data = tf.random.normal((1000, 4))
        sn.fit(data)
        x = tf.random.normal((32, 4))
        result = sn.normalize(x)
        assert result.shape == (32, 4)

    def test_fit_is_frozen_after_call(self):
        """fit() variables are non-trainable."""
        sn = StaticNormalizer(dim=2)
        assert sn.mean.trainable is False
        assert sn.std.trainable is False

    def test_refit_updates_stats(self):
        """Calling fit() a second time updates to new data statistics."""
        sn = StaticNormalizer(dim=1)
        sn.fit(tf.constant([[0.0], [2.0]]))
        mean_first = float(sn.mean[0])

        sn.fit(tf.constant([[100.0], [200.0]]))
        mean_second = float(sn.mean[0])

        assert mean_second != mean_first
        np.testing.assert_allclose(mean_second, 150.0, atol=1e-4)

    def test_epsilon_prevents_division_by_zero(self):
        """Constant-feature data (std=0) does not produce NaN or Inf."""
        sn = StaticNormalizer(dim=1)
        sn.fit(tf.constant([[5.0]] * 100))  # std = 0
        result = sn.normalize(tf.constant([[5.0]]))
        assert tf.math.is_finite(result[0, 0])

    def test_normalize_batched_input(self):
        """normalize() handles arbitrary leading batch dimensions."""
        sn = StaticNormalizer(dim=3)
        data = tf.random.normal((500, 3))
        sn.fit(data)
        x = tf.random.normal((8, 16, 3))
        result = sn.normalize(x)
        assert result.shape == (8, 16, 3)
