"""Tests for v2 smooth indicator functions."""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.utils.smooth_indicators import (
    indicator_abs_gt,
    indicator_lt,
    indicator_default,
)


class TestIndicatorAbsGt:
    """Tests for indicator_abs_gt: smooth gate for |x| > threshold."""

    def test_hard_mode_basic(self):
        """Hard mode returns exact 0/1 indicator."""
        x = tf.constant([-1.0, -0.0001, 0.0, 0.0001, 1.0])
        result = indicator_abs_gt(x, threshold=0.01, mode="hard")
        expected = [1.0, 0.0, 0.0, 0.0, 1.0]
        np.testing.assert_allclose(result.numpy(), expected)

    def test_soft_mode_far_from_threshold(self):
        """Soft mode returns ~1 well above threshold, ~0 well below."""
        x_above = tf.constant([10.0, -10.0])
        result_above = indicator_abs_gt(x_above, threshold=0.01,
                                        temperature=0.1, logit_clip=20.0,
                                        mode="soft")
        np.testing.assert_allclose(result_above.numpy(), [1.0, 1.0], atol=1e-4)

        x_below = tf.constant([0.0])
        result_below = indicator_abs_gt(x_below, threshold=1.0,
                                        temperature=0.1, logit_clip=20.0,
                                        mode="soft")
        assert result_below.numpy()[0] < 0.01

    def test_soft_approaches_hard_as_temperature_drops(self):
        """As temperature -> 0, soft gate -> hard gate."""
        x = tf.constant([0.5, -0.5, 0.001])
        hard = indicator_abs_gt(x, threshold=0.01, mode="hard").numpy()
        cold = indicator_abs_gt(x, threshold=0.01, temperature=1e-6,
                                logit_clip=20.0, mode="soft").numpy()
        np.testing.assert_allclose(cold, hard, atol=1e-3)

    def test_ste_forward_equals_hard(self):
        """STE mode: forward pass matches hard indicator."""
        x = tf.constant([1.0, 0.0, -1.0])
        hard = indicator_abs_gt(x, threshold=0.01, mode="hard")
        ste = indicator_abs_gt(x, threshold=0.01, temperature=0.1, mode="ste")
        np.testing.assert_allclose(ste.numpy(), hard.numpy(), atol=1e-6)

    def test_ste_has_gradient(self):
        """STE mode: gradients flow through the soft path."""
        x = tf.Variable([0.5])
        with tf.GradientTape() as tape:
            y = indicator_abs_gt(x, threshold=0.01, temperature=0.1,
                                 mode="ste")
        grad = tape.gradient(y, x)
        assert grad is not None
        assert grad.numpy()[0] != 0.0

    def test_output_shape(self):
        """Output shape matches input shape."""
        x = tf.random.normal((8, 4))
        result = indicator_abs_gt(x, threshold=0.01, temperature=0.1)
        assert result.shape == (8, 4)

    def test_soft_is_differentiable(self):
        """Soft mode produces non-zero gradients."""
        x = tf.Variable([0.1])
        with tf.GradientTape() as tape:
            y = indicator_abs_gt(x, threshold=0.01, temperature=0.1,
                                 mode="soft")
        grad = tape.gradient(y, x)
        assert grad is not None
        assert abs(grad.numpy()[0]) > 1e-8

    def test_output_range(self):
        """Soft gate output is in [0, 1]."""
        x = tf.random.normal((100,))
        result = indicator_abs_gt(x, threshold=0.01, temperature=0.5)
        assert tf.reduce_all(result >= 0.0)
        assert tf.reduce_all(result <= 1.0)


class TestIndicatorLt:
    """Tests for indicator_lt: smooth gate for x < -threshold."""

    def test_hard_mode_basic(self):
        """Hard mode returns exact 0/1 indicator."""
        x = tf.constant([-1.0, -0.001, 0.0, 0.001, 1.0])
        result = indicator_lt(x, threshold=0.01, mode="hard")
        expected = [1.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(result.numpy(), expected)

    def test_soft_mode_far_from_threshold(self):
        """Soft mode returns ~1 well below -threshold, ~0 well above."""
        x_below = tf.constant([-10.0])
        result = indicator_lt(x_below, threshold=0.01, temperature=0.1,
                              logit_clip=20.0)
        np.testing.assert_allclose(result.numpy(), [1.0], atol=1e-4)

        x_above = tf.constant([10.0])
        result = indicator_lt(x_above, threshold=0.01, temperature=0.1,
                              logit_clip=20.0)
        assert result.numpy()[0] < 1e-4

    def test_soft_approaches_hard_as_temperature_drops(self):
        """As temperature -> 0, soft gate -> hard gate."""
        x = tf.constant([-0.5, 0.5, -0.001])
        hard = indicator_lt(x, threshold=0.01, mode="hard").numpy()
        cold = indicator_lt(x, threshold=0.01, temperature=1e-6,
                            logit_clip=20.0, mode="soft").numpy()
        np.testing.assert_allclose(cold, hard, atol=1e-3)

    def test_ste_has_gradient(self):
        """STE mode: gradients flow through the soft path."""
        x = tf.Variable([-0.5])
        with tf.GradientTape() as tape:
            y = indicator_lt(x, threshold=0.01, temperature=0.1, mode="ste")
        grad = tape.gradient(y, x)
        assert grad is not None
        assert grad.numpy()[0] != 0.0

    def test_output_range(self):
        """Soft gate output is in [0, 1]."""
        x = tf.random.normal((100,))
        result = indicator_lt(x, threshold=0.01, temperature=0.5)
        assert tf.reduce_all(result >= 0.0)
        assert tf.reduce_all(result <= 1.0)


class TestIndicatorDefault:
    """Tests for indicator_default: Gumbel-sigmoid default probability."""

    def test_deterministic_mode_large_positive(self):
        """Large positive V_tilde_norm -> low default probability."""
        V = tf.constant([10.0])
        prob = indicator_default(V, temperature=0.1, noise=False)
        assert prob.numpy()[0] < 0.01

    def test_deterministic_mode_large_negative(self):
        """Large negative V_tilde_norm -> high default probability."""
        V = tf.constant([-10.0])
        prob = indicator_default(V, temperature=0.1, noise=False)
        assert prob.numpy()[0] > 0.99

    def test_noise_mode_produces_different_samples(self):
        """Noisy mode with different seeds gives different results."""
        V = tf.constant([0.0, 0.0, 0.0])
        p1 = indicator_default(V, temperature=1.0, noise=True,
                               noise_seed=tf.constant([1, 2]))
        p2 = indicator_default(V, temperature=1.0, noise=True,
                               noise_seed=tf.constant([3, 4]))
        # With different seeds, at least one sample should differ
        assert not np.allclose(p1.numpy(), p2.numpy(), atol=1e-6)

    def test_noise_mode_reproducible_with_same_seed(self):
        """Same seed produces identical results."""
        V = tf.constant([0.0, 0.0])
        seed = tf.constant([42, 7])
        p1 = indicator_default(V, temperature=1.0, noise=True, noise_seed=seed)
        p2 = indicator_default(V, temperature=1.0, noise=True, noise_seed=seed)
        np.testing.assert_allclose(p1.numpy(), p2.numpy())

    def test_output_range(self):
        """Output is in (0, 1)."""
        V = tf.random.normal((100,))
        prob = indicator_default(V, temperature=0.5, noise=False)
        assert tf.reduce_all(prob > 0.0)
        assert tf.reduce_all(prob < 1.0)

    def test_output_shape(self):
        """Output shape matches input shape."""
        V = tf.random.normal((8, 4))
        prob = indicator_default(V, temperature=0.5, noise=False)
        assert prob.shape == (8, 4)
