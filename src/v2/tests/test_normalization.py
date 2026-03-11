"""Tests for v2 normalization module: RunningZScore."""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.normalization import RunningZScore


# =============================================================================
# RunningZScore
# =============================================================================

class TestRunningZScore:
    """Tests for RunningZScore normalizer."""

    def test_initial_state(self):
        """Initial mean=0, var=1."""
        rz = RunningZScore(dim=3)
        np.testing.assert_allclose(rz.running_mean.numpy(), [0, 0, 0])
        np.testing.assert_allclose(rz.running_var.numpy(), [1, 1, 1])

    def test_normalize_initial(self):
        """With initial stats, normalize(x) = x / 1 = x."""
        rz = RunningZScore(dim=2)
        x = tf.constant([[1.0, 2.0]])
        result = rz.normalize(x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)

    def test_update_shifts_mean(self):
        """Updating with a batch shifts running_mean toward batch mean."""
        rz = RunningZScore(dim=1, momentum=0.1)
        batch = tf.constant([[10.0]] * 100)  # batch mean = 10
        rz.update(batch)
        # After one update: mean = 0.9 * 0 + 0.1 * 10 = 1.0
        assert float(rz.running_mean[0]) == pytest.approx(1.0, abs=1e-5)

    def test_explicit_update_normalize(self):
        """update() then normalize() produces correct output."""
        rz = RunningZScore(dim=1, momentum=1.0)  # full replacement
        batch = tf.constant([[5.0]] * 100)  # mean=5, var=0
        rz.update(batch)
        # After full update: mean=5, var=0, std=sqrt(0+eps)≈eps
        result = rz.normalize(tf.constant([[5.0]]))
        # (5 - 5) / sqrt(eps) ≈ 0
        assert abs(float(result[0, 0])) < 1e-2

    def test_inference_does_not_update(self):
        """normalize() without update() does not change stats."""
        rz = RunningZScore(dim=1)
        x = tf.constant([[100.0]] * 10)
        _ = rz.normalize(x)
        assert float(rz.running_mean[0]) == pytest.approx(0.0)
        assert int(rz.count) == 0

    def test_update_increments_count(self):
        """update() increments the count variable."""
        rz = RunningZScore(dim=1, momentum=0.5)
        x = tf.constant([[5.0]] * 10)
        rz.update(x)
        assert int(rz.count) == 1
        rz.update(x)
        assert int(rz.count) == 2

    def test_output_shape(self):
        """Output shape matches input shape."""
        rz = RunningZScore(dim=4)
        x = tf.constant(np.random.randn(32, 4).astype(np.float32))
        rz.update(x)
        result = rz.normalize(x)
        assert result.shape == (32, 4)
