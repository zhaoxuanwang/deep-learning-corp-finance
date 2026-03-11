"""Observation normalization for v2 generic framework.

Running Z-Score normalizer using exponential moving average (EMA) of
per-feature mean and variance. Standard approach in OpenAI Baselines,
Stable-Baselines3, and CleanRL.

Usage:
    normalizer = RunningZScore(dim=state_dim, momentum=0.01)

    # During training — update stats then normalize:
    normalizer.update(batch)
    x_norm = normalizer.normalize(batch)

    # During evaluation — normalize only (stats frozen):
    x_norm = normalizer.normalize(batch)
"""

import tensorflow as tf


class RunningZScore(tf.Module):
    """Exponential moving average z-score normalizer.

    Maintains running estimates of per-feature mean and variance via EMA:
        running_mean ← (1 - α) * running_mean + α * batch_mean
        running_var  ← (1 - α) * running_var  + α * batch_var

    Then normalizes:  x_norm = (x - running_mean) / sqrt(running_var + ε)

    Args:
        dim: number of input features.
        momentum: EMA update rate α. Higher = adapts faster, noisier.
        epsilon: small constant for numerical stability in division.
    """

    def __init__(self, dim: int, momentum: float = 0.001,
                 epsilon: float = 1e-8, name: str = "running_zscore"):
        super().__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = tf.Variable(
            tf.zeros(dim), trainable=False, name="running_mean")
        self.running_var = tf.Variable(
            tf.ones(dim), trainable=False, name="running_var")
        self.count = tf.Variable(0, dtype=tf.int64, trainable=False,
                                 name="count")

    def update(self, batch: tf.Tensor) -> None:
        """Update running statistics from a batch. Call during training only."""
        batch_mean = tf.reduce_mean(batch, axis=0)
        batch_var = tf.math.reduce_variance(batch, axis=0)
        alpha = self.momentum
        self.running_mean.assign(
            (1.0 - alpha) * self.running_mean + alpha * batch_mean)
        self.running_var.assign(
            (1.0 - alpha) * self.running_var + alpha * batch_var)
        self.count.assign_add(1)

    def normalize(self, x: tf.Tensor) -> tf.Tensor:
        """Normalize using current running statistics."""
        std = tf.sqrt(self.running_var + self.epsilon)
        return (x - self.running_mean) / std
