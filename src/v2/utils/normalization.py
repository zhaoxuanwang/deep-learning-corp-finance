"""Observation normalization for v2 generic framework.

Static Z-score normalizer: fitted once from the full training dataset
before gradient steps begin, then frozen for the entire round.

Design rationale
----------------
All v2 training methods use pre-generated fixed datasets, so exact
population statistics are computable in a single pass before training
starts.  There is no need for incremental (EMA) estimation during
training, and no warmup phase is required.

The exogenous and endogenous state components are fitted from different
data sources.  See fit_normalizer_traj() and fit_normalizer_flat() in
src/v2/data/pipeline.py for the construction details.

Usage
-----
    normalizer = StaticNormalizer(dim=state_dim)

    # Before training (once per round):
    normalizer.fit(s_all)          # s_all: (N, state_dim)

    # During training (always frozen):
    x_norm = normalizer.normalize(x)
"""

import tensorflow as tf


class StaticNormalizer(tf.Module):
    """Z-score normalizer fitted once from the full dataset.

    Computes exact per-feature mean and standard deviation from the
    supplied data tensor and stores them as non-trainable variables.
    After fit(), normalize() applies the frozen transform for the rest
    of the round.  Refit at the start of each round from fresh data.

    Args:
        dim:     Number of input features (state dimension).
        epsilon: Small constant added to std for numerical stability.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8,
                 name: str = "normalizer"):
        super().__init__(name=name)
        self.dim     = dim
        self.epsilon = epsilon
        self.mean = tf.Variable(tf.zeros(dim), trainable=False, name="mean")
        self.std  = tf.Variable(tf.ones(dim),  trainable=False, name="std")

    def fit(self, data: tf.Tensor) -> None:
        """Compute exact mean and std from the full dataset.

        Call once before gradient training begins.  data should contain
        all representative states for the round (see pipeline helpers).

        Args:
            data: Tensor of shape (N, dim) containing state observations.
        """
        self.mean.assign(tf.reduce_mean(data, axis=0))
        self.std.assign(tf.math.reduce_std(data, axis=0))

    def normalize(self, x: tf.Tensor) -> tf.Tensor:
        """Apply frozen z-score normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor, same shape as x.
        """
        return (x - self.mean) / (self.std + self.epsilon)
