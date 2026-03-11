"""Generic policy network: s → a with affine-rescaled output + clip.

The policy maps states to actions. The output head is affine-rescaled:
  action = center + scale * raw
where scale = sqrt(half_range) provides moderate gradient amplification.

When action bounds are finite, center and half_range are derived from
the bounds automatically.  When bounds are infinite (unbounded actions),
the caller must supply an explicit (action_center, action_half_range)
via the environment's action_scale_reference().

The output is clipped to [action_low, action_high].  With ±inf bounds
the clip is a no-op, and the only physical constraint comes from the
environment's _apply_action().
"""

import tensorflow as tf
from src.v2.networks.base import GenericNetwork


class PolicyNetwork(GenericNetwork):
    """Policy network: s → a.

    Args:
        state_dim: dimension of state vector s.
        action_dim: dimension of action vector a.
        action_low: lower clip bounds, shape (action_dim,). May be -inf.
        action_high: upper clip bounds, shape (action_dim,). May be +inf.
        action_center: explicit center for affine scaling (required when
            bounds contain inf, optional otherwise).
        action_half_range: explicit half-range for affine scaling (required
            when bounds contain inf, optional otherwise).
        n_layers: number of hidden layers.
        n_neurons: neurons per hidden layer.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 action_low: tf.Tensor, action_high: tf.Tensor,
                 action_center: tf.Tensor = None,
                 action_half_range: tf.Tensor = None,
                 n_layers: int = 2, n_neurons: int = 128,
                 name: str = "policy", **kwargs):
        super().__init__(input_dim=state_dim, n_layers=n_layers,
                         n_neurons=n_neurons,
                         name=name, **kwargs)
        self.action_dim = action_dim
        self.action_low = tf.cast(action_low, tf.float32)
        self.action_high = tf.cast(action_high, tf.float32)

        # Affine rescaling: sqrt-scale for moderate gradient amplification.
        if action_center is not None and action_half_range is not None:
            self.action_center = tf.cast(action_center, tf.float32)
            self.action_half_range = tf.cast(action_half_range, tf.float32)
        else:
            # Derive from finite bounds — guard against silent NaN/inf.
            if not (tf.reduce_all(tf.math.is_finite(self.action_low))
                    and tf.reduce_all(tf.math.is_finite(self.action_high))):
                raise ValueError(
                    "action_low/action_high contain non-finite values. "
                    "Pass explicit action_center and action_half_range."
                )
            self.action_center = (self.action_low + self.action_high) / 2.0
            self.action_half_range = (self.action_high - self.action_low) / 2.0
        self.action_scale = tf.sqrt(self.action_half_range)

        # Small kernel init so initial raw ≈ 0 → action ≈ center.
        self.output_head = tf.keras.layers.Dense(
            action_dim, use_bias=True, name="output_head",
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
            bias_initializer="zeros")

    def call(self, s, training=False, return_raw=False):
        """Forward pass.

        Args:
            s: state tensor, shape (batch, state_dim).
            training: whether to update running z-score stats.
            return_raw: if True, return (clipped, scaled) tuple.
                Used by MVE actor update where Q receives pre-clip action.

        Returns:
            Action tensor, shape (batch, action_dim).
            If return_raw: tuple (clipped, scaled).
        """
        h = self.forward_hidden(s, training=training)
        raw = self.output_head(h)
        # Affine rescale: raw ≈ 0 at init → action ≈ center.
        scaled = self.action_center + self.action_scale * raw
        # clip_by_value with ±inf bounds is a no-op.
        clipped = tf.clip_by_value(scaled, self.action_low, self.action_high)
        if return_raw:
            return clipped, scaled
        return clipped

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.input_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.numpy().tolist(),
            "action_high": self.action_high.numpy().tolist(),
        })
        return config
