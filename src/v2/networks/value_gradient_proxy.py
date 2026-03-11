"""Value gradient proxy network: s → ξ(s) ≈ ∇_s V(s).

Used by the three-network BRM trainer to decouple V's level-fitting
(Bellman residual) from the gradient information used by the policy (FOC).
"""

import tensorflow as tf
from src.v2.networks.base import GenericNetwork


class ValueGradientProxy(GenericNetwork):
    """Value gradient proxy: s → ξ(s) ∈ R^N (state_dim).

    Approximates ∇_s V(s). The FOC loss uses ξ(s') in place of
    autodiff ∇_s V_φ(s'), breaking the co-adaptation channel.

    Args:
        state_dim: dimension of state vector s (also the output dim).
        n_layers: number of hidden layers.
        n_neurons: neurons per hidden layer.
    """

    def __init__(self, state_dim: int,
                 n_layers: int = 2, n_neurons: int = 128,
                 name: str = "value_grad_proxy", **kwargs):
        super().__init__(input_dim=state_dim, n_layers=n_layers,
                         n_neurons=n_neurons, name=name, **kwargs)
        self.output_dim = state_dim
        self.output_head = tf.keras.layers.Dense(
            state_dim, use_bias=True, name="grad_head")

    def call(self, s, training=False):
        """Forward pass.

        Args:
            s: state tensor, shape (batch, state_dim).
            training: whether to update running z-score stats.

        Returns:
            Gradient proxy ξ(s), shape (batch, state_dim).
        """
        h = self.forward_hidden(s, training=training)
        return self.output_head(h)

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.input_dim})
        return config
