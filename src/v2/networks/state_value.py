"""Generic state-value network: s → V(s) scalar.

Distinct from CriticNetwork (Q-critic: (s, a) → Q(s, a)).
Used by the BRM trainer which learns V(s) directly, not Q(s, a).
"""

import tensorflow as tf
from src.v2.networks.base import GenericNetwork


class StateValueNetwork(GenericNetwork):
    """State value network: s → V(s).

    Input is the state vector s only (no action). The linear output head
    predicts V directly in level space.

    Args:
        state_dim: dimension of state vector s.
        n_layers: number of hidden layers.
        n_neurons: neurons per hidden layer.
    """

    def __init__(self, state_dim: int,
                 n_layers: int = 2, n_neurons: int = 128,
                 name: str = "state_value", **kwargs):
        super().__init__(input_dim=state_dim, n_layers=n_layers,
                         n_neurons=n_neurons, name=name, **kwargs)
        self.output_head = tf.keras.layers.Dense(
            1, use_bias=True, name="value_head")

    def call(self, s, training=False):
        """Forward pass.

        Args:
            s: state tensor, shape (batch, state_dim).
            training: whether to update running z-score stats.

        Returns:
            Value prediction in level space, shape (batch, 1).
        """
        h = self.forward_hidden(s, training=training)
        return self.output_head(h)

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.input_dim})
        return config
