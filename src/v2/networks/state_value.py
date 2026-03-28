"""Generic state-value network: s -> V(s) scalar.

Used by the BRM and SHAC trainers, which learn V(s) directly rather than a
state-action Q-function.
"""

import tensorflow as tf
from src.v2.networks.base import GenericNetwork
from src.v2.utils.seeding import make_seed_int


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
                 name: str = "state_value", seed: tuple = None,
                 activation: str = "silu", **kwargs):
        super().__init__(input_dim=state_dim, n_layers=n_layers,
                         n_neurons=n_neurons, name=name, seed=seed,
                         activation=activation, **kwargs)
        head_init = "glorot_uniform"
        if self.seed is not None:
            head_init = tf.keras.initializers.GlorotUniform(
                seed=make_seed_int(self.seed, "value_head"))
        self.output_head = tf.keras.layers.Dense(
            1, use_bias=True, name="value_head",
            kernel_initializer=head_init,
            bias_initializer="zeros")

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
        config.update({
            "state_dim": self.input_dim,
            "seed": list(self.seed) if self.seed is not None else None,
            "activation": self.activation,
        })
        return config
