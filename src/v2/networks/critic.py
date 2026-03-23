"""Generic Q-critic network: (s, a) → Q(s, a) in level space."""

import tensorflow as tf
from src.v2.networks.base import GenericNetwork
from src.v2.utils.seeding import make_seed_int


class CriticNetwork(GenericNetwork):
    """Q-value critic: (s, a) → scalar Q-value.

    Input is the concatenation [s, a]. The linear output head predicts
    Q directly in level space.

    Args:
        state_dim: dimension of state vector s.
        action_dim: dimension of action vector a.
        n_layers: number of hidden layers.
        n_neurons: neurons per hidden layer.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 n_layers: int = 2, n_neurons: int = 128,
                 name: str = "critic", seed: tuple = None, **kwargs):
        super().__init__(input_dim=state_dim + action_dim,
                         n_layers=n_layers, n_neurons=n_neurons,
                         name=name, seed=seed, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        head_init = "glorot_uniform"
        if self.seed is not None:
            head_init = tf.keras.initializers.GlorotUniform(
                seed=make_seed_int(self.seed, "q_head"))
        self.output_head = tf.keras.layers.Dense(
            1, use_bias=True, name="q_head",
            kernel_initializer=head_init,
            bias_initializer="zeros")

    def call(self, s, a, training=False):
        """Forward pass.

        Args:
            s: state tensor, shape (batch, state_dim).
            a: action tensor, shape (batch, action_dim).
            training: whether to update running z-score stats.

        Returns:
            Q prediction in level space, shape (batch, 1).
        """
        x = tf.concat([s, a], axis=-1)
        h = self.forward_hidden(x, training=training)
        return self.output_head(h)

    def q_level(self, s, a, training=False):
        """Return Q in level space. Same as call() — kept for API compat."""
        return self.call(s, a, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "seed": list(self.seed) if self.seed is not None else None,
        })
        return config
