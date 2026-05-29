"""Generic state-value network: s -> V(s) scalar.

Used by the BRM and SHAC trainers, which learn V(s) directly rather than a
state-action Q-function.

`ParameterizedStateValueNetwork` is a β-conditional variant whose value
function takes (s, β) as input — used by the SHAC surrogate trainer.
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


class ParameterizedStateValueNetwork(StateValueNetwork):
    """β-conditional state-value network: (s, β) → V.

    Internally concatenates `[s, β]` along the last axis and forwards
    through the inherited hidden stack + scalar value head.

    Args:
        state_dim: dimension of the raw state vector s.
        beta_dim:  dimension of the β vector.
        n_layers, n_neurons, seed, activation: see `StateValueNetwork`.
    """

    def __init__(self, state_dim: int, beta_dim: int,
                 n_layers: int = 2, n_neurons: int = 128,
                 name: str = "state_value_param", seed: tuple = None,
                 activation: str = "silu", **kwargs):
        super().__init__(
            state_dim=state_dim + beta_dim,
            n_layers=n_layers, n_neurons=n_neurons,
            name=name, seed=seed, activation=activation, **kwargs)
        self.raw_state_dim = state_dim
        self.beta_dim = beta_dim

    def call(self, s, beta=None, training=False):
        """Forward pass.

        Args:
            s:        state tensor, shape (batch, state_dim) OR a pre-
                      concatenated tensor of shape (batch, state_dim + beta_dim)
                      when `beta is None`. The concatenated form lets
                      target-value-network copies (built via the same
                      mechanism as target policies) keep working with the
                      `value(s)` single-arg call pattern.
            beta:     β tensor, shape (batch, beta_dim). Required unless
                      `s` is already pre-concatenated.
            training: passed through to the (frozen) normalizer.

        Returns:
            Value prediction in level space, shape (batch, 1).
        """
        if beta is None:
            x = s
        else:
            x = tf.concat([s, beta], axis=-1)
        return super().call(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "raw_state_dim": self.raw_state_dim,
            "beta_dim": self.beta_dim,
        })
        return config
