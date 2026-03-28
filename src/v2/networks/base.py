"""Generic network building blocks for v2.

Architecture: StaticNormalizer → [Dense(bias) → activation] × K → output
head. All networks share this hidden stack. Output heads are defined
per-network.
"""

from typing import Optional

import tensorflow as tf
from src.v2.utils.normalization import StaticNormalizer
from src.v2.utils.seeding import fold_in_seed, make_seed_int


class HiddenStack(tf.keras.layers.Layer):
    """Shared hidden stack: [Dense(bias) → activation] × K."""

    def __init__(self, n_layers: int = 2, n_neurons: int = 128,
                 name: str = "hidden_stack", seed: Optional[tuple] = None,
                 activation: str = "silu",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._seed = tuple(seed) if seed is not None else None
        self._activation = str(activation).lower()
        if self._activation not in ("silu", "relu"):
            raise ValueError(
                "HiddenStack activation must be one of {'silu', 'relu'}."
            )
        self.dense_layers = []
        for i in range(n_layers):
            kernel_init = "glorot_uniform"
            if self._seed is not None:
                kernel_init = tf.keras.initializers.GlorotUniform(
                    seed=make_seed_int(self._seed, "dense", i))
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    n_neurons, use_bias=True,
                    kernel_initializer=kernel_init,
                    bias_initializer="zeros",
                                      name=f"dense_{i}"))

    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
            if self._activation == "relu":
                x = tf.nn.relu(x)
            else:
                x = tf.nn.silu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
            "seed": list(self._seed) if self._seed is not None else None,
            "activation": self._activation,
        })
        return config


class GenericNetwork(tf.keras.Model):
    """Base class for all v2 networks.

    Subclasses override call() to define the output transformation.
    Input normalization and the hidden stack are shared.

    The normalizer is a StaticNormalizer fitted once from the full
    training dataset before gradient steps begin (see pipeline.py
    fit_normalizer_* helpers).  It is frozen during training.

    Args:
        input_dim: Number of input features.
        n_layers:  Number of hidden layers.
        n_neurons: Neurons per hidden layer.
    """

    def __init__(self, input_dim: int, n_layers: int = 2,
                 n_neurons: int = 128,
                 name: str = "generic_network",
                 seed: Optional[tuple] = None,
                 activation: str = "silu",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self.seed = tuple(seed) if seed is not None else None
        self.activation = str(activation).lower()
        self.normalizer   = StaticNormalizer(dim=input_dim)
        hidden_seed = (
            fold_in_seed(self.seed, "hidden_stack")
            if self.seed is not None else None
        )
        self.hidden_stack = HiddenStack(
            n_layers=n_layers,
            n_neurons=n_neurons,
            seed=hidden_seed,
            activation=self.activation,
        )

    def forward_hidden(self, x, training=False):
        """Normalize input and pass through hidden stack.

        The training flag is accepted for Keras API compatibility
        but has no effect: the normalizer is always frozen after fit().
        """
        x = self.normalizer.normalize(x)
        return self.hidden_stack(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "n_layers":  self._n_layers,
            "n_neurons": self._n_neurons,
            "seed": list(self.seed) if self.seed is not None else None,
            "activation": self.activation,
        })
        return config
