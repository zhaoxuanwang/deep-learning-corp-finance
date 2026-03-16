"""Generic network building blocks for v2.

Architecture: StaticNormalizer → [Dense(bias) → SiLU] × K → output head.
All networks share this hidden stack. Output heads are defined per-network.
"""

import tensorflow as tf
from src.v2.utils.normalization import StaticNormalizer


class HiddenStack(tf.keras.layers.Layer):
    """Shared hidden stack: [Dense(bias) → SiLU] × K."""

    def __init__(self, n_layers: int = 2, n_neurons: int = 128,
                 name: str = "hidden_stack", **kwargs):
        super().__init__(name=name, **kwargs)
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self.dense_layers = []
        for i in range(n_layers):
            self.dense_layers.append(
                tf.keras.layers.Dense(n_neurons, use_bias=True,
                                      name=f"dense_{i}"))

    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
            x = tf.nn.silu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"n_layers": self._n_layers, "n_neurons": self._n_neurons})
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
                 name: str = "generic_network", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self.normalizer   = StaticNormalizer(dim=input_dim)
        self.hidden_stack = HiddenStack(n_layers=n_layers,
                                        n_neurons=n_neurons)

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
        })
        return config
