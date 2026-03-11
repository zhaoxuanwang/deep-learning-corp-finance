"""Generic network building blocks for v2.

Architecture: RunningZScore → [Dense(bias) → SiLU] × K → output head.
All networks share this hidden stack. Output heads are defined per-network.
"""

import tensorflow as tf
from src.v2.normalization import RunningZScore


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

    Subclasses override _build_output_head() and call() to define
    the output transformation. The input normalization and hidden stack
    are shared.

    Args:
        input_dim: number of input features.
        n_layers: number of hidden layers.
        n_neurons: neurons per hidden layer.
    """

    def __init__(self, input_dim: int, n_layers: int = 2,
                 n_neurons: int = 128,
                 name: str = "generic_network", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self.obs_normalizer = RunningZScore(dim=input_dim)
        self.hidden_stack = HiddenStack(n_layers=n_layers,
                                        n_neurons=n_neurons)

    def update_normalizer(self, batch):
        """Update running z-score statistics. Call during training only."""
        self.obs_normalizer.update(batch)

    def normalize_input(self, x):
        """Apply observation normalization using current running statistics."""
        return self.obs_normalizer.normalize(x)

    def forward_hidden(self, x, training=False):
        """Normalize input and pass through hidden stack.

        The ``training`` flag controls only whether RunningZScore EMA
        statistics (mean, variance) are updated from ``x`` before
        normalizing.  No other layer in the stack has training-dependent
        behaviour (no Dropout, BatchNorm, etc.).

        Callers that perform multi-step rollouts should typically update
        the normalizer once on the initial (buffer-sampled) states and
        then call with ``training=False`` for subsequent rollout depths,
        so that the running statistics track the stationary state
        distribution rather than the on-policy trajectory distribution.
        See the LR trainer for the reference pattern.
        """
        if training:
            self.update_normalizer(x)
        x = self.normalize_input(x)
        return self.hidden_stack(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
        })
        return config
