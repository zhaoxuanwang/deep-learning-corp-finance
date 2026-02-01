"""
src/networks/network_basic.py

Neural network builders for Basic Model (Sec. 1).

Default Architecture (report_brief.md lines 341-357):
- 2 hidden layers with 16-32 neurons each
- SiLU (swish) activation for hidden layers
- Bounded sigmoid output for policy (k')
- Linear output for value (V)

=============================================================================
KEY DESIGN: INPUT/OUTPUT IN LEVELS, INTERNAL NORMALIZATION FOR STABILITY
=============================================================================

All networks follow a consistent pattern:
1. INPUTS: Accept (k, z) in LEVELS (actual economic values)
2. INTERNAL: Normalize inputs to [0, 1] for numerical stability
3. OUTPUTS: Return values in LEVELS (bounded sigmoid ensures output ∈ [k_min, k_max])

This design means:
- Trainers pass k values directly to economic functions (no scaling needed)
- The normalization is purely for NN stability, invisible to users
- Works with any bounds (model-based or arbitrary user-specified)
"""

import tensorflow as tf
from typing import Tuple

class BasicPolicyNetwork(tf.keras.Model):
    """
    Policy network for the Basic model: (k, z) -> k'

    Input transform: min-max normalization to [0, 1]
    Output activation: k' = k_min + (k_max - k_min) * sigmoid(raw)

    Args:
        k_min: Minimum capital
        k_max: Maximum capital
        logz_min: Minimum log productivity
        logz_max: Maximum log productivity
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation
    """

    def __init__(
        self,
        k_min: float,
        k_max: float,
        logz_min: float,
        logz_max: float,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()
        self.k_min = tf.constant(k_min, dtype=tf.float32)
        self.k_max = tf.constant(k_max, dtype=tf.float32)
        self.logz_min = tf.constant(logz_min, dtype=tf.float32)
        self.logz_max = tf.constant(logz_max, dtype=tf.float32)

        # Save config parameters for serialization
        self._k_min = k_min
        self._k_max = k_max
        self._logz_min = logz_min
        self._logz_max = logz_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # Hidden layers
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]

        # Raw output (no activation, we apply bounded sigmoid)
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            k: Capital stock (batch_size,) or (batch_size, 1), levels
            z: Productivity shock (batch_size,) or (batch_size, 1), levels

        Returns:
            k_next: Next period capital (batch_size, 1)
        """
        # Ensure 2D
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])

        # Normalize inputs to [0, 1]
        k_range = self.k_max - self.k_min
        k_norm = (k - self.k_min) / k_range

        log_z = tf.math.log(tf.maximum(z, 1e-8))
        logz_range = self.logz_max - self.logz_min
        logz_norm = (log_z - self.logz_min) / logz_range

        x = tf.concat([k_norm, logz_norm], axis=1)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Output: k' = k_min + (k_max - k_min) * sigmoid(raw)
        raw = self.output_layer(x)
        k_next = self.k_min + (self.k_max - self.k_min) * tf.nn.sigmoid(raw)

        return k_next

    def get_config(self):
        """Return configuration for serialization."""
        return {
            'k_min': self._k_min,
            'k_max': self._k_max,
            'logz_min': self._logz_min,
            'logz_max': self._logz_max,
            'n_layers': self._n_layers,
            'n_neurons': self._n_neurons,
            'activation': self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)

    def build_from_config(self, config):
        """Build model state from config (for Keras serialization)."""
        # Build by calling with dummy inputs
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        self(dummy_k, dummy_z)

    def get_build_config(self):
        """Return build configuration."""
        return self.get_config()


class BasicValueNetwork(tf.keras.Model):
    """
    Value network for the Basic model: (k, z) -> V(k, z)

    Input transform: min-max normalization to [0, 1]
    Output activation: linear (unbounded)

    Args:
        k_min: Minimum capital
        k_max: Maximum capital
        logz_min: Minimum log productivity
        logz_max: Maximum log productivity
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation
    """

    def __init__(
        self,
        k_min: float,
        k_max: float,
        logz_min: float,
        logz_max: float,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()
        self.k_min = tf.constant(k_min, dtype=tf.float32)
        self.k_max = tf.constant(k_max, dtype=tf.float32)
        self.logz_min = tf.constant(logz_min, dtype=tf.float32)
        self.logz_max = tf.constant(logz_max, dtype=tf.float32)

        # Save config parameters for serialization
        self._k_min = k_min
        self._k_max = k_max
        self._logz_min = logz_min
        self._logz_max = logz_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            k: Capital stock (batch_size,) or (batch_size, 1), levels
            z: Productivity shock (batch_size,) or (batch_size, 1), levels

        Returns:
            V: Value (batch_size, 1)
        """
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])

        # Normalize inputs to [0, 1]
        k_range = self.k_max - self.k_min
        k_norm = (k - self.k_min) / k_range

        log_z = tf.math.log(tf.maximum(z, 1e-8))
        logz_range = self.logz_max - self.logz_min
        logz_norm = (log_z - self.logz_min) / logz_range

        x = tf.concat([k_norm, logz_norm], axis=1)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)

    def get_config(self):
        """Return configuration for serialization."""
        return {
            'k_min': self._k_min,
            'k_max': self._k_max,
            'logz_min': self._logz_min,
            'logz_max': self._logz_max,
            'n_layers': self._n_layers,
            'n_neurons': self._n_neurons,
            'activation': self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)

    def build_from_config(self, config):
        """Build model state from config (for Keras serialization)."""
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        self(dummy_k, dummy_z)

    def get_build_config(self):
        """Return build configuration."""
        return self.get_config()


def build_basic_networks(
    k_min: float,
    k_max: float,
    logz_min: float,
    logz_max: float,
    n_layers: int,
    n_neurons: int,
    activation: str
) -> Tuple[BasicPolicyNetwork, BasicValueNetwork]:
    """
    Factory to build Basic model networks.

    Args:
        k_min: Minimum capital bound
        k_max: Maximum capital bound
        logz_min: Minimum log productivity bound
        logz_max: Maximum log productivity bound
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation function

    Returns:
        Tuple of (policy_net, value_net)
    """
    policy = BasicPolicyNetwork(
        k_min, k_max, logz_min, logz_max,
        n_layers, n_neurons, activation
    )
    value = BasicValueNetwork(
        k_min, k_max, logz_min, logz_max,
        n_layers, n_neurons, activation
    )

    # force build to initialize weights
    dummy_k = tf.constant([[1.0]], dtype=tf.float32)
    dummy_z = tf.constant([[1.0]], dtype=tf.float32)
    _ = policy(dummy_k, dummy_z)
    _ = value(dummy_k, dummy_z)

    return policy, value
