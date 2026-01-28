"""
src/networks/network_basic.py

Neural network builders for Basic Model (Sec. 1).

Default Architecture (report_brief.md lines 341-357):
- 2 hidden layers with 16-32 neurons each
- SiLU (swish) activation for hidden layers
- Bounded sigmoid output for policy (k')
- Linear output for value (V)

Input Convention:
- Networks accept (k, z) in levels
- Internally transform to (log k, log z) for numerical stability
"""

import tensorflow as tf
from typing import Tuple

class BasicPolicyNetwork(tf.keras.Model):
    """
    Policy network for the Basic model: (k, z) -> k'
    
    Input transform: (log k, log z)
    Output activation: k' = k_min + softplus(raw_output)
    
    Args:
        k_min: Minimum capital
        k_max: Maximum capital
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation
    """
    
    def __init__(
        self,
        k_min: float,
        k_max: float,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()
        self.k_min = tf.constant(k_min, dtype=tf.float32)
        self.k_max = tf.constant(k_max, dtype=tf.float32)

        # Save config parameters for serialization
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

        # Transform: (log k, log z)
        log_k = tf.math.log(tf.maximum(k, 1e-8))
        log_z = tf.math.log(tf.maximum(z, 1e-8))
        x = tf.concat([log_k, log_z], axis=1)

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
            'k_min': float(self.k_min),
            'k_max': float(self.k_max),
            'n_layers': self._n_layers,
            'n_neurons': self._n_neurons,
            'activation': self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)


class BasicValueNetwork(tf.keras.Model):
    """
    Value network for the Basic model: (k, z) -> V(k, z)
    
    Input transform: (log k, log z)
    Output activation: linear (unbounded)
    
    Args:
        n_layers: Number of hidden layers (default 2)
        n_neurons: Neurons per hidden layer (default 16)
        activation: Hidden layer activation (default 'tanh')
    """
    
    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()

        # Save config parameters for serialization
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

        log_k = tf.math.log(tf.maximum(k, 1e-8))
        log_z = tf.math.log(tf.maximum(z, 1e-8))
        x = tf.concat([log_k, log_z], axis=1)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)

    def get_config(self):
        """Return configuration for serialization."""
        return {
            'n_layers': self._n_layers,
            'n_neurons': self._n_neurons,
            'activation': self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)


def build_basic_networks(
    k_min: float,
    k_max: float,
    n_layers: int,
    n_neurons: int,
    activation: str
) -> Tuple[BasicPolicyNetwork, BasicValueNetwork]:
    """Factory to build Basic model networks."""
    policy = BasicPolicyNetwork(k_min, k_max, n_layers, n_neurons, activation)
    value = BasicValueNetwork(n_layers, n_neurons, activation)

    # force build to initialize weights
    dummy_k = tf.constant([[1.0]], dtype=tf.float32)
    dummy_z = tf.constant([[1.0]], dtype=tf.float32)
    _ = policy(dummy_k, dummy_z)
    _ = value(dummy_k, dummy_z)

    return policy, value
