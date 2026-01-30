"""
src/networks/network_risky.py

Neural network builders for Risky Debt Model (Sec. 2).
"""

import tensorflow as tf
from typing import Tuple

class RiskyPolicyNetwork(tf.keras.Model):
    """
    Policy network for Risky Debt model: (k, b, z) -> (k', b')

    Input transform: (log k, b/k, log z)
    Output activations:
        - k' = k_min + (k_max - k_min) * sigmoid(...)
        - b' = b_max * sigmoid(...)  [borrowing-only: b' >= 0]

    Args:
        k_min: Minimum capital
        k_max: Maximum capital
        b_min: Minimum debt (usually 0)
        b_max: Maximum debt
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation
    """

    def __init__(
        self,
        k_min: float,
        k_max: float,
        b_min: float,
        b_max: float,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()
        # Store config for serialization and cloning
        self._k_min = k_min
        self._k_max = k_max
        self._b_min = b_min
        self._b_max = b_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        self.k_min = tf.constant(k_min, dtype=tf.float32)
        self.k_max = tf.constant(k_max, dtype=tf.float32)
        self.b_min = tf.constant(b_min, dtype=tf.float32)
        self.b_max = tf.constant(b_max, dtype=tf.float32)

        # Shared hidden layers
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]

        # separate heads for k' and b'
        self.k_head = tf.keras.layers.Dense(1, activation=None, name="k_head")
        self.b_head = tf.keras.layers.Dense(1, activation=None, name="b_head")

    def get_config(self):
        """Return config for serialization."""
        return {
            "k_min": self._k_min,
            "k_max": self._k_max,
            "b_min": self._b_min,
            "b_max": self._b_max,
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
            "activation": self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create from config."""
        return cls(**config)
    
    def call(
        self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.
        
        Args:
            k: Capital stock (batch_size,) or (batch_size, 1), levels
            b: Debt (batch_size,) or (batch_size, 1), levels, b >= 0
            z: Productivity shock (batch_size,) or (batch_size, 1), levels
        
        Returns:
            k_next: Next period capital (batch_size, 1)
            b_next: Next period debt (batch_size, 1), b' >= 0
        """
        k = tf.reshape(k, [-1, 1])
        b = tf.reshape(b, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        
        # Transform: (log k, b/k, log z)
        log_k = tf.math.log(tf.maximum(k, 1e-8))
        safe_k = tf.maximum(k, 1e-8)
        leverage_ratio = b / safe_k
        log_z = tf.math.log(tf.maximum(z, 1e-8))
        
        x = tf.concat([log_k, leverage_ratio, log_z], axis=1)
        
        # Shared hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # k' = k_min + (k_max - k_min) * sigmoid(raw_k)
        raw_k = self.k_head(x)
        k_next = self.k_min + (self.k_max - self.k_min) * tf.nn.sigmoid(raw_k)
        
        # b' = b_max * sigmoid(raw_b)
        raw_b = self.b_head(x)
        b_next = self.b_max * tf.nn.sigmoid(raw_b)
        
        return k_next, b_next


class RiskyValueNetwork(tf.keras.Model):
    """
    Latent value network for Risky Debt model: (k, b, z) -> V_tilde(k, b, z)

    Input transform: (log k, b/k, log z)
    Output activation: linear (can be positive or negative)

    Args:
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation
    """

    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()
        # Store config for serialization and cloning
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def get_config(self):
        """Return config for serialization."""
        return {
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
            "activation": self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create from config."""
        return cls(**config)
    
    def call(self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            k: Capital stock (batch_size,) or (batch_size, 1), levels
            b: Debt (batch_size,) or (batch_size, 1), levels
            z: Productivity shock (batch_size,) or (batch_size, 1), levels
        
        Returns:
            V_tilde: Latent value (batch_size, 1)
        """
        k = tf.reshape(k, [-1, 1])
        b = tf.reshape(b, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        
        log_k = tf.math.log(tf.maximum(k, 1e-8))
        safe_k = tf.maximum(k, 1e-8)
        leverage_ratio = b / safe_k
        log_z = tf.math.log(tf.maximum(z, 1e-8))
        
        x = tf.concat([log_k, leverage_ratio, log_z], axis=1)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.output_layer(x)


class RiskyPriceNetwork(tf.keras.Model):
    """
    Bond pricing network for Risky Debt model: (k', b', z) -> q (bond price)

    Input transform: (log k', b'/k', log z)
    Output activation: q = (1/(1+r)) * sigmoid(...)

    Ensures q in [0, 1/(1+r)] where r is the risk-free rate.

    Reference:
        report_brief.md line 332: Bond price q(k',b',z) = (1/(1+r)) * Sigmoid(...)

    Args:
        r_risk_free: Risk-free interest rate
        n_layers: Number of hidden layers
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation
    """

    def __init__(
        self,
        r_risk_free: float,
        n_layers: int,
        n_neurons: int,
        activation: str
    ):
        super().__init__()
        # Store config for serialization and cloning
        self._r_risk_free = r_risk_free
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        self.r_risk_free = tf.constant(r_risk_free, dtype=tf.float32)
        # Maximum bond price = 1/(1+r) corresponds to risk-free lending
        self.q_max = tf.constant(1.0 / (1.0 + r_risk_free), dtype=tf.float32)

        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def get_config(self):
        """Return config for serialization."""
        return {
            "r_risk_free": self._r_risk_free,
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
            "activation": self._activation
        }

    @classmethod
    def from_config(cls, config):
        """Create from config."""
        return cls(**config)

    def call(
        self, k_next: tf.Tensor, b_next: tf.Tensor, z: tf.Tensor
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            k_next: Next period capital (batch_size,) or (batch_size, 1), levels
            b_next: Next period debt (batch_size,) or (batch_size, 1), levels
            z: Current productivity shock (batch_size,) or (batch_size, 1), levels

        Returns:
            q: Bond price (batch_size, 1), q in [0, 1/(1+r)]
        """
        k_next = tf.reshape(k_next, [-1, 1])
        b_next = tf.reshape(b_next, [-1, 1])
        z = tf.reshape(z, [-1, 1])

        log_k = tf.math.log(tf.maximum(k_next, 1e-8))
        safe_k = tf.maximum(k_next, 1e-8)
        leverage_ratio = b_next / safe_k
        log_z = tf.math.log(tf.maximum(z, 1e-8))

        x = tf.concat([log_k, leverage_ratio, log_z], axis=1)

        for layer in self.hidden_layers:
            x = layer(x)

        # q = q_max * sigmoid(raw) ensures q in [0, 1/(1+r)]
        # Reference: report_brief.md line 332
        raw = self.output_layer(x)
        q = self.q_max * tf.nn.sigmoid(raw)

        return q


def build_risky_networks(
    k_min: float,
    k_max: float,
    b_min: float,
    b_max: float,
    r_risk_free: float,
    n_layers: int,
    n_neurons: int,
    activation: str
) -> Tuple[RiskyPolicyNetwork, RiskyValueNetwork, RiskyPriceNetwork]:
    """Factory to build Risky Debt model networks."""
    policy = RiskyPolicyNetwork(k_min, k_max, b_min, b_max, n_layers, n_neurons, activation)
    value = RiskyValueNetwork(n_layers, n_neurons, activation)
    price = RiskyPriceNetwork(r_risk_free, n_layers, n_neurons, activation)
    return policy, value, price


def apply_limited_liability(
    V_tilde: tf.Tensor,
    leaky: bool = False,
    alpha: float = 0.01
) -> tf.Tensor:
    """
    Apply limited liability: V = max{0, V_tilde}.

    By default uses ReLU. Optionally use leaky ReLU to preserve
    non-zero gradients when V_tilde < 0 (useful for actor training).

    Note: For training, prefer compute_effective_value() which uses
    smooth Gumbel-Sigmoid approximation per report_brief.md.

    Args:
        V_tilde: Latent value (can be negative)
        leaky: If True, use leaky_relu instead of relu to preserve gradients
        alpha: Leaky ReLU slope for x < 0 (default 0.01)

    Returns:
        V: Actual value (V >= 0 for relu, V >= alpha*x for leaky)
    """
    if leaky:
        return tf.nn.leaky_relu(V_tilde, alpha=alpha)
    return tf.nn.relu(V_tilde)


def compute_effective_value(
    V_tilde: tf.Tensor,
    k: tf.Tensor,
    temperature: float,
    logit_clip: float = 20.0,
    noise: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute effective continuation value with smooth default probability.

    Implements V_eff = (1 - p) * V_tilde where p is the Gumbel-Sigmoid
    approximation of the default indicator.

    This avoids the dying ReLU problem in hard max{0, V} and provides
    gradient signal for the actor to escape default zones.

    Reference:
        report_brief.md lines 996-998:
        V_eff = (1 - p) * Gamma_value(k', b', z'; theta_value)

        report_brief.md lines 889-891 (Gumbel-Sigmoid):
        p = sigma((-V/k + log(u) - log(1-u)) / tau)

    Args:
        V_tilde: Latent value (batch_size, 1) - can be negative
        k: Capital for normalization (batch_size, 1)
        temperature: Annealing temperature tau (controls sigmoid sharpness)
        logit_clip: Clipping bound for normalized value (default 20.0)
        noise: If True, add Gumbel noise for exploration (training).
               If False, use deterministic sigmoid (evaluation).

    Returns:
        Tuple of:
            - V_eff: Effective value (batch_size, 1)
            - p_default: Default probability (batch_size, 1)
    """
    from src.utils.annealing import indicator_default

    # Normalize V by capital for numerical stability
    safe_k = tf.maximum(k, 1e-8)
    V_norm = V_tilde / safe_k

    # Compute default probability using Gumbel-Sigmoid
    p_default = indicator_default(V_norm, temperature, logit_clip=logit_clip, noise=noise)

    # Effective value: (1 - p) * V
    # When p -> 1 (default), V_eff -> 0
    # When p -> 0 (solvent), V_eff -> V_tilde
    V_eff = (1.0 - p_default) * V_tilde

    return V_eff, p_default
