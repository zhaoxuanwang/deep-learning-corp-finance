"""
src/dnn/networks.py

Neural network builders for policy, value, and price functions.
Implements exact input transforms and output activations from outline_v2.md.

Basic Model:
- Policy: (k,z) -> k' via (log k, log z) -> k_min + softplus(...)
- Value: (k,z) -> V via (log k, log z) -> linear(...)

Risky Debt Model:
- Policy: (k,b,z) -> (k', b') via (log k, b/k, log z)
- Value: (k,b,z) -> V_tilde via (log k, b/k, log z) -> linear(...)
- Price: (k',b',z) -> r_tilde via (log k', b'/k', log z) -> r + softplus(...)
"""

import tensorflow as tf
from typing import Tuple, Optional


# =============================================================================
# BASIC MODEL NETWORKS (Sec. 1)
# =============================================================================

class BasicPolicyNetwork(tf.keras.Model):
    """
    Policy network for the Basic model: (k, z) -> k'
    
    Input transform: (log k, log z)
    Output activation: k' = k_min + softplus(raw_output)
    
    Args:
        k_min: Minimum capital (default 1e-4)
        n_layers: Number of hidden layers (default 2)
        n_neurons: Neurons per hidden layer (default 16)
        activation: Hidden layer activation (default 'tanh')
    """
    
    def __init__(
        self,
        k_min: float = 1e-4,
        n_layers: int = 2,
        n_neurons: int = 16,
        activation: str = "tanh"
    ):
        super().__init__()
        self.k_min = tf.constant(k_min, dtype=tf.float32)
        
        # Hidden layers
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]
        
        # Raw output (no activation, we apply k_min + softplus)
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
        
        # Output: k' = k_min + softplus(raw)
        raw = self.output_layer(x)
        k_next = self.k_min + tf.nn.softplus(raw)
        
        return k_next


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
        n_layers: int = 2,
        n_neurons: int = 16,
        activation: str = "tanh"
    ):
        super().__init__()
        
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


# =============================================================================
# RISKY DEBT MODEL NETWORKS (Sec. 2)
# =============================================================================

class RiskyPolicyNetwork(tf.keras.Model):
    """
    Policy network for Risky Debt model: (k, b, z) -> (k', b')
    
    Input transform: (log k, b/k, log z)
    Output activations:
        - k' = k_min + softplus(...)
        - b' = k * leverage_scale * sigmoid(...)  [borrowing-only: b' >= 0]
    
    Uses CURRENT k for b' scaling (not k'), per spec line 84.
    
    Args:
        k_min: Minimum capital (default 1e-4)
        leverage_scale: Maximum b'/k ratio (default 1.0)
        n_layers: Number of hidden layers (default 2)
        n_neurons: Neurons per hidden layer (default 16)
        activation: Hidden layer activation (default 'tanh')
    """
    
    def __init__(
        self,
        k_min: float = 1e-4,
        leverage_scale: float = 1.0,
        n_layers: int = 2,
        n_neurons: int = 16,
        activation: str = "tanh"
    ):
        super().__init__()
        self.k_min = tf.constant(k_min, dtype=tf.float32)
        self.leverage_scale = tf.constant(leverage_scale, dtype=tf.float32)
        
        # Shared hidden layers
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]
        
        # Separate heads for k' and b'
        self.k_head = tf.keras.layers.Dense(1, activation=None, name="k_head")
        self.b_head = tf.keras.layers.Dense(1, activation=None, name="b_head")
    
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
        
        # k' = k_min + softplus(raw_k)
        raw_k = self.k_head(x)
        k_next = self.k_min + tf.nn.softplus(raw_k)
        
        # b' = k * leverage_scale * sigmoid(raw_b)
        # Uses CURRENT k for scaling (not k')
        raw_b = self.b_head(x)
        b_next = k * self.leverage_scale * tf.nn.sigmoid(raw_b)
        
        return k_next, b_next


class RiskyValueNetwork(tf.keras.Model):
    """
    Latent value network for Risky Debt model: (k, b, z) -> V_tilde(k, b, z)
    
    Input transform: (log k, b/k, log z)
    Output activation: linear (can be positive or negative)
    
    Args:
        n_layers: Number of hidden layers (default 2)
        n_neurons: Neurons per hidden layer (default 16)
        activation: Hidden layer activation (default 'tanh')
    """
    
    def __init__(
        self,
        n_layers: int = 2,
        n_neurons: int = 16,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]
        
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
    
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
    Bond pricing network for Risky Debt model: (k', b', z) -> r_tilde
    
    Input transform: (log k', b'/k', log z)
    Output activation: r_tilde = r_risk_free + softplus(...)
    
    Ensures r_tilde >= r_risk_free (risky rate lower bound).
    
    Args:
        r_risk_free: Risk-free interest rate (default 0.04)
        n_layers: Number of hidden layers (default 2)
        n_neurons: Neurons per hidden layer (default 16)
        activation: Hidden layer activation (default 'tanh')
    """
    
    def __init__(
        self,
        r_risk_free: float = 0.04,
        n_layers: int = 2,
        n_neurons: int = 16,
        activation: str = "tanh"
    ):
        super().__init__()
        self.r_risk_free = tf.constant(r_risk_free, dtype=tf.float32)
        
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=activation)
            for _ in range(n_layers)
        ]
        
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
    
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
            r_tilde: Risky interest rate (batch_size, 1), r_tilde >= r_risk_free
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
        
        # r_tilde = r + softplus(raw)
        raw = self.output_layer(x)
        r_tilde = self.r_risk_free + tf.nn.softplus(raw)
        
        return r_tilde


# =============================================================================
# LIMITED LIABILITY HELPER
# =============================================================================

def apply_limited_liability(
    V_tilde: tf.Tensor,
    leaky: bool = False,
    alpha: float = 0.01
) -> tf.Tensor:
    """
    Apply limited liability: V = max{0, V_tilde}.
    
    By default uses ReLU. Optionally use leaky ReLU to preserve
    non-zero gradients when V_tilde < 0 (useful for actor training).
    
    Per spec line 156-157: Use relu explicitly for subgradient behavior.
    
    Args:
        V_tilde: Latent value (can be negative)
        leaky: If True, use leaky_relu instead of relu to preserve gradients
        alpha: Leaky ReLU slope for x < 0 (default 0.01)
    
    Returns:
        V: Actual value (V >= 0 for relu, V >= alpha*x for leaky)
    
    Note:
        For actor training, `leaky=True` can help when most V_tilde < 0,
        by providing non-zero gradients through the continuation term.
        For critic training and final evaluation, use `leaky=False` (default).
    """
    if leaky:
        return tf.nn.leaky_relu(V_tilde, alpha=alpha)
    return tf.nn.relu(V_tilde)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_basic_networks(
    k_min: float = 1e-4,
    n_layers: int = 2,
    n_neurons: int = 16,
    activation: str = "tanh"
) -> Tuple[BasicPolicyNetwork, BasicValueNetwork]:
    """Factory to build Basic model networks."""
    policy = BasicPolicyNetwork(k_min, n_layers, n_neurons, activation)
    value = BasicValueNetwork(n_layers, n_neurons, activation)
    return policy, value


def build_risky_networks(
    k_min: float = 1e-4,
    leverage_scale: float = 1.0,
    r_risk_free: float = 0.04,
    n_layers: int = 2,
    n_neurons: int = 16,
    activation: str = "tanh"
) -> Tuple[RiskyPolicyNetwork, RiskyValueNetwork, RiskyPriceNetwork]:
    """Factory to build Risky Debt model networks."""
    policy = RiskyPolicyNetwork(k_min, leverage_scale, n_layers, n_neurons, activation)
    value = RiskyValueNetwork(n_layers, n_neurons, activation)
    price = RiskyPriceNetwork(r_risk_free, n_layers, n_neurons, activation)
    return policy, value, price
