"""
src/economy/shocks.py

Handles the exogenous stochastic processes for the model.
Focuses on continuous sampling for Deep Learning (Direct Optimization).
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union

from src.economy.parameters import EconomicParams, ShockParams


def step_ar1_tf(
    z: tf.Tensor,
    rho: float,
    sigma: float,
    mu: float = 0.0,
    eps: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Single-step AR(1) transition for productivity z (TensorFlow).
    
    Process:
        log(z') = (1 - rho) * mu + rho * log(z) + sigma * eps
    
    Args:
        z: Current productivity (any shape)
        rho: AR(1) persistence
        sigma: AR(1) volatility
        mu: AR(1) unconditional mean of log(z)
        eps: Pre-drawn standard normal shocks (default: draws internally)
    
    Returns:
        z_next: Next period productivity (same shape as z)
    """
    log_z = tf.math.log(tf.maximum(z, 1e-8))
    
    if eps is None:
        eps = tf.random.normal(tf.shape(z), dtype=tf.float32)
    
    log_z_next = (1 - rho) * mu + rho * log_z + sigma * eps
    return tf.exp(log_z_next)


def step_ar1_numpy(
    z: Union[float, np.ndarray],
    rho: float,
    sigma: float,
    mu: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """
    Single-step AR(1) transition for productivity z (NumPy).
    
    Args:
        z: Current productivity
        rho: Persistence
        sigma: Volatility
        mu: Mean
        rng: Random number generator (defaults to np.random.default_rng())
        
    Returns:
        z_next
    """
    if rng is None:
        rng = np.random.default_rng()
        
    log_z = np.log(np.maximum(z, 1e-8))
    
    if isinstance(z, np.ndarray):
        eps = rng.standard_normal(size=z.shape)
    else:
        eps = rng.standard_normal()
        
    log_z_next = (1 - rho) * mu + rho * log_z + sigma * eps
    
    if isinstance(z, float):
        return float(np.exp(log_z_next))
    return np.exp(log_z_next)


def draw_AiO_shocks(
    n: int,
    z_curr: tf.Tensor,
    rho: float,
    sigma: float,
    mu: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Draw two independent next-period productivity shocks for AiO method.
    
    log(z') = (1 - rho) * mu + rho * log(z) + sigma * epsilon
    
    Args:
        n: Batch size
        z_curr: Current productivity (n,) or (n, 1)
        rho: AR(1) persistence
        sigma: AR(1) volatility
        mu: AR(1) mean
        rng: NumPy random generator (for reproducibility)
    
    Returns:
        z_next_1, z_next_2: Two independent z' draws
    """
    z_curr = tf.reshape(z_curr, [-1, 1])
    
    # Use step_ar1_tf logic but with controlled shocks
    # We manually generate shocks to ensure independence and support seed
    if rng is not None:
        eps1 = tf.constant(rng.standard_normal(size=(n, 1)), dtype=tf.float32)
        eps2 = tf.constant(rng.standard_normal(size=(n, 1)), dtype=tf.float32)
    else:
        eps1 = tf.random.normal((n, 1), dtype=tf.float32)
        eps2 = tf.random.normal((n, 1), dtype=tf.float32)
    
    z_next_1 = step_ar1_tf(z_curr, rho, sigma, mu, eps=eps1)
    z_next_2 = step_ar1_tf(z_curr, rho, sigma, mu, eps=eps2)
    
    return z_next_1, z_next_2



def draw_initial_states(
        n_samples: int,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        shock_params: ShockParams,
        previous_states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Determines the starting states (z, k) for a training batch.

    Supports two modes:
    1. Random Initialization (Cold Start): Draws from Ergodic Z and Uniform K.
    2. Continuous Simulation (Warm Start): Uses terminal states from previous batch.

    Args:
        n_samples (int): Batch size (required for random generation).
        bounds (Tuple): ((k_min, k_max), (b_min, b_max)).
        shock_params (ShockParams): Shock parameters.
        previous_states (tuple, optional): (z_last, k_last) from previous batch.
                                           If provided, these are returned immediately.

    Returns:
        (z_init, k_init): Tensors of shape (n_samples,)
    """
    # --- Mode 1: Continuous Simulation ---
    if previous_states is not None:
        z_prev, k_prev = previous_states
        # Optional: Add shape check here if needed debugging
        return z_prev, k_prev

    # --- Mode 2: Random Initialization ---
    # Unpack bounds
    (k_min, k_max), _ = bounds

    # 1. Draw Capital Uniformly (Exploration)
    k_init = tf.random.uniform(
        shape=(n_samples,),
        minval=k_min,
        maxval=k_max,
        dtype=tf.float32
    )

    # 2. Draw Productivity from Ergodic Distribution
    # The ergodic distribution of log(z) is N(mu, sigma^2 / (1 - rho^2))
    ergodic_std = shock_params.sigma / np.sqrt(1 - shock_params.rho ** 2)

    log_z_init = tf.random.normal(
        shape=(n_samples,),
        mean= (1 - shock_params.rho) * shock_params.mu,
        stddev=ergodic_std,
        dtype=tf.float32
    )

    z_init = tf.exp(log_z_init)

    # Return in (z, k) order
    return z_init, k_init


