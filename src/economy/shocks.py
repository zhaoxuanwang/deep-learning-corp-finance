"""
src/economy/shocks.py

Handles the exogenous stochastic processes for the model.
Focuses on continuous sampling for Deep Learning (Direct Optimization).
"""

import tensorflow as tf
import numpy as np
import quantecon as qe
from typing import Optional, Tuple
from src.economy.parameters import EconomicParams
from typing import Union

from src.economy.parameters import EconomicParams
from typing import Union


def simulate_productivity_next(
        z_current: tf.Tensor,
        params: EconomicParams,
        epsilon: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Computes next period productivity z' given current z and parameters.
    Follows AR(1) process on log(z):

        log(z') = (1 - rho)*mu + rho * log(z) + sigma * epsilon

    Args:
        z_current (tf.Tensor): Current productivity levels (levels, not logs).
        params (EconomicParams): Contains rho, sigma, mu.
        epsilon (tf.Tensor, optional): Pre-drawn standard normal shocks.
                                       If None, draws new random shocks internally.

    Returns:
        tf.Tensor: Next period productivity z' (levels).
    """
    # 1. Ensure we have shocks
    if epsilon is None:
        # Generate shocks matching the shape of input z
        epsilon = tf.random.normal(shape=tf.shape(z_current), dtype=tf.float32)

    # 2. Extract parameters
    rho = params.rho
    sigma = params.sigma
    mu = params.mu

    # 3. Apply AR(1) in Log Space
    log_z = tf.math.log(tf.maximum(z_current, 1e-8))  # Safety floor

    log_z_next = (1 - rho) * mu + rho * log_z + sigma * epsilon

    # 4. Return to Levels
    return tf.exp(log_z_next)


def draw_initial_states(
        n_samples: int,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        params: EconomicParams,
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
        params (EconomicParams): Model parameters.
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
    ergodic_std = params.sigma / np.sqrt(1 - params.rho ** 2)

    log_z_init = tf.random.normal(
        shape=(n_samples,),
        mean= (1 - params.rho) * params.mu,
        stddev=ergodic_std,
        dtype=tf.float32
    )

    z_init = tf.exp(log_z_init)

    # Return in (z, k) order
    return z_init, k_init


def initialize_markov_process(
    params: EconomicParams, 
    z_size: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes the exogenous AR(1) shock process using Tauchen's method.

    Args:
        params (EconomicParams): The economic parameters containing rho, sigma, mu.
        z_size (int): Number of grid points for discretization.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - z_grid: 1D array of state values (productivity levels).
            - prob_matrix: 2D Transition probability matrix of shape (z_size, z_size).
    """
    mc = qe.tauchen(
        n=z_size,
        rho=params.rho,
        sigma=params.sigma,
        mu=params.mu,
        n_std=3  # Standard width coverage
    )

    z_grid = np.exp(mc.state_values)  # Convert log-states to levels
    prob_matrix = mc.P

    # Ensure strict row normalization (handling potential float precision issues)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

    return z_grid, prob_matrix


def get_sampling_bounds(
    params: EconomicParams,
    grid_config: "DDPGridConfig" = None
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Extracts the min/max bounds for DNN sampling using the existing grid logic.
    This ensures the DNN and DDP solve over the EXACT same state space.

    Args:
        params: Economic parameters.
        grid_config: Grid configuration (uses default if None).

    Returns:
        ((k_min, k_max), (b_min, b_max))
    """
    # Import here to avoid circular import
    from src.ddp.ddp_config import DDPGridConfig
    
    grid_config = grid_config or DDPGridConfig()
    
    # Generate capital grid using grid_config
    k_grid = grid_config.generate_capital_grid(params)
    k_min, k_max = float(k_grid[0]), float(k_grid[-1])

    # Get productivity bounds
    z_grid, _ = initialize_markov_process(params, grid_config.z_size)
    z_max = float(z_grid[-1])

    # Generate bond grid
    b_grid = grid_config.generate_bond_grid(params, k_max, z_max)
    b_min, b_max = float(b_grid[0]), float(b_grid[-1])

    return (k_min, k_max), (b_min, b_max)