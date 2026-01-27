"""
src/economy/bounds.py

Utilities for computing natural sampling bounds based on economic parameters.
Used to determine safe regions for state space sampling (grids or DNN training).
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from src.economy.parameters import EconomicParams, ShockParams
import warnings

logger = logging.getLogger(__name__)


def compute_ergodic_log_z_bounds(
    shock_params: ShockParams, 
    std_dev_multiplier: float
) -> Tuple[float, float]:
    """
    Compute natural bounds for log(z) based on the ergodic distribution of the AR(1) process.
    
    Bounds = mu +/- m * sigma_ergodic
    where sigma_ergodic = sigma / sqrt(1 - rho^2).
    
    Args:
        params: Economic parameters
        std_dev_multiplier: Number of standard deviations from mean
    
    Returns:
        (min_log_z, max_log_z)
    """
    rho = shock_params.rho
    sigma = shock_params.sigma
    mu = shock_params.mu
    
    if abs(rho) >= 1.0:
        logger.warning(f"AR(1) rho={rho} is >= 1. Cannot compute ergodic std. Using default bounds (-1, 1).")
        return -1.0, 1.0
    
    std_ergodic = sigma / np.sqrt(1 - rho**2)
    
    min_log_z = mu - std_dev_multiplier * std_ergodic
    max_log_z = mu + std_dev_multiplier * std_ergodic
    
    return float(min_log_z), float(max_log_z)


def compute_natural_k_bounds(
    theta: float,
    r: float,
    delta: float,
    log_z_bounds: Tuple[float, float],
    k_min_multiplier: float,
    k_max_multiplier: float
) -> Tuple[float, float]:
    """
    Compute natural bounds for capital k based on steady-state logic.
    
    k* = ((z * theta) / (r + delta)) ^ (1/(1-theta))
    k_min = multiplier_min * k*(z_min)
    k_max = multiplier_max * k*(z_max)
    
    Args:
        theta: Production elasticity
        r: Risk-free rate
        delta: Depreciation rate
        log_z_bounds: (min_log_z, max_log_z)
        k_min_multiplier: Factor for lower bound
        k_max_multiplier: Factor for upper bound
    
    Returns:
        (k_min, k_max)
    """
    min_log_z, max_log_z = log_z_bounds
    
    z_min = np.exp(min_log_z)
    z_max = np.exp(max_log_z)

    # Helper for steady state k given z
    def get_k_star(z_val):
        # Prevent division by zero or negative base if parameters are extreme
        # (Though parameters should be validated in EconomicParams)
        denom = r + delta
        if denom <= 0:
            return 1.0 # Fallback
        return ((z_val * theta) / denom) ** (1 / (1 - theta))
    
    k_star_low = get_k_star(z_min)
    k_star_high = get_k_star(z_max)
    
    k_min = k_min_multiplier * k_star_low
    k_max = k_max_multiplier * k_star_high
    
    return float(k_min), float(k_max)


def compute_natural_b_bound(
    theta: float,
    k_max: float,
    z_max: float
) -> float:
    """
    Compute natural borrowing limit (B_max).
    
    B_max = z_max * (k_max)^theta + k_max
    
    Represents maximum repayable amount in best state with full liquidation.
    
    Args:
        theta: Production elasticity
        k_max: Maximum capital bound
        z_max: Maximum productivity (level, not log)
        
    Returns:
        b_max (float)
    """
    prod = z_max * (k_max ** theta)
    return float(prod + k_max)


def generate_states_bounds(
    theta: float,
    r: float,
    delta: float,
    shock_params: ShockParams,
    std_dev_multiplier: float,
    k_min_multiplier: float,
    k_max_multiplier: float
) -> Dict[str, Tuple[float, float]]:
    """
    Generate consistent sampling bounds for (k, b, z).
    
    The bounds are generated sequentially to ensure consistency:
    1. log_z_bounds: computed from ergodic distribution (mu +/- m*sigma_ergodic)
    2. k_bounds: computed from steady-state k*(z) at z_min/z_max using multipliers
    3. b_bounds: b_max computed from max repayment capacity at z_max, k_max
    
    Args:
        theta: Production elasticity
        r: Risk-free rate
        delta: Depreciation rate
        std_dev_multiplier: Multiplier for log z bounds
        k_min_multiplier: Multiplier for k_min
        k_max_multiplier: Multiplier for k_max
        
    Returns:
        SamplingBounds objects properly configured
    """
    # 1. Z bounds
    log_z_bounds = compute_ergodic_log_z_bounds(
        shock_params, 
        std_dev_multiplier=std_dev_multiplier
    )
    
    # 2. K bounds (depend on Z bounds)
    k_bounds = compute_natural_k_bounds(
        theta, r, delta,
        log_z_bounds,
        k_min_multiplier=k_min_multiplier,
        k_max_multiplier=k_max_multiplier
    )
    
    # 3. B bounds (depend on Z max and K max)
    # b_min is always 0.0 for now
    _, k_max = k_bounds
    _, log_z_max = log_z_bounds
    z_max = np.exp(log_z_max)
    
    b_max = compute_natural_b_bound(theta, k_max, z_max)
    b_bounds = (0.0, b_max)
    
    return {
        "k": k_bounds,
        "b": b_bounds,
        "log_z": log_z_bounds
    }