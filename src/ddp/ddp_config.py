"""
DDP-specific grid configuration.

This module provides DDPGridConfig for DDP solver settings.
Keeps grid discretization separate from economic primitives.
"""

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from src.economy.parameters import EconomicParams


@dataclass
class DDPGridConfig:
    """
    DDP-specific grid configuration settings.
    
    This class holds discretization settings for the DDP (discrete dynamic
    programming) solver. These are NOT economic primitives — they are
    numerical settings for the solution method.
    
    Attributes:
        z_size: Number of productivity grid points
        k_size: Target number of capital grid points
        b_size: Number of bond/debt grid points
        grid_type: Capital grid generation method
    
    Example:
        config = DDPGridConfig(z_size=15, k_size=100, grid_type="delta_rule")
        k_grid = config.generate_capital_grid(params)
    """
    z_size: int = 15
    k_size: int = 100
    b_size: int = 80
    grid_type: Literal["delta_rule", "power_grid", "log_linear"] = "delta_rule"
    
    def __post_init__(self):
        """Validate grid settings."""
        valid_grid_types = {"delta_rule", "power_grid", "log_linear"}
        if self.grid_type not in valid_grid_types:
            raise ValueError(f"Unknown grid_type: {self.grid_type}. Valid options: {valid_grid_types}")
        
        if self.z_size < 2:
            raise ValueError(f"z_size must be >= 2. Got {self.z_size}")
        
        if self.k_size < 2:
            raise ValueError(f"k_size must be >= 2. Got {self.k_size}")
        
        if self.b_size < 2:
            raise ValueError(f"b_size must be >= 2. Got {self.b_size}")
    
    def compute_k_bounds(self, params: EconomicParams) -> Tuple[float, float]:
        """
        Compute capital grid bounds centered on steady state.
        
        Args:
            params: Economic parameters
        
        Returns:
            (k_min, k_max) tuple
        """
        k_ss = params.steady_state_k()
        k_min = max(k_ss * 0.1, 1e-6)
        k_max = k_ss * 2.0
        return k_min, k_max
    
    def generate_capital_grid(self, params: EconomicParams) -> np.ndarray:
        """
        Generate discretized capital grid.
        
        Args:
            params: Economic parameters
        
        Returns:
            1D array of capital grid points
        """
        k_min, k_max = self.compute_k_bounds(params)
        
        if self.grid_type == "delta_rule":
            density = 4
            multiplier = (1 / (1 - params.delta)) ** (1 / density)
            n_steps = np.log(k_max / k_min) / np.log(multiplier)
            actual_k_size = int(np.ceil(n_steps)) + 1
            return k_min * (multiplier ** np.arange(actual_k_size))
        
        elif self.grid_type == "power_grid":
            gamma = 2.0
            linear_steps = np.linspace(0, 1, self.k_size)
            return k_min + (k_max - k_min) * (linear_steps ** gamma)
        
        elif self.grid_type == "log_linear":
            return np.linspace(k_min, k_max, self.k_size)
        
        else:
            raise ValueError(f"Unknown grid_type: {self.grid_type}")
    
    def generate_bond_grid(
        self,
        params: EconomicParams,
        k_max: float,
        z_max: float
    ) -> np.ndarray:
        """
        Generate discretized bond/debt grid.
        
        Args:
            params: Economic parameters
            k_max: Maximum capital in grid
            z_max: Maximum productivity
        
        Returns:
            1D array of bond grid points
        """
        y_max = z_max * (k_max ** params.theta)
        
        collateral_value = params.frac_liquid * k_max
        tax_shield = params.tax * params.delta * k_max
        after_tax_profit = (1 - params.tax) * y_max
        
        b_max = after_tax_profit + tax_shield + collateral_value
        b_min = -1.5 * k_max
        
        return np.linspace(b_min, b_max, self.b_size)
