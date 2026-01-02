"""
Utility functions and configuration classes for the investment model.

This module handles the initialization of economic parameters,
discretization of state spaces (capital and productivity), and
integration with TensorFlow for the main solving engine.
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import quantecon as qe
import tensorflow as tf


@dataclass(frozen=True)
class ModelParameters:
    """
    Immutable container for economic parameters.

    Attributes:
        r_rate (float): Risk-free interest rate.
        delta (float): Depreciation rate.
        theta (float): Production elasticity/returns to scale (e.g., 0.7).
        cost_convex (float): Coefficient for convex adjustment costs.
        cost_fixed (float): Coefficient for fixed adjustment costs.
        rho (float): Persistence of the AR(1) shock process.
        sigma (float): Volatility (std dev) of the AR(1) shock process.
        mu (float): Unconditional mean of the shock process.
        z_size (int): Number of discretization points for the shock.
        k_size (int): Target number of grid points for capital stock.
        grid_type (str): Method to generate capital grid.
            Options: "delta_rule", "power_grid", "log_linear".
        b_size (int): Size of bond stock grid (Risky Debt Model).
        cost_default (float): Coefficient for default bond adjustment costs (alpha).
        tax (float): Corporate Income Tax Rate.
        frac_liquid (float): Fraction of capital that can be liquidated to repay debt.
        cost_inject_fixed (float): Fixed cost of external finance through equity injection.
        cost_inject_linear (float): Proportional cost of external finance.
    """
    # Basic Model Parameters
    r_rate: float = 0.04
    delta: float = 0.15
    theta: float = 0.7
    cost_convex: float = 0.01
    cost_fixed: float = 0.01
    rho: float = 0.7
    sigma: float = 0.15
    mu: float = 0.0
    z_size: int = 15
    k_size: int = 100
    grid_type: Literal["delta_rule", "power_grid", "log_linear"] = "delta_rule"

    # Risky Debt Model Parameters
    b_size: int = 80
    cost_default: float = 0.4
    tax: float = 0.3
    frac_liquid: float = 0.5
    cost_inject_fixed: float = 0.01
    cost_inject_linear: float = 0.01

    def __post_init__(self):
        """Validates parameters immediately after initialization."""
        # Enforce minimum number of grids
        if self.z_size < 2:
            raise ValueError(f"Number of productivity grids should be at least 2. Got {self.z_size}")

        if self.k_size < 2:
            raise ValueError(f"Number of capital grids should be at least 2. Got {self.k_size}")

        if self.b_size < 2:
            raise ValueError(f"Number of debt grids should be at least 2. Got {self.b_size}")

        # Enforce Risk-Free Rate bounds (0 < r < 1)
        if not (0.0 < self.r_rate < 1.0):
            raise ValueError(f"Depreciation rate (delta) must be between 0 and 1. Got {self.delta}")

        # Enforce Depreciation bounds (0 <= delta <= 1)
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"Depreciation rate (delta) must be between 0 and 1. Got {self.delta}")

        # Enforce Capital Adjustment Cost bounds (0 <= Coef)
        if self.cost_convex < 0:
            raise ValueError(f"Capital adjustment cost coefficients must be non-negative. Got {self.cost_convex}")

        if self.cost_fixed < 0:
            raise ValueError(f"Capital adjustment cost coefficients must be non-negative. Got {self.cost_fixed}")

        # Enforce Production Function Curvature (0 < theta < 1)
        if not (0.0 < self.theta < 1.0):
            raise ValueError(f"Theta must be strictly between 0 and 1. Got {self.theta}")

        # Enforce Volatility (sigma > 0)
        if self.sigma <= 0:
            raise ValueError(f"Volatility (sigma) must be positive. Got {self.sigma}")

        # Enforce Persistence (-1 < rho < 1)
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"Persistence (rho) must be between -1 and 1. Got {self.rho}")

        # Enforce Risky Debt Parameters
        if not (0.0 <= self.cost_default <= 1.0):
            raise ValueError(f"Coefficient for default costs must be between 0 and 1. ")

        if not (0.0 <= self.frac_liquid <= 1.0):
            raise ValueError(f"Max fraction of liquidated asset must be between 0 and 1. ")

        if not (0.0 <= self.tax <= 1.0):
            raise ValueError(f"Tax must be strictly between 0 and 1. Got {self.tax}")

        if not (0.0 <= self.cost_inject_fixed):
            raise ValueError(
                f"Cost coefficient of external financing must not be negative {self.cost_inject_fixed}"
            )

        if not (0.0 <= self.cost_inject_linear):
            raise ValueError(
                f"Cost coefficient of external financing must not be negative {self.cost_inject_linear}"
            )


def generate_capital_grid(params: ModelParameters) -> np.ndarray:
    """
    Generates a discretization grid for capital stock (k).

    Args:
        params (ModelParameters): The model configuration object.

    Returns:
        np.ndarray: A 1D array representing the capital grid.
    """

    # Center grids around Steady State capital stock (approximate)
    # User cost of capital ~ (r + delta)
    user_cost = params.r_rate + params.delta
    k_ss = (params.theta / user_cost) ** (1 / (1 - params.theta))

    # Define Boundaries relative to Steady State
    # We ensure k_min is never exactly 0 to avoid division by zero in cost functions
    k_min = max(k_ss * 0.1, 1e-6)
    k_max = k_ss * 2.0

    # Generate Grid based on strategy
    if params.grid_type == "delta_rule":
        # The 'Delta Rule' reduces interpolation error by aligning grid points
        # with the natural depreciation path: k_next = (1-delta)*k
        density = 4  # Points per unit of depreciation
        multiplier = (1 / (1 - params.delta)) ** (1 / density)

        # Calculate required steps to span the range
        n_steps = np.log(k_max / k_min) / np.log(multiplier)
        actual_k_size = int(np.ceil(n_steps)) + 1

        return k_min * (multiplier ** np.arange(actual_k_size))

    elif params.grid_type == "power_grid":
        # Places more points near the lower bound (high curvature area)
        gamma = 2.0
        linear_steps = np.linspace(0, 1, params.k_size)
        return k_min + (k_max - k_min) * (linear_steps ** gamma)

    elif params.grid_type == "log_linear":
        # Standard uniform grid
        return np.linspace(k_min, k_max, params.k_size)

    else:
        raise ValueError(f"Unknown grid_type: {params.grid_type}")


def initialize_markov_process(params: ModelParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes the exogenous AR(1) shock process using Tauchen's method.

    Args:
        params (ModelParameters): The model configuration containing rho, sigma, mu.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - z_grid: 1D array of state values (productivity levels).
            - prob_matrix: 2D Transition probability matrix of shape (z_size, z_size).
    """
    mc = qe.tauchen(
        n=params.z_size,
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


def generate_bond_grid(params: ModelParameters, k_max: float, z_max: float) -> np.ndarray:
    """
    Generates the debt grid based on the scale of the economy.

    This ensures the bond grid scales automatically if capital (k) or
    productivity (z) scales change.

    Args:
        params (ModelParameters): The model configuration.
        k_max (float): Maximum capital stock in the grid.
        z_max (float): Maximum productivity shock in the grid.

    Returns:
        np.ndarray: A 1D array representing the bond/debt grid.
    """
    # Unpack Economic Parameters
    delta = params.delta
    theta = params.theta
    tax = params.tax
    frac_liquid = params.frac_liquid

    # Calculate the Economic Scale (Max possible output)
    y_max = z_max * (k_max ** theta)

    # Upper Bound (Maximum Debt)
    # Logic: The max debt a firm can possibly repay if it liquidates everything
    # and has the best possible cash flow shock.
    # Formula: After-Tax Cash Flow + Tax Shield + Liquidation Value
    collateral_value = frac_liquid * k_max
    tax_shield = tax * delta * k_max
    after_tax_profit = (1 - tax) * y_max

    b_max = after_tax_profit + tax_shield + collateral_value

    # Lower Bound (Maximum Savings)
    # Logic: Allow firm to save enough that the constraint rarely binds.
    # Using k_max ensures the savings capacity scales with the size of the firm.
    b_min = -1.5 * k_max

    return np.linspace(b_min, b_max, params.b_size)


def convert_to_tf(*args: np.ndarray) -> List[tf.Tensor]:
    """
    Converts NumPy arrays into TensorFlow constants.

    This function is flexible and can handle varying numbers of input arrays,
    useful for switching between Baseline and Risky Debt models.

    Args:
        *args: Variable number of NumPy arrays.

    Returns:
        List[tf.Tensor]: A list of TensorFlow constants corresponding to input arrays.
    """
    return [tf.constant(arg, dtype=tf.float32) for arg in args]