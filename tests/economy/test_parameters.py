"""
Unit tests for the configuration and parameter management module.

This module validates the economic consistency of the ModelParameters class,
verifies the correct generation of state space grids (capital, bond, productivity),
and ensures proper integration with TensorFlow data types.
"""

import pytest
import numpy as np
import tensorflow as tf
from dataclasses import replace

# Import the module under test
from src.economy import parameters as params_module
from src.economy.parameters import EconomicParams, ShockParams
from src.ddp import DDPGridConfig


# --- Fixtures ---

@pytest.fixture
def default_params():
    """
    Fixture providing a standard EconomicParams instance.

    Returns:
        EconomicParams: A fresh instance with default values.
    """
    return EconomicParams()


# --- 1. Parameter Validation Tests (__post_init__) ---
def test_input_validation_logic():
    """
    Test that the __post_init__ validator catches invalid economic values.
    """
    base = EconomicParams()
    shock_base = ShockParams()

    # Case 1: Negative Volatility (ShockParams now)
    with pytest.raises(ValueError, match="sigma"):
        replace(shock_base, sigma=-0.1)

    # Case 2: Production returns non-diminishing
    with pytest.raises(ValueError, match="theta"):
        replace(base, theta=1.1)

    # Case 3: Tax > 100%
    with pytest.raises(ValueError, match="tax"):
        replace(base, tax=1.5)

    # Case 4: Negative Costs
    with pytest.raises(ValueError, match="cost_convex"):
        replace(base, cost_convex=-5.0)


# --- 2. Capital Grid Generation Tests ---

def test_grid_delta_rule_structure(default_params):
    """
    Test the 'delta_rule' grid generation strategy.
    """
    # Use DDPGridConfig with delta_rule
    params = replace(default_params, delta=0.10)
    grid_config = DDPGridConfig(grid_type="delta_rule")
    
    k_grid = grid_config.generate_capital_grid(params)

    # Check 1: Geometric spacing consistency
    ratios = k_grid[1:] / k_grid[:-1]
    expected_multiplier = (1 / (1 - params.delta)) ** (1 / 4)

    assert np.allclose(ratios, expected_multiplier, rtol=1e-4), \
        "Delta rule grid is not geometrically spaced as expected."

    # Check 2: Dynamic Sizing
    assert len(k_grid) > 2, "Grid generation failed to produce points."
    assert k_grid[-1] > k_grid[0], "Grid is not increasing."


def test_grid_power_grid_structure(default_params):
    """
    Test the 'power_grid' generation strategy.
    """
    target_size = 50
    grid_config = DDPGridConfig(grid_type="power_grid", k_size=target_size)

    k_grid = grid_config.generate_capital_grid(default_params)

    # Check 1: Exact size match
    assert len(k_grid) == target_size, \
        f"Power grid did not respect k_size. Expected {target_size}, got {len(k_grid)}."

    # Check 2: Monotonicity
    assert np.all(np.diff(k_grid) > 0), "Power grid is not strictly increasing."


def test_grid_unknown_type(default_params):
    """Test that specifying an unknown grid type raises a ValueError."""
    with pytest.raises(ValueError):
        DDPGridConfig(grid_type="magic_grid")  # Should fail validation


# --- 3. Bond Grid Tests ---

def test_bond_grid_scaling(default_params):
    """
    Test that the bond grid scales automatically with the economy size.
    """
    grid_config = DDPGridConfig()
    
    k_max_small = 10.0
    k_max_large = 1000.0
    z_max = 1.0

    b_grid_small = grid_config.generate_bond_grid(default_params, k_max_small, z_max)
    b_grid_large = grid_config.generate_bond_grid(default_params, k_max_large, z_max)

    # The upper bound of the large economy should be substantially higher
    scale_factor = b_grid_large[-1] / b_grid_small[-1]

    assert scale_factor > 10.0, \
        "Bond grid did not scale up when capital stock increased."


# --- 4. Stochastic Process Tests (Moved to test_shocks.py) ---


# --- 5. TensorFlow Compatibility Tests ---

def test_tf_conversion_types():
    """
    Test the conversion of NumPy arrays to TensorFlow tensors.

    Crucially, this checks that the output dtype is tf.float32, which is
    standard for Deep Learning, even if the input was float64.
    """
    # Create float64 inputs (standard NumPy default)
    arr1 = np.array([1.0, 2.0], dtype=np.float64)
    arr2 = np.array([3.0, 4.0], dtype=np.float64)

    tf_out = params_module.convert_to_tf(arr1, arr2)

    # Check return structure
    assert isinstance(tf_out, list), "Output should be a list of tensors."
    assert len(tf_out) == 2

    # Check Data Type Casting
    assert tf_out[0].dtype == tf.float32, \
        f"Tensor should be float32, but got {tf_out[0].dtype}."