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


# --- 2. Grid Generation Tests ---

def test_grid_multiplicative_capital_structure(default_params):
    """
    Multiplicative capital grid should use geometric spacing tied to depreciation.
    """
    params = replace(default_params, delta=0.10)
    grid_config = DDPGridConfig(capital_grid_type="multiplicative", z_size=7, b_size=9)
    bounds = {"k": (0.5, 5.0), "log_z": (-0.3, 0.3), "b": (0.0, 2.0)}

    k_grid, z_grid, b_grid = grid_config.generate_grids(bounds, delta=params.delta)

    ratios = k_grid[1:] / k_grid[:-1]
    expected_multiplier = 1.0 / (1.0 - params.delta)

    # Final step can be clipped to k_max, so check interior ratios.
    assert np.allclose(ratios[:-1], expected_multiplier, rtol=1e-4), \
        "Multiplicative grid is not geometrically spaced as expected."
    assert len(z_grid) == 7
    assert len(b_grid) == 9


def test_grid_linear_capital_structure():
    """
    Linear capital grid should respect k_size exactly via linspace.
    """
    target_size = 50
    grid_config = DDPGridConfig(capital_grid_type="linear", k_size=target_size)
    bounds = {"k": (1.0, 3.0), "log_z": (-0.2, 0.2), "b": (0.0, 1.0)}

    k_grid, _, _ = grid_config.generate_grids(bounds, delta=0.1)

    assert len(k_grid) == target_size
    assert np.allclose(k_grid, np.linspace(1.0, 3.0, target_size))
    assert np.all(np.diff(k_grid) > 0), "Linear capital grid is not strictly increasing."


def test_grid_unknown_type():
    """Test that specifying an unknown capital grid type raises a ValueError."""
    with pytest.raises(ValueError):
        DDPGridConfig(capital_grid_type="magic_grid")  # Should fail validation


def test_z_and_b_linear_spacing_invariant_to_capital_grid_type():
    """
    z (log-linear in levels) and b (linear) are fixed conventions regardless of capital grid type.
    """
    bounds = {"k": (0.8, 2.4), "log_z": (-0.3, 0.3), "b": (0.0, 4.0)}
    grid_mult = DDPGridConfig(capital_grid_type="multiplicative", z_size=5, b_size=6)
    grid_lin = DDPGridConfig(capital_grid_type="linear", z_size=5, b_size=6, k_size=9)

    _, z_mult, b_mult = grid_mult.generate_grids(bounds, delta=0.1)
    _, z_lin, b_lin = grid_lin.generate_grids(bounds, delta=0.1)

    expected_z = np.exp(np.linspace(-0.3, 0.3, 5))
    expected_b = np.linspace(0.0, 4.0, 6)
    assert np.allclose(z_mult, expected_z)
    assert np.allclose(z_lin, expected_z)
    assert np.allclose(b_mult, expected_b)
    assert np.allclose(b_lin, expected_b)


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
