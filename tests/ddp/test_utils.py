import pytest
import numpy as np
import tensorflow as tf
from src.ddp.utils import ModelParameters, generate_capital_grid, initialize_markov_process, convert_to_tf


@pytest.fixture
def default_params():
    return ModelParameters()


def test_delta_rule_grid_monotonicity(default_params):
    """Test that Delta Rule produces a strictly increasing grid."""
    k_grid = generate_capital_grid(default_params)

    assert len(k_grid) > 0
    # Check strict monotonicity (k[i+1] > k[i])
    assert np.all(np.diff(k_grid) > 0)
    # Check bounds
    assert k_grid[0] > 0  # k_min > 0
    assert k_grid[-1] > k_grid[0]


def test_power_grid_size(default_params):
    """Test that power_grid respects the requested k_size."""
    # We must use replace() because ModelParameters is frozen
    params = ModelParameters(grid_type="power_grid", k_size=50)
    k_grid = generate_capital_grid(params)

    assert len(k_grid) == 50


def test_invalid_grid_type():
    """Test that invalid grid types raise ValueError."""
    params = ModelParameters(grid_type="non_existent_type")
    with pytest.raises(ValueError):
        generate_capital_grid(params)


def test_markov_process_properties(default_params):
    """Test the shape and probability properties of the Markov Chain."""
    z_grid, prob_matrix = initialize_markov_process(default_params)

    # Check shapes
    assert len(z_grid) == default_params.z_size
    assert prob_matrix.shape == (default_params.z_size, default_params.z_size)

    # Check probabilities sum to 1
    row_sums = np.sum(prob_matrix, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_markov_process_moments(default_params):
    """
    Verifies that the transition matrix respects both the AR(1) persistence
    and the volatility using conditional moments.

    1. Conditional Mean: E[log(z') | z] approx (1-rho)*mu + rho*log(z)
    2. Conditional Variance: Var(log(z') | z) approx sigma^2
    """
    productivity_grid, prob_matrix = initialize_markov_process(default_params)
    log_productivity = np.log(productivity_grid)

    # --- 1. Check Persistence (First Moment) ---
    # E[x] = sum(Prob_ij * x_j) -> Matrix multiplication
    conditional_mean = prob_matrix @ log_productivity

    theoretical_mean = (
            (1 - default_params.rho) * default_params.mu
            + default_params.rho * log_productivity
    )

    # Check if the trend matches rho
    np.testing.assert_allclose(
        conditional_mean,
        theoretical_mean,
        atol=0.05,
        err_msg="Transition matrix failed Persistence (rho) check"
    )

    # --- 2. Check Volatility (Second Moment) ---
    # Var(x) = E[x^2] - (E[x])^2
    conditional_second_moment = prob_matrix @ (log_productivity ** 2)
    conditional_variance = conditional_second_moment - (conditional_mean ** 2)

    theoretical_variance = default_params.sigma ** 2

    # We check the variance at the middle of the grid to avoid boundary bias
    middle_idx = default_params.z_size // 2

    np.testing.assert_allclose(
        conditional_variance[middle_idx],
        theoretical_variance,
        rtol=0.1,  # 10% relative tolerance
        err_msg="Transition matrix failed Volatility (sigma) check"
    )


def test_tf_conversion(default_params):
    """Test that NumPy arrays are correctly converted to TF tensors."""
    z_grid, prob_matrix = initialize_markov_process(default_params)
    k_grid = generate_capital_grid(default_params)

    z_tf, prob_matrix_tf, k_tf = convert_to_tf(z_grid, prob_matrix, k_grid)

    assert isinstance(z_tf, tf.Tensor)
    assert z_tf.dtype == tf.float32
    assert z_tf.shape == z_grid.shape