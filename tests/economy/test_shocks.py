
import pytest
import tensorflow as tf
import numpy as np
from dataclasses import replace
from src.economy import shocks
from src.economy.parameters import EconomicParams

from src.economy.parameters import EconomicParams

# --- Fixtures ---

@pytest.fixture
def params():
    """Default test parameters."""
    return EconomicParams(
        rho=0.9,
        sigma=0.01,
        mu=0.0
    )

def test_simulate_productivity_shape(params):
    """Ensure output z' has the same shape as input z."""
    z_current = tf.ones((10, 1))
    z_next = shocks.simulate_productivity_next(z_current, params)
    
    assert z_next.shape == (10, 1)

def test_simulate_productivity_values(params):
    """
    Check AR(1) calculation with a fixed epsilon.
    log(z') = (1-rho)*mu + rho*log(z) + sigma*epsilon
    
    Let mu=0, z=1 (log_z=0).
    Then log(z') = 0 + 0 + sigma*epsilon
    z' = exp(sigma*epsilon)
    """
    z_current = tf.constant(1.0) # log(z) = 0
    epsilon = tf.constant(2.0)   # 2 std devs
    
    expected_log_z = params.sigma * 2.0
    expected_z = np.exp(expected_log_z)
    
    z_next = shocks.simulate_productivity_next(z_current, params, epsilon=epsilon)
    
    assert np.isclose(z_next, expected_z)

def test_draw_initial_states_distribution(params):
    """Verify statistical properties of the random draw."""
    n = 10000
    bounds = ((1.0, 10.0), (0.0, 0.0))
    
    z_init, k_init = shocks.draw_initial_states(n, bounds, params)
    
    # 1. Check Dimensions
    assert z_init.shape == (n,)
    assert k_init.shape == (n,)
    
    # 2. Check Z stats (Ergodic Log-Normal)
    # log(z) ~ N( (1-rho)mu, sigma^2 / (1-rho^2) )
    # With mu=0, mean log(z) should be near 0
    log_z = np.log(z_init.numpy())
    mean_log = np.mean(log_z)
    
    # Standard Error of Mean = std / sqrt(n). Allow 3-sigma tolerance.
    expected_std = params.sigma / np.sqrt(1 - params.rho**2)
    sem = expected_std / np.sqrt(n)
    
    assert np.abs(mean_log) < 3.0 * sem + 0.05 # Add margin for randomness
    
    # 3. Check K stats (Uniform)
    assert np.min(k_init) >= 1.0
    assert np.max(k_init) <= 10.0
    mean_k = np.mean(k_init)
    assert np.abs(mean_k - 5.5) < 0.1

def test_draw_initial_states_continuous(params):
    """Verify 'warm start' simply returns the previous states."""
    prev_z = tf.constant([1.5, 2.5])
    prev_k = tf.constant([5.0, 6.0])
    previous_states = (prev_z, prev_k)
    bounds = ((0,0), (0,0))
    
    z_out, k_out = shocks.draw_initial_states(10, bounds, params, previous_states=previous_states)
    
    assert z_out is prev_z
    assert k_out is prev_k

def test_markov_process_integrity(params):
    """
    Test the integrity of the Tauchen discretization method (moved from test_parameters).
    """
    z_size = 15  # Default z_size for test
    z_grid, prob_matrix = shocks.initialize_markov_process(params, z_size)

    # Check 1: Row Sums (Normalization)
    row_sums = np.sum(prob_matrix, axis=1)
    assert np.allclose(row_sums, 1.0), \
        "Transition probability matrix rows do not sum to 1.0."

    # Check 2: Domain (Productivity levels must be positive)
    assert np.all(z_grid > 0), "Productivity grid contains non-positive values."

    # Check 3: Dimensions
    assert len(z_grid) == z_size
    assert prob_matrix.shape == (z_size, z_size)

def test_sampling_bounds_structure(params):
    """Verify get_sampling_bounds returns correct structure."""
    bounds = shocks.get_sampling_bounds(params)
    (k_range, b_range) = bounds
    
    # Check K bounds
    assert len(k_range) == 2
    assert k_range[1] > k_range[0]
    
    # Check B bounds
    assert len(b_range) == 2
    assert b_range[1] > b_range[0]
