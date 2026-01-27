
import pytest
import tensorflow as tf
import numpy as np
from dataclasses import replace
from src.economy import shocks
from src.economy.parameters import EconomicParams, ShockParams

from src.economy.parameters import EconomicParams

# --- Fixtures ---

@pytest.fixture
def params():
    """Default shock parameters."""
    return ShockParams(
        rho=0.9,
        sigma=0.01,
        mu=0.0
    )

def test_simulate_productivity_shape(params):
    """Ensure output z' has the same shape as input z."""
    z_current = tf.ones((10, 1))
    z_next = shocks.step_ar1_tf(z_current, params.rho, params.sigma, params.mu)
    
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
    
    z_next = shocks.step_ar1_tf(z_current, params.rho, params.sigma, params.mu, eps=epsilon)
    
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






def test_step_ar1_tf_deterministic(params):
    """Test step_ar1_tf with zero shock (deterministic)."""
    # log(z) = 0 => z=1
    z = tf.constant(1.0)
    # log(z') = (1-rho)*0 + rho*0 + sigma*0 = 0 => z'=1
    z_next = shocks.step_ar1_tf(z, params.rho, params.sigma, params.mu, eps=tf.constant(0.0))
    assert abs(z_next - 1.0) < 1e-6


def test_step_ar1_numpy_deterministic(params):
    """Test step_ar1_numpy with zero shock (deterministic)."""
    # Mock RNG that returns 0
    class MockRNG:
        def standard_normal(self, size=None):
            if size is None: return 0.0
            return np.zeros(size)
    
    z = 1.0
    z_next = shocks.step_ar1_numpy(z, params.rho, params.sigma, params.mu, rng=MockRNG())
    assert abs(z_next - 1.0) < 1e-6


def test_draw_shocks_independent(params):
    """Test draw_AiO_shocks returns two independent draws."""
    z = tf.constant([1.0, 1.0, 1.0])
    z1, z2 = shocks.draw_AiO_shocks(3, z, params.rho, params.sigma, params.mu)
    
    # Should be different
    assert not np.allclose(z1.numpy(), z2.numpy())
    
    # Check shapes
    assert z1.shape == (3, 1)
    assert z2.shape == (3, 1)
