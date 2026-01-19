"""
tests/dnn/test_networks.py

Shape tests, output domain tests, and determinism tests for all networks.

Reference: outline_v2.md network specifications
"""

import pytest
import tensorflow as tf
import numpy as np

from src.dnn.networks import (
    BasicPolicyNetwork,
    BasicValueNetwork,
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    apply_limited_liability,
    build_basic_networks,
    build_risky_networks
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def basic_policy():
    return BasicPolicyNetwork(k_min=0.01, n_layers=2, n_neurons=16)


@pytest.fixture
def basic_value():
    return BasicValueNetwork(n_layers=2, n_neurons=16)


@pytest.fixture
def risky_policy():
    return RiskyPolicyNetwork(k_min=0.01, leverage_scale=1.5, n_layers=2, n_neurons=16)


@pytest.fixture
def risky_value():
    return RiskyValueNetwork(n_layers=2, n_neurons=16)


@pytest.fixture
def risky_price():
    return RiskyPriceNetwork(r_risk_free=0.04, n_layers=2, n_neurons=16)


# =============================================================================
# BASIC MODEL SHAPE TESTS
# =============================================================================

class TestBasicPolicyNetwork:
    """Tests for BasicPolicyNetwork."""
    
    def test_output_shape(self, basic_policy):
        """Policy output should be (batch, 1)."""
        k = tf.ones((16,))
        z = tf.ones((16,))
        k_next = basic_policy(k, z)
        assert k_next.shape == (16, 1)
    
    def test_output_shape_2d_input(self, basic_policy):
        """Should handle (batch, 1) inputs."""
        k = tf.ones((16, 1))
        z = tf.ones((16, 1))
        k_next = basic_policy(k, z)
        assert k_next.shape == (16, 1)
    
    def test_k_next_greater_than_kmin(self, basic_policy):
        """k' must be > k_min (enforced by k_min + softplus)."""
        # Test with various inputs including extreme values
        k = tf.constant([0.1, 1.0, 10.0, 100.0])
        z = tf.constant([0.5, 1.0, 2.0, 0.1])
        k_next = basic_policy(k, z)
        
        k_min = basic_policy.k_min.numpy()
        assert np.all(k_next.numpy() > k_min), f"k' must be > {k_min}"
    
    def test_k_next_positive_extreme_inputs(self, basic_policy):
        """k' should be positive even with extreme (near-zero) inputs."""
        k = tf.constant([1e-6, 1e-6])
        z = tf.constant([1e-6, 100.0])
        k_next = basic_policy(k, z)
        
        assert np.all(k_next.numpy() > 0), "k' must be positive"


class TestBasicValueNetwork:
    """Tests for BasicValueNetwork."""
    
    def test_output_shape(self, basic_value):
        """Value output should be (batch, 1)."""
        k = tf.ones((16,))
        z = tf.ones((16,))
        V = basic_value(k, z)
        assert V.shape == (16, 1)
    
    def test_output_can_be_negative(self, basic_value):
        """Value can be negative (linear activation)."""
        # Force network to produce negative by checking activation
        assert basic_value.output_layer.activation is None or \
               basic_value.output_layer.activation == tf.keras.activations.linear


# =============================================================================
# RISKY DEBT MODEL SHAPE TESTS
# =============================================================================

class TestRiskyPolicyNetwork:
    """Tests for RiskyPolicyNetwork."""
    
    def test_output_shapes(self, risky_policy):
        """Policy outputs (k', b') should both be (batch, 1)."""
        k = tf.ones((16,))
        b = tf.ones((16,)) * 0.5
        z = tf.ones((16,))
        k_next, b_next = risky_policy(k, b, z)
        
        assert k_next.shape == (16, 1)
        assert b_next.shape == (16, 1)
    
    def test_k_next_greater_than_kmin(self, risky_policy):
        """k' must be > k_min."""
        k = tf.constant([0.1, 1.0, 10.0])
        b = tf.constant([0.0, 0.5, 1.0])
        z = tf.constant([0.8, 1.0, 1.2])
        k_next, _ = risky_policy(k, b, z)
        
        k_min = risky_policy.k_min.numpy()
        assert np.all(k_next.numpy() > k_min)
    
    def test_b_next_non_negative(self, risky_policy):
        """b' >= 0 (borrowing-only constraint via sigmoid)."""
        k = tf.constant([0.1, 1.0, 10.0])
        b = tf.constant([0.0, 0.5, 1.0])
        z = tf.constant([0.8, 1.0, 1.2])
        _, b_next = risky_policy(k, b, z)
        
        assert np.all(b_next.numpy() >= 0), "b' must be >= 0 (borrowing-only)"
    
    def test_b_next_uses_current_k_for_scaling(self, risky_policy):
        """b' should scale with current k, not k'."""
        # Check that b' = k * leverage_scale * sigmoid(...)
        # Max b' should be approximately k * leverage_scale
        k = tf.constant([1.0, 2.0, 5.0])
        b = tf.zeros((3,))
        z = tf.ones((3,))
        _, b_next = risky_policy(k, b, z)
        
        leverage_scale = risky_policy.leverage_scale.numpy()
        max_b = k.numpy() * leverage_scale
        
        # Compare flattened arrays
        b_next_flat = b_next.numpy().flatten()
        assert np.all(b_next_flat <= max_b * 1.01), \
            f"b' should be bounded by k * leverage_scale, got {b_next_flat} vs {max_b}"


class TestRiskyValueNetwork:
    """Tests for RiskyValueNetwork."""
    
    def test_output_shape(self, risky_value):
        """V_tilde output should be (batch, 1)."""
        k = tf.ones((16,))
        b = tf.ones((16,)) * 0.5
        z = tf.ones((16,))
        V_tilde = risky_value(k, b, z)
        assert V_tilde.shape == (16, 1)
    
    def test_output_can_be_negative(self, risky_value):
        """V_tilde can be negative (latent value)."""
        assert risky_value.output_layer.activation is None or \
               risky_value.output_layer.activation == tf.keras.activations.linear


class TestRiskyPriceNetwork:
    """Tests for RiskyPriceNetwork."""
    
    def test_output_shape(self, risky_price):
        """r_tilde output should be (batch, 1)."""
        k_next = tf.ones((16,))
        b_next = tf.ones((16,)) * 0.5
        z = tf.ones((16,))
        r_tilde = risky_price(k_next, b_next, z)
        assert r_tilde.shape == (16, 1)
    
    def test_r_tilde_gte_r_risk_free(self, risky_price):
        """r_tilde >= r (enforced by r + softplus)."""
        k_next = tf.constant([0.5, 1.0, 2.0, 10.0])
        b_next = tf.constant([0.1, 0.5, 1.0, 5.0])
        z = tf.constant([0.8, 1.0, 1.2, 0.9])
        r_tilde = risky_price(k_next, b_next, z)
        
        r_rf = risky_price.r_risk_free.numpy()
        assert np.all(r_tilde.numpy() >= r_rf), f"r_tilde must be >= {r_rf}"


# =============================================================================
# LIMITED LIABILITY TESTS
# =============================================================================

class TestLimitedLiability:
    """Tests for apply_limited_liability."""
    
    def test_positive_values_unchanged(self):
        """V >= 0 remains unchanged."""
        V_tilde = tf.constant([[1.0], [2.0], [0.0]])
        V = apply_limited_liability(V_tilde)
        np.testing.assert_allclose(V.numpy(), V_tilde.numpy())
    
    def test_negative_values_zeroed(self):
        """V < 0 becomes 0."""
        V_tilde = tf.constant([[-1.0], [-0.5], [0.0], [0.5]])
        V = apply_limited_liability(V_tilde)
        expected = np.array([[0.0], [0.0], [0.0], [0.5]])
        np.testing.assert_allclose(V.numpy(), expected)


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for reproducibility with fixed seeds."""
    
    def test_basic_networks_deterministic(self):
        """Same seed produces same network outputs when same weights are used."""
        # Create one network and call it twice with same inputs
        # This tests that forward pass is deterministic (no randomness)
        tf.random.set_seed(42)
        net = BasicPolicyNetwork(k_min=0.01)
        
        k = tf.constant([1.0, 2.0])
        z = tf.constant([1.0, 1.0])
        
        out1 = net(k, z).numpy()
        out2 = net(k, z).numpy()
        
        np.testing.assert_allclose(out1, out2)
    
    def test_risky_networks_deterministic(self):
        """Same network produces deterministic outputs."""
        tf.random.set_seed(123)
        net = RiskyPolicyNetwork(k_min=0.01, leverage_scale=1.0)
        
        k = tf.constant([1.0, 2.0])
        b = tf.constant([0.5, 0.5])
        z = tf.constant([1.0, 1.0])
        
        k1, b1 = net(k, b, z)
        k2, b2 = net(k, b, z)  # Same network, same inputs
        
        np.testing.assert_allclose(k1.numpy(), k2.numpy())
        np.testing.assert_allclose(b1.numpy(), b2.numpy())


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for network factory functions."""
    
    def test_build_basic_networks(self):
        """Factory returns correct network types."""
        policy, value = build_basic_networks()
        assert isinstance(policy, BasicPolicyNetwork)
        assert isinstance(value, BasicValueNetwork)
    
    def test_build_risky_networks(self):
        """Factory returns correct network types."""
        policy, value, price = build_risky_networks()
        assert isinstance(policy, RiskyPolicyNetwork)
        assert isinstance(value, RiskyValueNetwork)
        assert isinstance(price, RiskyPriceNetwork)
