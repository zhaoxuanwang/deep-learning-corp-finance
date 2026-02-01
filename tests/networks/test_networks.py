"""
tests/networks/test_networks.py

Shape tests, output domain tests, and determinism tests for all networks.

Reference: outline_v2.md network specifications
"""

import pytest
import tensorflow as tf
import numpy as np

from src.networks import (
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
    return BasicPolicyNetwork(
        k_min=0.01, k_max=10.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16, activation="swish"
    )


@pytest.fixture
def basic_value():
    return BasicValueNetwork(
        k_min=0.01, k_max=10.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16, activation="swish"
    )


@pytest.fixture
def risky_policy():
    return RiskyPolicyNetwork(
        k_min=0.01, k_max=10.0,
        b_min=0.0, b_max=5.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16,
        activation="swish"
    )


@pytest.fixture
def risky_value():
    return RiskyValueNetwork(
        k_min=0.01, k_max=10.0, b_max=5.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16, activation="swish"
    )


@pytest.fixture
def risky_price():
    return RiskyPriceNetwork(
        k_min=0.01, k_max=10.0, b_max=5.0,
        logz_min=-1.0, logz_max=1.0,
        r_risk_free=0.04,
        n_layers=2, n_neurons=16, activation="swish"
    )


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

    def test_k_next_bounded(self, basic_policy):
        """k' must be in [k_min, k_max] (enforced by bounded sigmoid)."""
        # Test with various inputs including extreme values
        k = tf.constant([0.1, 1.0, 10.0, 100.0])
        z = tf.constant([0.5, 1.0, 2.0, 0.1])
        k_next = basic_policy(k, z)

        k_min = basic_policy.k_min.numpy()
        k_max = basic_policy.k_max.numpy()
        k_next_np = k_next.numpy().flatten()

        assert np.all(k_next_np >= k_min), f"k' must be >= {k_min}"
        assert np.all(k_next_np <= k_max), f"k' must be <= {k_max}"

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

    def test_k_next_bounded(self, risky_policy):
        """k' must be in [k_min, k_max]."""
        k = tf.constant([0.1, 1.0, 10.0])
        b = tf.constant([0.0, 0.5, 1.0])
        z = tf.constant([0.8, 1.0, 1.2])
        k_next, _ = risky_policy(k, b, z)

        k_min = risky_policy.k_min.numpy()
        k_max = risky_policy.k_max.numpy()
        k_next_np = k_next.numpy().flatten()

        assert np.all(k_next_np >= k_min)
        assert np.all(k_next_np <= k_max)

    def test_b_next_non_negative(self, risky_policy):
        """b' >= 0 (borrowing-only constraint via sigmoid)."""
        k = tf.constant([0.1, 1.0, 10.0])
        b = tf.constant([0.0, 0.5, 1.0])
        z = tf.constant([0.8, 1.0, 1.2])
        _, b_next = risky_policy(k, b, z)

        assert np.all(b_next.numpy() >= 0), "b' must be >= 0 (borrowing-only)"

    def test_b_next_is_bounded(self, risky_policy):
        """b' should be bounded by b_max."""
        k = tf.constant([1.0, 2.0, 5.0])
        b = tf.zeros((3,))
        z = tf.ones((3,))
        _, b_next = risky_policy(k, b, z)

        b_max = risky_policy.b_max.numpy()

        # Compare flattened arrays
        b_next_flat = b_next.numpy().flatten()
        assert np.all(b_next_flat <= b_max * 1.0001), \
            f"b' should be bounded by b_max={b_max}, got {b_next_flat}"


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
        """q output should be (batch, 1)."""
        k_next = tf.ones((16,))
        b_next = tf.ones((16,)) * 0.5
        z = tf.ones((16,))
        q = risky_price(k_next, b_next, z)
        assert q.shape == (16, 1)

    def test_q_bounded(self, risky_price):
        """q should be in [0, 1/(1+r)] (bounded sigmoid)."""
        k_next = tf.constant([0.5, 1.0, 2.0, 10.0])
        b_next = tf.constant([0.1, 0.5, 1.0, 5.0])
        z = tf.constant([0.8, 1.0, 1.2, 0.9])
        q = risky_price(k_next, b_next, z)

        q_max = risky_price.q_max.numpy()
        q_np = q.numpy().flatten()

        assert np.all(q_np >= 0), "q must be >= 0"
        assert np.all(q_np <= q_max * 1.0001), f"q must be <= {q_max}"


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
        net = BasicPolicyNetwork(
            k_min=0.01, k_max=10.0,
            logz_min=-1.0, logz_max=1.0,
            n_layers=2, n_neurons=16,
            activation="swish"
        )

        k = tf.constant([1.0, 2.0])
        z = tf.constant([1.0, 1.0])

        out1 = net(k, z).numpy()
        out2 = net(k, z).numpy()

        np.testing.assert_allclose(out1, out2)

    def test_risky_networks_deterministic(self):
        """Same network produces deterministic outputs."""
        tf.random.set_seed(123)
        net = RiskyPolicyNetwork(
            k_min=0.01, k_max=10.0,
            b_min=0.0, b_max=5.0,
            logz_min=-1.0, logz_max=1.0,
            n_layers=2, n_neurons=16,
            activation="swish"
        )

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
        policy, value = build_basic_networks(
            k_min=0.01, k_max=10.0,
            logz_min=-1.0, logz_max=1.0,
            n_layers=2, n_neurons=16,
            activation="swish"
        )
        assert isinstance(policy, BasicPolicyNetwork)
        assert isinstance(value, BasicValueNetwork)

    def test_build_risky_networks(self):
        """Factory returns correct network types."""
        policy, value, price = build_risky_networks(
            k_min=0.01, k_max=10.0,
            b_min=0.0, b_max=5.0,
            logz_min=-1.0, logz_max=1.0,
            r_risk_free=0.04,
            n_layers=2, n_neurons=16,
            activation="swish"
        )
        assert isinstance(policy, RiskyPolicyNetwork)
        assert isinstance(value, RiskyValueNetwork)
        assert isinstance(price, RiskyPriceNetwork)


# =============================================================================
# INPUT NORMALIZATION TESTS
# =============================================================================

class TestInputNormalization:
    """Tests to verify the new min-max input normalization."""

    def test_basic_policy_normalized_inputs(self):
        """Test that inputs are normalized to [0, 1]."""
        net = BasicPolicyNetwork(
            k_min=0.2, k_max=3.0,
            logz_min=-0.5, logz_max=0.5,
            n_layers=2, n_neurons=16,
            activation="swish"
        )

        # Test at boundary values
        k_at_min = tf.constant([[0.2]])
        k_at_max = tf.constant([[3.0]])
        z_at_mean = tf.constant([[1.0]])  # log(1.0) = 0, which is midpoint

        # Both should produce valid outputs
        k_next_1 = net(k_at_min, z_at_mean)
        k_next_2 = net(k_at_max, z_at_mean)

        # Outputs should be bounded
        assert k_next_1.numpy() >= 0.2
        assert k_next_1.numpy() <= 3.0
        assert k_next_2.numpy() >= 0.2
        assert k_next_2.numpy() <= 3.0

    def test_risky_policy_no_leverage_division(self):
        """Test that b/k division is no longer used (use b/b_max instead)."""
        net = RiskyPolicyNetwork(
            k_min=0.2, k_max=3.0,
            b_min=0.0, b_max=2.0,
            logz_min=-0.5, logz_max=0.5,
            n_layers=2, n_neurons=16,
            activation="swish"
        )

        # Test with small k (would cause issues with b/k)
        small_k = tf.constant([[0.2]])  # k_min
        large_b = tf.constant([[2.0]])  # b_max
        z = tf.constant([[1.0]])

        # Should not cause numerical issues
        k_next, b_next = net(small_k, large_b, z)

        # Outputs should be valid (no NaN, no Inf)
        assert np.isfinite(k_next.numpy()).all()
        assert np.isfinite(b_next.numpy()).all()

        # Outputs should be bounded
        assert k_next.numpy() >= 0.2
        assert k_next.numpy() <= 3.0
        assert b_next.numpy() >= 0.0
        assert b_next.numpy() <= 2.0
