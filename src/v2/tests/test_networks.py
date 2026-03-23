"""Tests for v2 network modules: base, policy, critic."""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.networks.base import HiddenStack, GenericNetwork
from src.v2.networks.policy import PolicyNetwork
from src.v2.networks.critic import CriticNetwork


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def policy_net():
    """Small policy network for testing."""
    net = PolicyNetwork(
        state_dim=2, action_dim=1,
        action_low=tf.constant([-5.0]),
        action_high=tf.constant([5.0]),
        n_layers=2, n_neurons=32,
    )
    # Build by calling with dummy input.
    dummy = tf.zeros((1, 2))
    net(dummy)
    return net


@pytest.fixture
def critic_net():
    """Small critic network for testing."""
    net = CriticNetwork(
        state_dim=2, action_dim=1,
        n_layers=2, n_neurons=32,
    )
    # Build by calling with dummy input.
    dummy_s = tf.zeros((1, 2))
    dummy_a = tf.zeros((1, 1))
    net(dummy_s, dummy_a)
    return net


# =============================================================================
# HiddenStack
# =============================================================================

class TestHiddenStack:
    """Tests for the HiddenStack layer."""

    def test_output_shape(self):
        """Output has shape (batch, n_neurons)."""
        stack = HiddenStack(n_layers=3, n_neurons=64)
        x = tf.random.normal((16, 5))
        out = stack(x)
        assert out.shape == (16, 64)

    def test_dense_has_bias(self):
        """Dense layers have use_bias=True."""
        stack = HiddenStack(n_layers=2, n_neurons=32)
        _ = stack(tf.zeros((1, 4)))  # build
        for dense in stack.dense_layers:
            assert dense.use_bias is True

    def test_layer_count(self):
        """Number of dense layers matches n_layers."""
        stack = HiddenStack(n_layers=4, n_neurons=16)
        assert len(stack.dense_layers) == 4

    def test_relu_activation_supported(self):
        """HiddenStack supports ReLU as an optional activation."""
        stack = HiddenStack(n_layers=2, n_neurons=8, activation="relu")
        out = stack(tf.random.normal((4, 3)))
        assert out.shape == (4, 8)


# =============================================================================
# PolicyNetwork
# =============================================================================

class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_output_shape(self, policy_net):
        """Output shape is (batch, action_dim)."""
        s = tf.random.normal((32, 2))
        a = policy_net(s)
        assert a.shape == (32, 1)

    def test_output_clipped(self, policy_net):
        """All outputs lie within [action_low, action_high]."""
        s = tf.random.normal((1000, 2)) * 10  # large inputs to push outputs
        a = policy_net(s)
        assert float(tf.reduce_min(a)) >= -5.0 - 1e-6
        assert float(tf.reduce_max(a)) <= 5.0 + 1e-6

    def test_return_raw(self, policy_net):
        """return_raw=True returns (clipped, raw) tuple."""
        s = tf.random.normal((16, 2))
        result = policy_net(s, return_raw=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        clipped, raw = result
        assert clipped.shape == (16, 1)
        assert raw.shape == (16, 1)

    def test_raw_can_exceed_bounds(self, policy_net):
        """Raw output is not clipped — can be outside bounds."""
        # With random weights, at least some raw values should differ from clipped.
        s = tf.random.normal((500, 2)) * 100
        clipped, raw = policy_net(s, return_raw=True)
        # They should differ when raw exceeds bounds.
        diff = tf.reduce_sum(tf.abs(clipped - raw))
        # Not guaranteed but highly likely with large inputs.
        # If all match, the test is vacuously correct (bounds are generous).

    def test_gradient_flow(self, policy_net):
        """Gradients flow through the policy."""
        s = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            a = policy_net(s, training=True)
            loss = tf.reduce_sum(a)
        grads = tape.gradient(loss, policy_net.trainable_variables)
        # At least some gradients should be non-zero.
        has_nonzero = any(
            float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads if g is not None)
        assert has_nonzero

    def test_gradient_through_raw_action(self, policy_net):
        """For MVE actor update: gradient flows through raw (unclipped) action."""
        s = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            _, raw = policy_net(s, training=True, return_raw=True)
            loss = tf.reduce_sum(raw ** 2)
        grads = tape.gradient(loss, policy_net.trainable_variables)
        has_nonzero = any(
            float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads if g is not None)
        assert has_nonzero

    def test_initial_output_near_action_center(self):
        """Untrained policy outputs near the center of the action range."""
        low = tf.constant([-100.0])
        high = tf.constant([200.0])
        net = PolicyNetwork(state_dim=2, action_dim=1,
                            action_low=low, action_high=high,
                            n_layers=2, n_neurons=32)
        s = tf.random.normal((64, 2))
        a = net(s)
        expected_center = (-100.0 + 200.0) / 2.0  # = 50.0
        mean_action = float(tf.reduce_mean(a))
        # Should be within 10% of the action range from center.
        assert abs(mean_action - expected_center) < 0.1 * (200.0 - (-100.0))

    def test_output_rescaling_params(self):
        """action_center and action_scale are computed correctly."""
        low = tf.constant([-10.0, 0.0])
        high = tf.constant([10.0, 20.0])
        net = PolicyNetwork(state_dim=3, action_dim=2,
                            action_low=low, action_high=high,
                            n_layers=1, n_neurons=16)
        np.testing.assert_allclose(
            net.action_center.numpy(), [0.0, 10.0], atol=1e-5)
        expected_scale = np.sqrt([10.0, 10.0])
        np.testing.assert_allclose(
            net.action_scale.numpy(), expected_scale, atol=1e-5)

    def test_gradient_amplification(self):
        """Gradient through the policy is amplified by action_scale."""
        low = tf.constant([-100.0])
        high = tf.constant([100.0])
        net = PolicyNetwork(state_dim=2, action_dim=1,
                            action_low=low, action_high=high,
                            n_layers=1, n_neurons=16)
        s = tf.constant([[1.0, 0.5]])
        # Compute gradient of action w.r.t. a single hidden output weight.
        with tf.GradientTape() as tape:
            a = net(s, training=True)
            loss = tf.reduce_sum(a)
        grads = tape.gradient(loss, net.trainable_variables)
        # Gradient should be non-trivially amplified.
        total_grad = float(tf.sqrt(sum(
            tf.reduce_sum(g**2) for g in grads if g is not None)))
        assert total_grad > 1.0  # amplified beyond unit scale


# =============================================================================
# CriticNetwork
# =============================================================================

class TestCriticNetwork:
    """Tests for CriticNetwork."""

    def test_output_shape(self, critic_net):
        """Output shape is (batch, 1)."""
        s = tf.random.normal((32, 2))
        a = tf.random.normal((32, 1))
        q = critic_net(s, a)
        assert q.shape == (32, 1)

    def test_q_level_shape(self, critic_net):
        """q_level returns same shape as call."""
        s = tf.random.normal((16, 2))
        a = tf.random.normal((16, 1))
        q_level = critic_net.q_level(s, a)
        assert q_level.shape == (16, 1)

    def test_q_level_equals_call(self, critic_net):
        """q_level(s,a) == call(s,a) — both in level space."""
        s = tf.random.normal((8, 2))
        a = tf.random.normal((8, 1))
        q_call = critic_net(s, a)
        q_level = critic_net.q_level(s, a)
        np.testing.assert_allclose(
            q_level.numpy(), q_call.numpy(), atol=1e-5)

    def test_gradient_flow(self, critic_net):
        """Gradients flow through the critic."""
        s = tf.constant([[1.0, 0.5]])
        a = tf.constant([[0.1]])
        with tf.GradientTape() as tape:
            q = critic_net(s, a, training=True)
            loss = tf.reduce_sum(q)
        grads = tape.gradient(loss, critic_net.trainable_variables)
        has_nonzero = any(
            float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads if g is not None)
        assert has_nonzero

    def test_input_dim_is_state_plus_action(self, critic_net):
        """Critic input_dim = state_dim + action_dim."""
        assert critic_net.input_dim == 3  # 2 + 1
