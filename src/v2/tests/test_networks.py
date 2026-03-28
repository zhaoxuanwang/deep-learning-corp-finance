"""Tests for the supported v2 network modules."""

import numpy as np
import pytest
import tensorflow as tf

from src.v2.networks.base import HiddenStack
from src.v2.networks.policy import PolicyNetwork
from src.v2.networks.state_value import StateValueNetwork


@pytest.fixture
def policy_net():
    net = PolicyNetwork(
        state_dim=2,
        action_dim=1,
        action_low=tf.constant([-5.0]),
        action_high=tf.constant([5.0]),
        n_layers=2,
        n_neurons=32,
    )
    net(tf.zeros((1, 2)))
    return net


@pytest.fixture
def value_net():
    net = StateValueNetwork(
        state_dim=2,
        n_layers=2,
        n_neurons=32,
    )
    net(tf.zeros((1, 2)))
    return net


class TestHiddenStack:
    def test_output_shape(self):
        stack = HiddenStack(n_layers=3, n_neurons=64)
        out = stack(tf.random.normal((16, 5)))
        assert out.shape == (16, 64)

    def test_dense_has_bias(self):
        stack = HiddenStack(n_layers=2, n_neurons=32)
        stack(tf.zeros((1, 4)))
        for dense in stack.dense_layers:
            assert dense.use_bias is True

    def test_relu_activation_supported(self):
        stack = HiddenStack(n_layers=2, n_neurons=8, activation="relu")
        out = stack(tf.random.normal((4, 3)))
        assert out.shape == (4, 8)


class TestPolicyNetwork:
    def test_output_shape(self, policy_net):
        a = policy_net(tf.random.normal((32, 2)))
        assert a.shape == (32, 1)

    def test_output_clipped(self, policy_net):
        a = policy_net(tf.random.normal((1000, 2)) * 10.0)
        assert float(tf.reduce_min(a)) >= -5.0 - 1e-6
        assert float(tf.reduce_max(a)) <= 5.0 + 1e-6

    def test_return_raw(self, policy_net):
        clipped, raw = policy_net(tf.random.normal((16, 2)), return_raw=True)
        assert clipped.shape == (16, 1)
        assert raw.shape == (16, 1)

    def test_gradient_flow(self, policy_net):
        s = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(policy_net(s, training=True))
        grads = tape.gradient(loss, policy_net.trainable_variables)
        assert any(float(tf.reduce_sum(tf.abs(g))) > 0.0 for g in grads if g is not None)

    def test_gradient_through_raw_action(self, policy_net):
        s = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            _, raw = policy_net(s, training=True, return_raw=True)
            loss = tf.reduce_sum(raw ** 2)
        grads = tape.gradient(loss, policy_net.trainable_variables)
        assert any(float(tf.reduce_sum(tf.abs(g))) > 0.0 for g in grads if g is not None)

    def test_initial_output_near_action_center(self):
        low = tf.constant([-100.0])
        high = tf.constant([200.0])
        net = PolicyNetwork(
            state_dim=2,
            action_dim=1,
            action_low=low,
            action_high=high,
            n_layers=2,
            n_neurons=32,
        )
        mean_action = float(tf.reduce_mean(net(tf.random.normal((64, 2)))))
        assert abs(mean_action - 50.0) < 30.0

    def test_output_rescaling_params(self):
        low = tf.constant([-10.0, 0.0])
        high = tf.constant([10.0, 20.0])
        net = PolicyNetwork(
            state_dim=3,
            action_dim=2,
            action_low=low,
            action_high=high,
            n_layers=1,
            n_neurons=16,
        )
        np.testing.assert_allclose(net.action_center.numpy(), [0.0, 10.0], atol=1e-5)
        np.testing.assert_allclose(net.action_scale.numpy(), [10.0, 10.0], atol=1e-5)

    def test_gradient_amplification(self):
        net = PolicyNetwork(
            state_dim=2,
            action_dim=1,
            action_low=tf.constant([-100.0]),
            action_high=tf.constant([100.0]),
            n_layers=1,
            n_neurons=16,
        )
        s = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(net(s, training=True))
        grads = tape.gradient(loss, net.trainable_variables)
        total_grad = float(
            tf.sqrt(sum(tf.reduce_sum(g ** 2) for g in grads if g is not None))
        )
        assert total_grad > 1.0


class TestStateValueNetwork:
    def test_output_shape(self, value_net):
        out = value_net(tf.random.normal((32, 2)))
        assert out.shape == (32, 1)

    def test_gradient_flow(self, value_net):
        s = tf.constant([[1.0, 0.5]])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(value_net(s, training=True))
        grads = tape.gradient(loss, value_net.trainable_variables)
        assert any(float(tf.reduce_sum(tf.abs(g))) > 0.0 for g in grads if g is not None)
