"""
tests/trainers/test_gradient_rules.py

Tests for specific gradient block rules:
- Critic should not update policy
- Actor should not update value
- Actor gradients must flow through value network
"""

import numpy as np
import tensorflow as tf

from src.networks import BasicPolicyNetwork, BasicValueNetwork
from src.trainers.io_transforms import (
    build_legacy_basic_transform_spec_from_networks,
    forward_basic_policy_levels,
    forward_basic_value_levels,
)


def _build_basic_networks():
    policy_net = BasicPolicyNetwork(
        k_min=0.01,
        k_max=10.0,
        logz_min=-1.0,
        logz_max=1.0,
        n_layers=2,
        n_neurons=8,
        hidden_activation="swish",
    )
    value_net = BasicValueNetwork(
        k_min=0.01,
        k_max=10.0,
        logz_min=-1.0,
        logz_max=1.0,
        n_layers=2,
        n_neurons=8,
        hidden_activation="swish",
    )
    # Build raw networks with normalized feature tensors.
    dummy_x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    _ = policy_net(dummy_x)
    _ = value_net(dummy_x)
    transform_spec = build_legacy_basic_transform_spec_from_networks(
        policy_net=policy_net,
        value_net=value_net,
    )
    return policy_net, value_net, transform_spec


class TestStopGradientRules:
    """
    Verify strict stop_gradient usage in BR-style blocks.
    """

    def test_critic_does_not_update_policy(self):
        """Critic step should not produce gradients for policy params."""
        policy_net, value_net, transform_spec = _build_basic_networks()
        k = tf.constant([[2.0], [3.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.2]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next = forward_basic_policy_levels(
                policy_net=policy_net,
                k=k,
                z=z,
                transform_spec=transform_spec,
                training=False,
                apply_output_clips=False,
            )
            k_next = tf.stop_gradient(k_next)
            v_curr = forward_basic_value_levels(
                value_net=value_net,
                k=k,
                z=z,
                transform_spec=transform_spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value_net,
                k=k_next,
                z=z,
                transform_spec=transform_spec,
                training=True,
                apply_output_clips=False,
            )
            target = tf.stop_gradient(v_next)
            loss = tf.reduce_mean(tf.square(v_curr - target))

        grads = tape.gradient(loss, policy_net.trainable_variables)
        for grad in grads:
            assert grad is None or np.allclose(grad.numpy(), 0.0)

    def test_actor_does_not_update_value(self):
        """Actor update should not mutate value weights when excluded from optimizer."""
        policy_net, value_net, transform_spec = _build_basic_networks()
        optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-3)
        init_value = [w.copy() for w in value_net.get_weights()]

        k = tf.constant([[2.0], [3.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.2]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            k_next = forward_basic_policy_levels(
                policy_net=policy_net,
                k=k,
                z=z,
                transform_spec=transform_spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value_net,
                k=k_next,
                z=z,
                transform_spec=transform_spec,
                training=False,
                apply_output_clips=False,
            )
            loss = -tf.reduce_mean(v_next)

        grads = tape.gradient(loss, policy_net.trainable_variables)
        optimizer_actor.apply_gradients(zip(grads, policy_net.trainable_variables))

        updated_value = value_net.get_weights()
        value_changed = any(
            not np.allclose(w0, w1, rtol=1e-6, atol=1e-8)
            for w0, w1 in zip(init_value, updated_value)
        )
        assert not value_changed

    def test_actor_has_gradient_through_value(self):
        """Actor loss should have gradient w.r.t. policy via value function."""
        policy_net, value_net, transform_spec = _build_basic_networks()
        k = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next = forward_basic_policy_levels(
                policy_net=policy_net,
                k=k,
                z=z,
                transform_spec=transform_spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value_net,
                k=k_next,
                z=z,
                transform_spec=transform_spec,
                training=False,
                apply_output_clips=False,
            )
            loss = -tf.reduce_mean(v_next)

        grads = tape.gradient(loss, policy_net.trainable_variables)
        assert any(g is not None for g in grads), "Policy should receive gradient through value net"
