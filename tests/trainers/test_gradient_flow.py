"""
tests/trainers/test_gradient_flow.py

Gradient-flow tests aligned with the transform-based refactor.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.networks import (
    BasicPolicyNetwork,
    BasicValueNetwork,
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    apply_limited_liability,
)
from src.trainers.io_transforms import (
    build_legacy_basic_transform_spec_from_networks,
    build_legacy_risky_transform_spec_from_networks,
    forward_basic_policy_levels,
    forward_basic_value_levels,
    forward_risky_policy_levels,
    forward_risky_value_levels,
    forward_risky_price_levels,
)


@pytest.fixture
def basic_networks():
    policy = BasicPolicyNetwork(
        k_min=0.01,
        k_max=10.0,
        logz_min=-1.0,
        logz_max=1.0,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    value = BasicValueNetwork(
        k_min=0.01,
        k_max=10.0,
        logz_min=-1.0,
        logz_max=1.0,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    dummy_x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    _ = policy(dummy_x)
    _ = value(dummy_x)
    spec = build_legacy_basic_transform_spec_from_networks(
        policy_net=policy,
        value_net=value,
    )
    return policy, value, spec


@pytest.fixture
def risky_networks():
    tf.random.set_seed(42)
    policy = RiskyPolicyNetwork(
        k_min=0.01,
        k_max=10.0,
        b_min=0.0,
        b_max=5.0,
        logz_min=-1.0,
        logz_max=1.0,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    value = RiskyValueNetwork(
        k_min=0.01,
        k_max=10.0,
        b_min=0.0,
        b_max=5.0,
        logz_min=-1.0,
        logz_max=1.0,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    price = RiskyPriceNetwork(
        k_min=0.01,
        k_max=10.0,
        b_min=0.0,
        b_max=5.0,
        logz_min=-1.0,
        logz_max=1.0,
        r_risk_free=0.04,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    dummy_x = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    _ = policy(dummy_x)
    _ = value(dummy_x)
    _ = price(dummy_x)
    spec = build_legacy_risky_transform_spec_from_networks(
        policy_net=policy,
        value_net=value,
        price_net=price,
        r_risk_free=0.04,
    )
    return policy, value, price, spec


class TestCriticBlockGradientFlow:
    def test_basic_critic_no_policy_gradient(self, basic_networks):
        policy, value, spec = basic_networks
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next = forward_basic_policy_levels(
                policy_net=policy,
                k=k,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            k_next_sg = tf.stop_gradient(k_next)
            v_curr = forward_basic_value_levels(
                value_net=value,
                k=k,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value,
                k=k_next_sg,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next_sg = tf.stop_gradient(v_next)
            loss = tf.reduce_mean(tf.square(v_curr - v_next_sg))

        policy_grads = tape.gradient(loss, policy.trainable_variables)
        for grad in policy_grads:
            assert grad is None or np.allclose(grad.numpy(), 0.0)

    def test_risky_critic_no_policy_gradient(self, risky_networks):
        policy, value, price, spec = risky_networks
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        b = tf.constant([[0.5], [0.5]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next, b_next = forward_risky_policy_levels(
                policy_net=policy,
                k=k,
                b=b,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            k_next_sg = tf.stop_gradient(k_next)
            b_next_sg = tf.stop_gradient(b_next)
            r_tilde = forward_risky_price_levels(
                price_net=price,
                k_next=k_next_sg,
                b_next=b_next_sg,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_curr = forward_risky_value_levels(
                value_net=value,
                k=k,
                b=b,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_risky_value_levels(
                value_net=value,
                k=k_next_sg,
                b=b_next_sg,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            loss = tf.reduce_mean(tf.square(v_curr - v_next) + tf.square(r_tilde))

        policy_grads = tape.gradient(loss, policy.trainable_variables)
        for grad in policy_grads:
            assert grad is None or np.allclose(grad.numpy(), 0.0)


class TestActorBlockGradientFlow:
    def test_basic_actor_policy_gradient_exists(self, basic_networks):
        policy, value, spec = basic_networks
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next = forward_basic_policy_levels(
                policy_net=policy,
                k=k,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value,
                k=k_next,
                z=z,
                transform_spec=spec,
                training=False,
                apply_output_clips=False,
            )
            loss = -tf.reduce_mean(v_next)

        policy_grads = tape.gradient(loss, policy.trainable_variables)
        assert any(
            grad is not None and not np.allclose(grad.numpy(), 0.0)
            for grad in policy_grads
        )

    def test_risky_actor_policy_gradient_exists(self, risky_networks):
        policy, value, price, spec = risky_networks
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        b = tf.constant([[0.5], [0.5]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next, b_next = forward_risky_policy_levels(
                policy_net=policy,
                k=k,
                b=b,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_risky_value_levels(
                value_net=value,
                k=k_next,
                b=b_next,
                z=z,
                transform_spec=spec,
                training=False,
                apply_output_clips=False,
            )
            r_tilde = forward_risky_price_levels(
                price_net=price,
                k_next=k_next,
                b_next=b_next,
                z=z,
                transform_spec=spec,
                training=False,
                apply_output_clips=False,
            )
            loss = -tf.reduce_mean(v_next) + 0.1 * tf.reduce_mean(r_tilde)

        policy_grads = tape.gradient(loss, policy.trainable_variables)
        assert any(
            grad is not None and not np.allclose(grad.numpy(), 0.0)
            for grad in policy_grads
        )

    def test_actor_value_receives_no_update(self, basic_networks):
        policy, value, spec = basic_networks
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        init_value = [w.copy() for w in value.get_weights()]
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next = forward_basic_policy_levels(
                policy_net=policy,
                k=k,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value,
                k=k_next,
                z=z,
                transform_spec=spec,
                training=False,
                apply_output_clips=False,
            )
            loss = -tf.reduce_mean(v_next)

        policy_grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(policy_grads, policy.trainable_variables))

        updated_value = value.get_weights()
        value_changed = any(
            not np.allclose(w0, w1, rtol=1e-6, atol=1e-8)
            for w0, w1 in zip(init_value, updated_value)
        )
        assert not value_changed


class TestCriticTargetDetachment:
    def test_rhs_continuation_detached(self, basic_networks):
        policy, value, spec = basic_networks
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)
        k_next = forward_basic_policy_levels(
            policy_net=policy,
            k=k,
            z=z,
            transform_spec=spec,
            training=False,
            apply_output_clips=False,
        )
        k_next_sg = tf.stop_gradient(k_next)

        with tf.GradientTape(persistent=True) as tape:
            v_curr = forward_basic_value_levels(
                value_net=value,
                k=k,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next = forward_basic_value_levels(
                value_net=value,
                k=k_next_sg,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            v_next_detached = tf.stop_gradient(v_next)
            y = tf.constant([[0.5], [0.5]], dtype=tf.float32) + 0.96 * v_next_detached
            loss = tf.reduce_mean(tf.square(v_curr - y))

        value_grads = tape.gradient(loss, value.trainable_variables)
        assert any(g is not None for g in value_grads)
        grad_v_next = tape.gradient(loss, v_next)
        assert grad_v_next is None
        del tape

    def test_lhs_remains_trainable(self, basic_networks):
        _, value, spec = basic_networks
        k = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            v_curr = forward_basic_value_levels(
                value_net=value,
                k=k,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            y = tf.constant([[1.0], [1.0]], dtype=tf.float32)
            loss = tf.reduce_mean(tf.square(v_curr - y))

        grad_v_curr = tape.gradient(loss, v_curr)
        assert grad_v_curr is not None
        assert not np.allclose(grad_v_curr.numpy(), 0.0)


class TestRiskyDebtGradientFlow:
    def test_price_gradients_in_actor_block(self, risky_networks):
        policy, _, price, spec = risky_networks
        k = tf.constant([[1.0]], dtype=tf.float32)
        b = tf.constant([[0.5]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            k_next, b_next = forward_risky_policy_levels(
                policy_net=policy,
                k=k,
                b=b,
                z=z,
                transform_spec=spec,
                training=True,
                apply_output_clips=False,
            )
            r_tilde = forward_risky_price_levels(
                price_net=price,
                k_next=k_next,
                b_next=b_next,
                z=z,
                transform_spec=spec,
                training=False,
                apply_output_clips=False,
            )
            loss = tf.reduce_mean(r_tilde)

        policy_grads = tape.gradient(loss, policy.trainable_variables)
        assert any(g is not None for g in policy_grads)

    def test_limited_liability_gradient_flow(self):
        with tf.GradientTape() as tape:
            v_tilde = tf.constant([[2.0], [-1.0]], dtype=tf.float32)
            tape.watch(v_tilde)
            v = apply_limited_liability(v_tilde)
            loss = tf.reduce_sum(v)

        grad = tape.gradient(loss, v_tilde)
        np.testing.assert_allclose(grad.numpy(), np.array([[1.0], [0.0]]))

    def test_limited_liability_leaky_gradient_flow(self):
        with tf.GradientTape() as tape:
            v_tilde = tf.constant([[2.0], [-1.0]], dtype=tf.float32)
            tape.watch(v_tilde)
            v = apply_limited_liability(v_tilde, leaky=True, alpha=0.01)
            loss = tf.reduce_sum(v)

        grad = tape.gradient(loss, v_tilde)
        np.testing.assert_allclose(grad.numpy(), np.array([[1.0], [0.01]]), atol=1e-6)
