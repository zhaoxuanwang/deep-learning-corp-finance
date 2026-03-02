"""
tests/networks/test_networks.py

Network tests aligned with raw-network + external-transform architecture.
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
    build_basic_networks,
    build_risky_networks,
)
from src.networks.output_heads import apply_output_head, validate_output_head
from src.trainers.io_transforms import (
    build_legacy_basic_transform_spec_from_networks,
    build_legacy_risky_transform_spec_from_networks,
    forward_basic_policy_levels,
    forward_risky_policy_levels,
    forward_risky_price_levels,
)


K_MIN, K_MAX = 0.01, 10.0
B_MIN, B_MAX = 0.0, 5.0
LOGZ_MIN, LOGZ_MAX = -1.0, 1.0
R_RISK_FREE = 0.04


@pytest.fixture
def basic_policy():
    net = BasicPolicyNetwork(
        k_min=K_MIN,
        k_max=K_MAX,
        logz_min=LOGZ_MIN,
        logz_max=LOGZ_MAX,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    _ = net(tf.constant([[0.0, 0.0]], dtype=tf.float32))
    return net


@pytest.fixture
def basic_value():
    net = BasicValueNetwork(
        k_min=K_MIN,
        k_max=K_MAX,
        logz_min=LOGZ_MIN,
        logz_max=LOGZ_MAX,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    _ = net(tf.constant([[0.0, 0.0]], dtype=tf.float32))
    return net


@pytest.fixture
def risky_policy():
    net = RiskyPolicyNetwork(
        k_min=K_MIN,
        k_max=K_MAX,
        b_min=B_MIN,
        b_max=B_MAX,
        logz_min=LOGZ_MIN,
        logz_max=LOGZ_MAX,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    _ = net(tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32))
    return net


@pytest.fixture
def risky_value():
    net = RiskyValueNetwork(
        k_min=K_MIN,
        k_max=K_MAX,
        b_min=B_MIN,
        b_max=B_MAX,
        logz_min=LOGZ_MIN,
        logz_max=LOGZ_MAX,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    _ = net(tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32))
    return net


@pytest.fixture
def risky_price():
    net = RiskyPriceNetwork(
        k_min=K_MIN,
        k_max=K_MAX,
        b_min=B_MIN,
        b_max=B_MAX,
        logz_min=LOGZ_MIN,
        logz_max=LOGZ_MAX,
        r_risk_free=R_RISK_FREE,
        n_layers=2,
        n_neurons=16,
        hidden_activation="swish",
    )
    _ = net(tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32))
    return net


@pytest.fixture
def basic_spec(basic_policy, basic_value):
    return build_legacy_basic_transform_spec_from_networks(
        policy_net=basic_policy,
        value_net=basic_value,
    )


@pytest.fixture
def risky_spec(risky_policy, risky_value, risky_price):
    return build_legacy_risky_transform_spec_from_networks(
        policy_net=risky_policy,
        value_net=risky_value,
        price_net=risky_price,
        r_risk_free=R_RISK_FREE,
    )


class TestBasicPolicyNetwork:
    def test_raw_output_shape(self, basic_policy):
        x = tf.ones((16, 2), dtype=tf.float32)
        k_raw = basic_policy(x)
        assert k_raw.shape == (16, 1)

    def test_raw_output_shape_1d_input(self, basic_policy):
        x = tf.constant([0.0, 0.0], dtype=tf.float32)
        k_raw = basic_policy(x)
        assert k_raw.shape == (1, 1)

    def test_direct_level_calls_raise(self, basic_policy):
        with pytest.raises(RuntimeError, match="deprecated"):
            _ = basic_policy(tf.ones((8, 1)), tf.ones((8, 1)))

    def test_level_k_bounded_via_transform(self, basic_policy, basic_spec):
        k = tf.constant([0.1, 1.0, 10.0, 100.0], dtype=tf.float32)
        z = tf.constant([0.5, 1.0, 2.0, 0.1], dtype=tf.float32)
        k_next = forward_basic_policy_levels(
            policy_net=basic_policy,
            k=k,
            z=z,
            transform_spec=basic_spec,
            training=False,
            apply_output_clips=False,
        )
        k_np = k_next.numpy().reshape(-1)
        assert np.all(k_np >= K_MIN)
        assert np.all(k_np <= K_MAX)


class TestBasicValueNetwork:
    def test_raw_output_shape(self, basic_value):
        x = tf.ones((16, 2), dtype=tf.float32)
        v_raw = basic_value(x)
        assert v_raw.shape == (16, 1)

    def test_output_can_be_negative(self, basic_value):
        x = tf.zeros((8, 2), dtype=tf.float32)
        v_raw = basic_value(x).numpy().reshape(-1)
        assert np.isfinite(v_raw).all()


class TestRiskyPolicyNetwork:
    def test_raw_output_shapes(self, risky_policy):
        x = tf.ones((16, 3), dtype=tf.float32)
        k_raw, b_raw = risky_policy(x)
        assert k_raw.shape == (16, 1)
        assert b_raw.shape == (16, 1)

    def test_direct_level_calls_raise(self, risky_policy):
        with pytest.raises(RuntimeError, match="deprecated"):
            _ = risky_policy(tf.ones((8, 1)), tf.ones((8, 1)), tf.ones((8, 1)))

    def test_level_outputs_bounded_via_transform(self, risky_policy, risky_spec):
        k = tf.constant([0.1, 1.0, 10.0], dtype=tf.float32)
        b = tf.constant([0.0, 0.5, 1.0], dtype=tf.float32)
        z = tf.constant([0.8, 1.0, 1.2], dtype=tf.float32)
        k_next, b_next = forward_risky_policy_levels(
            policy_net=risky_policy,
            k=k,
            b=b,
            z=z,
            transform_spec=risky_spec,
            training=False,
            apply_output_clips=False,
        )
        k_np = k_next.numpy().reshape(-1)
        b_np = b_next.numpy().reshape(-1)
        assert np.all(k_np >= K_MIN)
        assert np.all(k_np <= K_MAX)
        assert np.all(b_np >= B_MIN)
        assert np.all(b_np <= B_MAX)


class TestRiskyValueNetwork:
    def test_raw_output_shape(self, risky_value):
        x = tf.ones((16, 3), dtype=tf.float32)
        v_raw = risky_value(x)
        assert v_raw.shape == (16, 1)

    def test_output_can_be_negative(self, risky_value):
        x = tf.zeros((8, 3), dtype=tf.float32)
        v_raw = risky_value(x).numpy().reshape(-1)
        assert np.isfinite(v_raw).all()


class TestRiskyPriceNetwork:
    def test_raw_output_shape(self, risky_price):
        x = tf.ones((16, 3), dtype=tf.float32)
        q_raw = risky_price(x)
        assert q_raw.shape == (16, 1)

    def test_level_q_bounded_via_transform(self, risky_price, risky_spec):
        k_next = tf.constant([0.5, 1.0, 2.0, 10.0], dtype=tf.float32)
        b_next = tf.constant([0.1, 0.5, 1.0, 5.0], dtype=tf.float32)
        z = tf.constant([0.8, 1.0, 1.2, 0.9], dtype=tf.float32)
        q = forward_risky_price_levels(
            price_net=risky_price,
            k_next=k_next,
            b_next=b_next,
            z=z,
            transform_spec=risky_spec,
            training=False,
            apply_output_clips=False,
        )
        q_np = q.numpy().reshape(-1)
        q_max = 1.0 / (1.0 + R_RISK_FREE)
        assert np.all(q_np >= 0.0)
        assert np.all(q_np <= q_max * 1.0001)


class TestLimitedLiability:
    def test_positive_values_unchanged(self):
        v_tilde = tf.constant([[1.0], [2.0], [0.0]])
        v = apply_limited_liability(v_tilde)
        np.testing.assert_allclose(v.numpy(), v_tilde.numpy())

    def test_negative_values_zeroed(self):
        v_tilde = tf.constant([[-1.0], [-0.5], [0.0], [0.5]])
        v = apply_limited_liability(v_tilde)
        expected = np.array([[0.0], [0.0], [0.0], [0.5]])
        np.testing.assert_allclose(v.numpy(), expected)


class TestDeterminism:
    def test_basic_networks_deterministic(self):
        tf.random.set_seed(42)
        net = BasicPolicyNetwork(
            k_min=K_MIN,
            k_max=K_MAX,
            logz_min=LOGZ_MIN,
            logz_max=LOGZ_MAX,
            n_layers=2,
            n_neurons=16,
            hidden_activation="swish",
        )
        x = tf.constant([[0.0, 0.0], [0.3, -0.2]], dtype=tf.float32)
        out1 = net(x).numpy()
        out2 = net(x).numpy()
        np.testing.assert_allclose(out1, out2)

    def test_risky_networks_deterministic(self):
        tf.random.set_seed(123)
        net = RiskyPolicyNetwork(
            k_min=K_MIN,
            k_max=K_MAX,
            b_min=B_MIN,
            b_max=B_MAX,
            logz_min=LOGZ_MIN,
            logz_max=LOGZ_MAX,
            n_layers=2,
            n_neurons=16,
            hidden_activation="swish",
        )
        x = tf.constant([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]], dtype=tf.float32)
        k1, b1 = net(x)
        k2, b2 = net(x)
        np.testing.assert_allclose(k1.numpy(), k2.numpy())
        np.testing.assert_allclose(b1.numpy(), b2.numpy())


class TestFactoryFunctions:
    def test_build_basic_networks(self):
        policy, value = build_basic_networks(
            k_min=K_MIN,
            k_max=K_MAX,
            logz_min=LOGZ_MIN,
            logz_max=LOGZ_MAX,
            n_layers=2,
            n_neurons=16,
            hidden_activation="swish",
            policy_head="bounded_sigmoid",
            value_head="linear",
        )
        assert isinstance(policy, BasicPolicyNetwork)
        assert isinstance(value, BasicValueNetwork)

    def test_build_risky_networks(self):
        policy, value, price = build_risky_networks(
            k_min=K_MIN,
            k_max=K_MAX,
            b_min=B_MIN,
            b_max=B_MAX,
            logz_min=LOGZ_MIN,
            logz_max=LOGZ_MAX,
            r_risk_free=R_RISK_FREE,
            n_layers=2,
            n_neurons=16,
            hidden_activation="swish",
            policy_k_head="bounded_sigmoid",
            policy_b_head="bounded_sigmoid",
            value_head="linear",
            price_head="bounded_sigmoid",
        )
        assert isinstance(policy, RiskyPolicyNetwork)
        assert isinstance(value, RiskyValueNetwork)
        assert isinstance(price, RiskyPriceNetwork)


class TestInputNormalization:
    def test_basic_level_forward_is_finite(self, basic_policy, basic_spec):
        k = tf.constant([[0.2], [3.0]], dtype=tf.float32)
        z = tf.constant([[1.0], [1.0]], dtype=tf.float32)
        k_next = forward_basic_policy_levels(
            policy_net=basic_policy,
            k=k,
            z=z,
            transform_spec=basic_spec,
            training=False,
            apply_output_clips=False,
        )
        assert np.isfinite(k_next.numpy()).all()

    def test_risky_level_forward_no_division_issue(self, risky_policy, risky_spec):
        small_k = tf.constant([[0.2]], dtype=tf.float32)
        large_b = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)
        k_next, b_next = forward_risky_policy_levels(
            policy_net=risky_policy,
            k=small_k,
            b=large_b,
            z=z,
            transform_spec=risky_spec,
            training=False,
            apply_output_clips=False,
        )
        assert np.isfinite(k_next.numpy()).all()
        assert np.isfinite(b_next.numpy()).all()


class TestOutputHeads:
    def test_softplus_is_not_supported(self):
        with pytest.raises(ValueError, match="Unsupported head"):
            validate_output_head(output_name="basic_policy_k", head_name="softplus")

    def test_affine_exp_is_allowed_for_policy(self):
        validate_output_head(output_name="basic_policy_k", head_name="affine_exp")

    def test_affine_exp_not_allowed_for_value(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_output_head(output_name="basic_value", head_name="affine_exp")

    def test_affine_exp_output_positive(self):
        raw = tf.constant([[-2.0], [0.0], [2.0]], dtype=tf.float32)
        out = apply_output_head(
            raw,
            head_name="affine_exp",
            affine_mu=tf.constant(0.0, dtype=tf.float32),
            affine_std=tf.constant(1.0, dtype=tf.float32),
        )
        assert np.all(out.numpy() > 0.0)
