"""Tests for the β-conditional network variants.

What we want to be sure of:

  1. Construction with (state_dim, beta_dim, action_dim) gives a network
     whose effective input_dim is state_dim + beta_dim.
  2. Two-arg call(s, β) returns the right output shape and respects clipping.
  3. Single-arg call(s_concat) — used by `build_target_policy` copies —
     produces the same output as two-arg call(s, β) when given the
     concatenated tensor.
  4. Gradients flow into both s and β inputs (β is a real input, not a
     dummy passed through).
  5. Same seed → same outputs (reproducibility).
  6. Same for the value-network analog.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.v2.networks.policy import ParameterizedPolicyNetwork, PolicyNetwork
from src.v2.networks.state_value import (
    ParameterizedStateValueNetwork,
    StateValueNetwork,
)


STATE_DIM  = 2
BETA_DIM   = 5
ACTION_DIM = 1


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def policy_net():
    net = ParameterizedPolicyNetwork(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, beta_dim=BETA_DIM,
        action_low=tf.constant([-5.0]), action_high=tf.constant([5.0]),
        n_layers=2, n_neurons=32, seed=(7, 0))
    net(tf.zeros((1, STATE_DIM)), tf.zeros((1, BETA_DIM)))
    return net


@pytest.fixture
def value_net():
    net = ParameterizedStateValueNetwork(
        state_dim=STATE_DIM, beta_dim=BETA_DIM,
        n_layers=2, n_neurons=32, seed=(11, 0))
    net(tf.zeros((1, STATE_DIM)), tf.zeros((1, BETA_DIM)))
    return net


# --------------------------------------------------------------------- 1. construction

class TestConstruction:
    def test_policy_input_dim_is_augmented(self, policy_net):
        assert policy_net.input_dim == STATE_DIM + BETA_DIM
        assert policy_net.raw_state_dim == STATE_DIM
        assert policy_net.beta_dim == BETA_DIM
        assert policy_net.action_dim == ACTION_DIM

    def test_value_input_dim_is_augmented(self, value_net):
        assert value_net.input_dim == STATE_DIM + BETA_DIM
        assert value_net.raw_state_dim == STATE_DIM
        assert value_net.beta_dim == BETA_DIM

    def test_policy_is_subclass(self, policy_net):
        assert isinstance(policy_net, PolicyNetwork)

    def test_value_is_subclass(self, value_net):
        assert isinstance(value_net, StateValueNetwork)


# --------------------------------------------------------------------- 2. two-arg call

class TestTwoArgCall:
    def test_policy_output_shape(self, policy_net):
        s    = tf.random.normal((32, STATE_DIM))
        beta = tf.random.normal((32, BETA_DIM))
        a    = policy_net(s, beta)
        assert a.shape == (32, ACTION_DIM)

    def test_policy_output_clipped(self, policy_net):
        s    = tf.random.normal((1000, STATE_DIM)) * 10.0
        beta = tf.random.normal((1000, BETA_DIM)) * 10.0
        a    = policy_net(s, beta)
        assert float(tf.reduce_min(a)) >= -5.0 - 1e-6
        assert float(tf.reduce_max(a)) <= 5.0 + 1e-6

    def test_policy_return_raw(self, policy_net):
        s    = tf.random.normal((16, STATE_DIM))
        beta = tf.random.normal((16, BETA_DIM))
        clipped, raw = policy_net(s, beta, return_raw=True)
        assert clipped.shape == (16, ACTION_DIM)
        assert raw.shape == (16, ACTION_DIM)

    def test_value_output_shape(self, value_net):
        s    = tf.random.normal((32, STATE_DIM))
        beta = tf.random.normal((32, BETA_DIM))
        v    = value_net(s, beta)
        assert v.shape == (32, 1)


# --------------------------------------------------------------------- 3. pre-concat fallback (target-copy compatibility)

class TestPreConcatFallback:
    def test_policy_single_arg_matches_two_arg(self, policy_net):
        s    = tf.random.normal((8, STATE_DIM))
        beta = tf.random.normal((8, BETA_DIM))
        x    = tf.concat([s, beta], axis=-1)
        a_two = policy_net(s, beta)
        a_one = policy_net(x)
        np.testing.assert_allclose(a_two.numpy(), a_one.numpy(), rtol=1e-6)

    def test_value_single_arg_matches_two_arg(self, value_net):
        s    = tf.random.normal((8, STATE_DIM))
        beta = tf.random.normal((8, BETA_DIM))
        x    = tf.concat([s, beta], axis=-1)
        v_two = value_net(s, beta)
        v_one = value_net(x)
        np.testing.assert_allclose(v_two.numpy(), v_one.numpy(), rtol=1e-6)


# --------------------------------------------------------------------- 4. β is a real input

class TestBetaIsRealInput:
    def test_policy_output_changes_with_beta(self, policy_net):
        """Different β values at the same s produce different outputs.

        Holds at random init too, because Glorot weights are not identically
        zero on the β columns of the first dense layer (and even if they
        were, the output head would be zero everywhere, making the test
        vacuously trivial — we'd see a -infinity logical conclusion which
        would still pass).
        """
        s     = tf.constant([[1.0, 0.5]])
        beta1 = tf.constant([[0.5, 0.5, 0.2, 0.0, 0.0]])
        beta2 = tf.constant([[0.8, 0.2, 0.5, 0.3, 0.05]])
        a1 = policy_net(s, beta1).numpy()
        a2 = policy_net(s, beta2).numpy()
        assert float(np.max(np.abs(a1 - a2))) > 1e-6

    def test_policy_gradient_flows_into_beta(self, policy_net):
        s    = tf.constant([[1.0, 0.5]])
        beta = tf.Variable([[0.5, 0.5, 0.2, 0.1, 0.05]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(policy_net(s, beta, training=True))
        g = tape.gradient(loss, beta).numpy()
        assert float(np.max(np.abs(g))) > 0.0

    def test_value_output_changes_with_beta(self, value_net):
        s     = tf.constant([[1.0, 0.5]])
        beta1 = tf.constant([[0.5, 0.5, 0.2, 0.0, 0.0]])
        beta2 = tf.constant([[0.8, 0.2, 0.5, 0.3, 0.05]])
        v1 = value_net(s, beta1).numpy()
        v2 = value_net(s, beta2).numpy()
        assert float(np.max(np.abs(v1 - v2))) > 1e-6

    def test_value_gradient_flows_into_beta(self, value_net):
        s    = tf.constant([[1.0, 0.5]])
        beta = tf.Variable([[0.5, 0.5, 0.2, 0.1, 0.05]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(value_net(s, beta, training=True))
        g = tape.gradient(loss, beta).numpy()
        assert float(np.max(np.abs(g))) > 0.0


# --------------------------------------------------------------------- 5. reproducibility

class TestSeedReproducibility:
    def test_same_seed_same_policy_output(self):
        def build():
            net = ParameterizedPolicyNetwork(
                state_dim=STATE_DIM, action_dim=ACTION_DIM, beta_dim=BETA_DIM,
                action_low=tf.constant([-5.0]), action_high=tf.constant([5.0]),
                n_layers=2, n_neurons=32, seed=(42, 0))
            net(tf.zeros((1, STATE_DIM)), tf.zeros((1, BETA_DIM)))
            return net
        net1 = build()
        net2 = build()
        s    = tf.constant([[1.0, 0.5]])
        beta = tf.constant([[0.5, 0.5, 0.2, 0.1, 0.05]])
        np.testing.assert_allclose(net1(s, beta).numpy(), net2(s, beta).numpy())

    def test_same_seed_same_value_output(self):
        def build():
            net = ParameterizedStateValueNetwork(
                state_dim=STATE_DIM, beta_dim=BETA_DIM,
                n_layers=2, n_neurons=32, seed=(42, 0))
            net(tf.zeros((1, STATE_DIM)), tf.zeros((1, BETA_DIM)))
            return net
        net1 = build()
        net2 = build()
        s    = tf.constant([[1.0, 0.5]])
        beta = tf.constant([[0.5, 0.5, 0.2, 0.1, 0.05]])
        np.testing.assert_allclose(net1(s, beta).numpy(), net2(s, beta).numpy())


# --------------------------------------------------------------------- 6. target-policy copy via core.build_target_policy

def test_target_policy_copy_works_with_pre_concat_call(policy_net):
    """`build_target_policy` (existing core helper) builds a plain
    PolicyNetwork with the augmented input_dim. It calls `target(dummy)`
    with a single tensor — our subclass's fallback must support that path.
    """
    from src.v2.trainers.core import build_target_policy
    target = build_target_policy(policy_net)
    assert target.input_dim == STATE_DIM + BETA_DIM
    s    = tf.random.normal((4, STATE_DIM))
    beta = tf.random.normal((4, BETA_DIM))
    x    = tf.concat([s, beta], axis=-1)
    out  = target(x)
    assert out.shape == (4, ACTION_DIM)


def test_target_value_copy_works_with_pre_concat_call(value_net):
    from src.v2.trainers.core import build_target_value
    target = build_target_value(value_net)
    assert target.input_dim == STATE_DIM + BETA_DIM
    s    = tf.random.normal((4, STATE_DIM))
    beta = tf.random.normal((4, BETA_DIM))
    x    = tf.concat([s, beta], axis=-1)
    out  = target(x)
    assert out.shape == (4, 1)
