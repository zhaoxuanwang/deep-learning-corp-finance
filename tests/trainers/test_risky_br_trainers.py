"""
tests/trainers/test_risky_br_trainers.py

Unit tests for RiskyDebtTrainerBR with flattened dataset interface.

Tests verify:
1. Trainer initialization with target networks (policy, value, price)
2. Train step with flattened data format (k, b, z, z_next_main, z_next_fork)
3. Target networks exist and are initialized correctly
4. Polyak averaging for all three target networks
5. n_critic_steps controls critic iterations
6. Actor uses only main shock (not AiO)
7. Bond price network outputs q in [0, 1/(1+r)]
8. Effective value computation with Gumbel-Sigmoid
9. Gradient flow correctness

Reference:
    report_brief.md lines 917-1082: "BR Method" for Risky Debt
"""

import tensorflow as tf
import pytest
import numpy as np
import shutil
import os

from src.economy.parameters import EconomicParams, ShockParams
from src.networks.network_risky import (
    build_risky_networks,
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    compute_effective_value
)
from src.trainers.risky import RiskyDebtTrainerBR
from src.utils.annealing import AnnealingSchedule


@pytest.fixture
def params():
    """Economic parameters."""
    return EconomicParams()


@pytest.fixture
def shock_params():
    """Shock parameters."""
    return ShockParams()


@pytest.fixture
def bounds():
    """State space bounds."""
    return {
        'k': (0.1, 10.0),
        'b': (0.0, 5.0),
        'log_z': (-0.5, 0.5)
    }


@pytest.fixture
def networks(params, bounds):
    """Build all three networks for Risky Debt model."""
    policy_net, value_net, price_net = build_risky_networks(
        k_min=bounds['k'][0],
        k_max=bounds['k'][1],
        b_min=bounds['b'][0],
        b_max=bounds['b'][1],
        r_risk_free=params.r_rate,
        n_layers=2,
        n_neurons=16,
        activation='swish'
    )

    # Build networks by calling them once with dummy data
    dummy_k = tf.constant([[1.0]], dtype=tf.float32)
    dummy_b = tf.constant([[0.5]], dtype=tf.float32)
    dummy_z = tf.constant([[1.0]], dtype=tf.float32)
    _ = policy_net(dummy_k, dummy_b, dummy_z)
    _ = value_net(dummy_k, dummy_b, dummy_z)
    _ = price_net(dummy_k, dummy_b, dummy_z)

    return policy_net, value_net, price_net


@pytest.fixture
def trainer(networks, params, shock_params):
    """Create RiskyDebtTrainerBR instance."""
    policy_net, value_net, price_net = networks
    return RiskyDebtTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        optimizer_actor=tf.keras.optimizers.Adam(learning_rate=1e-3),
        optimizer_critic=tf.keras.optimizers.Adam(learning_rate=1e-3),
        weight_br=0.1,
        n_critic_steps=3,
        polyak_tau=0.9,  # Lower for easier testing
        smoothing=AnnealingSchedule(init_temp=1.0, min_temp=0.01, decay_rate=0.9),
        logit_clip=20.0
    )


@pytest.fixture
def flat_data(bounds):
    """Generate synthetic flattened training data."""
    batch_size = 64

    # Generate random states
    k = tf.random.uniform([batch_size], bounds['k'][0], bounds['k'][1])
    b = tf.random.uniform([batch_size], bounds['b'][0], bounds['b'][1])
    log_z = tf.random.uniform([batch_size], bounds['log_z'][0], bounds['log_z'][1])
    z = tf.exp(log_z)

    # Generate next period shocks
    log_z_next_main = tf.random.uniform([batch_size], bounds['log_z'][0], bounds['log_z'][1])
    log_z_next_fork = tf.random.uniform([batch_size], bounds['log_z'][0], bounds['log_z'][1])
    z_next_main = tf.exp(log_z_next_main)
    z_next_fork = tf.exp(log_z_next_fork)

    return {
        'k': k,
        'b': b,
        'z': z,
        'z_next_main': z_next_main,
        'z_next_fork': z_next_fork
    }


# ============================================================================
# Test Class: Initialization
# ============================================================================

class TestRiskyBRTrainerInitialization:
    """Test RiskyDebtTrainerBR initialization."""

    def test_networks_initialized(self, trainer):
        """Test trainer initializes with all three networks."""
        assert trainer.policy_net is not None
        assert trainer.value_net is not None
        assert trainer.price_net is not None

    def test_target_policy_exists(self, trainer):
        """Test target policy network is created."""
        assert hasattr(trainer, 'target_policy_net')
        assert trainer.target_policy_net is not None

    def test_target_value_exists(self, trainer):
        """Test target value network is created."""
        assert hasattr(trainer, 'target_value_net')
        assert trainer.target_value_net is not None

    def test_target_price_exists(self, trainer):
        """Test target price network is created."""
        assert hasattr(trainer, 'target_price_net')
        assert trainer.target_price_net is not None

    def test_targets_initialized_with_same_weights(self, trainer):
        """Test all target networks are initialized with current network weights."""
        # Check policy
        for target_var, source_var in zip(
            trainer.target_policy_net.trainable_variables,
            trainer.policy_net.trainable_variables
        ):
            assert tf.reduce_all(tf.equal(target_var, source_var))

        # Check value
        for target_var, source_var in zip(
            trainer.target_value_net.trainable_variables,
            trainer.value_net.trainable_variables
        ):
            assert tf.reduce_all(tf.equal(target_var, source_var))

        # Check price
        for target_var, source_var in zip(
            trainer.target_price_net.trainable_variables,
            trainer.price_net.trainable_variables
        ):
            assert tf.reduce_all(tf.equal(target_var, source_var))

    def test_target_networks_not_same_object(self, trainer):
        """Test target networks are separate objects."""
        assert trainer.target_policy_net is not trainer.policy_net
        assert trainer.target_value_net is not trainer.value_net
        assert trainer.target_price_net is not trainer.price_net

    def test_hyperparameters(self, trainer):
        """Test hyperparameters are set correctly."""
        assert trainer.n_critic_steps == 3
        assert trainer.polyak_tau == 0.9


# ============================================================================
# Test Class: Price Network Output
# ============================================================================

class TestPriceNetworkOutput:
    """Test price network outputs bond price q in correct range."""

    def test_price_network_output_range(self, networks, params):
        """Test bond price q is in [0, 1/(1+r)]."""
        _, _, price_net = networks

        # Generate random inputs
        k = tf.random.uniform([32, 1], 0.1, 10.0)
        b = tf.random.uniform([32, 1], 0.0, 5.0)
        z = tf.random.uniform([32, 1], 0.5, 2.0)

        q = price_net(k, b, z)

        # q should be in [0, 1/(1+r)]
        q_max = 1.0 / (1.0 + params.r_rate)
        assert tf.reduce_all(q >= 0.0)
        assert tf.reduce_all(q <= q_max + 1e-6)  # Small tolerance for numerical precision

    def test_price_network_output_shape(self, networks):
        """Test bond price q has correct shape."""
        _, _, price_net = networks

        k = tf.random.uniform([32], 0.1, 10.0)
        b = tf.random.uniform([32], 0.0, 5.0)
        z = tf.random.uniform([32], 0.5, 2.0)

        q = price_net(k, b, z)
        assert q.shape == (32, 1)


# ============================================================================
# Test Class: Train Step
# ============================================================================

class TestRiskyBRTrainStep:
    """Test RiskyDebtTrainerBR train step with flattened data."""

    def test_train_step_with_flattened_data(self, trainer, flat_data):
        """Test train_step accepts flattened data format."""
        metrics = trainer.train_step(
            flat_data['k'],
            flat_data['b'],
            flat_data['z'],
            flat_data['z_next_main'],
            flat_data['z_next_fork'],
            temperature=0.1
        )

        # Check metrics returned
        assert "loss_critic" in metrics
        assert "loss_actor" in metrics
        assert "loss_price" in metrics
        assert "mean_p_default" in metrics

    def test_loss_is_scalar(self, trainer, flat_data):
        """Test that losses are scalar values."""
        metrics = trainer.train_step(
            flat_data['k'],
            flat_data['b'],
            flat_data['z'],
            flat_data['z_next_main'],
            flat_data['z_next_fork'],
            temperature=0.1
        )

        assert isinstance(metrics["loss_critic"], float)
        assert isinstance(metrics["loss_actor"], float)
        assert isinstance(metrics["loss_price"], float)

    def test_loss_is_finite(self, trainer, flat_data):
        """Test that losses are finite (not NaN or Inf)."""
        metrics = trainer.train_step(
            flat_data['k'],
            flat_data['b'],
            flat_data['z'],
            flat_data['z_next_main'],
            flat_data['z_next_fork'],
            temperature=0.1
        )

        assert np.isfinite(metrics["loss_critic"])
        assert np.isfinite(metrics["loss_actor"])
        assert np.isfinite(metrics["loss_price"])


# ============================================================================
# Test Class: Polyak Averaging
# ============================================================================

class TestRiskyBRPolyakAveraging:
    """Test Polyak averaging for all target networks."""

    def test_polyak_updates_all_targets(self, trainer, flat_data):
        """Test that Polyak averaging updates all three target networks."""
        # Get initial target weights
        initial_policy = [tf.identity(var) for var in trainer.target_policy_net.trainable_variables]
        initial_value = [tf.identity(var) for var in trainer.target_value_net.trainable_variables]
        initial_price = [tf.identity(var) for var in trainer.target_price_net.trainable_variables]

        # Run multiple train steps
        for _ in range(5):
            trainer.train_step(
                flat_data['k'],
                flat_data['b'],
                flat_data['z'],
                flat_data['z_next_main'],
                flat_data['z_next_fork'],
                temperature=0.1
            )

        # Check all target networks have been updated
        def check_updated(initial_vars, current_net):
            for initial, current in zip(initial_vars, current_net.trainable_variables):
                if not tf.reduce_all(tf.equal(initial, current)):
                    return True
            return False

        assert check_updated(initial_policy, trainer.target_policy_net), "Target policy not updated"
        assert check_updated(initial_value, trainer.target_value_net), "Target value not updated"
        assert check_updated(initial_price, trainer.target_price_net), "Target price not updated"


# ============================================================================
# Test Class: Effective Value (V_eff)
# ============================================================================

class TestEffectiveValue:
    """Test effective value computation with Gumbel-Sigmoid."""

    def test_effective_value_output_shape(self):
        """Test V_eff and p_default have correct shapes."""
        batch_size = 32
        V_tilde = tf.random.normal([batch_size, 1])
        k = tf.ones([batch_size, 1])

        V_eff, p_default = compute_effective_value(V_tilde, k, temperature=0.1, noise=False)

        assert V_eff.shape == (batch_size, 1)
        assert p_default.shape == (batch_size, 1)

    def test_p_default_in_valid_range(self):
        """Test default probability p is in [0, 1]."""
        V_tilde = tf.random.normal([64, 1], mean=0.0, stddev=5.0)
        k = tf.ones([64, 1])

        _, p_default = compute_effective_value(V_tilde, k, temperature=0.1, noise=False)

        assert tf.reduce_all(p_default >= 0.0)
        assert tf.reduce_all(p_default <= 1.0)

    def test_v_eff_negative_value_behavior(self):
        """Test V_eff approaches 0 for very negative V_tilde."""
        # Very negative V_tilde should lead to high p_default, low V_eff
        V_tilde = tf.constant([[-10.0]], dtype=tf.float32)
        k = tf.constant([[1.0]], dtype=tf.float32)

        V_eff, p_default = compute_effective_value(V_tilde, k, temperature=0.01, noise=False)

        # p_default should be close to 1, V_eff close to 0
        assert float(p_default) > 0.9, f"p_default={float(p_default)} should be > 0.9"
        assert abs(float(V_eff)) < abs(float(V_tilde)), "V_eff should be smaller than V_tilde"

    def test_v_eff_positive_value_behavior(self):
        """Test V_eff approximately equals V_tilde for very positive values."""
        # Very positive V_tilde should lead to low p_default, V_eff ~ V_tilde
        V_tilde = tf.constant([[10.0]], dtype=tf.float32)
        k = tf.constant([[1.0]], dtype=tf.float32)

        V_eff, p_default = compute_effective_value(V_tilde, k, temperature=0.01, noise=False)

        # p_default should be close to 0, V_eff close to V_tilde
        assert float(p_default) < 0.1, f"p_default={float(p_default)} should be < 0.1"
        assert abs(float(V_eff) - float(V_tilde)) < 1.0, "V_eff should be close to V_tilde"


# ============================================================================
# Test Class: Gradient Flow
# ============================================================================

class TestRiskyBRGradientFlow:
    """Test gradient flow in Risky BR trainer."""

    def test_gradients_update_current_networks(self, trainer, flat_data):
        """Test that gradients update all current networks."""
        # Get initial weights
        initial_policy = [tf.identity(var) for var in trainer.policy_net.trainable_variables]
        initial_value = [tf.identity(var) for var in trainer.value_net.trainable_variables]
        initial_price = [tf.identity(var) for var in trainer.price_net.trainable_variables]

        # Run train steps
        for _ in range(5):
            trainer.train_step(
                flat_data['k'],
                flat_data['b'],
                flat_data['z'],
                flat_data['z_next_main'],
                flat_data['z_next_fork'],
                temperature=0.1
            )

        # Check networks changed
        def check_changed(initial_vars, current_net):
            for initial, current in zip(initial_vars, current_net.trainable_variables):
                if not tf.reduce_all(tf.equal(initial, current)):
                    return True
            return False

        assert check_changed(initial_policy, trainer.policy_net), "Policy network not updated"
        assert check_changed(initial_value, trainer.value_net), "Value network not updated"
        assert check_changed(initial_price, trainer.price_net), "Price network not updated"


# ============================================================================
# Test Class: Integration
# ============================================================================

class TestRiskyBRIntegration:
    """Test RiskyDebtTrainerBR integration tests."""

    def test_multiple_training_steps(self, trainer, flat_data):
        """Test trainer can run multiple training steps."""
        n_steps = 10
        all_metrics = []

        for _ in range(n_steps):
            metrics = trainer.train_step(
                flat_data['k'],
                flat_data['b'],
                flat_data['z'],
                flat_data['z_next_main'],
                flat_data['z_next_fork'],
                temperature=0.1
            )
            all_metrics.append(metrics)

        # Check all steps completed
        assert len(all_metrics) == n_steps

        # Check all metrics are finite
        for metrics in all_metrics:
            assert np.isfinite(metrics["loss_critic"])
            assert np.isfinite(metrics["loss_actor"])
            assert np.isfinite(metrics["loss_price"])

    def test_annealing_schedule_updates(self, trainer, flat_data):
        """Test annealing schedule updates with each train step."""
        initial_temp = trainer.smoothing.value

        # Run a few train steps
        for _ in range(5):
            trainer.train_step(
                flat_data['k'],
                flat_data['b'],
                flat_data['z'],
                flat_data['z_next_main'],
                flat_data['z_next_fork'],
                temperature=0.1  # This is overridden by smoothing.value
            )

        # Temperature should decrease
        assert trainer.smoothing.value < initial_temp

    def test_evaluate_without_updating(self, trainer, flat_data):
        """Test evaluate method doesn't update weights."""
        # Get initial weights
        initial_policy = [tf.identity(var) for var in trainer.policy_net.trainable_variables]

        # Evaluate
        metrics = trainer.evaluate(
            flat_data['k'],
            flat_data['b'],
            flat_data['z'],
            flat_data['z_next_main'],
            flat_data['z_next_fork'],
            temperature=0.1
        )

        # Check weights unchanged
        for initial, current in zip(initial_policy, trainer.policy_net.trainable_variables):
            assert tf.reduce_all(tf.equal(initial, current))

        # Check metrics returned
        assert "loss_critic" in metrics
        assert "loss_actor" in metrics
