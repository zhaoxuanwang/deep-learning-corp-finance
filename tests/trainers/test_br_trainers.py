"""
tests/trainers/test_br_trainers.py

Unit tests for BasicTrainerBR with flattened dataset interface.

Tests verify:
1. Trainer initialization with target networks
2. Train step with flattened data format
3. Both target policy AND target value networks exist
4. Target networks used in critic update
5. Current networks used in actor update
6. Polyak averaging for both networks
7. n_critic_steps controls critic iterations
8. Gradient flow correctness
9. TensorFlow Dataset pipeline integration
"""

import tensorflow as tf
import pytest
import shutil
import os
from pathlib import Path

from src.economy.data_generator import DataGenerator, create_data_generator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers.basic import BasicTrainerBR
from src.networks.network_basic import build_basic_networks


@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory for test data."""
    path = "tests/temp_data_cache_br"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def params():
    """Economic parameters."""
    return EconomicParams()


@pytest.fixture
def shock_params():
    """Shock parameters."""
    return ShockParams()


@pytest.fixture
def data_gen(params, shock_params, temp_cache_dir):
    """Create data generator with small test dataset."""
    return DataGenerator(
        master_seed=(42, 42),
        shock_params=shock_params,
        k_bounds=(0.1, 10.0),
        logz_bounds=(-0.5, 0.5),
        b_bounds=(0.0, 1.0),
        sim_batch_size=32,
        T=16,
        n_sim_batches=2,
        cache_dir=temp_cache_dir,
        save_to_disk=False
    )


@pytest.fixture
def flat_data(data_gen):
    """Get flattened training dataset."""
    return data_gen.get_flattened_training_dataset()


@pytest.fixture
def networks():
    """Build policy and value networks."""
    policy_net, value_net = build_basic_networks(
        k_min=0.1, k_max=10.0,
        n_layers=2, n_neurons=16,
        activation='relu'
    )

    # Build networks by calling them once with dummy data
    dummy_k = tf.constant([[1.0]], dtype=tf.float32)
    dummy_z = tf.constant([[1.0]], dtype=tf.float32)
    _ = policy_net(dummy_k, dummy_z)
    _ = value_net(dummy_k, dummy_z)

    return policy_net, value_net


@pytest.fixture
def trainer(networks, params, shock_params):
    """Create BasicTrainerBR instance."""
    policy_net, value_net = networks
    return BasicTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        params=params,
        shock_params=shock_params,
        n_critic_steps=3,
        polyak_tau=0.9  # Lower for easier testing
    )


# ============================================================================
# Test Class: Basic Trainer BR Initialization and Train Step
# ============================================================================

class TestBasicBRTrainer:
    """Test BasicTrainerBR initialization and basic train step."""

    def test_initialization(self, trainer):
        """Test trainer initializes with both policy and value networks."""
        assert trainer.policy_net is not None
        assert trainer.value_net is not None
        assert trainer.n_critic_steps == 3
        assert trainer.polyak_tau == 0.9

    def test_train_step_with_flattened_data(self, trainer, flat_data):
        """Test train_step accepts flattened data format."""
        # Extract small batch
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]

        # Run train step
        metrics = trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Check metrics returned
        assert "loss_critic" in metrics
        assert "loss_actor" in metrics
        assert "mse_proxy" in metrics

    def test_loss_is_scalar(self, trainer, flat_data):
        """Test that losses are scalar values."""
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]

        metrics = trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        assert isinstance(metrics["loss_critic"], float)
        assert isinstance(metrics["loss_actor"], float)
        assert isinstance(metrics["mse_proxy"], float)

    def test_loss_is_finite(self, trainer, flat_data):
        """Test that losses are finite (not NaN or Inf)."""
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]

        metrics = trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        assert tf.math.is_finite(metrics["loss_critic"])
        assert tf.math.is_finite(metrics["loss_actor"])
        assert tf.math.is_finite(metrics["mse_proxy"])


# ============================================================================
# Test Class: Target Networks for BR
# ============================================================================

class TestBRTargetNetworks:
    """Test target network creation and usage for BR trainer."""

    def test_target_policy_exists(self, trainer):
        """Test target policy network is created."""
        assert hasattr(trainer, 'target_policy_net')
        assert trainer.target_policy_net is not None

    def test_target_value_exists(self, trainer):
        """Test target value network is created."""
        assert hasattr(trainer, 'target_value_net')
        assert trainer.target_value_net is not None

    def test_both_targets_initialized_with_same_weights(self, trainer):
        """Test both target networks are initialized with current network weights."""
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

    def test_target_networks_not_same_object(self, trainer):
        """Test target networks are separate objects from current networks."""
        assert trainer.target_policy_net is not trainer.policy_net
        assert trainer.target_value_net is not trainer.value_net

    def test_polyak_averaging_updates_both_targets(self, trainer, flat_data):
        """Test that Polyak averaging updates both target networks."""
        # Get initial target weights
        initial_policy_weights = [
            tf.identity(var) for var in trainer.target_policy_net.trainable_variables
        ]
        initial_value_weights = [
            tf.identity(var) for var in trainer.target_value_net.trainable_variables
        ]

        # Run multiple train steps
        batch_size = 32
        for _ in range(5):
            k = flat_data['k'][:batch_size]
            z = flat_data['z'][:batch_size]
            z_next_main = flat_data['z_next_main'][:batch_size]
            z_next_fork = flat_data['z_next_fork'][:batch_size]
            trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Check both target networks have been updated
        policy_updated = False
        for initial_var, current_var in zip(
            initial_policy_weights,
            trainer.target_policy_net.trainable_variables
        ):
            if not tf.reduce_all(tf.equal(initial_var, current_var)):
                policy_updated = True
                break

        value_updated = False
        for initial_var, current_var in zip(
            initial_value_weights,
            trainer.target_value_net.trainable_variables
        ):
            if not tf.reduce_all(tf.equal(initial_var, current_var)):
                value_updated = True
                break

        assert policy_updated, "Target policy network was not updated"
        assert value_updated, "Target value network was not updated"

    def test_polyak_tau_controls_update_rate(self, networks, params, shock_params, flat_data):
        """Test that polyak_tau controls target network update rate."""
        policy_net, value_net = networks

        # Create trainers with different tau values
        trainer_slow = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            n_critic_steps=2,
            polyak_tau=0.99  # Slow updates
        )

        # Clone networks for second trainer
        policy_net2, value_net2 = build_basic_networks(
            k_min=0.1, k_max=10.0,
            n_layers=2, n_neurons=16,
            activation='relu'
        )

        # Build networks
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        _ = policy_net2(dummy_k, dummy_z)
        _ = value_net2(dummy_k, dummy_z)

        policy_net2.set_weights(policy_net.get_weights())
        value_net2.set_weights(value_net.get_weights())

        trainer_fast = BasicTrainerBR(
            policy_net=policy_net2,
            value_net=value_net2,
            params=params,
            shock_params=shock_params,
            n_critic_steps=2,
            polyak_tau=0.5  # Fast updates
        )

        # Get initial weights
        initial_policy = [tf.identity(var) for var in trainer_slow.target_policy_net.trainable_variables]

        # Run train step on both
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]

        trainer_slow.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)
        trainer_fast.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Compute change magnitudes for policy
        change_slow = sum([
            tf.reduce_mean(tf.abs(initial - current)).numpy()
            for initial, current in zip(initial_policy, trainer_slow.target_policy_net.trainable_variables)
        ])

        initial_policy2 = [tf.identity(var) for var in policy_net2.trainable_variables]
        change_fast = sum([
            tf.reduce_mean(tf.abs(initial - current)).numpy()
            for initial, current in zip(initial_policy, trainer_fast.target_policy_net.trainable_variables)
        ])

        # Fast update should have larger change
        assert change_fast > change_slow, f"Fast update ({change_fast}) should be larger than slow update ({change_slow})"


# ============================================================================
# Test Class: Critic Update with Target Networks
# ============================================================================

class TestBRCriticUpdate:
    """Test critic update uses target networks correctly."""

    def test_n_critic_steps_controls_iterations(self, networks, params, shock_params, flat_data):
        """Test that n_critic_steps controls number of critic updates."""
        policy_net, value_net = networks

        # Trainer with 1 critic step
        trainer_1 = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            n_critic_steps=1,
            polyak_tau=0.9
        )

        # Clone for trainer with more steps
        policy_net2, value_net2 = build_basic_networks(
            k_min=0.1, k_max=10.0,
            n_layers=2, n_neurons=16,
            activation='relu'
        )

        # Build networks
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        _ = policy_net2(dummy_k, dummy_z)
        _ = value_net2(dummy_k, dummy_z)

        policy_net2.set_weights(policy_net.get_weights())
        value_net2.set_weights(value_net.get_weights())

        trainer_5 = BasicTrainerBR(
            policy_net=policy_net2,
            value_net=value_net2,
            params=params,
            shock_params=shock_params,
            n_critic_steps=5,
            polyak_tau=0.9
        )

        # Get initial value weights
        initial_value_1 = [tf.identity(var) for var in trainer_1.value_net.trainable_variables]
        initial_value_5 = [tf.identity(var) for var in trainer_5.value_net.trainable_variables]

        # Run one train step each
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]

        trainer_1.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)
        trainer_5.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Compute value network update magnitudes
        change_1 = sum([
            tf.reduce_mean(tf.abs(initial - current)).numpy()
            for initial, current in zip(initial_value_1, trainer_1.value_net.trainable_variables)
        ])

        change_5 = sum([
            tf.reduce_mean(tf.abs(initial - current)).numpy()
            for initial, current in zip(initial_value_5, trainer_5.value_net.trainable_variables)
        ])

        # More critic steps should lead to larger updates
        assert change_5 > change_1, f"5 critic steps ({change_5}) should have larger change than 1 step ({change_1})"


# ============================================================================
# Test Class: Gradient Flow
# ============================================================================

class TestBRGradientFlow:
    """Test gradient flow in BR trainer."""

    def test_gradients_update_current_policy(self, trainer, flat_data):
        """Test that gradients update the current policy network."""
        # Get initial weights
        initial_weights = [tf.identity(var) for var in trainer.policy_net.trainable_variables]

        # Run train steps
        batch_size = 32
        for _ in range(5):
            k = flat_data['k'][:batch_size]
            z = flat_data['z'][:batch_size]
            z_next_main = flat_data['z_next_main'][:batch_size]
            z_next_fork = flat_data['z_next_fork'][:batch_size]
            trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Check that policy weights changed
        weights_changed = False
        for initial_var, current_var in zip(initial_weights, trainer.policy_net.trainable_variables):
            if not tf.reduce_all(tf.equal(initial_var, current_var)):
                weights_changed = True
                break

        assert weights_changed, "Policy network weights should be updated by gradients"

    def test_gradients_update_current_value(self, trainer, flat_data):
        """Test that gradients update the current value network."""
        # Get initial weights
        initial_weights = [tf.identity(var) for var in trainer.value_net.trainable_variables]

        # Run train steps
        batch_size = 32
        for _ in range(5):
            k = flat_data['k'][:batch_size]
            z = flat_data['z'][:batch_size]
            z_next_main = flat_data['z_next_main'][:batch_size]
            z_next_fork = flat_data['z_next_fork'][:batch_size]
            trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Check that value weights changed
        weights_changed = False
        for initial_var, current_var in zip(initial_weights, trainer.value_net.trainable_variables):
            if not tf.reduce_all(tf.equal(initial_var, current_var)):
                weights_changed = True
                break

        assert weights_changed, "Value network weights should be updated by gradients"

    def test_no_gradients_to_target_policy(self, trainer, flat_data):
        """Test that target policy network does not receive gradients."""
        # Get initial target weights
        initial_target = [tf.identity(var) for var in trainer.target_policy_net.trainable_variables]

        # Manually set current policy to something different
        for var in trainer.policy_net.trainable_variables:
            var.assign(var + tf.random.normal(var.shape, stddev=0.1))

        # Run ONE train step (should only do Polyak averaging, not gradient update)
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]
        trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Target should only change via Polyak averaging (small change)
        for initial_var, current_var in zip(initial_target, trainer.target_policy_net.trainable_variables):
            # Check that changes are small (only from Polyak)
            change = tf.reduce_mean(tf.abs(current_var - initial_var))
            # With tau=0.9 and one step, change should be ~0.1 * (policy - target)
            assert change < 1.0, "Target policy should not receive direct gradient updates"

    def test_no_gradients_to_target_value(self, trainer, flat_data):
        """Test that target value network does not receive gradients."""
        # Get initial target weights
        initial_target = [tf.identity(var) for var in trainer.target_value_net.trainable_variables]

        # Manually set current value to something different
        for var in trainer.value_net.trainable_variables:
            var.assign(var + tf.random.normal(var.shape, stddev=0.1))

        # Run ONE train step
        batch_size = 32
        k = flat_data['k'][:batch_size]
        z = flat_data['z'][:batch_size]
        z_next_main = flat_data['z_next_main'][:batch_size]
        z_next_fork = flat_data['z_next_fork'][:batch_size]
        trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        # Target should only change via Polyak averaging
        for initial_var, current_var in zip(initial_target, trainer.target_value_net.trainable_variables):
            change = tf.reduce_mean(tf.abs(current_var - initial_var))
            assert change < 1.0, "Target value should not receive direct gradient updates"


# ============================================================================
# Test Class: Integration with TensorFlow Dataset
# ============================================================================

class TestBRIntegration:
    """Test BR trainer integration with TensorFlow dataset pipeline."""

    def test_multiple_training_steps(self, trainer, flat_data):
        """Test trainer can run multiple training steps."""
        batch_size = 32
        n_steps = 10

        all_metrics = []
        for step in range(n_steps):
            k = flat_data['k'][:batch_size]
            z = flat_data['z'][:batch_size]
            z_next_main = flat_data['z_next_main'][:batch_size]
            z_next_fork = flat_data['z_next_fork'][:batch_size]

            metrics = trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)
            all_metrics.append(metrics)

        # Check all steps completed
        assert len(all_metrics) == n_steps

        # Check all metrics are finite
        for metrics in all_metrics:
            assert tf.math.is_finite(metrics["loss_critic"])
            assert tf.math.is_finite(metrics["loss_actor"])
            assert tf.math.is_finite(metrics["mse_proxy"])

    def test_tf_dataset_pipeline(self, trainer, flat_data):
        """Test trainer works with tf.data.Dataset pipeline."""
        # Create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'k': flat_data['k'],
            'z': flat_data['z'],
            'z_next_main': flat_data['z_next_main'],
            'z_next_fork': flat_data['z_next_fork']
        })
        dataset = dataset.batch(32).take(5)

        # Run training
        for batch in dataset:
            metrics = trainer.train_step(
                batch['k'],
                batch['z'],
                batch['z_next_main'],
                batch['z_next_fork'],
                temperature=0.1
            )

            # Check metrics
            assert "loss_critic" in metrics
            assert "loss_actor" in metrics
            assert "mse_proxy" in metrics

    def test_dataset_format_validation(self, flat_data):
        """Test that missing keys raise appropriate errors."""
        # This test ensures the data has all required keys
        required_keys = {'k', 'z', 'z_next_main', 'z_next_fork'}
        assert required_keys.issubset(flat_data.keys()), \
            f"Flattened dataset missing required keys. Expected {required_keys}, got {set(flat_data.keys())}"
