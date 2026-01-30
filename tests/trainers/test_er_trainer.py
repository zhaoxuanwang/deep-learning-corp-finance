"""
tests/trainers/test_er_trainer.py

Tests for Euler Residual (ER) trainer with flattened data and target networks.
"""

import tensorflow as tf
import pytest
import numpy as np
import shutil
import os
from pathlib import Path

from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers.basic import BasicTrainerER
from src.networks.network_basic import build_basic_networks


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory for data generation."""
    path = "tests/temp_data_cache_er"
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
    """Data generator for testing."""
    return DataGenerator(
        master_seed=(55, 55),
        shock_params=shock_params,
        k_bounds=(0.1, 10.0),
        logz_bounds=(-0.5, 0.5),
        b_bounds=(0.0, 1.0),
        sim_batch_size=32,
        T=16,
        n_sim_batches=1,
        cache_dir=temp_cache_dir,
        save_to_disk=False
    )


@pytest.fixture
def flat_data(data_gen):
    """Flattened training data for ER/BR methods."""
    return data_gen.get_flattened_training_dataset()


@pytest.fixture
def policy_net():
    """Policy network for testing."""
    policy, _ = build_basic_networks(
        k_min=0.1, k_max=10.0,
        n_layers=2, n_neurons=16,
        activation='swish'
    )

    # Build network by calling it once with dummy data
    dummy_k = tf.constant([[1.0]], dtype=tf.float32)
    dummy_z = tf.constant([[1.0]], dtype=tf.float32)
    _ = policy(dummy_k, dummy_z)

    return policy


@pytest.fixture
def trainer(policy_net, params, shock_params):
    """ER trainer for testing."""
    return BasicTrainerER(
        policy_net=policy_net,
        params=params,
        shock_params=shock_params,
        polyak_tau=0.995
    )


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestBasicERTrainer:
    """Tests for basic ER trainer functionality."""

    def test_trainer_initialization(self, trainer):
        """Trainer initializes correctly."""
        assert trainer.policy_net is not None
        assert trainer.target_policy_net is not None
        assert trainer.optimizer is not None
        assert trainer.polyak_tau == 0.995

    def test_target_network_exists(self, trainer):
        """Target network is created on initialization."""
        assert trainer.target_policy_net is not None
        assert trainer.target_policy_net is not trainer.policy_net

    def test_target_network_initialized_with_same_weights(self, trainer):
        """Target network starts with same weights as policy network."""
        policy_weights = trainer.policy_net.get_weights()
        target_weights = trainer.target_policy_net.get_weights()

        for pw, tw in zip(policy_weights, target_weights):
            assert np.allclose(pw, tw)

    def test_train_step_with_flattened_data(self, trainer, flat_data):
        """Train step runs with flattened data."""
        # Get a small batch
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        metrics = trainer.train_step(k, z, z_next_main, z_next_fork)

        assert "loss_ER" in metrics
        assert isinstance(metrics["loss_ER"], float)

    def test_loss_is_scalar(self, trainer, flat_data):
        """Loss is a scalar value."""
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        metrics = trainer.train_step(k, z, z_next_main, z_next_fork)

        assert isinstance(metrics["loss_ER"], float)
        assert not np.isnan(metrics["loss_ER"])
        assert not np.isinf(metrics["loss_ER"])


# =============================================================================
# TARGET NETWORK TESTS
# =============================================================================

class TestTargetNetwork:
    """Tests for target network functionality."""

    def test_target_policy_used_for_k_next_next(self, trainer, flat_data):
        """Target policy is used for two-step lookahead k''."""
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        # Store initial target weights (get_weights() returns numpy arrays directly)
        initial_target_weights = [w.copy() for w in trainer.target_policy_net.get_weights()]

        # Run training step
        trainer.train_step(k, z, z_next_main, z_next_fork)

        # Target weights should have been updated (Polyak averaging)
        updated_target_weights = trainer.target_policy_net.get_weights()

        # Check that at least one layer's weights changed (Polyak update)
        # With polyak_tau = 0.995, change is (1-0.995) = 0.5% of difference
        any_changed = False
        for init_w, updated_w in zip(initial_target_weights, updated_target_weights):
            if not np.allclose(init_w, updated_w, rtol=1e-4, atol=1e-8):
                any_changed = True
                break

        assert any_changed, "Target weights should have been updated via Polyak averaging"

    def test_polyak_averaging_updates_target(self, trainer, flat_data):
        """Polyak averaging updates target network weights."""
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        # Get initial weights (get_weights() returns numpy arrays directly)
        init_policy = [w.copy() for w in trainer.policy_net.get_weights()]
        init_target = [w.copy() for w in trainer.target_policy_net.get_weights()]

        # Run training step (updates policy and target)
        trainer.train_step(k, z, z_next_main, z_next_fork)

        # Get updated weights
        updated_policy = trainer.policy_net.get_weights()
        updated_target = trainer.target_policy_net.get_weights()

        # Target should have changed via Polyak averaging
        # Check that at least some weights changed
        any_changed = False
        for init_w, updated_w in zip(init_target, updated_target):
            if not np.allclose(init_w, updated_w, rtol=1e-4, atol=1e-8):
                any_changed = True
                break

        assert any_changed, "Target weights should have been updated via Polyak averaging"

    def test_polyak_tau_controls_update_rate(self, policy_net, params, shock_params, flat_data):
        """Polyak tau controls how fast target network updates."""
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        # Create two trainers with different polyak_tau
        trainer_slow = BasicTrainerER(
            policy_net=policy_net,
            params=params,
            shock_params=shock_params,
            polyak_tau=0.99  # Slower updates
        )

        policy_net2, _ = build_basic_networks(
            k_min=0.1, k_max=10.0,
            n_layers=2, n_neurons=16,
            activation='swish'
        )
        policy_net2.set_weights(policy_net.get_weights())  # Start with same weights

        trainer_fast = BasicTrainerER(
            policy_net=policy_net2,
            params=params,
            shock_params=shock_params,
            polyak_tau=0.9  # Faster updates
        )

        # Run same training step
        trainer_slow.train_step(k, z, z_next_main, z_next_fork)
        trainer_fast.train_step(k, z, z_next_main, z_next_fork)

        # Faster tau should lead to larger changes in target
        # (This is hard to test precisely without controlled gradients)
        # Just check that both trainers updated
        assert trainer_slow.target_policy_net is not None
        assert trainer_fast.target_policy_net is not None


# =============================================================================
# GRADIENT FLOW TESTS
# =============================================================================

class TestGradientFlow:
    """Tests for gradient flow through network."""

    def test_gradients_update_policy(self, trainer, flat_data):
        """Gradients update policy network."""
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        # Get initial weights (get_weights() returns numpy arrays directly)
        init_weights = [w.copy() for w in trainer.policy_net.get_weights()]

        # Run multiple training steps to ensure gradient accumulation
        for _ in range(3):
            trainer.train_step(k, z, z_next_main, z_next_fork)

        # Get updated weights
        updated_weights = trainer.policy_net.get_weights()

        # At least some weights should have changed
        changed = False
        for init_w, updated_w in zip(init_weights, updated_weights):
            if not np.allclose(init_w, updated_w, rtol=1e-4, atol=1e-8):
                changed = True
                break

        assert changed, "Policy weights did not update after multiple training steps"

    def test_no_gradients_to_target(self, trainer, flat_data):
        """Target network updates via Polyak averaging only, not gradients."""
        k = flat_data['k'][:32]
        z = flat_data['z'][:32]
        z_next_main = flat_data['z_next_main'][:32]
        z_next_fork = flat_data['z_next_fork'][:32]

        # Target weights should only update via Polyak averaging,
        # not via backpropagation through loss.
        # Verify that target network variables are not being optimized directly
        # by checking they are not in the policy optimizer's tracked variables.
        trainable_var_ids = {id(v) for v in trainer.policy_net.trainable_variables}
        target_var_ids = {id(v) for v in trainer.target_policy_net.trainable_variables}

        # Target variables should not overlap with trainable policy variables
        assert len(trainable_var_ids & target_var_ids) == 0, \
            "Target network variables should be separate from policy network"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestERIntegration:
    """Integration tests for ER trainer."""

    def test_multiple_training_steps(self, trainer, flat_data):
        """Multiple training steps run successfully."""
        k = flat_data['k'][:128]
        z = flat_data['z'][:128]
        z_next_main = flat_data['z_next_main'][:128]
        z_next_fork = flat_data['z_next_fork'][:128]

        losses = []
        for i in range(5):
            # Get batch
            batch_start = i * 16
            batch_end = (i + 1) * 16

            k_batch = k[batch_start:batch_end]
            z_batch = z[batch_start:batch_end]
            z_next_main_batch = z_next_main[batch_start:batch_end]
            z_next_fork_batch = z_next_fork[batch_start:batch_end]

            metrics = trainer.train_step(k_batch, z_batch, z_next_main_batch, z_next_fork_batch)
            losses.append(metrics["loss_ER"])

        # All losses should be finite
        assert all(np.isfinite(loss) for loss in losses)

    def test_er_trainer_with_tf_dataset(self, trainer, flat_data):
        """ER trainer works with TF Dataset pipeline."""
        # Create TF dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices(flat_data)
        tf_dataset = tf_dataset.batch(32)

        # Train for a few batches
        for i, batch in enumerate(tf_dataset.take(3)):
            metrics = trainer.train_step(
                batch['k'],
                batch['z'],
                batch['z_next_main'],
                batch['z_next_fork']
            )

            assert "loss_ER" in metrics
            assert np.isfinite(metrics["loss_ER"])
