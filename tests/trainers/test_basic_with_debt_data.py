"""
tests/trainers/test_basic_with_debt_data.py

Tests that Basic model trainers (LR, ER, BR) correctly ignore the debt column
when data is generated with include_debt=True.

This ensures that the shared data pipeline (Option A) works correctly:
- Data is generated once with include_debt=True
- Basic model uses k, z only (ignores b)
- Risky model uses k, b, z

Reference: The Basic model only has (k, z) as state variables.
"""

import tensorflow as tf
import pytest
import numpy as np
import shutil
import os

from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers.basic import BasicTrainerLR, BasicTrainerER, BasicTrainerBR
from src.networks.network_basic import build_basic_networks


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory for data generation."""
    path = "tests/temp_data_cache_debt"
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
def data_gen(shock_params, temp_cache_dir):
    """Data generator for testing."""
    return DataGenerator(
        master_seed=(99, 99),
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
def flat_data_with_debt(data_gen):
    """Flattened data WITH debt dimension."""
    return data_gen.get_flattened_training_dataset(include_debt=True)


@pytest.fixture
def flat_data_without_debt(data_gen):
    """Flattened data WITHOUT debt dimension."""
    return data_gen.get_flattened_training_dataset(include_debt=False)


@pytest.fixture
def traj_data(data_gen):
    """Trajectory data for LR method."""
    return data_gen.get_training_dataset()


@pytest.fixture
def networks():
    """Build policy and value networks."""
    policy_net, value_net = build_basic_networks(
        k_min=0.1, k_max=10.0,
        logz_min=-0.5, logz_max=0.5,
        n_layers=2, n_neurons=16,
        activation='swish'
    )

    # Build networks by calling once
    dummy_k = tf.constant([[1.0]], dtype=tf.float32)
    dummy_z = tf.constant([[1.0]], dtype=tf.float32)
    _ = policy_net(dummy_k, dummy_z)
    _ = value_net(dummy_k, dummy_z)

    return policy_net, value_net


# =============================================================================
# TEST: Data Contains Expected Keys
# =============================================================================

class TestDataFormat:
    """Test that data formats have expected keys."""

    def test_data_with_debt_has_b_key(self, flat_data_with_debt):
        """Data with include_debt=True contains 'b' key."""
        assert 'b' in flat_data_with_debt, \
            "Flattened data with include_debt=True should contain 'b' key"

    def test_data_without_debt_no_b_key(self, flat_data_without_debt):
        """Data with include_debt=False does not contain 'b' key."""
        assert 'b' not in flat_data_without_debt, \
            "Flattened data with include_debt=False should not contain 'b' key"

    def test_common_keys_present(self, flat_data_with_debt, flat_data_without_debt):
        """Both data formats have common required keys."""
        required_keys = {'k', 'z', 'z_next_main', 'z_next_fork'}

        assert required_keys.issubset(flat_data_with_debt.keys()), \
            f"Data with debt missing keys: {required_keys - set(flat_data_with_debt.keys())}"

        assert required_keys.issubset(flat_data_without_debt.keys()), \
            f"Data without debt missing keys: {required_keys - set(flat_data_without_debt.keys())}"

    def test_k_z_values_same(self, flat_data_with_debt, flat_data_without_debt):
        """k and z values are identical regardless of include_debt."""
        # The same data generator should produce identical k, z values
        assert np.allclose(flat_data_with_debt['k'].numpy(), flat_data_without_debt['k'].numpy()), \
            "k values should be identical"
        assert np.allclose(flat_data_with_debt['z'].numpy(), flat_data_without_debt['z'].numpy()), \
            "z values should be identical"


# =============================================================================
# TEST: Basic ER Trainer Ignores Debt
# =============================================================================

class TestBasicERIgnoresDebt:
    """Test that BasicTrainerER works correctly with data containing debt."""

    def test_er_train_step_with_debt_data(self, networks, params, shock_params, flat_data_with_debt):
        """ER trainer runs successfully with data containing 'b' key."""
        policy_net, _ = networks

        trainer = BasicTrainerER(
            policy_net=policy_net,
            params=params,
            shock_params=shock_params,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            polyak_tau=0.995
        )

        # Extract batch (ignoring 'b')
        k = flat_data_with_debt['k'][:32]
        z = flat_data_with_debt['z'][:32]
        z_next_main = flat_data_with_debt['z_next_main'][:32]
        z_next_fork = flat_data_with_debt['z_next_fork'][:32]

        # This should work - trainer uses only k, z
        metrics = trainer.train_step(k, z, z_next_main, z_next_fork)

        assert "loss_ER" in metrics
        assert np.isfinite(metrics["loss_ER"])

    def test_er_produces_same_loss_with_or_without_debt_data(
        self, params, shock_params, flat_data_with_debt, flat_data_without_debt
    ):
        """ER trainer produces identical loss whether data has 'b' or not."""
        # Build two identical networks
        policy_net1, _ = build_basic_networks(k_min=0.1, k_max=10.0, logz_min=-0.5, logz_max=0.5, n_layers=2, n_neurons=16, activation='swish')
        policy_net2, _ = build_basic_networks(k_min=0.1, k_max=10.0, logz_min=-0.5, logz_max=0.5, n_layers=2, n_neurons=16, activation='swish')

        # Build them
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        _ = policy_net1(dummy_k, dummy_z)
        _ = policy_net2(dummy_k, dummy_z)

        # Set identical weights
        policy_net2.set_weights(policy_net1.get_weights())

        # Create two trainers
        trainer1 = BasicTrainerER(
            policy_net=policy_net1,
            params=params,
            shock_params=shock_params,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            polyak_tau=0.995
        )

        trainer2 = BasicTrainerER(
            policy_net=policy_net2,
            params=params,
            shock_params=shock_params,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            polyak_tau=0.995
        )

        # Evaluate (not train) to get deterministic losses
        # Using same underlying k, z values
        # Note: BasicTrainerER.evaluate() does not take temperature argument
        batch_size = 32

        metrics1 = trainer1.evaluate(
            flat_data_with_debt['k'][:batch_size],
            flat_data_with_debt['z'][:batch_size],
            flat_data_with_debt['z_next_main'][:batch_size],
            flat_data_with_debt['z_next_fork'][:batch_size]
        )

        metrics2 = trainer2.evaluate(
            flat_data_without_debt['k'][:batch_size],
            flat_data_without_debt['z'][:batch_size],
            flat_data_without_debt['z_next_main'][:batch_size],
            flat_data_without_debt['z_next_fork'][:batch_size]
        )

        # Losses should be identical (trainer ignores 'b')
        assert np.isclose(metrics1["loss_ER"], metrics2["loss_ER"], rtol=1e-5), \
            f"ER loss should be identical: {metrics1['loss_ER']} vs {metrics2['loss_ER']}"


# =============================================================================
# TEST: Basic BR Trainer Ignores Debt
# =============================================================================

class TestBasicBRIgnoresDebt:
    """Test that BasicTrainerBR works correctly with data containing debt."""

    def test_br_train_step_with_debt_data(self, networks, params, shock_params, flat_data_with_debt):
        """BR trainer runs successfully with data containing 'b' key."""
        policy_net, value_net = networks

        trainer = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            optimizer_actor=tf.keras.optimizers.Adam(learning_rate=1e-3),
            optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
            n_critic_steps=3,
            logit_clip=20.0,
            polyak_tau=0.9
        )

        # Extract batch (ignoring 'b')
        k = flat_data_with_debt['k'][:32]
        z = flat_data_with_debt['z'][:32]
        z_next_main = flat_data_with_debt['z_next_main'][:32]
        z_next_fork = flat_data_with_debt['z_next_fork'][:32]

        # This should work - trainer uses only k, z
        metrics = trainer.train_step(k, z, z_next_main, z_next_fork, temperature=0.1)

        assert "loss_critic" in metrics
        assert "loss_actor" in metrics
        assert np.isfinite(metrics["loss_critic"])
        assert np.isfinite(metrics["loss_actor"])

    def test_br_multiple_steps_with_debt_data(self, networks, params, shock_params, flat_data_with_debt):
        """BR trainer runs multiple steps successfully with debt data."""
        policy_net, value_net = networks

        trainer = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            optimizer_actor=tf.keras.optimizers.Adam(learning_rate=1e-3),
            optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
            n_critic_steps=3,
            logit_clip=20.0,
            polyak_tau=0.9
        )

        losses = []
        for i in range(5):
            batch_start = i * 32
            batch_end = (i + 1) * 32

            metrics = trainer.train_step(
                flat_data_with_debt['k'][batch_start:batch_end],
                flat_data_with_debt['z'][batch_start:batch_end],
                flat_data_with_debt['z_next_main'][batch_start:batch_end],
                flat_data_with_debt['z_next_fork'][batch_start:batch_end],
                temperature=0.1
            )
            losses.append(metrics["loss_critic"])

        # All losses should be finite
        assert all(np.isfinite(loss) for loss in losses)


# =============================================================================
# TEST: Basic LR Trainer (Trajectory Data)
# =============================================================================

class TestBasicLRWithDebtContext:
    """
    Test that BasicTrainerLR works in the shared data context.

    Note: LR uses trajectory data, not flattened data with 'b'.
    However, we test that the data generator can produce both formats
    consistently.
    """

    def test_lr_train_step_runs(self, networks, params, shock_params, traj_data):
        """LR trainer runs successfully with trajectory data."""
        policy_net, _ = networks

        trainer = BasicTrainerLR(
            policy_net=policy_net,
            params=params,
            shock_params=shock_params,
            T=traj_data['z_path'].shape[1] - 1,  # T from data
            logit_clip=20.0,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
        )

        # Extract batch
        k0 = traj_data['k0'][:32]
        z_path = traj_data['z_path'][:32]

        metrics = trainer.train_step(k0, z_path, temperature=0.1)

        assert "loss_LR" in metrics
        assert np.isfinite(metrics["loss_LR"])


# =============================================================================
# TEST: TensorFlow Dataset Pipeline with Debt Data
# =============================================================================

class TestTFDatasetPipelineWithDebt:
    """Test that TF Dataset pipeline works with data containing 'b'."""

    def test_tf_dataset_er_with_debt_data(self, networks, params, shock_params, flat_data_with_debt):
        """ER trainer works with TF Dataset created from data with 'b'."""
        policy_net, _ = networks

        trainer = BasicTrainerER(
            policy_net=policy_net,
            params=params,
            shock_params=shock_params,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            polyak_tau=0.995
        )

        # Create TF dataset from data with 'b' key
        # The training code should only use k, z, z_next_main, z_next_fork
        dataset = tf.data.Dataset.from_tensor_slices(flat_data_with_debt)
        dataset = dataset.batch(32).take(3)

        for batch in dataset:
            # Trainer ignores 'b' even though it's in the batch
            metrics = trainer.train_step(
                batch['k'],
                batch['z'],
                batch['z_next_main'],
                batch['z_next_fork']
            )

            assert "loss_ER" in metrics
            assert np.isfinite(metrics["loss_ER"])

    def test_tf_dataset_br_with_debt_data(self, networks, params, shock_params, flat_data_with_debt):
        """BR trainer works with TF Dataset created from data with 'b'."""
        policy_net, value_net = networks

        trainer = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            optimizer_actor=tf.keras.optimizers.Adam(learning_rate=1e-3),
            optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
            n_critic_steps=2,
            logit_clip=20.0,
            polyak_tau=0.9
        )

        # Create TF dataset from data with 'b' key
        dataset = tf.data.Dataset.from_tensor_slices(flat_data_with_debt)
        dataset = dataset.batch(32).take(3)

        for batch in dataset:
            # Trainer ignores 'b' even though it's in the batch
            metrics = trainer.train_step(
                batch['k'],
                batch['z'],
                batch['z_next_main'],
                batch['z_next_fork'],
                temperature=0.1
            )

            assert "loss_critic" in metrics
            assert "loss_actor" in metrics
