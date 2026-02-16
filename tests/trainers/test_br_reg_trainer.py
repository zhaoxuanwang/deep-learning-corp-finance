"""
Unit tests for BasicTrainerBRRegression.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.networks.network_basic import build_basic_networks
from src.trainers.basic import BasicTrainerBRRegression


@pytest.fixture
def temp_cache_dir(tmp_path):
    path = tmp_path / "temp_data_cache_br_reg"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@pytest.fixture
def params():
    return EconomicParams()


@pytest.fixture
def shock_params():
    return ShockParams()


@pytest.fixture
def data_gen(shock_params, temp_cache_dir):
    return DataGenerator(
        master_seed=(19, 91),
        shock_params=shock_params,
        k_bounds=(0.1, 10.0),
        logz_bounds=(-0.5, 0.5),
        b_bounds=(0.0, 1.0),
        sim_batch_size=32,
        T=16,
        n_sim_batches=2,
        cache_dir=temp_cache_dir,
        save_to_disk=False,
    )


@pytest.fixture
def flat_data(data_gen):
    return data_gen.get_flattened_training_dataset()


@pytest.fixture
def networks():
    policy_net, value_net = build_basic_networks(
        k_min=0.1,
        k_max=10.0,
        logz_min=-0.5,
        logz_max=0.5,
        n_layers=2,
        n_neurons=16,
        activation="swish",
    )
    return policy_net, value_net


@pytest.fixture
def trainer(networks, params, shock_params):
    policy_net, value_net = networks
    return BasicTrainerBRRegression(
        policy_net=policy_net,
        value_net=value_net,
        params=params,
        shock_params=shock_params,
        optimizer_policy=tf.keras.optimizers.Adam(learning_rate=1e-3),
        optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
        logit_clip=20.0,
        loss_type="mse",
        weight_foc=1.0,
        weight_env=1.0,
        use_foc=True,
        use_env=True,
    )


class TestBasicBRRegressionTrainer:
    def test_initialization(self, trainer):
        assert trainer.policy_net is not None
        assert trainer.value_net is not None
        assert trainer.use_foc is True
        assert trainer.use_env is True
        assert trainer.loss_type == "mse"

    def test_train_step_with_flattened_data(self, trainer, flat_data):
        batch_size = 32
        metrics = trainer.train_step(
            flat_data["k"][:batch_size],
            flat_data["z"][:batch_size],
            flat_data["z_next_main"][:batch_size],
            flat_data["z_next_fork"][:batch_size],
            temperature=0.1,
        )

        assert "loss_BR_reg" in metrics
        assert "loss_BR" in metrics
        assert "loss_FOC" in metrics
        assert "loss_Env" in metrics

    def test_losses_are_finite(self, trainer, flat_data):
        batch_size = 32
        metrics = trainer.train_step(
            flat_data["k"][:batch_size],
            flat_data["z"][:batch_size],
            flat_data["z_next_main"][:batch_size],
            flat_data["z_next_fork"][:batch_size],
            temperature=0.1,
        )
        for key in ("loss_BR_reg", "loss_BR", "loss_FOC", "loss_Env"):
            assert np.isfinite(metrics[key]), f"{key} should be finite"

    def test_policy_and_value_weights_update(self, trainer, flat_data):
        batch_size = 32
        init_policy = [w.copy() for w in trainer.policy_net.get_weights()]
        init_value = [w.copy() for w in trainer.value_net.get_weights()]

        for _ in range(3):
            trainer.train_step(
                flat_data["k"][:batch_size],
                flat_data["z"][:batch_size],
                flat_data["z_next_main"][:batch_size],
                flat_data["z_next_fork"][:batch_size],
                temperature=0.1,
            )

        updated_policy = trainer.policy_net.get_weights()
        updated_value = trainer.value_net.get_weights()

        policy_changed = any(
            not np.allclose(w0, w1, rtol=1e-4, atol=1e-8)
            for w0, w1 in zip(init_policy, updated_policy)
        )
        value_changed = any(
            not np.allclose(w0, w1, rtol=1e-4, atol=1e-8)
            for w0, w1 in zip(init_value, updated_value)
        )

        assert policy_changed, "Policy weights should update during BR regression training"
        assert value_changed, "Value weights should update during BR regression training"

    def test_evaluate_returns_metrics(self, trainer, flat_data):
        batch_size = 16
        metrics = trainer.evaluate(
            flat_data["k"][:batch_size],
            flat_data["z"][:batch_size],
            flat_data["z_next_main"][:batch_size],
            flat_data["z_next_fork"][:batch_size],
            temperature=0.1,
        )

        assert "loss_BR_reg" in metrics
        assert "loss_BR" in metrics
        assert "loss_FOC" in metrics
        assert "loss_Env" in metrics
