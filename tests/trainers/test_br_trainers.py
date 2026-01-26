
import tensorflow as tf
import pytest
import shutil
import os
from pathlib import Path

from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers.basic import BasicTrainerBR
from src.trainers.risky import RiskyDebtTrainerBR
from src.networks.network_basic import build_basic_networks
from src.networks.network_risky import build_risky_networks
from src.utils.annealing import AnnealingSchedule

@pytest.fixture
def temp_cache_dir():
    path = "tests/temp_data_cache_br"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.fixture
def params():
    return EconomicParams()

@pytest.fixture
def shock_params():
    return ShockParams()

@pytest.fixture
def data_gen(params, shock_params, temp_cache_dir):
    return DataGenerator(
        master_seed=(42, 42),
        shock_params=shock_params,
        k_bounds=(0.1, 10.0),
        logz_bounds=(-0.5, 0.5), # approx
        b_bounds=(0.0, 1.0),
        sim_batch_size=32,
        T=16,
        n_sim_batches=1,
        cache_dir=temp_cache_dir,
        save_to_disk=False
    )

def test_basic_trainer_br_step(data_gen, params, shock_params):
    # 1. Get Batch
    batch = next(data_gen.get_training_batches())
    k0 = batch['k0']
    z_path = batch['z_path']
    z_fork = batch['z_fork']
    
    # 2. Build Networks
    policy_net, value_net = build_basic_networks(
        k_min=0.1, k_max=10.0,
        n_layers=2, n_neurons=16,
        activation='relu'
    )
    
    # 3. Init Trainer
    trainer = BasicTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        params=params,
        shock_params=shock_params,
        n_critic_steps=2
    )
    
    # 4. Train Step
    metrics = trainer.train_step(k0, z_path, z_fork, temperature=0.1)
    
    print(f"Basic BR Metrics: {metrics}")
    
    # Assertions
    assert "loss_critic" in metrics
    assert "loss_actor" in metrics
    assert metrics["loss_critic"] >= 0 # squared error
    # Actor loss can be negative
    
    # Check shape/type
    assert isinstance(metrics["loss_critic"], float)
    assert isinstance(metrics["loss_actor"], float)

def test_risky_trainer_br_step(data_gen, params, shock_params):
    # 1. Get Batch
    batch = next(data_gen.get_training_batches())
    k0 = batch['k0']
    b0 = batch['b0']
    z_path = batch['z_path']
    z_fork = batch['z_fork']
    
    # 2. Build Networks
    policy_net, value_net, price_net = build_risky_networks(
        k_min=0.1, k_max=10.0,
        b_min=0.0, b_max=1.0,
        r_risk_free=params.r_rate,
        n_layers=2, n_neurons=16,
        activation='relu'
    )
    
    # 3. Init Trainer
    trainer = RiskyDebtTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        batch_size=32,
        n_critic_steps=2
    )
    
    # 4. Train Step
    # Correct signature: train_step(self, k, b, z_path, z_fork, temperature=0.1)
    metrics = trainer.train_step(k0, b0, z_path, z_fork, temperature=0.1)
    
    print(f"Risky BR Metrics: {metrics}")
    
    assert "loss_critic" in metrics
    assert "loss_actor" in metrics
    assert "loss_price_critic" in metrics
