
import tensorflow as tf
import pytest
import shutil
import os
from pathlib import Path

from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers.basic import BasicTrainerLR
from src.trainers.risky import RiskyDebtTrainerLR
from src.networks.network_basic import build_basic_networks
from src.networks.network_risky import build_risky_networks

@pytest.fixture
def temp_cache_dir_lr():
    path = "tests/temp_data_cache_lr"
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
def data_gen(params, shock_params, temp_cache_dir_lr):
    # LR needs valid z_path
    return DataGenerator(
        master_seed=(99, 99),
        shock_params=shock_params,
        k_bounds=(0.1, 10.0),
        logz_bounds=(-0.5, 0.5), 
        b_bounds=(0.0, 1.0),
        sim_batch_size=32,
        T=16,
        n_sim_batches=1,
        cache_dir=temp_cache_dir_lr,
        save_to_disk=False
    )

def test_basic_trainer_lr_step(data_gen, params, shock_params):
    # 1. Get Batch
    batch = next(data_gen.get_training_batches())
    k0 = batch['k0']
    z_path = batch['z_path']
    # z_fork not strictly needed for Basic LR but usually yielded
    
    T = z_path.shape[1] - 1
    
    # 2. Build Networks
    policy_net, _ = build_basic_networks(
        k_min=0.1, k_max=10.0,
        n_layers=2, n_neurons=16,
        activation='relu'
    )
    
    # 3. Init Trainer
    trainer = BasicTrainerLR(
        policy_net=policy_net,
        params=params,
        shock_params=shock_params,
        T=T,
        logit_clip=20.0
    )
    
    # 4. Train Step
    metrics = trainer.train_step(k0, z_path, temperature=0.1)
    
    print(f"Basic LR Metrics: {metrics}")
    
    assert "loss_LR" in metrics
    assert "mean_reward" in metrics
    assert isinstance(metrics["loss_LR"], float)

def test_risky_trainer_lr_step(data_gen, params, shock_params):
    # 1. Get Batch
    batch = next(data_gen.get_training_batches())
    k0 = batch['k0']
    b0 = batch['b0']
    z_path = batch['z_path']
    z_fork = batch['z_fork'] # Required for Risky LR (pricing)
    
    T = z_path.shape[1] - 1
    
    # 2. Build Networks
    policy_net, value_net, price_net = build_risky_networks(
        k_min=0.1, k_max=10.0,
        b_min=0.0, b_max=1.0,
        r_risk_free=params.r_rate,
        n_layers=2, n_neurons=16,
        activation='relu'
    )
    
    # 3. Init Trainer
    trainer = RiskyDebtTrainerLR(
        policy_net=policy_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        T=T,
        batch_size=32,
        logit_clip=20.0,
        value_net_for_default=None # Optional, but good to test default None
    )
    
    # 4. Train Step
    metrics = trainer.train_step(k0, b0, z_path, z_fork, temperature=0.1)
    
    print(f"Risky LR Metrics: {metrics}")
    
    assert "loss_lr" in metrics
    assert "loss_price" in metrics
    assert isinstance(metrics["loss_lr"], float)
