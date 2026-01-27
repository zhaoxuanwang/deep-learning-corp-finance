
import tensorflow as tf
import pytest
import shutil
import os
from pathlib import Path

from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers.basic import BasicTrainerER
from src.networks.network_basic import build_basic_networks

@pytest.fixture
def temp_cache_dir():
    path = "tests/temp_data_cache_er"
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

def test_basic_trainer_er_step(data_gen, params, shock_params):
    # 1. Get Batch
    batch = next(data_gen.get_training_batches())
    k0 = batch['k0']
    z_path = batch['z_path']
    z_fork = batch['z_fork']
    
    # 2. Build Networks
    policy_net, _ = build_basic_networks(
        k_min=0.1, k_max=10.0,
        n_layers=2, n_neurons=16,
        activation='relu'
    )
    
    # 3. Init Trainer
    trainer = BasicTrainerER(
        policy_net=policy_net,
        params=params,
        shock_params=shock_params
    )
    
    # 4. Train Step
    metrics = trainer.train_step(k0, z_path, z_fork)
    
    print(f"Basic ER Metrics: {metrics}")
    
    # Assertions
    assert "loss_ER" in metrics
    assert metrics["loss_ER"] >= 0.0
    assert isinstance(metrics["loss_ER"], float)

    # Sanity Check: Loss should change if we train (though single step might not show huge diff)
    # Just checking it runs and returns valid number
