"""
Tests for DDP offline (dataset-driven) constructor path.

Verifies that BasicModelDDP and RiskyModelDDP can be constructed
from synthetic dataset + metadata instead of shock_params.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.economy.parameters import EconomicParams
from src.ddp import DDPGridConfig, BasicModelDDP, RiskyModelDDP


def _make_synthetic_dataset_and_metadata(n=500, seed=42):
    """Build a minimal synthetic dataset mimicking DataGenerator output."""
    rng = np.random.default_rng(seed)

    # Stationary AR(1) log-productivity
    mu, rho, sigma = 0.0, 0.8, 0.1
    log_z = rng.normal(mu, sigma / np.sqrt(1 - rho**2), size=n)
    log_z_next = rho * log_z + rng.normal(0, sigma, size=n)
    z = np.exp(log_z).astype(np.float32)
    z_next = np.exp(log_z_next).astype(np.float32)

    k = rng.uniform(0.5, 5.0, size=n).astype(np.float32)

    dataset = {
        "z": tf.constant(z),
        "z_next_main": tf.constant(z_next),
        "k": tf.constant(k),
    }
    metadata = {
        "bounds": {
            "k": (float(k.min()), float(k.max())),
            "log_z": (float(log_z.min()), float(log_z.max())),
            "b": (0.0, 50.0),
        }
    }
    return dataset, metadata


class TestBasicModelDDPOffline:
    """BasicModelDDP constructed from dataset instead of shock_params."""

    def test_construction_succeeds(self):
        """Offline constructor produces valid grids and reward matrix."""
        params = EconomicParams()
        dataset, metadata = _make_synthetic_dataset_and_metadata()
        model = BasicModelDDP(
            params,
            dataset=dataset,
            dataset_metadata=metadata,
            grid_config=DDPGridConfig(z_size=5),
        )
        assert model.z_grid.shape == (5,)
        assert model.prob_matrix.shape == (5, 5)
        assert model.nk > 0
        assert model.reward_matrix.shape[0] == 5

    def test_vfi_runs_to_completion(self):
        """VFI solver converges on offline-constructed model."""
        params = EconomicParams()
        dataset, metadata = _make_synthetic_dataset_and_metadata()
        model = BasicModelDDP(
            params,
            dataset=dataset,
            dataset_metadata=metadata,
            grid_config=DDPGridConfig(z_size=5),
        )
        v_star, policy = model.solve_basic_vfi(max_iter=50, tol=1e-3)
        assert v_star.shape == (5, model.nk)
        assert policy.shape == (5, model.nk)


class TestRiskyModelDDPOffline:
    """RiskyModelDDP constructed from dataset instead of shock_params."""

    def test_construction_succeeds(self):
        """Offline constructor produces valid grids and matrices."""
        params = EconomicParams()
        dataset, metadata = _make_synthetic_dataset_and_metadata()
        model = RiskyModelDDP(
            params,
            dataset=dataset,
            dataset_metadata=metadata,
            grid_config=DDPGridConfig(z_size=5, b_size=4),
        )
        assert model.z_grid.shape == (5,)
        assert model.prob_matrix.shape == (5, 5)
        assert model.nk > 0
        assert model.nb == 4
