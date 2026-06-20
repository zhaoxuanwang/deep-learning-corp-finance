"""Shared pytest fixtures for the v3 suite.

SMOKE-profile fixtures keep grids/epochs tiny so the whole suite runs in CI-time
while still exercising every code path (the oracle/property gates stay
meaningful). FULL/BASE values come from the spec and are used in notebooks.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.v3 import config as cfg
from src.v3.common import precision

# Run the test suite CPU-only: float64 numerics are unreliable on Apple Metal
# (review HW-1), and these tests gate the float64 oracle/property checks. Must
# happen before any TF op initializes the GPU (conftest is imported first).
precision.use_cpu_only()
precision.silence_logging()


@pytest.fixture(scope="session")
def ext_params() -> cfg.ExternalParams:
    return cfg.ExternalParams()


@pytest.fixture(scope="session")
def ref_params() -> cfg.ModelParams:
    return cfg.REFERENCE_ESTIMATES


@pytest.fixture(scope="session")
def bounds() -> cfg.ParamBounds:
    return cfg.ParamBounds.table_a1()


@pytest.fixture(scope="session")
def smoke_grid() -> cfg.GridConfig:
    """Tiny grid for fast tests (every dimension still > 1)."""
    return cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)


@pytest.fixture(scope="session")
def smoke_train() -> cfg.TrainConfig:
    return cfg.TrainConfig(batch_size=256, steps_per_epoch=20, gh_nodes=3)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260619)
