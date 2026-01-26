
import pytest
import numpy as np
from src.economy.parameters import EconomicParams, ShockParams
from src.ddp import ddp_config

@pytest.fixture
def params():
    """Default test parameters."""
    return EconomicParams()

@pytest.fixture
def shock_params():
    return ShockParams(
        rho=0.9,
        sigma=0.01,
        mu=0.0
    )

def test_get_sampling_bounds_structure(params, shock_params):
    """Verify get_sampling_bounds returns correct structure."""
    bounds = ddp_config.get_sampling_bounds(params, shock_params)
    (k_range, b_range) = bounds
    
    # Check K bounds
    assert len(k_range) == 2
    assert k_range[1] > k_range[0]
    
    # Check B bounds
    assert len(b_range) == 2
    assert b_range[1] > b_range[0]


def test_markov_process_integrity(shock_params):
    """
    Test the integrity of the Tauchen discretization method.
    """
    z_size = 15
    z_grid, prob_matrix = ddp_config.initialize_markov_process(shock_params, z_size)

    # Check 1: Row Sums (Normalization)
    row_sums = np.sum(prob_matrix, axis=1)
    assert np.allclose(row_sums, 1.0), \
        "Transition probability matrix rows do not sum to 1.0."

    # Check 2: Domain (Productivity levels must be positive)
    assert np.all(z_grid > 0), "Productivity grid contains non-positive values."

    # Check 3: Dimensions
    assert len(z_grid) == z_size
    assert prob_matrix.shape == (z_size, z_size)

