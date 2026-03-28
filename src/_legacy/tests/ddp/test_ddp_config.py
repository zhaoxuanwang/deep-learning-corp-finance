import numpy as np
from src.ddp import ddp_config


def test_transition_matrix_integrity():
    """
    Estimated transition matrix should be a valid stochastic matrix.
    """
    rng = np.random.default_rng(123)
    z_size = 15
    log_z = rng.normal(loc=0.0, scale=0.25, size=2_000)
    log_z_next = 0.85 * log_z + rng.normal(loc=0.0, scale=0.10, size=2_000)

    z_curr = np.exp(log_z)
    z_next_main = np.exp(log_z_next)
    log_z_bounds = (float(log_z.min()), float(log_z.max()))

    z_grid, prob_matrix = ddp_config.estimate_transition_matrix_from_flat_data(
        z_curr=z_curr,
        z_next_main=z_next_main,
        log_z_bounds=log_z_bounds,
        z_size=z_size,
        alpha=1.0,
    )

    # Check 1: Row sums (normalization)
    row_sums = np.sum(prob_matrix, axis=1)
    assert np.allclose(row_sums, 1.0), "Transition probability matrix rows do not sum to 1.0."

    # Check 2: Domain (productivity levels must be positive)
    assert np.all(z_grid > 0), "Productivity grid contains non-positive values."

    # Check 3: Dimensions
    assert len(z_grid) == z_size
    assert prob_matrix.shape == (z_size, z_size)


def test_transition_matrix_nearest_node_mapping():
    """
    Binning should use nearest-node mapping (round), not left-bin floor.
    """
    z_size = 5
    log_z_bounds = (0.0, 4.0)

    # 0.49 is nearest to node 0; 0.51 is nearest to node 1.
    z_vals = np.exp(np.array([0.49, 0.51], dtype=np.float64))
    _, prob_matrix = ddp_config.estimate_transition_matrix_from_flat_data(
        z_curr=z_vals,
        z_next_main=z_vals,
        log_z_bounds=log_z_bounds,
        z_size=z_size,
        alpha=0.0,
    )

    assert np.isclose(prob_matrix[0, 0], 1.0)
    assert np.isclose(prob_matrix[1, 1], 1.0)
