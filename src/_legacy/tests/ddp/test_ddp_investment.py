import pytest
import tensorflow as tf
import numpy as np

# Adjust imports to match your folder structure
from src.economy.parameters import EconomicParams
from src.ddp import DDPGridConfig, BasicModelDDP


def _make_synthetic_dataset_and_metadata(n=500, seed=42):
    """Build a minimal synthetic dataset mimicking flattened training data."""
    rng = np.random.default_rng(seed)

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
            "b": (0.0, 10.0),
        }
    }
    return dataset, metadata


@pytest.fixture
def model_ddp():
    """
    Standard fixture for creating a basic-model DDP instance.
    Uses a small grid to keep tests fast.
    """
    params = EconomicParams()
    dataset, metadata = _make_synthetic_dataset_and_metadata()
    grid_config = DDPGridConfig(z_size=5, k_size=10)
    return BasicModelDDP(
        params,
        grid_config=grid_config,
        dataset=dataset,
        dataset_metadata=metadata,
    )


def test_initialization_shapes(model_ddp):
    """
    Verify grids and matrices are created with correct TF shapes.
    """
    # 1. Get the "Truth" from the model instance itself
    expected_nz = model_ddp.nz  # or model_ddp.params.z_size
    expected_nk = model_ddp.nk  # or model_ddp.params.k_size

    # 2. Check Grids (Dynamic Assertion)
    assert model_ddp.z_grid.shape == (expected_nz,)
    assert model_ddp.k_grid.shape == (expected_nk,)
    assert model_ddp.prob_matrix.shape == (expected_nz, expected_nz)

    # 3. Check Reward Matrix
    # Note: If your reward matrix includes 'next k', it is (nz, nk, nk)
    assert model_ddp.reward_matrix.shape == (expected_nz, expected_nk, expected_nk)


def test_reward_values_no_costs():
    """
    Economic Sanity Check:
    If investment is zero and costs are zero, Reward should exactly equal Profit.
    """
    # Create a minimal model with no costs and full depreciation
    # If delta=1.0, then Investment I = k_next - (1-1)*k = k_next
    params = EconomicParams(
        cost_fixed=0.0,
        cost_convex=0.0,
        delta=1.0,
    )
    grid_config = DDPGridConfig(k_size=5, z_size=2, capital_grid_type="linear")
    dataset, metadata = _make_synthetic_dataset_and_metadata()
    model = BasicModelDDP(
        params,
        grid_config=grid_config,
        dataset=dataset,
        dataset_metadata=metadata,
    )

    # Extract values
    z = model.z_grid[0]
    k_current = model.k_grid[0]

    # Assume we choose to transition to index 0 (k_next = k_grid[0])
    k_next = model.k_grid[0]

    # Look at reward at index [0, 0, 0] -> state (z_0, k_0), action (k_0)
    reward_tf = model.reward_matrix[0, 0, 0]

    # Manual calculation based on economic logic
    investment = k_next
    profit = z * (k_current ** params.theta)
    expected_reward = profit - investment

    np.testing.assert_allclose(
        reward_tf.numpy(),
        expected_reward.numpy(),
        rtol=1e-5,
        err_msg="Reward calculation failed for simple no-cost case"
    )


def test_bellman_step_structure(model_ddp):
    """
    Verify Bellman operator accepts a guess and returns updated value/policy of correct shape.
    """
    nz, nk = model_ddp.nz, model_ddp.nk
    v_guess = tf.zeros((nz, nk), dtype=tf.float32)

    # Run one step
    v_new, policy_idx = model_ddp.bellman_step(v_guess)

    # Check output shapes
    assert v_new.shape == (nz, nk)
    assert policy_idx.shape == (nz, nk)

    # Check types
    assert v_new.dtype == tf.float32
    assert policy_idx.dtype == tf.int64


def test_solver_integration_vfi(model_ddp):
    """
    Runs the VFI solver and checks basic economic properties of the result.
    """
    # Solve with loose tolerance for speed in unit tests
    v_star, policy_k = model_ddp.solve_basic_vfi(tol=1e-4, max_iter=500)

    # 1. Check Shapes
    assert v_star.shape == (model_ddp.nz, model_ddp.nk)
    assert policy_k.shape == (model_ddp.nz, model_ddp.nk)

    # 2. Value Function Monotonicity (in Capital)
    # V(z, k) should generally increase with k (more assets = better)
    # We check: V(:, i+1) >= V(:, i)
    v_diff = v_star[:, 1:] - v_star[:, :-1]

    # Allow for tiny floating point errors (>-1e-5 instead of strictly >0)
    assert np.all(v_diff.numpy() > -1e-5), "Value function is not monotonically increasing in capital!"

    # 3. Policy Function Range check
    # Optimal capital choice must be within the grid bounds
    assert tf.reduce_min(policy_k) >= model_ddp.k_grid[0]
    assert tf.reduce_max(policy_k) <= model_ddp.k_grid[-1]


def test_solvers_consistency(model_ddp):
    """
    Ensures that Value Function Iteration (VFI) and Policy Function Iteration (PFI)
    converge to the same solution (within tolerance).
    """
    # 1. Solve using VFI (Robust)
    v_vfi, policy_vfi = model_ddp.solve_basic_vfi(tol=1e-5, max_iter=2000)

    # 2. Solve using PFI (Fast)
    v_pfi, policy_pfi = model_ddp.solve_basic_pfi(max_iter=50, eval_steps=500)

    # 3. Compare Value Functions
    diff_v = np.max(np.abs(v_vfi - v_pfi))
    assert diff_v < 2e-4, f"Solvers disagreed on Value Function! Max diff: {diff_v}"

    # 4. Compare Policy Functions
    # Both output actual capital values (floats), so we compare those directly
    diff_policy = np.max(np.abs(policy_vfi - policy_pfi))
    assert diff_policy < 1e-5, f"Solvers disagreed on Policy! Max diff: {diff_policy}"
