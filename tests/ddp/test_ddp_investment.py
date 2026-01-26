import pytest
import tensorflow as tf
import numpy as np

# Adjust imports to match your folder structure
from src.economy.parameters import EconomicParams, ShockParams
from src.ddp import DDPGridConfig
from src.ddp.ddp_investment import InvestmentModelDDP


@pytest.fixture
def model_ddp():
    """
    Standard fixture for creating a DDP model instance.
    Uses a small grid to keep tests fast.
    """
    params = EconomicParams()
    shock_params = ShockParams()
    grid_config = DDPGridConfig(z_size=5, k_size=10)
    return InvestmentModelDDP(params, shock_params, grid_config)


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
    grid_config = DDPGridConfig(k_size=5, z_size=2, grid_type="log_linear")
    shock_params = ShockParams()
    model = InvestmentModelDDP(params, shock_params, grid_config)

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
    v_star, policy_k = model_ddp.solve_invest_vfi(tol=1e-4, max_iter=500)

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
    v_vfi, policy_vfi = model_ddp.solve_invest_vfi(tol=1e-5, max_iter=2000)

    # 2. Solve using PFI (Fast)
    v_pfi, policy_pfi = model_ddp.solve_invest_pfi(max_iter=50, eval_steps=500)

    # 3. Compare Value Functions
    diff_v = np.max(np.abs(v_vfi - v_pfi))
    assert diff_v < 1e-4, f"Solvers disagreed on Value Function! Max diff: {diff_v}"

    # 4. Compare Policy Functions
    # Both output actual capital values (floats), so we compare those directly
    diff_policy = np.max(np.abs(policy_vfi - policy_pfi))
    assert diff_policy < 1e-5, f"Solvers disagreed on Policy! Max diff: {diff_policy}"