"""
tests/dnn/test_evaluation_common.py

Unit tests for src/dnn/evaluation/common.py
Tests grid generation, moment computation, and steady-state finding.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from src.dnn.evaluation.common import (
    get_eval_grids,
    compute_moments,
    compare_moments,
    find_steady_state_k,
)
from src.dnn import EconomicScenario, BasicPolicyNetwork, SamplingBounds


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_scenario():
    """Create a simple scenario for testing."""
    bounds = SamplingBounds(
        k_bounds=(1.0, 10.0),
        log_z_bounds=(-0.5, 0.5),
        b_bounds=(0.0, 2.0)  # borrowing only: b >= 0
    )
    return EconomicScenario.from_overrides(
        name="test_scenario",
        delta=0.1,
        r_rate=0.04,
        sampling=bounds
    )


@pytest.fixture
def mock_policy_net():
    """Create a simple mock policy network for testing."""
    net = BasicPolicyNetwork(k_min=0.1, n_layers=2, n_neurons=16)
    # Initialize with dummy forward pass
    k_dummy = tf.constant([[5.0]], dtype=tf.float32)
    z_dummy = tf.constant([[1.0]], dtype=tf.float32)
    _ = net(k_dummy, z_dummy)
    return net


# =============================================================================
# TEST GET_EVAL_GRIDS
# =============================================================================

class TestGetEvalGrids:
    """Test grid generation from scenarios."""

    def test_grid_bounds(self, simple_scenario):
        """Verify grids respect min/max bounds."""
        k_grid, z_grid, b_grid = get_eval_grids(
            simple_scenario, n_k=50, n_z=10, n_b=20
        )

        # Check k bounds
        assert k_grid.min() >= simple_scenario.sampling.k_bounds[0]
        assert k_grid.max() <= simple_scenario.sampling.k_bounds[1]

        # Check z bounds (in levels)
        z_min_expected = np.exp(simple_scenario.sampling.log_z_bounds[0])
        z_max_expected = np.exp(simple_scenario.sampling.log_z_bounds[1])
        assert k_grid.min() >= 0.99 * z_min_expected  # Slight tolerance
        assert z_grid.max() <= 1.01 * z_max_expected

        # Check b bounds
        assert b_grid.min() >= simple_scenario.sampling.b_bounds[0]
        assert b_grid.max() <= simple_scenario.sampling.b_bounds[1]

    def test_grid_structure_multiplicative(self, simple_scenario):
        """Verify k-grid follows (1-δ) multiplicative structure."""
        k_grid, _, _ = get_eval_grids(simple_scenario)

        delta = simple_scenario.params.delta
        ratio = 1.0 / (1.0 - delta)  # Expected growth factor

        # Compute actual ratios between consecutive points
        actual_ratios = k_grid[1:] / k_grid[:-1]

        # All ratios should be approximately equal to 1/(1-δ)
        # (except possibly the last point which might be clipped)
        tolerance = 0.05  # 5% tolerance
        for i, r in enumerate(actual_ratios[:-1]):  # Skip last ratio
            assert abs(r - ratio) / ratio < tolerance, (
                f"Grid point {i}: ratio {r:.4f} differs from expected {ratio:.4f}"
            )

    def test_grid_strictly_increasing(self, simple_scenario):
        """Ensure all grids are strictly increasing."""
        k_grid, z_grid, b_grid = get_eval_grids(simple_scenario)

        assert np.all(np.diff(k_grid) > 0), "k_grid not strictly increasing"
        assert np.all(np.diff(z_grid) > 0), "z_grid not strictly increasing"
        assert np.all(np.diff(b_grid) > 0), "b_grid not strictly increasing"

    def test_grid_from_history_dict(self, simple_scenario):
        """Test grid generation from training history dict."""
        history = {"_scenario": simple_scenario}

        k_grid, z_grid, b_grid = get_eval_grids(history, n_k=30, n_z=5, n_b=10)

        assert len(k_grid) >= 2
        assert len(z_grid) == 5
        assert len(b_grid) == 10

    def test_grid_auto_computed_n_k(self, simple_scenario):
        """Verify n_k is auto-computed based on depreciation structure."""
        k_grid_1, _, _ = get_eval_grids(simple_scenario, n_k=50)
        k_grid_2, _, _ = get_eval_grids(simple_scenario, n_k=100)

        # Both should have the same length (auto-computed)
        # because it's determined by k_bounds and delta
        assert len(k_grid_1) == len(k_grid_2), (
            "n_k should be auto-computed, not user-specified"
        )


# =============================================================================
# TEST COMPUTE_MOMENTS
# =============================================================================

class TestComputeMoments:
    """Test moment computation from evaluation data."""

    def test_moments_basic(self):
        """Test basic moment computation."""
        # Create synthetic data
        data = {
            "k": np.random.randn(10, 10),
            "z": np.random.randn(10, 10),
            "value": np.random.randn(10, 10) * 5 + 10
        }

        moments = compute_moments(data)

        # Check output structure
        assert isinstance(moments, pd.DataFrame)
        assert len(moments) == 3  # 3 variables
        assert set(moments.columns) == {
            "variable", "mean", "median", "std", "Q10", "Q90", "min", "max"
        }

    def test_moments_specific_keys(self):
        """Test moment computation for specific keys only."""
        data = {
            "k": np.ones((5, 5)),
            "z": np.ones((5, 5)) * 2,
            "ignored": np.ones((5, 5)) * 99
        }

        moments = compute_moments(data, keys=["k", "z"])

        assert len(moments) == 2
        assert set(moments["variable"]) == {"k", "z"}

    def test_moments_correctness(self):
        """Verify moment calculations are correct."""
        # Use known distribution
        data = {"x": np.arange(100).reshape(10, 10).astype(float)}

        moments = compute_moments(data)

        assert moments.loc[0, "variable"] == "x"
        assert abs(moments.loc[0, "mean"] - 49.5) < 0.1
        assert abs(moments.loc[0, "median"] - 49.5) < 0.1
        assert moments.loc[0, "min"] == 0.0
        assert moments.loc[0, "max"] == 99.0


# =============================================================================
# TEST FIND_STEADY_STATE_K
# =============================================================================

class TestFindSteadyStateK:
    """Test steady-state capital finding."""

    def test_find_steady_state_identity_policy(self, simple_scenario):
        """Test with identity policy k' = k."""
        # Create a policy that returns k (identity)
        class IdentityPolicy:
            def __call__(self, k, z):
                return k

        policy = IdentityPolicy()
        k_ss = find_steady_state_k(policy, simple_scenario, z_ss=1.0, n_grid=100)

        # Any k is a steady state for identity policy
        # Should return some value in bounds
        k_min, k_max = simple_scenario.sampling.k_bounds
        assert k_min <= k_ss <= k_max

    def test_find_steady_state_constant_policy(self, simple_scenario):
        """Test with constant policy k' = const."""
        target_k = 5.0

        class ConstantPolicy:
            def __call__(self, k, z):
                batch_size = tf.shape(k)[0]
                return tf.ones((batch_size, 1), dtype=tf.float32) * target_k

        policy = ConstantPolicy()
        k_ss = find_steady_state_k(policy, simple_scenario, z_ss=1.0, n_grid=200)

        # Steady state should be close to target_k
        assert abs(k_ss - target_k) < 0.5, (
            f"Expected k_ss ≈ {target_k}, got {k_ss}"
        )

    def test_find_steady_state_returns_float(self, mock_policy_net, simple_scenario):
        """Ensure return type is float, not tensor."""
        k_ss = find_steady_state_k(
            mock_policy_net, simple_scenario, z_ss=1.0, n_grid=100
        )

        assert isinstance(k_ss, float)
        assert not np.isnan(k_ss)
        assert not np.isinf(k_ss)

    def test_find_steady_state_within_bounds(self, mock_policy_net, simple_scenario):
        """Steady state should be within sampling bounds."""
        k_ss = find_steady_state_k(
            mock_policy_net, simple_scenario, z_ss=1.0, n_grid=150
        )

        k_min, k_max = simple_scenario.sampling.k_bounds
        assert k_min <= k_ss <= k_max, (
            f"k_ss = {k_ss} outside bounds [{k_min}, {k_max}]"
        )


# =============================================================================
# TEST COMPARE_MOMENTS
# =============================================================================

class TestCompareMoments:
    """Test moment comparison across models."""

    def test_compare_moments_basic(self, mock_policy_net, simple_scenario):
        """Test comparing moments from multiple models."""
        # Create mock histories
        history1 = {
            "_policy_net": mock_policy_net,
            "_scenario": simple_scenario
        }
        history2 = {
            "_policy_net": mock_policy_net,
            "_scenario": simple_scenario
        }

        histories = [history1, history2]
        labels = ["Model A", "Model B"]

        # Import evaluation function
        from src.dnn.evaluation.wrappers import evaluate_basic_policy

        k_grid, z_grid, _ = get_eval_grids(simple_scenario, n_k=10, n_z=5, n_b=5)

        moments_df = compare_moments(
            histories,
            labels,
            eval_fn=evaluate_basic_policy,
            grid_kwargs={"k_grid": k_grid, "z_grid": z_grid}
        )

        # Check structure
        assert isinstance(moments_df, pd.DataFrame)
        assert "scenario" in moments_df.columns
        assert set(moments_df["scenario"].unique()) == {"Model A", "Model B"}


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
