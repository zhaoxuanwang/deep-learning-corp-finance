"""Tests for v2 discrete solvers: grid construction, VFI, PFI.

Covers:
- GridAxis validation and 1-D grid construction (linear, log)
- GridConfig validation
- Transition matrix estimation
- VFI convergence on BasicInvestmentEnv (frictionless)
- PFI convergence on BasicInvestmentEnv (frictionless)
- VFI vs PFI policy agreement
- Multi-dimensional mock environment
"""

import numpy as np
import tensorflow as tf
import pytest

from src.v2.solvers.grid import (
    GridAxis,
    build_1d_grid,
    build_product_grid,
    build_grids,
    estimate_exo_transition_matrix,
    snap_to_grid,
)
from src.v2.solvers.config import GridConfig, VFIConfig, PFIConfig
from src.v2.solvers.vfi import solve_vfi
from src.v2.solvers.pfi import solve_pfi
from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.environments.base import MDPEnvironment
from src.v2.data.generator import DataGenerator, DataGeneratorConfig
from src.economy.parameters import EconomicParams, ShockParams


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def env():
    """Frictionless BasicInvestmentEnv (analytical solution available)."""
    return BasicInvestmentEnv(
        econ_params=EconomicParams(
            r_rate=0.04, delta=0.15, theta=0.7,
            cost_convex=0.0, cost_fixed=0.0),
        shock_params=ShockParams(mu=0.0, rho=0.7, sigma=0.15),
    )


@pytest.fixture
def flat_dataset(env):
    """Small flattened dataset for solver tests."""
    gen = DataGenerator(env, DataGeneratorConfig(n_paths=100, horizon=16))
    return gen.get_flattened_dataset("train")


@pytest.fixture
def small_grid_config():
    """Small grid config for fast test runs."""
    return GridConfig(
        exo_sizes=[5],
        endo_sizes=[8],
        action_sizes=[10],
        transition_alpha=1.0,
    )


# =============================================================================
# GridAxis tests
# =============================================================================

class TestGridAxis:

    def test_linear_grid(self):
        axis = GridAxis(0.0, 10.0, spacing="linear")
        grid = build_1d_grid(axis, 5)
        np.testing.assert_allclose(grid, [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_log_grid(self):
        axis = GridAxis(1.0, 100.0, spacing="log")
        grid = build_1d_grid(axis, 3)
        assert len(grid) == 3
        np.testing.assert_allclose(grid[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(grid[-1], 100.0, atol=1e-10)
        # Log-spacing: middle point is geometric mean
        np.testing.assert_allclose(grid[1], 10.0, atol=1e-10)

    def test_log_grid_respects_n(self):
        """Log grid always produces exactly n points."""
        axis = GridAxis(1.0, 100.0, spacing="log")
        for n in [5, 50, 200]:
            grid = build_1d_grid(axis, n)
            assert len(grid) == n
            assert grid[0] == pytest.approx(1.0, rel=1e-6)
            assert grid[-1] == pytest.approx(100.0, rel=1e-6)
            # Log-spaced: ratios should be constant
            ratios = grid[1:] / grid[:-1]
            np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)

    def test_log_grid_values_in_levels(self):
        """Log-spaced grid values should be in levels (positive), not in log."""
        axis = GridAxis(0.5, 2.0, spacing="log")
        grid = build_1d_grid(axis, 7)
        assert all(g > 0 for g in grid)
        assert grid[0] == pytest.approx(0.5, rel=1e-6)
        assert grid[-1] == pytest.approx(2.0, rel=1e-6)

    def test_invalid_low_ge_high(self):
        with pytest.raises(ValueError, match="low < high"):
            GridAxis(5.0, 3.0)

    def test_log_requires_positive(self):
        with pytest.raises(ValueError, match="low > 0"):
            GridAxis(-1.0, 10.0, spacing="log")

    def test_unknown_spacing(self):
        with pytest.raises(ValueError, match="Unknown spacing"):
            GridAxis(1.0, 10.0, spacing="chebyshev")

    def test_multiplicative_rejected(self):
        """Multiplicative spacing was removed — should raise."""
        with pytest.raises(ValueError, match="Unknown spacing"):
            GridAxis(1.0, 10.0, spacing="multiplicative")

    def test_min_grid_size(self):
        axis = GridAxis(0.0, 1.0)
        with pytest.raises(ValueError, match="must be >= 2"):
            build_1d_grid(axis, 1)


# =============================================================================
# Product grid tests
# =============================================================================

class TestProductGrid:

    def test_2d_product(self):
        g1 = np.array([1.0, 2.0])
        g2 = np.array([10.0, 20.0, 30.0])
        prod = build_product_grid([g1, g2])
        assert prod.shape == (6, 2)
        # First row: (g1[0], g2[0])
        np.testing.assert_allclose(prod[0], [1.0, 10.0])
        # Last row: (g1[-1], g2[-1])
        np.testing.assert_allclose(prod[-1], [2.0, 30.0])

    def test_1d_product_is_column(self):
        g1 = np.array([1.0, 2.0, 3.0])
        prod = build_product_grid([g1])
        assert prod.shape == (3, 1)
        np.testing.assert_allclose(prod[:, 0], [1.0, 2.0, 3.0])


# =============================================================================
# GridConfig validation tests
# =============================================================================

class TestGridConfig:

    def test_defaults(self):
        gc = GridConfig()
        assert gc.exo_sizes == [7]
        assert gc.endo_sizes == [25]
        assert gc.action_sizes == [25]

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="must be >= 2"):
            GridConfig(exo_sizes=[1])

    def test_empty_sizes(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            GridConfig(exo_sizes=[])

    def test_negative_alpha(self):
        with pytest.raises(ValueError, match="transition_alpha"):
            GridConfig(transition_alpha=-1.0)


# =============================================================================
# Transition matrix estimation tests
# =============================================================================

class TestTransitionEstimation:

    def test_rows_sum_to_one(self):
        """Estimated transition matrix rows must sum to 1."""
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.uniform(0.5, 2.0, (n, 1))
        z_next = z * 0.7 + 0.3 + rng.normal(0, 0.1, (n, 1))
        z_next = np.clip(z_next, 0.5, 2.0)

        grids_1d = [np.linspace(0.5, 2.0, 5)]
        prob = estimate_exo_transition_matrix(z, z_next, grids_1d, alpha=1.0)

        assert prob.shape == (5, 5)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-10)

    def test_all_probs_positive(self):
        """With Dirichlet smoothing, all entries should be > 0."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.uniform(1.0, 3.0, (n, 1))
        z_next = z + rng.normal(0, 0.2, (n, 1))
        z_next = np.clip(z_next, 1.0, 3.0)

        grids_1d = [np.linspace(1.0, 3.0, 4)]
        prob = estimate_exo_transition_matrix(z, z_next, grids_1d, alpha=1.0)
        assert np.all(prob > 0)

    def test_multidim_exo(self):
        """Transition estimation works with 2-D exogenous state."""
        rng = np.random.default_rng(42)
        n = 2000
        z = rng.uniform(0.5, 2.0, (n, 2))
        z_next = z * 0.8 + 0.2 + rng.normal(0, 0.05, (n, 2))
        z_next = np.clip(z_next, 0.5, 2.0)

        grids_1d = [np.linspace(0.5, 2.0, 3), np.linspace(0.5, 2.0, 3)]
        prob = estimate_exo_transition_matrix(z, z_next, grids_1d, alpha=1.0)

        assert prob.shape == (9, 9)  # 3x3 product grid
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-10)


# =============================================================================
# snap_to_grid tests
# =============================================================================

class TestSnapToGrid:

    def test_exact_match(self):
        grid = tf.constant([[1.0], [2.0], [3.0]])
        values = tf.constant([[1.0], [3.0], [2.0]])
        idx = snap_to_grid(values, grid)
        np.testing.assert_array_equal(idx.numpy(), [0, 2, 1])

    def test_nearest_neighbor(self):
        grid = tf.constant([[0.0], [1.0], [2.0]])
        values = tf.constant([[0.3], [1.8]])
        idx = snap_to_grid(values, grid)
        np.testing.assert_array_equal(idx.numpy(), [0, 2])

    def test_2d(self):
        grid = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        values = tf.constant([[0.1, 0.9], [0.9, 0.1]])
        idx = snap_to_grid(values, grid)
        np.testing.assert_array_equal(idx.numpy(), [1, 2])


# =============================================================================
# build_grids integration tests
# =============================================================================

class TestBuildGrids:

    def test_basic_env_grid_spec(self, env):
        """BasicInvestmentEnv provides grid_spec with log k, log z, linear I."""
        spec = env.grid_spec()
        assert spec is not None
        assert len(spec["endo"]) == 1
        assert len(spec["exo"]) == 1
        assert len(spec["action"]) == 1
        assert spec["endo"][0].spacing == "log"
        assert spec["exo"][0].spacing == "log"
        assert spec["action"][0].spacing == "linear"

    def test_build_grids_from_env(self, env):
        grids = build_grids(env, exo_sizes=[5], endo_sizes=[10], action_sizes=[8])
        assert len(grids["exo_grids_1d"]) == 1
        assert len(grids["endo_grids_1d"]) == 1
        assert len(grids["action_grids_1d"]) == 1
        assert grids["exo_product"].shape == (5, 1)
        assert grids["endo_product"].shape == (10, 1)
        assert grids["action_product"].shape == (8, 1)

    def test_exo_grid_in_levels(self, env):
        """Exo grid values should be in levels (positive), not log-space."""
        grids = build_grids(env, exo_sizes=[7], endo_sizes=[5], action_sizes=[5])
        z_grid = grids["exo_grids_1d"][0]
        assert all(z > 0 for z in z_grid)
        assert z_grid[0] == pytest.approx(env.z_min, rel=1e-4)
        assert z_grid[-1] == pytest.approx(env.z_max, rel=1e-4)


# =============================================================================
# VFI solver tests
# =============================================================================

class TestVFI:

    def test_convergence(self, env, flat_dataset, small_grid_config):
        """VFI converges on frictionless BasicInvestmentEnv."""
        config = VFIConfig(grid=small_grid_config, tol=1e-5, max_iter=2000)
        result = solve_vfi(env, flat_dataset, config)
        assert result["converged"]
        assert result["n_iter"] < 2000

    def test_value_positive(self, env, flat_dataset, small_grid_config):
        """Value function should be positive for the basic investment model."""
        config = VFIConfig(grid=small_grid_config, tol=1e-5, max_iter=2000)
        result = solve_vfi(env, flat_dataset, config)
        assert float(tf.reduce_min(result["value"])) >= -1e-3

    def test_output_shapes(self, env, flat_dataset, small_grid_config):
        """Check output tensor shapes."""
        config = VFIConfig(grid=small_grid_config, tol=1e-4, max_iter=500)
        result = solve_vfi(env, flat_dataset, config)

        n_exo = result["grids"]["exo_product"].shape[0]
        n_endo = result["grids"]["endo_product"].shape[0]

        assert result["value"].shape == (n_exo, n_endo)
        assert result["policy_action"].shape == (n_exo, n_endo, env.action_dim())
        assert result["policy_endo"].shape == (n_exo, n_endo, env.endo_dim())

    def test_frictionless_policy_monotone(self, env, flat_dataset):
        """In the frictionless model, k'(z) should be increasing in z."""
        config = VFIConfig(
            grid=GridConfig(exo_sizes=[7], endo_sizes=[15], action_sizes=[20]),
            tol=1e-5, max_iter=2000)
        result = solve_vfi(env, flat_dataset, config)

        # At a fixed endo state (middle of grid), check k'(z) is increasing
        n_endo = result["grids"]["endo_product"].shape[0]
        mid_k = n_endo // 2
        k_next = result["policy_endo"][:, mid_k, 0].numpy()
        # Allow small non-monotonicity from discretization
        diffs = np.diff(k_next)
        assert np.sum(diffs < -1e-2) <= 1, (
            f"k'(z) should be mostly increasing. Diffs: {diffs}")


# =============================================================================
# PFI solver tests
# =============================================================================

class TestPFI:

    def test_convergence(self, env, flat_dataset, small_grid_config):
        """PFI converges on frictionless BasicInvestmentEnv."""
        config = PFIConfig(grid=small_grid_config, max_iter=100, eval_steps=200)
        result = solve_pfi(env, flat_dataset, config)
        assert result["converged"]

    def test_value_positive(self, env, flat_dataset, small_grid_config):
        """Value function should be positive."""
        config = PFIConfig(grid=small_grid_config, max_iter=100, eval_steps=200)
        result = solve_pfi(env, flat_dataset, config)
        assert float(tf.reduce_min(result["value"])) >= -1e-3


# =============================================================================
# VFI vs PFI agreement
# =============================================================================

class TestVFIvsPFI:

    def test_policies_agree(self, env, flat_dataset):
        """VFI and PFI should produce similar policies on the same grid."""
        gc = GridConfig(exo_sizes=[5], endo_sizes=[10], action_sizes=[12])

        result_vfi = solve_vfi(env, flat_dataset,
                               VFIConfig(grid=gc, tol=1e-6, max_iter=2000))
        result_pfi = solve_pfi(env, flat_dataset,
                               PFIConfig(grid=gc, max_iter=100, eval_steps=400))

        # Value functions should be close
        value_gap = float(tf.reduce_max(
            tf.abs(result_vfi["value"] - result_pfi["value"])).numpy())
        assert value_gap < 1.0, f"VFI-PFI value gap too large: {value_gap}"

        # Policies should be close (in endo levels, not indices)
        policy_gap = float(tf.reduce_max(
            tf.abs(result_vfi["policy_endo"] - result_pfi["policy_endo"])).numpy())
        # Allow up to 2 grid spacings of difference
        endo_grid = result_vfi["grids"]["endo_grids_1d"][0]
        max_spacing = float(np.max(np.diff(endo_grid)))
        assert policy_gap < 3 * max_spacing, (
            f"VFI-PFI policy gap {policy_gap:.4f} > 3 * max_spacing {3*max_spacing:.4f}")


# =============================================================================
# Multi-dimensional mock environment
# =============================================================================

class TwoEndoMockEnv(MDPEnvironment):
    """Mock environment with 2 endogenous, 1 exogenous, 2 action variables.

    Simple linear dynamics for testing multi-dimensional grid handling.
    Reward: -||s||^2 - ||a||^2 (pushes toward origin).
    Endo transition: s_endo_next = 0.9 * s_endo + a
    Exo transition: z' = 0.8 * z + 0.2 + sigma * eps
    """

    def exo_dim(self) -> int:
        return 1

    def endo_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 2

    def exogenous_transition(self, s_exo, eps):
        z = s_exo[..., 0:1]
        return 0.8 * z + 0.2 + 0.05 * eps[..., 0:1]

    def endogenous_transition(self, s_endo, action, s_exo):
        return 0.9 * s_endo + action

    def reward(self, s, a, temperature=1e-6):
        return -tf.reduce_sum(s ** 2, axis=-1) - tf.reduce_sum(a ** 2, axis=-1)

    def discount(self):
        return 0.9

    def action_bounds(self):
        return (
            tf.constant([-1.0, -1.0], dtype=tf.float32),
            tf.constant([ 1.0,  1.0], dtype=tf.float32),
        )

    def sample_initial_endogenous(self, n, seed):
        return tf.random.stateless_uniform(
            [n, 2], seed=seed, minval=-1.0, maxval=1.0, dtype=tf.float32)

    def sample_initial_exogenous(self, n, seed):
        return tf.random.stateless_uniform(
            [n, 1], seed=seed, minval=0.5, maxval=1.5, dtype=tf.float32)


class TestMultiDim:

    @pytest.fixture
    def multi_env(self):
        return TwoEndoMockEnv()

    @pytest.fixture
    def multi_flat_dataset(self, multi_env):
        gen = DataGenerator(multi_env, DataGeneratorConfig(n_paths=50, horizon=8))
        return gen.get_flattened_dataset("train")

    def test_grid_construction(self, multi_env):
        """Grid construction works with 2-endo, 2-action env."""
        grids = build_grids(
            multi_env,
            exo_sizes=[3],
            endo_sizes=[4, 4],
            action_sizes=[3, 3],
        )
        assert grids["exo_product"].shape == (3, 1)
        assert grids["endo_product"].shape == (16, 2)   # 4x4
        assert grids["action_product"].shape == (9, 2)  # 3x3

    def test_vfi_runs(self, multi_env, multi_flat_dataset):
        """VFI runs to completion on multi-dim environment."""
        gc = GridConfig(
            exo_sizes=[3],
            endo_sizes=[4, 4],
            action_sizes=[3, 3],
        )
        config = VFIConfig(grid=gc, tol=1e-4, max_iter=500)
        result = solve_vfi(multi_env, multi_flat_dataset, config)

        n_exo = 3
        n_endo = 16   # 4*4
        assert result["value"].shape == (n_exo, n_endo)
        assert result["policy_action"].shape == (n_exo, n_endo, 2)
        assert result["policy_endo"].shape == (n_exo, n_endo, 2)

    def test_pfi_runs(self, multi_env, multi_flat_dataset):
        """PFI runs to completion on multi-dim environment."""
        gc = GridConfig(
            exo_sizes=[3],
            endo_sizes=[4, 4],
            action_sizes=[3, 3],
        )
        config = PFIConfig(grid=gc, max_iter=50, eval_steps=100)
        result = solve_pfi(multi_env, multi_flat_dataset, config)

        n_exo = 3
        n_endo = 16
        assert result["value"].shape == (n_exo, n_endo)
        assert result["policy_action"].shape == (n_exo, n_endo, 2)


# =============================================================================
# Config validation tests
# =============================================================================

class TestConfigValidation:

    def test_vfi_invalid_tol(self):
        with pytest.raises(ValueError, match="tol must be > 0"):
            VFIConfig(tol=0)

    def test_vfi_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            VFIConfig(max_iter=0)

    def test_pfi_invalid_eval_steps(self):
        with pytest.raises(ValueError, match="eval_steps must be >= 1"):
            PFIConfig(eval_steps=0)
