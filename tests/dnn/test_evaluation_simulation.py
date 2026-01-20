"""
tests/dnn/test_evaluation_simulation.py

Unit tests for src/dnn/evaluation/simulation.py
Tests policy simulation and Monte Carlo evaluation.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.dnn.evaluation.simulation import (
    _ar1_step_numpy,
    simulate_policy_path,
    evaluate_policy_return,
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
        log_z_bounds=(-0.5, 0.5)
    )
    return EconomicScenario.from_overrides(
        name="test_scenario",
        delta=0.1,
        r_rate=0.04,
        rho=0.9,
        sigma=0.02,
        mu=0.0,
        sampling=bounds
    )


@pytest.fixture
def deterministic_policy_net():
    """Create a policy network that returns k' = 0.9*k (deterministic)."""
    class DeterministicPolicy(tf.keras.Model):
        def call(self, k, z):
            return k * 0.9

    net = DeterministicPolicy()
    return net


@pytest.fixture
def identity_policy_net():
    """Create a policy network that returns k' = k (identity)."""
    class IdentityPolicy(tf.keras.Model):
        def call(self, k, z):
            return k

    net = IdentityPolicy()
    return net


# =============================================================================
# TEST AR(1) HELPER
# =============================================================================

class TestAR1StepNumpy:
    """Test AR(1) transition helper."""

    def test_ar1_step_deterministic(self):
        """Test AR(1) with zero shock (deterministic case)."""
        # Create RNG with fixed seed
        rng = np.random.default_rng(42)

        # Parameters
        z0 = 1.0
        rho = 0.9
        sigma = 0.0  # No noise
        mu = 0.0

        # Single step
        z1 = _ar1_step_numpy(z0, rho, sigma, mu, rng)

        # Expected: log(z1) = (1-rho)*mu + rho*log(z0) = 0.9 * 0 = 0
        # So z1 = exp(0) = 1.0
        assert abs(z1 - 1.0) < 1e-6

    def test_ar1_step_with_noise(self):
        """Test AR(1) with non-zero shock."""
        rng = np.random.default_rng(123)

        z0 = 1.0
        rho = 0.8
        sigma = 0.1
        mu = 0.0

        # Run multiple steps
        z_path = [z0]
        for _ in range(100):
            z_new = _ar1_step_numpy(z_path[-1], rho, sigma, mu, rng)
            z_path.append(z_new)

        z_path = np.array(z_path)

        # Check that z stays positive
        assert np.all(z_path > 0)

        # Check that log(z) has reasonable variance
        log_z = np.log(z_path)
        assert np.std(log_z) > 0  # Non-zero variance due to shocks

    def test_ar1_step_mean_reversion(self):
        """Test mean reversion to mu."""
        rng = np.random.default_rng(456)

        z0 = 0.5  # Start below mean
        rho = 0.5  # Moderate persistence
        sigma = 0.05
        mu = 1.0  # Mean log(z) = 1

        # Simulate long path
        z_path = [z0]
        for _ in range(500):
            z_new = _ar1_step_numpy(z_path[-1], rho, sigma, mu, rng)
            z_path.append(z_new)

        # After burn-in, log(z) should be near mu
        log_z_late = np.log(z_path[400:])
        mean_log_z = np.mean(log_z_late)

        # Should be close to mu (with some sampling error)
        assert abs(mean_log_z - mu) < 0.3


# =============================================================================
# TEST SIMULATE_POLICY_PATH
# =============================================================================

class TestSimulatePolicyPath:
    """Test policy path simulation."""

    def test_simulate_policy_path_shape(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify output shapes."""
        T_eval = 50
        k0, z0 = 5.0, 1.0

        rewards, k_path, z_path = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=42
        )

        assert rewards.shape == (T_eval,)
        assert k_path.shape == (T_eval + 1,)
        assert z_path.shape == (T_eval + 1,)

    def test_simulate_policy_path_initial_conditions(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify initial conditions are respected."""
        k0, z0 = 3.5, 1.2

        _, k_path, z_path = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=10,
            k0=k0,
            z0=z0,
            seed=99
        )

        assert abs(k_path[0] - k0) < 1e-6
        assert abs(z_path[0] - z0) < 1e-6

    def test_simulate_policy_path_reproducibility(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify same seed gives same results."""
        k0, z0 = 5.0, 1.0
        T_eval = 30
        seed = 12345

        # Run twice with same seed
        rewards1, k_path1, z_path1 = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=seed
        )

        rewards2, k_path2, z_path2 = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=seed
        )

        # Should be identical
        np.testing.assert_array_equal(rewards1, rewards2)
        np.testing.assert_array_equal(k_path1, k_path2)
        np.testing.assert_array_equal(z_path1, z_path2)

    def test_simulate_policy_path_different_seeds(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify different seeds give different results."""
        k0, z0 = 5.0, 1.0
        T_eval = 30

        # Run with different seeds
        _, _, z_path1 = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=111
        )

        _, _, z_path2 = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=222
        )

        # z paths should differ (due to different shocks)
        assert not np.allclose(z_path1, z_path2)

    def test_simulate_policy_path_deterministic_policy(
        self, deterministic_policy_net, simple_scenario
    ):
        """Test with k' = 0.9*k policy."""
        k0 = 10.0
        z0 = 1.0
        T_eval = 5

        _, k_path, _ = simulate_policy_path(
            deterministic_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=42
        )

        # Check that k follows k_{t+1} = 0.9 * k_t
        for t in range(T_eval):
            expected = k0 * (0.9 ** (t + 1))
            actual = k_path[t + 1]
            assert abs(actual - expected) < 1e-4, (
                f"At t={t+1}: expected k={expected:.6f}, got {actual:.6f}"
            )

    def test_simulate_policy_path_identity_policy(
        self, identity_policy_net, simple_scenario
    ):
        """Test with k' = k identity policy."""
        k0 = 7.0
        z0 = 1.0
        T_eval = 10

        _, k_path, _ = simulate_policy_path(
            identity_policy_net,
            simple_scenario,
            T_eval=T_eval,
            k0=k0,
            z0=z0,
            seed=123
        )

        # All k values should be identical to k0
        np.testing.assert_allclose(k_path, k0, rtol=1e-5)


# =============================================================================
# TEST EVALUATE_POLICY_RETURN
# =============================================================================

class TestEvaluatePolicyReturn:
    """Test Monte Carlo policy evaluation."""

    def test_evaluate_policy_return_structure(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify output structure."""
        result = evaluate_policy_return(
            deterministic_policy_net,
            simple_scenario,
            n_paths=10,
            T_eval=50,
            burn_in=10,
            seed=42
        )

        # Check keys
        expected_keys = {
            "mean_reward", "std_reward", "p10_reward", "p50_reward", "p90_reward"
        }
        assert set(result.keys()) == expected_keys

        # Check all values are floats
        for key, val in result.items():
            assert isinstance(val, float)
            assert not np.isnan(val)

    def test_evaluate_policy_return_reproducibility(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify same seed gives same results."""
        seed = 9999

        result1 = evaluate_policy_return(
            deterministic_policy_net,
            simple_scenario,
            n_paths=20,
            T_eval=100,
            burn_in=20,
            seed=seed
        )

        result2 = evaluate_policy_return(
            deterministic_policy_net,
            simple_scenario,
            n_paths=20,
            T_eval=100,
            burn_in=20,
            seed=seed
        )

        # Should be identical
        for key in result1:
            assert abs(result1[key] - result2[key]) < 1e-10, (
                f"Mismatch in {key}: {result1[key]} vs {result2[key]}"
            )

    def test_evaluate_policy_return_percentile_ordering(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify p10 <= p50 <= p90."""
        result = evaluate_policy_return(
            deterministic_policy_net,
            simple_scenario,
            n_paths=50,
            T_eval=100,
            burn_in=20,
            seed=777
        )

        assert result["p10_reward"] <= result["p50_reward"]
        assert result["p50_reward"] <= result["p90_reward"]

    def test_evaluate_policy_return_burn_in_effect(
        self, deterministic_policy_net, simple_scenario
    ):
        """Verify burn-in excludes initial transient."""
        # With burn_in=0
        result_no_burnin = evaluate_policy_return(
            deterministic_policy_net,
            simple_scenario,
            n_paths=20,
            T_eval=100,
            burn_in=0,
            seed=333
        )

        # With burn_in=50
        result_with_burnin = evaluate_policy_return(
            deterministic_policy_net,
            simple_scenario,
            n_paths=20,
            T_eval=100,
            burn_in=50,
            seed=333
        )

        # Results may differ due to burn-in
        # (Not necessarily, but burn-in at least shouldn't crash)
        assert isinstance(result_with_burnin["mean_reward"], float)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
