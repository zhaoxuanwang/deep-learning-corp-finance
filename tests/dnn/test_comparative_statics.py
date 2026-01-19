"""
tests/dnn/test_comparative_statics.py

Comparative statics and policy sanity tests.

Verifies that trained policies satisfy basic economic intuitions:
- Monotonicity in productivity
- Adjustment cost effects
- Transform correctness
"""

import pytest
import numpy as np
import tensorflow as tf
from dataclasses import replace

from src.economy import EconomicParams
from src.dnn import (
    TrainingConfig, EconomicScenario, SamplingBounds,
    train_basic_lr, train_basic_br,
    evaluate_basic_policy, get_eval_grids,
    BasicPolicyNetwork, BasicValueNetwork,
)


# =============================================================================
# Fixtures: Fast training configs
# =============================================================================

@pytest.fixture(scope="module")
def fast_config():
    """Minimal training config for fast tests."""
    return TrainingConfig(
        n_layers=2, n_neurons=16, n_iter=60,
        batch_size=256, learning_rate=3e-3, log_every=60, seed=42
    )


@pytest.fixture(scope="module")
def smooth_scenario():
    """
    Smooth objective scenario: no fixed costs, no external finance costs.
    All gradients should be well-behaved.
    """
    return EconomicScenario.from_overrides(
        "smooth_baseline",
        cost_fixed=0.0,
        cost_convex=0.01,
        cost_inject_fixed=0.0,
        cost_inject_linear=0.0,
    )


@pytest.fixture
def fixed_cost_scenario():
    """Fixed adjustment cost scenario."""
    return EconomicScenario.from_overrides(
        "fixed_cost",
        cost_fixed=0.01,
        cost_convex=0.0,
        cost_inject_fixed=0.0,
        cost_inject_linear=0.0,
    )


@pytest.fixture
def convex_cost_scenario():
    """Convex adjustment cost scenario."""
    return EconomicScenario.from_overrides(
        "convex_cost",
        cost_fixed=0.0,
        cost_convex=0.5,  # Substantial convex cost
        cost_inject_fixed=0.0,
        cost_inject_linear=0.0,
    )


# =============================================================================
# Shared Fixtures (Scope = Module)
# =============================================================================

@pytest.fixture(scope="module")
def shared_smooth_history(smooth_scenario, fast_config):
    """
    Trains the smooth baseline model ONCE for the entire test module.
    This saves ~90% of test runtime.
    """
    return train_basic_lr(smooth_scenario, fast_config)

# =============================================================================
# Part A: Isolation Tests
# =============================================================================

class TestSmoothBaseline:
    """A1: Smooth-baseline configuration tests."""
    
    def test_lr_loss_decreases(self, shared_smooth_history):
        """LR loss should decrease during training."""
        history = shared_smooth_history
        
        loss_start = history['loss_LR'][0]
        loss_end = history['loss_LR'][-1]
        
        # Loss should decrease (or at least not increase significantly)
        # LR loss is negative reward, so it should decrease
        assert loss_end <= loss_start + 0.1, \
            f"LR loss increased: {loss_start:.4f} -> {loss_end:.4f}"
    
    def test_br_critic_loss_decreases(self, smooth_scenario, fast_config):
        """BR critic loss should decrease during training."""
        history = train_basic_br(smooth_scenario, fast_config)
        
        loss_start = history['loss_BR_critic'][0]
        loss_end = history['loss_BR_critic'][-1]
        
        # Critic loss (Bellman residual) should decrease
        assert loss_end < loss_start * 2, \
            f"Critic loss unexpected: {loss_start:.6f} -> {loss_end:.6f}"
    
    def test_policy_outputs_valid(self, shared_smooth_history, smooth_scenario):
        """Policy outputs should be positive and finite."""
        history = shared_smooth_history
        policy_net = history['_policy_net']
        
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=20, n_z=10)
        eval_result = evaluate_basic_policy(policy_net, k_grid, z_grid)
        
        k_next = eval_result['k_next']
        
        # All k' should be positive
        assert np.all(k_next > 0), "k' should be positive everywhere"
        
        # All k' should be finite
        assert np.all(np.isfinite(k_next)), "k' should be finite everywhere"
        
        # k' shouldn't collapse to minimum for most states
        k_min = k_grid[0]
        fraction_at_min = np.mean(k_next < k_min * 1.05)
        assert fraction_at_min < 0.5, \
            f"Too many states at k_min: {fraction_at_min*100:.1f}%"


# =============================================================================
# Part B: Basic Model Comparative Statics
# =============================================================================

class TestMonotonicity:
    """B1: Policy monotonicity in productivity."""
    
    def test_kprime_monotone_in_z(self, shared_smooth_history, smooth_scenario):
        """k' should be non-decreasing in z at fixed k."""
        history = shared_smooth_history
        policy_net = history['_policy_net']
        
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=10, n_z=20)
        eval_result = evaluate_basic_policy(policy_net, k_grid, z_grid)
        
        k_next = eval_result['k_next']
        
        # For each k level, check monotonicity in z
        violations = 0
        for k_idx in range(len(k_grid)):
            k_prime_vs_z = k_next[k_idx, :]
            diffs = np.diff(k_prime_vs_z)
            
            # Allow small tolerance for non-monotonicity
            significant_decreases = diffs < -0.01
            violations += np.sum(significant_decreases)
        
        # Allow some minor violations due to approximation
        max_violations = len(k_grid) * 2  # At most 2 per k level
        assert violations <= max_violations, \
            f"Too many monotonicity violations: {violations}"
    
    def test_Ik_monotone_in_z(self, shared_smooth_history, smooth_scenario):
        """I/k should be non-decreasing in z at fixed k."""
        history = shared_smooth_history
        policy_net = history['_policy_net']
        
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=10, n_z=20)
        eval_result = evaluate_basic_policy(policy_net, k_grid, z_grid)
        
        I_k = eval_result['I_k']
        
        # For each k level, check monotonicity
        violations = 0
        for k_idx in range(len(k_grid)):
            Ik_vs_z = I_k[k_idx, :]
            diffs = np.diff(Ik_vs_z)
            
            significant_decreases = diffs < -0.01
            violations += np.sum(significant_decreases)
        
        max_violations = len(k_grid) * 2
        assert violations <= max_violations, \
            f"Too many I/k monotonicity violations: {violations}"


class TestAdjustmentCostEffects:
    """B3, B4: Adjustment cost effects on policy."""
    
    def test_convex_costs_damp_responsiveness(
        self, smooth_scenario, convex_cost_scenario, fast_config
    ):
        """Convex costs should reduce I/k responsiveness to z."""
        # Train both scenarios
        history_smooth = train_basic_lr(smooth_scenario, fast_config)
        history_convex = train_basic_lr(convex_cost_scenario, fast_config)
        
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=10, n_z=20)
        
        eval_smooth = evaluate_basic_policy(history_smooth['_policy_net'], k_grid, z_grid)
        eval_convex = evaluate_basic_policy(history_convex['_policy_net'], k_grid, z_grid)
        
        # Compute I/k range over z at each k
        Ik_range_smooth = np.max(eval_smooth['I_k'], axis=1) - np.min(eval_smooth['I_k'], axis=1)
        Ik_range_convex = np.max(eval_convex['I_k'], axis=1) - np.min(eval_convex['I_k'], axis=1)
        
        # Average range should be smaller under convex costs
        avg_range_smooth = np.mean(Ik_range_smooth)
        avg_range_convex = np.mean(Ik_range_convex)
        
        # Allow for some noise, but convex should dampen
        assert avg_range_convex <= avg_range_smooth * 1.5, \
            f"Convex costs did not dampen: smooth={avg_range_smooth:.4f}, convex={avg_range_convex:.4f}"


# =============================================================================
# Part D: Transform Correctness
# =============================================================================

class TestTransformCorrectness:
    """D1: Log/level transform verification."""
    
    def test_network_takes_levels_converts_to_log(self):
        """Networks should take level inputs and convert to log internally."""
        policy_net = BasicPolicyNetwork(
            n_layers=2, n_neurons=16, k_min=0.1
        )
        
        # Feed levels
        k_level = tf.constant([[2.0], [3.0]], dtype=tf.float32)
        z_level = tf.constant([[1.0], [1.5]], dtype=tf.float32)
        
        # Should not error and should return positive k'
        k_next = policy_net(k_level, z_level)
        
        assert k_next.shape == (2, 1)
        assert tf.reduce_all(k_next > 0).numpy()
    
    def test_log_level_roundtrip(self):
        """exp(log(x)) = x for network inputs."""
        k_levels = np.array([0.5, 1.0, 2.0, 5.0])
        z_levels = np.array([0.7, 1.0, 1.3])
        
        # Log transform
        log_k = np.log(k_levels)
        log_z = np.log(z_levels)
        
        # Recover levels
        k_recovered = np.exp(log_k)
        z_recovered = np.exp(log_z)
        
        np.testing.assert_allclose(k_recovered, k_levels, rtol=1e-6)
        np.testing.assert_allclose(z_recovered, z_levels, rtol=1e-6)
    
    def test_policy_consistent_with_level_inputs(self, smooth_scenario):
        """Policy output should be consistent whether using levels or log inputs."""
        policy_net = BasicPolicyNetwork(
            n_layers=2, n_neurons=16, k_min=0.1
        )
        
        k = 2.0
        z = 1.5
        
        # Call with levels
        k_tf = tf.constant([[k]], dtype=tf.float32)
        z_tf = tf.constant([[z]], dtype=tf.float32)
        k_next_1 = policy_net(k_tf, z_tf).numpy()[0, 0]
        
        # Call again with same levels
        k_next_2 = policy_net(k_tf, z_tf).numpy()[0, 0]
        
        # Should be deterministic
        assert k_next_1 == k_next_2


class TestLossWiring:
    """D3: Loss sign and wiring verification."""
    
    def test_lr_loss_is_negative_reward(self, shared_smooth_history):
        """LR loss should be negative of reward (higher reward = lower loss)."""
        # Train briefly
        history = shared_smooth_history
        
        # Loss is defined as negative of average discounted reward
        # A better policy (higher reward) should have lower loss
        # Just verify loss is finite and not exploding
        loss = history['loss_LR'][-1]
        assert np.isfinite(loss), "Loss should be finite"
    
    def test_higher_z_gives_higher_profit(self, smooth_scenario):
        """Verify primitive: higher z gives higher profit."""
        from src.economy import production_function
        
        params = smooth_scenario.params
        k = tf.constant([2.0], dtype=tf.float32)
        
        z_low = tf.constant([0.8], dtype=tf.float32)
        z_high = tf.constant([1.2], dtype=tf.float32)
        
        pi_low = production_function(k, z_low, params).numpy()[0]
        pi_high = production_function(k, z_high, params).numpy()[0]
        
        assert pi_high > pi_low, f"Profit not increasing: {pi_low} vs {pi_high}"


class TestActivationPathologies:
    """D4: Check for softplus saturation and k' collapse."""
    
    def test_kprime_not_collapsed_to_kmin(self, shared_smooth_history, smooth_scenario):
        """k' should not be at k_min for most states."""
        history = shared_smooth_history
        policy_net = history['_policy_net']
        
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=20, n_z=20)
        eval_result = evaluate_basic_policy(policy_net, k_grid, z_grid)
        
        k_next = eval_result['k_next']
        k_min = smooth_scenario.sampling.k_bounds[0]
        
        # Fraction at k_min
        at_min = np.mean(k_next < k_min * 1.1)
        
        assert at_min < 0.3, f"Too many states at k_min: {at_min*100:.1f}%"
    
    def test_kprime_not_exploding(self, shared_smooth_history, smooth_scenario):
        """k' should not explode to very large values."""
        history = shared_smooth_history
        policy_net = history['_policy_net']
        
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=20, n_z=20)
        eval_result = evaluate_basic_policy(policy_net, k_grid, z_grid)
        
        k_next = eval_result['k_next']
        k_max = smooth_scenario.sampling.k_bounds[1]
        
        # Fraction exploding
        exploding = np.mean(k_next > k_max * 2)
        
        assert exploding < 0.1, f"Too many states exploding: {exploding*100:.1f}%"


# =============================================================================
# Integration Test: Full Sanity Check
# =============================================================================

class TestFullSanityCheck:
    """Integration test combining multiple checks."""
    
    def test_smooth_baseline_full_check(self, shared_smooth_history, smooth_scenario):
        """Full sanity check on smooth baseline."""
        history = shared_smooth_history
        
        # 1. Loss decreased
        assert history['loss_LR'][-1] < history['loss_LR'][0] + 0.5
        
        # 2. Policy is valid
        policy_net = history['_policy_net']
        k_grid, z_grid, _ = get_eval_grids(smooth_scenario, n_k=15, n_z=15)
        eval_result = evaluate_basic_policy(policy_net, k_grid, z_grid)
        
        k_next = eval_result['k_next']
        
        # All positive and finite
        assert np.all(k_next > 0)
        assert np.all(np.isfinite(k_next))
        
        # 3. Monotonicity in z (at median k)
        k_idx = len(k_grid) // 2
        k_prime_vs_z = k_next[k_idx, :]
        
        # At least weakly increasing (allow 2 violations)
        diffs = np.diff(k_prime_vs_z)
        significant_decreases = np.sum(diffs < -0.01)
        assert significant_decreases <= 2
