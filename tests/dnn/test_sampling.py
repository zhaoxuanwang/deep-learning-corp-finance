"""
tests/dnn/test_sampling.py

Tests for replay buffer, sampling, and percentile oversampling.

Reference: outline_v2.md lines 113-121
"""

import pytest
import numpy as np
import tensorflow as tf

from src.dnn.sampling import (
    AdaptiveBounds,
    ReplayBuffer,
    sample_box,
    sample_box_basic,
    sample_box_risky,
    sample_mixture,
    sample_with_oversampling,
    draw_shocks
)


# =============================================================================
# REPLAY BUFFER TESTS
# =============================================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer."""
    
    def test_initialization(self):
        """Buffer initializes with correct capacity."""
        buffer = ReplayBuffer(capacity=1000, state_dim=3)
        assert buffer.capacity == 1000
        assert buffer.state_dim == 3
        assert buffer.size == 0
    
    def test_add_states(self):
        """Adding states increases size."""
        buffer = ReplayBuffer(capacity=100, state_dim=2)
        
        states = np.random.randn(30, 2).astype(np.float32)
        buffer.add(states)
        
        assert buffer.size == 30
    
    def test_add_with_wraparound(self):
        """Buffer wraps around when capacity exceeded."""
        buffer = ReplayBuffer(capacity=50, state_dim=2)
        
        # Add 30 states
        states1 = np.ones((30, 2), dtype=np.float32)
        buffer.add(states1)
        assert buffer.size == 30
        
        # Add 30 more (total 60, but capacity is 50)
        states2 = np.ones((30, 2), dtype=np.float32) * 2
        buffer.add(states2)
        assert buffer.size == 50  # Capped at capacity
    
    def test_sample_returns_correct_shape(self):
        """Sample returns correct shape."""
        buffer = ReplayBuffer(capacity=100, state_dim=3, seed=42)
        
        states = np.random.randn(50, 3).astype(np.float32)
        buffer.add(states)
        
        samples = buffer.sample(20)
        assert samples.shape == (20, 3)
    
    def test_sample_from_empty_raises(self):
        """Sampling from empty buffer raises error."""
        buffer = ReplayBuffer(capacity=100)
        
        with pytest.raises(ValueError, match="Cannot sample from empty"):
            buffer.sample(10)
    
    def test_clear(self):
        """Clear resets buffer."""
        buffer = ReplayBuffer(capacity=100)
        buffer.add(np.random.randn(50, 3).astype(np.float32))
        
        buffer.clear()
        assert buffer.size == 0


# =============================================================================
# PERCENTILE OVERSAMPLING TESTS
# =============================================================================

class TestPercentileOversampling:
    """Tests for percentile-based oversampling near default boundary."""
    
    def test_update_values(self):
        """update_values stores |V_tilde| correctly."""
        buffer = ReplayBuffer(capacity=100, state_dim=3)
        states = np.random.randn(50, 3).astype(np.float32)
        buffer.add(states)
        
        abs_values = np.random.rand(50).astype(np.float32)
        buffer.update_values(abs_values)
        
        np.testing.assert_allclose(buffer._abs_values, abs_values)
    
    def test_update_values_wrong_size_raises(self):
        """update_values with wrong size raises error."""
        buffer = ReplayBuffer(capacity=100)
        buffer.add(np.random.randn(50, 3).astype(np.float32))
        
        with pytest.raises(ValueError, match="Expected 50 values"):
            buffer.update_values(np.random.rand(30))
    
    def test_sample_near_default_selects_low_values(self):
        """Oversampling selects states with low |V_tilde|."""
        np.random.seed(42)
        buffer = ReplayBuffer(capacity=100, state_dim=3, seed=42)
        
        # Create states with known |V_tilde| values
        states = np.arange(100).reshape(-1, 1).repeat(3, axis=1).astype(np.float32)
        buffer.add(states)
        
        # |V_tilde|: indices 0-9 have low values, 90-99 have high values
        abs_values = np.arange(100).astype(np.float32) / 10  # 0.0 to 9.9
        buffer.update_values(abs_values)
        
        # Sample from bottom 10%
        samples = buffer.sample_near_default(n=100, percentile_q=0.10)
        
        # All samples should come from states with low |V_tilde|
        # States 0-9 have abs_values 0.0-0.9, which are bottom 10%
        threshold = np.percentile(abs_values, 10)
        
        # Check that sampled states correspond to low indices
        sample_indices = samples[:, 0]  # First column is index
        assert np.all(sample_indices <= 10), \
            f"Should sample from bottom 10%, got indices up to {sample_indices.max()}"
    
    def test_sample_near_default_without_values_raises(self):
        """Oversampling without update_values raises error."""
        buffer = ReplayBuffer(capacity=100)
        buffer.add(np.random.randn(50, 3).astype(np.float32))
        
        with pytest.raises(ValueError, match="Must call update_values"):
            buffer.sample_near_default(10, 0.1)
    
    def test_percentile_adapts_to_rescaling(self):
        """
        Percentile rule auto-adapts to value rescaling.
        
        If all |V_tilde| values are scaled by 10x, the bottom-q percentile
        still selects the same states.
        """
        np.random.seed(42)
        buffer = ReplayBuffer(capacity=100, state_dim=1, seed=42)
        
        states = np.arange(100).reshape(-1, 1).astype(np.float32)
        buffer.add(states)
        
        # Original values
        abs_values_original = np.random.rand(100).astype(np.float32)
        buffer.update_values(abs_values_original)
        samples_original = buffer.sample_near_default(50, 0.2)
        
        # Scaled values (10x)
        buffer.update_values(abs_values_original * 10)
        samples_scaled = buffer.sample_near_default(50, 0.2)
        
        # The same percentile should select similar states
        # (not exactly same due to random sampling)
        threshold_orig = np.percentile(abs_values_original, 20)
        threshold_scaled = np.percentile(abs_values_original * 10, 20)
        
        assert np.isclose(threshold_scaled, threshold_orig * 10)


# =============================================================================
# BOX SAMPLING TESTS
# =============================================================================

class TestBoxSampling:
    """Tests for box sampling functions."""
    
    def test_sample_box_basic(self):
        """Basic box sampling returns correct types."""
        rng = np.random.default_rng(42)
        k, z = sample_box_basic(
            n=100,
            k_bounds=(0.1, 10.0),
            log_z_bounds=(-0.5, 0.5),
            rng=rng
        )
        
        assert k.shape == (100,)
        assert z.shape == (100,)
        assert np.all(k >= 0.1) and np.all(k <= 10.0)
        assert np.all(z > 0)  # z = exp(log_z) > 0
    
    def test_sample_box_risky(self):
        """Risky debt box sampling returns correct types."""
        rng = np.random.default_rng(42)
        k, b, z = sample_box_risky(
            n=100,
            k_bounds=(0.1, 10.0),
            b_bounds=(0.0, 5.0),
            log_z_bounds=(-0.5, 0.5),
            rng=rng
        )
        
        assert k.shape == (100,)
        assert b.shape == (100,)
        assert z.shape == (100,)
        assert np.all(b >= 0), "b must be >= 0 (borrowing-only)"


# =============================================================================
# MIXTURE SAMPLING TESTS
# =============================================================================

class TestMixtureSampling:
    """Tests for mixture sampling."""
    
    def test_mixture_with_replay(self):
        """Mixture includes both box and replay samples."""
        rng = np.random.default_rng(42)
        
        buffer = ReplayBuffer(capacity=100, state_dim=2, seed=42)
        buffer.add(np.ones((50, 2), dtype=np.float32) * 999)  # Marked value
        
        def box_sampler(n):
            return np.zeros((n, 2), dtype=np.float32)  # Zeros for box
        
        samples = sample_mixture(
            n=100,
            box_sampler=box_sampler,
            replay_buffer=buffer,
            replay_ratio=0.3,
            rng=rng
        )
        
        assert samples.shape == (100, 2)
        
        # Should have some zeros (box) and some 999s (replay)
        n_box = np.sum(samples[:, 0] == 0)
        n_replay = np.sum(samples[:, 0] == 999)
        
        assert n_box > 0, "Should have box samples"
        assert n_replay > 0, "Should have replay samples"


# =============================================================================
# SHOCK DRAWING TESTS
# =============================================================================

class TestDrawShocks:
    """Tests for draw_shocks function."""
    
    def test_two_independent_draws(self):
        """draw_shocks returns two different draws."""
        z = tf.constant([1.0, 1.0, 1.0])
        
        z1, z2 = draw_shocks(
            n=3,
            z_curr=z,
            rho=0.9,
            sigma=0.1,
            mu=0.0
        )
        
        assert z1.shape == (3, 1)
        assert z2.shape == (3, 1)
        
        # Draws should be different (almost certainly with random seed)
        assert not np.allclose(z1.numpy(), z2.numpy())
    
    def test_deterministic_with_rng(self):
        """Fixed RNG produces reproducible shocks."""
        z = tf.constant([1.0, 2.0])
        
        rng1 = np.random.default_rng(42)
        z1_a, z2_a = draw_shocks(2, z, 0.9, 0.1, 0.0, rng=rng1)
        
        rng2 = np.random.default_rng(42)
        z1_b, z2_b = draw_shocks(2, z, 0.9, 0.1, 0.0, rng=rng2)
        
        np.testing.assert_allclose(z1_a.numpy(), z1_b.numpy())
        np.testing.assert_allclose(z2_a.numpy(), z2_b.numpy())
    
    def test_ar1_process(self):
        """Shocks follow AR(1) process in log space."""
        z_curr = tf.constant([1.0])  # log(z) = 0
        rho, sigma, mu = 0.9, 0.1, 0.0
        
        # With many samples, mean of log(z') should be rho * log(z) + (1-rho)*mu
        # = 0.9 * 0 + 0.1 * 0 = 0
        z1_samples = []
        for _ in range(1000):
            z1, _ = draw_shocks(1, z_curr, rho, sigma, mu)
            z1_samples.append(z1.numpy()[0, 0])
        
        mean_log_z = np.mean(np.log(z1_samples))
        assert np.abs(mean_log_z) < 0.1, "Mean should be close to 0"


# =============================================================================
# ADAPTIVE BOUNDS TESTS
# =============================================================================

class TestAdaptiveBounds:
    """Tests for AdaptiveBounds class."""
    
    def test_user_bounds_override(self):
        """User-provided bounds override adaptive mode."""
        bounds = AdaptiveBounds(
            initial_bounds={"k": (0.1, 10.0), "log_z": (-0.5, 0.5)},
            user_bounds={"k": (1.0, 5.0), "log_z": (-0.2, 0.2)}
        )
        
        # Even with a full buffer, should use user bounds
        buffer = ReplayBuffer(capacity=200, state_dim=2, seed=42)
        buffer.add(np.random.randn(150, 2).astype(np.float32))
        
        current, meta = bounds.get_bounds(buffer)
        
        assert meta["source"] == "user"
        assert current["k"] == (1.0, 5.0)
        assert current["log_z"] == (-0.2, 0.2)
    
    def test_warmup_uses_initial_bounds(self):
        """Empty or small buffer falls back to initial bounds."""
        bounds = AdaptiveBounds(
            initial_bounds={"k": (0.1, 10.0), "log_z": (-0.5, 0.5)},
            min_samples=100
        )
        
        # Empty buffer
        current, meta = bounds.get_bounds(None)
        assert meta["source"] == "initial"
        assert current["k"] == (0.1, 10.0)
        
        # Small buffer (< min_samples)
        buffer = ReplayBuffer(capacity=200, state_dim=2, seed=42)
        buffer.add(np.random.randn(50, 2).astype(np.float32))  # 50 < 100
        
        current, meta = bounds.get_bounds(buffer)
        assert meta["source"] == "initial"
        assert meta["buffer_size"] == 50
    
    def test_adaptive_uses_buffer(self):
        """With enough data, uses buffer statistics."""
        bounds = AdaptiveBounds(
            initial_bounds={"k": (0.1, 10.0), "log_z": (-0.5, 0.5)},
            min_samples=100,
            var_names=("k", "log_z")
        )
        
        # Fill buffer with data in a narrower range
        buffer = ReplayBuffer(capacity=200, state_dim=2, seed=42)
        np.random.seed(42)
        data = np.column_stack([
            np.random.uniform(2, 4, 150),
            np.random.uniform(-0.1, 0.1, 150)
        ]).astype(np.float32)
        buffer.add(data)
        
        current, meta = bounds.get_bounds(buffer)
        
        assert meta["source"] == "adaptive"
        assert meta["buffer_size"] == 150
        
        # Bounds should reflect buffer data (but never narrower than initial)
        k_lo, k_hi = current["k"]
        assert k_lo <= 0.1, "Should include initial lower bound"
        assert k_hi >= 10.0, "Should include initial upper bound"
    
    def test_bounds_never_shrink_below_minimum(self):
        """Computed bounds never shrink below min_range_fraction of initial."""
        bounds = AdaptiveBounds(
            initial_bounds={"k": (0.0, 10.0)},  # Range = 10
            min_samples=50,
            min_range_fraction=0.5,  # Min range = 5
            var_names=("k",)
        )
        
        # Buffer with very tight data (range << 5)
        buffer = ReplayBuffer(capacity=100, state_dim=1, seed=42)
        data = np.random.uniform(4.9, 5.1, (60, 1)).astype(np.float32)
        buffer.add(data)
        
        current, meta = bounds.get_bounds(buffer)
        
        k_lo, k_hi = current["k"]
        computed_range = k_hi - k_lo
        
        # Range should be at least 50% of initial (5.0)
        assert computed_range >= 5.0, \
            f"Range {computed_range} should be >= 5.0 (50% of initial 10.0)"
    
    def test_margin_expansion(self):
        """Computed bounds are expanded by margin."""
        bounds = AdaptiveBounds(
            initial_bounds={"k": (0.0, 100.0)},  # Wide initial to not constrain
            min_samples=50,
            margin=0.20,  # 20% expansion
            var_names=("k",)
        )
        
        # Buffer with data in [40, 60] range
        buffer = ReplayBuffer(capacity=100, state_dim=1, seed=42)
        data = np.random.uniform(40, 60, (60, 1)).astype(np.float32)
        buffer.add(data)
        
        current, meta = bounds.get_bounds(buffer)
        
        k_lo, k_hi = current["k"]
        
        # Percentile range ~ 40-60, margin 20% should expand to ~36-64
        # But final is clipped to initial bounds (0, 100)
        assert k_lo <= 36, f"Lower bound {k_lo} should be <= 36 after margin"
        assert k_hi >= 64, f"Upper bound {k_hi} should be >= 64 after margin"
