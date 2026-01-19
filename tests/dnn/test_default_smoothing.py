"""
tests/dnn/test_default_smoothing.py

Tests for default smoothing schedule and p^D computation.

Reference: outline_v2.md lines 122-135
"""

import pytest
import tensorflow as tf
import numpy as np

from src.dnn.default_smoothing import (
    DefaultSmoothingSchedule,
    compute_smooth_default_prob
)


# =============================================================================
# SCHEDULE TESTS
# =============================================================================

class TestDefaultSmoothingSchedule:
    """Tests for DefaultSmoothingSchedule."""
    
    def test_initialization(self):
        """Schedule initializes with correct values."""
        schedule = DefaultSmoothingSchedule(
            epsilon_D_0=0.5,
            epsilon_D_min=1e-4,
            decay_d=0.95,
            u_max=15.0
        )
        
        assert schedule.epsilon_D == 0.5
        assert schedule.epsilon_D_0 == 0.5
        assert schedule.epsilon_D_min == 1e-4
        assert schedule.decay_d == 0.95
        assert schedule.u_max == 15.0
    
    def test_update_decays_epsilon(self):
        """Each update should decay epsilon_D."""
        schedule = DefaultSmoothingSchedule(
            epsilon_D_0=1.0,
            epsilon_D_min=0.01,
            decay_d=0.9
        )
        
        schedule.update()
        assert np.isclose(schedule.epsilon_D, 0.9)
        
        schedule.update()
        assert np.isclose(schedule.epsilon_D, 0.81)
    
    def test_update_respects_minimum(self):
        """epsilon_D should not go below epsilon_D_min."""
        schedule = DefaultSmoothingSchedule(
            epsilon_D_0=0.1,
            epsilon_D_min=0.05,
            decay_d=0.5
        )
        
        # After first update: 0.05
        schedule.update()
        assert np.isclose(schedule.epsilon_D, 0.05)
        
        # Should stay at minimum
        schedule.update()
        assert np.isclose(schedule.epsilon_D, 0.05)
    
    def test_reset(self):
        """reset() should restore initial epsilon_D."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.5)
        
        schedule.update()
        schedule.update()
        assert schedule.epsilon_D < 0.5
        
        schedule.reset()
        assert schedule.epsilon_D == 0.5
    
    def test_get_and_set_state(self):
        """State should be saveable and restorable."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.5, decay_d=0.9)
        schedule.update()
        schedule.update()
        
        state = schedule.get_state()
        assert "epsilon_D" in state
        
        new_schedule = DefaultSmoothingSchedule()
        new_schedule.set_state(state)
        
        assert new_schedule.epsilon_D == schedule.epsilon_D


# =============================================================================
# CLIPPED LOGIT TESTS
# =============================================================================

class TestClippedLogitSigmoid:
    """Tests for clipped logit in p^D computation."""
    
    def test_clipping_bounds(self):
        """p^D uses clipped logit u in [-u_max, u_max]."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.1, u_max=5.0)
        
        # Very negative V_tilde -> large positive u -> p^D ~ 1
        V_tilde_neg = tf.constant([[-100.0]])
        p_neg = schedule.compute_default_prob(V_tilde_neg)
        
        # With clipping at u_max=5, sigmoid(5) ≈ 0.9933
        expected_max = tf.nn.sigmoid(5.0).numpy()
        assert np.isclose(p_neg.numpy()[0, 0], expected_max, rtol=0.01)
        
        # Very positive V_tilde -> large negative u -> p^D ~ 0
        V_tilde_pos = tf.constant([[100.0]])
        p_pos = schedule.compute_default_prob(V_tilde_pos)
        
        expected_min = tf.nn.sigmoid(-5.0).numpy()
        assert np.isclose(p_pos.numpy()[0, 0], expected_min, rtol=0.01)
    
    def test_without_clipping_extreme_values(self):
        """Without clipping, extreme V_tilde could cause numerical issues."""
        # This test documents that clipping prevents overflow
        schedule = DefaultSmoothingSchedule(epsilon_D_0=1e-6, u_max=50.0)
        
        V_tilde = tf.constant([[-1e10]])
        p = schedule.compute_default_prob(V_tilde)
        
        # Should not be NaN or Inf
        assert not np.isnan(p.numpy())
        assert not np.isinf(p.numpy())


# =============================================================================
# p^D PROBABILITY TESTS
# =============================================================================

class TestDefaultProbability:
    """Tests for smooth default probability computation."""
    
    def test_pD_at_zero_V_tilde(self):
        """At V_tilde = 0, p^D = sigmoid(0) = 0.5."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.1)
        
        V_tilde = tf.constant([[0.0]])
        p = schedule.compute_default_prob(V_tilde)
        
        assert np.isclose(p.numpy()[0, 0], 0.5)
    
    def test_pD_increases_as_V_decreases(self):
        """p^D should increase as V_tilde becomes more negative."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.1)
        
        V_values = tf.constant([[0.5], [0.0], [-0.5], [-1.0]])
        p_values = schedule.compute_default_prob(V_values).numpy()
        
        # p^D should be monotonically increasing
        assert p_values[0, 0] < p_values[1, 0]
        assert p_values[1, 0] < p_values[2, 0]
        assert p_values[2, 0] < p_values[3, 0]
    
    def test_pD_approaches_indicator_as_epsilon_shrinks(self):
        """As epsilon_D -> 0, p^D -> indicator{V_tilde < 0}."""
        V_tilde_neg = tf.constant([[-0.1]])
        V_tilde_pos = tf.constant([[0.1]])
        
        # Large epsilon: smooth transition
        schedule_large = DefaultSmoothingSchedule(epsilon_D_0=1.0)
        p_neg_large = schedule_large.compute_default_prob(V_tilde_neg).numpy()
        p_pos_large = schedule_large.compute_default_prob(V_tilde_pos).numpy()
        
        # Small epsilon: sharp transition
        schedule_small = DefaultSmoothingSchedule(epsilon_D_0=0.001, u_max=100)
        p_neg_small = schedule_small.compute_default_prob(V_tilde_neg).numpy()
        p_pos_small = schedule_small.compute_default_prob(V_tilde_pos).numpy()
        
        # With small epsilon, should be closer to hard threshold
        assert p_neg_small > p_neg_large, "Small epsilon should give higher p^D for V<0"
        assert p_pos_small < p_pos_large, "Small epsilon should give lower p^D for V>0"
    
    def test_pD_bounded_0_1(self):
        """p^D should always be in [0, 1]."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.1)
        
        V_tilde = tf.constant([[-100.0], [-1.0], [0.0], [1.0], [100.0]])
        p = schedule.compute_default_prob(V_tilde)
        
        assert np.all(p.numpy() >= 0)
        assert np.all(p.numpy() <= 1)


# =============================================================================
# FUNCTIONAL INTERFACE TESTS
# =============================================================================

class TestFunctionalInterface:
    """Tests for compute_smooth_default_prob function."""
    
    def test_matches_class_method(self):
        """Functional interface should match class method."""
        V_tilde = tf.constant([[0.5], [-0.5]])
        epsilon_D = 0.1
        u_max = 10.0
        
        # Class method
        schedule = DefaultSmoothingSchedule(epsilon_D_0=epsilon_D, u_max=u_max)
        p_class = schedule.compute_default_prob(V_tilde)
        
        # Functional
        p_func = compute_smooth_default_prob(V_tilde, epsilon_D, u_max)
        
        np.testing.assert_allclose(p_class.numpy(), p_func.numpy())
