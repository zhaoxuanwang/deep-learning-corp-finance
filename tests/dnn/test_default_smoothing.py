"""
tests/dnn/test_default_smoothing.py

Tests for default smoothing with AnnealingSchedule and compute_smooth_default_prob.

Reference: outline_v2.md lines 122-135
"""

import pytest
import tensorflow as tf
import numpy as np

from src.dnn.annealing import AnnealingSchedule, smooth_default_prob


# =============================================================================
# SCHEDULE TESTS
# =============================================================================

class TestAnnealingAsDefaultSmoothing:
    """Tests for AnnealingSchedule used for default smoothing."""
    
    def test_initialization(self):
        """Schedule initializes with correct values."""
        schedule = AnnealingSchedule(
            init=0.5,
            min=1e-4,
            decay=0.95,
            schedule="exponential"
        )
        
        assert schedule.value == 0.5
        assert schedule.init == 0.5
        assert schedule.min == 1e-4
        assert schedule.decay == 0.95
    
    def test_update_decays_value(self):
        """Each update should decay value."""
        schedule = AnnealingSchedule(init=1.0, min=0.01, decay=0.9)
        
        schedule.update()
        assert np.isclose(schedule.value, 0.9)
        
        schedule.update()
        assert np.isclose(schedule.value, 0.81)
    
    def test_update_respects_minimum(self):
        """Value should not go below min."""
        schedule = AnnealingSchedule(init=0.1, min=0.05, decay=0.5)
        
        schedule.update()
        assert np.isclose(schedule.value, 0.05)
        
        schedule.update()
        assert np.isclose(schedule.value, 0.05)
    
    def test_reset(self):
        """reset() should restore initial value."""
        schedule = AnnealingSchedule(init=0.5)
        
        schedule.update()
        schedule.update()
        assert schedule.value < 0.5
        
        schedule.reset()
        assert schedule.value == 0.5
    
    def test_get_and_set_state(self):
        """State should be saveable and restorable."""
        schedule = AnnealingSchedule(init=0.5, decay=0.9)
        schedule.update()
        schedule.update()
        
        state = schedule.get_state()
        assert "value" in state
        
        new_schedule = AnnealingSchedule()
        new_schedule.set_state(state)
        
        assert new_schedule.value == schedule.value


# =============================================================================
# p^D PROBABILITY TESTS
# =============================================================================

class TestDefaultProbability:
    """Tests for smooth default probability computation."""
    
    def test_pD_at_zero_V_tilde(self):
        """At V_tilde = 0, p^D = sigmoid(0) = 0.5."""
        V_tilde = tf.constant([[0.0]])
        p = smooth_default_prob(V_tilde, temperature=0.1)
        
        assert np.isclose(p.numpy()[0, 0], 0.5)
    
    def test_pD_increases_as_V_decreases(self):
        """p^D should increase as V_tilde becomes more negative."""
        V_values = tf.constant([[0.5], [0.0], [-0.5], [-1.0]])
        p_values = smooth_default_prob(V_values, temperature=0.1).numpy()
        
        assert p_values[0, 0] < p_values[1, 0]
        assert p_values[1, 0] < p_values[2, 0]
        assert p_values[2, 0] < p_values[3, 0]
    
    def test_pD_approaches_indicator_as_epsilon_shrinks(self):
        """As epsilon_D -> 0, p^D -> indicator{V_tilde < 0}."""
        V_tilde_neg = tf.constant([[-0.1]])
        V_tilde_pos = tf.constant([[0.1]])
        
        # Large epsilon: smooth transition
        p_neg_large = smooth_default_prob(V_tilde_neg, temperature=1.0).numpy()
        p_pos_large = smooth_default_prob(V_tilde_pos, temperature=1.0).numpy()
        
        # Small epsilon: sharp transition
        p_neg_small = smooth_default_prob(V_tilde_neg, temperature=0.001, u_max=100).numpy()
        p_pos_small = smooth_default_prob(V_tilde_pos, temperature=0.001, u_max=100).numpy()
        
        assert p_neg_small > p_neg_large
        assert p_pos_small < p_pos_large
    
    def test_pD_bounded_0_1(self):
        """p^D should always be in [0, 1]."""
        V_tilde = tf.constant([[-100.0], [-1.0], [0.0], [1.0], [100.0]])
        p = smooth_default_prob(V_tilde, temperature=0.1)
        
        assert np.all(p.numpy() >= 0)
        assert np.all(p.numpy() <= 1)


class TestClippedLogitSigmoid:
    """Tests for clipped logit in p^D computation."""
    
    def test_clipping_bounds(self):
        """p^D uses clipped logit u in [-u_max, u_max]."""
        # Very negative V_tilde -> large positive u -> p^D ~ 1
        V_tilde_neg = tf.constant([[-100.0]])
        p_neg = smooth_default_prob(V_tilde_neg, temperature=0.1, u_max=5.0)
        
        expected_max = tf.nn.sigmoid(5.0).numpy()
        assert np.isclose(p_neg.numpy()[0, 0], expected_max, rtol=0.01)
        
        # Very positive V_tilde -> large negative u -> p^D ~ 0
        V_tilde_pos = tf.constant([[100.0]])
        p_pos = smooth_default_prob(V_tilde_pos, temperature=0.1, u_max=5.0)
        
        expected_min = tf.nn.sigmoid(-5.0).numpy()
        assert np.isclose(p_pos.numpy()[0, 0], expected_min, rtol=0.01)
    
    def test_without_clipping_extreme_values(self):
        """Without clipping, extreme V_tilde could cause numerical issues."""
        V_tilde = tf.constant([[-1e10]])
        p = smooth_default_prob(V_tilde, temperature=1e-6, u_max=50.0)
        
        assert not np.isnan(p.numpy())
        assert not np.isinf(p.numpy())
