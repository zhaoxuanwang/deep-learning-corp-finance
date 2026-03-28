"""
tests/utils/test_annealing.py

Unit tests for AnnealingSchedule class.
"""

import pytest
import math
import numpy as np

from src.utils.annealing import AnnealingSchedule, indicator_default, indicator_abs_gt, indicator_lt


class TestAnnealingSchedule:
    """Unit tests for AnnealingSchedule."""
    
    def test_init_defaults(self):
        """Default initialization creates valid schedule."""
        # All defaults are sourced from src/_defaults.py
        from src._defaults import (
            DEFAULT_TEMPERATURE_INIT,
            DEFAULT_TEMPERATURE_MIN,
            DEFAULT_ANNEAL_DECAY,
            DEFAULT_ANNEAL_BUFFER
        )
        schedule = AnnealingSchedule()
        assert schedule.init_temp == DEFAULT_TEMPERATURE_INIT  # 1.0
        assert schedule.min_temp == DEFAULT_TEMPERATURE_MIN    # 1e-4
        assert schedule.decay_rate == DEFAULT_ANNEAL_DECAY     # 0.995
        assert schedule.buffer == DEFAULT_ANNEAL_BUFFER        # 0.25
        assert schedule.value == DEFAULT_TEMPERATURE_INIT      # starts at init_temp
        assert schedule.step == 0
    
    def test_n_anneal_calculation(self):
        """Test calculation of N_anneal against manual check."""
        schedule = AnnealingSchedule(init_temp=1.0, min_temp=0.01, decay_rate=0.9, buffer=0.25)
        # Expected: ceil( ((ln(0.01)-ln(1))/ln(0.9)) * 1.25 ) = ceil(43.7 * 1.25) = 55
        assert schedule.n_anneal == 55

    def test_exponential_decay_deterministic(self):
        """Exponential decay is deterministic."""
        s1 = AnnealingSchedule(init_temp=1.0, decay_rate=0.9, min_temp=0.01)
        s2 = AnnealingSchedule(init_temp=1.0, decay_rate=0.9, min_temp=0.01)
        
        for _ in range(10):
            s1.update()
            s2.update()
        
        assert s1.value == s2.value
    
    def test_monotone_decreasing(self):
        """Value monotonically decreases."""
        schedule = AnnealingSchedule(init_temp=1.0, decay_rate=0.9, min_temp=0.001)
        
        prev = schedule.value
        for _ in range(100):
            schedule.update()
            curr = schedule.value
            assert curr <= prev
            prev = curr
    
    def test_clamps_at_min(self):
        """Value never goes below min."""
        schedule = AnnealingSchedule(init_temp=1.0, decay_rate=0.5, min_temp=0.1)
        
        for _ in range(100):
            schedule.update()
        
        assert schedule.value >= schedule.min_temp
        assert schedule.value == 0.1
    
    def test_reaches_near_zero(self):
        """Value reaches near-zero relative to initial."""
        schedule = AnnealingSchedule(init_temp=1.0, min_temp=0.01, decay_rate=0.99)
        
        for _ in range(1000):
            schedule.update()
        
        assert schedule.value <= 0.01 * 1.01
    
    def test_reset(self):
        """Reset restores initial state."""
        schedule = AnnealingSchedule(init_temp=2.0)
        
        for _ in range(10):
            schedule.update()
        
        assert schedule.value < 2.0
        assert schedule.step > 0
        
        schedule.reset()
        
        assert schedule.value == 2.0
        assert schedule.step == 0
    
    def test_get_set_state(self):
        """State can be checkpointed and restored."""
        schedule = AnnealingSchedule(init_temp=1.0)
        
        for _ in range(5):
            schedule.update()
        
        state = schedule.get_state()
        
        for _ in range(5):
            schedule.update()
        
        assert schedule.step == 10
        
        schedule.set_state(state)
        
        assert schedule.step == 5
    
    def test_repr(self):
        """__repr__ produces informative string including N_anneal."""
        schedule = AnnealingSchedule(init_temp=1.0, min_temp=0.01, decay_rate=0.9, buffer=0.0)
        
        repr_str = repr(schedule)
        
        assert "AnnealingSchedule" in repr_str
        assert "init=1.0" in repr_str
        assert "min=0.01" in repr_str
        assert "N_anneal=" in repr_str

    def test_value_is_python_float(self):
        """Value property returns Python float, not tf.Tensor."""
        schedule = AnnealingSchedule()
        
        val = schedule.value
        
        assert isinstance(val, float)


class TestComputeSmoothDefaultProb:
    """Tests for refactored smooth_default_prob (Gumbel-Sigmoid)."""
    
    def test_deterministic_behavior(self):
        """When noise=False, acts as deterministic sigmoid."""
        import tensorflow as tf
        # Formula: sigma(-x/tau)
        V_norm = tf.constant([0.0]) # x=0 -> logit=0 -> p=0.5
        p_D = indicator_default(V_norm, temperature=1.0, noise=False)
        assert abs(p_D.numpy()[0] - 0.5) < 1e-5
        
        # V/k = 2.0 -> logit = -2.0 -> p = sigmoid(-2) approx 0.119
        V_pos = tf.constant([2.0])
        p_D_pos = indicator_default(V_pos, temperature=1.0, noise=False)
        expected = 1 / (1 + math.exp(2.0))
        assert abs(p_D_pos.numpy()[0] - expected) < 1e-5

    def test_stochastic_behavior(self):
        """When noise=True, output differs across runs."""
        import tensorflow as tf
        V_norm = tf.constant([0.0])
        # Two calls with same input should differ due to Gumbel noise
        p1 = indicator_default(V_norm, temperature=1.0, noise=True)
        p2 = indicator_default(V_norm, temperature=1.0, noise=True)
        assert p1.numpy()[0] != p2.numpy()[0]

    def test_clipping(self):
        """Value is clipped before processing."""
        import tensorflow as tf
        # V=100 would be clipped to 20
        # If noise=False, logit = -20/1 = -20.
        # sigmoid(-20) is very close to 0.
        V_huge = tf.constant([100.0])
        p_D = indicator_default(V_huge, temperature=1.0, logit_clip=20.0, noise=False)
        expected = 1 / (1 + math.exp(20.0))
        assert abs(p_D.numpy()[0] - expected) < 1e-7


class TestIndicators:
    """Tests for indicator_abs_gt and indicator_lt."""
    
    def test_indicator_abs_gt_epsilon_shift(self):
        """Test sigma((|x| - eps)/tau)."""
        import tensorflow as tf
        # x = eps -> logit = 0 -> p = 0.5
        eps = 0.5
        x = tf.constant([0.5]) 
        # indicator_abs_gt(x, threshold=eps, mode='soft')
        val = indicator_abs_gt(x, threshold=eps, temperature=1.0, mode='soft')
        assert abs(val.numpy()[0] - 0.5) < 1e-5
        
        # x > eps -> logit > 0 -> p > 0.5
        x_large = tf.constant([2.5]) # |2.5| - 0.5 = 2.0
        val_large = indicator_abs_gt(x_large, threshold=eps, temperature=1.0, mode='soft')
        expected = 1 / (1 + math.exp(-2.0))
        assert abs(val_large.numpy()[0] - expected) < 1e-5

    def test_indicator_lt_epsilon_shift(self):
        """Test sigma(-(x + eps)/tau)."""
        import tensorflow as tf
        # x = -eps -> logit = 0 -> p = 0.5
        eps = 0.5
        # indicator_lt checks x < -eps
        # formula: -(x + eps)/tau
        # if x = -0.5, then -( -0.5 + 0.5 ) = 0
        x = tf.constant([-0.5]) 
        val = indicator_lt(x, threshold=eps, temperature=1.0, mode='soft')
        assert abs(val.numpy()[0] - 0.5) < 1e-5
        
        # x < -eps (e.g. -2.5) -> -2.5 + 0.5 = -2.0 -> -(-2.0) = 2.0 -> p > 0.5
        x_small = tf.constant([-2.5])
        val_small = indicator_lt(x_small, threshold=eps, temperature=1.0, mode='soft')
        expected = 1 / (1 + math.exp(-2.0))
        assert abs(val_small.numpy()[0] - expected) < 1e-5
