"""
tests/utils/test_annealing.py

Unit tests for AnnealingSchedule class.
"""

import pytest
import numpy as np

from src.utils.annealing import AnnealingSchedule, smooth_default_prob


class TestAnnealingSchedule:
    """Unit tests for AnnealingSchedule."""
    
    def test_init_defaults(self):
        """Default initialization creates valid schedule."""
        schedule = AnnealingSchedule()
        assert schedule.init_temp == 1.0
        assert schedule.min == 0.0001
        assert schedule.decay == 0.9
        assert schedule.schedule == "exponential"
        assert schedule.value == 1.0
        assert schedule.step == 0
    
    def test_exponential_decay_deterministic(self):
        """Exponential decay is deterministic."""
        s1 = AnnealingSchedule(init_temp=1.0, decay=0.9, min=0.01)
        s2 = AnnealingSchedule(init_temp=1.0, decay=0.9, min=0.01)
        
        for _ in range(10):
            s1.update()
            s2.update()
        
        assert s1.value == s2.value
    
    def test_monotone_decreasing(self):
        """Value monotonically decreases."""
        schedule = AnnealingSchedule(init_temp=1.0, decay=0.9, min=0.001)
        
        prev = schedule.value
        for _ in range(100):
            schedule.update()
            curr = schedule.value
            assert curr <= prev
            prev = curr
    
    def test_clamps_at_min(self):
        """Value never goes below min."""
        schedule = AnnealingSchedule(init_temp=1.0, decay=0.5, min=0.1)
        
        for _ in range(100):
            schedule.update()
        
        assert schedule.value >= schedule.min
        assert schedule.value == 0.1
    
    def test_reaches_near_zero(self):
        """Value reaches near-zero relative to initial."""
        schedule = AnnealingSchedule(init_temp=1.0, min=0.01, decay=0.99)
        
        for _ in range(1000):
            schedule.update()
        
        assert schedule.value <= 0.01 * 1.01
    
    def test_linear_schedule(self):
        """Linear schedule decays linearly to min."""
        schedule = AnnealingSchedule(
            init_temp=1.0,
            min=0.0,
            schedule="linear",
            total_steps=100
        )
        
        for _ in range(50):
            schedule.update()
        
        assert abs(schedule.value - 0.5) < 0.02
        
        for _ in range(50):
            schedule.update()
        
        assert schedule.value == 0.0
    
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
    
    def test_invalid_schedule_raises(self):
        """Invalid schedule type raises ValueError."""
        schedule = AnnealingSchedule(schedule="invalid")
        
        with pytest.raises(ValueError, match="Unknown schedule"):
            schedule.update()
    
    def test_repr(self):
        """__repr__ produces informative string."""
        schedule = AnnealingSchedule(init_temp=1.0, min=0.01, schedule="exponential")
        
        repr_str = repr(schedule)
        
        assert "AnnealingSchedule" in repr_str
        assert "init=1.0" in repr_str
        assert "min=0.01" in repr_str

    def test_value_is_python_float(self):
        """Value property returns Python float, not tf.Tensor."""
        schedule = AnnealingSchedule()
        
        val = schedule.value
        
        assert isinstance(val, float)


class TestComputeSmoothDefaultProb:
    """Tests for functional smooth_default_prob."""
    
    def test_pD_at_zero_V_tilde(self):
        """p^D = 0.5 when V_tilde = 0."""
        import tensorflow as tf
        V_tilde = tf.constant([0.0])
        p_D = smooth_default_prob(V_tilde, temperature=0.1)
        assert abs(p_D.numpy()[0] - 0.5) < 1e-5
    
    def test_pD_increases_as_V_decreases(self):
        """p^D increases as V_tilde decreases."""
        import tensorflow as tf
        V_large = tf.constant([2.0])
        V_small = tf.constant([-2.0])
        
        p_D_large = smooth_default_prob(V_large, temperature=0.1)
        p_D_small = smooth_default_prob(V_small, temperature=0.1)
        
        assert p_D_small.numpy()[0] > p_D_large.numpy()[0]
    
    def test_pD_bounded_0_1(self):
        """p^D is bounded in [0, 1]."""
        import tensorflow as tf
        V = tf.constant([-100.0, -10.0, 0.0, 10.0, 100.0])
        
        p_D = smooth_default_prob(V, temperature=0.1)
        
        assert all(p_D.numpy() >= 0)
        assert all(p_D.numpy() <= 1)
