"""Tests for v2 annealing schedule."""

import math
import pytest

from src.v2.utils.annealing import AnnealingSchedule


class TestAnnealingSchedule:
    """Tests for AnnealingSchedule."""

    def test_initial_value(self):
        """Initial value equals init_temp."""
        s = AnnealingSchedule(init_temp=1.0)
        assert s.value == 1.0
        assert s.step == 0

    def test_decay_one_step(self):
        """After one update, value = init_temp * decay_rate."""
        s = AnnealingSchedule(init_temp=1.0, decay_rate=0.5, min_temp=1e-10)
        s.update()
        assert s.value == pytest.approx(0.5)
        assert s.step == 1

    def test_decay_multiple_steps(self):
        """After k steps, value = init_temp * decay_rate^k."""
        s = AnnealingSchedule(init_temp=2.0, decay_rate=0.9, min_temp=1e-10)
        for _ in range(10):
            s.update()
        expected = 2.0 * (0.9 ** 10)
        assert s.value == pytest.approx(expected, rel=1e-6)

    def test_min_temp_floor(self):
        """Value never drops below min_temp."""
        s = AnnealingSchedule(init_temp=1.0, decay_rate=0.1, min_temp=0.01)
        for _ in range(100):
            s.update()
        assert s.value == pytest.approx(0.01)

    def test_reset(self):
        """reset() restores initial state."""
        s = AnnealingSchedule(init_temp=1.0, decay_rate=0.5)
        for _ in range(5):
            s.update()
        s.reset()
        assert s.value == 1.0
        assert s.step == 0

    def test_n_anneal_positive(self):
        """n_anneal is positive for valid decay schedules."""
        s = AnnealingSchedule(init_temp=1.0, min_temp=1e-4, decay_rate=0.995)
        assert s.n_anneal > 0

    def test_n_anneal_zero_when_no_decay(self):
        """n_anneal is 0 if decay_rate >= 1 (no decay)."""
        s = AnnealingSchedule(init_temp=1.0, min_temp=1e-4, decay_rate=1.0)
        assert s.n_anneal == 0

    def test_n_anneal_zero_when_already_at_min(self):
        """n_anneal is 0 if init_temp <= min_temp."""
        s = AnnealingSchedule(init_temp=1e-4, min_temp=1e-4, decay_rate=0.99)
        assert s.n_anneal == 0

    def test_n_anneal_includes_buffer(self):
        """n_anneal > bare decay steps due to buffer fraction."""
        s_no_buf = AnnealingSchedule(init_temp=1.0, min_temp=1e-4,
                                      decay_rate=0.995, buffer=0.0)
        s_buf = AnnealingSchedule(init_temp=1.0, min_temp=1e-4,
                                   decay_rate=0.995, buffer=0.25)
        assert s_buf.n_anneal > s_no_buf.n_anneal

    def test_get_set_state_roundtrip(self):
        """get_state / set_state preserves schedule state."""
        s1 = AnnealingSchedule(init_temp=1.0, decay_rate=0.9)
        for _ in range(7):
            s1.update()
        state = s1.get_state()

        s2 = AnnealingSchedule(init_temp=1.0, decay_rate=0.9)
        s2.set_state(state)
        assert s2.value == pytest.approx(s1.value)
        assert s2.step == s1.step

    def test_repr(self):
        """__repr__ returns a readable string."""
        s = AnnealingSchedule()
        r = repr(s)
        assert "AnnealingSchedule" in r
        assert "init=" in r

    def test_value_monotonically_decreases(self):
        """Temperature decreases monotonically until hitting min_temp."""
        s = AnnealingSchedule(init_temp=1.0, min_temp=0.01, decay_rate=0.9)
        prev = s.value
        for _ in range(50):
            s.update()
            assert s.value <= prev
            prev = s.value
