"""Tests for v2 annealing schedules."""

import math
import pytest

from src.v2.utils.annealing import (
    AnnealingSchedule,
    build_geometric_stepwise_plan,
    geometric_temperature_ladder,
    HoldDecayFloorSchedule,
    StepwiseTemperatureSchedule,
    weighted_stage_steps,
)


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


class TestHoldDecayFloorSchedule:
    """Tests for hold-decay-floor continuation schedule."""

    def test_initial_value(self):
        s = HoldDecayFloorSchedule(
            init_temp=1e-2, min_temp=1e-3, hold_steps=3, decay_steps=4
        )
        assert s.value == pytest.approx(1e-2)
        assert s.step == 0
        assert s.n_anneal == 7

    def test_holds_then_decays_then_floors(self):
        s = HoldDecayFloorSchedule(
            init_temp=1e-2, min_temp=1e-3, hold_steps=2, decay_steps=3
        )
        values = [s.value]
        for _ in range(6):
            s.update()
            values.append(s.value)

        assert values[0] == pytest.approx(1e-2)
        assert values[1] == pytest.approx(1e-2)
        assert values[2] == pytest.approx(1e-2)
        assert values[3] < values[2]
        assert values[4] < values[3]
        assert values[5] == pytest.approx(1e-3, rel=1e-6)
        assert values[6] == pytest.approx(1e-3, rel=1e-6)

    def test_reset_and_state_roundtrip(self):
        s1 = HoldDecayFloorSchedule(
            init_temp=1e-2, min_temp=1e-3, hold_steps=1, decay_steps=4
        )
        for _ in range(3):
            s1.update()
        state = s1.get_state()

        s2 = HoldDecayFloorSchedule(
            init_temp=1e-2, min_temp=1e-3, hold_steps=1, decay_steps=4
        )
        s2.set_state(state)
        assert s2.value == pytest.approx(s1.value)
        assert s2.step == s1.step

        s2.reset()
        assert s2.value == pytest.approx(1e-2)
        assert s2.step == 0

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            HoldDecayFloorSchedule(init_temp=1e-3, min_temp=1e-2)
        with pytest.raises(ValueError):
            HoldDecayFloorSchedule(hold_steps=-1)
        with pytest.raises(ValueError):
            HoldDecayFloorSchedule(decay_steps=-1)


class TestStepwiseTemperatureSchedule:
    """Tests for piecewise-constant schedule."""

    def test_initial_value(self):
        s = StepwiseTemperatureSchedule(
            temperatures=(5e-3, 3e-3, 1e-3),
            stage_steps=(2, 3, 4),
        )
        assert s.value == pytest.approx(5e-3)
        assert s.step == 0
        assert s.n_anneal == 5
        assert s.stage_boundaries == (2, 5, 9)

    def test_stage_progression(self):
        s = StepwiseTemperatureSchedule(
            temperatures=(5e-3, 3e-3, 1e-3),
            stage_steps=(2, 2, 2),
        )
        values = [s.value]
        for _ in range(6):
            s.update()
            values.append(s.value)

        assert values[:3] == pytest.approx([5e-3, 5e-3, 3e-3])
        assert values[3:5] == pytest.approx([3e-3, 1e-3])
        assert values[5:] == pytest.approx([1e-3, 1e-3])

    def test_reset_and_state_roundtrip(self):
        s1 = StepwiseTemperatureSchedule(
            temperatures=(5e-3, 3e-3, 1e-3),
            stage_steps=(2, 3, 4),
        )
        for _ in range(5):
            s1.update()
        state = s1.get_state()

        s2 = StepwiseTemperatureSchedule(
            temperatures=(5e-3, 3e-3, 1e-3),
            stage_steps=(2, 3, 4),
        )
        s2.set_state(state)
        assert s2.value == pytest.approx(s1.value)
        assert s2.step == s1.step

        s2.reset()
        assert s2.value == pytest.approx(5e-3)
        assert s2.step == 0

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            StepwiseTemperatureSchedule(temperatures=(), stage_steps=())
        with pytest.raises(ValueError):
            StepwiseTemperatureSchedule(temperatures=(1e-3,), stage_steps=(1, 2))
        with pytest.raises(ValueError):
            StepwiseTemperatureSchedule(temperatures=(1e-3, -1e-4), stage_steps=(1, 1))
        with pytest.raises(ValueError):
            StepwiseTemperatureSchedule(temperatures=(1e-3, 1e-4), stage_steps=(1, 0))


class TestStepwiseBuilders:
    """Tests for geometric stepwise plan helpers."""

    def test_geometric_temperature_ladder_hits_endpoints(self):
        temps = geometric_temperature_ladder(5e-3, 1e-3, 6)
        assert temps[0] == pytest.approx(5e-3)
        assert temps[-1] == pytest.approx(1e-3)
        assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))

    def test_weighted_stage_steps_sum_and_backload(self):
        steps = weighted_stage_steps(210, 6, weight_mode="linear")
        assert sum(steps) == 210
        assert steps[0] < steps[-1]
        assert all(step >= 1 for step in steps)

    def test_weighted_stage_steps_quadratic_backloads_more_than_linear(self):
        linear = weighted_stage_steps(210, 6, weight_mode="linear")
        quadratic = weighted_stage_steps(210, 6, weight_mode="quadratic")
        assert sum(quadratic) == 210
        assert quadratic[-1] > linear[-1]
        assert quadratic[0] <= linear[0]

    def test_build_geometric_stepwise_plan(self):
        plan = build_geometric_stepwise_plan(
            start_temp=5e-3,
            floor_temp=1e-3,
            n_stages=6,
            total_steps=210,
            stage_weight_mode="linear",
        )
        temps = plan["temperatures"]
        steps = plan["stage_steps"]
        assert temps[0] == pytest.approx(5e-3)
        assert temps[-1] == pytest.approx(1e-3)
        assert len(temps) == len(steps) == 6
        assert sum(steps) == 210
        assert plan["stop_enable_step"] == sum(steps[:-1])
        assert plan["stage_boundaries"][-1] == 210

    def test_build_geometric_stepwise_plan_quadratic(self):
        plan = build_geometric_stepwise_plan(
            start_temp=5e-3,
            floor_temp=1e-3,
            n_stages=6,
            total_steps=210,
            stage_weight_mode="quadratic",
        )
        assert plan["stage_weight_mode"] == "quadratic"
        assert plan["stage_steps"][-1] > plan["stage_steps"][0]
