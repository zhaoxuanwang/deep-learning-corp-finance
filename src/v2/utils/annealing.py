"""Temperature annealing schedules for smooth indicator gates.

Three generic schedules are provided:

- ``AnnealingSchedule`` for simple multiplicative decay
- ``HoldDecayFloorSchedule`` for continuation-style training where the
  temperature is held high early, decayed through a middle phase, and
  then held fixed at a floor
- ``StepwiseTemperatureSchedule`` for piecewise-constant stage schedules
- helper builders for geometric temperature ladders and weighted stage spans

These schedules are stepped once per training iteration and expose the
same small interface used by the trainers: ``value``, ``step``,
``n_anneal``, ``update()``, ``reset()``, ``get_state()``, and
``set_state()``.
"""

import math
from dataclasses import dataclass


# Self-contained defaults (no external imports).
DEFAULT_INIT_TEMP = 1.0
DEFAULT_MIN_TEMP = 1e-6
DEFAULT_DECAY_RATE = 0.995
DEFAULT_BUFFER = 0.25


def geometric_temperature_ladder(start_temp: float,
                                 floor_temp: float,
                                 n_stages: int) -> tuple[float, ...]:
    """Build a monotonically decreasing geometric ladder.

    The first temperature equals ``start_temp`` and the final temperature
    equals ``floor_temp`` exactly. Intermediate stages are spaced evenly in
    log-temperature space.
    """
    if start_temp <= 0 or floor_temp <= 0:
        raise ValueError(
            "start_temp and floor_temp must be positive. "
            f"Got start_temp={start_temp}, floor_temp={floor_temp}"
        )
    if start_temp < floor_temp:
        raise ValueError(
            "start_temp must be >= floor_temp. "
            f"Got start_temp={start_temp}, floor_temp={floor_temp}"
        )
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1. Got {n_stages}")
    if n_stages == 1:
        return (float(floor_temp),)
    if start_temp == floor_temp:
        return tuple(float(start_temp) for _ in range(n_stages))

    ratio = (floor_temp / start_temp) ** (1.0 / (n_stages - 1))
    temps = [float(start_temp * (ratio ** idx)) for idx in range(n_stages)]
    temps[-1] = float(floor_temp)
    return tuple(temps)


def weighted_stage_steps(total_steps: int,
                         n_stages: int,
                         weight_mode: str = "linear") -> tuple[int, ...]:
    """Allocate stage lengths using a simple deterministic weighting rule."""
    if total_steps < n_stages:
        raise ValueError(
            "total_steps must be >= n_stages so each stage gets at least one "
            f"step. Got total_steps={total_steps}, n_stages={n_stages}"
        )
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1. Got {n_stages}")

    if weight_mode == "linear":
        weights = [float(idx) for idx in range(1, n_stages + 1)]
    elif weight_mode == "quadratic":
        weights = [float(idx ** 2) for idx in range(1, n_stages + 1)]
    elif weight_mode == "uniform":
        weights = [1.0] * n_stages
    else:
        raise ValueError(
            "Unsupported weight_mode. "
            f"Got {weight_mode!r}; expected 'linear', 'quadratic', or "
            "'uniform'."
        )

    weight_sum = sum(weights)
    raw = [total_steps * w / weight_sum for w in weights]
    stage_steps = [1] * n_stages
    remaining = total_steps - n_stages
    fractional_order = sorted(
        range(n_stages),
        key=lambda idx: (raw[idx] - math.floor(raw[idx]), weights[idx], -idx),
        reverse=True,
    )
    floor_add = [max(int(math.floor(val)) - 1, 0) for val in raw]
    used = sum(floor_add)
    for idx, extra in enumerate(floor_add):
        stage_steps[idx] += extra
    remaining -= used
    for idx in fractional_order[:remaining]:
        stage_steps[idx] += 1

    return tuple(stage_steps)


def build_geometric_stepwise_plan(start_temp: float,
                                  floor_temp: float,
                                  n_stages: int,
                                  total_steps: int,
                                  stage_weight_mode: str = "linear") -> dict:
    """Build a complete plan compatible with StepwiseTemperatureSchedule."""
    temperatures = geometric_temperature_ladder(
        start_temp=start_temp,
        floor_temp=floor_temp,
        n_stages=n_stages,
    )
    stage_steps = weighted_stage_steps(
        total_steps=total_steps,
        n_stages=n_stages,
        weight_mode=stage_weight_mode,
    )
    schedule_kwargs = {
        "temperatures": temperatures,
        "stage_steps": stage_steps,
    }
    preview = StepwiseTemperatureSchedule(**schedule_kwargs)
    return {
        "schedule_kwargs": schedule_kwargs,
        "temperatures": temperatures,
        "stage_steps": stage_steps,
        "stage_weight_mode": stage_weight_mode,
        "stop_enable_step": preview.n_anneal,
        "stage_boundaries": preview.stage_boundaries,
    }


@dataclass
class AnnealingSchedule:
    """Multiplicative annealing schedule for temperature parameters.

    Attributes:
        init_temp:  Starting temperature.
        min_temp:   Floor — temperature never drops below this.
        decay_rate: Per-step multiplicative factor (< 1 to decay).
        buffer:     Extra fraction added to n_anneal estimate for safety.
    """

    init_temp: float = DEFAULT_INIT_TEMP
    min_temp: float = DEFAULT_MIN_TEMP
    decay_rate: float = DEFAULT_DECAY_RATE
    buffer: float = DEFAULT_BUFFER

    def __post_init__(self):
        self._value: float = self.init_temp
        self._step: int = 0
        self._n_anneal: int = self._compute_n_anneal()

    def _compute_n_anneal(self) -> int:
        """Projected steps to reach min_temp (including buffer)."""
        if (self.init_temp <= self.min_temp
                or self.decay_rate >= 1.0
                or self.decay_rate <= 0.0):
            return 0
        n_decay = ((math.log(self.min_temp) - math.log(self.init_temp))
                   / math.log(self.decay_rate))
        return math.ceil(n_decay * (1.0 + self.buffer))

    @property
    def value(self) -> float:
        """Current temperature (clamped to min_temp)."""
        return max(self._value, self.min_temp)

    @property
    def step(self) -> int:
        """Number of update() calls so far."""
        return self._step

    @property
    def n_anneal(self) -> int:
        """Estimated steps to complete the schedule."""
        return self._n_anneal

    def reset(self):
        """Reset to initial state."""
        self._value = self.init_temp
        self._step = 0

    def update(self):
        """Decay by one step.  Call once per training iteration."""
        self._step += 1
        self._value = self.init_temp * (self.decay_rate ** self._step)

    # -- Checkpointing helpers --

    def get_state(self) -> dict:
        """Serializable state for checkpointing."""
        return {"value": self._value, "step": self._step}

    def set_state(self, state: dict):
        """Restore from a checkpoint dict."""
        self._value = state["value"]
        self._step = state["step"]

    def __repr__(self) -> str:
        return (
            f"AnnealingSchedule(init={self.init_temp}, min={self.min_temp}, "
            f"decay={self.decay_rate}, value={self.value:.6f}, "
            f"step={self._step}, n_anneal={self.n_anneal})"
        )


@dataclass
class HoldDecayFloorSchedule:
    """Piecewise schedule: hold high, decay, then hold at floor.

    The schedule is:

    - ``init_temp`` for the first ``hold_steps`` updates
    - exponential decay from ``init_temp`` to ``min_temp`` over
      ``decay_steps`` updates
    - ``min_temp`` thereafter
    """

    init_temp: float = DEFAULT_INIT_TEMP
    min_temp: float = 1e-3
    hold_steps: int = 0
    decay_steps: int = 1

    def __post_init__(self):
        if self.init_temp <= 0 or self.min_temp <= 0:
            raise ValueError(
                "init_temp and min_temp must be positive. "
                f"Got init_temp={self.init_temp}, min_temp={self.min_temp}"
            )
        if self.init_temp < self.min_temp:
            raise ValueError(
                "init_temp must be >= min_temp. "
                f"Got init_temp={self.init_temp}, min_temp={self.min_temp}"
            )
        if self.hold_steps < 0:
            raise ValueError(f"hold_steps must be >= 0. Got {self.hold_steps}")
        if self.decay_steps < 0:
            raise ValueError(
                f"decay_steps must be >= 0. Got {self.decay_steps}"
            )

        self._step: int = 0
        self._value: float = self.init_temp
        self._n_anneal: int = self.hold_steps + self.decay_steps
        self._decay_rate: float = self._compute_decay_rate()

    def _compute_decay_rate(self) -> float:
        if self.init_temp == self.min_temp or self.decay_steps == 0:
            return 1.0
        return (self.min_temp / self.init_temp) ** (1.0 / self.decay_steps)

    @property
    def value(self) -> float:
        """Current temperature."""
        return self._value

    @property
    def step(self) -> int:
        """Number of update() calls so far."""
        return self._step

    @property
    def n_anneal(self) -> int:
        """Steps after which the floor is first reached."""
        return self._n_anneal

    def _temperature_at_step(self, step: int) -> float:
        if step < self.hold_steps:
            return self.init_temp
        if step >= self.hold_steps + self.decay_steps:
            return self.min_temp
        decay_step = step - self.hold_steps
        return self.init_temp * (self._decay_rate ** decay_step)

    def reset(self):
        """Reset to initial state."""
        self._step = 0
        self._value = self.init_temp

    def update(self):
        """Advance by one step."""
        self._step += 1
        self._value = self._temperature_at_step(self._step)

    def get_state(self) -> dict:
        """Serializable state for checkpointing."""
        return {"value": self._value, "step": self._step}

    def set_state(self, state: dict):
        """Restore from a checkpoint dict."""
        self._value = state["value"]
        self._step = state["step"]

    def __repr__(self) -> str:
        return (
            "HoldDecayFloorSchedule("
            f"init={self.init_temp}, min={self.min_temp}, "
            f"hold_steps={self.hold_steps}, decay_steps={self.decay_steps}, "
            f"value={self.value:.6f}, step={self._step}, "
            f"n_anneal={self.n_anneal})"
        )


@dataclass
class StepwiseTemperatureSchedule:
    """Piecewise-constant temperature schedule.

    Attributes:
        temperatures: Stage temperatures in chronological order.
        stage_steps:  Number of updates spent in each stage.
    """

    temperatures: tuple[float, ...] = (1e-2, 1e-3)
    stage_steps: tuple[int, ...] = (10, 10)

    def __post_init__(self):
        self.temperatures = tuple(float(t) for t in self.temperatures)
        self.stage_steps = tuple(int(s) for s in self.stage_steps)
        if not self.temperatures:
            raise ValueError("temperatures must be non-empty.")
        if len(self.temperatures) != len(self.stage_steps):
            raise ValueError(
                "temperatures and stage_steps must have the same length. "
                f"Got {len(self.temperatures)} vs {len(self.stage_steps)}."
            )
        if any(t <= 0 for t in self.temperatures):
            raise ValueError(
                f"All temperatures must be positive. Got {self.temperatures}"
            )
        if any(s < 1 for s in self.stage_steps):
            raise ValueError(
                f"All stage_steps must be >= 1. Got {self.stage_steps}"
            )

        self._step: int = 0
        self._value: float = self.temperatures[0]
        self._boundaries: tuple[int, ...] = self._compute_boundaries()
        self._n_anneal: int = (
            0 if len(self.stage_steps) == 1 else sum(self.stage_steps[:-1])
        )

    def _compute_boundaries(self) -> tuple[int, ...]:
        total = 0
        bounds = []
        for span in self.stage_steps:
            total += span
            bounds.append(total)
        return tuple(bounds)

    @property
    def value(self) -> float:
        """Current temperature."""
        return self._value

    @property
    def step(self) -> int:
        """Number of update() calls so far."""
        return self._step

    @property
    def n_anneal(self) -> int:
        """First step index at which the final stage begins."""
        return self._n_anneal

    @property
    def stage_boundaries(self) -> tuple[int, ...]:
        """Cumulative step counts at which each stage ends."""
        return self._boundaries

    def _temperature_at_step(self, step: int) -> float:
        for temp, bound in zip(self.temperatures, self._boundaries):
            if step < bound:
                return temp
        return self.temperatures[-1]

    def reset(self):
        """Reset to initial state."""
        self._step = 0
        self._value = self.temperatures[0]

    def update(self):
        """Advance by one step."""
        self._step += 1
        self._value = self._temperature_at_step(self._step)

    def get_state(self) -> dict:
        """Serializable state for checkpointing."""
        return {"value": self._value, "step": self._step}

    def set_state(self, state: dict):
        """Restore from a checkpoint dict."""
        self._value = state["value"]
        self._step = state["step"]

    def __repr__(self) -> str:
        return (
            "StepwiseTemperatureSchedule("
            f"temperatures={self.temperatures}, "
            f"stage_steps={self.stage_steps}, "
            f"value={self.value:.6f}, step={self._step}, "
            f"n_anneal={self.n_anneal})"
        )
