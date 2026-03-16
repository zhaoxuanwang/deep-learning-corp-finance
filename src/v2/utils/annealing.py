"""Temperature annealing schedule for smooth indicator gates.

Implements multiplicative decay: tau[step] = max(min_temp, init_temp * decay^step).

As tau decays toward zero, soft sigmoid gates recover hard indicator
behaviour.  The schedule is typically stepped once per training iteration.

Usage::

    schedule = AnnealingSchedule(init_temp=1.0, min_temp=1e-6, decay_rate=0.995)
    for step in range(n_steps):
        temperature = schedule.value
        # ... training step using temperature ...
        schedule.update()
"""

import math
from dataclasses import dataclass


# Self-contained defaults (no external imports).
DEFAULT_INIT_TEMP = 1.0
DEFAULT_MIN_TEMP = 1e-6
DEFAULT_DECAY_RATE = 0.995
DEFAULT_BUFFER = 0.25


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
