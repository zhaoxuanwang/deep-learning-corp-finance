"""
Annealing schedule for decaying temperature parameters during training.

Provides a unified schedule for controlling sigmoid sharpness in:
- Soft investment/finance gates: sigmoid(x / tau)
- Smooth default probability: sigmoid(-V / epsilon)

As tau → 0, soft sigmoids → hard indicator functions.
"""

from typing import Literal
from dataclasses import dataclass


ScheduleType = Literal["exponential", "linear"]


@dataclass
class AnnealingSchedule:
    """
    Decaying schedule for temperature-like parameters.
    
    Attributes:
        init: Initial value
        min: Floor value (never decays below this)
        decay: Multiplicative decay factor per step (exponential only)
        schedule: "exponential" or "linear"
        total_steps: Steps to reach min (linear only)    
        
    Schedules:
        exponential: value_t = max(min, init * decay^t)
        linear:      value_t = max(min, init - (init - min) * t / total_steps)
    """
    init_temp: float = 1.0
    min: float = 1e-4
    decay: float = 0.9
    schedule: ScheduleType = "exponential"
    total_steps: int = 1000
    
    def __post_init__(self):
        self._value: float = self.init_temp
        self._step: int = 0
    
    @property
    def value(self) -> float:
        """Current temperature value (clamped to min)."""
        return max(self._value, self.min)
    
    @property
    def step(self) -> int:
        """Number of update() calls so far."""
        return self._step
    
    def reset(self):
        """Reset to initial state."""
        self._value = self.init_temp
        self._step = 0
    
    def update(self):
        """Decay value by one step. Call once per training iteration."""
        self._step += 1
        
        if self.schedule == "exponential":
            self._value = self.decay * self._value
        elif self.schedule == "linear":
            progress = min(self._step / self.total_steps, 1.0)
            self._value = self.init_temp - (self.init_temp - self.min) * progress
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        self._value = max(self._value, self.min)
    
    def get_state(self) -> dict:
        """Return state dict for checkpointing."""
        return {"value": self._value, "step": self._step}
    
    def set_state(self, state: dict):
        """Restore state from checkpoint."""
        self._value = state["value"]
        self._step = state["step"]
    
    def __repr__(self) -> str:
        return (
            f"AnnealingSchedule(init={self.init_temp}, min={self.min}, "
            f"value={self.value:.6f}, step={self._step}, schedule={self.schedule})"
        )


# =============================================================================
# GATE & INDICATOR FUNCTIONS
# =============================================================================

from typing import Union, Any
import tensorflow as tf

Numeric = Union[tf.Tensor, Any]

def _resolve_temp(arg: Union[float, AnnealingSchedule]) -> float:
    """Helper: extract float value from float or Schedule."""
    if isinstance(arg, AnnealingSchedule):
        return arg.value
    return float(arg)


def smooth_default_prob(
    V_tilde: Numeric, 
    temperature: Union[float, AnnealingSchedule], 
    logit_clip: float = 20.0
):
    """
    Compute smooth default probability.
    
    p = sigmoid(clip(-V / epsilon, -logit_clip, logit_clip))
    
    Args:
        V_tilde: Latent value tensor
        temperature: Temperature float or AnnealingSchedule
        logit_clip: Logit clipping bound (default 20.0)
    """
    V_tilde = tf.convert_to_tensor(V_tilde)
    dtype = V_tilde.dtype
    temp_val = _resolve_temp(temperature)
    temp_t = tf.maximum(tf.cast(temp_val, dtype), 1e-6)
    u = -V_tilde / temp_t
    u_clipped = tf.clip_by_value(u, -logit_clip, logit_clip)
    return tf.nn.sigmoid(u_clipped)


def indicator_abs_gt(
    x: Numeric,
    threshold: float = 1e-8,
    temperature: Union[float, AnnealingSchedule] = 0.1,
    logit_clip: float = 20.0,
    mode: Literal["hard", "ste", "soft"] = "soft"
) -> tf.Tensor:
    """
    Gate for |x| > threshold with soften indicator + annealing schedule.
    
    Args:
        x: Input tensor
        threshold: Threshold
        temperature: Hardness control (float or AnnealingSchedule)
        logit_clip: Logit clipping bound (default 20.0)
        mode: "hard", "ste", or "soft"
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    threshold_t = tf.cast(threshold, dtype)
    
    temp_val = _resolve_temp(temperature)
    temp_t = tf.maximum(tf.cast(temp_val, dtype), 1e-6)
    
    abs_x = tf.abs(x)
    g_hard = tf.cast(abs_x > threshold_t, dtype)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate logic
    logit = (abs_x - threshold_t) / temp_t
    logit = tf.clip_by_value(logit, -logit_clip, logit_clip)
    g_soft = tf.nn.sigmoid(logit)
    
    if mode == "soft":
        return g_soft
    
    # STE
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def indicator_lt(
    x: Numeric,
    threshold: float = 1e-8,
    temperature: Union[float, AnnealingSchedule] = 0.1,
    logit_clip: float = 20.0,
    mode: Literal["hard", "ste", "soft"] = "soft"
) -> tf.Tensor:
    """
    Gate for x < - threshold with soften indicator + annealing schedule.
    
    Args:
        x: Input tensor
        threshold: Comparison threshold
        temperature: Hardness control (float or AnnealingSchedule)
        logit_clip: Logit clipping bound (default 20.0)
        mode: "hard", "ste", or "soft"
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    threshold_t = tf.cast(threshold, dtype)
    
    temp_val = _resolve_temp(temperature)
    temp_t = tf.maximum(tf.cast(temp_val, dtype), 1e-8)
    
    g_hard = tf.cast(x < - threshold_t, dtype)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate logic
    logit = -(x + threshold_t) / temp_t
    logit = tf.clip_by_value(logit, -logit_clip, logit_clip)
    g_soft = tf.nn.sigmoid(logit)
    
    if mode == "soft":
        return g_soft
    
    # STE
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def hard_gate_abs_gt(x: Numeric,threshold: float = 1e-6) -> tf.Tensor:
    return indicator_abs_gt(x, threshold=threshold, mode="hard")


def hard_gate_lt(x: Numeric,threshold: float = 1e-6) -> tf.Tensor:
    return indicator_lt(x, threshold=threshold, mode="hard")
