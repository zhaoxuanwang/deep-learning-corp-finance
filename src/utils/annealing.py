"""
Annealing schedule for decaying temperature parameters during training.

Provides a unified schedule for controlling sigmoid sharpness in:
- Soft investment/finance gates: sigmoid(x / tau)
- Smooth default probability: sigmoid(-V / epsilon)

As tau → 0, soft sigmoids → hard indicator functions.
"""

import math
from typing import Literal, Union, Any
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class AnnealingSchedule:
    """
    Multiplicative annealing schedule for temperature parameters.
    
    Implements: tau[step] = max(min_temp, init_temp * decay_rate^step)
    
    Attributes:
        init_temp: Initial temperature 
        min_temp: Minimum temperature floor 
        decay_rate: Multiplicative decay factor per step 
        buffer: Stabilization buffer fraction (default 0.25)
    
    Properties:
        value: Current temperature
        step: Current iteration
        n_anneal: Projected iterations to reach convergence (including buffer)
    """
    init_temp: float = 1.0
    min_temp: float = 1e-4
    decay_rate: float = 0.9
    buffer: float = 0.25
    
    def __post_init__(self):
        self._value: float = self.init_temp
        self._step: int = 0
        self._n_anneal: int = self._compute_n_anneal()
    
    def _compute_n_anneal(self) -> int:
        """Compute expected annealing steps to reach min_temp plus buffer."""
        if self.init_temp <= self.min_temp or self.decay_rate >= 1.0 or self.decay_rate <= 0.0:
            return 0
            
        n_decay = (math.log(self.min_temp) - math.log(self.init_temp)) / math.log(self.decay_rate)
        return math.ceil(n_decay * (1.0 + self.buffer))
    
    @property
    def value(self) -> float:
        """Current temperature value (clamped to min_temp)."""
        return max(self._value, self.min_temp)
    
    @property
    def step(self) -> int:
        """Number of update() calls so far."""
        return self._step

    @property
    def n_anneal(self) -> int:
        """Expected number of steps to complete annealing schedule."""
        return self._n_anneal
    
    def reset(self):
        """Reset to initial state."""
        self._value = self.init_temp
        self._step = 0
    
    def update(self):
        """Decay value by one step. Call once per training iteration."""
        self._step += 1
        self._value = self.init_temp * (self.decay_rate ** self._step)
        # Implicitly clamped by value property
    
    def get_state(self) -> dict:
        """Return state dict for checkpointing."""
        return {"value": self._value, "step": self._step}
    
    def set_state(self, state: dict):
        """Restore state from checkpoint."""
        self._value = state["value"]
        self._step = state["step"]
    
    def __repr__(self) -> str:
        return (
            f"AnnealingSchedule(init={self.init_temp}, min={self.min_temp}, "
            f"decay={self.decay_rate}, buffer={self.buffer}, "
            f"value={self.value:.6f}, step={self._step}, N_anneal={self.n_anneal})"
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


def indicator_default(
    V_tilde_norm: Numeric, 
    temperature: Union[float, AnnealingSchedule], 
    logit_clip: float = 20.0,
    noise: bool = True
):
    """
    Compute smooth default probability using Gumbel-Sigmoid for exploration.
    
    Formula:
        sigma( (-V_tilde/k + log(u) - log(1-u)) / tau )
        
    Args:
        V_tilde_norm: Normalized latent value (V_tilde / k)
        temperature: Temperature parameter (tau)
        logit_clip: Clip range for the normalized value (default 20.0)
        noise: If True, add Gumbel noise. If False, use plain Sigmoid.
    """
    x = tf.convert_to_tensor(V_tilde_norm)
    dtype = x.dtype
    temp_val = _resolve_temp(temperature)
    temp_t = tf.maximum(tf.cast(temp_val, dtype), 1e-6)
    
    # 2. Clip the norm value between [-logit_clip, logit_clip]
    x_clipped = tf.clip_by_value(x, -logit_clip, logit_clip)
    
    if noise:
        # 3. Draw random noise u ~ Uniform(0,1)
        # 4. Gumbel noise term: log(u) - log(1-u)
        u = tf.random.uniform(tf.shape(x), minval=1e-6, maxval=1.0-1e-6, dtype=dtype)
        gumbel_noise = tf.math.log(u) - tf.math.log(1.0 - u)
        
        # 5. Compute Gumbel-Sigmoid
        # logit = (-V/k + gumbel) / tau
        logit = (-x_clipped + gumbel_noise) / temp_t
    else:
        # Fallback to deterministic sigmoid: sigma(-V/k / tau)
        logit = -x_clipped / temp_t
        
    return tf.nn.sigmoid(logit)


def indicator_abs_gt(
    x: Numeric,
    threshold: float = 1e-4,
    temperature: Union[float, AnnealingSchedule] = 0.1,
    logit_clip: float = 20.0,
    mode: Literal["hard", "ste", "soft"] = "soft"
) -> tf.Tensor:
    """
    Gate for |x| > threshold (epsilon).
    
    Soft approximation: sigma( (|x| - epsilon) / tau )
    
    Args:
        x: Input tensor (e.g. I/k)
        threshold: Epsilon tolerance (epsilon)
        temperature: Annealing temperature (tau)
        logit_clip: Logit clipping bound
        mode: "hard" (true indicator), "soft" (sigmoid), "ste" (straight-through)
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    eps = tf.cast(threshold, dtype)
    
    temp_val = _resolve_temp(temperature)
    temp_t = tf.maximum(tf.cast(temp_val, dtype), 1e-6)
    
    abs_x = tf.abs(x)
    g_hard = tf.cast(abs_x > eps, dtype)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate logic
    # logit = (|x| - epsilon) / tau
    logit = (abs_x - eps) / temp_t
    logit = tf.clip_by_value(logit, -logit_clip, logit_clip)
    g_soft = tf.nn.sigmoid(logit)
    
    if mode == "soft":
        return g_soft
    
    # STE
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def indicator_lt(
    x: Numeric,
    threshold: float = 1e-4,
    temperature: Union[float, AnnealingSchedule] = 0.1,
    logit_clip: float = 20.0,
    mode: Literal["hard", "ste", "soft"] = "soft"
) -> tf.Tensor:
    """
    Gate for x < -threshold.
    
    Soft approximation: sigma( -(x + epsilon) / tau )
    
    Args:
        x: Input tensor (e.g. e/k)
        threshold: Epsilon tolerance (epsilon)
        temperature: Annealing temperature (tau)
        logit_clip: Logit clipping bound
        mode: "hard" (true indicator), "soft" (sigmoid), "ste" (straight-through)
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    eps = tf.cast(threshold, dtype)
    
    temp_val = _resolve_temp(temperature)
    temp_t = tf.maximum(tf.cast(temp_val, dtype), 1e-6)
    
    g_hard = tf.cast(x < -eps, dtype)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate logic
    # logit = -(x + epsilon) / tau
    logit = -(x + eps) / temp_t
    logit = tf.clip_by_value(logit, -logit_clip, logit_clip)
    g_soft = tf.nn.sigmoid(logit)
    
    if mode == "soft":
        return g_soft
    
    # STE
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def hard_gate_abs_gt(x: Numeric, threshold: float = 1e-6) -> tf.Tensor:
    return indicator_abs_gt(x, threshold=threshold, mode="hard")


def hard_gate_lt(x: Numeric, threshold: float = 1e-6) -> tf.Tensor:
    return indicator_lt(x, threshold=threshold, mode="hard")
