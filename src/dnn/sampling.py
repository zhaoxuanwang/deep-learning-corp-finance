"""
src/dnn/sampling.py

State sampling and replay buffer for training.
Implements mixture sampling and percentile oversampling near default boundary.

Reference: outline_v2.md lines 113-121
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import warnings


# =============================================================================
# SAMPLING BOUNDS (DNN-specific)
# =============================================================================

@dataclass
class SamplingBounds:
    """
    DNN-specific sampling bounds.
    
    Defines the box region for uniform state sampling.
    Units: k in levels, z in log (log_z_bounds).
    
    This is separate from DDP grid bounds which serve a different purpose.
    
    Attributes:
        k_bounds: Capital bounds in levels (k_min, k_max)
        b_bounds: Debt bounds in levels (b_min, b_max), must be >= 0
        log_z_bounds: Productivity bounds in log (log_z_min, log_z_max)
    
    Example:
        bounds = SamplingBounds(k_bounds=(0.5, 5.0), log_z_bounds=(-0.3, 0.3))
    """
    k_bounds: Tuple[float, float] = (0.1, 10.0)
    b_bounds: Tuple[float, float] = (0.0, 5.0)
    log_z_bounds: Tuple[float, float] = (-0.5, 0.5)
    
    def __post_init__(self):
        """Validate bounds."""
        k_min, k_max = self.k_bounds
        if k_min <= 0:
            raise ValueError(f"k_min must be > 0. Got {k_min}")
        if k_min >= k_max:
            raise ValueError(f"k_min must be < k_max. Got {self.k_bounds}")
        
        b_min, b_max = self.b_bounds
        if b_min < 0:
            raise ValueError(f"b_min must be >= 0 (borrowing only). Got {b_min}")
        if b_min >= b_max:
            raise ValueError(f"b_min must be < b_max. Got {self.b_bounds}")
        
        log_z_min, log_z_max = self.log_z_bounds
        if log_z_min >= log_z_max:
            raise ValueError(f"log_z_min must be < log_z_max. Got {self.log_z_bounds}")
    
    def validate_consistency(self, steady_state_k: float):
        """
        Warn if bounds seem inconsistent with steady state.
        
        Args:
            steady_state_k: Approximate steady-state capital
        """
        k_min, k_max = self.k_bounds
        if k_max < steady_state_k * 0.5:
            warnings.warn(
                f"SamplingBounds k_max={k_max} is much lower than "
                f"steady-state k≈{steady_state_k:.2f}"
            )
        if k_min > steady_state_k * 2:
            warnings.warn(
                f"SamplingBounds k_min={k_min} is much higher than "
                f"steady-state k≈{steady_state_k:.2f}"
            )
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """Convert to dict format for AdaptiveBounds."""
        return {
            "k": self.k_bounds,
            "b": self.b_bounds,
            "log_z": self.log_z_bounds,
        }

@dataclass
class AdaptiveBounds:
    """
    Manages adaptive sampling bounds with optional user override.
    
    Default behavior: Adaptive mode — bounds auto-computed from replay buffer.
    If user_bounds is set, uses those (fixed mode, ignores buffer).
    
    Safety guarantees:
        - Conservative margin (20% expansion)
        - Wide percentiles (2.5th/97.5th)
        - Minimum range enforcement (never shrinks below 50% of initial)
        - Warmup fallback (uses initial bounds until min_samples reached)
    
    Args:
        initial_bounds: Starting bounds, used during warmup or as minimum
        user_bounds: If set, disables adaptive mode and uses these (fixed mode)
        margin: Fraction to expand computed bounds (default 20%)
        min_samples: Minimum buffer size before adapting
        percentile_low: Lower percentile for bounds computation
        percentile_high: Upper percentile for bounds computation
        min_range_fraction: Minimum range as fraction of initial (safety floor)
        var_names: Variable names in order (for ReplayBuffer column mapping)
    
    Example:
        # Adaptive mode (default)
        bounds = AdaptiveBounds(initial_bounds={"k": (0.1, 10.0), "log_z": (-0.5, 0.5)})
        current, meta = bounds.get_bounds(buffer)  # meta["source"] == "adaptive"
        
        # Fixed mode (user override)
        bounds = AdaptiveBounds(
            initial_bounds={"k": (0.1, 10.0)},
            user_bounds={"k": (1.0, 5.0)}  # Override
        )
        current, meta = bounds.get_bounds(buffer)  # meta["source"] == "user"
    """
    initial_bounds: Dict[str, Tuple[float, float]]
    user_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    margin: float = 0.20
    min_samples: int = 100
    percentile_low: float = 2.5
    percentile_high: float = 97.5
    min_range_fraction: float = 0.5
    var_names: Tuple[str, ...] = field(default_factory=lambda: ("k", "log_z"))
    
    def get_bounds(
        self,
        buffer: Optional["ReplayBuffer"] = None
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
        """
        Get current sampling bounds.
        
        Args:
            buffer: ReplayBuffer for adaptive bounds computation
        
        Returns:
            bounds: Dict mapping variable names to (min, max)
            metadata: Dict with source info:
                - "source": "user" | "adaptive" | "initial"
                - "buffer_size": int (if adaptive)
        """
        # User override mode (fixed)
        if self.user_bounds is not None:
            return dict(self.user_bounds), {"source": "user"}
        
        # Warmup: not enough data yet
        if buffer is None or buffer.size < self.min_samples:
            return dict(self.initial_bounds), {
                "source": "initial",
                "buffer_size": buffer.size if buffer else 0
            }
        
        # Adaptive mode: compute from buffer
        computed_bounds = self._compute_from_buffer(buffer)
        
        return computed_bounds, {
            "source": "adaptive",
            "buffer_size": buffer.size
        }
    
    def _compute_from_buffer(
        self,
        buffer: "ReplayBuffer"
    ) -> Dict[str, Tuple[float, float]]:
        """Compute adaptive bounds from buffer statistics with safety margins."""
        data = buffer.get_all_states()
        result = {}
        
        for i, var_name in enumerate(self.var_names):
            if i >= data.shape[1]:
                # Fallback if buffer has fewer columns
                result[var_name] = self.initial_bounds.get(var_name, (0.0, 1.0))
                continue
            
            col = data[:, i]
            
            # Percentile-based bounds
            p_low = np.percentile(col, self.percentile_low)
            p_high = np.percentile(col, self.percentile_high)
            
            # Expand by margin
            range_ = p_high - p_low
            p_low -= self.margin * range_
            p_high += self.margin * range_
            
            # Safety: ensure minimum range relative to initial
            if var_name in self.initial_bounds:
                init_lo, init_hi = self.initial_bounds[var_name]
                init_range = init_hi - init_lo
                min_range = self.min_range_fraction * init_range
                
                computed_range = p_high - p_low
                if computed_range < min_range:
                    # Expand symmetrically to meet minimum
                    expand = (min_range - computed_range) / 2
                    p_low -= expand
                    p_high += expand
                
                # Also ensure we never go narrower than initial bounds
                p_low = min(p_low, init_lo)
                p_high = max(p_high, init_hi)
            
            result[var_name] = (float(p_low), float(p_high))
        
        return result


class ReplayBuffer:
    """
    Replay buffer for storing and sampling training states.
    
    Stores states from previous training batches for experience replay.
    Supports oversampling near the default boundary based on |V_tilde|.
    
    Args:
        capacity: Maximum number of states to store
        state_dim: Dimension of state vector (2 for basic, 3 for risky debt)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 3,
        seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self._rng = np.random.default_rng(seed)
        
        # Storage
        self._buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self._size = 0
        self._ptr = 0
        
        # Cached |V_tilde| values for oversampling (updated externally)
        self._abs_values: Optional[np.ndarray] = None
    
    @property
    def size(self) -> int:
        """Current number of states in buffer."""
        return self._size
    
    def add(self, states: np.ndarray):
        """
        Add states to buffer.
        
        Args:
            states: Array of shape (N, state_dim) with states to add
        """
        n = states.shape[0]
        
        if n >= self.capacity:
            # Replace entire buffer
            self._buffer[:] = states[-self.capacity:]
            self._size = self.capacity
            self._ptr = 0
        else:
            # Add with wraparound
            end_ptr = self._ptr + n
            if end_ptr <= self.capacity:
                self._buffer[self._ptr:end_ptr] = states
            else:
                # Wraparound
                first_part = self.capacity - self._ptr
                self._buffer[self._ptr:] = states[:first_part]
                self._buffer[:n - first_part] = states[first_part:]
            
            self._ptr = end_ptr % self.capacity
            self._size = min(self._size + n, self.capacity)
        
        # Invalidate cached values
        self._abs_values = None
    
    def sample(self, n: int) -> np.ndarray:
        """
        Sample n states uniformly from buffer.
        
        Args:
            n: Number of states to sample
        
        Returns:
            Array of shape (n, state_dim)
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        indices = self._rng.choice(self._size, size=n, replace=True)
        return self._buffer[indices]
    
    def update_values(self, abs_values: np.ndarray):
        """
        Update cached |V_tilde| values for oversampling.
        
        Should be called after computing V_tilde for all buffer states.
        
        Args:
            abs_values: Array of shape (size,) with |V_tilde| values
        """
        if len(abs_values) != self._size:
            raise ValueError(f"Expected {self._size} values, got {len(abs_values)}")
        self._abs_values = abs_values.copy()
    
    def sample_near_default(self, n: int, percentile_q: float) -> np.ndarray:
        """
        Sample states near the default boundary using percentile rule.
        
        Selects states in the bottom q percentile of |V_tilde|.
        
        Args:
            n: Number of states to sample
            percentile_q: Percentile threshold (e.g., 0.05 for bottom 5%)
        
        Returns:
            Array of shape (n, state_dim)
        
        Reference: outline_v2.md lines 117-120
        """
        if self._abs_values is None:
            raise ValueError("Must call update_values() before oversampling")
        
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Find threshold for bottom q percentile
        threshold = np.percentile(self._abs_values[:self._size], percentile_q * 100)
        
        # Get indices of states below threshold
        mask = self._abs_values[:self._size] <= threshold
        near_default_indices = np.where(mask)[0]
        
        if len(near_default_indices) == 0:
            # Fallback to uniform if no states meet criterion
            return self.sample(n)
        
        # Sample from near-default states
        sampled_indices = self._rng.choice(
            near_default_indices, size=n, replace=True
        )
        return self._buffer[sampled_indices]
    
    def get_all_states(self) -> np.ndarray:
        """Return all states currently in buffer."""
        return self._buffer[:self._size].copy()
    
    def clear(self):
        """Clear the buffer."""
        self._size = 0
        self._ptr = 0
        self._abs_values = None


# =============================================================================
# TRAINING CONTEXT (Ergodic Sampling)
# =============================================================================

@dataclass
class TrainingContext:
    """
    Manages state sampling with optional replay for ergodic training.
    
    Mixes fresh random samples with states from a replay buffer containing
    terminal states from previous rollouts. This helps the training process
    explore the policy-induced stationary distribution rather than just
    the initial prior.
    
    Args:
        scenario: EconomicScenario with sampling bounds
        buffer: ReplayBuffer for storing terminal states
        replay_ratio: Fraction of batch to sample from buffer (0.0 = all fresh)
        warmup_iters: Number of iterations before mixing (buffer needs data)
        state_keys: Names of state variables for unpacking (e.g., ["k", "z"])
        rng: NumPy random generator
    
    Usage:
        ctx = TrainingContext(scenario=scenario, state_keys=["k", "z"])
        for i in range(n_iter):
            states = ctx.sample(batch_size)  # Returns dict {"k": ..., "z": ...}
            metrics = trainer.train_step(**states)
            ctx.update(metrics["terminal_states"])  # Add terminal states to buffer
    """
    scenario: "EconomicScenario"
    buffer: Optional[ReplayBuffer] = None
    replay_ratio: float = 0.5
    warmup_iters: int = 100
    state_keys: Tuple[str, ...] = ("k", "z")
    rng: Optional[np.random.Generator] = None
    
    # Internal tracking
    _iteration: int = field(default=0, repr=False)
    
    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        
        # Create buffer if not provided
        if self.buffer is None:
            state_dim = len(self.state_keys)
            self.buffer = ReplayBuffer(capacity=10000, state_dim=state_dim)
    
    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """
        Sample n states, mixing fresh and replay.
        
        During warmup (first warmup_iters iterations), samples are all fresh.
        After warmup, mixes (1-replay_ratio) fresh with replay_ratio from buffer.
        
        Args:
            n: Batch size
        
        Returns:
            Dict mapping state_keys to arrays of shape (n, 1)
        """
        self._iteration += 1
        
        # Warmup: all fresh
        if self._iteration <= self.warmup_iters or self.buffer.size < n:
            return self._sample_fresh(n)
        
        # Mix fresh + replay
        n_replay = int(n * self.replay_ratio)
        n_fresh = n - n_replay
        
        fresh_states = self._sample_fresh_raw(n_fresh) if n_fresh > 0 else None
        replay_states = self.buffer.sample(n_replay) if n_replay > 0 else None
        
        # Concatenate
        if fresh_states is not None and replay_states is not None:
            combined = np.vstack([fresh_states, replay_states])
        elif fresh_states is not None:
            combined = fresh_states
        else:
            combined = replay_states
        
        # Shuffle to avoid batch structure
        self.rng.shuffle(combined)
        
        return self._unpack_states(combined)
    
    def update(self, terminal_states: np.ndarray):
        """
        Add terminal states from training step to buffer.
        
        Args:
            terminal_states: Array of shape (batch_size, state_dim)
        """
        if terminal_states is not None:
            self.buffer.add(terminal_states)
    
    def _sample_fresh(self, n: int) -> Dict[str, np.ndarray]:
        """Sample fresh states from scenario bounds."""
        raw = self._sample_fresh_raw(n)
        return self._unpack_states(raw)
    
    def _sample_fresh_raw(self, n: int) -> np.ndarray:
        """Sample fresh states as raw numpy array."""
        bounds = self.scenario.sampling
        
        if len(self.state_keys) == 2 and "k" in self.state_keys and "z" in self.state_keys:
            # Basic model: (k, z)
            k, z = sample_box_basic(
                n=n,
                k_bounds=bounds.k_bounds,
                log_z_bounds=bounds.log_z_bounds,
                rng=self.rng
            )
            return np.column_stack([k, z])
        
        elif len(self.state_keys) == 3 and "b" in self.state_keys:
            # Risky model: (k, b, z)
            k, b, z = sample_box_risky(
                n=n,
                k_bounds=bounds.k_bounds,
                b_bounds=bounds.b_bounds,
                log_z_bounds=bounds.log_z_bounds,
                rng=self.rng
            )
            return np.column_stack([k, b, z])
        
        else:
            raise ValueError(f"Unknown state_keys: {self.state_keys}")
    
    def _unpack_states(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert (n, state_dim) array to dict of (n, 1) arrays."""
        return {
            key: states[:, i:i+1] 
            for i, key in enumerate(self.state_keys)
        }
    
    @property
    def buffer_size(self) -> int:
        """Current number of states in buffer."""
        return self.buffer.size
    
    def reset(self):
        """Reset iteration counter and clear buffer."""
        self._iteration = 0
        self.buffer.clear()


def sample_box(
    n: int,
    bounds: Dict[str, Tuple[float, float]],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample states uniformly from a box region.
    
    For Basic model: bounds = {"k": (k_min, k_max), "log_z": (log_z_min, log_z_max)}
    For Risky Debt: bounds = {"k": ..., "b": ..., "log_z": ...}
    
    Args:
        n: Number of samples
        bounds: Dictionary mapping variable names to (min, max) bounds
        rng: NumPy random generator
    
    Returns:
        Array of shape (n, len(bounds))
    """
    samples = []
    for var_name in sorted(bounds.keys()):  # Sorted for consistent ordering
        lo, hi = bounds[var_name]
        samples.append(rng.uniform(lo, hi, size=n))
    
    return np.stack(samples, axis=1).astype(np.float32)


def sample_box_basic(
    n: int,
    k_bounds: Tuple[float, float],
    log_z_bounds: Tuple[float, float],
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (k, z) for Basic model.
    
    Args:
        n: Number of samples
        k_bounds: (k_min, k_max)
        log_z_bounds: Bounds for log(z) (sampled from ergodic distribution)
        rng: NumPy random generator
    
    Returns:
        k: Array of shape (n,)
        z: Array of shape (n,)
    """
    k = rng.uniform(k_bounds[0], k_bounds[1], size=n).astype(np.float32)
    log_z = rng.uniform(log_z_bounds[0], log_z_bounds[1], size=n).astype(np.float32)
    z = np.exp(log_z)
    return k, z


def sample_box_risky(
    n: int,
    k_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    log_z_bounds: Tuple[float, float],
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (k, b, z) for Risky Debt model.
    
    Note: b >= 0 enforced (borrowing-only).
    
    Args:
        n: Number of samples
        k_bounds: (k_min, k_max)
        b_bounds: (b_min, b_max), should be >= 0
        log_z_bounds: Bounds for log(z)
        rng: NumPy random generator
    
    Returns:
        k, b, z: Arrays of shape (n,) each
    """
    k = rng.uniform(k_bounds[0], k_bounds[1], size=n).astype(np.float32)
    b = rng.uniform(max(0, b_bounds[0]), b_bounds[1], size=n).astype(np.float32)
    log_z = rng.uniform(log_z_bounds[0], log_z_bounds[1], size=n).astype(np.float32)
    z = np.exp(log_z)
    return k, b, z


def sample_mixture(
    n: int,
    box_sampler,
    replay_buffer: ReplayBuffer,
    replay_ratio: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample from mixture of box and replay buffer.
    
    Args:
        n: Total number of samples
        box_sampler: Callable that returns (n,) or (n, dim) array of box samples
        replay_buffer: Replay buffer to sample from
        replay_ratio: Fraction of samples from replay (0 to 1)
        rng: NumPy random generator
    
    Returns:
        Mixed samples
    """
    if replay_buffer.size == 0 or replay_ratio <= 0:
        return box_sampler(n)
    
    n_replay = int(n * replay_ratio)
    n_box = n - n_replay
    
    box_samples = box_sampler(n_box) if n_box > 0 else np.array([])
    replay_samples = replay_buffer.sample(n_replay) if n_replay > 0 else np.array([])
    
    if n_box == 0:
        return replay_samples
    if n_replay == 0:
        return box_samples
    
    return np.concatenate([box_samples, replay_samples], axis=0)


def sample_with_oversampling(
    n: int,
    replay_buffer: ReplayBuffer,
    percentile_q: float,
    oversample_fraction: float,
    box_sampler,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample with oversampling near default boundary.
    
    Combines:
    1. Regular samples (box + replay)
    2. Extra samples from near-default region
    
    Args:
        n: Total number of samples
        replay_buffer: Replay buffer with updated |V_tilde| values
        percentile_q: Percentile for near-default threshold (e.g., 0.05)
        oversample_fraction: Fraction of samples from near-default region
        box_sampler: Regular sample generator
        rng: NumPy random generator
    
    Returns:
        Combined samples
    
    Reference: outline_v2.md lines 117-120
    """
    if replay_buffer.size == 0 or replay_buffer._abs_values is None:
        return box_sampler(n)
    
    n_oversample = int(n * oversample_fraction)
    n_regular = n - n_oversample
    
    regular_samples = box_sampler(n_regular)
    oversample_samples = replay_buffer.sample_near_default(n_oversample, percentile_q)
    
    return np.concatenate([regular_samples, oversample_samples], axis=0)


# =============================================================================
# AR(1) SHOCK UTILITIES (Canonical Implementation)
# =============================================================================

# Minimum z floor to avoid log(0)
MIN_Z = 1e-8


def step_ar1_tf(
    z: tf.Tensor,
    rho: float,
    sigma: float,
    mu: float = 0.0,
    eps: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Single-step AR(1) transition for productivity z (TensorFlow).
    
    This is the CANONICAL implementation for all TF-based training code.
    Trainers should import and use this instead of inline AR(1) logic.
    
    Process:
        log(z') = (1 - rho) * mu + rho * log(z) + sigma * eps
    
    Args:
        z: Current productivity (any shape)
        rho: AR(1) persistence
        sigma: AR(1) volatility
        mu: AR(1) unconditional mean of log(z)
        eps: Pre-drawn standard normal shocks (default: draws internally)
    
    Returns:
        z_next: Next period productivity (same shape as z)
    """
    log_z = tf.math.log(tf.maximum(z, MIN_Z))
    
    if eps is None:
        eps = tf.random.normal(tf.shape(z), dtype=tf.float32)
    
    log_z_next = (1 - rho) * mu + rho * log_z + sigma * eps
    return tf.exp(log_z_next)


def draw_shocks(
    n: int,
    z_curr: tf.Tensor,
    rho: float,
    sigma: float,
    mu: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Draw two independent next-period productivity shocks for AiO method.
    
    log(z') = (1 - rho) * mu + rho * log(z) + sigma * epsilon
    
    Args:
        n: Batch size
        z_curr: Current productivity (n,) or (n, 1)
        rho: AR(1) persistence
        sigma: AR(1) volatility
        mu: AR(1) mean
        rng: NumPy random generator (for reproducibility)
    
    Returns:
        z_next_1, z_next_2: Two independent z' draws
    """
    z_curr = tf.reshape(z_curr, [-1, 1])
    log_z = tf.math.log(tf.maximum(z_curr, 1e-8))
    
    # Draw two independent epsilon
    if rng is not None:
        eps1 = tf.constant(rng.standard_normal(size=(n, 1)), dtype=tf.float32)
        eps2 = tf.constant(rng.standard_normal(size=(n, 1)), dtype=tf.float32)
    else:
        eps1 = tf.random.normal((n, 1), dtype=tf.float32)
        eps2 = tf.random.normal((n, 1), dtype=tf.float32)
    
    # AR(1) update
    log_z_next_1 = (1 - rho) * mu + rho * log_z + sigma * eps1
    log_z_next_2 = (1 - rho) * mu + rho * log_z + sigma * eps2
    
    z_next_1 = tf.exp(log_z_next_1)
    z_next_2 = tf.exp(log_z_next_2)
    
    return z_next_1, z_next_2
