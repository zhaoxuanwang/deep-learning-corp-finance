"""
src/economy/rng.py

Random number generation and seed management for reproducible experiments.

This module centralizes RNG seed scheduling for dataset generation using TensorFlow
stateless random functions (tf.random.stateless_uniform, tf.random.stateless_normal).

Key features:
- Deterministic seed generation from master seed pair (m0, m1)
- Disjoint RNG streams for train/validation/test splits
- Common random numbers across different methods for same split/step/variable
- TensorFlow-compatible seeds (tf.int32, shape [2])
- No global state or call-order dependencies

Seed Schedule:
--------------
Training:   s_train(j, x) = (m0 + 100 + VarID(x), m1 + j) for step j=1..J
Validation: s_val(x)      = (m0 + 200 + VarID(x), m1 + 0)
Test:       s_test(x)     = (m0 + 300 + VarID(x), m1 + 0)

Where VarID maps variables to permanent IDs: k0=1, z0=2, b0=3, eps1=4, eps2=5

Example:
--------
    from src.economy import SeedSchedule, SeedScheduleConfig, VariableID

    # Create seed schedule
    config = SeedScheduleConfig(master_seed=(42, 0), n_train_steps=100)
    schedule = SeedSchedule(config)

    # Get training seeds for step 1
    train_seeds = schedule.get_train_seeds(steps=[1])
    k_seed = train_seeds[1][VariableID.K0]  # tf.Tensor([143, 1], dtype=tf.int32)

    # Use with TensorFlow stateless RNG
    k_init = tf.random.stateless_uniform(
        shape=(batch_size,),
        seed=k_seed,
        minval=k_min,
        maxval=k_max
    )

    # Get validation seeds
    val_seeds = schedule.get_val_seeds()
    eps_seed = val_seeds[VariableID.EPS1]

    eps = tf.random.stateless_normal(shape=(n, 1), seed=eps_seed)
"""

from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Literal, Optional, Tuple


# =============================================================================
# VARIABLE ID MAPPING
# =============================================================================

class VariableID(IntEnum):
    """
    Permanent mapping of variables to integer IDs for seed generation.

    These IDs are STABLE and NEVER change. New variables must be added
    at the end to maintain backward compatibility and reproducibility.

    Variables:
        K0: Initial capital
        Z0: Initial productivity
        B0: Initial debt (risky model)
        EPS1: First shock realization
        EPS2: Second shock realization (for AiO methods)
    """
    K0 = 1      # Initial capital
    Z0 = 2      # Initial productivity
    B0 = 3      # Initial debt
    EPS1 = 4    # First shock draw
    EPS2 = 5    # Second shock draw


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _is_valid_int32(value: int) -> bool:
    """
    Check if value fits in int32 range.

    Args:
        value: Integer to check

    Returns:
        True if value is in [-2^31, 2^31 - 1], False otherwise
    """
    INT32_MIN = -(2**31)
    INT32_MAX = 2**31 - 1
    return INT32_MIN <= value <= INT32_MAX


def _wrap_to_int32(value: int) -> int:
    """
    Wrap integer to valid int32 range using modulo arithmetic.

    This ensures deterministic behavior when values exceed int32 bounds.
    Uses two's complement representation consistent with TensorFlow.

    Args:
        value: Integer value (may be outside int32 range)

    Returns:
        Wrapped value in int32 range [-2^31, 2^31 - 1]

    Example:
        >>> _wrap_to_int32(2**31)      # Overflow
        -2147483648
        >>> _wrap_to_int32(-(2**31) - 1)  # Underflow
        2147483647
    """
    INT32_RANGE = 2**32
    INT32_OFFSET = 2**31

    # Map to unsigned 32-bit range [0, 2^32), then to signed [-2^31, 2^31)
    wrapped = value % INT32_RANGE
    if wrapped >= INT32_OFFSET:
        wrapped -= INT32_RANGE

    return wrapped


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class SeedScheduleConfig:
    """
    Configuration for stateless RNG seed generation.

    Attributes:
        master_seed: Tuple of two int32 values (m0, m1) that seed all RNG streams
        train_offset: Offset for training seeds (default 100)
        val_offset: Offset for validation seeds (default 200)
        test_offset: Offset for test seeds (default 300)
        n_train_steps: Number of training steps for validation (optional)

    Example:
        >>> config = SeedScheduleConfig(master_seed=(42, 0))
        >>> config = SeedScheduleConfig(
        ...     master_seed=(42, 100),
        ...     train_offset=1000,
        ...     val_offset=2000,
        ...     test_offset=3000,
        ...     n_train_steps=10000
        ... )
    """
    master_seed: Tuple[int, int]
    train_offset: int = 100
    val_offset: int = 200
    test_offset: int = 300
    n_train_steps: Optional[int] = None

    def __post_init__(self):
        """Validate configuration at construction time."""
        # Validate master seed components are valid int32
        m0, m1 = self.master_seed
        if not _is_valid_int32(m0):
            raise ValueError(
                f"master_seed[0] must be valid int32 in [-2^31, 2^31-1]. Got {m0}"
            )
        if not _is_valid_int32(m1):
            raise ValueError(
                f"master_seed[1] must be valid int32 in [-2^31, 2^31-1]. Got {m1}"
            )

        # Validate offsets are distinct
        offsets = [self.train_offset, self.val_offset, self.test_offset]
        if len(set(offsets)) != 3:
            raise ValueError(
                f"Split offsets must be distinct. Got train={self.train_offset}, "
                f"val={self.val_offset}, test={self.test_offset}"
            )

        # Validate offsets don't cause overflow with max variable ID
        max_var_id = max(v.value for v in VariableID)
        for offset_name, offset in [("train", self.train_offset),
                                      ("val", self.val_offset),
                                      ("test", self.test_offset)]:
            seed_component = m0 + offset + max_var_id
            if not _is_valid_int32(seed_component):
                raise ValueError(
                    f"{offset_name}_offset={offset} causes int32 overflow with "
                    f"master_seed[0]={m0} and max variable ID={max_var_id}. "
                    f"Computed: {seed_component}"
                )

        # Validate n_train_steps if provided
        if self.n_train_steps is not None:
            if self.n_train_steps <= 0:
                raise ValueError(
                    f"n_train_steps must be > 0. Got {self.n_train_steps}"
                )


# =============================================================================
# SEED SCHEDULE
# =============================================================================

class SeedSchedule:
    """
    Generates deterministic TensorFlow seeds for stateless RNG operations.

    This class implements the exact seed schedule specification:
    - Training:   s_train(j, x) = (m0 + 100 + VarID(x), m1 + j) for j=1..J
    - Validation: s_val(x)      = (m0 + 200 + VarID(x), m1 + 0)
    - Test:       s_test(x)     = (m0 + 300 + VarID(x), m1 + 0)

    Seeds are returned as TensorFlow tensors (shape [2], dtype tf.int32) ready
    for use with tf.random.stateless_uniform and tf.random.stateless_normal.

    Example:
        >>> config = SeedScheduleConfig(master_seed=(42, 0), n_train_steps=100)
        >>> schedule = SeedSchedule(config)
        >>>
        >>> # Get training seeds for step 1
        >>> train_seeds = schedule.get_train_seeds(steps=[1])
        >>> train_seeds[1][VariableID.K0]  # tf.Tensor([143, 1], dtype=tf.int32)
        >>>
        >>> # Get all validation seeds
        >>> val_seeds = schedule.get_val_seeds()
        >>> val_seeds[VariableID.K0]  # tf.Tensor([243, 0], dtype=tf.int32)
    """

    def __init__(self, config: SeedScheduleConfig):
        """
        Initialize seed schedule with configuration.

        Args:
            config: SeedScheduleConfig instance
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate configuration (called at initialization)."""
        # Check if max training step causes overflow
        if self.config.n_train_steps is not None:
            m1 = self.config.master_seed[1]
            max_step = self.config.n_train_steps
            step_component = m1 + max_step
            if not _is_valid_int32(step_component):
                raise ValueError(
                    f"Training step {max_step} causes int32 overflow with "
                    f"master_seed[1]={m1}. Computed: {step_component}"
                )

    def _validate_step(self, step: int, context: str = ""):
        """
        Validate step index.

        Args:
            step: Step index to validate
            context: Context string for error messages
        """
        if step < 0:
            raise ValueError(f"{context}Step must be >= 0. Got {step}")

        if self.config.n_train_steps is not None and step > self.config.n_train_steps:
            raise ValueError(
                f"{context}Step {step} exceeds configured max "
                f"{self.config.n_train_steps}. Either increase n_train_steps or "
                f"use a smaller step index."
            )

    def _compute_seed(
        self,
        split: Literal["train", "val", "test"],
        variable: VariableID,
        step: int = 0
    ) -> tf.Tensor:
        """
        Compute a single seed tensor using the exact specification.

        Formula:
            seed = (m0 + offset + var_id, m1 + step_idx)

        Where:
            - offset: train_offset, val_offset, or test_offset
            - var_id: VariableID.value
            - step_idx: step for training, 0 for val/test

        Args:
            split: Data split ("train", "val", or "test")
            variable: Variable identifier
            step: Step index (only used for training, 0 for val/test)

        Returns:
            Seed tensor of shape [2] with dtype tf.int32
        """
        m0, m1 = self.config.master_seed

        # Get offset based on split
        offset_map = {
            "train": self.config.train_offset,
            "val": self.config.val_offset,
            "test": self.config.test_offset,
        }
        offset = offset_map[split]

        # Compute seed components
        seed_0 = m0 + offset + variable.value
        seed_1 = m1 + step

        # Wrap to int32 (handles overflow deterministically)
        seed_0 = _wrap_to_int32(seed_0)
        seed_1 = _wrap_to_int32(seed_1)

        return tf.constant([seed_0, seed_1], dtype=tf.int32)

    def get_train_seeds(
        self,
        steps: Optional[List[int]] = None,
        variables: Optional[List[VariableID]] = None
    ) -> Dict[int, Dict[VariableID, tf.Tensor]]:
        """
        Get training seeds for specified steps.

        Args:
            steps: Step indices (1-indexed). If None, returns all steps 1..n_train_steps.
            variables: Variables to include. If None, returns all variables.

        Returns:
            Dict mapping step -> variable -> seed tensor

        Raises:
            ValueError: If steps requested but n_train_steps not configured
            ValueError: If step index is invalid

        Example:
            >>> train_seeds = schedule.get_train_seeds(steps=[1, 2, 3])
            >>> step1_k0 = train_seeds[1][VariableID.K0]
            >>> # Use with TensorFlow
            >>> k = tf.random.stateless_uniform(
            ...     shape=(100,), seed=step1_k0, minval=0.5, maxval=5.0
            ... )
        """
        # Determine which steps to generate
        if steps is None:
            if self.config.n_train_steps is None:
                raise ValueError(
                    "Cannot generate all training steps: n_train_steps not configured. "
                    "Either provide explicit steps list or set n_train_steps in config."
                )
            steps = list(range(1, self.config.n_train_steps + 1))

        # Determine which variables to include
        if variables is None:
            variables = list(VariableID)

        # Generate seeds for each step and variable
        result = {}
        for step in steps:
            self._validate_step(step, context=f"get_train_seeds: ")
            step_seeds = {}
            for var in variables:
                step_seeds[var] = self._compute_seed("train", var, step)
            result[step] = step_seeds

        return result

    def get_val_seeds(
        self,
        variables: Optional[List[VariableID]] = None
    ) -> Dict[VariableID, tf.Tensor]:
        """
        Get validation seeds.

        Validation uses a single fixed dataset (step=0 in formula).

        Args:
            variables: Variables to include. If None, returns all variables.

        Returns:
            Dict mapping variable -> seed tensor

        Example:
            >>> val_seeds = schedule.get_val_seeds()
            >>> eps_seed = val_seeds[VariableID.EPS1]
            >>> eps = tf.random.stateless_normal(shape=(1000,), seed=eps_seed)
        """
        if variables is None:
            variables = list(VariableID)

        result = {}
        for var in variables:
            result[var] = self._compute_seed("val", var, step=0)

        return result

    def get_test_seeds(
        self,
        variables: Optional[List[VariableID]] = None
    ) -> Dict[VariableID, tf.Tensor]:
        """
        Get test seeds.

        Test uses a single fixed dataset (step=0 in formula).

        Args:
            variables: Variables to include. If None, returns all variables.

        Returns:
            Dict mapping variable -> seed tensor

        Example:
            >>> test_seeds = schedule.get_test_seeds()
            >>> k_seed = test_seeds[VariableID.K0]
            >>> z_seed = test_seeds[VariableID.Z0]
        """
        if variables is None:
            variables = list(VariableID)

        result = {}
        for var in variables:
            result[var] = self._compute_seed("test", var, step=0)

        return result

    def get_single_seed(
        self,
        split: Literal["train", "val", "test"],
        variable: VariableID,
        step: int = 0
    ) -> tf.Tensor:
        """
        Get a single seed for fine-grained control.

        Args:
            split: Data split ("train", "val", or "test")
            variable: Variable identifier
            step: Step index (only for training, ignored for val/test)

        Returns:
            Seed tensor of shape [2] with dtype tf.int32

        Example:
            >>> seed = schedule.get_single_seed("train", VariableID.K0, step=5)
            >>> seed  # tf.Tensor([143, 5], dtype=tf.int32)
        """
        if split == "train":
            self._validate_step(step, context=f"get_single_seed (split={split}): ")

        return self._compute_seed(split, variable, step)
