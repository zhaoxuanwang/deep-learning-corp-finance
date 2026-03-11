"""
src/v2/data/rng.py

Random number generation and seed management for reproducible experiments.

Duplicated from src/economy/rng.py (v1) with one addition:
    VariableID.SHUFFLE = 6  (for dataset shuffle operations)

All other behaviour is identical to v1 — see that module for full docs.

Seed Schedule:
--------------
Training:   s_train(j, x) = (m0 + 100 + VarID(x), m1 + j) for step j=1..J
Validation: s_val(x)      = (m0 + 200 + VarID(x), m1 + 0)
Test:       s_test(x)     = (m0 + 300 + VarID(x), m1 + 0)

VariableID mapping (stable, never change existing IDs):
    K0=1, Z0=2, B0=3, EPS1=4, EPS2=5, SHUFFLE=6
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
    """Permanent mapping of variables to integer IDs for seed generation.

    IDs are STABLE and NEVER change. New variables must be added at the end.

    Variables:
        K0:      Initial / sampled endogenous capital
        Z0:      Initial exogenous productivity
        B0:      Initial debt (risky model)
        EPS1:    Main AR(1) shock sequence
        EPS2:    Fork AR(1) shock sequence (AiO second draw)
        SHUFFLE: Dataset shuffle permutation
    """
    K0      = 1
    Z0      = 2
    B0      = 3
    EPS1    = 4
    EPS2    = 5
    SHUFFLE = 6


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _is_valid_int32(value: int) -> bool:
    INT32_MIN = -(2**31)
    INT32_MAX = 2**31 - 1
    return INT32_MIN <= value <= INT32_MAX


def _wrap_to_int32(value: int) -> int:
    INT32_RANGE = 2**32
    INT32_OFFSET = 2**31
    wrapped = value % INT32_RANGE
    if wrapped >= INT32_OFFSET:
        wrapped -= INT32_RANGE
    return wrapped


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class SeedScheduleConfig:
    """Configuration for stateless RNG seed generation.

    Attributes:
        master_seed:    Two int32 values (m0, m1) that seed all RNG streams.
        train_offset:   Offset for training seeds (default 100).
        val_offset:     Offset for validation seeds (default 200).
        test_offset:    Offset for test seeds (default 300).
        n_train_steps:  Number of training steps for overflow validation.
    """
    master_seed:    Tuple[int, int]
    train_offset:   int = 100
    val_offset:     int = 200
    test_offset:    int = 300
    n_train_steps:  Optional[int] = None

    def __post_init__(self):
        m0, m1 = self.master_seed
        if not _is_valid_int32(m0):
            raise ValueError(
                f"master_seed[0] must be valid int32. Got {m0}")
        if not _is_valid_int32(m1):
            raise ValueError(
                f"master_seed[1] must be valid int32. Got {m1}")

        offsets = [self.train_offset, self.val_offset, self.test_offset]
        if len(set(offsets)) != 3:
            raise ValueError(
                f"Split offsets must be distinct. Got train={self.train_offset}, "
                f"val={self.val_offset}, test={self.test_offset}")

        max_var_id = max(v.value for v in VariableID)
        for name, offset in [("train", self.train_offset),
                              ("val",   self.val_offset),
                              ("test",  self.test_offset)]:
            component = m0 + offset + max_var_id
            if not _is_valid_int32(component):
                raise ValueError(
                    f"{name}_offset={offset} causes int32 overflow with "
                    f"master_seed[0]={m0} and max variable ID={max_var_id}. "
                    f"Computed: {component}")

        if self.n_train_steps is not None and self.n_train_steps <= 0:
            raise ValueError(
                f"n_train_steps must be > 0. Got {self.n_train_steps}")


# =============================================================================
# SEED SCHEDULE
# =============================================================================

class SeedSchedule:
    """Deterministic TensorFlow seeds for stateless RNG operations.

    Implements:
        Training:   s_train(j, x) = (m0 + 100 + VarID(x), m1 + j)
        Validation: s_val(x)      = (m0 + 200 + VarID(x), m1 + 0)
        Test:       s_test(x)     = (m0 + 300 + VarID(x), m1 + 0)

    Seeds are returned as tf.constant([s0, s1], dtype=tf.int32).
    """

    def __init__(self, config: SeedScheduleConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        if self.config.n_train_steps is not None:
            m1 = self.config.master_seed[1]
            step_component = m1 + self.config.n_train_steps
            if not _is_valid_int32(step_component):
                raise ValueError(
                    f"n_train_steps={self.config.n_train_steps} causes int32 "
                    f"overflow with master_seed[1]={m1}.")

    def _validate_step(self, step: int, context: str = ""):
        if step < 0:
            raise ValueError(f"{context}Step must be >= 0. Got {step}")
        if (self.config.n_train_steps is not None
                and step > self.config.n_train_steps):
            raise ValueError(
                f"{context}Step {step} exceeds n_train_steps="
                f"{self.config.n_train_steps}.")

    def _compute_seed(
        self,
        split: Literal["train", "val", "test"],
        variable: VariableID,
        step: int = 0,
    ) -> tf.Tensor:
        m0, m1 = self.config.master_seed
        offset_map = {
            "train": self.config.train_offset,
            "val":   self.config.val_offset,
            "test":  self.config.test_offset,
        }
        offset = offset_map[split]
        seed_0 = _wrap_to_int32(m0 + offset + variable.value)
        seed_1 = _wrap_to_int32(m1 + step)
        return tf.constant([seed_0, seed_1], dtype=tf.int32)

    def get_train_seeds(
        self,
        steps: Optional[List[int]] = None,
        variables: Optional[List[VariableID]] = None,
    ) -> Dict[int, Dict[VariableID, tf.Tensor]]:
        """Get training seeds for specified steps.

        Returns:
            Dict mapping step -> {VariableID -> seed tensor}.
        """
        if steps is None:
            if self.config.n_train_steps is None:
                raise ValueError(
                    "Provide explicit steps or set n_train_steps in config.")
            steps = list(range(1, self.config.n_train_steps + 1))
        if variables is None:
            variables = list(VariableID)
        result = {}
        for step in steps:
            self._validate_step(step, context="get_train_seeds: ")
            result[step] = {
                var: self._compute_seed("train", var, step)
                for var in variables
            }
        return result

    def get_val_seeds(
        self,
        variables: Optional[List[VariableID]] = None,
    ) -> Dict[VariableID, tf.Tensor]:
        """Get validation seeds (fixed, step=0)."""
        if variables is None:
            variables = list(VariableID)
        return {var: self._compute_seed("val", var, step=0)
                for var in variables}

    def get_test_seeds(
        self,
        variables: Optional[List[VariableID]] = None,
    ) -> Dict[VariableID, tf.Tensor]:
        """Get test seeds (fixed, step=0)."""
        if variables is None:
            variables = list(VariableID)
        return {var: self._compute_seed("test", var, step=0)
                for var in variables}

    def get_single_seed(
        self,
        split: Literal["train", "val", "test"],
        variable: VariableID,
        step: int = 0,
    ) -> tf.Tensor:
        """Get a single seed tensor."""
        if split == "train":
            self._validate_step(step, context=f"get_single_seed (split={split}): ")
        return self._compute_seed(split, variable, step)
