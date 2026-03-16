"""
Economic parameters and configuration for the investment model.

This module provides:
- EconomicParams: Pure economic primitives (single source of truth)

It works in tandem with src.ddp.DDPGridConfig (for numerical grid settings), ensuring
a clean separation between economic fundamentals and solution methods.
"""

from __future__ import annotations
import dataclasses
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Tuple, Optional, Any, Dict

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# =============================================================================
# SHOCK PARAMS (Single Source of Truth)
# =============================================================================

@dataclass(frozen=True)
class ShockParams:
    """
    Immutable container for shock process parameters (AR(1)).
    Separated from EconomicParams to ensure data generation consistency.
    """
    rho: float = 0.7
    sigma: float = 0.15
    mu: float = 0.0

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0. Got {self.sigma}")
        
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1). Got {self.rho}")
       
    @classmethod
    def with_overrides(
        cls,
        base: Optional[ShockParams] = None,
        log_changes: bool = True,
        **overrides
    ) -> ShockParams:
        """
        Update ShockParams with strict validation and logging.
        
        Args:
            base: Existing parameters to update. If None, uses defaults.
            log_changes: Whether to log the differences.
            **overrides: Key-value pairs of parameters to update.
            
        Returns:
            New ShockParams instance.
        """
        base = base or cls()

        # 1. Validate keys to prevent typos
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        if unknown := set(overrides) - valid_keys:
            raise ValueError(f"Invalid override keys: {unknown}. Valid: {sorted(valid_keys)}")

        # 2. Log significant changes
        if log_changes:
            changes = [
                f"{k}: {getattr(base, k)} -> {v}"
                for k, v in overrides.items()
                if getattr(base, k) != v
            ]
            if changes:
                logger.info(f"ShockParams overrides: {', '.join(changes)}")

        return dataclasses.replace(base, **overrides)

# =============================================================================
# ECONOMIC PARAMS (Single Source of Truth)
# =============================================================================

@dataclass(frozen=True)
class EconomicParams:
    """
    Immutable container for pure economic primitives.
    
    This is the SINGLE SOURCE OF TRUTH for economic parameters.
    Both DNN and DDP methods should reference this class.
    
    Attributes:
        r_rate: Risk-free interest rate
        delta: Depreciation rate
        theta: Production elasticity/returns to scale
        cost_convex: Coefficient for convex adjustment costs (φ₀)
        cost_fixed: Coefficient for fixed adjustment costs (φ₁)
        tax: Corporate income tax rate
        cost_default: Default/bankruptcy cost (α)
        cost_inject_fixed: Fixed cost of external equity (η₀)
        cost_inject_linear: Proportional cost of external equity (η₁)
        frac_liquid: Fraction of capital that can be liquidated
    
    Example:
        params = EconomicParams()  # Use defaults
        params = EconomicParams(cost_convex=1.0)  # Override one field
        params = EconomicParams.with_overrides(cost_convex=1.0)  # Same, with logging
    """
    # Core production
    r_rate: float = 0.04
    delta: float = 0.15
    theta: float = 0.7
    
    # Adjustment costs
    cost_convex: float = 0.0
    cost_fixed: float = 0.0
    
    # Risky debt
    tax: float = 0.3
    cost_default: float = 0.4
    cost_inject_fixed: float = 0.0
    cost_inject_linear: float = 0.0
    frac_liquid: float = 0.5
    
    def __post_init__(self):
        """Validate parameters immediately after initialization."""
        # Basic economic parameters
        if not (0.0 < self.r_rate < 1.0):
            raise ValueError(f"r_rate must be in (0, 1). Got {self.r_rate}")
        
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"delta must be in [0, 1]. Got {self.delta}")
        
        if not (0.0 < self.theta < 1.0):
            raise ValueError(f"theta must be in (0, 1). Got {self.theta}")
        
        if self.cost_convex < 0:
            raise ValueError(f"cost_convex must be >= 0. Got {self.cost_convex}")
        
        if self.cost_fixed < 0:
            raise ValueError(f"cost_fixed must be >= 0. Got {self.cost_fixed}")
        
        # Risky debt
        if not (0.0 <= self.cost_default <= 1.0):
            raise ValueError(f"cost_default must be in [0, 1]. Got {self.cost_default}")
        
        if not (0.0 <= self.frac_liquid <= 1.0):
            raise ValueError(f"frac_liquid must be in [0, 1]. Got {self.frac_liquid}")
        
        if not (0.0 <= self.tax < 1.0):
            raise ValueError(f"tax must be in [0, 1). Got {self.tax}")
        
        if self.cost_inject_fixed < 0:
            raise ValueError(f"cost_inject_fixed must be >= 0. Got {self.cost_inject_fixed}")
        
        if self.cost_inject_linear < 0:
            raise ValueError(f"cost_inject_linear must be >= 0. Got {self.cost_inject_linear}")

    @classmethod
    def with_overrides(
        cls,
        base: Optional[EconomicParams] = None,
        log_changes: bool = True,
        **overrides
    ) -> EconomicParams:
        """
        Create (or update) EconomicParams with strict validation and logging.
        
        Args:
            base: Existing parameters to update. If None, uses defaults.
            log_changes: Whether to log the differences.
            **overrides: Key-value pairs of parameters to update.
            
        Returns:
            New EconomicParams instance.
        """
        base = base or cls()

        # 1. Validate keys to prevent typos
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        if unknown := set(overrides) - valid_keys:
            raise ValueError(f"Invalid override keys: {unknown}. Valid: {sorted(valid_keys)}")

        # 2. Log significant changes
        if log_changes:
            changes = [
                f"{k}: {getattr(base, k)} -> {v}"
                for k, v in overrides.items()
                if getattr(base, k) != v
            ]
            if changes:
                logger.info(f"EconomicParams overrides: {', '.join(changes)}")

        return dataclasses.replace(base, **overrides)
    
    def steady_state_k(self) -> float:
        """Compute approximate steady-state capital stock."""
        user_cost = self.r_rate + self.delta
        return (self.theta / user_cost) ** (1 / (1 - self.theta))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_to_tf(*args: np.ndarray) -> List[tf.Tensor]:
    """Converts NumPy arrays into TensorFlow constants."""
    return [tf.constant(arg, dtype=tf.float32) for arg in args]

