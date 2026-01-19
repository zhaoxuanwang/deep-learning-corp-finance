"""
src/economy/indicators.py

Centralized gate/indicator utilities for gradient-based training.

Hard indicators (e.g., 1{x>0}) have zero gradients almost everywhere.
STE (Straight-Through Estimator) provides gradient flow while preserving
hard forward semantics for economic correctness.

Indicators:
- Adjustment cost: 1{|I| > eps} - investment/disinvestment triggers fixed cost
- External finance: 1{e < 0} - negative cash flow requires equity injection
"""

import tensorflow as tf
from typing import Literal

Numeric = tf.Tensor


# =============================================================================
# CORE GATE UTILITIES
# =============================================================================

def ste_gate_abs_gt(
    x: Numeric,
    eps: float = 1e-6,
    temp: float = 0.1,
    mode: Literal["hard", "ste", "soft"] = "ste"
) -> Numeric:
    """
    Gate for |x| > eps with optional STE gradient.
    
    Forward: Returns 1 when |x| > eps, 0 otherwise.
    Backward (STE): Uses smooth sigmoid surrogate for gradient flow.
    
    Use case: Investment indicator 1{|I| > eps}
    
    Args:
        x: Input tensor (e.g., investment I)
        eps: Threshold (default 1e-6)
        temp: Sigmoid temperature for soft gate (default 0.1)
        mode: "hard" (no gradient), "ste" (STE), "soft" (smooth sigmoid)
    
    Returns:
        Gate value in [0, 1]
    """
    abs_x = tf.abs(x)
    
    # Hard gate (exact)
    g_hard = tf.cast(abs_x > eps, tf.float32)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate (smooth sigmoid)
    g_soft = tf.nn.sigmoid((abs_x - eps) / temp)
    
    if mode == "soft":
        return g_soft
    
    # STE: forward=hard, backward=soft gradient
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def ste_gate_lt(
    x: Numeric,
    threshold: float = 0.0,
    temp: float = 0.1,
    mode: Literal["hard", "ste", "soft"] = "ste"
) -> Numeric:
    """
    Gate for x < threshold with optional STE gradient.
    
    Forward: Returns 1 when x < threshold, 0 otherwise.
    Backward (STE): Uses smooth sigmoid surrogate for gradient flow.
    
    Use case: External finance indicator 1{e < 0}
    
    Args:
        x: Input tensor (e.g., cash flow e)
        threshold: Comparison threshold (default 0.0)
        temp: Sigmoid temperature for soft gate (default 0.1)
        mode: "hard" (no gradient), "ste" (STE), "soft" (smooth sigmoid)
    
    Returns:
        Gate value in [0, 1]
    """
    # Hard gate (exact)
    g_hard = tf.cast(x < threshold, tf.float32)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate: sigmoid(-(x - threshold)/temp) = high when x < threshold
    g_soft = tf.nn.sigmoid(-(x - threshold) / temp)
    
    if mode == "soft":
        return g_soft
    
    # STE: forward=hard, backward=soft gradient
    return tf.stop_gradient(g_hard - g_soft) + g_soft


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

def hard_gate_abs_gt(x: Numeric, eps: float = 1e-6) -> Numeric:
    """Hard gate for |x| > eps. Zero gradient."""
    return ste_gate_abs_gt(x, eps=eps, mode="hard")


def hard_gate_lt(x: Numeric, threshold: float = 0.0) -> Numeric:
    """Hard gate for x < threshold. Zero gradient."""
    return ste_gate_lt(x, threshold=threshold, mode="hard")
