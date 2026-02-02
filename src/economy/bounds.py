"""
src/economy/bounds.py

Utilities for computing sampling bounds for state space (grids or DNN training).

=============================================================================
KEY DESIGN PRINCIPLE: BOUNDS ARE RETURNED IN LEVELS
=============================================================================

This module provides TWO ways to specify bounds:

1. MODEL-BASED AUTO-COMPUTATION (recommended for economic models):
   - User specifies bounds as MULTIPLIERS on steady-state k*
   - Example: k_min_multiplier=0.2, k_max_multiplier=3.0
   - This means k ∈ [0.2 * k*, 3.0 * k*] where k* is computed from economic params
   - The returned bounds are already converted to LEVELS for direct use

2. DIRECT SPECIFICATION (for custom data or arbitrary units):
   - User provides bounds directly in whatever units their data uses
   - No economic model needed - framework is unit-agnostic
   - Simply pass bounds like k_bounds=(5000, 10000) directly to DataGenerator

IMPORTANT:
- All functions return bounds in LEVELS (actual values), NOT as multipliers
- Networks internally normalize inputs to [0,1] via bounded sigmoid for stability
- Trainers pass k values directly to economic functions without any scaling
- k_star is stored for documentation/reference only, not for post-hoc conversion

Input Constraints (for model-based auto-computation):
- m ∈ (2, 5): Standard deviations for log_z bounds
- k_min ∈ (0, 0.5): Lower bound as multiplier on k*
- k_max ∈ (1.5, 5): Upper bound as multiplier on k*
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field
from src.economy.parameters import EconomicParams, ShockParams
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# BOUNDS CONFIGURATION
# =============================================================================

@dataclass
class BoundsConfig:
    """
    Configuration for model-based bounds computation.

    This config is used for AUTO-COMPUTING bounds from economic parameters.
    User specifies bounds as MULTIPLIERS on k*, and the module returns
    bounds in LEVELS ready for direct use by networks.

    For custom/arbitrary bounds, skip this config and pass bounds directly
    to DataGenerator or generate_states_bounds with validate=False.

    Attributes:
        m: Standard deviation multiplier for log_z bounds (2 < m < 5)
        k_min: Capital lower bound as multiplier on k* (0 < k_min < 0.5)
        k_max: Capital upper bound as multiplier on k* (1.5 < k_max < 5)
        k_star_override: Optional override for steady-state k* calculation

    Example:
        >>> config = BoundsConfig(m=3.0, k_min=0.2, k_max=3.0)
        >>> bounds = generate_bounds_from_config(config, theta, r, delta, shock_params)
        >>> # bounds['k'] is now in LEVELS, e.g., (15.4, 231.7) for k*=77.24
    """
    m: float = 3.0
    k_min: float = 0.2
    k_max: float = 3.0
    k_star_override: Optional[float] = None

    def __post_init__(self):
        """Validate bounds constraints per report_brief.md lines 111-114."""
        if not (2.0 < self.m < 5.0):
            raise ValueError(
                f"m (std_dev_multiplier) must be in (2, 5), got {self.m}. "
                f"This ensures log_z bounds cover the ergodic distribution "
                f"while avoiding extreme rare events."
            )
        if not (0.0 < self.k_min < 0.5):
            raise ValueError(
                f"k_min must be in (0, 0.5), got {self.k_min}. "
                f"This ensures normalized capital space avoids zero "
                f"while allowing for small initial capital."
            )
        if not (1.5 < self.k_max < 5.0):
            raise ValueError(
                f"k_max must be in (1.5, 5), got {self.k_max}. "
                f"This ensures normalized capital space covers above steady-state "
                f"while avoiding gradient explosion."
            )


def compute_ergodic_log_z_bounds(
    shock_params: ShockParams, 
    std_dev_multiplier: float
) -> Tuple[float, float]:
    """
    Compute natural bounds for log(z) based on the ergodic distribution of the AR(1) process.
    
    Bounds = mu +/- m * sigma_ergodic
    where sigma_ergodic = sigma / sqrt(1 - rho^2).
    
    Args:
        params: Economic parameters
        std_dev_multiplier: Number of standard deviations from mean
    
    Returns:
        (min_log_z, max_log_z)
    """
    rho = shock_params.rho
    sigma = shock_params.sigma
    mu = shock_params.mu
    
    if abs(rho) >= 1.0:
        logger.warning(f"AR(1) rho={rho} is >= 1. Cannot compute ergodic std. Using default bounds (-1, 1).")
        return -1.0, 1.0
    
    std_ergodic = sigma / np.sqrt(1 - rho**2)
    
    min_log_z = mu - std_dev_multiplier * std_ergodic
    max_log_z = mu + std_dev_multiplier * std_ergodic
    
    return float(min_log_z), float(max_log_z)


def compute_k_star(
    theta: float,
    r: float,
    delta: float,
    mu: float
) -> float:
    """
    Compute steady-state capital k* at the stationary mean productivity z = e^μ.

    The steady-state condition is MPK = r + δ:
        k* = ((z * θ) / (r + δ)) ^ (1/(1-θ))

    Reference:
        report_brief.md lines 71-74: Steady-state capital formula

    Args:
        theta: Production elasticity (γ in the report)
        r: Risk-free rate
        delta: Depreciation rate
        mu: Unconditional mean of log(z) (stationary mean)

    Returns:
        k_star: Steady-state capital level at z = e^μ
    """
    z_mean = np.exp(mu)
    denom = r + delta

    if denom <= 0:
        logger.warning(f"r + delta = {denom} <= 0. Using fallback k_star = 1.0")
        return 1.0

    k_star = ((z_mean * theta) / denom) ** (1 / (1 - theta))
    return float(k_star)


def compute_k_bounds_levels(
    k_min_multiplier: float,
    k_max_multiplier: float,
    theta: float,
    r: float,
    delta: float,
    mu: float,
    k_star_override: Optional[float] = None
) -> Tuple[Tuple[float, float], float]:
    """
    Compute capital bounds in LEVELS, anchored to steady-state k*.

    User specifies bounds as multipliers on k* (e.g., 0.2 to 3.0 meaning
    20% to 300% of steady-state). This function converts to actual level
    bounds that can be used directly by networks and economic functions.

    This design separates:
    - Economic anchoring: bounds are meaningful fractions of steady-state
    - NN normalization: network internally normalizes to [0,1] for stability

    Reference:
        report_brief.md lines 89-94: Normalization approach

    Args:
        k_min_multiplier: Lower bound as fraction of k* (0 < k_min < 0.5)
        k_max_multiplier: Upper bound as fraction of k* (1.5 < k_max < 5)
        theta: Production elasticity
        r: Risk-free rate
        delta: Depreciation rate
        mu: Unconditional mean of log(z)
        k_star_override: Optional override for k* (if None, computed from params)

    Returns:
        Tuple of:
            - k_bounds: (k_min, k_max) in LEVELS
            - k_star: Steady-state capital (for reference/documentation)
    """
    # Compute or use override for k*
    if k_star_override is not None:
        k_star = k_star_override
        logger.info(f"Using k_star override: {k_star:.4f}")
    else:
        k_star = compute_k_star(theta, r, delta, mu)
        logger.debug(f"Auto-computed k_star = {k_star:.4f} at z = e^{mu:.4f}")

    # Convert multipliers to LEVEL bounds
    k_min_level = k_min_multiplier * k_star
    k_max_level = k_max_multiplier * k_star

    k_bounds = (float(k_min_level), float(k_max_level))

    logger.debug(
        f"k bounds in levels: ({k_min_level:.2f}, {k_max_level:.2f}) "
        f"= ({k_min_multiplier}, {k_max_multiplier}) × k*={k_star:.2f}"
    )

    return k_bounds, k_star


# Legacy function for backward compatibility
def compute_natural_k_bounds(
    theta: float,
    r: float,
    delta: float,
    log_z_bounds: Tuple[float, float],
    k_min_multiplier: float,
    k_max_multiplier: float
) -> Tuple[float, float]:
    """
    [DEPRECATED] Compute natural bounds for capital k based on steady-state logic.

    NOTE: This function is deprecated. Use compute_k_bounds_levels() instead.
    The new approach computes k* at the stationary mean z=e^μ and returns
    bounds directly in LEVELS.

    This legacy function computes k* at z_min and z_max separately, which
    gives different bounds than the recommended anchoring to stationary k*.

    Args:
        theta: Production elasticity
        r: Risk-free rate
        delta: Depreciation rate
        log_z_bounds: (min_log_z, max_log_z)
        k_min_multiplier: Factor for lower bound
        k_max_multiplier: Factor for upper bound

    Returns:
        (k_min, k_max) in LEVEL space
    """
    warnings.warn(
        "compute_natural_k_bounds is deprecated. Use compute_k_bounds_levels() "
        "which anchors bounds to stationary k* at z=e^μ.",
        DeprecationWarning,
        stacklevel=2
    )

    min_log_z, max_log_z = log_z_bounds

    z_min = np.exp(min_log_z)
    z_max = np.exp(max_log_z)

    # Helper for steady state k given z
    def get_k_star(z_val):
        denom = r + delta
        if denom <= 0:
            return 1.0  # Fallback
        return ((z_val * theta) / denom) ** (1 / (1 - theta))

    k_star_low = get_k_star(z_min)
    k_star_high = get_k_star(z_max)

    k_min = k_min_multiplier * k_star_low
    k_max = k_max_multiplier * k_star_high

    return float(k_min), float(k_max)


def compute_b_bound_levels(
    theta: float,
    k_max: float,
    z_min: float,
    tax: float,
    delta: float,
    frac_liquid: float
) -> float:
    """
    Compute borrowing limit in LEVELS using collateral constraint.

    The maximum borrowing is capped by the collateral constraint:
        b_max = (1-τ) π(k_max, z_min) + τ δ k_max + s_liquid · k_max

    This represents the worst-case liquidation value of the firm at z_min,
    which is much tighter than the old "natural borrowing limit" that used z_max.

    The constraint ensures firms cannot borrow more than they could repay
    even in the worst productivity state, preventing excessive leverage.

    Reference:
        report_brief.md "Debt Bounds" section: Collateral constraint formula

    Args:
        theta: Production elasticity (θ in the report)
        k_max: Maximum capital in LEVELS
        z_min: Minimum productivity LEVEL (not log) - worst case scenario
        tax: Corporate tax rate (τ)
        delta: Depreciation rate (δ)
        frac_liquid: Liquidation fraction (s_liquid) in [0, 1]

    Returns:
        b_max: Borrowing limit in LEVELS
    """
    # Production at worst state: π(k_max, z_min) = z_min * k_max^θ
    pi_worst = z_min * (k_max ** theta)

    # Collateral constraint components:
    # 1. After-tax production value: (1-τ) * π
    # 2. Tax shield on depreciation: τ * δ * k
    # 3. Liquidation value of capital: s_liquid * k
    b_max = (1 - tax) * pi_worst + tax * delta * k_max + frac_liquid * k_max

    return float(b_max)


# Legacy function for backward compatibility
def compute_natural_b_bound(
    theta: float,
    k_max: float,
    z_max: float
) -> float:
    """
    [DEPRECATED] Compute natural borrowing limit (B_max) in level space.

    NOTE: This function is deprecated and uses the OLD formula.
    Use compute_b_bound_levels() instead, which implements the tighter
    collateral constraint from report_brief.md.

    OLD formula: B_max = z_max * (k_max)^theta + k_max
    NEW formula: B_max = (1-τ) π(k_max, z_min) + τ δ k_max + s_liquid · k_max

    The new formula is much tighter and prevents excessive leverage.

    Args:
        theta: Production elasticity
        k_max: Maximum capital bound in LEVELS
        z_max: Maximum productivity (level, not log)

    Returns:
        b_max (float) in LEVEL space (using OLD formula)
    """
    warnings.warn(
        "compute_natural_b_bound is deprecated and uses the OLD formula. "
        "Use compute_b_bound_levels() with collateral constraint parameters instead.",
        DeprecationWarning,
        stacklevel=2
    )
    prod = z_max * (k_max ** theta)
    return float(prod + k_max)


def generate_states_bounds(
    theta: float,
    r: float,
    delta: float,
    shock_params: ShockParams,
    std_dev_multiplier: float,
    k_min_multiplier: float,
    k_max_multiplier: float,
    k_star_override: Optional[float] = None,
    validate: bool = True,
    # Collateral constraint parameters (with defaults from EconomicParams)
    tax: float = 0.3,
    frac_liquid: float = 0.5
) -> Dict[str, any]:
    """
    Generate consistent sampling bounds for (k, b, z) in LEVELS.

    This function computes economically-anchored bounds:
    - k bounds are specified as multipliers on k* (steady-state), returned in LEVELS
    - b bounds use collateral constraint at worst-case z_min (tighter than old formula)
    - log_z bounds cover the ergodic distribution

    The network's internal normalization (to [0,1]) is separate from these
    economic bounds. Networks receive level bounds and handle normalization
    internally via bounded sigmoid outputs.

    Debt Bound Formula (Collateral Constraint):
        b_max = (1-τ) π(k_max, z_min) + τ δ k_max + s_liquid · k_max

    This is tighter than the old "natural borrowing limit" (which used z_max)
    and prevents excessive leverage that destabilizes training.

    Reference:
        report_brief.md "Debt Bounds" section: Collateral constraint

    Args:
        theta: Production elasticity (θ in the report)
        r: Risk-free rate
        delta: Depreciation rate
        shock_params: AR(1) shock parameters (μ, σ, ρ)
        std_dev_multiplier: m, number of std devs for log_z bounds (2 < m < 5)
        k_min_multiplier: k_min as multiplier on k* (0 < k_min < 0.5)
        k_max_multiplier: k_max as multiplier on k* (1.5 < k_max < 5)
        k_star_override: Optional override for k* (if None, auto-computed)
        validate: If True, validate input constraints per report_brief.md
        tax: Corporate tax rate τ (default: 0.3 from EconomicParams)
        frac_liquid: Liquidation fraction s_liquid (default: 0.5 from EconomicParams)

    Returns:
        Dict containing:
            - "k": (k_min, k_max) capital bounds in LEVELS
            - "b": (0, b_max) debt bounds in LEVELS (using collateral constraint)
            - "log_z": (log_z_min, log_z_max) shock bounds
            - "k_star": steady-state capital (for reference/documentation)

    Raises:
        ValueError: If validate=True and constraints are violated
    """
    # Validate input constraints if requested
    if validate:
        if not (2.0 < std_dev_multiplier < 5.0):
            raise ValueError(
                f"std_dev_multiplier (m) must be in (2, 5), got {std_dev_multiplier}"
            )
        if not (0.0 < k_min_multiplier < 0.5):
            raise ValueError(
                f"k_min_multiplier must be in (0, 0.5), got {k_min_multiplier}"
            )
        if not (1.5 < k_max_multiplier < 5.0):
            raise ValueError(
                f"k_max_multiplier must be in (1.5, 5), got {k_max_multiplier}"
            )
        if not (0.0 <= tax < 1.0):
            raise ValueError(
                f"tax must be in [0, 1), got {tax}"
            )
        if not (0.0 <= frac_liquid <= 1.0):
            raise ValueError(
                f"frac_liquid must be in [0, 1], got {frac_liquid}"
            )

    # 1. Z bounds (unchanged)
    log_z_bounds = compute_ergodic_log_z_bounds(
        shock_params,
        std_dev_multiplier=std_dev_multiplier
    )

    # 2. K bounds in LEVELS - anchored to k* at stationary mean z = e^μ
    k_bounds, k_star = compute_k_bounds_levels(
        k_min_multiplier=k_min_multiplier,
        k_max_multiplier=k_max_multiplier,
        theta=theta,
        r=r,
        delta=delta,
        mu=shock_params.mu,
        k_star_override=k_star_override
    )

    # 3. B bounds in LEVELS - uses collateral constraint at WORST case z_min
    # This is much tighter than the old formula that used z_max
    log_z_min, _ = log_z_bounds
    z_min = np.exp(log_z_min)  # Worst-case productivity
    _, k_max = k_bounds  # k_max is now in LEVELS

    b_max = compute_b_bound_levels(
        theta=theta,
        k_max=k_max,
        z_min=z_min,
        tax=tax,
        delta=delta,
        frac_liquid=frac_liquid
    )
    b_bounds = (0.0, b_max)

    logger.debug(
        f"Generated bounds in LEVELS: k={k_bounds}, b={b_bounds}, "
        f"log_z={log_z_bounds}, k_star={k_star:.4f}"
    )

    return {
        "k": k_bounds,
        "b": b_bounds,
        "log_z": log_z_bounds,
        "k_star": k_star
    }


# =============================================================================
# CONVENIENCE FUNCTION WITH BOUNDS CONFIG
# =============================================================================

def generate_bounds_from_config(
    bounds_config: BoundsConfig,
    theta: float,
    r: float,
    delta: float,
    shock_params: ShockParams,
    tax: float = 0.3,
    frac_liquid: float = 0.5
) -> Dict[str, any]:
    """
    Generate bounds using a BoundsConfig object.

    This is a convenience wrapper around generate_states_bounds that
    uses the validated BoundsConfig dataclass for input.

    Args:
        bounds_config: Validated bounds configuration
        theta: Production elasticity
        r: Risk-free rate
        delta: Depreciation rate
        shock_params: AR(1) shock parameters
        tax: Corporate tax rate for collateral constraint (default: 0.3)
        frac_liquid: Liquidation fraction for collateral constraint (default: 0.5)

    Returns:
        Dict with 'k', 'b', 'log_z', and 'k_star' keys
    """
    return generate_states_bounds(
        theta=theta,
        r=r,
        delta=delta,
        shock_params=shock_params,
        std_dev_multiplier=bounds_config.m,
        k_min_multiplier=bounds_config.k_min,
        k_max_multiplier=bounds_config.k_max,
        k_star_override=bounds_config.k_star_override,
        validate=False,  # Already validated in BoundsConfig.__post_init__
        tax=tax,
        frac_liquid=frac_liquid
    )