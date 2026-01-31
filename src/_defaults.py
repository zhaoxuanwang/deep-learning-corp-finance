"""
src/_defaults.py

Centralized default constants for training configuration.

This module contains ONLY constants with NO imports to avoid circular dependencies.
Both trainers/config.py and utils/annealing.py import from here to maintain a single
source of truth without creating import cycles.

Usage:
    # In config.py
    from src._defaults import DEFAULT_TEMPERATURE_INIT, ...

    # In annealing.py
    from src._defaults import DEFAULT_TEMPERATURE_INIT, ...
"""

# =============================================================================
# TRAINING LOOP DEFAULTS
# =============================================================================

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_ITER = 1000
DEFAULT_LOG_EVERY = 10

DEFAULT_N_CRITIC = 5  # Critic updates per actor update (BR methods)
DEFAULT_POLYAK_TAU = 0.995  # Polyak averaging coefficient for target networks

DEFAULT_WEIGHT_BR = 0.1  # Bellman residual weight for Risky Debt BR method


# =============================================================================
# ANNEALING & NUMERICAL DEFAULTS
# =============================================================================
# Centralized defaults for annealing schedules and indicator functions.
# These control the smooth approximation of discrete operations (gates, indicators).
#
# Reference: report_brief.md lines 420-433 (Soft Gates)

# Temperature annealing schedule (AnnealingSchedule, AnnealingConfig)
DEFAULT_TEMPERATURE_INIT = 1.0     # Initial temperature for soft gates
DEFAULT_TEMPERATURE_MIN = 1e-4     # Minimum temperature floor
DEFAULT_ANNEAL_DECAY = 0.995       # Per-step multiplicative decay: tau[j+1] = decay * tau[j]
DEFAULT_ANNEAL_BUFFER = 0.25       # Stabilization buffer fraction for n_anneal calculation
DEFAULT_LOGIT_CLIP = 20.0          # Logit clipping for smooth indicators (prevents saturation)

# Indicator function defaults (indicator_abs_gt, indicator_lt, indicator_default)
DEFAULT_INDICATOR_THRESHOLD = 1e-4  # Default threshold for |x| > eps gates
DEFAULT_INDICATOR_LOGIT_CLIP = 5.0  # Tighter logit clip for indicator functions

# Numerical safety
DEFAULT_SAFE_EPSILON = 1e-8         # Safety threshold for division operations in logic.py


# =============================================================================
# EARLY STOPPING / CONVERGENCE DEFAULTS
# =============================================================================

DEFAULT_PATIENCE = 5  # Consecutive checks before stopping
DEFAULT_EVAL_FREQ = 100  # Evaluate validation metrics every N steps
DEFAULT_LR_EPSILON = 1e-4  # LR relative improvement threshold
DEFAULT_LR_WINDOW = 100  # LR window size for improvement evaluation
DEFAULT_MA_WINDOW = 10  # Moving average window for smoothing
DEFAULT_ER_EPSILON = 1e-4  # ER absolute threshold (unit-free, ~1% accuracy)
DEFAULT_BR_CRITIC_EPSILON = 1e-5  # BR critic threshold
DEFAULT_BR_ACTOR_EPSILON = 1e-4  # BR actor relative improvement threshold
DEFAULT_DIVISION_EPSILON = 1e-10  # Safety threshold for division operations
