"""
src/trainers/

Training algorithms for DNN-based economic models.

This package implements three solution methods:
- LR (Lifetime Reward): Maximizes discounted lifetime rewards
- ER (Euler Residual): Minimizes Euler equation residuals
- BR (Bellman Residual): Actor-critic framework

Modules:
    basic: LR, ER, and BR trainers for basic investment model
    risky: BR trainer for risky debt model
    losses: Loss computation utilities
    config: Configuration dataclasses
    core: Training loop infrastructure
    stopping: Early stopping and convergence criteria

References:
    report/report_brief.md lines 442-1084: Training Methods
"""

from src.trainers.config import (
    NetworkConfig,
    OptimizationConfig,
    AnnealingConfig,
    EarlyStoppingConfig,
    MethodConfig,
    RiskyDebtConfig,
)

from src.trainers.basic import (
    train_basic_lr,
    train_basic_er,
    train_basic_br,
)

from src.trainers.risky import (
    train_risky_br,
)

__all__ = [
    # Configuration
    "NetworkConfig",
    "OptimizationConfig",
    "AnnealingConfig",
    "EarlyStoppingConfig",
    "MethodConfig",
    "RiskyDebtConfig",
    # Training functions
    "train_basic_lr",
    "train_basic_er",
    "train_basic_br",
    "train_risky_br",
]
