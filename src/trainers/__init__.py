"""
src/trainers/

Training algorithms for DNN-based economic models.

This package implements three solution methods:
- LR (Lifetime Reward): Maximizes discounted lifetime rewards
- ER (Euler Residual): Minimizes Euler equation residuals
- BR Actor-Critic (Bellman Residual): Actor-critic framework
- BR Multitask (Maliar-style): Joint BR + FOC/Envelope weighted objective

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
    ExperimentConfig,
    DataConfig,
)
from src.trainers.method_specs import (
    MethodSpec,
    register_method_spec,
    get_method_spec,
    list_method_specs,
)

from src.trainers.basic import (
    train_basic_lr,
    train_basic_er,
    train_basic_br,
    train_basic_br_reg,
    train_basic_br_actor_critic,
    train_basic_br_multitask,
)

from src.trainers.risky import (
    train_risky_br,
    train_risky_br_actor_critic,
)
from src.trainers.api import train, train_from_dataset_bundle
from src.trainers.results import TrainingResult

__all__ = [
    # Configuration
    "NetworkConfig",
    "OptimizationConfig",
    "AnnealingConfig",
    "EarlyStoppingConfig",
    "MethodConfig",
    "RiskyDebtConfig",
    "ExperimentConfig",
    "DataConfig",
    "MethodSpec",
    "register_method_spec",
    "get_method_spec",
    "list_method_specs",
    # Training functions
    "train_basic_lr",
    "train_basic_er",
    "train_basic_br",
    "train_basic_br_reg",
    "train_basic_br_actor_critic",
    "train_basic_br_multitask",
    "train_risky_br",
    "train_risky_br_actor_critic",
    "train",
    "train_from_dataset_bundle",
    "TrainingResult",
]
