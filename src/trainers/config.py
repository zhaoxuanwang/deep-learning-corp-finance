"""
src/trainers/config.py

Hierarchical configuration system for DNN experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.economy.parameters import EconomicParams


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    n_layers: int = 2
    n_neurons: int = 16
    activation: str = "swish"


@dataclass
class OptimizationConfig:
    """Configuration for training loop and optimizer."""
    learning_rate: float = 1e-3
    learning_rate_critic: Optional[float] = None  # If None, use learning_rate
    batch_size: int = 128  # Trainer-controlled batch size
    n_iter: int = 1000
    log_every: int = 10


@dataclass
class AnnealingConfig:
    """Configuration for temperature annealing (soft gates)."""
    temperature_init: float = 1.0
    temperature_min: float = 1e-4
    decay: float = 0.9
    schedule: str = "exponential"  # "exponential" or "linear"
    logit_clip: float = 20.0


@dataclass
class RiskyDebtConfig:
    """Configuration specific to Risky Debt models."""
    # BR method: price constraint weights
    lambda_1: float = 1.0
    lambda_2: float = 1.0

    # LR method: adaptive Lagrange multiplier parameters
    lambda_price_init: float = 1.0
    learning_rate_lambda: float = 0.01
    epsilon_price: float = 0.01
    polyak_weight: float = 0.1
    n_value_update_freq: int = 1
    learning_rate_value: Optional[float] = None

    # Default probability smoothing (shared by LR and BR)
    epsilon_D_0: float = 0.1
    epsilon_D_min: float = 1e-4
    decay_d: float = 0.99


@dataclass
class MethodConfig:
    """Configuration for the solution method (Algorithm)."""
    name: str  # e.g., "basic_lr", "basic_er", "risky_br", etc.
    n_critic: int = 5   # Critic updates per actor update (BR methods)
    risky: Optional[RiskyDebtConfig] = None