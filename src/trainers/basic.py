"""
Compatibility facade for Basic model trainers and entrypoints.

Implementation has been split into:
- src/trainers/basic_trainers.py (trainer classes)
- src/trainers/basic_api.py (high-level training entrypoints)
"""

from src.trainers.basic_trainers import (
    BasicTrainerLR,
    BasicTrainerER,
    BasicTrainerBR,
    BasicTrainerBRRegression,
)
from src.trainers.basic_api import (
    train_basic_lr,
    train_basic_er,
    train_basic_br,
    train_basic_br_reg,
    train_basic_br_actor_critic,
    train_basic_br_multitask,
)

__all__ = [
    "BasicTrainerLR",
    "BasicTrainerER",
    "BasicTrainerBR",
    "BasicTrainerBRRegression",
    "train_basic_lr",
    "train_basic_er",
    "train_basic_br",
    "train_basic_br_reg",
    "train_basic_br_actor_critic",
    "train_basic_br_multitask",
]
