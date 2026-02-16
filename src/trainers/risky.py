"""
Compatibility facade for Risky Debt trainers and entrypoints.

Implementation has been split into:
- src/trainers/risky_trainers.py (trainer classes)
- src/trainers/risky_api.py (high-level training entrypoints)
"""

from src.trainers.risky_trainers import RiskyDebtTrainerBR
from src.trainers.risky_api import train_risky_br, train_risky_br_actor_critic

__all__ = [
    "RiskyDebtTrainerBR",
    "train_risky_br",
    "train_risky_br_actor_critic",
]
