"""Supported control methods used by the ablation notebook."""

from src.v2.ablations.brm_joint import train_brm_joint
from src.v2.ablations.er_original import train_er_original

__all__ = [
    "train_brm_joint",
    "train_er_original",
]
