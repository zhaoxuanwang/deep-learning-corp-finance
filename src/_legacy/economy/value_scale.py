"""
Utilities for Bellman-residual normalization scales.

This module computes a model-level value-scale proxy that can be evaluated
before training from economic primitives only.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.bounds import compute_k_star


@dataclass(frozen=True)
class FrictionlessValueBenchmark:
    """
    Frictionless steady-state benchmark used as BR normalizer.

    Attributes:
        k_star: Frictionless steady-state capital at z*=exp(mu)
        z_star: Stationary mean productivity level
        reward_star: Frictionless steady-state period reward
        beta: Discount factor implied by r_rate
        value_star: Frictionless steady-state value reward_star / (1-beta)
        scale: Safe positive scale max(abs(value_star), epsilon)
    """

    k_star: float
    z_star: float
    reward_star: float
    beta: float
    value_star: float
    scale: float


def compute_frictionless_value_benchmark(
    params: EconomicParams,
    shock_params: ShockParams,
    *,
    epsilon: float = 1e-8,
) -> FrictionlessValueBenchmark:
    """
    Compute frictionless benchmark scale for BR normalization.

    The benchmark follows:
        z* = exp(mu)
        k* = ((theta * z*) / (r + delta))^(1/(1-theta))
        r* = z* * (k*)^theta - delta * k*
        beta = 1/(1+r)
        V* = r*/(1-beta)

    Args:
        params: Economic parameters.
        shock_params: Shock process parameters.
        epsilon: Lower bound for positive normalization scale.

    Returns:
        FrictionlessValueBenchmark with raw value and safe scale.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    k_star = compute_k_star(
        theta=params.theta,
        r=params.r_rate,
        delta=params.delta,
        mu=shock_params.mu,
    )
    z_star = float(np.exp(shock_params.mu))
    reward_star = float(z_star * (k_star ** params.theta) - params.delta * k_star)
    beta = float(1.0 / (1.0 + params.r_rate))
    one_minus_beta = 1.0 - beta
    if one_minus_beta <= 0:
        raise ValueError(
            f"Invalid discount factor beta={beta:.8f}. Need beta < 1."
        )
    value_star = float(reward_star / one_minus_beta)
    scale = float(max(abs(value_star), epsilon))
    return FrictionlessValueBenchmark(
        k_star=float(k_star),
        z_star=z_star,
        reward_star=reward_star,
        beta=beta,
        value_star=value_star,
        scale=scale,
    )
