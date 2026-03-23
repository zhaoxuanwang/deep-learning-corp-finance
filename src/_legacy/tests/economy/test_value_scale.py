"""
Tests for frictionless value benchmark (BR normalization scale).
"""
import numpy as np
import pytest

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.value_scale import (
    FrictionlessValueBenchmark,
    compute_frictionless_value_benchmark,
)


def test_benchmark_known_values():
    """Verify formula against hand computation with default parameters."""
    params = EconomicParams()
    shock_params = ShockParams()

    bm = compute_frictionless_value_benchmark(params, shock_params)

    # z* = exp(mu)
    z_star = np.exp(shock_params.mu)
    assert np.isclose(bm.z_star, z_star)

    # k* = ((theta * z*) / (r + delta))^(1/(1-theta))
    k_star = ((params.theta * z_star) / (params.r_rate + params.delta)) ** (
        1 / (1 - params.theta)
    )
    assert np.isclose(bm.k_star, k_star, rtol=1e-6)

    # r* = z* * k*^theta - delta * k*
    reward_star = z_star * (k_star ** params.theta) - params.delta * k_star
    assert np.isclose(bm.reward_star, reward_star, rtol=1e-6)

    # beta = 1/(1+r), V* = r*/(1-beta)
    beta = 1.0 / (1.0 + params.r_rate)
    value_star = reward_star / (1.0 - beta)
    assert np.isclose(bm.value_star, value_star, rtol=1e-6)

    # Scale must be positive
    assert bm.scale > 0
    assert bm.scale == max(abs(value_star), 1e-8)


def test_benchmark_scale_positive_for_small_reward():
    """Epsilon guard ensures scale stays positive even for tiny rewards."""
    # Use very low productivity to get near-zero reward
    params = EconomicParams(theta=0.01, delta=0.99)
    shock_params = ShockParams(mu=-10.0)

    bm = compute_frictionless_value_benchmark(params, shock_params, epsilon=1.0)
    assert bm.scale >= 1.0
