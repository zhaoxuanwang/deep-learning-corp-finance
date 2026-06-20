"""Investment and capital adjustment costs (DF26 Sec 1.4).

``i`` is the investment RATE with ``k' = (1 + i - delta) k`` (Sec 1.2), so the
investment LEVEL is ``I = i k = k' - (1 - delta) k``.
"""
from __future__ import annotations


def investment_level(i_rate, k):
    """I = i * k (Sec 1.2)."""
    return i_rate * k


def k_next(i_rate, k, delta):
    """k' = (1 + i - delta) k (Eq 10)."""
    return (1.0 + i_rate - delta) * k


def convex_cost(invest_level, k, gamma1):
    """Convex adjustment cost 0.5 * gamma1 * I**2 / k (Sec 1.4)."""
    return 0.5 * gamma1 * invest_level * invest_level / k
