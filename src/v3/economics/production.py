"""Technology, operating surplus, and steady-state capital (DF26 Sec 1.3, 3.2).

All functions follow the dtype of their tensor inputs. ``theta`` and ``alpha`` may
be Python floats or tensors (batched over parameter vectors).
"""
from __future__ import annotations


def nu(theta, alpha):
    """nu = 1 - (1 - alpha) * theta (Eq 3)."""
    return 1.0 - (1.0 - alpha) * theta


def xi(theta, alpha):
    """xi = alpha * theta / nu (the capital exponent in operating surplus, Eq 3)."""
    return alpha * theta / nu(theta, alpha)


def a_pi(theta, alpha):
    """A_pi = nu * ((1-alpha) theta) ** ((1-alpha) theta / nu) (Sec 1.3, w = 1)."""
    n = nu(theta, alpha)
    p = (1.0 - alpha) * theta
    return n * p ** (p / n)


def gross_output(z, k, theta, alpha):
    """z * A_pi * k**xi: the reduced-form output term used in d2 (Eq 8)."""
    return z * a_pi(theta, alpha) * k ** xi(theta, alpha)


def operating_surplus(z, k, theta, alpha, cf):
    """pi(z, k) = z * A_pi * k**xi - cf (Eq 3)."""
    return gross_output(z, k, theta, alpha) - cf


def k_steady_state(z, theta, alpha, delta, rf):
    """Frictionless steady-state capital with z held permanently (Sec 3.2):

    k_ss(z) = [xi * z * A_pi / (delta + rf)] ** (1 / (1 - xi)),
    the solution of the user-cost condition xi z A_pi k**(xi-1) = delta + rf.
    """
    x = xi(theta, alpha)
    return (x * z * a_pi(theta, alpha) / (delta + rf)) ** (1.0 / (1.0 - x))
