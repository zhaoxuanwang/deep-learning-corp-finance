"""Primitives and grids for the CEO short-termism contract (Marinovic-Varas 2019).

Continuous-time dynamic principal-agent model summarised in
``docs/paper/ceo_contract.md``.  The contract reduces to a single state ``z``
(duration of deferred pay) governed by an HJB equation on ``[0, T]``.  This
module holds the economic primitives (flow payoff and the semi-explicit optimal
policies ``a``, ``m``, ``sigma_z``) plus the ``(z, t)`` grid builder.  It is the
single source of truth shared by the FD-PFI solver and the SDE simulator.

Like the Nikolov LP baselines, this is intentionally *not* an ``MDPEnvironment``:
the model is a continuous-time finite-difference problem, not a discrete-time MDP.

Sign conventions (verified against the value function ``F``, concave with
``F_z <= 0``, ``F_zz < 0``):
    * ``sigma_z <= 0``  -- positive shocks accelerate vesting (reduce z).
    * ``m >= 0``        -- manipulation floored at zero.
    * ``a >= 0``        -- productive effort is non-negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CEOContractParams:
    """Economic parameters of the Marinovic-Varas CEO contract model.

    NOTE: the defaults are PLACEHOLDERS chosen only to yield a well-behaved HJB
    for smoke-testing the pipeline.  Replace with the calibrated values before
    drawing economic conclusions.  ``__post_init__`` enforces sign/range
    constraints regardless of the numbers supplied.
    """

    r: float = 0.10        # risk-free rate / discount rate
    gamma: float = 5.0     # CARA risk aversion
    sigma: float = 1.0     # exogenous cash-flow volatility
    kappa: float = 0.30    # manipulation-stock depreciation rate
    theta: float = 0.50    # marginal effect of manipulation on current cash flow
    g: float = 1.0         # manipulation cost coefficient g(m) = g m^2 / 2
    T: float = 10.0        # deterministic retirement date
    tau: float = 2.0       # clawback horizon (post-retirement); informational
    C: float = 1.0         # terminal cost coefficient: F(z, T) = -1/2 C z^2

    def __post_init__(self) -> None:
        if self.r <= 0.0:
            raise ValueError(f"r must be > 0. Got {self.r}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0. Got {self.gamma}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0. Got {self.sigma}")
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be >= 0. Got {self.kappa}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be > 0. Got {self.theta}")
        if self.g <= 0.0:
            raise ValueError(f"g must be > 0. Got {self.g}")
        if self.T <= 0.0:
            raise ValueError(f"T must be > 0. Got {self.T}")
        if self.tau < 0.0:
            raise ValueError(f"tau must be >= 0. Got {self.tau}")
        if self.C < 0.0:
            raise ValueError(f"C must be >= 0. Got {self.C}")

    @property
    def lambda_(self) -> float:
        """Value-destroying effect of manipulation: theta / (r + kappa) - 1."""

        return self.theta / (self.r + self.kappa) - 1.0

    @property
    def phi(self) -> float:
        """Manipulation deterrence coefficient: theta / (r * gamma)."""

        return self.theta / (self.r * self.gamma)


@dataclass(frozen=True)
class CEOContractGridConfig:
    """Finite-difference grid and numerical-guard configuration."""

    z_bounds: tuple[float, float] = (0.0, 2.0)
    n_z: int = 201
    n_t: int = 201

    vzz_floor: float = -1e-6     # enforce strict concavity for matrix stability
    a_max: float = 1.0e3         # safety cap on effort magnitude
    sigma_z_max: float = 50.0    # safety cap on |sigma_z| (bounds the diffusion)

    def __post_init__(self) -> None:
        low, high = map(float, self.z_bounds)
        if not (low < high):
            raise ValueError(f"z_bounds must satisfy low < high. Got {self.z_bounds}")
        if self.n_z < 3:
            raise ValueError(f"n_z must be >= 3. Got {self.n_z}")
        if self.n_t < 2:
            raise ValueError(f"n_t must be >= 2. Got {self.n_t}")
        if self.vzz_floor >= 0.0:
            raise ValueError(f"vzz_floor must be < 0. Got {self.vzz_floor}")
        if self.a_max <= 0.0:
            raise ValueError(f"a_max must be > 0. Got {self.a_max}")
        if self.sigma_z_max <= 0.0:
            raise ValueError(f"sigma_z_max must be > 0. Got {self.sigma_z_max}")


def build_ceo_grids(
    config: CEOContractGridConfig,
    params: CEOContractParams,
) -> dict[str, np.ndarray]:
    """Build the uniform ``z`` grid over ``z_bounds`` and ``t`` grid over ``[0, T]``."""

    low, high = map(float, config.z_bounds)
    z_grid = np.linspace(low, high, config.n_z, dtype=np.float64)
    t_grid = np.linspace(0.0, params.T, config.n_t, dtype=np.float64)
    return {
        "z": z_grid,
        "t": t_grid,
        "dz": float((high - low) / (config.n_z - 1)),
        "dt": float(params.T / (config.n_t - 1)),
    }


def flow_payoff(
    a: np.ndarray,
    m: np.ndarray,
    z: np.ndarray,
    params: CEOContractParams,
) -> np.ndarray:
    """Principal flow payoff pi(a, m, z) entering the HJB.

    pi = a - lambda*m - a^2/2 - g*m^2/2 - sigma^2*(r*gamma*a)^2/(2*r*gamma),
    with the last term equal to sigma^2 * r * gamma * a^2 / 2 (the risk premium
    cost of the short-term incentive beta = r*gamma*a).
    """

    a = np.asarray(a, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    risk_premium = 0.5 * params.sigma ** 2 * params.r * params.gamma * a ** 2
    return (
        a
        - params.lambda_ * m
        - 0.5 * a ** 2
        - 0.5 * params.g * m ** 2
        - risk_premium
    )


def optimal_manipulation(
    a: np.ndarray,
    z: np.ndarray,
    params: CEOContractParams,
) -> np.ndarray:
    """Optimal manipulation m(a, z) = (1/g) * max(a - phi*z, 0)."""

    a = np.asarray(a, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(a - params.phi * z, 0.0) / params.g


def optimal_sigma_z(
    a: np.ndarray,
    Fz: np.ndarray,
    Fzz: np.ndarray,
    params: CEOContractParams,
) -> np.ndarray:
    """Optimal vesting sensitivity sigma_z = -r*gamma*sigma*a*F_z/F_zz.

    Caller must supply ``Fzz`` already clamped strictly negative.
    """

    a = np.asarray(a, dtype=np.float64)
    Fz = np.asarray(Fz, dtype=np.float64)
    Fzz = np.asarray(Fzz, dtype=np.float64)
    return -params.r * params.gamma * params.sigma * a * (Fz / Fzz)


def optimal_effort(
    z: np.ndarray,
    Fz: np.ndarray,
    sigma_z: np.ndarray,
    params: CEOContractParams,
    *,
    a_max: float = 1.0e3,
) -> np.ndarray:
    """Optimal effort best-response a(z, F_z, sigma_z), handling the kink.

    This is the effort FOC *given* ``sigma_z`` (the decoupled form), whose
    denominator is always positive.  Iterated with :func:`optimal_sigma_z` it
    converges to the doc's substituted policy 3 wherever the second-order
    condition holds, but it is numerically robust where the substituted form's
    ``F_z^2 / F_zz`` denominator would blow up.

    Two regimes separated by the manipulation floor at ``a = phi*z``:
      * interior (m > 0): the full effort FOC.
      * boundary (m = 0): the g-cost-of-m terms drop from the FOC.
    The interior maximiser is used where it exceeds ``phi*z``; the boundary
    maximiser where it falls below ``phi*z``; otherwise the optimum sits at the
    kink ``a = phi*z`` (manipulation just deterred).
    """

    z = np.asarray(z, dtype=np.float64)
    Fz = np.asarray(Fz, dtype=np.float64)
    sigma_z = np.asarray(sigma_z, dtype=np.float64)

    rg = params.r * params.gamma
    rgs2 = rg * params.sigma ** 2
    drift_term = rg * (params.sigma * sigma_z - 1.0) * Fz

    a_int = (params.g - params.lambda_ + params.phi * z + params.g * drift_term) / (
        1.0 + params.g * (1.0 + rgs2)
    )
    a_0 = (1.0 + drift_term) / (1.0 + rgs2)

    phi_z = params.phi * z
    a = np.where(a_int > phi_z, a_int, np.where(a_0 < phi_z, a_0, phi_z))
    return np.clip(a, 0.0, a_max)


def drift_z(
    z: np.ndarray,
    a: np.ndarray,
    sigma_z: np.ndarray,
    params: CEOContractParams,
) -> np.ndarray:
    """Drift of the state z: (r+kappa)*z + a*r*gamma*(sigma*sigma_z - 1)."""

    z = np.asarray(z, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    sigma_z = np.asarray(sigma_z, dtype=np.float64)
    return (params.r + params.kappa) * z + a * params.r * params.gamma * (
        params.sigma * sigma_z - 1.0
    )


def terminal_value(z: np.ndarray, params: CEOContractParams) -> np.ndarray:
    """Terminal condition F(z, T) = -1/2 C z^2."""

    z = np.asarray(z, dtype=np.float64)
    return -0.5 * params.C * z ** 2
