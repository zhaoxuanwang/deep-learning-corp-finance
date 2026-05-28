"""Implicit upwind finite-difference solver (FD-PFI) for the CEO contract model.

Solves the HJB equation of the Marinovic-Varas (2019) CEO short-termism model
(``docs/paper/ceo_contract.md``) backward in time on ``[0, T]``:

    r F = max_{a, sigma_z} pi(a, z) + F_t + mu_z(a, sigma_z) F_z + 1/2 sigma_z^2 F_zz

with terminal condition ``F(z, T) = -1/2 C z^2``.  Each backward time step is a
backward-Euler implicit solve wrapped in a policy-iteration ("Howard") fixed
point: the semi-explicit policies ``a, m, sigma_z`` are recomputed from the
current value iterate, the upwind generator is assembled, and the implicit
linear system is solved, until the value iterate stops moving.  The RHS of the
implicit system freezes the *previous time level* ``V_prev`` throughout the inner
loop (correcting the drifting-RHS in the doc's pseudo-code).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from src.v2.environments.ceo_contract import (
    CEOContractGridConfig,
    CEOContractParams,
    build_ceo_grids,
    drift_z,
    flow_payoff,
    optimal_effort,
    optimal_manipulation,
    optimal_sigma_z,
    terminal_value,
)


def _fd_derivatives(
    V: np.ndarray, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward/backward first differences and the central second difference."""

    n = V.size
    Fz_f = np.empty(n, dtype=np.float64)
    Fz_b = np.empty(n, dtype=np.float64)
    Fzz = np.empty(n, dtype=np.float64)

    Fz_f[:-1] = (V[1:] - V[:-1]) / dz
    Fz_f[-1] = (V[-1] - V[-2]) / dz
    Fz_b[1:] = (V[1:] - V[:-1]) / dz
    Fz_b[0] = (V[1] - V[0]) / dz

    Fzz[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dz ** 2
    Fzz[0] = Fzz[1]
    Fzz[-1] = Fzz[-2]
    return Fz_f, Fz_b, Fzz


def _update_policies(
    z: np.ndarray,
    V: np.ndarray,
    dz: float,
    params: CEOContractParams,
    config: CEOContractGridConfig,
    *,
    n_inner: int = 60,
    inner_tol: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compute the upwind-consistent optimal policies and derivatives from V.

    Alternates the decoupled effort FOC with the ``sigma_z`` FOC (capping
    ``|sigma_z|``) and reselects the upwind first derivative each pass, to a
    fixed point.  This resolves the mutual a/sigma_z dependence robustly.
    """

    Fz_f, Fz_b, Fzz = _fd_derivatives(V, dz)
    Fzz = np.minimum(Fzz, config.vzz_floor)

    Fz_up = 0.5 * (Fz_f + Fz_b)
    sigma_z = np.zeros_like(z)
    a = np.zeros_like(z)
    for _ in range(n_inner):
        a_new = optimal_effort(z, Fz_up, sigma_z, params, a_max=config.a_max)
        sigma_z = np.clip(
            optimal_sigma_z(a_new, Fz_up, Fzz, params),
            -config.sigma_z_max,
            config.sigma_z_max,
        )
        mu_z = drift_z(z, a_new, sigma_z, params)
        Fz_up = np.where(mu_z > 0.0, Fz_f, Fz_b)
        if np.max(np.abs(a_new - a)) < inner_tol:
            a = a_new
            break
        a = a_new

    m = optimal_manipulation(a, z, params)
    mu_z = drift_z(z, a, sigma_z, params)
    return {"a": a, "m": m, "sigma_z": sigma_z, "mu_z": mu_z, "Fz": Fz_up, "Fzz": Fzz}


def _build_generator(
    mu_z: np.ndarray, sigma_z: np.ndarray, dz: float
) -> sp.csc_matrix:
    """Upwind tridiagonal generator with linear (zero-curvature) boundaries.

    Boundary nodes use a ghost point set by linear extrapolation
    (``V[-1] = 2 V[0] - V[1]``, ``V[N] = 2 V[N-1] - V[N-2]``), i.e. ``F_zz = 0``
    at the edges.  This avoids the spurious convex boundary layer that a
    zero-flux (reflecting) condition produces when the terminal slope is nonzero.
    Rows still sum to zero, so the matrix remains a valid (conservative) generator.
    """

    diffusion = 0.5 * sigma_z ** 2 / dz ** 2
    lower = np.maximum(-mu_z, 0.0) / dz + diffusion  # coefficient on V[i-1]
    upper = np.maximum(mu_z, 0.0) / dz + diffusion   # coefficient on V[i+1]
    diag = -(lower + upper)

    # Fold the linear-extrapolation ghost nodes into the boundary rows.
    diag[0] += 2.0 * lower[0]
    upper[0] -= lower[0]
    lower[0] = 0.0

    diag[-1] += 2.0 * upper[-1]
    lower[-1] -= upper[-1]
    upper[-1] = 0.0

    return sp.diags(
        [lower[1:], diag, upper[:-1]],
        offsets=[-1, 0, 1],
        format="csc",
    )


def solve_ceo_contract(
    config: CEOContractGridConfig | None = None,
    params: CEOContractParams | None = None,
    *,
    howard_tol: float = 1e-8,
    max_howard_iter: int = 500,
) -> dict[str, Any]:
    """Solve the CEO-contract HJB by backward-Euler FD with Howard iteration.

    Returns a dict with the numerical value function ``value`` of shape
    ``(n_t, n_z)``, the semi-parametric ``policy`` fields (``a``, ``m``,
    ``sigma_z``) of the same shape, the ``grids``, the maximum interior HJB
    residual, a ``converged`` flag, and a ``diagnostics`` dict.
    """

    config = config or CEOContractGridConfig()
    params = params or CEOContractParams()
    grids = build_ceo_grids(config, params)

    z = grids["z"]
    dz = grids["dz"]
    dt = grids["dt"]
    n_z = z.size
    n_t = config.n_t

    value = np.empty((n_t, n_z), dtype=np.float64)
    pol_a = np.empty((n_t, n_z), dtype=np.float64)
    pol_m = np.empty((n_t, n_z), dtype=np.float64)
    pol_s = np.empty((n_t, n_z), dtype=np.float64)

    identity = sp.identity(n_z, format="csc")
    one_plus_rdt = 1.0 + params.r * dt

    value[-1] = terminal_value(z, params)
    term_pol = _update_policies(z, value[-1], dz, params, config)
    pol_a[-1], pol_m[-1], pol_s[-1] = term_pol["a"], term_pol["m"], term_pol["sigma_z"]

    max_iter_used = 0
    n_not_converged = 0

    for n in range(n_t - 2, -1, -1):
        v_prev = value[n + 1]
        v_it = v_prev.copy()
        converged_step = False
        for it in range(max_howard_iter):
            pol = _update_policies(z, v_it, dz, params, config)
            generator = _build_generator(pol["mu_z"], pol["sigma_z"], dz)
            payoff = flow_payoff(pol["a"], pol["m"], z, params)
            lhs = (one_plus_rdt * identity - dt * generator).tocsc()
            v_new = spsolve(lhs, v_prev + dt * payoff)
            diff = float(np.max(np.abs(v_new - v_it)))
            v_it = v_new
            if diff < howard_tol:
                converged_step = True
                max_iter_used = max(max_iter_used, it + 1)
                break
        if not converged_step:
            n_not_converged += 1
            max_iter_used = max(max_iter_used, max_howard_iter)

        value[n] = v_it
        pol = _update_policies(z, value[n], dz, params, config)
        pol_a[n], pol_m[n], pol_s[n] = pol["a"], pol["m"], pol["sigma_z"]

    hjb_residual_max, min_fzz, max_fzz = _hjb_diagnostics(value, grids, params, config)

    return {
        "value": value,
        "policy": {"a": pol_a, "m": pol_m, "sigma_z": pol_s},
        "grids": grids,
        "hjb_residual": hjb_residual_max,
        "converged": n_not_converged == 0,
        "diagnostics": {
            "hjb_residual_max": hjb_residual_max,
            "min_interior_fzz": min_fzz,
            "max_interior_fzz": max_fzz,
            "max_howard_iter_used": max_iter_used,
            "n_steps_not_converged": n_not_converged,
            "howard_tol": howard_tol,
        },
    }


def _hjb_diagnostics(
    value: np.ndarray,
    grids: dict[str, np.ndarray],
    params: CEOContractParams,
    config: CEOContractGridConfig,
) -> tuple[float, float, float]:
    """Max absolute interior HJB residual and the interior F_zz range."""

    z = grids["z"]
    dz = grids["dz"]
    dt = grids["dt"]
    n_t = value.shape[0]

    max_residual = 0.0
    min_fzz = np.inf
    max_fzz = -np.inf
    for n in range(n_t - 1):
        pol = _update_policies(z, value[n], dz, params, config)
        payoff = flow_payoff(pol["a"], pol["m"], z, params)
        v_t = (value[n + 1] - value[n]) / dt
        lhs = params.r * value[n]
        rhs = (
            payoff
            + v_t
            + pol["mu_z"] * pol["Fz"]
            + 0.5 * pol["sigma_z"] ** 2 * pol["Fzz"]
        )
        residual = np.abs(lhs - rhs)[1:-1]  # interior nodes only
        if residual.size:
            max_residual = max(max_residual, float(residual.max()))
        _, _, fzz_raw = _fd_derivatives(value[n], dz)
        min_fzz = min(min_fzz, float(fzz_raw[1:-1].min()))
        max_fzz = max(max_fzz, float(fzz_raw[1:-1].max()))

    return max_residual, float(min_fzz), float(max_fzz)
