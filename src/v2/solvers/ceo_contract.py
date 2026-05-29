"""Implicit upwind finite-difference solver for the CEO contract model.

Reproduces the Marinovic-Varas (2019) Internet Appendix scheme (Section III).
The HJB is solved in reverse time ``s = T - t`` for ``f(z, s) = F(z, T - s)``,

    f_s = max_{a in [0, a_max], sigma_z} { L^{a,sigma_z} f + pi(a, z) },
    L^{a,sigma_z} f = [(r+kappa)z + a r gamma (sigma sigma_z - 1)] f_z
                      + 1/2 sigma_z^2 f_zz - r f,

with Dirichlet conditions ``f(0, s) = 0`` and the absorbing-boundary value at
``z_max`` (where ``sigma_z = 0`` and ``a = (r+kappa) z_max / (r gamma)``), and the
terminal/initial condition ``f(z, 0) = -1/2 C z^2``.

Key point vs. a naive scheme: the per-node maximisation over ``(a, sigma_z)`` is
done by **grid search** over bounded control grids (uniform ``a``, ``sigma_z``
refined near 0), *not* by the closed-form FOC ``sigma_z = -r gamma sigma a F_z/F_zz``.
The FOC is unbounded where ``F_zz -> 0`` and, plugged into the explicit diffusion,
triggers a self-reinforcing terminal-layer collapse; the bounded grid search is
robust there and keeps ``F`` correctly concave.  Each implicit time step is solved
by policy iteration (monotone, converges to the unique viscosity solution per
Barles-Souganidis).  Final policies are then extracted in closed form from the
solved value function (smooth, since ``F_zz`` stays healthy).
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


def _control_grids(config: CEOContractGridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Flattened ``(a, sigma_z)`` control grids: uniform ``a``, ``sigma_z`` near 0."""

    a_axis = np.linspace(0.0, config.a_max, config.n_a)
    u = np.linspace(-1.0, 1.0, config.n_sigma)
    sigma_axis = config.sigma_z_max * np.sign(u) * u ** 2   # refined near sigma_z = 0
    A, S = np.meshgrid(a_axis, sigma_axis)
    return A.ravel(), S.ravel()


def _upwind_coeffs(
    A: np.ndarray, S: np.ndarray, z_int: np.ndarray, dz: float, params: CEOContractParams
) -> tuple[np.ndarray, np.ndarray]:
    """Positive-coefficient upwind weights alpha (on f_{i+1}) and rho (on f_{i-1}).

    Forward/backward switch chosen so both weights stay non-negative (monotone).
    Shapes broadcast controls ``(P,)`` against interior nodes ``(n_z-2,)``.
    """

    rg = params.r * params.gamma
    drift = (params.r + params.kappa) * z_int[None, :] + A[:, None] * rg * (
        params.sigma * S[:, None] - 1.0
    )
    diffusion = (S[:, None] ** 2) / (2.0 * dz ** 2)
    test = diffusion + drift / dz
    alpha = np.where(test > 0.0, diffusion + drift / dz, diffusion)
    rho = np.where(test > 0.0, diffusion, diffusion - drift / dz)
    return alpha, rho


def _upper_boundary_value(
    s: float, z_max: float, params: CEOContractParams
) -> float:
    """Absorbing-boundary value f(z_max, s) (Marinovic-Varas IA).

    At ``z_max`` the state is absorbed (``sigma_z = 0``, ``a = (r+kappa)z_max/(r
    gamma)``), giving ``f(z_max, s) = pi(a_max, m)/r (1 - e^{-rs}) - e^{-rs} 1/2 C
    z_max^2``.
    """

    a_b = (params.r + params.kappa) * z_max / (params.r * params.gamma)
    m_b = max(0.0, (a_b - params.phi * z_max) / params.g)
    pi_b = float(flow_payoff(np.array(a_b), np.array(m_b), np.array(z_max), params))
    disc = np.exp(-params.r * s)
    return pi_b / params.r * (1.0 - disc) - disc * 0.5 * params.C * z_max ** 2


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
    """Closed-form policy extraction from a solved value function ``V``.

    Alternates the decoupled effort FOC with the ``sigma_z`` FOC to a fixed point.
    Used only as a post-processing step on the grid-search solution, where ``F_zz``
    is healthy, so the ``1/F_zz`` in ``sigma_z`` is well behaved (no blow-up).
    """

    Fz_f, Fz_b, Fzz = _fd_derivatives(V, dz)
    Fzz = np.minimum(Fzz, config.vzz_floor)

    Fz_up = 0.5 * (Fz_f + Fz_b)
    sigma_z = np.zeros_like(z)
    a = np.zeros_like(z)
    for _ in range(n_inner):
        a_new = optimal_effort(z, Fz_up, sigma_z, params, a_max=config.a_max)
        sigma_z = optimal_sigma_z(a_new, Fz_up, Fzz, params)
        mu_z = drift_z(z, a_new, sigma_z, params)
        Fz_up = np.where(mu_z > 0.0, Fz_f, Fz_b)
        if np.max(np.abs(a_new - a)) < inner_tol:
            a = a_new
            break
        a = a_new

    m = optimal_manipulation(a, z, params)
    mu_z = drift_z(z, a, sigma_z, params)
    return {"a": a, "m": m, "sigma_z": sigma_z, "mu_z": mu_z, "Fz": Fz_up, "Fzz": Fzz}


def solve_ceo_contract(
    config: CEOContractGridConfig | None = None,
    params: CEOContractParams | None = None,
    *,
    tol: float = 1e-8,
    max_policy_iter: int = 200,
) -> dict[str, Any]:
    """Solve the CEO-contract HJB by the Marinovic-Varas IA grid-search scheme.

    Returns a dict with the value function ``value`` of shape ``(n_t, n_z)``
    indexed by forward time ``t`` (``value[-1]`` is the terminal ``t = T``), the
    semi-parametric ``policy`` fields (``a``, ``m``, ``sigma_z``) of the same
    shape, the ``grids``, the max interior HJB residual, a ``converged`` flag, and
    a ``diagnostics`` dict.
    """

    config = config or CEOContractGridConfig()
    params = params or CEOContractParams()
    grids = build_ceo_grids(config, params)

    z = grids["z"]
    dz = grids["dz"]
    ds = grids["dt"]            # reverse-time step s = T - t
    n_z = z.size
    n_t = config.n_t
    z_max = float(z[-1])
    r = params.r

    A, S = _control_grids(config)
    z_int = z[1:-1]
    alpha, rho = _upwind_coeffs(A, S, z_int, dz, params)            # (P, n_z-2)
    m_ctrl = optimal_manipulation(A[:, None], z_int[None, :], params)
    payoff_ctrl = flow_payoff(A[:, None], m_ctrl, z_int[None, :], params)  # (P, n_z-2)

    value = np.empty((n_t, n_z), dtype=np.float64)
    f = terminal_value(z, params)        # f(z, 0), i.e. t = T
    value[n_t - 1] = f

    idx = np.arange(n_z - 2)
    max_iter_used = 0
    n_not_converged = 0

    for ns in range(1, n_t):
        s = ns * ds
        f_up = _upper_boundary_value(s, z_max, params)
        f_it = f.copy()
        converged_step = False
        for it in range(max_policy_iter):
            f_im1, f_i, f_ip1 = f_it[:-2], f_it[1:-1], f_it[2:]
            # policy improvement: argmax over control pairs of (L f)_i + pi
            val = alpha * f_ip1 + rho * f_im1 - (alpha + rho + r) * f_i + payoff_ctrl
            kbest = np.argmax(val, axis=0)
            al = alpha[kbest, idx]
            rh = rho[kbest, idx]
            pa = payoff_ctrl[kbest, idx]
            # implicit step (1 - ds L) f^{k+1} = f^n + ds pi on the interior
            diag = 1.0 + ds * (al + rh + r)
            lhs = sp.diags(
                [(-ds * rh)[1:], diag, (-ds * al)[:-1]], offsets=[-1, 0, 1], format="csc"
            )
            rhs = f[1:-1] + ds * pa
            rhs[-1] += ds * al[-1] * f_up        # upper Dirichlet f(z_max, s)
            f_int = spsolve(lhs, rhs)            # lower node pinned to 0 implicitly
            f_new = f.copy()
            f_new[0] = 0.0
            f_new[-1] = f_up
            f_new[1:-1] = f_int
            rel = np.max(
                np.abs(f_new[1:-1] - f_it[1:-1]) / np.maximum(1.0, np.abs(f_new[1:-1]))
            )
            f_it = f_new
            if rel < tol:
                converged_step = True
                max_iter_used = max(max_iter_used, it + 1)
                break
        if not converged_step:
            n_not_converged += 1
            max_iter_used = max(max_iter_used, max_policy_iter)
        f = f_it
        value[n_t - 1 - ns] = f          # s = ns*ds maps to t = T - s

    # closed-form policy extraction from the solved value function
    pol_a = np.empty((n_t, n_z), dtype=np.float64)
    pol_m = np.empty((n_t, n_z), dtype=np.float64)
    pol_s = np.empty((n_t, n_z), dtype=np.float64)
    for n in range(n_t):
        pol = _update_policies(z, value[n], dz, params, config)
        pol_a[n], pol_m[n], pol_s[n] = pol["a"], pol["m"], pol["sigma_z"]

    hjb_max, hjb_median, min_fzz, max_fzz = _hjb_diagnostics(value, grids, params, config)

    return {
        "value": value,
        "policy": {"a": pol_a, "m": pol_m, "sigma_z": pol_s},
        "grids": grids,
        "hjb_residual": hjb_max,
        "converged": n_not_converged == 0,
        "diagnostics": {
            "hjb_residual_max": hjb_max,
            "hjb_residual_median": hjb_median,
            "min_interior_fzz": min_fzz,
            "max_interior_fzz": max_fzz,
            "max_abs_sigma_z": float(np.max(np.abs(pol_s))),
            "max_policy_iter_used": max_iter_used,
            "n_steps_not_converged": n_not_converged,
            "tol": tol,
        },
    }


def _hjb_diagnostics(
    value: np.ndarray,
    grids: dict[str, np.ndarray],
    params: CEOContractParams,
    config: CEOContractGridConfig,
) -> tuple[float, float, float, float]:
    """Interior HJB residual (max and median) and the interior F_zz range.

    The max is typically attained at the manipulation-kink node, where the value
    function has a curvature jump and central differences are unreliable; the
    median reflects bulk accuracy.  The Monte-Carlo value reconciliation is the
    authoritative correctness check.
    """

    z = grids["z"]
    dz = grids["dz"]
    dt = grids["dt"]
    n_t = value.shape[0]

    residuals = []
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
        residuals.append(np.abs(lhs - rhs)[1:-1])  # interior nodes only
        _, _, fzz_raw = _fd_derivatives(value[n], dz)
        min_fzz = min(min_fzz, float(fzz_raw[1:-1].min()))
        max_fzz = max(max_fzz, float(fzz_raw[1:-1].max()))

    res = np.concatenate(residuals) if residuals else np.zeros(1)
    return float(res.max()), float(np.median(res)), float(min_fzz), float(max_fzz)
