"""Risky-debt nested VFI solver with z-interpolation and adaptive b' bounds.

Solves the Bellman equation on a coarse z grid (n_z_solve nodes, a subset
of the fine Tauchen grid) and uses 1-D linear interpolation in z for:

  1. Continuation values — V interpolated from coarse to fine z for
     accurate Tauchen quadrature.
  2. Pricing updates — V at fine z for accurate default-boundary detection.

The inner-loop effective transition matrix is::

    M_eff = P_sub @ W          (n_z_solve, n_z_solve)

where::

    P_sub = P_full[coarse_idx]  (n_z_solve, n_z)
    W     = interp matrix       (n_z, n_z_solve)

The reward tensor and argmax are computed only at n_z_solve current-z
nodes, giving a linear speedup proportional to n_z / n_z_solve in the
inner loop.  Pricing updates retain full fine-z resolution.

After convergence, a single-pass re-evaluation at full z resolution
produces output in the same format as ``solve_nested_vfi()``.

When ``config.adaptive=True`` (default), the solver first runs a fast
internal boundary-discovery solve on a small grid to find the b' range
where default risk is non-trivial, then re-solves on tight b' bounds.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict

import numpy as np

from src.v2.solvers.config import RiskyDebtSolverConfig
from src.v2.solvers.grid import (
    GridAxis,
    build_1d_grid,
    build_product_grid,
    tauchen_transition_matrix,
)
from src.v2.solvers.nested_vfi_np import (
    _compute_pricing_update_np,
    _extract_policy_np,
    _is_risk_free_equilibrium,
    _pricing_discount_sup_norm_diff,
    _reward_tensor_np,
    _solve_inner_bellman_np,
    _zero_profit_residual_np,
)


# ======================================================================
# Module-level defaults
# ======================================================================

# Convergence — generous enough that both loops always converge.
# Use coarser grids for faster runs, not fewer iterations.
MAX_INNER = 2000
MAX_OUTER = 500
TOL_INNER = 1e-6
TOL_OUTER = 1e-6

# Pessimistic initial risky rate for positive-debt cells.
AUTARKY_RATE = 10.0

# Boundary discovery grid (internal, not user-configured).
BOUNDARY_N_K = 15
BOUNDARY_N_B = 15
BOUNDARY_N_Z = 15
BOUNDARY_N_Z_SOLVE = 6


# ======================================================================
# Grid construction (risky-debt specific)
# ======================================================================


def _build_risky_debt_grids(env, n_k, n_b, n_z, k_spacing, b_spacing, z_spacing):
    """Build state/choice grids for the risky-debt model."""

    k_axis = GridAxis(env.k_min, env.k_max, spacing=k_spacing)
    b_axis = GridAxis(env.b_min, env.b_max, spacing=b_spacing)
    z_axis = GridAxis(env.z_min, env.z_max, spacing=z_spacing)

    k_grid = build_1d_grid(k_axis, n_k)
    b_grid = build_1d_grid(b_axis, n_b)
    z_grid = build_1d_grid(z_axis, n_z)

    endo_product = build_product_grid([k_grid, b_grid])

    return {
        "exo_grids_1d": [z_grid],
        "endo_grids_1d": [k_grid, b_grid],
        "action_grids_1d": [k_grid, b_grid],
        "choice_grids_1d": [k_grid, b_grid],
        "exo_product": build_product_grid([z_grid]),
        "endo_product": endo_product,
        "choice_product": endo_product,
    }


# ======================================================================
# z-interpolation helpers
# ======================================================================


def _select_coarse_indices(n_z: int, n_z_solve: int) -> np.ndarray:
    """Select *n_z_solve* uniformly-spaced indices from ``[0, n_z-1]``.

    Always includes endpoints.
    """
    if n_z_solve >= n_z:
        return np.arange(n_z)
    return np.unique(np.round(np.linspace(0, n_z - 1, n_z_solve)).astype(int))


def _build_z_interp_matrix(
    z_coarse: np.ndarray,
    z_fine: np.ndarray,
) -> np.ndarray:
    """Linear-interpolation matrix ``W`` such that ``V_fine ≈ W @ V_coarse``.

    Returns shape ``(n_z, n_z_solve)``.
    """
    n_fine = len(z_fine)
    n_coarse = len(z_coarse)
    W = np.zeros((n_fine, n_coarse), dtype=np.float64)

    for l in range(n_fine):
        zf = z_fine[l]
        idx_hi = int(np.searchsorted(z_coarse, zf, side="right"))
        idx_hi = min(idx_hi, n_coarse - 1)
        idx_lo = max(idx_hi - 1, 0)

        if idx_lo == idx_hi:
            W[l, idx_lo] = 1.0
        else:
            span = z_coarse[idx_hi] - z_coarse[idx_lo]
            frac = (zf - z_coarse[idx_lo]) / max(span, 1e-15)
            frac = np.clip(frac, 0.0, 1.0)
            W[l, idx_lo] = 1.0 - frac
            W[l, idx_hi] = frac

    return W


# ======================================================================
# Pricing update (coarse current-z, fine future-z')
# ======================================================================


def _compute_pricing_update_interp(
    env,
    grids: Dict[str, np.ndarray],
    prob_sub: np.ndarray,
    z_fine: np.ndarray,
    value_fine_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update ``r_tilde`` using fine-z value function at coarse-z nodes.

    Default boundary is detected on the full fine-z grid for accuracy.
    Expectations use ``prob_sub`` (coarse rows of the Tauchen matrix)
    so the resulting ``r_tilde`` has shape ``(n_z_solve, n_k, n_b)``.
    """
    k_grid = np.asarray(grids["endo_grids_1d"][0], dtype=np.float64)
    b_grid = np.asarray(grids["endo_grids_1d"][1], dtype=np.float64)
    n_z_fine = len(z_fine)
    n_z_solve = prob_sub.shape[0]
    n_k = len(k_grid)
    n_b = len(b_grid)

    default_mask = np.asarray(value_fine_3d, dtype=np.float64) <= 0.0

    recovery = env.recovery_value(
        k_grid[None, :, None],
        z_fine[:, None, None],
    )
    recovery_flat = np.repeat(recovery, n_b, axis=2).reshape(n_z_fine, -1)
    default_flat = default_mask.reshape(n_z_fine, -1).astype(np.float64)
    solvent_flat = 1.0 - default_flat

    expected_recovery = prob_sub @ (default_flat * recovery_flat)
    solvent_probability = prob_sub @ solvent_flat

    b_flat = np.asarray(grids["choice_product"], dtype=np.float64)[:, 1][None, :]
    b_matrix = np.broadcast_to(b_flat, expected_recovery.shape)
    r_rf = env.econ.interest_rate
    b_positive = b_flat > 1e-12
    funded = b_positive & (solvent_probability > 1e-12)

    r_tilde_new = np.full_like(expected_recovery, np.inf, dtype=np.float64)
    r_tilde_new[:, ~b_positive.reshape(-1)] = r_rf
    r_tilde_new[funded] = (
        (1.0 + r_rf) - expected_recovery[funded] / b_matrix[funded]
    ) / solvent_probability[funded] - 1.0

    return (
        r_tilde_new.reshape(n_z_solve, n_k, n_b),
        default_mask,
        expected_recovery.reshape(n_z_solve, n_k, n_b),
        solvent_probability.reshape(n_z_solve, n_k, n_b),
    )


# ======================================================================
# Core solver (single solve at given grids)
# ======================================================================


def _solve_single(
    env,
    n_k, n_b, n_z, n_z_solve,
    k_spacing, b_spacing, z_spacing,
    value_init=None,
    r_tilde_init=None,
    eval_callback=None,
):
    """Run one nested VFI solve with z-interpolation. Internal workhorse."""

    grids = _build_risky_debt_grids(env, n_k, n_b, n_z, k_spacing, b_spacing, z_spacing)
    z_fine = np.asarray(grids["exo_grids_1d"][0], dtype=np.float64)
    n_z_fine = len(z_fine)

    coarse_indices = _select_coarse_indices(n_z_fine, n_z_solve)
    n_z_coarse = len(coarse_indices)
    z_coarse = z_fine[coarse_indices]

    # Transition matrices
    prob_full = tauchen_transition_matrix(env.shocks, grids["exo_grids_1d"])
    prob_sub = prob_full[coarse_indices]  # (n_z_coarse, n_z_fine)

    # Interpolation: V_fine ≈ W @ V_coarse
    W = _build_z_interp_matrix(z_coarse, z_fine)  # (n_z_fine, n_z_coarse)
    M_eff = prob_sub @ W  # (n_z_coarse, n_z_coarse)

    # Dimensions
    endo_product = grids["endo_product"]
    n_state = endo_product.shape[0]
    b_grid = grids["endo_grids_1d"][1]

    # Initialization (coarse z)
    if value_init is not None:
        value_full = np.asarray(value_init, dtype=np.float64).reshape(
            n_z_fine, n_state
        )
        value = value_full[coarse_indices]
    else:
        value = np.zeros((n_z_coarse, n_state), dtype=np.float64)

    if r_tilde_init is not None:
        r_full = np.asarray(r_tilde_init, dtype=np.float64).reshape(
            n_z_fine, n_k, n_b
        )
        r_tilde_grid = r_full[coarse_indices]
    else:
        r_tilde_grid = np.full(
            (n_z_coarse, n_k, n_b),
            env.econ.interest_rate,
            dtype=np.float64,
        )
        r_tilde_grid[:, :, b_grid > 1e-12] = AUTARKY_RATE

    # Shallow copy of grids with z_coarse for reward computation
    grids_coarse = dict(grids)
    grids_coarse["exo_grids_1d"] = [z_coarse]

    # ------------------------------------------------------------------
    # Solve (nested loop on coarse z)
    # ------------------------------------------------------------------
    start_time = time.perf_counter()
    inner_diff_history: list[float] = []
    outer_value_diff_history: list[float] = []
    pricing_diff_history: list[float] = []
    eval_history: Dict = {}
    converged_inner = False
    converged_outer = False
    n_inner_last = 0
    stop_reason = "max_outer"
    n_outer = 0
    inner_diff = float("inf")

    # Initial inner solve
    reward = _reward_tensor_np(env, grids_coarse, r_tilde_grid)
    (
        value,
        policy_idx,
        converged_inner,
        n_inner_last,
        inner_diff,
        inner_history_local,
    ) = _solve_inner_bellman_np(
        env, M_eff, reward, value, MAX_INNER, TOL_INNER,
    )
    inner_diff_history.extend(inner_history_local)

    if not converged_inner:
        stop_reason = "max_inner"
    else:
        value_prev_outer = value.copy()

        for outer in range(MAX_OUTER):
            # Interpolate V to fine z for pricing
            value_fine = W @ value_prev_outer
            value_fine_3d = value_fine.reshape(n_z_fine, n_k, n_b)

            (
                r_tilde_new,
                _default_mask_fine,
                _expected_recovery,
                _solvent_probability,
            ) = _compute_pricing_update_interp(
                env, grids, prob_sub, z_fine, value_fine_3d,
            )
            pricing_diff = _pricing_discount_sup_norm_diff(
                r_tilde_new, r_tilde_grid,
            )

            reward = _reward_tensor_np(env, grids_coarse, r_tilde_new)
            (
                value,
                policy_idx,
                converged_inner,
                n_inner_last,
                inner_diff,
                inner_history_local,
            ) = _solve_inner_bellman_np(
                env,
                M_eff,
                reward,
                value_prev_outer,
                MAX_INNER,
                TOL_INNER,
            )
            inner_diff_history.extend(inner_history_local)
            outer_value_diff = float(
                np.max(np.abs(value - value_prev_outer))
            )

            outer_value_diff_history.append(outer_value_diff)
            pricing_diff_history.append(pricing_diff)
            n_outer = outer + 1
            r_tilde_grid = r_tilde_new

            if eval_callback is not None:
                value_fine_cb = W @ value
                value_fine_3d_cb = value_fine_cb.reshape(n_z_fine, n_k, n_b)
                default_mask_cb = value_fine_3d_cb <= 0.0
                policy_action_cb, policy_endo_cb = _extract_policy_np(
                    env, grids_coarse, policy_idx,
                )
                partial_result = {
                    "value": value_fine_cb,
                    "policy_action": policy_action_cb,
                    "policy_endo": policy_endo_cb,
                    "grids": grids,
                    "r_tilde_grid": r_tilde_grid,
                    "default_mask": default_mask_cb,
                    "backend": "numpy",
                    "dtype": "float64",
                    "device": "CPU:0",
                }
                metrics = eval_callback(outer, partial_result) or {}
                elapsed = time.perf_counter() - start_time
                eval_history.setdefault("outer_iter", []).append(outer)
                eval_history.setdefault("elapsed_sec", []).append(elapsed)
                eval_history.setdefault("n_inner", []).append(n_inner_last)
                eval_history.setdefault("inner_diff", []).append(inner_diff)
                eval_history.setdefault("outer_value_diff", []).append(
                    outer_value_diff,
                )
                eval_history.setdefault("pricing_diff", []).append(
                    pricing_diff,
                )
                for key, val in metrics.items():
                    eval_history.setdefault(key, []).append(val)

            inner_status = (
                "converged"
                if converged_inner
                else f"max({MAX_INNER})"
            )
            print(
                f"  outer={outer:3d} | "
                f"inner={inner_status} ({n_inner_last} iters, "
                f"diff={inner_diff:.2e}) | "
                f"outer_value_diff={outer_value_diff:.2f} | "
                f"pricing_diff={pricing_diff:.2f}"
            )

            if not converged_inner:
                stop_reason = "max_inner"
                break

            if outer_value_diff < TOL_OUTER:
                converged_outer = True
                stop_reason = "converged_outer_value"
                break

            value_prev_outer = value.copy()

    # ------------------------------------------------------------------
    # Final re-evaluation at full z resolution
    # ------------------------------------------------------------------
    value_fine = W @ value
    value_fine_3d = value_fine.reshape(n_z_fine, n_k, n_b)

    (
        r_tilde_fine,
        default_mask_fine,
        expected_recovery_fine,
        solvent_probability_fine,
    ) = _compute_pricing_update_np(env, grids, prob_full, value_fine_3d)

    reward_fine = _reward_tensor_np(env, grids, r_tilde_fine)
    beta = env.discount()
    expected_continuation_fine = prob_full @ value_fine
    rhs_fine = reward_fine + beta * expected_continuation_fine[:, None, :]
    policy_idx_fine = rhs_fine.argmax(axis=2)
    value_fine_reeval = np.maximum(rhs_fine.max(axis=2), 0.0)

    value_reeval_3d = value_fine_reeval.reshape(n_z_fine, n_k, n_b)
    (
        r_tilde_final,
        default_mask_final,
        expected_recovery_final,
        solvent_probability_final,
    ) = _compute_pricing_update_np(env, grids, prob_full, value_reeval_3d)

    policy_action, policy_endo = _extract_policy_np(
        env, grids, policy_idx_fine,
    )
    zero_profit_residual, funded_mask = _zero_profit_residual_np(
        env,
        grids,
        r_tilde_final,
        expected_recovery_final,
        solvent_probability_final,
    )

    risk_free_equilibrium = _is_risk_free_equilibrium(
        env,
        {
            "stop_reason": stop_reason,
            "default_mask": default_mask_final,
            "r_tilde_grid": r_tilde_final,
            "funded_mask": funded_mask,
        },
    )
    risk_free_fixed_point = risk_free_equilibrium and n_outer <= 1
    risk_free_fixed_point_warning = None
    if risk_free_fixed_point:
        risk_free_fixed_point_warning = (
            "Solver converged to a risk-free fixed point in ≤1 outer "
            "iteration.  See solve_nested_vfi docstring for details."
        )
        warnings.warn(
            risk_free_fixed_point_warning, RuntimeWarning, stacklevel=2,
        )

    env.install_r_tilde_grid(grids, r_tilde_final.astype(np.float32))

    return {
        "value": value_fine_reeval,
        "value_coarse": value,
        "policy_idx": policy_idx_fine,
        "policy_action": policy_action,
        "policy_endo": policy_endo,
        "grids": grids,
        "prob_matrix": prob_full,
        "r_tilde_grid": r_tilde_final,
        "default_mask": default_mask_final,
        "zero_profit_residual": zero_profit_residual,
        "funded_mask": funded_mask,
        "converged_inner": converged_inner,
        "converged_outer": converged_outer,
        "n_outer": n_outer,
        "n_inner_last": n_inner_last,
        "inner_diff_history": inner_diff_history,
        "outer_value_diff_history": outer_value_diff_history,
        "pricing_diff_history": pricing_diff_history,
        "history": eval_history,
        "stop_reason": stop_reason,
        "risk_free_equilibrium": risk_free_equilibrium,
        "risk_free_fixed_point": risk_free_fixed_point,
        "risk_free_fixed_point_warning": risk_free_fixed_point_warning,
        "n_z_solve": n_z_coarse,
        "n_z": n_z_fine,
        "z_coarse": z_coarse,
        "coarse_indices": coarse_indices,
        "interp_matrix": W,
        "backend": "numpy",
        "dtype": "float64",
        "device": "CPU:0",
    }


# ======================================================================
# Adaptive b-bounds helper
# ======================================================================


def _extract_active_b_bounds(
    result: Dict,
    buffer_frac: float,
) -> tuple[float, float] | None:
    """Extract the b' range where default risk is non-trivial.

    Returns ``(b_lo, b_hi)`` with buffer, or None if the mixed region
    is degenerate (all-solvent or all-default).
    """
    grids = result["grids"]
    b_grid = np.asarray(grids["endo_grids_1d"][1], dtype=np.float64)
    z_grid = np.asarray(grids["exo_grids_1d"][0], dtype=np.float64)
    n_z = len(z_grid)
    n_k = len(grids["endo_grids_1d"][0])
    n_b = len(b_grid)

    value_3d = np.asarray(result["value"], dtype=np.float64).reshape(
        n_z, n_k, n_b,
    )

    all_solvent = np.all(value_3d > 0, axis=0)
    all_default = np.all(value_3d <= 0, axis=0)
    mixed = ~all_solvent & ~all_default

    mixed_any_k = np.any(mixed, axis=0)

    if not mixed_any_k.any():
        return None

    idx_lo = int(np.argmax(mixed_any_k))
    idx_hi = int(n_b - 1 - np.argmax(mixed_any_k[::-1]))

    b_range = float(b_grid[idx_hi] - b_grid[idx_lo])
    if b_range < 1e-10:
        return None

    buffer = buffer_frac * b_range
    b_lo = float(max(b_grid[0], b_grid[idx_lo] - buffer))
    b_hi = float(min(b_grid[-1], b_grid[idx_hi] + buffer))

    return b_lo, b_hi


# ======================================================================
# Public entry point
# ======================================================================


def solve_risky_debt(
    env,
    config: RiskyDebtSolverConfig | None = None,
    eval_callback=None,
) -> Dict:
    """Solve the risky-debt model via nested VFI with z-interpolation.

    Args:
        env:    RiskyDebtEnv instance.
        config: RiskyDebtSolverConfig (optional, uses defaults).
        eval_callback: Optional callable(outer_iter, partial_result) -> dict.

    Returns:
        Dict with value, policy, r_tilde, default_mask, grids, etc.
        When adaptive=True, also includes ``adaptive_b_bounds`` and
        ``coarse_wall_sec``.
    """
    config = config or RiskyDebtSolverConfig()

    if config.adaptive:
        # Stage 1: boundary discovery on internal small grid
        from src.v2.environments.risky_debt import RiskyDebtEnv as _RiskyDebtEnv

        t0 = time.perf_counter()
        print(
            f"solve_risky_debt: boundary discovery "
            f"({BOUNDARY_N_K}x{BOUNDARY_N_B}x{BOUNDARY_N_Z}) ..."
        )
        result_coarse = _solve_single(
            env,
            n_k=BOUNDARY_N_K,
            n_b=BOUNDARY_N_B,
            n_z=BOUNDARY_N_Z,
            n_z_solve=BOUNDARY_N_Z_SOLVE,
            k_spacing=config.k_spacing,
            b_spacing=config.b_spacing,
            z_spacing=config.z_spacing,
        )
        coarse_wall = time.perf_counter() - t0

        bounds = _extract_active_b_bounds(result_coarse, config.buffer_frac)

        if bounds is not None:
            b_lo, b_hi = bounds
            print(
                f"solve_risky_debt: active b' in [{b_lo:.1f}, {b_hi:.1f}] "
                f"(discovered in {coarse_wall:.1f}s)"
            )
            env_tight = _RiskyDebtEnv(
                econ_params=env.econ,
                shock_params=env.shocks,
                k_min_mult=env.k_min_mult,
                k_max_mult=env.k_max_mult,
                b_max_mult=env.b_max_mult,
                b_min_mult=env.b_min_mult,
                z_sd_mult=env.z_sd_mult,
                b_min_override=b_lo,
                b_max_override=b_hi,
            )
        else:
            print(
                f"solve_risky_debt: no mixed region found, "
                f"using full b' range ({coarse_wall:.1f}s)"
            )
            env_tight = env
            b_lo, b_hi = env.b_min, env.b_max

        # Stage 2: fine solve on tight bounds
        print(
            f"solve_risky_debt: fine solve "
            f"({config.n_k}x{config.n_b}x{config.n_z}, "
            f"z_solve={config.n_z_solve}) ..."
        )
        result = _solve_single(
            env_tight,
            n_k=config.n_k,
            n_b=config.n_b,
            n_z=config.n_z,
            n_z_solve=config.n_z_solve,
            k_spacing=config.k_spacing,
            b_spacing=config.b_spacing,
            z_spacing=config.z_spacing,
            eval_callback=eval_callback,
        )
        result["adaptive_b_bounds"] = (b_lo, b_hi)
        result["coarse_wall_sec"] = coarse_wall
        return result

    else:
        # Single solve on full b' range
        print(
            f"solve_risky_debt: single solve "
            f"({config.n_k}x{config.n_b}x{config.n_z}, "
            f"z_solve={config.n_z_solve}) ..."
        )
        return _solve_single(
            env,
            n_k=config.n_k,
            n_b=config.n_b,
            n_z=config.n_z,
            n_z_solve=config.n_z_solve,
            k_spacing=config.k_spacing,
            b_spacing=config.b_spacing,
            z_spacing=config.z_spacing,
            eval_callback=eval_callback,
        )
