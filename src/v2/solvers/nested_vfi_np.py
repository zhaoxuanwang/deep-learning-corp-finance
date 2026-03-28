"""NumPy reference nested VFI for the risky-debt model.

The inner loop solves the Bellman equation with a fixed risky-rate
schedule r_tilde(z, k', b'). After the Bellman max is computed, the
value is clamped to zero to enforce limited liability:

    V_{s+1}(k, b, z) = max(0, max_{k', b'} RHS)

The outer loop updates r_tilde from the lender zero-profit condition
using the default partition defined by V(k', b', z') = 0.
The expectation terms use a single discrete Markov transition matrix,
and each outer iteration hard-copies r_tilde^{(n+1)} into the next
iteration without damping.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from src.v2.solvers.config import NestedVFIConfig
from src.v2.solvers.grid import (
    GridAxis,
    build_1d_grid,
    build_product_grid,
    tauchen_transition_matrix,
)

def _production_np(k, z, alpha):
    return z * np.power(k, alpha)


def _apply_grid_axis_overrides(
    axes: list[GridAxis],
    spacings: list[str] | None,
    powers: list[float] | None,
) -> list[GridAxis]:
    """Apply solver-side sampling overrides while keeping env bounds fixed."""

    resolved_axes: list[GridAxis] = []
    for i, axis in enumerate(axes):
        spacing = spacings[i] if spacings is not None else axis.spacing
        power = powers[i] if powers is not None else axis.power
        resolved_axes.append(
            GridAxis(
                low=axis.low,
                high=axis.high,
                spacing=spacing,
                power=power,
            )
        )
    return resolved_axes


def _build_nested_vfi_grids(env, grid_config) -> Dict[str, np.ndarray]:
    """Build the risky-debt state and choice grids used by nested VFI."""

    spec = env.grid_spec()
    exo_axes = spec["exo"]
    endo_axes = spec["endo"]
    exo_axes = _apply_grid_axis_overrides(
        exo_axes,
        grid_config.exo_spacings,
        grid_config.exo_powers,
    )
    endo_axes = _apply_grid_axis_overrides(
        endo_axes,
        grid_config.endo_spacings,
        grid_config.endo_powers,
    )

    if len(exo_axes) != len(grid_config.exo_sizes):
        raise ValueError(
            "grid_config.exo_sizes does not match env exogenous dimension. "
            f"Got {len(grid_config.exo_sizes)} sizes for {len(exo_axes)} variables."
        )
    if len(endo_axes) != len(grid_config.endo_sizes):
        raise ValueError(
            "grid_config.endo_sizes does not match env endogenous dimension. "
            f"Got {len(grid_config.endo_sizes)} sizes for {len(endo_axes)} variables."
        )
    if len(endo_axes) != 2 or len(exo_axes) != 1:
        raise ValueError(
            "RiskyDebtEnv nested VFI expects 1 exogenous dimension (z) "
            "and 2 endogenous dimensions (k, b)."
        )

    exo_grids_1d = [
        build_1d_grid(axis, n) for axis, n in zip(exo_axes, grid_config.exo_sizes)
    ]
    endo_grids_1d = [
        build_1d_grid(axis, n) for axis, n in zip(endo_axes, grid_config.endo_sizes)
    ]
    endo_product = build_product_grid(endo_grids_1d)
    exo_product = build_product_grid(exo_grids_1d)

    return {
        "exo_grids_1d": exo_grids_1d,
        "endo_grids_1d": endo_grids_1d,
        "action_grids_1d": endo_grids_1d,
        "choice_grids_1d": endo_grids_1d,
        "exo_product": exo_product,
        "endo_product": endo_product,
        "choice_product": endo_product,
    }


def _debt_price_from_r_tilde(r_tilde: np.ndarray) -> np.ndarray:
    """Compute 1 / (1 + r_tilde), with zero for infinite rates."""

    r_tilde = np.asarray(r_tilde, dtype=np.float64)
    debt_price = np.zeros_like(r_tilde, dtype=np.float64)
    finite = np.isfinite(r_tilde)
    debt_price[finite] = 1.0 / np.maximum(1.0 + r_tilde[finite], 1e-12)
    return debt_price


def _pricing_discount_sup_norm_diff(
    r_new: np.ndarray,
    r_old: np.ndarray,
) -> float:
    """Diagnostic sup-norm difference in bounded debt-discount space.

    The canonical algorithm stores and updates the risky rate ``r_tilde``.
    For outer-loop convergence diagnostics, however, we compare the implied
    debt discount factor ``1 / (1 + r_tilde)`` because this is the bounded
    object that directly enters firm cash flow. In particular, all-default
    cells with ``r_tilde = +inf`` map cleanly to zero proceeds.
    """

    discount_new = _debt_price_from_r_tilde(r_new)
    discount_old = _debt_price_from_r_tilde(r_old)
    return float(np.max(np.abs(discount_new - discount_old)))


def _reward_tensor_np(
    env,
    grids: Dict[str, np.ndarray],
    r_tilde_grid: np.ndarray,
) -> np.ndarray:
    """Compute reward(z, state, choice) under a fixed risky-rate schedule."""

    z_grid = np.asarray(grids["exo_grids_1d"][0], dtype=np.float64)
    endo_product = np.asarray(grids["endo_product"], dtype=np.float64)
    choice_product = np.asarray(grids["choice_product"], dtype=np.float64)

    k_current = endo_product[:, 0]
    b_current = endo_product[:, 1]
    k_next = choice_product[:, 0]
    b_next = choice_product[:, 1]

    delta = env.econ.depreciation_rate
    tau = env.econ.tax
    alpha = env.econ.production_elasticity
    rate_rf = env.econ.interest_rate

    investment = k_next[None, :] - (1.0 - delta) * k_current[:, None]
    safe_k = np.maximum(k_current[:, None], 1e-12)
    adjustment_cost = 0.5 * env.econ.cost_convex * (investment ** 2) / safe_k

    after_tax_profit = (
        (1.0 - tau) * _production_np(k_current[None, :], z_grid[:, None], alpha)
    )
    debt_discount = np.zeros_like(r_tilde_grid, dtype=np.float64)
    finite = np.isfinite(r_tilde_grid)
    debt_discount[finite] = 1.0 / np.maximum(1.0 + r_tilde_grid[finite], 1e-12)
    debt_discount = debt_discount.reshape(len(z_grid), -1)
    debt_proceeds = b_next[None, :] * debt_discount
    tax_shield = tau * b_next[None, :] * (1.0 - debt_discount) / (1.0 + rate_rf)

    cash_flow = (
        after_tax_profit[:, :, None]
        - adjustment_cost[None, :, :]
        - investment[None, :, :]
        - b_current[None, :, None]
        + debt_proceeds[:, None, :]
        + tax_shield[:, None, :]
    )

    shortfall = np.maximum(-cash_flow, 0.0)
    issuance_cost = env.econ.cost_inject_linear * shortfall
    if env.econ.cost_inject_fixed > 0.0:
        issuance_cost = issuance_cost + env.econ.cost_inject_fixed * (shortfall > 0.0)

    return cash_flow - issuance_cost


def _extract_policy_np(
    env,
    grids: Dict[str, np.ndarray],
    policy_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract policy_action=(I,b') and policy_endo=(k',b')."""

    endo_product = grids["endo_product"]
    choice_product = grids["choice_product"]

    policy_endo = choice_product[policy_idx]
    current_k = endo_product[:, 0][None, :]
    investment = policy_endo[..., 0] - (1.0 - env.econ.depreciation_rate) * current_k
    policy_action = np.stack([investment, policy_endo[..., 1]], axis=-1)
    return policy_action, policy_endo


def _compute_pricing_update_np(
    env,
    grids: Dict[str, np.ndarray],
    prob_matrix: np.ndarray,
    value_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update r_tilde from the future-state default partition.

    Timing matters here:
    - ``value_3d[z', k', b']`` is the next-period equity value.
    - Default is therefore determined from the next-period state only,
      via ``V(k', b', z') = 0``.
    - Pricing is still indexed by the current shock ``z`` because lenders
      integrate that future default set using the conditional transition
      matrix ``p(z' | z)``.
    """

    z_grid = np.asarray(grids["exo_grids_1d"][0], dtype=np.float64)
    k_grid = np.asarray(grids["endo_grids_1d"][0], dtype=np.float64)
    b_grid = np.asarray(grids["endo_grids_1d"][1], dtype=np.float64)

    n_z = len(z_grid)
    n_b = len(b_grid)

    default_mask = np.asarray(value_3d, dtype=np.float64) <= 0.0
    recovery = env.recovery_value(
        k_grid[None, :, None],
        z_grid[:, None, None],
    )
    recovery_flat = np.repeat(recovery, n_b, axis=2).reshape(n_z, -1)
    default_flat = default_mask.reshape(n_z, -1).astype(np.float64)
    solvent_flat = 1.0 - default_flat

    expected_recovery = prob_matrix @ (default_flat * recovery_flat)
    solvent_probability = prob_matrix @ solvent_flat

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
        r_tilde_new.reshape(n_z, len(k_grid), len(b_grid)),
        default_mask,
        expected_recovery.reshape(n_z, len(k_grid), len(b_grid)),
        solvent_probability.reshape(n_z, len(k_grid), len(b_grid)),
    )


def _zero_profit_residual_np(
    env,
    grids: Dict[str, np.ndarray],
    r_tilde_grid: np.ndarray,
    expected_recovery: np.ndarray,
    solvent_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute zero-profit residuals on funded cells."""
    b_grid = np.asarray(grids["endo_grids_1d"][1], dtype=np.float64)
    b_3d = np.broadcast_to(
        b_grid[None, None, :],
        np.asarray(r_tilde_grid).shape,
    ).astype(np.float64)
    funded_mask = (
        (b_3d > 1e-12)
        & np.isfinite(r_tilde_grid)
        & (solvent_probability > 1e-12)
    )

    lhs = b_3d * (1.0 + env.econ.interest_rate)
    residual = np.full_like(r_tilde_grid, np.nan, dtype=np.float64)
    rhs = np.full_like(r_tilde_grid, np.nan, dtype=np.float64)
    rhs[funded_mask] = (
        expected_recovery[funded_mask]
        + b_3d[funded_mask]
        * (1.0 + r_tilde_grid[funded_mask])
        * solvent_probability[funded_mask]
    )
    residual[funded_mask] = lhs[funded_mask] - rhs[funded_mask]
    return residual, funded_mask


def _solve_inner_bellman_np(
    env,
    prob_matrix: np.ndarray,
    reward: np.ndarray,
    value_init: np.ndarray,
    max_iter_inner: int,
    tol_inner: float,
) -> tuple[np.ndarray, np.ndarray, bool, int, float, list[float]]:
    """Solve the inner Bellman problem under a fixed pricing schedule."""

    value = np.array(value_init, copy=True)
    policy_idx = np.zeros((reward.shape[0], reward.shape[1]), dtype=np.int64)
    inner_diff_history_local: list[float] = []
    inner_diff = float("inf")
    converged_inner = False
    n_inner = max_iter_inner
    beta = env.discount()

    for inner in range(max_iter_inner):
        expected_continuation = prob_matrix @ value
        rhs = reward + beta * expected_continuation[:, None, :]
        rhs_max = rhs.max(axis=2)
        policy_idx = rhs.argmax(axis=2)
        value_next = np.maximum(rhs_max, 0.0)

        inner_diff = float(np.max(np.abs(value_next - value)))
        value = value_next
        inner_diff_history_local.append(inner_diff)
        if inner_diff < tol_inner:
            converged_inner = True
            n_inner = inner + 1
            break

    return value, policy_idx, converged_inner, n_inner, inner_diff, inner_diff_history_local


def solve_nested_vfi(
    env,
    train_dataset: Dict[str, object] | None = None,
    config: NestedVFIConfig | None = None,
    eval_callback=None,
) -> Dict:
    """Solve the canonical risky-debt benchmark via the NumPy reference path."""

    del train_dataset
    config = config or NestedVFIConfig()

    grids = _build_nested_vfi_grids(env, config.grid)
    prob_matrix = tauchen_transition_matrix(env.shocks, grids["exo_grids_1d"])

    z_grid = grids["exo_grids_1d"][0]
    endo_product = grids["endo_product"]
    n_z = len(z_grid)
    n_state = endo_product.shape[0]

    value = np.zeros((n_z, n_state), dtype=np.float64)
    policy_idx = np.zeros((n_z, n_state), dtype=np.int64)
    r_tilde_grid = np.full(
        (n_z, len(grids["endo_grids_1d"][0]), len(grids["endo_grids_1d"][1])),
        env.econ.interest_rate,
        dtype=np.float64,
    )

    start_time = time.perf_counter()
    inner_diff_history = []
    outer_value_diff_history = []
    pricing_diff_history = []
    eval_history = {}
    converged_inner = False
    converged_outer = False
    n_inner_last = 0
    stop_reason = "max_outer"
    default_mask = np.zeros_like(r_tilde_grid, dtype=bool)
    expected_recovery = np.zeros_like(r_tilde_grid, dtype=np.float64)
    solvent_probability = np.zeros_like(r_tilde_grid, dtype=np.float64)
    n_outer = 0
    inner_diff = float("inf")

    reward = _reward_tensor_np(env, grids, r_tilde_grid)
    (
        value,
        policy_idx,
        converged_inner,
        n_inner_last,
        inner_diff,
        inner_history_local,
    ) = _solve_inner_bellman_np(
        env,
        prob_matrix,
        reward,
        value,
        config.max_iter_inner,
        config.tol_inner,
    )
    inner_diff_history.extend(inner_history_local)

    if not converged_inner:
        stop_reason = "max_inner"
    else:
        value_prev_outer = value.copy()

        for outer in range(config.max_iter_outer):
            value_prev_3d = value_prev_outer.reshape(
                n_z,
                len(grids["endo_grids_1d"][0]),
                len(grids["endo_grids_1d"][1]),
            )
            (
                r_tilde_new,
                default_mask,
                expected_recovery,
                solvent_probability,
            ) = _compute_pricing_update_np(
                env,
                grids,
                prob_matrix,
                value_prev_3d,
            )
            pricing_diff = _pricing_discount_sup_norm_diff(r_tilde_new, r_tilde_grid)

            reward = _reward_tensor_np(env, grids, r_tilde_new)
            (
                value,
                policy_idx,
                converged_inner,
                n_inner_last,
                inner_diff,
                inner_history_local,
            ) = _solve_inner_bellman_np(
                env,
                prob_matrix,
                reward,
                value_prev_outer,
                config.max_iter_inner,
                config.tol_inner,
            )
            inner_diff_history.extend(inner_history_local)
            outer_value_diff = float(np.max(np.abs(value - value_prev_outer)))

            outer_value_diff_history.append(outer_value_diff)
            pricing_diff_history.append(pricing_diff)
            n_outer = outer + 1
            r_tilde_grid = r_tilde_new

            value_3d = value.reshape(
                n_z,
                len(grids["endo_grids_1d"][0]),
                len(grids["endo_grids_1d"][1]),
            )
            default_mask = value_3d <= 0.0

            policy_action, policy_endo = _extract_policy_np(env, grids, policy_idx)
            partial_result = {
                "value": value,
                "policy_action": policy_action,
                "policy_endo": policy_endo,
                "grids": grids,
                "r_tilde_grid": r_tilde_grid,
                "default_mask": default_mask,
                "backend": "numpy",
                "dtype": "float64",
                "device": "CPU:0",
            }

            if eval_callback is not None:
                metrics = eval_callback(outer, partial_result) or {}
                elapsed = time.perf_counter() - start_time
                eval_history.setdefault("outer_iter", []).append(outer)
                eval_history.setdefault("elapsed_sec", []).append(elapsed)
                eval_history.setdefault("n_inner", []).append(n_inner_last)
                eval_history.setdefault("inner_diff", []).append(inner_diff)
                eval_history.setdefault("outer_value_diff", []).append(outer_value_diff)
                eval_history.setdefault("pricing_diff", []).append(pricing_diff)
                for key, val in metrics.items():
                    eval_history.setdefault(key, []).append(val)

            inner_status = "converged" if converged_inner else f"max({config.max_iter_inner})"
            print(
                f"Nested VFI outer={outer:3d} | "
                f"inner={inner_status} ({n_inner_last} iters, diff={inner_diff:.2e}) | "
                f"outer_value_diff={outer_value_diff:.2f} | "
                f"pricing_diff={pricing_diff:.2f}"
            )

            if not converged_inner:
                stop_reason = "max_inner"
                break

            if outer_value_diff < config.tol_outer_value:
                converged_outer = True
                stop_reason = "converged_outer_value"
                break

            value_prev_outer = value.copy()

    value_3d = value.reshape(
        n_z,
        len(grids["endo_grids_1d"][0]),
        len(grids["endo_grids_1d"][1]),
    )
    (
        r_tilde_grid,
        default_mask,
        expected_recovery,
        solvent_probability,
    ) = _compute_pricing_update_np(env, grids, prob_matrix, value_3d)

    policy_action, policy_endo = _extract_policy_np(env, grids, policy_idx)
    zero_profit_residual, funded_mask = _zero_profit_residual_np(
        env,
        grids,
        r_tilde_grid,
        expected_recovery,
        solvent_probability,
    )

    env.install_r_tilde_grid(grids, r_tilde_grid.astype(np.float32))

    return {
        "value": value,
        "policy_action": policy_action,
        "policy_endo": policy_endo,
        "grids": grids,
        "prob_matrix": prob_matrix,
        "r_tilde_grid": r_tilde_grid,
        "default_mask": default_mask,
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
        "backend": "numpy",
        "dtype": "float64",
        "device": "CPU:0",
    }
