"""Finite-grid LP baseline for the Nikolov trade-off model."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from src.v2.environments.nikolov import (
    NikolovGridConfig,
    NikolovParams,
    adjustment_cost,
    build_nikolov_grids,
    eta_grid,
    profit_pre_tax,
)


def solve_nikolov_to(
    config: NikolovGridConfig | None = None,
    params: NikolovParams | None = None,
    *,
    pricing_tol: float = 1e-10,
    ll_tol: float = 1e-10,
    max_premium: float = 100.0,
    linprog_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the trade-off model with a full finite-action LP."""

    config = config or NikolovGridConfig()
    params = params or NikolovParams()
    grids = build_nikolov_grids(config)
    premium_grid, pricing_residual, default_probability_grid = compute_to_premiums(
        grids,
        params,
        pricing_tol=pricing_tol,
        max_premium=max_premium,
    )

    lp = _assemble_to_lp(
        grids,
        params,
        premium_grid,
        ll_tol=ll_tol,
    )

    n_vars = lp["n_vars"]
    value_bounds = [(None, None)] * n_vars
    lp_result = linprog(
        np.ones(n_vars, dtype=np.float64),
        A_ub=lp["A_ub"],
        b_ub=lp["b_ub"],
        bounds=value_bounds,
        method="highs",
        options=linprog_options,
    )
    if not lp_result.success:
        raise RuntimeError(f"Nikolov TO LP failed: {lp_result.message}")

    value = np.asarray(lp_result.x, dtype=np.float64).reshape(lp["state_shape"])
    policy = _recover_to_policy(
        value,
        grids,
        params,
        premium_grid,
        ll_tol=ll_tol,
    )
    lp_slack = lp["A_ub"].dot(lp_result.x) - lp["b_ub"]

    diagnostics = {
        "lp_value_bounds_free": True,
        "max_lp_violation": float(np.max(lp_slack)) if lp_slack.size else 0.0,
        "max_abs_pricing_residual": float(np.max(np.abs(pricing_residual))),
        "max_premium": float(np.max(premium_grid)),
        "states_with_no_feasible_actions": int(
            np.sum(lp["feasible_action_counts"] == 0)
        ),
    }

    return {
        "value": value,
        "policy": policy,
        "grids": grids,
        "prob_matrix": grids["Q"],
        "feasible_action_counts": lp["feasible_action_counts"],
        "lp_result": lp_result,
        "diagnostics": diagnostics,
        "premium_grid": premium_grid,
        "pricing_residual": pricing_residual,
        "default_probability_grid": default_probability_grid,
        "default_diagnostics": {
            "default_probability_grid": default_probability_grid,
        },
    }


def compute_to_premiums(
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    *,
    pricing_tol: float = 1e-10,
    max_premium: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Delta(k_choice, b_old, z_old) by finite-threshold search."""

    k_grid = grids["k"]
    b_grid = grids["b"]
    z_grid = grids["z"]
    q = grids["Q"]
    eta_values, eta_probs = eta_grid(params)

    premium_grid = np.zeros((len(k_grid), len(b_grid), len(z_grid)), dtype=np.float64)
    residual_grid = np.zeros_like(premium_grid)
    default_probability_grid = np.zeros_like(premium_grid)

    for ik, k_choice in enumerate(k_grid):
        for ib, b_old in enumerate(b_grid):
            for iz, _z_old in enumerate(z_grid):
                premium, residual, default_prob = _solve_to_premium_cell(
                    k_choice,
                    b_old,
                    z_grid,
                    q[iz],
                    eta_values,
                    eta_probs,
                    params,
                    pricing_tol=pricing_tol,
                    max_premium=max_premium,
                )
                premium_grid[ik, ib, iz] = premium
                residual_grid[ik, ib, iz] = residual
                default_probability_grid[ik, ib, iz] = default_prob

    return premium_grid, residual_grid, default_probability_grid


def _assemble_to_lp(
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    premium_grid: np.ndarray,
    *,
    ll_tol: float,
) -> dict[str, Any]:
    k_grid = grids["k"]
    b_grid = grids["b"]
    z_grid = grids["z"]
    q = grids["Q"]
    eta_values, eta_probs = eta_grid(params)

    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    n_vars = n_k * n_b * n_z
    beta = 1.0 / (1.0 + params.r)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    feasible_counts = np.zeros((n_k, n_b, n_z), dtype=np.int64)
    row_id = 0

    for ik, k_current in enumerate(k_grid):
        for ib, b_old in enumerate(b_grid):
            for iz, z_old in enumerate(z_grid):
                current_idx = _state_index(ik, ib, iz, n_b, n_z)
                for ikp, k_next in enumerate(k_grid):
                    premium = premium_grid[ikp, ib, iz]
                    det_flow = _to_deterministic_flow(k_next, k_current, params)
                    default_mask = _to_default_mask(
                        k_next,
                        b_old,
                        premium,
                        z_grid,
                        eta_values,
                        params,
                    )
                    for ibp, b_next in enumerate(b_grid):
                        if not _to_limited_liability_feasible(
                            k_next,
                            b_old,
                            b_next,
                            premium,
                            det_flow,
                            z_grid,
                            eta_values,
                            default_mask,
                            params,
                            ll_tol=ll_tol,
                        ):
                            continue
                        flow, continuation = _to_flow_and_continuation(
                            ikp,
                            ibp,
                            k_next,
                            b_old,
                            premium,
                            det_flow,
                            z_grid,
                            q[iz],
                            eta_values,
                            eta_probs,
                            default_mask,
                            params,
                            n_b,
                            n_z,
                        )
                        rows.append(row_id)
                        cols.append(current_idx)
                        data.append(-1.0)
                        for cont_idx, prob in continuation:
                            rows.append(row_id)
                            cols.append(cont_idx)
                            data.append(beta * prob)
                        rhs.append(-beta * flow)
                        row_id += 1
                        feasible_counts[ik, ib, iz] += 1

                if feasible_counts[ik, ib, iz] == 0:
                    state = (float(k_current), float(b_old), float(z_old))
                    raise ValueError(
                        "Nikolov TO grid has no feasible action at state "
                        f"(k, b, z)={state}."
                    )

    a_ub = coo_matrix((data, (rows, cols)), shape=(row_id, n_vars)).tocsc()
    return {
        "A_ub": a_ub,
        "b_ub": np.asarray(rhs, dtype=np.float64),
        "n_vars": n_vars,
        "state_shape": (n_k, n_b, n_z),
        "feasible_action_counts": feasible_counts,
    }


def _recover_to_policy(
    value: np.ndarray,
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    premium_grid: np.ndarray,
    *,
    ll_tol: float,
) -> dict[str, np.ndarray]:
    k_grid = grids["k"]
    b_grid = grids["b"]
    z_grid = grids["z"]
    q = grids["Q"]
    eta_values, eta_probs = eta_grid(params)
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    beta = 1.0 / (1.0 + params.r)
    value_flat = value.ravel()

    policy_k_idx = np.zeros((n_k, n_b, n_z), dtype=np.int64)
    policy_b_idx = np.zeros((n_k, n_b, n_z), dtype=np.int64)
    policy_value = np.full((n_k, n_b, n_z), -np.inf, dtype=np.float64)
    repayment_premium = np.zeros((n_k, n_b, n_z), dtype=np.float64)
    issuance_premium = np.zeros((n_k, n_b, n_z), dtype=np.float64)

    for ik, k_current in enumerate(k_grid):
        for ib, b_old in enumerate(b_grid):
            for iz, _z_old in enumerate(z_grid):
                best_rhs = -np.inf
                best_ikp = 0
                best_ibp = 0
                best_repayment_premium = 0.0
                best_issuance_premium = 0.0
                for ikp, k_next in enumerate(k_grid):
                    premium = premium_grid[ikp, ib, iz]
                    det_flow = _to_deterministic_flow(k_next, k_current, params)
                    default_mask = _to_default_mask(
                        k_next,
                        b_old,
                        premium,
                        z_grid,
                        eta_values,
                        params,
                    )
                    for ibp, b_next in enumerate(b_grid):
                        if not _to_limited_liability_feasible(
                            k_next,
                            b_old,
                            b_next,
                            premium,
                            det_flow,
                            z_grid,
                            eta_values,
                            default_mask,
                            params,
                            ll_tol=ll_tol,
                        ):
                            continue
                        flow, continuation = _to_flow_and_continuation(
                            ikp,
                            ibp,
                            k_next,
                            b_old,
                            premium,
                            det_flow,
                            z_grid,
                            q[iz],
                            eta_values,
                            eta_probs,
                            default_mask,
                            params,
                            n_b,
                            n_z,
                        )
                        cont_value = sum(
                            prob * value_flat[cont_idx]
                            for cont_idx, prob in continuation
                        )
                        rhs = beta * (flow + cont_value)
                        if rhs > best_rhs:
                            best_rhs = rhs
                            best_ikp = ikp
                            best_ibp = ibp
                            best_repayment_premium = premium
                            best_issuance_premium = premium_grid[ikp, ibp, iz]

                policy_k_idx[ik, ib, iz] = best_ikp
                policy_b_idx[ik, ib, iz] = best_ibp
                policy_value[ik, ib, iz] = best_rhs
                repayment_premium[ik, ib, iz] = best_repayment_premium
                issuance_premium[ik, ib, iz] = best_issuance_premium

    return {
        "k_next_idx": policy_k_idx,
        "b_next_idx": policy_b_idx,
        "k_next": k_grid[policy_k_idx],
        "b_next": b_grid[policy_b_idx],
        "value": policy_value,
        "premium": issuance_premium,
        "repayment_premium": repayment_premium,
    }


def _solve_to_premium_cell(
    k_choice: float,
    b_old: float,
    z_grid: np.ndarray,
    z_probs: np.ndarray,
    eta_values: np.ndarray,
    eta_probs: np.ndarray,
    params: NikolovParams,
    *,
    pricing_tol: float,
    max_premium: float,
) -> tuple[float, float, float]:
    if b_old <= 1e-12:
        return 0.0, 0.0, 0.0

    weights = (z_probs[:, None] * eta_probs[None, :]).ravel()
    z_realized = np.repeat(z_grid, len(eta_values))
    eta_realized = np.tile(eta_values, len(z_grid))
    resources = (
        (1.0 - params.tau)
        * profit_pre_tax(k_choice, z_realized, eta_realized, params)
        + (1.0 - params.delta) * k_choice
        + params.tau * params.delta * k_choice
    )
    thresholds = (
        resources / ((1.0 - params.tau) * b_old)
        - 1.0 / (1.0 - params.tau)
        - params.r
    )
    recovery_per_debt = params.xi * (1.0 - params.delta) * k_choice / b_old
    target = 1.0 + params.r

    candidates = {0.0, float(max_premium)}
    finite_thresholds = np.unique(
        thresholds[np.isfinite(thresholds) & (thresholds >= 0.0)]
    )
    for threshold in finite_thresholds:
        if threshold <= max_premium:
            candidates.add(float(threshold))
            candidates.add(float(min(threshold + pricing_tol, max_premium)))

    bounds = [0.0]
    bounds.extend(float(t) for t in finite_thresholds if 0.0 < t < max_premium)
    bounds.append(float(max_premium))
    bounds = sorted(set(bounds))

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        test_delta = lo + 0.5 * (hi - lo)
        default = test_delta > thresholds
        solvent_weight = float(np.sum(weights[~default]))
        base = float(
            np.sum(weights[~default] * (1.0 + params.r))
            + np.sum(weights[default] * recovery_per_debt)
        )
        if solvent_weight <= 1e-14:
            continue
        root = (target - base) / solvent_weight
        if lo - pricing_tol <= root <= hi + pricing_tol:
            candidates.add(float(np.clip(root, 0.0, max_premium)))

    evaluated = []
    for delta_premium in sorted(candidates):
        payoff, default_prob = _to_expected_lender_payoff(
            delta_premium,
            thresholds,
            recovery_per_debt,
            weights,
            params,
        )
        evaluated.append((delta_premium, payoff, default_prob))

    feasible = [
        item for item in evaluated if item[1] >= target - pricing_tol
    ]
    if feasible:
        premium, payoff, default_prob = min(feasible, key=lambda item: item[0])
    else:
        premium, payoff, default_prob = max(
            evaluated,
            key=lambda item: (item[1], -item[0]),
        )
    return float(premium), float(payoff - target), float(default_prob)


def _to_expected_lender_payoff(
    premium: float,
    thresholds: np.ndarray,
    recovery_per_debt: float,
    weights: np.ndarray,
    params: NikolovParams,
) -> tuple[float, float]:
    default = premium > thresholds
    payoff = float(
        np.sum(weights[~default] * (1.0 + params.r + premium))
        + np.sum(weights[default] * recovery_per_debt)
    )
    default_prob = float(np.sum(weights[default]))
    return payoff, default_prob


def _to_default_mask(
    k_next: float,
    b_old: float,
    premium: float,
    z_grid: np.ndarray,
    eta_values: np.ndarray,
    params: NikolovParams,
) -> np.ndarray:
    if b_old <= 1e-12:
        return np.zeros((len(z_grid), len(eta_values)), dtype=bool)
    resources = (
        (1.0 - params.tau)
        * profit_pre_tax(k_next, z_grid[:, None], eta_values[None, :], params)
        + (1.0 - params.delta) * k_next
        + params.tau * params.delta * k_next
    )
    repayment = (1.0 + (1.0 - params.tau) * (params.r + premium)) * b_old
    return resources < repayment


def _to_limited_liability_feasible(
    k_next: float,
    b_old: float,
    b_next: float,
    premium: float,
    det_flow: float,
    z_grid: np.ndarray,
    eta_values: np.ndarray,
    default_mask: np.ndarray,
    params: NikolovParams,
    *,
    ll_tol: float,
) -> bool:
    after_tax_profit = (
        (1.0 - params.tau)
        * profit_pre_tax(k_next, z_grid[:, None], eta_values[None, :], params)
    )
    repayment = (1.0 + (1.0 - params.tau) * (params.r + premium)) * b_old
    dividend = after_tax_profit + det_flow - repayment + b_next
    return bool(np.all(dividend[~default_mask] >= -ll_tol))


def _to_flow_and_continuation(
    ikp: int,
    ibp: int,
    k_next: float,
    b_old: float,
    premium: float,
    det_flow: float,
    z_grid: np.ndarray,
    z_probs: np.ndarray,
    eta_values: np.ndarray,
    eta_probs: np.ndarray,
    default_mask: np.ndarray,
    params: NikolovParams,
    n_b: int,
    n_z: int,
) -> tuple[float, list[tuple[int, float]]]:
    expected_flow = 0.0
    continuation_by_state: dict[int, float] = {}
    for izp, z_next in enumerate(z_grid):
        for ieta, eta_next in enumerate(eta_values):
            prob = float(z_probs[izp] * eta_probs[ieta])
            is_default = bool(default_mask[izp, ieta])
            profit_flow = (
                (1.0 - params.tau)
                * profit_pre_tax(
                    k_next,
                    z_next,
                    eta_next,
                    params,
                )
            )
            tax_shield = (
                params.tau * (params.r + premium) * b_old
                if not is_default
                else 0.0
            )
            deadweight_loss = (
                (1.0 - params.xi)
                * ((1.0 - params.delta) * k_next
                   + params.tau * params.delta * k_next)
                if is_default
                else 0.0
            )
            expected_flow += prob * (profit_flow + tax_shield - deadweight_loss)
            if not is_default:
                cont_idx = _state_index(ikp, ibp, izp, n_b, n_z)
                continuation_by_state[cont_idx] = (
                    continuation_by_state.get(cont_idx, 0.0) + prob
                )
    return det_flow + expected_flow, list(continuation_by_state.items())


def _to_deterministic_flow(
    k_next: float,
    k_current: float,
    params: NikolovParams,
) -> float:
    return float(
        -k_next
        + (1.0 - params.delta) * k_next
        - adjustment_cost(k_next, k_current, params)
        + params.tau * params.delta * k_next
    )


def _state_index(ik: int, ib: int, iz: int, n_b: int, n_z: int) -> int:
    return (ik * n_b + ib) * n_z + iz
