"""Finite-grid LP baseline for the Nikolov limited-enforcement model."""

from __future__ import annotations

from itertools import product
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


def solve_nikolov_le(
    config: NikolovGridConfig | None = None,
    params: NikolovParams | None = None,
    *,
    be_tol: float = 1e-10,
    ll_tol: float = 1e-10,
    collateral_tol: float = 1e-10,
    linprog_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the limited-enforcement model with exhaustive fixed contracts."""

    config = config or NikolovGridConfig()
    params = params or NikolovParams()
    grids = build_nikolov_grids(config)
    actions, feasible_counts = _enumerate_le_actions(
        grids,
        params,
        be_tol=be_tol,
        ll_tol=ll_tol,
        collateral_tol=collateral_tol,
    )
    lp = _assemble_le_lp(grids, params, actions)

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
        raise RuntimeError(f"Nikolov LE LP failed: {lp_result.message}")

    value = np.asarray(lp_result.x, dtype=np.float64).reshape(lp["state_shape"])
    policy = _recover_le_policy(value, grids, params, actions)
    lp_slack = lp["A_ub"].dot(lp_result.x) - lp["b_ub"]

    diagnostics = {
        "lp_value_bounds_free": True,
        "max_lp_violation": float(np.max(lp_slack)) if lp_slack.size else 0.0,
        "states_with_no_feasible_actions": int(np.sum(feasible_counts == 0)),
        "total_feasible_actions": int(np.sum(feasible_counts)),
    }
    return {
        "value": value,
        "policy": policy,
        "grids": grids,
        "prob_matrix": grids["Q"],
        "feasible_action_counts": feasible_counts,
        "lp_result": lp_result,
        "diagnostics": diagnostics,
    }


def _enumerate_le_actions(
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    *,
    be_tol: float,
    ll_tol: float,
    collateral_tol: float,
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    k_grid = grids["k"]
    b_grid = grids["b"]
    z_grid = grids["z"]
    p_grid = grids["p"]
    q = grids["Q"]
    eta_values, eta_probs = eta_grid(params)

    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    n_eta = len(eta_values)
    n_shocks = n_z * n_eta
    n_states = n_k * n_b * n_z
    actions: list[list[dict[str, Any]]] = [[] for _ in range(n_states)]
    feasible_counts = np.zeros((n_k, n_b, n_z), dtype=np.int64)

    pair_indices = np.array(
        list(product(range(n_b), range(len(p_grid)))),
        dtype=np.int64,
    )

    z_next_idx = np.repeat(np.arange(n_z), n_eta)
    eta_next_idx = np.tile(np.arange(n_eta), n_z)
    z_next_values = z_grid[z_next_idx]
    eta_next_values = eta_values[eta_next_idx]

    for ik, k_current in enumerate(k_grid):
        for ib, b_current in enumerate(b_grid):
            for iz, z_current in enumerate(z_grid):
                state_idx = _state_index(ik, ib, iz, n_b, n_z)
                shock_probs = (
                    np.repeat(q[iz], n_eta) * np.tile(eta_probs, n_z)
                )
                for ikp, k_next in enumerate(k_grid):
                    capital_flow = _capital_flow(k_next, k_current, params)
                    tax_flow = params.tau * params.r * b_current
                    deterministic_flow = capital_flow + tax_flow
                    resource = (
                        (1.0 - params.tau)
                        * profit_pre_tax(
                            k_next,
                            z_next_values,
                            eta_next_values,
                            params,
                        )
                        + deterministic_flow
                    )
                    for contract_pairs in product(
                        range(len(pair_indices)),
                        repeat=n_shocks,
                    ):
                        contract_pairs_arr = pair_indices[list(contract_pairs)]
                        b_idx = contract_pairs_arr[:, 0]
                        p_idx = contract_pairs_arr[:, 1]
                        b_next = b_grid[b_idx]
                        p_next = p_grid[p_idx]

                        break_even = (
                            np.dot(shock_probs, p_next + b_next)
                            / (1.0 + params.r)
                        )
                        if abs(break_even - b_current) > be_tol:
                            continue
                        if np.any(p_next - resource > ll_tol):
                            continue
                        collateral = params.theta * (1.0 - params.delta) * k_next
                        if np.any(p_next + b_next - collateral > collateral_tol):
                            continue

                        flow = float(
                            deterministic_flow
                            + np.dot(
                                shock_probs,
                                (1.0 - params.tau)
                                * profit_pre_tax(
                                    k_next,
                                    z_next_values,
                                    eta_next_values,
                                    params,
                                ),
                            )
                        )
                        continuation = _continuation_entries(
                            ikp,
                            b_idx,
                            z_next_idx,
                            shock_probs,
                            n_b,
                            n_z,
                        )
                        actions[state_idx].append(
                            {
                                "k_idx": ikp,
                                "b_idx": b_idx.copy(),
                                "p_idx": p_idx.copy(),
                                "b_next": b_next.copy(),
                                "p": p_next.copy(),
                                "flow": flow,
                                "continuation": continuation,
                            }
                        )

                feasible_counts[ik, ib, iz] = len(actions[state_idx])
                if feasible_counts[ik, ib, iz] == 0:
                    state = (float(k_current), float(b_current), float(z_current))
                    raise ValueError(
                        "Nikolov LE grid has no feasible action at state "
                        f"(k, b, z)={state}."
                    )

    return actions, feasible_counts


def _assemble_le_lp(
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    actions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    n_k, n_b, n_z = len(grids["k"]), len(grids["b"]), len(grids["z"])
    n_vars = n_k * n_b * n_z
    beta = 1.0 / (1.0 + params.r)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_id = 0
    for state_idx, state_actions in enumerate(actions):
        for action in state_actions:
            rows.append(row_id)
            cols.append(state_idx)
            data.append(-1.0)
            for cont_idx, prob in action["continuation"]:
                rows.append(row_id)
                cols.append(cont_idx)
                data.append(beta * prob)
            rhs.append(-beta * action["flow"])
            row_id += 1

    return {
        "A_ub": coo_matrix((data, (rows, cols)), shape=(row_id, n_vars)).tocsc(),
        "b_ub": np.asarray(rhs, dtype=np.float64),
        "n_vars": n_vars,
        "state_shape": (n_k, n_b, n_z),
    }


def _recover_le_policy(
    value: np.ndarray,
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    actions: list[list[dict[str, Any]]],
) -> dict[str, np.ndarray]:
    k_grid = grids["k"]
    b_grid = grids["b"]
    z_grid = grids["z"]
    eta_values, _ = eta_grid(params)
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    n_eta = len(eta_values)
    beta = 1.0 / (1.0 + params.r)
    value_flat = value.ravel()

    policy_k_idx = np.zeros((n_k, n_b, n_z), dtype=np.int64)
    policy_value = np.full((n_k, n_b, n_z), -np.inf, dtype=np.float64)
    contract_b = np.zeros((n_k, n_b, n_z, n_z, n_eta), dtype=np.float64)
    contract_p = np.zeros_like(contract_b)
    contract_b_idx = np.zeros((n_k, n_b, n_z, n_z, n_eta), dtype=np.int64)
    contract_p_idx = np.zeros_like(contract_b_idx)

    for ik in range(n_k):
        for ib in range(n_b):
            for iz in range(n_z):
                state_idx = _state_index(ik, ib, iz, n_b, n_z)
                best_rhs = -np.inf
                best_action = actions[state_idx][0]
                for action in actions[state_idx]:
                    cont_value = sum(
                        prob * value_flat[cont_idx]
                        for cont_idx, prob in action["continuation"]
                    )
                    rhs = beta * (action["flow"] + cont_value)
                    if rhs > best_rhs:
                        best_rhs = rhs
                        best_action = action

                policy_k_idx[ik, ib, iz] = best_action["k_idx"]
                policy_value[ik, ib, iz] = best_rhs
                contract_b[ik, ib, iz] = best_action["b_next"].reshape(n_z, n_eta)
                contract_p[ik, ib, iz] = best_action["p"].reshape(n_z, n_eta)
                contract_b_idx[ik, ib, iz] = best_action["b_idx"].reshape(
                    n_z,
                    n_eta,
                )
                contract_p_idx[ik, ib, iz] = best_action["p_idx"].reshape(
                    n_z,
                    n_eta,
                )

    return {
        "k_next_idx": policy_k_idx,
        "k_next": k_grid[policy_k_idx],
        "b_next_idx": contract_b_idx,
        "p_idx": contract_p_idx,
        "b_next": contract_b,
        "p": contract_p,
        "value": policy_value,
    }


def _capital_flow(
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


def _continuation_entries(
    ikp: int,
    b_idx: np.ndarray,
    z_idx: np.ndarray,
    probs: np.ndarray,
    n_b: int,
    n_z: int,
) -> list[tuple[int, float]]:
    continuation_by_state: dict[int, float] = {}
    for shock_id, prob in enumerate(probs):
        cont_idx = _state_index(
            ikp,
            int(b_idx[shock_id]),
            int(z_idx[shock_id]),
            n_b,
            n_z,
        )
        continuation_by_state[cont_idx] = (
            continuation_by_state.get(cont_idx, 0.0) + float(prob)
        )
    return list(continuation_by_state.items())


def _state_index(ik: int, ib: int, iz: int, n_b: int, n_z: int) -> int:
    return (ik * n_b + ib) * n_z + iz
