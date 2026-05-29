"""Finite-grid LP baseline for the Nikolov moral-hazard model."""

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


def solve_nikolov_mh(
    config: NikolovGridConfig | None = None,
    params: NikolovParams | None = None,
    *,
    pk_tol: float = 1e-10,
    ic_tol: float = 1e-10,
    linprog_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the moral-hazard model with exhaustive fixed contracts."""

    config = config or NikolovGridConfig()
    params = params or NikolovParams()
    grids = build_nikolov_grids(config)
    actions, feasible_counts = _enumerate_mh_actions(
        grids,
        params,
        pk_tol=pk_tol,
        ic_tol=ic_tol,
    )
    lp = _assemble_mh_lp(grids, params, actions)

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
        raise RuntimeError(f"Nikolov MH LP failed: {lp_result.message}")

    value = np.asarray(lp_result.x, dtype=np.float64).reshape(lp["state_shape"])
    policy = _recover_mh_policy(value, grids, params, actions)
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


def _enumerate_mh_actions(
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    *,
    pk_tol: float,
    ic_tol: float,
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    k_grid = grids["k"]
    v_grid = grids["v"]
    z_grid = grids["z"]
    d_grid = grids["d"]
    q = grids["Q"]
    eta_values, eta_probs = eta_grid(params)

    n_k, n_v, n_z = len(k_grid), len(v_grid), len(z_grid)
    n_eta = len(eta_values)
    n_shocks = n_z * n_eta
    n_states = n_k * n_v * n_z
    actions: list[list[dict[str, Any]]] = [[] for _ in range(n_states)]
    feasible_counts = np.zeros((n_k, n_v, n_z), dtype=np.int64)

    pair_indices = np.array(
        list(product(range(n_v), range(len(d_grid)))),
        dtype=np.int64,
    )

    z_next_idx = np.repeat(np.arange(n_z), n_eta)
    eta_next_idx = np.tile(np.arange(n_eta), n_z)
    z_next_values = z_grid[z_next_idx]
    eta_next_values = eta_values[eta_next_idx]

    for ik, k_current in enumerate(k_grid):
        for iv, v_current in enumerate(v_grid):
            for iz, z_current in enumerate(z_grid):
                state_idx = _state_index(ik, iv, iz, n_v, n_z)
                shock_probs = (
                    np.repeat(q[iz], n_eta) * np.tile(eta_probs, n_z)
                )
                for ikp, k_next in enumerate(k_grid):
                    deterministic_flow = (
                        _capital_flow(k_next, k_current, params)
                        - params.r * params.tau * v_current
                    )
                    for contract_pairs in product(
                        range(len(pair_indices)),
                        repeat=n_shocks,
                    ):
                        contract_pairs_arr = pair_indices[list(contract_pairs)]
                        v_idx = contract_pairs_arr[:, 0]
                        d_idx = contract_pairs_arr[:, 1]
                        v_next = v_grid[v_idx]
                        d_next = d_grid[d_idx]

                        promise = (
                            np.dot(shock_probs, d_next + v_next)
                            / (1.0 + params.r)
                        )
                        if abs(promise - v_current) > pk_tol:
                            continue
                        if not _mh_ic_feasible(
                            k_next,
                            z_grid,
                            eta_values,
                            v_next.reshape(n_z, n_eta),
                            d_next.reshape(n_z, n_eta),
                            params,
                            ic_tol=ic_tol,
                        ):
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
                            v_idx,
                            z_next_idx,
                            shock_probs,
                            n_v,
                            n_z,
                        )
                        actions[state_idx].append(
                            {
                                "k_idx": ikp,
                                "v_idx": v_idx.copy(),
                                "d_idx": d_idx.copy(),
                                "V_next": v_next.copy(),
                                "d": d_next.copy(),
                                "flow": flow,
                                "continuation": continuation,
                            }
                        )

                feasible_counts[ik, iv, iz] = len(actions[state_idx])
                if feasible_counts[ik, iv, iz] == 0:
                    state = (float(k_current), float(v_current), float(z_current))
                    raise ValueError(
                        "Nikolov MH grid has no feasible action at state "
                        f"(k, V, z)={state}."
                    )

    return actions, feasible_counts


def _assemble_mh_lp(
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    actions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    n_k, n_v, n_z = len(grids["k"]), len(grids["v"]), len(grids["z"])
    n_vars = n_k * n_v * n_z
    beta_firm = 1.0 / (1.0 + (1.0 - params.tau) * params.r)

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
                data.append(beta_firm * prob)
            rhs.append(-beta_firm * action["flow"])
            row_id += 1

    return {
        "A_ub": coo_matrix((data, (rows, cols)), shape=(row_id, n_vars)).tocsc(),
        "b_ub": np.asarray(rhs, dtype=np.float64),
        "n_vars": n_vars,
        "state_shape": (n_k, n_v, n_z),
    }


def _recover_mh_policy(
    value: np.ndarray,
    grids: dict[str, np.ndarray],
    params: NikolovParams,
    actions: list[list[dict[str, Any]]],
) -> dict[str, np.ndarray]:
    k_grid = grids["k"]
    v_grid = grids["v"]
    z_grid = grids["z"]
    eta_values, _ = eta_grid(params)
    n_k, n_v, n_z = len(k_grid), len(v_grid), len(z_grid)
    n_eta = len(eta_values)
    beta_firm = 1.0 / (1.0 + (1.0 - params.tau) * params.r)
    value_flat = value.ravel()

    policy_k_idx = np.zeros((n_k, n_v, n_z), dtype=np.int64)
    policy_value = np.full((n_k, n_v, n_z), -np.inf, dtype=np.float64)
    contract_v = np.zeros((n_k, n_v, n_z, n_z, n_eta), dtype=np.float64)
    contract_d = np.zeros_like(contract_v)
    contract_p = np.zeros_like(contract_v)
    contract_v_idx = np.zeros((n_k, n_v, n_z, n_z, n_eta), dtype=np.int64)
    contract_d_idx = np.zeros_like(contract_v_idx)

    for ik, k_current in enumerate(k_grid):
        for iv in range(n_v):
            for iz in range(n_z):
                state_idx = _state_index(ik, iv, iz, n_v, n_z)
                best_rhs = -np.inf
                best_action = actions[state_idx][0]
                for action in actions[state_idx]:
                    cont_value = sum(
                        prob * value_flat[cont_idx]
                        for cont_idx, prob in action["continuation"]
                    )
                    rhs = beta_firm * (action["flow"] + cont_value)
                    if rhs > best_rhs:
                        best_rhs = rhs
                        best_action = action

                policy_k_idx[ik, iv, iz] = best_action["k_idx"]
                policy_value[ik, iv, iz] = best_rhs
                contract_v[ik, iv, iz] = best_action["V_next"].reshape(n_z, n_eta)
                contract_d[ik, iv, iz] = best_action["d"].reshape(n_z, n_eta)
                contract_v_idx[ik, iv, iz] = best_action["v_idx"].reshape(
                    n_z,
                    n_eta,
                )
                contract_d_idx[ik, iv, iz] = best_action["d_idx"].reshape(
                    n_z,
                    n_eta,
                )

                k_next = k_grid[best_action["k_idx"]]
                capital_flow = _capital_flow(k_next, k_current, params)
                for izp, z_next in enumerate(z_grid):
                    for ieta, eta_next in enumerate(eta_values):
                        v_cont = contract_v[ik, iv, iz, izp, ieta]
                        d_cont = contract_d[ik, iv, iz, izp, ieta]
                        v_cont_idx = contract_v_idx[ik, iv, iz, izp, ieta]
                        w_cont = value[
                            best_action["k_idx"],
                            v_cont_idx,
                            izp,
                        ]
                        contract_p[ik, iv, iz, izp, ieta] = (
                            capital_flow
                            + params.tau * params.r * (w_cont - v_cont)
                            + (1.0 - params.tau)
                            * profit_pre_tax(k_next, z_next, eta_next, params)
                            - d_cont
                        )

    return {
        "k_next_idx": policy_k_idx,
        "k_next": k_grid[policy_k_idx],
        "V_next_idx": contract_v_idx,
        "d_idx": contract_d_idx,
        "V_next": contract_v,
        "d": contract_d,
        "p": contract_p,
        "value": policy_value,
    }


def _mh_ic_feasible(
    k_next: float,
    z_grid: np.ndarray,
    eta_values: np.ndarray,
    v_next: np.ndarray,
    d_next: np.ndarray,
    params: NikolovParams,
    *,
    ic_tol: float,
) -> bool:
    n_z, n_eta = v_next.shape
    for iz in range(n_z):
        for true_eta in range(n_eta):
            for report_eta in range(n_eta):
                if true_eta == report_eta:
                    continue
                diversion = params.lambda_diversion * (1.0 - params.tau) * (
                    profit_pre_tax(k_next, z_grid[iz], eta_values[true_eta], params)
                    - profit_pre_tax(k_next, z_grid[iz], eta_values[report_eta], params)
                )
                truth_payoff = d_next[iz, true_eta] + v_next[iz, true_eta]
                report_payoff = (
                    d_next[iz, report_eta]
                    + v_next[iz, report_eta]
                    + diversion
                )
                if truth_payoff + ic_tol < report_payoff:
                    return False
    return True


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
    v_idx: np.ndarray,
    z_idx: np.ndarray,
    probs: np.ndarray,
    n_v: int,
    n_z: int,
) -> list[tuple[int, float]]:
    continuation_by_state: dict[int, float] = {}
    for shock_id, prob in enumerate(probs):
        cont_idx = _state_index(
            ikp,
            int(v_idx[shock_id]),
            int(z_idx[shock_id]),
            n_v,
            n_z,
        )
        continuation_by_state[cont_idx] = (
            continuation_by_state.get(cont_idx, 0.0) + float(prob)
        )
    return list(continuation_by_state.items())


def _state_index(ik: int, iv: int, iz: int, n_v: int, n_z: int) -> int:
    return (ik * n_v + iv) * n_z + iz
