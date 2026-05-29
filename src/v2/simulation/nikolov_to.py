"""Grid-state panel simulation for the Nikolov trade-off model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.v2.environments.nikolov import (
    NikolovParams,
    adjustment_cost,
    eta_grid,
    profit_pre_tax,
)


@dataclass(frozen=True)
class NikolovTOSimulationConfig:
    """Runtime settings for a TO model panel simulation."""

    n_firms: int = 1000
    horizon: int = 40
    burn_in: int = 20
    seed: int = 20210527
    default_restart: bool = True

    def __post_init__(self) -> None:
        if self.n_firms < 1:
            raise ValueError(f"n_firms must be >= 1. Got {self.n_firms}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1. Got {self.horizon}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0. Got {self.burn_in}")


def simulate_nikolov_to_panel(
    result: dict[str, Any],
    params: NikolovParams,
    config: NikolovTOSimulationConfig | None = None,
) -> pd.DataFrame:
    """Simulate a firm panel from a solved TO policy using grid-only states."""

    config = config or NikolovTOSimulationConfig()
    grids = result["grids"]
    policy = result["policy"]
    rng = np.random.default_rng(config.seed)

    k_grid = np.asarray(grids["k"], dtype=np.float64)
    b_grid = np.asarray(grids["b"], dtype=np.float64)
    z_grid = np.asarray(grids["z"], dtype=np.float64)
    q = np.asarray(result["prob_matrix"], dtype=np.float64)
    eta_values, eta_probs = eta_grid(params)

    n_firms = config.n_firms
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    n_eta = len(eta_values)

    firm_id = np.arange(n_firms, dtype=np.int64)
    k_idx = rng.integers(0, n_k, size=n_firms, dtype=np.int64)
    b_idx = rng.integers(0, n_b, size=n_firms, dtype=np.int64)
    z_idx = rng.integers(0, n_z, size=n_firms, dtype=np.int64)
    eta_idx = rng.choice(n_eta, size=n_firms, p=eta_probs).astype(np.int64)

    rows: list[pd.DataFrame] = []
    total_steps = config.burn_in + config.horizon

    for step in range(total_steps):
        k = k_grid[k_idx]
        b = b_grid[b_idx]
        z = z_grid[z_idx]
        eta = eta_values[eta_idx]

        k_next_idx = policy["k_next_idx"][k_idx, b_idx, z_idx].astype(np.int64)
        b_next_idx = policy["b_next_idx"][k_idx, b_idx, z_idx].astype(np.int64)
        k_next = k_grid[k_next_idx]
        b_next = b_grid[b_next_idx]
        premium = policy["repayment_premium"][k_idx, b_idx, z_idx]
        issuance_premium = policy["premium"][k_idx, b_idx, z_idx]

        z_next_idx = _draw_next_z_indices(rng, q, z_idx)
        eta_next_idx = rng.choice(n_eta, size=n_firms, p=eta_probs).astype(np.int64)
        z_next = z_grid[z_next_idx]
        eta_next = eta_values[eta_next_idx]

        resources = (
            (1.0 - params.tau)
            * profit_pre_tax(k_next, z_next, eta_next, params)
            + (1.0 - params.delta) * k_next
            + params.tau * params.delta * k_next
        )
        repayment = (1.0 + (1.0 - params.tau) * (params.r + premium)) * b
        default = resources < repayment

        deterministic_flow = (
            -k_next
            + (1.0 - params.delta) * k_next
            - adjustment_cost(k_next, k, params)
            + params.tau * params.delta * k_next
        )
        dividend_next = (
            (1.0 - params.tau) * profit_pre_tax(k_next, z_next, eta_next, params)
            + deterministic_flow
            - repayment
            + b_next
        )
        dividend_next = np.where(default, 0.0, dividend_next)

        if step >= config.burn_in:
            rows.append(
                pd.DataFrame(
                    {
                        "firm_id": firm_id,
                        "t": step - config.burn_in,
                        "k_idx": k_idx,
                        "b_idx": b_idx,
                        "z_idx": z_idx,
                        "eta_idx": eta_idx,
                        "k": k,
                        "b": b,
                        "z": z,
                        "eta": eta,
                        "k_next_idx": k_next_idx,
                        "b_next_idx": b_next_idx,
                        "z_next_idx": z_next_idx,
                        "eta_next_idx": eta_next_idx,
                        "k_next": k_next,
                        "b_next": b_next,
                        "z_next": z_next,
                        "eta_next": eta_next,
                        "profit_pre_tax": profit_pre_tax(k, z, eta, params),
                        "dividend_next": dividend_next,
                        "repayment_premium": premium,
                        "issuance_premium": issuance_premium,
                        "repayment": repayment,
                        "default": default,
                        "delta": params.delta,
                    }
                )
            )

        if config.default_restart and np.any(default):
            restart_count = int(np.sum(default))
            restart_high = max(1, n_k // 2 + 1)
            k_next_idx = k_next_idx.copy()
            b_next_idx = b_next_idx.copy()
            k_next_idx[default] = rng.integers(
                0,
                restart_high,
                size=restart_count,
                dtype=np.int64,
            )
            b_next_idx[default] = 0

        k_idx = k_next_idx
        b_idx = b_next_idx
        z_idx = z_next_idx
        eta_idx = eta_next_idx

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def construct_to_observables(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct observable states/actions for empirical policy estimation."""

    required = {
        "k",
        "b",
        "profit_pre_tax",
        "k_next",
        "b_next",
        "dividend_next",
        "delta",
        "default",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    out = panel.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_k"] = np.log(out["k"])
        out["profitability"] = out["profit_pre_tax"] / out["k"]
        out["leverage"] = out["b"] / out["k"]
        out["investment_rate"] = (
            out["k_next"] - (1.0 - out["delta"]) * out["k"]
        ) / out["k"]
        out["future_leverage"] = out["b_next"] / out["k_next"]
        out["payout_rate"] = out["dividend_next"] / out["k_next"]
    out.loc[out["default"].astype(bool), "payout_rate"] = 0.0
    return out


def _draw_next_z_indices(
    rng: np.random.Generator,
    q: np.ndarray,
    z_idx: np.ndarray,
) -> np.ndarray:
    draws = np.empty_like(z_idx, dtype=np.int64)
    for iz in np.unique(z_idx):
        mask = z_idx == iz
        draws[mask] = rng.choice(q.shape[1], size=int(np.sum(mask)), p=q[iz])
    return draws
