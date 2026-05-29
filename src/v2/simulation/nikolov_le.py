"""Grid-state panel simulation for the Nikolov limited-enforcement model."""

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
from src.v2.simulation.nikolov_to import _draw_next_z_indices


@dataclass(frozen=True)
class NikolovLESimulationConfig:
    """Runtime settings for an LE model panel simulation."""

    n_firms: int = 1000
    horizon: int = 40
    burn_in: int = 20
    seed: int = 20210527

    def __post_init__(self) -> None:
        if self.n_firms < 1:
            raise ValueError(f"n_firms must be >= 1. Got {self.n_firms}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1. Got {self.horizon}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0. Got {self.burn_in}")


def simulate_nikolov_le_panel(
    result: dict[str, Any],
    params: NikolovParams,
    config: NikolovLESimulationConfig | None = None,
) -> pd.DataFrame:
    """Simulate a firm panel from a solved LE policy using grid-only states.

    The LE policy is shock-contingent: the next-period debt balance ``b_next``
    and the payment to investors ``p`` are chosen as a schedule over the next
    shock ``(z', eta')``, so the realized contract is selected once ``(z', eta')``
    is drawn.
    """

    config = config or NikolovLESimulationConfig()
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
        k_next = k_grid[k_next_idx]

        z_next_idx = _draw_next_z_indices(rng, q, z_idx)
        eta_next_idx = rng.choice(n_eta, size=n_firms, p=eta_probs).astype(np.int64)
        z_next = z_grid[z_next_idx]
        eta_next = eta_values[eta_next_idx]

        # Realized shock-contingent contract for the drawn (z', eta').
        b_next_idx = policy["b_next_idx"][
            k_idx, b_idx, z_idx, z_next_idx, eta_next_idx
        ].astype(np.int64)
        b_next = b_grid[b_next_idx]
        payment = policy["p"][k_idx, b_idx, z_idx, z_next_idx, eta_next_idx]

        deterministic_flow = (
            -k_next
            + (1.0 - params.delta) * k_next
            - adjustment_cost(k_next, k, params)
            + params.tau * params.delta * k_next
        )
        tax_flow = params.tau * params.r * b
        resource = (
            (1.0 - params.tau) * profit_pre_tax(k_next, z_next, eta_next, params)
            + deterministic_flow
            + tax_flow
        )
        # Distribution to equity is the budget residual after paying investors.
        dividend_next = resource - payment
        default = np.zeros(n_firms, dtype=bool)

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
                        "payment": payment,
                        "dividend_next": dividend_next,
                        "default": default,
                        "delta": params.delta,
                    }
                )
            )

        k_idx = k_next_idx
        b_idx = b_next_idx
        z_idx = z_next_idx
        eta_idx = eta_next_idx

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def construct_le_observables(panel: pd.DataFrame) -> pd.DataFrame:
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
    return out
