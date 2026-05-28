"""Grid-state panel simulation for the Nikolov moral-hazard model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.v2.environments.nikolov import (
    NikolovParams,
    eta_grid,
    profit_pre_tax,
)
from src.v2.simulation.nikolov_to import _draw_next_z_indices


@dataclass(frozen=True)
class NikolovMHSimulationConfig:
    """Runtime settings for an MH model panel simulation."""

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


def simulate_nikolov_mh_panel(
    result: dict[str, Any],
    params: NikolovParams,
    config: NikolovMHSimulationConfig | None = None,
) -> pd.DataFrame:
    """Simulate a firm panel from a solved MH policy using grid-only states.

    The MH state is ``(k, V, z)`` with ``V`` the promised equity value (no debt
    state). The policy is shock-contingent: the continuation value ``V_next`` and
    dividend ``d`` form a schedule over the next shock ``(z', eta')``. Implied
    debt follows the paper's identity ``b = W - V`` (firm value minus equity
    value), with ``W`` read from the solved value function.
    """

    config = config or NikolovMHSimulationConfig()
    grids = result["grids"]
    policy = result["policy"]
    value = np.asarray(result["value"], dtype=np.float64)
    rng = np.random.default_rng(config.seed)

    k_grid = np.asarray(grids["k"], dtype=np.float64)
    v_grid = np.asarray(grids["v"], dtype=np.float64)
    z_grid = np.asarray(grids["z"], dtype=np.float64)
    q = np.asarray(result["prob_matrix"], dtype=np.float64)
    eta_values, eta_probs = eta_grid(params)

    n_firms = config.n_firms
    n_k, n_v, n_z = len(k_grid), len(v_grid), len(z_grid)
    n_eta = len(eta_values)

    firm_id = np.arange(n_firms, dtype=np.int64)
    k_idx = rng.integers(0, n_k, size=n_firms, dtype=np.int64)
    v_idx = rng.integers(0, n_v, size=n_firms, dtype=np.int64)
    z_idx = rng.integers(0, n_z, size=n_firms, dtype=np.int64)
    eta_idx = rng.choice(n_eta, size=n_firms, p=eta_probs).astype(np.int64)

    rows: list[pd.DataFrame] = []
    total_steps = config.burn_in + config.horizon

    for step in range(total_steps):
        k = k_grid[k_idx]
        v_promised = v_grid[v_idx]
        z = z_grid[z_idx]
        eta = eta_values[eta_idx]
        w_firm = value[k_idx, v_idx, z_idx]

        k_next_idx = policy["k_next_idx"][k_idx, v_idx, z_idx].astype(np.int64)
        k_next = k_grid[k_next_idx]

        z_next_idx = _draw_next_z_indices(rng, q, z_idx)
        eta_next_idx = rng.choice(n_eta, size=n_firms, p=eta_probs).astype(np.int64)
        z_next = z_grid[z_next_idx]
        eta_next = eta_values[eta_next_idx]

        # Realized shock-contingent contract for the drawn (z', eta').
        v_next_idx = policy["V_next_idx"][
            k_idx, v_idx, z_idx, z_next_idx, eta_next_idx
        ].astype(np.int64)
        v_next = v_grid[v_next_idx]
        dividend_next = policy["d"][k_idx, v_idx, z_idx, z_next_idx, eta_next_idx]
        w_firm_next = value[k_next_idx, v_next_idx, z_next_idx]
        default = np.zeros(n_firms, dtype=bool)

        if step >= config.burn_in:
            rows.append(
                pd.DataFrame(
                    {
                        "firm_id": firm_id,
                        "t": step - config.burn_in,
                        "k_idx": k_idx,
                        "v_idx": v_idx,
                        "z_idx": z_idx,
                        "eta_idx": eta_idx,
                        "k": k,
                        "V": v_promised,
                        "W": w_firm,
                        "z": z,
                        "eta": eta,
                        "k_next_idx": k_next_idx,
                        "v_next_idx": v_next_idx,
                        "z_next_idx": z_next_idx,
                        "eta_next_idx": eta_next_idx,
                        "k_next": k_next,
                        "V_next": v_next,
                        "W_next": w_firm_next,
                        "z_next": z_next,
                        "eta_next": eta_next,
                        "profit_pre_tax": profit_pre_tax(k, z, eta, params),
                        "dividend_next": dividend_next,
                        "default": default,
                        "delta": params.delta,
                    }
                )
            )

        k_idx = k_next_idx
        v_idx = v_next_idx
        z_idx = z_next_idx
        eta_idx = eta_next_idx

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def construct_mh_observables(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct observable states/actions for empirical policy estimation.

    Leverage uses implied debt ``b = W - V`` (firm value minus promised equity
    value); on a coarse grid this can be slightly negative, which is left as-is.
    """

    required = {
        "k",
        "V",
        "W",
        "profit_pre_tax",
        "k_next",
        "V_next",
        "W_next",
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
        out["leverage"] = (out["W"] - out["V"]) / out["k"]
        out["investment_rate"] = (
            out["k_next"] - (1.0 - out["delta"]) * out["k"]
        ) / out["k"]
        out["future_leverage"] = (out["W_next"] - out["V_next"]) / out["k_next"]
        out["payout_rate"] = out["dividend_next"] / out["k_next"]
    return out
