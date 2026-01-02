"""
visuals.py

Visualization utilities for Corporate Finance Structural Models.
This module is a PURE CONSUMER of the results produced by simulation.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any


# --- 1. Global Style Settings ---
def set_plot_style():
    """Sets professional publication-quality standards."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


# =============================================================================
# SECTION A: Basic Investment Model (2D States: z, k)
# =============================================================================

def plot_investment_policies(results_dict: Dict[str, Any], z_idx: int = None):
    """
    Plots a 2x2 dashboard for the Basic Investment Model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_config = [
        {"key": "policy", "is_moment": False, "title": "Policy Function $k'(k)$", "ylabel": "Next Capital $k'$"},
        {"key": "value", "is_moment": False, "title": "Value Function $V(k)$", "ylabel": "Value"},
        {"key": "invest_rate", "is_moment": True, "title": "Investment Rate $i/k$", "ylabel": "Rate"},
        {"key": "financial_surplus", "is_moment": True, "title": "Financial Surplus", "ylabel": "Ratio (CF/k)"}
    ]

    # Defaults
    first_key = next(iter(results_dict))
    if z_idx is None:
        z_idx = results_dict[first_key]['grids']['z'].shape[0] // 2
        print(f"Plotting for median productivity shock: z index {z_idx}")

    for name, data in results_dict.items():
        k_grid = data['grids']['k']

        for i, ax in enumerate(axes):
            cfg = plot_config[i]
            # Handle nested keys
            if cfg['is_moment']:
                matrix = data['moments'][cfg['key']]
            else:
                matrix = data[cfg['key']]

            y_values = matrix[z_idx, :]
            ax.plot(k_grid, y_values, linewidth=2.5, label=name)

    # Formatting
    for i, ax in enumerate(axes):
        cfg = plot_config[i]
        ax.set_title(cfg["title"], fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel("Capital Stock ($k$)", fontsize=14)
        ax.set_ylabel(cfg["ylabel"], fontsize=14)
        ax.legend(frameon=True, fancybox=True, loc='best')

        if i == 0:  # 45-degree line for policy
            min_val, max_val = ax.get_xlim()
            ax.plot([min_val, max_val], [min_val, max_val],
                    color='gray', linestyle='--', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.show()


def _find_investment_steady_state(data: Dict[str, Any], z_idx: int):
    """
    Helper: Finds the endogenous steady state capital k_ss where k' ~ k.
    """
    k_grid = data['grids']['k']
    # Policy is (nz, nk), we grab the slice at the specific z index
    policy_k = data['policy'][z_idx, :]

    # Find index where distance |k'(k) - k| is minimized
    dist = np.abs(policy_k - k_grid)
    k_idx_ss = np.argmin(dist)

    return k_idx_ss, k_grid[k_idx_ss]


def plot_investment_moments_vs_shocks(results_dict: Dict[str, Any]):
    """
    Plots Investment Rate and Surplus against Log Productivity (log z).

    Fixed at the Baseline's Endogenous Steady State (k_ss).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes = axes.flatten()

    plot_config = [
        {"key": "invest_rate", "title": "Investment Rate vs. Shock", "ylabel": "Inv. Rate ($i/k$)"},
        {"key": "financial_surplus", "title": "Financial Surplus vs. Shock", "ylabel": "Surplus Ratio"}
    ]

    # 1. Determine Slice Point (Steady State of Baseline)
    first_key = next(iter(results_dict))
    base_data = results_dict[first_key]

    # Find SS at median Z (long-run mean)
    z_mid = base_data['grids']['z'].shape[0] // 2
    k_idx_ss, k_ss = _find_investment_steady_state(base_data, z_mid)

    print(f"Plotting Investment Moments at Baseline Steady State:\n  k_ss = {k_ss:.3f} (Index {k_idx_ss})")

    # --- Plotting Loop ---
    for name, data in results_dict.items():
        z_grid = data['grids']['z']

        # TRANSFORMATION: Convert z to log(z) for centering
        z_log = np.log(z_grid)

        for i, ax in enumerate(axes):
            key = plot_config[i]["key"]
            matrix = data['moments'][key]

            # Slice at the calculated Steady State Index
            y_values = matrix[:, k_idx_ss]

            # Plot against Log Z
            ax.plot(z_log, y_values, marker='o', markersize=6, linewidth=2.5, label=name)

    # --- Formatting ---
    fig.suptitle(f"Moments at Steady State Capital $k_{{ss}} \\approx {k_ss:.2f}$", fontsize=16, y=1.05)

    for i, ax in enumerate(axes):
        ax.set_title(plot_config[i]["title"], fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Log Productivity ($\log z$)", fontsize=13)
        ax.set_ylabel(plot_config[i]["ylabel"], fontsize=13)
        ax.legend()

        # Reference lines
        ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION B: Debt Model (3D States: z, k, b)
# =============================================================================

def plot_debt_policies(results_dict: Dict[str, Any], z_idx: int = None, k_idx: int = None):
    """
    Visualizes Lender's Pricing (Interest Rate) and Firm's Debt Policy.
    Slices the 3D state space at fixed (z, k).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_rate, ax_policy = axes

    # Defaults
    first_key = next(iter(results_dict))
    first_data = results_dict[first_key]
    b_grid = first_data['grids']['b']

    if z_idx is None: z_idx = first_data['grids']['z'].shape[0] // 2
    if k_idx is None: k_idx = first_data['grids']['k'].shape[0] // 2

    k_val = first_data['grids']['k'][k_idx]
    z_val = first_data['grids']['z'][z_idx]

    for name, data in results_dict.items():
        # 1. Risky Rate from 'prices' (As defined in simulation.py)
        r_values = data['prices']['risky_rate'][z_idx, k_idx, :]
        ax_rate.plot(b_grid, r_values, linewidth=2.5, label=name)

        # 2. Debt Policy
        b_next_values = data['policy']['b'][z_idx, k_idx, :]
        ax_policy.plot(b_grid, b_next_values, linewidth=2.5, label=name)

    # Formatting
    ax_rate.set_title(f"Lender's Required Return\n($z={z_val:.2f}, k={k_val:.1f}$)", fontsize=14, fontweight='bold')
    ax_rate.set_xlabel("Next Period Debt ($b'$)", fontsize=13)
    ax_rate.set_ylabel(r"Risky Rate ($\tilde{r}$)", fontsize=13)

    r_rate = first_data['params'].r_rate
    ax_rate.axhline(r_rate, color='grey', linestyle='--', label=f"Risk-Free ({r_rate:.1%})")
    ax_rate.legend(loc='upper left')

    ax_policy.set_title(f"Debt Issuance Policy\n($z={z_val:.2f}, k={k_val:.1f}$)", fontsize=14, fontweight='bold')
    ax_policy.set_xlabel("Current Debt ($b$)", fontsize=13)
    ax_policy.set_ylabel("Optimal Next Debt ($b'$)", fontsize=13)

    min_b, max_b = b_grid[0], b_grid[-1]
    ax_policy.plot([min_b, max_b], [min_b, max_b], color='grey', linestyle=':', label="45-deg line")
    ax_policy.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def _find_joint_steady_state(data: Dict[str, Any], z_idx: int):
    """Vectorized search for endogenous steady state (k_ss, b_ss)."""
    k_grid = data['grids']['k']
    b_grid = data['grids']['b']

    pol_k = data['policy']['k'][z_idx, :, :]
    pol_b = data['policy']['b'][z_idx, :, :]

    k_mesh = k_grid[:, np.newaxis]
    b_mesh = b_grid[np.newaxis, :]

    dist = np.abs(pol_k - k_mesh) + np.abs(pol_b - b_mesh)
    min_idx_flat = np.argmin(dist)
    k_idx_ss, b_idx_ss = np.unravel_index(min_idx_flat, dist.shape)

    return k_idx_ss, b_idx_ss, k_grid[k_idx_ss], b_grid[b_idx_ss]


def plot_debt_moments_vs_shocks(results_dict: Dict[str, Any]):
    """
    Plots key Debt Model moments (Inv, Lev, Div) against productivity shocks.
    Fixed at the Baseline's Endogenous Steady State (k_ss).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_inv, ax_lev, ax_div = axes.flatten()

    # Determine Slice Point
    first_key = next(iter(results_dict))
    base_data = results_dict[first_key]
    z_mid = base_data['grids']['z'].shape[0] // 2
    k_idx_ss, b_idx_ss, k_ss, b_ss = _find_joint_steady_state(base_data, z_mid)

    print(f"Moments vs Shocks (Slice at Baseline SS):\n  k_ss = {k_ss:.3f}, b_ss = {b_ss:.3f}")

    for name, data in results_dict.items():
        z_grid_log = np.log(data['grids']['z'])

        # Inv Rate
        inv_vals = data['moments']['invest_rate'][:, k_idx_ss, b_idx_ss]
        ax_inv.plot(z_grid_log, inv_vals, marker='o', markersize=4, label=name)

        # Leverage
        lev_vals = data['moments']['leverage_ratio'][:, k_idx_ss, b_idx_ss]
        ax_lev.plot(z_grid_log, lev_vals, marker='s', markersize=4, label=name)

        # Dividends
        div_vals = data['moments']['dividends'][:, k_idx_ss, b_idx_ss] / k_ss
        ax_div.plot(z_grid_log, div_vals, marker='^', markersize=4, label=name)

    titles = ["Investment Rate ($i/k$)", "Leverage Ratio ($b'/k$)", "Distributions ($d/k$)"]
    for i, ax in enumerate([ax_inv, ax_lev, ax_div]):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel("Log Productivity ($\log z$)", fontsize=12)
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
        ax.legend()

    fig.suptitle(f"Response to Shocks at Steady State ($k={k_ss:.2f}, b={b_ss:.2f}$)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()