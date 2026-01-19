# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DNN Experiments: Basic Model Sanity Suite
# 
# Structured comparison of LR vs ER vs BR training methods across two scenarios:
# - **S1 (smooth_reward)**: cost_fixed=0, no external finance cost → smooth reward
# - **S2 (full_costs)**: all costs enabled → hard indicators (STE)
# 
# Fast debug mode for iteration; full mode for quality results.

# %%
# =============================================================================
# 0. IMPORTS & SETUP
# =============================================================================
import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from src.dnn import (
    TrainingConfig, EconomicScenario, SamplingBounds,
    train_basic_lr, train_basic_er, train_basic_br,
    get_eval_grids, evaluate_basic_policy,
    find_steady_state_k, simulate_policy_path,
    evaluate_policy_return, eval_euler_residual_basic,
)

# Output directory
OUTPUT_DIR = Path("./outputs/dnn_experiments/basic_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# 0. DEBUG / FULL CONFIGURATION
# =============================================================================

RUN_MODE = "debug"  # "debug" (fast) or "full" (slow but accurate)
SEED = 33
VERBOSE = False

# Plotting defaults
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10

# --- Config presets ---
if RUN_MODE == "debug":
    CONFIG = TrainingConfig(
        n_layers=2, n_neurons=16, n_iter=50,
        batch_size=128, learning_rate=3e-3, log_every=10, seed=SEED
    )
    print(f"DEBUG MODE: n_iter={CONFIG.n_iter}, batch={CONFIG.batch_size}, neurons={CONFIG.n_neurons}")
else:
    CONFIG = TrainingConfig(
        n_layers=2, n_neurons=32, n_iter=200,
        batch_size=512, learning_rate=1e-3, log_every=10, seed=SEED
    )
    print(f"FULL MODE: n_iter={CONFIG.n_iter}, batch={CONFIG.batch_size}, neurons={CONFIG.n_neurons}")

# %%
# =============================================================================
# 0. SCENARIO DEFINITIONS
# =============================================================================

# S1: Smooth reward (no hard indicators)
SCENARIO_S1 = EconomicScenario.from_overrides(
    name="smooth_reward",
    cost_fixed=0.0,           # No fixed adjustment cost
    cost_convex=0.1,          # Keep convex cost for some friction
    cost_inject_fixed=0.0,    # No external finance cost
    cost_inject_linear=0.0,   # No external finance cost
)

# S2: Full costs (hard indicators enabled)
SCENARIO_S2 = EconomicScenario.from_overrides(
    name="full_costs",
    cost_fixed=0.05,          # Fixed adjustment cost → inaction region
    cost_convex=0.1,          # Convex cost
    cost_inject_fixed=0.0,    # Keep external finance off for basic model
    cost_inject_linear=0.0,
)

print(f"S1 (smooth_reward): cost_fixed={SCENARIO_S1.params.cost_fixed}")
print(f"S2 (full_costs): cost_fixed={SCENARIO_S2.params.cost_fixed}")


# %%
# =============================================================================
# HELPER: Run Training Suite
# =============================================================================

def run_basic_training_suite(scenario: EconomicScenario, config: TrainingConfig) -> Dict[str, Dict]:
    """Run LR, ER, BR training for a given scenario."""
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Training: {scenario.name}")
    print(f"{'='*60}")
    
    # --- LR ---
    print("  [1/3] Training LR...", end=" ")
    history_lr = train_basic_lr(scenario, config)
    results["LR"] = {
        "history": history_lr,
        "policy_net": history_lr["_policy_net"],
        "final_loss": history_lr["loss_LR"][-1],
    }
    print(f"done. Final loss: {results['LR']['final_loss']:.4f}")
    
    # --- ER ---
    print("  [2/3] Training ER...", end=" ")
    history_er = train_basic_er(scenario, config)
    results["ER"] = {
        "history": history_er,
        "policy_net": history_er["_policy_net"],
        "final_loss": history_er["loss_ER"][-1],
    }
    print(f"done. Final loss: {results['ER']['final_loss']:.6f}")
    
    # --- BR ---
    print("  [3/3] Training BR...", end=" ")
    history_br = train_basic_br(scenario, config)
    results["BR"] = {
        "history": history_br,
        "policy_net": history_br["_policy_net"],
        "final_loss_critic": history_br["loss_BR_critic"][-1],
        "final_loss_actor": history_br["loss_BR_actor"][-1],
    }
    print(f"done. Critic: {results['BR']['final_loss_critic']:.6f}, Actor: {results['BR']['final_loss_actor']:.4f}")
    
    return results

# %%
# =============================================================================
# 2. RUN TRAINING SUITE
# =============================================================================

# Train both scenarios
results_s1 = run_basic_training_suite(SCENARIO_S1, CONFIG)
results_s2 = run_basic_training_suite(SCENARIO_S2, CONFIG)

# Combined results dict
ALL_RESULTS = {
    ("smooth_reward", "LR"): results_s1["LR"],
    ("smooth_reward", "ER"): results_s1["ER"],
    ("smooth_reward", "BR"): results_s1["BR"],
    ("full_costs", "LR"): results_s2["LR"],
    ("full_costs", "ER"): results_s2["ER"],
    ("full_costs", "BR"): results_s2["BR"],
}

# %%
# =============================================================================
# 3. PLOT: Loss Curves (3 columns: LR / ER / BR)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# LR losses
ax = axes[0]
ax.plot(results_s1["LR"]["history"]["loss_LR"], 'b-', label="S1: smooth", alpha=0.8)
ax.plot(results_s2["LR"]["history"]["loss_LR"], 'r--', label="S2: full_costs", alpha=0.8)
ax.set_xlabel("Iteration (10x)")
ax.set_ylabel("LR Loss (neg. reward)")
ax.set_title("LR: Lifetime Reward")
ax.legend()
ax.grid(True, alpha=0.3)

# ER losses
ax = axes[1]
ax.plot(results_s1["ER"]["history"]["loss_ER"], 'b-', label="S1: smooth", alpha=0.8)
ax.plot(results_s2["ER"]["history"]["loss_ER"], 'r--', label="S2: full_costs", alpha=0.8)
ax.set_xlabel("Iteration (10x)")
ax.set_ylabel("ER Loss (Euler residual)")
ax.set_title("ER: Euler Residual")
ax.legend()
ax.grid(True, alpha=0.3)

# BR losses
ax = axes[2]
ax.plot(results_s1["BR"]["history"]["loss_BR_critic"], 'b-', label="S1 Critic", alpha=0.8)
ax.plot(results_s1["BR"]["history"]["loss_BR_actor"], 'b:', label="S1 Actor", alpha=0.8)
ax.plot(results_s2["BR"]["history"]["loss_BR_critic"], 'r--', label="S2 Critic", alpha=0.8)
ax.plot(results_s2["BR"]["history"]["loss_BR_actor"], 'r:', label="S2 Actor", alpha=0.8)
ax.set_xlabel("Iteration (10x)")
ax.set_ylabel("BR Loss")
ax.set_title("BR: Actor-Critic (cross-product)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 4. EVALUATION GRIDS
# =============================================================================

# Get evaluation grids
k_grid, z_grid, _ = get_eval_grids(SCENARIO_S1, n_k=50, n_z=20)
delta = SCENARIO_S1.params.delta

# Find steady-state k for z=1
k_ss_s1 = {m: find_steady_state_k(results_s1[m]["policy_net"], SCENARIO_S1) for m in ["LR", "ER", "BR"]}
k_ss_s2 = {m: find_steady_state_k(results_s2[m]["policy_net"], SCENARIO_S2) for m in ["LR", "ER", "BR"]}

print("\nSteady-state capital (z=1):")
for m in ["LR", "ER", "BR"]:
    print(f"  {m}: S1={k_ss_s1[m]:.3f}, S2={k_ss_s2[m]:.3f}")

# %%
# =============================================================================
# 4.1 PLOT: k' vs k (z fixed at z=1)
# =============================================================================

z_slice_idx = len(z_grid) // 2  # z=1 (median)
z_slice = z_grid[z_slice_idx]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for idx, method in enumerate(["LR", "ER", "BR"]):
    ax = axes[idx]
    
    eval_s1 = evaluate_basic_policy(results_s1[method]["policy_net"], k_grid, z_grid)
    eval_s2 = evaluate_basic_policy(results_s2[method]["policy_net"], k_grid, z_grid)
    
    ax.plot(k_grid, k_grid, 'k--', label="45° (k'=k)", alpha=0.5)
    ax.plot(k_grid, eval_s1["k_next"][:, z_slice_idx], 'b-', label="S1: smooth", alpha=0.8)
    ax.plot(k_grid, eval_s2["k_next"][:, z_slice_idx], 'r--', label="S2: full_costs", alpha=0.8)
    
    ax.set_xlabel("Current k")
    ax.set_ylabel("k' (next period)")
    ax.set_title(f"{method}: k' vs k (z={z_slice:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kprime_vs_k.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 4.2 PLOT: k' vs log(z) (k fixed at k_ss)
# =============================================================================

# Each method uses its own steady-state k for the plots
ln_z = np.log(z_grid)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for idx, method in enumerate(["LR", "ER", "BR"]):
    ax = axes[idx]
    
    # Use this method's k_ss (from S1 scenario)
    k_slice = k_ss_s1[method]
    k_slice_idx = np.argmin(np.abs(k_grid - k_slice))
    
    eval_s1 = evaluate_basic_policy(results_s1[method]["policy_net"], k_grid, z_grid)
    eval_s2 = evaluate_basic_policy(results_s2[method]["policy_net"], k_grid, z_grid)
    
    ax.plot(ln_z, eval_s1["k_next"][k_slice_idx, :], 'b-', label="S1: smooth", alpha=0.8)
    ax.plot(ln_z, eval_s2["k_next"][k_slice_idx, :], 'r--', label="S2: full_costs", alpha=0.8)
    
    ax.set_xlabel("ln(z)")
    ax.set_ylabel("k' (next period)")
    ax.set_title(f"{method}: k' vs ln(z) (k={k_slice:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kprime_vs_lnz.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 4.3 PLOT: Investment Rate I/k vs log(z)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for idx, method in enumerate(["LR", "ER", "BR"]):
    ax = axes[idx]
    
    # Use this method's k_ss (from S1 scenario)
    k_slice = k_ss_s1[method]
    k_slice_idx = np.argmin(np.abs(k_grid - k_slice))
    
    eval_s1 = evaluate_basic_policy(results_s1[method]["policy_net"], k_grid, z_grid)
    eval_s2 = evaluate_basic_policy(results_s2[method]["policy_net"], k_grid, z_grid)
    
    # I/k = (k' - (1-delta)*k) / k
    Ik_s1 = (eval_s1["k_next"][k_slice_idx, :] - (1 - delta) * k_grid[k_slice_idx]) / k_grid[k_slice_idx]
    Ik_s2 = (eval_s2["k_next"][k_slice_idx, :] - (1 - delta) * k_grid[k_slice_idx]) / k_grid[k_slice_idx]
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.plot(ln_z, Ik_s1, 'b-', label="S1: smooth", alpha=0.8)
    ax.plot(ln_z, Ik_s2, 'r--', label="S2: full_costs", alpha=0.8)
    
    ax.set_xlabel("ln(z)")
    ax.set_ylabel("I/k (investment rate)")
    ax.set_title(f"{method}: Investment Rate (k={k_slice:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "investment_rate_vs_lnz.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 5. SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Scenario':<15} {'Method':<6} {'Final Loss':<15} {'k_ss (z=1)':<10}")
print("-"*70)

for method in ["LR", "ER", "BR"]:
    loss_s1 = results_s1[method].get("final_loss", results_s1[method].get("final_loss_actor", 0))
    loss_s2 = results_s2[method].get("final_loss", results_s2[method].get("final_loss_actor", 0))
    print(f"{'smooth_reward':<15} {method:<6} {loss_s1:<15.6f} {k_ss_s1[method]:<10.3f}")
    print(f"{'full_costs':<15} {method:<6} {loss_s2:<15.6f} {k_ss_s2[method]:<10.3f}")

print("="*70)
print(f"\nFigures saved to: {OUTPUT_DIR.absolute()}")

# %%

