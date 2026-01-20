# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LR Horizon Sensitivity Diagnostics
#
# Dedicated script for comparing Lifetime Reward (LR) training across different rollout horizons $T$.
# Longer horizons reduce bias but increase variance. This study helps select the optimal $T$.

# %%
import sys
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import replace as dc_replace
from typing import Dict, Any

# Robust root finding
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dnn import (
    TrainingConfig, EconomicScenario, SamplingBounds,
    train_basic_lr,
    get_eval_grids, find_steady_state_k,
    evaluate_basic_policy,
    evaluate_policy_return,
    eval_euler_residual_basic,
)

# Output directory
OUTPUT_DIR = Path("./experiments/outputs/lr_study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_MODE = "debug"  # "debug" or "full"
SEED = 33

# Re-define scenario for self-contained execution
CUSTOM_BOUNDS = SamplingBounds(
    k_bounds=(1e-3, 10.0),
    log_z_bounds=(-5.0, 5.0),
)

SCENARIO_S1 = EconomicScenario.from_overrides(
    name="smooth_reward",
    cost_fixed=0.0,
    cost_convex=0.1,
    cost_inject_fixed=0.0,
    cost_inject_linear=0.0,
    sampling=CUSTOM_BOUNDS  # Separate from param_overrides
)


def run_lr_horizon_study(scenario: EconomicScenario, base_config: TrainingConfig, 
                         T_list: list) -> Dict[int, Dict]:
    """
    Train LR under multiple horizons T, keeping all other hyperparameters fixed.
    
    Returns:
        {T: {"history": ..., "policy_net": ..., "config": ...}}
    """
    results = {}
    for i, T in enumerate(T_list):
        print(f"  [{i+1}/{len(T_list)}] T={T}...", end=" ")
        config_T = dc_replace(base_config, T=T)
        history = train_basic_lr(scenario, config_T)
        results[T] = {
            "history": history,
            "policy_net": history["_policy_net"],
            "config": history["_config"],
            "scenario": history["_scenario"],
        }
        print(f"done. final_loss={history['loss_LR'][-1]:.4f}")
    return results

    # %%
    # =============================================================================
    # 1.1 Run LR Horizon Study
    # =============================================================================

    print("\n" + "="*70)
    print(f"LR HORIZON SENSITIVITY STUDY (Mode: {RUN_MODE})")
    print("="*70)

    # Use smooth cost scenario for cleaner comparison
    LR_SCENARIO = SCENARIO_S1

    # Base config for LR study
    LR_BASE_CONFIG = TrainingConfig(
        n_layers=2, n_neurons=16, 
        n_iter=100 if RUN_MODE == "debug" else 300,
        batch_size=256, learning_rate=3e-3, 
        log_every=10, seed=SEED, T=32
    )

    # Horizons to compare
    T_LIST = [8, 16, 32, 64] if RUN_MODE == "debug" else [8, 16, 32, 64, 128]

    print(f"Training LR with horizons: {T_LIST}")
    lr_horizon_results = run_lr_horizon_study(LR_SCENARIO, LR_BASE_CONFIG, T_LIST)

    # %%
    # =============================================================================
    # 1.2 Plot: LR Loss Curves by Horizon
    # =============================================================================

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(T_LIST)))
    for idx, T in enumerate(T_LIST):
        h = lr_horizon_results[T]["history"]
        ax.plot(h["iteration"], h["loss_LR"], color=colors[idx], label=f"T={T}", alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("LR Loss (negative discounted return)")
    ax.set_title("LR Training: Loss by Horizon T")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lr_horizon_loss.png", dpi=150)
    # plt.show()

    # %%
    # =============================================================================
    # 1.3 Policy Slice Plots by T
    # =============================================================================

    k_grid_lr, z_grid_lr, _ = get_eval_grids(LR_SCENARIO)
    delta_lr = LR_SCENARIO.params.delta

    # Plot k' vs k at z=1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    z_idx_one = np.argmin(np.abs(z_grid_lr - 1.0))
    z_one = z_grid_lr[z_idx_one]

    ax = axes[0]
    ax.plot(k_grid_lr, k_grid_lr, 'k--', label="45° line", alpha=0.5)
    for idx, T in enumerate(T_LIST):
        policy = lr_horizon_results[T]["policy_net"]
        k_tf = tf.constant(k_grid_lr.reshape(-1, 1), dtype=tf.float32)
        z_tf = tf.constant(np.full_like(k_grid_lr, z_one).reshape(-1, 1), dtype=tf.float32)
        k_next = policy(k_tf, z_tf).numpy().flatten()
        ax.plot(k_grid_lr, k_next, color=colors[idx], label=f"T={T}", alpha=0.8)

    ax.set_xlabel("k")
    ax.set_ylabel("k'")
    ax.set_title("Policy k' vs k at z=1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot I/k vs ln(z) at k_ss
    ax = axes[1]
    ln_z = np.log(z_grid_lr)

    for idx, T in enumerate(T_LIST):
        policy = lr_horizon_results[T]["policy_net"]
        k_ss = find_steady_state_k(policy, LR_SCENARIO)
        k_ss_idx = np.argmin(np.abs(k_grid_lr - k_ss))
        
        k_tf = tf.constant(np.full_like(z_grid_lr, k_grid_lr[k_ss_idx]).reshape(-1, 1), dtype=tf.float32)
        z_tf = tf.constant(z_grid_lr.reshape(-1, 1), dtype=tf.float32)
        k_next = policy(k_tf, z_tf).numpy().flatten()
        Ik = (k_next - (1 - delta_lr) * k_grid_lr[k_ss_idx]) / k_grid_lr[k_ss_idx]
        ax.plot(ln_z, Ik, color=colors[idx], label=f"T={T}", alpha=0.8)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("ln(z)")
    ax.set_ylabel("I/k")
    ax.set_title("Investment Rate vs ln(z) at Steady-State k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lr_horizon_policy.png", dpi=150)
    # plt.show()

    # %%
    # =============================================================================
    # 1.4 Evaluate Policies: Long-Run Return + Euler Residuals
    # =============================================================================

    print("\n" + "-"*60)
    print("Evaluating LR policies (return + Euler residual)...")
    print("-"*60)

    lr_eval_results = {}
    for T in T_LIST:
        policy = lr_horizon_results[T]["policy_net"]
        print(f"  T={T}: ", end="")
        
        # Long-run return
        ret = evaluate_policy_return(policy, LR_SCENARIO, n_paths=50, T_eval=200, burn_in=50, seed=SEED)
        
        # Euler residual (only if smooth scenario)
        try:
            euler = eval_euler_residual_basic(policy, LR_SCENARIO, n_states=200, seed=SEED)
        except Exception as e:
            euler = {"mean_abs": float('nan'), "median_abs": float('nan'), "p90_abs": float('nan')}
            print(f"(Euler skipped: {e}) ", end="")
        
        lr_eval_results[T] = {"return": ret, "euler": euler}
        print(f"mean_reward={ret['mean_reward']:.4f}, euler_mean={euler['mean_abs']:.4f}")

    # %%
    # =============================================================================
    # 1.5 Summary Table
    # =============================================================================

    print("\n" + "="*70)
    print("LR HORIZON STUDY SUMMARY")
    print("="*70)

    print(f"\n{'T':<6} {'Final Loss':<12} {'Mean Reward':<12} {'Euler MAE':<12} {'k_ss':<8}")
    print("-"*55)

    for T in T_LIST:
        h = lr_horizon_results[T]["history"]
        policy = lr_horizon_results[T]["policy_net"]
        k_ss = find_steady_state_k(policy, LR_SCENARIO)
        ret = lr_eval_results[T]["return"]
        euler = lr_eval_results[T]["euler"]
        
        print(f"{T:<6} {h['loss_LR'][-1]:<12.4f} {ret['mean_reward']:<12.4f} "
              f"{euler['mean_abs']:<12.4f} {k_ss:<8.3f}")

    print("-"*55)
    print(f"Figures saved to: {OUTPUT_DIR.absolute()}")
