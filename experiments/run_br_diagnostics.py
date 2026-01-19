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
# # Bellman Residual Hyperparameter Study
#
# Dedicated script for comparing BR training stability across:
# 1. Actor vs Critic Learning Rates
# 2. Critic steps per actor step
#
# This experiment can be slow, so it is separated from the main runner.

# %%
import sys
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Robust root finding
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dnn import (
    TrainingConfig, EconomicScenario, SamplingBounds,
    train_basic_br
)
from src.dnn.networks import BasicPolicyNetwork, BasicValueNetwork
from src.dnn.trainer_basic import BasicTrainerBR
from src.economy import EconomicParams

# Output directory
OUTPUT_DIR = Path("./experiments/outputs/br_study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_MODE = "debug"  # "debug" or "full"
SEED = 33

# Re-define scenarios if needed, or import common ones if you refactor definitions
# For now, defining locally to be self-contained
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
    sampling=CUSTOM_BOUNDS
)

SCENARIO_S2 = EconomicScenario.from_overrides(
    name="full_costs",
    cost_fixed=0.05,
    cost_convex=0.1,
    cost_inject_fixed=0.0,
    cost_inject_linear=0.0,
    sampling=CUSTOM_BOUNDS
)

BR_SCENARIOS = {"S1_smooth": SCENARIO_S1, "S2_full_costs": SCENARIO_S2}
BR_N_ITER = 80 if RUN_MODE == "debug" else 200


def train_br(scenario, *, actor_lr=1e-3, critic_lr=1e-3, n_critic_steps=10, n_iter=BR_N_ITER):
    """Train BR with specified hyperparameters. Returns (history, policy_net, value_net)."""
    tf.random.set_seed(SEED)
    
    policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=16)
    value_net = BasicValueNetwork(n_layers=2, n_neurons=16)
    
    trainer = BasicTrainerBR(
        policy_net, value_net, scenario.params,
        actor_lr=actor_lr, critic_lr=critic_lr,
        batch_size=512, n_critic_steps=n_critic_steps, seed=SEED
    )
    
    history = {"iter": [], "critic": [], "actor": []}
    k_min, k_max = scenario.sampling.k_bounds
    z_min, z_max = [np.exp(x) for x in scenario.sampling.log_z_bounds]
    
    for i in range(n_iter):
        k = tf.random.uniform((512,), k_min, k_max)
        z = tf.random.uniform((512,), z_min, z_max)
        m = trainer.train_step(k, z)
        if i % 5 == 0:
            history["iter"].append(i)
            history["critic"].append(m["loss_critic"])
            history["actor"].append(m["loss_actor"])
    
    return history, policy_net, value_net


def compare_variants(scenarios, variants, title, save_name):
    """Run variants across scenarios and plot comparison."""
    # Train all combinations
    results = {}
    runs = [(s, v) for s in scenarios for v in variants]
    for i, (s_name, v_name) in enumerate(runs):
        print(f"  [{i+1}/{len(runs)}] {s_name} + {v_name}...", end=" ")
        h, _, _ = train_br(scenarios[s_name], **variants[v_name])
        results.setdefault(s_name, {})[v_name] = h
        print(f"done")
    
    # Plot
    colors = ['b', 'r', 'g', 'orange', 'purple']
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 6), sharex=True)
    if n == 1: axes = axes.reshape(-1, 1)
    
    for col, s_name in enumerate(scenarios):
        for row, metric in enumerate(["critic", "actor"]):
            ax = axes[row, col]
            for idx, (v_name, h) in enumerate(results[s_name].items()):
                ax.plot(h["iter"], h[metric], color=colors[idx], label=v_name, alpha=0.8)
            ax.set_ylabel(f"{metric.title()} Loss")
            ax.set_title(f"{s_name}: {metric.title()}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if row == 1: ax.set_xlabel("Iteration")
    
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    # Ensure directory exists before saving
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, dpi=150)
    print(f"Experiments saved to {save_path}")
    plt.show()
    return results


# %%
# =============================================================================
# 6.1 Compare: Actor LR vs Critic LR
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"BR: Comparing Actor/Critic Learning Rates (Mode: {RUN_MODE})")
    print("="*70)

    LR_VARIANTS = {
        "Slow Actor (1e-4)": {"actor_lr": 1e-4, "critic_lr": 1e-3},
        "Equal LR (1e-3)":   {"actor_lr": 1e-3, "critic_lr": 1e-3},
        "Fast Actor (3e-3)": {"actor_lr": 3e-3, "critic_lr": 1e-3},
    }

    results_lr = compare_variants(BR_SCENARIOS, LR_VARIANTS, 
                                  "BR: Actor LR Comparison", "br_lr_comparison.png")

    # %%
    # =============================================================================
    # 6.2 Compare: Critic Steps per Actor Step
    # =============================================================================

    print("\n" + "="*70)
    print("BR: Comparing Critic Steps per Actor Update")
    print("="*70)

    CRITIC_STEPS_VARIANTS = {
        "5 critic steps":  {"n_critic_steps": 5},
        "10 critic steps": {"n_critic_steps": 10},
        "20 critic steps": {"n_critic_steps": 20},
    }

    results_critic = compare_variants(BR_SCENARIOS, CRITIC_STEPS_VARIANTS,
                                      "BR: Critic Steps Comparison", "br_critic_steps_comparison.png")

    # %%
    # =============================================================================
    # 6.3 SUMMARY TABLE
    # =============================================================================

    def print_summary(results, title):
        print(f"\n{title}")
        print("-"*50)
        print(f"{'Scenario':<15} {'Variant':<18} {'Critic':>8} {'Actor':>8}")
        print("-"*50)
        for s, variants in results.items():
            for v, h in variants.items():
                print(f"{s:<15} {v:<18} {h['critic'][-1]:>8.4f} {h['actor'][-1]:>8.4f}")

    print("\n" + "="*70)
    print("BR SUMMARY")
    print("="*70)
    print_summary(results_lr, "Learning Rate Comparison")
    print_summary(results_critic, "Critic Steps Comparison")
