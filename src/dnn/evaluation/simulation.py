"""
src/dnn/evaluation/simulation.py

Policy simulation and Monte Carlo evaluation.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from src.economy import compute_cash_flow_basic

# =============================================================================
# AR(1) HELPER (NumPy version for simulation loops)
# =============================================================================

def _ar1_step_numpy(z: float, rho: float, sigma: float, mu: float, 
                    rng: np.random.Generator) -> float:
    """
    Single-step AR(1) transition for productivity z.
    
    log(z') = (1 - rho) * mu + rho * log(z) + sigma * eps
    
    Same process as BasicTrainerLR and draw_shocks, but for numpy scalars.
    """
    log_z = np.log(max(z, 1e-8))
    eps = rng.standard_normal()
    log_z_next = (1 - rho) * mu + rho * log_z + sigma * eps
    return float(np.exp(log_z_next))


# =============================================================================
# POLICY SIMULATION
# =============================================================================

def simulate_policy_path(policy_net, scenario, *, T_eval, k0, z0, seed=None):
    """
    Simulate one path under policy with AR(1) z transitions.
    
    Uses the same shock process as BasicTrainerLR:
        log_z_next = (1-rho)*mu + rho*log_z + sigma*eps
    
    Args:
        policy_net: Trained BasicPolicyNetwork
        scenario: EconomicScenario (provides params with rho, sigma, mu)
        T_eval: Number of steps to simulate
        k0, z0: Initial capital and productivity
        seed: Random seed for reproducibility
    
    Returns:
        rewards: (T_eval,) array of per-step rewards
        k_path: (T_eval+1,) array of capital states
        z_path: (T_eval+1,) array of productivity states
    """
    rng = np.random.default_rng(seed)
    params = scenario.params
    
    k_path = [k0]
    z_path = [z0]
    rewards = []
    
    k, z = k0, z0
    for t in range(T_eval):
        k_tf = tf.constant([[k]], dtype=tf.float32)
        z_tf = tf.constant([[z]], dtype=tf.float32)
        k_next = float(policy_net(k_tf, z_tf).numpy()[0, 0])
        
        # Reward
        e = compute_cash_flow_basic(k_tf, tf.constant([[k_next]]), z_tf, params)
        rewards.append(float(e.numpy()[0, 0]))
        
        # AR(1) transition for z using params
        z = _ar1_step_numpy(z, params.rho, params.sigma, params.mu, rng)
        k = k_next
        
        k_path.append(k)
        z_path.append(z)
    
    return np.array(rewards), np.array(k_path), np.array(z_path)


def evaluate_policy_return(policy_net, scenario, *, n_paths=100, T_eval=500, 
                           burn_in=100, seed=42) -> Dict:
    """
    Evaluate long-run return of a policy via Monte Carlo simulation.
    
    Args:
        policy_net: Trained BasicPolicyNetwork
        scenario: EconomicScenario
        n_paths: Number of independent paths
        T_eval: Steps per path
        burn_in: Steps to discard at start
        seed: Random seed
    
    Returns:
        Dict with mean/std/percentiles of per-period reward (after burn-in).
    """
    rng = np.random.default_rng(seed)
    # Using sampling bounds to initialize paths
    bounds = scenario.sampling
    k_min, k_max = bounds.k_bounds
    z_min, z_max = [np.exp(x) for x in bounds.log_z_bounds]
    
    all_rewards = []
    for p in range(n_paths):
        k0 = rng.uniform(k_min, k_max)
        z0 = rng.uniform(z_min, z_max)
        rewards, _, _ = simulate_policy_path(policy_net, scenario, T_eval=T_eval, 
                                             k0=k0, z0=z0, seed=seed + p)
        all_rewards.append(rewards[burn_in:])
    
    all_rewards = np.concatenate(all_rewards)
    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "p10_reward": float(np.percentile(all_rewards, 10)),
        "p50_reward": float(np.percentile(all_rewards, 50)),
        "p90_reward": float(np.percentile(all_rewards, 90)),
    }
