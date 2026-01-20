"""
src/dnn/evaluation/residuals.py

Physics-based model checks (Euler residuals, Bellman residuals).
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional

from src.economy import (
    euler_chi,
    euler_m,
)
from src.dnn.sampling import draw_shocks

def eval_euler_residual_basic(policy_net, scenario, *, n_states=500, seed=42) -> Dict:
    """
    Compute Euler residual diagnostics for a trained Basic policy (out-of-sample).
    
    Evaluates how well the policy satisfies Euler equation:
        chi(k,z) = beta * E[m(k', z')]
    
    Uses the same Euler residual formulation as ER trainer with AiO estimator.
    
    Args:
        policy_net: Trained BasicPolicyNetwork
        scenario: EconomicScenario
        n_states: Number of test states to sample
        seed: Random seed
    
    Returns:
        Dict with mean_abs, median_abs, p90_abs, p95_abs, max_abs of residuals.
    """
    rng = np.random.default_rng(seed)
    bounds = scenario.sampling
    params = scenario.params
    beta = 1.0 / (1.0 + params.r_rate)
    
    k_min, k_max = bounds.k_bounds
    z_min, z_max = [np.exp(x) for x in bounds.log_z_bounds]
    
    # Sample states
    k = rng.uniform(k_min, k_max, n_states)
    z = rng.uniform(z_min, z_max, n_states)
    
    k_tf = tf.constant(k.reshape(-1, 1), dtype=tf.float32)
    z_tf = tf.constant(z.reshape(-1, 1), dtype=tf.float32)
    
    # k' = policy(k, z)
    k_next = policy_net(k_tf, z_tf)
    
    # Current chi = 1 + psi_I (marginal cost of investment)
    chi_curr = euler_chi(k_tf, k_next, params)
    
    # Two independent z' draws for AiO estimator
    n = tf.shape(k_tf)[0]
    z_next_1, z_next_2 = draw_shocks(n, z_tf, params.rho, params.sigma, params.mu)
    
    # For each z', compute k'' = policy(k', z')
    k_next_1 = policy_net(k_next, z_next_1)
    k_next_2 = policy_net(k_next, z_next_2)
    
    # Compute m(k', z', k'') using euler_m from economy module
    m_1 = euler_m(k_next, k_next_1, z_next_1, params)
    m_2 = euler_m(k_next, k_next_2, z_next_2, params)
    
    # Euler residual: f = chi - beta * E[m]
    # Using AiO estimator: E[m] ≈ 0.5 * (m_1 + m_2)
    f = chi_curr - beta * 0.5 * (m_1 + m_2)
    f_abs = tf.abs(f).numpy().flatten()
    
    return {
        "mean_abs": float(np.mean(f_abs)),
        "median_abs": float(np.median(f_abs)),
        "p90_abs": float(np.percentile(f_abs, 90)),
        "p95_abs": float(np.percentile(f_abs, 95)),
        "max_abs": float(np.max(f_abs)),
    }
