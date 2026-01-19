"""
src/dnn/evaluation.py

Evaluation utilities for trained DNN models.
Computes policy, value, and economic quantities on grids in LEVELS.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

# Economy module imports
from src.economy import (
    compute_cash_flow_basic,
    euler_chi,      # Was incorrectly called compute_chi 
    euler_m,        # Computes RHS of Euler equation
    adjustment_costs,
    production_function,
)

# Sampling utilities (AR(1) shock process)
from src.dnn.sampling import draw_shocks


def get_eval_grids(
    source: Union[Dict, "EconomicScenario"],
    n_k: int = 50,
    n_z: int = 10,
    n_b: int = 20
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate evaluation grids from scenario/history bounds.
    
    Uses the same bounds as training to avoid extrapolation.
    
    Args:
        source: Either a training history dict (with "_scenario" key)
                or an EconomicScenario directly
        n_k: Number of points for capital grid
        n_z: Number of points for productivity grid
        n_b: Number of points for debt grid (risky model)
    
    Returns:
        (k_grid, z_grid, b_grid) where:
            - k_grid: 1D array of capital values in levels
            - z_grid: 1D array of productivity values in levels
            - b_grid: 1D array of debt values (or None for basic model)
    
    Example:
        k_grid, z_grid, _ = get_eval_grids(history)
        eval_result = evaluate_basic_policy(history["_policy_net"], k_grid, z_grid)
    """
    # Extract bounds from source
    if isinstance(source, dict):
        # Training history dict
        scenario = source.get("_scenario")
        if scenario is None:
            raise ValueError("History dict must contain '_scenario' key")
        bounds = scenario.sampling  # EconomicScenario.sampling is SamplingBounds
    else:
        # Assume it's an EconomicScenario
        bounds = source.sampling  # EconomicScenario.sampling is SamplingBounds
    
    # Extract individual bounds
    k_min, k_max = bounds.k_bounds
    log_z_min, log_z_max = bounds.log_z_bounds
    b_min, b_max = bounds.b_bounds
    
    # Get delta from scenario params (single source of truth)
    if isinstance(source, dict):
        delta = source["_scenario"].params.delta
    else:
        delta = source.params.delta
    
    # Create k_grid using (1-δ) multiplicative grid
    # Grid points satisfy: k_grid[i] = (1-δ) * k_grid[i+1]  (i.e., I=0 inaction)
    # Equivalently: k_grid[i+1] / k_grid[i] = 1 / (1-δ)
    # 
    # Auto-compute n_k to span [k_min, k_max] with ratio r = 1/(1-δ):
    #   k_max = k_min * r^{n-1}  =>  n = 1 + log(k_max/k_min) / log(r)
    g = 1 - delta  # decay factor
    r = 1 / g      # growth factor per grid step
    
    # Compute number of grid points needed to span [k_min, k_max]
    n_k_auto = int(np.ceil(1 + np.log(k_max / k_min) / np.log(r)))
    n_k_auto = max(n_k_auto, 2)  # at least 2 points
    
    # Use auto-computed n_k (override the input n_k to match depreciation structure)
    n_k = n_k_auto
    
    # Build grid: k_grid[i] = k_min * r^i for i = 0, 1, ..., n_k-1
    exponents = np.arange(n_k)
    k_grid = k_min * (r ** exponents)
    
    # Clip last point to exactly k_max if overshoot
    k_grid = np.clip(k_grid, k_min, k_max)
    
    # Ensure strictly increasing (handle any numerical duplicates)
    for i in range(1, len(k_grid)):
        if k_grid[i] <= k_grid[i-1]:
            k_grid[i] = k_grid[i-1] + 1e-8
    
    # z_grid and b_grid unchanged (linear/log as before)
    z_grid = np.exp(np.linspace(log_z_min, log_z_max, n_z))  # Convert from log to levels
    b_grid = np.linspace(b_min, b_max, n_b)
    
    return k_grid, z_grid, b_grid


def evaluate_basic_policy(
    policy_net,
    k_grid: np.ndarray,
    z_grid: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Evaluate Basic policy on a meshgrid.
    
    Args:
        policy_net: Trained BasicPolicyNetwork
        k_grid: 1D array of capital values (levels)
        z_grid: 1D array of productivity values (levels)
    
    Returns:
        Dict with:
            - k: meshgrid of k values (n_k, n_z)
            - z: meshgrid of z values (n_k, n_z)
            - k_next: policy output k'(k, z) (n_k, n_z)
            - I_k: investment rate I/k (n_k, n_z)
    """
    n_k, n_z = len(k_grid), len(z_grid)
    
    # Create meshgrid
    k_mesh, z_mesh = np.meshgrid(k_grid, z_grid, indexing='ij')
    
    # Flatten for network
    k_flat = k_mesh.flatten()
    z_flat = z_mesh.flatten()
    
    k_tf = tf.constant(k_flat, dtype=tf.float32)
    z_tf = tf.constant(z_flat, dtype=tf.float32)
    
    # Forward pass
    k_next_flat = policy_net(k_tf, z_tf).numpy().flatten()
    
    # Reshape
    k_next = k_next_flat.reshape(n_k, n_z)
    
    # Compute investment rate I/k
    # Assumes delta is not available, return raw I/k without depreciation
    # User can compute full I = k' - (1-delta)*k separately
    I_k = (k_next - k_mesh) / k_mesh
    
    return {
        "k": k_mesh,
        "z": z_mesh,
        "k_next": k_next,
        "I_k": I_k
    }


def evaluate_basic_value(
    value_net,
    k_grid: np.ndarray,
    z_grid: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Evaluate Basic value function on a meshgrid.
    
    Args:
        value_net: Trained BasicValueNetwork
        k_grid: 1D array of capital values
        z_grid: 1D array of productivity values
    
    Returns:
        Dict with k, z meshgrids and V(k, z)
    """
    n_k, n_z = len(k_grid), len(z_grid)
    k_mesh, z_mesh = np.meshgrid(k_grid, z_grid, indexing='ij')
    
    k_flat = k_mesh.flatten()
    z_flat = z_mesh.flatten()
    
    k_tf = tf.constant(k_flat, dtype=tf.float32)
    z_tf = tf.constant(z_flat, dtype=tf.float32)
    
    V_flat = value_net(k_tf, z_tf).numpy().flatten()
    V = V_flat.reshape(n_k, n_z)
    
    return {
        "k": k_mesh,
        "z": z_mesh,
        "V": V
    }


def evaluate_risky_policy(
    policy_net,
    price_net,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_val: float
) -> Dict[str, np.ndarray]:
    """
    Evaluate Risky Debt policy at fixed z on (k, b) meshgrid.
    
    Args:
        policy_net: Trained RiskyPolicyNetwork
        price_net: Trained RiskyPriceNetwork (or None)
        k_grid: 1D array of capital values
        b_grid: 1D array of debt values
        z_val: Fixed productivity level
    
    Returns:
        Dict with k, b meshgrids, k_next, b_next, r_tilde, I_k, leverage
    """
    n_k, n_b = len(k_grid), len(b_grid)
    k_mesh, b_mesh = np.meshgrid(k_grid, b_grid, indexing='ij')
    z_mesh = np.full_like(k_mesh, z_val)
    
    k_flat = k_mesh.flatten()
    b_flat = b_mesh.flatten()
    z_flat = z_mesh.flatten()
    
    k_tf = tf.constant(k_flat, dtype=tf.float32)
    b_tf = tf.constant(b_flat, dtype=tf.float32)
    z_tf = tf.constant(z_flat, dtype=tf.float32)
    
    # Policy
    k_next_tf, b_next_tf = policy_net(k_tf, b_tf, z_tf)
    k_next = k_next_tf.numpy().flatten().reshape(n_k, n_b)
    b_next = b_next_tf.numpy().flatten().reshape(n_k, n_b)
    
    # Price
    if price_net is not None:
        r_tilde_tf = price_net(k_next_tf, b_next_tf, z_tf)
        r_tilde = r_tilde_tf.numpy().flatten().reshape(n_k, n_b)
    else:
        r_tilde = None
    
    # Investment rate I/k
    I_k = (k_next - k_mesh) / k_mesh
    
    # Leverage b'/k
    leverage = b_next / k_mesh
    
    return {
        "k": k_mesh,
        "b": b_mesh,
        "z": z_mesh,
        "k_next": k_next,
        "b_next": b_next,
        "r_tilde": r_tilde,
        "I_k": I_k,
        "leverage": leverage
    }


def evaluate_risky_value(
    value_net,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_val: float
) -> Dict[str, np.ndarray]:
    """
    Evaluate Risky Debt latent value V_tilde at fixed z.
    
    Args:
        value_net: Trained RiskyValueNetwork
        k_grid: 1D array of capital values
        b_grid: 1D array of debt values
        z_val: Fixed productivity level
    
    Returns:
        Dict with k, b meshgrids, V_tilde, V (limited liability)
    """
    n_k, n_b = len(k_grid), len(b_grid)
    k_mesh, b_mesh = np.meshgrid(k_grid, b_grid, indexing='ij')
    z_mesh = np.full_like(k_mesh, z_val)
    
    k_flat = k_mesh.flatten()
    b_flat = b_mesh.flatten()
    z_flat = z_mesh.flatten()
    
    k_tf = tf.constant(k_flat, dtype=tf.float32)
    b_tf = tf.constant(b_flat, dtype=tf.float32)
    z_tf = tf.constant(z_flat, dtype=tf.float32)
    
    V_tilde_flat = value_net(k_tf, b_tf, z_tf).numpy().flatten()
    V_tilde = V_tilde_flat.reshape(n_k, n_b)
    
    # Limited liability
    V = np.maximum(V_tilde, 0)
    
    return {
        "k": k_mesh,
        "b": b_mesh,
        "z": z_mesh,
        "V_tilde": V_tilde,
        "V": V
    }


def compute_moments(
    data: Dict[str, np.ndarray],
    keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute summary moments for evaluation data.
    
    Args:
        data: Dict of arrays from evaluate_* functions
        keys: Keys to compute moments for (default: all numeric arrays)
    
    Returns:
        DataFrame with mean, median, std, Q10, Q90 for each key
    """
    if keys is None:
        keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v is not None]
    
    rows = []
    for key in keys:
        arr = data.get(key)
        if arr is None:
            continue
        flat = arr.flatten()
        rows.append({
            "variable": key,
            "mean": np.mean(flat),
            "median": np.median(flat),
            "std": np.std(flat),
            "Q10": np.percentile(flat, 10),
            "Q90": np.percentile(flat, 90),
            "min": np.min(flat),
            "max": np.max(flat)
        })
    
    return pd.DataFrame(rows)


def compare_moments(
    histories: List[Dict],
    labels: List[str],
    eval_fn,
    grid_kwargs: Dict
) -> pd.DataFrame:
    """
    Compare moments across multiple trained models.
    
    Args:
        histories: List of training history dicts (containing _policy_net, etc.)
        labels: Labels for each model
        eval_fn: Evaluation function (e.g., evaluate_basic_policy)
        grid_kwargs: Arguments for eval_fn (k_grid, z_grid, etc.)
    
    Returns:
        Combined moments DataFrame with scenario column
    """
    all_moments = []
    
    for hist, label in zip(histories, labels):
        # Extract networks from history
        policy_net = hist.get("_policy_net")
        if policy_net is None:
            continue
        
        # Evaluate
        if "price_net" in str(eval_fn.__code__.co_varnames):
            price_net = hist.get("_price_net")
            data = eval_fn(policy_net, price_net, **grid_kwargs)
        else:
            data = eval_fn(policy_net, **grid_kwargs)
        
        moments = compute_moments(data)
        moments["scenario"] = label
        all_moments.append(moments)
    
    return pd.concat(all_moments, ignore_index=True)


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
# STEADY-STATE FINDER
# =============================================================================

def find_steady_state_k(
    policy_net,
    scenario,
    z_ss: float = 1.0,
    n_grid: int = 500
) -> float:
    """
    Find k_ss such that k'(k_ss, z_ss) ≈ k_ss.
    
    Selection logic:
    1. Find all crossings where g(k) = k' - k changes sign
    2. Among stable crossings (slope < 1), pick largest k
    3. Fallback: largest k among all crossings, or argmin|g|
    
    Args:
        policy_net: Trained BasicPolicyNetwork
        scenario: EconomicScenario with sampling bounds
        z_ss: Steady-state productivity (default 1.0)
        n_grid: Grid density for root finding
    
    Returns:
        k_ss: Steady-state capital
    """
    k_min, k_max = scenario.sampling.k_bounds
    k_grid = np.linspace(k_min, k_max, n_grid)
    
    # Evaluate policy on grid
    k_tf = tf.constant(k_grid.reshape(-1, 1), dtype=tf.float32)
    z_tf = tf.constant(np.full(n_grid, z_ss).reshape(-1, 1), dtype=tf.float32)
    k_next = policy_net(k_tf, z_tf).numpy().flatten()
    
    g = k_next - k_grid  # Zero at fixed point
    
    # Find sign changes (crossings)
    sign_changes = np.where(np.diff(np.sign(g)) != 0)[0]
    
    # Ignore saturated region where k' ≈ k_min
    is_saturated = k_next < k_min * 1.01
    
    # Collect valid crossings: (k_root, slope)
    crossings = []
    for idx in sign_changes:
        if is_saturated[idx] or is_saturated[idx + 1]:
            continue
        # Linear interpolation for root
        k1, k2 = k_grid[idx], k_grid[idx + 1]
        g1, g2 = g[idx], g[idx + 1]
        k_root = k1 - g1 * (k2 - k1) / (g2 - g1)
        slope = (k_next[idx + 1] - k_next[idx]) / (k2 - k1)
        crossings.append((k_root, slope))
    
    if crossings:
        # Prefer stable crossings (slope < 1), pick largest k
        stable = [(k, s) for k, s in crossings if s < 1.0]
        if stable:
            return float(max(stable, key=lambda x: x[0])[0])
        # Fallback: largest k among all
        return float(max(crossings, key=lambda x: x[0])[0])
    
    # No crossings: return k with smallest |g| (excluding saturated)
    valid = ~is_saturated
    if valid.any():
        best_idx = np.where(valid)[0][np.argmin(np.abs(g[valid]))]
        return float(k_grid[best_idx])
    
    return float(k_max)


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
    # Note: compute_cash_flow_basic imported at module top
    
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


# =============================================================================
# EULER RESIDUAL DIAGNOSTICS
# =============================================================================

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
    # Note: euler_chi, euler_m, draw_shocks imported at module top
    
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
