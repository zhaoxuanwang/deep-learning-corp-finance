"""
src/dnn/evaluation/common.py

Shared utilities for evaluation:
- Grid generation
- Moment computation across models
- Steady state finding
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union

from src.economy.grids import generate_state_grids

def get_eval_grids(
    source: Union[Dict, "EconomicScenario"],
    n_k: int = 50,
    n_z: int = 10,
    n_b: int = 20
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate evaluation grids from scenario/history bounds.
    
    Wrapper around src.economy.grids.generate_state_grids.
    
    Args:
        source: Either a training history dict (with "_scenario" key)
                or an EconomicScenario directly
        n_k: Target number of points for capital grid
        n_z: Number of points for productivity grid
        n_b: Number of points for debt grid (risky model)
    
    Returns:
        (k_grid, z_grid, b_grid)
    """
    # Extract bounds and delta from source
    if isinstance(source, dict):
        # Training history dict
        scenario = source.get("_scenario")
        if scenario is None:
            raise ValueError("History dict must contain '_scenario' key")
        bounds = scenario.sampling
        params = scenario.params
    else:
        # Assume it's an EconomicScenario
        # Lazy import to avoid circular dependency
        from src.dnn.experiments import ensure_scenario_bounds
        ensure_scenario_bounds(source)
        bounds = source.sampling
        params = source.params
    
    return generate_state_grids(params, bounds, n_k, n_z, n_b)


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
