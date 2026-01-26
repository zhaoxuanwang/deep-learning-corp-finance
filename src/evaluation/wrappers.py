"""
src/dnn/evaluation/wrappers.py

High-level wrappers for evaluating networks on grids.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional

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
