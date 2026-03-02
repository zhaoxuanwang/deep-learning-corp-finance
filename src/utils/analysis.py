"""
src/utils/analysis.py

Utilities for analyzing and visualizing generated datasets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from src.economy.grids import generate_state_grids
from src.economy.parameters import EconomicParams

def summarize_stored_metadata(file_path: str):
    """
    Load and summarize the metadata stored in a .npz dataset file.

    Args:
        file_path: Path to the .npz file
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = np.load(file_path)
        if "_metadata" not in data:
            print("No metadata found in file.")
            return

        meta_item = data["_metadata"]
        # Handle 0-d array wrapping the string
        if hasattr(meta_item, "item"):
            meta_str = meta_item.item()
        else:
            meta_str = str(meta_item)

        metadata = json.loads(meta_str)
        print("\nDataset Metadata Summary:")
        print("-" * 40)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Schema Version: {metadata.get('schema_version', 'N/A')}")
        print(f"Created: {metadata.get('creation_timestamp', 'N/A')}")
        print(f"Master Seed: {metadata.get('master_seed')}")
        
        dims = metadata.get("dims", {})
        print("\nDimensions:")
        for k, v in dims.items():
            print(f"  {k}: {v}")
            
        bounds = metadata.get("bounds", {})
        if bounds:
            print("\nBounds:")
            for k, v in bounds.items():
                print(f"  {k}: {v}")

        shock_params = metadata.get("shock_params", {})
        if shock_params:
            print("\nShock Params:")
            for k, v in shock_params.items():
                print(f"  {k}: {v}")

        print("-" * 40)

    except Exception as e:
        print(f"Error reading metadata: {e}")


def plot_dataset_stats(data: dict, save_path: str):
    """
    Generate basic statistical plots for the dataset.
    """
    # Extract keys
    keys = ['k0', 'z0', 'b0']
    if 'z_path' in data:
        # Plot z_path trajectories for first few samples
        z_path = data['z_path']
        if hasattr(z_path, 'numpy'):
            z_path = z_path.numpy()
        
        plt.figure(figsize=(10, 6))
        for i in range(min(5, z_path.shape[0])):
            plt.plot(z_path[i], label=f'Path {i}')
        plt.title("Sample Productivity Paths (z_path)")
        plt.xlabel("Time")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path.replace(".png", "_zpath.png"))
        plt.close()
        print(f"Saved z_path plot to {save_path.replace('.png', '_zpath.png')}")

    # Histograms of initial states
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, key in enumerate(keys):
        if key in data:
            val = data[key]
            if hasattr(val, 'numpy'):
                val = val.numpy()
            
            # Special handling for z0 to show uniformity in log-space
            if key == 'z0':
                val = np.log(val)
                plot_title = "Distribution of log(z0)"
                plot_xlabel = "log(z0)"
            else:
                plot_title = f"Distribution of {key}"
                plot_xlabel = key
            
            axes[i].hist(val, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[i].set_title(plot_title)
            axes[i].set_xlabel(plot_xlabel)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved distribution plot to {save_path}")


def summarize_batch(batch: dict, batch_idx: int = 1):
    """
    Print a summary of a batch.

    Args:
        batch: Dictionary with batch data
        batch_idx: Batch index for display
    """
    print(f"\nBatch {batch_idx} Summary:")
    print("-" * 60)

    for key in ['k0', 'z0', 'b0', 'eps1', 'eps2', 'z_path']:
        if key in batch:
            tensor = batch[key]
            values = tensor.numpy()

            print(f"{key:10} shape={str(tensor.shape):15} "
                  f"mean={values.mean():.4f} std={values.std():.4f} "
                  f"min={values.min():.4f} max={values.max():.4f}")


# =============================================================================
# ANALYTICAL SOLUTIONS
# =============================================================================

def compute_frictionless_policy(z, params, shock_params):
    """
    Compute analytical optimal policy for basic model without adjustment costs.

    The frictionless solution (no adjustment costs) has closed-form:

        k'(z) = [γ * E[z' | z] / (r + δ)]^(1/(1-γ))

    where E[z' | z] is the expected next-period productivity given current z.

    For the AR(1) process: log(z') = (1-ρ)*μ + ρ*log(z) + σ*ε, ε ~ N(0,1)

        E[z' | z] = exp((1-ρ)*μ) * z^ρ * exp(0.5*σ²)

    This gives the final formula:

        k'(z) = [γ * exp((1-ρ)*μ + 0.5*σ²) * z^ρ / (r + δ)]^(1/(1-γ))

    Note: The optimal policy depends only on z, not on current capital k.

    Args:
        z: Productivity level(s). Can be scalar or array.
        params: EconomicParams with theta (γ), r_rate, delta.
        shock_params: ShockParams with rho, sigma, mu.

    Returns:
        Optimal next-period capital k'(z) in levels. Same shape as input z.
    """
    gamma = params.theta
    rho = shock_params.rho
    sigma = shock_params.sigma
    mu = shock_params.mu

    r = params.r_rate
    delta = params.delta

    # E[z' | z] = exp((1-rho)*mu) * z^rho * exp(0.5*sigma^2)
    # k' = [gamma * E[z' | z] / (r + delta)]^(1/(1-gamma))
    #
    # Combining the exp() terms for clarity:
    # exp_correction = exp((1-rho)*mu + 0.5*sigma^2)
    exp_correction = np.exp((1 - rho) * mu + 0.5 * sigma ** 2)

    numerator = gamma * (z ** rho) * exp_correction
    denominator = r + delta
    k_prime = (numerator / denominator) ** (1 / (1 - gamma))

    return k_prime


# =============================================================================
# POLICY EVALUATION
# =============================================================================

def _resolve_delta_from_result(result_dict: Dict[str, Any]) -> float:
    """
    Resolve depreciation rate for multiplicative capital grid construction.
    """
    default_delta = float(EconomicParams.delta)
    params_obj = result_dict.get("params") or result_dict.get("_params")
    if params_obj is None:
        return default_delta
    if isinstance(params_obj, dict):
        return float(params_obj.get("delta", default_delta))
    return float(getattr(params_obj, "delta", default_delta))


def evaluate_policy(
    result: Dict[str, Any],
    k_bounds: tuple,
    logz_bounds: tuple,
    b_bounds: Optional[tuple] = None,
    n_z: int = 50,
    n_b: int = 100,
    # Fixed indices (alternative to fixed values)
    fixed_k_idx: Optional[int] = None,
    fixed_z_idx: Optional[int] = None,
    fixed_b_idx: Optional[int] = None,
    # Fixed values (alternative to fixed indices)
    fixed_k_val: Optional[float] = None,
    fixed_logz_val: Optional[float] = None,
    fixed_b_val: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained policy network on mesh grids for visualization and analysis.

    Discretizes the learned optimal policy onto regular grids. Supports both
    the basic model (k, z) and risky debt model (k, b, z).

    The capital grid size is determined automatically by the multiplicative
    grid scheme (depends on k_bounds and the depreciation rate delta).

    Args:
        result: Training result dictionary from train_basic_* or train_risky_*.
                Must contain '_policy_net' key with the trained policy network.
        k_bounds: (min, max) for capital grid.
        logz_bounds: (min, max) for log-productivity grid.
        b_bounds: (min, max) for debt grid. Required for risky debt model,
                  ignored for basic model.
        n_z: Number of grid points for productivity (default: 50).
        n_b: Number of grid points for debt (default: 100). Ignored for basic model.
        fixed_k_idx: Index for fixed capital slice.
        fixed_z_idx: Index for fixed productivity slice (0 to n_z-1).
        fixed_b_idx: Index for fixed debt slice (0 to n_b-1). Ignored for basic model.
        fixed_k_val: Fixed capital value (overrides fixed_k_idx if provided).
        fixed_logz_val: Fixed log-productivity value (overrides fixed_z_idx if provided).
        fixed_b_val: Fixed debt value (overrides fixed_b_idx if provided).

    Returns:
        Dictionary with:
            Basic Model (2D grids):
                - 'k': Capital mesh (n_k, n_z)
                - 'z': Productivity mesh (n_k, n_z)
                - 'log_z': Log-productivity mesh (n_k, n_z)
                - 'logz': Legacy alias of 'log_z'
                - 'k_next': Policy output for k' (n_k, n_z)
                - 'b_next': None (not applicable)
                - 'fixed_k_val': Scalar value of fixed k
                - 'fixed_z_val': Scalar value of fixed z
                - 'fixed_logz_val': Scalar value of fixed log(z)
                - 'fixed_b_val': None (not applicable)
                - 'model_type': 'basic'

            Risky Debt Model (3D grids):
                - 'k': Capital mesh (n_k, n_z, n_b)
                - 'z': Productivity mesh (n_k, n_z, n_b)
                - 'log_z': Log-productivity mesh (n_k, n_z, n_b)
                - 'logz': Legacy alias of 'log_z'
                - 'b': Debt mesh (n_k, n_z, n_b)
                - 'k_next': Policy output for k' (n_k, n_z, n_b)
                - 'b_next': Policy output for b' (n_k, n_z, n_b)
                - 'fixed_k_val': Scalar value of fixed k
                - 'fixed_z_val': Scalar value of fixed z
                - 'fixed_logz_val': Scalar value of fixed log(z)
                - 'fixed_b_val': Scalar value of fixed b
                - 'model_type': 'risky'

            Common:
                - 'k_vals': 1D array of capital grid values
                - 'z_vals': 1D array of productivity grid values
                - 'log_z_vals': 1D array of log-productivity grid values
                - 'logz_vals': Legacy alias of 'log_z_vals'
                - 'b_vals': 1D array of debt grid values (or None for basic)
                - 'fixed_k_idx': Index used for fixed k
                - 'fixed_z_idx': Index used for fixed z
                - 'fixed_b_idx': Index used for fixed b (or None for basic)

    Example:
        >>> # Basic model
        >>> result_br = train_basic_br(dataset, ...)
        >>> grid = evaluate_policy(result_br, k_bounds=(0.5, 3.0), logz_bounds=(-0.3, 0.3))
        >>> plt.contourf(grid['k'], grid['z'], grid['k_next'])
        >>>
        >>> # Risky debt model
        >>> result_risky = train_risky_br(dataset, ...)
        >>> grid = evaluate_policy(
        ...     result_risky,
        ...     k_bounds=(0.5, 3.0),
        ...     logz_bounds=(-0.3, 0.3),
        ...     b_bounds=(0.0, 2.0),
        ...     fixed_b_val=1.0  # Fix debt at 1.0 for 2D slice
        ... )
    """
    import tensorflow as tf
    from src.trainers.io_transforms import build_inference_helper_from_result

    # Extract policy network from result
    if '_policy_net' not in result:
        raise ValueError(
            "Result dictionary must contain '_policy_net' key. "
            "Expected output from train_basic_* or train_risky_* functions."
        )
    infer = build_inference_helper_from_result(result)
    is_risky_model = infer.model == "risky"

    if is_risky_model and b_bounds is None:
        raise ValueError("b_bounds is required for risky debt model evaluation.")

    # Create model-consistent state grids:
    # - k: multiplicative
    # - log_z/z: log-linear
    # - b: linear
    grid_bounds = {
        "k": tuple(k_bounds),
        "log_z": tuple(logz_bounds),
        "b": tuple(b_bounds) if b_bounds is not None else (0.0, 0.0),
    }
    delta = _resolve_delta_from_result(result)
    k_vals, z_vals, b_grid = generate_state_grids(
        grid_bounds,
        delta=delta,
        n_z=n_z,
        n_b=n_b,
    )
    logz_vals = np.log(np.maximum(z_vals, 1e-12))
    b_vals = b_grid if is_risky_model else None

    # === Resolve fixed indices ===
    # For each dimension, use fixed_*_val if provided, else use fixed_*_idx, else use midpoint

    def resolve_fixed_idx(vals: np.ndarray, fixed_idx: Optional[int], fixed_val: Optional[float], name: str) -> int:
        """Resolve fixed index from value or index, defaulting to midpoint."""
        if fixed_val is not None:
            # Find nearest grid point to fixed_val
            idx = int(np.argmin(np.abs(vals - fixed_val)))
            return idx
        elif fixed_idx is not None:
            if not (0 <= fixed_idx < len(vals)):
                raise ValueError(f"fixed_{name}_idx={fixed_idx} out of range [0, {len(vals)-1}]")
            return fixed_idx
        else:
            # Default to midpoint
            return len(vals) // 2

    fixed_k_idx_resolved = resolve_fixed_idx(k_vals, fixed_k_idx, fixed_k_val, 'k')
    fixed_z_idx_resolved = resolve_fixed_idx(z_vals, fixed_z_idx, np.exp(fixed_logz_val) if fixed_logz_val is not None else None, 'z')

    if is_risky_model:
        fixed_b_idx_resolved = resolve_fixed_idx(b_vals, fixed_b_idx, fixed_b_val, 'b')
    else:
        fixed_b_idx_resolved = None

    # === Evaluate policy on grid ===
    if is_risky_model:
        # 3D grid for risky debt model: (n_k_actual, n_z, n_b)
        K, Z, B = np.meshgrid(k_vals, z_vals, b_vals, indexing='ij')
        LOGZ = np.log(Z)

        # Flatten for batch prediction
        flat_k = K.reshape(-1, 1)
        flat_z = Z.reshape(-1, 1)
        flat_b = B.reshape(-1, 1)

        # Convert to Tensor
        tensor_k = tf.constant(flat_k, dtype=tf.float32)
        tensor_b = tf.constant(flat_b, dtype=tf.float32)
        tensor_z = tf.constant(flat_z, dtype=tf.float32)

        # Predict in level space via inference helper.
        k_next_tensor, b_next_tensor = infer.policy(tensor_k, tensor_b, tensor_z)

        k_next_np = k_next_tensor.numpy().reshape(len(k_vals), len(z_vals), len(b_vals))
        b_next_np = b_next_tensor.numpy().reshape(len(k_vals), len(z_vals), len(b_vals))

        return {
            # 3D grids
            'k': K,
            'z': Z,
            'log_z': LOGZ,
            'logz': LOGZ,
            'b': B,
            'k_next': k_next_np,
            'b_next': b_next_np,
            # 1D grid arrays (for convenience)
            'k_vals': k_vals,
            'z_vals': z_vals,
            'log_z_vals': logz_vals,
            'logz_vals': logz_vals,
            'b_vals': b_vals,
            # Fixed values
            'fixed_k_val': k_vals[fixed_k_idx_resolved],
            'fixed_z_val': z_vals[fixed_z_idx_resolved],
            'fixed_logz_val': logz_vals[fixed_z_idx_resolved],
            'fixed_b_val': b_vals[fixed_b_idx_resolved],
            # Fixed indices
            'fixed_k_idx': fixed_k_idx_resolved,
            'fixed_z_idx': fixed_z_idx_resolved,
            'fixed_b_idx': fixed_b_idx_resolved,
            # Metadata
            'model_type': 'risky',
            'grid_shape': (len(k_vals), len(z_vals), len(b_vals)),
        }

    else:
        # 2D grid for basic model: (n_k_actual, n_z)
        K, Z = np.meshgrid(k_vals, z_vals, indexing='ij')
        LOGZ = np.log(Z)

        # Flatten for batch prediction
        flat_k = K.reshape(-1, 1)
        flat_z = Z.reshape(-1, 1)

        # Convert to Tensor
        tensor_k = tf.constant(flat_k, dtype=tf.float32)
        tensor_z = tf.constant(flat_z, dtype=tf.float32)

        # Predict in level space via inference helper.
        k_next_tensor = infer.policy(tensor_k, tensor_z)
        k_next_np = k_next_tensor.numpy().reshape(len(k_vals), len(z_vals))

        return {
            # 2D grids
            'k': K,
            'z': Z,
            'log_z': LOGZ,
            'logz': LOGZ,
            'b': None,
            'k_next': k_next_np,
            'b_next': None,
            # 1D grid arrays (for convenience)
            'k_vals': k_vals,
            'z_vals': z_vals,
            'log_z_vals': logz_vals,
            'logz_vals': logz_vals,
            'b_vals': None,
            # Fixed values
            'fixed_k_val': k_vals[fixed_k_idx_resolved],
            'fixed_z_val': z_vals[fixed_z_idx_resolved],
            'fixed_logz_val': logz_vals[fixed_z_idx_resolved],
            'fixed_b_val': None,
            # Fixed indices
            'fixed_k_idx': fixed_k_idx_resolved,
            'fixed_z_idx': fixed_z_idx_resolved,
            'fixed_b_idx': None,
            # Metadata
            'model_type': 'basic',
            'grid_shape': (len(k_vals), len(z_vals)),
        }


def get_steady_state_policy(
    result: Dict[str, Any],
    k_bounds: tuple,
    logz_bounds: tuple,
    b_bounds: Optional[tuple] = None,
    n_grid: int = 500,
    outlier_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Find steady state indices and values where policy crosses 45-degree line.

    Locates the steady state capital k* (where k' = k) and debt b* (where b' = b)
    by finding zero-crossings of (policy_output - state). Uses a sequential approach
    for risky models: first finds k*, then uses k* to find b*.

    Steady State Selection Rules:
        1. Find all crossings where (policy - state) changes sign
        2. Exclude crossings in the bottom `outlier_pct` of the grid (near-zero outliers)
        3. Select the smallest valid crossing (capital accumulation from below)

    Args:
        result: Training result dictionary from train_basic_* or train_risky_*.
                Must contain '_policy_net' key.
        k_bounds: (min, max) for capital grid.
        logz_bounds: (min, max) for log-productivity grid.
        b_bounds: (min, max) for debt grid. Required for risky model.
        n_grid: Number of grid points for 1D search (default: 200).
        outlier_pct: Fraction of grid range to exclude from bottom (default: 0.05).

    Returns:
        Dictionary with:
            - 'k_star_idx': Index of steady state k in a grid of size n_grid
            - 'k_star_val': Value of steady state k
            - 'z_mid_idx': Index of midpoint z (used for k* search)
            - 'z_mid_val': Value of midpoint z
            - 'b_star_idx': Index of steady state b (None for basic model)
            - 'b_star_val': Value of steady state b (None for basic model)
            - 'model_type': 'basic' or 'risky'
            - 'k_bounds': Tuple of k bounds used
            - 'logz_bounds': Tuple of logz bounds used
            - 'b_bounds': Tuple of b bounds used (None for basic model)
            - 'n_grid': Grid resolution used

    Example:
        >>> result = train_basic_br(dataset, ...)
        >>> ss = get_steady_state_policy(result, k_bounds=(0.5, 3), logz_bounds=(-0.3, 0.3))
        >>> print(f"Steady state k* = {ss['k_star_val']:.3f}")
        >>>
        >>> # Use with evaluate_policy for visualization at steady state
        >>> grid = evaluate_policy(
        ...     result, k_bounds=(0.5, 3), logz_bounds=(-0.3, 0.3),
        ...     fixed_k_val=ss['k_star_val']
        ... )
    """
    import tensorflow as tf
    from src.trainers.io_transforms import build_inference_helper_from_result

    # Extract policy network
    if '_policy_net' not in result:
        raise ValueError(
            "Result dictionary must contain '_policy_net' key. "
            "Expected output from train_basic_* or train_risky_* functions."
        )
    infer = build_inference_helper_from_result(result)
    # Keep b_bounds override behavior for compatibility.
    is_risky_model = infer.model == "risky" or b_bounds is not None

    if is_risky_model and b_bounds is None:
        raise ValueError("b_bounds is required for risky debt model.")

    # Create model-consistent state grids from shared grid utilities.
    grid_bounds = {
        "k": tuple(k_bounds),
        "log_z": tuple(logz_bounds),
        "b": tuple(b_bounds) if b_bounds is not None else (0.0, 0.0),
    }
    delta = _resolve_delta_from_result(result)
    k_vals, z_vals, b_grid = generate_state_grids(
        grid_bounds,
        delta=delta,
        n_z=n_grid,
        n_b=n_grid,
    )

    # Midpoint indices
    z_mid_idx = len(z_vals) // 2
    z_mid_val = z_vals[z_mid_idx]

    # Outlier threshold: exclude bottom outlier_pct of range
    k_outlier_threshold = float(k_vals[0]) + outlier_pct * float(k_vals[-1] - k_vals[0])

    def find_crossing(state_vals: np.ndarray, policy_vals: np.ndarray,
                      outlier_threshold: float) -> tuple:
        """
        Find the smallest valid crossing where policy = state (45-degree line).

        Returns (index, value) of the crossing point.
        """
        # Compute difference: policy - state
        diff = policy_vals - state_vals

        # Find sign changes (zero crossings)
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

        if len(sign_changes) == 0:
            # No crossing found - fallback to point closest to 45-degree line
            closest_idx = int(np.argmin(np.abs(diff)))
            return closest_idx, state_vals[closest_idx]

        # Filter out crossings in outlier region (bottom outlier_pct)
        valid_crossings = []
        for idx in sign_changes:
            crossing_val = state_vals[idx]
            if crossing_val >= outlier_threshold:
                valid_crossings.append(idx)

        if len(valid_crossings) == 0:
            # All crossings are outliers - use smallest non-outlier crossing
            # or fallback to the first crossing above threshold
            for idx in sign_changes:
                if state_vals[idx] >= outlier_threshold * 0.5:  # Relax threshold
                    valid_crossings.append(idx)

            if len(valid_crossings) == 0:
                # Still no valid crossings - use first crossing
                valid_crossings = [sign_changes[0]]

        # Select the smallest valid crossing (accumulating from below)
        selected_idx = min(valid_crossings)

        # Refine crossing point using linear interpolation
        if selected_idx < len(state_vals) - 1:
            # Linear interpolation between idx and idx+1
            d0, d1 = diff[selected_idx], diff[selected_idx + 1]
            if d0 != d1:
                alpha = -d0 / (d1 - d0)
                crossing_val = state_vals[selected_idx] + alpha * (
                    state_vals[selected_idx + 1] - state_vals[selected_idx]
                )
            else:
                crossing_val = state_vals[selected_idx]
        else:
            crossing_val = state_vals[selected_idx]

        return selected_idx, float(crossing_val)

    if is_risky_model:
        b_vals = b_grid
        if b_vals is None:
            raise ValueError("b_bounds must produce a valid debt grid for risky model.")
        b_mid_idx = len(b_vals) // 2
        b_mid_val = b_vals[b_mid_idx]
        b_outlier_threshold = float(b_vals[0]) + outlier_pct * float(b_vals[-1] - b_vals[0])

        # === Step 1: Find k* with b and z fixed at midpoints ===
        # Evaluate policy(k, b_mid, z_mid) for all k
        tensor_k = tf.constant(k_vals.reshape(-1, 1), dtype=tf.float32)
        tensor_b = tf.constant(np.full((len(k_vals), 1), b_mid_val), dtype=tf.float32)
        tensor_z = tf.constant(np.full((len(k_vals), 1), z_mid_val), dtype=tf.float32)

        k_next_tensor, _ = infer.policy(tensor_k, tensor_b, tensor_z)
        k_next_vals = k_next_tensor.numpy().flatten()

        k_star_idx, k_star_val = find_crossing(k_vals, k_next_vals, k_outlier_threshold)

        # === Step 2: Find b* with k fixed at k* and z at midpoint ===
        # Evaluate policy(k*, b, z_mid) for all b
        tensor_k_star = tf.constant(np.full((len(b_vals), 1), k_star_val), dtype=tf.float32)
        tensor_b_all = tf.constant(b_vals.reshape(-1, 1), dtype=tf.float32)
        tensor_z_mid = tf.constant(np.full((len(b_vals), 1), z_mid_val), dtype=tf.float32)

        _, b_next_tensor = infer.policy(tensor_k_star, tensor_b_all, tensor_z_mid)
        b_next_vals = b_next_tensor.numpy().flatten()

        b_star_idx, b_star_val = find_crossing(b_vals, b_next_vals, b_outlier_threshold)

        return {
            'k_star_idx': k_star_idx,
            'k_star_val': k_star_val,
            'z_mid_idx': z_mid_idx,
            'z_mid_val': z_mid_val,
            'b_star_idx': b_star_idx,
            'b_star_val': b_star_val,
            'model_type': 'risky',
            'k_bounds': k_bounds,
            'log_z_bounds': logz_bounds,
            'logz_bounds': logz_bounds,
            'b_bounds': b_bounds,
            'n_grid': len(k_vals),
        }

    else:
        # Basic model: find k* with z fixed at midpoint
        tensor_k = tf.constant(k_vals.reshape(-1, 1), dtype=tf.float32)
        tensor_z = tf.constant(np.full((len(k_vals), 1), z_mid_val), dtype=tf.float32)

        k_next_tensor = infer.policy(tensor_k, tensor_z)
        k_next_vals = k_next_tensor.numpy().flatten()

        k_star_idx, k_star_val = find_crossing(k_vals, k_next_vals, k_outlier_threshold)

        return {
            'k_star_idx': k_star_idx,
            'k_star_val': k_star_val,
            'z_mid_idx': z_mid_idx,
            'z_mid_val': z_mid_val,
            'b_star_idx': None,
            'b_star_val': None,
            'model_type': 'basic',
            'k_bounds': k_bounds,
            'log_z_bounds': logz_bounds,
            'logz_bounds': logz_bounds,
            'b_bounds': None,
            'n_grid': len(k_vals),
        }


# =============================================================================
# TRAINING SUMMARY
# =============================================================================

def print_training_summary(
    results: Dict[str, Any],
    model_type: str = "basic"
) -> None:
    """
    Print a clean summary table of training results.

    For basic model, expects results structured as:
        {'scenario_name': {'lr': {...}, 'er': {...}, 'br': {...}}, ...}

    For risky model, expects results structured as:
        {'scenario_name': {'history': {...}, ...}, ...}

    Args:
        results: Dict of training results
        model_type: "basic" or "risky"
    """
    print("\n" + "=" * 70)
    print(f"TRAINING SUMMARY ({model_type.upper()} MODEL)")
    print("=" * 70)

    if model_type == "basic":
        # Header
        print(f"{'Scenario':<20} {'LR Loss':>12} {'ER Loss':>12} {'BR Actor':>12} {'BR RelMSE':>12}")
        print("-" * 70)

        for scenario_name, scenario_results in results.items():
            lr_loss = scenario_results['lr']['history']['loss_LR'][-1]
            er_loss = scenario_results['er']['history']['loss_ER'][-1]
            br_actor = scenario_results['br']['history']['loss_actor'][-1]
            br_rel_mse = scenario_results['br']['history'].get('rel_mse', [0])[-1]

            print(f"{scenario_name:<20} {lr_loss:>12.4f} {er_loss:>12.6f} {br_actor:>12.4f} {br_rel_mse:>12.6f}")

    elif model_type == "risky":
        # Header
        print(f"{'Scenario':<20} {'Actor':>12} {'Price':>12} {'RelMSE':>12} {'ValueScale':>12}")
        print("-" * 70)

        for scenario_name, result in results.items():
            history = result['history']
            actor = history['loss_actor'][-1]
            price = history['loss_price'][-1]
            rel_mse = history.get('rel_mse', [0])[-1]
            scale = history.get('mean_value_scale', [0])[-1]

            print(f"{scenario_name:<20} {actor:>12.4f} {price:>12.6f} {rel_mse:>12.6f} {scale:>12.2f}")

    print("=" * 70 + "\n")
        
