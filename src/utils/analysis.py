"""
src/utils/analysis.py

Utilities for analyzing and visualizing generated datasets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

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
            
        params = metadata.get("params", {})
        print("\nParameters:")
        for k, v in params.items():
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


def evaluate_policy(
    policy_net,
    k_bounds: tuple,
    logz_bounds: tuple,
    n_k: int = 50,
    n_z: int = 15,
    fixed_k_idx: int = 25,
    fixed_z_idx: int = 7
) -> Dict[str, np.ndarray]:
    """
    Evaluate policy network on a 2D grid.
    
    Args:
        policy_net: Keras model or callable
        k_bounds: (min, max) for capital
        logz_bounds: (min, max) for log productivity
        n_k: Grid points for k
        n_z: Grid points for z
        fixed_k_idx: Index to slice for "Fixed k" plot
        fixed_z_idx: Index to slice for "Fixed z" plot
        
    Returns:
        Dictionary with:
        - 'k': Grid mesh (N_k, N_z)
        - 'z': Grid mesh (N_k, N_z)
        - 'k_next': Policy Output (N_k, N_z)
        - 'fixed_k_val': Scalar value of fixed k
        - 'fixed_z_val': Scalar value of fixed z
    """
    import tensorflow as tf
    
    # Create grids
    k_vals = np.linspace(k_bounds[0], k_bounds[1], n_k)
    logz_vals = np.linspace(logz_bounds[0], logz_bounds[1], n_z)
    z_vals = np.exp(logz_vals)
    
    K, Z = np.meshgrid(k_vals, z_vals, indexing='ij')
    
    # Flatten for batch prediction
    flat_k = K.reshape(-1, 1)
    flat_z = Z.reshape(-1, 1)
    
    # Convert to Tensor
    tensor_k = tf.constant(flat_k, dtype=tf.float32)
    tensor_z = tf.constant(flat_z, dtype=tf.float32)
    
    # Predict
    k_next = policy_net(tensor_k, tensor_z)

    k_next_np = k_next.numpy().reshape(n_k, n_z)
    
    return {
        'k': K,
        'z': Z,
        'k_next': k_next_np,
        'fixed_k_val': k_vals[fixed_k_idx],
        'fixed_z_val': z_vals[fixed_z_idx]
    }
