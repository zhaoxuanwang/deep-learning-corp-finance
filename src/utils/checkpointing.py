"""
src/utils/checkpointing.py

Save and load training results to/from disk.

Handles Keras model serialization and training history/config persistence.
Allows resuming analysis without re-running expensive training.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict, is_dataclass

import tensorflow as tf


def _get_custom_objects() -> Dict[str, Any]:
    """
    Get custom objects dict for Keras model loading.

    Imports all custom network classes so Keras can deserialize them.
    """
    # Import custom network classes (must be imported before loading models)
    from src.networks.network_basic import BasicPolicyNetwork, BasicValueNetwork
    from src.networks.network_risky import RiskyPolicyNetwork, RiskyValueNetwork, RiskyPriceNetwork

    return {
        'BasicPolicyNetwork': BasicPolicyNetwork,
        'BasicValueNetwork': BasicValueNetwork,
        'RiskyPolicyNetwork': RiskyPolicyNetwork,
        'RiskyValueNetwork': RiskyValueNetwork,
        'RiskyPriceNetwork': RiskyPriceNetwork,
    }


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================

def save_training_result(
    result: Dict[str, Any],
    save_dir: str,
    name: str,
    overwrite: bool = False,
    verbose: bool = True
) -> str:
    """
    Save a training result dictionary to disk.

    Saves Keras models as .keras files and metadata (history, configs) as JSON.
    Creates a structured directory with all components.

    Args:
        result: Training result dict from train_basic_*/train_risky_* functions.
                Must contain 'history' and '_policy_net' at minimum.
        save_dir: Base directory for saving (e.g., '../checkpoints').
        name: Name for this result (e.g., 'basic_lr', 'risky_br').
        overwrite: If True, overwrite existing checkpoint. Default False.
        verbose: If True, print save progress.

    Returns:
        Path to the saved checkpoint directory.

    Directory Structure:
        {save_dir}/{name}/
            metadata.json       # history, configs, params (serializable)
            policy_net.keras    # Main policy network
            value_net.keras     # Value network (if present)
            price_net.keras     # Price network (risky model only)
            target_policy_net.keras  # Target networks (if present)
            target_value_net.keras
            target_price_net.keras

    Example:
        >>> result = train_basic_br(dataset, ...)
        >>> save_training_result(result, '../checkpoints', 'basic_br')
        Saved checkpoint to ../checkpoints/basic_br/
    """
    checkpoint_dir = Path(save_dir) / name

    if checkpoint_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at {checkpoint_dir}. "
            f"Use overwrite=True to replace."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Separate models from serializable data
    models = {}
    metadata = {}

    for key, value in result.items():
        if isinstance(value, tf.keras.Model):
            models[key] = value
        elif is_dataclass(value):
            metadata[key] = asdict(value)
        elif key == '_configs':
            # Configs is a dict of dataclasses
            metadata[key] = {
                k: asdict(v) if is_dataclass(v) else v
                for k, v in value.items()
            }
        elif key == '_params':
            # EconomicParams dataclass
            metadata[key] = asdict(value) if is_dataclass(value) else value
        elif _is_json_serializable(value):
            metadata[key] = value
        else:
            if verbose:
                print(f"  Warning: Skipping non-serializable key '{key}'")

    # Save metadata as JSON
    metadata_path = checkpoint_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    if verbose:
        print(f"Saved metadata to {metadata_path}")

    # Save each Keras model
    model_name_map = {
        '_policy_net': 'policy_net.keras',
        '_value_net': 'value_net.keras',
        '_price_net': 'price_net.keras',
        '_target_policy_net': 'target_policy_net.keras',
        '_target_value_net': 'target_value_net.keras',
        '_target_price_net': 'target_price_net.keras',
    }

    for key, model in models.items():
        filename = model_name_map.get(key, f"{key.strip('_')}.keras")
        model_path = checkpoint_dir / filename
        model.save(model_path)
        if verbose:
            print(f"  Saved {key} to {model_path}")

    if verbose:
        print(f"Checkpoint saved to {checkpoint_dir}/")

    return str(checkpoint_dir)


def save_all_results(
    results: Dict[str, Dict[str, Any]],
    save_dir: str,
    overwrite: bool = False,
    verbose: bool = True
) -> str:
    """
    Save multiple training results to a single directory.

    Convenience function for saving all results from a notebook session.

    Args:
        results: Dict mapping names to result dicts.
                 E.g., {'basic_lr': result_lr, 'basic_er': result_er, ...}
        save_dir: Base directory for saving.
        overwrite: If True, overwrite existing checkpoints.
        verbose: If True, print save progress.

    Returns:
        Path to the save directory.

    Example:
        >>> results = {
        ...     'basic_lr': result_lr,
        ...     'basic_er': result_er,
        ...     'basic_br': result_br,
        ...     'risky_br': result_risky_br
        ... }
        >>> save_all_results(results, '../checkpoints/demo_run')
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Saving {name}...")
            print('='*50)
        save_training_result(result, save_dir, name, overwrite=overwrite, verbose=verbose)

    # Save an index file listing all results
    index = {
        'results': list(results.keys()),
        'save_dir': str(save_path)
    }
    index_path = save_path / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    if verbose:
        print(f"\nAll results saved to {save_path}/")
        print(f"Index file: {index_path}")

    return str(save_path)


# =============================================================================
# LOAD FUNCTIONS
# =============================================================================

def load_training_result(
    checkpoint_dir: str,
    load_target_nets: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load a training result from disk.

    Reconstructs the result dict with Keras models and metadata.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        load_target_nets: If True, load target networks (if present).
        verbose: If True, print load progress.

    Returns:
        Reconstructed result dict matching the original training output.

    Example:
        >>> result_br = load_training_result('../checkpoints/basic_br')
        >>> # Use result_br just like the original training output
        >>> grid = evaluate_policy(result_br, k_bounds, logz_bounds)
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    result = {}

    # Load metadata
    metadata_path = checkpoint_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        result.update(metadata)
        if verbose:
            print(f"Loaded metadata from {metadata_path}")

    # Load Keras models
    model_files = {
        'policy_net.keras': '_policy_net',
        'value_net.keras': '_value_net',
        'price_net.keras': '_price_net',
        'target_policy_net.keras': '_target_policy_net',
        'target_value_net.keras': '_target_value_net',
        'target_price_net.keras': '_target_price_net',
    }

    # Get custom objects for Keras deserialization
    custom_objects = _get_custom_objects()

    for filename, key in model_files.items():
        model_path = checkpoint_path / filename

        # Skip target nets if not requested
        if not load_target_nets and 'target' in key:
            continue

        if model_path.exists():
            model = tf.keras.models.load_model(
                model_path, compile=False, custom_objects=custom_objects
            )
            result[key] = model
            if verbose:
                print(f"  Loaded {key} from {model_path}")

    if verbose:
        print(f"Checkpoint loaded from {checkpoint_path}/")

    return result


def load_all_results(
    save_dir: str,
    names: Optional[List[str]] = None,
    load_target_nets: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple training results from a directory.

    Args:
        save_dir: Directory containing saved results.
        names: List of result names to load. If None, loads all from index.
        load_target_nets: If True, load target networks.
        verbose: If True, print load progress.

    Returns:
        Dict mapping names to loaded result dicts.

    Example:
        >>> results = load_all_results('../checkpoints/demo_run')
        >>> result_lr = results['basic_lr']
        >>> result_br = results['basic_br']
    """
    save_path = Path(save_dir)

    # Try to load names from index
    if names is None:
        index_path = save_path / 'index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
            names = index.get('results', [])
        else:
            # Discover subdirectories with metadata.json
            names = [
                d.name for d in save_path.iterdir()
                if d.is_dir() and (d / 'metadata.json').exists()
            ]

    results = {}
    for name in names:
        checkpoint_dir = save_path / name
        if verbose:
            print(f"\n{'='*50}")
            print(f"Loading {name}...")
            print('='*50)
        results[name] = load_training_result(
            str(checkpoint_dir),
            load_target_nets=load_target_nets,
            verbose=verbose
        )

    if verbose:
        print(f"\nLoaded {len(results)} results from {save_path}/")

    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _is_json_serializable(value: Any) -> bool:
    """Check if a value can be JSON serialized."""
    try:
        json.dumps(value, default=_json_default)
        return True
    except (TypeError, ValueError):
        return False


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder for numpy types and other special objects."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, '__dict__'):
        return obj.__dict__

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def list_checkpoints(save_dir: str) -> List[str]:
    """
    List all available checkpoints in a directory.

    Args:
        save_dir: Directory to search.

    Returns:
        List of checkpoint names.
    """
    save_path = Path(save_dir)

    if not save_path.exists():
        return []

    # Check for index file first
    index_path = save_path / 'index.json'
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
        return index.get('results', [])

    # Otherwise discover subdirectories
    return [
        d.name for d in save_path.iterdir()
        if d.is_dir() and (d / 'metadata.json').exists()
    ]


def checkpoint_exists(save_dir: str, name: str) -> bool:
    """Check if a specific checkpoint exists."""
    checkpoint_path = Path(save_dir) / name / 'metadata.json'
    return checkpoint_path.exists()
