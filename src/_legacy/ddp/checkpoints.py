"""
Checkpoint helpers for DDP solver outputs.

This mirrors the neural-network checkpoint workflow with lightweight
NumPy/JSON serialization for grid-based DDP solutions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf


def _to_numpy(value: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
    if isinstance(value, tf.Tensor):
        return value.numpy()
    return np.asarray(value)


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float16, np.float32, np.float64, np.integer)):
        return value.item()
    return value


def save_ddp_solution(
    *,
    save_dir: str,
    model_name: str,
    scenario_name: str,
    solver_name: str,
    value: Union[np.ndarray, tf.Tensor],
    policy_k: Union[np.ndarray, tf.Tensor],
    policy_b: Optional[Union[np.ndarray, tf.Tensor]] = None,
    bond_price: Optional[Union[np.ndarray, tf.Tensor]] = None,
    params: Any = None,
    shock_params: Any = None,
    grid_config: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> str:
    """
    Save a single DDP solve result to disk.

    Directory layout:
        {save_dir}/ddp/{model_name}/{scenario_name}/{solver_name}/
            arrays.npz
            metadata.json
    """
    checkpoint_dir = (
        Path(save_dir)
        / "ddp"
        / model_name
        / scenario_name
        / solver_name
    )

    if checkpoint_dir.exists() and not overwrite:
        raise FileExistsError(
            f"DDP checkpoint exists at {checkpoint_dir}. Use overwrite=True to replace."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {
        "value": _to_numpy(value),
        "policy_k": _to_numpy(policy_k),
    }
    if policy_b is not None:
        arrays["policy_b"] = _to_numpy(policy_b)
    if bond_price is not None:
        arrays["bond_price"] = _to_numpy(bond_price)

    arrays_path = checkpoint_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)

    metadata = {
        "model_name": model_name,
        "scenario_name": scenario_name,
        "solver_name": solver_name,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "params": _to_serializable(params),
        "shock_params": _to_serializable(shock_params),
        "grid_config": _to_serializable(grid_config),
        "metrics": {k: _to_serializable(v) for k, v in (metrics or {}).items()},
    }

    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Saved DDP arrays -> {arrays_path}")
        print(f"Saved DDP metadata -> {metadata_path}")

    return str(checkpoint_dir)


def load_ddp_solution(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Load a saved DDP checkpoint from disk.
    """
    ckpt = Path(checkpoint_dir)
    arrays_path = ckpt / "arrays.npz"
    metadata_path = ckpt / "metadata.json"

    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing arrays file: {arrays_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with np.load(arrays_path) as data:
        arrays = {k: data[k] for k in data.files}

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return {
        "arrays": arrays,
        "metadata": metadata,
        "path": str(ckpt),
    }
