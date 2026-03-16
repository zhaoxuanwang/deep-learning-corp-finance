"""
Disk cache helpers for reproducible datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.economy.parameters import ShockParams


def default_cache_dir(reference_file: str) -> str:
    """
    Resolve default cache directory as PROJECT_ROOT/data.
    """
    project_root = Path(reference_file).resolve().parent.parent.parent
    return str(project_root / "data")


def compute_config_hash(
    master_seed: Tuple[int, int],
    sim_batch_size: int,
    horizon: int,
    n_sim_batches: int,
    n_val: int,
    n_test: int,
    k_bounds: Tuple[float, float],
    logz_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    shock_params: ShockParams,
) -> str:
    """
    Compute deterministic hash for cache file naming.
    """
    config_tuple = (
        master_seed,
        sim_batch_size,
        horizon,
        n_sim_batches,
        n_val,
        n_test,
        k_bounds,
        logz_bounds,
        b_bounds,
        shock_params.rho,
        shock_params.sigma,
        shock_params.mu,
    )
    config_str = str(config_tuple)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:12]


def build_cache_path(cache_dir: str, split: str, config_hash: str) -> str:
    """
    Build .npz path for a split/config.
    """
    filename = f"{split}_{config_hash}.npz"
    return os.path.join(cache_dir, filename)


def build_metadata(
    master_seed: Tuple[int, int],
    sim_batch_size: int,
    horizon: int,
    n_sim_batches: int,
    n_val: int,
    n_test: int,
    k_bounds: Tuple[float, float],
    logz_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    shock_params: ShockParams,
) -> Dict[str, Any]:
    """
    Build cache metadata payload.
    """
    return {
        "schema_version": "1.0",
        "master_seed": [int(x) for x in master_seed],
        "dims": {
            "n": sim_batch_size,
            "T": horizon,
            "J": n_sim_batches,
            "N_val": n_val,
            "N_test": n_test,
        },
        "bounds": {
            "k": [float(x) for x in k_bounds],
            # Canonical key for log-productivity bounds.
            # Legacy key "logz" is still accepted when loading older cache files.
            "log_z": [float(x) for x in logz_bounds],
            "b": [float(x) for x in b_bounds],
        },
        "shock_params": {
            "rho": float(shock_params.rho),
            "sigma": float(shock_params.sigma),
            "mu": float(shock_params.mu),
        },
        "creation_timestamp": str(datetime.now()),
    }


def _canonicalize_bounds_section(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize bounds key naming for compatibility across schema revisions.
    """
    out = dict(meta)
    bounds = dict(out.get("bounds", {}))
    if "log_z" not in bounds and "logz" in bounds:
        bounds["log_z"] = bounds["logz"]
    if "logz" in bounds:
        # Keep only canonical key in normalized representation.
        bounds.pop("logz")
    out["bounds"] = bounds
    return out


def save_dataset_to_disk(
    dataset: Dict[str, tf.Tensor],
    path: str,
    *,
    save_to_disk: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save tensors as compressed NPZ with optional metadata.
    """
    if not save_to_disk:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    numpy_data = {key: tensor.numpy() for key, tensor in dataset.items()}
    if metadata is not None:
        numpy_data["_metadata"] = json.dumps(metadata)
    np.savez_compressed(path, **numpy_data)


def _metadata_section_mismatch(
    stored_meta: Dict[str, Any],
    current_meta: Dict[str, Any],
    section: str,
) -> bool:
    return stored_meta.get(section) != current_meta.get(section)


def load_dataset_from_disk(
    path: str,
    *,
    current_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, tf.Tensor]:
    """
    Load tensors from compressed NPZ and optionally validate metadata.
    """
    result: Dict[str, tf.Tensor] = {}
    with np.load(path) as data:
        if current_metadata is not None and "_metadata" in data:
            stored_meta = _canonicalize_bounds_section(json.loads(str(data["_metadata"])))
            current_metadata_norm = _canonicalize_bounds_section(current_metadata)
            if (
                _metadata_section_mismatch(stored_meta, current_metadata_norm, "master_seed")
                or _metadata_section_mismatch(stored_meta, current_metadata_norm, "dims")
                or _metadata_section_mismatch(stored_meta, current_metadata_norm, "bounds")
                or _metadata_section_mismatch(stored_meta, current_metadata_norm, "shock_params")
            ):
                raise ValueError(
                    "Cached dataset metadata mismatch!\n"
                    f"Stored: {json.dumps(stored_meta, indent=2)}\n"
                    f"Current: {json.dumps(current_metadata_norm, indent=2)}\n"
                    f"File: {path}"
                )

        for key in data.keys():
            if key != "_metadata":
                result[key] = tf.constant(data[key])

    return result
