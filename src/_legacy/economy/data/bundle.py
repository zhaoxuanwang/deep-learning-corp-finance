"""
Dataset bundle utilities for metadata-first offline workflows.

This module standardizes three concerns:
1. Loading flat datasets with metadata.
2. Canonicalizing metadata keys (especially bounds naming).
3. Exposing a single bundle object consumed by trainers and DDP solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf


BoundsDict = Dict[str, Tuple[float, float]]


@dataclass(frozen=True)
class DatasetBundle:
    """
    Container for offline training/solver inputs.

    Attributes:
        data: Tensor dictionary loaded from npz.
        metadata: Parsed metadata dictionary.
        bounds: Canonical bounds with keys {'k', 'log_z', 'b'}.
        path: Source file path.
        fingerprint: Stable content hash of dataset arrays.
    """

    data: Dict[str, tf.Tensor]
    metadata: Dict[str, Any]
    bounds: BoundsDict
    path: str
    fingerprint: str


def _decode_metadata(npz_data: np.lib.npyio.NpzFile) -> Optional[Dict[str, Any]]:
    if "_metadata" not in npz_data:
        return None
    raw = npz_data["_metadata"]
    meta_str = raw.item() if hasattr(raw, "item") else str(raw)
    return json.loads(meta_str)


def _array_fingerprint(npz_data: np.lib.npyio.NpzFile) -> str:
    hasher = hashlib.sha256()
    keys = sorted(k for k in npz_data.files if k != "_metadata")
    for key in keys:
        arr = np.asarray(npz_data[key])
        hasher.update(key.encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(arr.tobytes(order="C"))
    return hasher.hexdigest()[:16]


def canonicalize_bounds(metadata: Mapping[str, Any]) -> BoundsDict:
    """
    Extract canonical bounds from metadata.

    Supports both legacy key 'logz' and canonical key 'log_z'.
    """
    if "bounds" not in metadata:
        raise ValueError("Dataset metadata missing required 'bounds' section.")
    bounds_meta = metadata["bounds"]
    if not isinstance(bounds_meta, Mapping):
        raise ValueError("metadata['bounds'] must be a mapping.")

    if "k" not in bounds_meta:
        raise ValueError("metadata['bounds'] missing key 'k'.")
    if "b" not in bounds_meta:
        raise ValueError("metadata['bounds'] missing key 'b'.")

    logz_key = "log_z" if "log_z" in bounds_meta else "logz"
    if logz_key not in bounds_meta:
        raise ValueError("metadata['bounds'] missing key 'log_z' (or legacy 'logz').")

    k_bounds = tuple(float(x) for x in bounds_meta["k"])
    b_bounds = tuple(float(x) for x in bounds_meta["b"])
    logz_bounds = tuple(float(x) for x in bounds_meta[logz_key])

    return {
        "k": (k_bounds[0], k_bounds[1]),
        "log_z": (logz_bounds[0], logz_bounds[1]),
        "b": (b_bounds[0], b_bounds[1]),
    }


def infer_bounds_from_flat_data(dataset: Mapping[str, np.ndarray]) -> BoundsDict:
    """
    Infer bounds from flat data arrays when metadata is unavailable.
    """
    required = {"k", "z"}
    missing = required - set(dataset.keys())
    if missing:
        raise ValueError(f"Cannot infer bounds. Missing flat dataset keys: {sorted(missing)}")

    k_arr = np.asarray(dataset["k"], dtype=np.float64)
    z_arr = np.asarray(dataset["z"], dtype=np.float64)
    b_arr = np.asarray(dataset["b"], dtype=np.float64) if "b" in dataset else np.array([0.0], dtype=np.float64)

    return {
        "k": (float(np.min(k_arr)), float(np.max(k_arr))),
        "log_z": (float(np.min(np.log(np.maximum(z_arr, 1e-12)))), float(np.max(np.log(np.maximum(z_arr, 1e-12))))),
        "b": (float(np.min(b_arr)), float(np.max(b_arr))),
    }


def load_dataset_bundle(
    dataset_path: str,
    *,
    require_metadata: bool = True,
    infer_bounds_if_missing: bool = False,
) -> DatasetBundle:
    """
    Load npz dataset + metadata into a DatasetBundle.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path) as npz_data:
        metadata = _decode_metadata(npz_data)
        fingerprint = _array_fingerprint(npz_data)
        arrays = {
            key: np.asarray(npz_data[key])
            for key in npz_data.files
            if key != "_metadata"
        }

    if metadata is None:
        if require_metadata and not infer_bounds_if_missing:
            raise ValueError(
                "Dataset metadata not found. "
                "Run an ingestion/preprocessing step to create metadata first."
            )
        metadata = {"schema_version": "inferred-1.0", "bounds": {}}

    if infer_bounds_if_missing and ("bounds" not in metadata or not metadata["bounds"]):
        inferred = infer_bounds_from_flat_data(arrays)
        metadata = dict(metadata)
        metadata["bounds"] = {
            "k": list(inferred["k"]),
            "log_z": list(inferred["log_z"]),
            "b": list(inferred["b"]),
        }

    bounds = canonicalize_bounds(metadata)
    tensor_data = {key: tf.constant(val) for key, val in arrays.items()}

    return DatasetBundle(
        data=tensor_data,
        metadata=dict(metadata),
        bounds=bounds,
        path=str(path),
        fingerprint=fingerprint,
    )

