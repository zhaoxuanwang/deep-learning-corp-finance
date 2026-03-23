"""Compatibility wrappers for dataset persistence.

The single implementation authority now lives in `src.v2.benchmarks.storage`.
This module remains as a lightweight import-compatible shim.
"""

from src.v2.benchmarks.storage import (
    load_dataset,
    load_dataset_manifest as load_manifest,
    save_dataset,
)

__all__ = ["save_dataset", "load_dataset", "load_manifest"]
