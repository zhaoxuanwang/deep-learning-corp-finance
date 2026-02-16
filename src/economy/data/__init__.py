"""
Data generation helper modules.

These utilities keep DataGenerator as a thin orchestrator while preserving
the same public API.
"""

from src.economy.data.cache import (
    default_cache_dir,
    compute_config_hash,
    build_cache_path,
    build_metadata,
    save_dataset_to_disk,
    load_dataset_from_disk,
)
from src.economy.data.sampling import (
    generate_initial_states,
    generate_shocks,
    rollout_forked_path,
    generate_batch,
)
from src.economy.data.flattening import (
    build_flattened_dataset,
)
from src.economy.data.bundle import (
    DatasetBundle,
    canonicalize_bounds,
    infer_bounds_from_flat_data,
    load_dataset_bundle,
)

__all__ = [
    "default_cache_dir",
    "compute_config_hash",
    "build_cache_path",
    "build_metadata",
    "save_dataset_to_disk",
    "load_dataset_from_disk",
    "generate_initial_states",
    "generate_shocks",
    "rollout_forked_path",
    "generate_batch",
    "build_flattened_dataset",
    "DatasetBundle",
    "canonicalize_bounds",
    "infer_bounds_from_flat_data",
    "load_dataset_bundle",
]
