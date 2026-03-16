"""
Lightweight, config-driven training pipelines.
"""

from src.pipeline.part1 import (
    run_generate_data,
    run_gc_datasets,
    run_clean_results,
    run_train_ddp,
    run_train_nn,
    run_compare,
    run_all,
)

__all__ = [
    "run_generate_data",
    "run_gc_datasets",
    "run_clean_results",
    "run_train_ddp",
    "run_train_nn",
    "run_compare",
    "run_all",
]
