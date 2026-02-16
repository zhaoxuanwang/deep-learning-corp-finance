#!/usr/bin/env python3
"""
CLI for the lightweight Part-1 config-driven pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure repository root is importable when running via:
# python scripts/run_part1_pipeline.py ...
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep matplotlib cache in a writable temp path for CLI runs.
MPL_CACHE_DIR = Path("/tmp/dl_corp_finance_mpl")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from src.pipeline.part1 import (
    run_generate_data,
    run_gc_datasets,
    run_clean_results,
    run_train_ddp,
    run_train_nn,
    run_compare,
    run_all,
)


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline Part-1 pipeline stages (data -> DDP -> NN -> compare)."
    )
    parser.add_argument(
        "--config",
        default="configs/pipelines/part1_baseline.json",
        help="Path to pipeline config (.json, optional .yaml if PyYAML installed).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root directory from config.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("generate-data", help="Generate/reuse shared dataset artifact.")
    p_data.add_argument("--dataset-id", default=None, help="Optional explicit dataset id (e.g., ds_xxxxx).")

    p_ddp = sub.add_parser("train-ddp", help="Run DDP VFI/PFI for basic+risky baseline.")
    p_ddp.add_argument("--run-id", required=True, help="Existing run id (e.g., run-YYYYMMDD-HHMMSS).")
    p_ddp.add_argument("--dataset-id", required=True, help="Dataset id under output_root/datasets.")

    p_nn = sub.add_parser("train-nn", help="Run NN methods for basic+risky baseline.")
    p_nn.add_argument("--run-id", required=True, help="Existing run id (e.g., run-YYYYMMDD-HHMMSS).")
    p_nn.add_argument("--dataset-id", required=True, help="Dataset id under output_root/datasets.")

    p_cmp = sub.add_parser("compare", help="Compare DDP and NN checkpoints and save figures/metrics.")
    p_cmp.add_argument("--run-id", required=True, help="Existing run id (e.g., run-YYYYMMDD-HHMMSS).")
    p_cmp.add_argument("--dataset-id", required=True, help="Dataset id under output_root/datasets.")

    p_all = sub.add_parser("run-all", help="Run full pipeline end-to-end.")
    p_all.add_argument("--run-id", default=None, help="Optional run id. If omitted, auto-generated.")
    p_all.add_argument("--dataset-id", default=None, help="Optional dataset id. If omitted, config-hash based.")

    p_gc = sub.add_parser("gc-datasets", help="Garbage-collect old unreferenced dataset artifacts.")
    p_gc.add_argument("--min-age-days", type=int, default=14, help="Minimum age for deletion candidates.")
    p_gc.add_argument("--dry-run", action="store_true", help="Preview deletions only (default safe mode).")
    p_gc.add_argument("--apply", action="store_true", help="Actually delete candidates.")

    p_clean = sub.add_parser("clean-results", help="Archive legacy/deprecated result folders.")
    p_clean.add_argument("--keep-last-n-runs", type=int, default=3, help="Keep latest N run-* folders.")
    p_clean.add_argument("--dry-run", action="store_true", help="Preview cleanup only (default safe mode).")
    p_clean.add_argument("--apply", action="store_true", help="Actually archive candidates.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    common = {
        "config_path": args.config,
        "output_root_override": args.output_root,
    }

    if args.command == "generate-data":
        result = run_generate_data(
            **common,
            dataset_id=args.dataset_id,
        )
        _print_json(result)
        return

    if args.command == "train-ddp":
        result = run_train_ddp(
            **common,
            run_id=args.run_id,
            dataset_id=args.dataset_id,
        )
        _print_json(result)
        return

    if args.command == "train-nn":
        result = run_train_nn(
            **common,
            run_id=args.run_id,
            dataset_id=args.dataset_id,
        )
        _print_json(result)
        return

    if args.command == "compare":
        result = run_compare(
            **common,
            run_id=args.run_id,
            dataset_id=args.dataset_id,
        )
        _print_json(result)
        return

    if args.command == "run-all":
        result = run_all(
            **common,
            run_id=args.run_id,
            dataset_id=args.dataset_id,
        )
        _print_json(result)
        return

    if args.command == "gc-datasets":
        dry_run = True
        if args.apply:
            dry_run = False
        elif args.dry_run:
            dry_run = True
        result = run_gc_datasets(
            **common,
            min_age_days=args.min_age_days,
            dry_run=dry_run,
        )
        _print_json(result)
        return

    if args.command == "clean-results":
        dry_run = True
        if args.apply:
            dry_run = False
        elif args.dry_run:
            dry_run = True
        result = run_clean_results(
            **common,
            keep_last_n_runs=args.keep_last_n_runs,
            dry_run=dry_run,
        )
        _print_json(result)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
