"""Run directory + manifest + figure/array/summary helpers (v2 artifacts pattern).

A run context dict carries the directory layout. Arrays are saved as ``.npz``,
metadata as ``manifest.json``, summaries as CSV; a ``latest`` symlink points at the
most recent run (best-effort).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict


def prepare_run(experiment_name: str, results_root: str = "outputs/v3",
                run_tag: str = "run", save: bool = True) -> Dict:
    """Create (or describe) a run directory and return the run context."""
    root = Path(results_root) / experiment_name
    run_dir = root / run_tag
    ctx = {
        "experiment": experiment_name,
        "run_dir": run_dir,
        "figures_dir": run_dir / "figures",
        "arrays_dir": run_dir / "arrays",
        "manifest_path": run_dir / "manifest.json",
        "save": save,
    }
    if save:
        ctx["figures_dir"].mkdir(parents=True, exist_ok=True)
        ctx["arrays_dir"].mkdir(parents=True, exist_ok=True)
        latest = root / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        except OSError:
            pass  # symlinks may be unavailable; not fatal
    return ctx


def save_manifest(ctx: Dict, **sections) -> None:
    if not ctx.get("save"):
        return
    ctx["manifest_path"].write_text(json.dumps(sections, indent=2, default=str))


def save_figure(ctx: Dict, fig, name: str, dpi: int = 150):
    if not ctx.get("save"):
        return None
    path = ctx["figures_dir"] / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_arrays(ctx: Dict, name: str, **arrays) -> None:
    if not ctx.get("save"):
        return
    import numpy as np
    np.savez(ctx["arrays_dir"] / f"{name}.npz", **arrays)


def save_summary(ctx: Dict, rows, name: str = "summary.csv") -> None:
    if not ctx.get("save") or not rows:
        return
    path = ctx["run_dir"] / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
