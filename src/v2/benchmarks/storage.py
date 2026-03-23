"""Persistence helpers for benchmark notebook runs and saved training results."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if tf.is_tensor(obj):
        return obj.numpy().tolist()
    return str(obj)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_run_dir(run_dir: Optional[str]) -> Optional[Path]:
    if run_dir is None:
        return None
    return Path(run_dir).expanduser().resolve()


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _subdirs(run_dir: Path) -> Dict[str, Path]:
    """Sub-directories that live inside a run folder.

    Datasets are NOT persisted (they are fully reproducible from the saved
    seed/config metadata in manifest.json).
    """
    return {
        "run_dir":        run_dir,
        "figures_dir":    run_dir / "figures",
        "plot_inputs_dir": run_dir / "plot_inputs",
        "pfi_dir":        run_dir / "pfi",
        "methods_dir":    run_dir / "methods",
    }


def _append_run_index(root: Path, run_id: str, run_tag: Optional[str]) -> None:
    """Append a one-line record to the experiment-level runs.json index."""
    index_path = root / "runs.json"
    index: list = []
    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            index = []

    # Avoid duplicate entries (e.g. if run_tag is reused)
    if not any(r.get("run_id") == run_id for r in index):
        index.append({
            "run_id":     run_id,
            "tag":        run_tag,
            "created_at": datetime.now().isoformat(),
        })

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def prepare_benchmark_run(experiment_name: str,
                           save_run: bool = True,
                           load_run_dir: Optional[str] = None,
                           results_root: str = "outputs/notebooks",
                           run_tag: Optional[str] = None) -> Dict[str, Any]:
    """Create or attach to a benchmark run directory.

    Directory layout::

        {results_root}/{experiment_name}/{run_id}/
            manifest.json
            runs.json          ← experiment-level index (one record per run)
            figures/
            plot_inputs/
            pfi/
            methods/{LR,ER,BRM,SHAC}/
                config.json
                result.json
                history.npz
                checkpoint_metrics.npz
                selected.json
                snapshots/

    Run ID policy:
        - If ``run_tag`` is provided it becomes the folder name directly,
          giving a stable, human-readable identifier (e.g. "paper-final").
        - Otherwise a local-time timestamp ``YYYYMMDD-HHMMSS`` is used.

    Datasets are intentionally NOT stored. They are regenerated from
    ``master_seed`` + ``DataGeneratorConfig`` recorded in manifest.json,
    which takes only a few seconds and keeps the run folders small.

    Returns a run-context dict consumed by the notebooks.
    """
    root = Path(results_root).expanduser().resolve() / experiment_name
    _ensure_dir(root)

    # ── Load mode ────────────────────────────────────────────────────────────
    if load_run_dir is not None:
        run_dir = _coerce_run_dir(load_run_dir)
        if run_dir is None or not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {load_run_dir}")
        ctx = {"mode": "load", "root_dir": root}
        ctx.update(_subdirs(run_dir))
        ctx["manifest_path"] = run_dir / "manifest.json"
        return ctx

    # ── Memory-only mode (save_run=False) ────────────────────────────────────
    if not save_run:
        ctx = {"mode": "memory", "root_dir": root}
        ctx.update({k: None for k in ("run_dir", "figures_dir", "plot_inputs_dir",
                                      "pfi_dir", "methods_dir", "manifest_path")})
        return ctx

    # ── Save mode ────────────────────────────────────────────────────────────
    # run_tag → stable folder name; otherwise local-time timestamp
    if run_tag:
        run_id = run_tag
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = root / run_id
    subdirs = _subdirs(run_dir)
    _ensure_dir(run_dir)
    for key, path in subdirs.items():
        if key != "run_dir":
            _ensure_dir(path)

    # Update "latest" symlink
    latest_link = root / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        if latest_link.is_dir() and not latest_link.is_symlink():
            shutil.rmtree(latest_link)
        else:
            latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    _append_run_index(root, run_id, run_tag)

    ctx = {"mode": "save", "root_dir": root, "run_id": run_id}
    ctx.update(subdirs)
    ctx["manifest_path"] = run_dir / "manifest.json"
    return ctx


def load_benchmark_run(run_dir: str) -> Dict[str, Any]:
    """Attach to an existing benchmark run directory."""
    return prepare_benchmark_run(
        experiment_name=Path(run_dir).resolve().parent.name,
        save_run=False,
        load_run_dir=run_dir,
    )


# ── Manifest ──────────────────────────────────────────────────────────────────

def save_manifest_sections(run_ctx: Dict[str, Any], **sections) -> Optional[Path]:
    """Merge notebook metadata sections into manifest.json."""
    manifest_path = run_ctx.get("manifest_path")
    if manifest_path is None:
        return None
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest.update(sections)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    return manifest_path


def load_manifest(run_ctx: Dict[str, Any]) -> dict:
    manifest_path = run_ctx.get("manifest_path")
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError("manifest.json is missing for this run.")
    with open(manifest_path) as f:
        return json.load(f)


# ── Low-level dataset I/O (kept for standalone scripts; not used by notebooks) ─

def save_dataset(data: dict, path: str, fmt: str,
                 env_config: Optional[dict] = None,
                 generator_config: Optional[dict] = None) -> str:
    """Save a dataset dict to disk with a manifest."""
    target = Path(path)
    _ensure_dir(target)
    filename = {"trajectory": "trajectory.npz", "flat": "flat.npz"}
    if fmt not in filename:
        raise ValueError(f"fmt must be 'trajectory' or 'flat', got '{fmt}'")
    arrays = {k: _to_numpy(v) for k, v in data.items()}
    np.savez(target / filename[fmt], **arrays)
    manifest = {}
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest[fmt] = {
        "keys": list(arrays), "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "saved_at": datetime.now().isoformat(),
    }
    if env_config is not None:
        manifest["env_config"] = env_config
    if generator_config is not None:
        manifest["generator_config"] = generator_config
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    return str(target)


def load_dataset(path: str, fmt: str) -> dict:
    """Load a persisted trajectory or flattened dataset."""
    target = Path(path)
    filename = {"trajectory": "trajectory.npz", "flat": "flat.npz"}
    if fmt not in filename:
        raise ValueError(f"fmt must be 'trajectory' or 'flat', got '{fmt}'")
    npz_path = target / filename[fmt]
    if not npz_path.exists():
        available = sorted(p.name for p in target.iterdir()) if target.exists() else []
        raise FileNotFoundError(
            f"No {filename[fmt]} found at {target}. Available files: {available}"
        )
    npz = np.load(npz_path)
    return {k: tf.constant(npz[k]) for k in npz.files}


def load_dataset_manifest(path: str) -> dict:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found at {path}")
    with open(manifest_path) as f:
        return json.load(f)


# ── Plot inputs ───────────────────────────────────────────────────────────────

def save_plot_inputs(run_ctx: Dict[str, Any], name: str,
                     arrays: Dict[str, Any]) -> Optional[Path]:
    if run_ctx.get("plot_inputs_dir") is None:
        return None
    path = run_ctx["plot_inputs_dir"] / f"{name}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load_plot_inputs(run_ctx: Dict[str, Any], name: str) -> Dict[str, np.ndarray]:
    path = run_ctx["plot_inputs_dir"] / f"{name}.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ── Figures ───────────────────────────────────────────────────────────────────

def save_figure(run_ctx: Dict[str, Any], fig, filename: str,
                dpi: int = 150) -> Optional[Path]:
    figures_dir = run_ctx.get("figures_dir")
    if figures_dir is None:
        return None
    path = figures_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ── PFI bundle ────────────────────────────────────────────────────────────────

def _solver_bundle_dir(run_ctx: Dict[str, Any],
                       name: Optional[str] = None) -> Optional[Path]:
    pfi_dir = run_ctx.get("pfi_dir")
    if pfi_dir is None:
        return None
    return pfi_dir if not name else pfi_dir / str(name)


def save_pfi_bundle(run_ctx: Dict[str, Any], result: dict,
                    summary: Optional[dict] = None,
                    name: Optional[str] = None) -> Optional[Path]:
    pfi_dir = _solver_bundle_dir(run_ctx, name=name)
    if pfi_dir is None:
        return None
    _ensure_dir(pfi_dir)

    solution_payload = {k: _to_numpy(result[k])
                        for k in ("value", "policy_action", "policy_endo", "prob_matrix")
                        if k in result}
    np.savez_compressed(pfi_dir / "solution.npz", **solution_payload)

    grid_payload = {}
    for prefix, key in (("exo", "exo_grids_1d"), ("endo", "endo_grids_1d"),
                        ("action", "action_grids_1d")):
        for i, grid in enumerate(result["grids"][key]):
            grid_payload[f"{prefix}_{i}"] = np.asarray(grid)
    np.savez_compressed(pfi_dir / "grids.npz", **grid_payload)

    merged_summary = {"converged": bool(result.get("converged", False)),
                      "n_iter": int(result.get("n_iter", 0))}
    if summary:
        merged_summary.update(summary)
    with open(pfi_dir / "summary.json", "w") as f:
        json.dump(merged_summary, f, indent=2, default=_json_default)
    return pfi_dir


def load_pfi_bundle(run_ctx: Dict[str, Any], name: Optional[str] = None) -> dict:
    pfi_dir = _solver_bundle_dir(run_ctx, name=name)
    if pfi_dir is None:
        raise FileNotFoundError("No solver bundle directory is configured for this run.")
    solution = np.load(pfi_dir / "solution.npz")
    grids    = np.load(pfi_dir / "grids.npz")
    with open(pfi_dir / "summary.json") as f:
        summary = json.load(f)

    def _load_axis(prefix: str):
        items, i = [], 0
        while f"{prefix}_{i}" in grids.files:
            items.append(grids[f"{prefix}_{i}"])
            i += 1
        return items

    result = {
        "value":        tf.constant(solution["value"]) if "value" in solution else None,
        "policy_action": tf.constant(solution["policy_action"]),
        "policy_endo":  tf.constant(solution["policy_endo"]),
        "prob_matrix":  tf.constant(solution["prob_matrix"]) if "prob_matrix" in solution else None,
        "grids": {
            "exo_grids_1d":    _load_axis("exo"),
            "endo_grids_1d":   _load_axis("endo"),
            "action_grids_1d": _load_axis("action"),
        },
        "converged": summary.get("converged", False),
        "n_iter":    summary.get("n_iter", 0),
    }
    return {"result": result, "summary": summary}


# ── Snapshot / checkpoint helpers ─────────────────────────────────────────────

def save_snapshot_weights(path: Path, weights: Sequence[np.ndarray]) -> Path:
    payload = {f"w_{i:03d}": np.asarray(w, dtype=np.float32) for i, w in enumerate(weights)}
    np.savez_compressed(path, **payload)
    return path


def load_snapshot_weights(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    return [data[key] for key in sorted(data.files)]


def _normalize_checkpoint_entry(entry: Mapping[str, Any]) -> dict:
    step = int(entry["step"])
    models = {}
    for model_name, weights in dict(entry.get("models", {})).items():
        if weights is None:
            continue
        models[str(model_name)] = [np.asarray(w, dtype=np.float32) for w in weights]
    return {"step": step, "models": models}


def _checkpoint_entry_from_weight_history(step: int,
                                           weights: Sequence[np.ndarray]) -> dict:
    return {"step": int(step),
            "models": {"policy": [np.asarray(w, dtype=np.float32) for w in weights]}}


def normalize_checkpoint_history(
    checkpoint_history: Optional[Iterable[Mapping[str, Any]]] = None,
    weight_history: Optional[Iterable[tuple[int, Sequence[np.ndarray]]]] = None,
) -> list[dict]:
    if checkpoint_history is not None:
        return [_normalize_checkpoint_entry(e) for e in checkpoint_history]
    if weight_history is not None:
        return [_checkpoint_entry_from_weight_history(s, w) for s, w in weight_history]
    return []


def checkpoint_history_to_weight_history(
    checkpoint_history: Iterable[Mapping[str, Any]],
) -> list[tuple[int, list[np.ndarray]]]:
    projected = []
    for entry in checkpoint_history:
        norm = _normalize_checkpoint_entry(entry)
        if "policy" not in norm["models"]:
            continue
        projected.append((norm["step"], norm["models"]["policy"]))
    return projected


def save_checkpoint_history(method_dir: Path,
                             checkpoint_history: Iterable[Mapping[str, Any]]) -> None:
    snapshots_dir = _ensure_dir(method_dir / "snapshots")
    snapshot_index = []
    for entry in checkpoint_history:
        norm = _normalize_checkpoint_entry(entry)
        model_files = {}
        for model_name, weights in norm["models"].items():
            filename = f"step_{norm['step']:06d}_{model_name}.npz"
            save_snapshot_weights(snapshots_dir / filename, weights)
            model_files[model_name] = filename
        snapshot_index.append({"step": norm["step"], "models": model_files})
    with open(method_dir / "snapshots.json", "w") as f:
        json.dump(snapshot_index, f, indent=2)


def load_checkpoint_history(method_dir: Path) -> list[dict]:
    snapshots_index_path = method_dir / "snapshots.json"
    if not snapshots_index_path.exists():
        return []
    with open(snapshots_index_path) as f:
        snapshot_index = json.load(f)
    checkpoint_history = []
    for item in snapshot_index:
        models = {}
        if "models" in item:
            for model_name, filename in item["models"].items():
                models[model_name] = load_snapshot_weights(
                    method_dir / "snapshots" / filename)
        elif "file" in item:
            # Backward compatibility: legacy policy-only snapshot index.
            models["policy"] = load_snapshot_weights(
                method_dir / "snapshots" / item["file"])
        checkpoint_history.append({"step": int(item["step"]), "models": models})
    return checkpoint_history


# ── Method bundle ─────────────────────────────────────────────────────────────

def _serialize_result_payload(result: Optional[dict]) -> dict:
    if result is None:
        return {}
    return {k: v for k, v in result.items()
            if k not in {"policy", "value_net", "history", "config"}}


def _serialize_config_payload(config_obj) -> dict:
    if config_obj is None:
        return {}
    if is_dataclass(config_obj):
        payload = asdict(config_obj)
    elif isinstance(config_obj, dict):
        payload = dict(config_obj)
    else:
        payload = {"repr": repr(config_obj)}
    payload.pop("weight_history", None)
    payload.pop("checkpoint_history", None)
    return payload


def _save_history_npz(path: Path, history: Optional[dict]) -> None:
    if history is None:
        return
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in history.items()})


def _load_history_npz(path: Path) -> dict:
    data = np.load(path)
    return {k: data[k].tolist() for k in data.files}


def save_method_bundle(run_ctx: Dict[str, Any], method: str,
                       result: Optional[dict] = None,
                       checkpoint_history: Optional[Iterable[Mapping[str, Any]]] = None,
                       weight_history: Optional[Iterable[tuple[int, Sequence[np.ndarray]]]] = None,
                       checkpoint_metrics: Optional[dict] = None,
                       selected: Optional[dict] = None,
                       extra_sections: Optional[dict] = None) -> Optional[Path]:
    """Persist one NN method bundle for later reload and analysis."""
    methods_dir = run_ctx.get("methods_dir")
    if methods_dir is None:
        return None

    method_dir = methods_dir / method
    _ensure_dir(method_dir)

    if result is not None and "config" in result:
        with open(method_dir / "config.json", "w") as f:
            json.dump(_serialize_config_payload(result["config"]), f,
                      indent=2, default=_json_default)

    if result is not None and "history" in result:
        _save_history_npz(method_dir / "history.npz", result["history"])

    result_payload = _serialize_result_payload(result)
    if extra_sections:
        result_payload.update(extra_sections)
    if result_payload:
        with open(method_dir / "result.json", "w") as f:
            json.dump(result_payload, f, indent=2, default=_json_default)

    if checkpoint_metrics is not None:
        _save_history_npz(method_dir / "checkpoint_metrics.npz", checkpoint_metrics)

    if selected is not None:
        with open(method_dir / "selected.json", "w") as f:
            json.dump(selected, f, indent=2, default=_json_default)

    normalized = normalize_checkpoint_history(
        checkpoint_history=checkpoint_history, weight_history=weight_history)
    if normalized:
        save_checkpoint_history(method_dir, normalized)

    return method_dir


def load_method_bundle(run_ctx: Dict[str, Any], method: str) -> dict:
    method_dir = run_ctx["methods_dir"] / method
    with open(method_dir / "config.json") as f:
        config = json.load(f)
    with open(method_dir / "result.json") as f:
        result_payload = json.load(f)

    history = _load_history_npz(method_dir / "history.npz")

    checkpoint_metrics = None
    metrics_path = method_dir / "checkpoint_metrics.npz"
    if metrics_path.exists():
        checkpoint_metrics = _load_history_npz(metrics_path)

    selected = None
    selected_path = method_dir / "selected.json"
    if selected_path.exists():
        with open(selected_path) as f:
            selected = json.load(f)

    checkpoint_history = load_checkpoint_history(method_dir)
    weight_history = checkpoint_history_to_weight_history(checkpoint_history)

    result = dict(result_payload)
    result["config"] = config
    result["history"] = history
    return {
        "result":             result,
        "checkpoint_history": checkpoint_history,
        "weight_history":     weight_history,
        "checkpoint_metrics": checkpoint_metrics,
        "selected":           selected,
    }


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary_rows(run_ctx: Dict[str, Any], rows: Sequence[dict],
                      filename: str = "summary.csv") -> Optional[Path]:
    if run_ctx.get("run_dir") is None or not rows:
        return None
    path = run_ctx["run_dir"] / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_summary_rows(run_ctx: Dict[str, Any],
                      filename: str = "summary.csv") -> list[dict]:
    path = run_ctx["run_dir"] / filename
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Legacy training result bundle (kept for standalone scripts) ───────────────

def save_training_result_bundle(result, run_dir: str,
                                 policy=None, value_net=None) -> str:
    run_path   = _ensure_dir(Path(run_dir))
    weights_dir = _ensure_dir(run_path / "weights")
    with open(run_path / "config.json", "w") as f:
        json.dump(result.config, f, indent=2, default=_json_default)
    with open(run_path / "metadata.json", "w") as f:
        json.dump(result.metadata, f, indent=2, default=_json_default)
    _save_history_npz(run_path / "history.npz", result.history)
    if policy is not None:
        policy.save_weights(str(weights_dir / "policy.weights.h5"))
    if value_net is not None:
        value_net.save_weights(str(weights_dir / "value_net.weights.h5"))
    return str(run_path)


def load_training_result_bundle(run_dir: str) -> dict:
    run_path = Path(run_dir)
    with open(run_path / "config.json") as f:
        config = json.load(f)
    with open(run_path / "metadata.json") as f:
        metadata = json.load(f)
    history = _load_history_npz(run_path / "history.npz")
    return {"config": config, "history": history, "metadata": metadata,
            "run_dir": str(run_path)}
