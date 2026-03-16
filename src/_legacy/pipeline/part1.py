"""
Part-1 lightweight config-driven pipeline.

Stages:
1. generate-data
2. train-ddp
3. train-nn
4. compare
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.ddp import DDPGridConfig, BasicModelDDP, RiskyModelDDP, save_ddp_solution, load_ddp_solution
from src.economy.data import load_dataset_bundle, save_dataset_to_disk
from src.economy.data_generator import DataGenerator
from src.economy.parameters import EconomicParams, ShockParams
from src.trainers import (
    NetworkConfig,
    OptimizationConfig,
    AnnealingConfig,
    EarlyStoppingConfig,
    MethodConfig,
    RiskyDebtConfig,
    ExperimentConfig,
    DataConfig,
    train,
)
from src.trainers.io_transforms import build_inference_helper_from_result
from src.trainers.config import (
    OptimizerConfig,
    ObservationNormalizationConfig,
    InferenceClipConfig,
)
from src.utils.checkpointing import save_training_result, load_training_result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Use JSON config or install pyyaml."
            ) from exc
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported config extension: {suffix}. Use .json or .yaml.")


def _stable_hash(payload: Mapping[str, Any], n_chars: int = 12) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n_chars]


def _to_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _safe_update_symlink(link_path: Path, target: Path) -> None:
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        # Non-symlink "latest" is unexpected; keep it untouched.
        return
    link_path.symlink_to(target)


def _build_optimizer_config(raw: Mapping[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(**dict(raw))


def _build_early_stopping(raw: Optional[Mapping[str, Any]]) -> Optional[EarlyStoppingConfig]:
    if raw is None:
        return None
    return EarlyStoppingConfig(**dict(raw))


def _build_optimization_config(raw: Mapping[str, Any]) -> OptimizationConfig:
    payload = dict(raw)
    payload["optimizer"] = _build_optimizer_config(payload.get("optimizer", {}))
    payload["early_stopping"] = _build_early_stopping(payload.get("early_stopping"))
    return OptimizationConfig(**payload)


def _build_method_config(raw: Mapping[str, Any]) -> MethodConfig:
    payload = dict(raw)
    risky = payload.get("risky")
    if risky is not None:
        payload["risky"] = RiskyDebtConfig(**dict(risky))
    return MethodConfig(**payload)


def _build_network_config(raw: Mapping[str, Any]) -> NetworkConfig:
    payload = dict(raw)
    payload.pop("use_batch_norm", None)
    payload.pop("allow_unsafe_heads", None)
    obs_raw = payload.get("observation_normalization")
    if obs_raw is not None:
        payload["observation_normalization"] = ObservationNormalizationConfig(
            **dict(obs_raw)
        )
    clip_raw = payload.get("inference_clips")
    if clip_raw is not None:
        payload["inference_clips"] = InferenceClipConfig(**dict(clip_raw))
    return NetworkConfig(**payload)


def _mk_experiment_config(
    *,
    name: str,
    params: EconomicParams,
    shock_params: ShockParams,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizationConfig,
    anneal_cfg: AnnealingConfig,
    method_cfg: MethodConfig,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        params=params,
        shock_params=shock_params,
        network=net_cfg,
        optimization=opt_cfg,
        annealing=anneal_cfg,
        method=method_cfg,
    )


def _resolve_roots(config: Mapping[str, Any], output_root_override: Optional[str]) -> Dict[str, Any]:
    output_root = Path(output_root_override or config.get("output_root", "results"))
    run_layout = str(config.get("run_layout", "notebook")).strip().lower()
    if run_layout not in {"notebook", "pipeline"}:
        raise ValueError(f"Invalid run_layout='{run_layout}'. Use 'notebook' or 'pipeline'.")

    datasets_dirname = str(config.get("datasets_dirname", "datasets"))
    output_root.mkdir(parents=True, exist_ok=True)
    datasets_root = output_root / datasets_dirname
    runs_root = output_root if run_layout == "notebook" else output_root / "runs"
    datasets_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    return {
        "output_root": output_root,
        "datasets_root": datasets_root,
        "runs_root": runs_root,
        "run_layout": run_layout,
        "datasets_dirname": datasets_dirname,
    }


def _create_run_layout(paths: Mapping[str, Any], run_id: Optional[str], prefix: str) -> Dict[str, Path]:
    output_root = Path(paths["output_root"])
    runs_root = Path(paths["runs_root"])
    run_layout = str(paths["run_layout"])

    final_run_id = run_id or _to_run_id(prefix=prefix)
    run_dir = runs_root / final_run_id
    data_dir = run_dir / "data"
    checkpoints = run_dir / "checkpoints"
    figures = run_dir / "figures"
    for directory in (run_dir, data_dir, checkpoints, figures):
        directory.mkdir(parents=True, exist_ok=True)

    latest_target = Path(final_run_id) if run_layout == "notebook" else Path("runs") / final_run_id
    _safe_update_symlink(output_root / "latest", latest_target)
    return {
        "run_id": Path(final_run_id),
        "run_dir": run_dir,
        "data_dir": data_dir,
        "checkpoints": checkpoints,
        "figures": figures,
        "manifest_path": run_dir / "manifest.json",
    }


def _link_dataset_into_run(
    *,
    data_dir: Path,
    dataset_dir: Path,
    dataset_id: str,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_link = data_dir / "dataset"
    if dataset_link.is_symlink() or dataset_link.exists():
        if dataset_link.is_symlink() or dataset_link.is_file():
            dataset_link.unlink()
        else:
            shutil.rmtree(dataset_link, ignore_errors=True)

    rel_target = Path(os.path.relpath(dataset_dir, start=data_dir))
    dataset_link.symlink_to(rel_target)
    (data_dir / "dataset_id.txt").write_text(f"{dataset_id}\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _initialize_manifest(
    manifest_path: Path,
    *,
    pipeline_name: str,
    config_path: str,
    run_id: str,
) -> None:
    manifest = {
        "pipeline_name": pipeline_name,
        "config_path": config_path,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "stages": {},
    }
    _write_json(manifest_path, manifest)


def _append_stage(manifest_path: Path, stage_name: str, payload: Mapping[str, Any]) -> None:
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    stages = manifest.setdefault("stages", {})
    stages[stage_name] = dict(payload)
    manifest["updated_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)


def _build_bounds_dict(raw_bounds: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    return {
        "k": (float(raw_bounds["k"][0]), float(raw_bounds["k"][1])),
        "log_z": (float(raw_bounds["log_z"][0]), float(raw_bounds["log_z"][1])),
        "b": (float(raw_bounds["b"][0]), float(raw_bounds["b"][1])),
    }


def _last_scalar_metrics(history: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in history.items():
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, (float, int, np.floating, np.integer)):
                out[key] = float(last)
    return out


def _save_dataset_npz(path: Path, dataset: Dict[str, tf.Tensor], metadata: Mapping[str, Any]) -> None:
    save_dataset_to_disk(dataset, str(path), save_to_disk=True, metadata=dict(metadata))


def _dataset_id_payload(config: Mapping[str, Any], bounds: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "version": 1,
        "pipeline_name": config.get("pipeline_name", "part1_baseline"),
        "data": config["data"],
        "shock_params": config["shock_params"],
        "base_params": config["base_params"],
        "bounds": bounds,
    }


def _load_dataset_manifest(dataset_dir: Path) -> Dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest missing at {manifest_path}")
    manifest = _read_json(manifest_path)
    required = {"train_flat", "val_flat", "train_traj", "val_traj"}
    missing = required - set(manifest.get("files", {}).keys())
    if missing:
        raise ValueError(f"Dataset manifest missing files: {sorted(missing)}")
    return manifest


def _build_economic_params(config: Mapping[str, Any], key: str) -> EconomicParams:
    base = EconomicParams(**dict(config["base_params"]))
    overrides = dict(config.get(key, {}))
    return base.with_overrides(base=base, log_changes=False, **overrides)


def _build_data_generator(config: Mapping[str, Any]) -> Tuple[DataGenerator, ShockParams, Dict[str, Any]]:
    shock_params = ShockParams(**dict(config["shock_params"]))
    base_params = EconomicParams(**dict(config["base_params"]))
    data_cfg = DataConfig(**dict(config["data"]))
    generator, _, bounds = data_cfg.create_generator(
        params=base_params,
        shock_params=shock_params,
        verbose=False,
    )
    return generator, shock_params, bounds


def _dataset_files_from_manifest(manifest: Mapping[str, Any], dataset_dir: Path) -> Dict[str, Path]:
    return {
        key: dataset_dir / rel_path
        for key, rel_path in manifest["files"].items()
    }


def _collect_dataset_ids(obj: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "dataset_id" and isinstance(value, str):
                ids.add(value)
            else:
                ids |= _collect_dataset_ids(value)
    elif isinstance(obj, list):
        for item in obj:
            ids |= _collect_dataset_ids(item)
    return ids


def run_generate_data(
    config_path: str,
    *,
    output_root_override: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    output_root = Path(paths["output_root"])
    datasets_root = Path(paths["datasets_root"])
    generator, shock_params, bounds = _build_data_generator(config)
    bounds_dict = _build_bounds_dict(bounds)

    resolved_dataset_id = dataset_id or f"ds_{_stable_hash(_dataset_id_payload(config, bounds), n_chars=12)}"
    dataset_dir = datasets_root / resolved_dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = _load_dataset_manifest(dataset_dir)
    else:
        train_traj = generator.get_training_dataset()
        val_traj = generator.get_validation_dataset()
        train_flat = generator.get_flattened_training_dataset(include_debt=True)
        val_flat = generator.get_flattened_validation_dataset(include_debt=True)

        base_meta = {
            "schema_version": "pipeline-dataset-1.0",
            "pipeline_name": config.get("pipeline_name", "part1_baseline"),
            "dataset_id": resolved_dataset_id,
            "created_at_utc": _utc_now(),
            "bounds": {
                "k": [float(bounds_dict["k"][0]), float(bounds_dict["k"][1])],
                "log_z": [float(bounds_dict["log_z"][0]), float(bounds_dict["log_z"][1])],
                "b": [float(bounds_dict["b"][0]), float(bounds_dict["b"][1])],
            },
            "shock_params": asdict(shock_params),
            "data_config": asdict(DataConfig(**dict(config["data"]))),
        }

        files = {
            "train_traj": "train_traj.npz",
            "val_traj": "val_traj.npz",
            "train_flat": "train_flat.npz",
            "val_flat": "val_flat.npz",
        }

        _save_dataset_npz(
            dataset_dir / files["train_traj"],
            train_traj,
            {**base_meta, "split": "train", "kind": "trajectory"},
        )
        _save_dataset_npz(
            dataset_dir / files["val_traj"],
            val_traj,
            {**base_meta, "split": "validation", "kind": "trajectory"},
        )
        _save_dataset_npz(
            dataset_dir / files["train_flat"],
            train_flat,
            {**base_meta, "split": "train", "kind": "flat"},
        )
        _save_dataset_npz(
            dataset_dir / files["val_flat"],
            val_flat,
            {**base_meta, "split": "validation", "kind": "flat"},
        )

        train_flat_bundle = load_dataset_bundle(str(dataset_dir / files["train_flat"]), require_metadata=True)
        manifest = {
            "dataset_id": resolved_dataset_id,
            "dataset_dir": str(dataset_dir),
            "created_at_utc": _utc_now(),
            "files": files,
            "bounds": train_flat_bundle.bounds,
            "fingerprint": train_flat_bundle.fingerprint,
        }
        _write_json(manifest_path, manifest)

    _safe_update_symlink(
        output_root / "latest_dataset",
        Path(str(paths["datasets_dirname"])) / resolved_dataset_id,
    )
    return manifest


def run_gc_datasets(
    config_path: str,
    *,
    output_root_override: Optional[str] = None,
    min_age_days: int = 14,
    dry_run: bool = True,
) -> Dict[str, Any]:
    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    output_root = Path(paths["output_root"])
    datasets_root = Path(paths["datasets_root"])

    referenced: set[str] = set()
    for manifest_path in output_root.rglob("manifest.json"):
        try:
            payload = _read_json(manifest_path)
        except Exception:
            continue
        referenced |= _collect_dataset_ids(payload)

    now = time.time()
    min_age_sec = float(min_age_days) * 24.0 * 3600.0
    deleted: list[str] = []
    kept: list[str] = []
    candidates: list[str] = []

    for ds_dir in datasets_root.glob("ds_*"):
        if not ds_dir.is_dir():
            continue
        ds_id = ds_dir.name
        age_sec = now - ds_dir.stat().st_mtime
        if ds_id in referenced:
            kept.append(ds_id)
            continue
        if age_sec < min_age_sec:
            kept.append(ds_id)
            continue
        candidates.append(ds_id)
        if not dry_run:
            shutil.rmtree(ds_dir, ignore_errors=True)
            deleted.append(ds_id)

    return {
        "output_root": str(output_root),
        "dry_run": dry_run,
        "min_age_days": min_age_days,
        "referenced_dataset_ids": sorted(referenced),
        "gc_candidates": sorted(candidates),
        "deleted_dataset_ids": sorted(deleted),
        "kept_dataset_ids": sorted(kept),
    }


def run_clean_results(
    config_path: str,
    *,
    output_root_override: Optional[str] = None,
    keep_last_n_runs: int = 3,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Archive legacy/deprecated results and optionally prune old runs.

    Safety:
    - Dry-run by default.
    - Moves to `_archive/cleanup-<timestamp>/` when apply mode is used.
    """
    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    output_root = Path(paths["output_root"])
    datasets_dirname = str(paths["datasets_dirname"])

    run_like_dirs = []
    legacy_dirs = []
    run_name_pattern = re.compile(r"^run-\d{8}-\d{6}$")
    for child in output_root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        name = child.name
        if name in {datasets_dirname, "_archive"}:
            continue
        if run_name_pattern.match(name) or (child / "manifest.json").exists():
            run_like_dirs.append(child)
            continue
        if name.startswith(("part1-", "pipeline", "run-ddp-")):
            legacy_dirs.append(child)

    run_like_dirs = sorted(run_like_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    keep_runs = {p.name for p in run_like_dirs[: max(0, keep_last_n_runs)]}

    latest_link = output_root / "latest"
    if latest_link.is_symlink():
        latest_target = latest_link.resolve()
        if latest_target.exists() and latest_target.parent == output_root:
            keep_runs.add(latest_target.name)

    to_archive = []
    for run_dir in run_like_dirs:
        if run_dir.name not in keep_runs:
            to_archive.append(run_dir)
    to_archive.extend(legacy_dirs)

    archive_root = output_root / "_archive" / f"cleanup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    archived = []
    for src in to_archive:
        if dry_run:
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        dst = archive_root / src.name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(src), str(dst))
        archived.append(src.name)

    # If apply mode changed runs, repoint latest to newest kept run.
    if not dry_run:
        remaining_runs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("run-")]
        if remaining_runs:
            remaining_runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            _safe_update_symlink(output_root / "latest", Path(remaining_runs[0].name))

    return {
        "output_root": str(output_root),
        "dry_run": dry_run,
        "keep_last_n_runs": keep_last_n_runs,
        "keep_runs": sorted(keep_runs),
        "archive_target": str(archive_root),
        "archive_candidates": sorted([p.name for p in to_archive]),
        "archived": sorted(archived),
    }


def run_train_ddp(
    config_path: str,
    *,
    run_id: str,
    dataset_id: str,
    output_root_override: Optional[str] = None,
) -> Dict[str, Any]:
    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    output_root = Path(paths["output_root"])
    datasets_root = Path(paths["datasets_root"])
    runs_root = Path(paths["runs_root"])
    run_dir = runs_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    data_dir = run_dir / "data"
    manifest_path = run_dir / "manifest.json"
    if not run_dir.exists():
        layout = _create_run_layout(
            paths=paths,
            run_id=run_id,
            prefix=config.get("run_name_prefix", "run"),
        )
        run_dir = layout["run_dir"]
        data_dir = layout["data_dir"]
        checkpoints_dir = layout["checkpoints"]
        manifest_path = layout["manifest_path"]
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _initialize_manifest(
            manifest_path,
            pipeline_name=config.get("pipeline_name", "part1_baseline"),
            config_path=config_path,
            run_id=run_id,
        )

    dataset_dir = datasets_root / dataset_id
    _link_dataset_into_run(data_dir=data_dir, dataset_dir=dataset_dir, dataset_id=dataset_id)
    dataset_manifest = _load_dataset_manifest(dataset_dir)
    dataset_files = _dataset_files_from_manifest(dataset_manifest, dataset_dir)
    train_flat_bundle = load_dataset_bundle(str(dataset_files["train_flat"]), require_metadata=True)

    basic_params = _build_economic_params(config, "basic_baseline_params")
    risky_params = _build_economic_params(config, "risky_baseline_params")
    shock_params = ShockParams(**dict(config["shock_params"]))

    ddp_cfg = dict(config["ddp"])
    grid_basic = DDPGridConfig(**dict(ddp_cfg["grid_basic"]))
    grid_risky = DDPGridConfig(**dict(ddp_cfg["grid_risky"]))

    basic_model = BasicModelDDP.from_dataset_bundle(
        params=basic_params,
        bundle=train_flat_bundle,
        grid_config=grid_basic,
    )
    v_basic_vfi, p_basic_vfi = basic_model.solve_basic_vfi(**dict(ddp_cfg["basic_vfi"]))
    v_basic_pfi, p_basic_pfi = basic_model.solve_basic_pfi(**dict(ddp_cfg["basic_pfi"]))
    basic_value_gap = float(np.max(np.abs(v_basic_vfi.numpy() - v_basic_pfi.numpy())))
    basic_policy_gap = float(np.max(np.abs(p_basic_vfi.numpy() - p_basic_pfi.numpy())))

    save_ddp_solution(
        save_dir=str(checkpoints_dir),
        model_name="basic",
        scenario_name="baseline",
        solver_name="vfi",
        value=v_basic_vfi,
        policy_k=p_basic_vfi,
        params=basic_params,
        shock_params=shock_params,
        grid_config=grid_basic,
        metrics={
            "dataset_id": dataset_id,
            "value_pfi_gap_max_abs": basic_value_gap,
            "policy_k_pfi_gap_max_abs": basic_policy_gap,
        },
        overwrite=True,
        verbose=False,
    )
    save_ddp_solution(
        save_dir=str(checkpoints_dir),
        model_name="basic",
        scenario_name="baseline",
        solver_name="pfi",
        value=v_basic_pfi,
        policy_k=p_basic_pfi,
        params=basic_params,
        shock_params=shock_params,
        grid_config=grid_basic,
        metrics={
            "dataset_id": dataset_id,
            "value_vfi_gap_max_abs": basic_value_gap,
            "policy_k_vfi_gap_max_abs": basic_policy_gap,
        },
        overwrite=True,
        verbose=False,
    )

    risky_model = RiskyModelDDP.from_dataset_bundle(
        params=risky_params,
        bundle=train_flat_bundle,
        grid_config=grid_risky,
    )
    v_risky_vfi, (pk_vfi, pb_vfi), q_vfi = risky_model.solve_risky_vfi(**dict(ddp_cfg["risky_vfi"]))
    v_risky_pfi, (pk_pfi, pb_pfi), q_pfi = risky_model.solve_risky_pfi(**dict(ddp_cfg["risky_pfi"]))
    risky_value_gap = float(np.max(np.abs(v_risky_vfi.numpy() - v_risky_pfi.numpy())))
    risky_k_gap = float(np.max(np.abs(pk_vfi.numpy() - pk_pfi.numpy())))
    risky_b_gap = float(np.max(np.abs(pb_vfi.numpy() - pb_pfi.numpy())))
    risky_q_gap = float(np.max(np.abs(q_vfi.numpy() - q_pfi.numpy())))

    save_ddp_solution(
        save_dir=str(checkpoints_dir),
        model_name="risky",
        scenario_name="baseline",
        solver_name="vfi",
        value=v_risky_vfi,
        policy_k=pk_vfi,
        policy_b=pb_vfi,
        bond_price=q_vfi,
        params=risky_params,
        shock_params=shock_params,
        grid_config=grid_risky,
        metrics={
            "dataset_id": dataset_id,
            "value_pfi_gap_max_abs": risky_value_gap,
            "policy_k_pfi_gap_max_abs": risky_k_gap,
            "policy_b_pfi_gap_max_abs": risky_b_gap,
            "q_pfi_gap_max_abs": risky_q_gap,
        },
        overwrite=True,
        verbose=False,
    )
    save_ddp_solution(
        save_dir=str(checkpoints_dir),
        model_name="risky",
        scenario_name="baseline",
        solver_name="pfi",
        value=v_risky_pfi,
        policy_k=pk_pfi,
        policy_b=pb_pfi,
        bond_price=q_pfi,
        params=risky_params,
        shock_params=shock_params,
        grid_config=grid_risky,
        metrics={
            "dataset_id": dataset_id,
            "value_vfi_gap_max_abs": risky_value_gap,
            "policy_k_vfi_gap_max_abs": risky_k_gap,
            "policy_b_vfi_gap_max_abs": risky_b_gap,
            "q_vfi_gap_max_abs": risky_q_gap,
        },
        overwrite=True,
        verbose=False,
    )

    summary = {
        "dataset_id": dataset_id,
        "basic": {
            "value_max_abs": basic_value_gap,
            "policy_k_max_abs": basic_policy_gap,
        },
        "risky": {
            "value_max_abs": risky_value_gap,
            "policy_k_max_abs": risky_k_gap,
            "policy_b_max_abs": risky_b_gap,
            "q_max_abs": risky_q_gap,
        },
    }
    _append_stage(manifest_path, "train_ddp", summary)
    return summary


def run_train_nn(
    config_path: str,
    *,
    run_id: str,
    dataset_id: str,
    output_root_override: Optional[str] = None,
) -> Dict[str, Any]:
    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    output_root = Path(paths["output_root"])
    datasets_root = Path(paths["datasets_root"])
    runs_root = Path(paths["runs_root"])
    run_dir = runs_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    data_dir = run_dir / "data"
    manifest_path = run_dir / "manifest.json"
    if not run_dir.exists():
        layout = _create_run_layout(
            paths=paths,
            run_id=run_id,
            prefix=config.get("run_name_prefix", "run"),
        )
        run_dir = layout["run_dir"]
        data_dir = layout["data_dir"]
        checkpoints_dir = layout["checkpoints"]
        manifest_path = layout["manifest_path"]
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _initialize_manifest(
            manifest_path,
            pipeline_name=config.get("pipeline_name", "part1_baseline"),
            config_path=config_path,
            run_id=run_id,
        )

    dataset_dir = datasets_root / dataset_id
    _link_dataset_into_run(data_dir=data_dir, dataset_dir=dataset_dir, dataset_id=dataset_id)
    dataset_manifest = _load_dataset_manifest(dataset_dir)
    dataset_files = _dataset_files_from_manifest(dataset_manifest, dataset_dir)

    train_flat = load_dataset_bundle(str(dataset_files["train_flat"]), require_metadata=True)
    val_flat = load_dataset_bundle(str(dataset_files["val_flat"]), require_metadata=True)
    train_traj = load_dataset_bundle(str(dataset_files["train_traj"]), require_metadata=True)
    val_traj = load_dataset_bundle(str(dataset_files["val_traj"]), require_metadata=True)

    basic_params = _build_economic_params(config, "basic_baseline_params")
    risky_params = _build_economic_params(config, "risky_baseline_params")
    shock_params = ShockParams(**dict(config["shock_params"]))

    nn_cfg = dict(config["nn"])
    net_cfg = _build_network_config(nn_cfg["network"])
    basic_opt = _build_optimization_config(nn_cfg["basic_optimization"])
    basic_anneal = AnnealingConfig(**dict(nn_cfg["basic_annealing"]))
    risky_opt = _build_optimization_config(nn_cfg["risky_optimization"])
    risky_anneal = AnnealingConfig(**dict(nn_cfg["risky_annealing"]))

    basic_methods = {k: _build_method_config(v) for k, v in nn_cfg["basic_methods"].items()}
    risky_method = _build_method_config(nn_cfg["risky_method"])

    nn_metrics: Dict[str, Any] = {"dataset_id": dataset_id, "basic": {}, "risky": {}}

    basic_ckpt_root = checkpoints_dir / "basic" / "baseline"
    risky_ckpt_root = checkpoints_dir / "risky" / "baseline"

    _write_json(
        checkpoints_dir / "bounds.json",
        {
            "k": list(train_flat.bounds["k"]),
            "log_z": list(train_flat.bounds["log_z"]),
            "b": list(train_flat.bounds["b"]),
        },
    )
    _write_json(
        checkpoints_dir / "dataset_ref.json",
        {
            "dataset_id": dataset_id,
            "dataset_dir": str(dataset_dir),
        },
    )

    # Train LR/ER from random init first, then warm-start supported BR variants from ER.
    method_order = [m for m in ("lr", "er", "br") if m in basic_methods]

    basic_results: Dict[str, Dict[str, Any]] = {}

    for method_alias in method_order:
        method_cfg = basic_methods[method_alias]
        exp_cfg = _mk_experiment_config(
            name=f"basic_baseline_{method_alias}",
            params=basic_params,
            shock_params=shock_params,
            net_cfg=net_cfg,
            opt_cfg=basic_opt,
            anneal_cfg=basic_anneal,
            method_cfg=method_cfg,
        )
        if method_alias == "lr":
            train_data = train_traj.data
            val_data = val_traj.data
            metadata = train_traj.metadata
        else:
            train_data = train_flat.data
            val_data = val_flat.data
            metadata = train_flat.metadata

        warm_start_policy = None
        if method_alias == "br":
            warm_start_policy = basic_results.get("er")
            if warm_start_policy is None:
                raise RuntimeError(
                    "Pipeline warm-start for BR methods requires ER result first. "
                    f"Missing ER result before method '{method_alias}'."
                )

        result = train(
            model="basic",
            method=method_alias,
            config=exp_cfg,
            train_data=train_data,
            dataset_metadata=metadata,
            validation_data=val_data,
            warm_start_policy=warm_start_policy,
            return_legacy_dict=True,
        )
        basic_results[method_alias] = result

        save_training_result(
            result=result,
            save_dir=str(basic_ckpt_root),
            name=method_alias,
            overwrite=True,
            verbose=False,
        )

        history = result.get("history", {})
        nn_metrics["basic"][method_alias] = {
            "final_metrics": _last_scalar_metrics(history),
            "warm_start_applied": bool(result.get("_meta", {}).get("warm_start_applied", False)),
            "warm_start_source": result.get("_meta", {}).get("warm_start_source"),
        }

    risky_exp_cfg = _mk_experiment_config(
        name="risky_baseline_br",
        params=risky_params,
        shock_params=shock_params,
        net_cfg=net_cfg,
        opt_cfg=risky_opt,
        anneal_cfg=risky_anneal,
        method_cfg=risky_method,
    )
    risky_result = train(
        model="risky",
        method="br",
        config=risky_exp_cfg,
        train_data=train_flat.data,
        dataset_metadata=train_flat.metadata,
        validation_data=val_flat.data,
        return_legacy_dict=True,
    )
    save_training_result(
        result=risky_result,
        save_dir=str(risky_ckpt_root),
        name="br",
        overwrite=True,
        verbose=False,
    )

    risky_hist = risky_result.get("history", {})
    nn_metrics["risky"]["br"] = {
        "final_metrics": _last_scalar_metrics(risky_hist),
    }

    _append_stage(manifest_path, "train_nn", nn_metrics)
    return nn_metrics


def run_compare(
    config_path: str,
    *,
    run_id: str,
    dataset_id: str,
    output_root_override: Optional[str] = None,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    output_root = Path(paths["output_root"])
    datasets_root = Path(paths["datasets_root"])
    runs_root = Path(paths["runs_root"])
    run_dir = runs_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"
    data_dir = run_dir / "data"
    manifest_path = run_dir / "manifest.json"
    if not run_dir.exists():
        layout = _create_run_layout(
            paths=paths,
            run_id=run_id,
            prefix=config.get("run_name_prefix", "run"),
        )
        run_dir = layout["run_dir"]
        data_dir = layout["data_dir"]
        checkpoints_dir = layout["checkpoints"]
        figures_dir = layout["figures"]
        manifest_path = layout["manifest_path"]
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _initialize_manifest(
            manifest_path,
            pipeline_name=config.get("pipeline_name", "part1_baseline"),
            config_path=config_path,
            run_id=run_id,
        )

    dataset_dir = datasets_root / dataset_id
    _link_dataset_into_run(data_dir=data_dir, dataset_dir=dataset_dir, dataset_id=dataset_id)
    dataset_manifest = _load_dataset_manifest(dataset_dir)
    dataset_files = _dataset_files_from_manifest(dataset_manifest, dataset_dir)
    train_flat_bundle = load_dataset_bundle(str(dataset_files["train_flat"]), require_metadata=True)
    bounds = train_flat_bundle.bounds

    basic_params = _build_economic_params(config, "basic_baseline_params")
    ddp_cfg = dict(config["ddp"])
    grid_basic = DDPGridConfig(**dict(ddp_cfg["grid_basic"]))
    grid_risky = DDPGridConfig(**dict(ddp_cfg["grid_risky"]))
    k_basic, z_basic, _ = grid_basic.generate_grids(bounds, delta=basic_params.delta)

    basic_vfi = load_ddp_solution(str(checkpoints_dir / "ddp" / "basic" / "baseline" / "vfi"))
    basic_pfi = load_ddp_solution(str(checkpoints_dir / "ddp" / "basic" / "baseline" / "pfi"))
    policy_basic_vfi = basic_vfi["arrays"]["policy_k"]
    policy_basic_pfi = basic_pfi["arrays"]["policy_k"]

    z_idx_basic = len(z_basic) // 2
    z_fixed_basic = float(z_basic[z_idx_basic])
    basic_slice_vfi = policy_basic_vfi[z_idx_basic, :]
    basic_slice_pfi = policy_basic_pfi[z_idx_basic, :]

    plt.figure(figsize=(10, 6))
    plt.plot(k_basic, basic_slice_vfi, label="DDP VFI", linewidth=2.2)
    plt.plot(k_basic, basic_slice_pfi, "--", label="DDP PFI", linewidth=2.2)

    basic_rmse: Dict[str, float] = {}
    for method in ("lr", "er", "br"):
        nn_result = load_training_result(
            str(checkpoints_dir / "basic" / "baseline" / method),
            load_target_nets=False,
            verbose=False,
        )
        infer = build_inference_helper_from_result(nn_result)
        k_in = tf.constant(k_basic.reshape(-1, 1), dtype=tf.float32)
        z_in = tf.constant(np.full((len(k_basic), 1), z_fixed_basic), dtype=tf.float32)
        k_next = infer.policy(k_in, z_in).numpy().reshape(-1)
        basic_rmse[method] = float(np.sqrt(np.mean((k_next - basic_slice_pfi) ** 2)))
        plt.plot(k_basic, k_next, label=f"NN {method.upper()}")

    plt.plot(k_basic, k_basic, ":", color="gray", label="45-degree")
    plt.title(f"Basic Baseline Policy Comparison at median z={z_fixed_basic:.3f}")
    plt.xlabel("Current capital k")
    plt.ylabel("Next capital k'")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    basic_fig = figures_dir / "compare_basic_baseline.png"
    plt.savefig(basic_fig, dpi=180, bbox_inches="tight")
    plt.close()

    risky_params = _build_economic_params(config, "risky_baseline_params")
    k_risky, z_risky, b_risky = grid_risky.generate_grids(bounds, delta=risky_params.delta)
    if b_risky is None:
        raise ValueError("Risky DDP comparison requires a valid debt grid.")

    risky_vfi = load_ddp_solution(str(checkpoints_dir / "ddp" / "risky" / "baseline" / "vfi"))
    risky_pfi = load_ddp_solution(str(checkpoints_dir / "ddp" / "risky" / "baseline" / "pfi"))
    pk_vfi = risky_vfi["arrays"]["policy_k"]
    pb_vfi = risky_vfi["arrays"]["policy_b"]
    pk_pfi = risky_pfi["arrays"]["policy_k"]
    pb_pfi = risky_pfi["arrays"]["policy_b"]

    z_idx = len(z_risky) // 2
    k_idx = len(k_risky) // 2
    z_fixed = float(z_risky[z_idx])
    k_fixed = float(k_risky[k_idx])

    b_policy_vfi = pb_vfi[z_idx, k_idx, :]
    b_policy_pfi = pb_pfi[z_idx, k_idx, :]
    k_policy_vfi = pk_vfi[z_idx, k_idx, :]
    k_policy_pfi = pk_pfi[z_idx, k_idx, :]

    risky_nn = load_training_result(
        str(checkpoints_dir / "risky" / "baseline" / "br"),
        load_target_nets=False,
        verbose=False,
    )
    risky_infer = build_inference_helper_from_result(risky_nn)
    k_in = tf.constant(np.full((len(b_risky), 1), k_fixed), dtype=tf.float32)
    b_in = tf.constant(b_risky.reshape(-1, 1), dtype=tf.float32)
    z_in = tf.constant(np.full((len(b_risky), 1), z_fixed), dtype=tf.float32)
    k_next_nn, b_next_nn = risky_infer.policy(k_in, b_in, z_in)
    k_next_nn = k_next_nn.numpy().reshape(-1)
    b_next_nn = b_next_nn.numpy().reshape(-1)

    risky_rmse_b = float(np.sqrt(np.mean((b_next_nn - b_policy_pfi) ** 2)))
    risky_rmse_k = float(np.sqrt(np.mean((k_next_nn - k_policy_pfi) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_b, ax_k = axes

    ax_b.plot(b_risky, b_policy_vfi, label="DDP VFI", linewidth=2.2)
    ax_b.plot(b_risky, b_policy_pfi, "--", label="DDP PFI", linewidth=2.2)
    ax_b.plot(b_risky, b_next_nn, label="NN BR")
    ax_b.plot(b_risky, b_risky, ":", color="gray", label="45-degree")
    ax_b.set_title(f"Debt Policy at z={z_fixed:.3f}, k={k_fixed:.2f}")
    ax_b.set_xlabel("Current debt b")
    ax_b.set_ylabel("Next debt b'")
    ax_b.grid(alpha=0.3)
    ax_b.legend()

    ax_k.plot(b_risky, k_policy_vfi, label="DDP VFI", linewidth=2.2)
    ax_k.plot(b_risky, k_policy_pfi, "--", label="DDP PFI", linewidth=2.2)
    ax_k.plot(b_risky, k_next_nn, label="NN BR")
    ax_k.set_title(f"Capital Policy at z={z_fixed:.3f}, k={k_fixed:.2f}")
    ax_k.set_xlabel("Current debt b")
    ax_k.set_ylabel("Next capital k'")
    ax_k.grid(alpha=0.3)
    ax_k.legend()

    plt.tight_layout()
    risky_fig = figures_dir / "compare_risky_baseline.png"
    plt.savefig(risky_fig, dpi=180, bbox_inches="tight")
    plt.close()

    comparison = {
        "dataset_id": dataset_id,
        "basic": {
            "z_fixed": z_fixed_basic,
            "ddp_vfi_pfi_policy_gap_max_abs": float(np.max(np.abs(basic_slice_vfi - basic_slice_pfi))),
            "nn_rmse_vs_ddp_pfi": basic_rmse,
            "figure_path": str(basic_fig),
        },
        "risky": {
            "z_fixed": z_fixed,
            "k_fixed": k_fixed,
            "ddp_vfi_pfi_policy_b_gap_max_abs": float(np.max(np.abs(b_policy_vfi - b_policy_pfi))),
            "ddp_vfi_pfi_policy_k_gap_max_abs": float(np.max(np.abs(k_policy_vfi - k_policy_pfi))),
            "nn_br_rmse_vs_ddp_pfi": {
                "b_next": risky_rmse_b,
                "k_next": risky_rmse_k,
            },
            "figure_path": str(risky_fig),
        },
    }

    _write_json(run_dir / "comparison_summary.json", comparison)
    _append_stage(manifest_path, "compare", comparison)
    return comparison


def run_all(
    config_path: str,
    *,
    output_root_override: Optional[str] = None,
    run_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = _load_config(config_path)
    paths = _resolve_roots(config, output_root_override)
    run_layout = _create_run_layout(
        paths=paths,
        run_id=run_id,
        prefix=config.get("run_name_prefix", "run"),
    )

    _initialize_manifest(
        run_layout["manifest_path"],
        pipeline_name=config.get("pipeline_name", "part1_baseline"),
        config_path=config_path,
        run_id=str(run_layout["run_id"]),
    )

    dataset_manifest = run_generate_data(
        config_path=config_path,
        output_root_override=output_root_override,
        dataset_id=dataset_id,
    )
    ds_id = dataset_manifest["dataset_id"]
    _append_stage(run_layout["manifest_path"], "generate_data", dataset_manifest)

    ddp_summary = run_train_ddp(
        config_path=config_path,
        run_id=str(run_layout["run_id"]),
        dataset_id=ds_id,
        output_root_override=output_root_override,
    )
    nn_summary = run_train_nn(
        config_path=config_path,
        run_id=str(run_layout["run_id"]),
        dataset_id=ds_id,
        output_root_override=output_root_override,
    )
    compare_summary = run_compare(
        config_path=config_path,
        run_id=str(run_layout["run_id"]),
        dataset_id=ds_id,
        output_root_override=output_root_override,
    )

    summary = {
        "run_id": str(run_layout["run_id"]),
        "run_dir": str(run_layout["run_dir"]),
        "dataset_id": ds_id,
        "ddp": ddp_summary,
        "nn": nn_summary,
        "compare": compare_summary,
    }
    _write_json(run_layout["run_dir"] / "run_summary.json", summary)
    return summary
