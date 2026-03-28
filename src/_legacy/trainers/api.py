"""
Unified trainer API entrypoint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from src.economy.data import DatasetBundle, canonicalize_bounds
from src.trainers.basic_api import (
    train_basic_lr,
    train_basic_er,
    train_basic_br_actor_critic,
)
from src.trainers.config import ExperimentConfig
from src.trainers.method_names import canonicalize_method_name
from src.trainers.results import TrainingResult
from src.trainers.risky_api import train_risky_br_actor_critic


_MODEL_ALIASES = {
    "basic": "basic",
    "risky": "risky",
    "risky_debt": "risky",
}


def _canonicalize_manual_bounds(
    bounds: Optional[Dict[str, Tuple[float, float]]],
) -> Optional[Dict[str, Tuple[float, float]]]:
    if bounds is None:
        return None
    out = dict(bounds)
    if "log_z" not in out and "logz" in out:
        out["log_z"] = out["logz"]
    return out


def _canonicalize_model(model: str) -> str:
    normalized = model.strip().lower().replace("-", "_")
    if normalized not in _MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{model}'. Supported models: {sorted(_MODEL_ALIASES)}."
        )
    return _MODEL_ALIASES[normalized]


def _coerce_seed_pair(
    raw_seed: Any,
    *,
    seed_label: str,
) -> Tuple[int, int]:
    """
    Coerce seed input to a 2-int stateless seed pair.
    """
    if isinstance(raw_seed, bool):
        raise ValueError(f"{seed_label} must be int or length-2 seed pair; got bool.")

    if isinstance(raw_seed, int):
        return int(raw_seed), 0

    if isinstance(raw_seed, (list, tuple)) and len(raw_seed) == 2:
        return int(raw_seed[0]), int(raw_seed[1])

    raise ValueError(
        f"{seed_label} must be int or length-2 list/tuple, got {raw_seed!r}."
    )


def _seed_pair_to_global_int(seed_pair: Tuple[int, int]) -> int:
    """
    Mix a 2-int seed pair into a deterministic non-negative 31-bit integer.
    """
    s0, s1 = int(seed_pair[0]), int(seed_pair[1])
    mixed = ((s0 & 0xFFFFFFFF) * 1_000_003 + (s1 & 0xFFFFFFFF) + 0x9E3779B9) & 0x7FFFFFFF
    return int(mixed)


def _resolve_training_seed(
    *,
    config: ExperimentConfig,
    dataset_metadata: Optional[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    """
    Resolve runtime training seed with explicit precedence:
    1) config.optimization.training_seed
    2) dataset_metadata['training_seed']
    3) dataset_metadata['master_seed'] (backward-compatible fallback)
    """
    config_seed = config.optimization.training_seed
    if config_seed is not None:
        return _coerce_seed_pair(
            config_seed,
            seed_label="config.optimization.training_seed",
        )

    if dataset_metadata is None:
        return None

    if "training_seed" in dataset_metadata and dataset_metadata["training_seed"] is not None:
        return _coerce_seed_pair(
            dataset_metadata["training_seed"],
            seed_label="dataset_metadata['training_seed']",
        )

    if "master_seed" in dataset_metadata and dataset_metadata["master_seed"] is not None:
        return _coerce_seed_pair(
            dataset_metadata["master_seed"],
            seed_label="dataset_metadata['master_seed']",
        )

    return None


def _configure_runtime_seed(
    *,
    config: ExperimentConfig,
    training_seed: Optional[Tuple[int, int]],
) -> None:
    """
    Configure framework-level runtime seed and deterministic-op mode.
    """
    if training_seed is not None:
        tf.keras.utils.set_random_seed(_seed_pair_to_global_int(training_seed))

    if config.optimization.deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Best effort only: some environments/devices may not support full determinism.
            pass


def train(
    *,
    model: str,
    method: str,
    config: ExperimentConfig,
    train_data: Dict[str, tf.Tensor],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    warm_start_policy: Optional[Any] = None,
    return_legacy_dict: bool = False,
) -> Union[TrainingResult, Dict[str, Any]]:
    """
    Unified training entrypoint for basic and risky models.

    Args:
        model: Model family ('basic' or 'risky').
        method: Method id or alias ('lr', 'er', 'br', etc.).
        config: ExperimentConfig containing params/shocks/network/opt/method/anneal.
        train_data: Training dataset dictionary.
        bounds: Optional manual bounds dictionary.
            Prefer metadata-driven bounds extraction via dataset_metadata.
        dataset_metadata: Optional metadata dictionary containing 'bounds' and
            optional seed fields ('training_seed' / 'master_seed'). If provided,
            bounds are auto-extracted and used as source of truth.
        validation_data: Optional validation dataset dictionary.
        warm_start_policy: Optional policy source used to initialize BR policy
            networks. Supported only for basic BR methods.
        return_legacy_dict: If True, return legacy dict schema.

    Returns:
        TrainingResult by default, or legacy result dict if return_legacy_dict=True.
    """
    canonical_model = _canonicalize_model(model)
    requested_method = canonicalize_method_name(method, model=canonical_model)
    configured_method = canonicalize_method_name(config.method.name, model=canonical_model)

    manual_bounds = _canonicalize_manual_bounds(bounds)
    metadata_bounds = None
    if dataset_metadata is not None:
        metadata_bounds = canonicalize_bounds(dataset_metadata)

    required_bounds_keys = {"k", "log_z"} if canonical_model == "basic" else {"k", "log_z", "b"}
    for source_name, candidate in [("manual bounds", manual_bounds), ("dataset metadata bounds", metadata_bounds)]:
        if candidate is None:
            continue
        missing = required_bounds_keys - set(candidate.keys())
        if missing:
            raise ValueError(f"{source_name} missing required keys for model '{canonical_model}': {sorted(missing)}")

    if metadata_bounds is not None and manual_bounds is not None:
        if (
            tuple(manual_bounds["k"]) != tuple(metadata_bounds["k"])
            or tuple(manual_bounds["log_z"]) != tuple(metadata_bounds["log_z"])
            or (
                "b" in required_bounds_keys
                and tuple(manual_bounds["b"]) != tuple(metadata_bounds["b"])
            )
        ):
            raise ValueError(
                "Manual bounds mismatch dataset metadata bounds. "
                "Use metadata-derived bounds to avoid configuration drift."
            )

    resolved_bounds = metadata_bounds or manual_bounds
    if resolved_bounds is None:
        raise ValueError(
            "Bounds are required. Provide dataset_metadata with bounds "
            "or pass bounds explicitly."
        )

    if requested_method != configured_method:
        raise ValueError(
            f"Method mismatch: train(..., method='{method}') resolves to "
            f"'{requested_method}' but config.method.name='{config.method.name}' "
            f"resolves to '{configured_method}'."
        )

    method_config = (
        replace(config.method, name=configured_method)
        if config.method.name != configured_method
        else config.method
    )
    training_seed = _resolve_training_seed(
        config=config,
        dataset_metadata=dataset_metadata,
    )
    _configure_runtime_seed(config=config, training_seed=training_seed)

    dispatch = {
        ("basic", "basic_lr"): train_basic_lr,
        ("basic", "basic_er"): train_basic_er,
        ("basic", "basic_br_actor_critic"): train_basic_br_actor_critic,
        ("risky", "risky_br_actor_critic"): train_risky_br_actor_critic,
    }
    key = (canonical_model, requested_method)
    if key not in dispatch:
        raise ValueError(
            f"Unsupported model/method combination: model='{canonical_model}', "
            f"method='{requested_method}'."
        )

    call_kwargs = dict(
        dataset=train_data,
        net_config=config.network,
        opt_config=config.optimization,
        method_config=method_config,
        anneal_config=config.annealing,
        params=config.params,
        shock_params=config.shock_params,
        bounds=resolved_bounds,
        validation_data=validation_data,
        training_seed=training_seed,
    )

    if warm_start_policy is not None:
        warm_start_methods = {
            ("basic", "basic_br_actor_critic"),
        }
        if key not in warm_start_methods:
            raise ValueError(
                "warm_start_policy is only supported for basic BR methods: "
                "basic_br_actor_critic."
            )
        call_kwargs["warm_start_policy"] = warm_start_policy

    payload = dispatch[key](**call_kwargs)

    if isinstance(payload, TrainingResult):
        result_obj = payload
    else:
        result_obj = TrainingResult.from_legacy_dict(payload)

    if return_legacy_dict:
        return result_obj.to_legacy_dict()
    return result_obj


def train_from_dataset_bundle(
    *,
    model: str,
    method: str,
    config: ExperimentConfig,
    train_bundle: DatasetBundle,
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    warm_start_policy: Optional[Any] = None,
    return_legacy_dict: bool = False,
) -> Union[TrainingResult, Dict[str, Any]]:
    """
    Metadata-first offline training entrypoint.

    This wrapper enforces bounds extraction from dataset metadata and avoids
    manual bounds overrides.
    """
    return train(
        model=model,
        method=method,
        config=config,
        train_data=train_bundle.data,
        dataset_metadata=train_bundle.metadata,
        bounds=None,
        validation_data=validation_data,
        warm_start_policy=warm_start_policy,
        return_legacy_dict=return_legacy_dict,
    )
