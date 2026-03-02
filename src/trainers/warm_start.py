"""
Warm-start helpers for policy-network initialization.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict, Iterable

import numpy as np
import tensorflow as tf

from src.trainers.results import TrainingResult


def extract_policy_model(source: Any) -> tf.keras.Model:
    """
    Resolve a policy model from supported warm-start sources.

    Supported source types:
    - tf.keras.Model (used directly)
    - TrainingResult (artifacts['policy_net'])
    - legacy/new result dict with '_policy_net' or 'policy_net'
    """
    if isinstance(source, tf.keras.Model):
        return source

    if isinstance(source, TrainingResult):
        model = source.artifacts.get("policy_net")
        if isinstance(model, tf.keras.Model):
            return model
        raise ValueError(
            "Warm-start TrainingResult is missing artifacts['policy_net'] "
            "or it is not a tf.keras.Model."
        )

    if isinstance(source, dict):
        for key in ("_policy_net", "policy_net"):
            model = source.get(key)
            if isinstance(model, tf.keras.Model):
                return model
        raise ValueError(
            "Warm-start dict must contain '_policy_net' or 'policy_net' "
            "with a tf.keras.Model value."
        )

    raise TypeError(
        "warm_start_policy must be a tf.keras.Model, TrainingResult, "
        f"or result dict. Got type: {type(source).__name__}."
    )


def describe_policy_source(source: Any) -> str:
    """
    Return a short human-readable source description for metadata/logging.
    """
    if isinstance(source, tf.keras.Model):
        return "model"
    if isinstance(source, TrainingResult):
        return "TrainingResult.artifacts['policy_net']"
    if isinstance(source, dict):
        if isinstance(source.get("_policy_net"), tf.keras.Model):
            return "dict['_policy_net']"
        if isinstance(source.get("policy_net"), tf.keras.Model):
            return "dict['policy_net']"
    return type(source).__name__


def _numeric_equal(a: Any, b: Any, atol: float = 1e-8) -> bool:
    if isinstance(a, Number) and isinstance(b, Number):
        return bool(np.isclose(float(a), float(b), atol=atol, rtol=0.0))
    return a == b


def _iter_config_keys() -> Iterable[str]:
    # Keys that most directly affect policy-network compatibility.
    return (
        "k_min",
        "k_max",
        "logz_min",
        "logz_max",
        "b_min",
        "b_max",
        "n_layers",
        "n_neurons",
        "hidden_activation",
        "policy_head",
        "value_head",
        "policy_k_head",
        "policy_b_head",
        "price_head",
        "basic_policy_head",
        "basic_value_head",
        "risky_policy_k_head",
        "risky_policy_b_head",
        "risky_price_head",
        "observation_normalizer",
        "inference_clips",
    )


def _collect_config_mismatches(
    source_cfg: Dict[str, Any],
    target_cfg: Dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for key in _iter_config_keys():
        has_source = key in source_cfg
        has_target = key in target_cfg
        if has_source and has_target:
            src_val = source_cfg[key]
            tgt_val = target_cfg[key]
            if not _numeric_equal(src_val, tgt_val):
                mismatches.append(
                    f"config mismatch '{key}': source={src_val}, target={tgt_val}"
                )
        elif has_source != has_target:
            src_val = source_cfg.get(key, "<missing>")
            tgt_val = target_cfg.get(key, "<missing>")
            mismatches.append(
                f"config key presence mismatch '{key}': source={src_val}, target={tgt_val}"
            )
    return mismatches


def ensure_policy_compatibility(
    source_policy: tf.keras.Model,
    target_policy: tf.keras.Model,
) -> None:
    """
    Validate compatibility and raise detailed errors on mismatch.
    """
    mismatches: list[str] = []

    if type(source_policy) is not type(target_policy):
        mismatches.append(
            "model class mismatch: "
            f"source={type(source_policy).__name__}, "
            f"target={type(target_policy).__name__}"
        )

    source_cfg = source_policy.get_config() if hasattr(source_policy, "get_config") else {}
    target_cfg = target_policy.get_config() if hasattr(target_policy, "get_config") else {}
    if isinstance(source_cfg, dict) and isinstance(target_cfg, dict):
        mismatches.extend(_collect_config_mismatches(source_cfg, target_cfg))

    source_weights = source_policy.get_weights()
    target_weights = target_policy.get_weights()

    if len(source_weights) != len(target_weights):
        mismatches.append(
            "weight tensor count mismatch: "
            f"source={len(source_weights)}, target={len(target_weights)}"
        )
    else:
        for idx, (src_w, tgt_w) in enumerate(zip(source_weights, target_weights)):
            if src_w.shape != tgt_w.shape:
                mismatches.append(
                    f"weight shape mismatch at index {idx}: "
                    f"source={src_w.shape}, target={tgt_w.shape}"
                )

    if mismatches:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(
            "Warm-start policy is incompatible with target policy network.\n"
            f"{details}\n"
            "Please match the network architecture and bounds."
        )


def copy_policy_weights(
    warm_start_policy: Any,
    target_policy: tf.keras.Model,
) -> tf.keras.Model:
    """
    Copy policy weights from warm_start_policy source into target policy.

    Returns:
        The resolved source policy model.
    """
    source_policy = extract_policy_model(warm_start_policy)
    ensure_policy_compatibility(source_policy, target_policy)
    target_policy.set_weights(source_policy.get_weights())
    return source_policy
