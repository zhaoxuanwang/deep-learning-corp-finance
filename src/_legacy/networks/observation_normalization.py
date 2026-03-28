"""
Observation normalization utilities used at network input boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


SUPPORTED_NORMALIZATION_SCHEMES = {"none", "minmax", "zscore", "log", "log_zscore"}
SUPPORTED_Z_INPUT_SPACES = {"level", "log"}


def _as_tensor_1d(value: Any) -> tf.Tensor:
    tensor = value if isinstance(value, tf.Tensor) else tf.convert_to_tensor(value, dtype=tf.float32)
    return tf.reshape(tf.cast(tensor, tf.float32), [-1])


def _extract_feature(dataset: Dict[str, tf.Tensor], feature: str) -> Optional[tf.Tensor]:
    """
    Extract a flattened feature vector from common dataset schemas.
    """
    if feature == "k":
        if "k" in dataset:
            return _as_tensor_1d(dataset["k"])
        if "k0" in dataset:
            return _as_tensor_1d(dataset["k0"])
        return None

    if feature == "z":
        if "z" in dataset:
            return _as_tensor_1d(dataset["z"])
        if "z0" in dataset:
            return _as_tensor_1d(dataset["z0"])
        if "z_path" in dataset:
            return _as_tensor_1d(dataset["z_path"])
        return None

    if feature == "b":
        if "b" in dataset:
            return _as_tensor_1d(dataset["b"])
        if "b0" in dataset:
            return _as_tensor_1d(dataset["b0"])
        return None

    return None


def _base_transform(
    x: tf.Tensor,
    *,
    feature: str,
    scheme: str,
    z_input_space: str,
    epsilon: float,
) -> tf.Tensor:
    """
    Apply feature-space transform before affine normalization.
    """
    eps = tf.constant(epsilon, dtype=tf.float32)

    if feature == "z":
        if z_input_space == "level":
            # z is passed in levels by dataset generation, so normalize in log-space.
            z_base = tf.math.log(tf.maximum(x, eps))
        elif z_input_space == "log":
            z_base = x
        else:
            raise ValueError(f"Unsupported z_input_space='{z_input_space}'")
        return z_base

    if scheme in {"log", "log_zscore"}:
        return tf.math.log(tf.maximum(x, eps))

    return x


def _fit_feature_stats(
    *,
    feature: str,
    scheme: str,
    values: Optional[tf.Tensor],
    bounds: Dict[str, Any],
    z_input_space: str,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Fit normalization stats for a single feature.
    """
    payload: Dict[str, Any] = {"scheme": scheme}
    if scheme in {"none", "log"}:
        return payload

    if scheme == "minmax":
        if feature == "k":
            f_min, f_max = bounds["k"]
        elif feature == "b":
            f_min, f_max = bounds["b"]
        elif feature == "z":
            # z normalization is performed in log-space by design.
            f_min, f_max = bounds["log_z"]
        else:
            raise ValueError(f"Unsupported feature='{feature}'")

        span = max(float(f_max) - float(f_min), float(epsilon))
        payload.update({"min": float(f_min), "max": float(f_max), "span": float(span)})
        return payload

    if scheme in {"zscore", "log_zscore"}:
        if values is None:
            raise ValueError(
                f"Cannot fit zscore stats for feature '{feature}': missing data in dataset."
            )
        base = _base_transform(
            tf.cast(values, tf.float32),
            feature=feature,
            scheme=scheme,
            z_input_space=z_input_space,
            epsilon=epsilon,
        )
        mean = tf.reduce_mean(base)
        std = tf.math.reduce_std(base)
        std_safe = tf.maximum(std, tf.constant(epsilon, dtype=tf.float32))
        payload.update({"mean": float(mean.numpy()), "std": float(std_safe.numpy())})
        return payload

    raise ValueError(f"Unsupported normalization scheme '{scheme}'")


def build_observation_normalizer(
    *,
    dataset: Dict[str, tf.Tensor],
    bounds: Dict[str, Any],
    k_scheme: str,
    z_scheme: str,
    b_scheme: str,
    z_input_space: str,
    epsilon: float,
    include_debt: bool,
) -> Dict[str, Any]:
    """
    Fit and return a serializable observation-normalizer payload.
    """
    for scheme in (k_scheme, z_scheme, b_scheme):
        if scheme not in SUPPORTED_NORMALIZATION_SCHEMES:
            raise ValueError(
                f"Unsupported normalization scheme '{scheme}'. "
                f"Supported: {sorted(SUPPORTED_NORMALIZATION_SCHEMES)}"
            )
    if z_input_space not in SUPPORTED_Z_INPUT_SPACES:
        raise ValueError(
            f"Unsupported z_input_space '{z_input_space}'. "
            f"Supported: {sorted(SUPPORTED_Z_INPUT_SPACES)}"
        )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    values_k = _extract_feature(dataset, "k")
    values_z = _extract_feature(dataset, "z")
    values_b = _extract_feature(dataset, "b") if include_debt else None

    features = {
        "k": _fit_feature_stats(
            feature="k",
            scheme=k_scheme,
            values=values_k,
            bounds=bounds,
            z_input_space=z_input_space,
            epsilon=epsilon,
        ),
        "z": _fit_feature_stats(
            feature="z",
            scheme=z_scheme,
            values=values_z,
            bounds=bounds,
            z_input_space=z_input_space,
            epsilon=epsilon,
        ),
        "b": (
            _fit_feature_stats(
                feature="b",
                scheme=b_scheme,
                values=values_b,
                bounds=bounds,
                z_input_space=z_input_space,
                epsilon=epsilon,
            )
            if include_debt
            else {"scheme": "none"}
        ),
    }

    return {
        "epsilon": float(epsilon),
        "z_input_space": z_input_space,
        "features": features,
    }


def _apply_feature_normalization(
    x: tf.Tensor,
    *,
    feature: str,
    normalizer: Dict[str, Any],
) -> tf.Tensor:
    feature_cfg = normalizer["features"][feature]
    scheme = feature_cfg["scheme"]
    epsilon = float(normalizer["epsilon"])
    z_input_space = str(normalizer["z_input_space"])

    base = _base_transform(
        x,
        feature=feature,
        scheme=scheme,
        z_input_space=z_input_space,
        epsilon=epsilon,
    )

    if scheme in {"none", "log"}:
        return base

    if scheme == "minmax":
        f_min = tf.constant(float(feature_cfg["min"]), dtype=tf.float32)
        span = tf.constant(float(feature_cfg["span"]), dtype=tf.float32)
        return (base - f_min) / tf.maximum(span, tf.constant(epsilon, dtype=tf.float32))

    if scheme in {"zscore", "log_zscore"}:
        mean = tf.constant(float(feature_cfg["mean"]), dtype=tf.float32)
        std = tf.constant(float(feature_cfg["std"]), dtype=tf.float32)
        return (base - mean) / tf.maximum(std, tf.constant(epsilon, dtype=tf.float32))

    raise ValueError(f"Unsupported normalization scheme '{scheme}'")


def transform_basic_observations(
    *,
    k: tf.Tensor,
    z: tf.Tensor,
    normalizer: Dict[str, Any],
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Normalize (k, z) observations for basic-model networks.
    """
    return (
        _apply_feature_normalization(k, feature="k", normalizer=normalizer),
        _apply_feature_normalization(z, feature="z", normalizer=normalizer),
    )


def transform_risky_observations(
    *,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    normalizer: Dict[str, Any],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Normalize (k, b, z) observations for risky-model networks.
    """
    return (
        _apply_feature_normalization(k, feature="k", normalizer=normalizer),
        _apply_feature_normalization(b, feature="b", normalizer=normalizer),
        _apply_feature_normalization(z, feature="z", normalizer=normalizer),
    )
