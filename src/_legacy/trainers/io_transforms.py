"""
Input/output transform utilities shared by trainers and inference callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from src.networks.observation_normalization import (
    transform_basic_observations,
    transform_risky_observations,
)
from src.networks.output_heads import apply_output_head, validate_output_head


ClipValue = Optional[Union[float, str]]


def _resolve_clip_value(
    value: ClipValue,
    *,
    k_bounds: Tuple[float, float],
    b_bounds: Optional[Tuple[float, float]],
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "k_max":
            return float(k_bounds[1])
        if value == "b_max":
            if b_bounds is None:
                raise ValueError("clip token 'b_max' requires b_bounds.")
            return float(b_bounds[1])
        raise ValueError(f"Unknown clip token '{value}'.")
    return float(value)


def _build_head_transform(
    *,
    output_name: str,
    head_name: str,
    lower: Optional[float],
    upper: Optional[float],
    normalizer: Dict[str, Any],
    affine_feature: Optional[str],
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> Dict[str, Any]:
    validate_output_head(output_name=output_name, head_name=head_name)

    payload: Dict[str, Any] = {
        "head": head_name,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }

    if head_name == "bounded_sigmoid":
        if lower is None or upper is None:
            raise ValueError(
                f"{output_name} with bounded_sigmoid requires lower and upper bounds."
            )
        payload["lower"] = float(lower)
        payload["upper"] = float(upper)
        return payload

    if head_name == "affine_exp":
        if affine_feature is None:
            raise ValueError(f"{output_name} affine_exp requires affine_feature.")
        feat_cfg = normalizer["features"][affine_feature]
        scheme = str(feat_cfg["scheme"])
        if scheme not in {"zscore", "log_zscore"}:
            raise ValueError(
                f"{output_name} affine_exp requires zscore/log_zscore stats on "
                f"feature '{affine_feature}', got scheme '{scheme}'."
            )
        payload["affine_mu"] = float(feat_cfg["mean"])
        payload["affine_std"] = float(feat_cfg["std"])
        return payload

    # linear head
    return payload


def build_basic_transform_spec(
    *,
    normalizer: Dict[str, Any],
    k_bounds: Tuple[float, float],
    policy_head: str,
    value_head: str,
    clip_policy_k_min: ClipValue,
    clip_policy_k_max: ClipValue,
    clip_value_min: ClipValue,
    clip_value_max: ClipValue,
) -> Dict[str, Any]:
    policy_k = _build_head_transform(
        output_name="basic_policy_k",
        head_name=policy_head,
        lower=float(k_bounds[0]),
        upper=float(k_bounds[1]),
        normalizer=normalizer,
        affine_feature="k",
        clip_min=_resolve_clip_value(
            clip_policy_k_min,
            k_bounds=k_bounds,
            b_bounds=None,
        ),
        clip_max=_resolve_clip_value(
            clip_policy_k_max,
            k_bounds=k_bounds,
            b_bounds=None,
        ),
    )
    value = _build_head_transform(
        output_name="basic_value",
        head_name=value_head,
        lower=None,
        upper=None,
        normalizer=normalizer,
        affine_feature=None,
        clip_min=_resolve_clip_value(
            clip_value_min,
            k_bounds=k_bounds,
            b_bounds=None,
        ),
        clip_max=_resolve_clip_value(
            clip_value_max,
            k_bounds=k_bounds,
            b_bounds=None,
        ),
    )
    return {
        "model": "basic",
        "observation_normalizer": normalizer,
        "outputs": {
            "policy_k": policy_k,
            "value": value,
        },
    }


def build_risky_transform_spec(
    *,
    normalizer: Dict[str, Any],
    k_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float],
    r_risk_free: float,
    policy_k_head: str,
    policy_b_head: str,
    value_head: str,
    price_head: str,
    clip_policy_k_min: ClipValue,
    clip_policy_k_max: ClipValue,
    clip_policy_b_min: ClipValue,
    clip_policy_b_max: ClipValue,
    clip_value_min: ClipValue,
    clip_value_max: ClipValue,
    clip_price_min: ClipValue,
    clip_price_max: ClipValue,
) -> Dict[str, Any]:
    q_max = float(1.0 / (1.0 + r_risk_free))

    policy_k = _build_head_transform(
        output_name="risky_policy_k",
        head_name=policy_k_head,
        lower=float(k_bounds[0]),
        upper=float(k_bounds[1]),
        normalizer=normalizer,
        affine_feature="k",
        clip_min=_resolve_clip_value(
            clip_policy_k_min,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
        clip_max=_resolve_clip_value(
            clip_policy_k_max,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
    )
    policy_b = _build_head_transform(
        output_name="risky_policy_b",
        head_name=policy_b_head,
        lower=float(b_bounds[0]),
        upper=float(b_bounds[1]),
        normalizer=normalizer,
        affine_feature="b",
        clip_min=_resolve_clip_value(
            clip_policy_b_min,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
        clip_max=_resolve_clip_value(
            clip_policy_b_max,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
    )
    value = _build_head_transform(
        output_name="risky_value",
        head_name=value_head,
        lower=None,
        upper=None,
        normalizer=normalizer,
        affine_feature=None,
        clip_min=_resolve_clip_value(
            clip_value_min,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
        clip_max=_resolve_clip_value(
            clip_value_max,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
    )
    price_q = _build_head_transform(
        output_name="risky_price_q",
        head_name=price_head,
        lower=0.0,
        upper=q_max,
        normalizer=normalizer,
        affine_feature=None,
        clip_min=_resolve_clip_value(
            clip_price_min,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
        clip_max=_resolve_clip_value(
            clip_price_max,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
        ),
    )

    return {
        "model": "risky",
        "observation_normalizer": normalizer,
        "outputs": {
            "policy_k": policy_k,
            "policy_b": policy_b,
            "value": value,
            "price_q": price_q,
        },
    }


def _apply_output_transform(
    raw: tf.Tensor,
    *,
    output_cfg: Dict[str, Any],
    apply_output_clips: bool,
    return_preclip: bool,
):
    kwargs: Dict[str, Any] = {"head_name": output_cfg["head"]}
    if "lower" in output_cfg:
        kwargs["lower"] = tf.constant(float(output_cfg["lower"]), dtype=raw.dtype)
    if "upper" in output_cfg:
        kwargs["upper"] = tf.constant(float(output_cfg["upper"]), dtype=raw.dtype)
    if "affine_mu" in output_cfg:
        kwargs["affine_mu"] = tf.constant(float(output_cfg["affine_mu"]), dtype=raw.dtype)
    if "affine_std" in output_cfg:
        kwargs["affine_std"] = tf.constant(float(output_cfg["affine_std"]), dtype=raw.dtype)

    preclip = apply_output_head(raw, **kwargs)
    out = preclip

    if apply_output_clips:
        clip_min = output_cfg.get("clip_min")
        clip_max = output_cfg.get("clip_max")
        if clip_min is not None:
            out = tf.maximum(out, tf.constant(float(clip_min), dtype=raw.dtype))
        if clip_max is not None:
            out = tf.minimum(out, tf.constant(float(clip_max), dtype=raw.dtype))

    if return_preclip:
        return out, preclip
    return out


def forward_basic_policy_levels(
    *,
    policy_net: tf.keras.Model,
    k: tf.Tensor,
    z: tf.Tensor,
    transform_spec: Dict[str, Any],
    training: bool,
    apply_output_clips: bool,
    return_preclip: bool = False,
):
    k = tf.reshape(tf.cast(k, tf.float32), [-1, 1])
    z = tf.reshape(tf.cast(z, tf.float32), [-1, 1])
    k_norm, z_norm = transform_basic_observations(
        k=k,
        z=z,
        normalizer=transform_spec["observation_normalizer"],
    )
    x = tf.concat([k_norm, z_norm], axis=1)
    raw = policy_net(x, training=training)
    return _apply_output_transform(
        raw,
        output_cfg=transform_spec["outputs"]["policy_k"],
        apply_output_clips=apply_output_clips,
        return_preclip=return_preclip,
    )


def forward_basic_value_levels(
    *,
    value_net: tf.keras.Model,
    k: tf.Tensor,
    z: tf.Tensor,
    transform_spec: Dict[str, Any],
    training: bool,
    apply_output_clips: bool,
):
    k = tf.reshape(tf.cast(k, tf.float32), [-1, 1])
    z = tf.reshape(tf.cast(z, tf.float32), [-1, 1])
    k_norm, z_norm = transform_basic_observations(
        k=k,
        z=z,
        normalizer=transform_spec["observation_normalizer"],
    )
    x = tf.concat([k_norm, z_norm], axis=1)
    raw = value_net(x, training=training)
    return _apply_output_transform(
        raw,
        output_cfg=transform_spec["outputs"]["value"],
        apply_output_clips=apply_output_clips,
        return_preclip=False,
    )


def forward_risky_policy_levels(
    *,
    policy_net: tf.keras.Model,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    transform_spec: Dict[str, Any],
    training: bool,
    apply_output_clips: bool,
    return_preclip: bool = False,
):
    k = tf.reshape(tf.cast(k, tf.float32), [-1, 1])
    b = tf.reshape(tf.cast(b, tf.float32), [-1, 1])
    z = tf.reshape(tf.cast(z, tf.float32), [-1, 1])
    k_norm, b_norm, z_norm = transform_risky_observations(
        k=k,
        b=b,
        z=z,
        normalizer=transform_spec["observation_normalizer"],
    )
    x = tf.concat([k_norm, b_norm, z_norm], axis=1)
    raw_k, raw_b = policy_net(x, training=training)

    k_next = _apply_output_transform(
        raw_k,
        output_cfg=transform_spec["outputs"]["policy_k"],
        apply_output_clips=apply_output_clips,
        return_preclip=return_preclip,
    )
    b_next = _apply_output_transform(
        raw_b,
        output_cfg=transform_spec["outputs"]["policy_b"],
        apply_output_clips=apply_output_clips,
        return_preclip=return_preclip,
    )

    if return_preclip:
        k_post, k_pre = k_next
        b_post, b_pre = b_next
        return (k_post, b_post), (k_pre, b_pre)

    return k_next, b_next


def forward_risky_value_levels(
    *,
    value_net: tf.keras.Model,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    transform_spec: Dict[str, Any],
    training: bool,
    apply_output_clips: bool,
):
    k = tf.reshape(tf.cast(k, tf.float32), [-1, 1])
    b = tf.reshape(tf.cast(b, tf.float32), [-1, 1])
    z = tf.reshape(tf.cast(z, tf.float32), [-1, 1])
    k_norm, b_norm, z_norm = transform_risky_observations(
        k=k,
        b=b,
        z=z,
        normalizer=transform_spec["observation_normalizer"],
    )
    x = tf.concat([k_norm, b_norm, z_norm], axis=1)
    raw = value_net(x, training=training)
    return _apply_output_transform(
        raw,
        output_cfg=transform_spec["outputs"]["value"],
        apply_output_clips=apply_output_clips,
        return_preclip=False,
    )


def forward_risky_price_levels(
    *,
    price_net: tf.keras.Model,
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    z: tf.Tensor,
    transform_spec: Dict[str, Any],
    training: bool,
    apply_output_clips: bool,
):
    k_next = tf.reshape(tf.cast(k_next, tf.float32), [-1, 1])
    b_next = tf.reshape(tf.cast(b_next, tf.float32), [-1, 1])
    z = tf.reshape(tf.cast(z, tf.float32), [-1, 1])
    k_norm, b_norm, z_norm = transform_risky_observations(
        k=k_next,
        b=b_next,
        z=z,
        normalizer=transform_spec["observation_normalizer"],
    )
    x = tf.concat([k_norm, b_norm, z_norm], axis=1)
    raw = price_net(x, training=training)
    return _apply_output_transform(
        raw,
        output_cfg=transform_spec["outputs"]["price_q"],
        apply_output_clips=apply_output_clips,
        return_preclip=False,
    )


def compute_k_clip_diagnostics(
    *,
    k_postclip: tf.Tensor,
    k_preclip: tf.Tensor,
) -> Dict[str, float]:
    k_post = tf.reshape(tf.cast(k_postclip, tf.float32), [-1])
    k_pre = tf.reshape(tf.cast(k_preclip, tf.float32), [-1])
    clipped = tf.abs(k_post - k_pre) > tf.constant(1e-10, dtype=tf.float32)
    return {
        "clip_fraction_k": float(tf.reduce_mean(tf.cast(clipped, tf.float32))),
        "preclip_max_k": float(tf.reduce_max(k_pre)),
    }


def build_legacy_basic_transform_spec_from_networks(
    *,
    policy_net: tf.keras.Model,
    value_net: Optional[tf.keras.Model] = None,
) -> Dict[str, Any]:
    """
    Best-effort fallback transform spec for direct-trainer legacy call sites.
    """
    policy_cfg = policy_net.get_config() if hasattr(policy_net, "get_config") else {}
    value_cfg = value_net.get_config() if (value_net is not None and hasattr(value_net, "get_config")) else {}

    k_min = float(policy_cfg.get("k_min", 0.0))
    k_max = float(policy_cfg.get("k_max", 1.0))
    logz_min = float(policy_cfg.get("logz_min", -1.0))
    logz_max = float(policy_cfg.get("logz_max", 1.0))

    normalizer = policy_cfg.get("observation_normalizer")
    if not isinstance(normalizer, dict):
        eps = 1e-8
        normalizer = {
            "epsilon": eps,
            "z_input_space": "level",
            "features": {
                "k": {
                    "scheme": "minmax",
                    "min": k_min,
                    "max": k_max,
                    "span": max(k_max - k_min, eps),
                },
                "z": {
                    "scheme": "minmax",
                    "min": logz_min,
                    "max": logz_max,
                    "span": max(logz_max - logz_min, eps),
                },
                "b": {"scheme": "none"},
            },
        }

    policy_head = str(
        policy_cfg.get("policy_head", policy_cfg.get("basic_policy_head", "bounded_sigmoid"))
    )
    value_head = str(value_cfg.get("value_head", value_cfg.get("basic_value_head", "linear")))

    return build_basic_transform_spec(
        normalizer=normalizer,
        k_bounds=(k_min, k_max),
        policy_head=policy_head,
        value_head=value_head,
        clip_policy_k_min=None,
        clip_policy_k_max=None,
        clip_value_min=None,
        clip_value_max=None,
    )


def build_legacy_risky_transform_spec_from_networks(
    *,
    policy_net: tf.keras.Model,
    value_net: Optional[tf.keras.Model] = None,
    price_net: Optional[tf.keras.Model] = None,
    r_risk_free: float = 0.04,
) -> Dict[str, Any]:
    """
    Best-effort fallback transform spec for direct-trainer legacy call sites.
    """
    policy_cfg = policy_net.get_config() if hasattr(policy_net, "get_config") else {}
    value_cfg = value_net.get_config() if (value_net is not None and hasattr(value_net, "get_config")) else {}
    price_cfg = price_net.get_config() if (price_net is not None and hasattr(price_net, "get_config")) else {}

    k_min = float(policy_cfg.get("k_min", 0.0))
    k_max = float(policy_cfg.get("k_max", 1.0))
    b_min = float(policy_cfg.get("b_min", 0.0))
    b_max = float(policy_cfg.get("b_max", 1.0))
    logz_min = float(policy_cfg.get("logz_min", -1.0))
    logz_max = float(policy_cfg.get("logz_max", 1.0))
    r_rate = float(policy_cfg.get("r_risk_free", price_cfg.get("r_risk_free", r_risk_free)))

    normalizer = policy_cfg.get("observation_normalizer")
    if not isinstance(normalizer, dict):
        eps = 1e-8
        normalizer = {
            "epsilon": eps,
            "z_input_space": "level",
            "features": {
                "k": {
                    "scheme": "minmax",
                    "min": k_min,
                    "max": k_max,
                    "span": max(k_max - k_min, eps),
                },
                "b": {
                    "scheme": "minmax",
                    "min": b_min,
                    "max": b_max,
                    "span": max(b_max - b_min, eps),
                },
                "z": {
                    "scheme": "minmax",
                    "min": logz_min,
                    "max": logz_max,
                    "span": max(logz_max - logz_min, eps),
                },
            },
        }

    policy_k_head = str(
        policy_cfg.get("policy_k_head", policy_cfg.get("k_head", policy_cfg.get("risky_policy_k_head", "bounded_sigmoid")))
    )
    policy_b_head = str(
        policy_cfg.get("policy_b_head", policy_cfg.get("b_head", policy_cfg.get("risky_policy_b_head", "bounded_sigmoid")))
    )
    value_head = str(value_cfg.get("value_head", value_cfg.get("risky_value_head", "linear")))
    price_head = str(price_cfg.get("price_head", price_cfg.get("risky_price_head", "bounded_sigmoid")))

    return build_risky_transform_spec(
        normalizer=normalizer,
        k_bounds=(k_min, k_max),
        b_bounds=(b_min, b_max),
        r_risk_free=r_rate,
        policy_k_head=policy_k_head,
        policy_b_head=policy_b_head,
        value_head=value_head,
        price_head=price_head,
        clip_policy_k_min=None,
        clip_policy_k_max=None,
        clip_policy_b_min=None,
        clip_policy_b_max=None,
        clip_value_min=None,
        clip_value_max=None,
        clip_price_min=None,
        clip_price_max=None,
    )


@dataclass
class LevelInferenceHelper:
    model: str
    transform_spec: Dict[str, Any]
    policy_net: tf.keras.Model
    value_net: Optional[tf.keras.Model] = None
    price_net: Optional[tf.keras.Model] = None

    def policy(self, *args):
        if self.model == "basic":
            if len(args) != 2:
                raise ValueError("Basic policy inference expects (k, z).")
            k, z = args
            return forward_basic_policy_levels(
                policy_net=self.policy_net,
                k=k,
                z=z,
                transform_spec=self.transform_spec,
                training=False,
                apply_output_clips=True,
            )
        if len(args) != 3:
            raise ValueError("Risky policy inference expects (k, b, z).")
        k, b, z = args
        return forward_risky_policy_levels(
            policy_net=self.policy_net,
            k=k,
            b=b,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=True,
        )

    def value(self, *args):
        if self.value_net is None:
            raise ValueError("value_net is not available in this helper.")
        if self.model == "basic":
            if len(args) != 2:
                raise ValueError("Basic value inference expects (k, z).")
            k, z = args
            return forward_basic_value_levels(
                value_net=self.value_net,
                k=k,
                z=z,
                transform_spec=self.transform_spec,
                training=False,
                apply_output_clips=True,
            )
        if len(args) != 3:
            raise ValueError("Risky value inference expects (k, b, z).")
        k, b, z = args
        return forward_risky_value_levels(
            value_net=self.value_net,
            k=k,
            b=b,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=True,
        )

    def price(self, *args):
        if self.price_net is None:
            raise ValueError("price_net is not available in this helper.")
        if self.model != "risky":
            raise ValueError("Price inference is only available for risky model.")
        if len(args) != 3:
            raise ValueError("Risky price inference expects (k_next, b_next, z).")
        k_next, b_next, z = args
        return forward_risky_price_levels(
            price_net=self.price_net,
            k_next=k_next,
            b_next=b_next,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=True,
        )


def build_inference_helper_from_result(result: Dict[str, Any]) -> LevelInferenceHelper:
    """
    Reconstruct a level-space inference helper from a result payload.
    """
    if not isinstance(result, dict) and hasattr(result, "to_legacy_dict"):
        result = result.to_legacy_dict()

    meta = result.get("_meta", result.get("meta", {}))
    policy_net = result.get("_policy_net", result.get("policy_net"))
    value_net = result.get("_value_net", result.get("value_net"))
    price_net = result.get("_price_net", result.get("price_net"))
    if policy_net is None:
        raise ValueError("Result is missing policy network ('_policy_net').")

    transform_spec = meta.get("io_transforms")
    inferred_model = str(meta.get("model", "")).strip().lower()
    if inferred_model not in {"basic", "risky"}:
        inferred_model = "risky" if price_net is not None else "basic"

    if transform_spec is None:
        # Backward-compatible reconstruction for old checkpoints/results.
        if inferred_model == "risky":
            params = result.get("_params", result.get("params"))
            r_rate = float(getattr(params, "r_rate", 0.04)) if params is not None else 0.04
            transform_spec = build_legacy_risky_transform_spec_from_networks(
                policy_net=policy_net,
                value_net=value_net,
                price_net=price_net,
                r_risk_free=r_rate,
            )
        else:
            transform_spec = build_legacy_basic_transform_spec_from_networks(
                policy_net=policy_net,
                value_net=value_net,
            )

    model = str(transform_spec.get("model", inferred_model)).strip().lower()
    if model not in {"basic", "risky"}:
        raise ValueError(f"Could not infer model type from result payload: {model!r}")

    return LevelInferenceHelper(
        model=model,
        transform_spec=transform_spec,
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
    )
