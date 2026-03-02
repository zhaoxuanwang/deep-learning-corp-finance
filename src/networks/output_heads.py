"""
Output-head registry and safety guards for policy/value/price networks.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


SUPPORTED_OUTPUT_HEADS = {"bounded_sigmoid", "linear", "affine_exp"}

# Heads that are structurally valid for each output channel.
ALLOWED_OUTPUT_HEADS = {
    "basic_policy_k": {"bounded_sigmoid", "linear", "affine_exp"},
    "basic_value": {"linear"},
    "risky_policy_k": {"bounded_sigmoid", "linear", "affine_exp"},
    "risky_policy_b": {"bounded_sigmoid", "linear"},
    "risky_value": {"linear"},
    "risky_price_q": {"bounded_sigmoid", "linear"},
}


def validate_output_head(
    *,
    output_name: str,
    head_name: str,
) -> None:
    """
    Validate output-head choice for a specific output channel.
    """
    if head_name not in SUPPORTED_OUTPUT_HEADS:
        raise ValueError(
            f"Unsupported head '{head_name}' for {output_name}. "
            f"Supported: {sorted(SUPPORTED_OUTPUT_HEADS)}"
        )

    allowed_heads = ALLOWED_OUTPUT_HEADS.get(output_name)
    if allowed_heads is None:
        raise ValueError(f"Unknown output_name='{output_name}'")
    if head_name not in allowed_heads:
        raise ValueError(
            f"Head '{head_name}' is not allowed for {output_name}. "
            f"Allowed: {sorted(allowed_heads)}"
        )


def apply_output_head(
    raw: tf.Tensor,
    *,
    head_name: str,
    lower: Optional[tf.Tensor] = None,
    upper: Optional[tf.Tensor] = None,
    affine_mu: Optional[tf.Tensor] = None,
    affine_std: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Transform raw network output into economic-level output.
    """
    if head_name == "linear":
        return raw

    eps = tf.constant(1e-8, dtype=raw.dtype)

    if head_name == "bounded_sigmoid":
        s = tf.nn.sigmoid(raw)
        if lower is None and upper is None:
            return s
        if lower is None or upper is None:
            raise ValueError(
                "bounded_sigmoid requires both lower and upper when using bounded scaling."
        )
        return lower + (upper - lower) * s

    if head_name == "affine_exp":
        if affine_mu is None or affine_std is None:
            raise ValueError("affine_exp requires affine_mu and affine_std.")
        # Clip exponent argument for numeric stability in float32 training.
        exp_arg = tf.clip_by_value(affine_mu + affine_std * raw, -20.0, 20.0)
        return tf.exp(exp_arg)

    raise ValueError(f"Unsupported head '{head_name}'")
