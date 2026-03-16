"""
Normalization helpers for Bellman-residual trainers.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.value_scale import compute_frictionless_value_benchmark


@dataclass(frozen=True)
class BRNormalizationInfo:
    """
    Resolved normalization info for BR losses.

    Attributes:
        mode: One of {"none", "frictionless", "custom"}.
        scale: Positive scalar used to normalize BR residuals.
        source: Human-readable source description.
    """

    mode: str
    scale: float
    source: str


def resolve_br_normalization_scale(
    *,
    mode: str,
    custom_value: float | None,
    epsilon: float,
    params: EconomicParams,
    shock_params: ShockParams,
) -> BRNormalizationInfo:
    """
    Resolve BR normalization scale before trainer construction.
    """
    if epsilon <= 0:
        raise ValueError(f"Normalization epsilon must be > 0, got {epsilon}")

    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode == "none":
        return BRNormalizationInfo(mode="none", scale=1.0, source="disabled")

    if normalized_mode == "custom":
        if custom_value is None:
            raise ValueError(
                "br_normalization='custom' requires br_normalizer_value."
            )
        scale = max(abs(float(custom_value)), epsilon)
        return BRNormalizationInfo(
            mode="custom",
            scale=float(scale),
            source=f"custom({float(custom_value):.6g})",
        )

    if normalized_mode == "frictionless":
        benchmark = compute_frictionless_value_benchmark(
            params=params,
            shock_params=shock_params,
            epsilon=epsilon,
        )
        return BRNormalizationInfo(
            mode="frictionless",
            scale=benchmark.scale,
            source=f"frictionless(V0*={benchmark.value_star:.6g})",
        )

    raise ValueError(
        "br_normalization must be one of {'none', 'frictionless', 'custom'}, "
        f"got '{mode}'."
    )
