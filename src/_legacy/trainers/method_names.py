"""
Method-name normalization and validation helpers for trainer APIs.
"""

from __future__ import annotations

from typing import Optional


_ALIASES = {
    # Basic model aliases
    "basic_lifetime_reward": "basic_lr",
    "basic_euler_residual": "basic_er",
    "basic_bellman_residual": "basic_br_actor_critic",
    "basic_br": "basic_br_actor_critic",
    "basic_br_actor": "basic_br_actor_critic",
    "basic_br_actor_critic": "basic_br_actor_critic",
    # Risky model aliases
    "risky_bellman_residual": "risky_br_actor_critic",
    "risky_debt_bellman_residual": "risky_br_actor_critic",
    "risky_br": "risky_br_actor_critic",
    "risky_br_actor_critic": "risky_br_actor_critic",
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def canonicalize_method_name(name: str, *, model: Optional[str] = None) -> str:
    """
    Map user-facing aliases to canonical registry names.

    Canonical method names are:
    - basic_lr
    - basic_er
    - basic_br_actor_critic
    - risky_br_actor_critic
    """
    normalized = _normalize(name)
    if normalized in {
        "br_constrained",
        "basic_br_constrained",
        "basic_br_lagrangian",
    }:
        raise ValueError(
            "Method 'br_constrained' is experimental and disabled in production APIs. "
            "Use src.experimental.br_constrained for research-only experiments."
        )
    if normalized in {
        "br_multitask",
        "br_reg",
        "basic_br_multitask",
        "basic_br_reg",
        "basic_br_regression",
        "basic_bellman_residual_regression",
    }:
        raise ValueError(
            "Method 'br_multitask' has been moved to src/experimental/br_multitask.py "
            "due to structural identification failure. "
            "See report/br_multitask_structural_issues.md for details."
        )
    normalized = _ALIASES.get(normalized, normalized)

    # Model-specific shorthand aliases.
    if normalized in {"lr", "er", "br"} and model is not None:
        model_norm = _normalize(model)
        if model_norm == "basic":
            if normalized == "br":
                return "basic_br_actor_critic"
            return f"basic_{normalized}"
        if model_norm in {"risky", "risky_debt"}:
            if normalized != "br":
                raise ValueError(
                    f"Model '{model}' only supports method 'br'. Got '{name}'."
                )
            return "risky_br_actor_critic"

    return normalized


def validate_method_config_name(
    configured_name: str,
    *,
    expected_name: str,
    model: Optional[str] = None,
) -> str:
    """
    Validate method config coherence and return canonical configured name.
    """
    configured_canonical = canonicalize_method_name(configured_name, model=model)
    expected_canonical = canonicalize_method_name(expected_name, model=model)

    if configured_canonical != expected_canonical:
        raise ValueError(
            f"MethodConfig.name='{configured_name}' does not match requested "
            f"method '{expected_name}' (canonical='{expected_canonical}')."
        )

    return configured_canonical
