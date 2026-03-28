"""
Method specification registry for training entrypoints.

This makes adding new trainer methods explicit and non-invasive:
- declare required dataset keys
- declare runtime batch ordering mode
- register once
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class MethodSpec:
    name: str
    required_keys: FrozenSet[str]
    batch_order: str = "as_is"


_METHOD_SPECS: Dict[str, MethodSpec] = {}


def register_method_spec(spec: MethodSpec) -> None:
    _METHOD_SPECS[spec.name] = spec


def get_method_spec(name: str) -> MethodSpec:
    if name not in _METHOD_SPECS:
        raise KeyError(
            f"Method '{name}' is not registered. "
            "Register a MethodSpec in src/trainers/method_specs.py."
        )
    return _METHOD_SPECS[name]


def list_method_specs() -> Dict[str, MethodSpec]:
    return dict(_METHOD_SPECS)


# Built-in methods
register_method_spec(MethodSpec(
    name="basic_lr",
    required_keys=frozenset({"k0", "z_path"}),
    batch_order="as_is",
))
register_method_spec(MethodSpec(
    name="basic_er",
    required_keys=frozenset({"k", "z", "z_next_main", "z_next_fork"}),
    batch_order="as_is",
))
register_method_spec(MethodSpec(
    name="basic_br_actor_critic",
    required_keys=frozenset({"k", "z", "z_next_main", "z_next_fork"}),
    batch_order="as_is",
))
register_method_spec(MethodSpec(
    name="risky_br_actor_critic",
    required_keys=frozenset({"k", "b", "z", "z_next_main", "z_next_fork"}),
    batch_order="as_is",
))

# Backward-compatible aliases (legacy method ids)
register_method_spec(MethodSpec(
    name="basic_br",
    required_keys=frozenset({"k", "z", "z_next_main", "z_next_fork"}),
    batch_order="as_is",
))
register_method_spec(MethodSpec(
    name="risky_br",
    required_keys=frozenset({"k", "b", "z", "z_next_main", "z_next_fork"}),
    batch_order="as_is",
))
