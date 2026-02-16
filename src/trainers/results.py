"""
Structured training-result object with legacy-dict compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainingResult:
    """
    Canonical result container for trainer APIs.

    Attributes:
        history: Training/eval metrics and metadata.
        artifacts: Runtime objects (e.g., policy/value/price networks).
                   Use plain names like 'policy_net', 'value_net', etc.
        config: Config bundle used for this run.
        params: Economic parameters used for this run.
        meta: Optional metadata (e.g., model/method ids).
    """

    history: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    params: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dict schema expected by notebooks/utilities.
        """
        out: Dict[str, Any] = {"history": self.history}

        for key, value in self.artifacts.items():
            legacy_key = key if key.startswith("_") else f"_{key}"
            out[legacy_key] = value

        if self.config:
            out["_configs"] = self.config
        if self.params is not None:
            out["_params"] = self.params
        if self.meta:
            out["_meta"] = self.meta

        return out

    @classmethod
    def from_legacy_dict(cls, payload: Dict[str, Any]) -> "TrainingResult":
        """
        Build TrainingResult from the existing legacy dict schema.
        """
        artifacts: Dict[str, Any] = {}
        for key, value in payload.items():
            if key.startswith("_") and key not in {"_configs", "_params", "_meta"}:
                artifacts[key.lstrip("_")] = value

        return cls(
            history=payload.get("history", {}),
            artifacts=artifacts,
            config=payload.get("_configs", {}),
            params=payload.get("_params"),
            meta=payload.get("_meta", {}),
        )

