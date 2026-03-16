"""Lightweight checkpointing for v2 training runs.

Saves everything needed to reload a completed training run:
  - Trained model weights (.weights.h5)
  - Training config (JSON)
  - Training history (npz)
  - Run metadata (JSON)

Directory layout for a saved run::

    <run_dir>/
        config.json          # serialized TrainingConfig
        metadata.json        # timestamp, method, wall_time, etc.
        history.npz          # loss curves and eval metrics per step
        weights/
            policy.weights.h5
            value_net.weights.h5   # if present

Usage::

    from src.v2.utils.checkpointing import TrainingResult, save_run, load_run

    # After training:
    result = TrainingResult.from_trainer_output(output_dict, method="lr")
    save_run(result, "outputs/runs/basic_investment/lr_20260314_153000")

    # Later analysis (no retraining):
    result = load_run("outputs/runs/basic_investment/lr_20260314_153000")
    print(result.history["loss"][-1])

    # To restore weights into live models:
    result.load_weights(policy=policy_net)
    result.load_weights(policy=policy_net, value_net=value_net)
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclass
class TrainingResult:
    """Container for a completed training run.

    This is a data-only object that can be saved/loaded without
    instantiating TensorFlow models.  Call load_weights() to restore
    saved weights into live Keras models.

    Attributes:
        method:     Method name ("lr", "er", "brm", "shac").
        config:     Serialized TrainingConfig dict.
        history:    Training history dict (lists of floats per metric).
        metadata:   Run metadata (timestamps, wall time, etc.).
        run_dir:    Path where this result is saved (set by save_run).
    """
    method:   str
    config:   dict
    history:  dict
    metadata: dict  = field(default_factory=dict)
    run_dir:  Optional[str] = None

    @classmethod
    def from_trainer_output(cls, output: dict, method: str,
                            wall_time: float = None) -> "TrainingResult":
        """Build a TrainingResult from a trainer's return dict.

        Args:
            output:    Dict returned by train_lr / train_er / train_brm / train_shac.
            method:    Method name string.
            wall_time: Optional wall-clock seconds for the training run.
        """
        config_obj = output["config"]
        # Serialize dataclass to dict; fall back to str for non-serializable
        try:
            from dataclasses import asdict as _asdict
            config_dict = _asdict(config_obj)
        except TypeError:
            config_dict = {"repr": repr(config_obj)}

        metadata = {
            "method": method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if wall_time is not None:
            metadata["wall_time_seconds"] = round(wall_time, 2)

        return cls(
            method=method,
            config=config_dict,
            history=output["history"],
            metadata=metadata,
        )

    def load_weights(self, policy=None, value_net=None) -> None:
        """Restore saved weights into live Keras model instances.

        Models must already be built (i.e. called at least once) before
        loading weights so that the variable shapes are known.

        Args:
            policy:    PolicyNetwork to load policy weights into.
            value_net: StateValueNetwork to load critic weights into (BRM/SHAC).

        Raises:
            FileNotFoundError: if run_dir is not set or weight files missing.
        """
        if self.run_dir is None:
            raise FileNotFoundError("run_dir is not set; load_run() first.")
        weights_dir = os.path.join(self.run_dir, "weights")
        if policy is not None:
            path = os.path.join(weights_dir, "policy.weights.h5")
            policy.load_weights(path)
        if value_net is not None:
            path = os.path.join(weights_dir, "value_net.weights.h5")
            value_net.load_weights(path)


def save_run(result: TrainingResult, run_dir: str,
             policy=None, value_net=None) -> str:
    """Save a TrainingResult to disk.

    Args:
        result:    TrainingResult to save.
        run_dir:   Directory path for this run.
        policy:    Live PolicyNetwork whose weights to save.
        value_net: Live StateValueNetwork whose weights to save (optional).

    Returns:
        The run_dir path (for convenience).
    """
    os.makedirs(run_dir, exist_ok=True)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(result.config, f, indent=2, default=str)

    # Metadata
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(result.metadata, f, indent=2, default=str)

    # History
    history_arrays = {k: np.array(v) for k, v in result.history.items()}
    np.savez(os.path.join(run_dir, "history.npz"), **history_arrays)

    # Weights
    if policy is not None:
        policy.save_weights(os.path.join(weights_dir, "policy.weights.h5"))
    if value_net is not None:
        value_net.save_weights(
            os.path.join(weights_dir, "value_net.weights.h5"))

    result.run_dir = run_dir
    return run_dir


def load_run(run_dir: str) -> TrainingResult:
    """Load a TrainingResult from disk (weights loaded lazily via load_weights).

    Args:
        run_dir: Path to a saved run directory.

    Returns:
        TrainingResult with config, history, metadata populated.
        Call result.load_weights(policy=...) to restore model weights.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    with open(os.path.join(run_dir, "metadata.json")) as f:
        metadata = json.load(f)

    history_data = np.load(os.path.join(run_dir, "history.npz"))
    history = {k: history_data[k].tolist() for k in history_data.files}

    return TrainingResult(
        method=metadata.get("method", "unknown"),
        config=config,
        history=history,
        metadata=metadata,
        run_dir=run_dir,
    )
