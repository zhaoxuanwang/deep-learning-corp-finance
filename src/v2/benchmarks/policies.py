"""Policy helpers shared across benchmark notebooks."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


class InterpolatedGridPolicy:
    """Callable policy wrapper built from a regular-grid action interpolator."""

    def __init__(self, action_interp, action_dim: int):
        self.action_interp = action_interp
        self.action_dim = int(action_dim)

    def __call__(self, s, training: bool = False):
        del training
        s_np = s.numpy() if tf.is_tensor(s) else np.asarray(s, dtype=np.float32)
        s_np = np.asarray(s_np, dtype=np.float32)
        if s_np.ndim == 1:
            s_np = s_np[None, :]
        points = np.column_stack([s_np[:, 1], s_np[:, 0]])
        action = self.action_interp(points)
        action = np.asarray(action, dtype=np.float32).reshape(-1, self.action_dim)
        return tf.convert_to_tensor(action, dtype=tf.float32)


def build_action_grid_policy(result: dict, action_dim: int,
                             method: str = "linear") -> InterpolatedGridPolicy:
    """Build an interpolated policy from a solver result dict."""
    exo_1d = result["grids"]["exo_grids_1d"][0]
    endo_1d = result["grids"]["endo_grids_1d"][0]
    action_grid = _to_numpy(result["policy_action"])[:, :, 0]
    action_interp = RegularGridInterpolator(
        (exo_1d, endo_1d),
        action_grid,
        method=method,
        bounds_error=False,
        fill_value=None,
    )
    return InterpolatedGridPolicy(action_interp, action_dim)


def restore_snapshot_weights(model, weights: Sequence[np.ndarray]):
    """Load a saved snapshot weight list into a built Keras model."""
    model.set_weights([np.asarray(w, dtype=np.float32) for w in weights])
    return model


def extract_model_snapshots(checkpoint_history: Iterable[Mapping],
                            model_name: str = "policy"):
    """Project generic checkpoint history onto one named model."""
    projected = []
    for entry in checkpoint_history:
        step = int(entry["step"])
        models = dict(entry.get("models", {}))
        if model_name in models:
            projected.append((step, models[model_name]))
    return projected


def restore_selected_models(models: Mapping[str, object],
                            checkpoint_history: Iterable[Mapping],
                            selected_step: int):
    """Restore one or more named models from a generic checkpoint record."""
    for entry in checkpoint_history:
        if int(entry["step"]) != int(selected_step):
            continue
        available = dict(entry.get("models", {}))
        for model_name, model in models.items():
            if model is None:
                continue
            if model_name not in available:
                raise ValueError(
                    f"Checkpoint step {selected_step} does not contain model "
                    f"{model_name!r}. Available: {sorted(available)}"
                )
            restore_snapshot_weights(model, available[model_name])
        return models
    raise ValueError(f"Selected step {selected_step} not found in checkpoint history.")


def restore_selected_snapshot(model,
                              weight_history: Iterable,
                              selected_step: int):
    """Restore the policy snapshot whose step matches selected_step.

    Accepts either:
      - legacy weight_history: [(step, weights), ...]
      - generic checkpoint_history: [{"step": ..., "models": {...}}, ...]
    """
    history = list(weight_history)
    if not history:
        raise ValueError("No saved snapshots are available.")
    first = history[0]
    if isinstance(first, Mapping):
        restore_selected_models({"policy": model}, history, selected_step)
        return model
    for step, weights in history:
        if int(step) == int(selected_step):
            return restore_snapshot_weights(model, weights)
    raise ValueError(f"Selected step {selected_step} not found in weight history.")
