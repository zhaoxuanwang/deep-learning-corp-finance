"""Scale presets: one knob to move the whole pipeline between test scale and paper scale.

A :class:`Profile` bundles every size that trades accuracy for cost: the state/control
grids, Block-1 training length, panel simulation size, Block-2 collection size, the
surrogate/LM settings, and the Section-6 controller resolution. Three presets:

* ``SMOKE`` -- tiny; the pytest/CI scale (seconds), grids and counts kept minimal.
* ``MEDIUM`` -- CPU-affordable demo scale; runs end-to-end on an M-series Mac in minutes
  and lands the recovery in the right region without paper-grade precision.
* ``FULL`` -- the DF26 paper scale (Table A2): 11x15x35 / 81x91x71 grids, 5000x300
  panels, 10k-row collections, 31-point minimum-loss curves. Built for a CUDA GPU
  (float64 on device); affordable on M1 only as an overnight/rented-GPU run.

Same code path at every scale; only these numbers change. ``configure_devices("auto")``
(see :mod:`src.v3.common.precision`) then puts FULL on the GPU when one is present.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.v3.config import ControllerConfig, GridConfig, TrainConfig

PROFILE_NAMES = ("SMOKE", "MEDIUM", "FULL")


@dataclass(frozen=True)
class Profile:
    """All scale knobs for one run; see module docstring."""

    name: str
    grid: GridConfig
    train: TrainConfig
    controller: ControllerConfig
    # Block-1 training.
    train_epochs: int
    # Block-2 simulation + collection.
    n_firms: int
    T: int
    burn_in: int
    refine_rounds: int
    collect_rows: int
    collect_batch_size: int
    # Block-3 surrogate + estimation.
    surrogate_passes: int
    n_restarts: int
    # Recovery (Sec 12.2) and controller cadence.
    recovery_draws: int
    controller_every: int   # run the controller min-loss/shrink step every N epochs

    def with_(self, **overrides) -> "Profile":
        """Return a copy with fields overridden (e.g. profile.with_(recovery_draws=2))."""
        from dataclasses import replace
        return replace(self, **overrides)


_SMOKE = Profile(
    name="SMOKE",
    grid=GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7),
    train=TrainConfig(batch_size=512, steps_per_epoch=30),
    controller=ControllerConfig(n_loss_points=7, n_restarts=6, n_folds=4,
                                warmup_epochs=0, n_volume_steps=8,
                                containment_recent=40, containment_closest=8,
                                surrogate_max_obs=500),
    train_epochs=6,
    n_firms=300, T=50, burn_in=15, refine_rounds=6,
    collect_rows=32, collect_batch_size=8,
    surrogate_passes=40, n_restarts=6,
    recovery_draws=4, controller_every=1,
)

_MEDIUM = Profile(
    name="MEDIUM",
    grid=GridConfig(n_z=7, n_k=10, n_b=15, n_kp=25, n_bp=25, n_cp=15),
    train=TrainConfig(batch_size=4096, steps_per_epoch=200),
    controller=ControllerConfig(n_loss_points=15, n_restarts=15, n_folds=10,
                                warmup_epochs=2, n_volume_steps=16,
                                containment_recent=200, containment_closest=25,
                                surrogate_max_obs=4000),
    train_epochs=30,
    n_firms=2000, T=120, burn_in=60, refine_rounds=6,
    collect_rows=400, collect_batch_size=16,
    surrogate_passes=200, n_restarts=30,
    recovery_draws=20, controller_every=5,
)

_FULL = Profile(
    name="FULL",
    grid=GridConfig(),  # spec default 11x15x35 / 81x91x71
    train=TrainConfig(),  # spec default batch 8192, 500 steps
    controller=ControllerConfig(),  # spec default 31 pts, 30 restarts, 10 folds, warm-up 200
    train_epochs=1000,
    n_firms=5000, T=300, burn_in=200, refine_rounds=6,
    # collect_batch_size is memory-bound at the FULL state grid: the batched dense
    # policy-evaluation solve is [B, S, S] with S = n_z*n_k*n_b = 5775, i.e. ~266 MB
    # per batch element in float64, and the batched panel records [B, n_firms, T]. B=8
    # fits a 40 GB A100 with headroom; raise toward 16 on an A100 only (see notebook).
    collect_rows=10000, collect_batch_size=8,
    surrogate_passes=200, n_restarts=30,
    recovery_draws=40, controller_every=1,
)

_PROFILES = {"SMOKE": _SMOKE, "MEDIUM": _MEDIUM, "FULL": _FULL}


def get_profile(name: str) -> Profile:
    """Return the named scale preset (one of :data:`PROFILE_NAMES`)."""
    key = name.upper()
    if key not in _PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {PROFILE_NAMES}")
    return _PROFILES[key]
