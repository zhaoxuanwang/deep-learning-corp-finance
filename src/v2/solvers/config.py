"""
src/v2/solvers/config.py

Configuration dataclasses for the supported discrete solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GridConfig:
    """Per-variable grid sizes for discrete solvers.

    Each list has one entry per variable dimension.  For example,
    BasicInvestmentEnv has exo_sizes=[7], endo_sizes=[25], action_sizes=[25].
    A risky-debt model with (k, b) endo and (z,) exo would use
    exo_sizes=[7], endo_sizes=[25, 20], action_sizes=[25, 20].

    Defaults are deliberately small for fast debugging / test iteration.
    Production runs should override with larger values.

    Attributes:
        exo_sizes:        Grid points per exogenous variable.
        endo_sizes:       Grid points per endogenous variable.
        action_sizes:     Grid points per action variable.
        transition_alpha: Dirichlet smoothing mass for transition estimation.
    """
    exo_sizes:        List[int] = field(default_factory=lambda: [7])
    endo_sizes:       List[int] = field(default_factory=lambda: [25])
    action_sizes:     List[int] = field(default_factory=lambda: [25])
    transition_alpha: float     = 1.0

    def __post_init__(self):
        for name, sizes in [("exo_sizes", self.exo_sizes),
                            ("endo_sizes", self.endo_sizes),
                            ("action_sizes", self.action_sizes)]:
            if not sizes:
                raise ValueError(f"{name} must be non-empty.")
            for i, s in enumerate(sizes):
                if s < 2:
                    raise ValueError(
                        f"{name}[{i}] must be >= 2. Got {s}")
        if self.transition_alpha < 0:
            raise ValueError(
                f"transition_alpha must be >= 0. Got {self.transition_alpha}")


_VALID_INTERPOLATION_MODES = {"auto", "linear", "none"}


@dataclass
class VFIConfig:
    """Configuration for Value Function Iteration.

    Attributes:
        grid:               Grid discretization settings.
        tol:                Sup-norm convergence tolerance.
        max_iter:           Maximum Bellman iterations.
        eval_interval:      Evaluate via eval_callback every N Bellman iterations.
        interpolation:      Continuation-value interpolation mode:
                            - "auto" (default): use linear interpolation when
                              per-variable 1-D grids are available, fall back to
                              snap-to-grid otherwise.
                            - "linear": require linear interpolation (1-D) or
                              N-linear interpolation (multi-D).
                            - "none": always snap to nearest grid point.
        monitor:            Metric name to watch for early stopping (optional).
        threshold:          Stop when monitor metric satisfies the criterion.
        threshold_patience: Consecutive eval checkpoints that must satisfy rule.
    """
    grid:               GridConfig = field(default_factory=GridConfig)
    tol:                float      = 1e-6
    max_iter:           int        = 2000
    eval_interval:      int        = 50
    interpolation:      str        = "auto"
    monitor:            Optional[str]   = None
    threshold:          Optional[float] = None
    threshold_patience: int = 1

    def __post_init__(self):
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0. Got {self.tol}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1. Got {self.max_iter}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1. Got {self.eval_interval}")
        if self.threshold_patience < 1:
            raise ValueError(f"threshold_patience must be >= 1. Got {self.threshold_patience}")
        if self.interpolation not in _VALID_INTERPOLATION_MODES:
            raise ValueError(
                f"interpolation must be one of {sorted(_VALID_INTERPOLATION_MODES)}. "
                f"Got {self.interpolation!r}"
            )
@dataclass
class PFIConfig:
    """Configuration for Policy Function Iteration (Howard's method).

    Attributes:
        grid:               Grid discretization settings.
        max_iter:           Maximum policy improvement iterations.
        eval_steps:         Number of Bellman evaluations per policy
                            evaluation step.
        interpolation:      Continuation-value interpolation mode (same
                            semantics as VFIConfig.interpolation).
        monitor:            Metric name to watch for early stopping (optional).
        threshold:          Stop when monitor metric satisfies the criterion.
        threshold_patience: Consecutive eval checkpoints that must satisfy rule.
    """
    grid:               GridConfig = field(default_factory=GridConfig)
    max_iter:           int        = 200
    eval_steps:         int        = 400
    interpolation:      str        = "auto"
    monitor:            Optional[str]   = None
    threshold:          Optional[float] = None
    threshold_patience: int = 1

    def __post_init__(self):
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1. Got {self.max_iter}")
        if self.eval_steps < 1:
            raise ValueError(f"eval_steps must be >= 1. Got {self.eval_steps}")
        if self.threshold_patience < 1:
            raise ValueError(f"threshold_patience must be >= 1. Got {self.threshold_patience}")
        if self.interpolation not in _VALID_INTERPOLATION_MODES:
            raise ValueError(
                f"interpolation must be one of {sorted(_VALID_INTERPOLATION_MODES)}. "
                f"Got {self.interpolation!r}"
            )
@dataclass
class NestedVFIGridConfig:
    """Grid sizes for the canonical risky-debt nested VFI.

    Unlike the generic grid solvers, the risky-debt benchmark searches
    directly over the endogenous next-state grid (k', b'). There is no
    separate action grid in this algorithm.

    The environment remains the authority for bounds, but the solver config
    may override how each axis is sampled via per-axis spacing rules.
    """

    exo_sizes:  List[int] = field(default_factory=lambda: [7])
    endo_sizes: List[int] = field(default_factory=lambda: [25, 20])
    exo_spacings: Optional[List[str]] = None
    endo_spacings: Optional[List[str]] = None
    exo_powers: Optional[List[float]] = None
    endo_powers: Optional[List[float]] = None

    def __post_init__(self):
        for name, sizes in [("exo_sizes", self.exo_sizes),
                            ("endo_sizes", self.endo_sizes)]:
            if not sizes:
                raise ValueError(f"{name} must be non-empty.")
            for i, s in enumerate(sizes):
                if s < 2:
                    raise ValueError(
                        f"{name}[{i}] must be >= 2. Got {s}")
        valid_spacings = {"linear", "log", "geometric", "zero_power"}
        for name, spacings, dims in [
            ("exo_spacings", self.exo_spacings, len(self.exo_sizes)),
            ("endo_spacings", self.endo_spacings, len(self.endo_sizes)),
        ]:
            if spacings is None:
                continue
            if len(spacings) != dims:
                raise ValueError(
                    f"{name} must have length {dims}. Got {len(spacings)}."
                )
            for i, spacing in enumerate(spacings):
                if spacing not in valid_spacings:
                    raise ValueError(
                        f"{name}[{i}] has unknown spacing {spacing!r}. "
                        f"Valid: {sorted(valid_spacings)}"
                    )
        for name, powers, dims in [
            ("exo_powers", self.exo_powers, len(self.exo_sizes)),
            ("endo_powers", self.endo_powers, len(self.endo_sizes)),
        ]:
            if powers is None:
                continue
            if len(powers) != dims:
                raise ValueError(
                    f"{name} must have length {dims}. Got {len(powers)}."
                )
            for i, power in enumerate(powers):
                if power <= 0:
                    raise ValueError(
                        f"{name}[{i}] must be > 0. Got {power}."
                    )


@dataclass
class NestedVFIConfig:
    """Configuration for the canonical risky-debt nested VFI benchmark.

    The implementation follows docs/environments/risky_debt.md:
    one discrete Markov transition matrix, standard Bellman iteration
    with post-max clamping, hard outer updates of r_tilde, and outer
    convergence monitored on consecutive value-function iterates.
    """

    grid:               NestedVFIGridConfig = field(default_factory=NestedVFIGridConfig)
    max_iter_inner:     int = 2000
    tol_inner:          float = 1e-6
    max_iter_outer:     int = 50
    tol_outer_value:    float = 1e-6

    def __post_init__(self):
        if self.max_iter_inner < 1:
            raise ValueError(
                f"max_iter_inner must be >= 1. Got {self.max_iter_inner}")
        if self.tol_inner <= 0:
            raise ValueError(f"tol_inner must be > 0. Got {self.tol_inner}")
        if self.max_iter_outer < 1:
            raise ValueError(
                f"max_iter_outer must be >= 1. Got {self.max_iter_outer}")
        if self.tol_outer_value <= 0:
            raise ValueError(
                f"tol_outer_value must be > 0. Got {self.tol_outer_value}")


@dataclass
class InterpNestedVFIConfig(NestedVFIConfig):
    """Nested VFI with z-interpolation speedup.

    Solves the Bellman on a coarse z grid (n_z_coarse nodes selected from
    the fine Tauchen grid) and interpolates the value function to the full
    fine grid for continuation values and pricing updates.  The fine grid
    size is ``grid.exo_sizes[0]``.
    """

    n_z_coarse: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.n_z_coarse < 2:
            raise ValueError(
                f"n_z_coarse must be >= 2. Got {self.n_z_coarse}"
            )
        n_z_fine = self.grid.exo_sizes[0]
        if self.n_z_coarse >= n_z_fine:
            raise ValueError(
                f"n_z_coarse ({self.n_z_coarse}) must be < "
                f"exo_sizes[0] ({n_z_fine})."
            )


@dataclass
class RiskyDebtSolverConfig:
    """Risky-debt nested VFI solver with z-interpolation and adaptive b' bounds.

    Grid parameters:
        n_k:         Capital grid points (shared by state k and action k').
        n_b:         Debt grid points (shared by state b and action b').
        n_z:         Productivity nodes for Tauchen quadrature (fine grid).
        n_z_solve:   Productivity nodes for the Bellman solve (coarse grid).
                     Must be < n_z.  The value function is solved at n_z_solve
                     nodes and linearly interpolated to n_z for continuation
                     values and bond pricing.

    Spacing:
        k_spacing:   "geometric" (default, denser at low k) or "linear".
        b_spacing:   "linear" (default).
        z_spacing:   "log" (default, natural for log-AR(1)).

    Adaptive b' bounds:
        adaptive:    If True (default), run a fast internal solve on a small
                     grid to discover the economically active b' region, then
                     re-solve with tight bounds.
        buffer_frac: Fractional buffer around the discovered active region.
    """

    # Grid
    n_k: int = 50
    n_b: int = 50
    n_z: int = 30
    n_z_solve: int = 10
    k_spacing: str = "geometric"
    b_spacing: str = "linear"
    z_spacing: str = "log"

    # Adaptive b' bounds
    adaptive: bool = True
    buffer_frac: float = 0.01

    _VALID_SPACINGS = {"linear", "log", "geometric"}

    def __post_init__(self):
        for name, val in [("n_k", self.n_k), ("n_b", self.n_b), ("n_z", self.n_z)]:
            if val < 2:
                raise ValueError(f"{name} must be >= 2. Got {val}")
        if self.n_z_solve < 2:
            raise ValueError(f"n_z_solve must be >= 2. Got {self.n_z_solve}")
        if self.n_z_solve >= self.n_z:
            raise ValueError(
                f"n_z_solve ({self.n_z_solve}) must be < n_z ({self.n_z})."
            )
        for name, val in [
            ("k_spacing", self.k_spacing),
            ("b_spacing", self.b_spacing),
            ("z_spacing", self.z_spacing),
        ]:
            if val not in self._VALID_SPACINGS:
                raise ValueError(
                    f"{name} must be one of {sorted(self._VALID_SPACINGS)}. "
                    f"Got {val!r}"
                )


@dataclass
class NestedVFITFRuntimeConfig:
    """Backend-only runtime settings for the TF-native nested VFI solver."""

    dtype: str = "float32"
    choice_block_size: int = 128
    jit_compile: bool = True
    device: Optional[str] = None
    record_inner_history: bool = False

    def __post_init__(self):
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(
                f"dtype must be 'float32' or 'float64'. Got {self.dtype!r}"
            )
        if self.choice_block_size < 1:
            raise ValueError(
                "choice_block_size must be >= 1. "
                f"Got {self.choice_block_size}"
            )
