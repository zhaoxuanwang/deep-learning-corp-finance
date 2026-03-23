"""
src/v2/solvers/config.py

Configuration dataclasses for discrete VFI and PFI solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


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


@dataclass
class VFIConfig:
    """Configuration for Value Function Iteration.

    Attributes:
        grid:               Grid discretization settings.
        tol:                Sup-norm convergence tolerance.
        max_iter:           Maximum Bellman iterations.
        eval_interval:      Evaluate via eval_callback every N Bellman iterations.
        monitor:            Metric name to watch for early stopping (optional).
        threshold:          Stop when monitor metric satisfies the criterion.
        threshold_patience:     Consecutive eval checkpoints that must satisfy rule.
        reward_temperature: Temperature passed into env.reward() when building
                            reward tables.
        reward_gate_mode:   Gate mode passed into env.reward() when building
                            reward tables.
    """
    grid:               GridConfig = field(default_factory=GridConfig)
    tol:                float      = 1e-6
    max_iter:           int        = 2000
    eval_interval:      int        = 50
    monitor:            Optional[str]   = None
    threshold:          Optional[float] = None
    threshold_patience:     int        = 1
    reward_temperature: float      = 1e-6
    reward_gate_mode:   Literal["hard", "soft", "ste"] = "hard"

    def __post_init__(self):
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0. Got {self.tol}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1. Got {self.max_iter}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1. Got {self.eval_interval}")
        if self.threshold_patience < 1:
            raise ValueError(f"threshold_patience must be >= 1. Got {self.threshold_patience}")
        if self.reward_temperature <= 0:
            raise ValueError(
                "reward_temperature must be > 0. "
                f"Got {self.reward_temperature}"
            )
        if self.reward_gate_mode not in {"hard", "soft", "ste"}:
            raise ValueError(
                "reward_gate_mode must be one of {'hard', 'soft', 'ste'}. "
                f"Got {self.reward_gate_mode!r}"
            )


@dataclass
class PFIConfig:
    """Configuration for Policy Function Iteration (Howard's method).

    Attributes:
        grid:               Grid discretization settings.
        max_iter:           Maximum policy improvement iterations.
        eval_steps:         Number of Bellman evaluations per policy
                            evaluation step.
        monitor:            Metric name to watch for early stopping (optional).
        threshold:          Stop when monitor metric satisfies the criterion.
        threshold_patience:     Consecutive eval checkpoints that must satisfy rule.
        reward_temperature: Temperature passed into env.reward() when building
                            reward tables.
        reward_gate_mode:   Gate mode passed into env.reward() when building
                            reward tables.
    """
    grid:               GridConfig = field(default_factory=GridConfig)
    max_iter:           int        = 200
    eval_steps:         int        = 400
    monitor:            Optional[str]   = None
    threshold:          Optional[float] = None
    threshold_patience:     int        = 1
    reward_temperature: float      = 1e-6
    reward_gate_mode:   Literal["hard", "soft", "ste"] = "hard"

    def __post_init__(self):
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1. Got {self.max_iter}")
        if self.eval_steps < 1:
            raise ValueError(f"eval_steps must be >= 1. Got {self.eval_steps}")
        if self.threshold_patience < 1:
            raise ValueError(f"threshold_patience must be >= 1. Got {self.threshold_patience}")
        if self.reward_temperature <= 0:
            raise ValueError(
                "reward_temperature must be > 0. "
                f"Got {self.reward_temperature}"
            )
        if self.reward_gate_mode not in {"hard", "soft", "ste"}:
            raise ValueError(
                "reward_gate_mode must be one of {'hard', 'soft', 'ste'}. "
                f"Got {self.reward_gate_mode!r}"
            )
