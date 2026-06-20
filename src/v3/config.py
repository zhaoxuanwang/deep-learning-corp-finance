"""Configuration objects for the v3 DF26 pipeline.

Frozen dataclasses with ``__post_init__`` validation (the v2 ``solvers/config.py``
house style). The estimated parameter vector has a single canonical order,
:data:`PARAM_NAMES`; bounds and values are always bound BY NAME, never by table
position (the spec warns that the paper's table order is cosmetic, Sec 0).

This module covers the first increment (Block 1 + VFI). ``EstimationConfig`` and
``ControllerConfig`` are added in later increments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Canonical order of the 8 estimated parameters (DF26 Sec 0).
PARAM_NAMES: Tuple[str, ...] = (
    "theta",   # returns to scale, in (0, 1)
    "rho",     # AR(1) autocorrelation of log productivity
    "sigma",   # SD of the productivity innovation
    "delta",   # depreciation rate
    "gamma1",  # convex capital adjustment cost coefficient
    "gamma0",  # fixed capital adjustment cost coefficient
    "chi",     # default recovery rate
    "cf",      # fixed operating cost
)
N_PARAMS = len(PARAM_NAMES)


@dataclass(frozen=True)
class ExternalParams:
    """Fixed (non-estimated) parameters (DF26 Sec 1.1, Gao-Whited-Zhang 2021)."""

    rf: float = 0.02       # risk-free rate; discount factor is 1/(1+rf)
    alpha: float = 0.30    # capital share
    tau: float = 0.20      # corporate tax rate
    lambda0: float = 0.007  # fixed equity issuance cost
    lambda1: float = 0.054  # proportional equity issuance cost
    rc: float = 0.0        # interest earned on cash (so iota_c = 1)
    wage: float = 1.0      # normalized wage (partial equilibrium)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if self.rf <= -1.0:
            raise ValueError(f"rf must exceed -1, got {self.rf}")
        if not 0.0 <= self.tau < 1.0:
            raise ValueError(f"tau must be in [0,1), got {self.tau}")
        for nm in ("lambda0", "lambda1"):
            if getattr(self, nm) < 0.0:
                raise ValueError(f"{nm} must be >= 0, got {getattr(self, nm)}")

    @property
    def iota_c(self) -> float:
        """Cash carry factor 1/(1 + rc*rf*(1-tau)); equals 1 when rc=0 (Eq 8)."""
        return 1.0 / (1.0 + self.rc * self.rf * (1.0 - self.tau))

    @property
    def discount(self) -> float:
        """Firm discount factor 1/(1+rf) (Eq 10)."""
        return 1.0 / (1.0 + self.rf)

    @property
    def bond_discount(self) -> float:
        """Bond discount 1/(1 + rf*(1-tau)) (Eq 6)."""
        return 1.0 / (1.0 + self.rf * (1.0 - self.tau))


@dataclass(frozen=True)
class ModelParams:
    """The 8 estimated parameters (DF26 Sec 1.1), in canonical order."""

    theta: float
    rho: float
    sigma: float
    delta: float
    gamma1: float
    gamma0: float
    chi: float
    cf: float

    def __post_init__(self) -> None:
        if not 0.0 < self.theta < 1.0:
            raise ValueError(f"theta must be in (0,1), got {self.theta}")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(f"rho must be in [0,1), got {self.rho}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not 0.0 <= self.delta <= 1.0:
            raise ValueError(f"delta must be in [0,1], got {self.delta}")
        if self.gamma1 < 0.0:
            raise ValueError(f"gamma1 must be >= 0, got {self.gamma1}")
        if self.gamma0 < 0.0:
            raise ValueError(f"gamma0 must be >= 0, got {self.gamma0}")
        if not 0.0 <= self.chi <= 1.0:
            raise ValueError(f"chi must be in [0,1], got {self.chi}")
        if self.cf < 0.0:
            raise ValueError(f"cf must be >= 0, got {self.cf}")

    def to_array(self) -> np.ndarray:
        """Return the parameters as a float64 vector in canonical order."""
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=np.float64)

    def as_dict(self) -> Dict[str, float]:
        return {n: float(getattr(self, n)) for n in PARAM_NAMES}

    @classmethod
    def from_array(cls, arr) -> "ModelParams":
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.shape[0] != N_PARAMS:
            raise ValueError(f"expected {N_PARAMS} params, got {a.shape[0]}")
        return cls(**{n: float(v) for n, v in zip(PARAM_NAMES, a)})


@dataclass(frozen=True)
class ParamBounds:
    """Lower/upper bounds for each estimated parameter, in canonical order."""

    lower: Tuple[float, ...]
    upper: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != N_PARAMS or len(self.upper) != N_PARAMS:
            raise ValueError(f"bounds must have length {N_PARAMS}")
        for lo, hi, nm in zip(self.lower, self.upper, PARAM_NAMES):
            if not lo < hi:
                raise ValueError(f"{nm}: lower {lo} must be < upper {hi}")

    def lower_array(self) -> np.ndarray:
        return np.array(self.lower, dtype=np.float64)

    def upper_array(self) -> np.ndarray:
        return np.array(self.upper, dtype=np.float64)

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return {n: (lo, hi) for n, lo, hi in zip(PARAM_NAMES, self.lower, self.upper)}

    @classmethod
    def from_dict(cls, d: Dict[str, Tuple[float, float]]) -> "ParamBounds":
        missing = set(PARAM_NAMES) - set(d)
        if missing:
            raise ValueError(f"missing bounds for {sorted(missing)}")
        return cls(
            lower=tuple(float(d[n][0]) for n in PARAM_NAMES),
            upper=tuple(float(d[n][1]) for n in PARAM_NAMES),
        )

    @classmethod
    def table_a1(cls) -> "ParamBounds":
        """Initial bounds from DF26 Table A1 (Sec 9)."""
        return cls.from_dict({
            "theta": (0.60, 0.82),
            "rho": (0.50, 0.80),
            "sigma": (0.05, 0.20),
            "delta": (0.05, 0.20),
            "gamma1": (0.05, 1.00),
            "gamma0": (0.001, 0.20),
            "chi": (0.001, 0.90),
            "cf": (0.001, 0.25),
        })


# Reference estimates for validation only (DF26 Table 1, Panel A). Not an input.
REFERENCE_ESTIMATES = ModelParams(
    theta=0.796, rho=0.597, sigma=0.187, delta=0.066,
    gamma1=0.945, gamma0=0.014, chi=0.003, cf=0.028,
)

# Standard errors of the reference estimates (Table 1, Panel A); chi weakly identified.
REFERENCE_ESTIMATE_SE = np.array(
    [0.002, 0.007, 0.002, 0.000, 0.014, 0.000, 0.024, 0.001], dtype=np.float64)

# Data-moment targets m-hat (DF26 Table 1, Panel B), in the Sec 4.3 moment order. Used
# directly as the estimation targets when the Compustat micro data is unavailable (Sec 9).
REFERENCE_TARGETS = np.array(
    [0.065, 0.045, 0.095, 0.100, 0.495, 0.271, 0.148, 0.096, 0.080, 0.094, 0.039],
    dtype=np.float64)


@dataclass(frozen=True)
class GridConfig:
    """State and control grid sizes/bounds (DF26 Sec 3.2, 4.1, Table A2)."""

    # State grid.
    n_z: int = 11
    n_k: int = 15
    n_b: int = 35
    # Control grid.
    n_kp: int = 81
    n_bp: int = 91
    n_cp: int = 71
    # Tauchen coverage (unconditional +/- m SD).
    tauchen_m: float = 2.5
    # Net-debt box (Sec 3.2): b in [-1, 2].
    b_lo: float = -1.0
    b_hi: float = 2.0
    # Gross-debt control box: b' in [0, 2].
    bp_lo: float = 0.0
    bp_hi: float = 2.0
    # Cash control box: c' in [0, 1].
    cp_lo: float = 0.0
    cp_hi: float = 1.0

    def __post_init__(self) -> None:
        for nm in ("n_z", "n_k", "n_b", "n_kp", "n_bp", "n_cp"):
            if getattr(self, nm) < 2:
                raise ValueError(f"{nm} must be >= 2, got {getattr(self, nm)}")
        if self.tauchen_m <= 0:
            raise ValueError(f"tauchen_m must be > 0, got {self.tauchen_m}")
        if not self.b_lo < self.b_hi:
            raise ValueError("require b_lo < b_hi")
        if not self.bp_lo < self.bp_hi:
            raise ValueError("require bp_lo < bp_hi")
        if not self.cp_lo < self.cp_hi:
            raise ValueError("require cp_lo < cp_hi")

    @property
    def n_states(self) -> int:
        return self.n_z * self.n_k * self.n_b


@dataclass(frozen=True)
class NetworkConfig:
    """FiLM value/policy network architecture (DF26 Sec 3.3, Table A2)."""

    trunk_layers: int = 3
    trunk_units: int = 128
    film_hidden: int = 32
    activation: str = "silu"

    def __post_init__(self) -> None:
        if self.trunk_layers < 1:
            raise ValueError("trunk_layers must be >= 1")
        if self.trunk_units < 1 or self.film_hidden < 1:
            raise ValueError("layer widths must be >= 1")
        if self.activation != "silu":
            raise ValueError("only 'silu' is supported (Table A2)")


@dataclass(frozen=True)
class TrainConfig:
    """Block-1 policy-iteration training settings (DF26 Sec 3.4, Table A2)."""

    batch_size: int = 8192
    steps_per_epoch: int = 500
    learning_rate: float = 1e-3
    lr_decay: float = 0.99   # multiplicative, per epoch
    lr_floor: float = 1e-6
    gh_nodes: int = 5        # Gauss-Hermite quadrature nodes Q
    smooth_tau: float = 1e-3  # temperature for smoothed payoffs (Eqs 31-33)
    bprime_init_bias: float = -4.0  # b' raw-output bias (Sec 3.3)

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.steps_per_epoch < 1:
            raise ValueError("batch_size and steps_per_epoch must be >= 1")
        if not 0.0 < self.lr_decay <= 1.0:
            raise ValueError("lr_decay must be in (0,1]")
        if self.learning_rate <= 0 or self.lr_floor <= 0:
            raise ValueError("learning rates must be > 0")
        if self.gh_nodes < 1:
            raise ValueError("gh_nodes must be >= 1")
        if self.smooth_tau <= 0:
            raise ValueError("smooth_tau must be > 0")


@dataclass(frozen=True)
class ControllerConfig:
    """Minimum-loss, adaptive-shrinkage, and identification settings (DF26 Sec 6, Table A2)."""

    n_loss_points: int = 31        # evenly spaced beta^j values per minimum-loss curve (Eq 44)
    n_restarts: int = 30           # LM restarts per fold in the inner min over beta^{-j}
    n_folds: int = 10              # CV folds (must match the surrogate)
    warmup_epochs: int = 200       # epochs before any bound shrink is attempted
    target_volume: float = 0.80    # initial post-shrink volume fraction v (a 20% cut)
    volume_min: float = 0.05       # smallest admissible v searched
    volume_max: float = 1.00       # largest v (no shrink)
    n_volume_steps: int = 32       # grid resolution of the v search in [volume_min, volume_max]
    id_guard_sd: float = 3.0       # 90th-pct loss must exceed the minimum by this many SD to shrink
    id_percentile: float = 90.0    # percentile used by the identification guard
    containment_recent: int = 500  # window of most-recent sampled parameters (containment set 1)
    containment_closest: int = 50  # closest-by-GMM of that window that must stay inside the bounds
    surrogate_max_obs: int = 10000  # 10k most-recent observations cap on the surrogate dataset (Sec 5.3)

    def __post_init__(self) -> None:
        if self.n_loss_points < 3:
            raise ValueError("n_loss_points must be >= 3")
        if not 0.0 < self.volume_min <= self.target_volume <= self.volume_max <= 1.0:
            raise ValueError("require 0 < volume_min <= target_volume <= volume_max <= 1")
        if self.id_guard_sd < 0 or not 0.0 < self.id_percentile < 100.0:
            raise ValueError("invalid identification-guard settings")
        if self.containment_closest > self.containment_recent:
            raise ValueError("containment_closest must be <= containment_recent")
        if self.surrogate_max_obs < 1:
            raise ValueError("surrogate_max_obs must be >= 1")


@dataclass(frozen=True)
class RunConfig:
    """Top-level run identity and reproducibility settings (DF26 Sec 10)."""

    master_seed: int = 20260619
    experiment_name: str = "df26_block1"
    results_root: str = "outputs/v3"
    profile: str = "BASE"

    def __post_init__(self) -> None:
        if not isinstance(self.master_seed, int):
            raise ValueError("master_seed must be an int")
        if self.profile not in ("SMOKE", "BASE", "FULL"):
            raise ValueError("profile must be one of SMOKE/BASE/FULL")
