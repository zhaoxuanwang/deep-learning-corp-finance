"""Generic Simulated Method of Moments (SMM) for v2.

The core layer is intentionally model-agnostic. It knows only about:
  - parameter vectors and bounds
  - target moments
  - simulated per-panel moments
  - objective construction, two-step weighting, inference, and Monte Carlo

All model-specific simulation, shock construction, and moment logic must live
outside this module (in the environment files).

---------------------------------------------------------------------------
COST WARNING — READ BEFORE USING
---------------------------------------------------------------------------
The SMM optimization loop calls ``simulate_panel_moments`` (passed via
``SMMSpec``) **once per candidate β** that the optimizer evaluates.  Each
such call typically involves:

  1. Re-solving the model at β (e.g., VFI, PFI, or NN training) — this is
     the dominant cost, ranging from seconds (PFI on a coarse grid) to
     minutes (full PFI or NN-based training).
  2. Simulating S panels using the solved policy.
  3. Computing moments from those S panels.

A single ``solve_smm`` run may call ``simulate_panel_moments`` hundreds to
thousands of times (``nfev`` in ``SMMStageResult`` reports the count per
stage).  For expensive models, budget accordingly.  The Common Random
Numbers (CRN) design (fixed simulation seed across all β candidates) makes
Q(β) smooth in β, reducing the optimizer iterations needed, but does not
reduce the per-call model-solve cost.

The frictionless analytical-policy path (Stage A of the basic investment
model) is the exception: the policy is a closed-form formula, so each call
costs microseconds instead of seconds.
---------------------------------------------------------------------------

Algorithm structure (matches SMM.md Parts 1–4):

  Part 1 — Setup:      _make_cached_evaluator()
  Part 2 — Estimation (inner loop owned by scipy optimizer):
                       _run_smm_stage()
  Part 3 — Optimal weight: _build_omega_hat()
  Part 4 — Inference:  _compute_smm_inference()

``solve_smm`` orchestrates Parts 1–4.

Shared panel-statistics helpers used by environment moment calculators:
  _panel_covariance()
  _panel_serial_correlation()
  _panel_iv_first_diff_ar1()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, least_squares, minimize
from scipy.stats import chi2

from src.v2.utils.seeding import fold_in_seed, make_seed_int


ErrorType = Literal["level", "percent"]
GlobalMethod = Literal[
    "dual_annealing", "differential_evolution", "L-BFGS-B", "Powell", "Nelder-Mead"
]
LocalMethod = Literal["Powell", "least_squares"]

# Keep old name as alias for backward compatibility in type annotations.
OptimizerName = GlobalMethod

_SUPPORTED_GLOBAL_METHODS = {
    "dual_annealing",
    "differential_evolution",
    "L-BFGS-B",
    "Powell",
    "Nelder-Mead",
}
_SUPPORTED_LOCAL_METHODS = {"Powell", "least_squares"}

# Ridge added to Omega_hat before inversion to stabilise near-singular cases
# (e.g., S close to R, or panels with near-identical shocks by chance).
# Requires S > R for full rank; in production use S >> R (e.g., 50x).
# This value is negligible when Omega_hat diagonal entries are O(1e-3) or
# larger.  If cond(Omega_hat) >> 1e6 before ridge, increase S rather than
# increasing this constant.
_OMEGA_RIDGE = 1e-8

# Condition-number threshold above which the J-test result is unreliable.
# When cond(Omega_hat) exceeds this before ridge regularisation, the optimal
# weighting matrix is numerically unstable and the J-statistic is inflated.
_OMEGA_COND_WARN = 1e6

_FD_REL_STEP = 1e-4
_FD_ABS_STEP = 1e-8
# Fraction of the bounds range used as a lower bound on the FD step.
# Prevents vanishing steps for small-magnitude parameters on coarse grids
# (e.g., psi1=0.008 → 1e-4*0.008 = 8e-7, which cannot flip a discrete
# argmax on a 10×25×25 grid).  1% of the feasible range is a conservative
# floor: large enough to move the policy, small enough for a reasonable
# linear approximation.
_FD_BOUNDS_FRAC = 0.01
_J_TEST_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SMMRunConfig:
    """Flat runtime configuration for one SMM estimation run.

    Attributes:
        n_firms:           Number of simulated firms per panel.
        horizon:           Number of periods *recorded* per panel
                           (post-burn-in).  Total simulation length is
                           ``burn_in + horizon``.  Do not confuse with the
                           full simulation period — only the last ``horizon``
                           steps are kept; ``burn_in`` steps are discarded.
        burn_in:           Number of initial periods discarded to let the
                           simulated economy reach its stationary
                           distribution before recording.
        n_sim_panels:      Number of independently simulated panels S per
                           evaluation of Q(β).  Must exceed n_moments for
                           Omega_hat to be full-rank.  Use S >> R in
                           production (e.g., 50× more panels than moments).
        master_seed:       Length-2 integer tuple used to derive all
                           sub-seeds via the seeding utilities.
        error_type:        ``'level'`` (default) uses raw differences
                           ``m_sim - m_target``; ``'percent'`` normalises
                           by ``|m_target|`` (with a small floor for
                           near-zero targets).  Level is recommended
                           when moments differ in magnitude because
                           percent amplifies moments whose target is
                           small, inflating Omega_hat's condition
                           number.
        percent_denom_floor:
                           Minimum absolute denominator for percent errors,
                           preventing division by near-zero targets.
        global_method:     Global optimizer for Stage 1 exploration.
                           ``'dual_annealing'`` (default) is recommended
                           when the objective landscape is unknown.
        local_method:      Local refinement optimizer used after the global
                           search (Stage 1 polish) and for Stage 2.
                           ``'Powell'`` (default) is derivative-free.
                           ``'least_squares'`` exploits the least-squares
                           structure of the SMM objective via Cholesky-
                           transformed residuals and trust-region steps.
        optimizer_maxiter: Maximum iterations for the global optimizer.
                           The local refinement uses its own internal
                           stopping criteria and is not governed by this.
        optimizer_popsize: Population size multiplier for
                           ``differential_evolution``.  Actual population
                           is ``popsize * n_params``.  Ignored by other
                           global methods.
    """

    n_firms: int = 256
    horizon: int = 32
    burn_in: int = 64
    n_sim_panels: int = 32
    master_seed: tuple[int, int] = (20, 26)
    error_type: ErrorType = "level"
    percent_denom_floor: float = 1e-6
    global_method: GlobalMethod = "dual_annealing"
    local_method: LocalMethod = "Powell"
    optimizer_maxiter: int = 200
    optimizer_popsize: int = 15

    @property
    def optimizer_name(self) -> str:
        """Backward-compatible alias for global_method."""
        return self.global_method

    def __post_init__(self):
        if self.n_firms < 1:
            raise ValueError(f"n_firms must be >= 1. Got {self.n_firms}")
        if self.horizon < 2:
            raise ValueError(f"horizon must be >= 2. Got {self.horizon}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0. Got {self.burn_in}")
        if self.n_sim_panels < 1:
            raise ValueError(
                f"n_sim_panels must be >= 1. Got {self.n_sim_panels}"
            )
        if self.error_type not in {"level", "percent"}:
            raise ValueError(
                f"error_type must be 'level' or 'percent'. Got {self.error_type!r}"
            )
        if self.percent_denom_floor <= 0:
            raise ValueError(
                "percent_denom_floor must be > 0. "
                f"Got {self.percent_denom_floor}"
            )
        if self.global_method not in _SUPPORTED_GLOBAL_METHODS:
            raise ValueError(
                "global_method must be one of "
                f"{sorted(_SUPPORTED_GLOBAL_METHODS)}. "
                f"Got {self.global_method!r}"
            )
        if self.local_method not in _SUPPORTED_LOCAL_METHODS:
            raise ValueError(
                "local_method must be one of "
                f"{sorted(_SUPPORTED_LOCAL_METHODS)}. "
                f"Got {self.local_method!r}"
            )
        if self.optimizer_maxiter < 1:
            raise ValueError(
                "optimizer_maxiter must be >= 1. "
                f"Got {self.optimizer_maxiter}"
            )
        if self.master_seed is None or len(self.master_seed) != 2:
            raise ValueError(
                "master_seed must be a length-2 integer tuple. "
                f"Got {self.master_seed!r}"
            )


@dataclass(frozen=True)
class SMMMonteCarloConfig:
    """Monte Carlo validation settings."""

    n_replications: int = 1

    def __post_init__(self):
        if self.n_replications < 1:
            raise ValueError(
                f"n_replications must be >= 1. Got {self.n_replications}"
            )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SMMTargetMoments:
    """Fixed target moments for one SMM run."""

    values: np.ndarray
    n_observations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "values", values)
        if values.ndim != 1:
            raise ValueError(
                f"values must be a 1-D array. Got shape {values.shape}."
            )
        if self.n_observations < 1:
            raise ValueError(
                "n_observations must be >= 1. "
                f"Got {self.n_observations}"
            )


@dataclass(frozen=True)
class SMMPanelMoments:
    """Per-panel simulated moments for a candidate parameter vector."""

    panel_moments: np.ndarray
    n_observations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        panel_moments = np.asarray(self.panel_moments, dtype=np.float64)
        object.__setattr__(self, "panel_moments", panel_moments)
        if panel_moments.ndim != 2:
            raise ValueError(
                "panel_moments must be a 2-D array of shape "
                f"(n_panels, n_moments). Got {panel_moments.shape}."
            )
        if panel_moments.shape[0] < 1:
            raise ValueError("panel_moments must contain at least one panel.")
        if self.n_observations < 1:
            raise ValueError(
                "n_observations must be >= 1. "
                f"Got {self.n_observations}"
            )

    @property
    def average_moments(self) -> np.ndarray:
        return np.mean(self.panel_moments, axis=0)

    @property
    def n_panels(self) -> int:
        return int(self.panel_moments.shape[0])


# ---------------------------------------------------------------------------
# Simulator callable types
# ---------------------------------------------------------------------------

PanelMomentSimulator = Callable[
    [Sequence[float], SMMRunConfig, tuple[int, int]],
    SMMPanelMoments,
]
"""Callable(beta, config, seed) -> SMMPanelMoments.

Called once per candidate β by the optimizer (Part 2).  For models with an
expensive solve step (VFI, PFI, NN training), each call can take seconds to
minutes.  The total call count per stage is in ``SMMStageResult.optimizer_nfev``.
"""

TargetMomentSimulator = Callable[
    [Sequence[float], SMMRunConfig, tuple[int, int]],
    SMMTargetMoments,
]
"""Callable(beta, config, seed) -> SMMTargetMoments.

Used only during Monte Carlo validation to generate fake-real target moments.
Called once per replication, not inside the optimizer loop.
"""


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SMMSpec:
    """Flat SMM problem specification consumed by the generic core.

    Attributes:
        parameter_names:         Names of the K structural parameters.
        moment_names:            Names of the R moment conditions (R >= K).
        bounds:                  (lower, upper) bounds for each parameter.
        initial_guess:           Starting point for the outer optimizer.
                                 Defaults to the midpoint of bounds when
                                 constructed via ``env.make_smm_spec()``.
                                 Pass ``env.smm_true_beta()`` explicitly
                                 only for oracle debugging — starting from
                                 a wrong initial point is the standard
                                 validation test for SMM correctness.
        simulate_panel_moments:  See ``PanelMomentSimulator``.  This is the
                                 inner-loop function called once per
                                 candidate β by the optimizer.
        simulate_target_moments: See ``TargetMomentSimulator``.  Only
                                 needed for Monte Carlo validation via
                                 ``validate_smm``.
    """

    parameter_names: tuple[str, ...]
    moment_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    initial_guess: np.ndarray
    simulate_panel_moments: PanelMomentSimulator
    simulate_target_moments: Optional[TargetMomentSimulator] = None

    def __post_init__(self):
        parameter_names = tuple(self.parameter_names)
        moment_names = tuple(self.moment_names)
        bounds = tuple(tuple(map(float, pair)) for pair in self.bounds)
        initial_guess = np.asarray(self.initial_guess, dtype=np.float64)

        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "moment_names", moment_names)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "initial_guess", initial_guess)

        if initial_guess.ndim != 1:
            raise ValueError(
                "initial_guess must be a 1-D array. "
                f"Got shape {initial_guess.shape}."
            )
        if len(parameter_names) != initial_guess.size:
            raise ValueError(
                "parameter_names and initial_guess must have matching lengths. "
                f"Got {len(parameter_names)} and {initial_guess.size}."
            )
        if len(bounds) != initial_guess.size:
            raise ValueError(
                "bounds and initial_guess must have matching lengths. "
                f"Got {len(bounds)} and {initial_guess.size}."
            )
        for lower, upper in bounds:
            if lower > upper:
                raise ValueError(
                    "Each bound must satisfy lower <= upper. "
                    f"Got ({lower}, {upper})."
                )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SMMStageResult:
    beta: np.ndarray
    objective: float
    moment_fit_errors: np.ndarray
    """Moment error vector e(β̂) at the solution (same scale as error_type).

    For ``error_type='percent'`` each entry is ``(m_sim - m_target)/|m_target|``.
    """
    average_moments: np.ndarray
    panel_moments: np.ndarray
    weighting_matrix: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    optimizer_nit: int
    optimizer_nfev: int
    trace: dict[str, list[Any]]


@dataclass
class SMMSolveResult:
    parameter_names: tuple[str, ...]
    moment_names: tuple[str, ...]
    target_moments: np.ndarray
    target_n_observations: int
    simulation_seed: tuple[int, int]
    error_type: ErrorType
    stage1: SMMStageResult
    stage2: SMMStageResult
    omega_hat: np.ndarray
    omega_hat_condition_number: float
    jacobian: np.ndarray
    """Jacobian D (R × K) of the moment function w.r.t. β at β̂."""
    asymptotic_variance: np.ndarray
    """Asymptotic variance matrix V (K × K).  NaN if D'WD is singular."""
    standard_errors: np.ndarray
    """Per-parameter standard errors SE_k = sqrt(V_kk / N_d).  NaN if D'WD is singular."""
    j_statistic: float
    j_p_value: float
    j_df: int
    j_test_valid: bool
    """False when Omega_hat is ill-conditioned or the optimizer did not terminate normally."""

    @property
    def beta_hat(self) -> np.ndarray:
        return self.stage2.beta


@dataclass
class SMMMonteCarloSummary:
    parameter_names: tuple[str, ...]
    beta_true: np.ndarray
    mean_beta_hat: np.ndarray
    bias: np.ndarray
    sd: np.ndarray
    mean_standard_error: np.ndarray
    rmse: np.ndarray
    coverage_95: np.ndarray
    j_test_size: float


@dataclass
class SMMMonteCarloResult:
    parameter_names: tuple[str, ...]
    moment_names: tuple[str, ...]
    beta_true: np.ndarray
    replications: list[SMMSolveResult]
    summary: SMMMonteCarloSummary


@dataclass
class _CachedEvaluation:
    beta: np.ndarray
    panel_result: SMMPanelMoments
    error_vector: np.ndarray


# ---------------------------------------------------------------------------
# Part 1 — Setup: cached evaluator
# ---------------------------------------------------------------------------

def _make_cached_evaluator(
    spec: SMMSpec,
    target: SMMTargetMoments,
    config: SMMRunConfig,
    simulation_seed: tuple[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
) -> Callable[[Sequence[float]], _CachedEvaluation]:
    """Part 1 of the SMM algorithm — build the cached Q(β) evaluator.

    Returns ``evaluate_beta(β) -> _CachedEvaluation``.

    The simulation seed is fixed once here and reused for every β candidate
    (Common Random Numbers, CRN).  CRN ensures the same underlying shocks
    drive all simulated panels regardless of β, making Q(β) smooth in β and
    reducing the number of optimizer iterations needed.

    Each call to ``evaluate_beta`` that misses the cache:
      1. Clips β to bounds.
      2. Calls ``spec.simulate_panel_moments(β, config, seed)`` — the
         model-solve + simulate + moment-compute step.  This is the dominant
         cost: for VFI/PFI/NN-based models each call can take seconds to
         minutes; the optimizer may issue hundreds of such calls per stage.
      3. Computes the error vector e(β) in level or percent form.
      4. Caches by β bytes so identical candidates are never re-solved.
    """
    cache: dict[bytes, _CachedEvaluation] = {}
    moment_names = spec.moment_names

    def evaluate_beta(beta: Sequence[float]) -> _CachedEvaluation:
        beta_arr = np.clip(_as_1d_float(beta, "beta"), lower, upper)
        key = beta_arr.tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached

        # Model solve + simulate + moments.  Expensive for VFI/PFI/NN models.
        panel_result = spec.simulate_panel_moments(beta_arr, config, simulation_seed)
        if panel_result.panel_moments.shape[1] != len(moment_names):
            raise ValueError(
                "simulate_panel_moments returned the wrong number of moments. "
                f"Expected {len(moment_names)}, got "
                f"{panel_result.panel_moments.shape[1]}."
            )

        error_vector = _moment_errors(
            panel_result.average_moments,
            target.values,
            config.error_type,
            config.percent_denom_floor,
        )
        result = _CachedEvaluation(
            beta=beta_arr,
            panel_result=panel_result,
            error_vector=error_vector,
        )
        cache[key] = result
        return result

    return evaluate_beta


# ---------------------------------------------------------------------------
# Part 2 — Estimation: one optimizer stage
# ---------------------------------------------------------------------------

def _run_smm_stage(
    evaluate_beta: Callable[[Sequence[float]], _CachedEvaluation],
    weighting_matrix: np.ndarray,
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    config: SMMRunConfig,
    optimizer_seed: tuple[int, int],
    *,
    local_only: bool = False,
) -> SMMStageResult:
    """Part 2 of the SMM algorithm — one full optimization stage.

    Minimizes Q(β) = e(β)^T W e(β) starting from x0.

    When ``local_only=False`` (default, stage 1):
      The configured global optimizer (e.g. ``dual_annealing``) runs first
      for global exploration, followed by a local refinement pass from its
      best point.

    When ``local_only=True`` (stage 2):
      Only local refinement runs from x0.  Stage 2 warm-starts from β̂₁
      (already a consistent estimator), so global search is unnecessary —
      only the weighting matrix has changed.

    Local refinement is controlled by ``config.local_method``:
      - ``"Powell"``: derivative-free conjugate-direction search on the
        scalar Q(β).
      - ``"least_squares"``: Cholesky-transforms the SMM residual and uses
        ``scipy.optimize.least_squares`` with trust-region reflective
        steps, exploiting the Jacobian for quadratic convergence.
    """
    trace: dict[str, list[Any]] = {"beta": [], "objective": []}

    def objective_fn(beta: Sequence[float]) -> float:
        evaluation = evaluate_beta(beta)
        objective = _objective_value(evaluation.error_vector, weighting_matrix)
        trace["beta"].append(evaluation.beta.copy())
        trace["objective"].append(float(objective))
        return objective

    if local_only:
        outer_best_x = np.asarray(x0, dtype=np.float64)
        opt_result = None
    else:
        opt_result = _call_optimizer(
            objective_fn=objective_fn,
            x0=x0,
            bounds=bounds,
            config=config,
            optimizer_seed=optimizer_seed,
        )
        outer_best_x = np.asarray(opt_result.x, dtype=np.float64)

    # --- Local refinement ---
    if config.local_method == "least_squares":
        # Cholesky-transform: W = LL^T, residual r(β) = L^T e(β),
        # so ||r||^2 = e^T W e = Q(β).
        L = np.linalg.cholesky(weighting_matrix)
        L_T = L.T.copy()
        lower_arr = np.array([b[0] for b in bounds], dtype=np.float64)
        upper_arr = np.array([b[1] for b in bounds], dtype=np.float64)

        def residual_fn(beta: Sequence[float]) -> np.ndarray:
            evaluation = evaluate_beta(beta)
            objective = _objective_value(
                evaluation.error_vector, weighting_matrix
            )
            trace["beta"].append(evaluation.beta.copy())
            trace["objective"].append(float(objective))
            return L_T @ evaluation.error_vector

        polish_result = least_squares(
            residual_fn,
            x0=outer_best_x,
            bounds=(lower_arr, upper_arr),
            method="trf",
        )
    else:
        polish_result = minimize(
            objective_fn,
            x0=outer_best_x,
            method="Powell",
            bounds=bounds,
        )
    best_x = np.asarray(polish_result.x, dtype=np.float64)

    evaluation = evaluate_beta(best_x)
    error_at_solution = evaluation.error_vector
    return SMMStageResult(
        beta=evaluation.beta.copy(),
        objective=float(
            _objective_value(evaluation.error_vector, weighting_matrix)
        ),
        moment_fit_errors=error_at_solution.copy(),
        average_moments=evaluation.panel_result.average_moments.copy(),
        panel_moments=evaluation.panel_result.panel_moments.copy(),
        weighting_matrix=np.asarray(weighting_matrix, dtype=np.float64).copy(),
        optimizer_success=bool(
            getattr(opt_result, "success", True)
            and getattr(polish_result, "success", True)
        ),
        optimizer_message=str(getattr(polish_result, "message", "")),
        optimizer_nit=int(
            getattr(opt_result, "nit", 0) + getattr(polish_result, "nit", 0)
        ),
        optimizer_nfev=len(trace["objective"]),
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Part 3 — Optimal weight: build Omega_hat
# ---------------------------------------------------------------------------

def _build_omega_hat(
    stage1_panel_moments: np.ndarray,
    target_values: np.ndarray,
    config: SMMRunConfig,
    n_moments: int,
) -> tuple[np.ndarray, float]:
    """Part 3 of the SMM algorithm — construct the optimal weighting matrix.

    Returns (omega_hat, condition_number_before_ridge).

    Ω̂ = (1/S) E E^T where E is the (S × R) matrix of per-panel moment
    errors at the stage-1 estimate.  Uses the same error form (level or
    percent) as the main objective so W = Ω̂⁻¹ is dimensionally consistent
    with e(β).

    A small ridge (_OMEGA_RIDGE) is added before inversion to handle near-
    singular cases — see the module constant for full rationale.  The
    condition number *before* adding the ridge is returned so callers can
    flag unreliable J-tests when it is large.
    """
    identity = np.eye(n_moments, dtype=np.float64)
    error_matrix = _panel_error_matrix(
        stage1_panel_moments,
        target_values,
        config.error_type,
        config.percent_denom_floor,
    )
    omega_hat_raw = (
        error_matrix.T @ error_matrix
    ) / float(error_matrix.shape[0])
    cond = float(np.linalg.cond(omega_hat_raw))
    omega_hat = omega_hat_raw + _OMEGA_RIDGE * identity
    return omega_hat, cond


# ---------------------------------------------------------------------------
# Part 4 — Inference
# ---------------------------------------------------------------------------

def _compute_smm_inference(
    beta_hat: np.ndarray,
    evaluate_beta: Callable[[Sequence[float]], _CachedEvaluation],
    weighting_matrix: np.ndarray,
    omega_hat: np.ndarray,
    final_eval: _CachedEvaluation,
    lower: np.ndarray,
    upper: np.ndarray,
    n_params: int,
    n_moments: int,
    n_sim_panels: int,
    *,
    efficient: bool = True,
) -> dict[str, Any]:
    """Part 4 of the SMM algorithm — inference at the final estimate.

    All inference uses the panel count S (not firm-years), matching the
    level at which Omega_hat is estimated.

    When ``efficient=True`` (W = Ω̂⁻¹, the two-step efficient estimator):
      - V = (1 + 1/S) · (D'WD)⁻¹
      - J = S/(S+1) · Q(β̂) ~ χ²(R-K)

    When ``efficient=False`` (W ≠ Ω̂⁻¹, e.g., W = I fallback):
      - V = (1 + 1/S) · (D'WD)⁻¹ D'WΩ̂WD (D'WD)⁻¹  (sandwich formula)
      - J-test is not available (requires W = Ω̂⁻¹ for the χ² result).
    """
    jacobian = _finite_difference_jacobian(
        evaluate_beta=evaluate_beta,
        beta_hat=beta_hat,
        lower=lower,
        upper=upper,
    )

    S = float(n_sim_panels)
    dwd = jacobian.T @ weighting_matrix @ jacobian  # K x K
    sim_correction = 1.0 + 1.0 / S
    try:
        dwd_inv = np.linalg.inv(dwd)
    except np.linalg.LinAlgError:
        warnings.warn(
            "D'WD is singular: the Jacobian does not have full column rank "
            "at beta_hat, indicating weak or failed local identification. "
            "Standard errors are undefined.",
            RuntimeWarning,
            stacklevel=3,
        )
        dwd_inv = None

    if dwd_inv is not None:
        if efficient:
            # V = (1 + 1/S) (D'WD)^{-1}
            asymptotic_variance = sim_correction * dwd_inv
        else:
            # Sandwich: V = (1 + 1/S) (D'WD)^{-1} D'W Ω̂ W D (D'WD)^{-1}
            dw = jacobian.T @ weighting_matrix  # K x R
            meat = dw @ omega_hat @ dw.T         # K x K
            asymptotic_variance = sim_correction * dwd_inv @ meat @ dwd_inv
    else:
        asymptotic_variance = np.full(
            (n_params, n_params), np.nan, dtype=np.float64
        )
    standard_errors = np.sqrt(
        np.clip(np.diag(asymptotic_variance), 0.0, None)
    )

    # J-test: only valid when W = Ω̂⁻¹ (efficient estimator) AND the
    # Jacobian has full column rank (model is locally identified).
    j_df = n_moments - n_params
    if efficient and j_df > 0 and dwd_inv is not None:
        objective_final = _objective_value(
            final_eval.error_vector, weighting_matrix
        )
        j_statistic = float(S / (S + 1.0) * objective_final)
        j_p_value = float(chi2.sf(j_statistic, j_df))
    else:
        j_statistic = float("nan")
        j_p_value = float("nan")

    return dict(
        jacobian=jacobian,
        asymptotic_variance=asymptotic_variance,
        standard_errors=standard_errors,
        j_statistic=j_statistic,
        j_p_value=j_p_value,
        j_df=j_df,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def solve_smm(
    spec: SMMSpec,
    target: SMMTargetMoments,
    config: SMMRunConfig | None = None,
    simulation_seed: Optional[tuple[int, int]] = None,
) -> SMMSolveResult:
    """Run the generic two-step SMM estimator.

    Orchestrates Parts 1–4 of the algorithm documented in SMM.md:

      Part 1 — Build cached evaluator (fixed CRN seed).
      Part 2 — Stage 1: minimise Q(β) with W = I.
      Part 3 — Build Ω̂ from stage-1 per-panel moments.
      Part 2 — Stage 2: minimise Q(β) with W = Ω̂⁻¹, then Powell polish.
      Part 4 — Jacobian, standard errors, J-statistic.

    Total model solves ≈ stage1.optimizer_nfev + stage2.optimizer_nfev
    + 2 × K (Jacobian finite differences).  For expensive model solves
    (VFI/PFI/NN) this dominates wall time — see module docstring.
    """
    config = config or SMMRunConfig()
    simulation_seed = simulation_seed or fold_in_seed(
        config.master_seed, "smm", "crn"
    )

    parameter_names = tuple(spec.parameter_names)
    moment_names = tuple(spec.moment_names)
    initial_guess = _as_1d_float(spec.initial_guess, "initial_guess")
    bounds = tuple(tuple(map(float, pair)) for pair in spec.bounds)

    if target.values.size != len(moment_names):
        raise ValueError(
            "target moment count does not match spec.moment_names. "
            f"Got {target.values.size} and {len(moment_names)}."
        )
    if config.n_sim_panels <= len(moment_names):
        raise ValueError(
            "The two-step SMM estimator requires n_sim_panels > n_moments "
            f"to form Omega_hat. Got n_sim_panels={config.n_sim_panels} "
            f"and n_moments={len(moment_names)}."
        )

    lower = np.array([pair[0] for pair in bounds], dtype=np.float64)
    upper = np.array([pair[1] for pair in bounds], dtype=np.float64)
    if np.any(initial_guess < lower) or np.any(initial_guess > upper):
        raise ValueError("initial_guess must lie inside the supplied bounds.")

    n_params = len(parameter_names)
    n_moments = len(moment_names)
    identity = np.eye(n_moments, dtype=np.float64)

    # --- Part 1: Setup ---
    evaluate_beta = _make_cached_evaluator(
        spec=spec,
        target=target,
        config=config,
        simulation_seed=simulation_seed,
        lower=lower,
        upper=upper,
    )

    # --- Part 2 (stage 1): Minimise Q with W = I ---
    stage1_seed = fold_in_seed(
        simulation_seed, "smm", "stage1", "optimizer", config.global_method
    )
    stage1 = _run_smm_stage(
        evaluate_beta=evaluate_beta,
        weighting_matrix=identity,
        x0=initial_guess,
        bounds=bounds,
        config=config,
        optimizer_seed=stage1_seed,
    )

    # --- Part 3: Build Omega_hat at stage-1 estimate ---
    omega_hat, omega_cond = _build_omega_hat(
        stage1_panel_moments=stage1.panel_moments,
        target_values=target.values,
        config=config,
        n_moments=n_moments,
    )

    j_test_valid = True
    if omega_cond > _OMEGA_COND_WARN:
        warnings.warn(
            f"Omega_hat condition number {omega_cond:.2e} exceeds "
            f"{_OMEGA_COND_WARN:.0e}. W = Omega_hat^{{-1}} is numerically "
            "unstable. Falling back to W = I for Stage 2 (first-step "
            "estimator). J-test and efficient standard errors are not "
            "available. Increase n_sim_panels (recommended S >> R, e.g., "
            "50x n_moments) to obtain the efficient estimator.",
            RuntimeWarning,
            stacklevel=2,
        )
        j_test_valid = False
        # Fall back to W = I: the first-step estimator is consistent,
        # and using an ill-conditioned W would distort the estimate.
        weighting_matrix = identity
    else:
        weighting_matrix = np.linalg.inv(omega_hat)

    # --- Part 2 (stage 2): Minimise Q with W ---
    # Stage 2 warm-starts from β̂₁ (already consistent) and only runs
    # local refinement.  When W = Omega_hat^{-1}, this is the efficient
    # two-step estimator.  When W = I (fallback), Stage 2 reconverges
    # to β̂₁ immediately — the two-step gracefully degrades to one-step.
    stage2_seed = fold_in_seed(
        simulation_seed, "smm", "stage2", "optimizer", config.global_method
    )
    stage2 = _run_smm_stage(
        evaluate_beta=evaluate_beta,
        weighting_matrix=weighting_matrix,
        x0=stage1.beta.copy(),
        bounds=bounds,
        config=config,
        optimizer_seed=stage2_seed,
        local_only=True,
    )
    if not stage2.optimizer_success:
        j_test_valid = False

    # --- Part 4: Inference ---
    beta_hat = stage2.beta.copy()
    final_eval = evaluate_beta(beta_hat)
    inference = _compute_smm_inference(
        beta_hat=beta_hat,
        evaluate_beta=evaluate_beta,
        weighting_matrix=weighting_matrix,
        omega_hat=omega_hat,
        final_eval=final_eval,
        lower=lower,
        upper=upper,
        n_params=n_params,
        n_moments=n_moments,
        n_sim_panels=config.n_sim_panels,
        efficient=j_test_valid,
    )

    return SMMSolveResult(
        parameter_names=parameter_names,
        moment_names=moment_names,
        target_moments=target.values.copy(),
        target_n_observations=int(target.n_observations),
        simulation_seed=tuple(map(int, simulation_seed)),
        error_type=config.error_type,
        stage1=stage1,
        stage2=stage2,
        omega_hat=omega_hat,
        omega_hat_condition_number=omega_cond,
        jacobian=inference["jacobian"],
        asymptotic_variance=inference["asymptotic_variance"],
        standard_errors=inference["standard_errors"],
        j_statistic=inference["j_statistic"],
        j_p_value=inference["j_p_value"],
        j_df=inference["j_df"],
        j_test_valid=j_test_valid,
    )


def validate_smm(
    spec: SMMSpec,
    beta_true: Sequence[float],
    run_config: SMMRunConfig | None = None,
    mc_config: SMMMonteCarloConfig | None = None,
) -> SMMMonteCarloResult:
    """Run Monte Carlo validation for an SMM specification.

    Each replication generates an independent fake-real target dataset at
    beta_true, then runs the full two-step SMM to recover beta_hat.  Target
    and simulation seeds are independent across replications.

    Requires ``spec.simulate_target_moments`` to be provided.

    Only replications where ``j_test_valid=True`` are counted in
    ``j_test_size``, so a poor Omega_hat in one replication does not
    silently contaminate the reported rejection rate.
    """
    run_config = run_config or SMMRunConfig()
    mc_config = mc_config or SMMMonteCarloConfig()
    beta_true = _as_1d_float(beta_true, "beta_true")

    if spec.simulate_target_moments is None:
        raise ValueError(
            "validate_smm requires spec.simulate_target_moments to be provided."
        )

    replications: list[SMMSolveResult] = []
    for rep in range(mc_config.n_replications):
        target_seed = fold_in_seed(
            run_config.master_seed, "smm", "mc_rep", rep, "target"
        )
        simulation_seed = fold_in_seed(
            run_config.master_seed, "smm", "mc_rep", rep, "crn"
        )
        target = spec.simulate_target_moments(beta_true, run_config, target_seed)
        result = solve_smm(
            spec=spec,
            target=target,
            config=run_config,
            simulation_seed=simulation_seed,
        )
        replications.append(result)

    beta_hats = np.stack([result.beta_hat for result in replications], axis=0)
    standard_errors = np.stack(
        [result.standard_errors for result in replications], axis=0
    )
    if beta_hats.shape[0] > 1:
        sd = np.std(beta_hats, axis=0, ddof=1)
    else:
        sd = np.zeros(beta_hats.shape[1], dtype=np.float64)

    mean_beta_hat = np.mean(beta_hats, axis=0)
    bias = mean_beta_hat - beta_true
    mean_standard_error = np.mean(standard_errors, axis=0)
    rmse = np.sqrt(np.mean((beta_hats - beta_true[None, :]) ** 2, axis=0))
    coverage_95 = np.mean(
        np.abs(beta_hats - beta_true[None, :]) <= 1.96 * standard_errors,
        axis=0,
    )

    # Only count replications with a valid Omega_hat in the J-test size.
    reject_flags = []
    for result in replications:
        if (
            result.j_test_valid
            and np.isfinite(result.j_statistic)
            and result.j_df > 0
        ):
            critical = float(chi2.ppf(1.0 - _J_TEST_ALPHA, result.j_df))
            reject_flags.append(result.j_statistic > critical)
    j_test_size = float(np.mean(reject_flags)) if reject_flags else float("nan")

    summary = SMMMonteCarloSummary(
        parameter_names=tuple(spec.parameter_names),
        beta_true=beta_true.copy(),
        mean_beta_hat=mean_beta_hat,
        bias=bias,
        sd=sd,
        mean_standard_error=mean_standard_error,
        rmse=rmse,
        coverage_95=coverage_95,
        j_test_size=j_test_size,
    )
    return SMMMonteCarloResult(
        parameter_names=tuple(spec.parameter_names),
        moment_names=tuple(spec.moment_names),
        beta_true=beta_true.copy(),
        replications=replications,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal optimizer wrapper
# ---------------------------------------------------------------------------

def _call_optimizer(
    *,
    objective_fn: Callable[[Sequence[float]], float],
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    config: SMMRunConfig,
    optimizer_seed: tuple[int, int],
):
    """Dispatch to the configured global optimizer.

    ``dual_annealing`` is seeded via ``make_seed_int`` for reproducibility.
    Local optimizers (L-BFGS-B, Powell, Nelder-Mead) are deterministic
    given x0 and bounds; their reproducibility is guaranteed by the fixed
    CRN simulation seed passed to ``evaluate_beta`` — no additional
    optimizer seed is needed or accepted by scipy for these methods.

    ``differential_evolution`` with ``workers != 1`` uses a fork-based
    process pool so that closure-based objective functions work without
    pickling.  The ``deferred`` updating strategy is required when
    workers > 1.
    """
    method = config.global_method
    if method == "dual_annealing":
        return dual_annealing(
            objective_fn,
            bounds=bounds,
            maxiter=config.optimizer_maxiter,
            seed=make_seed_int(optimizer_seed, "dual_annealing"),
            x0=np.asarray(x0, dtype=np.float64),
        )

    if method == "differential_evolution":
        return differential_evolution(
            objective_fn,
            bounds=bounds,
            maxiter=config.optimizer_maxiter,
            seed=make_seed_int(optimizer_seed, "differential_evolution"),
            popsize=config.optimizer_popsize,
            workers=1,
            polish=False,  # our own local_method handles polishing
            tol=1e-6,
        )

    options = {"maxiter": config.optimizer_maxiter}
    kwargs: dict[str, Any] = {
        "fun": objective_fn,
        "x0": np.asarray(x0, dtype=np.float64),
        "method": method,
        "options": options,
    }
    if method != "Nelder-Mead":
        kwargs["bounds"] = bounds
    return minimize(**kwargs)


# ---------------------------------------------------------------------------
# Shared panel statistics — imported by environment moment calculators
# ---------------------------------------------------------------------------

def _panel_covariance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Population covariance between two flattened panel arrays."""
    lhs_flat = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs_flat = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_centered = lhs_flat - np.mean(lhs_flat)
    rhs_centered = rhs_flat - np.mean(rhs_flat)
    return float(np.mean(lhs_centered * rhs_centered))


def _panel_serial_correlation(values: np.ndarray) -> float:
    """Serial correlation of a (firms × time) panel array.

    Uses all flattened (t, t-1) pairs across firms, normalised by the
    product of standard deviations.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] < 2:
        return 0.0
    current = values[:, 1:].reshape(-1)
    lagged = values[:, :-1].reshape(-1)
    current_centered = current - np.mean(current)
    lagged_centered = lagged - np.mean(lagged)
    denominator = np.sqrt(
        np.mean(current_centered ** 2) * np.mean(lagged_centered ** 2)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.mean(current_centered * lagged_centered) / denominator)


def _panel_iv_first_diff_ar1(values: np.ndarray) -> tuple[float, float]:
    """AR(1) persistence and shock std-dev via IV first-difference estimator.

    Implements the panel regression from Hennessy & Whited (2007):

        Δy_{it} = Δδ_t + β₁ Δy_{i,t-1} + Δu_{it}

    where Δ is the first difference and y_{i,t-2} instruments for
    Δy_{i,t-1}.  The exclusion restriction E[y_{i,t-2} · u_{it}] = 0
    holds by construction under an AR(1) error process.

    This estimator is applied to *both* simulated and real data so that
    the SMM moment conditions match exactly.  Using OLS on simulated data
    (where the true AR(1) is known) would introduce a moment inconsistency:
    OLS is valid under the known exogenous process but real Compustat data
    may exhibit endogeneity (income/asset ratio depends on capital choice).
    IV eliminates this source of moment mismatch uniformly.

    Args:
        values: (n_firms, horizon) panel in log scale.  Callers pass
                log(z · k^{α-1}), the log income/asset ratio, which is
                an observable proxy for the latent log(z) process.

    Returns:
        (beta1, sigma_u): AR(1) slope and residual std-dev.
        Returns (0.0, std(y)) on degenerate inputs (horizon < 3).
    """
    values = np.asarray(values, dtype=np.float64)
    n_firms, horizon = values.shape
    if n_firms < 1 or horizon < 3:
        return 0.0, float(np.std(values))

    diff = values[:, 1:] - values[:, :-1]   # (n_firms, horizon-1)
    dep = diff[:, 1:]           # Δy_{t},   (n_firms, horizon-2)
    endog = diff[:, :-1]        # Δy_{t-1}, (n_firms, horizon-2)
    instruments = values[:, :-2]  # y_{t-2}, (n_firms, horizon-2)

    # Within-time demeaning absorbs time fixed effects Δδ_t
    dep = dep - np.mean(dep, axis=0, keepdims=True)
    endog = endog - np.mean(endog, axis=0, keepdims=True)
    instruments = instruments - np.mean(instruments, axis=0, keepdims=True)

    y = dep.reshape(-1, 1)
    x = endog.reshape(-1, 1)
    z = instruments.reshape(-1, 1)

    zz = z.T @ z
    if float(zz[0, 0]) <= 1e-12:
        return 0.0, float(np.std(y))

    x_hat = z @ np.linalg.solve(zz, z.T @ x)
    xhx = float((x_hat.T @ x)[0, 0])
    if xhx <= 1e-12:
        return 0.0, float(np.std(y))

    beta1 = float((x_hat.T @ y)[0, 0] / xhx)
    resid = y - x * beta1
    return beta1, float(np.std(resid))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _as_1d_float(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array. Got shape {arr.shape}.")
    return arr


def _percent_denominator(target_moments: np.ndarray, floor: float) -> np.ndarray:
    sign = np.where(target_moments >= 0.0, 1.0, -1.0)
    stabilized = np.where(
        np.abs(target_moments) >= floor,
        target_moments,
        sign * floor,
    )
    return stabilized.astype(np.float64)


def _moment_errors(
    simulated_moments: np.ndarray,
    target_moments: np.ndarray,
    error_type: ErrorType,
    percent_denom_floor: float,
) -> np.ndarray:
    simulated_moments = np.asarray(simulated_moments, dtype=np.float64)
    target_moments = np.asarray(target_moments, dtype=np.float64)
    if error_type == "level":
        return simulated_moments - target_moments
    denom = _percent_denominator(target_moments, percent_denom_floor)
    return (simulated_moments - target_moments) / denom


def _panel_error_matrix(
    panel_moments: np.ndarray,
    target_moments: np.ndarray,
    error_type: ErrorType,
    percent_denom_floor: float,
) -> np.ndarray:
    panel_moments = np.asarray(panel_moments, dtype=np.float64)
    target_moments = np.asarray(target_moments, dtype=np.float64)
    if error_type == "level":
        return panel_moments - target_moments[None, :]
    denom = _percent_denominator(target_moments, percent_denom_floor)
    return (panel_moments - target_moments[None, :]) / denom[None, :]


def _objective_value(error_vector: np.ndarray, weighting_matrix: np.ndarray) -> float:
    return float(error_vector.T @ weighting_matrix @ error_vector)



def _finite_difference_jacobian(
    *,
    evaluate_beta: Callable[[Sequence[float]], _CachedEvaluation],
    beta_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """R × K Jacobian of e(β) via central finite differences.

    Requires 2K calls to ``evaluate_beta`` (each a full model solve for
    expensive models).  Falls back to one-sided differences at boundaries.
    """
    beta_hat = np.asarray(beta_hat, dtype=np.float64)
    base_error = evaluate_beta(beta_hat).error_vector
    n_moments = base_error.size
    n_params = beta_hat.size
    jacobian = np.zeros((n_moments, n_params), dtype=np.float64)

    for idx in range(n_params):
        bounds_floor = _FD_BOUNDS_FRAC * (upper[idx] - lower[idx])
        base_step = max(
            _FD_REL_STEP * abs(beta_hat[idx]), bounds_floor, _FD_ABS_STEP
        )
        max_down = beta_hat[idx] - lower[idx]
        max_up = upper[idx] - beta_hat[idx]

        if max_down > 0.0 and max_up > 0.0:
            step = min(base_step, max_down, max_up)
            beta_plus = beta_hat.copy()
            beta_minus = beta_hat.copy()
            beta_plus[idx] += step
            beta_minus[idx] -= step
            error_plus = evaluate_beta(beta_plus).error_vector
            error_minus = evaluate_beta(beta_minus).error_vector
            jacobian[:, idx] = (error_plus - error_minus) / (2.0 * step)
        elif max_up > 0.0:
            step = min(base_step, max_up)
            beta_plus = beta_hat.copy()
            beta_plus[idx] += step
            error_plus = evaluate_beta(beta_plus).error_vector
            jacobian[:, idx] = (error_plus - base_error) / step
        elif max_down > 0.0:
            step = min(base_step, max_down)
            beta_minus = beta_hat.copy()
            beta_minus[idx] -= step
            error_minus = evaluate_beta(beta_minus).error_vector
            jacobian[:, idx] = (base_error - error_minus) / step
        else:
            jacobian[:, idx] = 0.0

    return jacobian
