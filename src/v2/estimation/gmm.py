"""Generic Generalized Method of Moments (GMM) for v2.

The core layer is model-agnostic.  It knows only about:
  - parameter vectors and bounds
  - moment functions g(beta) and per-observation contributions g_it(beta)
  - objective construction, two-step weighting, inference, and Monte Carlo

All model-specific moment conditions, instruments, and panel data must live
outside this module (in the environment files).

Unlike SMM, GMM does not require solving the model per evaluation.  Each
Q(beta) evaluation is arithmetic on the data, making it orders of magnitude
faster for models where closed-form structural restrictions (e.g., Euler
equations) are available.

Algorithm structure (matches GMM.md):

  Part 1 -- Setup:       _make_cached_evaluator()
  Part 2 -- Estimation:  _run_gmm_stage()
  Part 3 -- Optimal weight: _build_omega_hat_gmm()
  Part 4 -- Inference:   _compute_gmm_inference()

``solve_gmm`` orchestrates Parts 1-4.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.stats import chi2

from src.v2.utils.seeding import fold_in_seed, make_seed_int


OptimizerName = Literal["dual_annealing", "L-BFGS-B", "Powell", "Nelder-Mead"]

_SUPPORTED_OPTIMIZERS = {
    "dual_annealing",
    "L-BFGS-B",
    "Powell",
    "Nelder-Mead",
}

_OMEGA_RIDGE = 1e-8
_OMEGA_COND_WARN = 1e6
_FD_REL_STEP = 1e-4
_FD_ABS_STEP = 1e-8
_J_TEST_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GMMRunConfig:
    """Runtime configuration for one GMM estimation run.

    Unlike SMMRunConfig, there is no n_firms / horizon / n_sim_panels
    because GMM works on a single fixed data panel.
    """

    master_seed: tuple[int, int] = (20, 26)
    optimizer_name: OptimizerName = "dual_annealing"
    optimizer_maxiter: int = 200

    def __post_init__(self):
        if self.optimizer_name not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {self.optimizer_name!r}. "
                f"Choose from {sorted(_SUPPORTED_OPTIMIZERS)}."
            )
        if self.optimizer_maxiter < 1:
            raise ValueError(
                f"optimizer_maxiter must be >= 1. Got {self.optimizer_maxiter}."
            )


@dataclass(frozen=True)
class GMMMonteCarloConfig:
    """Monte Carlo validation settings."""

    n_replications: int = 1

    def __post_init__(self):
        if self.n_replications < 1:
            raise ValueError(
                f"n_replications must be >= 1. Got {self.n_replications}."
            )


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GMMSpec:
    """Flat GMM problem specification consumed by the generic core.

    Unlike SMMSpec, there is no simulation loop.  The moment function g(beta)
    is a closed-form function of the data panel and candidate beta, evaluated
    via the callables below.

    Attributes:
        parameter_names: K parameter names.
        moment_names: R moment condition names (R >= K).
        bounds: (lower, upper) for each parameter.
        initial_guess: K-dim starting point.
        n_observations: effective NT for SE scaling and J-test.
        compute_moments: beta -> g(beta), shape (R,).
        compute_moment_contributions: beta -> g_it(beta), shape (n_obs, R).
            Called once per stage for Omega_hat construction.
        resample_spec: for Monte Carlo -- (beta_true, seed) -> new GMMSpec
            bound to a fresh simulated panel.  Optional.
    """

    parameter_names: tuple[str, ...]
    moment_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    initial_guess: np.ndarray
    n_observations: int
    n_firms: int
    n_periods: int
    compute_moments: Callable[[np.ndarray], np.ndarray]
    compute_moment_contributions: Callable[[np.ndarray], np.ndarray]
    resample_spec: Optional[Callable[..., "GMMSpec"]] = None

    def __post_init__(self):
        n_params = len(self.parameter_names)
        n_moments = len(self.moment_names)
        if n_moments < n_params:
            raise ValueError(
                f"GMM requires R >= K. Got {n_moments} moments for "
                f"{n_params} parameters."
            )
        if len(self.bounds) != n_params:
            raise ValueError(
                f"bounds length ({len(self.bounds)}) != "
                f"parameter count ({n_params})."
            )
        guess = np.asarray(self.initial_guess, dtype=np.float64)
        if guess.shape != (n_params,):
            raise ValueError(
                f"initial_guess shape {guess.shape} != ({n_params},)."
            )
        object.__setattr__(self, "initial_guess", guess)
        if self.n_observations < 1:
            raise ValueError(
                f"n_observations must be >= 1. Got {self.n_observations}."
            )
        if self.n_firms < 1:
            raise ValueError(
                f"n_firms must be >= 1. Got {self.n_firms}."
            )
        if self.n_periods < 1:
            raise ValueError(
                f"n_periods must be >= 1. Got {self.n_periods}."
            )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GMMStageResult:
    """Result of one GMM optimization stage."""

    beta: np.ndarray
    objective: float
    moment_vector: np.ndarray
    weighting_matrix: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    optimizer_nit: int
    optimizer_nfev: int
    trace: dict[str, list[Any]]


@dataclass
class GMMSolveResult:
    """Full two-step GMM result."""

    parameter_names: tuple[str, ...]
    moment_names: tuple[str, ...]
    n_observations: int
    stage1: GMMStageResult
    stage2: GMMStageResult
    omega_hat: np.ndarray
    omega_hat_condition_number: float
    jacobian: np.ndarray
    asymptotic_variance: np.ndarray
    standard_errors: np.ndarray
    j_statistic: float
    j_p_value: float
    j_df: int
    j_test_valid: bool

    @property
    def beta_hat(self) -> np.ndarray:
        return self.stage2.beta


@dataclass
class GMMMonteCarloSummary:
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
class GMMMonteCarloResult:
    parameter_names: tuple[str, ...]
    moment_names: tuple[str, ...]
    beta_true: np.ndarray
    replications: list[GMMSolveResult]
    summary: GMMMonteCarloSummary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_1d_float(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array. Got shape {arr.shape}.")
    return arr


@dataclass
class _CachedEvaluation:
    beta: np.ndarray
    moment_vector: np.ndarray


def _objective_value(
    moment_vector: np.ndarray,
    weighting_matrix: np.ndarray,
) -> float:
    return float(moment_vector.T @ weighting_matrix @ moment_vector)


# ---------------------------------------------------------------------------
# Part 1 -- Setup: cached evaluator
# ---------------------------------------------------------------------------

def _make_cached_evaluator(
    spec: GMMSpec,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Callable[[Sequence[float]], _CachedEvaluation]:
    """Build a cached g(beta) evaluator.

    Each evaluation is arithmetic on the data (no model solve), so caching
    matters only for deduplication during optimizer polling, not for cost.
    """

    cache: dict[bytes, _CachedEvaluation] = {}

    def evaluate_beta(beta: Sequence[float]) -> _CachedEvaluation:
        beta_arr = np.clip(_as_1d_float(beta, "beta"), lower, upper)
        key = beta_arr.tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached

        g = np.asarray(spec.compute_moments(beta_arr), dtype=np.float64)
        result = _CachedEvaluation(
            beta=beta_arr,
            moment_vector=g,
        )
        cache[key] = result
        return result

    return evaluate_beta


# ---------------------------------------------------------------------------
# Part 2 -- Estimation: one optimizer stage
# ---------------------------------------------------------------------------

def _call_optimizer(
    *,
    objective_fn: Callable[[Sequence[float]], float],
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    config: GMMRunConfig,
    optimizer_seed: tuple[int, int],
):
    """Dispatch to the configured scipy optimizer."""

    if config.optimizer_name == "dual_annealing":
        return dual_annealing(
            objective_fn,
            bounds=bounds,
            maxiter=config.optimizer_maxiter,
            seed=make_seed_int(optimizer_seed, "dual_annealing"),
            x0=np.asarray(x0, dtype=np.float64),
        )

    options = {"maxiter": config.optimizer_maxiter}
    kwargs: dict[str, Any] = {
        "fun": objective_fn,
        "x0": np.asarray(x0, dtype=np.float64),
        "method": config.optimizer_name,
        "options": options,
    }
    if config.optimizer_name != "Nelder-Mead":
        kwargs["bounds"] = bounds
    return minimize(**kwargs)


def _run_gmm_stage(
    evaluate_beta: Callable[[Sequence[float]], _CachedEvaluation],
    weighting_matrix: np.ndarray,
    x0: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    config: GMMRunConfig,
    optimizer_seed: tuple[int, int],
    *,
    local_only: bool = False,
) -> GMMStageResult:
    """One full GMM optimization stage.

    When ``local_only=False`` (stage 1): outer optimizer + Powell polish.
    When ``local_only=True``  (stage 2): Powell only from x0.
    """

    trace: dict[str, list[Any]] = {"beta": [], "objective": []}

    def objective_fn(beta: Sequence[float]) -> float:
        evaluation = evaluate_beta(beta)
        objective = _objective_value(evaluation.moment_vector, weighting_matrix)
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

    polish_result = minimize(
        objective_fn,
        x0=outer_best_x,
        method="Powell",
        bounds=bounds,
    )
    best_x = np.asarray(polish_result.x, dtype=np.float64)

    evaluation = evaluate_beta(best_x)
    return GMMStageResult(
        beta=evaluation.beta.copy(),
        objective=float(
            _objective_value(evaluation.moment_vector, weighting_matrix)
        ),
        moment_vector=evaluation.moment_vector.copy(),
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
# Part 3 -- Optimal weight
# ---------------------------------------------------------------------------

def _build_omega_hat_gmm(
    spec: GMMSpec,
    beta_hat: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Construct Omega_hat via Newey-West HAC at beta_hat.

    Accounts for within-firm serial correlation using Bartlett kernel
    weights with bandwidth L = floor(T^{1/3}).

    The contributions are reshaped to (N, T, R) to compute within-firm
    autocovariances.  Cross-sectional independence across firms is assumed.

    Returns (omega_hat, condition_number_before_ridge).
    """

    contributions_flat = np.asarray(
        spec.compute_moment_contributions(beta_hat), dtype=np.float64
    )
    N = spec.n_firms
    T = spec.n_periods
    n_moments = contributions_flat.shape[1]

    # Reshape to (N, T, R) for within-firm autocovariance computation.
    contributions = contributions_flat.reshape(N, T, n_moments)

    # Bandwidth: standard choice from Newey-West (1994).
    L = max(int(T ** (1.0 / 3.0)), 1)

    # Lag-0 autocovariance: Gamma_0 = (1/NT) sum_i sum_t g_it g_it'
    gamma_0 = np.einsum("ntr,nts->rs", contributions, contributions) / float(N * T)

    # Lag-l autocovariances with Bartlett kernel weights.
    omega_hat_raw = gamma_0.copy()
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        # Gamma_l = (1/NT) sum_i sum_{t=lag}^{T-1} g_it g_{i,t-lag}'
        gamma_l = np.einsum(
            "ntr,nts->rs",
            contributions[:, lag:, :],
            contributions[:, :-lag, :],
        ) / float(N * T)
        omega_hat_raw += w * (gamma_l + gamma_l.T)

    cond = float(np.linalg.cond(omega_hat_raw))
    omega_hat = omega_hat_raw + _OMEGA_RIDGE * np.eye(n_moments, dtype=np.float64)
    return omega_hat, cond


# ---------------------------------------------------------------------------
# Part 4 -- Inference
# ---------------------------------------------------------------------------

def _finite_difference_jacobian(
    evaluate_beta: Callable[[Sequence[float]], _CachedEvaluation],
    beta_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """R x K Jacobian of g(beta) via central finite differences.

    Requires 2K calls to ``evaluate_beta``.  Each call is cheap (arithmetic
    on data, no model solve).
    """

    beta_hat = np.asarray(beta_hat, dtype=np.float64)
    base_g = evaluate_beta(beta_hat).moment_vector
    n_moments = base_g.size
    n_params = beta_hat.size
    jacobian = np.zeros((n_moments, n_params), dtype=np.float64)

    for idx in range(n_params):
        base_step = max(_FD_REL_STEP * abs(beta_hat[idx]), _FD_ABS_STEP)
        max_down = beta_hat[idx] - lower[idx]
        max_up = upper[idx] - beta_hat[idx]

        if max_down > 0.0 and max_up > 0.0:
            step = min(base_step, max_down, max_up)
            beta_plus = beta_hat.copy()
            beta_minus = beta_hat.copy()
            beta_plus[idx] += step
            beta_minus[idx] -= step
            g_plus = evaluate_beta(beta_plus).moment_vector
            g_minus = evaluate_beta(beta_minus).moment_vector
            jacobian[:, idx] = (g_plus - g_minus) / (2.0 * step)
        elif max_up > 0.0:
            step = min(base_step, max_up)
            beta_plus = beta_hat.copy()
            beta_plus[idx] += step
            g_plus = evaluate_beta(beta_plus).moment_vector
            jacobian[:, idx] = (g_plus - base_g) / step
        elif max_down > 0.0:
            step = min(base_step, max_down)
            beta_minus = beta_hat.copy()
            beta_minus[idx] -= step
            g_minus = evaluate_beta(beta_minus).moment_vector
            jacobian[:, idx] = (base_g - g_minus) / step
        else:
            jacobian[:, idx] = 0.0

    return jacobian


def _compute_gmm_inference(
    beta_hat: np.ndarray,
    evaluate_beta: Callable[[Sequence[float]], _CachedEvaluation],
    omega_hat_inv: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_params: int,
    n_moments: int,
    n_observations: int,
) -> dict[str, Any]:
    """Part 4 -- inference at the final estimate.

    Computes:
      - Jacobian D (R x K) via central finite differences.
      - V = (D' Omega_hat^{-1} D)^{-1}
      - SE_k = sqrt(V_kk / NT)
      - J = NT * Q(beta_hat), J ~ chi2(R - K)
    """

    jacobian = _finite_difference_jacobian(
        evaluate_beta=evaluate_beta,
        beta_hat=beta_hat,
        lower=lower,
        upper=upper,
    )

    NT = float(n_observations)
    dwd = jacobian.T @ omega_hat_inv @ jacobian

    try:
        asymptotic_variance = np.linalg.inv(dwd)
    except np.linalg.LinAlgError:
        warnings.warn(
            "D'W D is singular: the Jacobian does not have full column rank "
            "at beta_hat, indicating weak or failed local identification. "
            "Standard errors are undefined.",
            RuntimeWarning,
            stacklevel=3,
        )
        asymptotic_variance = np.full(
            (n_params, n_params), np.nan, dtype=np.float64
        )

    standard_errors = np.sqrt(
        np.clip(np.diag(asymptotic_variance), 0.0, None) / NT
    )

    j_df = n_moments - n_params
    final_g = evaluate_beta(beta_hat).moment_vector
    q_final = _objective_value(final_g, omega_hat_inv)
    if j_df > 0:
        j_statistic = float(NT * q_final)
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
# Public API
# ---------------------------------------------------------------------------

def solve_gmm(
    spec: GMMSpec,
    config: GMMRunConfig | None = None,
) -> GMMSolveResult:
    """Run the generic two-step GMM estimator.

    Parts:
      Part 1 -- Build cached evaluator
      Part 2 -- Stage 1: minimize Q(beta) with W = I
      Part 3 -- Build Omega_hat from per-observation contributions
      Part 2 -- Stage 2: minimize Q(beta) with W = Omega_hat^{-1}
      Part 4 -- Jacobian, standard errors, J-statistic

    Each evaluation of Q is arithmetic on the data (no model solve), so
    typical wall time is seconds, not hours.
    """

    config = config or GMMRunConfig()
    n_params = len(spec.parameter_names)
    n_moments = len(spec.moment_names)

    lower = np.array([b[0] for b in spec.bounds], dtype=np.float64)
    upper = np.array([b[1] for b in spec.bounds], dtype=np.float64)

    evaluate_beta = _make_cached_evaluator(spec, lower, upper)

    # -- Stage 1: W = I --
    identity = np.eye(n_moments, dtype=np.float64)
    stage1_seed = fold_in_seed(config.master_seed, "gmm", "stage1")

    stage1 = _run_gmm_stage(
        evaluate_beta=evaluate_beta,
        weighting_matrix=identity,
        x0=spec.initial_guess.copy(),
        bounds=spec.bounds,
        config=config,
        optimizer_seed=stage1_seed,
    )

    # -- Build Omega_hat --
    omega_hat, omega_cond = _build_omega_hat_gmm(spec, stage1.beta)

    j_test_valid = True
    if omega_cond > _OMEGA_COND_WARN:
        warnings.warn(
            f"Omega_hat condition number is {omega_cond:.2e} (threshold "
            f"{_OMEGA_COND_WARN:.0e}). The J-test may be unreliable. "
            f"Consider increasing the panel size (NT).",
            RuntimeWarning,
            stacklevel=2,
        )
        j_test_valid = False

    try:
        omega_hat_inv = np.linalg.inv(omega_hat)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Omega_hat is singular even after ridge regularisation. "
            "Stage 2 will use W = I.",
            RuntimeWarning,
            stacklevel=2,
        )
        omega_hat_inv = identity.copy()
        j_test_valid = False

    # -- Stage 2: W = Omega_hat^{-1}, local only --
    stage2_seed = fold_in_seed(config.master_seed, "gmm", "stage2")

    stage2 = _run_gmm_stage(
        evaluate_beta=evaluate_beta,
        weighting_matrix=omega_hat_inv,
        x0=stage1.beta.copy(),
        bounds=spec.bounds,
        config=config,
        optimizer_seed=stage2_seed,
        local_only=True,
    )

    # -- Inference --
    inference = _compute_gmm_inference(
        beta_hat=stage2.beta,
        evaluate_beta=evaluate_beta,
        omega_hat_inv=omega_hat_inv,
        lower=lower,
        upper=upper,
        n_params=n_params,
        n_moments=n_moments,
        n_observations=spec.n_observations,
    )

    j_test_valid = j_test_valid and stage2.optimizer_success

    return GMMSolveResult(
        parameter_names=tuple(spec.parameter_names),
        moment_names=tuple(spec.moment_names),
        n_observations=spec.n_observations,
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


def validate_gmm(
    spec: GMMSpec,
    beta_true: Sequence[float],
    run_config: GMMRunConfig | None = None,
    mc_config: GMMMonteCarloConfig | None = None,
) -> GMMMonteCarloResult:
    """Run Monte Carlo validation for a GMM specification.

    Each replication generates an independent fake-real data panel at
    beta_true via ``spec.resample_spec``, then runs the full two-step GMM
    to recover beta_hat.

    Requires ``spec.resample_spec`` to be provided.
    """

    run_config = run_config or GMMRunConfig()
    mc_config = mc_config or GMMMonteCarloConfig()
    beta_true_arr = _as_1d_float(beta_true, "beta_true")

    if spec.resample_spec is None:
        raise ValueError(
            "validate_gmm requires spec.resample_spec to be provided."
        )

    replications: list[GMMSolveResult] = []
    for rep in range(mc_config.n_replications):
        rep_seed = fold_in_seed(
            run_config.master_seed, "gmm", "mc_rep", rep,
        )
        rep_spec = spec.resample_spec(beta_true_arr, rep_seed)
        result = solve_gmm(rep_spec, config=run_config)
        replications.append(result)

    beta_hats = np.stack(
        [result.beta_hat for result in replications], axis=0
    )
    standard_errors = np.stack(
        [result.standard_errors for result in replications], axis=0
    )
    if beta_hats.shape[0] > 1:
        sd = np.std(beta_hats, axis=0, ddof=1)
    else:
        sd = np.zeros(beta_hats.shape[1], dtype=np.float64)

    mean_beta_hat = np.mean(beta_hats, axis=0)
    bias = mean_beta_hat - beta_true_arr
    mean_standard_error = np.mean(standard_errors, axis=0)
    rmse = np.sqrt(
        np.mean((beta_hats - beta_true_arr[None, :]) ** 2, axis=0)
    )
    coverage_95 = np.mean(
        np.abs(beta_hats - beta_true_arr[None, :]) <= 1.96 * standard_errors,
        axis=0,
    )

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

    summary = GMMMonteCarloSummary(
        parameter_names=tuple(spec.parameter_names),
        beta_true=beta_true_arr.copy(),
        mean_beta_hat=mean_beta_hat,
        bias=bias,
        sd=sd,
        mean_standard_error=mean_standard_error,
        rmse=rmse,
        coverage_95=coverage_95,
        j_test_size=j_test_size,
    )
    return GMMMonteCarloResult(
        parameter_names=tuple(spec.parameter_names),
        moment_names=tuple(spec.moment_names),
        beta_true=beta_true_arr.copy(),
        replications=replications,
        summary=summary,
    )
