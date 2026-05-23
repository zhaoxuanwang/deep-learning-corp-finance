"""Generic estimation tools for v2.

Methods
-------
smm       — Simulated Method of Moments (generic two-step estimator)
gmm       — Generalized Method of Moments (closed-form moment conditions)
bayesian  — Bayesian estimation via TFP (Kalman or particle filter + NUTS
            or RW-MH); per-env factories live in sibling modules such as
            ``bayesian_basic_investment``.

Shared panel-statistics helpers (_panel_covariance, _panel_serial_correlation,
_panel_iv_first_diff_ar1) live in smm.py and are imported by environment
moment calculators.
"""

from src.v2.estimation.bayesian import (
    BayesianCoverageConfig,
    BayesianCoverageResult,
    BayesianMCMCResult,
    BayesianRunConfig,
    BayesianSpec,
    FilterKind,
    SamplerKind,
    run_coverage_check,
    run_mcmc,
)
from src.v2.estimation.beta_sampler import (
    BETA_DIM,
    BETA_DIM_NAMES,
    BetaSampler,
)
from src.v2.estimation.gmm import (
    GMMMonteCarloConfig,
    GMMMonteCarloResult,
    GMMMonteCarloSummary,
    GMMRunConfig,
    GMMSpec,
    GMMSolveResult,
    GMMStageResult,
    solve_gmm,
    validate_gmm,
)
from src.v2.estimation.smm import (
    SMMMonteCarloConfig,
    SMMMonteCarloResult,
    SMMMonteCarloSummary,
    SMMPanelMoments,
    SMMRunConfig,
    SMMSpec,
    SMMSolveResult,
    SMMStageResult,
    SMMTargetMoments,
    solve_smm,
    validate_smm,
)
__all__ = [
    "BayesianCoverageConfig",
    "BayesianCoverageResult",
    "BayesianMCMCResult",
    "BayesianRunConfig",
    "BayesianSpec",
    "FilterKind",
    "SamplerKind",
    "run_coverage_check",
    "run_mcmc",
    "BETA_DIM",
    "BETA_DIM_NAMES",
    "BetaSampler",
    "GMMMonteCarloConfig",
    "GMMMonteCarloResult",
    "GMMMonteCarloSummary",
    "GMMRunConfig",
    "GMMSpec",
    "GMMSolveResult",
    "GMMStageResult",
    "solve_gmm",
    "validate_gmm",
    "SMMMonteCarloConfig",
    "SMMMonteCarloResult",
    "SMMMonteCarloSummary",
    "SMMPanelMoments",
    "SMMRunConfig",
    "SMMSpec",
    "SMMSolveResult",
    "SMMStageResult",
    "SMMTargetMoments",
    "solve_smm",
    "validate_smm",
]
