"""Generic estimation tools for v2.

Methods
-------
smm  — Simulated Method of Moments (generic two-step estimator)
gmm  — Generalized Method of Moments (closed-form moment conditions)

Shared panel-statistics helpers (_panel_covariance, _panel_serial_correlation,
_panel_iv_first_diff_ar1) live in smm.py and are imported by environment
moment calculators.
"""

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
