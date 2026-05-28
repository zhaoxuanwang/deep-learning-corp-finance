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

from src.v2.estimation.beta_sampler import (
    BETA_DIM,
    BETA_DIM_NAMES,
    BetaSampler,
)
from src.v2.estimation.empirical_policy import (
    EmpiricalPolicyConfig,
    EmpiricalPolicyFit,
    EmpiricalPolicyOutcome,
    empirical_policy_coefficients,
    fit_empirical_policy,
    partial_dependence_slices,
    predict_policy_slices,
)
from src.v2.estimation.nikolov_ii import (
    NikolovIIConfig,
    build_nikolov_grid_config,
    build_nikolov_params,
    compute_h_data,
    make_nikolov_ii_spec,
    make_nikolov_ii_target,
    pack_beta,
    run_nikolov_ii,
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
    "BETA_DIM",
    "BETA_DIM_NAMES",
    "BetaSampler",
    "EmpiricalPolicyConfig",
    "EmpiricalPolicyFit",
    "EmpiricalPolicyOutcome",
    "empirical_policy_coefficients",
    "fit_empirical_policy",
    "partial_dependence_slices",
    "predict_policy_slices",
    "NikolovIIConfig",
    "build_nikolov_grid_config",
    "build_nikolov_params",
    "compute_h_data",
    "make_nikolov_ii_spec",
    "make_nikolov_ii_target",
    "pack_beta",
    "run_nikolov_ii",
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

try:
    from src.v2.estimation.bayesian import (
        BayesianCoverageConfig,
        BayesianCoverageResult,
        BayesianMCMCResult,
        BayesianRunConfig,
        BayesianSpec,
        FilterKind,
        PosteriorPredictiveConfig,
        PosteriorPredictiveResult,
        PriorSensitivityResult,
        SamplerKind,
        default_rw_scales_from_prior,
        run_coverage_check,
        run_mcmc,
        run_posterior_predictive_check,
        run_prior_sensitivity,
    )
except ImportError as exc:
    # ModuleNotFoundError covers a missing optional stack; a plain ImportError
    # covers a version-incompatible one (tensorflow_probability raises
    # ImportError when the installed TensorFlow is too old, with exc.name=None).
    _msg = str(exc).lower()
    _tfp_related = (
        getattr(exc, "name", None) in {"tensorflow_probability", "tf_keras"}
        or "tensorflow probability" in _msg
        or "tensorflow_probability" in _msg
        or "tf_keras" in _msg
    )
    if not _tfp_related:
        raise
    # Bayesian estimators require a tensorflow_probability/tf_keras stack that
    # is compatible with the installed TensorFlow.  Keep GMM/SMM, empirical
    # policy, indirect inference, and model modules importable when that
    # optional stack is missing or version-incompatible.
    pass
else:
    __all__ = [
        "BayesianCoverageConfig",
        "BayesianCoverageResult",
        "BayesianMCMCResult",
        "BayesianRunConfig",
        "BayesianSpec",
        "FilterKind",
        "PosteriorPredictiveConfig",
        "PosteriorPredictiveResult",
        "PriorSensitivityResult",
        "SamplerKind",
        "default_rw_scales_from_prior",
        "run_coverage_check",
        "run_mcmc",
        "run_posterior_predictive_check",
        "run_prior_sensitivity",
    ] + __all__
