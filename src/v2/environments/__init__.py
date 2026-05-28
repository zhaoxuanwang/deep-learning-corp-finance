from src.v2.environments.base import MDPEnvironment
from src.v2.environments.ceo_contract import (
    CEOContractGridConfig,
    CEOContractParams,
    build_ceo_grids,
    drift_z,
    flow_payoff,
    optimal_effort,
    optimal_manipulation,
    optimal_sigma_z,
    terminal_value,
)
from src.v2.environments.nikolov import (
    NikolovGridConfig,
    NikolovParams,
    ar1_z_bounds,
    build_nikolov_grids,
    calibrated_k_bounds,
    conservative_debt_bound,
    reference_capital,
)

__all__ = [
    "MDPEnvironment",
    "CEOContractGridConfig",
    "CEOContractParams",
    "build_ceo_grids",
    "drift_z",
    "flow_payoff",
    "optimal_effort",
    "optimal_manipulation",
    "optimal_sigma_z",
    "terminal_value",
    "NikolovGridConfig",
    "NikolovParams",
    "ar1_z_bounds",
    "build_nikolov_grids",
    "calibrated_k_bounds",
    "conservative_debt_bound",
    "reference_capital",
]

try:
    from src.v2.environments.basic_investment import (
        BasicInvestmentEnv,
        BasicInvestmentSMMPanelData,
        BasicInvestmentSMMSolverBundle,
        BasicInvestmentSMMSolverConfig,
    )
    from src.v2.environments.parameterized_basic_investment import (
        ParameterizedBasicInvestmentEnv,
    )
    from src.v2.environments.risky_debt import RiskyDebtEnv, RiskyDebtSMMPanelData
except ModuleNotFoundError as exc:
    if exc.name not in {"tensorflow_probability", "tf_keras"}:
        raise
    # Keep lightweight modules such as environments.nikolov importable when
    # optional estimation dependencies are unavailable in a local environment.
    pass
else:
    __all__.extend(
        [
            "BasicInvestmentEnv",
            "BasicInvestmentSMMPanelData",
            "BasicInvestmentSMMSolverConfig",
            "BasicInvestmentSMMSolverBundle",
            "ParameterizedBasicInvestmentEnv",
            "RiskyDebtEnv",
            "RiskyDebtSMMPanelData",
        ]
    )
