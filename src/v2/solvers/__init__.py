"""
src/v2/solvers/

Generic discrete solvers plus the canonical risky-debt nested VFI.

Public API:
    solve_vfi(env, train_dataset, config) -> Dict
    solve_pfi(env, train_dataset, config) -> Dict
    solve_nested_vfi(env, train_dataset, config) -> Dict
    solve_nested_vfi_tf(env, train_dataset, config, runtime_config) -> Dict
    solve_risky_debt(env, config) -> Dict
    solve_nikolov_to(config, params) -> Dict
    solve_nikolov_le(config, params) -> Dict
    solve_nikolov_mh(config, params) -> Dict
    VFIConfig, PFIConfig, NestedVFIConfig, NestedVFITFRuntimeConfig, GridConfig — configuration
    RiskyDebtSolverConfig — risky-debt solver configuration
    GridAxis              — per-variable grid spec
"""

from src.v2.solvers.config import (
    GridConfig,
    InterpNestedVFIConfig,
    NestedVFIConfig,
    NestedVFIGridConfig,
    NestedVFITFRuntimeConfig,
    PFIConfig,
    RiskyDebtSolverConfig,
    VFIConfig,
)
from src.v2.solvers.grid import GridAxis
from src.v2.solvers.vfi import solve_vfi
from src.v2.solvers.pfi import solve_pfi
from src.v2.solvers.nested_vfi_np import solve_nested_vfi
from src.v2.solvers.nested_vfi_tf import solve_nested_vfi_tf
from src.v2.solvers.nested_vfi_interp import solve_risky_debt
from src.v2.solvers.nikolov_to import compute_to_premiums, solve_nikolov_to
from src.v2.solvers.nikolov_le import solve_nikolov_le
from src.v2.solvers.nikolov_mh import solve_nikolov_mh
from src.v2.solvers.ceo_contract import solve_ceo_contract

__all__ = [
    "solve_vfi",
    "solve_pfi",
    "solve_nested_vfi",
    "solve_nested_vfi_tf",
    "solve_risky_debt",
    "solve_nikolov_to",
    "solve_nikolov_le",
    "solve_nikolov_mh",
    "solve_ceo_contract",
    "compute_to_premiums",
    "VFIConfig",
    "PFIConfig",
    "InterpNestedVFIConfig",
    "NestedVFIConfig",
    "NestedVFIGridConfig",
    "NestedVFITFRuntimeConfig",
    "RiskyDebtSolverConfig",
    "GridConfig",
    "GridAxis",
]
