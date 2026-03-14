"""
src/v2/solvers/

Generic discrete solvers (VFI, PFI) for MDPEnvironment.

Public API:
    solve_vfi(env, train_dataset, config) -> Dict
    solve_pfi(env, train_dataset, config) -> Dict
    VFIConfig, PFIConfig, GridConfig       — configuration
    GridAxis                               — per-variable grid specification
"""

from src.v2.solvers.config import GridConfig, VFIConfig, PFIConfig
from src.v2.solvers.grid import GridAxis
from src.v2.solvers.vfi import solve_vfi
from src.v2.solvers.pfi import solve_pfi

__all__ = [
    "solve_vfi",
    "solve_pfi",
    "VFIConfig",
    "PFIConfig",
    "GridConfig",
    "GridAxis",
]
