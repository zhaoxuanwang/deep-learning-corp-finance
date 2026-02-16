"""
src/ddp/__init__.py

Public API for DDP (Discrete Dynamic Programming) solvers.
"""

from src.ddp.ddp_config import DDPGridConfig

__all__ = [
    "DDPGridConfig",
    "BasicModelDDP",
    "RiskyModelDDP",
    "save_ddp_solution",
    "load_ddp_solution",
]


def __getattr__(name):
    # Lazy import heavy solver modules to keep package import lightweight.
    if name == "BasicModelDDP":
        from src.ddp.ddp_basic import BasicModelDDP
        return BasicModelDDP
    if name == "RiskyModelDDP":
        from src.ddp.ddp_risky import RiskyModelDDP
        return RiskyModelDDP
    if name == "save_ddp_solution":
        from src.ddp.checkpoints import save_ddp_solution
        return save_ddp_solution
    if name == "load_ddp_solution":
        from src.ddp.checkpoints import load_ddp_solution
        return load_ddp_solution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
