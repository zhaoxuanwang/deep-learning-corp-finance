"""
src/economy/__init__.py

Public API for economic model primitives.
"""

from src.economy.parameters import (
    EconomicParams,
    ShockParams,
    convert_to_tf,
)

# NOTE: DDPGridConfig is now in src.ddp.ddp_config
# Import directly: from src.ddp import DDPGridConfig

from src.economy.shocks import (
    draw_AiO_shocks,
    draw_initial_states,
)

from src.economy.logic import (
    # Primitives
    production_function,
    compute_investment,
    adjustment_costs,
    # Cash flows
    compute_cash_flow_basic,
    cash_flow_risky_debt_q,
    external_financing_cost,
    # Recovery & default
    recovery_value,
    compute_lender_payoff,
    # Euler primitives
    euler_chi,
    euler_m,
    # Pricing
    pricing_residual_zero_profit,
    pricing_residual_bond_price,
    # AutoDiff helper
    take_derivative,
)

from src.economy.rng import (
    SeedSchedule,
    SeedScheduleConfig,
    VariableID,
)

from src.economy.data_generator import (
    DataGenerator,
    create_data_generator,
)
from src.economy.data import (
    DatasetBundle,
    load_dataset_bundle,
    canonicalize_bounds,
)
from src.economy.value_scale import (
    FrictionlessValueBenchmark,
    compute_frictionless_value_benchmark,
)



__all__ = [
    # Parameters
    "EconomicParams",
    "ShockParams",
    "convert_to_tf",
    # Shocks
    "draw_AiO_shocks",
    "draw_initial_states",
    # Primitives
    "production_function",
    "compute_investment",
    "adjustment_costs",
    # Cash flows
    "compute_cash_flow_basic",
    "cash_flow_risky_debt_q",
    "external_financing_cost",
    # Recovery & default
    "recovery_value",
    "compute_lender_payoff",
    # Euler
    "euler_chi",
    "euler_m",
    # Pricing
    "pricing_residual_zero_profit",
    "pricing_residual_bond_price",
    # AutoDiff
    "take_derivative",
    # RNG & Reproducibility
    "SeedSchedule",
    "SeedScheduleConfig",
    "VariableID",
    # Data Generation
    "DataGenerator",
    "create_data_generator",
    "DatasetBundle",
    "load_dataset_bundle",
    "canonicalize_bounds",
    # BR normalization benchmark
    "FrictionlessValueBenchmark",
    "compute_frictionless_value_benchmark",
]
