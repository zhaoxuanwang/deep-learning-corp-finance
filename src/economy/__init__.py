"""
src/economy/__init__.py

Public API for economic model primitives.
"""

from src.economy.parameters import (
    EconomicParams,
    convert_to_tf,
)

# NOTE: DDPGridConfig is now in src.ddp.ddp_config
# Import directly: from src.ddp import DDPGridConfig

from src.economy.shocks import (
    simulate_productivity_next,
    draw_initial_states,
    initialize_markov_process,
    get_sampling_bounds,
)

from src.economy.logic import (
    # Primitives
    production_function,
    compute_investment,
    adjustment_costs,
    investment_gate_ste,
    # Cash flows
    compute_cash_flow_basic,
    cash_flow_risky_debt,
    external_financing_cost,
    # Recovery & default
    recovery_value,
    compute_lender_payoff,
    # Euler primitives
    euler_chi,
    euler_m,
    # Pricing
    pricing_residual_zero_profit,
    # AutoDiff helper
    take_derivative,
)

from src.economy.indicators import (
    ste_gate_abs_gt,
    ste_gate_lt,
    hard_gate_abs_gt,
    hard_gate_lt,
)

__all__ = [
    # Parameters
    "EconomicParams",
    "convert_to_tf",
    # Shocks
    "simulate_productivity_next",
    "draw_initial_states",
    "initialize_markov_process",
    "get_sampling_bounds",
    # Primitives
    "production_function",
    "compute_investment",
    "adjustment_costs",
    "investment_gate_ste",
    # Cash flows
    "compute_cash_flow_basic",
    "cash_flow_risky_debt",
    "external_financing_cost",
    # Recovery & default
    "recovery_value",
    "compute_lender_payoff",
    # Euler
    "euler_chi",
    "euler_m",
    # Pricing
    "pricing_residual_zero_profit",
    # AutoDiff
    "take_derivative",
]

