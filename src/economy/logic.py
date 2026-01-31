"""
src/economy/logic.py

The 'Physics Engine' for the Corporate Finance Model.
Defines core economic equations. Uses Automatic Differentiation for derivatives.

Design:
- Backend Agnostic: Accepts NumPy arrays or TF Tensors.
- AutoDiff: Uses tf.GradientTape to compute marginal values dynamically.
"""

import tensorflow as tf
from typing import Any, Union, Callable
from src.economy.parameters import EconomicParams

# Type alias: Accepts Tensors, NumPy arrays, or floats
Numeric = Union[tf.Tensor, Any]


# --- 1. Primitives (Production & Motion) ---

def production_function(k: Numeric, z: Numeric, params: EconomicParams) -> Numeric:
    """ Cobb-Douglas Production: y = z * k^theta """
    return z * (k ** params.theta)

def compute_investment(k: Numeric, k_next: Numeric, params: EconomicParams) -> Numeric:
    """ Law of motion: I = k' - (1 - delta) * k """
    return k_next - (1 - params.delta) * k


def adjustment_costs(
    k: Numeric,
    k_next: Numeric,
    params: EconomicParams,
    temperature: float,
    logit_clip: float
) -> Numeric:
    """
    Total Adjustment Costs (Convex + Fixed).
    
    Args:
        k: Current capital
        k_next: Next period capital
        params: Model parameters (cost_convex, cost_fixed)
        temperature: Temperature for fixed cost gate (if soft/ste)
    """
    from src.utils.annealing import indicator_abs_gt

    investment = compute_investment(k, k_next, params)
    safe_k = tf.maximum(k, 1e-8)

    # 1. Convex Costs: (φ/2) * (I^2 / k)
    adj_convex = (params.cost_convex / 2.0) * (investment ** 2) / safe_k

    # 2. Fixed Costs: cost_fixed * k * 1{|I/k| > eps}
    # Uses smooth gate for gradient flow during training
    # Note: Report specifies 1{|I/k| > eps}. We normalize I by k.
    i_normalized = investment / safe_k
    
    # Use epsilon=1e-6 as hardcoded threshold for now, matching previous default
    is_investing = indicator_abs_gt(
        i_normalized, 
        threshold=1e-8, 
        temperature=temperature, 
        logit_clip=logit_clip
    )
    
    adj_fixed = params.cost_fixed * safe_k * is_investing

    return adj_convex + adj_fixed

def compute_cash_flow_basic(
    k: Numeric, 
    k_next: Numeric, 
    z: Numeric, 
    params: EconomicParams,
    temperature: float,
    logit_clip: float
) -> Numeric:
    """
    Dividend d_t = Profit - Investment - Costs
    """
    profit = production_function(k, z, params)
    investment = compute_investment(k, k_next, params)
    costs_adjust = adjustment_costs(k, k_next, params, temperature=temperature, logit_clip=logit_clip)

    return profit - investment - costs_adjust


def external_financing_cost(
    e: Numeric,
    k: Numeric,
    params: EconomicParams,
    temperature: float,
    logit_clip: float
) -> Numeric:
    """
    External equity injection cost per report_brief.md lines 432-433.

    η(e) = (η₀ + η₁ · |e|) · 1_{e/k < -ε}

    The indicator is evaluated on normalized e/k to avoid sigmoid saturation
    when cash flows are large in levels.

    Reference:
        report_brief.md line 432-433:
        "Equity injection: σ(-(e/k + ε)/τ)"

    Args:
        e: Cash flow / dividends (levels)
        k: Current capital (for normalization of indicator)
        params: Contains cost_inject_fixed (η₀), cost_inject_linear (η₁)
        temperature: Sharpness of the < 0 gate
        logit_clip: Logit clipping bound for smooth indicator

    Returns:
        External financing cost (levels, same units as e)
    """
    from src.utils.annealing import indicator_lt

    # Normalize e by k for the indicator to avoid sigmoid saturation
    safe_k = tf.maximum(k, 1e-8)
    e_normalized = e / safe_k

    # Evaluate indicator on normalized value: 1_{e/k < -eps}
    is_negative = indicator_lt(e_normalized, threshold=1e-8, temperature=temperature, logit_clip=logit_clip)

    # Cost is still computed on levels (not normalized)
    return is_negative * (params.cost_inject_fixed + params.cost_inject_linear * tf.abs(e))


# --- 2. The AutoDiff Engine (Rigorous Derivatives) ---

def take_derivative(
    func: Callable,
    wrt_tensor: tf.Tensor,
    *args,
    **kwargs
) -> tf.Tensor:
    """
    Computes the gradient of a scalar function 'func' with respect to 'wrt_tensor'.

    Args:
        func: The function to differentiate (e.g., adjustment_costs).
        wrt_tensor: The tensor variable to differentiate against.
                    IMPORTANT: Must be a tf.Tensor, not a NumPy array.
        *args, **kwargs: Arguments to pass to 'func'.

    Returns:
        The gradient df/dx of the same shape as wrt_tensor.
    """
    with tf.GradientTape() as tape:
        tape.watch(wrt_tensor)
        y = func(*args, **kwargs)

    grad = tape.gradient(y, wrt_tensor)

    # Safety: If gradient is None (variable not used in calculation), return zeros
    if grad is None:
        return tf.zeros_like(wrt_tensor)

    return grad


# === SECTION 3: Risky Debt Logic (Default & Recovery) ===

def recovery_value(
        k_next: Numeric,
        z_next: Numeric,
        params: EconomicParams
) -> Numeric:
    """
    Calculates R(k', z'): The total value seized by creditors in default.
    Matches Equation (3.27).

    Formula: (1 - alpha) * [ (1 - tau) * pi(k', z') + (1 - delta) * k' ]
    """
    # 1. Future Operating Profit
    profit_next = production_function(k_next, z_next, params)

    # 2. Liquidation Value of Capital
    capital_val = (1 - params.delta) * k_next

    # 3. Total Pre-Friction Value
    # (1 - tau) * Profit + Capital
    gross_value = (1 - params.tax) * profit_next + capital_val

    # 4. Apply Deadweight Default Costs (alpha)
    # Ensure non-negative recovery (though k and pi usually > 0)
    recovery = (1 - params.cost_default) * gross_value

    return tf.maximum(recovery, 0.0)


def compute_lender_payoff(
        k_next: Numeric,
        b_next: Numeric,
        z_next: Numeric,
        is_default: Numeric,  # Boolean/Float Tensor (1.0 = Default, 0.0 = Solvent)
        params: EconomicParams
) -> Numeric:
    """
    Calculates the ex-post payoff to the lender in a specific future state (z').
    
    Per outline_v2.md line 308:
        Lender payoff = (1-D) * b'(1+r̃) + D * R(k', z')
    
    Note: We assume b' >= 0 (borrowing only, no savings).
    
    Args:
        k_next: Next period capital
        b_next: Next period debt (≥ 0)
        z_next: Next period productivity
        is_default: 1.0 if V(z', k', b') <= 0, else 0.0
        params: Model parameters
    
    Returns:
        Payoff to lender in this state
    """
    # Recovery Value R(k', z') - does NOT depend on b'
    recovery_val = recovery_value(k_next, z_next, params)
    
    # Payoff in default: lender seizes recovery value
    # Capped at face value (cannot recover more than owed)
    payoff_default = tf.minimum(recovery_val, b_next)
    
    # Total Payoff:
    # - Solvent: Get full face value b'
    # - Default: Get recovery value (capped at b')
    payoff = (1.0 - is_default) * b_next + is_default * payoff_default
    
    return payoff


# === SECTION 4: Euler Equation Primitives ===

def euler_chi(k: Numeric, k_next: Numeric, params: EconomicParams) -> Numeric:
    """
    Euler equation chi: marginal cost of investment.
    
    χ = 1 + ψ_I = 1 + φ₀ · I / k
    
    where I = k' - (1-δ)k
    
    Args:
        k: Current capital (levels)
        k_next: Next period capital (levels)
        params: Contains delta, cost_convex (φ₀)
    
    Returns:
        χ: Marginal adjustment cost (same shape as inputs)
    """
    I = compute_investment(k, k_next, params)
    safe_k = tf.maximum(k, 1e-8)
    psi_I = params.cost_convex * I / safe_k
    return 1.0 + psi_I


def euler_m(
    k_next: Numeric,
    k_next_next: Numeric,
    z_next: Numeric,
    params: EconomicParams
) -> Numeric:
    """
    Euler equation RHS component.
    
    m = π_k - ψ_k + (1-δ)χ
    
    where:
        π_k = θ·z·k^(θ-1)  (marginal product of capital)
        ψ_k = -(φ₀/2)·I²/k²  (marginal adjustment cost w.r.t. k)
        χ = euler_chi(k', k'')
    
    Args:
        k_next: Next period capital
        k_next_next: Capital after next (from policy at z')
        z_next: Next period productivity
        params: Model parameters
    
    Returns:
        m: Euler RHS term (same shape as inputs)
    """
    safe_k = tf.maximum(k_next, 1e-8)
    
    # π_k = d(z*k^θ)/dk = θ * z * k^(θ-1)
    pi_k = params.theta * z_next * tf.pow(safe_k, params.theta - 1)
    
    # I' = k'' - (1-δ)k'
    I_next = compute_investment(k_next, k_next_next, params)
    
    # ψ = (φ₀/2) * I²/k => ψ_k = -(φ₀/2) * I²/k²
    psi_k = -(params.cost_convex / 2.0) * tf.square(I_next) / tf.square(safe_k)
    
    # χ(k', z') = 1 + ψ_I'
    chi_next = euler_chi(k_next, k_next_next, params)
    
    return pi_k - psi_k + (1 - params.delta) * chi_next


# === SECTION 5: Pricing Primitives ===

# === SECTION 6: Terminal Value for LR Method ===

def compute_terminal_value(
    k_terminal: Numeric,
    z_terminal: Numeric,
    params: EconomicParams,
    beta: float,
    temperature: float,
    logit_clip: float
) -> Numeric:
    """
    Compute terminal value for LR method truncation correction.

    References:
        report_brief.md lines 503-514: Terminal Value for LR

    Assumes:
        - With long enough T horizon, k_T has converged to steady state
        - At steady state: k_SS = k_T = k_{T+1} = ... forever
        - Investment at steady state: I_SS = δ · k_SS
        - Terminal value is infinite sum of discounted cash flow at (k_SS, z_T)

    Formula:
        V^term(k_T, z_T) = e(k_SS, k_SS, z_T) / (1 - β)

    Args:
        k_terminal: Terminal capital k_T (batch_size, 1)
        z_terminal: Terminal productivity z_T (batch_size, 1)
        params: Economic parameters
        beta: Discount factor 1/(1+r)
        temperature: Gate temperature for adjustment costs
        logit_clip: Logit clipping for indicator functions

    Returns:
        Terminal value V^term (batch_size, 1)
    """
    # At steady state, k_SS = k_T
    k_ss = k_terminal

    # Compute steady-state cash flow: e(k_SS, k_SS, z_T)
    # Investment I_SS = k_SS - (1-δ)k_SS = δ · k_SS (replace depreciation only)
    e_ss = compute_cash_flow_basic(
        k_ss, k_ss, z_terminal, params,
        temperature=temperature,
        logit_clip=logit_clip
    )

    # V^term = e_ss / (1 - β)
    # This is the infinite geometric sum of discounted steady-state cash flows
    v_term = e_ss / (1.0 - beta)

    return v_term


def pricing_residual_zero_profit(
    b_next: Numeric,
    r_risk_free: float,
    r_tilde: Numeric,
    p_default: Numeric,
    recovery: Numeric
) -> Numeric:
    """
    Lender zero-profit pricing residual (rate-based formulation).

    f = b'(1+r) - [p^D · R + (1-p^D) · b'(1+r̃)]

    When f = 0, the bond is fairly priced (zero expected profit for lender).

    Note: Prefer pricing_residual_bond_price() for BR method as it uses
    bond price q directly per report_brief.md.

    Args:
        b_next: Next period debt (levels)
        r_risk_free: Risk-free rate (scalar float)
        r_tilde: Risky rate (batch,)
        p_default: Default probability (batch,)
        recovery: Recovery value R(k', z') (batch,)

    Returns:
        f: Pricing residual (batch,). Positive = lender earns profit.
    """
    lhs = b_next * (1 + r_risk_free)

    # Expected payoff to lender:
    # If default: get recovery R
    # If solvent: get full repayment b'*(1+r̃)
    rhs = p_default * recovery + (1 - p_default) * b_next * (1 + r_tilde)

    return lhs - rhs


def pricing_residual_bond_price(
    q: Numeric,
    b_next: Numeric,
    r_risk_free: float,
    p_default: Numeric,
    recovery: Numeric
) -> Numeric:
    """
    Lender zero-profit pricing residual (bond price formulation).

    f = q · b' · (1+r) - β · [(1-p) · b' + p · R]

    where β = 1/(1+r) is the discount factor.

    Reference:
        report_brief.md lines 1046-1059:
        "Compute Lender Payoff P_ell: P_ell = β[(1-p_ell)·b' + p_ell·R]"
        "f = (q * b' * (1+r) - P_ell)"

    When f = 0, the bond is fairly priced (zero expected profit for lender).

    Args:
        q: Bond price (batch,) in [0, 1/(1+r)]
        b_next: Next period debt face value (batch,)
        r_risk_free: Risk-free rate (scalar float)
        p_default: Default probability (batch,)
        recovery: Recovery value R(k', z') (batch,)

    Returns:
        f: Pricing residual (batch,). Positive = lender earns profit.
    """
    beta = 1.0 / (1.0 + r_risk_free)

    # LHS: Market value of bond * gross risk-free return
    # This represents what the lender pays (discounted) times the opportunity cost
    lhs = q * b_next * (1 + r_risk_free)

    # RHS: Expected discounted payoff to lender
    # If solvent: get full repayment b'
    # If default: get recovery R
    expected_payoff = (1 - p_default) * b_next + p_default * recovery
    rhs = beta * expected_payoff

    return lhs - rhs


def cash_flow_risky_debt_q(
    k: Numeric,
    k_next: Numeric,
    b: Numeric,
    b_next: Numeric,
    z: Numeric,
    q: Numeric,
    params: EconomicParams,
    temperature: float,
    logit_clip: float
) -> Numeric:
    """
    Cash flow for risky debt model, using bond price q directly.

    Reference:
        report_brief.md lines 795-799:
        e = (1-τ)π(k,z) - ψ(I,k) - I + b'·q + τ·r̃·b'·q/(1+r) - b

    Note: Since q = 1/(1+r̃), we have r̃ = 1/q - 1

    Args:
        k: Current capital
        k_next: Next period capital
        b: Current debt (repayment)
        b_next: Next period debt (face value)
        z: Current productivity
        q: Bond price in [0, 1/(1+r)]
        params: Economic parameters
        temperature: Gate temperature
        logit_clip: Logit clipping bound

    Returns:
        e: Cash flow (can be negative, triggering external financing cost)
    """
    # Basic cash flow components
    profit = production_function(k, z, params)
    investment = compute_investment(k, k_next, params)
    costs_adjust = adjustment_costs(k, k_next, params, temperature=temperature, logit_clip=logit_clip)

    # Debt proceeds: b' * q (market value of new debt)
    debt_proceeds = b_next * q

    # Implied risky rate: r̃ = 1/q - 1
    safe_q = tf.maximum(q, 1e-8)
    r_tilde = 1.0 / safe_q - 1.0

    # Tax shield: τ·r̃·b'·q / (1+r)
    # Simplifies to: τ·(1-q)·b' / (1+r) since r̃·q = 1 - q
    tax_shield = params.tax * r_tilde * debt_proceeds / (1 + params.r_rate)

    # Repayment of old debt
    debt_repay = b

    # Net cash flow
    cash_flow = (1 - params.tax) * profit - investment - costs_adjust + debt_proceeds + tax_shield - debt_repay

    return cash_flow