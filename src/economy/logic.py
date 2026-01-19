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

def investment_gate_ste(
    investment: Numeric,
    eps: float = 1e-6,
    temp: float = 0.1,
    mode: str = "ste"
) -> Numeric:
    """
    Investment indicator gate with optional STE for gradient flow.
    
    Forward: Returns 1 when |I| > eps, 0 otherwise (hard gate).
    Backward: Uses smooth sigmoid surrogate for gradient flow (STE mode).
    
    Args:
        investment: I = k' - (1-δ)k, can be positive (investment) or negative (disinvestment)
        eps: Threshold for "investing" (default 1e-6)
        temp: Sigmoid temperature for soft gate (default 0.1, should be ~ typical |I| scale)
        mode: Gate mode:
            - "hard": Pure hard gate, zero gradient (for evaluation)
            - "ste": Straight-Through Estimator (forward=hard, backward=soft gradient)
            - "soft": Pure smooth sigmoid (for debugging)
    
    Returns:
        g: Gate value in [0, 1]
    
    Note:
        STE pattern: g = stop_gradient(g_hard - g_soft) + g_soft
        - Forward pass uses g_hard (exact discrete gate)
        - Backward pass uses grad(g_soft) (smooth gradient)
    """
    abs_I = tf.abs(investment)
    
    # Hard gate (exact economic logic)
    g_hard = tf.cast(abs_I > eps, tf.float32)
    
    if mode == "hard":
        return g_hard
    
    # Soft gate (smooth sigmoid for gradients)
    # sigmoid((|I| - eps) / temp): centered at |I| = eps, sharpness controlled by temp
    g_soft = tf.nn.sigmoid((abs_I - eps) / temp)
    
    if mode == "soft":
        return g_soft
    
    # STE: forward uses hard, backward uses soft gradient
    # tf.stop_gradient(g_hard - g_soft) has value (g_hard - g_soft) but zero gradient
    # Adding g_soft gives: forward = g_hard, backward = grad(g_soft)
    return tf.stop_gradient(g_hard - g_soft) + g_soft


def adjustment_costs(
    k: Numeric,
    k_next: Numeric,
    params: EconomicParams,
    fixed_cost_gate: str = "ste"
) -> Numeric:
    """
    Total Adjustment Costs (Convex + Fixed).
    
    Args:
        k: Current capital
        k_next: Next period capital
        params: Model parameters (cost_convex, cost_fixed)
        fixed_cost_gate: Gate mode for fixed cost indicator:
            - "hard": Zero gradient (for evaluation)
            - "ste": Straight-Through Estimator (default, for training)
            - "soft": Pure smooth gate (for debugging)
    
    Returns:
        Total adjustment cost: convex + fixed
    """
    investment = compute_investment(k, k_next, params)
    safe_k = tf.maximum(k, 1e-8)

    # 1. Convex Costs: (φ/2) * (I^2 / k)
    adj_convex = (params.cost_convex / 2.0) * (investment ** 2) / safe_k

    # 2. Fixed Costs: cost_fixed * k * 1{|I| > eps}
    # Uses STE gate for gradient flow during training
    is_investing = investment_gate_ste(investment, mode=fixed_cost_gate)
    adj_fixed = params.cost_fixed * safe_k * is_investing

    return adj_convex + adj_fixed

def compute_cash_flow_basic(k: Numeric, k_next: Numeric, z: Numeric, params: EconomicParams) -> Numeric:
    """
    Dividend d_t = Profit - Investment - Costs
    """
    profit = production_function(k, z, params)
    investment = compute_investment(k, k_next, params)
    costs_adjust = adjustment_costs(k, k_next, params)

    return profit - investment - costs_adjust


def external_financing_cost(
    e: Numeric,
    params: EconomicParams,
    gate_mode: str = "ste"
) -> Numeric:
    """
    External equity injection cost per outline_v2.md.
    
    η(e) = (η₀ + η₁ · |e|) · 1_{e < 0}
    
    Args:
        e: Cash flow / dividends (levels), shape (batch,) or (batch, 1)
        params: Contains cost_inject_fixed (η₀), cost_inject_linear (η₁)
        gate_mode: "ste" (default, gradient flow), "hard" (no gradient), "soft"
    
    Returns:
        η: Financing cost, same shape as e. Zero when e >= 0.
    
    Note:
        Uses STE by default so gradients flow through the indicator during
        training. For evaluation, use gate_mode="hard".
    """
    from src.economy.indicators import ste_gate_lt
    
    is_negative = ste_gate_lt(e, threshold=0.0, mode=gate_mode)
    return is_negative * (params.cost_inject_fixed + params.cost_inject_linear * tf.abs(e))


def cash_flow_risky_debt(
    k: Numeric,
    k_next: Numeric,
    b: Numeric,
    b_next: Numeric,
    z: Numeric,
    r_tilde: Numeric,
    params: EconomicParams,
) -> Numeric:
    """
    Cash flow for risky debt model, using interest rate directly.
    
    e = (1-τ)π - ψ - I + b'/(1+r̃) + τ·r̃·b'/[(1+r̃)(1+r)] - b
    
    Args:
        k: Current capital (levels)
        k_next: Next period capital (levels)
        b: Current debt (levels)
        b_next: Next period debt (levels)
        z: Productivity shock (levels)
        r_tilde: Risky interest rate (NOT bond price)
        params: Model parameters
    
    Returns:
        e: Cash flow (same shape as inputs)
    """
    # Basic cash flow components
    profit = production_function(k, z, params)
    investment = compute_investment(k, k_next, params)
    costs_adjust = adjustment_costs(k, k_next, params)

    # Bond price from rate: q = 1/(1+r̃)
    safe_r = tf.maximum(1.0 + r_tilde, 1e-8)
    bond_price = 1.0 / safe_r

    # Market value of the debt issuance
    debt_proceeds = b_next * bond_price

    # Repayment of old debt
    debt_repay = b

    # Tax Shield: τ·r̃·b'/[(1+r̃)(1+r)]
    tax_shield_debt = params.tax * r_tilde * debt_proceeds / (1 + params.r_rate)

    # Net cash flow with risky debt
    cash_flow = (1 - params.tax) * profit - investment - costs_adjust + debt_proceeds + tax_shield_debt - debt_repay

    return cash_flow


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

def pricing_residual_zero_profit(
    b_next: Numeric,
    r_risk_free: float,
    r_tilde: Numeric,
    p_default: Numeric,
    recovery: Numeric
) -> Numeric:
    """
    Lender zero-profit pricing residual.
    
    f = b'(1+r) - [p^D · R + (1-p^D) · b'(1+r̃)]
    
    When f = 0, the bond is fairly priced (zero expected profit for lender).
    
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