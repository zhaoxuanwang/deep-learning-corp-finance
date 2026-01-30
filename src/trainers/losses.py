"""
src/trainers/losses.py

Loss functions for training DNN approximations.
Implements exact formulations from outline_v2.md.

Loss Types:
- LR: Lifetime Reward (maximize discounted rewards)
- ER: Euler Residual (AiO form for squared residual)
- BR: Bellman Residual (actor-critic with proper detachment)
- Price: Lender zero-profit residual (AiO form)
"""

import tensorflow as tf
from typing import Dict, Tuple


# =============================================================================
# LIFETIME REWARD LOSS (Sec. 1.2, 2.3)
# =============================================================================

def compute_lr_loss(
    rewards: tf.Tensor,
    beta: float,
    terminal_value: tf.Tensor = None
) -> tf.Tensor:
    """
    Compute Lifetime Reward loss with optional terminal value correction.

    L_LR = -mean(sum_t beta^t * e_t + beta^T * V^term)

    We MINIMIZE this, so we take the negative of the total discounted reward.

    Reference:
        report_brief.md lines 499-514: LR loss with terminal value

    Args:
        rewards: Tensor of shape (batch, T) with per-period rewards
        beta: Discount factor
        terminal_value: Optional terminal value V^term(k_T, z_T) of shape (batch, 1)

    Returns:
        Scalar loss
    """
    T = tf.shape(rewards)[1]
    # Discount factors: [1, beta, beta^2, ..., beta^(T-1)]
    t_indices = tf.cast(tf.range(T), tf.float32)
    discount_factors = tf.pow(beta, t_indices)

    # Discounted sum per trajectory
    discounted_rewards = rewards * discount_factors[None, :]
    total_rewards = tf.reduce_sum(discounted_rewards, axis=1, keepdims=True)

    # Add terminal value if provided (report lines 503-514)
    if terminal_value is not None:
        # V^term is discounted by beta^T
        T_float = tf.cast(T, tf.float32)
        terminal_discount = tf.pow(beta, T_float)
        total_rewards = total_rewards + terminal_discount * terminal_value

    # Loss = -mean(J)
    return -tf.reduce_mean(total_rewards)


def compute_lr_loss_risky(
    rewards: tf.Tensor,
    beta: float,
    price_loss: tf.Tensor,
    lambda_price: float
) -> tf.Tensor:
    """
    Compute Lifetime Reward loss for Risky Debt model.
    
    L_LR = -mean(sum_t beta^t * u_t) + lambda_price * L_price
    
    where u_t = e_t - eta(e_t) (cash flow minus equity issuance cost)
    
    Args:
        rewards: Tensor of shape (batch, T) with per-period utility
        beta: Discount factor
        price_loss: Price loss from AiO form
        lambda_price: Weight on price loss
    
    Returns:
        Scalar loss
    
    Reference: outline_v2.md lines 342-344
    """
    lr_component = compute_lr_loss(rewards, beta)
    return lr_component + lambda_price * price_loss


# =============================================================================
# EULER RESIDUAL LOSS (Sec. 1.3)
# =============================================================================

def compute_er_loss_aio(
    f1: tf.Tensor,
    f2: tf.Tensor
) -> tf.Tensor:
    """
    Compute Euler Residual loss using All-in-One (AiO) method.
    
    L_ER = mean(f1 * f2)
    
    where f = chi(k,z) - beta * m(k',z')
    and chi = 1 + psi_I(I, k), m = pi_k - psi_k + (1-delta)*chi
    
    The AiO trick: E[(f)^2] = E[f1 * f2] when f1, f2 are computed with
    independent draws of z'.
    
    Args:
        f1: Euler residual computed with shock draw 1
        f2: Euler residual computed with shock draw 2
    
    Returns:
        Scalar loss
    
    Reference: outline_v2.md lines 225-229
    """
    return tf.reduce_mean(f1 * f2)


def compute_lifetime_er_loss_aio(
    f1_path: tf.Tensor,
    f2_path: tf.Tensor
) -> tf.Tensor:
    """
    Compute Lifetime Euler Residual loss using AiO method over a trajectory.
    
    L_Lifetime_ER = mean_{i, t} (f1_{i,t} * f2_{i,t})
    
    Args:
        f1_path: Tensor (batch, T) of residuals with shock sequence 1
        f2_path: Tensor (batch, T) of residuals with shock sequence 2
    
    Returns:
        Scalar loss
    """
    return tf.reduce_mean(f1_path * f2_path)



# =============================================================================
# BELLMAN RESIDUAL LOSS (Sec. 1.4, 2.4)
# =============================================================================

def compute_br_critic_loss_aio(
    V_curr: tf.Tensor,
    y1: tf.Tensor,
    y2: tf.Tensor
) -> tf.Tensor:
    """
    Compute Bellman Residual critic loss using AiO method.
    
    L_critic = mean((V - y1) * (V - y2))
    
    where y = e + beta * V_next (with V_next DETACHED).
    
    IMPORTANT: y1, y2 should be computed with detached V_next values.
    The LHS V_curr remains trainable.
    
    Args:
        V_curr: Current value prediction (trainable)
        y1: Target computed with shock draw 1 (contains detached continuation)
        y2: Target computed with shock draw 2 (contains detached continuation)
    
    Returns:
        Scalar loss
    
    Reference: outline_v2.md lines 262-264
    """
    delta1 = V_curr - y1
    delta2 = V_curr - y2
    return tf.reduce_mean(delta1 * delta2)


def compute_br_critic_diagnostics(
    V_curr: tf.Tensor,
    y1: tf.Tensor,
    y2: tf.Tensor
) -> Dict[str, float]:
    """
    Compute diagnostic metrics for BR critic.
    
    The cross-product loss mean(delta1 * delta2) is unbiased but can be negative
    and non-monotone. These diagnostics provide more interpretable metrics.
    
    Args:
        V_curr: Current value prediction
        y1: Target computed with shock draw 1
        y2: Target computed with shock draw 2
    
    Returns:
        Dict with:
            1. cross_product (the actual loss): mean(delta1 * delta2)
            2. mse_proxy (always positive): mean(0.5 * (delta1^2 + delta2^2)) 
            3. mae_proxy (always positive): mean(0.5 * (|delta1| + |delta2|))
    """
    delta1 = V_curr - y1
    delta2 = V_curr - y2
    
    cross_product = float(tf.reduce_mean(delta1 * delta2))
    mse_proxy = float(tf.reduce_mean(0.5 * (delta1**2 + delta2**2)))
    mae_proxy = float(tf.reduce_mean(0.5 * (tf.abs(delta1) + tf.abs(delta2))))
    
    return {
        "cross_product": cross_product,
        "mse_proxy": mse_proxy,
        "mae_proxy": mae_proxy,
    }

def compute_br_actor_loss(
    e: tf.Tensor,
    V_next_1: tf.Tensor,
    V_next_2: tf.Tensor,
    beta: float
) -> tf.Tensor:
    """
    Compute Bellman Residual actor loss.
    
    L_actor = -mean(e + beta * (V'_1 + V'_2) / 2)
    
    Uses AVERAGED continuation values (not cross-product).
    This is different from critic which uses product of residuals.
    
    For actor: we want to MAXIMIZE the RHS, so we minimize the negative.
    
    IMPORTANT: V_next values should NOT be detached - gradients must flow
    through (k', b') to policy parameters.
    
    Args:
        e: Current period reward (batch,)
        V_next_1: Continuation value with shock draw 1
        V_next_2: Continuation value with shock draw 2
        beta: Discount factor
    
    Returns:
        Scalar loss
    
    Reference: outline_v2.md lines 267-271
    """
    # Average the two continuation values for variance reduction
    V_next_avg = 0.5 * (V_next_1 + V_next_2)
    
    # RHS of Bellman = e + beta * E[V']
    rhs = e + beta * V_next_avg
    
    # MAXIMIZE RHS => minimize -RHS
    return -tf.reduce_mean(rhs)


def compute_br_actor_loss_risky(
    e: tf.Tensor,
    eta: tf.Tensor,
    V_next_1: tf.Tensor,
    V_next_2: tf.Tensor,
    beta: float
) -> tf.Tensor:
    """
    Compute BR actor loss for Risky Debt model.
    
    L_actor = -mean(e - eta + beta * (V'_1 + V'_2) / 2)
    
    where V' = max{0, V_tilde'} (limited liability applied)
    
    Args:
        e: Cash flow (batch,)
        eta: External financing cost (batch,)
        V_next_1: Continuation value with shock draw 1 (after relu)
        V_next_2: Continuation value with shock draw 2 (after relu)
        beta: Discount factor
    
    Returns:
        Scalar loss
    """
    V_next_avg = V_next_1
    payout = e - eta
    rhs = payout + beta * V_next_avg
    return -tf.reduce_mean(rhs)


# =============================================================================
# PRICE LOSS (Sec. 2.2)
# =============================================================================

def compute_price_loss_aio(
    f1: tf.Tensor,
    f2: tf.Tensor
) -> tf.Tensor:
    """
    Compute price loss using AiO method.
    
    L_price = mean(f1 * f2)
    
    where f = b'(1+r) - [p^D * R + (1-p^D) * b'*(1+r_tilde)]
    
    Args:
        f1: Price residual with shock draw 1
        f2: Price residual with shock draw 2
    
    Returns:
        Scalar loss
    
    Reference: outline_v2.md lines 322-325
    """
    return tf.reduce_mean(f1 * f2)


def compute_price_residual(
    b_next: tf.Tensor,
    r_risk_free: float,
    r_tilde: tf.Tensor,
    p_default: tf.Tensor,
    recovery: tf.Tensor
) -> tf.Tensor:
    """
    Compute pricing residual for a single shock draw.
    
    f = b'(1+r) - [p^D * R + (1-p^D) * b'*(1+r_tilde)]
    
    Args:
        b_next: Next period debt
        r_risk_free: Risk-free rate
        r_tilde: Risky interest rate
        p_default: Smooth default probability
        recovery: Recovery value R(k', z')
    
    Returns:
        Residuals (batch,)
    
    Reference: outline_v2.md line 322
    """
    lhs = b_next * (1 + r_risk_free)
    
    # Payoff if default: recovery R
    # Payoff if solvent: full repayment b'*(1+r_tilde)
    rhs = p_default * recovery + (1 - p_default) * b_next * (1 + r_tilde)
    
    return lhs - rhs


# =============================================================================
# COMBINED LOSSES (Sec. 2.5)
# =============================================================================

def compute_critic_objective(
    br_critic_loss: tf.Tensor,
    price_loss: tf.Tensor,
    weight_br: float = 0.1
) -> tf.Tensor:
    """
    Combined critic objective for Risky Debt BR training.

    L_critic = weight_br * L_BR + L_price

    The price weight is implicitly 1.0 for better numerical stability.
    BR loss is typically 100x larger than price loss (sums lifetime rewards),
    so we normalize to price and weight down BR instead.

    Args:
        br_critic_loss: Bellman residual critic loss
        price_loss: Price loss
        weight_br: Weight on BR loss for critic (default 0.1)

    Returns:
        Combined loss

    Reference: report_brief.md lines 1062-1063
    """
    return weight_br * br_critic_loss + price_loss
