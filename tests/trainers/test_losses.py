"""
tests/trainers/test_losses.py

Tests for loss function computations.

Reference: outline_v2.md loss specifications
"""

import pytest
import tensorflow as tf
import numpy as np

from src.trainers.losses import (
    compute_lr_loss,
    compute_lr_loss_risky,
    compute_er_loss_aio,
    compute_br_critic_loss_aio,
    compute_br_actor_loss,
    compute_br_actor_loss_risky,
    compute_price_loss_aio,
    compute_critic_objective,
)
from src.economy.logic import (
    euler_chi,
    pricing_residual_zero_profit
)


# =============================================================================
# LR LOSS TESTS
# =============================================================================

class TestLRLoss:
    """Tests for Lifetime Reward loss."""
    
    def test_lr_loss_shape(self):
        """LR loss returns scalar."""
        rewards = tf.constant([[1.0, 0.9, 0.8], [1.0, 0.9, 0.8]])
        loss = compute_lr_loss(rewards, beta=0.96)
        assert loss.shape == ()
    
    def test_lr_loss_is_negative_mean_discounted_reward(self):
        """L_LR = -mean(sum_t beta^t * e_t)."""
        rewards = tf.constant([[1.0, 1.0]])  # batch=1, T=2
        beta = 0.5
        
        loss = compute_lr_loss(rewards, beta)
        
        # Expected: -(1.0 * 1 + 1.0 * 0.5) = -1.5
        expected = -1.5
        assert np.isclose(loss.numpy(), expected)
    
    def test_lr_loss_risky_includes_price(self):
        """Risky LR loss includes lambda_price * L_price."""
        rewards = tf.constant([[1.0, 1.0]])
        price_loss = tf.constant(2.0)
        lambda_price = 0.5
        
        loss = compute_lr_loss_risky(rewards, 0.96, price_loss, lambda_price)
        lr_only = compute_lr_loss(rewards, 0.96)
        
        expected = lr_only + 0.5 * 2.0
        assert np.isclose(loss.numpy(), expected.numpy())


# =============================================================================
# ER LOSS TESTS
# =============================================================================

class TestERLoss:
    """Tests for Euler Residual loss."""
    
    def test_er_aio_loss(self):
        """ER AiO loss = mean(f1 * f2)."""
        f1 = tf.constant([[1.0], [2.0]])
        f2 = tf.constant([[3.0], [4.0]])
        
        loss = compute_er_loss_aio(f1, f2)
        
        # (1*3 + 2*4) / 2 = (3 + 8) / 2 = 5.5
        expected = 5.5
        assert np.isclose(loss.numpy(), expected)
    
    def test_euler_chi(self):
        """euler_chi = 1 + psi_I where psi_I = phi_0 * I / k."""
        from src.economy.parameters import EconomicParams
        
        k = tf.constant([[1.0]])
        k_next = tf.constant([[1.2]])
        params = EconomicParams(delta=0.1, cost_convex=0.5)
        
        # I = 1.2 - 0.9 * 1.0 = 0.3
        # psi_I = 0.5 * 0.3 / 1.0 = 0.15
        # chi = 1.15
        chi = euler_chi(k, k_next, params)
        
        assert np.isclose(chi.numpy()[0, 0], 1.15)


# =============================================================================
# BR LOSS TESTS
# =============================================================================

class TestBRLoss:
    """Tests for Bellman Residual losses."""
    
    def test_critic_loss_aio(self):
        """Critic loss = mean((V - y1) * (V - y2))."""
        V = tf.constant([[1.0], [2.0]])
        y1 = tf.constant([[0.8], [1.8]])
        y2 = tf.constant([[0.9], [1.9]])
        
        loss = compute_br_critic_loss_aio(V, y1, y2)
        
        # deltas: (0.2, 0.2) and (0.1, 0.1)
        # products: 0.2*0.1 = 0.02 and 0.2*0.1 = 0.02
        # mean: 0.02
        expected = ((1.0-0.8)*(1.0-0.9) + (2.0-1.8)*(2.0-1.9)) / 2
        assert np.isclose(loss.numpy(), expected)
    
    def test_actor_loss_main_shock_only(self):
        """Actor loss uses only main shock continuation value."""
        e = tf.constant([[1.0], [1.0]])
        V_next = tf.constant([[2.0], [2.0]])
        beta = 0.96

        loss = compute_br_actor_loss(e, V_next, beta)

        # RHS = 1 + 0.96 * 2 = 2.92
        # loss = -mean(2.92) = -2.92
        expected = -2.92
        assert np.isclose(loss.numpy(), expected)
    
    def test_actor_loss_risky_includes_eta(self):
        """Risky actor loss subtracts eta and uses main shock only."""
        e = tf.constant([[1.0]])
        eta = tf.constant([[0.2]])
        V_next = tf.constant([[2.0]])
        beta = 0.96

        loss = compute_br_actor_loss_risky(e, eta, V_next, beta)

        # payout = e - eta = 1 - 0.2 = 0.8
        # RHS = 0.8 + 0.96 * 2 = 2.72
        # loss = -2.72
        expected = -2.72
        assert np.isclose(loss.numpy(), expected)


# =============================================================================
# PRICE LOSS TESTS
# =============================================================================

class TestPriceLoss:
    """Tests for price loss."""
    
    def test_price_loss_aio(self):
        """Price loss = mean(f1 * f2)."""
        f1 = tf.constant([[0.1], [0.2]])
        f2 = tf.constant([[0.1], [0.2]])
        
        loss = compute_price_loss_aio(f1, f2)
        
        expected = (0.01 + 0.04) / 2
        assert np.isclose(loss.numpy(), expected)
    
    def test_pricing_residual_formula(self):
        """f = b'(1+r) - [p^D*R + (1-p^D)*b'*(1+r_tilde)]."""
        b_next = tf.constant([[1.0]])
        r_rf = 0.04
        r_tilde = tf.constant([[0.06]])
        p_D = tf.constant([[0.2]])
        R = tf.constant([[0.5]])
        
        f = pricing_residual_zero_profit(b_next, r_rf, r_tilde, p_D, R)
        
        # LHS = 1.0 * 1.04 = 1.04
        # RHS = 0.2 * 0.5 + 0.8 * 1.0 * 1.06 = 0.1 + 0.848 = 0.948
        # f = 1.04 - 0.948 = 0.092
        lhs = 1.0 * 1.04
        rhs = 0.2 * 0.5 + 0.8 * 1.0 * 1.06
        expected = lhs - rhs
        
        assert np.isclose(f.numpy()[0, 0], expected)


# =============================================================================
# COMBINED OBJECTIVE TESTS
# =============================================================================

class TestCombinedObjectives:
    """Tests for combined critic objective."""

    def test_critic_objective(self):
        """L_critic = weight_br * L_BR + L_price."""
        br_loss = tf.constant(100.0)  # BR loss is typically much larger
        price_loss = tf.constant(2.0)
        weight_br = 0.5

        total = compute_critic_objective(br_loss, price_loss, weight_br)

        expected = 0.5 * 100.0 + 2.0  # = 52.0
        assert np.isclose(total.numpy(), expected)

    def test_critic_objective_default_weight(self):
        """L_critic with default weight_br=0.1."""
        br_loss = tf.constant(100.0)
        price_loss = tf.constant(2.0)

        total = compute_critic_objective(br_loss, price_loss)

        expected = 0.1 * 100.0 + 2.0  # = 12.0
        assert np.isclose(total.numpy(), expected)
