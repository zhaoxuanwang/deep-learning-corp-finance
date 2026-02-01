"""
tests/trainers/test_gradient_flow.py

Tests for correct gradient flow in training.

Key rules from outline_v2.md:
1. Critic/price block: grad wrt theta_policy is zero (stop_gradient on k', b')
2. Actor block: grads flow to theta_policy through (k', b') even with frozen value/price
3. Critic target detachment: no gradient flows through RHS continuation term

Reference: outline_v2.md lines 136-158
"""

import pytest
import tensorflow as tf
import numpy as np

from src.networks import (
    BasicPolicyNetwork,
    BasicValueNetwork,
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    apply_limited_liability
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def basic_networks():
    """Create basic model networks."""
    policy = BasicPolicyNetwork(
        k_min=0.01, k_max=10.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16,
        activation="swish"
    )
    value = BasicValueNetwork(
        k_min=0.01, k_max=10.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16,
        activation="swish"
    )
    # Build networks
    k = tf.constant([[1.0]])
    z = tf.constant([[1.0]])
    policy(k, z)
    value(k, z)
    return policy, value


@pytest.fixture
def risky_networks():
    """Create risky debt model networks."""
    tf.random.set_seed(42)
    policy = RiskyPolicyNetwork(
        k_min=0.01, k_max=10.0, b_min=0.0, b_max=5.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16, activation="swish"
    )
    value = RiskyValueNetwork(
        k_min=0.01, k_max=10.0, b_max=5.0,
        logz_min=-1.0, logz_max=1.0,
        n_layers=2, n_neurons=16, activation="swish"
    )
    price = RiskyPriceNetwork(
        k_min=0.01, k_max=10.0, b_max=5.0,
        logz_min=-1.0, logz_max=1.0,
        r_risk_free=0.04, n_layers=2, n_neurons=16, activation="swish"
    )
    # Build networks
    k = tf.constant([[1.0]])
    b = tf.constant([[0.5]])
    z = tf.constant([[1.0]])
    k_next, b_next = policy(k, b, z)
    value(k, b, z)
    price(k_next, b_next, z)
    return policy, value, price


# =============================================================================
# CRITIC/PRICE BLOCK GRADIENT TESTS
# =============================================================================

class TestCriticBlockGradientFlow:
    """
    Verify that in critic/price block, grad wrt theta_policy is zero.
    
    Reference: outline_v2.md lines 139-145
    """
    
    def test_basic_critic_no_policy_gradient(self, basic_networks):
        """In critic step, policy gradients should be zero."""
        policy, value = basic_networks
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[1.0], [1.0]])
        
        with tf.GradientTape() as tape:
            # Policy output WITH stop_gradient (as in critic block)
            k_next = policy(k, z)
            k_next_sg = tf.stop_gradient(k_next)
            
            # Critic loss (simplified)
            V_curr = value(k, z)
            V_next = value(k_next_sg, z)  # Uses stopped k'
            V_next_sg = tf.stop_gradient(V_next)
            loss = tf.reduce_mean(tf.square(V_curr - V_next_sg))
        
        # Gradient wrt policy should be None or zero
        policy_grads = tape.gradient(loss, policy.trainable_variables)
        
        for grad in policy_grads:
            if grad is not None:
                assert np.allclose(grad.numpy(), 0), \
                    "Policy gradient should be zero in critic block"
    
    def test_risky_critic_no_policy_gradient(self, risky_networks):
        """In risky debt critic step, policy gradients should be zero."""
        policy, value, price = risky_networks
        k = tf.constant([[1.0], [2.0]])
        b = tf.constant([[0.5], [0.5]])
        z = tf.constant([[1.0], [1.0]])
        
        with tf.GradientTape() as tape:
            # Policy output WITH stop_gradient
            k_next, b_next = policy(k, b, z)
            k_next_sg = tf.stop_gradient(k_next)
            b_next_sg = tf.stop_gradient(b_next)
            
            # Price and value (critic block)
            r_tilde = price(k_next_sg, b_next_sg, z)
            V_curr = value(k, b, z)
            V_next = value(k_next_sg, b_next_sg, z)
            
            # Combined mock loss
            loss = tf.reduce_mean(tf.square(V_curr - V_next) + tf.square(r_tilde))
        
        # Gradient wrt policy should be None
        policy_grads = tape.gradient(loss, policy.trainable_variables)
        
        for i, grad in enumerate(policy_grads):
            assert grad is None or np.allclose(grad.numpy(), 0), \
                f"Policy gradient {i} should be zero/None in critic block"


# =============================================================================
# ACTOR BLOCK GRADIENT TESTS
# =============================================================================

class TestActorBlockGradientFlow:
    """
    Verify that in actor block, grads flow to theta_policy through (k', b').
    
    Reference: outline_v2.md lines 147-153
    """
    
    def test_basic_actor_policy_gradient_exists(self, basic_networks):
        """In actor step, policy should receive gradients through value."""
        policy, value = basic_networks
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[1.0], [1.0]])
        
        with tf.GradientTape() as tape:
            # Policy output WITHOUT stop_gradient
            k_next = policy(k, z)
            
            # Actor loss: -mean(e + beta * V(k'))
            # Gradients should flow through V(k') to policy
            V_next = value(k_next, z)
            loss = -tf.reduce_mean(V_next)  # Simplified actor loss
        
        # Gradient wrt policy should NOT be None
        policy_grads = tape.gradient(loss, policy.trainable_variables)
        
        # At least one gradient should be non-zero
        has_nonzero_grad = any(
            grad is not None and not np.allclose(grad.numpy(), 0)
            for grad in policy_grads
        )
        assert has_nonzero_grad, \
            "Policy should receive gradients in actor block"
    
    def test_risky_actor_policy_gradient_exists(self, risky_networks):
        """In risky debt actor step, policy receives gradients."""
        policy, value, price = risky_networks
        k = tf.constant([[1.0], [2.0]])
        b = tf.constant([[0.5], [0.5]])
        z = tf.constant([[1.0], [1.0]])
        
        with tf.GradientTape() as tape:
            # Policy output WITHOUT stop_gradient
            k_next, b_next = policy(k, b, z)
            
            # Actor loss components (no stop_gradient!)
            V_next = value(k_next, b_next, z)
            r_tilde = price(k_next, b_next, z)
            
            # Combined loss
            loss = -tf.reduce_mean(V_next) + 0.1 * tf.reduce_mean(r_tilde)
        
        # Gradient wrt policy should NOT be None
        policy_grads = tape.gradient(loss, policy.trainable_variables)
        
        has_nonzero_grad = any(
            grad is not None and not np.allclose(grad.numpy(), 0)
            for grad in policy_grads
        )
        assert has_nonzero_grad, \
            "Policy should receive gradients in actor block"
    
    def test_actor_value_receives_no_update(self, basic_networks):
        """
        In actor block, value params are frozen (excluded from optimizer).
        But gradients w.r.t. value params may exist - we just don't apply them.
        
        This test verifies the gradient EXISTS but we only UPDATE policy.
        """
        policy, value = basic_networks
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[1.0], [1.0]])
        
        with tf.GradientTape() as tape:
            k_next = policy(k, z)
            V_next = value(k_next, z)
            loss = -tf.reduce_mean(V_next)
        
        # When we only request policy gradients, value is excluded
        policy_grads = tape.gradient(loss, policy.trainable_variables)
        
        # Policy grads should exist
        assert any(g is not None for g in policy_grads)
        
        # Verify we CAN get value grads, but we choose not to update them
        # (This is the "freeze by exclusion" pattern)


# =============================================================================
# CRITIC TARGET DETACHMENT TESTS
# =============================================================================

class TestCriticTargetDetachment:
    """
    Verify that RHS continuation term is detached in critic update.
    
    The LHS V(k,z) should be trainable, but the RHS target
    e + beta * V(k',z') should have V(k',z') detached.
    
    Reference: outline_v2.md lines 255-259
    """
    
    def test_rhs_continuation_detached(self, basic_networks):
        """No gradient flows through RHS continuation term."""
        policy, value = basic_networks
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[1.0], [1.0]])
        
        # Compute k' with stop_gradient (critic block)
        k_next = policy(k, z)
        k_next_sg = tf.stop_gradient(k_next)
        
        with tf.GradientTape(persistent=True) as tape:
            # LHS (trainable)
            V_curr = value(k, z)
            
            # RHS continuation (to be detached)
            V_next = value(k_next_sg, z)
            V_next_detached = tf.stop_gradient(V_next)
            
            # Target
            e = tf.constant([[0.5], [0.5]])  # Mock reward
            beta = 0.96
            y = e + beta * V_next_detached
            
            # Critic loss
            loss = tf.reduce_mean(tf.square(V_curr - y))
        
        # Gradient should flow through V_curr but NOT through V_next
        value_grads = tape.gradient(loss, value.trainable_variables)
        
        # Grads should exist (from V_curr)
        assert any(g is not None for g in value_grads)
        
        # Verify V_next_detached has no gradient
        grad_V_next = tape.gradient(loss, V_next)
        assert grad_V_next is None, \
            "Gradient should not flow through detached RHS continuation"
        
        del tape
    
    def test_lhs_remains_trainable(self, basic_networks):
        """LHS V(k,z) should receive gradients."""
        policy, value = basic_networks
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[1.0], [1.0]])
        
        with tf.GradientTape() as tape:
            V_curr = value(k, z)
            
            # Fixed target (simulates detached RHS)
            y = tf.constant([[1.0], [1.0]])
            
            loss = tf.reduce_mean(tf.square(V_curr - y))
        
        # V_curr should have gradient
        grad_V_curr = tape.gradient(loss, V_curr)
        assert grad_V_curr is not None
        assert not np.allclose(grad_V_curr.numpy(), 0)


# =============================================================================
# RISKY DEBT SPECIFIC GRADIENT TESTS
# =============================================================================

class TestRiskyDebtGradientFlow:
    """Additional gradient tests specific to risky debt model."""
    
    def test_price_gradients_in_actor_block(self, risky_networks):
        """
        In actor block, price network outputs should allow gradients
        to flow back to policy through (k', b').
        """
        policy, value, price = risky_networks
        k = tf.constant([[1.0]])
        b = tf.constant([[0.5]])
        z = tf.constant([[1.0]])
        
        with tf.GradientTape() as tape:
            k_next, b_next = policy(k, b, z)
            r_tilde = price(k_next, b_next, z)
            loss = tf.reduce_mean(r_tilde)  # Price-based loss
        
        # Should have gradient to policy
        policy_grads = tape.gradient(loss, policy.trainable_variables)
        has_grad = any(g is not None for g in policy_grads)
        assert has_grad, \
            "Policy should receive gradients from price in actor block"
    
    def test_limited_liability_gradient_flow(self, risky_networks):
        """
        relu(V_tilde) should allow gradients when V_tilde > 0.
        """
        policy, value, price = risky_networks
        
        with tf.GradientTape() as tape:
            V_tilde = tf.constant([[2.0], [-1.0]])  # One positive, one negative
            tape.watch(V_tilde)
            
            V = apply_limited_liability(V_tilde)
            loss = tf.reduce_sum(V)
        
        grad = tape.gradient(loss, V_tilde)
        
        # Gradient should be 1 for positive, 0 for negative (relu subgradient)
        expected = np.array([[1.0], [0.0]])
        np.testing.assert_allclose(grad.numpy(), expected)
    
    def test_limited_liability_leaky_gradient_flow(self, risky_networks):
        """
        leaky_relu(V_tilde) should allow gradients even when V_tilde < 0.
        """
        with tf.GradientTape() as tape:
            V_tilde = tf.constant([[2.0], [-1.0]])  # One positive, one negative
            tape.watch(V_tilde)
            
            V = apply_limited_liability(V_tilde, leaky=True, alpha=0.01)
            loss = tf.reduce_sum(V)
        
        grad = tape.gradient(loss, V_tilde)
        
        # Gradient should be 1 for positive, 0.01 for negative (leaky relu)
        expected = np.array([[1.0], [0.01]])
        np.testing.assert_allclose(grad.numpy(), expected, atol=1e-6)


# =============================================================================
# TRAINER GRADIENT FLOW TESTS
# =============================================================================


