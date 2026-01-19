"""
tests/dnn/test_gradient_rules.py

Tests for gradient flow correctness in BR training.
Verifies stop-gradient rules and hard indicator handling.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.economy import EconomicParams
from src.dnn import (
    BasicPolicyNetwork, BasicValueNetwork,
    RiskyPolicyNetwork, RiskyValueNetwork, RiskyPriceNetwork,
    BasicTrainerBR,
    apply_limited_liability,
    DefaultSmoothingSchedule,
)
from src.economy.logic import investment_gate_ste, adjustment_costs


# =============================================================================
# Stop-Gradient Correctness
# =============================================================================

class TestStopGradientRules:
    """Verify stop-gradient is applied correctly in BR training."""
    
    def test_critic_does_not_update_policy(self):
        """Critic step should not produce gradients for policy params."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            batch_size=16, n_critic_steps=1, seed=42
        )
        
        # Get initial policy weights
        policy_weights_before = [w.numpy().copy() for w in policy_net.trainable_variables]
        
        # Run critic step only
        k = tf.random.uniform((16, 1), 0.5, 5.0)
        z = tf.random.uniform((16, 1), 0.7, 1.3)
        trainer._critic_step(k, z)
        
        # Policy weights should be unchanged
        policy_weights_after = [w.numpy() for w in policy_net.trainable_variables]
        
        for before, after in zip(policy_weights_before, policy_weights_after):
            np.testing.assert_array_equal(before, after, 
                "Policy weights changed during critic step!")
    
    def test_actor_does_not_update_value(self):
        """Actor step should not update value network parameters."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            batch_size=16, n_critic_steps=1, seed=42
        )
        
        # Get initial value weights
        value_weights_before = [w.numpy().copy() for w in value_net.trainable_variables]
        
        # Run actor step only
        k = tf.random.uniform((16, 1), 0.5, 5.0)
        z = tf.random.uniform((16, 1), 0.7, 1.3)
        trainer._actor_step(k, z)
        
        # Value weights should be unchanged
        value_weights_after = [w.numpy() for w in value_net.trainable_variables]
        
        for before, after in zip(value_weights_before, value_weights_after):
            np.testing.assert_array_equal(before, after,
                "Value weights changed during actor step!")
    
    def test_actor_has_gradient_through_value(self):
        """Actor loss should have gradient w.r.t. policy via value function."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        k = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Policy output
            k_next = policy_net(k, z)
            
            # Continuation value (NOT detached for actor)
            V_next = value_net(k_next, z)
            
            # Actor loss approximation
            loss = -tf.reduce_mean(V_next)
        
        # Gradient should exist for policy
        grads = tape.gradient(loss, policy_net.trainable_variables)
        
        assert all(g is not None for g in grads), \
            "Actor should have gradients through value function"
        assert any(tf.reduce_sum(tf.abs(g)).numpy() > 0 for g in grads), \
            "At least some gradients should be non-zero"


# =============================================================================
# Hard Indicator Handling
# =============================================================================

class TestHardIndicatorGradients:
    """Verify hard indicators have appropriate gradient handling."""
    
    def test_ste_gate_has_nonzero_gradient(self):
        """STE gate should have non-zero gradient."""
        I = tf.Variable([0.3], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            gate = investment_gate_ste(I, mode="ste")
            loss = tf.reduce_sum(gate)
        
        grad = tape.gradient(loss, I)
        
        assert grad is not None, "STE should provide gradient"
        assert grad.numpy()[0] != 0, "STE gradient should be non-zero"
    
    def test_hard_gate_has_zero_gradient(self):
        """Hard gate should have zero gradient."""
        I = tf.Variable([0.3], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            gate = investment_gate_ste(I, mode="hard")
            loss = tf.reduce_sum(gate)
        
        grad = tape.gradient(loss, I)
        
        # Hard gate gradient is zero
        assert grad is None or grad.numpy()[0] == 0, \
            "Hard gate should have zero gradient"
    
    def test_leaky_relu_preserves_gradient(self):
        """Leaky ReLU should preserve gradients when V_tilde < 0."""
        V_tilde = tf.Variable([-0.5], dtype=tf.float32)  # Negative
        
        with tf.GradientTape() as tape:
            V = apply_limited_liability(V_tilde, leaky=True)
            loss = tf.reduce_sum(V)
        
        grad = tape.gradient(loss, V_tilde)
        
        assert grad is not None, "Leaky ReLU should provide gradient"
        assert grad.numpy()[0] != 0, "Leaky ReLU gradient should be non-zero"
    
    def test_relu_kills_gradient_when_negative(self):
        """Standard ReLU should have zero gradient when V_tilde < 0."""
        V_tilde = tf.Variable([-0.5], dtype=tf.float32)  # Negative
        
        with tf.GradientTape() as tape:
            V = apply_limited_liability(V_tilde, leaky=False)
            loss = tf.reduce_sum(V)
        
        grad = tape.gradient(loss, V_tilde)
        
        # ReLU gradient is zero when input < 0
        assert grad is None or grad.numpy()[0] == 0, \
            "ReLU should have zero gradient when input < 0"
    
    def test_limited_liability_defaults_to_relu(self):
        """Default apply_limited_liability should use ReLU (not leaky)."""
        V_tilde = tf.constant([-0.5], dtype=tf.float32)
        
        # Default call (no leaky argument)
        V_default = apply_limited_liability(V_tilde)
        
        # Explicit ReLU
        V_relu = apply_limited_liability(V_tilde, leaky=False)
        
        # Default should equal ReLU behavior
        np.testing.assert_array_equal(V_default.numpy(), V_relu.numpy(),
            "Default should be ReLU (leaky=False)")
        
        # Specifically, output should be 0 for negative input
        assert V_default.numpy()[0] == 0, \
            "Default ReLU should output 0 for V_tilde < 0"


# =============================================================================
# Boundary Cases
# =============================================================================

class TestBoundaryCases:
    """Verify no NaN/Inf at boundary values."""
    
    def test_investment_gate_at_zero(self):
        """Investment gate should be stable at I = 0."""
        I_values = [0.0, 1e-8, -1e-8, 1e-6, -1e-6]
        
        for I_val in I_values:
            I = tf.constant([I_val], dtype=tf.float32)
            gate = investment_gate_ste(I, mode="ste")
            
            assert tf.reduce_all(tf.math.is_finite(gate)).numpy(), \
                f"Gate should be finite at I = {I_val}"
    
    def test_limited_liability_at_zero(self):
        """Limited liability should be stable at V_tilde = 0."""
        V_values = [0.0, 1e-8, -1e-8]
        
        for V_val in V_values:
            V_tilde = tf.constant([V_val], dtype=tf.float32)
            
            V_hard = apply_limited_liability(V_tilde, leaky=False)
            V_leaky = apply_limited_liability(V_tilde, leaky=True)
            
            assert tf.reduce_all(tf.math.is_finite(V_hard)).numpy(), \
                f"Hard limited liability should be finite at V_tilde = {V_val}"
            assert tf.reduce_all(tf.math.is_finite(V_leaky)).numpy(), \
                f"Leaky limited liability should be finite at V_tilde = {V_val}"
    
    def test_default_smoothing_at_zero(self):
        """Default smoothing should be stable at V_tilde = 0."""
        schedule = DefaultSmoothingSchedule(epsilon_D_0=0.1)
        
        V_values = [0.0, 1e-8, -1e-8]
        
        for V_val in V_values:
            V_tilde = tf.constant([V_val], dtype=tf.float32)
            p_D = schedule.compute_default_prob(V_tilde)
            
            assert tf.reduce_all(tf.math.is_finite(p_D)).numpy(), \
                f"Default prob should be finite at V_tilde = {V_val}"
            
            # At V_tilde = 0, p_D should be 0.5 (centered)
            if V_val == 0.0:
                np.testing.assert_allclose(p_D.numpy(), 0.5, atol=0.01,
                    err_msg="p_D should be ~0.5 at V_tilde = 0")
    
    def test_adjustment_costs_at_zero_investment(self):
        """Adjustment costs should be stable when I = 0."""
        params = EconomicParams(cost_fixed=0.1, cost_convex=0.1)
        
        k = tf.constant([2.0], dtype=tf.float32)
        delta = params.delta
        k_next_no_invest = (1 - delta) * k  # I = 0 exactly
        
        cost = adjustment_costs(k, k_next_no_invest, params)
        
        assert tf.reduce_all(tf.math.is_finite(cost)).numpy(), \
            "Adjustment cost should be finite at I = 0"
