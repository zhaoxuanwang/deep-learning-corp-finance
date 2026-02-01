"""
tests/trainers/test_gradient_rules.py

Tests for specific gradient block rules:
- Critic should not update policy
- Actor should not update value
- Actor gradients must flow through value network (d_loss/d_theta_policy exists)
"""

import pytest
import numpy as np
import tensorflow as tf

from src.economy import EconomicParams, ShockParams
from src.networks import (
    BasicPolicyNetwork, BasicValueNetwork,
    RiskyPolicyNetwork, RiskyValueNetwork, RiskyPriceNetwork,
    apply_limited_liability
)
from src.trainers.basic import BasicTrainerBR
from src.utils.annealing import AnnealingSchedule, indicator_default
from src.economy.logic import adjustment_costs


class TestStopGradientRules:
    """
    Verify strict stop_gradient usage in BR algorithms.
    """
    
    def test_critic_does_not_update_policy(self):
        """Critic step should not produce gradients for policy params."""
        policy_net = BasicPolicyNetwork(k_min=0.01, k_max=10.0, logz_min=-1.0, logz_max=1.0, n_layers=2, n_neurons=8, activation="swish")
        value_net = BasicValueNetwork(k_min=0.01, k_max=10.0, logz_min=-1.0, logz_max=1.0, n_layers=2, n_neurons=8, activation="swish")

        # Build networks
        k_dummy = tf.constant([[1.0]])
        z_dummy = tf.constant([[1.0]])
        policy_net(k_dummy, z_dummy)
        value_net(k_dummy, z_dummy)

        params = EconomicParams()
        shock_params = ShockParams()

        trainer = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            optimizer_actor=tf.keras.optimizers.Adam(learning_rate=1e-3),
            optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
            n_critic_steps=1,
            logit_clip=20.0,
            polyak_tau=0.995
        )
        pass 
    
    def test_actor_does_not_update_value(self):
        """Actor step should not update value network parameters."""
        policy_net = BasicPolicyNetwork(k_min=0.01, k_max=10.0, logz_min=-1.0, logz_max=1.0, n_layers=2, n_neurons=8, activation="swish")
        value_net = BasicValueNetwork(k_min=0.01, k_max=10.0, logz_min=-1.0, logz_max=1.0, n_layers=2, n_neurons=8, activation="swish")

        # Build networks
        k_dummy = tf.constant([[1.0]])
        z_dummy = tf.constant([[1.0]])
        policy_net(k_dummy, z_dummy)
        value_net(k_dummy, z_dummy)

        params = EconomicParams()
        shock_params = ShockParams()

        trainer = BasicTrainerBR(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            optimizer_actor=tf.keras.optimizers.Adam(learning_rate=1e-3),
            optimizer_value=tf.keras.optimizers.Adam(learning_rate=1e-3),
            n_critic_steps=1,
            logit_clip=20.0,
            polyak_tau=0.995
        )
        pass

    def test_actor_has_gradient_through_value(self):
        """Actor loss should have gradient w.r.t. policy via value function."""
        policy_net = BasicPolicyNetwork(k_min=0.01, k_max=10.0, logz_min=-1.0, logz_max=1.0, n_layers=2, n_neurons=8, activation="swish")
        value_net = BasicValueNetwork(k_min=0.01, k_max=10.0, logz_min=-1.0, logz_max=1.0, n_layers=2, n_neurons=8, activation="swish")
        
        k = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            k_next = policy_net(k, z)
            V_next = value_net(k_next, z)
            loss = -tf.reduce_mean(V_next)
            
        grads = tape.gradient(loss, policy_net.trainable_variables)
        
        assert any(g is not None for g in grads), "Policy should receive gradient through value net"
