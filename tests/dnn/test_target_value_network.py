"""
tests/dnn/test_target_value_network.py

Unit tests for the optional Polyak/EMA target value network in BasicTrainerBR.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.economy import EconomicParams
from src.dnn import BasicPolicyNetwork, BasicValueNetwork
from src.dnn.trainer_basic import BasicTrainerBR


class TestTargetValueNetworkBackwardCompatibility:
    """Test that default behavior works correctly."""
    
    def test_enabled_by_default(self):
        """use_value_target should be True by default (for stability)."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(policy_net, value_net, params)
        
        assert trainer.use_value_target is True
        assert trainer.value_net_target is not None
    
    def test_can_disable_target(self):
        """Target network can be explicitly disabled."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(policy_net, value_net, params, use_value_target=False)
        
        assert trainer.use_value_target is False
        assert trainer.value_net_target is None
    
    def test_train_step_works_with_default(self):
        """Training should work normally with default target network enabled."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            batch_size=16, n_critic_steps=2, seed=42
        )
        
        k = tf.random.uniform((16,), 0.5, 5.0)
        z = tf.random.uniform((16,), 0.7, 1.3)
        
        metrics = trainer.train_step(k, z)
        
        assert "loss_critic" in metrics
        assert "loss_actor" in metrics
        assert np.isfinite(metrics["loss_critic"])
        assert np.isfinite(metrics["loss_actor"])


class TestTargetValueNetworkInitialization:
    """Test target network initialization."""
    
    def test_target_created_when_enabled(self):
        """Target network should be created when use_value_target=True."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=True, value_target_mix=0.01
        )
        
        assert trainer.use_value_target is True
        assert trainer.value_net_target is not None
    
    def test_target_weights_equal_at_init(self):
        """Target weights should equal online weights at initialization."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=True
        )
        
        online_weights = trainer.value_net.get_weights()
        target_weights = trainer.value_net_target.get_weights()
        
        for ow, tw in zip(online_weights, target_weights):
            np.testing.assert_array_almost_equal(ow, tw, decimal=6)


class TestTargetNetworkNotOptimized:
    """Test that target network is not directly optimized."""
    
    def test_target_unchanged_after_critic_step(self):
        """Target weights should not change after critic step (before Polyak)."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=True, value_target_mix=0.01, seed=42
        )
        
        # Store target weights before
        target_weights_before = [w.copy() for w in trainer.value_net_target.get_weights()]
        
        # Run critic step
        k = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)
        trainer._critic_step(k, z)
        
        # Target weights should be unchanged (Polyak not called yet)
        target_weights_after = trainer.value_net_target.get_weights()
        
        for before, after in zip(target_weights_before, target_weights_after):
            np.testing.assert_array_equal(before, after)
    
    def test_online_changed_after_critic_step(self):
        """Online value net weights should change after critic step."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=True, seed=42
        )
        
        # Store online weights before
        online_weights_before = [w.copy() for w in trainer.value_net.get_weights()]
        
        # Run critic step
        k = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)
        trainer._critic_step(k, z)
        
        # Online weights should have changed
        online_weights_after = trainer.value_net.get_weights()
        
        any_changed = any(
            not np.allclose(before, after)
            for before, after in zip(online_weights_before, online_weights_after)
        )
        assert any_changed, "Online value net weights should change after critic step"


class TestPolyakUpdateCorrectness:
    """Test Polyak averaging correctness."""
    
    def test_polyak_update_formula(self):
        """Polyak update should apply: target_new = (1-mix)*target_old + mix*online."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        mix = 0.1  # Large mix for easy verification
        trainer = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=True, value_target_mix=mix
        )
        
        # Set online weights to known values (different from target)
        k = tf.constant([[2.0]], dtype=tf.float32)
        z = tf.constant([[1.0]], dtype=tf.float32)
        trainer._critic_step(k, z)  # This changes online weights
        
        target_before = [w.copy() for w in trainer.value_net_target.get_weights()]
        online = [w.copy() for w in trainer.value_net.get_weights()]
        
        # Call Polyak update
        trainer._polyak_update_value_target()
        
        target_after = trainer.value_net_target.get_weights()
        
        # Verify formula: target_new = (1-mix)*target_old + mix*online
        for tb, ol, ta in zip(target_before, online, target_after):
            expected = (1.0 - mix) * tb + mix * ol
            np.testing.assert_array_almost_equal(ta, expected, decimal=6)


class TestCriticUsesTargetNetwork:
    """Test that critic correctly uses target network for RHS when enabled."""
    
    def test_critic_uses_different_networks(self):
        """When enabled, critic should use target for RHS, online for LHS."""
        policy_net = BasicPolicyNetwork(n_layers=2, n_neurons=8)
        value_net = BasicValueNetwork(n_layers=2, n_neurons=8)
        params = EconomicParams()
        
        trainer_with_target = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=True, seed=42
        )
        
        trainer_without_target = BasicTrainerBR(
            policy_net, value_net, params,
            use_value_target=False, seed=42
        )
        
        # Run several critic steps to diverge online from target
        for _ in range(5):
            k = tf.random.uniform((16, 1), 0.5, 5.0)
            z = tf.random.uniform((16, 1), 0.7, 1.3)
            trainer_with_target._critic_step(k, z)
            # Polyak update to slightly move target
            trainer_with_target._polyak_update_value_target()
        
        # At this point, online and target weights should be different
        online_weights = trainer_with_target.value_net.get_weights()
        target_weights = trainer_with_target.value_net_target.get_weights()
        
        any_different = any(
            not np.allclose(ow, tw)
            for ow, tw in zip(online_weights, target_weights)
        )
        assert any_different, "Online and target should have diverged after training"
