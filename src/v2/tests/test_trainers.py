"""Tests for v2 trainer modules: core utilities, LR, ER, BRM, MVE.

All training smoke tests use very small configs (few steps, small batch)
to verify correctness without long runtimes.
"""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.networks.policy import PolicyNetwork
from src.v2.networks.critic import CriticNetwork
from src.v2.networks.state_value import StateValueNetwork
from src.v2.trainers.core import (
    ReplayBuffer, SeedSchedule, polyak_update, hard_update,
    collect_transitions, generate_eval_dataset,
    evaluate_euler_residual, evaluate_bellman_residual,
    evaluate_bellman_residual_v,
)
from src.v2.trainers.config import (
    LRConfig, ERConfig, BRMConfig, MVEConfig, OptimizerConfig,
)
from src.v2.trainers.lr import train_lr
from src.v2.trainers.er import train_er
from src.v2.trainers.brm import train_brm
from src.v2.experimental.mve import train_mve


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def env():
    """BasicInvestmentEnv with default parameters."""
    return BasicInvestmentEnv()


@pytest.fixture
def policy(env):
    """Small policy network for smoke tests."""
    net = PolicyNetwork(
        state_dim=env.state_dim(), action_dim=env.action_dim(),
        **env.action_spec(), n_layers=2, n_neurons=32,
    )
    dummy = tf.zeros((1, env.state_dim()))
    net(dummy)
    return net


@pytest.fixture
def critic(env):
    """Small critic network for smoke tests."""
    net = CriticNetwork(
        state_dim=env.state_dim(), action_dim=env.action_dim(),
        n_layers=2, n_neurons=32,
    )
    dummy_s = tf.zeros((1, env.state_dim()))
    dummy_a = tf.zeros((1, env.action_dim()))
    net(dummy_s, dummy_a)
    return net


@pytest.fixture
def value_net(env):
    """Small state-value network for smoke tests."""
    net = StateValueNetwork(
        state_dim=env.state_dim(),
        n_layers=2, n_neurons=32,
    )
    dummy = tf.zeros((1, env.state_dim()))
    net(dummy)
    return net


def _small_optimizer():
    """Shared small optimizer config for fast smoke tests."""
    return OptimizerConfig(learning_rate=1e-3, clipnorm=100.0)


# =============================================================================
# ReplayBuffer
# =============================================================================

class TestReplayBuffer:
    """Tests for the ReplayBuffer."""

    def test_empty_buffer(self):
        """New buffer has size 0."""
        buf = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        assert len(buf) == 0

    def test_add_and_size(self):
        """Adding transitions increases buffer size."""
        buf = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        s = np.random.randn(10, 2).astype(np.float32)
        a = np.random.randn(10, 1).astype(np.float32)
        r = np.random.randn(10).astype(np.float32)
        s_next = np.random.randn(10, 2).astype(np.float32)
        eps = np.random.randn(10, 1).astype(np.float32)
        buf.add(s, a, r, s_next, eps)
        assert len(buf) == 10

    def test_capacity_limit(self):
        """Buffer does not exceed capacity."""
        buf = ReplayBuffer(capacity=20, state_dim=2, action_dim=1)
        for _ in range(5):
            s = np.random.randn(10, 2).astype(np.float32)
            a = np.random.randn(10, 1).astype(np.float32)
            r = np.random.randn(10).astype(np.float32)
            s_next = np.random.randn(10, 2).astype(np.float32)
            eps = np.random.randn(10, 1).astype(np.float32)
            buf.add(s, a, r, s_next, eps)
        assert len(buf) == 20

    def test_sample_returns_tf_tensors(self):
        """Sampled batch values are TF tensors."""
        buf = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        s = np.random.randn(50, 2).astype(np.float32)
        a = np.random.randn(50, 1).astype(np.float32)
        r = np.random.randn(50).astype(np.float32)
        s_next = np.random.randn(50, 2).astype(np.float32)
        eps = np.random.randn(50, 1).astype(np.float32)
        buf.add(s, a, r, s_next, eps)

        batch = buf.sample(16)
        assert isinstance(batch["s"], tf.Tensor)
        assert batch["s"].shape == (16, 2)
        assert batch["a"].shape == (16, 1)
        assert batch["r"].shape == (16, 1)
        assert batch["s_next"].shape == (16, 2)

    def test_sample_deterministic_with_seed(self):
        """Same seed produces same sample."""
        buf = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)
        s = np.random.randn(50, 2).astype(np.float32)
        a = np.random.randn(50, 1).astype(np.float32)
        r = np.random.randn(50).astype(np.float32)
        s_next = np.random.randn(50, 2).astype(np.float32)
        eps = np.random.randn(50, 1).astype(np.float32)
        buf.add(s, a, r, s_next, eps)

        b1 = buf.sample(8, seed=(1, 2))
        b2 = buf.sample(8, seed=(1, 2))
        np.testing.assert_allclose(b1["s"].numpy(), b2["s"].numpy())


# =============================================================================
# SeedSchedule
# =============================================================================

class TestSeedSchedule:
    """Tests for the SeedSchedule."""

    def test_training_seeds_unique(self):
        """Different steps produce different seeds."""
        ss = SeedSchedule((20, 26))
        s1 = ss.training_seed(0, SeedSchedule.VAR_STATE)
        s2 = ss.training_seed(1, SeedSchedule.VAR_STATE)
        assert s1 != s2

    def test_different_vars_different_seeds(self):
        """Different variable types produce different seeds at same step."""
        ss = SeedSchedule((20, 26))
        s1 = ss.training_seed(0, SeedSchedule.VAR_STATE)
        s2 = ss.training_seed(0, SeedSchedule.VAR_SHOCK_MAIN)
        assert s1 != s2

    def test_eval_seed_deterministic(self):
        """Eval seeds are deterministic."""
        ss = SeedSchedule((20, 26))
        e1 = ss.eval_seed(SeedSchedule.VAR_STATE)
        e2 = ss.eval_seed(SeedSchedule.VAR_STATE)
        assert e1 == e2


# =============================================================================
# Target network operations
# =============================================================================

class TestTargetOps:
    """Tests for polyak_update and hard_update."""

    def test_hard_update_copies_weights(self, policy, env):
        """hard_update makes target weights equal to source."""
        target = PolicyNetwork(
            state_dim=env.state_dim(), action_dim=env.action_dim(),
            **env.action_spec(), n_layers=2, n_neurons=32,
            name="policy_target",
        )
        dummy = tf.zeros((1, env.state_dim()))
        target(dummy)

        hard_update(policy, target)
        for src, tgt in zip(policy.trainable_variables,
                            target.trainable_variables):
            np.testing.assert_allclose(
                src.numpy(), tgt.numpy(), atol=1e-7)

    def test_polyak_update_interpolates(self, policy, env):
        """polyak_update moves target toward source."""
        target = PolicyNetwork(
            state_dim=env.state_dim(), action_dim=env.action_dim(),
            **env.action_spec(), n_layers=2, n_neurons=32,
            name="policy_target",
        )
        dummy = tf.zeros((1, env.state_dim()))
        target(dummy)

        # Record initial target weights.
        before = [v.numpy().copy() for v in target.trainable_variables]

        # Polyak with tau=0.5.
        polyak_update(policy, target, tau=0.5)

        # Check interpolation: w_target = 0.5 * w_old + 0.5 * w_source.
        for src, tgt, bef in zip(policy.trainable_variables,
                                 target.trainable_variables, before):
            expected = 0.5 * bef + 0.5 * src.numpy()
            np.testing.assert_allclose(
                tgt.numpy(), expected, atol=1e-6)


# =============================================================================
# Data pipeline
# =============================================================================

class TestDataPipeline:
    """Tests for data collection and evaluation utilities."""

    def test_collect_transitions_shape(self, env, policy):
        """collect_transitions returns correct shapes."""
        ss = SeedSchedule((20, 26))
        data = collect_transitions(env, policy, n_samples=32,
                                   seed_schedule=ss, step=0)
        assert data["s"].shape == (32, 2)
        assert data["a"].shape == (32, 1)
        assert data["s_next"].shape == (32, 2)
        assert data["eps"].shape == (32, 1)

    def test_generate_eval_dataset(self, env):
        """generate_eval_dataset returns states and two shock sets."""
        ss = SeedSchedule((20, 26))
        eval_data = generate_eval_dataset(env, n_samples=64, seed_schedule=ss)
        assert "s" in eval_data
        assert "eps" in eval_data
        assert "eps_fork" in eval_data
        assert eval_data["s"].shape == (64, 2)

    def test_evaluate_euler_residual_finite(self, env, policy):
        """Euler residual evaluation returns a finite number."""
        ss = SeedSchedule((20, 26))
        eval_data = generate_eval_dataset(env, n_samples=32, seed_schedule=ss)
        er = evaluate_euler_residual(env, policy, eval_data)
        assert np.isfinite(er)

    def test_evaluate_bellman_residual_finite(self, env, policy, critic):
        """Bellman residual evaluation returns a finite number."""
        ss = SeedSchedule((20, 26))
        eval_data = generate_eval_dataset(env, n_samples=32, seed_schedule=ss)
        br = evaluate_bellman_residual(env, policy, critic, eval_data)
        assert np.isfinite(br)


# =============================================================================
# LR trainer smoke test
# =============================================================================

class TestTrainLR:
    """Smoke test for the LR trainer."""

    def test_train_lr_runs(self, env, policy):
        """train_lr completes without error for a few steps."""
        config = LRConfig(
            n_steps=5, batch_size=16, horizon=4,
            eval_interval=2, eval_size=32,
            policy_optimizer=_small_optimizer(),
        )
        result = train_lr(env, policy, config=config)
        assert "policy" in result
        assert "history" in result
        assert len(result["history"]["step"]) > 0

    def test_train_lr_loss_recorded(self, env, policy):
        """LR trainer records loss values."""
        config = LRConfig(
            n_steps=3, batch_size=16, horizon=4,
            eval_interval=1, eval_size=32,
            policy_optimizer=_small_optimizer(),
        )
        result = train_lr(env, policy, config=config)
        losses = result["history"]["loss"]
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)


# =============================================================================
# ER trainer smoke test
# =============================================================================

class TestTrainER:
    """Smoke test for the ER trainer."""

    def test_train_er_runs(self, env, policy):
        """train_er completes without error for a few steps."""
        config = ERConfig(
            n_steps=5, batch_size=16,
            eval_interval=2, eval_size=32,
            replay_buffer_size=200,
            policy_optimizer=_small_optimizer(),
        )
        result = train_er(env, policy, config=config)
        assert "policy" in result
        assert "history" in result

    def test_train_er_crossprod_and_mse(self, env):
        """Both loss types run without error."""
        for loss_type in ["crossprod", "mse"]:
            p = PolicyNetwork(
                state_dim=env.state_dim(), action_dim=env.action_dim(),
                **env.action_spec(), n_layers=2, n_neurons=32,
            )
            p(tf.zeros((1, env.state_dim())))
            config = ERConfig(
                n_steps=3, batch_size=16, loss_type=loss_type,
                eval_interval=2, eval_size=32,
                replay_buffer_size=200,
                policy_optimizer=_small_optimizer(),
            )
            result = train_er(env, p, config=config)
            assert len(result["history"]["step"]) > 0


# =============================================================================
# MVE trainer smoke test
# =============================================================================

class TestTrainMVE:
    """Smoke test for the MVE trainer."""

    def test_train_mve_runs(self, env, policy, critic):
        """train_mve completes without error for a few steps."""
        config = MVEConfig(
            n_steps=5, batch_size=16, mve_horizon=3,
            eval_interval=2, eval_size=32,
            replay_buffer_size=200,
            policy_optimizer=_small_optimizer(),
            critic_optimizer=_small_optimizer(),
        )
        result = train_mve(env, policy, critic, config=config)
        assert "policy" in result
        assert "critic" in result
        assert "history" in result

    def test_train_mve_records_both_losses(self, env, policy, critic):
        """MVE trainer records both actor and critic losses."""
        config = MVEConfig(
            n_steps=3, batch_size=16, mve_horizon=2,
            eval_interval=1, eval_size=32,
            replay_buffer_size=200,
            policy_optimizer=_small_optimizer(),
            critic_optimizer=_small_optimizer(),
        )
        result = train_mve(env, policy, critic, config=config)
        h = result["history"]
        assert len(h["actor_loss"]) > 0
        assert len(h["critic_loss"]) > 0
        assert all(np.isfinite(l) for l in h["actor_loss"])
        assert all(np.isfinite(l) for l in h["critic_loss"])

    def test_lambda_preprocessing_reduces_critic_loss(self, env):
        """λ-preprocessing brings critic loss to O(1) scale."""
        from src.v2.trainers.core import SeedSchedule
        seed = SeedSchedule().pretraining_seed(SeedSchedule.VAR_BELLMAN_SCALE)
        lam = env.compute_reward_scale(seed=seed)

        def _fresh_nets():
            tf.random.set_seed(0)
            p = PolicyNetwork(
                state_dim=env.state_dim(), action_dim=env.action_dim(),
                **env.action_spec(), n_layers=2, n_neurons=32)
            c = CriticNetwork(
                state_dim=env.state_dim(), action_dim=env.action_dim(),
                n_layers=2, n_neurons=32)
            p(tf.zeros((1, env.state_dim())))
            c(tf.zeros((1, env.state_dim())), tf.zeros((1, env.action_dim())))
            return p, c

        base_cfg = dict(
            n_steps=10, batch_size=32, mve_horizon=3,
            eval_interval=5, eval_size=64,
            replay_buffer_size=500,
            master_seed=(99, 0),
            policy_optimizer=_small_optimizer(),
            critic_optimizer=_small_optimizer(),
        )

        p1, c1 = _fresh_nets()
        r1 = train_mve(env, p1, c1,
                        config=MVEConfig(**base_cfg, reward_scale=1.0))

        p2, c2 = _fresh_nets()
        r2 = train_mve(env, p2, c2,
                        config=MVEConfig(**base_cfg, reward_scale=lam))

        # The λ-normalized run should have much smaller critic loss.
        loss_unnorm = r1["history"]["critic_loss"][-1]
        loss_norm = r2["history"]["critic_loss"][-1]
        assert loss_norm < loss_unnorm, (
            f"λ-normalized loss ({loss_norm:.4f}) should be smaller than "
            f"unnormalized ({loss_unnorm:.4f})"
        )


# =============================================================================
# BRM trainer smoke test
# =============================================================================

class TestTrainBRM:
    """Smoke test for the BRM trainer."""

    def test_train_brm_runs(self, env, policy, value_net):
        """train_brm completes without error for a few steps."""
        config = BRMConfig(
            n_steps=5, batch_size=16,
            eval_interval=2, eval_size=32,
            replay_buffer_size=200,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, policy, value_net, config=config)
        assert "policy" in result
        assert "value_net" in result
        assert "history" in result
        assert len(result["history"]["step"]) > 0

    def test_train_brm_crossprod_and_mse(self, env):
        """Both loss types run without error."""
        for loss_type in ["crossprod", "mse"]:
            p = PolicyNetwork(
                state_dim=env.state_dim(), action_dim=env.action_dim(),
                **env.action_spec(), n_layers=2, n_neurons=32,
            )
            p(tf.zeros((1, env.state_dim())))
            v = StateValueNetwork(
                state_dim=env.state_dim(), n_layers=2, n_neurons=32,
            )
            v(tf.zeros((1, env.state_dim())))
            config = BRMConfig(
                n_steps=3, batch_size=16, loss_type=loss_type,
                eval_interval=2, eval_size=32,
                replay_buffer_size=200,
                policy_optimizer=_small_optimizer(),
            )
            result = train_brm(env, p, v, config=config)
            assert len(result["history"]["step"]) > 0

    def test_train_brm_foc_finite(self, env):
        """BRM autodiff FOC loss is finite."""
        p = PolicyNetwork(
            state_dim=env.state_dim(), action_dim=env.action_dim(),
            **env.action_spec(), n_layers=2, n_neurons=32,
        )
        p(tf.zeros((1, env.state_dim())))
        v = StateValueNetwork(
            state_dim=env.state_dim(), n_layers=2, n_neurons=32,
        )
        v(tf.zeros((1, env.state_dim())))
        config = BRMConfig(
            n_steps=3, batch_size=16,
            eval_interval=2, eval_size=32,
            replay_buffer_size=200,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, p, v, config=config)
        assert len(result["history"]["step"]) > 0
        assert all(np.isfinite(l) for l in result["history"]["loss_foc"])

    def test_train_brm_records_all_losses(self, env, policy, value_net):
        """BRM trainer records all component losses."""
        config = BRMConfig(
            n_steps=3, batch_size=16,
            eval_interval=1, eval_size=32,
            replay_buffer_size=200,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, policy, value_net, config=config)
        h = result["history"]
        assert len(h["loss_br"]) == 3
        assert len(h["loss_foc"]) == 3
        assert all(np.isfinite(l) for l in h["loss"])
        assert all(np.isfinite(l) for l in h["loss_br"])

    def test_bellman_residual_v_finite(self, env, policy, value_net):
        """evaluate_bellman_residual_v returns a finite number."""
        ss = SeedSchedule((20, 26))
        eval_data = generate_eval_dataset(env, n_samples=32,
                                          seed_schedule=ss)
        br = evaluate_bellman_residual_v(env, policy, value_net, eval_data)
        assert np.isfinite(br)


# =============================================================================
# Environment interface: euler_residual with temperature
# =============================================================================

class TestEnvironmentResiduals:
    """Tests for environment-level residual methods."""

    def test_euler_residual_with_temperature(self, env, policy):
        """euler_residual accepts temperature and returns finite values."""
        ss = SeedSchedule((20, 26))
        s = env.sample_initial_states(32, seed=ss.eval_seed(1))
        eps = env.sample_shocks(32, seed=ss.eval_seed(4))
        a = policy(s, training=False)
        s_next = env.transition(s, a, eps)
        a_next = policy(s_next, training=False)

        for temp in [1e-6, 0.01, 0.1]:
            res = env.euler_residual(s, a, s_next, a_next,
                                     temperature=temp)
            assert res.shape == (32,)
            assert tf.reduce_all(tf.math.is_finite(res))
