"""Tests for v2 trainer modules: core utilities, LR, ER, BRM, SHAC.

All training smoke tests use very small configs (few steps, small batch)
to verify correctness without long runtimes.
"""

import tensorflow as tf
import numpy as np
import pytest

from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.networks.policy import PolicyNetwork
from src.v2.networks.state_value import StateValueNetwork
from src.v2.data.generator import DataGenerator, DataGeneratorConfig
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys,
    fit_normalizer_traj, fit_normalizer_flat,
)
from src.v2.trainers.core import (
    polyak_update, hard_update, build_target_policy, build_target_value,
    evaluate_euler_residual, evaluate_bellman_residual, StopTracker,
)
from src.v2.trainers.config import (
    LRConfig, ERConfig, BRMConfig, SHACConfig, OptimizerConfig,
)
from src.v2.trainers.lr import train_lr
from src.v2.trainers.er import train_er
from src.v2.trainers.brm import train_brm
from src.v2.trainers.shac import train_shac
from src.v2.experimental.brm_joint import train_brm_joint


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
    net(tf.zeros((1, env.state_dim())))
    return net


@pytest.fixture
def value_net(env):
    """Small state-value network for smoke tests."""
    net = StateValueNetwork(
        state_dim=env.state_dim(),
        n_layers=2, n_neurons=32,
    )
    net(tf.zeros((1, env.state_dim())))
    return net


@pytest.fixture
def traj_dataset(env):
    """Small trajectory-format dataset for LR/SHAC tests."""
    gen = DataGenerator(env, DataGeneratorConfig(n_paths=32, horizon=8))
    return gen.get_trajectory_dataset("train")


@pytest.fixture
def flat_dataset(env):
    """Small flattened-format dataset for ER/BRM tests."""
    gen = DataGenerator(env, DataGeneratorConfig(n_paths=32, horizon=8))
    return gen.get_flattened_dataset("train")


@pytest.fixture
def val_dataset(env):
    """Small flattened-format validation dataset."""
    gen = DataGenerator(env, DataGeneratorConfig(n_paths=16, horizon=8))
    return gen.get_flattened_dataset("val")


def _small_optimizer():
    return OptimizerConfig(learning_rate=1e-3, clipnorm=100.0)


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
        target(tf.zeros((1, env.state_dim())))
        hard_update(policy, target)
        for src, tgt in zip(policy.trainable_variables,
                            target.trainable_variables):
            np.testing.assert_allclose(src.numpy(), tgt.numpy(), atol=1e-7)

    def test_polyak_update_interpolates(self, policy, env):
        """polyak_update moves target toward source."""
        target = PolicyNetwork(
            state_dim=env.state_dim(), action_dim=env.action_dim(),
            **env.action_spec(), n_layers=2, n_neurons=32,
            name="policy_target",
        )
        target(tf.zeros((1, env.state_dim())))
        before = [v.numpy().copy() for v in target.trainable_variables]
        polyak_update(policy, target, tau=0.5)
        for src, tgt, bef in zip(policy.trainable_variables,
                                  target.trainable_variables, before):
            expected = 0.5 * bef + 0.5 * src.numpy()
            np.testing.assert_allclose(tgt.numpy(), expected, atol=1e-6)

    def test_build_target_policy(self, policy, env):
        """build_target_policy creates a target with identical initial weights."""
        target = build_target_policy(policy)
        for src, tgt in zip(policy.trainable_variables,
                             target.trainable_variables):
            np.testing.assert_allclose(src.numpy(), tgt.numpy(), atol=1e-7)

    def test_build_target_value(self, value_net):
        """build_target_value creates a target with identical initial weights."""
        target = build_target_value(value_net)
        for src, tgt in zip(value_net.trainable_variables,
                             target.trainable_variables):
            np.testing.assert_allclose(src.numpy(), tgt.numpy(), atol=1e-7)


# =============================================================================
# Data pipeline utilities
# =============================================================================

class TestDataPipeline:
    """Tests for pipeline utilities."""

    def test_validate_dataset_keys_passes(self, flat_dataset):
        """validate_dataset_keys raises nothing when all keys present."""
        validate_dataset_keys(flat_dataset,
                              ["s_endo", "z", "z_next_main", "z_next_fork"],
                              "test")

    def test_validate_dataset_keys_raises(self, flat_dataset):
        """validate_dataset_keys raises ValueError on missing key."""
        with pytest.raises(ValueError, match="missing required keys"):
            validate_dataset_keys(flat_dataset, ["s_endo", "missing_key"],
                                  "test")

    def test_build_iterator_shape(self, flat_dataset, env):
        """build_iterator yields batches with correct shapes."""
        ds = build_iterator(flat_dataset, batch_size=16)
        batch = next(iter(ds))
        assert batch["s_endo"].shape == (16, env.endo_dim())
        assert batch["z"].shape == (16, env.exo_dim())

    def test_fit_normalizer_flat(self, env, flat_dataset, policy):
        """fit_normalizer_flat sets non-trivial mean on the policy network."""
        np.testing.assert_allclose(policy.normalizer.mean.numpy(),
                                   np.zeros(env.state_dim()), atol=1e-6)
        fit_normalizer_flat(env, flat_dataset, policy)
        assert not np.allclose(policy.normalizer.mean.numpy(),
                               np.zeros(env.state_dim()))

    def test_fit_normalizer_traj(self, env, traj_dataset, policy):
        """fit_normalizer_traj sets non-trivial mean on the policy network."""
        np.testing.assert_allclose(policy.normalizer.mean.numpy(),
                                   np.zeros(env.state_dim()), atol=1e-6)
        fit_normalizer_traj(env, traj_dataset, policy)
        assert not np.allclose(policy.normalizer.mean.numpy(),
                               np.zeros(env.state_dim()))

    def test_fit_normalizer_multiple_networks(self, env, flat_dataset,
                                              policy, value_net):
        """fit_normalizer_flat gives identical stats to all networks."""
        fit_normalizer_flat(env, flat_dataset, policy, value_net)
        np.testing.assert_allclose(
            policy.normalizer.mean.numpy(),
            value_net.normalizer.mean.numpy(), atol=1e-6)
        np.testing.assert_allclose(
            policy.normalizer.std.numpy(),
            value_net.normalizer.std.numpy(), atol=1e-6)


# =============================================================================
# Evaluation metrics
# =============================================================================

class TestEvaluationMetrics:

    def test_evaluate_euler_residual_finite(self, env, policy, val_dataset):
        """evaluate_euler_residual returns a finite number."""
        er = evaluate_euler_residual(env, policy, val_dataset)
        assert np.isfinite(er)

    def test_evaluate_bellman_residual_finite(self, env, policy,
                                              value_net, val_dataset):
        """evaluate_bellman_residual returns a finite number."""
        br = evaluate_bellman_residual(env, policy, value_net, val_dataset)
        assert np.isfinite(br)


class TestStopTracker:
    """Unit tests for shared threshold / stopping metadata."""

    def test_stop_tracker_confirms_on_consecutive_hits(self):
        tracker = StopTracker(monitor="metric", threshold=0.1, threshold_patience=2)

        assert not tracker.record_eval(
            step=0, elapsed_sec=1.0, metrics={"metric": 0.09})
        assert tracker.record_eval(
            step=1, elapsed_sec=2.0, metrics={"metric": 0.08})
        assert tracker.converged is True
        assert tracker.stop_reason == "converged"
        assert tracker.threshold_step == 0
        assert tracker.stop_step == 1
        assert tracker.threshold_elapsed_sec == 1.0
        assert tracker.stop_elapsed_sec == 2.0

    def test_stop_tracker_plateau_after_patience(self):
        tracker = StopTracker(
            monitor="metric",
            mode="min",
            plateau_patience=2,
            plateau_min_delta=0.01,
        )

        assert not tracker.record_eval(
            step=0, elapsed_sec=1.0, metrics={"metric": 1.00})
        assert not tracker.record_eval(
            step=1, elapsed_sec=2.0, metrics={"metric": 0.90})
        assert not tracker.record_eval(
            step=2, elapsed_sec=3.0, metrics={"metric": 0.905})
        assert tracker.record_eval(
            step=3, elapsed_sec=4.0, metrics={"metric": 0.908})

        assert tracker.converged is False
        assert tracker.stop_reason == "plateau"
        assert tracker.best_step == 1
        assert tracker.best_elapsed_sec == 2.0
        assert tracker.best_metric_value == pytest.approx(0.90)

    def test_stop_tracker_relative_plateau_after_patience(self):
        tracker = StopTracker(
            monitor="metric",
            mode="min",
            plateau_patience=2,
            plateau_min_delta=0.0,
            plateau_rel_delta=0.05,
        )

        assert not tracker.record_eval(
            step=0, elapsed_sec=1.0, metrics={"metric": 100.0})
        assert not tracker.record_eval(
            step=1, elapsed_sec=2.0, metrics={"metric": 97.0})  # < 5% improvement
        assert tracker.record_eval(
            step=2, elapsed_sec=3.0, metrics={"metric": 96.5})

        assert tracker.stop_reason == "plateau"
        assert tracker.best_step == 2
        assert tracker.best_metric_value == pytest.approx(96.5)

    def test_stop_tracker_freezes_terminal_plateau_metadata(self):
        tracker = StopTracker(
            monitor="metric",
            mode="min",
            plateau_patience=2,
            plateau_min_delta=0.01,
        )

        assert not tracker.record_eval(
            step=0, elapsed_sec=1.0, metrics={"metric": 1.00})
        assert not tracker.record_eval(
            step=1, elapsed_sec=2.0, metrics={"metric": 0.90})
        assert not tracker.record_eval(
            step=2, elapsed_sec=3.0, metrics={"metric": 0.905})
        assert tracker.record_eval(
            step=3, elapsed_sec=4.0, metrics={"metric": 0.908})

        # Later accidental calls should preserve the original terminal state.
        assert tracker.record_eval(
            step=4, elapsed_sec=5.0, metrics={"metric": 0.50})
        assert tracker.stop_reason == "plateau"
        assert tracker.stop_step == 3
        assert tracker.best_step == 1
        assert tracker.best_metric_value == pytest.approx(0.90)


# =============================================================================
# LR trainer smoke test
# =============================================================================

class TestTrainLR:
    """Smoke test for the LR trainer."""

    def test_train_lr_runs(self, env, policy, traj_dataset):
        """train_lr completes without error for a few steps."""
        config = LRConfig(
            n_steps=5, batch_size=16, horizon=4,
            eval_interval=2,
            policy_optimizer=_small_optimizer(),
        )
        result = train_lr(env, policy, traj_dataset, config=config)
        assert "policy" in result
        assert "history" in result
        assert len(result["history"]["step"]) > 0

    def test_train_lr_loss_recorded(self, env, policy, traj_dataset,
                                    val_dataset):
        """LR trainer records finite loss values and euler residual."""
        config = LRConfig(
            n_steps=3, batch_size=16, horizon=4,
            eval_interval=1,
            policy_optimizer=_small_optimizer(),
        )
        result = train_lr(env, policy, traj_dataset, val_dataset=val_dataset,
                          config=config)
        losses = result["history"]["loss"]
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)

    def test_train_lr_horizon_exceeds_data_raises(self, env, policy,
                                                   traj_dataset):
        """train_lr raises if horizon > dataset horizon."""
        config = LRConfig(n_steps=2, batch_size=16, horizon=999,
                          policy_optimizer=_small_optimizer())
        with pytest.raises(ValueError, match="horizon"):
            train_lr(env, policy, traj_dataset, config=config)

    def test_train_lr_threshold_stop_records_metadata(self, env, policy,
                                                      traj_dataset, val_dataset):
        """LR records elapsed time and stops on confirmed threshold hits."""
        config = LRConfig(
            n_steps=5, batch_size=16, horizon=4,
            eval_interval=1,
            stop_euler_threshold=1.0,
            threshold_patience=2,
            policy_optimizer=_small_optimizer(),
        )
        result = train_lr(env, policy, traj_dataset, val_dataset=val_dataset,
                          config=config)
        hist = result["history"]
        assert result["converged"] is True
        assert result["stop_reason"] == "converged"
        assert result["threshold_step"] == 0
        assert result["stop_step"] == 1
        assert len(hist["elapsed_sec"]) == len(hist["step"])
        assert len(hist["train_temperature"]) == len(hist["step"])
        assert hist["elapsed_sec"][0] <= hist["elapsed_sec"][-1]

    def test_train_lr_eval_temperature_override(self, env, policy, traj_dataset,
                                                val_dataset, monkeypatch):
        """LR validation uses eval_temperature when provided."""
        seen = []

        def fake_eval(_env, _policy, _val_dataset, temperature=0.0):
            seen.append(float(temperature))
            return 2.0

        monkeypatch.setattr("src.v2.trainers.core.evaluate_euler_residual", fake_eval)
        config = LRConfig(
            n_steps=1, batch_size=16, horizon=4,
            eval_interval=1,
            eval_temperature=1e-6,
            policy_optimizer=_small_optimizer(),
        )
        train_lr(env, policy, traj_dataset, val_dataset=val_dataset, config=config)
        assert seen == [pytest.approx(1e-6)]

    def test_train_lr_callback_metric_supports_plateau_stopping(self, env, policy,
                                                                traj_dataset):
        """LR can stop on a custom eval callback without a validation dataset."""
        series = {
            0: 5.0,
            1: 4.0,
            2: 4.03,
            3: 4.05,
        }

        def eval_callback(step, _env, _policy, _value_net, _val_dataset,
                          train_temperature):
            assert train_temperature > 0.0
            return {
                "benchmark_policy_mae": series[step],
                "lifetime_reward_val": 10.0 - 0.1 * step,
            }

        config = LRConfig(
            n_steps=6,
            batch_size=16,
            horizon=4,
            eval_interval=1,
            monitor="benchmark_policy_mae",
            mode="min",
            plateau_patience=2,
            plateau_min_delta=0.01,
            policy_optimizer=_small_optimizer(),
        )
        result = train_lr(
            env, policy, traj_dataset, config=config, eval_callback=eval_callback)
        hist = result["history"]
        assert result["converged"] is False
        assert result["stop_reason"] == "plateau"
        assert result["best_step"] == 1
        assert result["stop_step"] == 3
        assert hist["benchmark_policy_mae"] == pytest.approx([5.0, 4.0, 4.03, 4.05])
        assert hist["lifetime_reward_val"] == pytest.approx([10.0, 9.9, 9.8, 9.7])
        assert len(hist["train_temperature"]) == 4


# =============================================================================
# ER trainer smoke test
# =============================================================================

class TestTrainER:
    """Smoke test for the ER trainer."""

    def test_train_er_runs(self, env, policy, flat_dataset):
        """train_er completes without error for a few steps."""
        config = ERConfig(
            n_steps=5, batch_size=16,
            eval_interval=2,
            policy_optimizer=_small_optimizer(),
        )
        result = train_er(env, policy, flat_dataset, config=config)
        assert "policy" in result
        assert "history" in result
        assert len(result["history"]["step"]) > 0

    def test_train_er_crossprod_and_mse(self, env, flat_dataset):
        """Both loss types run without error."""
        for loss_type in ["crossprod", "mse"]:
            p = PolicyNetwork(
                state_dim=env.state_dim(), action_dim=env.action_dim(),
                **env.action_spec(), n_layers=2, n_neurons=32,
            )
            p(tf.zeros((1, env.state_dim())))
            config = ERConfig(
                n_steps=3, batch_size=16, loss_type=loss_type,
                eval_interval=2,
                policy_optimizer=_small_optimizer(),
            )
            result = train_er(env, p, flat_dataset, config=config)
            assert len(result["history"]["step"]) > 0

    def test_train_er_missing_keys_raises(self, env, policy):
        """train_er raises ValueError if dataset missing required keys."""
        bad_dataset = {"s_endo": tf.zeros((10, env.endo_dim()))}
        config = ERConfig(n_steps=1, batch_size=4,
                          policy_optimizer=_small_optimizer())
        with pytest.raises(ValueError, match="missing required keys"):
            train_er(env, policy, bad_dataset, config=config)

    def test_train_er_threshold_stop_records_metadata(self, env, policy,
                                                      flat_dataset, val_dataset):
        """ER records threshold-stop metadata and elapsed history."""
        config = ERConfig(
            n_steps=5, batch_size=16,
            eval_interval=1,
            stop_euler_threshold=1.0,
            threshold_patience=2,
            policy_optimizer=_small_optimizer(),
        )
        result = train_er(env, policy, flat_dataset, val_dataset=val_dataset,
                          config=config)
        assert result["converged"] is True
        assert result["stop_reason"] == "converged"
        assert result["stop_step"] == 1
        assert len(result["history"]["elapsed_sec"]) == 2
        assert len(result["history"]["train_temperature"]) == 2


# =============================================================================
# BRM trainer smoke test
# =============================================================================

class TestTrainBRM:
    """Smoke test for the BRM trainer."""

    def test_train_brm_runs(self, env, policy, value_net, flat_dataset):
        """train_brm completes without error for a few steps."""
        config = BRMConfig(
            n_steps=5, batch_size=16,
            eval_interval=2,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, policy, value_net, flat_dataset, config=config)
        assert "policy" in result
        assert "value_net" in result
        assert "history" in result
        assert len(result["history"]["step"]) > 0

    def test_train_brm_crossprod_and_mse(self, env, flat_dataset):
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
                eval_interval=2,
                policy_optimizer=_small_optimizer(),
            )
            result = train_brm(env, p, v, flat_dataset, config=config)
            assert len(result["history"]["step"]) > 0

    def test_train_brm_records_all_losses(self, env, policy, value_net,
                                          flat_dataset):
        """BRM trainer records all component losses."""
        config = BRMConfig(
            n_steps=3, batch_size=16,
            eval_interval=1,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, policy, value_net, flat_dataset, config=config)
        h = result["history"]
        assert len(h["loss_br"]) == 3
        assert len(h["loss_foc"]) == 3
        assert all(np.isfinite(l) for l in h["loss"])
        assert all(np.isfinite(l) for l in h["loss_br"])

    def test_train_brm_foc_finite(self, env, flat_dataset):
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
            eval_interval=2,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, p, v, flat_dataset, config=config)
        assert all(np.isfinite(l) for l in result["history"]["loss_foc"])

    def test_train_brm_threshold_stop_records_metadata(self, env, policy,
                                                       value_net, flat_dataset,
                                                       val_dataset):
        """BRM exposes stop metadata and elapsed seconds."""
        config = BRMConfig(
            n_steps=5, batch_size=16,
            eval_interval=1,
            stop_euler_threshold=1.0,
            threshold_patience=2,
            warm_start_steps=0,
            policy_optimizer=_small_optimizer(),
        )
        result = train_brm(env, policy, value_net, flat_dataset,
                           val_dataset=val_dataset, config=config)
        assert result["converged"] is True
        assert result["stop_reason"] == "converged"
        assert result["stop_step"] == 1
        assert len(result["history"]["elapsed_sec"]) == 2
        assert len(result["history"]["train_temperature"]) == 2


class TestTrainBRMJoint:
    """Smoke tests for the experimental joint-update BRM control."""

    def test_train_brm_joint_runs(self, env, policy, value_net, flat_dataset):
        config = BRMConfig(
            n_steps=4,
            batch_size=16,
            eval_interval=2,
            warm_start_steps=0,
            policy_optimizer=_small_optimizer(),
            critic_optimizer=_small_optimizer(),
        )
        result = train_brm_joint(env, policy, value_net, flat_dataset, config=config)
        assert "policy" in result
        assert "value_net" in result
        assert len(result["history"]["step"]) > 0
        assert all(np.isfinite(x) for x in result["history"]["loss"])

    def test_train_brm_joint_supports_eval_callback_and_checkpoints(
        self, env, policy, value_net, flat_dataset
    ):
        checkpoint_history = []

        def eval_callback(step, _env, _policy, _value_net, _val_dataset,
                          train_temperature):
            assert train_temperature > 0.0
            return {
                "benchmark_policy_mae": 10.0 - step,
                "lifetime_reward_val": train_temperature,
            }

        config = BRMConfig(
            n_steps=3,
            batch_size=16,
            eval_interval=1,
            warm_start_steps=0,
            checkpoint_history=checkpoint_history,
            snapshot_targets=("policy", "value_net"),
            policy_optimizer=_small_optimizer(),
            critic_optimizer=_small_optimizer(),
        )
        result = train_brm_joint(
            env,
            policy,
            value_net,
            flat_dataset,
            config=config,
            eval_callback=eval_callback,
        )
        hist = result["history"]
        assert hist["benchmark_policy_mae"] == pytest.approx([10.0, 9.0, 8.0])
        assert len(checkpoint_history) == 3
        assert "policy" in checkpoint_history[0]["models"]
        assert "value_net" in checkpoint_history[0]["models"]


# =============================================================================
# SHAC trainer (DDPG-style variant) tests
# =============================================================================

def _shac_config(**overrides):
    """Build a small SHACConfig for testing."""
    defaults = dict(
        n_steps=4, batch_size=16, horizon=8, short_horizon=4,
        n_critic=2, eval_interval=2,
        policy_optimizer=_small_optimizer(),
        critic_optimizer=_small_optimizer(),
    )
    defaults.update(overrides)
    return SHACConfig(**defaults)


def _make_policy_and_value(env):
    """Create fresh small policy + value_net for isolation tests."""
    p = PolicyNetwork(
        state_dim=env.state_dim(), action_dim=env.action_dim(),
        **env.action_spec(), n_layers=2, n_neurons=32,
    )
    p(tf.zeros((1, env.state_dim())))
    v = StateValueNetwork(
        state_dim=env.state_dim(), n_layers=2, n_neurons=32,
    )
    v(tf.zeros((1, env.state_dim())))
    return p, v


class TestTrainSHAC:
    """Tests for SHAC trainer (DDPG-style variant)."""

    # ---- Smoke / integration tests ----

    def test_train_shac_runs(self, env, policy, value_net, traj_dataset):
        """train_shac completes without error for a few steps."""
        config = _shac_config()
        result = train_shac(env, policy, value_net, traj_dataset, config=config)
        assert "policy" in result
        assert "value_net" in result
        assert "history" in result
        assert len(result["history"]["step"]) > 0

    def test_train_shac_records_losses(self, env, traj_dataset, val_dataset):
        """SHAC trainer records finite actor and critic losses."""
        p, v = _make_policy_and_value(env)
        config = _shac_config(eval_interval=1)
        result = train_shac(env, p, v, traj_dataset,
                            val_dataset=val_dataset, config=config)
        hist = result["history"]
        assert len(hist["loss_actor"]) == 4
        assert all(np.isfinite(l) for l in hist["loss_actor"])
        assert all(np.isfinite(l) for l in hist["loss_critic"])
        assert all(np.isfinite(r) for r in hist["euler_residual_val"])
        assert all(np.isfinite(r) for r in hist["bellman_residual"])
        assert len(hist["train_temperature"]) == len(hist["step"])

    # ---- Input validation ----

    def test_train_shac_horizon_exceeds_data_raises(self, env, policy,
                                                     value_net, traj_dataset):
        """train_shac raises if horizon > dataset horizon."""
        config = _shac_config(horizon=999, short_horizon=3)
        with pytest.raises(ValueError, match="horizon"):
            train_shac(env, policy, value_net, traj_dataset, config=config)

    def test_train_shac_window_not_divisible_raises(self, env, policy,
                                                     value_net, traj_dataset):
        """train_shac raises if horizon % short_horizon != 0."""
        config = _shac_config(horizon=8, short_horizon=3)
        with pytest.raises(ValueError, match="divisible"):
            train_shac(env, policy, value_net, traj_dataset, config=config)

    def test_train_shac_step_counting(self, env, traj_dataset):
        """Each mini-batch produces horizon/short_horizon steps."""
        p, v = _make_policy_and_value(env)
        # horizon=8, short_horizon=4 → 2 windows per batch
        # n_steps=6 → exactly 6 steps (3 batches × 2 windows)
        config = _shac_config(
            n_steps=6, n_critic=1,
            eval_interval=100,  # only log at step 0 and step 5 (last)
        )
        result = train_shac(env, p, v, traj_dataset, config=config)
        steps = result["history"]["step"]
        assert steps[0] == 0
        assert steps[-1] == 5  # 0-indexed, last step is n_steps - 1

    def test_train_shac_threshold_stop_records_metadata(self, env, traj_dataset,
                                                        val_dataset):
        """SHAC can stop early on confirmed threshold hits."""
        p, v = _make_policy_and_value(env)
        config = _shac_config(
            n_steps=6,
            eval_interval=1,
            stop_euler_threshold=1.0,
            threshold_patience=2,
        )
        result = train_shac(env, p, v, traj_dataset,
                            val_dataset=val_dataset, config=config)
        assert result["converged"] is True
        assert result["stop_reason"] == "converged"
        assert result["stop_step"] == 1
        assert len(result["history"]["elapsed_sec"]) == 2

    def test_train_shac_callback_metric_supports_plateau_stopping(
        self, env, traj_dataset
    ):
        """SHAC must exit immediately once plateau stopping triggers."""
        p, v = _make_policy_and_value(env)
        series = {
            0: 5.0,
            1: 4.0,
            2: 4.03,
            3: 4.05,
        }

        def eval_callback(step, _env, _policy, _value_net, _val_dataset,
                          train_temperature):
            assert train_temperature > 0.0
            return {
                "benchmark_policy_mae": series[step],
                "lifetime_reward_val": 10.0 - 0.1 * step,
            }

        config = _shac_config(
            n_steps=10,
            n_critic=1,
            eval_interval=1,
            monitor="benchmark_policy_mae",
            mode="min",
            plateau_patience=2,
            plateau_min_delta=0.01,
        )
        result = train_shac(
            env, p, v, traj_dataset, config=config, eval_callback=eval_callback)
        hist = result["history"]
        assert result["converged"] is False
        assert result["stop_reason"] == "plateau"
        assert result["best_step"] == 1
        assert result["stop_step"] == 3
        assert hist["step"] == [0, 1, 2, 3]
        assert hist["benchmark_policy_mae"] == pytest.approx([5.0, 4.0, 4.03, 4.05])
        assert hist["lifetime_reward_val"] == pytest.approx([10.0, 9.9, 9.8, 9.7])

    # ---- Config shape ----

    def test_shac_config_no_warm_start(self):
        """SHACConfig has no warm_start_steps field (cold start only)."""
        config = SHACConfig()
        assert not hasattr(config, 'warm_start_steps')
        assert not hasattr(config, 'td_lambda')
        assert not hasattr(config, 'n_mb')

    def test_shac_config_reward_normalization_defaults(self):
        """SHACConfig defaults to normalize_rewards=True."""
        config = SHACConfig()
        assert config.normalize_rewards is True
        assert config.reward_scale_override is None

    def test_shac_reward_scale_auto(self, env, traj_dataset):
        """normalize_rewards=True (default) produces reward_scale != 1.0."""
        from src.v2.trainers.shac import _resolve_reward_scale
        config = SHACConfig()
        rs = _resolve_reward_scale(env, config)
        assert rs != 1.0
        assert rs > 0.0

    def test_shac_reward_scale_disabled(self, env):
        """normalize_rewards=False gives reward_scale = 1.0."""
        from src.v2.trainers.shac import _resolve_reward_scale
        config = SHACConfig(normalize_rewards=False)
        rs = _resolve_reward_scale(env, config)
        assert rs == 1.0

    def test_shac_reward_scale_override(self, env):
        """reward_scale_override takes priority over normalize_rewards."""
        from src.v2.trainers.shac import _resolve_reward_scale
        config = SHACConfig(normalize_rewards=True,
                            reward_scale_override=0.005)
        rs = _resolve_reward_scale(env, config)
        assert rs == 0.005

    # ---- Target network management ----

    def test_shac_training_updates_both_networks(self, env, traj_dataset):
        """After training, both policy and value weights have changed."""
        p, v = _make_policy_and_value(env)
        w_policy_before = [w.numpy().copy() for w in p.trainable_variables]
        w_value_before = [w.numpy().copy() for w in v.trainable_variables]

        config = _shac_config(n_steps=2, n_critic=1, eval_interval=100)
        train_shac(env, p, v, traj_dataset, config=config)

        policy_changed = any(
            not np.allclose(w.numpy(), wb, atol=1e-7)
            for w, wb in zip(p.trainable_variables, w_policy_before))
        value_changed = any(
            not np.allclose(w.numpy(), wb, atol=1e-7)
            for w, wb in zip(v.trainable_variables, w_value_before))
        assert policy_changed, "Policy weights should change during training"
        assert value_changed, "Value weights should change during training"

    # ---- Actor uses current V bootstrap (not target V̄) ----

    def test_actor_bootstrap_uses_current_v(self, env, traj_dataset):
        """Verify actor loss depends on current V, not target V̄.

        Strategy: compute the actor objective with the real V and with a
        corrupted V. If the actor uses current V, the results must differ.
        """
        from src.v2.data.pipeline import fit_normalizer_traj

        p, v = _make_policy_and_value(env)
        fit_normalizer_traj(env, traj_dataset, p, v)

        gamma = env.discount()
        window_len = 4
        discount_powers = tf.constant(
            [gamma ** t for t in range(window_len + 1)], dtype=tf.float32)

        # Get a batch
        gen = iter(tf.data.Dataset.from_tensor_slices(traj_dataset)
                   .batch(16))
        batch = next(gen)
        k_endo = batch["s_endo_0"]
        z_window = batch["z_path"][:, :window_len + 1, :]

        def compute_actor_objective(policy_net, value_network):
            """Compute actor objective (no gradient step)."""
            k_current = k_endo
            total_r = tf.zeros(tf.shape(k_endo)[0])
            for tau in range(window_len):
                z_t = z_window[:, tau, :]
                s_t = env.merge_state(k_current, z_t)
                a_t = policy_net(s_t, training=False)
                r_t = tf.reshape(env.reward(s_t, a_t), [-1])
                total_r = total_r + discount_powers[tau] * r_t
                k_current = env.endogenous_transition(k_current, a_t, z_t)
            s_end = env.merge_state(k_current, z_window[:, window_len, :])
            v_boot = tf.squeeze(value_network(s_end, training=False))
            total_r = total_r + discount_powers[window_len] * v_boot
            return -tf.reduce_mean(total_r)

        loss_with_current_v = compute_actor_objective(p, v)

        # Create a corrupted value net (large bias)
        v_corrupt = StateValueNetwork(
            state_dim=env.state_dim(), n_layers=2, n_neurons=32)
        v_corrupt(tf.zeros((1, env.state_dim())))
        for var in v_corrupt.trainable_variables:
            var.assign(var + 999.0)

        loss_with_corrupt_v = compute_actor_objective(p, v_corrupt)

        assert not np.isclose(float(loss_with_current_v),
                              float(loss_with_corrupt_v), rtol=1e-3), \
            "Actor loss must depend on the value network (current V bootstrap)"

    # ---- Critic uses stop_gradient on Bellman target ----

    def test_critic_target_is_stopped(self, env, traj_dataset):
        """Verify critic gradient doesn't flow through policy.

        Strategy: compute critic loss using target π̄ + target V̄ with
        stop_gradient on the Bellman target y. Then check that
        tape.gradient(loss, policy.trainable_variables) returns all None.
        """
        from src.v2.trainers.core import build_target_value, build_target_policy
        from src.v2.data.pipeline import fit_normalizer_traj

        p, v = _make_policy_and_value(env)
        fit_normalizer_traj(env, traj_dataset, p, v)

        target_v = build_target_value(v)
        target_p = build_target_policy(p)
        fit_normalizer_traj(env, traj_dataset, target_p, target_v)

        discount_tf = tf.constant(env.discount(), dtype=tf.float32)

        # Get some states
        gen = iter(tf.data.Dataset.from_tensor_slices(traj_dataset)
                   .batch(16))
        batch = next(gen)
        s_endo_0 = batch["s_endo_0"]
        z_path = batch["z_path"]

        s = env.merge_state(s_endo_0, z_path[:, 0, :])
        z_next = z_path[:, 1, :]

        # Compute Bellman target — uses target_p + target_v, detached
        a_target = target_p(s, training=False)
        r_target = tf.reshape(env.reward(s, a_target), [-1])
        s_endo, s_exo = env.split_state(s)
        k_next = env.endogenous_transition(s_endo, a_target, s_exo)
        s_next = env.merge_state(k_next, z_next)
        v_next = tf.squeeze(target_v(s_next, training=False))
        bellman_target = tf.stop_gradient(r_target + discount_tf * v_next)

        # Only the MSE on current V should be inside the tape
        with tf.GradientTape() as tape:
            v_pred = tf.squeeze(v(s, training=True), axis=-1)
            loss_c = tf.reduce_mean((v_pred - bellman_target) ** 2)

        # Gradient w.r.t. value_net should exist
        grads_wrt_value = tape.gradient(loss_c, v.trainable_variables)
        assert any(g is not None for g in grads_wrt_value), \
            "Critic loss must produce gradients w.r.t. value network"

        # bellman_target is detached, so no gradient flows to policy
        with tf.GradientTape() as tape2:
            v_pred2 = tf.squeeze(v(s, training=True), axis=-1)
            loss_c2 = tf.reduce_mean((v_pred2 - bellman_target) ** 2)
        grads_wrt_policy = tape2.gradient(loss_c2, p.trainable_variables)
        assert all(g is None for g in grads_wrt_policy), \
            "Critic loss must not produce gradients w.r.t. policy variables"
