"""Smoke and convergence tests for the β-conditional SHAC trainer."""

import numpy as np
import pytest
import tensorflow as tf

from src.v2.environments.basic_investment import EconomicParams, ShockParams
from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
)
from src.v2.estimation.beta_sampler import BetaSampler
from src.v2.networks.policy import ParameterizedPolicyNetwork
from src.v2.networks.state_value import ParameterizedStateValueNetwork
from src.v2.trainers.config import (
    NetworkConfig,
    OptimizerConfig,
    SHACConfig,
)
from src.v2.trainers.shac_param import train_shac_param


@pytest.fixture(scope="module")
def env():
    econ   = EconomicParams(
        interest_rate=0.04, depreciation_rate=0.10, production_elasticity=0.5)
    shocks = ShockParams(rho=0.5, sigma=0.24, mu=0.0)
    return ParameterizedBasicInvestmentEnv(
        nominal_econ=econ, nominal_shocks=shocks,
        k_min_mult=0.1, k_max_mult=10.0, z_sd_mult=3.0)


@pytest.fixture(scope="module")
def beta_sampler():
    return BetaSampler()


def _build_dataset(env, n=512, seed=(1, 2)):
    s_endo = env.sample_initial_endogenous(
        n, seed=tf.constant(seed, dtype=tf.int32))
    s_exo  = env.sample_initial_exogenous(
        n, seed=tf.constant([seed[0] + 1, seed[1]], dtype=tf.int32))
    return {"s_endo": s_endo, "z": s_exo}


def _build_nets(env, seed_pol=(7, 0), seed_val=(11, 0)):
    pol = ParameterizedPolicyNetwork(
        state_dim=env.state_dim(), action_dim=env.action_dim(),
        beta_dim=5, **env.action_spec(),
        n_layers=2, n_neurons=32, seed=seed_pol)
    pol(tf.zeros((1, env.state_dim())), tf.zeros((1, 5)))
    val = ParameterizedStateValueNetwork(
        state_dim=env.state_dim(), beta_dim=5,
        n_layers=2, n_neurons=32, seed=seed_val)
    val(tf.zeros((1, env.state_dim())), tf.zeros((1, 5)))
    return pol, val


def test_smoke_runs(env, beta_sampler):
    pol, val = _build_nets(env)
    ds = _build_dataset(env, n=128)
    cfg = SHACConfig(
        n_steps=4, batch_size=64,
        horizon=16, short_horizon=8, n_critic=2,
        eval_interval=2,
        master_seed=(1, 1),
        normalize_rewards=False,
        reward_scale_override=0.01,
        network=NetworkConfig(n_layers=2, n_neurons=32),
        policy_optimizer=OptimizerConfig(learning_rate=1e-3),
        critic_optimizer=OptimizerConfig(learning_rate=1e-3),
    )
    out = train_shac_param(env, pol, val, beta_sampler, ds, config=cfg)
    assert out["policy"] is pol
    assert out["value_net"] is val
    assert len(out["history"].get("loss_actor", [])) >= 1
    assert out["wall_time_sec"] > 0


def test_actor_loss_descends(env, beta_sampler):
    """Across a modest number of windows the actor loss (= -E[return]) should
    decrease — i.e., the surrogate finds higher-return actions over time."""
    pol, val = _build_nets(env, seed_pol=(21, 0), seed_val=(22, 0))
    ds = _build_dataset(env, n=256, seed=(3, 4))
    cfg = SHACConfig(
        n_steps=40, batch_size=64,
        horizon=24, short_horizon=8, n_critic=4,
        eval_interval=4,
        master_seed=(2, 2),
        normalize_rewards=False,
        reward_scale_override=0.01,
        network=NetworkConfig(n_layers=2, n_neurons=32),
        policy_optimizer=OptimizerConfig(learning_rate=2e-3),
        critic_optimizer=OptimizerConfig(learning_rate=5e-3),
    )
    out = train_shac_param(env, pol, val, beta_sampler, ds, config=cfg)
    actor_losses = np.array(out["history"]["loss_actor"])
    initial = float(actor_losses[0])
    best    = float(np.min(actor_losses))
    # Actor loss = -E[return]; lower (more negative) is better.
    assert best < initial, (
        f"actor loss did not improve: initial={initial:.4f}, best={best:.4f}")


def test_reproducibility(env, beta_sampler):
    def run():
        pol, val = _build_nets(env, seed_pol=(99, 0), seed_val=(98, 0))
        ds = _build_dataset(env, n=64, seed=(8, 9))
        cfg = SHACConfig(
            n_steps=4, batch_size=32,
            horizon=8, short_horizon=4, n_critic=2,
            eval_interval=2,
            master_seed=(7, 7),
            normalize_rewards=False,
            reward_scale_override=0.01,
            network=NetworkConfig(n_layers=2, n_neurons=16),
        )
        out = train_shac_param(env, pol, val, beta_sampler, ds, config=cfg)
        return [w.numpy() for w in out["policy"].trainable_weights]
    w1 = run()
    w2 = run()
    for a, b in zip(w1, w2):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
