"""Tests for the β-conditional ER trainer.

What we want to be sure of:

  1. The trainer runs end-to-end for a small step budget without crashing
     and produces a non-empty history.
  2. Training loss decreases over the run (best step's loss ≪ initial loss).
  3. The trained policy roughly recovers the analytical frictionless policy
     when evaluated at the φ = 0 slice — i.e. β-amortization actually works,
     not just shape correctness.
  4. Reproducibility: same master_seed → same trained policy weights.

The third test is the most expensive and uses a modest training budget
(~600 steps on a small dataset). Total runtime for the file is targeted
at well under a minute on CPU.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.v2.environments.basic_investment import EconomicParams, ShockParams
from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
)
from src.v2.estimation.beta_sampler import BetaSampler
from src.v2.networks.policy import ParameterizedPolicyNetwork
from src.v2.trainers.config import ERConfig, NetworkConfig, OptimizerConfig
from src.v2.trainers.er_param import train_er_param


# --------------------------------------------------------------------- fixtures

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


def _build_dataset(env, n=1000, seed=(1, 2)):
    """Synthesize a small flat dataset by uniform sampling inside bounds."""
    s_endo = env.sample_initial_endogenous(
        n, seed=tf.constant(seed, dtype=tf.int32))
    s_exo  = env.sample_initial_exogenous(
        n, seed=tf.constant([seed[0] + 1, seed[1]], dtype=tf.int32))
    return {"s_endo": s_endo, "z": s_exo}


def _build_policy(env, seed=(7, 0)):
    pol = ParameterizedPolicyNetwork(
        state_dim=env.state_dim(), action_dim=env.action_dim(),
        beta_dim=5,
        **env.action_spec(),
        n_layers=2, n_neurons=32, seed=seed)
    pol(tf.zeros((1, env.state_dim())), tf.zeros((1, 5)))
    return pol


# --------------------------------------------------------------------- 1. end-to-end smoke

def test_smoke_runs(env, beta_sampler):
    policy = _build_policy(env)
    ds = _build_dataset(env, n=512)
    cfg = ERConfig(
        n_steps=10, batch_size=128,
        eval_interval=5,
        master_seed=(1, 1),
        network=NetworkConfig(n_layers=2, n_neurons=32),
        policy_optimizer=OptimizerConfig(learning_rate=1e-3),
    )
    out = train_er_param(env, policy, beta_sampler, ds, config=cfg)
    assert out["policy"] is policy
    assert "history" in out and "loss" in out["history"]
    assert len(out["history"]["loss"]) >= 1
    assert out["wall_time_sec"] > 0


# --------------------------------------------------------------------- 2. loss decreases

def test_loss_decreases(env, beta_sampler):
    """Run for enough steps to see a clear loss drop."""
    policy = _build_policy(env, seed=(13, 0))
    ds = _build_dataset(env, n=2048, seed=(2, 3))
    cfg = ERConfig(
        n_steps=300, batch_size=256,
        eval_interval=20,
        master_seed=(2, 2),
        loss_type="mse",
        network=NetworkConfig(n_layers=2, n_neurons=32),
        policy_optimizer=OptimizerConfig(learning_rate=3e-3),
    )
    out = train_er_param(env, policy, beta_sampler, ds, config=cfg)
    losses = np.array(out["history"]["loss"])
    initial = float(losses[0])
    final   = float(np.min(losses[-3:]))
    assert final < 0.5 * initial, (
        f"loss did not decrease enough: initial={initial:.4f}, final={final:.4f}")


# --------------------------------------------------------------------- 3. recovers analytical at φ=0

def test_recovers_frictionless_analytical(env, beta_sampler):
    """After training, the policy should track the closed-form k'(z;β) on
    the frictionless slice within a generous tolerance."""
    policy = _build_policy(env, seed=(31, 0))
    ds = _build_dataset(env, n=4096, seed=(4, 5))
    cfg = ERConfig(
        n_steps=600, batch_size=256,
        eval_interval=100,
        master_seed=(3, 3),
        loss_type="crossprod",
        polyak_rate=0.99,
        network=NetworkConfig(n_layers=2, n_neurons=32),
        policy_optimizer=OptimizerConfig(learning_rate=3e-3),
    )
    out = train_er_param(env, policy, beta_sampler, ds, config=cfg)
    policy = out["policy"]

    # Evaluation on the φ=0 slice
    rng = np.random.default_rng(0)
    n_eval = 256
    k = rng.uniform(env.k_min * 1.5, env.k_max * 0.5, size=n_eval).astype(np.float32)
    z = rng.uniform(env.z_min * 1.2, env.z_max * 0.8, size=n_eval).astype(np.float32)
    s = tf.stack([tf.constant(k), tf.constant(z)], axis=-1)

    # β = prior mean restricted to the frictionless slice.
    beta = BetaSampler(freeze_dims=(3, 4)).prior_mean()
    beta_eval = tf.broadcast_to(beta, [n_eval, 5])

    a_pred = policy(s, beta_eval).numpy().ravel()
    a_true = env.analytical_policy(s, beta_eval).numpy().ravel()

    # Compare on the relative scale of the action range. Loose tolerance.
    action_range = float(env.I_max - env.I_min)
    mae_rel = float(np.mean(np.abs(a_pred - a_true))) / action_range
    assert mae_rel < 0.15, (
        f"trained policy MAE vs analytical at prior-mean β: {mae_rel:.3f} of action range")


# --------------------------------------------------------------------- 4. reproducibility

def test_per_step_beta_seed_is_deterministic():
    """Pin the reproducibility contract for per-step β draws.

    `train_er_param` resamples β every minibatch rather than baking it into
    the dataset. Reproducibility relies on the per-step seed being a pure
    function of (master_seed, trainer name, "step", step, "beta") via
    `fold_in_seed`. This test pins that contract: two independent
    invocations must produce bit-identical β draws at the same step.
    """
    from src.v2.estimation.beta_sampler import BetaSampler
    from src.v2.utils.seeding import fold_in_seed
    master = (20, 26)

    # 1. Seed derivation is deterministic.
    seed_step_100_a = fold_in_seed(master, "train_er_param", "step", 100, "beta")
    seed_step_100_b = fold_in_seed(master, "train_er_param", "step", 100, "beta")
    assert seed_step_100_a == seed_step_100_b

    # 2. Sub-seeds across steps are distinct (no collisions inside the run).
    seeds_first_10 = [
        fold_in_seed(master, "train_er_param", "step", i, "beta")
        for i in range(10)
    ]
    assert len(set(seeds_first_10)) == 10

    # 3. Sibling sub-namespace 'eps_main' derives a different seed.
    seed_eps_100 = fold_in_seed(master, "train_er_param", "step", 100, "eps_main")
    assert seed_step_100_a != seed_eps_100

    # 4. Same seed → bit-identical β draws from the BetaSampler.
    sampler = BetaSampler()
    b1 = sampler.sample(8, seed=seed_step_100_a).numpy()
    b2 = sampler.sample(8, seed=seed_step_100_a).numpy()
    np.testing.assert_array_equal(b1, b2)


def test_reproducibility_same_seed(env, beta_sampler):
    """Identical config and seed → identical final policy weights."""
    def run():
        pol = _build_policy(env, seed=(99, 0))
        ds  = _build_dataset(env, n=512, seed=(8, 9))
        cfg = ERConfig(
            n_steps=20, batch_size=128,
            eval_interval=10,
            master_seed=(7, 7),
            strict_reproducibility=False,
            network=NetworkConfig(n_layers=2, n_neurons=16),
        )
        out = train_er_param(env, pol, beta_sampler, ds, config=cfg)
        return [w.numpy() for w in out["policy"].trainable_weights]

    w1 = run()
    w2 = run()
    for a, b in zip(w1, w2):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
