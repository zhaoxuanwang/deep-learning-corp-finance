"""β-conditional Euler Residual trainer for the neural surrogate.

Parallel to `train_er` (`src/v2/trainers/er.py`) but consumes:
  * a `ParameterizedBasicInvestmentEnv` whose β-aware methods take a per-
    sample β tensor of shape (B, 5),
  * a `ParameterizedPolicyNetwork` whose call signature is `(s, β)`,
  * a `BetaSampler` that draws β ∼ prior per minibatch.

Data contract
-------------
The flattened `train_dataset` only needs the (s_endo, z) columns:

    s_endo: (N, endo_dim)   uniform-in-bounds endogenous state
    z:      (N, exo_dim)    pre-generated exogenous state samples

Any `z_next_main`/`z_next_fork` columns produced by the standard
`DataGenerator` are *ignored* — the trainer resamples ε per minibatch and
computes `z_next` under the per-minibatch β.

Loss
----
Per minibatch:
    β        ~ BetaSampler.sample(B, seed=step_seed_β)
    ε_main   ~ N(0, 1)        shape (B, exo_dim)
    ε_fork   ~ N(0, 1)        shape (B, exo_dim)
    z_next_• = env.exogenous_transition(z, ε_•, β)
    a        = policy(s, β)
    k_next   = env.endogenous_transition(s_endo, a, z, β)
    s_next_• = merge(k_next, z_next_•)
    a_next_• = target_policy([s_next_• | β])      # single-arg call
    f_•      = env.euler_residual(s, a, s_next_•, a_next_•, β)
    loss     = E[f_main · f_fork]    (crossprod)
            or E[f_main²]            (mse)

Existing `train_er` and the production env stay untouched.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import tensorflow as tf

from src.v2.data.pipeline import build_iterator, validate_dataset_keys
from src.v2.environments.parameterized_basic_investment import (
    ParameterizedBasicInvestmentEnv,
)
from src.v2.estimation.beta_sampler import BetaSampler
from src.v2.networks.policy import ParameterizedPolicyNetwork
from src.v2.trainers.config import ERConfig
from src.v2.trainers.core import (
    StopTracker,
    append_history_row,
    build_target_policy,
    capture_checkpoint,
    polyak_update,
)
from src.v2.utils.seeding import fold_in_seed, make_seed_int, seed_runtime


_DATASET_KEYS = ["s_endo", "z"]


def fit_normalizer_param_flat(
    env: ParameterizedBasicInvestmentEnv,
    dataset: Mapping[str, tf.Tensor],
    beta_sampler: BetaSampler,
    *networks,
    seed,
) -> None:
    """Fit each network's StaticNormalizer on the joint (s, β) distribution.

    Approach: pair every (s_endo[i], z[i]) row with one β-draw from the
    prior. The resulting (N, state_dim + beta_dim) tensor is fed to each
    network's normalizer. After this call, the normalizer sees one
    full-coverage sample of the prior; per-minibatch β draws encountered
    during training will be in-distribution.

    Args:
        env:          ParameterizedBasicInvestmentEnv (for merge_state).
        dataset:      Flat dict with keys s_endo, z.
        beta_sampler: BetaSampler instance.
        networks:     Networks to fit (any GenericNetwork with .normalizer.fit).
        seed:         MasterSeed pair for the β draws used during fit.
    """
    s_endo = dataset["s_endo"]
    z      = dataset["z"]
    s      = env.merge_state(s_endo, z)              # (N, state_dim)
    n      = int(s.shape[0])
    beta   = beta_sampler.sample(n, seed=seed)       # (N, beta_dim)
    x      = tf.concat([s, beta], axis=-1)           # (N, state_dim + beta_dim)
    for net in networks:
        net.normalizer.fit(x)


def train_er_param(
    env: ParameterizedBasicInvestmentEnv,
    policy: ParameterizedPolicyNetwork,
    beta_sampler: BetaSampler,
    train_dataset: Mapping[str, tf.Tensor],
    val_dataset: Optional[Mapping[str, tf.Tensor]] = None,
    config: Optional[ERConfig] = None,
    eval_callback: Optional[Any] = None,
    normalizer_beta_sampler: Optional[BetaSampler] = None,
) -> dict:
    """Train a β-conditional policy via Euler-residual minimization.

    Args:
        env:           ParameterizedBasicInvestmentEnv instance.
        policy:        ParameterizedPolicyNetwork instance to train.
        beta_sampler:  BetaSampler that draws β ∼ prior per minibatch.
        train_dataset: Flat dict with keys (s_endo, z). Any extra keys are
                       ignored.
        val_dataset:   Optional validation dict; passed to `eval_callback`
                       if provided. Built-in defaults skip eval when None.
        config:        ERConfig (reuses the existing trainer config).
        eval_callback: Optional callable
                       `(step, env, policy, value_net, val_dataset) -> dict`.
                       The β-conditional setting has no canonical default
                       eval metric, so leave this None unless the notebook
                       supplies one.

    Returns:
        dict with keys: policy, history, config, wall_time_sec, plus any
        StopTracker output.
    """
    config = config or ERConfig()
    env.validate_nn_training_support("train_er_param")
    seed_runtime(
        config.master_seed, "train_er_param",
        strict_reproducibility=config.strict_reproducibility,
    )

    validate_dataset_keys(
        train_dataset, _DATASET_KEYS, "train_er_param", "train_dataset")

    # Target policy: a vanilla PolicyNetwork with input_dim = state_dim + beta_dim.
    # build_target_policy reads `policy.input_dim` and matches it exactly.
    target_policy = build_target_policy(policy)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)

    # ------------------------------------------------------------------
    # Fit normalizer on joint (s, β) distribution.
    # If `normalizer_beta_sampler` is provided, use it instead of the
    # training sampler — useful for curriculum stages where the training
    # distribution changes but we want a stable normalizer baseline.
    # ------------------------------------------------------------------
    norm_sampler = normalizer_beta_sampler if normalizer_beta_sampler is not None else beta_sampler
    fit_seed = fold_in_seed(config.master_seed, "train_er_param", "normalizer_beta")
    fit_normalizer_param_flat(
        env, train_dataset, norm_sampler, policy, target_policy, seed=fit_seed)

    # ------------------------------------------------------------------
    # Mini-batch iterator
    # ------------------------------------------------------------------
    train_iter = build_iterator(
        {"s_endo": train_dataset["s_endo"], "z": train_dataset["z"]},
        config.batch_size,
        shuffle_seed=make_seed_int(
            config.master_seed, "train_er_param", "batch_shuffle"),
    )

    # ------------------------------------------------------------------
    # Stop tracker (only fires when config.monitor is set)
    # ------------------------------------------------------------------
    stop_tracker = StopTracker(
        monitor=config.monitor,
        mode=config.mode,
        threshold=config.threshold,
        threshold_patience=config.threshold_patience,
        plateau_patience=config.plateau_patience,
        plateau_min_delta=config.plateau_min_delta,
        plateau_rel_delta=config.plateau_rel_delta,
        min_steps_before_stop=config.min_steps_before_stop,
    )

    exo_dim   = env.exo_dim()
    batch_sz  = config.batch_size

    @tf.function
    def _train_step(s_endo, z, beta, eps_main, eps_fork):
        z_next_main = env.exogenous_transition(z, eps_main, beta)
        z_next_fork = env.exogenous_transition(z, eps_fork, beta)
        with tf.GradientTape() as tape:
            s = env.merge_state(s_endo, z)
            a = policy(s, beta, training=False)

            k_next       = env.endogenous_transition(s_endo, a, z, beta)
            s_next_main  = env.merge_state(k_next, z_next_main)
            s_next_fork  = env.merge_state(k_next, z_next_fork)

            # Target-policy call uses pre-concatenated [s, β] (single arg).
            x_next_main  = tf.concat([s_next_main, beta], axis=-1)
            x_next_fork  = tf.concat([s_next_fork, beta], axis=-1)
            a_next_main  = target_policy(x_next_main, training=False)
            a_next_fork  = target_policy(x_next_fork, training=False)

            f1 = env.euler_residual(s, a, s_next_main, a_next_main, beta)
            f2 = env.euler_residual(s, a, s_next_fork, a_next_fork, beta)

            if config.loss_type == "crossprod":
                loss = tf.reduce_mean(f1 * f2)
            else:
                loss = tf.reduce_mean(f1 ** 2)
        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))
        polyak_update(policy, target_policy, tau=config.polyak_rate)
        return loss, f1, f2

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: dict = {}
    train_start = time.perf_counter()
    last_step = -1

    for step, batch in enumerate(train_iter.take(config.n_steps)):
        last_step = step

        # Per-step seeds — different namespaces avoid β/ε draw collisions.
        beta_seed = fold_in_seed(
            config.master_seed, "train_er_param", "step", step, "beta")
        main_seed = tf.constant(fold_in_seed(
            config.master_seed, "train_er_param", "step", step, "eps_main"),
            dtype=tf.int32)
        fork_seed = tf.constant(fold_in_seed(
            config.master_seed, "train_er_param", "step", step, "eps_fork"),
            dtype=tf.int32)

        beta_t   = beta_sampler.sample(batch_sz, seed=beta_seed)
        eps_main = tf.random.stateless_normal(
            (batch_sz, exo_dim), seed=main_seed, dtype=tf.float32)
        eps_fork = tf.random.stateless_normal(
            (batch_sz, exo_dim), seed=fork_seed, dtype=tf.float32)

        loss, f1, f2 = _train_step(
            batch["s_endo"], batch["z"], beta_t, eps_main, eps_fork)

        # Evaluation cadence
        elapsed_sec = time.perf_counter() - train_start
        cap_reached = (
            config.max_wall_time_sec is not None
            and elapsed_sec >= config.max_wall_time_sec
        )

        is_eval = (
            step % config.eval_interval == 0
            or step == config.n_steps - 1
            or cap_reached
        )
        if is_eval:
            base_scalars = {
                "loss": float(loss),
                "euler_residual_mse": float(tf.reduce_mean(f1 ** 2)),
            }
            eval_metrics: dict = {}
            if eval_callback is not None:
                from src.v2.trainers.core import run_eval_callback
                eval_metrics = run_eval_callback(
                    step, env, policy, None, val_dataset,
                    eval_callback=eval_callback)

            stop_on_threshold = stop_tracker.record_eval(
                step, elapsed_sec, {**base_scalars, **eval_metrics})
            if not stop_on_threshold and cap_reached:
                stop_tracker.finalize("max_wall_time", step, elapsed_sec)
            elif not stop_on_threshold and step == config.n_steps - 1:
                stop_tracker.finalize("max_steps", step, elapsed_sec)

            append_history_row(
                history, step, elapsed_sec,
                base_scalars=base_scalars,
                eval_metrics=eval_metrics)
            print(
                f"ER-param step {step:5d} | loss={float(loss):.6f} "
                f"| residual_mse={base_scalars['euler_residual_mse']:.6f} "
                f"| elapsed={elapsed_sec:.1f}s"
            )
            capture_checkpoint(step, config, policy=policy)
            if stop_on_threshold or cap_reached:
                break

    wall_time_sec = time.perf_counter() - train_start
    if stop_tracker.stop_reason is None and last_step >= 0:
        stop_tracker.finalize("max_steps", last_step, wall_time_sec)

    return {
        "policy": policy,
        "target_policy": target_policy,
        "history": history,
        "config": config,
        "wall_time_sec": wall_time_sec,
        **stop_tracker.result_dict(),
    }
