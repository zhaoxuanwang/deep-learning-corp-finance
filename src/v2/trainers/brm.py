"""Bellman Residual Minimization (BRM) trainer — offline, generic.

Jointly trains policy π_θ and value V_ϕ by minimizing:
    L_BR:  Bellman residual  V(s) - r(s,a) - γV(s')
    L_FOC: autodiff FOC  ∂r/∂a + γ(∂f_endo/∂a)^T ∇V

Uses AiO cross-product estimator (two independent z-shock draws) for
unbiased squared-expectation estimation.

Data contract
-------------
Both train_dataset and val_dataset are in flattened format:
    s_endo:       (N, endo_dim)    endogenous state k (i.i.d. uniform)
    z:            (N, exo_dim)     current exogenous state
    z_next_main:  (N, exo_dim)     next z from main AR(1) path
    z_next_fork:  (N, exo_dim)     next z from fork path (AiO second draw)

No shocks are sampled inside this trainer — all transition data comes
from the pre-generated dataset.
"""

import time

import tensorflow as tf
from src.v2.trainers.config import BRMConfig
from src.v2.trainers.core import (
    warm_start_value_net,
    StopTracker,
    append_history_row,
    capture_checkpoint,
    run_eval_callback,
)
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys, fit_normalizer_flat,
)
from src.v2.utils.seeding import make_seed_int, seed_runtime


_DATASET_KEYS = ["s_endo", "z", "z_next_main", "z_next_fork"]


def _autodiff_foc(env, s, s_endo, a, z, z_next, dV_ds_next, gamma):
    """FOC residual via auto-diff: ∂r/∂a + γ·(∂s_next/∂a)^T·dV_ds_next.

    s_next = merge_state(endogenous_transition(s_endo, a, z), z_next)
    Only the endogenous part of s_next depends on a; z_next is fixed data.

    Args:
        env:         MDPEnvironment.
        s:           Current merged state (batch, state_dim).
        s_endo:      Current endogenous state (batch, endo_dim).
        a:           Current action (batch, action_dim).
        z:           Current exogenous state (batch, exo_dim).
        z_next:      Next exogenous state (batch, exo_dim) — fixed.
        dV_ds_next:  ∇_s' V(s') (batch, state_dim) — VJP direction.
        gamma:       Discount factor.

    Returns:
        FOC residual: shape (batch, action_dim).
    """
    with tf.GradientTape() as tape:
        tape.watch(a)
        r = env.reward(s, a)
    dr_da = tape.gradient(r, a)

    with tf.GradientTape() as tape:
        tape.watch(a)
        k_next = env.endogenous_transition(s_endo, a, z)
        s_next = env.merge_state(k_next, z_next)
    vjp = tape.gradient(s_next, a, output_gradients=dV_ds_next)

    return dr_da + gamma * vjp


def train_brm(env, policy, value_net, train_dataset: dict,
              val_dataset: dict = None, config: BRMConfig = None,
              eval_callback=None):
    """Train policy and value networks using BRM on a fixed dataset.

    Args:
        env:           MDPEnvironment instance.
        policy:        PolicyNetwork instance.
        value_net:     StateValueNetwork instance.
        train_dataset: Flattened dataset dict (see module docstring).
        val_dataset:   Flattened dataset for evaluation (optional).
        config:        BRMConfig with hyperparameters.
        eval_callback: Optional callable returning eval metrics at checkpoints.

    Returns:
        dict with keys: policy, value_net, history, config.
    """
    config      = config or BRMConfig()
    env.validate_nn_training_support("train_brm")
    seed_runtime(
        config.master_seed, "train_brm",
        strict_reproducibility=config.strict_reproducibility,
    )
    gamma       = env.discount()

    validate_dataset_keys(train_dataset, _DATASET_KEYS, "train_brm", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _DATASET_KEYS, "train_brm", "val_dataset")
    if config.monitor is not None and eval_callback is None and val_dataset is None:
        raise ValueError(
            "train_brm requires val_dataset when monitor is set and no "
            "eval_callback is provided."
        )

    train_start = time.perf_counter()

    value_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.critic_optimizer.learning_rate,
        clipnorm=config.critic_optimizer.clipnorm)
    policy_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)

    # ------------------------------------------------------------------
    # Fit normalizer from full dataset (once, before gradient steps)
    # ------------------------------------------------------------------
    fit_normalizer_flat(env, train_dataset, policy, value_net)

    # ------------------------------------------------------------------
    # Warm-start critic on analytical V (before gradient steps)
    # ------------------------------------------------------------------
    if config.warm_start_epochs > 0:
        warm_start_value_net(env, value_net, train_dataset,
                             n_epochs=config.warm_start_epochs,
                             shuffle_seed=make_seed_int(
                                 config.master_seed,
                                 "train_brm", "warm_start_shuffle",
                             ))
    elif config.warm_start_steps > 0:
        warm_start_value_net(env, value_net, train_dataset,
                             n_steps=config.warm_start_steps,
                             shuffle_seed=make_seed_int(
                                 config.master_seed,
                                 "train_brm", "warm_start_shuffle",
                             ))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(
        train_dataset, config.batch_size,
        shuffle_seed=make_seed_int(
            config.master_seed, "train_brm", "batch_shuffle"),
    )
    history = {}
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
    last_step = -1

    for step, batch in enumerate(train_iter.take(config.n_steps)):
        last_step = step

        s_endo      = batch["s_endo"]        # (B, endo_dim)
        z           = batch["z"]             # (B, exo_dim)
        z_next_main = batch["z_next_main"]   # (B, exo_dim)
        z_next_fork = batch["z_next_fork"]   # (B, exo_dim)

        s = env.merge_state(s_endo, z)       # (B, state_dim)

        # ========== L_BR: Bellman residual (trains V) ==========
        with tf.GradientTape() as tape_v:
            a     = policy(s, training=False)
            v_s   = tf.squeeze(value_net(s, training=True))

            r     = env.reward(s, a)
            r     = tf.squeeze(r) if r.shape.rank > 1 else r

            k_next      = env.endogenous_transition(s_endo, a, z)
            s_next_main = env.merge_state(k_next, z_next_main)
            s_next_fork = env.merge_state(k_next, z_next_fork)

            v_next_main = tf.squeeze(value_net(s_next_main, training=False))
            v_next_fork = tf.squeeze(value_net(s_next_fork, training=False))

            f_br_main = (v_s - r - gamma * v_next_main) / config.br_scale
            f_br_fork = (v_s - r - gamma * v_next_fork) / config.br_scale

            if config.loss_type == "crossprod":
                loss_br = tf.reduce_mean(f_br_main * f_br_fork)
            else:
                loss_br = tf.reduce_mean(f_br_main ** 2)

        grads_v = tape_v.gradient(loss_br, value_net.trainable_variables)
        value_optimizer.apply_gradients(zip(grads_v, value_net.trainable_variables))

        # ========== L_FOC: autodiff FOC (trains π) ==========
        with tf.GradientTape() as tape_p:
            a           = policy(s, training=False)
            k_next      = env.endogenous_transition(s_endo, a, z)
            s_next_main = env.merge_state(k_next, z_next_main)
            s_next_fork = env.merge_state(k_next, z_next_fork)

            # ∇_s' V(s') for VJP in _autodiff_foc
            with tf.GradientTape(watch_accessed_variables=False) as t1:
                t1.watch(s_next_main)
                vm = value_net(s_next_main, training=False)
            dV_main = t1.gradient(vm, s_next_main)

            with tf.GradientTape(watch_accessed_variables=False) as t2:
                t2.watch(s_next_fork)
                vf = value_net(s_next_fork, training=False)
            dV_fork = t2.gradient(vf, s_next_fork)

            f1 = _autodiff_foc(env, s, s_endo, a, z, z_next_main, dV_main, gamma)
            f2 = _autodiff_foc(env, s, s_endo, a, z, z_next_fork, dV_fork, gamma)

            if config.loss_type == "crossprod":
                loss_foc = tf.reduce_mean(tf.reduce_sum(f1 * f2, axis=-1))
            else:
                loss_foc = tf.reduce_mean(tf.reduce_sum(f1 ** 2, axis=-1))

        grads_p = tape_p.gradient(loss_foc, policy.trainable_variables)
        policy_optimizer.apply_gradients(zip(grads_p, policy.trainable_variables))

        loss = loss_br + config.weight_foc * loss_foc

        # Evaluation
        elapsed_sec = time.perf_counter() - train_start
        cap_reached = (
            config.max_wall_time_sec is not None
            and elapsed_sec >= config.max_wall_time_sec
        )
        if step % config.eval_interval == 0 or step == config.n_steps - 1 or cap_reached:
            eval_metrics = run_eval_callback(
                step,
                env,
                policy,
                value_net,
                val_dataset,
                eval_callback=eval_callback,
            )
            elapsed_sec = time.perf_counter() - train_start
            stop_on_threshold = stop_tracker.record_eval(
                step, elapsed_sec, eval_metrics)
            if not stop_on_threshold and cap_reached:
                stop_tracker.finalize("max_wall_time", step, elapsed_sec)
            elif not stop_on_threshold and step == config.n_steps - 1:
                stop_tracker.finalize("max_steps", step, elapsed_sec)

            append_history_row(
                history,
                step,
                elapsed_sec,
                base_scalars={
                    "loss": float(loss),
                    "loss_br": float(loss_br),
                    "loss_foc": float(loss_foc),
                },
                eval_metrics=eval_metrics,
            )
            status = ""
            if stop_on_threshold:
                status = f" | stop={stop_tracker.stop_reason}"
            elif cap_reached:
                status = " | stop=max_wall_time"
            monitor_name = stop_tracker.monitor or next(iter(eval_metrics), None)
            monitor_text = ""
            if monitor_name is not None and monitor_name in eval_metrics:
                monitor_text = (
                    f" | {monitor_name}={float(eval_metrics[monitor_name]):.6f}"
                )
            print(
                f"BRM step {step:5d} | loss={float(loss):.6f} | "
                f"loss_br={float(loss_br):.6f} | "
                f"loss_foc={float(loss_foc):.6f}"
                f"{monitor_text} | elapsed={elapsed_sec:.1f}s{status}"
            )
            capture_checkpoint(step, config, policy=policy, value_net=value_net)
            if stop_on_threshold or cap_reached:
                break

    wall_time_sec = time.perf_counter() - train_start
    if stop_tracker.stop_reason is None and last_step >= 0:
        stop_tracker.finalize("max_steps", last_step, wall_time_sec)

    return {
        "policy":    policy,
        "value_net": value_net,
        "history":   history,
        "config":    config,
        "wall_time_sec": wall_time_sec,
        **stop_tracker.result_dict(),
    }
