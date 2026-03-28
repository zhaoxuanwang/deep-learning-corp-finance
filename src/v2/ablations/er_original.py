"""Supported ER control used by the ablation notebook.

Faithful control implementation of the original Euler Residual method:
the same policy network is used for both sides of the Euler equation,
with no target network or Polyak averaging.  This creates a moving-target
bias — the loss landscape shifts every gradient step because the "target"
next-period actions change with the policy.

This trainer stays outside ``src/v2/trainers`` because it is a comparison
method for notebook 02 rather than part of the core production trainer set.
The implementation reuses the modern v2 infrastructure so comparisons against
the refined ER are fair.
"""

import time

import tensorflow as tf
from src.v2.trainers.config import ERConfig
from src.v2.trainers.core import (
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


def train_er_original(env, policy, train_dataset: dict, val_dataset: dict = None,
                      config: ERConfig = None, eval_callback=None):
    """Train a policy using the original (single-network) ER method.

    Unlike the refined ER trainer, this uses the *same* policy for both
    current and next-period action evaluation — no target network, no
    Polyak updates.

    Args:
        env:           MDPEnvironment instance (must implement euler_residual).
        policy:        PolicyNetwork instance.
        train_dataset: Flattened dataset dict (see module docstring).
        val_dataset:   Flattened dataset for evaluation (optional).
        config:        ERConfig with hyperparameters (polyak_rate is ignored).
        eval_callback: Optional callable returning eval metrics at checkpoints.

    Returns:
        dict with keys: policy, history, config.
    """
    config = config or ERConfig()
    env.validate_nn_training_support("train_er_original")
    seed_runtime(
        config.master_seed, "train_er_original",
        strict_reproducibility=config.strict_reproducibility,
    )

    validate_dataset_keys(train_dataset, _DATASET_KEYS, "train_er_original", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _DATASET_KEYS, "train_er_original", "val_dataset")
    if config.monitor is not None and eval_callback is None and val_dataset is None:
        raise ValueError(
            "train_er_original requires val_dataset when monitor is set and no "
            "eval_callback is provided."
        )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)

    # ------------------------------------------------------------------
    # Fit normalizer from full dataset (once, before gradient steps)
    # ------------------------------------------------------------------
    fit_normalizer_flat(env, train_dataset, policy)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(
        train_dataset, config.batch_size,
        shuffle_seed=make_seed_int(
            config.master_seed, "train_er_original", "batch_shuffle"),
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
    train_start = time.perf_counter()
    last_step = -1

    for step, batch in enumerate(train_iter.take(config.n_steps)):
        last_step = step

        s_endo      = batch["s_endo"]        # (B, endo_dim)
        z           = batch["z"]             # (B, exo_dim)
        z_next_main = batch["z_next_main"]   # (B, exo_dim)
        z_next_fork = batch["z_next_fork"]   # (B, exo_dim)

        with tf.GradientTape() as tape:
            s = env.merge_state(s_endo, z)
            a = policy(s, training=False)

            # Endogenous next state (same k' for both shock draws)
            k_next = env.endogenous_transition(s_endo, a, z)

            # Two next states differing only in z (AiO cross-product)
            s_next_main = env.merge_state(k_next, z_next_main)
            s_next_fork = env.merge_state(k_next, z_next_fork)

            # Next actions from SAME policy (no target network)
            a_next_main = policy(s_next_main, training=False)
            a_next_fork = policy(s_next_fork, training=False)

            # Euler residuals
            f1 = env.euler_residual(s, a, s_next_main, a_next_main)
            f2 = env.euler_residual(s, a, s_next_fork, a_next_fork)

            if config.loss_type == "crossprod":
                loss = tf.reduce_mean(f1 * f2)
            else:   # mse
                loss = tf.reduce_mean(f1 ** 2)

        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

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
                None,
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
                base_scalars={"loss": float(loss)},
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
                f"ER-Orig step {step:5d} | loss={float(loss):.6f}"
                f"{monitor_text} | elapsed={elapsed_sec:.1f}s{status}"
            )
            capture_checkpoint(step, config, policy=policy)
            if stop_on_threshold or cap_reached:
                break

    wall_time_sec = time.perf_counter() - train_start
    if stop_tracker.stop_reason is None and last_step >= 0:
        stop_tracker.finalize("max_steps", last_step, wall_time_sec)

    return {
        "policy": policy,
        "history": history,
        "config": config,
        "wall_time_sec": wall_time_sec,
        **stop_tracker.result_dict(),
    }
