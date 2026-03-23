"""Lifetime Reward (LR) trainer — offline, generic.

Maximizes discounted cumulative rewards over T-period trajectory rollouts.
Gradients flow backward through the endogenous capital trajectory (BPTT).

Data contract
-------------
train_dataset (trajectory format):
    s_endo_0:  (N, endo_dim)         initial endogenous state (k0)
    z_path:    (N, T+1, exo_dim)     pre-computed exogenous trajectory
    z_fork:    (N, T, exo_dim)       one-step alternative branches

val_dataset (flattened format, for evaluation):
    s_endo:      (M, endo_dim)
    z:           (M, exo_dim)
    z_next_main: (M, exo_dim)
    z_next_fork: (M, exo_dim)

The exogenous path z is fixed and comes from the dataset.
Only the endogenous state k is evolved endogenously under the policy,
so BPTT flows through k but not z.
"""

import time

import tensorflow as tf
from src.v2.trainers.config import LRConfig
from src.v2.trainers.core import (
    StopTracker,
    append_history_row,
    capture_checkpoint,
    run_eval_callback,
)
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys, fit_normalizer_traj,
)
from src.v2.utils.seeding import make_seed_int, seed_runtime


_TRAIN_KEYS = ["s_endo_0", "z_path", "z_fork"]
_VAL_KEYS   = ["s_endo", "z", "z_next_main", "z_next_fork"]


def train_lr(env, policy, train_dataset: dict, val_dataset: dict = None,
             config: LRConfig = None, eval_callback=None):
    """Train a policy using the Lifetime Reward method on a fixed dataset.

    Args:
        env:           MDPEnvironment instance.
        policy:        PolicyNetwork instance.
        train_dataset: Trajectory-format dataset dict (see module docstring).
        val_dataset:   Flattened-format dataset for evaluation (optional).
        config:        LRConfig with hyperparameters.
        eval_callback: Optional callable returning eval metrics at checkpoints.

    Returns:
        dict with keys: policy, history, config.
    """
    config      = config or LRConfig()
    seed_runtime(
        config.master_seed, "train_lr",
        strict_reproducibility=config.strict_reproducibility,
    )
    gamma       = env.discount()
    T           = config.horizon

    # Annealing: env decides; tf.Variable so @tf.function reads it live.
    schedule = env.annealing_schedule()
    temp_var = tf.Variable(
        schedule.value if schedule else config.temperature,
        dtype=tf.float32, trainable=False)

    validate_dataset_keys(train_dataset, _TRAIN_KEYS, "train_lr", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _VAL_KEYS, "train_lr", "val_dataset")
    if config.monitor is not None and eval_callback is None and val_dataset is None:
        raise ValueError(
            "train_lr requires val_dataset when monitor is set and no "
            "eval_callback is provided."
        )

    T_data = train_dataset["z_path"].shape[1] - 1
    if T > T_data:
        raise ValueError(
            f"LRConfig.horizon={T} exceeds dataset horizon={T_data}. "
            f"Re-generate data with DataGeneratorConfig.horizon >= {T}."
        )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)

    # ------------------------------------------------------------------
    # Fit normalizer from full dataset (once, before gradient steps)
    # ------------------------------------------------------------------
    fit_normalizer_traj(env, train_dataset, policy)

    # ------------------------------------------------------------------
    # Compiled training step (tf.function with static unroll)
    # ------------------------------------------------------------------
    # The Python for-loop is statically unrolled during tf.function
    # tracing — produces a fixed graph with no dynamic control flow.
    # This is faster than tf.while_loop on Metal/M1 where dynamic
    # TensorArray ops fall back to CPU.
    use_terminal = config.terminal_value

    @tf.function
    def train_step(s_endo_0, z_path):
        with tf.GradientTape() as tape:
            k            = s_endo_0
            total_reward = tf.zeros(tf.shape(s_endo_0)[0])
            discount_t   = 1.0

            for t in range(T):
                z_t = z_path[:, t, :]
                s_t = env.merge_state(k, z_t)
                a_t = policy(s_t, training=False)
                r_t = tf.reshape(
                    env.reward(s_t, a_t, temperature=temp_var), [-1])
                total_reward = total_reward + discount_t * r_t
                k = env.endogenous_transition(k, a_t, z_t)
                discount_t = discount_t * gamma

            # Terminal value: V^term(s_endo_T) = r(s̄, ā) / (1-γ).
            # Gradients flow through k_T but not through the policy.
            if use_terminal:
                v_term = tf.reshape(
                    env.terminal_value(k, temperature=temp_var), [-1])
                total_reward = total_reward + discount_t * v_term

            loss = -tf.reduce_mean(total_reward)

        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(
        train_dataset, config.batch_size,
        shuffle_seed=make_seed_int(
            config.master_seed, "train_lr", "batch_shuffle"),
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
        loss = train_step(batch["s_endo_0"], batch["z_path"])

        if schedule:
            schedule.update()
            temp_var.assign(schedule.value)

        # Evaluation
        elapsed_sec = time.perf_counter() - train_start
        cap_reached = (
            config.max_wall_time_sec is not None
            and elapsed_sec >= config.max_wall_time_sec
        )
        if step % config.eval_interval == 0 or step == config.n_steps - 1 or cap_reached:
            train_temperature = float(temp_var)
            eval_metrics = run_eval_callback(
                step,
                env,
                policy,
                None,
                val_dataset,
                train_temperature,
                eval_callback=eval_callback,
                eval_temperature=(
                    config.eval_temperature
                    if eval_callback is None else None
                ),
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
                train_temperature,
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
                f"LR step {step:5d} | loss={float(loss):.4f}"
                f"{monitor_text} | temp={train_temperature:.6g} | "
                f"elapsed={elapsed_sec:.1f}s{status}"
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
