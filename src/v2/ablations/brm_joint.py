"""Supported BRM control used by the ablation notebook.

Faithful control implementation of the original joint-regression BRM idea:
the Bellman residual and autodiff FOC losses are summed into one objective,
and a single optimizer step updates both the policy and value network
simultaneously.

This trainer stays outside ``src/v2/trainers`` because it is a comparison
method for notebook 02 rather than part of the core production trainer set.
The implementation reuses the modern v2 infrastructure so comparisons against
the refined BRM are fair.
"""

from __future__ import annotations

import time

import tensorflow as tf

from src.v2.data.pipeline import (
    build_iterator,
    fit_normalizer_flat,
    validate_dataset_keys,
)
from src.v2.trainers.brm import _autodiff_foc
from src.v2.trainers.config import BRMConfig
from src.v2.trainers.core import (
    StopTracker,
    append_history_row,
    capture_checkpoint,
    run_eval_callback,
    warm_start_value_net,
)
from src.v2.utils.seeding import make_seed_int, seed_runtime


_DATASET_KEYS = ["s_endo", "z", "z_next_main", "z_next_fork"]


def train_brm_joint(
    env,
    policy,
    value_net,
    train_dataset: dict,
    val_dataset: dict | None = None,
    config: BRMConfig | None = None,
    eval_callback=None,
):
    """Train BRM with a single joint loss and one optimizer update.

    The loss is:

        L = L_BR + weight_foc * L_FOC

    and the gradient is taken jointly with respect to both policy and value
    parameters.  This preserves the key structural defect of the original BRM:
    Bellman-residual gradients act directly on the policy, so self-consistency
    and optimality are mixed inside one update.
    """

    config = config or BRMConfig()
    env.validate_nn_training_support("train_brm_joint")
    seed_runtime(
        config.master_seed,
        "train_brm_joint",
        strict_reproducibility=config.strict_reproducibility,
    )
    gamma = env.discount()

    validate_dataset_keys(
        train_dataset, _DATASET_KEYS, "train_brm_joint", "train_dataset"
    )
    if val_dataset is not None:
        validate_dataset_keys(
            val_dataset, _DATASET_KEYS, "train_brm_joint", "val_dataset"
        )
    if config.monitor is not None and eval_callback is None and val_dataset is None:
        raise ValueError(
            "train_brm_joint requires val_dataset when monitor is set and no "
            "eval_callback is provided."
        )

    if (
        config.policy_optimizer.learning_rate != config.critic_optimizer.learning_rate
        or config.policy_optimizer.clipnorm != config.critic_optimizer.clipnorm
    ):
        raise ValueError(
            "train_brm_joint uses a single joint optimizer. "
            "Set policy_optimizer and critic_optimizer to identical values."
        )

    train_start = time.perf_counter()

    joint_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm,
    )

    fit_normalizer_flat(env, train_dataset, policy, value_net)

    if config.warm_start_steps > 0:
        warm_start_value_net(
            env,
            value_net,
            train_dataset,
            n_steps=config.warm_start_steps,
            shuffle_seed=make_seed_int(
                config.master_seed, "train_brm_joint", "warm_start_shuffle"
            ),
        )

    train_iter = build_iterator(
        train_dataset,
        config.batch_size,
        shuffle_seed=make_seed_int(
            config.master_seed, "train_brm_joint", "batch_shuffle"
        ),
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

    joint_variables = (
        list(policy.trainable_variables) + list(value_net.trainable_variables)
    )

    for step, batch in enumerate(train_iter.take(config.n_steps)):
        last_step = step

        s_endo = batch["s_endo"]
        z = batch["z"]
        z_next_main = batch["z_next_main"]
        z_next_fork = batch["z_next_fork"]
        s = env.merge_state(s_endo, z)

        with tf.GradientTape() as tape:
            a = policy(s, training=True)
            v_s = tf.squeeze(value_net(s, training=True))

            r = env.reward(s, a)
            r = tf.squeeze(r) if r.shape.rank > 1 else r

            k_next = env.endogenous_transition(s_endo, a, z)
            s_next_main = env.merge_state(k_next, z_next_main)
            s_next_fork = env.merge_state(k_next, z_next_fork)

            v_next_main = tf.squeeze(value_net(s_next_main, training=True))
            v_next_fork = tf.squeeze(value_net(s_next_fork, training=True))

            f_br_main = (v_s - r - gamma * v_next_main) / config.br_scale
            f_br_fork = (v_s - r - gamma * v_next_fork) / config.br_scale
            if config.loss_type == "crossprod":
                loss_br = tf.reduce_mean(f_br_main * f_br_fork)
            else:
                loss_br = tf.reduce_mean(f_br_main ** 2)

            # FOC term remains the same as the refined BRM, but now its
            # higher-order gradient also flows into the critic parameters.
            with tf.GradientTape(watch_accessed_variables=False) as t1:
                t1.watch(s_next_main)
                vm = value_net(s_next_main, training=True)
            dV_main = t1.gradient(vm, s_next_main)

            with tf.GradientTape(watch_accessed_variables=False) as t2:
                t2.watch(s_next_fork)
                vf = value_net(s_next_fork, training=True)
            dV_fork = t2.gradient(vf, s_next_fork)

            f1 = _autodiff_foc(
                env, s, s_endo, a, z, z_next_main, dV_main, gamma
            )
            f2 = _autodiff_foc(
                env, s, s_endo, a, z, z_next_fork, dV_fork, gamma
            )

            if config.loss_type == "crossprod":
                loss_foc = tf.reduce_mean(tf.reduce_sum(f1 * f2, axis=-1))
            else:
                loss_foc = tf.reduce_mean(tf.reduce_sum(f1 ** 2, axis=-1))

            loss = loss_br + config.weight_foc * loss_foc

        grads = tape.gradient(loss, joint_variables)
        joint_optimizer.apply_gradients(zip(grads, joint_variables))

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
            stop_now = stop_tracker.record_eval(step, elapsed_sec, eval_metrics)
            if not stop_now and cap_reached:
                stop_tracker.finalize("max_wall_time", step, elapsed_sec)
            elif not stop_now and step == config.n_steps - 1:
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
            if stop_now:
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
                f"BRM-Joint step {step:5d} | loss={float(loss):.6f} | "
                f"loss_br={float(loss_br):.6f} | "
                f"loss_foc={float(loss_foc):.6f}"
                f"{monitor_text} | elapsed={elapsed_sec:.1f}s{status}"
            )
            capture_checkpoint(step, config, policy=policy, value_net=value_net)
            if stop_now or cap_reached:
                break

    wall_time_sec = time.perf_counter() - train_start
    if stop_tracker.stop_reason is None and last_step >= 0:
        stop_tracker.finalize("max_steps", last_step, wall_time_sec)

    return {
        "policy": policy,
        "value_net": value_net,
        "history": history,
        "config": config,
        "wall_time_sec": wall_time_sec,
        **stop_tracker.result_dict(),
    }
