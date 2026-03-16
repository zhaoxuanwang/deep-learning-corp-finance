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

import tensorflow as tf
from src.v2.trainers.config import BRMConfig
from src.v2.trainers.core import (
    evaluate_euler_residual,
    evaluate_bellman_residual_v,
    warm_start_value_net,
)
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys, fit_normalizer_flat,
)


_DATASET_KEYS = ["s_endo", "z", "z_next_main", "z_next_fork"]


def _autodiff_foc(env, s, s_endo, a, z, z_next, dV_ds_next, gamma, temperature):
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
        temperature: Smooth-gate temperature.

    Returns:
        FOC residual: shape (batch, action_dim).
    """
    with tf.GradientTape() as tape:
        tape.watch(a)
        r = env.reward(s, a, temperature=temperature)
    dr_da = tape.gradient(r, a)

    with tf.GradientTape() as tape:
        tape.watch(a)
        k_next = env.endogenous_transition(s_endo, a, z)
        s_next = env.merge_state(k_next, z_next)
    vjp = tape.gradient(s_next, a, output_gradients=dV_ds_next)

    return dr_da + gamma * vjp


def train_brm(env, policy, value_net, train_dataset: dict,
              val_dataset: dict = None, config: BRMConfig = None):
    """Train policy and value networks using BRM on a fixed dataset.

    Args:
        env:           MDPEnvironment instance.
        policy:        PolicyNetwork instance.
        value_net:     StateValueNetwork instance.
        train_dataset: Flattened dataset dict (see module docstring).
        val_dataset:   Flattened dataset for evaluation (optional).
        config:        BRMConfig with hyperparameters.

    Returns:
        dict with keys: policy, value_net, history, config.
    """
    config      = config or BRMConfig()
    gamma       = env.discount()
    schedule    = env.annealing_schedule()

    validate_dataset_keys(train_dataset, _DATASET_KEYS, "train_brm", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _DATASET_KEYS, "train_brm", "val_dataset")

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
    if config.warm_start_steps > 0:
        warm_start_value_net(env, value_net, train_dataset,
                             n_steps=config.warm_start_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(train_dataset, config.batch_size)
    history = {
        "step": [], "loss": [], "loss_br": [], "loss_foc": [],
        "euler_residual": [], "bellman_residual": [],
    }

    for step, batch in enumerate(train_iter.take(config.n_steps)):
        temperature = schedule.value if schedule else config.temperature

        s_endo      = batch["s_endo"]        # (B, endo_dim)
        z           = batch["z"]             # (B, exo_dim)
        z_next_main = batch["z_next_main"]   # (B, exo_dim)
        z_next_fork = batch["z_next_fork"]   # (B, exo_dim)

        s = env.merge_state(s_endo, z)       # (B, state_dim)

        # ========== L_BR: Bellman residual (trains V) ==========
        with tf.GradientTape() as tape_v:
            a     = policy(s, training=False)
            v_s   = tf.squeeze(value_net(s, training=True))

            r     = env.reward(s, a, temperature=temperature)
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

            f1 = _autodiff_foc(env, s, s_endo, a, z, z_next_main,
                               dV_main, gamma, temperature)
            f2 = _autodiff_foc(env, s, s_endo, a, z, z_next_fork,
                               dV_fork, gamma, temperature)

            if config.loss_type == "crossprod":
                loss_foc = tf.reduce_mean(tf.reduce_sum(f1 * f2, axis=-1))
            else:
                loss_foc = tf.reduce_mean(tf.reduce_sum(f1 ** 2, axis=-1))

        grads_p = tape_p.gradient(loss_foc, policy.trainable_variables)
        policy_optimizer.apply_gradients(zip(grads_p, policy.trainable_variables))

        loss = loss_br + config.weight_foc * loss_foc

        if schedule:
            schedule.update()

        # Evaluation
        if step % config.eval_interval == 0 or step == config.n_steps - 1:
            er = br = float("nan")
            if val_dataset is not None:
                er = evaluate_euler_residual(
                    env, policy, val_dataset, temperature=temperature)
                br = evaluate_bellman_residual_v(
                    env, policy, value_net, val_dataset, temperature=temperature)
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["loss_br"].append(float(loss_br))
            history["loss_foc"].append(float(loss_foc))
            history["euler_residual"].append(er)
            history["bellman_residual"].append(br)
            print(f"BRM step {step:5d} | loss={float(loss):.6f} | "
                  f"L_br={float(loss_br):.6f} | "
                  f"L_foc={float(loss_foc):.6f} | "
                  f"euler={er:.6f} | bellman={br:.6f}")

    return {
        "policy":    policy,
        "value_net": value_net,
        "history":   history,
        "config":    config,
    }
