"""Euler Residual (ER) trainer — offline, generic.

Minimizes squared Euler equation residuals using AiO cross-product
or MSE loss. Requires the environment to implement euler_residual().

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
from src.v2.trainers.config import ERConfig
from src.v2.trainers.core import (
    evaluate_euler_residual,
    polyak_update,
    build_target_policy,
)
from src.v2.data.pipeline import build_iterator, validate_dataset_keys


_DATASET_KEYS = ["s_endo", "z", "z_next_main", "z_next_fork"]


def train_er(env, policy, train_dataset: dict, val_dataset: dict = None,
             config: ERConfig = None):
    """Train a policy using the Euler Residual method on a fixed dataset.

    Args:
        env:           MDPEnvironment instance (must implement euler_residual).
        policy:        PolicyNetwork instance.
        train_dataset: Flattened dataset dict (see module docstring).
        val_dataset:   Flattened dataset for evaluation (optional).
        config:        ERConfig with hyperparameters.

    Returns:
        dict with keys: policy, history, config.
    """
    config = config or ERConfig()

    validate_dataset_keys(train_dataset, _DATASET_KEYS, "train_er", "train_dataset")
    if val_dataset is not None:
        validate_dataset_keys(val_dataset, _DATASET_KEYS, "train_er", "val_dataset")

    target_policy = build_target_policy(policy)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)

    # ------------------------------------------------------------------
    # Normalizer warm-up — freeze after warm-up
    # ------------------------------------------------------------------
    if config.warmup_steps > 0:
        warmup_iter = build_iterator(train_dataset, config.batch_size)
        for batch in warmup_iter.take(config.warmup_steps):
            s = env.merge_state(batch["s_endo"], batch["z"])
            policy.update_normalizer(s)
    # Normalizer is now warm. All subsequent policy calls use training=False
    # so update_normalizer is never triggered during gradient steps.

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_iter = build_iterator(train_dataset, config.batch_size)
    history = {"step": [], "loss": [], "euler_residual": []}

    for step, batch in enumerate(train_iter.take(config.n_steps)):
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

            # Next actions from TARGET policy (no gradient)
            a_next_main = target_policy(s_next_main, training=False)
            a_next_fork = target_policy(s_next_fork, training=False)

            # Euler residuals
            f1 = env.euler_residual(s, a, s_next_main, a_next_main,
                                    temperature=config.temperature)
            f2 = env.euler_residual(s, a, s_next_fork, a_next_fork,
                                    temperature=config.temperature)

            if config.loss_type == "crossprod":
                loss = tf.reduce_mean(f1 * f2)
            else:   # mse
                loss = tf.reduce_mean(f1 ** 2)

        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))
        polyak_update(policy, target_policy, tau=config.polyak_rate)

        # Evaluation
        if step % config.eval_interval == 0 or step == config.n_steps - 1:
            er = (evaluate_euler_residual(env, policy, val_dataset,
                                          temperature=config.temperature)
                  if val_dataset is not None else float("nan"))
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["euler_residual"].append(er)
            print(f"ER step {step:5d} | loss={float(loss):.6f} | "
                  f"euler_resid={er:.6f}")

    return {"policy": policy, "history": history, "config": config}
