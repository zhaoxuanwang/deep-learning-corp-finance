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

import tensorflow as tf
from src.v2.trainers.config import LRConfig
from src.v2.trainers.core import evaluate_euler_residual
from src.v2.data.pipeline import (
    build_iterator, validate_dataset_keys, fit_normalizer_traj,
)


_TRAIN_KEYS = ["s_endo_0", "z_path", "z_fork"]
_VAL_KEYS   = ["s_endo", "z", "z_next_main", "z_next_fork"]


def train_lr(env, policy, train_dataset: dict, val_dataset: dict = None,
             config: LRConfig = None):
    """Train a policy using the Lifetime Reward method on a fixed dataset.

    Args:
        env:           MDPEnvironment instance.
        policy:        PolicyNetwork instance.
        train_dataset: Trajectory-format dataset dict (see module docstring).
        val_dataset:   Flattened-format dataset for evaluation (optional).
                       If None, Euler residual is not reported.
        config:        LRConfig with hyperparameters.

    Returns:
        dict with keys: policy, history, config.
    """
    config      = config or LRConfig()
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
    train_iter = build_iterator(train_dataset, config.batch_size)
    history = {"step": [], "loss": [], "euler_residual": []}

    for step, batch in enumerate(train_iter.take(config.n_steps)):
        loss = train_step(batch["s_endo_0"], batch["z_path"])

        if schedule:
            schedule.update()
            temp_var.assign(schedule.value)

        # Evaluation
        if step % config.eval_interval == 0 or step == config.n_steps - 1:
            temperature = float(temp_var)
            er = (evaluate_euler_residual(env, policy, val_dataset,
                                          temperature=temperature)
                  if val_dataset is not None else float("nan"))
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["euler_residual"].append(er)
            print(f"LR step {step:5d} | loss={float(loss):.4f} | "
                  f"euler_resid={er:.6f}")

    return {"policy": policy, "history": history, "config": config}
