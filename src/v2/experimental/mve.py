"""Model-Based Value Expansion (MVE-DDPG) trainer — generic.

Actor-critic method with Q(s,a) critic. The critic is trained on
multi-step MVE targets. The actor maximizes Q(s, pi(s)) using the
deterministic policy gradient (no differentiation through r or f).

With λ-preprocessing (reward_scale = (1-γ)/μ̂_{|r|}), the critic
predicts unit-free Q̂ ≈ O(1) at the Bellman fixed point.
"""

import tensorflow as tf
from src.v2.trainers.config import MVEConfig
from src.v2.trainers.core import (
    SeedSchedule, ReplayBuffer, collect_transitions,
    generate_eval_dataset, evaluate_euler_residual,
    evaluate_bellman_residual,
    polyak_update, hard_update,
)
from src.v2.networks.policy import PolicyNetwork
from src.v2.networks.critic import CriticNetwork


def _build_target(network, name_suffix="_target"):
    """Clone a network and copy weights for use as a target network."""
    if isinstance(network, PolicyNetwork):
        target = PolicyNetwork(
            state_dim=network.input_dim,
            action_dim=network.action_dim,
            action_low=network.action_low,
            action_high=network.action_high,
            action_center=network.action_center,
            action_half_range=network.action_half_range,
            n_layers=len(network.hidden_stack.dense_layers),
            n_neurons=network.hidden_stack.dense_layers[0].units,
            name=network.name + name_suffix,
        )
        dummy_s = tf.zeros((1, network.input_dim))
        target(dummy_s)
    elif isinstance(network, CriticNetwork):
        target = CriticNetwork(
            state_dim=network.state_dim,
            action_dim=network.action_dim,
            n_layers=len(network.hidden_stack.dense_layers),
            n_neurons=network.hidden_stack.dense_layers[0].units,
            name=network.name + name_suffix,
        )
        dummy_s = tf.zeros((1, network.state_dim))
        dummy_a = tf.zeros((1, network.action_dim))
        target(dummy_s, dummy_a)
    else:
        raise TypeError(f"Unsupported network type: {type(network)}")
    hard_update(network, target)
    return target


def _mve_rollout_and_targets(env, target_policy, target_critic,
                             s_batch, seed_schedule, step, config):
    """Imagination rollout and MVE target construction.

    Rolls out T steps using target policy and known dynamics.
    Builds TD-k mixture targets at each depth.  Rewards are scaled
    by ``config.reward_scale`` (λ-preprocessing); the terminal
    bootstrap uses the raw target critic output (already in λ-space
    when the critic is trained on λ-scaled targets).

    Uses VAR_SHOCK_ROLLOUT seeds to avoid collision with data
    collection seeds (VAR_SHOCK_MAIN).

    Returns:
        list of (s_t, a_t, q_target_t) tuples for t=0..T-1.
    """
    gamma = env.discount()
    T = config.mve_horizon
    lam = config.reward_scale
    batch_size = s_batch.shape[0]

    # Collect rollout states and actions.
    states = [s_batch]
    actions = []
    rewards = []

    s = s_batch
    for t in range(T):
        a = target_policy(s, training=False)
        r = env.reward(s, a)
        r = tf.reshape(r, [-1, 1]) if r.shape.rank == 1 else r
        r = lam * r
        actions.append(a)
        rewards.append(r)

        shock_seed = seed_schedule.training_seed(
            step * T + t, SeedSchedule.VAR_SHOCK_ROLLOUT)
        eps = env.sample_shocks(batch_size, seed=shock_seed)
        s_next = env.transition(s, a, eps)
        states.append(s_next)
        s = s_next

    # Terminal bootstrap: target critic already predicts in λ-space.
    a_T = target_policy(states[T], training=False)
    q_terminal = target_critic(states[T], a_T, training=False)

    # Build MVE targets at each depth t: sum of discounted λ-rewards
    # from t to T-1, plus discounted terminal bootstrap.
    targets = []
    for t in range(T):
        # Accumulate rewards from t to T-1.
        q_target = tf.zeros_like(rewards[0])
        discount_j = 1.0
        for j in range(t, T):
            q_target = q_target + discount_j * rewards[j]
            discount_j = discount_j * gamma
        # Add terminal bootstrap.
        q_target = q_target + discount_j * q_terminal

        targets.append((states[t], actions[t], q_target))

    return targets


def train_mve(env, policy, critic, config: MVEConfig = None):
    """Train policy and critic using MVE-DDPG.

    Args:
        env: MDPEnvironment instance.
        policy: PolicyNetwork instance (actor).
        critic: CriticNetwork instance (Q-function).
        config: MVEConfig with hyperparameters.

    Returns:
        dict with training history, final policy and critic.
    """
    config = config or MVEConfig()
    seed_schedule = SeedSchedule(config.master_seed)
    gamma = env.discount()

    # Auto-compute λ-preprocessing if not explicitly set.
    if config.reward_scale is None:
        lam_seed = seed_schedule.pretraining_seed(SeedSchedule.VAR_BELLMAN_SCALE)
        config.reward_scale = env.compute_reward_scale(seed=lam_seed)
        print(f"MVE: auto λ = {config.reward_scale:.6f} "
              f"(Q* · λ ≈ 1)")

    # Build target networks.
    target_policy = _build_target(policy)
    target_critic = _build_target(critic)

    policy_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)
    critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.critic_optimizer.learning_rate,
        clipnorm=config.critic_optimizer.clipnorm)

    # Replay buffer and eval data.
    buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        state_dim=env.state_dim(),
        action_dim=env.action_dim())
    eval_data = generate_eval_dataset(env, config.eval_size, seed_schedule)

    history = {
        "step": [], "actor_loss": [], "critic_loss": [],
        "euler_residual": [], "bellman_residual": [],
    }

    for step in range(config.n_steps):
        # Collect transitions using current policy (online mode).
        if config.data_mode == "online":
            transitions = collect_transitions(
                env, policy, config.batch_size, seed_schedule, step)
            buffer.add(
                transitions["s"], transitions["a"], transitions["r"],
                transitions["s_next"], transitions["eps"])

        if len(buffer) < config.batch_size:
            continue

        # --- Critic Update ---
        for _ in range(config.critic_updates_per_step):
            batch = buffer.sample(config.batch_size)
            s = batch["s"]

            # MVE rollout and target construction (λ-scaled).
            mve_targets = _mve_rollout_and_targets(
                env, target_policy, target_critic,
                s, seed_schedule, step, config)

            # Update critic normalizer once on buffer-sampled states,
            # not on rolled-out states that drift toward the target
            # policy attractor. Matches the LR trainer convention.
            a0 = target_policy(s, training=False)
            critic.update_normalizer(tf.concat([s, a0], axis=-1))

            with tf.GradientTape() as tape:
                # TD-k mixture loss (critic and targets both in λ-space).
                # training=False: normalizer stats frozen during rollout.
                critic_loss = 0.0
                for s_t, a_t, q_target in mve_targets:
                    q_pred = critic(s_t, a_t, training=False)
                    critic_loss = critic_loss + tf.reduce_mean(
                        tf.square(q_pred - tf.stop_gradient(q_target)))
                critic_loss = critic_loss / len(mve_targets)

            critic_grads = tape.gradient(
                critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(
                zip(critic_grads, critic.trainable_variables))

        # --- Actor Update ---
        batch = buffer.sample(config.batch_size)
        s = batch["s"]

        # Update policy normalizer on buffer-sampled states only.
        policy.update_normalizer(s)

        with tf.GradientTape() as tape:
            # DDPG actor gradient: pass raw (unclipped) action to Q.
            _, a_raw = policy(s, training=False, return_raw=True)
            q_value = critic(s, a_raw)
            actor_loss = -tf.reduce_mean(q_value)

        actor_grads = tape.gradient(actor_loss, policy.trainable_variables)
        policy_optimizer.apply_gradients(
            zip(actor_grads, policy.trainable_variables))

        # --- Target Updates ---
        polyak_update(policy, target_policy, tau=config.polyak_rate)
        polyak_update(critic, target_critic, tau=config.polyak_rate)

        # --- Evaluation ---
        if step % config.eval_interval == 0 or step == config.n_steps - 1:
            er = evaluate_euler_residual(env, policy, eval_data)
            br = evaluate_bellman_residual(env, policy, critic, eval_data)
            history["step"].append(step)
            history["actor_loss"].append(float(actor_loss))
            history["critic_loss"].append(float(critic_loss))
            history["euler_residual"].append(er)
            history["bellman_residual"].append(br)
            print(f"MVE step {step:5d} | actor={float(actor_loss):.4f} "
                  f"critic={float(critic_loss):.6f} | "
                  f"euler={er:.6f} bellman={br:.4f}")

    return {
        "policy": policy, "critic": critic,
        "history": history, "config": config,
    }
