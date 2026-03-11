"""BRM trainer with all three FOC variants — experimental archive.

Backs up the three-way FOC design (euler / autodiff / proxy) that was
explored and ultimately ruled out in favour of the minimal autodiff-only
BRM trainer in src/v2/trainers/brm.py.

FOC methods:
  "euler"    — hand-coded Euler residual from env (no V dependency)
  "autodiff" — ∂r/∂a + γ(∂f/∂a)^T ∇V via auto-diff (depends on V)
  "proxy"    — 3-net: ξ_ψ(s) replaces ∇V in FOC + L_GC tether

Config and trainer are self-contained.  Shared utilities are imported
from src.v2.trainers.core (read-only dependency).
"""

import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional

from src.v2.trainers.core import (
    SeedSchedule, ReplayBuffer, collect_transitions,
    generate_eval_dataset, evaluate_euler_residual,
    evaluate_bellman_residual_v, polyak_update, build_target_policy,
)
from src.v2.trainers.config import NetworkConfig, OptimizerConfig, TrainingConfig


# ============================================================================
# Config
# ============================================================================

@dataclass
class BRMFocConfig(TrainingConfig):
    """BRM configuration with full three-way FOC support.

    foc_method options:
      "euler"    — hand-coded Euler residual from env (no V dep)
      "autodiff" — ∂r/∂a + γ(∂f/∂a)^T ∇V via auto-diff (depends on V)
      "proxy"    — 3-net: ξ_ψ(s) ≈ ∇_s V replaces ∇V in FOC, breaking
                   co-adaptation.  Adds L_GC gradient-consistency loss.
    """
    loss_type: str = "crossprod"       # "crossprod" (AiO) or "mse"
    weight_foc: float = 1.0            # FOC / OPT loss weight (ω_OPT)
    use_foc: bool = True               # enable FOC loss for policy
    foc_method: str = "euler"          # "euler", "autodiff", or "proxy"
    br_scale: float = 1.0              # normalizer for Bellman residual

    # --- proxy (3-net) specific ---
    weight_gc: float = 1.0             # ω_GC: gradient consistency loss weight
    gc_detach_value: bool = False      # if True, stop-gradient on φ in L_GC


# ============================================================================
# FOC helpers
# ============================================================================

def _autodiff_foc(env, s, a, eps, dV_ds_next, gamma, temperature):
    """FOC residual via auto-diff: ∂r/∂a + γ·(∂f/∂a)^T·dV_ds_next."""
    with tf.GradientTape() as tape:
        tape.watch(a)
        r = env.reward(s, a, temperature=temperature)
    dr_da = tape.gradient(r, a)

    with tf.GradientTape() as tape:
        tape.watch(a)
        s_next = env.transition(s, a, eps)
    vjp = tape.gradient(s_next, a, output_gradients=dV_ds_next)

    return dr_da + gamma * vjp


def _proxy_foc_step(env, policy, value_net, grad_proxy, s,
                    eps_main, eps_fork, gamma, temperature, config):
    """Compute L_OPT and L_GC for the 3-net proxy method.

    Returns (loss_opt, loss_gc, grads_policy, grads_proxy_opt,
             grads_value_gc, grads_proxy_gc).
    """
    # ---- L_OPT: FOC with ξ_ψ replacing ∇V ----
    # Trains θ (policy) and ψ (grad proxy).  φ (value) is NOT updated.
    with tf.GradientTape(persistent=True) as tape_opt:
        a = policy(s, training=True)
        s_next_main = env.transition(s, a, eps_main)
        s_next_fork = env.transition(s, a, eps_fork)

        # ξ_ψ(s') replaces ∇_s V_φ(s').
        xi_main = grad_proxy(s_next_main, training=True)
        xi_fork = grad_proxy(s_next_fork, training=True)

        f1 = _autodiff_foc(env, s, a, eps_main, xi_main, gamma, temperature)
        f2 = _autodiff_foc(env, s, a, eps_fork, xi_fork, gamma, temperature)

        if config.loss_type == "crossprod":
            loss_opt = tf.reduce_mean(tf.reduce_sum(f1 * f2, axis=-1))
        else:
            loss_opt = tf.reduce_mean(tf.reduce_sum(f1 ** 2, axis=-1))

    grads_policy = tape_opt.gradient(loss_opt, policy.trainable_variables)
    grads_proxy_opt = tape_opt.gradient(loss_opt, grad_proxy.trainable_variables)
    del tape_opt

    # ---- L_GC: gradient consistency  ‖∇_s V_φ(s') − ξ_ψ(s')‖² ----
    # Stop-gradient on θ: use detached action to compute s'.
    a_detached = policy(s, training=False)
    s_next_gc = env.transition(s, a_detached, eps_main)
    s_next_gc = tf.stop_gradient(s_next_gc)  # fix evaluation points

    if not config.gc_detach_value:
        # ∇_s V_φ inside outer tape so ∂L_GC/∂φ flows (2nd-order).
        with tf.GradientTape(persistent=True) as tape_gc:
            with tf.GradientTape(
                    watch_accessed_variables=False) as tape_dv:
                tape_dv.watch(s_next_gc)
                v_next = value_net(s_next_gc, training=False)
            dV_target = tape_dv.gradient(v_next, s_next_gc)
            xi_gc = grad_proxy(s_next_gc, training=True)
            loss_gc = tf.reduce_mean(
                tf.reduce_sum(tf.square(dV_target - xi_gc), axis=-1))
        grads_proxy_gc = tape_gc.gradient(
            loss_gc, grad_proxy.trainable_variables)
        grads_value_gc = tape_gc.gradient(
            loss_gc, value_net.trainable_variables)
        del tape_gc
    else:
        # Detached: ∇_s V_φ is a fixed target — only ψ gets gradients.
        with tf.GradientTape(
                watch_accessed_variables=False) as tape_dv:
            tape_dv.watch(s_next_gc)
            v_next = value_net(s_next_gc, training=False)
        dV_target = tf.stop_gradient(tape_dv.gradient(v_next, s_next_gc))
        with tf.GradientTape() as tape_gc:
            xi_gc = grad_proxy(s_next_gc, training=True)
            loss_gc = tf.reduce_mean(
                tf.reduce_sum(tf.square(dV_target - xi_gc), axis=-1))
        grads_proxy_gc = tape_gc.gradient(
            loss_gc, grad_proxy.trainable_variables)
        grads_value_gc = None

    return (loss_opt, loss_gc, grads_policy, grads_proxy_opt,
            grads_value_gc, grads_proxy_gc)


# ============================================================================
# Trainer
# ============================================================================

def train_brm_foc(env, policy, value_net, config: BRMFocConfig = None,
                  grad_proxy=None):
    """Train policy and value networks using BRM with selectable FOC method.

    Args:
        env: MDPEnvironment instance.
        policy: PolicyNetwork instance.
        value_net: StateValueNetwork instance.
        config: BRMFocConfig with hyperparameters.
        grad_proxy: ValueGradientProxy instance (required for foc_method="proxy").

    Returns:
        dict with training history, final policy, and value network.
    """
    config = config or BRMFocConfig()
    seed_schedule = SeedSchedule(config.master_seed)
    gamma = env.discount()
    temperature = config.temperature

    value_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.critic_optimizer.learning_rate,
        clipnorm=config.critic_optimizer.clipnorm)
    policy_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.policy_optimizer.learning_rate,
        clipnorm=config.policy_optimizer.clipnorm)

    proxy_optimizer = None
    if config.foc_method == "proxy":
        if grad_proxy is None:
            raise ValueError(
                "grad_proxy is required for foc_method='proxy'")
        proxy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.critic_optimizer.learning_rate,
            clipnorm=config.critic_optimizer.clipnorm)

    target_policy = build_target_policy(policy)

    buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        state_dim=env.state_dim(),
        action_dim=env.action_dim())
    eval_data = generate_eval_dataset(env, config.eval_size, seed_schedule)

    history = {
        "step": [], "loss": [], "loss_br": [], "loss_foc": [],
        "euler_residual": [], "bellman_residual": [],
    }
    if config.foc_method == "proxy":
        history["loss_gc"] = []

    for step in range(config.n_steps):
        # --- Data collection ---
        if config.data_mode == "online":
            transitions = collect_transitions(
                env, policy, config.batch_size, seed_schedule, step,
                temperature=temperature)
            buffer.add(
                transitions["s"], transitions["a"], transitions["r"],
                transitions["s_next"], transitions["eps"])

        if len(buffer) < config.batch_size:
            continue
        batch = buffer.sample(config.batch_size)

        shock_seed_main = seed_schedule.training_seed(
            step, SeedSchedule.VAR_SHOCK_MAIN)
        shock_seed_fork = seed_schedule.training_seed(
            step, SeedSchedule.VAR_SHOCK_FORK)
        eps_main = env.sample_shocks(config.batch_size, seed=shock_seed_main)
        eps_fork = env.sample_shocks(config.batch_size, seed=shock_seed_fork)

        s = batch["s"]

        # ========== L_BR: Bellman residual (trains V) ==========
        with tf.GradientTape() as tape_v:
            a = policy(s, training=False)
            v_s = tf.squeeze(value_net(s, training=True))

            r = env.reward(s, a, temperature=temperature)
            r = tf.squeeze(r) if r.shape.rank > 1 else r

            s_next_main = env.transition(s, a, eps_main)
            s_next_fork = env.transition(s, a, eps_fork)

            v_next_main = tf.squeeze(value_net(s_next_main, training=False))
            v_next_fork = tf.squeeze(value_net(s_next_fork, training=False))

            f_br_main = (v_s - r - gamma * v_next_main) / config.br_scale
            f_br_fork = (v_s - r - gamma * v_next_fork) / config.br_scale

            if config.loss_type == "crossprod":
                loss_br = tf.reduce_mean(f_br_main * f_br_fork)
            else:
                loss_br = tf.reduce_mean(f_br_main ** 2)

        grads_v = tape_v.gradient(loss_br, value_net.trainable_variables)
        value_optimizer.apply_gradients(
            zip(grads_v, value_net.trainable_variables))

        # ========== L_FOC: policy optimality condition ==========
        loss_foc_val = 0.0
        loss_gc_val = 0.0
        if config.use_foc:
            if config.foc_method == "euler":
                with tf.GradientTape() as tape_p:
                    a = policy(s, training=True)
                    s_next_main = env.transition(s, a, eps_main)
                    s_next_fork = env.transition(s, a, eps_fork)
                    a_next_main = target_policy(s_next_main, training=False)
                    a_next_fork = target_policy(s_next_fork, training=False)
                    f1 = env.euler_residual(s, a, s_next_main, a_next_main,
                                            temperature=temperature)
                    f2 = env.euler_residual(s, a, s_next_fork, a_next_fork,
                                            temperature=temperature)
                    if config.loss_type == "crossprod":
                        loss_foc_val = tf.reduce_mean(f1 * f2)
                    else:
                        loss_foc_val = tf.reduce_mean(f1 ** 2)
                grads_p = tape_p.gradient(
                    loss_foc_val, policy.trainable_variables)
                policy_optimizer.apply_gradients(
                    zip(grads_p, policy.trainable_variables))

            elif config.foc_method == "autodiff":
                with tf.GradientTape() as tape_p:
                    a = policy(s, training=True)
                    s_next_main = env.transition(s, a, eps_main)
                    s_next_fork = env.transition(s, a, eps_fork)
                    with tf.GradientTape(
                            watch_accessed_variables=False) as t1:
                        t1.watch(s_next_main)
                        vm = value_net(s_next_main, training=False)
                    dV_main = t1.gradient(vm, s_next_main)
                    with tf.GradientTape(
                            watch_accessed_variables=False) as t2:
                        t2.watch(s_next_fork)
                        vf = value_net(s_next_fork, training=False)
                    dV_fork = t2.gradient(vf, s_next_fork)
                    f1 = _autodiff_foc(env, s, a, eps_main, dV_main,
                                       gamma, temperature)
                    f2 = _autodiff_foc(env, s, a, eps_fork, dV_fork,
                                       gamma, temperature)
                    if config.loss_type == "crossprod":
                        loss_foc_val = tf.reduce_mean(
                            tf.reduce_sum(f1 * f2, axis=-1))
                    else:
                        loss_foc_val = tf.reduce_mean(
                            tf.reduce_sum(f1 ** 2, axis=-1))
                grads_p = tape_p.gradient(
                    loss_foc_val, policy.trainable_variables)
                policy_optimizer.apply_gradients(
                    zip(grads_p, policy.trainable_variables))

            elif config.foc_method == "proxy":
                (loss_foc_val, loss_gc_val,
                 grads_p, grads_psi_opt,
                 grads_v_gc, grads_psi_gc) = _proxy_foc_step(
                    env, policy, value_net, grad_proxy, s,
                    eps_main, eps_fork, gamma, temperature, config)

                policy_optimizer.apply_gradients(
                    zip(grads_p, policy.trainable_variables))

                combined_psi = [
                    g_opt + config.weight_gc * g_gc
                    for g_opt, g_gc in zip(grads_psi_opt, grads_psi_gc)]
                proxy_optimizer.apply_gradients(
                    zip(combined_psi, grad_proxy.trainable_variables))

                if grads_v_gc is not None:
                    pairs = [(config.weight_gc * g, v)
                             for g, v in zip(grads_v_gc,
                                             value_net.trainable_variables)
                             if g is not None]
                    if pairs:
                        value_optimizer.apply_gradients(pairs)

        polyak_update(policy, target_policy, tau=config.polyak_rate)

        loss = (loss_br + config.weight_foc * loss_foc_val
                + config.weight_gc * loss_gc_val)

        # --- Evaluation ---
        if step % config.eval_interval == 0 or step == config.n_steps - 1:
            er = evaluate_euler_residual(
                env, policy, eval_data, temperature=temperature)
            br = evaluate_bellman_residual_v(
                env, policy, value_net, eval_data, temperature=temperature)
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["loss_br"].append(float(loss_br))
            history["loss_foc"].append(float(loss_foc_val))
            history["euler_residual"].append(er)
            history["bellman_residual"].append(br)
            if config.foc_method == "proxy":
                history["loss_gc"].append(float(loss_gc_val))
                print(f"BRM step {step:5d} | loss={float(loss):.6f} | "
                      f"L_br={float(loss_br):.6f} | "
                      f"L_opt={float(loss_foc_val):.6f} | "
                      f"L_gc={float(loss_gc_val):.6f} | "
                      f"euler={er:.6f} | bellman={br:.6f}")
            else:
                print(f"BRM step {step:5d} | loss={float(loss):.6f} | "
                      f"L_br={float(loss_br):.6f} | "
                      f"L_foc={float(loss_foc_val):.6f} | "
                      f"euler={er:.6f} | bellman={br:.6f}")

    result = {
        "policy": policy, "value_net": value_net,
        "history": history, "config": config,
    }
    if config.foc_method == "proxy":
        result["grad_proxy"] = grad_proxy
    return result
