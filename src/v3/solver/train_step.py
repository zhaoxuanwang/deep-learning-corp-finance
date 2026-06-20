"""One policy-iteration gradient step (DF26 Sec 3.4 Steps 2-3).

Step 2 (policy evaluation): update the value network to minimize the mean squared
Bellman residual (Eq 28), policy networks fixed. Step 3 (policy improvement):
update the policy networks to maximize the Bellman right-hand side (Eq 29) using the
just-updated value network, value network fixed. Two separate Adam steps.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.solver import bellman


def make_step_fn(bundle, ext, grid_cfg, x_nodes, w_nodes, tau, value_opt, policy_opt,
                 compile_step: bool = True):
    """Build the per-step training function (optionally graph-compiled)."""

    def step(batch):
        # Step 2: policy evaluation (value updated, policy fixed).
        with tf.GradientTape() as tape:
            residual, _, _ = bellman.residual_and_rhs(
                bundle, batch, ext, grid_cfg, x_nodes, w_nodes, tau)
            loss_v = tf.reduce_mean(tf.square(residual))
        grads_v = tape.gradient(loss_v, bundle.value_variables)
        value_opt.apply_gradients(zip(grads_v, bundle.value_variables))

        # Step 3: policy improvement (policy updated, value fixed, uses updated value).
        with tf.GradientTape() as tape:
            _, rhs, _ = bellman.residual_and_rhs(
                bundle, batch, ext, grid_cfg, x_nodes, w_nodes, tau)
            loss_pi = -tf.reduce_mean(rhs)
        grads_pi = tape.gradient(loss_pi, bundle.policy_variables)
        policy_opt.apply_gradients(zip(grads_pi, bundle.policy_variables))
        return loss_v, loss_pi

    return tf.function(step, reduce_retracing=True) if compile_step else step
