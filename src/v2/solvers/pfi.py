"""
src/v2/solvers/pfi.py

Generic Policy Function Iteration (Howard's method) for MDPEnvironment.

Algorithm
---------
Uses the same grid construction and precomputation as VFI (see vfi.py).
The iteration alternates between:

1. Policy evaluation: for a fixed policy (action indices at each state),
   iterate V = R_policy + gamma * P * V[trans_policy] for eval_steps.
2. Policy improvement: one full Bellman maximization step to find a
   better policy.
3. Convergence check: policy indices unchanged.

PFI generally converges in far fewer outer iterations than VFI (quadratic
vs linear rate), at the cost of the inner evaluation loop.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from src.v2.solvers.config import PFIConfig
from src.v2.solvers.grid import (
    build_grids,
    estimate_exo_transition_matrix,
)
from src.v2.solvers.vfi import _precompute_tables, _bellman_step


def _evaluate_policy(
    policy_idx: tf.Tensor,
    v_init: tf.Tensor,
    tables: Dict[str, tf.Tensor],
    eval_steps: int,
) -> tf.Tensor:
    """Policy evaluation: iterate V = R_h + gamma * P * V[trans_h].

    Args:
        policy_idx: (n_exo, n_endo) int32 — action index at each state.
        v_init:     (n_exo, n_endo) initial value estimate.
        tables:     Precomputed tables from _precompute_tables.
        eval_steps: Number of evaluation iterations.

    Returns:
        v_eval: (n_exo, n_endo) evaluated value function.
    """
    n_exo    = tables["n_exo"]
    n_endo   = tables["n_endo"]
    n_action = tables["n_action"]
    gamma    = tables["gamma"]
    prob     = tables["prob_matrix"]   # (n_exo, n_exo)
    reward   = tables["reward"]        # (n_exo, n_endo, n_action)
    trans    = tables["trans_idx"]     # (n_exo, n_endo, n_action)

    # Extract reward and transition for the fixed policy
    # policy_idx: (n_exo, n_endo) -> gather along action axis
    # reward_policy[z, k] = reward[z, k, policy_idx[z, k]]
    pi_flat = tf.reshape(policy_idx, [-1])  # (n_exo * n_endo,)

    reward_flat = tf.reshape(reward, [n_exo * n_endo, n_action])
    reward_policy = tf.gather(reward_flat, tf.expand_dims(pi_flat, 1), batch_dims=1)
    reward_policy = tf.reshape(tf.squeeze(reward_policy, -1), [n_exo, n_endo])

    trans_flat = tf.reshape(trans, [n_exo * n_endo, n_action])
    trans_policy = tf.gather(trans_flat, tf.expand_dims(pi_flat, 1), batch_dims=1)
    trans_policy = tf.reshape(tf.squeeze(trans_policy, -1), [n_exo, n_endo])
    # trans_policy[z, k] = endo_next index under the fixed policy

    # Iterate V = R_h + gamma * E[V(z', k_h')]
    v = v_init
    for _ in range(eval_steps):
        # E_z'[V(z', k_h')] = sum_{z'} P(z, z') * V(z', trans_policy[z, k])
        # Loop over z' to avoid materializing (n_exo, n_exo*n_endo) tensor.
        ev = tf.zeros([n_exo, n_endo], dtype=v.dtype)
        for zp in range(n_exo):
            v_zp = tf.gather(v[zp], trans_policy)  # (n_exo, n_endo)
            w = tf.reshape(prob[:, zp], [n_exo, 1])
            ev = ev + w * v_zp

        v = reward_policy + gamma * ev

    return v


def solve_pfi(
    env,
    train_dataset: Dict[str, tf.Tensor],
    config: PFIConfig = None,
) -> Dict:
    """Generic Policy Function Iteration solver.

    Args:
        env:           MDPEnvironment instance.
        train_dataset: Flattened dataset dict with keys 's_endo', 'z',
                       'z_next_main' (same format as ER/BRM trainers).
        config:        PFIConfig (defaults to small debug-friendly grid).

    Returns:
        Same dict structure as solve_vfi.
    """
    config = config or PFIConfig()
    gc = config.grid

    # 1. Build grids
    grids = build_grids(env, gc.exo_sizes, gc.endo_sizes, gc.action_sizes)

    # 2. Estimate transition matrix
    z_curr = train_dataset["z"].numpy()
    z_next = train_dataset["z_next_main"].numpy()
    prob_matrix = estimate_exo_transition_matrix(
        z_curr, z_next, grids["exo_grids_1d"], alpha=gc.transition_alpha)

    # 3. Precompute tables
    tables = _precompute_tables(env, grids, prob_matrix)

    n_exo  = tables["n_exo"]
    n_endo = tables["n_endo"]

    # 4. PFI iteration
    v_curr = tf.zeros([n_exo, n_endo], dtype=tf.float32)
    # Initial policy: first Bellman step
    _, policy_idx = _bellman_step(v_curr, tables)

    converged = False
    for iteration in range(config.max_iter):
        # Policy evaluation
        v_eval = _evaluate_policy(policy_idx, v_curr, tables, config.eval_steps)

        # Policy improvement
        v_greedy, new_policy_idx = _bellman_step(v_eval, tables)

        # Check convergence: policy unchanged
        policy_changed = bool(tf.reduce_any(
            tf.not_equal(new_policy_idx, policy_idx)).numpy())

        if not policy_changed:
            converged = True
            print(f"PFI converged: {iteration + 1} policy updates")
            v_curr = v_greedy
            policy_idx = new_policy_idx
            break

        policy_idx = new_policy_idx
        v_curr = v_greedy

    if not converged:
        print(f"PFI did not converge after {config.max_iter} iterations")

    # 5. Extract policies in levels
    action_product = tf.constant(grids["action_product"], dtype=tf.float32)
    endo_product   = tf.constant(grids["endo_product"], dtype=tf.float32)

    policy_action = tf.gather(action_product, tf.reshape(policy_idx, [-1]))
    policy_action = tf.reshape(
        policy_action, [n_exo, n_endo, env.action_dim()])

    trans_at_policy = tf.gather(
        tf.reshape(tables["trans_idx"], [n_exo * n_endo, -1]),
        tf.reshape(policy_idx, [n_exo * n_endo, 1]),
        batch_dims=1,
    )
    trans_at_policy = tf.squeeze(trans_at_policy, axis=-1)
    policy_endo = tf.gather(endo_product, trans_at_policy)
    policy_endo = tf.reshape(policy_endo, [n_exo, n_endo, env.endo_dim()])

    return {
        "value":         v_curr,
        "policy_action": policy_action,
        "policy_endo":   policy_endo,
        "grids":         grids,
        "prob_matrix":   prob_matrix,
        "converged":     converged,
        "n_iter":        iteration + 1 if converged else config.max_iter,
        "history":       [],  # PFI tracks policy changes, not value diffs
    }
