"""
src/v2/solvers/vfi.py

Generic Value Function Iteration (VFI) solver for MDPEnvironment.

Algorithm
---------
1. Build discrete grids over (endo, exo, action) using env.grid_spec()
   or linspace fallback.
2. Estimate Markov transition P(z'|z) from the flattened dataset's
   (z, z_next_main) columns.
3. Precompute:
   - Reward matrix R[i_exo, i_endo, i_action] by evaluating env.reward()
     on all (state, action) grid point combinations.
   - Transition map T[i_exo, i_endo, i_action] -> i_endo_next by evaluating
     env.endogenous_transition() and snapping to the nearest endo grid point.
4. Bellman iteration until sup-norm convergence:
     V_new[z, k] = max_a { R[z, k, a] + gamma * sum_{z'} P[z, z'] * V[z', T[z, k, a]] }

The solver is model-agnostic: the only interface to the model is through
env.reward(), env.endogenous_transition(), env.discount(), and env.grid_spec().
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from src.v2.solvers.config import VFIConfig
from src.v2.solvers.grid import (
    build_grids,
    estimate_exo_transition_matrix,
    snap_to_grid,
)


def _precompute_tables(
    env,
    grids: Dict,
    prob_matrix: np.ndarray,
) -> Dict[str, tf.Tensor]:
    """Precompute reward matrix and transition map on the discrete grid.

    Returns dict with:
        reward:       (n_exo, n_endo, n_action) float32
        trans_idx:    (n_exo, n_endo, n_action) int32  — endo_next flat index
        prob_matrix:  (n_exo, n_exo) float32
        gamma:        scalar float32
    """
    exo_pts    = grids["exo_product"]     # (n_exo, exo_dim)
    endo_pts   = grids["endo_product"]    # (n_endo, endo_dim)
    action_pts = grids["action_product"]  # (n_action, action_dim)

    n_exo    = exo_pts.shape[0]
    n_endo   = endo_pts.shape[0]
    n_action = action_pts.shape[0]
    endo_dim = endo_pts.shape[1]

    # Build all (exo, endo, action) combinations
    # Use meshgrid indices then gather
    idx_exo, idx_endo, idx_action = np.meshgrid(
        np.arange(n_exo), np.arange(n_endo), np.arange(n_action),
        indexing="ij",
    )
    idx_exo    = idx_exo.ravel()
    idx_endo   = idx_endo.ravel()
    idx_action = idx_action.ravel()
    N_total = len(idx_exo)

    s_exo_all    = tf.constant(exo_pts[idx_exo],    dtype=tf.float32)  # (N, exo_dim)
    s_endo_all   = tf.constant(endo_pts[idx_endo],   dtype=tf.float32)  # (N, endo_dim)
    action_all   = tf.constant(action_pts[idx_action], dtype=tf.float32) # (N, act_dim)

    # Full state for reward: s = [s_endo | s_exo]
    s_all = env.merge_state(s_endo_all, s_exo_all)  # (N, state_dim)

    # --- Reward ---
    # Evaluate in chunks to avoid OOM on very large grids
    CHUNK = 500_000
    reward_parts = []
    for start in range(0, N_total, CHUNK):
        end = min(start + CHUNK, N_total)
        r_chunk = env.reward(s_all[start:end], action_all[start:end],
                             temperature=1e-6)
        # Flatten to (chunk,)
        r_chunk = tf.reshape(r_chunk, [-1])
        reward_parts.append(r_chunk)
    reward_flat = tf.concat(reward_parts, axis=0)  # (N_total,)
    reward = tf.reshape(reward_flat, [n_exo, n_endo, n_action])

    # --- Transition map: endo_next = f_endo(s_endo, action, s_exo) ---
    trans_parts = []
    endo_product_tf = tf.constant(endo_pts, dtype=tf.float32)
    for start in range(0, N_total, CHUNK):
        end = min(start + CHUNK, N_total)
        s_endo_next = env.endogenous_transition(
            s_endo_all[start:end], action_all[start:end], s_exo_all[start:end])
        # Snap to nearest endo grid point
        idx_chunk = snap_to_grid(
            tf.cast(s_endo_next, tf.float32), endo_product_tf)
        trans_parts.append(idx_chunk)
    trans_flat = tf.concat(trans_parts, axis=0)  # (N_total,)
    trans_idx = tf.reshape(trans_flat, [n_exo, n_endo, n_action])

    return {
        "reward":      reward,
        "trans_idx":   trans_idx,
        "prob_matrix": tf.constant(prob_matrix, dtype=tf.float32),
        "gamma":       tf.constant(env.discount(), dtype=tf.float32),
        "n_exo":       n_exo,
        "n_endo":      n_endo,
        "n_action":    n_action,
    }


def _bellman_step(
    v_curr: tf.Tensor,
    tables: Dict[str, tf.Tensor],
) -> tuple:
    """One Bellman operator application.

    Args:
        v_curr: Value function, shape (n_exo, n_endo).
        tables: Precomputed tables from _precompute_tables.

    Returns:
        (v_next, policy_idx):
            v_next:     (n_exo, n_endo) updated values.
            policy_idx: (n_exo, n_endo) optimal action flat index.
    """
    n_exo    = tables["n_exo"]
    n_endo   = tables["n_endo"]
    n_action = tables["n_action"]
    gamma    = tables["gamma"]
    prob     = tables["prob_matrix"]   # (n_exo, n_exo)
    reward   = tables["reward"]        # (n_exo, n_endo, n_action)
    trans    = tables["trans_idx"]     # (n_exo, n_endo, n_action)

    # Expected continuation: E[V(z', k')] for each (z, k, a)
    # EV[z, k, a] = sum_{z'} P(z, z') * V(z', trans[z, k, a])
    #
    # Loop over z' to avoid materializing the full (n_exo, n_exo, n_endo, n_action)
    # tensor, which would OOM for large grids. Memory per iteration: O(n_endo*n_action).
    ev = tf.zeros([n_exo, n_endo, n_action], dtype=v_curr.dtype)
    for zp in range(n_exo):
        # V(z', trans[z,k,a]) for this specific z'
        v_zp = tf.gather(v_curr[zp], trans)  # (n_exo, n_endo, n_action)
        # Weight by P(z, z') — broadcast (n_exo, 1, 1) * (n_exo, n_endo, n_action)
        w = tf.reshape(prob[:, zp], [n_exo, 1, 1])
        ev = ev + w * v_zp

    # RHS of Bellman: R + gamma * EV
    rhs = reward + gamma * ev  # (n_exo, n_endo, n_action)

    # Maximize over actions
    v_next     = tf.reduce_max(rhs, axis=-1)        # (n_exo, n_endo)
    policy_idx = tf.argmax(rhs, axis=-1, output_type=tf.int32)  # (n_exo, n_endo)

    return v_next, policy_idx


def solve_vfi(
    env,
    train_dataset: Dict[str, tf.Tensor],
    config: VFIConfig = None,
) -> Dict:
    """Generic Value Function Iteration solver.

    Args:
        env:           MDPEnvironment instance.
        train_dataset: Flattened dataset dict with keys 's_endo', 'z',
                       'z_next_main' (same format as ER/BRM trainers).
        config:        VFIConfig (defaults to small debug-friendly grid).

    Returns:
        Dict with keys:
            value:          (n_exo, n_endo) converged value function.
            policy_action:  (n_exo, n_endo, action_dim) optimal actions in levels.
            policy_endo:    (n_exo, n_endo, endo_dim) optimal next endo state.
            grids:          Grid dict from build_grids.
            prob_matrix:    (n_exo, n_exo) estimated transition matrix.
            converged:      bool.
            n_iter:         number of iterations used.
            final_diff:     final sup-norm difference.
            history:        list of per-iteration diffs.
    """
    config = config or VFIConfig()
    gc = config.grid

    # 1. Build grids
    grids = build_grids(env, gc.exo_sizes, gc.endo_sizes, gc.action_sizes)

    # 2. Estimate transition matrix from dataset
    z_curr = train_dataset["z"].numpy()
    z_next = train_dataset["z_next_main"].numpy()
    prob_matrix = estimate_exo_transition_matrix(
        z_curr, z_next, grids["exo_grids_1d"], alpha=gc.transition_alpha)

    # 3. Precompute tables
    tables = _precompute_tables(env, grids, prob_matrix)

    n_exo  = tables["n_exo"]
    n_endo = tables["n_endo"]

    # 4. Bellman iteration
    v_curr = tf.zeros([n_exo, n_endo], dtype=tf.float32)
    policy_idx = tf.zeros([n_exo, n_endo], dtype=tf.int32)
    history = []

    converged = False
    for iteration in range(config.max_iter):
        v_next, policy_idx = _bellman_step(v_curr, tables)
        diff = float(tf.reduce_max(tf.abs(v_next - v_curr)).numpy())
        history.append(diff)

        if diff < config.tol:
            converged = True
            print(f"VFI converged: {iteration + 1} iterations, diff={diff:.2e}")
            v_curr = v_next
            break
        v_curr = v_next

    if not converged:
        print(f"VFI did not converge after {config.max_iter} iterations, "
              f"diff={history[-1]:.2e}")

    # 5. Extract policies in levels
    action_product = tf.constant(grids["action_product"], dtype=tf.float32)
    endo_product   = tf.constant(grids["endo_product"], dtype=tf.float32)

    # Optimal action at each (z, k)
    policy_action = tf.gather(action_product, tf.reshape(policy_idx, [-1]))
    policy_action = tf.reshape(
        policy_action, [n_exo, n_endo, env.action_dim()])

    # Optimal next endo state at each (z, k)
    trans_at_policy = tf.gather(
        tf.reshape(tables["trans_idx"], [n_exo * n_endo, -1]),
        tf.reshape(policy_idx, [n_exo * n_endo, 1]),
        batch_dims=1,
    )
    trans_at_policy = tf.squeeze(trans_at_policy, axis=-1)  # (n_exo * n_endo,)
    policy_endo = tf.gather(endo_product, trans_at_policy)
    policy_endo = tf.reshape(policy_endo, [n_exo, n_endo, env.endo_dim()])

    return {
        "value":         v_curr,
        "policy_action": policy_action,
        "policy_endo":   policy_endo,
        "grids":         grids,
        "prob_matrix":   prob_matrix,
        "converged":     converged,
        "n_iter":        len(history),
        "final_diff":    history[-1] if history else float("inf"),
        "history":       history,
    }
