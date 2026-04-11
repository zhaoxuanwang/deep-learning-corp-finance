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

import time
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
    interp_mode: str = "auto",
) -> tf.Tensor:
    """Policy evaluation: iterate V = R_h + gamma * P * V[trans_h].

    Args:
        policy_idx:  (n_exo, n_endo) int32 — action index at each state.
        v_init:      (n_exo, n_endo) initial value estimate.
        tables:      Precomputed tables from _precompute_tables.
        eval_steps:  Number of evaluation iterations.
        interp_mode: "auto" | "linear" | "none".  Controls how the
                     continuation value is computed (same as _bellman_step).

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
    pi_exp  = tf.expand_dims(pi_flat, 1)    # (n_exo * n_endo, 1)

    reward_flat = tf.reshape(reward, [n_exo * n_endo, n_action])
    reward_policy = tf.gather(reward_flat, pi_exp, batch_dims=1)
    reward_policy = tf.reshape(tf.squeeze(reward_policy, -1), [n_exo, n_endo])

    trans_flat = tf.reshape(trans, [n_exo * n_endo, n_action])
    trans_policy = tf.gather(trans_flat, pi_exp, batch_dims=1)
    trans_policy = tf.reshape(tf.squeeze(trans_policy, -1), [n_exo, n_endo])
    # trans_policy[z, k] = endo_next index under the fixed policy

    # --- Determine interpolation strategy ---
    use_snap = (interp_mode == "none")
    has_1d = "trans_lo" in tables
    has_nd = "trans_lo_0" in tables

    use_interp_1d = (not use_snap) and has_1d
    use_interp_nd = (not use_snap) and (not has_1d) and has_nd

    if use_interp_1d:
        lo_flat = tf.reshape(tables["trans_lo"], [n_exo * n_endo, n_action])
        trans_lo_policy = tf.gather(lo_flat, pi_exp, batch_dims=1)
        trans_lo_policy = tf.reshape(tf.squeeze(trans_lo_policy, -1), [n_exo, n_endo])

        frac_flat = tf.reshape(tables["trans_frac"], [n_exo * n_endo, n_action])
        frac_policy = tf.gather(frac_flat, pi_exp, batch_dims=1)
        frac_policy = tf.reshape(tf.squeeze(frac_policy, -1), [n_exo, n_endo])

        trans_hi_policy = tf.minimum(trans_lo_policy + 1, n_endo - 1)

    if use_interp_nd:
        endo_var_sizes = tables["endo_var_sizes"]  # [n_0, n_1, ...]
        endo_dim_count = len(endo_var_sizes)
        # Row-major strides
        nd_strides = [0] * endo_dim_count
        nd_strides[-1] = 1
        for d in range(endo_dim_count - 2, -1, -1):
            nd_strides[d] = nd_strides[d + 1] * endo_var_sizes[d + 1]
        # Extract per-policy lo/frac for each endo variable
        lo_policy_d = []
        frac_policy_d = []
        hi_policy_d = []
        for d in range(endo_dim_count):
            lo_flat_d = tf.reshape(tables[f"trans_lo_{d}"], [n_exo * n_endo, n_action])
            lo_pol = tf.gather(lo_flat_d, pi_exp, batch_dims=1)
            lo_pol = tf.reshape(tf.squeeze(lo_pol, -1), [n_exo, n_endo])
            lo_policy_d.append(lo_pol)

            frac_flat_d = tf.reshape(tables[f"trans_frac_{d}"], [n_exo * n_endo, n_action])
            frac_pol = tf.gather(frac_flat_d, pi_exp, batch_dims=1)
            frac_pol = tf.reshape(tf.squeeze(frac_pol, -1), [n_exo, n_endo])
            frac_policy_d.append(frac_pol)

            hi_policy_d.append(tf.minimum(lo_pol + 1, endo_var_sizes[d] - 1))

    # Iterate V = R_h + gamma * E[V(z', k_h')]
    v = v_init
    for _ in range(eval_steps):
        # E_z'[V(z', k_h')] = sum_{z'} P(z, z') * V(z', trans_policy[z, k])
        # Loop over z' to avoid materializing (n_exo, n_exo*n_endo) tensor.
        ev = tf.zeros([n_exo, n_endo], dtype=v.dtype)
        for zp in range(n_exo):
            if use_interp_1d:
                v_lo = tf.gather(v[zp], trans_lo_policy)   # (n_exo, n_endo)
                v_hi = tf.gather(v[zp], trans_hi_policy)
                v_zp = (1.0 - frac_policy) * v_lo + frac_policy * v_hi
            elif use_interp_nd:
                v_zp = tf.zeros([n_exo, n_endo], dtype=v.dtype)
                for corner in range(1 << endo_dim_count):
                    weight = tf.ones([n_exo, n_endo], dtype=v.dtype)
                    flat_idx = tf.zeros([n_exo, n_endo], dtype=tf.int32)
                    for d in range(endo_dim_count):
                        if corner & (1 << d):
                            idx_d = hi_policy_d[d]
                            w_d = frac_policy_d[d]
                        else:
                            idx_d = lo_policy_d[d]
                            w_d = 1.0 - frac_policy_d[d]
                        weight = weight * w_d
                        flat_idx = flat_idx + idx_d * nd_strides[d]
                    v_corner = tf.gather(v[zp], flat_idx)
                    v_zp = v_zp + weight * v_corner
            else:
                v_zp = tf.gather(v[zp], trans_policy)      # (n_exo, n_endo)
            w = tf.reshape(prob[:, zp], [n_exo, 1])
            ev = ev + w * v_zp

        v = reward_policy + gamma * ev

    return v


def solve_pfi(
    env,
    train_dataset: Dict[str, tf.Tensor],
    config: PFIConfig = None,
    eval_callback=None,
    value_init: Optional[tf.Tensor] = None,
) -> Dict:
    """Generic Policy Function Iteration solver.

    Args:
        env:           MDPEnvironment instance.
        train_dataset: Flattened dataset dict with keys 's_endo', 'z',
                       'z_next_main' (same format as ER/BRM trainers).
        config:        PFIConfig (defaults to small debug-friendly grid).
        eval_callback: Optional callable(iteration, partial_result) -> dict[str, float].
                       Called after each policy improvement iteration.
                       partial_result contains 'value', 'policy_action', 'policy_endo', 'grids'.
        value_init:    Optional initial value function for warm-starting,
                       shape (n_exo, n_endo).  When None (default), starts
                       from zeros (cold start).  A warm V also produces a
                       warm initial policy via the first Bellman step.

    Returns:
        Dict with keys:
            value:          (n_exo, n_endo) converged value function.
            policy_action:  (n_exo, n_endo, action_dim) optimal actions in levels.
            policy_endo:    (n_exo, n_endo, endo_dim) optimal next endo state.
            grids:          Grid dict from build_grids.
            prob_matrix:    (n_exo, n_exo) estimated transition matrix.
            converged:      bool.
            n_iter:         number of iterations used.
            history:        dict-of-lists with eval_callback metrics (empty if no callback).
            stop_reason:    'converged', 'max_iter', or 'early_stop'.
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

    n_exo    = tables["n_exo"]
    n_endo   = tables["n_endo"]
    n_action = tables["n_action"]
    action_product = tf.constant(grids["action_product"], dtype=tf.float32)
    endo_product   = tf.constant(grids["endo_product"],   dtype=tf.float32)

    def _extract_policy(v, pidx):
        pa = tf.gather(action_product, tf.reshape(pidx, [-1]))
        pa = tf.reshape(pa, [n_exo, n_endo, env.action_dim()])
        tap = tf.gather(
            tf.reshape(tables["trans_idx"], [n_exo * n_endo, n_action]),
            tf.reshape(pidx, [n_exo * n_endo, 1]),
            batch_dims=1,
        )
        tap = tf.squeeze(tap, axis=-1)
        pe  = tf.gather(endo_product, tap)
        pe  = tf.reshape(pe, [n_exo, n_endo, env.endo_dim()])
        return pa, pe

    def _append_row(hist, step, elapsed, base_scalars, eval_metrics):
        for k, v in [("step", step), ("elapsed_sec", elapsed)]:
            hist.setdefault(k, []).append(v)
        for k, v in list(base_scalars.items()) + list(eval_metrics.items()):
            hist.setdefault(k, []).append(v)

    def _should_stop(hist, monitor, threshold, threshold_patience):
        if monitor is None or threshold is None or monitor not in hist:
            return False
        vals = hist[monitor]
        return len(vals) >= threshold_patience and all(
            v <= threshold for v in vals[-threshold_patience:])

    # 4. PFI iteration
    interp_mode = config.interpolation
    if value_init is not None:
        v_curr = tf.cast(tf.reshape(value_init, [n_exo, n_endo]), tf.float32)
    else:
        v_curr = tf.zeros([n_exo, n_endo], dtype=tf.float32)
    _, policy_idx = _bellman_step(v_curr, tables, interp_mode)

    eval_history = {}
    stop_reason  = None
    converged    = False
    start_time   = time.perf_counter()

    for iteration in range(config.max_iter):
        # Policy evaluation
        v_eval = _evaluate_policy(
            policy_idx, v_curr, tables, config.eval_steps, interp_mode)

        # Policy improvement
        v_greedy, new_policy_idx = _bellman_step(v_eval, tables, interp_mode)

        # Eval checkpoint (before convergence check so we log the last state)
        if eval_callback is not None:
            elapsed = time.perf_counter() - start_time
            policy_action, policy_endo = _extract_policy(v_greedy, new_policy_idx)
            partial = {
                "value":         v_greedy,
                "policy_action": policy_action,
                "policy_endo":   policy_endo,
                "grids":         grids,
            }
            metrics = eval_callback(iteration, partial)
            _append_row(eval_history, iteration, elapsed,
                        {"policy_changed": int(tf.reduce_any(
                            tf.not_equal(new_policy_idx, policy_idx)).numpy())},
                        metrics)
            if _should_stop(eval_history, config.monitor,
                            config.threshold, config.threshold_patience):
                stop_reason = "early_stop"
                policy_idx = new_policy_idx
                v_curr = v_greedy
                print(f"PFI early stop at iteration {iteration}: "
                      f"{config.monitor}={metrics.get(config.monitor):.6g}")
                converged = True
                break

        # Check convergence: policy unchanged
        policy_changed = bool(tf.reduce_any(
            tf.not_equal(new_policy_idx, policy_idx)).numpy())

        policy_idx = new_policy_idx
        v_curr = v_greedy

        if not policy_changed:
            converged = True
            stop_reason = "converged"
            print(f"PFI converged: {iteration + 1} policy updates")
            break

    if not converged and stop_reason is None:
        stop_reason = "max_iter"
        print(f"PFI did not converge after {config.max_iter} iterations")

    # 5. Extract final policies in levels
    policy_action, policy_endo = _extract_policy(v_curr, policy_idx)

    return {
        "value":         v_curr,
        "policy_action": policy_action,
        "policy_endo":   policy_endo,
        "grids":         grids,
        "prob_matrix":   prob_matrix,
        "converged":     converged,
        "n_iter":        iteration + 1,
        "history":       eval_history,
        "stop_reason":   stop_reason,
    }
