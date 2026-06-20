"""VFI benchmark solver (DF26 Sec 12.1): the trusted oracle.

Plain value-function iteration on the discrete grid, from a cold start, with a
GLOBAL search over the full control grid at every state node and the joint
value/bond-price fixed point (q is recomputed from the current V each sweep). It
uses the exact (non-smoothed) dividend and the 11-state Tauchen expectation.

Intentionally exact and slow; it is the ground truth the network/refined solutions
are checked against, and is never used inside the estimation loop. float64, CPU.
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.config import ExternalParams, GridConfig, ModelParams
from src.v3.economics import debt as _debt
from src.v3.economics import dividends as _div
from src.v3.solver import grid as _grid
from src.v3.solver.interp import interp_grid
from src.v3.solver.refine_evaluate import policy_evaluate


class VFIResult(NamedTuple):
    value: tf.Tensor       # [n_z, n_k, n_b] equity value to shareholders, max(V, 0)
    value_raw: tf.Tensor   # [n_z, n_k, n_b] unclamped V (Eq 10); V < 0 marks default for simulation
    policy_i: tf.Tensor    # [n_z, n_k, n_b] investment rate
    policy_kp: tf.Tensor   # [n_z, n_k, n_b] next capital k'
    policy_bp: tf.Tensor   # [n_z, n_k, n_b] gross debt b'
    policy_cp: tf.Tensor   # [n_z, n_k, n_b] cash c'
    grids: _grid.Grids
    n_sweeps: int
    converged: bool
    final_change: float


def _flatten_controls(g: _grid.Grids):
    kpm, bpm, cpm = tf.meshgrid(g.kp, g.bp, g.cp, indexing="ij")
    kp_f = tf.reshape(kpm, [-1])
    bp_f = tf.reshape(bpm, [-1])
    cp_f = tf.reshape(cpm, [-1])
    return kp_f, bp_f, cp_f


def _flatten_states(g: _grid.Grids):
    zm, km, bm = tf.meshgrid(g.z, g.k, g.b, indexing="ij")
    z_f = tf.reshape(zm, [-1])
    k_f = tf.reshape(km, [-1])
    b_f = tf.reshape(bm, [-1])
    n_z, n_k, n_b = g.z.shape[0], g.k.shape[0], g.b.shape[0]
    zi_f = tf.repeat(tf.range(n_z), n_k * n_b)
    return z_f, k_f, b_f, zi_f


def solve_vfi(params: ModelParams, ext: ExternalParams, grid_cfg: GridConfig,
              *, tol: float = 1e-8, max_sweeps: int = 2000, control_chunk: int = 8192,
              v_init=None, mode: str = "value_iteration", eval_tol: float = 1e-10,
              eval_newton_max: int = 50, verbose: bool = False) -> VFIResult:
    """Solve the model on the grid as the trusted oracle.

    ``mode="value_iteration"`` (default) iterates the one-step Bellman operator with
    a global control search (simple, slow; the small-grid cross-check). ``mode="howard"``
    is the spec's benchmark (Sec 12.1): each sweep does the same global policy
    improvement, then exact policy evaluation via the dense linear solve
    (:func:`solver.refine_evaluate.policy_evaluate` with ``method="dense"``), which
    converges in far fewer sweeps and makes finer grids feasible.
    """
    dtype = TF_FLOAT_NUM
    g = _grid.build_grids(params, ext, grid_cfg, dtype)
    n_z, n_k, n_b = g.z.shape[0], g.k.shape[0], g.b.shape[0]
    n_states = n_z * n_k * n_b

    logk = tf.math.log(g.k)
    kp_f, bp_f, cp_f = _flatten_controls(g)
    bpp_f = bp_f - cp_f                      # next-period net debt b' - c'
    n_controls = int(kp_f.shape[0])
    z_f, k_f, b_f, zi_f = _flatten_states(g)

    disc = tf.constant(ext.discount, dtype)
    bond_disc = tf.constant(ext.bond_discount, dtype)
    delta = tf.constant(params.delta, dtype)
    chi = tf.constant(params.chi, dtype)
    pkw = dict(theta=params.theta, alpha=ext.alpha, delta=params.delta,
               gamma1=params.gamma1, gamma0=params.gamma0, cf=params.cf,
               lambda0=ext.lambda0, lambda1=ext.lambda1, iota_c=ext.iota_c)
    neg_inf = tf.constant(float("-inf"), dtype)

    V = tf.zeros([n_z, n_k, n_b], dtype) if v_init is None else tf.cast(v_init, dtype)
    prev_policy = tf.fill([n_states], -1)
    converged = False
    final_change = float("inf")
    sweep = 0

    for sweep in range(1, max_sweeps + 1):
        # Continuation pieces (independent of the current state's k, b).
        vprime = interp_grid(V, logk, g.b, kp_f, bpp_f)        # [n_z', C]
        cont = tf.matmul(g.P, tf.maximum(vprime, 0.0))         # [n_z, C]
        pdef = tf.matmul(g.P, tf.cast(vprime < 0.0, dtype))    # [n_z, C]
        q = _debt.bond_price(pdef, kp_f, cp_f, bp_f, chi, delta, bond_disc)  # [n_z, C]

        best_val = tf.fill([n_states], neg_inf)
        best_idx = tf.zeros([n_states], tf.int32)
        for c0 in range(0, n_controls, control_chunk):
            c1 = min(c0 + control_chunk, n_controls)
            kpc, bpc, cpc = kp_f[c0:c1], bp_f[c0:c1], cp_f[c0:c1]   # [CB]
            q_sc = tf.gather(q[:, c0:c1], zi_f, axis=0)             # [S, CB]
            cont_sc = tf.gather(cont[:, c0:c1], zi_f, axis=0)       # [S, CB]
            i_rate = kpc[None, :] / k_f[:, None] - (1.0 - delta)    # [S, CB]
            D = _div.dividend(z_f[:, None], k_f[:, None], b_f[:, None],
                              i_rate, bpc[None, :], cpc[None, :], q_sc,
                              **pkw, smooth=False).D
            rhs = D + disc * cont_sc                                # [S, CB]
            cbest = tf.reduce_max(rhs, axis=1)
            carg = tf.cast(tf.argmax(rhs, axis=1), tf.int32) + c0
            upd = cbest > best_val
            best_val = tf.where(upd, cbest, best_val)
            best_idx = tf.where(upd, carg, best_idx)

        if mode == "howard":
            # Exact policy evaluation under the improved policy (Sec 12.1 inner step).
            kp_s = tf.gather(kp_f, best_idx)
            bp_s = tf.gather(bp_f, best_idx)
            cp_s = tf.gather(cp_f, best_idx)
            V_new, _ = policy_evaluate(V, z_f, k_f, b_f, zi_f, kp_s, bp_s, cp_s,
                                       g.P, logk, g.b, params, ext,
                                       method="dense", tol=eval_tol, max_iter=eval_newton_max)
        elif mode == "value_iteration":
            V_new = tf.reshape(tf.maximum(best_val, 0.0), [n_z, n_k, n_b])  # limited liability
        else:
            raise ValueError(f"unknown mode {mode!r}")
        final_change = float(tf.reduce_max(tf.abs(V_new - V)))
        policy_stable = bool(tf.reduce_all(best_idx == prev_policy))
        V = V_new
        prev_policy = best_idx
        if verbose and sweep % 50 == 0:
            print(f"  VFI sweep {sweep}: dV={final_change:.3e}")
        if final_change < tol and policy_stable:
            converged = True
            break

    def grid_of(flat):
        return tf.reshape(tf.gather(flat, best_idx), [n_z, n_k, n_b])

    kp_star = tf.gather(kp_f, best_idx)
    i_star = tf.reshape(kp_star / k_f - (1.0 - delta), [n_z, n_k, n_b])
    return VFIResult(
        value=tf.maximum(V, 0.0),  # report equity value to shareholders (Howard V is unclamped)
        value_raw=V,               # unclamped (negative => default; used by the simulation)
        policy_i=i_star,
        policy_kp=grid_of(kp_f),
        policy_bp=grid_of(bp_f),
        policy_cp=grid_of(cp_f),
        grids=g, n_sweeps=sweep, converged=converged, final_change=final_change,
    )
