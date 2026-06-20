"""Grid refinement (DF26 Sec 4.1): bring the network solution to VFI accuracy.

Initialize the value and policy from the trained networks, then run six fixed rounds
of (local policy improvement -> policy evaluation), with the bond price recomputed
exactly from the current V. Uses the exact (non-smoothed) dividend and the Tauchen
expectation. float64, CPU.
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.config import ExternalParams, GridConfig, ModelParams, ParamBounds
from src.v3.solver import grid as _grid
from src.v3.solver.refine_evaluate import policy_evaluate
from src.v3.solver.refine_improve import policy_improve
from src.v3.validation import evaluate as _evaluate


class RefineResult(NamedTuple):
    value: tf.Tensor       # [n_z, n_k, n_b] equity value to shareholders, max(V, 0)
    value_raw: tf.Tensor   # [n_z, n_k, n_b] unclamped V (Eq 10); V < 0 marks default for simulation
    policy_i: tf.Tensor
    policy_kp: tf.Tensor
    policy_bp: tf.Tensor
    policy_cp: tf.Tensor
    grids: _grid.Grids
    n_rounds: int


def _nearest_index(grid_1d, query):
    """Nearest grid index for each query value (grid is small)."""
    diff = tf.abs(query[..., None] - grid_1d)
    return tf.cast(tf.argmin(diff, axis=-1), tf.int32)


def refine(bundle, params: ModelParams, ext: ExternalParams, grid_cfg: GridConfig,
           bounds: ParamBounds, *, n_rounds=6, eval_method="dense",
           eval_tol=1e-10, eval_max_iter=50) -> RefineResult:
    net = _evaluate.network_on_grid(bundle, params, ext, grid_cfg, bounds)
    g = net.grids
    shape = net.value.shape
    n_z, n_k, n_b = shape
    logk = tf.math.log(g.k)
    delta = params.delta

    zm, km, bm = tf.meshgrid(g.z, g.k, g.b, indexing="ij")
    z, k, b = tf.reshape(zm, [-1]), tf.reshape(km, [-1]), tf.reshape(bm, [-1])
    zi = tf.repeat(tf.range(n_z), n_k * n_b)

    # Initialize policy indices from the network policy (nearest grid point).
    kp_net = (1.0 + net.policy_i - delta) * km
    a = tf.reshape(_nearest_index(g.kp, tf.reshape(kp_net, [-1])), shape)
    c = tf.reshape(_nearest_index(g.bp, tf.reshape(net.policy_bp, [-1])), shape)
    d = tf.reshape(_nearest_index(g.cp, tf.reshape(net.policy_cp, [-1])), shape)
    V = tf.maximum(tf.cast(net.value, TF_FLOAT_NUM), 0.0)

    kp = bp = cp = None
    for _ in range(n_rounds):
        a, c, d = policy_improve(V, a, c, d, z, k, b, zi, g, logk, params, ext)
        kp = tf.gather(g.kp, tf.reshape(a, [-1]))
        bp = tf.gather(g.bp, tf.reshape(c, [-1]))
        cp = tf.gather(g.cp, tf.reshape(d, [-1]))
        V, _ = policy_evaluate(V, z, k, b, zi, kp, bp, cp, g.P, logk, g.b, params, ext,
                               method=eval_method, tol=eval_tol, max_iter=eval_max_iter)

    i_ref = tf.reshape(kp / k - (1.0 - delta), shape)
    return RefineResult(
        # Report equity value to shareholders; the dense policy eval keeps V unclamped
        # internally (Eq 10) so the active set can identify the default region.
        value=tf.maximum(V, 0.0), value_raw=V, policy_i=i_ref,
        policy_kp=tf.reshape(kp, shape),
        policy_bp=tf.reshape(bp, shape),
        policy_cp=tf.reshape(cp, shape),
        grids=g, n_rounds=n_rounds,
    )
