"""Comparative statics of the Block-1 policies versus each parameter (DF26 Sec 12.3).

Sweep one parameter across its Table A1 range, holding the other seven at the box
midpoint and the state at the central grid node, and read each policy (i, b', c') from
the NN (Block 1) and from the refined NN. Raw VFI is omitted: a VFI solve per swept
point is infeasible (and unnecessary here). Uses the batched refine over the whole sweep
so all points solve at once. float64.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.config import PARAM_NAMES
from src.v3.solver.batched import build_grids_batch, network_on_grid_batch, refine_batch


def comparative_statics(bundle, param, bounds, ext, grid_cfg, *, n_points=25, refine_rounds=6):
    """Sweep parameter ``param`` (name or index); return NN and refined-NN policies (i, b', c')
    at the central grid node across the sweep."""
    j = PARAM_NAMES.index(param) if isinstance(param, str) else int(param)
    lo = tf.constant(bounds.lower_array(), TF_FLOAT_NUM)
    hi = tf.constant(bounds.upper_array(), TF_FLOAT_NUM)
    ref = 0.5 * (lo + hi)
    sweep = tf.linspace(lo[j], hi[j], n_points)
    beta = tf.tile(ref[None, :], [n_points, 1])
    beta = tf.concat([beta[:, :j], sweep[:, None], beta[:, j + 1:]], axis=1)   # [n_points, 8]

    g = build_grids_batch(beta, ext, grid_cfg)
    _, pi_nn, bp_nn, cp_nn = network_on_grid_batch(bundle, beta, ext, grid_cfg, bounds, g)
    refined = refine_batch(bundle, beta, ext, grid_cfg, bounds, n_rounds=refine_rounds)

    mz, mk, mb = grid_cfg.n_z // 2, grid_cfg.n_k // 2, grid_cfg.n_b // 2
    pick = lambda a: a[:, mz, mk, mb].numpy()
    return {
        "param": PARAM_NAMES[j], "values": sweep.numpy(),
        "i": {"network": pick(pi_nn), "refined": pick(refined.policy_i)},
        "bp": {"network": pick(bp_nn), "refined": pick(refined.policy_bp)},
        "cp": {"network": pick(cp_nn), "refined": pick(refined.policy_cp)},
    }
