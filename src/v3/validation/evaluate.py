"""Evaluate the Block-1 networks on the discrete state grid (for VFI comparison).

Produces value and policy arrays of the same shape as the VFI benchmark
(``[n_z, n_k, n_b]``) so the network and VFI solutions can be compared directly
(Sec 12.3). Networks run in float32; outputs are returned as float64 for clean
comparison with the float64 VFI grids.
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET, TF_FLOAT_NUM
from src.v3.config import ExternalParams, GridConfig, ModelParams, ParamBounds
from src.v3.economics import bounds as _bounds
from src.v3.solver import grid as _grid


class NetworkOnGrid(NamedTuple):
    value: tf.Tensor      # [n_z, n_k, n_b]
    policy_i: tf.Tensor
    policy_bp: tf.Tensor
    policy_cp: tf.Tensor
    grids: _grid.Grids


def _grid_states(g: _grid.Grids):
    zm, km, bm = tf.meshgrid(g.logz, g.k, g.b, indexing="ij")
    return tf.stack([tf.reshape(zm, [-1]), tf.reshape(km, [-1]), tf.reshape(bm, [-1])], axis=-1)


def network_on_grid(bundle, params: ModelParams, ext: ExternalParams,
                    grid_cfg: GridConfig, bounds: ParamBounds) -> NetworkOnGrid:
    """Evaluate the bundle's value and three policies on the VFI state grid."""
    g = _grid.build_grids(params, ext, grid_cfg, dtype=TF_FLOAT_NUM)
    shape = [g.logz.shape[0], g.k.shape[0], g.b.shape[0]]

    state_raw = tf.cast(_grid_states(g), TF_FLOAT_NET)               # [S, 3]
    box = _bounds.state_box(params.theta, ext.alpha, params.delta, ext.rf,
                            params.rho, params.sigma, grid_cfg.tauchen_m,
                            grid_cfg.b_lo, grid_cfg.b_hi)
    state_norm = _bounds.normalize_state(state_raw, box)
    scaler = ParamScaler(bounds, dtype=TF_FLOAT_NET)
    param_norm = tf.tile(scaler.normalize(params.to_array())[None, :], [state_raw.shape[0], 1])

    value = bundle.value(state_norm, param_norm)

    k = state_raw[:, 1]
    i_lo, i_hi = _bounds.investment_rate_bounds(k, tf.cast(box.k_lo, TF_FLOAT_NET),
                                                tf.cast(box.k_hi, TF_FLOAT_NET),
                                                tf.cast(params.delta, TF_FLOAT_NET))
    pi = bundle.policy_i(state_norm, param_norm, i_lo, i_hi)
    pbp = bundle.policy_bp(state_norm, param_norm, grid_cfg.bp_lo, grid_cfg.bp_hi)
    pcp = bundle.policy_cp(state_norm, param_norm, grid_cfg.cp_lo, grid_cfg.cp_hi)

    def to_grid(x):
        return tf.reshape(tf.cast(x, TF_FLOAT_NUM), shape)

    return NetworkOnGrid(to_grid(value), to_grid(pi), to_grid(pbp), to_grid(pcp), g)
