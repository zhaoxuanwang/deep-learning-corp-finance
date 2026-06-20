"""Mini-batch sampling for Block-1 training (DF26 Sec 3.4 Step 1).

Draw parameter vectors uniformly from the current bounds; for each, compute its
parameter-dependent state box (Sec 3.2) and draw (log z, k, b) uniformly within it.
All draws are stateless and keyed by (master, TRAIN, epoch, step) (Sec 10).
"""
from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET
from src.v3.config import ExternalParams, GridConfig, ParamBounds
from src.v3.economics import bounds as _bounds


class Batch(NamedTuple):
    param_raw: tf.Tensor    # [N, 8]
    param_norm: tf.Tensor   # [N, 8] in [-1, 1]
    state_raw: tf.Tensor    # [N, 3] = (log z, k, b)
    state_norm: tf.Tensor   # [N, 3] in [-1, 1]
    box: _bounds.StateBox    # per-sample state bounds


def make_batch(n: int, bounds: ParamBounds, ext: ExternalParams, grid_cfg: GridConfig,
               param_scaler: ParamScaler, master_seed: int, epoch: int, step: int,
               dtype=TF_FLOAT_NET) -> Batch:
    lo = tf.constant(bounds.lower_array(), dtype)
    hi = tf.constant(bounds.upper_array(), dtype)
    u = seeding.uniform([n, 8], master_seed, seeding.Purpose.TRAIN, epoch, step, 0, dtype=dtype)
    param_raw = lo + u * (hi - lo)

    theta, rho, sigma, delta = (param_raw[:, 0], param_raw[:, 1],
                                param_raw[:, 2], param_raw[:, 3])
    box = _bounds.state_box(theta, ext.alpha, delta, ext.rf, rho, sigma,
                            grid_cfg.tauchen_m, grid_cfg.b_lo, grid_cfg.b_hi)
    lo_s = tf.stack([box.logz_lo, box.k_lo, box.b_lo], axis=-1)
    hi_s = tf.stack([box.logz_hi, box.k_hi, box.b_hi], axis=-1)
    us = seeding.uniform([n, 3], master_seed, seeding.Purpose.TRAIN, epoch, step, 1, dtype=dtype)
    state_raw = lo_s + us * (hi_s - lo_s)

    return Batch(
        param_raw=param_raw,
        param_norm=param_scaler.normalize(param_raw),
        state_raw=state_raw,
        state_norm=_bounds.normalize_state(state_raw, box),
        box=box,
    )
