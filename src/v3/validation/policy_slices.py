"""Policy-slice data for Fig V3 (DF26 Sec 12.3): VFI vs network vs refined.

Extracts on-grid k-slices of V and the three policies at a fixed productivity node
and net-debt level (the headline DF26 Appendix Fig A1 panel set), and reports the
refined-vs-VFI and network-vs-VFI deviations in the good region (V > 0).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

_FIELDS = [("V", "value"), ("i", "policy_i"), ("bp", "policy_bp"), ("cp", "policy_cp")]


def k_slices(vfi_res, net_grid, refined, z_index=None, b_value=-0.03):
    """Return on-grid k-slices of V/i/b'/c' at a fixed (z node, nearest b)."""
    g = vfi_res.grids
    zi = g.z.shape[0] // 2 if z_index is None else z_index
    bi = int(tf.argmin(tf.abs(g.b - b_value)))

    def sl(arr):
        return arr.numpy()[zi, :, bi]

    out = {"k": g.k.numpy(), "z": float(g.z[zi]), "b": float(g.b[bi])}
    for key, attr in _FIELDS:
        out[key] = {
            "vfi": sl(getattr(vfi_res, attr)),
            "network": sl(getattr(net_grid, attr)),
            "refined": sl(getattr(refined, attr)),
        }
    return out


def deviation_report(vfi_res, refined, net_grid):
    """Per-quantity max relative / mean absolute deviation in the good region."""
    good = vfi_res.value.numpy() > 1e-6
    rows = []
    for key, attr in _FIELDS:
        vv = getattr(vfi_res, attr).numpy()[good]
        rr = getattr(refined, attr).numpy()[good]
        nn = getattr(net_grid, attr).numpy()[good]
        rng = max(vv.max() - vv.min(), 1e-12)
        rows.append({
            "quantity": key,
            "refined_max_rel": float(np.max(np.abs(rr - vv)) / rng),
            "network_max_rel": float(np.max(np.abs(nn - vv)) / rng),
            "refined_mean_abs": float(np.mean(np.abs(rr - vv))),
        })
    return rows
