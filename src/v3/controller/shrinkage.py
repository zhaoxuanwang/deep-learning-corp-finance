"""Adaptive bound shrinkage (DF26 Sec 6.2).

Narrow the parameter bounds during training to concentrate computation near the
estimates, using the minimum loss functions (Sec 6.1). Three safeguards:

* **Identification guard.** Only shrink parameter j if its loss curve is sharp: the
  90th-percentile loss must exceed the minimum by at least ``id_guard_sd`` across-fold
  standard deviations (SD = median over grid points of the per-fold-recentered SD).
  Flat (weakly identified) parameters, e.g. chi, are never shrunk.
* **Level rule.** For an identified parameter and a tolerance Delta, the new interval is
  the contiguous region around the curve's minimum where L <= L_min + Delta. Computed per
  fold and spanned (lowest left, highest right) across folds.
* **Containment guard.** A candidate region is admissible only if it still contains both
  (1) the ``containment_closest`` of the ``containment_recent`` most-recent sampled
  parameters whose simulated moments are closest to the target in the weighted-GMM norm,
  and (2) every LM estimate from the current estimation step (all restarts x folds).

Delta is chosen via a target volume fraction v (product of identified-interval widths
over the pre-shrink volume), searching for the smallest admissible v in
[volume_min, volume_max]; if none is admissible, the bounds are left unchanged.
Pure NumPy (operates on the :class:`LossProfile` and small parameter arrays).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.v3.config import ParamBounds


class ShrinkResult(NamedTuple):
    bounds: ParamBounds        # new bounds (unchanged if no admissible shrink)
    shrunk: bool
    identified: np.ndarray     # [P] bool: which parameters passed the identification guard
    chosen_v: float            # achieved volume fraction (1.0 if no shrink)
    volume_fraction: float     # realized product-of-widths fraction over identified params


def _identified_mask(profile, id_guard_sd, id_percentile):
    """Per parameter: is the loss curve sharp enough to shrink? (identification guard)."""
    L = profile.L                                            # [P, npts] median curve
    L_min = L.min(axis=1)
    pct = np.percentile(L, id_percentile, axis=1)
    sd = np.median(profile.L_sd, axis=1)                    # median over grid of across-fold SD
    return (pct - L_min) >= id_guard_sd * sd                # [P] bool


def _contiguous_interval(grid, curve, delta):
    """Contiguous [lo, hi] around the curve's minimum where curve <= min + delta."""
    i = int(np.argmin(curve))
    thr = curve[i] + delta
    lo_i = i
    while lo_i > 0 and curve[lo_i - 1] <= thr:
        lo_i -= 1
    hi_i = i
    while hi_i < len(curve) - 1 and curve[hi_i + 1] <= thr:
        hi_i += 1
    return grid[lo_i], grid[hi_i]


def _param_interval(grid_j, folds_j, delta):
    """Span the per-fold contiguous intervals: (min left, max right) across folds."""
    los, his = [], []
    for k in range(folds_j.shape[0]):
        a, b = _contiguous_interval(grid_j, folds_j[k], delta)
        los.append(a); his.append(b)
    return min(los), max(his)


def _volume_fraction(delta, profile, identified, width0):
    """Product of identified-parameter interval widths at ``delta`` over the pre-shrink widths."""
    frac = 1.0
    for j in np.where(identified)[0]:
        a, b = _param_interval(profile.grid[j], profile.L_folds[j], delta)
        frac *= (b - a) / width0[j]
    return frac


def _delta_for_volume(target_v, profile, identified, width0, *, n_bisect=40):
    """Smallest Delta whose identified-volume fraction reaches ``target_v`` (monotone bisection)."""
    # Delta range: 0 -> point intervals (vol ~ 0); max curve span -> full bounds (vol = 1).
    hi = float(max(profile.L_folds[j].max() - profile.L_folds[j].min()
                   for j in np.where(identified)[0]))
    lo = 0.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _volume_fraction(mid, profile, identified, width0) < target_v:
            lo = mid
        else:
            hi = mid
    return hi


def _contains(intervals, identified, vectors):
    """Do all ``vectors`` [N, P] lie inside ``intervals`` on every identified dimension?"""
    if vectors.size == 0:
        return True
    ok = np.ones(vectors.shape[0], dtype=bool)
    for j in np.where(identified)[0]:
        ok &= (vectors[:, j] >= intervals[j][0] - 1e-12) & (vectors[:, j] <= intervals[j][1] + 1e-12)
    return bool(ok.all())


def _gmm_closest(recent_betas, recent_moments, target_m, W, n_closest):
    """The ``n_closest`` recent parameters whose moments are nearest the target in the W-norm."""
    if recent_betas.shape[0] == 0:
        return np.zeros((0, recent_betas.shape[1] if recent_betas.ndim == 2 else 8))
    r = recent_moments - target_m[None, :]
    d = np.einsum("ni,ij,nj->n", r, W, r)
    keep = np.argsort(d)[:min(n_closest, d.shape[0])]
    return recent_betas[keep]


def shrink_bounds(profile, bounds, ctrl, *, recent_betas, recent_moments, target_m, W,
                  lm_estimates) -> ShrinkResult:
    """Attempt one bound shrink (Sec 6.2). Returns the new bounds (unchanged if inadmissible)."""
    lo0 = bounds.lower_array()
    hi0 = bounds.upper_array()
    width0 = hi0 - lo0
    P = lo0.shape[0]
    identified = _identified_mask(profile, ctrl.id_guard_sd, ctrl.id_percentile)
    if not identified.any():
        return ShrinkResult(bounds, False, identified, 1.0, 1.0)

    recent_betas = np.asarray(recent_betas, np.float64).reshape(-1, P)
    recent_moments = np.asarray(recent_moments, np.float64).reshape(-1, target_m.shape[0])
    W = np.asarray(W, np.float64)
    last = slice(max(0, recent_betas.shape[0] - ctrl.containment_recent), None)
    closest = _gmm_closest(recent_betas[last], recent_moments[last],
                           np.asarray(target_m, np.float64), W, ctrl.containment_closest)
    contain = np.concatenate([closest, np.asarray(lm_estimates, np.float64).reshape(-1, P)], axis=0)

    # Smallest admissible volume fraction (most aggressive shrink the guards allow).
    idx_id = np.where(identified)[0]
    for v in np.linspace(ctrl.volume_min, ctrl.volume_max, ctrl.n_volume_steps):
        if v >= ctrl.volume_max:
            break  # the no-shrink end of the range
        delta = _delta_for_volume(v, profile, identified, width0)
        intervals = {}
        for j in range(P):
            if identified[j]:
                a, b = _param_interval(profile.grid[j], profile.L_folds[j], delta)
                intervals[j] = (max(a, lo0[j]), min(b, hi0[j]))
            else:
                intervals[j] = (lo0[j], hi0[j])
        new_lo = np.array([intervals[j][0] for j in range(P)])
        new_hi = np.array([intervals[j][1] for j in range(P)])
        realized = float(np.prod([(new_hi[j] - new_lo[j]) / width0[j] for j in idx_id]))
        # Reject non-shrinks (volume rounds back to full) and degenerate (zero-width) intervals.
        if realized >= 1.0 - 1e-9 or np.any(new_hi[idx_id] <= new_lo[idx_id]):
            continue
        if _contains(intervals, identified, contain):
            new_bounds = ParamBounds(lower=tuple(new_lo), upper=tuple(new_hi))
            return ShrinkResult(new_bounds, True, identified, float(v), realized)

    return ShrinkResult(bounds, False, identified, 1.0, 1.0)
