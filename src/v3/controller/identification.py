"""Global identification diagnostic (DF26 Sec 6.3, Eq 45).

To separate weak identification from misspecification, recompute the minimum loss
functions using the SIMULATED moments at the estimated parameters as the target rather
than the data moments:

    L(beta^j) = min_{beta^{-j}} (tilde_m - g(beta^j, beta^{-j}))' W (tilde_m - g(...)),   (45)

where tilde_m are the model-implied moments at beta-hat (Sec 4 simulation). The diagnostic
is twofold: (a) does each L(beta^j) have a sharp unique minimum (identified) versus a flat
curve (weakly identified); and (b) does this second estimation recover the parameter vector
that generated tilde_m. float64.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf

from src.v3.controller.min_loss import LossProfile, minimum_loss_profile
from src.v3.controller.shrinkage import _identified_mask
from src.v3.estimation.estimate import estimate


class IdentificationResult(NamedTuple):
    profile: LossProfile       # Eq-45 minimum loss curves (target = simulated moments)
    identified: np.ndarray     # [P] bool: sharp unique minimum (passes the 3-SD guard)
    beta_recovered: tf.Tensor  # [P] self-recovered estimate against tilde_m
    recovery_abs_err: np.ndarray  # [P] |beta_recovered - beta_generating| (if provided)


def identification_diagnostic(surrogate, tilde_m, W, bounds, master_seed, ctrl,
                              *, beta_generating=None) -> IdentificationResult:
    """Run the Eq-45 diagnostic: sharpness of each curve + self-recovery of beta-hat."""
    profile = minimum_loss_profile(surrogate, tilde_m, W, bounds, master_seed, ctrl,
                                   target_label="simulated")
    identified = _identified_mask(profile, ctrl.id_guard_sd, ctrl.id_percentile)
    est = estimate(surrogate, tilde_m, W, bounds, master_seed, n_restarts=ctrl.n_restarts)
    beta_rec = est["beta_hat"]
    err = (np.abs(beta_rec.numpy() - np.asarray(beta_generating, np.float64))
           if beta_generating is not None else np.full(beta_rec.shape, np.nan))
    return IdentificationResult(profile, identified, beta_rec, err)
