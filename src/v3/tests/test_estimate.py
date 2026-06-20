"""Estimation driver known-answer test (DF26 Sec 5.4-5.6).

Uses a fake surrogate that is an exact known function whose first 8 outputs are the
normalized parameters (so the inverse is unique), to verify estimate -> batched LM ->
reparam -> median-across-folds recovers the true parameter vector.
"""
import numpy as np
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common.normalization import ParamScaler
from src.v3.estimation.estimate import estimate

DT = tf.float64


class _FakeSurrogate:
    F = 4  # folds (fewer for speed); estimate medians across them

    def forward_fold(self, beta_norm, fold):
        b = beta_norm
        return tf.stack([b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5], b[:, 6], b[:, 7],
                         b[:, 0] * b[:, 1], b[:, 2] + b[:, 3], b[:, 4] ** 2], axis=1)


def test_estimate_recovers_truth():
    b = cfg.ParamBounds.table_a1()
    lo, hi = b.lower_array(), b.upper_array()
    beta_star = lo + np.array([0.6, 0.4, 0.5, 0.55, 0.45, 0.5, 0.5, 0.6]) * (hi - lo)
    fake = _FakeSurrogate()
    target = fake.forward_fold(ParamScaler(b, dtype=DT).normalize(tf.constant(beta_star[None], DT)), 0)[0]
    res = estimate(fake, target, tf.eye(11, dtype=DT), b, master_seed=5, n_restarts=20, max_iter=50)
    np.testing.assert_allclose(res["beta_hat"].numpy(), beta_star, rtol=1e-3, atol=1e-4)
    assert res["fold_betas"].shape == (4, 8)
    assert res["all_betas"].shape == (4, 20, 8)
