"""Moment-surrogate ensemble tests (DF26 Sec 5.1-5.3; review NN-1 no leakage)."""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common.normalization import ParamScaler
from src.v3.estimation.surrogate import SurrogateEnsemble

DT = tf.float64


def _dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    b = cfg.ParamBounds.table_a1()
    lo, hi = b.lower_array(), b.upper_array()
    beta = lo + rng.uniform(size=(n, 8)) * (hi - lo)
    bt = 2.0 * (beta - lo) / (hi - lo) - 1.0           # normalized
    A = rng.normal(size=(11, 8)) * 0.5
    m = bt @ A.T + 0.3 * (bt[:, 0] * bt[:, 1])[:, None]  # smooth, learnable
    return beta.astype(np.float64), m.astype(np.float64), b


def test_forward_fold_matches_forward_all():
    beta, m, b = _dataset(40)
    s = SurrogateEnsemble(master_seed=1)
    s.bounds = b
    bn = ParamScaler(b, dtype=DT).normalize(tf.constant(beta, DT))
    allp = s.forward_all(bn).numpy()
    for f in range(s.F):
        np.testing.assert_allclose(s.forward_fold(bn, f).numpy(), allp[:, :, f], atol=1e-10)


@pytest.mark.slow
def test_fits_known_function_oos():
    beta, m, b = _dataset(1500)
    s = SurrogateEnsemble(master_seed=3)
    s.train(tf.constant(beta, DT), tf.constant(m, DT), b, master_seed=3, passes=200, batch=256)
    r2 = s.oos_r2(tf.constant(beta, DT), tf.constant(m, DT)).numpy()
    assert (r2 > 0.9).all(), f"OOS R^2 too low: {r2}"
