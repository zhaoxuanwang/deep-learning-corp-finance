"""Min-max normalization round-trip and ParamScaler (DF26 Eq 24, Sec 3.2)."""
import numpy as np
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common import normalization as N


def test_to_from_unit_roundtrip():
    x = tf.constant([0.1, 0.5, 0.9], tf.float64)
    lo = tf.constant(0.0, tf.float64)
    hi = tf.constant(1.0, tf.float64)
    u = N.to_unit(x, lo, hi)
    np.testing.assert_allclose(u.numpy(), [-0.8, 0.0, 0.8], atol=1e-12)
    np.testing.assert_allclose(N.from_unit(u, lo, hi).numpy(), x.numpy(), atol=1e-12)


def test_param_scaler_maps_bounds_to_pm1():
    b = cfg.ParamBounds.table_a1()
    sc = N.ParamScaler(b)
    lo, hi = b.lower_array(), b.upper_array()
    np.testing.assert_allclose(sc.normalize(lo).numpy(), -np.ones(8), atol=1e-12)
    np.testing.assert_allclose(sc.normalize(hi).numpy(), np.ones(8), atol=1e-12)
    mid = 0.5 * (lo + hi)
    np.testing.assert_allclose(sc.normalize(mid).numpy(), np.zeros(8), atol=1e-12)


def test_param_scaler_roundtrip_batched():
    b = cfg.ParamBounds.table_a1()
    sc = N.ParamScaler(b)
    raw = np.stack([b.lower_array(), b.upper_array(),
                    0.5 * (b.lower_array() + b.upper_array())])
    back = sc.denormalize(sc.normalize(raw))
    np.testing.assert_allclose(back.numpy(), raw, atol=1e-12)
