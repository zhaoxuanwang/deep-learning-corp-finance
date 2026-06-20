"""Analytic default probability vs brute-force quadrature (DF26 Sec 3.5)."""
import numpy as np
import tensorflow as tf
from scipy.stats import norm

from src.v3.solver import default_prob as dp

DT = tf.float64


def _brute(V0, a, c, n=2_000_001):
    eps = np.linspace(-12.0, 12.0, n)
    quad = 0.5 * c * eps ** 2 + a * eps + V0
    return float(np.trapz(norm.pdf(eps) * (quad < 0.0), eps))


# (V0, a, c) cases spanning every branch.
CASES = [
    (1.0, 0.2, 0.5),     # c>0, no roots -> P=0
    (-0.5, 0.3, 0.5),    # c>0, roots    -> interval
    (0.5, 0.4, -0.3),    # c<0, roots    -> two tails
    (1.0, 0.1, -0.05),   # c<0, near all-positive small region
    (0.3, 0.5, 0.0),     # linear, a>0
    (0.3, -0.5, 0.0),    # linear, a<0
    (-0.2, 0.4, 1.0),    # c>0 deep, default interval
    (2.0, -0.1, 0.2),    # c>0, V0 large -> P~0
    (-2.0, 0.1, 0.2),    # c>0, V0 very negative -> large P
]


def test_matches_brute_force():
    for V0, a, c in CASES:
        analytic = float(dp.default_probability(
            tf.constant(V0, DT), tf.constant(a, DT), tf.constant(c, DT)))
        brute = _brute(V0, a, c)
        assert abs(analytic - brute) < 2e-3, f"({V0},{a},{c}): {analytic} vs {brute}"


def test_in_unit_interval_and_batched():
    V0 = tf.constant([c[0] for c in CASES], DT)
    a = tf.constant([c[1] for c in CASES], DT)
    c = tf.constant([c[2] for c in CASES], DT)
    p = dp.default_probability(V0, a, c).numpy()
    assert (p >= 0).all() and (p <= 1).all()
    brute = np.array([_brute(*case) for case in CASES])
    np.testing.assert_allclose(p, brute, atol=2e-3)


def test_constant_region_when_a_and_c_zero():
    # No eps-dependence: P = 1{V0 < 0}.
    assert float(dp.default_probability(tf.constant(-0.5, DT), tf.constant(0.0, DT),
                                        tf.constant(0.0, DT))) == 1.0
    assert float(dp.default_probability(tf.constant(0.5, DT), tf.constant(0.0, DT),
                                        tf.constant(0.0, DT))) == 0.0


def test_differentiable():
    V0 = tf.Variable(0.2, dtype=DT)
    a = tf.Variable(0.3, dtype=DT)
    c = tf.Variable(0.4, dtype=DT)
    with tf.GradientTape() as t:
        p = dp.default_probability(V0, a, c)
    g = t.gradient(p, [V0, a, c])
    assert all(x is not None for x in g)
