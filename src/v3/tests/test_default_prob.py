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


@tf.function(reduce_retracing=True)
def _outer_grad_through_taylor(net, logz, kprime, bpp, pn, rho, sigma, lo, hi):
    """The training-step-3 structure: outer reverse gradient (wrt a policy-fed input,
    kprime) through the Taylor default probability. Old nested-reverse taylor_coeffs hit
    an empty-gradient AddN here on some batches; forward-mode is reverse-over-forward."""
    def nrm(lz, kp, bb):
        from src.v3.common.normalization import to_unit
        return to_unit(tf.stack([lz, kp, bb], axis=-1), lo, hi)
    with tf.GradientTape() as tape:
        tape.watch(kprime)
        v0, a, c = dp.taylor_coeffs(net, logz, kprime, bpp, pn, rho, sigma, nrm)
        loss = tf.reduce_sum(dp.default_probability(v0, a, c))
    return loss, tape.gradient(loss, kprime)


def test_taylor_outer_gradient_robust_on_adversarial_batches():
    """The reverse-over-forward gradient through taylor_coeffs stays finite across corner /
    extreme / boundary batches under tf.function (data-independence guard for the GPU fix)."""
    from src.v3 import config as cfg
    from src.v3.common.normalization import ParamScaler
    from src.v3.networks.film import FiLMNetwork

    f32 = tf.float32
    net = FiLMNetwork(3, 8, cfg.NetworkConfig(), master_seed=2, net_id=0, dtype=f32)
    bounds = cfg.ParamBounds.table_a1()
    scaler = ParamScaler(bounds, dtype=f32)
    plo = tf.constant(bounds.lower_array(), f32)
    phi = tf.constant(bounds.upper_array(), f32)
    lo = tf.constant([-1.0, 0.2, -1.0], f32)
    hi = tf.constant([1.0, 6.0, 2.0], f32)
    gen = tf.random.Generator.from_seed(11)
    N = 256

    for trial in range(8):
        u = gen.uniform([N, 8], dtype=f32)
        if trial < 4:
            u = tf.round(u)                              # parameter-box corners
        params = plo + u * (phi - plo)
        pn = scaler.normalize(params)
        rho, sigma = params[:, 1], params[:, 2]
        logz = gen.normal([N], dtype=f32) * 2.0          # extreme productivity
        kprime = tf.exp(gen.normal([N], dtype=f32)) * 0.5 + 0.21   # tiny .. large capital
        bpp = gen.uniform([N], -1.0, 2.0, dtype=f32)     # full net-debt range incl boundaries
        loss, grad = _outer_grad_through_taylor(net, logz, kprime, bpp, pn, rho, sigma, lo, hi)
        assert bool(tf.math.is_finite(loss)), f"trial {trial}: non-finite loss"
        assert bool(tf.reduce_all(tf.math.is_finite(grad))), f"trial {trial}: non-finite gradient"
