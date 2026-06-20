"""Known-answer tests for the economic primitives (DF26 Sec 1)."""
import numpy as np
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.economics import adjustment as adj
from src.v3.economics import bounds as bnd
from src.v3.economics import debt
from src.v3.economics import dividends as div
from src.v3.economics import production as prod

DT = tf.float64


def test_production_constants():
    theta, alpha = 0.7, 0.3
    nu = 1.0 - (1.0 - alpha) * theta
    p = (1.0 - alpha) * theta
    assert np.isclose(float(prod.nu(theta, alpha)), nu)
    assert np.isclose(float(prod.xi(theta, alpha)), alpha * theta / nu)
    assert np.isclose(float(prod.a_pi(theta, alpha)), nu * p ** (p / nu))


def test_operating_surplus_is_gross_minus_cf():
    z, k = tf.constant(1.3, DT), tf.constant(0.6, DT)
    g = prod.gross_output(z, k, 0.7, 0.3)
    s = prod.operating_surplus(z, k, 0.7, 0.3, cf=0.1)
    np.testing.assert_allclose(float(g) - 0.1, float(s), atol=1e-12)


def test_k_steady_state_satisfies_user_cost():
    theta, alpha, delta, rf, z = 0.7, 0.3, 0.1, 0.02, 1.3
    k = float(prod.k_steady_state(z, theta, alpha, delta, rf))
    x = float(prod.xi(theta, alpha))
    ap = float(prod.a_pi(theta, alpha))
    # xi z A_pi k^(xi-1) = delta + rf at the steady state.
    np.testing.assert_allclose(x * z * ap * k ** (x - 1.0), delta + rf, rtol=1e-10)


def test_investment_and_capital_law():
    k, i, delta = 2.0, 0.05, 0.1
    np.testing.assert_allclose(adj.k_next(i, k, delta), (1 + i - delta) * k)
    np.testing.assert_allclose(adj.investment_level(i, k), i * k)


def test_dividend_known_answer_no_investment():
    # z=k=1, b=0, i=0 (I=0, so no fixed/convex cost), bp=0, cp=0.
    # Then d1 = -cf < 0 (equity issuance), d2 = gross output, no d2 issuance.
    e = cfg.ExternalParams()
    kw = dict(theta=0.7, alpha=0.3, delta=0.1, gamma1=0.5, gamma0=0.05, cf=0.1,
              lambda0=e.lambda0, lambda1=e.lambda1, iota_c=e.iota_c)
    one = tf.constant(1.0, DT)
    zero = tf.constant(0.0, DT)
    out = div.dividend(one, one, zero, i_rate=zero, bp=zero, cp=zero, q=tf.constant(0.9, DT), **kw)
    gross = float(prod.gross_output(1.0, 1.0, 0.7, 0.3))
    expected_D = gross - 0.1 - (e.lambda0 + e.lambda1 * 0.1)  # d2 + min(d1,0) - issue1
    np.testing.assert_allclose(float(out.D), expected_D, atol=1e-9)
    np.testing.assert_allclose(float(out.d1), -0.1, atol=1e-12)
    np.testing.assert_allclose(float(out.kprime), 0.9, atol=1e-12)


def test_dividend_smooth_approaches_exact_away_from_kinks():
    # Choose a scenario where d1, d2, I are all clearly away from zero, so the
    # smoothed dividend matches the exact one to high precision.
    e = cfg.ExternalParams()
    kw = dict(theta=0.7, alpha=0.3, delta=0.1, gamma1=0.3, gamma0=0.02, cf=0.05,
              lambda0=e.lambda0, lambda1=e.lambda1, iota_c=e.iota_c)
    args = dict(z=tf.constant(1.2, DT), k=tf.constant(0.8, DT), b=tf.constant(0.1, DT),
                i_rate=tf.constant(0.15, DT), bp=tf.constant(0.5, DT),
                cp=tf.constant(0.05, DT), q=tf.constant(0.9, DT))
    exact = div.dividend(**args, **kw, smooth=False)
    smooth = div.dividend(**args, **kw, smooth=True, tau=1e-3)
    np.testing.assert_allclose(float(smooth.D), float(exact.D), atol=1e-2)


def test_recovery_and_gate():
    kp, cp, bp, chi, delta = (tf.constant(v, DT) for v in (1.0, 0.1, 0.5, 0.3, 0.1))
    R = debt.recovery(kp, cp, chi, delta)
    np.testing.assert_allclose(float(R), 0.1 + 0.3 * 0.9, atol=1e-12)
    assert float(debt.gate(kp, cp, bp, chi, delta)) == 1.0  # R=0.37 < face=0.5
    np.testing.assert_allclose(float(debt.recovery_ratio(kp, cp, bp, chi, delta)),
                               0.37 / 0.5, atol=1e-12)


def test_bond_price_riskfree_when_gate_zero():
    e = cfg.ExternalParams()
    bd = e.bond_discount
    # R = 0.5 + 0.27 = 0.77 >= face = 0.1 -> g = 0 -> risk-free.
    q = debt.bond_price(tf.constant(0.5, DT), tf.constant(1.0, DT), tf.constant(0.5, DT),
                        tf.constant(0.1, DT), tf.constant(0.3, DT), tf.constant(0.1, DT), bd)
    np.testing.assert_allclose(float(q), bd, atol=1e-12)


def test_bond_price_bounds_and_extreme():
    e = cfg.ExternalParams()
    bd = e.bond_discount
    # cp=0, chi=0 -> R=0, face=1 -> g=1, ratio=0 -> q = bd (1 - pdef).
    for pdef in (0.0, 0.5, 1.0):
        q = float(debt.bond_price(tf.constant(pdef, DT), tf.constant(1.0, DT),
                                  tf.constant(0.0, DT), tf.constant(1.0, DT),
                                  tf.constant(0.0, DT), tf.constant(0.1, DT), bd))
        assert -1e-12 <= q <= bd + 1e-12
    q1 = float(debt.bond_price(tf.constant(1.0, DT), tf.constant(1.0, DT),
                               tf.constant(0.0, DT), tf.constant(1.0, DT),
                               tf.constant(0.0, DT), tf.constant(0.1, DT), bd))
    np.testing.assert_allclose(q1, 0.0, atol=1e-12)


def test_state_box_param_dependent():
    e = cfg.ExternalParams()
    box = bnd.state_box(theta=tf.constant(0.7, DT), alpha=tf.constant(e.alpha, DT),
                        delta=tf.constant(0.1, DT), rf=tf.constant(e.rf, DT),
                        rho=tf.constant(0.6, DT), sigma=tf.constant(0.2, DT), m=2.5)
    assert float(box.logz_hi) > 0 and float(box.logz_lo) < 0
    np.testing.assert_allclose(float(box.logz_hi), -float(box.logz_lo), atol=1e-12)
    assert float(box.k_lo) < float(box.k_hi)
    # higher sigma widens the log z box.
    box2 = bnd.state_box(theta=tf.constant(0.7, DT), alpha=tf.constant(e.alpha, DT),
                         delta=tf.constant(0.1, DT), rf=tf.constant(e.rf, DT),
                         rho=tf.constant(0.6, DT), sigma=tf.constant(0.3, DT), m=2.5)
    assert float(box2.logz_hi) > float(box.logz_hi)
