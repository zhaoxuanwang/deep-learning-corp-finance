"""Moment construction tests (DF26 Sec 4.3): numpy cross-check + hand-computed cases."""
import numpy as np
import tensorflow as tf

from src.v3.simulation import moments as M

DT = tf.float64


def _panel(z, k, bnet, c, V, A_pi, xi, cf, delta, bi, T):
    return dict(z=tf.constant(z, DT), k=tf.constant(k, DT), b_net=tf.constant(bnet, DT),
                c=tf.constant(c, DT), V=tf.constant(V, DT),
                burn_in=bi, T=T, A_pi=A_pi, xi=xi, cf=cf, delta=delta)


def _np_moments(z, k, bnet, c, V, A_pi, xi, cf, delta, bi, T):
    cur, nxt = slice(bi, T - 1), slice(bi + 1, T)
    K, C, Bn = k[:, cur], c[:, cur] * k[:, cur], bnet[:, cur] * k[:, cur]
    inc = (z[:, cur] * A_pi * K ** xi - cf) / (K + C)
    net = Bn / (K + C)
    Kp, Cp = k[:, nxt], c[:, nxt] * k[:, nxt]
    Bgp = (bnet[:, nxt] + c[:, nxt]) * k[:, nxt]
    incp = (z[:, nxt] * A_pi * Kp ** xi - cf) / (Kp + Cp)
    inv = (Kp - (1.0 - delta) * K) / K
    d, cash, dc = Bgp / (Kp + Cp), Cp / (Kp + Cp), (Cp - C) / (Kp + Cp)
    good = ((V[:, cur] > 0) & (V[:, nxt] > 0)).ravel()
    f = lambda a: a.ravel()[good]
    inv, inc, incp, d, cash, dc, net = map(f, [inv, inc, incp, d, cash, dc, net])
    slope = lambda x, y: ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    X = np.stack([net - net.mean(), inc - inc.mean()], axis=1)
    bb = np.linalg.lstsq(X, dc - dc.mean(), rcond=None)[0]
    return np.array([inv.mean(), inv.std(), inc.mean(), inc.std(), slope(inc, incp),
                     d.mean(), d.std(), cash.mean(), cash.std(), bb[0], bb[1]])


def test_matches_numpy_reference():
    rng = np.random.default_rng(0)
    nf, T, bi = 20, 30, 10
    z = rng.uniform(0.5, 1.5, (nf, T)); k = rng.uniform(0.5, 3.0, (nf, T))
    bnet = rng.uniform(-0.5, 1.5, (nf, T)); c = rng.uniform(0.0, 0.5, (nf, T))
    V = rng.uniform(-0.2, 2.0, (nf, T))
    A_pi, xi, cf, delta = 0.3, 0.45, 0.05, 0.1
    m = M.compute_moments(_panel(z, k, bnet, c, V, A_pi, xi, cf, delta, bi, T)).numpy()
    ref = _np_moments(z, k, bnet, c, V, A_pi, xi, cf, delta, bi, T)
    np.testing.assert_allclose(m, ref, atol=1e-8, rtol=1e-6)


def test_investment_rate_hand_computed():
    # k grows 1.2x per period with delta=0.1 -> investment rate is a constant 0.3.
    k = np.array([[1.0, 1.2, 1.44, 1.728]])
    z, bnet, c, V = np.ones((1, 4)), np.zeros((1, 4)), np.zeros((1, 4)), np.ones((1, 4))
    m = M.compute_moments(_panel(z, k, bnet, c, V, 1.0, 0.5, 0.0, 0.1, 0, 4)).numpy()
    np.testing.assert_allclose(m[0], 0.3, atol=1e-9)  # mean investment rate
    np.testing.assert_allclose(m[1], 0.0, atol=1e-9)  # sd investment rate


def test_good_obs_filter_excludes_defaults():
    k = np.array([[1.0, 1.2, 1.44, 1.728]])
    z, bnet, c = np.ones((1, 4)), np.zeros((1, 4)), np.zeros((1, 4))
    V = np.array([[1.0, -1.0, 1.0, 1.0]])  # default at t=1 -> only the (2,3) pair is good
    m = M.compute_moments(_panel(z, k, bnet, c, V, 1.0, 0.5, 0.0, 0.1, 0, 4)).numpy()
    np.testing.assert_allclose(m[0], 0.3, atol=1e-9)
    np.testing.assert_allclose(m[1], 0.0, atol=1e-9)


def test_autocorr_slope_known():
    # inc' = 0.5 * inc exactly -> OLS slope of inc' on inc is 0.5.
    rng = np.random.default_rng(1)
    nf, T = 30, 3
    k = np.ones((nf, T))
    # choose z so that inc varies; with k=1, c=0, A_pi=1, xi=1, cf=0: inc = z.
    z = np.zeros((nf, T))
    z[:, 0] = rng.uniform(0.5, 1.5, nf)
    z[:, 1] = 0.5 * z[:, 0]          # inc' (period 1) = 0.5 * inc (period 0)
    z[:, 2] = 0.25 * z[:, 0]
    c, bnet, V = np.zeros((nf, T)), np.zeros((nf, T)), np.ones((nf, T))
    m = M.compute_moments(_panel(z, k, bnet, c, V, 1.0, 1.0, 0.0, 0.1, 0, T)).numpy()
    np.testing.assert_allclose(m[4], 0.5, atol=1e-6)
