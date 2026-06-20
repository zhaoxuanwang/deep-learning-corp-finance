"""Training-time analytic default probability (DF26 Sec 3.5).

Treat the continuation value V'(eps) = V_target(z'(eps), k', b'-c') as a function of
the standard-normal shock eps (with z'(eps) = exp(rho log z + sigma eps)) and take a
2nd-order Taylor expansion around eps = 0:

    V'(eps) ~ V0 + a eps + 0.5 c eps^2,

with a = dV'/deps|0, c = d^2V'/deps^2|0 from automatic differentiation (the target
network). The default region {eps : V'(eps) < 0} is then a union of intervals whose
standard-normal measure has a closed form via Phi. Used ONLY during training; grid
refinement computes P_def exactly over the Tauchen nodes.
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def default_probability(V0, a, c, c_tol=1e-6, a_tol=1e-12):
    """P_def = Pr(0.5 c eps^2 + a eps + V0 < 0), eps ~ N(0,1). Differentiable, in [0,1]."""
    dtype = V0.dtype
    normal = tfd.Normal(tf.zeros([], dtype), tf.ones([], dtype))
    phi = normal.cdf
    zero = tf.zeros_like(V0)
    one = tf.ones_like(V0)

    # Quadratic case (|c| >= c_tol).
    disc = a * a - 2.0 * c * V0
    has_roots = disc > 0.0
    sqrt_disc = tf.sqrt(tf.maximum(disc, zero))
    safe_c = tf.where(tf.abs(c) < c_tol, one, c)
    r_a = (-a - sqrt_disc) / safe_c
    r_b = (-a + sqrt_disc) / safe_c
    lo = tf.minimum(r_a, r_b)
    hi = tf.maximum(r_a, r_b)
    # c > 0: parabola opens up, negative between the roots (empty if no real roots).
    p_cpos = tf.where(has_roots, phi(hi) - phi(lo), zero)
    # c < 0: opens down, negative outside the roots (all of R if no real roots).
    p_cneg = tf.where(has_roots, phi(lo) + (one - phi(hi)), one)
    p_quad = tf.where(c > 0.0, p_cpos, p_cneg)

    # Linear case (|c| < c_tol): V0 + a eps < 0.
    safe_a = tf.where(tf.abs(a) < a_tol, one, a)
    thresh = -V0 / safe_a
    p_lin = tf.where(a > 0.0, phi(thresh),
                     tf.where(a < 0.0, one - phi(thresh), tf.cast(V0 < 0.0, dtype)))

    p = tf.where(tf.abs(c) < c_tol, p_lin, p_quad)
    return tf.clip_by_value(p, 0.0, 1.0)


def taylor_coeffs(target_net, logz, kprime, bpp, param_norm, rho, sigma, normalize_next):
    """Return ``(V0, a, c)`` for the Taylor expansion of V'(eps) at eps = 0.

    ``normalize_next(logz_p, kprime, bpp)`` maps the next-period raw state to the
    network's normalized input. Uses nested gradient tapes (Sec 11).
    """
    eps = tf.zeros_like(logz)
    with tf.GradientTape() as tape2:
        tape2.watch(eps)
        with tf.GradientTape() as tape1:
            tape1.watch(eps)
            logz_p = rho * logz + sigma * eps
            state_norm = normalize_next(logz_p, kprime, bpp)
            vprime = target_net(state_norm, param_norm)          # V'(eps)  [N]
        # gradient() of a vector target sums over its elements; per-sample
        # independence makes the cross-derivatives zero, so this is the
        # per-sample dV'/deps. (Reductions must stay inside the recording tape.)
        a = tape1.gradient(vprime, eps)                          # dV'/deps  [N]
    c = tape2.gradient(a, eps)                                   # d2V'/deps2 [N]
    return vprime, a, c
