"""Equilibrium-channel diagnostics for Block-1 training (DF26 Sec 3.5).

The firm prices its own default risk through the indirect policy gradient

    actions (k', b'-c')  ->  target V(k', b'-c')  ->  P_def  ->  q  ->  D,

so a riskier debt choice raises P_def, lowers the bond price q, and the loss carries
that penalty back into the policy. Two things can switch this channel off WITHOUT
breaking training, so we monitor both:

* The bond price multiplies P_def by the gate ``g = 1{R < b'k'}``. In the safe regime
  (debt fully collateralized, ``R >= b'k'``) ``g = 0``, ``q`` is risk-free, and the
  channel is correctly inactive (no default risk to price). At initialization the
  debt policy is biased low (``raw_bias=-4``), so ``g = 0`` everywhere and the channel
  is dormant by design -- it activates only once the policy levers into risky debt. A
  run whose ``gate_active_frac`` never rises off zero has a *muted* channel: the policy
  never priced its own default risk, which would invalidate the capital-structure
  result even with a small Bellman residual.
* The reverse-over-forward q-gradient through the Taylor P_def only differentiates
  under ``tf.function``; eager mode raises (see :mod:`src.v3.solver.train_step`).

``channel_stats`` logs the cheap per-epoch activity (gate fraction, leverage, default
risk) so a muted run is visible *after* training. ``qchannel_grad_norms`` measures how
much of the policy gradient flows through q, a regression guard against an accidental
``stop_gradient`` on q (the exact silent bug Sec 3.5 warns about).
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET as _F
from src.v3.economics import debt as _debt
from src.v3.economics import dividends as _div
from src.v3.solver import bellman
from src.v3.solver.sampling import make_batch


def channel_stats(bundle, batch, ext, grid_cfg) -> dict:
    """Cheap per-epoch activity of the default-risk pricing channel (one forward pass, no grad).

    Returns:
      * ``gate_active_frac`` -- fraction of states with ``g=1`` (risky debt chosen); ~0
        throughout training signals a muted channel.
      * ``mean_bp`` -- mean gross debt ``b'`` (leverage; starts near 0.04 at init).
      * ``frac_default_risk`` -- fraction of states with ``P_def > 1e-2``.
      * ``pdef_default_regime`` -- mean ``P_def`` among gated (risky) states.
    """
    k = batch.state_raw[:, 1]
    rho, sigma, delta = batch.param_raw[:, 1], batch.param_raw[:, 2], batch.param_raw[:, 3]
    chi = batch.param_raw[:, 6]
    i, bp, cp = bellman.policy_controls(bundle, batch, ext, grid_cfg)
    kprime = (1.0 + i - delta) * k
    g = _debt.gate(kprime, cp, bp, chi, delta)
    _, pdef = bellman.bond_price_training(bundle.target, batch.state_raw[:, 0], kprime,
                                          bp - cp, cp, bp, batch.param_norm, rho, sigma,
                                          batch.box, chi, delta, ext.bond_discount)
    gated = tf.reduce_sum(g)
    return {
        "gate_active_frac": float(tf.reduce_mean(g)),
        "mean_bp": float(tf.reduce_mean(bp)),
        "frac_default_risk": float(tf.reduce_mean(tf.cast(pdef > 1e-2, _F))),
        "pdef_default_regime": float(tf.reduce_sum(g * pdef) / tf.maximum(gated, 1.0)),
    }


def make_eval_fn(bounds, ext, grid_cfg, master_seed, n: int = 4096, seed_epoch: int = 10 ** 9):
    """Build an ``eval_fn(epoch, bundle)`` for :func:`trainer.train_block1` that logs
    :func:`channel_stats` each epoch on a FIXED held-out batch (so the per-epoch series is
    comparable). The ``seed_epoch`` key is far from any training epoch, so the eval batch
    is disjoint from the training draws."""
    scaler = ParamScaler(bounds, dtype=_F)
    batch = make_batch(n, bounds, ext, grid_cfg, scaler, master_seed, seed_epoch, 0)
    return lambda epoch, bundle: channel_stats(bundle, batch, ext, grid_cfg)


def _rhs_mean(bundle, batch, ext, grid_cfg, x_nodes, w_nodes, tau, detach_q: bool):
    """Mean Bellman RHS (the Step-3 objective), optionally with q detached from the controls."""
    logz = batch.state_raw[:, 0]; k = batch.state_raw[:, 1]; b = batch.state_raw[:, 2]
    z = tf.exp(logz)
    theta, rho, sigma, delta = (batch.param_raw[:, 0], batch.param_raw[:, 1],
                                batch.param_raw[:, 2], batch.param_raw[:, 3])
    gamma1, gamma0, chi, cf = (batch.param_raw[:, 4], batch.param_raw[:, 5],
                               batch.param_raw[:, 6], batch.param_raw[:, 7])
    i, bp, cp = bellman.policy_controls(bundle, batch, ext, grid_cfg)
    kprime = (1.0 + i - delta) * k
    bpp = bp - cp
    q, _ = bellman.bond_price_training(bundle.target, logz, kprime, bpp, cp, bp,
                                       batch.param_norm, rho, sigma, batch.box, chi,
                                       delta, ext.bond_discount)
    if detach_q:
        q = tf.stop_gradient(q)
    D = _div.dividend(z, k, b, i, bp, cp, q, theta=theta, alpha=ext.alpha, delta=delta,
                      gamma1=gamma1, gamma0=gamma0, cf=cf, lambda0=ext.lambda0,
                      lambda1=ext.lambda1, iota_c=ext.iota_c, smooth=True, tau=tau).D
    cont = bellman.continuation(bundle.value, logz, kprime, bpp, batch.param_norm,
                                rho, sigma, batch.box, x_nodes, w_nodes)
    return -tf.reduce_mean(D + ext.discount * cont)


def qchannel_grad_norms(bundle, batch, ext, grid_cfg, x_nodes, w_nodes, tau):
    """Measure how much of the ``b'``-policy gradient flows through the bond price q.

    Returns ``(grad_norm, qchannel_norm)``: the norm of the full gradient and the norm of
    the part removed by detaching q (``||grad_full - grad_detached_q||``). In the safe
    regime (gate off) ``qchannel_norm == 0`` (channel inactive); in the risky regime it is
    a large, finite contribution (the pricing penalty that disciplines leverage). A
    regression test asserts it is non-zero in the risky regime, catching an accidental
    detach of q. Builds the gradient under ``tf.function`` (the q-gradient is
    reverse-over-forward and does not differentiate eagerly)."""
    bp_vars = list(bundle.policy_bp.trainable_variables)

    @tf.function
    def grads(detach_q: bool):
        with tf.GradientTape() as tape:
            loss = _rhs_mean(bundle, batch, ext, grid_cfg, x_nodes, w_nodes, tau, detach_q)
        return tape.gradient(loss, bp_vars)

    g_full = grads(False)
    g_det = grads(True)
    grad_norm = float(tf.linalg.global_norm(g_full))
    qchannel_norm = float(tf.linalg.global_norm([a - b for a, b in zip(g_full, g_det)]))
    return grad_norm, qchannel_norm
