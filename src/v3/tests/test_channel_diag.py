"""Equilibrium-channel diagnostics and gradient-flow guards (DF26 Sec 3.5).

These lock in the default-risk pricing channel: actions -> target V -> P_def -> q -> D.
They guard against (a) the channel being silently detached (an accidental stop_gradient
on q, which finiteness tests miss), (b) a muted-channel solution where the policy never
levers into risky debt, and (c) running the policy step eagerly (the q-gradient is
reverse-over-forward and needs tf.function).
"""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET as f32
from src.v3.common.precision import make_adam
from src.v3.common.quadrature import gauss_hermite
from src.v3.networks.bundle import NetworkBundle
from src.v3.solver import channel_diag as cd
from src.v3.solver.sampling import make_batch
from src.v3.solver.train_step import make_step_fn
from src.v3.solver.trainer import train_block1

SEED = 20260619
EXT = cfg.ExternalParams()
GRID = cfg.GridConfig()
BOUNDS = cfg.ParamBounds.table_a1()
SCALER = ParamScaler(BOUNDS, dtype=f32)
TRAIN = cfg.TrainConfig()
XN, WN = gauss_hermite(TRAIN.gh_nodes, dtype=f32)
TAU = tf.constant(TRAIN.smooth_tau, f32)


def _batch(n=512):
    return make_batch(n, BOUNDS, EXT, GRID, SCALER, SEED, 7, 0)


def test_channel_stats_keys_and_ranges():
    s = cd.channel_stats(NetworkBundle(cfg.NetworkConfig(), SEED), _batch(), EXT, GRID)
    assert set(s) >= {"gate_active_frac", "mean_bp", "frac_default_risk", "pdef_default_regime"}
    assert 0.0 <= s["gate_active_frac"] <= 1.0
    assert 0.0 <= s["frac_default_risk"] <= 1.0
    assert s["mean_bp"] >= 0.0


def test_qchannel_inactive_in_safe_regime():
    # Low-debt init: gate off, so detaching q changes nothing (channel correctly dormant).
    safe = NetworkBundle(cfg.NetworkConfig(), SEED, bprime_init_bias=-4.0)
    batch = _batch()
    assert cd.channel_stats(safe, batch, EXT, GRID)["gate_active_frac"] < 0.05
    _, qnorm = cd.qchannel_grad_norms(safe, batch, EXT, GRID, XN, WN, TAU)
    assert qnorm < 1e-7


def test_qchannel_active_and_informative_in_risky_regime():
    # High-debt bias: gate on, so the pricing channel MUST contribute a finite, non-zero
    # part of the policy gradient. This is the guard against an accidental detach of q.
    risky = NetworkBundle(cfg.NetworkConfig(), SEED, bprime_init_bias=4.0)
    batch = _batch()
    assert cd.channel_stats(risky, batch, EXT, GRID)["gate_active_frac"] > 0.1
    gnorm, qnorm = cd.qchannel_grad_norms(risky, batch, EXT, GRID, XN, WN, TAU)
    assert np.isfinite(gnorm) and np.isfinite(qnorm)
    assert qnorm > 0.0


def test_channel_activates_during_training_not_muted():
    # The muted-fixed-point check: starting from the low-debt init, a short run must lever
    # the firm into risky debt (mean_bp rises, gate turns on), so the channel is not
    # permanently muted. Also confirms the monitor is recorded in the persisted history.
    bundle = NetworkBundle(cfg.NetworkConfig(), SEED)
    ev = cd.make_eval_fn(BOUNDS, EXT, GRID, SEED, n=512)
    init = ev(0, bundle)
    hist = train_block1(bundle, BOUNDS, EXT, GRID, TRAIN, SEED, n_epochs=8,
                        steps_per_epoch=25, batch_size=512, eval_fn=ev, compile_step=True)
    assert "gate_active_frac" in hist[-1]              # monitor recorded each epoch
    assert hist[-1]["mean_bp"] > 1.5 * init["mean_bp"]  # firm levers up
    assert hist[-1]["gate_active_frac"] > 0.02          # channel switches on


def test_policy_step_runs_compiled():
    # Production path: the combined value+policy step differentiates under tf.function.
    bundle = NetworkBundle(cfg.NetworkConfig(), SEED)
    step = make_step_fn(bundle, EXT, GRID, XN, WN, TAU, make_adam(1e-3), make_adam(1e-3),
                        compile_step=True)
    lv, lp = step(_batch())
    assert bool(tf.math.is_finite(lv)) and bool(tf.math.is_finite(lp))


def test_policy_step_eager_unsupported():
    # Documents the constraint: Step 3's reverse-over-forward q-gradient does NOT
    # differentiate in eager mode. If a future TF lifts this, drop the eager guard/doc.
    bundle = NetworkBundle(cfg.NetworkConfig(), SEED)
    step = make_step_fn(bundle, EXT, GRID, XN, WN, TAU, make_adam(1e-3), make_adam(1e-3),
                        compile_step=False)
    with pytest.raises(Exception):
        step(_batch())
