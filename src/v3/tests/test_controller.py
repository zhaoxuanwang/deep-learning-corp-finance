"""Adaptive controller tests (DF26 Sec 6): minimum loss, shrinkage guards, identification.

The shrinkage math is tested deterministically on a hand-built LossProfile (sharp vs flat
curves), and the minimum-loss / identification machinery on a surrogate trained on a known
linear map where some parameters are identified and one (a stand-in for chi) is not.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common import precision
from src.v3.common.normalization import ParamScaler
from src.v3.controller import shrinkage as SH
from src.v3.controller.dataset import DatasetBuffer
from src.v3.controller.identification import identification_diagnostic
from src.v3.controller.min_loss import LossProfile, minimum_loss_profile
from src.v3.estimation.surrogate import SurrogateEnsemble

precision.configure_devices("cpu")

BOUNDS = cfg.ParamBounds.table_a1()
P = 8
WEAK = 6  # chi, the deliberately unidentified parameter


def test_dataset_buffer_caps_to_most_recent():
    buf = DatasetBuffer(max_obs=10)
    for s in range(4):
        buf.add(tf.ones([4, 8], tf.float64) * s, tf.ones([4, 11], tf.float64) * s)
    assert len(buf) == 10
    # The tail is the most-recent batch (value 3).
    assert float(buf.betas[-1, 0]) == 3.0
    rb, _ = buf.recent(3)
    assert rb.shape[0] == 3


def _synthetic_profile(sharp_params, n_points=21, n_folds=6, seed=0):
    """LossProfile (in parameter units) with a sharp V-curve on ``sharp_params`` and
    noise-dominated (weakly identified) curves elsewhere."""
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS.lower_array(), BOUNDS.upper_array()
    grid = np.stack([np.linspace(lo[j], hi[j], n_points) for j in range(P)])   # [P, npts]
    idx = np.arange(n_points)
    c = n_points // 2
    L_folds = np.empty([P, n_folds, n_points])
    for j in range(P):
        for k in range(n_folds):
            if j in sharp_params:    # sharp min at the centre, folds agree (tiny noise)
                L_folds[j, k] = 1.0 + 0.5 * (idx - c) ** 2 + 1e-3 * rng.standard_normal(n_points)
            else:                    # noise-dominated: folds disagree at the across-point scale
                L_folds[j, k] = 1.0 + 0.3 * rng.standard_normal(n_points)
    L = np.median(L_folds, axis=1)
    centered = L_folds - L_folds.min(axis=2, keepdims=True)
    return LossProfile(grid=grid, L=L, L_sd=centered.std(axis=1), L_folds=L_folds,
                       argmin=grid[np.arange(P), L.argmin(axis=1)])


def test_identification_guard_skips_flat_params():
    prof = _synthetic_profile(sharp_params={0, 3})
    ident = SH._identified_mask(prof, id_guard_sd=3.0, id_percentile=90.0)
    assert ident[0] and ident[3]
    assert not ident[1] and not ident[WEAK]


def test_shrinkage_level_rule_and_volume():
    prof = _synthetic_profile(sharp_params=set(range(P)) - {WEAK})
    ctrl = cfg.ControllerConfig(n_loss_points=21, warmup_epochs=0, volume_min=0.05)
    # Containment vectors near the centre so an aggressive shrink is admissible.
    recent_b = 0.5 * (BOUNDS.lower_array() + BOUNDS.upper_array())[None, :] + np.zeros([20, P])
    recent_m = np.zeros([20, 11])
    res = SH.shrink_bounds(prof, BOUNDS, ctrl, recent_betas=recent_b, recent_moments=recent_m,
                           target_m=np.zeros(11), W=np.eye(11), lm_estimates=recent_b[:1])
    assert res.shrunk and res.volume_fraction < 1.0
    new = res.bounds
    # The weak parameter keeps its full range; an identified one narrows.
    assert new.lower[WEAK] == BOUNDS.lower[WEAK] and new.upper[WEAK] == BOUNDS.upper[WEAK]
    assert (new.upper[0] - new.lower[0]) < (BOUNDS.upper[0] - BOUNDS.lower[0])


def test_shrinkage_containment_blocks_when_estimate_outside():
    prof = _synthetic_profile(sharp_params=set(range(P)) - {WEAK})
    ctrl = cfg.ControllerConfig(n_loss_points=21, warmup_epochs=0, volume_min=0.05, volume_max=1.0)
    # An LM estimate pinned at the upper corner: any real shrink would exclude it -> no shrink.
    corner = BOUNDS.upper_array()[None, :]
    res = SH.shrink_bounds(prof, BOUNDS, ctrl, recent_betas=corner, recent_moments=np.zeros([1, 11]),
                           target_m=np.zeros(11), W=np.eye(11), lm_estimates=corner)
    assert not res.shrunk


# --- surrogate-based minimum-loss / identification (known linear map) ---------

@pytest.fixture(scope="module")
def linear_surrogate():
    """Surrogate trained on m = A . normalize(beta), with column WEAK zeroed (unidentified)."""
    dtype = tf.float64
    lo = tf.constant(BOUNDS.lower_array(), dtype)
    hi = tf.constant(BOUNDS.upper_array(), dtype)
    scaler = ParamScaler(BOUNDS, dtype=dtype)
    A = np.zeros([11, P])
    for j in range(P):
        if j != WEAK:
            A[j, j] = 1.0
    A = tf.constant(A, dtype)
    u = tf.random.stateless_uniform([4000, P], seed=[1, 2], dtype=dtype)
    betas = lo + u * (hi - lo)
    m = tf.einsum("mi,bi->bm", A, scaler.normalize(betas))
    surr = SurrogateEnsemble(7, n_folds=5)
    surr.train(betas, m, BOUNDS, 7, passes=150)
    beta_star = cfg.REFERENCE_ESTIMATES.to_array()
    target = tf.einsum("mi,i->m", A, scaler.normalize(beta_star[None])[0])
    return surr, target, beta_star


def test_min_loss_recovers_truth(linear_surrogate):
    surr, target, beta_star = linear_surrogate
    ctrl = cfg.ControllerConfig(n_loss_points=15, n_restarts=8, n_folds=5, warmup_epochs=0)
    prof = minimum_loss_profile(surr, target, tf.eye(11, dtype=tf.float64), BOUNDS, 11, ctrl)
    span = BOUNDS.upper_array() - BOUNDS.lower_array()
    for j in range(P):
        if j != WEAK:    # identified params: curve minimized near the true value
            assert abs(prof.argmin[j] - beta_star[j]) < 0.2 * span[j]
    ident = SH._identified_mask(prof, ctrl.id_guard_sd, ctrl.id_percentile)
    assert ident.sum() >= 6 and not ident[WEAK]   # weak param flagged unidentified


@pytest.mark.slow
def test_controller_loop_runs():
    # End-to-end serial controller (train + collect + estimate + shrink) at SMOKE scale.
    from src.v3.controller.loop import run_controller
    from src.v3.networks.bundle import NetworkBundle
    from src.v3.profiles import get_profile

    prof = get_profile("SMOKE")
    bundle = NetworkBundle(cfg.NetworkConfig(), master_seed=7)
    out = run_controller(bundle, BOUNDS, cfg.ExternalParams(), prof, master_seed=7,
                         target_m=tf.constant(cfg.REFERENCE_TARGETS),
                         W=tf.eye(11, dtype=tf.float64), n_epochs=2)
    assert out["beta_hat"] is not None
    assert bool(tf.reduce_all(tf.math.is_finite(out["beta_hat"])))
    assert len(out["history"]["bounds"]) == 2


@pytest.mark.slow
def test_identification_diagnostic_self_recovers(linear_surrogate):
    surr, target, beta_star = linear_surrogate
    ctrl = cfg.ControllerConfig(n_loss_points=11, n_restarts=8, n_folds=5, warmup_epochs=0)
    res = identification_diagnostic(surr, target, tf.eye(11, dtype=tf.float64), BOUNDS, 11, ctrl,
                                    beta_generating=beta_star)
    span = BOUNDS.upper_array() - BOUNDS.lower_array()
    # Self-recovery: identified params recovered; chi (WEAK) allowed to miss.
    good = [j for j in range(P) if j != WEAK and res.recovery_abs_err[j] < 0.25 * span[j]]
    assert len(good) >= 6 and not res.identified[WEAK]
