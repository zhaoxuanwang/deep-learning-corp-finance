"""Regression golden test (DF26 Sec 12.5): a thin drift guard, not a correctness test.

It does NOT re-run training. It freezes one deterministic configuration (a seed-frozen
network + surrogate, equivalent to a committed checkpoint since the init is fully
reproducible) and re-checks only the fast, deterministic pieces against committed goldens:

  (a) the network forward pass (V and the three controls) at fixed inputs;
  (b) the bond price q and dividend D at fixed inputs (Eqs 6-9);
  (c) one on-grid refinement round from the network solution;
  (d) a short fixed-seed simulation and its moments (and the integer default count);
  (e) one LM solve on the frozen surrogate against beta_0.

Floats compared at rtol 1e-5 / atol 1e-7; integer/structural outputs (default count)
exactly. Run CPU-only with op-level determinism (Sec 10). On the first run (no goldens
file) it writes the goldens and skips; commit that file so later runs guard against drift.
Regenerate intentionally (delete the file) when a change alters the goldens on purpose.
"""
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.v3.common import precision

precision.configure_determinism()

from src.v3 import config as cfg  # noqa: E402
from src.v3.common.normalization import ParamScaler  # noqa: E402
from src.v3.economics import bounds as ebounds  # noqa: E402
from src.v3.economics import debt as edebt  # noqa: E402
from src.v3.economics import dividends as ediv  # noqa: E402
from src.v3.estimation.estimate import estimate  # noqa: E402
from src.v3.estimation.surrogate import SurrogateEnsemble  # noqa: E402
from src.v3.estimation.weighting import weighting_matrix  # noqa: E402
from src.v3.networks.bundle import NetworkBundle  # noqa: E402
from src.v3.simulation.moments import compute_moments  # noqa: E402
from src.v3.simulation.panel import simulate_panel  # noqa: E402
from src.v3.solver import grid as Grid  # noqa: E402
from src.v3.solver import refine as _refine  # noqa: E402

GOLDEN = Path(__file__).parent / "regression_fixtures" / "goldens.npz"
SEED = 12345
GRID = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)
EXT = cfg.ExternalParams()
BOUNDS = cfg.ParamBounds.table_a1()
PARAMS = cfg.REFERENCE_ESTIMATES
F64 = tf.float64


def _compute_pieces():
    precision.configure_determinism()
    bundle = NetworkBundle(cfg.NetworkConfig(), master_seed=SEED)
    g = Grid.build_grids(PARAMS, EXT, GRID)
    scaler32 = ParamScaler(BOUNDS, dtype=tf.float32)

    # (a) network forward pass at fixed inputs.
    z0, k0, b0 = float(g.z[2]), float(g.k[2]), float(g.b[3])
    box = ebounds.state_box(PARAMS.theta, EXT.alpha, PARAMS.delta, EXT.rf, PARAMS.rho,
                            PARAMS.sigma, GRID.tauchen_m, GRID.b_lo, GRID.b_hi)
    state_norm = ebounds.normalize_state(tf.constant([[np.log(z0), k0, b0]], tf.float32), box)
    pnorm = scaler32.normalize(PARAMS.to_array()[None])
    i_lo, i_hi = ebounds.investment_rate_bounds(
        tf.constant([k0], tf.float32), tf.cast(box.k_lo, tf.float32),
        tf.cast(box.k_hi, tf.float32), tf.cast(PARAMS.delta, tf.float32))
    net_fwd = np.array([
        float(bundle.value(state_norm, pnorm)[0]),
        float(bundle.policy_i(state_norm, pnorm, i_lo, i_hi)[0]),
        float(bundle.policy_bp(state_norm, pnorm, GRID.bp_lo, GRID.bp_hi)[0]),
        float(bundle.policy_cp(state_norm, pnorm, GRID.cp_lo, GRID.cp_hi)[0]),
    ])

    # (b) bond price + dividend at fixed inputs (Eqs 6-9).
    kp, bpv, cpv, pdef = 1.0, 0.5, 0.1, 0.3
    c = lambda x: tf.constant(x, F64)
    q = edebt.bond_price(c(pdef), c(kp), c(cpv), c(bpv), c(PARAMS.chi), c(PARAMS.delta), EXT.bond_discount)
    D = ediv.dividend(c(z0), c(k0), c(b0), c(kp / k0 - (1.0 - PARAMS.delta)), c(bpv), c(cpv), q,
                      theta=PARAMS.theta, alpha=EXT.alpha, delta=PARAMS.delta, gamma1=PARAMS.gamma1,
                      gamma0=PARAMS.gamma0, cf=PARAMS.cf, lambda0=EXT.lambda0, lambda1=EXT.lambda1,
                      iota_c=EXT.iota_c, smooth=False).D
    econ = np.array([float(q), float(D)])

    # (c) one on-grid refinement round.
    refined1 = _refine.refine(bundle, PARAMS, EXT, GRID, BOUNDS, n_rounds=1)
    refine_v = refined1.value_raw.numpy()

    # (d) short fixed-seed simulation + moments + default count.
    panel = simulate_panel(refined1, PARAMS, EXT, GRID, SEED, n_firms=200, T=40, burn_in=10)
    moments = compute_moments(panel).numpy()
    n_default = int(tf.reduce_sum(tf.cast(panel["V"] < 0.0, tf.int32)))
    W, _ = weighting_matrix(panel)

    # (e) one LM solve on the frozen (seed-init) surrogate against beta_0 = the reference targets.
    surr = SurrogateEnsemble(SEED, n_folds=3)
    lo = tf.constant(BOUNDS.lower_array(), F64)
    hi = tf.constant(BOUNDS.upper_array(), F64)
    betas = lo + tf.cast(tf.linspace(0.0, 1.0, 60)[:, None], F64) * (hi - lo)
    surr.train(betas, tf.tile(tf.constant(cfg.REFERENCE_TARGETS)[None], [60, 1]), BOUNDS, SEED, passes=5)
    beta_hat = estimate(surr, tf.constant(cfg.REFERENCE_TARGETS), W, BOUNDS, SEED, n_restarts=4)["beta_hat"].numpy()

    return {"net_fwd": net_fwd, "econ": econ, "refine_v": refine_v,
            "moments": moments, "n_default": np.array(n_default), "beta_hat": beta_hat}


def test_regression_goldens():
    pieces = _compute_pieces()
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GOLDEN, **pieces)
        pytest.skip(f"regression goldens written to {GOLDEN}; commit and rerun to guard drift")
    gold = np.load(GOLDEN)
    assert int(pieces["n_default"]) == int(gold["n_default"])      # structural: exact
    for key in ("net_fwd", "econ", "refine_v", "moments", "beta_hat"):
        np.testing.assert_allclose(pieces[key], gold[key], rtol=1e-5, atol=1e-7,
                                   err_msg=f"regression drift in {key}")
