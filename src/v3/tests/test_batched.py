"""Batched-over-parameters pipeline equivalence (DF26 Sec 11, GPU-default path).

The batched grid/refine/simulate/moments carry a ``[B, ...]`` parameter-batch
dimension so B parameter vectors are processed at once (near-B x GPU throughput).
These tests pin that the batched path is the single-parameter path, vectorized:

* grids, interpolation, and the moment formulas are bit-identical;
* batched refine reproduces the single-parameter refine to solver tolerance;
* simulation is reproducible and grouping-invariant.

End-to-end (collector) the batched and serial paths agree EXCEPT for rare draws
where a firm sits within ~1e-13 of the limited-liability default boundary (V < 0):
there the batched [B,S,S] LAPACK solve and the single [S,S] solve differ at float
precision and the discontinuity flips one firm, moving a noisy OLS-slope moment.
``test_collector_batch_size_betas_identical`` documents this (betas identical;
moments identical on all but at most a couple of boundary rows).
"""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common import precision
from src.v3.networks.bundle import NetworkBundle
from src.v3.simulation import batched as SB
from src.v3.simulation import moments as M
from src.v3.solver import batched as BB
from src.v3.solver import grid as G
from src.v3.solver import refine as R
from src.v3.solver.interp import interp_grid

precision.configure_devices("cpu")

GRID = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)
EXT = cfg.ExternalParams()
BOUNDS = cfg.ParamBounds.table_a1()
BETAS = np.stack([
    cfg.REFERENCE_ESTIMATES.to_array(),
    cfg.ModelParams(0.70, 0.60, 0.15, 0.10, 0.5, 0.05, 0.30, 0.10).to_array(),
    cfg.ModelParams(0.85, 0.70, 0.30, 0.15, 0.7, 0.02, 0.50, 0.05).to_array(),
]).astype(np.float64)


@pytest.fixture(scope="module")
def bundle():
    return NetworkBundle(cfg.NetworkConfig(), master_seed=7)


def test_grids_batch_match_single():
    bg = BB.build_grids_batch(tf.constant(BETAS), EXT, GRID)
    for i, bb in enumerate(BETAS):
        sg = G.build_grids(cfg.ModelParams.from_array(bb), EXT, GRID)
        assert float(tf.reduce_max(tf.abs(bg.k[i] - sg.k))) < 1e-12
        assert float(tf.reduce_max(tf.abs(bg.kp[i] - sg.kp))) < 1e-12
        assert float(tf.reduce_max(tf.abs(bg.P[i] - sg.P))) < 1e-12
        assert float(tf.reduce_max(tf.abs(bg.stationary[i] - sg.stationary))) < 1e-12


def test_interp_batch_match_single():
    bg = BB.build_grids_batch(tf.constant(BETAS), EXT, GRID)
    V = tf.random.stateless_normal([3, GRID.n_z, GRID.n_k, GRID.n_b], seed=[1, 2], dtype=tf.float64)
    kq = tf.constant([[1.0, 1.5], [0.8, 1.2], [1.1, 0.9]], tf.float64)
    bq = tf.constant([[0.0, 0.5], [0.1, -0.3], [0.2, 0.4]], tf.float64)
    out = BB.interp_batch(V, tf.math.log(bg.k), bg.b, kq, bq)
    for i in range(3):
        ref = interp_grid(V[i], tf.math.log(bg.k[i]), bg.b, kq[i], bq[i])
        assert float(tf.reduce_max(tf.abs(out[i] - ref))) < 1e-12


def test_moments_formula_match_single():
    # Identical synthetic panel through both moment functions -> bit-identical (no solve/boundary).
    nf, T, bi = 50, 30, 8
    rng = np.random.default_rng(0)
    z = np.exp(rng.normal(0, 0.2, (nf, T)))
    k = np.exp(rng.normal(0, 0.3, (nf, T)))
    b_net = rng.normal(0, 0.2, (nf, T))
    c = np.abs(rng.normal(0.1, 0.05, (nf, T)))
    V = rng.normal(0.5, 1.0, (nf, T))  # some V < 0 -> exercises the good-obs mask
    sca = dict(A_pi=0.5, xi=0.6, cf=0.05, delta=0.12)

    single = {"z": tf.constant(z), "k": tf.constant(k), "b_net": tf.constant(b_net),
              "c": tf.constant(c), "V": tf.constant(V), "burn_in": bi, "T": T, **sca}
    m_single = M.compute_moments(single)

    batch = {key: tf.constant(val)[None] for key, val in
             [("z", z), ("k", k), ("b_net", b_net), ("c", c), ("V", V)]}
    batch.update(burn_in=bi, T=T,
                 **{key: tf.constant([val], tf.float64) for key, val in sca.items()})
    m_batch = SB.compute_moments_batch(batch)[0]
    assert float(tf.reduce_max(tf.abs(m_batch - m_single))) < 1e-12


def test_simulation_reproducible(bundle):
    rb = BB.refine_batch(bundle, tf.constant(BETAS), EXT, GRID, BOUNDS, n_rounds=4)
    seeds = [11, 22, 33]
    kw = dict(n_firms=200, T=40, burn_in=10)
    p1 = SB.simulate_panel_batch(rb, tf.constant(BETAS), EXT, GRID, seeds, **kw)
    p2 = SB.simulate_panel_batch(rb, tf.constant(BETAS), EXT, GRID, seeds, **kw)
    assert float(tf.reduce_max(tf.abs(p1["k"] - p2["k"]))) == 0.0
    assert float(tf.reduce_max(tf.abs(SB.compute_moments_batch(p1)
                                      - SB.compute_moments_batch(p2)))) == 0.0


def test_simulation_grouping_invariant(bundle):
    # The same parameter, alone or inside a batch, simulates bit-identically.
    solo = BB.refine_batch(bundle, tf.constant(BETAS[2:3]), EXT, GRID, BOUNDS, n_rounds=4)
    grp = BB.refine_batch(bundle, tf.constant(BETAS), EXT, GRID, BOUNDS, n_rounds=4)
    kw = dict(n_firms=200, T=40, burn_in=10)
    ps = SB.simulate_panel_batch(solo, tf.constant(BETAS[2:3]), EXT, GRID, [33], **kw)
    pg = SB.simulate_panel_batch(grp, tf.constant(BETAS), EXT, GRID, [11, 22, 33], **kw)
    ms = SB.compute_moments_batch(ps)[0]
    mg = SB.compute_moments_batch(pg)[2]
    assert float(tf.reduce_max(tf.abs(ms - mg))) == 0.0


@pytest.mark.slow
def test_refine_batch_matches_single(bundle):
    rb = BB.refine_batch(bundle, tf.constant(BETAS), EXT, GRID, BOUNDS, n_rounds=6)
    for i, bb in enumerate(BETAS):
        rs = R.refine(bundle, cfg.ModelParams.from_array(bb), EXT, GRID, BOUNDS, n_rounds=6)
        good = rs.value.numpy() > 1e-6
        assert np.max(np.abs(rb.value_raw[i].numpy()[good] - rs.value_raw.numpy()[good])) < 1e-9
        # Identical policy grid nodes; the gathered control values differ only by the
        # grids' own ~1e-16 float noise (k/k' built per-batch vs per-parameter).
        assert np.max(np.abs(rb.policy_kp[i].numpy() - rs.policy_kp.numpy())) < 1e-12
        assert np.max(np.abs(rb.policy_bp[i].numpy() - rs.policy_bp.numpy())) < 1e-12
        assert np.max(np.abs(rb.policy_cp[i].numpy() - rs.policy_cp.numpy())) < 1e-12


@pytest.mark.slow
def test_collector_batch_size_betas_identical(bundle):
    from src.v3.estimation.collector import collect_dataset_batch
    kw = dict(refine_rounds=6, n_firms=300, T=50, burn_in=15)
    b1, m1 = collect_dataset_batch(bundle, BOUNDS, EXT, GRID, 7, 10, batch_size=1, **kw)
    b5, m5 = collect_dataset_batch(bundle, BOUNDS, EXT, GRID, 7, 10, batch_size=5, **kw)
    # Same draws kept in the same order -> identical parameter rows.
    assert float(tf.reduce_max(tf.abs(b1 - b5))) == 0.0
    # Moments identical on all but at most a couple of default-boundary-sensitive rows.
    differing = int(tf.reduce_sum(tf.cast(
        tf.reduce_max(tf.abs(m1 - m5), axis=1) > 1e-9, tf.int32)))
    assert differing <= 2
