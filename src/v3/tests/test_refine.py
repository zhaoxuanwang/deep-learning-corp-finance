"""Grid-refinement tests (DF26 Sec 4.1, 12.3).

The rigorous correctness gate does not depend on training quality: (1) policy
evaluation reproduces the VFI fixed point exactly, and (2) the VFI policy is a fixed
point of the local improvement search. We additionally check that refining a trained
network moves its value toward VFI. The strict 1%-policy match of Sec 12.3 needs the
fine BASE control grid (where plain-VI VFI is infeasible) and is demonstrated in the
demo notebook; on the coarse SMOKE grid policy deviations are bounded by grid spacing.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.networks.bundle import NetworkBundle
from src.v3.solver import refine
from src.v3.solver import refine_evaluate as RE
from src.v3.solver import refine_improve as RI
from src.v3.solver import trainer
from src.v3.validation import evaluate, vfi

pytestmark = pytest.mark.slow
GRID = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)


def _flat_states(grids):
    nz, nk, nb = grids.z.shape[0], grids.k.shape[0], grids.b.shape[0]
    zm, km, bm = tf.meshgrid(grids.z, grids.k, grids.b, indexing="ij")
    return (tf.reshape(zm, [-1]), tf.reshape(km, [-1]), tf.reshape(bm, [-1]),
            tf.repeat(tf.range(nz), nk * nb))


def _nearest(grid_1d, values):
    return tf.cast(tf.argmin(tf.abs(values[:, None] - grid_1d), axis=-1), tf.int32)


@pytest.fixture(scope="module")
def setup():
    bounds = cfg.ParamBounds.table_a1()
    ext = cfg.ExternalParams()
    params = cfg.REFERENCE_ESTIMATES
    vfi_sol = vfi.solve_vfi(params, ext, GRID, tol=1e-10, max_sweeps=4000)
    tc = cfg.TrainConfig(batch_size=1024, steps_per_epoch=40, gh_nodes=3)
    bundle = NetworkBundle(cfg.NetworkConfig(), master_seed=7, bprime_init_bias=tc.bprime_init_bias)
    trainer.train_block1(bundle, bounds, ext, GRID, tc, master_seed=7, n_epochs=8, compile_step=True)
    refined = refine.refine(bundle, params, ext, GRID, bounds, n_rounds=6)
    net = evaluate.network_on_grid(bundle, params, ext, GRID, bounds)
    return dict(vfi=vfi_sol, refined=refined, net=net, params=params, ext=ext)


def test_policy_evaluate_reproduces_vfi_fixed_point(setup):
    v = setup["vfi"]
    g = v.grids
    z, k, b, zi = _flat_states(g)
    logk = tf.math.log(g.k)
    kp = tf.reshape(v.policy_kp, [-1])
    bp = tf.reshape(v.policy_bp, [-1])
    cp = tf.reshape(v.policy_cp, [-1])
    v_eval, _ = RE.policy_evaluate(tf.maximum(v.value, 0.0), z, k, b, zi, kp, bp, cp,
                                   g.P, logk, g.b, setup["params"], setup["ext"],
                                   tol=1e-10, max_iter=4000)
    good = v.value.numpy() > 1e-6
    assert np.max(np.abs(v_eval.numpy()[good] - v.value.numpy()[good])) < 1e-7


def test_dense_policy_evaluate_reproduces_vfi(setup):
    # The semismooth-Newton + dense linear solve reproduces the VFI fixed point.
    v = setup["vfi"]
    g = v.grids
    z, k, b, zi = _flat_states(g)
    logk = tf.math.log(g.k)
    kp = tf.reshape(v.policy_kp, [-1])
    bp = tf.reshape(v.policy_bp, [-1])
    cp = tf.reshape(v.policy_cp, [-1])
    v_dense, _ = RE.policy_evaluate(tf.maximum(v.value, 0.0), z, k, b, zi, kp, bp, cp,
                                    g.P, logk, g.b, setup["params"], setup["ext"],
                                    method="dense", tol=1e-10, max_iter=50)
    good = v.value.numpy() > 1e-6
    assert np.max(np.abs(v_dense.numpy()[good] - v.value.numpy()[good])) < 1e-7


def test_vfi_policy_is_local_optimum(setup):
    v = setup["vfi"]
    g = v.grids
    shape = v.value.shape
    z, k, b, zi = _flat_states(g)
    logk = tf.math.log(g.k)
    a = tf.reshape(_nearest(g.kp, tf.reshape(v.policy_kp, [-1])), shape)
    c = tf.reshape(_nearest(g.bp, tf.reshape(v.policy_bp, [-1])), shape)
    d = tf.reshape(_nearest(g.cp, tf.reshape(v.policy_cp, [-1])), shape)
    na, nc, nd = RI.policy_improve(tf.maximum(v.value, 0.0), a, c, d, z, k, b, zi,
                                   g, logk, setup["params"], setup["ext"])
    changed = np.mean((na.numpy() != a.numpy()) | (nc.numpy() != c.numpy()) | (nd.numpy() != d.numpy()))
    assert changed < 0.02  # the global optimum is a fixed point of the local search


def test_refinement_moves_value_toward_vfi(setup):
    v, r, net = setup["vfi"], setup["refined"], setup["net"]
    good = v.value.numpy() > 1e-6
    vv = v.value.numpy()[good]
    err_refined = np.max(np.abs(r.value.numpy()[good] - vv))
    err_network = np.max(np.abs(np.maximum(net.value.numpy(), 0.0)[good] - vv))
    assert err_refined <= err_network + 1e-9


def test_refined_value_within_tolerance(setup):
    v, r = setup["vfi"], setup["refined"]
    good = v.value.numpy() > 1e-6
    vv = v.value.numpy()[good]
    rng = vv.max() - vv.min()
    assert np.max(np.abs(r.value.numpy()[good] - vv)) / rng < 0.12


def test_refined_passes_property_checks(setup):
    # Group 1 (bounds) + Group 2 (monotonicity) hard checks on the refined solution.
    from src.v3.validation.properties import check_properties
    checks = check_properties(setup["refined"], GRID)
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"refined solution failed property checks: {failed}"
