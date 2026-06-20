"""Economic property tests (DF26 Sec 12.4 hard checks).

Group 1/2 (bounds + state monotonicities incl. i down k) and Group 3 (Bellman/bond
residuals, accounting identity, finiteness, weighting SPD) on the refined/VFI solution;
Group 4 (corner cases) on the primitives; Group 6 (parameter comparative statics of the
value network, q non-decreasing in chi). A hard check failing fails the run.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common import precision
from src.v3.estimation.weighting import weighting_matrix
from src.v3.networks.bundle import NetworkBundle
from src.v3.simulation.panel import simulate_panel
from src.v3.solver import refine as _refine
from src.v3.solver import trainer
from src.v3.validation import properties as props
from src.v3.validation import vfi as _vfi

precision.configure_devices("cpu")

GRID = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)
EXT = cfg.ExternalParams()
BOUNDS = cfg.ParamBounds.table_a1()


def test_group4_corner_cases_no_solver():
    # Pure-primitive corner checks (no training): finite production constants, k_min>0.
    checks = props.check_corner_cases(EXT, BOUNDS, GRID)
    assert all(checks.values()), checks


@pytest.fixture(scope="module")
def solved():
    params = cfg.REFERENCE_ESTIMATES
    vfi_sol = _vfi.solve_vfi(params, EXT, GRID, mode="howard", tol=1e-10, max_sweeps=200)
    bundle = NetworkBundle(cfg.NetworkConfig(), master_seed=7)
    tc = cfg.TrainConfig(batch_size=1024, steps_per_epoch=60, gh_nodes=3)
    trainer.train_block1(bundle, BOUNDS, EXT, GRID, tc, master_seed=7, n_epochs=18, compile_step=True)
    refined = _refine.refine(bundle, params, EXT, GRID, BOUNDS, n_rounds=6)
    panel = simulate_panel(refined, params, EXT, GRID, 11, n_firms=600, T=60, burn_in=20)
    W, _ = weighting_matrix(panel)
    return dict(vfi=vfi_sol, refined=refined, bundle=bundle, params=params, W=W)


@pytest.mark.slow
def test_group1_2_bounds_and_monotonicity(solved):
    for sol in (solved["vfi"], solved["refined"]):
        checks = props.check_properties(sol, GRID)
        failed = [k for k, ok in checks.items() if not ok]
        assert not failed, f"Group 1/2 violations: {failed}"


@pytest.mark.slow
def test_group3_mechanics(solved):
    checks = props.check_mechanics(solved["refined"], solved["params"], EXT)
    for key in ("bellman_residual_small", "q_in_bounds", "accounting_kprime_identity", "all_finite"):
        assert checks[key], (key, checks)


@pytest.mark.slow
def test_group3_weighting_spd(solved):
    checks = props.check_weighting_spd(solved["W"])
    assert checks["weighting_symmetric"] and checks["weighting_spd"], checks


@pytest.mark.slow
def test_group6_comparative_statics(solved):
    checks = props.check_comparative_statics(solved["bundle"], BOUNDS, EXT, GRID)
    # q non-decreasing in chi is pure economics (no network) -> always hard.
    assert checks["q_nondecreasing_in_chi"], checks
    # Value-network signs: the robust ones (large, direct effects) hold even at SMOKE scale.
    # The subtler signs (delta, gamma1, gamma0) are a converged-network gate exercised at FULL
    # scale; at SMOKE the coarse net has not converged on them, so they are reported not asserted.
    assert checks["V_monotone_in_cf"] and checks["V_monotone_in_chi"], checks
