"""Panel simulation tests (DF26 Sec 4.2; review EST-1 CRN, Group-1 box)."""
import types

import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.simulation import panel as P
from src.v3.solver import grid as G
from src.v3.validation import vfi

GRID = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)
PARAMS, EXT = cfg.REFERENCE_ESTIMATES, cfg.ExternalParams()


@pytest.fixture(scope="module")
def solution():
    return vfi.solve_vfi(PARAMS, EXT, GRID, mode="howard", tol=1e-8, max_sweeps=200)


@pytest.mark.slow
def test_states_stay_in_box(solution):
    pan = P.simulate_panel(solution, PARAMS, EXT, GRID, master_seed=1,
                           n_firms=500, T=80, burn_in=40)
    k, b = pan["k"].numpy(), pan["b_net"].numpy()
    assert np.isfinite(k).all() and np.isfinite(pan["V"].numpy()).all()
    assert k.min() >= float(solution.grids.k_lo) - 1e-9
    assert k.max() <= float(solution.grids.k_hi) + 1e-9
    assert b.min() >= GRID.b_lo - 1e-9 and b.max() <= GRID.b_hi + 1e-9


@pytest.mark.slow
def test_reproducible_crn(solution):
    a = P.simulate_panel(solution, PARAMS, EXT, GRID, master_seed=2, n_firms=300, T=60, burn_in=30)
    b = P.simulate_panel(solution, PARAMS, EXT, GRID, master_seed=2, n_firms=300, T=60, burn_in=30)
    c = P.simulate_panel(solution, PARAMS, EXT, GRID, master_seed=9, n_firms=300, T=60, burn_in=30)
    assert np.array_equal(a["V"].numpy(), b["V"].numpy())
    assert not np.array_equal(a["V"].numpy(), c["V"].numpy())


def test_reseed_keeps_firms_in_box_when_all_default():
    # value_raw all negative -> every firm defaults each period -> all reseed to grid nodes.
    g = G.build_grids(PARAMS, EXT, GRID)
    shape = [GRID.n_z, GRID.n_k, GRID.n_b]
    fake = types.SimpleNamespace(
        value_raw=-tf.ones(shape, tf.float64), value=tf.zeros(shape, tf.float64),
        policy_i=tf.zeros(shape, tf.float64), policy_bp=tf.zeros(shape, tf.float64),
        policy_cp=tf.zeros(shape, tf.float64), grids=g)
    pan = P.simulate_panel(fake, PARAMS, EXT, GRID, master_seed=4, n_firms=200, T=20, burn_in=10)
    k, b = pan["k"].numpy(), pan["b_net"].numpy()
    assert (pan["V"].numpy() < 0).all()  # every observation is a default
    assert k.min() >= float(g.k_lo) - 1e-9 and k.max() <= float(g.k_hi) + 1e-9
    assert b.min() >= GRID.b_lo - 1e-9 and b.max() <= GRID.b_hi + 1e-9
