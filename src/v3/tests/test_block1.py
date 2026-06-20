"""Block-1 training tests (DF26 Sec 3.4; review TF-1 retracing, NUM-1 finiteness).

The loose oracle gate (Sec 12.3 precursor): the trained network correlates with the
VFI benchmark. The tight 1% gate comes after grid refinement (M5).
"""
import numpy as np
import pytest
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET, make_adam
from src.v3.common.quadrature import gauss_hermite
from src.v3.networks.bundle import NetworkBundle
from src.v3.solver import trainer
from src.v3.solver.sampling import make_batch
from src.v3.solver.train_step import make_step_fn
from src.v3.validation import evaluate, vfi

NETCFG = cfg.NetworkConfig()
SMOKE_GRID = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)


def test_step_does_not_retrace():
    # review TF-1: the compiled step must trace once, not per call.
    bounds = cfg.ParamBounds.table_a1()
    ext = cfg.ExternalParams()
    tc = cfg.TrainConfig(batch_size=256, steps_per_epoch=2, gh_nodes=3)
    bundle = NetworkBundle(NETCFG, master_seed=1)
    scaler = ParamScaler(bounds, dtype=TF_FLOAT_NET)
    x, w = gauss_hermite(tc.gh_nodes, dtype=TF_FLOAT_NET)
    tau = tf.constant(tc.smooth_tau, TF_FLOAT_NET)
    step = make_step_fn(bundle, ext, SMOKE_GRID, x, w, tau,
                        make_adam(1e-3), make_adam(1e-3), compile_step=True)
    for s in range(3):
        step(make_batch(256, bounds, ext, SMOKE_GRID, scaler, 1, 0, s))
    count_after_3 = step.experimental_get_tracing_count()
    for s in range(3, 8):
        step(make_batch(256, bounds, ext, SMOKE_GRID, scaler, 1, 0, s))
    # Tracing must stabilize, not grow per call (reduce_retracing settles at <= 2).
    assert step.experimental_get_tracing_count() == count_after_3
    assert count_after_3 <= 2


@pytest.fixture(scope="module")
def trained():
    bounds = cfg.ParamBounds.table_a1()
    ext = cfg.ExternalParams()
    params = cfg.REFERENCE_ESTIMATES
    tc = cfg.TrainConfig(batch_size=1024, steps_per_epoch=40, gh_nodes=3)
    bundle = NetworkBundle(NETCFG, master_seed=7, bprime_init_bias=tc.bprime_init_bias)
    history = trainer.train_block1(bundle, bounds, ext, SMOKE_GRID, tc,
                                   master_seed=7, n_epochs=8, compile_step=True)
    net_grid = evaluate.network_on_grid(bundle, params, ext, SMOKE_GRID, bounds)
    vfi_sol = vfi.solve_vfi(params, ext, SMOKE_GRID, tol=1e-8, max_sweeps=3000)
    return history, net_grid, vfi_sol


@pytest.mark.slow
def test_training_reduces_residual_and_finite(trained):
    history, _, _ = trained
    losses = [h["loss_v"] for h in history]
    assert all(np.isfinite(h["loss_v"]) and np.isfinite(h["loss_pi"]) for h in history)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]  # meaningful reduction


@pytest.mark.slow
def test_value_correlates_with_vfi(trained):
    _, net_grid, vfi_sol = trained
    vn = net_grid.value.numpy().ravel()
    vv = vfi_sol.value.numpy().ravel()
    good = vv > 1e-6  # compare in the non-default region
    corr = np.corrcoef(vn[good], vv[good])[0, 1]
    assert corr > 0.85, f"value correlation with VFI too low: {corr:.3f}"


@pytest.mark.slow
def test_investment_correlates_with_vfi(trained):
    _, net_grid, vfi_sol = trained
    good = vfi_sol.value.numpy().ravel() > 1e-6
    inet = net_grid.policy_i.numpy().ravel()[good]
    ivfi = vfi_sol.policy_i.numpy().ravel()[good]
    corr = np.corrcoef(inet, ivfi)[0, 1]
    assert corr > 0.4, f"investment correlation with VFI too low: {corr:.3f}"
