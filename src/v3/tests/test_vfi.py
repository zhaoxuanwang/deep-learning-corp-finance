"""VFI benchmark oracle tests (DF26 Sec 12.1; review OPT-3, property Groups 1-3).

The VFI solve is shared across assertions via a module-scoped fixture (one solve).
Marked slow because plain value iteration takes ~1000 sweeps.
"""
import numpy as np
import pytest

from src.v3 import config as cfg
from src.v3.validation import vfi

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def solution():
    p = cfg.REFERENCE_ESTIMATES
    e = cfg.ExternalParams()
    g = cfg.GridConfig(n_z=5, n_k=5, n_b=7, n_kp=9, n_bp=9, n_cp=7)
    return vfi.solve_vfi(p, e, g, tol=1e-8, max_sweeps=3000), e, g


def test_converges(solution):
    r, _, _ = solution
    assert r.converged
    assert r.final_change < 1e-8


def test_value_finite_and_nonnegative(solution):
    r, _, _ = solution
    V = r.value.numpy()
    assert np.isfinite(V).all()
    assert (V >= 0).all()  # limited liability


def test_value_increasing_in_z(solution):
    # Hard property (Group 2): V increasing in z, pointwise where V > 0.
    r, _, _ = solution
    V = r.value.numpy()  # [n_z, n_k, n_b]
    d = np.diff(V, axis=0)
    both_pos = (V[:-1] > 1e-9) & (V[1:] > 1e-9)
    assert np.all(d[both_pos] >= -1e-9)


def test_value_decreasing_in_b(solution):
    # Hard property (Group 2): V decreasing in net debt b, where V > 0.
    r, _, _ = solution
    V = r.value.numpy()  # [n_z, n_k, n_b]
    d = np.diff(V, axis=2)
    both_pos = (V[..., :-1] > 1e-9) & (V[..., 1:] > 1e-9)
    assert np.all(d[both_pos] <= 1e-9)


def test_policies_within_bounds(solution):
    # Group 1: controls land in their boxes; net debt b'-c' in [-1, 2].
    r, _, g = solution
    bp, cp, kp = r.policy_bp.numpy(), r.policy_cp.numpy(), r.policy_kp.numpy()
    assert bp.min() >= g.bp_lo - 1e-12 and bp.max() <= g.bp_hi + 1e-12
    assert cp.min() >= g.cp_lo - 1e-12 and cp.max() <= g.cp_hi + 1e-12
    net = bp - cp
    assert net.min() >= g.b_lo - 1e-9 and net.max() <= g.b_hi + 1e-9
    assert kp.min() >= float(r.grids.k_lo) - 1e-9
    assert kp.max() <= float(r.grids.k_hi) + 1e-9


def test_howard_matches_value_iteration():
    # Howard PFI (global improve + dense policy eval) reaches the same fixed point
    # as plain value iteration, in far fewer sweeps (Sec 12.1).
    p = cfg.REFERENCE_ESTIMATES
    e = cfg.ExternalParams()
    g = cfg.GridConfig(n_z=5, n_k=4, n_b=5, n_kp=7, n_bp=7, n_cp=5)
    vi = vfi.solve_vfi(p, e, g, tol=1e-8, max_sweeps=4000, mode="value_iteration")
    hw = vfi.solve_vfi(p, e, g, tol=1e-8, max_sweeps=200, mode="howard")
    assert hw.converged and hw.n_sweeps < 50
    good = vi.value.numpy() > 1e-6
    np.testing.assert_allclose(hw.value.numpy()[good], vi.value.numpy()[good], atol=1e-5)


def test_start_independent():
    # Contraction (OPT-3): cold and warm starts reach the same fixed point.
    p = cfg.REFERENCE_ESTIMATES
    e = cfg.ExternalParams()
    g = cfg.GridConfig(n_z=4, n_k=4, n_b=5, n_kp=7, n_bp=7, n_cp=5)
    cold = vfi.solve_vfi(p, e, g, tol=1e-9, max_sweeps=5000)
    warm_init = 50.0 * np.ones((4, 4, 5))
    warm = vfi.solve_vfi(p, e, g, tol=1e-9, max_sweeps=5000, v_init=warm_init)
    np.testing.assert_allclose(cold.value.numpy(), warm.value.numpy(), atol=1e-6)
