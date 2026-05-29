from __future__ import annotations

import numpy as np
import pytest

from src.v2.environments.ceo_contract import (
    CEOContractGridConfig,
    CEOContractParams,
    build_ceo_grids,
    flow_payoff,
    optimal_effort,
    optimal_manipulation,
    optimal_sigma_z,
    terminal_value,
)
from src.v2.simulation.ceo_contract import (
    CEOContractSimulationConfig,
    simulate_ceo_contract_panel,
)
from src.v2.solvers.ceo_contract import _fd_derivatives, solve_ceo_contract


@pytest.fixture(scope="module")
def params():
    return CEOContractParams(
        r=0.1, gamma=5.0, sigma=1.0, kappa=0.3, theta=0.5, g=1.0, T=2.0, tau=1.0
    )


@pytest.fixture(scope="module")
def config():
    return CEOContractGridConfig(z_bounds=(0.0, 2.0), n_z=61, n_t=61, sigma_z_max=10.0)


@pytest.fixture(scope="module")
def solved(config, params):
    return solve_ceo_contract(config, params)


# --- parameter / config validation -----------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0.0},
        {"gamma": -1.0},
        {"sigma": 0.0},
        {"kappa": -0.1},
        {"theta": 0.0},
        {"g": -1.0},
        {"T": 0.0},
        {"tau": -1.0},
        {"tau": 0.0},
        {"C_override": -1.0},
    ],
)
def test_params_validation_raises(kwargs):
    base = dict(r=0.1, gamma=5.0, sigma=1.0, kappa=0.3, theta=0.5, g=1.0, T=2.0, tau=1.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        CEOContractParams(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"z_bounds": (2.0, 1.0)},
        {"z_bounds": (0.1, 0.3)},  # f0_dirichlet requires lower bound 0
        {"f0_dirichlet": False},
        {"n_z": 2},
        {"n_t": 1},
        {"vzz_floor": 0.0},
        {"a_max": 0.0},
        {"sigma_z_max": -1.0},
    ],
)
def test_grid_config_validation_raises(kwargs):
    with pytest.raises(ValueError):
        CEOContractGridConfig(**kwargs)


def test_dirichlet_lower_bound_cannot_be_disabled():
    with pytest.raises(ValueError, match="f0_dirichlet=False is not implemented"):
        CEOContractGridConfig(z_bounds=(0.0, 0.3), f0_dirichlet=False)


def test_derived_parameters(params):
    assert params.lambda_ == pytest.approx(params.theta / (params.r + params.kappa) - 1.0)
    assert params.phi == pytest.approx(params.theta / (params.r * params.gamma))


def test_terminal_cost_closed_form():
    # Marinovic-Varas (2019) Eq. 10 at the paper's Figure-1 baseline.
    p = CEOContractParams(r=0.1, gamma=1.0, sigma=2.0, kappa=0.3, theta=0.4, g=1.0, T=10.0, tau=5.0)
    rate = p.r + 2.0 * p.kappa
    expected = p.sigma ** 2 * rate / (p.r * p.gamma * (1.0 - np.exp(-rate * p.tau)))
    assert p.C == pytest.approx(expected)
    assert p.C == pytest.approx(28.8719, abs=1e-3)
    # tau -> infinity floor and tau -> 0 blow-up
    floor = p.sigma ** 2 * rate / (p.r * p.gamma)
    long_tau = CEOContractParams(r=0.1, gamma=1.0, sigma=2.0, kappa=0.3, theta=0.4, T=10.0, tau=1e6)
    assert long_tau.C == pytest.approx(floor)
    short_tau = CEOContractParams(r=0.1, gamma=1.0, sigma=2.0, kappa=0.3, theta=0.4, T=10.0, tau=1e-3)
    assert short_tau.C > 100.0 * floor


def test_C_override():
    p = CEOContractParams(C_override=3.5)
    assert p.C == 3.5


def test_grid_construction(config, params):
    grids = build_ceo_grids(config, params)
    z, t = grids["z"], grids["t"]
    assert z[0] == pytest.approx(config.z_bounds[0])
    assert z[-1] == pytest.approx(config.z_bounds[1])
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(params.T)
    assert grids["dz"] == pytest.approx((z[-1] - z[0]) / (config.n_z - 1))
    assert grids["dt"] == pytest.approx(params.T / (config.n_t - 1))


# --- primitives --------------------------------------------------------------

def test_terminal_value(config, params):
    z = build_ceo_grids(config, params)["z"]
    np.testing.assert_allclose(terminal_value(z, params), -0.5 * params.C * z ** 2)


def test_manipulation_floor_and_kink(params):
    z = np.linspace(0.0, 2.0, 50)
    # effort below phi*z -> manipulation floored at zero
    a_low = 0.5 * params.phi * z
    np.testing.assert_allclose(optimal_manipulation(a_low, z, params), 0.0)
    # effort above phi*z -> linear in (a - phi z)
    a_high = params.phi * z + 1.3
    np.testing.assert_allclose(
        optimal_manipulation(a_high, z, params), 1.3 / params.g
    )


def test_optimal_effort_clipped_nonnegative(params):
    z = np.linspace(0.0, 2.0, 50)
    Fz = np.full_like(z, 5.0)  # large positive Fz drives the raw FOC negative
    sigma_z = np.zeros_like(z)
    a = optimal_effort(z, Fz, sigma_z, params, a_max=1e3)
    assert np.all(a >= 0.0)
    assert np.all(a <= 1e3)


def test_sigma_z_sign_matches_fz(params):
    # sigma_z = -r gamma sigma a Fz / Fzz, Fzz<0 -> sign(sigma_z) = -sign(Fz)
    a = np.array([1.0, 1.0])
    Fz = np.array([-2.0, 2.0])
    Fzz = np.array([-1.0, -1.0])
    s = optimal_sigma_z(a, Fz, Fzz, params)
    assert s[0] < 0.0 and s[1] > 0.0


# --- solver ------------------------------------------------------------------

def test_solver_converges_and_shapes(solved, config):
    assert solved["converged"] is True
    n_t, n_z = config.n_t, config.n_z
    assert solved["value"].shape == (n_t, n_z)
    for key in ("a", "m", "sigma_z"):
        assert solved["policy"][key].shape == (n_t, n_z)


def test_solver_terminal_condition(solved, params):
    z = solved["grids"]["z"]
    np.testing.assert_allclose(solved["value"][-1], -0.5 * params.C * z ** 2)


def test_solver_value_is_concave(solved):
    # F_zz < 0 on the interior at every time level
    z = solved["grids"]["z"]
    dz = solved["grids"]["dz"]
    for n in range(solved["value"].shape[0]):
        _, _, fzz = _fd_derivatives(solved["value"][n], dz)
        assert fzz[1:-1].max() < 1e-6


def test_solver_policy_signs(solved, config):
    sigma_z = solved["policy"]["sigma_z"]
    assert np.all(solved["policy"]["m"] >= 0.0)
    assert np.all(np.abs(sigma_z) <= config.sigma_z_max + 1e-9)
    # Under the F(0,t)=0 BC, F is decreasing (F_z <= 0), so sigma_z <= 0 up to
    # small numerical noise (the paper's sign property): no large positive values.
    assert sigma_z.max() <= 0.1


def test_solver_dirichlet_lower_boundary(solved):
    # f0_dirichlet enforces F(0, t) = 0 at every time level
    np.testing.assert_allclose(solved["value"][:, 0], 0.0, atol=1e-10)


def test_solver_low_hjb_residual(solved):
    # bulk HJB accuracy (the max is dominated by the manipulation-kink node, where
    # central differences are unreliable; the Monte-Carlo value match is the
    # authoritative correctness check).
    assert solved["diagnostics"]["hjb_residual_median"] < 0.05


def test_horizon_problem_manipulation_rises(solved):
    # mean manipulation across z is larger near retirement than at the start
    m = solved["policy"]["m"]
    assert m[-2].mean() > m[0].mean()


# --- simulator ---------------------------------------------------------------

def test_simulation_is_deterministic(solved, params):
    cfg = CEOContractSimulationConfig(n_paths=200, seed=7)
    a = simulate_ceo_contract_panel(solved, params, cfg)
    b = simulate_ceo_contract_panel(solved, params, cfg)
    assert a.equals(b)


def test_simulation_stays_on_grid_and_finite(solved, params, config):
    panel = simulate_ceo_contract_panel(
        solved, params, CEOContractSimulationConfig(n_paths=300, seed=3)
    )
    assert panel["z"].between(config.z_bounds[0] - 1e-9, config.z_bounds[1] + 1e-9).all()
    cols = ["z", "a", "m", "sigma_z", "M", "X"]
    assert np.isfinite(panel[cols].to_numpy(dtype=np.float64)).all()
    assert (panel["m"] >= 0.0).all()


def test_simulation_horizon_effect(solved, params):
    panel = simulate_ceo_contract_panel(
        solved, params, CEOContractSimulationConfig(n_paths=2000, seed=5)
    )
    by_t = panel.groupby("t")["m"].mean()
    assert by_t.iloc[-1] > by_t.iloc[0]


def test_monte_carlo_value_matches_F():
    # mean simulated discounted payoff reconciles with F(z0, 0), on the
    # paper-aligned baseline (derived C, F(0,t)=0 BC, z_max=0.3).
    p = CEOContractParams()
    cfg = CEOContractGridConfig(n_z=151, n_t=401)
    res = solve_ceo_contract(cfg, p)
    panel = simulate_ceo_contract_panel(
        res, p, CEOContractSimulationConfig(n_paths=4000, seed=11)
    )
    z = res["grids"]["z"]
    V0 = res["value"][0]
    dt = res["grids"]["dt"]
    disc = np.exp(-p.r * panel["t"].to_numpy())
    pi = flow_payoff(panel["a"].to_numpy(), panel["m"].to_numpy(), panel["z"].to_numpy(), p)
    n_t = res["value"].shape[0]
    contrib = (pi * disc * dt).reshape(-1, n_t)  # (n_paths, n_t) row-major by path
    flow = contrib[:, :-1].sum(axis=1)
    z_paths = panel["z"].to_numpy().reshape(-1, n_t)
    z0 = panel["z0"].to_numpy().reshape(-1, n_t)[:, 0]
    terminal = np.exp(-p.r * p.T) * (-0.5 * p.C * z_paths[:, -1] ** 2)
    J = flow + terminal
    F0 = np.interp(z0, z, V0)
    assert np.mean(np.abs(J - F0)) < 0.05
