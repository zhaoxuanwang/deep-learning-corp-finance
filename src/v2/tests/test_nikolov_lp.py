from __future__ import annotations

import numpy as np
import pytest

from src.v2.environments.nikolov import (
    NikolovGridConfig,
    NikolovParams,
    adjustment_cost,
    ar1_z_bounds,
    build_nikolov_grids,
    calibrated_k_bounds,
    conservative_debt_bound,
    eta_grid,
    profit_pre_tax,
    reference_capital,
)
from src.v2.solvers.nikolov_le import solve_nikolov_le
from src.v2.solvers.nikolov_mh import solve_nikolov_mh
from src.v2.solvers.nikolov_to import compute_to_premiums, solve_nikolov_to


@pytest.fixture
def tiny_transition():
    return np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)


@pytest.fixture
def tiny_config(tiny_transition):
    return NikolovGridConfig(
        k_bounds=(1.0, 1.2),
        b_bounds=(0.0, 0.5),
        v_bounds=(0.0, 0.5),
        z_bounds=(1.0, 1.1),
        p_bounds=(0.0, 0.02),
        d_bounds=(0.0, 0.02),
        n_k=2,
        n_b=2,
        n_v=2,
        n_z=2,
        n_p=2,
        n_d=2,
        z_transition=tiny_transition,
    )


@pytest.fixture
def tiny_params():
    return NikolovParams(
        tau=0.2,
        alpha=0.5,
        f=0.0,
        delta=0.1,
        psi=0.0,
        r=0.04,
        kappa=0.5,
        eta_bar=0.0,
        xi=0.5,
        theta=1.0,
        lambda_diversion=0.0,
    )


def test_nikolov_grids_include_financial_zero_and_valid_transition(tiny_config):
    grids = build_nikolov_grids(tiny_config)

    for key in ["b", "v", "p", "d"]:
        assert grids[key][0] == 0.0
        assert np.all(grids[key] >= 0.0)

    np.testing.assert_allclose(grids["Q"].sum(axis=1), 1.0)


def test_nikolov_ar1_calibration_helpers(tiny_params):
    z_bounds = ar1_z_bounds(rho=0.7, sigma=0.15, mu=0.0, z_sd_mult=2.0)
    assert 0.0 < z_bounds[0] < 1.0 < z_bounds[1]

    k_ref = reference_capital(tiny_params, z_bar=1.0)
    assert k_ref > 0.0

    k_bounds = calibrated_k_bounds(
        tiny_params,
        low_mult=0.5,
        high_mult=1.5,
        z_bar=1.0,
    )
    assert k_bounds == pytest.approx((0.5 * k_ref, 1.5 * k_ref))

    b_max = conservative_debt_bound(
        tiny_params,
        k_low=k_bounds[0],
        pledge_fraction=0.5,
    )
    assert 0.0 < b_max < tiny_params.theta * (1.0 - tiny_params.delta) * k_bounds[0]

    config = NikolovGridConfig(
        k_bounds=k_bounds,
        b_bounds=(0.0, b_max),
        v_bounds=(0.1, 0.5),
        z_bounds=z_bounds,
        p_bounds=(0.0, tiny_params.r * b_max),
        d_bounds=(tiny_params.r * 0.1, tiny_params.r * 0.5),
        n_k=3,
        n_b=2,
        n_v=2,
        n_z=3,
        n_p=2,
        n_d=2,
        z_rho=0.7,
        z_sigma=0.15,
        z_mu=0.0,
    )
    grids = build_nikolov_grids(config)
    assert np.all(grids["z"] > 0.0)
    np.testing.assert_allclose(grids["Q"].sum(axis=1), 1.0)
    np.testing.assert_allclose(grids["d"], tiny_params.r * grids["v"])


def test_nikolov_lp_solvers_use_free_value_bounds(tiny_config, tiny_params):
    to_result = solve_nikolov_to(tiny_config, tiny_params)
    le_result = solve_nikolov_le(tiny_config, tiny_params)
    mh_result = solve_nikolov_mh(tiny_config, tiny_params)

    assert to_result["diagnostics"]["lp_value_bounds_free"] is True
    assert le_result["diagnostics"]["lp_value_bounds_free"] is True
    assert mh_result["diagnostics"]["lp_value_bounds_free"] is True


def test_to_zero_debt_has_zero_premium_and_no_default(tiny_config, tiny_params):
    grids = build_nikolov_grids(tiny_config)
    premium, residual, default_prob = compute_to_premiums(grids, tiny_params)

    np.testing.assert_allclose(premium[:, 0, :], 0.0)
    np.testing.assert_allclose(default_prob[:, 0, :], 0.0)
    assert np.max(np.abs(residual)) < 1e-8


def test_to_policy_satisfies_bellman_fixed_point(tiny_config, tiny_params):
    result = solve_nikolov_to(tiny_config, tiny_params)

    np.testing.assert_allclose(
        result["policy"]["value"],
        result["value"],
        rtol=1e-7,
        atol=1e-7,
    )
    assert result["diagnostics"]["max_lp_violation"] < 1e-7


def test_le_policy_contracts_are_feasible(tiny_config, tiny_params):
    result = solve_nikolov_le(tiny_config, tiny_params)
    grids = result["grids"]
    eta_values, eta_probs = eta_grid(tiny_params)
    k_grid = grids["k"]
    b_grid = grids["b"]
    z_grid = grids["z"]
    q = grids["Q"]
    policy = result["policy"]

    for ik, k_current in enumerate(k_grid):
        for ib, b_current in enumerate(b_grid):
            for iz in range(len(z_grid)):
                k_next = policy["k_next"][ik, ib, iz]
                p = policy["p"][ik, ib, iz]
                b_next = policy["b_next"][ik, ib, iz]
                probs = q[iz, :, None] * eta_probs[None, :]

                break_even = np.sum(probs * (p + b_next)) / (1.0 + tiny_params.r)
                assert break_even == pytest.approx(b_current, abs=1e-10)
                assert np.all(
                    p + b_next
                    <= tiny_params.theta * (1.0 - tiny_params.delta) * k_next
                    + 1e-10
                )

                resources = (
                    (1.0 - tiny_params.tau)
                    * profit_pre_tax(
                        k_next,
                        z_grid[:, None],
                        eta_values[None, :],
                        tiny_params,
                    )
                    - k_next
                    + (1.0 - tiny_params.delta) * k_next
                    - adjustment_cost(k_next, k_current, tiny_params)
                    + tiny_params.tau * tiny_params.delta * k_next
                    + tiny_params.tau * tiny_params.r * b_current
                )
                assert np.all(p <= resources + 1e-10)

    np.testing.assert_allclose(result["policy"]["value"], result["value"], atol=1e-7)


def test_mh_policy_contracts_are_feasible(tiny_config, tiny_params):
    result = solve_nikolov_mh(tiny_config, tiny_params)
    grids = result["grids"]
    eta_values, eta_probs = eta_grid(tiny_params)
    v_grid = grids["v"]
    z_grid = grids["z"]
    q = grids["Q"]
    policy = result["policy"]

    for ik in range(len(grids["k"])):
        for iv, v_current in enumerate(v_grid):
            for iz in range(len(z_grid)):
                v_next = policy["V_next"][ik, iv, iz]
                d = policy["d"][ik, iv, iz]
                probs = q[iz, :, None] * eta_probs[None, :]

                promise = np.sum(probs * (d + v_next)) / (1.0 + tiny_params.r)
                assert promise == pytest.approx(v_current, abs=1e-10)
                assert np.all(d >= 0.0)
                assert np.all(v_next >= 0.0)
                assert np.all(np.isfinite(policy["p"][ik, iv, iz]))

                for izp, z_next in enumerate(z_grid):
                    for true_eta in range(len(eta_values)):
                        for report_eta in range(len(eta_values)):
                            if true_eta == report_eta:
                                continue
                            diversion = (
                                tiny_params.lambda_diversion
                                * (1.0 - tiny_params.tau)
                                * (
                                    profit_pre_tax(
                                        policy["k_next"][ik, iv, iz],
                                        z_next,
                                        eta_values[true_eta],
                                        tiny_params,
                                    )
                                    - profit_pre_tax(
                                        policy["k_next"][ik, iv, iz],
                                        z_next,
                                        eta_values[report_eta],
                                        tiny_params,
                                    )
                                )
                            )
                            assert (
                                d[izp, true_eta] + v_next[izp, true_eta] + 1e-10
                                >= d[izp, report_eta]
                                + v_next[izp, report_eta]
                                + diversion
                            )

    np.testing.assert_allclose(result["policy"]["value"], result["value"], atol=1e-7)


def test_le_and_mh_raise_on_empty_feasible_states(tiny_transition, tiny_params):
    bad_config = NikolovGridConfig(
        k_bounds=(1.0, 1.2),
        b_bounds=(0.0, 0.5),
        v_bounds=(0.0, 0.5),
        z_bounds=(1.0, 1.1),
        p_bounds=(0.0, 0.001),
        d_bounds=(0.0, 0.001),
        n_k=2,
        n_b=2,
        n_v=2,
        n_z=2,
        n_p=2,
        n_d=2,
        z_transition=tiny_transition,
    )

    with pytest.raises(ValueError, match="Nikolov LE grid has no feasible action"):
        solve_nikolov_le(bad_config, tiny_params)

    with pytest.raises(ValueError, match="Nikolov MH grid has no feasible action"):
        solve_nikolov_mh(bad_config, tiny_params)
