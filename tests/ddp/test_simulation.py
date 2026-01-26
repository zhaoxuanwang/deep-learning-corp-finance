"""
test_simulation.py

Unit tests for the simulation pipeline (simulation.py).
Verifies:
1. Integration: Parameter Sweep -> Solver -> Post-Processor works without crashing.
2. Consistency: Grid sizes match input (unless delta_rule is used).
3. Math: Derived moments (investment rate, leverage) match manual calculations.
"""

import pytest
import numpy as np
import sys
import os
from typing import Literal

# Ensure src is in path
sys.path.append(os.path.abspath('..'))

from src.ddp import simulation, DDPGridConfig
from src.economy.parameters import EconomicParams, ShockParams
from src.ddp.ddp_investment import InvestmentModelDDP
from src.ddp.ddp_debt import DebtModelDDP


# --- TEST 1: Investment Model Pipeline & Grid Consistency ---

@pytest.mark.parametrize("grid_type", ["power_grid", "log_linear", "delta_rule"])
def test_investment_pipeline_consistency(
        grid_type: Literal["power_grid", "log_linear", "delta_rule"]
):
    """
    Runs the full Investment Model pipeline.
    Checks if output dimensions match inputs and if moment calculations are correct.
    """
    # 1. Setup
    input_n_z = 5
    input_n_k = 40
    scenario_name = f"Test-{grid_type}"

    params = EconomicParams(r_rate=0.04, delta=0.06, theta=0.65)
    shock_params = ShockParams(sigma=0.15)
    grid_config = DDPGridConfig(z_size=input_n_z, k_size=input_n_k, grid_type=grid_type)

    # 2. Action: Run the Pipeline (Sweep -> Process)
    # Step A: Run Sweep (Returns Raw Tensors)
    raw_res = simulation.run_parameter_sweep(
        model_class=InvestmentModelDDP,
        base_params=params,
        scenarios={scenario_name: {}},  # Empty dict means no overrides
        solver_method="solve_invest_vfi",
        solver_kwargs={"max_iter": 50},  # Reduced iters for speed
        base_grid_config=grid_config,
        base_shock_params=shock_params
    )

    # Step B: Process Output (Returns Clean NumPy)
    clean_res = simulation.process_investment_output(raw_res)
    data = clean_res[scenario_name]  # Extract the specific scenario

    # 3. Dynamic Dimension Checking
    k_grid = data['grids']['k']
    actual_n_k = len(k_grid)

    # Check Z dimension (should always match input)
    assert data['grids']['z'].shape[0] == input_n_z

    # Check Internal Consistency
    # Policy shape must match (Nz, Actual_Nk)
    expected_shape = (input_n_z, actual_n_k)
    assert data['policy'].shape == expected_shape, \
        f"Policy shape mismatch! Expected {expected_shape}, got {data['policy'].shape}"

    # Check Fixed Grid Logic
    if grid_type != "delta_rule":
        assert actual_n_k == input_n_k, \
            f"For {grid_type}, output grid size ({actual_n_k}) should match input ({input_n_k})"

    # 4. Math Consistency Check (Manual Recalculation)
    policy = data['policy']
    invest_rate_model = data['moments']['invest_rate']

    # Manually calculate i/k = (k' - (1-delta)k) / k
    # Broadcasting: (Nz, Nk) - (Nk,)
    expected_investment = policy - (1 - params.delta) * k_grid

    with np.errstate(divide='ignore', invalid='ignore'):
        expected_rate = expected_investment / k_grid
        expected_rate = np.nan_to_num(expected_rate)

    np.testing.assert_allclose(
        invest_rate_model,
        expected_rate,
        rtol=1e-5,
        err_msg="Investment Rate moment calculation doesn't match manual verification!"
    )


# --- TEST 2: Debt Model Pipeline & Broadcasting Safety ---
@pytest.mark.parametrize("grid_type", ["power_grid", "log_linear", "delta_rule"])
def test_debt_pipeline_broadcasting(
        grid_type: Literal["power_grid", "log_linear", "delta_rule"]
):
    """
    Test for the Debt Model to ensure 3D array broadcasting works correctly
    across ALL grid types (including dynamic ones like delta_rule).
    """
    # 1. Setup
    input_n_z = 3
    input_n_k = 10
    input_n_b = 10
    scenario_name = f"Debt-Test-{grid_type}"

    params = EconomicParams()
    shock_params = ShockParams()
    grid_config = DDPGridConfig(z_size=input_n_z, k_size=input_n_k, b_size=input_n_b, grid_type=grid_type)

    # 2. Action: Run Pipeline
    raw_res = simulation.run_parameter_sweep(
        model_class=DebtModelDDP,
        base_params=params,
        scenarios={scenario_name: {}},
        solver_method="solve_risky_debt_vfi",
        solver_kwargs={"max_iter": 5},
        base_grid_config=grid_config,
        base_shock_params=shock_params
    )

    clean_res = simulation.process_debt_output(raw_res)
    data = clean_res[scenario_name]

    # 3. Dynamic Dimension Checking
    # Because 'delta_rule' might change k_size, we trust the output grid
    actual_k_grid = data['grids']['k']
    actual_n_k = len(actual_k_grid)

    # Construct expected shape based on ACTUAL output dimensions
    expected_shape = (input_n_z, actual_n_k, input_n_b)

    # Verify Value Function shape
    assert data['value'].shape == expected_shape, \
        f"Value shape mismatch! Expected {expected_shape}, got {data['value'].shape}"

    # Verify Derived Moments shape (Leverage, Risky Rate)
    lev_ratio = data['moments']['leverage_ratio']
    assert lev_ratio.shape == expected_shape
    assert data['prices']['risky_rate'].shape == expected_shape

    # 4. Check Input/Output Consistency
    # Only enforce strict input size matching if we aren't using the dynamic rule
    if grid_type != "delta_rule":
        assert actual_n_k == input_n_k, \
            f"For {grid_type}, k_size changed from {input_n_k} to {actual_n_k}!"

    # 5. Math Check: Leverage Ratio Logic
    # Verify b'/k calculation for a random point
    # We must pick a valid index within actual_n_k
    safe_k_idx = actual_n_k // 2
    idx = (1, safe_k_idx, 5)  # z=1, middle k, b=5

    chosen_b_next = data['policy']['b'][idx]  # From policy
    current_k = actual_k_grid[safe_k_idx]  # From grid

    model_leverage = lev_ratio[idx]
    manual_leverage = chosen_b_next / current_k

    # Use nan_to_num on manual calculation to match production code safety
    if np.isnan(manual_leverage): manual_leverage = 0.0

    assert np.isclose(model_leverage, manual_leverage, rtol=1e-5), \
        f"Leverage broadcasting failed at index {idx}"