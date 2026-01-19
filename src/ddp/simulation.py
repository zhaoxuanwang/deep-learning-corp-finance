"""
simulation.py

Executes model simulations and post-processes results.

This module acts as the computational engine of the project. It strictly separates
the "Solving" phase (heavy computation) from the "Analysis" phase (calculating
derived economic moments), ensuring modularity and reproducibility.
"""

import time
import logging
from dataclasses import replace
from typing import Dict, Any, Type, Optional

import numpy as np

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def run_parameter_sweep(
    model_class: Type,
    base_params: Any,
    scenarios: Dict[str, Dict[str, float]],
    solver_method: str = "solve",
    solver_kwargs: Optional[Dict[str, Any]] = None,
    base_grid_config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes a batch of scenarios (parameter sweep) for a given economic model.

    This function acts as a polymorphic runner. It can accept ANY model class
    (Investment or Debt) and run the specified solver method for every scenario.

    Args:
        model_class (Type): The class of the model to run (e.g., DebtModelDDP).
        base_params (Any): The baseline parameter dataclass.
        scenarios (Dict): A dictionary where keys are scenario names and values
            are dictionaries of parameter overrides (e.g., {"tax": 0.40}).
        solver_method (str): The name of the method to call on the model instance.
        solver_kwargs (Dict, optional): Additional arguments passed to the solver.
        base_grid_config (Any, optional): Grid config for DDP models (DDPGridConfig).

    Returns:
        Dict[str, Any]: Raw simulation results containing:
            {
                "ScenarioName": {
                    "params": UpdatedParams,
                    "solution": (v_star, policy, ...), # Raw Tensors/Arrays
                    "model": ModelInstance,
                    "duration": float
                },
                ...
            }
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    results = {}
    total_scenarios = len(scenarios)

    print(f"\n--- Starting Parameter Sweep for {model_class.__name__} ---")

    for i, (name, param_overrides) in enumerate(scenarios.items(), 1):
        start_time = time.time()
        print(f"[{i}/{total_scenarios}] Running Scenario: '{name}'...")

        # 1. Update Parameters (Immutable replacement to prevent side effects)
        scenario_params = replace(base_params, **param_overrides)

        # 2. Instantiate Model
        if base_grid_config is not None:
            model = model_class(scenario_params, base_grid_config)
        else:
            model = model_class(scenario_params)

        # 3. Solve (Reflection)
        if not hasattr(model, solver_method):
            raise AttributeError(f"Model {model_class.__name__} has no method '{solver_method}'")

        solve_func = getattr(model, solver_method)
        solution = solve_func(**solver_kwargs)

        duration = time.time() - start_time
        print(f"   > Completed in {duration:.2f} seconds.")

        # 4. Store Raw Results
        results[name] = {
            "params": scenario_params,
            "solution": solution,
            "model": model,
            "duration": duration
        }

    return results


def process_investment_output(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-processes raw output from the Basic Investment Model.

    Calculations:
    1. Investment Rate: i/k
    2. Financial Surplus: (Profit - Investment) / k
    3. External Financing: Max(0, Deficit) / k

    Args:
        raw_results (Dict): Output from run_parameter_sweep.

    Returns:
        Dict: Clean dictionary with NumPy arrays and derived moments.
    """
    print("\n--- Processing Investment Output ---")
    processed_data = {}

    for name, data in raw_results.items():
        model = data['model']
        # Unpack solution (Tuple: Value, Policy)
        v_star_tf, policy_k_tf = data['solution']

        # 1. Convert Tensors to NumPy
        k_grid = model.k_grid.numpy()
        z_grid = model.z_grid.numpy()
        policy_np = policy_k_tf.numpy()
        v_star_np = v_star_tf.numpy()

        # 2. Calculate Moments

        # A. Investment Rate: i/k = (k' - (1-delta)k) / k
        investment = policy_np - (1 - model.params.delta) * k_grid

        with np.errstate(divide='ignore', invalid='ignore'):
            invest_rate = investment / k_grid
            invest_rate = np.nan_to_num(invest_rate, nan=0.0, posinf=0.0, neginf=0.0)

        # B. Financial Surplus Ratio
        z_grid_col = z_grid.reshape(-1, 1)
        profit = (1 - model.params.tax) * z_grid_col * (k_grid ** model.params.theta)
        cash_flow = profit - investment

        with np.errstate(divide='ignore', invalid='ignore'):
            surplus_ratio = cash_flow / k_grid
            surplus_ratio = np.nan_to_num(surplus_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # C. External Financing Ratio
        # Definition: Absolute value of deficit, normalized by k. 0 if surplus is positive.
        ext_financing = np.maximum(0.0, -cash_flow)

        with np.errstate(divide='ignore', invalid='ignore'):
            ext_fin_ratio = ext_financing / k_grid
            ext_fin_ratio = np.nan_to_num(ext_fin_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        processed_data[name] = {
            "name": name,
            "params": data['params'],
            "grids": {"k": k_grid, "z": z_grid},
            "policy": policy_np,
            "value": v_star_np,
            "moments": {
                "invest_rate": invest_rate,
                "financial_surplus": surplus_ratio,
                "ext_financing": ext_fin_ratio
            }
        }

    return processed_data


def process_debt_output(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-processes raw output from the Debt Model.

    Computes 4 Key Moments:
    1. Risky Bond Yield: (1/Price) - 1
    2. Leverage Ratio: Chosen Debt / Current Assets (b'/k)
    3. Investment Rate: i/k
    4. Dividends: V - Beta * E[V_next] (Backing out flow of funds)

    Args:
        raw_results (Dict): Output from run_parameter_sweep.

    Returns:
        Dict: Clean dictionary with NumPy arrays and derived moments.
    """
    print("\n--- Processing Debt Output ---")
    processed_data = {}

    for name, data in raw_results.items():
        model = data['model']
        params = model.params

        # Unpack Tuple: (v_star, (policy_k, policy_b), q_star)
        v_star_tf, policies_tf, q_star_tf = data['solution']
        policy_k_tf, policy_b_tf = policies_tf

        # 1. Convert Tensors to NumPy
        k_grid = model.k_grid.numpy()
        b_grid = model.b_grid.numpy()
        z_grid = model.z_grid.numpy()
        prob_matrix = model.prob_matrix.numpy()
        beta = float(model.beta)  # Ensure scalar float

        v_star = v_star_tf.numpy()
        q_star = q_star_tf.numpy()
        pol_k = policy_k_tf.numpy()
        pol_b = policy_b_tf.numpy()

        nz, nk, nb = v_star.shape

        # --- MOMENT 1: Implied Yield ---
        # r = (1/q) - 1. Handle defaults (q ~ 0) safely.
        safe_q = np.maximum(q_star, 1e-4)
        risky_rate = (1.0 / safe_q) - 1.0

        # --- MOMENT 2: Future Leverage (b'/k) ---
        # Broadcast k (nk,) to (1, nk, 1) to divide pol_b (nz, nk, nb)
        k_grid_broad = k_grid.reshape(1, -1, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            leverage_ratio = pol_b / k_grid_broad
            leverage_ratio = np.nan_to_num(leverage_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # --- MOMENT 3: Investment Rate (i/k) ---
        # Investment = k_next - (1 - delta) * k_curr
        investment = pol_k - (1 - params.delta) * k_grid_broad

        with np.errstate(divide='ignore', invalid='ignore'):
            invest_rate = investment / k_grid_broad
            invest_rate = np.nan_to_num(invest_rate, nan=0.0, posinf=0.0, neginf=0.0)

        # --- MOMENT 4: Dividends (Distribution) ---
        # Formula: Div = V_curr - Beta * E[V_next | Chosen Policies]

        # Step A: Calculate Expected Value Surface E[V(z', k', b')] for ALL (k', b')
        # Flatten V to (nz, nk*nb) for matrix multiplication
        v_flat = v_star.reshape(nz, -1)
        # E[V] = TransitionMatrix(nz, nz) @ V(nz, nk*nb) -> (nz, nk*nb)
        ev_flat = np.dot(prob_matrix, v_flat)
        # Reshape back to cube: ev_grid[z_curr, k_next, b_next]
        ev_grid = ev_flat.reshape(nz, nk, nb)

        # Step B: Look up the specific EV for the chosen policies (k', b')
        # We need indices because ev_grid is indexed by (k_idx, b_idx), not values.
        # Since this is Discrete DP, pol_k values match k_grid values exactly.
        idx_k_next = np.searchsorted(k_grid, pol_k)
        idx_b_next = np.searchsorted(b_grid, pol_b)

        # Clip indices to prevent floating point edge case errors
        idx_k_next = np.clip(idx_k_next, 0, nk - 1)
        idx_b_next = np.clip(idx_b_next, 0, nb - 1)

        # Step C: Advanced Indexing to extract specific expected values
        # We want: chosen_ev[z, k, b] = ev_grid[z, idx_k_next[z,k,b], idx_b_next[z,k,b]]
        idx_z_curr = np.arange(nz)[:, None, None]  # Broadcast to (nz, 1, 1)
        ev_chosen = ev_grid[idx_z_curr, idx_k_next, idx_b_next]

        # Step D: Final Calculation
        dividends = v_star - (beta * ev_chosen)

        # Pack Results
        processed_data[name] = {
            "name": name,
            "params": data['params'],
            "grids": {"k": k_grid, "b": b_grid, "z": z_grid},
            "policy": {"k": pol_k, "b": pol_b},
            "prices": {"q": q_star, "risky_rate": risky_rate},
            "value": v_star,
            "moments": {
                "leverage_ratio": leverage_ratio,
                "invest_rate": invest_rate,
                "dividends": dividends
            }
        }

    return processed_data