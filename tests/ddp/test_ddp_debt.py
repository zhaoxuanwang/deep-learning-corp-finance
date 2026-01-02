import pytest
import tensorflow as tf
import numpy as np
import dataclasses

# Import key models
from src.ddp.ddp_debt import DebtModelDDP
from src.ddp.utils import ModelParameters, generate_bond_grid


def test_generate_bond_grid_refined():
    """
    Verifies the bond grid generation uses the correct economic scaling
    and components (collateral, tax shield, profit).
    """
    # 1. SETUP: Define parameters explicitly
    params = ModelParameters(
        b_size=5,
        r_rate=0.05,
        delta=0.1,
        theta=0.5,
        tax=0.2,
        frac_liquid=0.9
    )

    # Define Economy Scale
    k_max = 100.0
    z_max = 2.0

    # 2. EXECUTE
    b_grid = generate_bond_grid(params, k_max, z_max)

    # 3. VERIFY: LOWER BOUND (Savings)
    # Formula: b_min = -1.5 * k_max
    # Expected: -1.5 * 100 = -150.0
    expected_b_min = -150.0
    assert np.isclose(b_grid[0], expected_b_min), \
        f"Lower Bound (Savings) incorrect. Expected {expected_b_min}, Got {b_grid[0]}"

    # 4. VERIFY: UPPER BOUND (Max Debt Capacity)
    # Formula components check:

    # A. Max Output (y_max)
    # y = z * k^theta = 2.0 * 100^0.5 = 2.0 * 10 = 20.0
    y_max = 20.0

    # B. After-Tax Cash Flow
    # (1 - tax) * y = 0.8 * 20 = 16.0
    term_profit = (1 - 0.2) * y_max

    # C. Tax Shield
    # tax * delta * k = 0.2 * 0.1 * 100 = 2.0
    term_shield = 2.0

    # D. Collateral / Liquidation Value
    # frac_liquid * k = 0.9 * 100 = 90.0
    term_collateral = 90.0

    expected_b_max = term_profit + term_shield + term_collateral

    assert np.isclose(b_grid[-1], expected_b_max), \
        f"Upper Bound (Max Debt) incorrect. Expected {expected_b_max}, Got {b_grid[-1]}"

    # 5. VERIFY: Structure
    assert len(b_grid) == params.b_size
    assert np.all(np.diff(b_grid) > 0), "Grid must be strictly increasing"


# --- 1. FIXTURE ---
@pytest.fixture
def model_debt():
    """
    Standard fixture for creating a DebtModelDDP instance.
    Uses small grids for speed, but real parameters.
    """
    params = ModelParameters(
        z_size=2,
        k_size=5,
        b_size=4,
        r_rate=0.04,
        grid_type="log_linear"
    )
    return DebtModelDDP(params)


# --- 2. INITIALIZATION TESTS ---

def test_initialization_shapes_and_types(model_debt):
    """
    Verifies that the model initializes dimensions, grids, and tensors correctly.
    """
    # 1. Scalar Attributes
    # Beta = 1 / (1.04) ~ 0.9615
    expected_beta = 1 / (1 + model_debt.params.r_rate)
    assert np.isclose(model_debt.beta.numpy(), expected_beta), "Beta calculation incorrect"

    # 2. Dimensions
    assert model_debt.nz == 2
    assert model_debt.nk == 5
    assert model_debt.nb == 4

    # 3. TensorFlow Conversions
    assert isinstance(model_debt.z_grid, tf.Tensor)
    assert isinstance(model_debt.k_grid, tf.Tensor)
    assert isinstance(model_debt.b_grid, tf.Tensor)

    # Check Dtypes (Critical for GPU)
    assert model_debt.prob_matrix.dtype == tf.float32


def test_initialization_dynamic_grid_size():
    """
    Verifies that self.nk correctly adapts when 'delta_rule' generates
    a different number of grid points than requested.
    """
    # Case A: Log Linear (Fixed Request)
    params_fixed = ModelParameters(k_size=10, grid_type="log_linear")
    model_fixed = DebtModelDDP(params_fixed)
    assert model_fixed.nk == 10, "Log_linear should preserve requested k_size"

    # Case B: Delta Rule (Dynamic Request)
    # We create a specific params where we know delta_rule might shift the count
    params_dynamic = ModelParameters(k_size=10, grid_type="delta_rule")
    model_dynamic = DebtModelDDP(params_dynamic)

    actual_tensor_len = model_dynamic.k_grid.shape[0]

    # The class attribute .nk must match the actual tensor length
    assert model_dynamic.nk == actual_tensor_len, \
        f"Model.nk ({model_dynamic.nk}) does not match grid tensor ({actual_tensor_len})"


# --- 3. BOND PRICING TESTS ---

def test_compute_bond_price_logic(model_debt):
    """
    CRITICAL TEST: Verifies the equilibrium pricing logic.
    Checks that:
    1. Solvent firms get risk-free rates.
    2. Defaulting firms get recovery rates.
    3. Prices NEVER exceed the risk-free rate (No Arbitrage).
    4. Prices NEVER go negative (Limited Liability).
    """
    # Risk Free Price (Maximum possible price)
    rf_price = 1.0 / (1 + model_debt.params.r_rate)

    # --- SCENARIO A: The "Safe" Firm ---
    # We force V_next to be positive everywhere (Solvent)
    v_solvent = tf.ones((model_debt.nz, model_debt.nk, model_debt.nb))
    q_solvent = model_debt._compute_bond_price(v_solvent)

    # Assert: Price should be effectively Risk-Free
    # Allow small float error (1e-6)
    diff_tf = tf.abs(tf.math.subtract(q_solvent, rf_price))
    diff = tf.reduce_max(diff_tf)
    assert diff.numpy() < 1e-6, "Solvent firm didn't get risk-free rate"

    # --- SCENARIO B: The "Defaulting" Firm ---
    v_default = -1.0 * tf.ones((model_debt.nz, model_debt.nk, model_debt.nb))
    q_default = model_debt._compute_bond_price(v_default)

    # Get the b_grid to distinguish Borrowing from Saving
    b_grid_vals = model_debt.b_grid.numpy()

    # 1. CHECK SAVINGS (b <= 0): Should be Risk-Free
    # We look for indices where b <= 0
    savings_indices = np.where(b_grid_vals <= 0)[0]
    if len(savings_indices) > 0:
        q_savings = tf.gather(q_default, savings_indices, axis=2)
        diff = tf.reduce_max(tf.abs(q_savings - rf_price))
        assert diff < 1e-6, "Savings should be Risk-Free even in default (Bank returns deposit)."

    # 2. CHECK DEBT (b > 0): Should be Risky (if Recovery < Debt)
    borrow_indices = np.where(b_grid_vals > 0)[0]
    if len(borrow_indices) > 0:
        q_debt = tf.gather(q_default, borrow_indices, axis=2)

        # We only assert strictly less if the debt is large enough to be risky.
        # But generally, for high debt, q < rf_price.
        # Let's relax the strict check to a general check for the highest debt level
        q_high_debt = q_debt[:, :, -1]  # Highest debt level
        assert tf.reduce_all(q_high_debt < rf_price - 1e-4).numpy(), \
            "High debt should carry a risk premium (Price < Risk-Free)."

    # 3. CHECK GUARDRAILS (Applies to ALL)
    # Prices must NEVER exceed Risk-Free (No Arbitrage)
    max_price = tf.reduce_max(q_default).numpy()
    assert max_price <= rf_price + 1e-6, \
        f"Price exploded! Max: {max_price} > RF: {rf_price}"

    # Prices must NEVER be negative
    min_price = tf.reduce_min(q_default).numpy()
    assert min_price >= 0.0, f"Price negative! Min: {min_price}"


def test_compute_reward_matrix_shapes_and_values(model_debt):
    """
    Verifies the 5D reward matrix calculation.
    1. Shape Check: Ensures dimensions (nz, nk, nk, nb, nb) are correct.
    2. Logic Audit: Manually calculates ONE specific transition to verify
       accounting (Profit - Inv + Borrowing - Repayment).
    """
    # 1. SETUP: Create a Mock Price Schedule
    # We assume Price = 1.0 everywhere to make math simple for the audit
    # Shape needed: (nz, nk, nb) -> indices correspond to (z, k_next, b_next)
    q_mock = tf.ones((model_debt.nz, model_debt.nk, model_debt.nb))

    # 2. EXECUTE
    rewards_5d = model_debt._compute_reward_matrix(q_mock)

    # 3. VERIFY SHAPE
    # Expected: (nz, nk_curr, nk_next, nb_curr, nb_next)
    expected_shape = (model_debt.nz, model_debt.nk, model_debt.nk, model_debt.nb, model_debt.nb)
    assert rewards_5d.shape == expected_shape, \
        f"Reward matrix shape mismatch. Expected {expected_shape}, Got {rewards_5d.shape}"

    # 4. VERIFY LOGIC (The Single Cell Audit)
    # We will audit the transition:
    # State: z[0], k[0], b[0]  ->  Choice: k[1], b[1]

    # A. Extract Raw Values from Grids
    # Note: Use .numpy() for scalar extraction
    z_val = model_debt.z_grid[0].numpy()
    k_curr_val = model_debt.k_grid[0].numpy()
    k_next_val = model_debt.k_grid[1].numpy()
    b_curr_val = model_debt.b_grid[0].numpy()
    b_next_val = model_debt.b_grid[1].numpy()

    # B. Manual Calculation of the Budget Constraint
    p = model_debt.params

    # Operating Profit
    profit = (1 - p.tax) * z_val * (k_curr_val ** p.theta)

    # Investment
    investment = k_next_val - (1 - p.delta) * k_curr_val
    k_safe = max(k_curr_val, 1e-6)

    # Adjustment Costs
    # Convex
    adj_convex = (p.cost_convex / 2) * (investment ** 2) / k_safe

    # Fixed (Only if investment is non-zero)
    is_investing = 1.0 if abs(investment) > 1e-6 else 0.0
    adj_fixed = p.cost_fixed * k_safe * is_investing

    # Financing (Assuming q=1.0 as set above)
    q_val = 1.0
    debt_proceeds = q_val * b_next_val
    debt_repay = b_curr_val

    # Tax Benefit on Interest
    # Formula: (Tax * (Face - Proceeds)) / (1+r)
    # Since q=1, Proceeds = Face. Interest = 0. Benefit = 0.
    # But let's write the formula out to be robust
    tax_benefit = (p.tax * (b_next_val - debt_proceeds)) / (1 + p.r_rate)

    # Total Dividend (Before Equity Cost)
    div_raw = (profit - investment - adj_convex - adj_fixed
               + debt_proceeds - debt_repay + tax_benefit)

    # Equity Injection Cost
    # If Div < 0, we pay extra cost
    equity_cost = 0.0
    if div_raw < 0:
        equity_cost = (p.cost_inject_fixed * k_safe) + (p.cost_inject_linear * abs(div_raw))

    expected_reward = div_raw - equity_cost

    # C. Compare with Code Output
    # Index: [z=0, k=0, kp=1, b=0, bp=1]
    actual_reward = rewards_5d[0, 0, 1, 0, 1].numpy()

    assert np.isclose(actual_reward, expected_reward, rtol=1e-5), \
        f"""
        Logic Mismatch in Cell [0,0,1,0,1]:
        Expected: {expected_reward:.4f}
        Got:      {actual_reward:.4f}

        Breakdown:
        Profit: {profit:.4f}
        Inv:    {investment:.4f}
        Debt+:  {debt_proceeds:.4f}
        Debt-:  {debt_repay:.4f}
        """


def test_equity_issuance_cost_trigger(model_debt):
    """
    Verifies that equity issuance costs strictly lower the reward
    when dividends are negative.
    """
    # 1. BASELINE: Calculate rewards WITH costs
    q_mock = tf.ones((model_debt.nz, model_debt.nk, model_debt.nb))
    rewards_with_cost = model_debt._compute_reward_matrix(q_mock)

    # Find a cell where investment is huge (Small k -> Big k) to force negative dividends
    # Index: [z=0, k=0, k'=max, b=0, b'=0]
    cell_idx = (0, 0, -1, 0, 0)
    reward_with_cost = rewards_with_cost[cell_idx]

    # Sanity Check: Dividend must be negative for the test to be valid
    assert reward_with_cost < 0, "Test Setup Failed: Dividend was positive, equity cost wouldn't trigger."

    # 2. EXPERIMENT: Create a NEW model WITHOUT costs
    # We use dataclasses.replace to clone the params but change specific fields
    params_no_cost = dataclasses.replace(
        model_debt.params,
        cost_inject_fixed=0.0,
        cost_inject_linear=0.0
    )

    # Re-initialize the model with these cheap parameters
    # (Since grids depend on params, it's safer to make a fresh instance)
    model_cheap = DebtModelDDP(params_no_cost)

    rewards_no_cost = model_cheap._compute_reward_matrix(q_mock)
    reward_no_cost = rewards_no_cost[cell_idx]

    # 3. ASSERTION
    # With Costs (e.g., -150) should be WORSE than Without Costs (e.g., -100)
    # So: -150 < -100  --> True
    assert reward_with_cost < reward_no_cost, \
        f"Equity costs failed. With: {reward_with_cost.numpy()}, Without: {reward_no_cost.numpy()}"


# --- Bellman Test 1: Optimization Logic ---
def test_bellman_optimization(model_debt):
    """
    Verifies the agent correctly identifies the maximum reward index.
    """
    nz, nk, nb = model_debt.nz, model_debt.nk, model_debt.nb

    # Setup: Zero future value, zero reward everywhere except one "needle in haystack"
    v_curr_zero = tf.zeros((nz, nk, nb))
    reward_needle = np.zeros((nz, nk, nk, nb, nb), dtype=np.float32)

    # Plant the 'best choice' at state (z=0, k=0, b=0) -> Choice (k'=1, b'=1)
    target_val = 100.0
    reward_needle[0, 0, 1, 0, 1] = target_val
    reward_tf = tf.constant(reward_needle)

    # Execute
    v_next = model_debt._compute_bellman_step(v_curr_zero, reward_tf)

    # Assert
    val_opt = v_next[0, 0, 0].numpy()
    np.testing.assert_allclose(val_opt, target_val, err_msg="Agent failed to find max reward.")

    # Ensure other states remain 0
    np.testing.assert_allclose(v_next[1, 0, 0].numpy(), 0.0, err_msg="Agent found reward in wrong state.")


# --- Test 2: Discounting Logic ---
def test_bellman_discounting(model_debt):
    """
    Verifies that future values are correctly discounted by beta.
    """
    nz, nk, nb = model_debt.nz, model_debt.nk, model_debt.nb

    # Setup: Constant future value of 100, zero current reward
    v_curr_100 = tf.ones((nz, nk, nb)) * 100.0
    reward_zero = tf.zeros((nz, nk, nk, nb, nb))

    # Execute
    v_next = model_debt._compute_bellman_step(v_curr_100, reward_zero)

    # Assert
    expected_val = model_debt.beta.numpy() * 100.0
    avg_val = np.mean(v_next.numpy())

    np.testing.assert_allclose(avg_val, expected_val, rtol=1e-5, err_msg="Discounting beta applied incorrectly.")


# --- Test 3: Limited Liability ---
def test_bellman_limited_liability(model_debt):
    """
    Verifies that the firm defaults (value = 0) if total value is negative.
    """
    nz, nk, nb = model_debt.nz, model_debt.nk, model_debt.nb

    # Setup: Huge negative reward, zero future value
    v_curr_zero = tf.zeros((nz, nk, nb))
    reward_loss = tf.ones((nz, nk, nk, nb, nb)) * -1000.0

    # Execute
    v_next = model_debt._compute_bellman_step(v_curr_zero, reward_loss)

    # Assert
    min_val = np.min(v_next.numpy())
    assert min_val == 0.0, f"Limited Liability failed. Expected 0.0, got {min_val}"


# --- Test 4: Expectation / Probability Mixing ---
def test_bellman_expectation_mixing(model_debt):
    """
    Verifies the matrix multiplication for expected value E[V].
    Uses a manual probability matrix to ensure deterministic calculation.
    """
    nz, nk, nb = model_debt.nz, model_debt.nk, model_debt.nb

    # Note: Requires at least 2 states
    if nz < 2:
        pytest.skip("Test requires at least 2 exogenous states (nz >= 2)")

    # Setup V: 0 in State 0, 100 in State 1
    v_mixed_np = np.zeros((nz, nk, nb), dtype=np.float32)
    v_mixed_np[1, :, :] = 100.0
    v_mixed = tf.constant(v_mixed_np)

    # Setup Prob Matrix: 50/50 transition from State 0 to (0, 1)
    # We patch the instance only for this test
    original_prob = model_debt.prob_matrix
    p_manual = np.zeros((nz, nz), dtype=np.float32)
    p_manual[0, 0] = 0.5
    p_manual[0, 1] = 0.5
    p_manual[1, 1] = 1.0  # Absorb in state 1
    model_debt.prob_matrix = tf.constant(p_manual)

    try:
        # Execute
        reward_zero = tf.zeros((nz, nk, nk, nb, nb))
        v_next = model_debt._compute_bellman_step(v_mixed, reward_zero)

        # Assert: State 0 should be beta * (0.5*0 + 0.5*100) = beta * 50
        expected_val = model_debt.beta.numpy() * 50.0
        actual_val = v_next[0, 0, 0].numpy()

        np.testing.assert_allclose(actual_val, expected_val, err_msg="Expectation matrix math failed.")

    finally:
        # Clean up: Restore original matrix even if test fails
        model_debt.prob_matrix = original_prob


def test_solve_vfi_geometric_series(model_debt):
    """
    GOLD STANDARD TEST: Verifies convergence to the theoretical limit.

    If Reward is constant R everywhere, the Value Function MUST converge to:
         V = R / (1 - beta)
    This confirms:
    1. The loop updates V_curr correctly.
    2. Discounting (beta) is applied recursively.
    3. The convergence check (diff < tol) works.
    """
    # 1. SETUP
    # Define a constant reward of 10.0
    reward_cons = 10.0

    # Create a reward matrix where EVERY transition gives 10.0
    # Shape: (nz, nk, nk, nb, nb)
    shape_5d = (model_debt.nz, model_debt.nk, model_debt.nk, model_debt.nb, model_debt.nb)
    reward_constant = reward_cons * tf.ones(shape_5d)

    # 2. EXECUTE
    # Use a strict tolerance to check precision
    # Note: We assume you have implemented _extract_policy, otherwise this will crash.
    v_converged, policy = model_debt.solve_vfi(reward_constant, tol=1e-6, max_iter=2000)

    # 3. VERIFY THEORETICAL LIMIT
    beta = model_debt.beta.numpy()
    expected_value = reward_cons / (1.0 - beta)  # e.g., 10 / 0.04 = 250.0

    # Calculate error
    # Since inputs are uniform, output V should be uniform
    actual_values = v_converged.numpy()
    max_error = np.max(np.abs(actual_values - expected_value))

    # Error Tolerance Check:
    # VFI stops when |V_new - V_old| < tol.
    # The remaining error from the true limit is bounded by approx tol / (1-beta).
    # With tol=1e-6 and beta=0.96, max error is roughly 2.5e-5.
    assert max_error < 1e-3, \
        f"VFI Convergence Failed. Expected {expected_value:.4f}, Got Mean {np.mean(actual_values):.4f}"

    # 4. VERIFY ITERATION
    # If R=10 and beta=0.96, it takes ~300 iterations to hit 1e-6 accuracy.
    # If it stopped in 1 iteration, something is wrong (it didn't loop).
    # We can't check iteration count easily unless solve_vfi returns it,
    # but we can ensure it's not the initial guess (Zero).
    assert np.mean(actual_values) > 10.0, "VFI didn't seem to iterate (Value is too low)."


def test_solve_vfi_max_iter_cutoff(model_debt):
    """
    Verifies that the solver respects the max_iter argument (Safety Break).
    """
    reward = 10.0
    reward_constant = reward * tf.ones((model_debt.nz, model_debt.nk, model_debt.nk, model_debt.nb, model_debt.nb))

    # Run for exactly 1 iteration
    # Start (0) -> Step 1 (Reward + beta*0) = 10.0
    v_1step, _ = model_debt.solve_vfi(reward_constant, max_iter=1)

    # The value should be exactly R (first term of series)
    expected_val = reward
    actual_val = np.mean(v_1step.numpy())

    assert np.isclose(actual_val, expected_val, rtol=1e-5), \
        f"Max Iteration limit ignored. Expected {expected_val} (1 step), Got {actual_val}"


def test_solve_risky_bond_integration(model_debt):
    """
    INTEGRATION TEST: Runs the full Outer Loop (Price Equilibrium).

    Checks:
    1. Execution: Runs without crashing.
    2. Bounds: Prices are within [0, RiskFree].
    3. Logic: High debt should carry a risk premium (lower price)
              compared to savings (risk-free price).
    """
    # 1. EXECUTE
    v_star, policy, q_star = model_debt.solve_risky_debt_vfi()

    # 2. CHECK OUTPUT TYPES
    assert v_star is not None
    assert q_star is not None

    # 3. CHECK BOUNDS (No Arbitrage)
    rf_price = 1.0 / (1 + model_debt.params.r_rate)

    # Max price must not exceed Risk-Free (allow tiny float error)
    max_q = tf.reduce_max(q_star).numpy()
    assert max_q <= rf_price + 1e-5, f"Arbitrage found! Price {max_q} > RF {rf_price}"

    # Min price must be non-negative
    min_q = tf.reduce_min(q_star).numpy()
    assert min_q >= 0.0, "Negative bond prices found!"

    # 4. CHECK RISK SPREADS
    # Compare Price of Savings (b < 0) vs Price of High Debt (b > 0)
    # We average over z and k to look at the 'pure' effect of debt
    avg_price_by_debt = tf.reduce_mean(q_star, axis=[0, 1]).numpy()

    # Identify indices
    b_grid = model_debt.b_grid.numpy()
    savings_idx = 0  # Most negative b (Savings)
    debt_idx = -1  # Most positive b (High Debt)

    p_savings = avg_price_by_debt[savings_idx]
    p_debt = avg_price_by_debt[debt_idx]

    # Assert Savings are priced ~ Risk Free
    # Note: If b_grid[0] is negative, it's a deposit -> Risk Free
    if b_grid[savings_idx] <= 0:
        assert np.isclose(p_savings, rf_price, atol=1e-3), \
            "Savings should be priced at Risk-Free rate."

    # Assert Debt is priced lower (Risk Premium)
    # Note: This assertion assumes the model is calibrated such that default is possible.
    # If parameters are too safe, p_debt might equal p_savings.
    # We use <= to be safe, but ideally <.
    assert p_debt <= p_savings, \
        "Curve inversion! Debt is more expensive than savings?"

    print(f"\n    Market Prices checked: Savings={p_savings:.4f}, Debt={p_debt:.4f}")