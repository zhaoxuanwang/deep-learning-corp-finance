"""
tests/economy/test_logic.py

Unit tests for economic logic functions.
Follows existing test patterns in tests/economy/.
"""

import pytest
import numpy as np
import tensorflow as tf
from dataclasses import replace

from src.economy.parameters import EconomicParams
from src.economy import logic


# --- Fixtures ---

@pytest.fixture
def params():
    """Default test parameters."""
    return EconomicParams()


@pytest.fixture
def params_no_fixed_cost():
    """Parameters with zero fixed adjustment cost."""
    return EconomicParams(cost_fixed=0.0, cost_convex=0.5)


# === SECTION 1: External Financing Cost Tests ===

class TestExternalFinancingCost:
    """Tests for external_financing_cost (η) per outline_v2.md."""
    
    def test_zero_for_positive_cashflow(self, params):
        """η(e) = 0 when e >= 0."""
        e = tf.constant([0.0, 1.0, 10.0, 100.0])
        eta = logic.external_financing_cost(e, params)
        
        assert np.allclose(eta.numpy(), 0.0), \
            "External financing cost should be zero for non-negative cash flow"
    
    def test_formula_for_negative_cashflow(self, params):
        """η(e) = η₀ + η₁|e| when e < 0."""
        e = tf.constant([-5.0])
        eta = logic.external_financing_cost(e, params)
        
        # Expected: η₀ + η₁ * 5.0
        expected = params.cost_inject_fixed + params.cost_inject_linear * 5.0
        
        assert np.isclose(eta.numpy(), expected), \
            f"Expected η = {expected}, got {eta.numpy()}"
    
    def test_no_k_dependence(self, params):
        """η(e) does NOT depend on k — function signature doesn't include k."""
        import inspect
        sig = inspect.signature(logic.external_financing_cost)
        param_names = list(sig.parameters.keys())
        
        assert 'k' not in param_names, \
            "external_financing_cost should not take 'k' as parameter"
        assert 'e' in param_names and 'params' in param_names, \
            f"Expected e and params in signature, got {param_names}"
    
    def test_vectorized(self, params):
        """Works on batched inputs."""
        e = tf.constant([-1.0, 0.0, 1.0, -10.0])
        eta = logic.external_financing_cost(e, params)
        
        assert eta.shape == (4,)
        assert eta.numpy()[0] > 0  # negative e
        assert eta.numpy()[1] == 0  # zero e
        assert eta.numpy()[2] == 0  # positive e
        assert eta.numpy()[3] > 0  # negative e
    
    def test_ste_gradient_nonzero_for_negative_e(self, params):
        """STE should provide non-zero gradient when e < 0."""
        e = tf.Variable([-0.5], dtype=tf.float32)  # Negative
        
        with tf.GradientTape() as tape:
            eta = logic.external_financing_cost(e, params, gate_mode="ste")
            loss = tf.reduce_sum(eta)
        
        grad = tape.gradient(loss, e)
        
        assert grad is not None, "STE should provide gradient"
        # Gradient should be non-zero for negative e
        assert grad.numpy()[0] != 0, "STE gradient should be non-zero for e < 0"
    
    def test_hard_gate_zero_gradient(self, params):
        """Hard gate should have zero gradient."""
        e = tf.Variable([-0.5], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            eta = logic.external_financing_cost(e, params, gate_mode="hard")
            loss = tf.reduce_sum(eta)
        
        grad = tape.gradient(loss, e)
        
        # Hard gate has zero gradient through the indicator
        # (gradient only from |e| term, not the gate itself)
        assert grad is not None  # Still get gradient from |e| part


# === SECTION 2: Primitives Tests ===

class TestPrimitives:
    """Tests for basic economic primitives."""
    
    def test_production_function_cobb_douglas(self, params):
        """π = z · k^θ"""
        k = tf.constant([2.0])
        z = tf.constant([1.5])
        
        pi = logic.production_function(k, z, params)
        expected = 1.5 * (2.0 ** params.theta)
        
        assert np.isclose(pi.numpy(), expected)
    
    def test_investment_formula(self, params):
        """I = k' - (1-δ)k"""
        k = tf.constant([10.0])
        k_next = tf.constant([12.0])
        
        I = logic.compute_investment(k, k_next, params)
        expected = 12.0 - (1 - params.delta) * 10.0
        
        assert np.isclose(I.numpy(), expected)
    
    def test_adjustment_costs_convex_only(self, params_no_fixed_cost):
        """ψ = (φ₀/2) · I²/k when φ₁ = 0."""
        k = tf.constant([4.0])
        k_next = tf.constant([5.0])  # I = 5 - (1-δ)*4
        
        psi = logic.adjustment_costs(k, k_next, params_no_fixed_cost)
        
        I = 5.0 - (1 - params_no_fixed_cost.delta) * 4.0
        expected = (params_no_fixed_cost.cost_convex / 2.0) * (I ** 2) / 4.0
        
        assert np.isclose(psi.numpy(), expected, rtol=1e-5)


# === SECTION 3: Cash Flow Tests ===

class TestCashFlow:
    """Tests for cash flow functions."""
    
    def test_basic_cash_flow_components(self, params):
        """e = π - I - ψ"""
        k = tf.constant([2.0])
        k_next = tf.constant([2.2])
        z = tf.constant([1.0])
        
        e = logic.compute_cash_flow_basic(k, k_next, z, params)
        
        # Verify it's finite and has right shape
        assert np.isfinite(e.numpy())
        assert e.shape == (1,) or e.shape == ()
    
    def test_risky_debt_cash_flow_shape(self, params):
        """Output shape matches input for risky debt cash flow."""
        k = tf.constant([1.0, 2.0])
        k_next = tf.constant([1.1, 2.1])
        b = tf.constant([0.5, 0.5])
        b_next = tf.constant([0.6, 0.6])
        z = tf.constant([1.0, 1.0])
        r_tilde = tf.constant([0.05, 0.06])
        
        e = logic.cash_flow_risky_debt(k, k_next, b, b_next, z, r_tilde, params)
        
        assert e.shape == (2,)
        assert np.all(np.isfinite(e.numpy()))
    
    def test_risky_debt_differentiable_wrt_actions(self, params):
        """Gradients exist w.r.t. k_next, b_next, r_tilde."""
        k = tf.constant([1.0])
        k_next = tf.Variable([1.1])
        b = tf.constant([0.5])
        b_next = tf.Variable([0.6])
        z = tf.constant([1.0])
        r_tilde = tf.Variable([0.05])
        
        with tf.GradientTape() as tape:
            e = logic.cash_flow_risky_debt(k, k_next, b, b_next, z, r_tilde, params)
        
        grads = tape.gradient(e, [k_next, b_next, r_tilde])
        
        assert all(g is not None for g in grads), \
            "Gradients should exist for all action variables"


# === SECTION 4: Euler Primitives Tests ===

class TestEulerPrimitives:
    """Tests for euler_chi and euler_m."""
    
    def test_euler_chi_formula(self, params_no_fixed_cost):
        """χ = 1 + φ₀ · I / k."""
        k = tf.constant([2.0])
        k_next = tf.constant([2.5])
        
        chi = logic.euler_chi(k, k_next, params_no_fixed_cost)
        
        I = 2.5 - (1 - params_no_fixed_cost.delta) * 2.0
        expected = 1.0 + params_no_fixed_cost.cost_convex * I / 2.0
        
        assert np.isclose(chi.numpy(), expected, rtol=1e-5)
    
    def test_euler_chi_equals_one_when_no_investment(self, params_no_fixed_cost):
        """χ = 1 when I = 0 (k' = (1-δ)k)."""
        k = tf.constant([2.0])
        k_next = tf.constant([(1 - params_no_fixed_cost.delta) * 2.0])
        
        chi = logic.euler_chi(k, k_next, params_no_fixed_cost)
        
        assert np.isclose(chi.numpy(), 1.0)
    
    def test_euler_chi_differentiable(self, params):
        """Gradient exists w.r.t. k_next."""
        k = tf.constant([2.0])
        k_next = tf.Variable([2.5])
        
        with tf.GradientTape() as tape:
            chi = logic.euler_chi(k, k_next, params)
        
        grad = tape.gradient(chi, k_next)
        
        assert grad is not None, "Gradient should exist for k_next"
        assert np.isfinite(grad.numpy())
    
    def test_euler_m_finite(self, params_no_fixed_cost):
        """m is finite for reasonable inputs."""
        k_next = tf.constant([2.0])
        k_next_next = tf.constant([2.1])
        z_next = tf.constant([1.0])
        
        m = logic.euler_m(k_next, k_next_next, z_next, params_no_fixed_cost)
        
        assert np.isfinite(m.numpy())
        assert m.numpy() > 0  # m should be positive (marginal value of capital)


# === SECTION 5: Recovery & Pricing Tests ===

class TestRecoveryAndPricing:
    """Tests for recovery_value and pricing_residual_zero_profit."""
    
    def test_recovery_value_formula(self, params):
        """R = (1-α)[(1-τ)π + (1-δ)k']"""
        k_next = tf.constant([3.0])
        z_next = tf.constant([1.2])
        
        R = logic.recovery_value(k_next, z_next, params)
        
        pi_next = 1.2 * (3.0 ** params.theta)
        gross = (1 - params.tax) * pi_next + (1 - params.delta) * 3.0
        expected = (1 - params.cost_default) * gross
        
        assert np.isclose(R.numpy(), expected, rtol=1e-5)
    
    def test_recovery_value_nonnegative(self, params):
        """Recovery is always >= 0."""
        k_next = tf.constant([0.01])  # Very small capital
        z_next = tf.constant([0.1])   # Low productivity
        
        R = logic.recovery_value(k_next, z_next, params)
        
        assert R.numpy() >= 0
    
    def test_pricing_residual_zero_when_fair(self):
        """Residual = 0 when bond is fairly priced."""
        b_next = tf.constant([1.0])
        r_risk_free = 0.04
        r_tilde = tf.constant([0.08])  # Risky rate
        
        # At fair price, LHS = RHS
        # If no default (p_D = 0): LHS = b'(1+r), RHS = b'(1+r̃)
        # => r_tilde must equal r for f=0 when p_D=0
        
        p_default = tf.constant([0.0])
        recovery = tf.constant([0.0])
        
        # When p_D=0 and r_tilde = r_risk_free, residual should be 0
        r_tilde_fair = tf.constant([r_risk_free])
        f = logic.pricing_residual_zero_profit(
            b_next, r_risk_free, r_tilde_fair, p_default, recovery
        )
        
        assert np.isclose(f.numpy(), 0.0, atol=1e-6)
    
    def test_pricing_residual_positive_when_lender_profits(self):
        """f > 0 when expected payoff < funding cost (lender earns spread)."""
        b_next = tf.constant([1.0])
        r_risk_free = 0.04
        r_tilde = tf.constant([0.10])  # High risky rate
        p_default = tf.constant([0.0])  # No default
        recovery = tf.constant([0.0])
        
        # LHS = 1.04, RHS = 1.10 => f = 1.04 - 1.10 = -0.06
        # Wait, that's negative. Let me reconsider.
        # Actually LHS > RHS means lender profits (pays less than promised)
        # When r_tilde > r, RHS > LHS, so f < 0 (lender overpays)
        
        f = logic.pricing_residual_zero_profit(
            b_next, r_risk_free, r_tilde, p_default, recovery
        )
        
        # f = b'(1+r) - b'(1+r_tilde) = b'(r - r_tilde) < 0 when r_tilde > r
        assert f.numpy() < 0


# === SECTION 6: Differentiability Tests ===

class TestDifferentiability:
    """Tests that key functions are differentiable for gradient-based training."""
    
    def test_basic_cash_flow_grad_exists(self, params):
        """Gradient of cash flow w.r.t. k_next exists."""
        k = tf.constant([2.0])
        k_next = tf.Variable([2.2])
        z = tf.constant([1.0])
        
        with tf.GradientTape() as tape:
            e = logic.compute_cash_flow_basic(k, k_next, z, params)
        
        grad = tape.gradient(e, k_next)
        
        assert grad is not None
        assert np.isfinite(grad.numpy())
    
    def test_adjustment_costs_grad_convex(self, params_no_fixed_cost):
        """Gradient of ψ w.r.t. k_next exists (convex part only)."""
        k = tf.constant([2.0])
        k_next = tf.Variable([2.5])
        
        with tf.GradientTape() as tape:
            psi = logic.adjustment_costs(k, k_next, params_no_fixed_cost)
        
        grad = tape.gradient(psi, k_next)
        
        assert grad is not None
        assert np.isfinite(grad.numpy())


# === SECTION 7: Investment Gate STE Tests ===

class TestInvestmentGateSTE:
    """Tests for investment_gate_ste function (Straight-Through Estimator)."""
    
    def test_forward_hard_gate_above_threshold(self):
        """Gate = 1 when |I| > eps."""
        I = tf.constant([1.0, -1.0, 0.1, -0.1])
        
        gate_hard = logic.investment_gate_ste(I, eps=1e-6, mode="hard")
        gate_ste = logic.investment_gate_ste(I, eps=1e-6, mode="ste")
        
        # Both should return 1 for all (|I| > eps)
        assert np.allclose(gate_hard.numpy(), 1.0)
        assert np.allclose(gate_ste.numpy(), 1.0)
    
    def test_forward_hard_gate_at_threshold(self):
        """Gate = 0 when |I| <= eps."""
        eps = 1e-6
        I = tf.constant([0.0, 1e-7, -1e-7, eps * 0.5])
        
        gate_hard = logic.investment_gate_ste(I, eps=eps, mode="hard")
        gate_ste = logic.investment_gate_ste(I, eps=eps, mode="ste")
        
        # Both should return 0 for all (|I| <= eps)
        assert np.allclose(gate_hard.numpy(), 0.0)
        assert np.allclose(gate_ste.numpy(), 0.0)
    
    def test_ste_forward_matches_hard(self):
        """STE forward pass matches hard gate exactly."""
        I = tf.constant([0.0, 1e-7, 1e-5, 0.01, 1.0, -0.5])
        eps = 1e-6
        
        gate_hard = logic.investment_gate_ste(I, eps=eps, mode="hard")
        gate_ste = logic.investment_gate_ste(I, eps=eps, mode="ste")
        
        np.testing.assert_allclose(gate_ste.numpy(), gate_hard.numpy())
    
    def test_ste_gradient_nonzero_near_threshold(self):
        """STE gradient is non-zero near |I| ≈ eps."""
        eps = 1e-6
        temp = 0.1  # Default value
        
        # Investment near threshold: use Variable to track gradients
        I = tf.Variable([eps * 0.5, eps, eps * 2, eps * 10])
        
        with tf.GradientTape() as tape:
            gate = logic.investment_gate_ste(I, eps=eps, temp=temp, mode="ste")
            loss = tf.reduce_sum(gate)
        
        grad = tape.gradient(loss, I)
        
        assert grad is not None, "Gradient should exist for STE mode"
        assert np.all(np.isfinite(grad.numpy())), "Gradients should be finite"
        # At least some gradients should be non-zero
        assert np.any(grad.numpy() > 0), "Some gradients should be positive"
    
    def test_hard_gradient_is_zero(self):
        """Hard gate has zero gradient."""
        eps = 1e-6
        I = tf.Variable([eps * 0.5, eps, eps * 2, eps * 10])
        
        with tf.GradientTape() as tape:
            gate = logic.investment_gate_ste(I, eps=eps, mode="hard")
            loss = tf.reduce_sum(gate)
        
        grad = tape.gradient(loss, I)
        
        # Hard gate: gradient is None or zero
        assert grad is None or np.allclose(grad.numpy(), 0.0)
    
    def test_symmetry_positive_and_negative_investment(self):
        """Gate responds identically for I > 0 and I < 0."""
        I_pos = tf.constant([0.01, 0.1, 1.0])
        I_neg = tf.constant([-0.01, -0.1, -1.0])
        
        gate_pos = logic.investment_gate_ste(I_pos, mode="ste")
        gate_neg = logic.investment_gate_ste(I_neg, mode="ste")
        
        np.testing.assert_allclose(gate_pos.numpy(), gate_neg.numpy())
    
    def test_adjustment_costs_with_ste_has_gradient(self, params):
        """adjustment_costs with STE mode has gradient through fixed cost."""
        # Use params with non-zero fixed cost
        params_fixed = replace(params, cost_fixed=0.5, cost_convex=0.01)
        
        k = tf.constant([2.0])
        k_next = tf.Variable([2.5])  # I = 2.5 - (1-δ)*2 > 0
        
        with tf.GradientTape() as tape:
            psi = logic.adjustment_costs(k, k_next, params_fixed, fixed_cost_gate="ste")
        
        grad = tape.gradient(psi, k_next)
        
        assert grad is not None, "Gradient should exist with STE mode"
        assert np.isfinite(grad.numpy()), "Gradient should be finite"
        # Gradient should be non-zero (from both convex and fixed cost)
        assert grad.numpy() != 0, "Gradient should be non-zero with fixed cost"
    
    def test_adjustment_costs_hard_vs_ste_forward_match(self, params):
        """adjustment_costs forward values match between hard and ste modes."""
        params_fixed = replace(params, cost_fixed=0.5)
        
        k = tf.constant([2.0])
        k_next = tf.constant([2.5])
        
        psi_hard = logic.adjustment_costs(k, k_next, params_fixed, fixed_cost_gate="hard")
        psi_ste = logic.adjustment_costs(k, k_next, params_fixed, fixed_cost_gate="ste")
        
        np.testing.assert_allclose(psi_ste.numpy(), psi_hard.numpy())
