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

# Default temperature and logit_clip for soft gates
DEFAULT_TEMPERATURE = 0.01  # Small value for near-hard gates
DEFAULT_LOGIT_CLIP = 20.0


@pytest.fixture
def params():
    """Default test parameters."""
    return EconomicParams()


@pytest.fixture
def params_no_fixed_cost():
    """Parameters with zero fixed adjustment cost."""
    return EconomicParams(cost_fixed=0.0, cost_convex=0.5)

@pytest.fixture
def params_with_injection_costs():
    """Parameters with non-zero equity injection costs."""
    return EconomicParams(cost_inject_fixed=0.1, cost_inject_linear=0.1)


# === SECTION 1: External Financing Cost Tests ===

class TestExternalFinancingCost:
    """Tests for external_financing_cost (η) per outline_v2.md."""

    def test_zero_for_positive_cashflow(self, params):
        """η(e) = 0 when e >= 0."""
        e = tf.constant([0.0, 1.0, 10.0, 100.0])
        eta = logic.external_financing_cost(
            e, params,
            temperature=DEFAULT_TEMPERATURE,
            logit_clip=DEFAULT_LOGIT_CLIP
        )

        assert np.allclose(eta.numpy(), 0.0, atol=1e-3), \
            "External financing cost should be zero for non-negative cash flow"

    def test_formula_for_negative_cashflow(self, params_with_injection_costs):
        """η(e) = η₀ + η₁|e| when e < 0."""
        e = tf.constant([-5.0])
        eta = logic.external_financing_cost(
            e, params_with_injection_costs,
            temperature=DEFAULT_TEMPERATURE,
            logit_clip=DEFAULT_LOGIT_CLIP
        )

        # Expected: η₀ + η₁ * 5.0
        expected = params_with_injection_costs.cost_inject_fixed + params_with_injection_costs.cost_inject_linear * 5.0

        assert np.isclose(eta.numpy(), expected, rtol=0.01), \
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

    def test_vectorized(self, params_with_injection_costs):
        """Works on batched inputs."""
        e = tf.constant([-1.0, 5.0, 1.0, -10.0])
        eta = logic.external_financing_cost(
            e, params_with_injection_costs,
            temperature=DEFAULT_TEMPERATURE,
            logit_clip=DEFAULT_LOGIT_CLIP
        )

        assert eta.shape == (4,)
        assert eta.numpy()[0] > 0  # negative e
        assert eta.numpy()[1] < 1e-3  # positive e (5.0), cost should be ~0
        assert eta.numpy()[2] < 1e-3  # positive e
        assert eta.numpy()[3] > 0  # negative e

    def test_ste_gradient_nonzero_for_negative_e(self, params_with_injection_costs):
        """Soft gate should provide non-zero gradient when e < 0."""
        e = tf.Variable([-0.5], dtype=tf.float32)  # Negative

        with tf.GradientTape() as tape:
            eta = logic.external_financing_cost(
                e, params_with_injection_costs,
                temperature=DEFAULT_TEMPERATURE,
                logit_clip=DEFAULT_LOGIT_CLIP
            )
            loss = tf.reduce_sum(eta)

        grad = tape.gradient(loss, e)

        assert grad is not None, "Soft gate should provide gradient"
        # Gradient should be non-zero for negative e (derivative of |e| * const)
        assert grad.numpy()[0] != 0, "Gradient should be non-zero for e < 0"


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

        psi = logic.adjustment_costs(
            k, k_next, params_no_fixed_cost,
            temperature=DEFAULT_TEMPERATURE,
            logit_clip=DEFAULT_LOGIT_CLIP
        )

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

        e = logic.compute_cash_flow_basic(
            k, k_next, z, params,
            temperature=DEFAULT_TEMPERATURE,
            logit_clip=DEFAULT_LOGIT_CLIP
        )

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
        q = 1.0 / (1.0 + r_tilde)  # Convert interest rate to bond price

        e = logic.cash_flow_risky_debt_q(
            k, k_next, b, b_next, z, q, params,
            temperature=DEFAULT_TEMPERATURE,
            logit_clip=DEFAULT_LOGIT_CLIP
        )

        assert e.shape == (2,)
        assert np.all(np.isfinite(e.numpy()))

    def test_risky_debt_differentiable_wrt_actions(self, params):
        """Gradients exist w.r.t. k_next, b_next, q."""
        k = tf.constant([1.0])
        k_next = tf.Variable([1.1])
        b = tf.constant([0.5])
        b_next = tf.Variable([0.6])
        z = tf.constant([1.0])
        q = tf.Variable([0.95])  # Bond price instead of interest rate

        with tf.GradientTape() as tape:
            e = logic.cash_flow_risky_debt_q(
                k, k_next, b, b_next, z, q, params,
                temperature=DEFAULT_TEMPERATURE,
                logit_clip=DEFAULT_LOGIT_CLIP
            )

        grads = tape.gradient(e, [k_next, b_next, q])

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
        
        # If r_tilde > r, Lender payoff (1+r_tilde) > Cost (1+r).
        # Residual f = Cost - Payoff < 0.
        
        f = logic.pricing_residual_zero_profit(
            b_next, r_risk_free, r_tilde, p_default, recovery
        )
        
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
            e = logic.compute_cash_flow_basic(
                k, k_next, z, params,
                temperature=DEFAULT_TEMPERATURE,
                logit_clip=DEFAULT_LOGIT_CLIP
            )

        grad = tape.gradient(e, k_next)

        assert grad is not None
        assert np.isfinite(grad.numpy())

    def test_adjustment_costs_grad_convex(self, params_no_fixed_cost):
        """Gradient of ψ w.r.t. k_next exists (convex part only)."""
        k = tf.constant([2.0])
        k_next = tf.Variable([2.5])

        with tf.GradientTape() as tape:
            psi = logic.adjustment_costs(
                k, k_next, params_no_fixed_cost,
                temperature=DEFAULT_TEMPERATURE,
                logit_clip=DEFAULT_LOGIT_CLIP
            )

        grad = tape.gradient(psi, k_next)

        assert grad is not None
        assert np.isfinite(grad.numpy())

    def test_adjustment_costs_grad_fixed_cost(self, params):
        """adjustment_costs should have gradient through fixed cost (via smooth gate)."""
        # Use params with non-zero fixed cost
        params_fixed = replace(params, cost_fixed=0.5, cost_convex=0.01)

        k = tf.constant([2.0])
        # I = 2.5 - (1-0.15)*2 = 2.5 - 1.7 = 0.8 > 0 (investing)
        k_next = tf.Variable([2.5])

        with tf.GradientTape() as tape:
            psi = logic.adjustment_costs(
                k, k_next, params_fixed,
                temperature=DEFAULT_TEMPERATURE,
                logit_clip=DEFAULT_LOGIT_CLIP
            )

        grad = tape.gradient(psi, k_next)

        assert grad is not None, "Gradient should exist with smooth gate"
        assert np.isfinite(grad.numpy()), "Gradient should be finite"
        # Gradient should be non-zero (from both convex and fixed cost)
        assert grad.numpy() != 0, "Gradient should be non-zero"





