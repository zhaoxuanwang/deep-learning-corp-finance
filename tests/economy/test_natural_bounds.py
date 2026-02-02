"""
tests/economy/test_natural_bounds.py

Tests for bounds calculation in src.economy.bounds.

Reference:
    report_brief.md lines 84-122: Observation Normalization

Note: Bounds are now returned in LEVELS (not as multipliers on k*).
The user specifies bounds as multipliers, but the returned bounds are
already converted to level space for direct use by networks.
"""

import pytest
import numpy as np
import warnings
from src.economy.bounds import (
    BoundsConfig,
    compute_ergodic_log_z_bounds,
    compute_k_star,
    compute_k_bounds_levels,
    compute_b_bound_levels,
    generate_states_bounds,
    generate_bounds_from_config,
    # Legacy functions (deprecated)
    compute_natural_k_bounds,
    compute_natural_b_bound,
)
from src.economy.parameters import EconomicParams, ShockParams


class TestBoundsConfig:
    """Tests for BoundsConfig validation."""

    def test_valid_config(self):
        """Valid config within all constraints."""
        config = BoundsConfig(m=3.0, k_min=0.2, k_max=3.0)
        assert config.m == 3.0
        assert config.k_min == 0.2
        assert config.k_max == 3.0

    def test_m_constraint_lower(self):
        """m must be > 2."""
        with pytest.raises(ValueError, match="m.*must be in.*2.*5"):
            BoundsConfig(m=2.0, k_min=0.2, k_max=3.0)

    def test_m_constraint_upper(self):
        """m must be < 5."""
        with pytest.raises(ValueError, match="m.*must be in.*2.*5"):
            BoundsConfig(m=5.0, k_min=0.2, k_max=3.0)

    def test_k_min_constraint_lower(self):
        """k_min must be > 0."""
        with pytest.raises(ValueError, match="k_min.*must be in.*0.*0.5"):
            BoundsConfig(m=3.0, k_min=0.0, k_max=3.0)

    def test_k_min_constraint_upper(self):
        """k_min must be < 0.5."""
        with pytest.raises(ValueError, match="k_min.*must be in.*0.*0.5"):
            BoundsConfig(m=3.0, k_min=0.5, k_max=3.0)

    def test_k_max_constraint_lower(self):
        """k_max must be > 1.5."""
        with pytest.raises(ValueError, match="k_max.*must be in.*1.5.*5"):
            BoundsConfig(m=3.0, k_min=0.2, k_max=1.5)

    def test_k_max_constraint_upper(self):
        """k_max must be < 5."""
        with pytest.raises(ValueError, match="k_max.*must be in.*1.5.*5"):
            BoundsConfig(m=3.0, k_min=0.2, k_max=5.0)

    def test_k_star_override(self):
        """k_star_override is optional and not validated."""
        config = BoundsConfig(m=3.0, k_min=0.2, k_max=3.0, k_star_override=100.0)
        assert config.k_star_override == 100.0


class TestErgodicLogZBounds:
    """Tests for compute_ergodic_log_z_bounds (unchanged from original)."""

    def test_standard_case(self):
        """Standard AR(1) case matches formula: mu +/- m * sigma_ergodic."""
        # Set specific params: rho=0.8, sigma=0.6, mu=1.0
        # sigma_ergodic = sigma / sqrt(1 - rho^2) = 0.6 / sqrt(1 - 0.64) = 0.6 / 0.6 = 1.0
        # Bounds (m=3.0) should be [1.0 - 3*1.0, 1.0 + 3*1.0] = [-2.0, 4.0]
        shock_params = ShockParams(rho=0.8, sigma=0.6, mu=1.0)

        min_log_z, max_log_z = compute_ergodic_log_z_bounds(
            shock_params=shock_params,
            std_dev_multiplier=3.0
        )

        assert np.isclose(min_log_z, -2.0), f"Expected -2.0, got {min_log_z}"
        assert np.isclose(max_log_z, 4.0), f"Expected 4.0, got {max_log_z}"


class TestComputeKStar:
    """Tests for compute_k_star function."""

    def test_k_star_at_mean(self):
        """k* computed at stationary mean z = e^mu."""
        # Setup: theta = 0.5, r = 0.04, delta = 0.06 -> r + delta = 0.1
        # mu = 0 -> z = e^0 = 1.0
        # k* = ((1.0 * 0.5) / 0.1) ** (1 / 0.5) = (5) ** 2 = 25.0
        k_star = compute_k_star(theta=0.5, r=0.04, delta=0.06, mu=0.0)
        assert np.isclose(k_star, 25.0), f"Expected k_star=25.0, got {k_star}"

    def test_k_star_with_nonzero_mu(self):
        """k* at non-zero mu uses z = e^mu."""
        # mu = ln(2) -> z = 2.0
        # k* = ((2.0 * 0.5) / 0.1) ** 2 = (10) ** 2 = 100.0
        mu = np.log(2.0)
        k_star = compute_k_star(theta=0.5, r=0.04, delta=0.06, mu=mu)
        assert np.isclose(k_star, 100.0), f"Expected k_star=100.0, got {k_star}"


class TestKBoundsLevels:
    """Tests for compute_k_bounds_levels function."""

    def test_k_bounds_are_levels(self):
        """k bounds are returned in LEVELS (multiplier * k_star)."""
        k_bounds, k_star = compute_k_bounds_levels(
            k_min_multiplier=0.2,
            k_max_multiplier=3.0,
            theta=0.5,
            r=0.04,
            delta=0.06,
            mu=0.0
        )

        # k_star should be 25.0 (from previous test)
        assert np.isclose(k_star, 25.0), f"Expected k_star=25.0, got {k_star}"

        # Bounds should be multipliers * k_star = (0.2 * 25, 3.0 * 25) = (5.0, 75.0)
        assert np.isclose(k_bounds[0], 5.0), f"Expected k_min=5.0, got {k_bounds[0]}"
        assert np.isclose(k_bounds[1], 75.0), f"Expected k_max=75.0, got {k_bounds[1]}"

    def test_k_star_override(self):
        """k_star_override bypasses auto-calculation."""
        k_bounds, k_star = compute_k_bounds_levels(
            k_min_multiplier=0.2,
            k_max_multiplier=3.0,
            theta=0.5,
            r=0.04,
            delta=0.06,
            mu=0.0,
            k_star_override=100.0
        )

        # k_star should be the override value
        assert k_star == 100.0, f"Expected k_star=100.0, got {k_star}"

        # Bounds should be multipliers * k_star = (0.2 * 100, 3.0 * 100) = (20.0, 300.0)
        assert np.isclose(k_bounds[0], 20.0), f"Expected k_min=20.0, got {k_bounds[0]}"
        assert np.isclose(k_bounds[1], 300.0), f"Expected k_max=300.0, got {k_bounds[1]}"


class TestBBoundLevels:
    """Tests for compute_b_bound_levels function with collateral constraint."""

    def test_b_bound_levels_formula(self):
        """
        b_max = (1-τ) * π(k_max, z_min) + τ * δ * k_max + s_liquid * k_max

        This is the collateral constraint formula from report_brief.md.
        """
        # Parameters: k_max=75.0, z_min=1.0, theta=0.5, tax=0.3, delta=0.06, frac_liquid=0.5
        # π(k_max, z_min) = 1.0 * 75^0.5 = 8.66
        # b_max = 0.7 * 8.66 + 0.3 * 0.06 * 75 + 0.5 * 75 = 6.06 + 1.35 + 37.5 = 44.91
        b_max = compute_b_bound_levels(
            theta=0.5, k_max=75.0, z_min=1.0,
            tax=0.3, delta=0.06, frac_liquid=0.5
        )
        pi_worst = 1.0 * (75.0 ** 0.5)
        expected = 0.7 * pi_worst + 0.3 * 0.06 * 75.0 + 0.5 * 75.0
        assert np.isclose(b_max, expected), f"Expected {expected}, got {b_max}"

    def test_b_bound_scales_with_z_min_and_k(self):
        """b_max scales with z_min and k_max under collateral constraint."""
        # Lower z_min should reduce b_max (worse productivity = lower collateral value)
        b_max_high_z = compute_b_bound_levels(
            theta=0.5, k_max=100.0, z_min=1.0,
            tax=0.3, delta=0.06, frac_liquid=0.5
        )
        b_max_low_z = compute_b_bound_levels(
            theta=0.5, k_max=100.0, z_min=0.5,
            tax=0.3, delta=0.06, frac_liquid=0.5
        )
        # Lower z_min should give lower b_max
        assert b_max_low_z < b_max_high_z, (
            f"Lower z_min should reduce b_max: z_min=0.5 gave {b_max_low_z}, "
            f"z_min=1.0 gave {b_max_high_z}"
        )


class TestGenerateStatesBounds:
    """Tests for generate_states_bounds (main integration function)."""

    def test_full_bounds_generation(self):
        """Integration test for bounds generation in LEVELS with collateral constraint."""
        shock_params = ShockParams(rho=0.0, sigma=1e-8, mu=0.0)  # approx deterministic z=1

        # Use explicit collateral constraint parameters
        tax = 0.3
        delta = 0.06
        frac_liquid = 0.5

        bounds = generate_states_bounds(
            theta=0.5,
            r=0.04,
            delta=delta,
            shock_params=shock_params,
            std_dev_multiplier=3.0,
            k_min_multiplier=0.2,
            k_max_multiplier=3.0,
            tax=tax,
            frac_liquid=frac_liquid
        )

        # Check all expected keys exist
        assert "k" in bounds
        assert "b" in bounds
        assert "log_z" in bounds
        assert "k_star" in bounds

        # k_star should be 25.0
        assert np.isclose(bounds["k_star"], 25.0), f"Expected k_star=25.0, got {bounds['k_star']}"

        # k bounds should be in LEVELS: (0.2 * 25, 3.0 * 25) = (5.0, 75.0)
        assert np.isclose(bounds["k"][0], 5.0), f"Expected k_min=5.0, got {bounds['k'][0]}"
        assert np.isclose(bounds["k"][1], 75.0), f"Expected k_max=75.0, got {bounds['k'][1]}"

        # b bounds: (0, b_max) using COLLATERAL CONSTRAINT:
        # b_max = (1-τ) * π(k_max, z_min) + τ * δ * k_max + s_liquid * k_max
        # With z_min ≈ 1, k_max = 75.0 (level), theta = 0.5, tax=0.3, delta=0.06, frac_liquid=0.5:
        # π(k_max, z_min) = 1.0 * 75^0.5 = 8.66
        # b_max = 0.7 * 8.66 + 0.3 * 0.06 * 75 + 0.5 * 75 = 6.06 + 1.35 + 37.5 ≈ 44.91
        pi_worst = 1.0 * (75.0 ** 0.5)
        expected_b_max = (1 - tax) * pi_worst + tax * delta * 75.0 + frac_liquid * 75.0
        assert bounds["b"][0] == 0.0
        assert np.isclose(bounds["b"][1], expected_b_max, rtol=0.01), (
            f"Expected b_max={expected_b_max:.2f}, got {bounds['b'][1]:.2f}"
        )

    def test_validation_enforced(self):
        """Validation rejects out-of-range parameters."""
        shock_params = ShockParams()

        # m out of range
        with pytest.raises(ValueError, match="std_dev_multiplier.*must be in"):
            generate_states_bounds(
                theta=0.5, r=0.04, delta=0.06,
                shock_params=shock_params,
                std_dev_multiplier=1.5,  # Invalid: < 2
                k_min_multiplier=0.2,
                k_max_multiplier=3.0
            )

    def test_validation_can_be_disabled(self):
        """validate=False allows out-of-range parameters (for testing)."""
        shock_params = ShockParams()

        # Should not raise with validate=False
        bounds = generate_states_bounds(
            theta=0.5, r=0.04, delta=0.06,
            shock_params=shock_params,
            std_dev_multiplier=1.5,  # Would be invalid
            k_min_multiplier=0.2,
            k_max_multiplier=3.0,
            validate=False
        )

        assert "k" in bounds


class TestGenerateBoundsFromConfig:
    """Tests for generate_bounds_from_config convenience function."""

    def test_from_config(self):
        """Bounds generated from BoundsConfig object are in LEVELS."""
        config = BoundsConfig(m=3.0, k_min=0.2, k_max=3.0)
        shock_params = ShockParams(rho=0.0, sigma=1e-8, mu=0.0)

        bounds = generate_bounds_from_config(
            bounds_config=config,
            theta=0.5,
            r=0.04,
            delta=0.06,
            shock_params=shock_params
        )

        # k_star = 25.0
        assert np.isclose(bounds["k_star"], 25.0)

        # k bounds in LEVELS: (0.2 * 25, 3.0 * 25) = (5.0, 75.0)
        assert np.isclose(bounds["k"][0], 5.0)
        assert np.isclose(bounds["k"][1], 75.0)


class TestLegacyFunctions:
    """Tests for deprecated legacy functions."""

    def test_compute_natural_k_bounds_deprecated(self):
        """Legacy function emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            k_min, k_max = compute_natural_k_bounds(
                theta=0.5, r=0.04, delta=0.06,
                log_z_bounds=(0.0, 0.0),
                k_min_multiplier=0.1,
                k_max_multiplier=2.0
            )

            # Check deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_compute_natural_b_bound_deprecated(self):
        """Legacy function emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            b_max = compute_natural_b_bound(theta=0.5, k_max=50.0, z_max=1.0)

            # Check deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
