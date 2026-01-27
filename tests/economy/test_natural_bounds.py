"""
tests/economy/test_natural_bounds.py

Tests for auto-calculation of log_z bounds using src.economy.bounds.
"""

import pytest
import numpy as np
from src.economy.bounds import (
    compute_ergodic_log_z_bounds,
    compute_natural_k_bounds,
    generate_states_bounds
)
from src.economy.parameters import EconomicParams, ShockParams

class TestNaturalBounds:
    """Tests for bounds calculation functions."""
    
    def test_ergodic_log_z_bounds(self):
        """Standard AR(1) case matches formula: mu +/- m * sigma_ergodic."""
        # Set specific params: rho=0.8, sigma=0.6, mu=1.0
        # sigma_ergodic = sigma / sqrt(1 - rho^2) = 0.6 / sqrt(1 - 0.64) = 0.6 / 0.6 = 1.0
        # Bounds (m=2.0) should be [1.0 - 2*1.0, 1.0 + 2*1.0] = [-1.0, 3.0]
        
        shock_params = ShockParams(rho=0.8, sigma=0.6, mu=1.0)
        
        min_log_z, max_log_z = compute_ergodic_log_z_bounds(
            shock_params=shock_params,
            std_dev_multiplier=2.0
        )
        
        assert np.isclose(min_log_z, -1.0), f"Expected -1.0, got {min_log_z}"
        assert np.isclose(max_log_z, 3.0), f"Expected 3.0, got {max_log_z}"


    
    def test_natural_k_bounds_calculation(self):
        """Standard case: k_bounds computed from z_max and multipliers."""
        
        # Setup parameters where math is easy
        # r = 0.04, delta = 0.06 -> r + delta = 0.1
        # theta = 0.5 -> 1 - theta = 0.5 -> exponent = 2
        
        theta = 0.5
        r = 0.04
        delta = 0.06
        
        # log_z_bounds = (0, 0) -> z = 1.0
        # k* = ((1.0 * 0.5) / 0.1) ** 2 = (5) ** 2 = 25.0
        log_z_bounds = (0.0, 0.0)
        
        k_min, k_max = compute_natural_k_bounds(
            theta=theta, r=r, delta=delta,
            log_z_bounds=log_z_bounds,
            k_min_multiplier=0.1,
            k_max_multiplier=2.0
        )
        
        # Expected:
        # k_min = 0.1 * 25.0 = 2.5
        # k_max = 2.0 * 25.0 = 50.0
        assert np.isclose(k_min, 2.5), f"Expected k_min=2.5, got {k_min}"
        assert np.isclose(k_max, 50.0), f"Expected k_max=50.0, got {k_max}"
    
    def test_generate_states_bounds_integration(self):
        """Integration test for full bounds generation."""
        params = EconomicParams.with_overrides(r_rate=0.04, delta=0.06, theta=0.5)
        shock_params = ShockParams(rho=0.0, sigma=1e-8, mu=0.0) # approx deterministic z=1
        
        bounds = generate_states_bounds(
            theta=params.theta,
            r=params.r_rate,
            delta=params.delta,
            shock_params=shock_params,
            std_dev_multiplier=2.0,
            k_min_multiplier=0.1,
            k_max_multiplier=2.0
        )
        
        assert "k" in bounds
        assert "b" in bounds
        assert "log_z" in bounds
        
        # k max should be ~50 (same as above)
        assert np.isclose(bounds["k"][1], 50.0)
