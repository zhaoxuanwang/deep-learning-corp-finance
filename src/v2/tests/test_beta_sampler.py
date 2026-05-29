"""Tests for the BetaSampler (uniform-only training sampler).

Asserted invariants:

  1. Output shape is (n, 5), columns ordered (α, ρ, σ_ε, φ_quad, φ_prop).
  2. Each column lies within its uniform box.
  3. Same seed → bit-identical output (stateless).
  4. Different seeds → different output.
  5. Sub-seed namespacing per coord → near-zero correlation between α and ρ
     even though they share the same default box.
  6. Empirical column means converge to box midpoints.
  7. `freeze_dims` zeros selected columns and leaves the rest untouched
     when only the freeze_dims changes between two samplers.
  8. `prior_mean()` returns box midpoints (zeros at frozen dims).
  9. Validation errors on bad inputs.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.v2.estimation.beta_sampler import (
    BETA_DIM,
    BETA_DIM_NAMES,
    DEFAULT_UNIFORM_BOUNDS,
    BetaSampler,
)


# --------------------------------------------------------------------- 1. shape

def test_dim_constants():
    assert BETA_DIM == 5
    assert BETA_DIM_NAMES == (
        "alpha", "rho", "sigma_epsilon", "phi_quad", "phi_prop")


def test_sample_shape_and_dtype():
    s = BetaSampler()
    beta = s.sample(64, seed=(0, 0))
    assert beta.shape == (64, 5)
    assert beta.dtype == tf.float32


# --------------------------------------------------------------------- 2. support

def test_each_column_in_box():
    s = BetaSampler()
    beta = s.sample(4096, seed=(1, 2)).numpy()
    for j, name in enumerate(BETA_DIM_NAMES):
        lo, hi = DEFAULT_UNIFORM_BOUNDS[name]
        assert beta[:, j].min() >= lo - 1e-6, (name, beta[:, j].min(), lo)
        assert beta[:, j].max() <= hi + 1e-6, (name, beta[:, j].max(), hi)


def test_custom_uniform_bounds():
    custom = (
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),
    )
    s = BetaSampler(uniform_bounds=custom)
    beta = s.sample(1024, seed=(9, 9)).numpy()
    assert beta.max() <= 0.5 + 1e-6
    assert beta.min() >= 0.0 - 1e-6


# --------------------------------------------------------------------- 3-4. seed determinism

def test_same_seed_same_output():
    s = BetaSampler()
    a = s.sample(128, seed=(11, 22)).numpy()
    b = s.sample(128, seed=(11, 22)).numpy()
    np.testing.assert_array_equal(a, b)


def test_different_seeds_different_output():
    s = BetaSampler()
    a = s.sample(128, seed=(11, 22)).numpy()
    b = s.sample(128, seed=(11, 23)).numpy()
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------- 5. sub-seed independence

def test_alpha_rho_decorrelated():
    """α and ρ share a Uniform(0, 1)-style box; per-coord sub-seed
    namespacing should yield near-zero correlation."""
    s = BetaSampler(
        uniform_bounds=(
            (0.0, 1.0), (0.0, 1.0),
            DEFAULT_UNIFORM_BOUNDS["sigma_epsilon"],
            DEFAULT_UNIFORM_BOUNDS["phi_quad"],
            DEFAULT_UNIFORM_BOUNDS["phi_prop"]),
    )
    beta = s.sample(4096, seed=(0, 0)).numpy()
    alpha, rho = beta[:, 0], beta[:, 1]
    assert abs(np.corrcoef(alpha, rho)[0, 1]) < 0.05


# --------------------------------------------------------------------- 6. empirical mean

def test_empirical_mean_matches_midpoint():
    s = BetaSampler()
    beta = s.sample(20000, seed=(5, 6)).numpy()
    for j, name in enumerate(BETA_DIM_NAMES):
        lo, hi = DEFAULT_UNIFORM_BOUNDS[name]
        midpoint = 0.5 * (lo + hi)
        observed = float(beta[:, j].mean())
        np.testing.assert_allclose(
            observed, midpoint, atol=0.01 * (hi - lo) + 0.005)


# --------------------------------------------------------------------- 7. freeze_dims

def test_freeze_dims_zeros_selected_columns():
    s = BetaSampler(freeze_dims=(3, 4))
    beta = s.sample(128, seed=(7, 8)).numpy()
    np.testing.assert_array_equal(beta[:, 3], np.zeros(128, dtype=np.float32))
    np.testing.assert_array_equal(beta[:, 4], np.zeros(128, dtype=np.float32))
    assert beta[:, 0].std() > 0.05
    assert beta[:, 2].std() > 0.05


def test_freeze_dims_leaves_unfrozen_columns_unchanged():
    s_full = BetaSampler()
    s_frictionless = BetaSampler(freeze_dims=(3, 4))
    beta_full = s_full.sample(128, seed=(9, 9)).numpy()
    beta_fric = s_frictionless.sample(128, seed=(9, 9)).numpy()
    np.testing.assert_array_equal(beta_full[:, 0:3], beta_fric[:, 0:3])


# --------------------------------------------------------------------- 8. prior_mean (box midpoint)

def test_prior_mean_is_box_midpoint():
    s = BetaSampler()
    mu = s.prior_mean().numpy()
    assert mu.shape == (1, 5)
    for j, name in enumerate(BETA_DIM_NAMES):
        lo, hi = DEFAULT_UNIFORM_BOUNDS[name]
        np.testing.assert_allclose(mu[0, j], 0.5 * (lo + hi), atol=1e-6)


def test_prior_mean_zeros_frozen_dims():
    s = BetaSampler(freeze_dims=(3, 4))
    mu = s.prior_mean().numpy()
    assert mu[0, 3] == 0.0 and mu[0, 4] == 0.0
    # First three are still box midpoints
    lo_a, hi_a = DEFAULT_UNIFORM_BOUNDS["alpha"]
    assert mu[0, 0] == pytest.approx(0.5 * (lo_a + hi_a))


# --------------------------------------------------------------------- 9. validation

class TestValidation:
    def test_bad_freeze_dim(self):
        with pytest.raises(ValueError, match="out of range"):
            BetaSampler(freeze_dims=(7,))

    def test_duplicate_freeze_dim(self):
        with pytest.raises(ValueError, match="duplicate"):
            BetaSampler(freeze_dims=(1, 1))

    def test_bad_uniform_bounds_length(self):
        with pytest.raises(ValueError, match="uniform_bounds must have"):
            BetaSampler(uniform_bounds=((0.0, 1.0),))

    def test_bad_uniform_bounds_low_ge_high(self):
        bad = ((0.5, 0.2),) + tuple((0.0, 1.0) for _ in range(4))
        with pytest.raises(ValueError, match="must be < high"):
            BetaSampler(uniform_bounds=bad)

    def test_bad_batch_size(self):
        s = BetaSampler()
        with pytest.raises(ValueError, match="batch_size"):
            s.sample(0, seed=(0, 0))
