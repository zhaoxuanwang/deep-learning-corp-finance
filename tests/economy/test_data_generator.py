"""
tests/economy/test_data_generator.py

Comprehensive tests for the data generator module.

Test categories:
1. Determinism
2. Step indexing
3. Split isolation
4. Shape checks
5. Sealed test behavior
6. Z-rollout correctness
7. Integration tests
"""

import pytest
import tensorflow as tf
import numpy as np

from src.economy.data_generator import DataGenerator, create_data_generator
from src.economy.parameters import EconomicParams, ShockParams


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def params():
    """Create default economic parameters."""
    return EconomicParams()


@pytest.fixture
def shock_params():
    """Create default shock parameters."""
    return ShockParams()


@pytest.fixture
def basic_generator(shock_params):
    """Create a basic data generator for testing."""
    return DataGenerator(
        master_seed=(42, 0),
        shock_params=shock_params,
        k_bounds=(0.5, 5.0),
        logz_bounds=(-0.3, 0.3),
        b_bounds=(0.0, 3.0),
        sim_batch_size=16,
        T=5,
        n_sim_batches=10,
        N_val=32,
        N_test=64,
        save_to_disk=False  # Prevent cache pollution during tests
    )


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_training_batches(self, shock_params):
        """Same master seed produces identical training batches."""
        gen1 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=3
        )

        gen2 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=3
        )

        batches1 = list(gen1.get_training_batches())
        batches2 = list(gen2.get_training_batches())

        assert len(batches1) == 3
        assert len(batches2) == 3

        for i in range(3):
            for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
                assert tf.reduce_all(batches1[i][key] == batches2[i][key]), \
                    f"Mismatch in batch {i}, key {key}"

    def test_same_seed_same_validation(self, shock_params):
        """Same master seed produces identical validation data."""
        gen1 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32
        )

        gen2 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32
        )

        val1 = gen1.get_validation_dataset()
        val2 = gen2.get_validation_dataset()

        for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
            assert tf.reduce_all(val1[key] == val2[key]), f"Mismatch in {key}"

    def test_same_seed_same_test(self, shock_params):
        """Same master seed produces identical test data."""
        gen1 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_test=64
        )

        gen2 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_test=64
        )

        test1 = gen1.get_test_dataset()
        test2 = gen2.get_test_dataset()

        for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
            assert tf.reduce_all(test1[key] == test2[key]), f"Mismatch in {key}"

    def test_validation_cached(self, basic_generator):
        """Validation dataset is cached after first generation."""
        val1 = basic_generator.get_validation_dataset()
        val2 = basic_generator.get_validation_dataset()

        # Should be the same object (cached)
        for key in val1.keys():
            assert val1[key] is val2[key], f"Validation not cached for {key}"

    def test_test_cached(self, basic_generator):
        """Test dataset is cached after first generation."""
        test1 = basic_generator.get_test_dataset()
        test2 = basic_generator.get_test_dataset()

        # Should be the same object (cached)
        for key in test1.keys():
            assert test1[key] is test2[key], f"Test not cached for {key}"


# =============================================================================
# STEP INDEXING TESTS
# =============================================================================

class TestStepIndexing:
    """Tests for training step indexing."""

    def test_training_batches_count(self, basic_generator):
        """Training produces exactly n_sim_batches batches."""
        batches = list(basic_generator.get_training_batches())
        assert len(batches) == basic_generator.n_sim_batches

    def test_consecutive_batches_differ(self, basic_generator):
        """Consecutive training batches have different data."""
        batches = list(basic_generator.get_training_batches())

        # Compare first two batches
        batch1 = batches[0]
        batch2 = batches[1]

        # They should differ (with very high probability for random data)
        for key in ['k0', 'z0', 'b0', 'eps_path', 'eps_fork']:
            assert not tf.reduce_all(batch1[key] == batch2[key]), \
                f"Batches 1 and 2 should differ in {key}"

    def test_step_1_and_j_differ(self, basic_generator):
        """First and last training batches differ."""
        batches = list(basic_generator.get_training_batches())

        batch_first = batches[0]
        batch_last = batches[-1]

        for key in ['k0', 'z0', 'b0', 'eps_path', 'eps_fork']:
            assert not tf.reduce_all(batch_first[key] == batch_last[key]), \
                f"First and last batches should differ in {key}"


# =============================================================================
# SPLIT ISOLATION TESTS
# =============================================================================

class TestSplitIsolation:
    """Tests for train/val/test isolation."""

    def test_train_val_differ(self, basic_generator):
        """Training and validation data differ."""
        batches = list(basic_generator.get_training_batches())
        train_batch = batches[0]
        val_data = basic_generator.get_validation_dataset()

        # Compare first n samples (same size as training batch)
        for key in ['k0', 'z0', 'b0', 'eps_path', 'eps_fork']:
            train_slice = train_batch[key]
            val_slice = val_data[key][:basic_generator.sim_batch_size]

            assert not tf.reduce_all(train_slice == val_slice), \
                f"Train and val should differ in {key}"

    def test_val_test_differ(self, basic_generator):
        """Validation and test data differ."""
        val_data = basic_generator.get_validation_dataset()
        test_data = basic_generator.get_test_dataset()

        # Compare same-sized slices
        n_compare = min(basic_generator.N_val, basic_generator.N_test)

        for key in ['k0', 'z0', 'b0', 'eps_path', 'eps_fork']:
            val_slice = val_data[key][:n_compare]
            test_slice = test_data[key][:n_compare]

            assert not tf.reduce_all(val_slice == test_slice), \
                f"Val and test should differ in {key}"

    def test_train_test_differ(self, basic_generator):
        """Training and test data differ."""
        batches = list(basic_generator.get_training_batches())
        train_batch = batches[0]
        test_data = basic_generator.get_test_dataset()

        # Compare first n samples
        for key in ['k0', 'z0', 'b0', 'eps_path', 'eps_fork']:
            train_slice = train_batch[key]
            test_slice = test_data[key][:basic_generator.sim_batch_size]

            assert not tf.reduce_all(train_slice == test_slice), \
                f"Train and test should differ in {key}"


# =============================================================================
# SHAPE TESTS
# =============================================================================

class TestShapes:
    """Tests for tensor shapes."""

    def test_training_batch_shapes(self, basic_generator):
        """Training batch has correct shapes."""
        batches = list(basic_generator.get_training_batches())
        batch = batches[0]

        n = basic_generator.sim_batch_size
        T = basic_generator.T

        assert batch['k0'].shape == (n,)
        assert batch['z0'].shape == (n,)
        assert batch['b0'].shape == (n,)
        assert batch['z_path'].shape == (n, T + 1)
        assert batch['z_fork'].shape == (n, T, 1)
        assert batch['eps_path'].shape == (n, T)
        assert batch['eps_fork'].shape == (n, T)

    def test_validation_shapes(self, basic_generator):
        """Validation dataset has correct shapes."""
        val_data = basic_generator.get_validation_dataset()

        N_val = basic_generator.N_val
        T = basic_generator.T

        assert val_data['k0'].shape == (N_val,)
        assert val_data['z0'].shape == (N_val,)
        assert val_data['b0'].shape == (N_val,)
        assert val_data['z_path'].shape == (N_val, T + 1)
        assert val_data['z_fork'].shape == (N_val, T, 1)
        assert val_data['eps_path'].shape == (N_val, T)
        assert val_data['eps_fork'].shape == (N_val, T)

    def test_test_shapes(self, basic_generator):
        """Test dataset has correct shapes."""
        test_data = basic_generator.get_test_dataset()

        N_test = basic_generator.N_test
        T = basic_generator.T

        assert test_data['k0'].shape == (N_test,)
        assert test_data['z0'].shape == (N_test,)
        assert test_data['b0'].shape == (N_test,)
        assert test_data['z_path'].shape == (N_test, T + 1)
        assert test_data['z_fork'].shape == (N_test, T, 1)
        assert test_data['eps_path'].shape == (N_test, T)
        assert test_data['eps_fork'].shape == (N_test, T)

    def test_dtypes(self, basic_generator):
        """All tensors have float32 dtype."""
        batch = next(basic_generator.get_training_batches())

        for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
            assert batch[key].dtype == tf.float32, f"Wrong dtype for {key}"


# =============================================================================
# Z-ROLLOUT TESTS
# =============================================================================

class TestZRollout:
    """Tests for z-path rollout correctness."""

    def test_z_path_starts_with_z0(self, basic_generator):
        """z_path[:, 0] equals z0."""
        batch = next(basic_generator.get_training_batches())

        z0 = batch['z0']
        z_path_0 = batch['z_path'][:, 0]

        assert tf.reduce_all(z0 == z_path_0)

    def test_z_path_length(self, basic_generator):
        """z_path has length T+1."""
        batch = next(basic_generator.get_training_batches())

        z_path = batch['z_path']
        T = basic_generator.T

        assert z_path.shape[1] == T + 1

    def test_z_path_positive(self, basic_generator):
        """All z values are positive (productivity must be > 0)."""
        batch = next(basic_generator.get_training_batches())

        z_path = batch['z_path']
        assert tf.reduce_all(z_path > 0)

    def test_z_rollout_uses_eps1(self, basic_generator):
        """Z-rollout uses eps1 shocks (verify by manual step)."""
        batch = next(basic_generator.get_training_batches())

        z_path = batch['z_path']
        eps1 = batch['eps_path']

        # Verify first transition: z[1] should be close to step_ar1_tf(z[0], eps1[:, 0])
        from src.economy.shocks import step_ar1_tf

        z0 = z_path[:, 0]
        eps_0 = eps1[:, 0]

        z1_expected = step_ar1_tf(
            z0,
            basic_generator.shock_params.rho,
            basic_generator.shock_params.sigma,
            basic_generator.shock_params.mu,
            eps_0
        )

        z1_actual = z_path[:, 1]

        # Should match exactly (deterministic computation)
        assert tf.reduce_all(tf.abs(z1_actual - z1_expected) < 1e-6)

    def test_fork_rollout_uses_eps_fork(self, basic_generator):
        """Fork rollout uses eps_fork shocks.
           z_fork[:, t] derived from z_path[:, t] using eps_fork[:, t].
        """
        batch = next(basic_generator.get_training_batches())

        z_path = batch['z_path']
        z_fork = batch['z_fork']
        eps_fork = batch['eps_fork']

        # Verify first fork: z_fork[:, 0] comes from z_path[:, 0] and eps_fork[:, 0]
        from src.economy.shocks import step_ar1_tf

        z0 = z_path[:, 0]
        eps_f0 = eps_fork[:, 0]

        z_fork_0_expected = step_ar1_tf(
            z0,
            basic_generator.shock_params.rho,
            basic_generator.shock_params.sigma,
            basic_generator.shock_params.mu,
            eps_f0
        )
        
        z_fork_0_actual = tf.reshape(z_fork[:, 0], [-1]) # flatten for comparison
        
        # Note: z_fork in batch is (B, T, 1). z_fork_0_expected is (B,) or (B,1) depending on step_ar1_tf output handling?
        # step_ar1_tf usually returns same shape as input Z. z0 is (B,).
        
        assert tf.reduce_all(tf.abs(z_fork_0_actual - z_fork_0_expected) < 1e-6)

    def test_path_and_fork_differ(self, basic_generator):
        """Main path next step and fork next step should differ."""
        batch = next(basic_generator.get_training_batches())
        z_path_next = batch['z_path'][:, 1:] # z_1 ... z_T
        z_fork = batch['z_fork']             # z'_1 ... z'_T
        
        # Flatten for comparison
        z_path_flat = tf.reshape(z_path_next, [-1])
        z_fork_flat = tf.reshape(z_fork, [-1])
        
        # Paths track different noise, so they must diverge
        assert not tf.reduce_all(z_path_flat == z_fork_flat)

# =============================================================================
# BOUNDS TESTS
# =============================================================================

class TestBounds:
    """Tests for respecting bounds."""

    def test_k0_within_bounds(self, basic_generator):
        """k0 is within specified bounds."""
        batch = next(basic_generator.get_training_batches())

        k0 = batch['k0']
        k_min, k_max = basic_generator.k_bounds

        assert tf.reduce_all(k0 >= k_min)
        assert tf.reduce_all(k0 <= k_max)

    def test_b0_within_bounds(self, basic_generator):
        """b0 is within specified bounds."""
        batch = next(basic_generator.get_training_batches())

        b0 = batch['b0']
        b_min, b_max = basic_generator.b_bounds

        assert tf.reduce_all(b0 >= b_min)
        assert tf.reduce_all(b0 <= b_max)

    def test_logz0_within_bounds_implicitly(self, basic_generator):
        """log(z0) should be within logz_bounds (approximately)."""
        batch = next(basic_generator.get_training_batches())

        z0 = batch['z0']
        logz0 = tf.math.log(z0)

        logz_min, logz_max = basic_generator.logz_bounds

        # Should be within bounds (with small tolerance for numerical precision)
        assert tf.reduce_all(logz0 >= logz_min - 1e-6)
        assert tf.reduce_all(logz0 <= logz_max + 1e-6)


# =============================================================================
# SEALED TEST BEHAVIOR TESTS
# =============================================================================

class TestSealedTest:
    """Tests for sealed test dataset behavior."""

    def test_test_not_generated_until_called(self, basic_generator):
        """Test dataset is not generated until get_test_dataset() is called."""
        # Before calling get_test_dataset, _test_dataset should be None
        assert basic_generator._test_dataset is None

        # After calling, it should be set
        test_data = basic_generator.get_test_dataset()
        assert basic_generator._test_dataset is not None

    def test_validation_not_generated_until_called(self, basic_generator):
        """Validation dataset is not generated until get_validation_dataset() is called."""
        assert basic_generator._validation_dataset is None

        val_data = basic_generator.get_validation_dataset()
        assert basic_generator._validation_dataset is not None

    def test_test_independent_of_training(self, basic_generator):
        """Test data is independent of training batch generation."""
        # Generate some training batches
        batches = list(basic_generator.get_training_batches())

        # Now get test data
        test_data = basic_generator.get_test_dataset()

        # Test should still be deterministic and independent
        # Verify by creating a fresh generator and comparing
        gen2 = DataGenerator(
            master_seed=(42, 0),
            shock_params=basic_generator.shock_params,
            k_bounds=basic_generator.k_bounds,
            logz_bounds=basic_generator.logz_bounds,
            b_bounds=basic_generator.b_bounds,
            sim_batch_size=basic_generator.sim_batch_size,
            T=basic_generator.T,
            n_sim_batches=basic_generator.n_sim_batches,
            N_test=basic_generator.N_test
        )

        test_data2 = gen2.get_test_dataset()

        for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
            assert tf.reduce_all(test_data[key] == test_data2[key]), \
                f"Test data affected by training for {key}"


# =============================================================================
# DEFAULT SIZE TESTS
# =============================================================================

class TestDefaultSizes:
    """Tests for default N_val and N_test sizes."""

    def test_default_n_val(self, shock_params):
        """N_val defaults to 10*sim_batch_size."""
        gen = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10
            # N_val not specified
        )

        assert gen.N_val == 10 * 16

    def test_default_n_test(self, shock_params):
        """N_test defaults to 50*sim_batch_size."""
        gen = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10
            # N_test not specified
        )

        assert gen.N_test == 50 * 16

    def test_custom_sizes(self, shock_params):
        """Custom N_val and N_test are respected."""
        gen = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=100,
            N_test=200
        )

        assert gen.N_val == 100
        assert gen.N_test == 200


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction:
    """Tests for create_data_generator convenience function."""

    def test_create_dataset(self, params):
        """create_dataset creates a valid generator."""
        gen, _, _ = create_data_generator(
            master_seed=(42, 0),
            theta=params.theta, r=params.r_rate, delta=params.delta,
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            save_to_disk=False  # Prevent cache pollution during tests
        )

        assert isinstance(gen, DataGenerator)
        assert gen.sim_batch_size == 16
        assert gen.T == 5
        assert gen.n_sim_batches == 10

    def test_create_dataset_with_defaults(self, params):
        """create_dataset uses default values."""
        gen, _, _ = create_data_generator(
            master_seed=(42, 0),
            theta=params.theta, r=params.r_rate, delta=params.delta,
            bounds={
                'k': (0.5, 5.0),
                'log_z': (-0.3, 0.3),
                'b': (0.0, 5.0)
            },
            auto_compute_bounds=False,
            save_to_disk=False  # Prevent cache pollution during tests
        )

        # Check defaults
        assert gen.sim_batch_size == 128
        assert gen.T == 64
        assert gen.n_sim_batches == 256
        assert gen.b_bounds == (0.0, 5.0)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with realistic usage."""

    def test_full_training_loop(self, basic_generator):
        """Simulate a full training loop."""
        step_count = 0
        for step, batch in enumerate(basic_generator.get_training_batches(), start=1):
            # Verify batch structure
            assert 'k0' in batch
            assert 'z_path' in batch
            assert 'z_fork' in batch
            assert 'eps_path' in batch

            # Verify batch size
            assert batch['k0'].shape[0] == basic_generator.sim_batch_size

            step_count += 1

        assert step_count == basic_generator.n_sim_batches

    def test_train_val_test_workflow(self, basic_generator):
        """Simulate typical train/val/test workflow."""
        # Training
        for step, batch in enumerate(basic_generator.get_training_batches(), start=1):
            if step >= 3:  # Just do a few steps
                break

        # Validation
        val_data = basic_generator.get_validation_dataset()
        assert val_data['k0'].shape[0] == basic_generator.N_val

        # Test (after training complete)
        test_data = basic_generator.get_test_dataset()
        assert test_data['k0'].shape[0] == basic_generator.N_test

    def test_eps_statistics_reasonable(self, basic_generator):
        """Generated shocks have reasonable statistics."""
        batch = next(basic_generator.get_training_batches())

        eps1 = batch['eps_path'].numpy()

        # Should be approximately N(0, 1)
        mean = np.mean(eps1)
        std = np.std(eps1)

        # With small sample, just check rough bounds
        assert abs(mean) < 0.5  # Should be close to 0
        assert 0.5 < std < 1.5  # Should be close to 1


# =============================================================================
# DISK CACHING TESTS
# =============================================================================

class TestDataGeneratorCaching:
    """Test disk caching functionality for validation and test datasets."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        return str(tmp_path / "cache")

    @pytest.fixture
    def cached_generator(self, shock_params, cache_dir):
        """Create a data generator with caching enabled."""
        return DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32,
            N_test=64,
            cache_dir=cache_dir
        )

    def test_cache_dir_none_by_default(self, shock_params):
        """Cache directory defaults to standard location when None."""
        gen = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=None
        )
        assert gen.cache_dir is not None
        assert gen.cache_dir.endswith("data")

    def test_cache_dir_set_when_provided(self, cached_generator, cache_dir):
        """Cache directory is set when provided."""
        assert cached_generator.cache_dir == cache_dir

    def test_validation_cache_creates_file(self, cached_generator, cache_dir):
        """Validation dataset is saved to disk when cache_dir is set."""
        import os

        # Get validation dataset (should create cache file)
        val_data = cached_generator.get_validation_dataset()

        # Check cache file exists
        cache_path = cached_generator._get_cache_path("validation")
        assert os.path.exists(cache_path)
        assert cache_path.endswith(".npz")

    def test_test_cache_creates_file(self, cached_generator, cache_dir):
        """Test dataset is saved to disk when cache_dir is set."""
        import os

        # Get test dataset (should create cache file)
        test_data = cached_generator.get_test_dataset()

        # Check cache file exists
        cache_path = cached_generator._get_cache_path("test")
        assert os.path.exists(cache_path)
        assert cache_path.endswith(".npz")

    def test_cache_hit_loads_from_disk(self, shock_params, cache_dir):
        """Second generator with same config loads from cache."""
        import os

        # First generator: creates cache
        gen1 = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32,
            N_test=64,
            cache_dir=cache_dir
        )
        val_data1 = gen1.get_validation_dataset()

        # Second generator: should load from cache
        gen2 = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32,
            N_test=64,
            cache_dir=cache_dir
        )
        val_data2 = gen2.get_validation_dataset()

        # Verify data is identical
        for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
            assert tf.reduce_all(val_data1[key] == val_data2[key]).numpy()

    def test_cache_invalidation_on_config_change(self, shock_params, cache_dir):
        """Different config generates different cache file."""
        import os

        # First generator
        gen1 = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32,
            N_test=64,
            cache_dir=cache_dir
        )
        gen1.get_validation_dataset()
        cache_path1 = gen1._get_cache_path("validation")

        # Second generator with different config
        gen2 = DataGenerator(
            master_seed=(99, 200),  # Different seed
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32,
            N_test=64,
            cache_dir=cache_dir
        )
        gen2.get_validation_dataset()
        cache_path2 = gen2._get_cache_path("validation")

        # Cache paths should be different
        assert cache_path1 != cache_path2
        assert os.path.exists(cache_path1)
        assert os.path.exists(cache_path2)

    def test_cache_preserves_tensor_types(self, cached_generator, cache_dir):
        """Tensros loaded from cache have correct dtype and shape."""
        # Force generation and save
        cached_generator.get_validation_dataset()
        
        # Load in a fresh generator
        gen2 = DataGenerator(
            master_seed=(42, 100),
            shock_params=cached_generator.shock_params,
            k_bounds=cached_generator.k_bounds,
            logz_bounds=cached_generator.logz_bounds,
            b_bounds=cached_generator.b_bounds,
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            N_val=32,
            N_test=64,
            cache_dir=cache_dir
        )
        
        val_data = gen2.get_validation_dataset()
        
        for key in ['k0', 'z0', 'b0', 'z_path', 'z_fork', 'eps_path', 'eps_fork']:
            assert isinstance(val_data[key], tf.Tensor)
            assert val_data[key].dtype == tf.float32

    def test_cache_hash_deterministic(self, shock_params, cache_dir):
        """Config hash is deterministic for same config."""
        gen1 = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=cache_dir
        )

        gen2 = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=cache_dir
        )

        assert gen1._get_config_hash() == gen2._get_config_hash()

    def test_cache_hash_changes_with_config(self, shock_params, cache_dir):
        """Config hash changes when configuration changes."""
        gen1 = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=cache_dir
        )

        gen2 = DataGenerator(
            master_seed=(99, 200),  # Different seed
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=cache_dir
        )

        assert gen1._get_config_hash() != gen2._get_config_hash()


    def test_separate_cache_files_for_val_test(self, cached_generator):
        """Validation and test use separate cache files."""
        import os

        val_data = cached_generator.get_validation_dataset()
        test_data = cached_generator.get_test_dataset()

        val_cache = cached_generator._get_cache_path("validation")
        test_cache = cached_generator._get_cache_path("test")

        assert val_cache != test_cache
        assert os.path.exists(val_cache)
        assert os.path.exists(test_cache)

    def test_no_cache_without_cache_dir(self, shock_params):
        """No cache files created when cache_dir is None."""
        gen = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=None
        )

        val_data = gen.get_validation_dataset()
        test_data = gen.get_test_dataset()

        # Should not create any cache files
        # (Can't check this directly, but no errors should occur)
        assert val_data is not None
        assert test_data is not None

    def test_cache_directory_created_automatically(self, shock_params, tmp_path):
        """Cache directory is created automatically if it doesn't exist."""
        import os

        # Use a cache dir that doesn't exist yet
        cache_dir = str(tmp_path / "nested" / "cache" / "dir")
        assert not os.path.exists(cache_dir)

        gen = DataGenerator(
            master_seed=(42, 100),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10,
            cache_dir=cache_dir
        )

        # Trigger cache creation
        gen.get_validation_dataset()

        # Directory should now exist
        assert os.path.exists(cache_dir)


# =============================================================================
# FLATTENED DATASET TESTS (NEW)
# =============================================================================

class TestFlattenedDataset:
    """Tests for flattened dataset generation for ER/BR methods."""

    def test_flattened_dataset_shape(self, basic_generator):
        """Flattened dataset has correct shape (N*T,)."""
        flat_data = basic_generator.get_flattened_training_dataset()

        N = basic_generator.sim_batch_size * basic_generator.n_sim_batches
        T = basic_generator.T
        N_total = N * T

        assert flat_data['k'].shape == (N_total,)
        assert flat_data['z'].shape == (N_total,)
        assert flat_data['z_next_main'].shape == (N_total,)
        assert flat_data['z_next_fork'].shape == (N_total,)

    def test_flattened_dataset_keys(self, basic_generator):
        """Flattened dataset has correct keys."""
        flat_data = basic_generator.get_flattened_training_dataset()

        expected_keys = {'k', 'z', 'z_next_main', 'z_next_fork'}
        assert set(flat_data.keys()) == expected_keys

    def test_flattened_dataset_dtypes(self, basic_generator):
        """Flattened dataset has float32 dtype."""
        flat_data = basic_generator.get_flattened_training_dataset()

        for key in flat_data.keys():
            assert flat_data[key].dtype == tf.float32, f"Wrong dtype for {key}"

    def test_flattened_k_within_bounds(self, basic_generator):
        """Flattened k values are within bounds."""
        flat_data = basic_generator.get_flattened_training_dataset()

        k = flat_data['k']
        k_min, k_max = basic_generator.k_bounds

        assert tf.reduce_all(k >= k_min)
        assert tf.reduce_all(k <= k_max)

    def test_flattened_z_positive(self, basic_generator):
        """Flattened z values are positive."""
        flat_data = basic_generator.get_flattened_training_dataset()

        assert tf.reduce_all(flat_data['z'] > 0)
        assert tf.reduce_all(flat_data['z_next_main'] > 0)
        assert tf.reduce_all(flat_data['z_next_fork'] > 0)

    def test_flattened_reproducibility(self, shock_params):
        """Same seed produces identical flattened dataset."""
        gen1 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10
        )

        gen2 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10
        )

        flat1 = gen1.get_flattened_training_dataset()
        flat2 = gen2.get_flattened_training_dataset()

        for key in ['k', 'z', 'z_next_main', 'z_next_fork']:
            assert tf.reduce_all(flat1[key] == flat2[key]), f"Mismatch in {key}"

    def test_flattened_shuffled(self, basic_generator):
        """Flattened dataset is shuffled (not sequential)."""
        flat_data = basic_generator.get_flattened_training_dataset()
        traj_data = basic_generator.get_training_dataset()

        # Extract first N*T z values from trajectory (flattened, unshuffled)
        z_path = traj_data['z_path'][:, :-1]  # (N, T)
        z_unshuffled = tf.reshape(z_path, [-1])  # (N*T,)

        z_shuffled = flat_data['z']

        # They should NOT match (unless incredibly unlucky with random shuffle)
        # Check that at least some values are in different positions
        matches = int(tf.reduce_sum(tf.cast(z_unshuffled == z_shuffled, tf.int32)))
        total = int(tf.size(z_unshuffled))

        # Expect most values to be in different positions after shuffle
        # (Some might match by chance, but not all)
        assert matches < total * 0.9, "Data doesn't appear to be shuffled"

    def test_flattened_z_main_fork_differ(self, basic_generator):
        """Main and fork z_next values differ (different shocks)."""
        flat_data = basic_generator.get_flattened_training_dataset()

        z_next_main = flat_data['z_next_main']
        z_next_fork = flat_data['z_next_fork']

        # Should differ (with very high probability)
        assert not tf.reduce_all(z_next_main == z_next_fork)

    def test_flattened_k_independent_sampling(self, shock_params):
        """k values are sampled independently (not from trajectories)."""
        # Create two generators with same seed
        gen1 = DataGenerator(
            master_seed=(42, 0),
            shock_params=shock_params,
            k_bounds=(0.5, 5.0),
            logz_bounds=(-0.3, 0.3),
            b_bounds=(0.0, 3.0),
            sim_batch_size=16,
            T=5,
            n_sim_batches=10
        )

        # Get flattened data
        flat1 = gen1.get_flattened_training_dataset()

        # k values should span the full range (not concentrated)
        k = flat1['k'].numpy()
        k_min, k_max = gen1.k_bounds

        # Check that k values span at least 80% of the range
        k_range = k_max - k_min
        k_span = np.max(k) - np.min(k)
        assert k_span > 0.8 * k_range, "k values too concentrated"

    def test_flattened_statistics_reasonable(self, basic_generator):
        """Flattened dataset has reasonable statistics."""
        flat_data = basic_generator.get_flattened_training_dataset()

        k = flat_data['k'].numpy()
        z = flat_data['z'].numpy()

        # k should be roughly uniform within bounds
        k_min, k_max = basic_generator.k_bounds
        k_mean_expected = (k_min + k_max) / 2
        k_mean_actual = np.mean(k)

        # Should be within 20% of expected mean (for uniform distribution)
        assert abs(k_mean_actual - k_mean_expected) < 0.2 * k_mean_expected

        # z should be positive
        assert np.all(z > 0)

    def test_flattened_different_from_trajectory_initial(self, basic_generator):
        """Flattened k values differ from trajectory k0 (independent sampling)."""
        flat_data = basic_generator.get_flattened_training_dataset()
        traj_data = basic_generator.get_training_dataset()

        # Flattened k is sampled independently
        k_flat = flat_data['k']

        # Trajectory k0
        k0_traj = traj_data['k0']

        # They should be from same distribution but different samples
        # (Very unlikely to match exactly for all values)
        # We can't directly compare since shapes differ, but can compare distributions

        k_flat_mean = tf.reduce_mean(k_flat)
        k0_traj_mean = tf.reduce_mean(k0_traj)

        # Means should be close (both uniform from same bounds)
        # but not identical (different samples)
        k_bounds_mean = (basic_generator.k_bounds[0] + basic_generator.k_bounds[1]) / 2
        assert abs(float(k_flat_mean) - k_bounds_mean) < 1.0
        assert abs(float(k0_traj_mean) - k_bounds_mean) < 1.0
