"""
tests/economy/test_rng.py

Comprehensive tests for the stateless RNG seed scheduling module.

Test categories:
1. Configuration validation
2. Seed generation (formula verification)
3. Determinism
4. Disjointness
5. Edge cases
6. Integration with TensorFlow stateless functions
"""

import pytest
import tensorflow as tf
import numpy as np

from src.economy.rng import (
    SeedSchedule,
    SeedScheduleConfig,
    VariableID,
    _is_valid_int32,
    _wrap_to_int32,
)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_valid_int32_true(self):
        """Valid int32 values return True."""
        assert _is_valid_int32(0)
        assert _is_valid_int32(42)
        assert _is_valid_int32(-42)
        assert _is_valid_int32(2**31 - 1)  # Max int32
        assert _is_valid_int32(-(2**31))   # Min int32

    def test_is_valid_int32_false(self):
        """Invalid int32 values return False."""
        assert not _is_valid_int32(2**31)      # Overflow
        assert not _is_valid_int32(-(2**31) - 1)  # Underflow
        assert not _is_valid_int32(2**32)

    def test_wrap_to_int32_no_overflow(self):
        """Values within int32 range are unchanged."""
        assert _wrap_to_int32(0) == 0
        assert _wrap_to_int32(42) == 42
        assert _wrap_to_int32(-42) == -42
        assert _wrap_to_int32(2**31 - 1) == 2**31 - 1
        assert _wrap_to_int32(-(2**31)) == -(2**31)

    def test_wrap_to_int32_overflow(self):
        """Overflow wraps correctly."""
        # 2^31 overflows to -2^31
        assert _wrap_to_int32(2**31) == -(2**31)
        # 2^31 + 1 wraps to -2^31 + 1
        assert _wrap_to_int32(2**31 + 1) == -(2**31) + 1

    def test_wrap_to_int32_underflow(self):
        """Underflow wraps correctly."""
        # -(2^31) - 1 wraps to 2^31 - 1
        assert _wrap_to_int32(-(2**31) - 1) == 2**31 - 1


# =============================================================================
# CONFIGURATION VALIDATION TESTS
# =============================================================================

class TestSeedScheduleConfig:
    """Tests for SeedScheduleConfig validation."""

    def test_valid_config_creation(self):
        """Valid config creates successfully."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        assert config.master_seed == (42, 0)
        assert config.train_offset == 100
        assert config.val_offset == 200
        assert config.test_offset == 300

    def test_valid_config_with_custom_offsets(self):
        """Config with custom offsets works."""
        config = SeedScheduleConfig(
            master_seed=(42, 0),
            train_offset=1000,
            val_offset=2000,
            test_offset=3000
        )
        assert config.train_offset == 1000

    def test_invalid_master_seed_m0_raises(self):
        """Invalid master_seed[0] raises ValueError."""
        with pytest.raises(ValueError, match="master_seed\\[0\\].*int32"):
            SeedScheduleConfig(master_seed=(2**31, 0))

    def test_invalid_master_seed_m1_raises(self):
        """Invalid master_seed[1] raises ValueError."""
        with pytest.raises(ValueError, match="master_seed\\[1\\].*int32"):
            SeedScheduleConfig(master_seed=(42, 2**31))

    def test_duplicate_train_val_offsets_raises(self):
        """Duplicate train and val offsets raise ValueError."""
        with pytest.raises(ValueError, match="distinct"):
            SeedScheduleConfig(
                master_seed=(42, 0),
                train_offset=100,
                val_offset=100  # Duplicate!
            )

    def test_duplicate_val_test_offsets_raises(self):
        """Duplicate val and test offsets raise ValueError."""
        with pytest.raises(ValueError, match="distinct"):
            SeedScheduleConfig(
                master_seed=(42, 0),
                val_offset=200,
                test_offset=200  # Duplicate!
            )

    def test_overflow_detection_train(self):
        """Config detects potential int32 overflow for train offset."""
        with pytest.raises(ValueError, match="train_offset.*overflow"):
            SeedScheduleConfig(
                master_seed=(2**30, 0),
                train_offset=2**30  # Will overflow
            )

    def test_overflow_detection_val(self):
        """Config detects potential int32 overflow for val offset."""
        with pytest.raises(ValueError, match="val_offset.*overflow"):
            SeedScheduleConfig(
                master_seed=(2**30, 0),
                val_offset=2**30  # Will overflow
            )

    def test_invalid_n_train_steps_raises(self):
        """Invalid n_train_steps raises ValueError."""
        with pytest.raises(ValueError, match="n_train_steps"):
            SeedScheduleConfig(master_seed=(42, 0), n_train_steps=0)

        with pytest.raises(ValueError, match="n_train_steps"):
            SeedScheduleConfig(master_seed=(42, 0), n_train_steps=-10)


# =============================================================================
# SEED SCHEDULE INITIALIZATION TESTS
# =============================================================================

class TestSeedScheduleInit:
    """Tests for SeedSchedule initialization."""

    def test_init_success(self):
        """SeedSchedule initializes with valid config."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)
        assert schedule.config == config

    def test_init_overflow_validation(self):
        """SeedSchedule validates step overflow at init."""
        config = SeedScheduleConfig(
            master_seed=(0, 2**30),
            n_train_steps=2**30  # Will overflow
        )
        with pytest.raises(ValueError, match="overflow"):
            SeedSchedule(config)


# =============================================================================
# SEED GENERATION TESTS
# =============================================================================

class TestSeedGeneration:
    """Tests for seed generation formulas."""

    @pytest.fixture
    def schedule(self):
        """Create a basic schedule for testing."""
        config = SeedScheduleConfig(master_seed=(42, 100), n_train_steps=10)
        return SeedSchedule(config)

    def test_seed_shape_and_dtype(self, schedule):
        """Seeds have correct shape and dtype."""
        val_seeds = schedule.get_val_seeds()

        for var_id, seed in val_seeds.items():
            assert seed.shape == (2,), f"Wrong shape for {var_id}"
            assert seed.dtype == tf.int32, f"Wrong dtype for {var_id}"

    def test_train_seed_formula_step1_k0(self, schedule):
        """Training seed for step 1, K0 follows exact formula."""
        # s_train(j=1, K0) = (m0 + 100 + 1, m1 + 1)
        #                   = (42 + 100 + 1, 100 + 1)
        #                   = (143, 101)
        train_seeds = schedule.get_train_seeds(steps=[1])
        k0_seed = train_seeds[1][VariableID.K0]

        expected = [143, 101]
        assert k0_seed.numpy().tolist() == expected

    def test_train_seed_formula_step1_z0(self, schedule):
        """Training seed for step 1, Z0 follows exact formula."""
        # s_train(j=1, Z0) = (m0 + 100 + 2, m1 + 1)
        #                   = (42 + 100 + 2, 100 + 1)
        #                   = (144, 101)
        train_seeds = schedule.get_train_seeds(steps=[1])
        z0_seed = train_seeds[1][VariableID.Z0]

        expected = [144, 101]
        assert z0_seed.numpy().tolist() == expected

    def test_train_seed_formula_step2_k0(self, schedule):
        """Training seed for step 2, K0 follows exact formula."""
        # s_train(j=2, K0) = (m0 + 100 + 1, m1 + 2)
        #                   = (42 + 100 + 1, 100 + 2)
        #                   = (143, 102)
        train_seeds = schedule.get_train_seeds(steps=[2])
        k0_seed = train_seeds[2][VariableID.K0]

        expected = [143, 102]
        assert k0_seed.numpy().tolist() == expected

    def test_train_seed_formula_all_variables(self, schedule):
        """Training seeds for step 1 include all variables with correct values."""
        train_seeds = schedule.get_train_seeds(steps=[1])
        step1 = train_seeds[1]

        # m0=42, m1=100, train_offset=100, step=1
        # Base: (42 + 100, 100 + 1) = (142, 101)
        expected = {
            VariableID.K0: [142 + 1, 101],    # var_id=1
            VariableID.Z0: [142 + 2, 101],    # var_id=2
            VariableID.B0: [142 + 3, 101],    # var_id=3
            VariableID.EPS1: [142 + 4, 101],  # var_id=4
            VariableID.EPS2: [142 + 5, 101],  # var_id=5
        }

        for var_id, expected_seed in expected.items():
            actual = step1[var_id].numpy().tolist()
            assert actual == expected_seed, f"Mismatch for {var_id}"

    def test_val_seed_formula_k0(self, schedule):
        """Validation seed for K0 follows exact formula."""
        # s_val(K0) = (m0 + 200 + 1, m1 + 0)
        #           = (42 + 200 + 1, 100 + 0)
        #           = (243, 100)
        val_seeds = schedule.get_val_seeds()
        k0_seed = val_seeds[VariableID.K0]

        expected = [243, 100]
        assert k0_seed.numpy().tolist() == expected

    def test_val_seed_formula_all_variables(self, schedule):
        """Validation seeds include all variables with correct values."""
        val_seeds = schedule.get_val_seeds()

        # m0=42, m1=100, val_offset=200, step=0
        # Base: (42 + 200, 100 + 0) = (242, 100)
        expected = {
            VariableID.K0: [242 + 1, 100],    # var_id=1
            VariableID.Z0: [242 + 2, 100],    # var_id=2
            VariableID.B0: [242 + 3, 100],    # var_id=3
            VariableID.EPS1: [242 + 4, 100],  # var_id=4
            VariableID.EPS2: [242 + 5, 100],  # var_id=5
        }

        for var_id, expected_seed in expected.items():
            actual = val_seeds[var_id].numpy().tolist()
            assert actual == expected_seed, f"Mismatch for {var_id}"

    def test_test_seed_formula_k0(self, schedule):
        """Test seed for K0 follows exact formula."""
        # s_test(K0) = (m0 + 300 + 1, m1 + 0)
        #            = (42 + 300 + 1, 100 + 0)
        #            = (343, 100)
        test_seeds = schedule.get_test_seeds()
        k0_seed = test_seeds[VariableID.K0]

        expected = [343, 100]
        assert k0_seed.numpy().tolist() == expected

    def test_test_seed_formula_all_variables(self, schedule):
        """Test seeds include all variables with correct values."""
        test_seeds = schedule.get_test_seeds()

        # m0=42, m1=100, test_offset=300, step=0
        # Base: (42 + 300, 100 + 0) = (342, 100)
        expected = {
            VariableID.K0: [342 + 1, 100],    # var_id=1
            VariableID.Z0: [342 + 2, 100],    # var_id=2
            VariableID.B0: [342 + 3, 100],    # var_id=3
            VariableID.EPS1: [342 + 4, 100],  # var_id=4
            VariableID.EPS2: [342 + 5, 100],  # var_id=5
        }

        for var_id, expected_seed in expected.items():
            actual = test_seeds[var_id].numpy().tolist()
            assert actual == expected_seed, f"Mismatch for {var_id}"

    def test_get_single_seed_train(self, schedule):
        """get_single_seed returns correct training seed."""
        seed = schedule.get_single_seed("train", VariableID.K0, step=5)
        # s_train(j=5, K0) = (42 + 100 + 1, 100 + 5) = (143, 105)
        expected = [143, 105]
        assert seed.numpy().tolist() == expected

    def test_get_single_seed_val(self, schedule):
        """get_single_seed returns correct validation seed."""
        seed = schedule.get_single_seed("val", VariableID.EPS1)
        # s_val(EPS1) = (42 + 200 + 4, 100 + 0) = (246, 100)
        expected = [246, 100]
        assert seed.numpy().tolist() == expected

    def test_get_train_seeds_multiple_steps(self, schedule):
        """get_train_seeds returns seeds for multiple steps."""
        train_seeds = schedule.get_train_seeds(steps=[1, 2, 3])

        assert len(train_seeds) == 3
        assert 1 in train_seeds
        assert 2 in train_seeds
        assert 3 in train_seeds

        # Verify step 3, K0
        k0_seed_step3 = train_seeds[3][VariableID.K0]
        expected = [143, 103]  # (42 + 100 + 1, 100 + 3)
        assert k0_seed_step3.numpy().tolist() == expected

    def test_get_train_seeds_subset_variables(self, schedule):
        """get_train_seeds with variable subset works."""
        train_seeds = schedule.get_train_seeds(
            steps=[1],
            variables=[VariableID.K0, VariableID.Z0]
        )

        step1 = train_seeds[1]
        assert len(step1) == 2
        assert VariableID.K0 in step1
        assert VariableID.Z0 in step1
        assert VariableID.B0 not in step1

    def test_get_train_seeds_all_steps(self, schedule):
        """get_train_seeds without steps argument returns all steps."""
        train_seeds = schedule.get_train_seeds()

        # Should have steps 1..10 (n_train_steps=10)
        assert len(train_seeds) == 10
        for step in range(1, 11):
            assert step in train_seeds

    def test_get_train_seeds_all_steps_without_n_train_steps_raises(self):
        """get_train_seeds without steps fails if n_train_steps not set."""
        config = SeedScheduleConfig(master_seed=(42, 0))  # No n_train_steps
        schedule = SeedSchedule(config)

        with pytest.raises(ValueError, match="n_train_steps not configured"):
            schedule.get_train_seeds()

    def test_get_train_seeds_invalid_step_raises(self, schedule):
        """get_train_seeds with invalid step raises ValueError."""
        with pytest.raises(ValueError, match="Step must be >= 0"):
            schedule.get_train_seeds(steps=[-1])

    def test_get_train_seeds_exceeds_max_step_raises(self, schedule):
        """get_train_seeds with step > n_train_steps raises ValueError."""
        with pytest.raises(ValueError, match="exceeds configured max"):
            schedule.get_train_seeds(steps=[100])  # n_train_steps=10


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_config_same_seeds(self):
        """Same config produces identical seeds."""
        config = SeedScheduleConfig(master_seed=(42, 0), n_train_steps=3)

        schedule1 = SeedSchedule(config)
        schedule2 = SeedSchedule(config)

        seeds1 = schedule1.get_train_seeds(steps=[1, 2, 3])
        seeds2 = schedule2.get_train_seeds(steps=[1, 2, 3])

        for step in [1, 2, 3]:
            for var_id in VariableID:
                assert tf.reduce_all(seeds1[step][var_id] == seeds2[step][var_id])

    def test_val_seeds_deterministic(self):
        """Validation seeds are deterministic."""
        config = SeedScheduleConfig(master_seed=(42, 0))

        schedule1 = SeedSchedule(config)
        schedule2 = SeedSchedule(config)

        val1 = schedule1.get_val_seeds()
        val2 = schedule2.get_val_seeds()

        for var_id in VariableID:
            assert tf.reduce_all(val1[var_id] == val2[var_id])

    def test_test_seeds_deterministic(self):
        """Test seeds are deterministic."""
        config = SeedScheduleConfig(master_seed=(42, 0))

        schedule1 = SeedSchedule(config)
        schedule2 = SeedSchedule(config)

        test1 = schedule1.get_test_seeds()
        test2 = schedule2.get_test_seeds()

        for var_id in VariableID:
            assert tf.reduce_all(test1[var_id] == test2[var_id])

    def test_stateless_sampling_deterministic(self):
        """Stateless TensorFlow sampling is deterministic."""
        seed = tf.constant([42, 0], dtype=tf.int32)

        sample1 = tf.random.stateless_uniform(
            shape=(100,), seed=seed, minval=0.0, maxval=1.0
        )
        sample2 = tf.random.stateless_uniform(
            shape=(100,), seed=seed, minval=0.0, maxval=1.0
        )

        assert tf.reduce_all(sample1 == sample2)


# =============================================================================
# DISJOINTNESS TESTS
# =============================================================================

class TestDisjointness:
    """Tests for disjointness of train/val/test seeds."""

    def test_train_val_test_seeds_disjoint(self):
        """Train/val/test seeds are completely disjoint."""
        config = SeedScheduleConfig(master_seed=(42, 0), n_train_steps=100)
        schedule = SeedSchedule(config)

        train_seeds = schedule.get_train_seeds()
        val_seeds = schedule.get_val_seeds()
        test_seeds = schedule.get_test_seeds()

        # Collect all seed pairs
        all_seeds = set()

        # Add training seeds
        for step_seeds in train_seeds.values():
            for seed in step_seeds.values():
                seed_tuple = tuple(seed.numpy())
                assert seed_tuple not in all_seeds, "Duplicate seed in training!"
                all_seeds.add(seed_tuple)

        # Add validation seeds (should not collide)
        for seed in val_seeds.values():
            seed_tuple = tuple(seed.numpy())
            assert seed_tuple not in all_seeds, "Val seed collides with train!"
            all_seeds.add(seed_tuple)

        # Add test seeds (should not collide)
        for seed in test_seeds.values():
            seed_tuple = tuple(seed.numpy())
            assert seed_tuple not in all_seeds, "Test seed collides with train/val!"

    def test_different_variables_different_seeds(self):
        """Different variables get different seeds at same step."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        train_seeds = schedule.get_train_seeds(steps=[1])
        step1_seeds = train_seeds[1]

        # All seeds should be unique
        seed_set = set()
        for var_id in VariableID:
            seed_tuple = tuple(step1_seeds[var_id].numpy())
            assert seed_tuple not in seed_set, f"Duplicate seed for {var_id}"
            seed_set.add(seed_tuple)

    def test_different_steps_different_seeds(self):
        """Different steps get different seeds for same variable."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        train_seeds = schedule.get_train_seeds(steps=[1, 2, 3])

        for var_id in VariableID:
            seeds = [train_seeds[s][var_id] for s in [1, 2, 3]]

            # All should be different (second component differs)
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    assert not tf.reduce_all(seeds[i] == seeds[j])

    def test_train_val_first_component_differs(self):
        """Train and val seeds differ in first component by at least 100."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        train_seeds = schedule.get_train_seeds(steps=[1])
        val_seeds = schedule.get_val_seeds()

        train_k0 = train_seeds[1][VariableID.K0].numpy()[0]  # 42 + 100 + 1 = 143
        val_k0 = val_seeds[VariableID.K0].numpy()[0]        # 42 + 200 + 1 = 243

        # Difference should be exactly 100 (val_offset - train_offset)
        assert abs(val_k0 - train_k0) == 100


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_int32_overflow_handling(self):
        """Int32 overflow is handled gracefully via wrapping."""
        # Use large master seed near boundary
        config = SeedScheduleConfig(master_seed=(2**30, 2**30 - 1000))
        schedule = SeedSchedule(config)

        # Should not raise
        seeds = schedule.get_train_seeds(steps=[1])

        # Seeds should be valid int32
        for var_id in VariableID:
            seed = seeds[1][var_id]
            assert seed.dtype == tf.int32
            # Verify values are in int32 range
            for component in seed.numpy():
                assert _is_valid_int32(int(component))

    def test_negative_master_seed(self):
        """Negative master seeds work correctly."""
        config = SeedScheduleConfig(master_seed=(-100, -50))
        schedule = SeedSchedule(config)

        seeds = schedule.get_val_seeds()
        assert all(s.dtype == tf.int32 for s in seeds.values())

        # Verify formula for K0: (-100 + 200 + 1, -50 + 0) = (101, -50)
        k0_seed = seeds[VariableID.K0].numpy()
        assert k0_seed[0] == 101
        assert k0_seed[1] == -50

    def test_zero_master_seed(self):
        """Zero master seed is valid."""
        config = SeedScheduleConfig(master_seed=(0, 0))
        schedule = SeedSchedule(config)

        seeds = schedule.get_val_seeds()
        k0_seed = seeds[VariableID.K0]

        # s_val(K0) = (0 + 200 + 1, 0 + 0) = (201, 0)
        expected = [201, 0]
        assert k0_seed.numpy().tolist() == expected

    def test_large_step_count(self):
        """Large step counts work correctly."""
        config = SeedScheduleConfig(master_seed=(42, 0), n_train_steps=10000)
        schedule = SeedSchedule(config)

        # Should handle step 9999 without issues
        seeds = schedule.get_train_seeds(steps=[9999])
        assert 9999 in seeds

        # Verify formula for step 9999, K0: (42 + 100 + 1, 0 + 9999) = (143, 9999)
        k0_seed = seeds[9999][VariableID.K0].numpy()
        assert k0_seed[0] == 143
        assert k0_seed[1] == 9999

    def test_maximum_int32_master_seed(self):
        """Maximum valid int32 master seed works."""
        max_int32 = 2**31 - 1
        # Use smaller offsets to avoid overflow
        config = SeedScheduleConfig(
            master_seed=(max_int32 - 500, max_int32 - 500),
            train_offset=10,
            val_offset=20,
            test_offset=30
        )
        schedule = SeedSchedule(config)

        # Should not raise
        val_seeds = schedule.get_val_seeds()
        assert all(s.dtype == tf.int32 for s in val_seeds.values())

    def test_minimum_int32_master_seed(self):
        """Minimum valid int32 master seed works."""
        min_int32 = -(2**31)
        config = SeedScheduleConfig(master_seed=(min_int32, min_int32))
        schedule = SeedSchedule(config)

        # Should not raise
        val_seeds = schedule.get_val_seeds()
        assert all(s.dtype == tf.int32 for s in val_seeds.values())


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with TensorFlow stateless functions."""

    def test_stateless_uniform_sampling(self):
        """Integration with tf.random.stateless_uniform."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        val_seeds = schedule.get_val_seeds()

        # Sample capital
        k = tf.random.stateless_uniform(
            shape=(100,),
            seed=val_seeds[VariableID.K0],
            minval=0.5,
            maxval=5.0,
            dtype=tf.float32
        )

        assert k.shape == (100,)
        assert tf.reduce_all(k >= 0.5)
        assert tf.reduce_all(k <= 5.0)

    def test_stateless_normal_sampling(self):
        """Integration with tf.random.stateless_normal."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        val_seeds = schedule.get_val_seeds()

        # Sample shocks
        eps = tf.random.stateless_normal(
            shape=(10000,),
            seed=val_seeds[VariableID.EPS1],
            dtype=tf.float32
        )

        # Check approximate standard normal properties
        mean = tf.reduce_mean(eps)
        std = tf.math.reduce_std(eps)

        # Should be close to N(0, 1)
        assert abs(mean.numpy()) < 0.05  # Close to 0
        assert abs(std.numpy() - 1.0) < 0.05  # Close to 1

    def test_reproducible_across_calls(self):
        """Seeds produce same samples across multiple calls."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        seed = schedule.get_single_seed("val", VariableID.K0)

        sample1 = tf.random.stateless_uniform(
            shape=(100,), seed=seed, minval=0.0, maxval=1.0
        )
        sample2 = tf.random.stateless_uniform(
            shape=(100,), seed=seed, minval=0.0, maxval=1.0
        )

        assert tf.reduce_all(sample1 == sample2)

    def test_different_seeds_different_samples(self):
        """Different seeds produce different samples."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        val_seeds = schedule.get_val_seeds()

        sample_k0 = tf.random.stateless_uniform(
            shape=(100,),
            seed=val_seeds[VariableID.K0],
            minval=0.0,
            maxval=1.0
        )
        sample_z0 = tf.random.stateless_uniform(
            shape=(100,),
            seed=val_seeds[VariableID.Z0],
            minval=0.0,
            maxval=1.0
        )

        # Should be different (with very high probability)
        assert not tf.reduce_all(sample_k0 == sample_z0)

    def test_train_val_test_produce_different_samples(self):
        """Train/val/test seeds produce different samples."""
        config = SeedScheduleConfig(master_seed=(42, 0))
        schedule = SeedSchedule(config)

        train_seeds = schedule.get_train_seeds(steps=[1])
        val_seeds = schedule.get_val_seeds()
        test_seeds = schedule.get_test_seeds()

        train_sample = tf.random.stateless_uniform(
            shape=(100,),
            seed=train_seeds[1][VariableID.K0],
            minval=0.0,
            maxval=1.0
        )
        val_sample = tf.random.stateless_uniform(
            shape=(100,),
            seed=val_seeds[VariableID.K0],
            minval=0.0,
            maxval=1.0
        )
        test_sample = tf.random.stateless_uniform(
            shape=(100,),
            seed=test_seeds[VariableID.K0],
            minval=0.0,
            maxval=1.0
        )

        # All should be different
        assert not tf.reduce_all(train_sample == val_sample)
        assert not tf.reduce_all(val_sample == test_sample)
        assert not tf.reduce_all(train_sample == test_sample)
