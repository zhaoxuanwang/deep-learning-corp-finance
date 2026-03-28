"""
src/economy/data_generator.py

Reproducible data generation for training/validation/test using TensorFlow stateless RNG.

This module generates synthetic datasets for the corporate finance model, ensuring:
- Full reproducibility: identical master seed + inputs -> identical datasets
- Common random numbers: different methods see identical draws for same split/step/variable
- No leakage: train/val/test use disjoint RNG streams (enforced by seed schedule)
- Stateless RNG only: all randomness via tf.random.stateless_* functions

Key Features:
- Uses SeedSchedule from rng.py for deterministic seed management
- Generates training batches (J steps), validation dataset, test dataset
- Includes full z-path rollout using AR(1) transition
- All operations vectorized over batch dimension
- Provides TWO training dataset formats: trajectories (for LR) and flattened (for ER/BR)

Data Representation:
- log(z) is drawn uniformly in log space, then z = exp(log_z)
- All datasets include b0 (ignored by basic model if not used)
- z_path includes z0 at index 0: shape (n, T+1)

TWO TRAINING DATASET FORMATS:
=============================

1. TRAJECTORY DATA (for Lifetime Reward / LR method):
   - Use: get_training_dataset() or get_training_batches()
   - Structure: Full trajectories with shape (N, T+1) for z_path
   - Contains: k0, z_path, z_fork, eps_path, eps_fork
   - Purpose: Compute discounted sum of rewards over horizon T
   - Reference: report_brief.md lines 407-456 (LR Method)

2. FLATTENED DATA (for Euler Residual / ER and Bellman Residual / BR methods):
   - Use: get_flattened_training_dataset()
   - Structure: Individual i.i.d. transitions with shape (N*T,)
   - Contains: k, z, z_next_main, z_next_fork
   - Purpose: One-step policy optimization with i.i.d. samples for SGD
   - Reference: report_brief.md lines 157-167 (Flatten Data for ER and BR)

Why Two Formats?
- LR requires full trajectories to compute lifetime rewards Σ β^t e_t
- ER/BR operate on individual transitions to minimize residuals E[f^2]
- Flattening + shuffling ensures i.i.d. property for gradient descent

Example:
--------
    from src.economy import DataGenerator, EconomicParams

    # Create generator
    params = EconomicParams()
    generator = DataGenerator(
        master_seed=(42, 0),
        params=params,
        k_bounds=(0.5, 5.0),
        logz_bounds=(-0.3, 0.3),
        b_bounds=(0.0, 5.0),
        sim_batch_size=128,
        horizon=10,
        n_sim_batches=1000,
        N_val=1280,
        N_test=6400
    )

    # === FOR LR METHOD ===
    # Use trajectory data
    for step, batch in enumerate(generator.get_training_batches(), start=1):
        k0 = batch['k0']          # (128,)
        z_path = batch['z_path']  # (128, 11) - includes z0
        # ... LR training code ...

    # === FOR ER/BR METHODS ===
    # Use flattened data
    flat_data = generator.get_flattened_training_dataset()
    k = flat_data['k']                    # (N*T,) independent samples
    z = flat_data['z']                    # (N*T,)
    z_next_main = flat_data['z_next_main']  # (N*T,)
    z_next_fork = flat_data['z_next_fork']  # (N*T,)
    # ... ER/BR training code ...

    # Validation dataset (trajectory format)
    val_data = generator.get_validation_dataset()

    # Test dataset (trajectory format, sealed accessor)
    test_data = generator.get_test_dataset()
"""

from __future__ import annotations

import os
from pathlib import Path
import tensorflow as tf
from typing import Dict, Iterator, Optional, Tuple, Any, List

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.rng import SeedSchedule, SeedScheduleConfig, VariableID
from src.economy.data import (
    default_cache_dir,
    compute_config_hash,
    build_cache_path,
    build_metadata,
    save_dataset_to_disk,
    load_dataset_from_disk,
    generate_initial_states,
    generate_shocks,
    rollout_forked_path,
    generate_batch,
    build_flattened_dataset,
)


# =============================================================================
# DATA GENERATOR
# =============================================================================

class DataGenerator:
    """
    Reproducible data generator using TensorFlow stateless RNG.

    Generates synthetic datasets for training, validation, and test with:
    - Full reproducibility from master seed
    - Disjoint RNG streams for train/val/test
    - Complete z-path rollout over horizon T
    - Optional disk caching for val/test datasets

    All randomness uses tf.random.stateless_* functions with seeds from
    SeedSchedule. No stateful RNG or global seeds are used.

    Attributes:
        master_seed: Master seed pair (m0, m1)
        shock_params: Shock parameters (for AR(1) transition)
        k_bounds: Capital bounds (k_min, k_max)
        logz_bounds: Log productivity bounds (logz_min, logz_max)
        b_bounds: Debt bounds (b_min, b_max)
        sim_batch_size: Number of trajectories per simulation batch
        T: Rollout horizon
        n_sim_batches: Number of simulation batches (total samples = sim_batch_size * n_sim_batches)
        N_val: Validation dataset size (default: 10*sim_batch_size)
        N_test: Test dataset size (default: 50*sim_batch_size)
        cache_dir: Optional directory for caching val/test datasets to disk
    """

    def __init__(
        self,
        master_seed: Tuple[int, int],
        shock_params: ShockParams,
        k_bounds: Tuple[float, float],
        logz_bounds: Tuple[float, float],
        b_bounds: Tuple[float, float],
        sim_batch_size: int,
        T: int,
        n_sim_batches: int,
        N_val: Optional[int] = None,
        N_test: Optional[int] = None,
        cache_dir: Optional[str] = None, # If None, defaults to PROJECT_ROOT/data/interim
        save_to_disk: bool = True
    ):
        """
        Initialize data generator.

        Args:
            master_seed: Master seed pair (m0, m1) for RNG schedule
            shock_params: Shock parameters for AR(1) transition
            k_bounds: Capital bounds (k_min, k_max)
            logz_bounds: Log productivity bounds (logz_min, logz_max)
            b_bounds: Debt bounds (b_min, b_max)
            sim_batch_size: Simulation batch size (vectorization width)
            T: Rollout horizon (number of transitions)
            n_sim_batches: Number of simulation batches
            N_val: Validation size (default: 10*sim_batch_size)
            N_test: Test size (default: 50*sim_batch_size)
            cache_dir: Directory for caching datasets. If None, uses default standard path.
            save_to_disk: Whether to save generated datasets to disk (default: True)
        """
        self.master_seed = master_seed
        self.shock_params = shock_params
        self.k_bounds = k_bounds
        self.logz_bounds = logz_bounds
        self.b_bounds = b_bounds
        self.sim_batch_size = sim_batch_size
        self.T = T
        self.n_sim_batches = n_sim_batches
        self.N_val = N_val if N_val is not None else 10 * sim_batch_size
        self.N_test = N_test if N_test is not None else 50 * sim_batch_size
        self.save_to_disk = save_to_disk

        if cache_dir is None:
            # Default to standard data directory: PROJECT_ROOT/data
            self.cache_dir = default_cache_dir(__file__)
        else:
            self.cache_dir = cache_dir

        # Create seed schedule
        config = SeedScheduleConfig(master_seed=master_seed, n_train_steps=n_sim_batches)
        self.seed_schedule = SeedSchedule(config)

        # Cache validation and test datasets on first access
        self._validation_dataset: Optional[Dict[str, tf.Tensor]] = None
        self._test_dataset: Optional[Dict[str, tf.Tensor]] = None
        self._training_dataset: Optional[Dict[str, tf.Tensor]] = None
        self._flattened_training_dataset: Optional[Dict[str, tf.Tensor]] = None
        self._flattened_validation_dataset: Optional[Dict[str, tf.Tensor]] = None
        # Separate caches for debt variants (include_debt=True)
        self._flattened_training_dataset_debt: Optional[Dict[str, tf.Tensor]] = None
        self._flattened_validation_dataset_debt: Optional[Dict[str, tf.Tensor]] = None

    def _get_config_hash(self) -> str:
        """
        Generate deterministic hash from generator configuration.

        Used to create unique cache filenames that change when configuration
        changes (master_seed, bounds, sizes, economic params).

        Returns:
            12-character hex string
        """
        return compute_config_hash(
            master_seed=self.master_seed,
            sim_batch_size=self.sim_batch_size,
            horizon=self.T,
            n_sim_batches=self.n_sim_batches,
            n_val=self.N_val,
            n_test=self.N_test,
            k_bounds=self.k_bounds,
            logz_bounds=self.logz_bounds,
            b_bounds=self.b_bounds,
            shock_params=self.shock_params,
        )

    def _get_cache_path(self, split: str) -> str:
        """
        Generate cache file path for a dataset split.

        Args:
            split: Dataset split name ("validation" or "test")

        Returns:
            Absolute path to cache file (.npz format)
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir is None, cannot generate cache path")

        config_hash = self._get_config_hash()
        return build_cache_path(self.cache_dir, split, config_hash)

    def _get_metadata(self) -> Dict[str, Any]:
        """
        Generate metadata for dataset verification.
        """
        return build_metadata(
            master_seed=self.master_seed,
            sim_batch_size=self.sim_batch_size,
            horizon=self.T,
            n_sim_batches=self.n_sim_batches,
            n_val=self.N_val,
            n_test=self.N_test,
            k_bounds=self.k_bounds,
            logz_bounds=self.logz_bounds,
            b_bounds=self.b_bounds,
            shock_params=self.shock_params,
        )

    def _save_to_disk(self, dataset: Dict[str, tf.Tensor], path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save dataset to disk in .npz format with metadata.

        Args:
            dataset: Dictionary of tensors to save
            path: File path for saving (.npz file)
            metadata: Optional dictionary of metadata to save
        """
        save_dataset_to_disk(
            dataset,
            path,
            save_to_disk=self.save_to_disk,
            metadata=metadata,
        )

    def _load_from_disk(self, path: str) -> Dict[str, tf.Tensor]:
        """
        Load dataset from disk (.npz format) and validate metadata.

        Args:
            path: File path to load (.npz file)

        Returns:
            Dictionary of tensors
        
        Raises:
            ValueError: If metadata does not match current generator configuration.
        """
        return load_dataset_from_disk(path, current_metadata=self._get_metadata())

    def _generate_initial_states(
        self,
        batch_size: int,
        k_seed: tf.Tensor,
        z_seed: tf.Tensor,
        b_seed: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate initial states (k0, z0, b0) using stateless RNG.

        Args:
            batch_size: Number of samples
            k_seed: Seed for k0 sampling (shape [2], dtype tf.int32)
            z_seed: Seed for z0 sampling (shape [2], dtype tf.int32)
            b_seed: Seed for b0 sampling (shape [2], dtype tf.int32)

        Returns:
            k0: Initial capital (batch_size,)
            z0: Initial productivity (batch_size,)
            b0: Initial debt (batch_size,)
        """
        return generate_initial_states(
            batch_size,
            self.k_bounds,
            self.logz_bounds,
            self.b_bounds,
            k_seed=k_seed,
            z_seed=z_seed,
            b_seed=b_seed,
        )

    def _generate_shocks(
        self,
        batch_size: int,
        T: int,
        eps1_seed: tf.Tensor,
        eps2_seed: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate shock sequences (eps1, eps2) using stateless RNG.

        Args:
            batch_size: Number of samples
            T: Horizon length
            eps1_seed: Seed for eps1 (shape [2], dtype tf.int32)
            eps2_seed: Seed for eps2 (shape [2], dtype tf.int32)

        Returns:
            eps1: First shock sequence (batch_size, T)
            eps2: Second shock sequence (batch_size, T)
        """
        return generate_shocks(
            batch_size,
            T,
            eps1_seed=eps1_seed,
            eps2_seed=eps2_seed,
        )

    def _rollout_forked_path(
        self,
        z0: tf.Tensor,
        eps_main: tf.Tensor,
        eps_fork: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Roll out main z-path and pre-compute auxiliary forks using eps_fork.
        
        Args:
            z0: Initial productivity (batch_size,)
            eps_main: Main shock sequence (batch_size, T)
            eps_fork: Auxiliary shock sequence (batch_size, T)
            
        Returns:
            z_path: Main trajectory (batch_size, T+1)
            z_fork: Forked next states (batch_size, T, 1) corresponding to z_path[:,t]
        """
        return rollout_forked_path(z0, eps_main, eps_fork, self.shock_params)

    def _generate_batch(
        self,
        seeds_dict: Dict[VariableID, tf.Tensor],
        batch_size: int
    ) -> Dict[str, tf.Tensor]:
        """
        Generate a single batch of data.

        Args:
            seeds_dict: Dictionary mapping VariableID -> seed tensor
            batch_size: Number of samples

        Returns:
            Dictionary with keys:
                - 'k0': Initial capital (batch_size,)
                - 'z0': Initial productivity (batch_size,)
                - 'b0': Initial debt (batch_size,)
                - 'z_path_1': Full z trajectory 1 (batch_size, T+1)
                - 'z_path_2': Full z trajectory 2 (batch_size, T+1)
                - 'eps1': First shock sequence (batch_size, T)
                - 'eps2': Second shock sequence (batch_size, T)
        """
        return generate_batch(
            seeds_dict=seeds_dict,
            batch_size=batch_size,
            horizon=self.T,
            k_bounds=self.k_bounds,
            logz_bounds=self.logz_bounds,
            b_bounds=self.b_bounds,
            shock_params=self.shock_params,
        )

    def get_training_batches(self) -> Iterator[Dict[str, tf.Tensor]]:
        """
        Generate training batches (n_sim_batches batches of size sim_batch_size).

        Returns:
            Iterator yielding dictionaries with batch data

        Example:
            >>> for step, batch in enumerate(gen.get_training_batches(), start=1):
            ...     k0 = batch['k0']  # (n,)
            ...     z_path = batch['z_path']  # (n, T+1)
            ...     # ... training code ...
        """
        # Get all training seeds
        train_seeds = self.seed_schedule.get_train_seeds()

        # Generate batches sequentially
        for step in range(1, self.n_sim_batches + 1):
            seeds_dict = train_seeds[step]
            batch = self._generate_batch(seeds_dict, self.sim_batch_size)
            yield batch

    def get_training_dataset(self) -> Dict[str, tf.Tensor]:
        """
        Get entire training dataset (n_sim_batches * sim_batch_size samples).

        Combines all training batches into single large tensors.
        If cache_dir is provided, the dataset is cached to disk for faster loading.

        Returns:
            Dictionary with training data:
                - 'k0': (J*n,)
                - 'z0': (J*n,)
                - 'b0': (J*n,)
                - 'z_path': (J*n, T+1)
                - 'eps1': (J*n, T)
                - 'eps2': (J*n, T)
        """
        if self._training_dataset is None:
            # Try loading from disk cache if enabled
            if self.cache_dir is not None:
                cache_path = self._get_cache_path("training")
                if os.path.exists(cache_path):
                    self._training_dataset = self._load_from_disk(cache_path)
                    return self._training_dataset

            # Generate all batches and concatenate
            batches = []
            for batch in self.get_training_batches():
                batches.append(batch)
            
            # Concatenate
            self._training_dataset = {}
            for key in batches[0].keys():
                # Stack along 0-th dimension (batch dimension)
                tensors = [b[key] for b in batches]
                self._training_dataset[key] = tf.concat(tensors, axis=0)

            # Save to disk cache if enabled
            if self.cache_dir is not None:
                cache_path = self._get_cache_path("training")
                self._save_to_disk(
                    self._training_dataset, 
                    cache_path,
                    metadata=self._get_metadata()
                )

        return self._training_dataset

    def get_validation_dataset(self) -> Dict[str, tf.Tensor]:
        """
        Get validation dataset (N_val samples).

        Validation uses a single fixed seed configuration. The dataset
        is cached in memory after first generation. If cache_dir is provided,
        the dataset is also cached to disk for faster loading in future runs.

        Returns:
            Dictionary with validation data:
                - 'k0': (N_val,)
                - 'z0': (N_val,)
                - 'b0': (N_val,)
                - 'z_path': (N_val, T+1)
                - 'eps1': (N_val, T)
                - 'eps2': (N_val, T)

        Example:
            >>> val_data = gen.get_validation_dataset()
            >>> val_k0 = val_data['k0']  # (N_val,)
        """
        if self._validation_dataset is None:
            # Try loading from disk cache if enabled
            if self.cache_dir is not None:
                cache_path = self._get_cache_path("validation")
                if os.path.exists(cache_path):
                    self._validation_dataset = self._load_from_disk(cache_path)
                    return self._validation_dataset

            # Generate dataset
            val_seeds = self.seed_schedule.get_val_seeds()
            self._validation_dataset = self._generate_batch(val_seeds, self.N_val)

            # Save to disk cache if enabled
            if self.cache_dir is not None:
                cache_path = self._get_cache_path("validation")
                self._save_to_disk(
                    self._validation_dataset, 
                    cache_path, 
                    metadata=self._get_metadata()
                )

        return self._validation_dataset

    def get_test_dataset(self) -> Dict[str, tf.Tensor]:
        """
        Get test dataset (N_test samples) - SEALED accessor.

        Test uses a separate fixed seed configuration distinct from
        training and validation. The dataset is cached in memory after first
        generation. If cache_dir is provided, the dataset is also cached to
        disk for faster loading in future runs.

        This method provides "sealed" access: test data is only generated
        on explicit call to this method, preventing accidental regeneration
        or mutation during training/validation phases.

        Returns:
            Dictionary with test data:
                - 'k0': (N_test,)
                - 'z0': (N_test,)
                - 'b0': (N_test,)
                - 'z_path': (N_test, T+1)
                - 'eps1': (N_test, T)
                - 'eps2': (N_test, T)

        Example:
            >>> # Only call this when ready to evaluate on test set
            >>> test_data = gen.get_test_dataset()
            >>> test_k0 = test_data['k0']  # (N_test,)
        """
        if self._test_dataset is None:
            # Try loading from disk cache if enabled
            if self.cache_dir is not None:
                cache_path = self._get_cache_path("test")
                if os.path.exists(cache_path):
                    self._test_dataset = self._load_from_disk(cache_path)
                    return self._test_dataset
            
            # Generate dataset
            test_seeds = self.seed_schedule.get_test_seeds()
            self._test_dataset = self._generate_batch(test_seeds, self.N_test)

            # Save to disk cache if enabled
            if self.cache_dir is not None:
                cache_path = self._get_cache_path("test")
                self._save_to_disk(
                    self._test_dataset, 
                    cache_path,
                    metadata=self._get_metadata()
                )

        return self._test_dataset

    def get_flattened_training_dataset(self, include_debt: bool = False) -> Dict[str, tf.Tensor]:
        """
        Get flattened training dataset for ER/BR methods (N*T i.i.d. transitions).

        This method transforms trajectory-based training data into individual i.i.d.
        transitions suitable for Euler Residual (ER) and Bellman Residual (BR) methods.

        Transformation Process:
        1. Load full training dataset with trajectories (N firms, T timesteps)
        2. Flatten from (N, T) to (N*T,) individual state transitions
        3. For each transition: sample k (and optionally b) independently
        4. Shuffle all observations to ensure i.i.d. property for SGD

        Why Independent k/b Sampling:
        - At data generation time, we don't have a policy to generate k_1, k_2, ..., k_T
        - We only have initial k_0 for each trajectory
        - To create N*T i.i.d. samples, we sample k ~ Uniform(k_min, k_max) for each transition
        - This treats each (k, z) pair as an independent draw from the ergodic state distribution
        - Same logic applies to b when include_debt=True

        Data Format Difference:
        - LR method: Uses trajectory data with shape (N, T+1) to compute lifetime rewards
        - ER/BR methods: Use flattened data with shape (N*T,) for one-step transitions

        Args:
            include_debt: If True, also sample b independently from b_bounds.
                          Use for risky debt models. Default False for backward compatibility.

        Returns:
            Dictionary with flattened training data:
                - 'k': Current capital (N*T,) - sampled independently
                - 'z': Current productivity (N*T,) - from z_path[:, t]
                - 'z_next_main': Next productivity, main path (N*T,) - from z_path[:, t+1]
                - 'z_next_fork': Next productivity, fork path (N*T,) - from z_fork[:, t, 0]
                - 'b': Current debt (N*T,) - sampled independently (only if include_debt=True)

        Reference:
            report_brief.md lines 157-167: "Flatten Data for ER and BR"
            Each observation represents a single transition (k, z) -> (k', z')

        Example:
            >>> # For basic ER/BR training (no debt)
            >>> flat_data = gen.get_flattened_training_dataset()
            >>> k = flat_data['k']  # (N*T,) independent capital samples
            >>>
            >>> # For risky debt BR training (with debt)
            >>> flat_data = gen.get_flattened_training_dataset(include_debt=True)
            >>> k, b = flat_data['k'], flat_data['b']  # Both sampled i.i.d.
        """
        # Select appropriate cache based on include_debt flag
        if include_debt:
            cache_attr = '_flattened_training_dataset_debt'
            cache_name = "training_flattened_debt"
        else:
            cache_attr = '_flattened_training_dataset'
            cache_name = "training_flattened"

        # Check memory cache first
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached

        # Try loading from disk cache if enabled
        if self.cache_dir is not None:
            cache_path = self._get_cache_path(cache_name)
            if os.path.exists(cache_path):
                dataset = self._load_from_disk(cache_path)
                setattr(self, cache_attr, dataset)
                return dataset

        # Generate flattened dataset from trajectory data.
        traj_data = self.get_training_dataset()
        result = build_flattened_dataset(
            traj_data=traj_data,
            seed_schedule=self.seed_schedule,
            split="train",
            k_bounds=self.k_bounds,
            b_bounds=self.b_bounds,
            include_debt=include_debt,
            shuffle=True,
            seed_step=0,
        )

        # Store in memory cache
        setattr(self, cache_attr, result)

        # Save to disk cache if enabled
        if self.cache_dir is not None:
            cache_path = self._get_cache_path(cache_name)
            self._save_to_disk(
                result,
                cache_path,
                metadata=self._get_metadata()
            )

        return result

    def get_flattened_validation_dataset(self, include_debt: bool = False) -> Dict[str, tf.Tensor]:
        """
        Get flattened validation dataset for ER/BR methods (N_val*T i.i.d. transitions).

        This method transforms trajectory-based validation data into individual i.i.d.
        transitions suitable for Euler Residual (ER) and Bellman Residual (BR) methods.

        The transformation follows the same logic as get_flattened_training_dataset():
        1. Load validation dataset with trajectories (N_val firms, T timesteps)
        2. Flatten from (N_val, T) to (N_val*T,) individual state transitions
        3. For each transition: sample k (and optionally b) independently
        4. NO shuffling for validation (deterministic evaluation)

        Args:
            include_debt: If True, also sample b independently from b_bounds.
                          Use for risky debt models. Default False for backward compatibility.

        Returns:
            Dictionary with flattened validation data:
                - 'k': Current capital (N_val*T,) - sampled independently
                - 'z': Current productivity (N_val*T,) - from z_path[:, t]
                - 'z_next_main': Next productivity, main path (N_val*T,) - from z_path[:, t+1]
                - 'z_next_fork': Next productivity, fork path (N_val*T,) - from z_fork[:, t, 0]
                - 'b': Current debt (N_val*T,) - sampled independently (only if include_debt=True)

        Example:
            >>> # For ER/BR validation
            >>> flat_val = gen.get_flattened_validation_dataset()
            >>> k = flat_val['k']  # (N_val*T,)
            >>>
            >>> # For risky debt validation
            >>> flat_val = gen.get_flattened_validation_dataset(include_debt=True)
            >>> k, b = flat_val['k'], flat_val['b']
        """
        # Select appropriate cache based on include_debt flag
        if include_debt:
            cache_attr = '_flattened_validation_dataset_debt'
            cache_name = "validation_flattened_debt"
        else:
            cache_attr = '_flattened_validation_dataset'
            cache_name = "validation_flattened"

        # Check memory cache first
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached

        # Try loading from disk cache if enabled
        if self.cache_dir is not None:
            cache_path = self._get_cache_path(cache_name)
            if os.path.exists(cache_path):
                dataset = self._load_from_disk(cache_path)
                setattr(self, cache_attr, dataset)
                return dataset

        # Generate flattened dataset from trajectory validation data.
        traj_data = self.get_validation_dataset()
        result = build_flattened_dataset(
            traj_data=traj_data,
            seed_schedule=self.seed_schedule,
            split="val",
            k_bounds=self.k_bounds,
            b_bounds=self.b_bounds,
            include_debt=include_debt,
            shuffle=False,
            seed_step=None,
        )

        # Store in memory cache
        setattr(self, cache_attr, result)

        # Save to disk cache if enabled
        if self.cache_dir is not None:
            cache_path = self._get_cache_path(cache_name)
            self._save_to_disk(
                result,
                cache_path,
                metadata=self._get_metadata()
            )

        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_data_generator(
    master_seed: Tuple[int, int],
    T: int = 64,
    sim_batch_size: int = 128,  # Batch size (Simulation only)
    n_sim_batches: int = 256,  # Number of batches (Simulation only)
    N_val: Optional[int] = None,
    N_test: Optional[int] = None,
    # Decoupled Economic Parameters (Defaults from source of truth)
    theta: float = EconomicParams.theta,
    r: float = EconomicParams.r_rate,
    delta: float = EconomicParams.delta,
    shock_params: Optional[ShockParams] = None,
    # Bounds configuration
    # Allow for user input bound overrides
    bounds: Optional[Dict[str, Any]] = None,
    # Auto-bounds parameters
    auto_compute_bounds: bool = True,
    std_dev_multiplier: float = 3.0,
    k_min_multiplier: float = 0.2,
    k_max_multiplier: float = 3.0,
    k_star_override: Optional[float] = None,
    # Collateral constraint parameters for debt bounds
    tax: float = EconomicParams.tax,
    frac_liquid: float = EconomicParams.frac_liquid,
    # Config
    cache_dir: Optional[str] = None,
    save_to_disk: bool = True,
    verbose: bool = False
) -> Tuple[DataGenerator, ShockParams, Dict[str, Any]]:
    """
    Factory function to create a DataGenerator with economically-anchored bounds.

    This function supports TWO modes:

    1. MODEL-BASED AUTO-COMPUTATION (default, recommended for economic models):
       - Specify bounds as multipliers on steady-state k* (e.g., k_min=0.2, k_max=3.0)
       - Bounds are converted to LEVELS internally and returned ready for use
       - Networks receive level values directly; no post-hoc conversion needed

    2. DIRECT SPECIFICATION (for custom data or arbitrary units):
       - Pass bounds directly via the `bounds` parameter
       - Set auto_compute_bounds=False to skip economic model
       - Use any units your data requires - framework is unit-agnostic

    The returned bounds dict contains all values in LEVELS (actual values, not
    multipliers). The k_star field is included for reference/documentation only.

    Args:
        master_seed: Master seed pair (m0, m1)
        T: Rollout horizon (default: 64)
        sim_batch_size: Batch size for data generation (default: 128)
        n_sim_batches: Number of simulation batches (default: 256)
        N_val: Validation size (default: 10*n)
        N_test: Test size (default: 50*n)
        theta: Production elasticity (default: from EconomicParams)
        r: Risk-free rate (default: from EconomicParams)
        delta: Depreciation rate (default: from EconomicParams)
        shock_params: Shock parameters (if None, uses defaults)
        bounds: Dictionary of bounds in LEVELS (k, log_z, b). Can be partial if auto_compute_bounds=True.
            For custom data, pass bounds directly in your data's units.
        auto_compute_bounds: If True, missing bounds are auto-generated from economic params.
            Set to False when using custom bounds with arbitrary units.
        std_dev_multiplier: m, number of std devs for log_z bounds (must be in (2, 5), default 3.0)
        k_min_multiplier: k_min as multiplier on k* (must be in (0, 0.5), default 0.2)
        k_max_multiplier: k_max as multiplier on k* (must be in (1.5, 5), default 3.0)
        k_star_override: Optional override for k* (if None, auto-computed at z=e^μ)
        tax: Corporate tax rate for collateral constraint (default: from EconomicParams)
        frac_liquid: Liquidation fraction for collateral constraint (default: from EconomicParams)
        cache_dir: Directory for caching (default: None -> PROJECT_ROOT/data)
        save_to_disk: Whether to save to disk (default: True)
        verbose: Print configuration summary

    Returns:
        Tuple containing:
        - DataGenerator instance
        - ShockParams used
        - Bounds dictionary in LEVELS (includes 'k', 'b', 'log_z', 'k_star')
          Note: k_star is for reference only; k bounds are already in levels.
    """
    from src.economy.bounds import generate_states_bounds

    # Step 1: Initialize parameters
    if shock_params is None:
        shock_params = ShockParams()
        if verbose:
            print(" Using default ShockParams")
    else:
        if verbose:
            print(" Using provided ShockParams")

    # Step 2: Generate/Resolve bounds
    if bounds is None:
        bounds = {}

    # Auto-compute missing bounds if requested
    if auto_compute_bounds:
        if 'k' not in bounds or 'log_z' not in bounds or 'b' not in bounds or 'k_star' not in bounds:
            auto_bounds = generate_states_bounds(
                theta=theta,
                r=r,
                delta=delta,
                shock_params=shock_params,
                std_dev_multiplier=std_dev_multiplier,
                k_min_multiplier=k_min_multiplier,
                k_max_multiplier=k_max_multiplier,
                k_star_override=k_star_override,
                validate=True,  # Enforce constraints from report_brief.md
                tax=tax,
                frac_liquid=frac_liquid
            )
            # Fill missing
            if 'k' not in bounds: bounds['k'] = auto_bounds['k']
            if 'log_z' not in bounds: bounds['log_z'] = auto_bounds['log_z']
            if 'b' not in bounds: bounds['b'] = auto_bounds['b']
            if 'k_star' not in bounds: bounds['k_star'] = auto_bounds['k_star']

            if verbose:
                print(" Auto-generated bounds in LEVELS (from economic parameters)")
    else:
        # Strict mode: verify all bounds exist
        missing = []
        if 'k' not in bounds: missing.append('k')
        if 'log_z' not in bounds: missing.append('log_z')
        if 'b' not in bounds: missing.append('b')

        if missing:
            raise ValueError(f"auto_compute_bounds=False but bounds are missing: {missing}")

        # If k_star not provided in strict mode, compute it
        if 'k_star' not in bounds:
            from src.economy.bounds import compute_k_star
            bounds['k_star'] = compute_k_star(theta, r, delta, shock_params.mu)
            if verbose:
                print(f" Auto-computed k_star = {bounds['k_star']:.4f}")

    if verbose:
        print(f"  k_bounds (LEVELS): {bounds['k']}")
        print(f"  b_bounds (LEVELS): {bounds['b']}")
        print(f"  log_z_bounds: {bounds['log_z']}")
        print(f"  k_star (reference): {bounds.get('k_star', 'N/A')}")
        print(f"  Master seed: {master_seed}")

    # Step 3: Create data generator
    generator = DataGenerator(
        master_seed=master_seed,
        shock_params=shock_params,
        k_bounds=bounds['k'],
        logz_bounds=bounds['log_z'],
        b_bounds=bounds['b'],
        sim_batch_size=sim_batch_size,
        T=T,
        n_sim_batches=n_sim_batches,
        N_val=N_val,
        N_test=N_test,
        cache_dir=cache_dir,
        save_to_disk=save_to_disk
    )

    if verbose:
        print(" === DataGenerator created ===")
        print(f"  Simulation batch size (n): {generator.sim_batch_size}")
        print(f"  Horizon (T): {generator.T}")
        print(f"  Simulation batches (J): {generator.n_sim_batches}")
        print(f"  Total samples: {generator.sim_batch_size * generator.n_sim_batches}")
        if 'k_star' in bounds:
            print(f"  NOTE: Bounds are in LEVELS. k_star={bounds['k_star']:.4f} is for reference only.")

    return generator, shock_params, bounds


def cleanup_cache(
    cache_dir: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Clean up cached data files from the data directory.

    This utility removes old cached datasets before starting a new experiment,
    ensuring fresh data generation with current configuration.

    Args:
        cache_dir: Directory to clean. If None, uses PROJECT_ROOT/data.
        patterns: List of glob patterns to match (default: ["*.npz", "*.png"])
        verbose: Print deleted files

    Returns:
        Dict with 'deleted_count' and 'total_bytes' removed

    Example:
        >>> # Clean default cache directory
        >>> cleanup_cache()

        >>> # Clean specific directory with custom patterns
        >>> cleanup_cache(cache_dir="../data", patterns=["*.npz"])

        >>> # Silent mode
        >>> stats = cleanup_cache(verbose=False)
        >>> print(f"Removed {stats['deleted_count']} files")
    """
    from pathlib import Path

    # Default patterns
    if patterns is None:
        patterns = ["*.npz", "*.png"]

    # Resolve cache directory
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_dir = str(project_root / "data")

    cache_path = Path(cache_dir)

    stats = {"deleted_count": 0, "total_bytes": 0}

    if not cache_path.exists():
        if verbose:
            print(f"Cache directory does not exist: {cache_path}")
        return stats

    # Find and delete matching files
    for pattern in patterns:
        for filepath in cache_path.glob(pattern):
            try:
                file_size = filepath.stat().st_size
                filepath.unlink()
                stats["deleted_count"] += 1
                stats["total_bytes"] += file_size
                if verbose:
                    print(f"  Deleted: {filepath.name}")
            except OSError as e:
                if verbose:
                    print(f"  Error deleting {filepath}: {e}")

    if verbose:
        if stats["deleted_count"] > 0:
            mb = stats["total_bytes"] / (1024 * 1024)
            print(f"Cleaned {stats['deleted_count']} files ({mb:.2f} MB)")
        else:
            print("No cached files found to clean")

    return stats
