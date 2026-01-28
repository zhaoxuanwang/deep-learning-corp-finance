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
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from typing import Dict, Iterator, Optional, Tuple, Any
from dataclasses import dataclass

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.rng import SeedSchedule, SeedScheduleConfig, VariableID
from src.economy.shocks import step_ar1_tf


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
            # This file is in src/economy/data_generator.py -> root is ../..
            project_root = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = str(project_root / "data")
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

    def _get_config_hash(self) -> str:
        """
        Generate deterministic hash from generator configuration.

        Used to create unique cache filenames that change when configuration
        changes (master_seed, bounds, sizes, economic params).

        Returns:
            12-character hex string
        """
        # Include all configuration that affects dataset generation
        # Only include parameters that affect z-path rollout
        config_tuple = (
            self.master_seed,
            self.sim_batch_size,
            self.T,
            self.n_sim_batches,
            self.N_val,
            self.N_test,
            self.k_bounds,
            self.logz_bounds,
            self.b_bounds,
            self.shock_params.rho,      # AR(1) persistence
            self.shock_params.sigma,    # AR(1) variance
            self.shock_params.mu        # AR(1) unconditional mean
        )
        config_str = str(config_tuple)
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:12]

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
        filename = f"{split}_{config_hash}.npz"
        return os.path.join(self.cache_dir, filename)

    def _get_metadata(self) -> Dict[str, Any]:
        """
        Generate metadata for dataset verification.
        """
        return {
            "schema_version": "1.0",
            "master_seed": [int(x) for x in self.master_seed],
            "dims": {
                "n": self.sim_batch_size,
                "T": self.T,
                "J": self.n_sim_batches,
                "N_val": self.N_val,
                "N_test": self.N_test
            },
            "bounds": {
                "k": [float(x) for x in self.k_bounds],
                "logz": [float(x) for x in self.logz_bounds],
                "b": [float(x) for x in self.b_bounds]
            },
            "shock_params": {
                "rho": float(self.shock_params.rho),
                "sigma": float(self.shock_params.sigma),
                "mu": float(self.shock_params.mu)
            },
            "creation_timestamp": str(datetime.now())
        }

    def _save_to_disk(self, dataset: Dict[str, tf.Tensor], path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save dataset to disk in .npz format with metadata.

        Args:
            dataset: Dictionary of tensors to save
            path: File path for saving (.npz file)
            metadata: Optional dictionary of metadata to save
        """
        if not self.save_to_disk:
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert tensors to numpy arrays
        numpy_data = {key: tensor.numpy() for key, tensor in dataset.items()}
        
        # Add metadata if provided
        if metadata is not None:
            numpy_data["_metadata"] = json.dumps(metadata)

        np.savez_compressed(path, **numpy_data)

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
        # Load numpy arrays
        data = np.load(path)
        result = {}
        
        # Check metadata
        if "_metadata" in data:
            stored_meta = json.loads(str(data["_metadata"]))
            current_meta = self._get_metadata()
            
            # Helper to compare dictionaries ignoring timestamp
            def check_mismatch(section):
                if stored_meta.get(section) != current_meta.get(section):
                    return True
                return False

            if (check_mismatch("master_seed") or 
                check_mismatch("dims") or 
                check_mismatch("bounds") or 
                check_mismatch("params")):
                 
                 raise ValueError(
                     f"Cached dataset metadata mismatch!\n"
                     f"Stored: {json.dumps(stored_meta, indent=2)}\n"
                     f"Current: {json.dumps(current_meta, indent=2)}\n"
                     f"File: {path}"
                 )
        
        # Convert to TensorFlow tensors (excluding metadata)
        for key in data.keys():
            if key != "_metadata":
                result[key] = tf.constant(data[key])
                
        return result

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
        k_min, k_max = self.k_bounds
        logz_min, logz_max = self.logz_bounds
        b_min, b_max = self.b_bounds

        # Sample k0 uniformly
        k0 = tf.random.stateless_uniform(
            shape=(batch_size,),
            seed=k_seed,
            minval=k_min,
            maxval=k_max,
            dtype=tf.float32
        )

        # Sample log(z0) uniformly, then convert to z0
        logz0 = tf.random.stateless_uniform(
            shape=(batch_size,),
            seed=z_seed,
            minval=logz_min,
            maxval=logz_max,
            dtype=tf.float32
        )
        z0 = tf.exp(logz0)

        # Sample b0 uniformly
        b0 = tf.random.stateless_uniform(
            shape=(batch_size,),
            seed=b_seed,
            minval=b_min,
            maxval=b_max,
            dtype=tf.float32
        )

        return k0, z0, b0

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
        # Generate standard normal shocks
        eps1 = tf.random.stateless_normal(
            shape=(batch_size, T),
            seed=eps1_seed,
            dtype=tf.float32
        )

        eps2 = tf.random.stateless_normal(
            shape=(batch_size, T),
            seed=eps2_seed,
            dtype=tf.float32
        )

        return eps1, eps2

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
        batch_size = tf.shape(z0)[0]
        T = tf.shape(eps_main)[1]
        
        z_main_list = [z0]
        z_fork_list = []
        
        z_current = z0
        
        for t in range(T):
            # 1. Main Path Step
            z_next_main = step_ar1_tf(
                z_current, self.shock_params.rho, self.shock_params.sigma, self.shock_params.mu, eps=eps_main[:, t]
            )
            z_main_list.append(z_next_main)
            
            # 2. Fork Step (from SAME current state, using different shock)
            z_next_fork = step_ar1_tf(
                z_current, self.shock_params.rho, self.shock_params.sigma, self.shock_params.mu, eps=eps_fork[:, t]
            )
            z_fork_list.append(tf.reshape(z_next_fork, [-1, 1]))
            
            # Update state along MAIN path
            z_current = z_next_main
            
        z_path = tf.stack(z_main_list, axis=1) # (B, T+1)
        z_fork = tf.stack(z_fork_list, axis=1) # (B, T, 1)
        
        return z_path, z_fork

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
        # Generate initial states
        k0, z0, b0 = self._generate_initial_states(
            batch_size,
            k_seed=seeds_dict[VariableID.K0],
            z_seed=seeds_dict[VariableID.Z0],
            b_seed=seeds_dict[VariableID.B0]
        )

        # Generate shocks
        eps1, eps2 = self._generate_shocks(
            batch_size,
            self.T,
            eps1_seed=seeds_dict[VariableID.EPS1],
            eps2_seed=seeds_dict[VariableID.EPS2]
        )

        # Rollout z-path (Main) and Forks
        # We use eps1 for main path, eps2 for forks
        z_path, z_fork = self._rollout_forked_path(z0, eps1, eps2)

        return {
            'k0': k0,
            'z0': z0,
            'b0': b0,
            'z_path': z_path, # Main trajectory
            'z_fork': z_fork, # Pre-computed forks for AiO
            'eps_path': eps1,
            'eps_fork': eps2
        }

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

    def get_flattened_training_dataset(self) -> Dict[str, tf.Tensor]:
        """
        Get flattened training dataset for ER/BR methods (N*T i.i.d. transitions).

        This method transforms trajectory-based training data into individual i.i.d.
        transitions suitable for Euler Residual (ER) and Bellman Residual (BR) methods.

        Transformation Process:
        1. Load full training dataset with trajectories (N firms, T timesteps)
        2. Flatten from (N, T) to (N*T,) individual state transitions
        3. For each transition: sample k independently from ergodic distribution
        4. Shuffle all observations to ensure i.i.d. property for SGD

        Why Independent k Sampling:
        - At data generation time, we don't have a policy to generate k_1, k_2, ..., k_T
        - We only have initial k_0 for each trajectory
        - To create N*T i.i.d. samples, we sample k ~ Uniform(k_min, k_max) for each transition
        - This treats each (k, z) pair as an independent draw from the ergodic state distribution

        Data Format Difference:
        - LR method: Uses trajectory data with shape (N, T+1) to compute lifetime rewards
        - ER/BR methods: Use flattened data with shape (N*T,) for one-step transitions

        Returns:
            Dictionary with flattened training data:
                - 'k': Current capital (N*T,) - sampled independently
                - 'z': Current productivity (N*T,) - from z_path[:, t]
                - 'z_next_main': Next productivity, main path (N*T,) - from z_path[:, t+1]
                - 'z_next_fork': Next productivity, fork path (N*T,) - from z_fork[:, t, 0]

        Reference:
            report_brief.md lines 157-167: "Flatten Data for ER and BR"
            Each observation represents a single transition (k, z) -> (k', z')

        Example:
            >>> # For ER/BR training
            >>> flat_data = gen.get_flattened_training_dataset()
            >>> k = flat_data['k']  # (N*T,) independent capital samples
            >>> z = flat_data['z']  # (N*T,) productivity states
            >>>
            >>> # Create TF dataset for batching
            >>> tf_dataset = tf.data.Dataset.from_tensor_slices(flat_data)
            >>> tf_dataset = tf_dataset.shuffle(10000).batch(256).repeat()
        """
        # Check memory cache first
        if self._flattened_training_dataset is not None:
            return self._flattened_training_dataset

        # Try loading from disk cache if enabled
        if self.cache_dir is not None:
            cache_path = self._get_cache_path("training_flattened")
            if os.path.exists(cache_path):
                self._flattened_training_dataset = self._load_from_disk(cache_path)
                return self._flattened_training_dataset

        # Generate flattened dataset
        # Get full trajectory dataset
        traj_data = self.get_training_dataset()

        # Extract relevant arrays
        z_path = traj_data['z_path']  # (N, T+1)
        z_fork = traj_data['z_fork']  # (N, T, 1)

        # Get dimensions
        N = tf.shape(z_path)[0]  # Number of trajectories
        T = tf.shape(z_path)[1] - 1  # Horizon (z_path is T+1)
        N_total = N * T  # Total flattened samples

        # === FLATTENING ===
        # For each trajectory i and timestep t (t=0..T-1), create observation:
        #   z[i,t] = z_path[i, t]
        #   z_next_main[i,t] = z_path[i, t+1]
        #   z_next_fork[i,t] = z_fork[i, t, 0]

        # Extract z (current): take z_path[:, 0:T]
        z_curr = z_path[:, :-1]  # (N, T)

        # Extract z_next_main: take z_path[:, 1:T+1]
        z_next_main = z_path[:, 1:]  # (N, T)

        # Extract z_next_fork: z_fork is already (N, T, 1), squeeze last dim
        z_next_fork = tf.squeeze(z_fork, axis=-1)  # (N, T)

        # Flatten from (N, T) to (N*T,)
        z_flat = tf.reshape(z_curr, [-1])  # (N*T,)
        z_next_main_flat = tf.reshape(z_next_main, [-1])  # (N*T,)
        z_next_fork_flat = tf.reshape(z_next_fork, [-1])  # (N*T,)

        # === INDEPENDENT K SAMPLING ===
        # Sample k independently for each of the N*T observations
        # Use deterministic seed for reproducibility: derive from master seed
        k_seed = tf.constant([
            self.master_seed[0] + 400,  # Offset for flattened k sampling
            self.master_seed[1] + 0
        ], dtype=tf.int32)

        k_min, k_max = self.k_bounds
        k_flat = tf.random.stateless_uniform(
            shape=[N_total],
            seed=k_seed,
            minval=k_min,
            maxval=k_max,
            dtype=tf.float32
        )

        # === SHUFFLING ===
        # Shuffle all arrays together using the same permutation
        # This ensures i.i.d. property for SGD
        shuffle_seed = tf.constant([
            self.master_seed[0] + 500,  # Offset for shuffling
            self.master_seed[1] + 0
        ], dtype=tf.int32)

        # Create permutation indices
        indices = tf.range(N_total, dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=shuffle_seed)

        # Apply permutation to all arrays
        k_shuffled = tf.gather(k_flat, shuffled_indices)
        z_shuffled = tf.gather(z_flat, shuffled_indices)
        z_next_main_shuffled = tf.gather(z_next_main_flat, shuffled_indices)
        z_next_fork_shuffled = tf.gather(z_next_fork_flat, shuffled_indices)

        # Store in memory cache
        self._flattened_training_dataset = {
            'k': k_shuffled,
            'z': z_shuffled,
            'z_next_main': z_next_main_shuffled,
            'z_next_fork': z_next_fork_shuffled
        }

        # Save to disk cache if enabled
        if self.cache_dir is not None:
            cache_path = self._get_cache_path("training_flattened")
            self._save_to_disk(
                self._flattened_training_dataset,
                cache_path,
                metadata=self._get_metadata()
            )

        return self._flattened_training_dataset


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
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    auto_compute_bounds: bool = True,
    # Auto-bounds parameters
    std_dev_multiplier: float = 3.0,
    k_min_multiplier: float = 0.2,
    k_max_multiplier: float = 3.0,
    # Config
    cache_dir: Optional[str] = None,
    save_to_disk: bool = True,
    verbose: bool = False
) -> Tuple[DataGenerator, ShockParams, Dict[str, Tuple[float, float]]]:
    """
    Factory function to create a DataGenerator.

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
        bounds: Dictionary of bounds (k, log_z, b). Can be partial if auto_compute_bounds=True.
        auto_compute_bounds: If True, missing bounds are auto-generated using economic params.
        std_dev_multiplier: Multiplier for log z bounds (default 3.0)
        k_min_multiplier: Multiplier for k_min (default 0.2)
        k_max_multiplier: Multiplier for k_max (default 3.0)
        cache_dir: Directory for caching (default: None -> PROJECT_ROOT/data)
        save_to_disk: Whether to save to disk (default: True)
        verbose: Print configuration summary

    Returns:
        Tuple containing:
        - DataGenerator instance
        - ShockParams used
        - Bounds dictionary used
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
        if 'k' not in bounds or 'log_z' not in bounds or 'b' not in bounds:
            auto_bounds = generate_states_bounds(
                theta=theta,
                r=r,
                delta=delta,
                shock_params=shock_params,
                std_dev_multiplier=std_dev_multiplier,
                k_min_multiplier=k_min_multiplier,
                k_max_multiplier=k_max_multiplier
            )
            # Fill missing
            if 'k' not in bounds: bounds['k'] = auto_bounds['k']
            if 'log_z' not in bounds: bounds['log_z'] = auto_bounds['log_z']
            if 'b' not in bounds: bounds['b'] = auto_bounds['b']
            
            if verbose:
                print(" Auto-generated missing SamplingBounds")
    else:
        # Strict mode: verify all bounds exist
        missing = []
        if 'k' not in bounds: missing.append('k')
        if 'log_z' not in bounds: missing.append('log_z')
        if 'b' not in bounds: missing.append('b')
        
        if missing:
            raise ValueError(f"auto_compute_bounds=False but bounds are missing: {missing}")

    if verbose:
        print(f"  k_bounds: {bounds['k']}")
        print(f"  log_z_bounds: {bounds['log_z']}")
        print(f"  b_bounds: {bounds['b']}")
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
        
    return generator, shock_params, bounds
