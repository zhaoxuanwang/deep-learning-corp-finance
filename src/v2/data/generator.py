"""
src/v2/data/generator.py

Offline dataset generator for v2 trainers.

Design principles:
- ALL randomness is governed by the master seed pair via SeedSchedule.
- Data generation is fully separated from training — trainers receive
  pre-built dataset dicts and never call RNG directly.
- Exogenous state sequences (z_path, z_fork) are pre-computed via
  env.exogenous_transition(). Endogenous states (k) are sampled i.i.d.
  uniformly from the state space bounds.

Dataset formats
---------------
Trajectory (for LR trainer):
    s_endo_0:  (N, endo_dim)         initial endogenous state (k0)
    z_path:    (N, T+1, exo_dim)     pre-rolled exogenous trajectory
    z_fork:    (N, T, exo_dim)       one-step alternative branches

Flattened (for ER / BRM trainers):
    s_endo:       (N*T, endo_dim)    endogenous state, i.i.d. uniform
    z:            (N*T, exo_dim)     z_curr from z_path[:, :-1]
    z_next_main:  (N*T, exo_dim)     z from main AR(1) path
    z_next_fork:  (N*T, exo_dim)     z from fork path (AiO second draw)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import tensorflow as tf

from src.v2.data.rng import SeedSchedule, SeedScheduleConfig, VariableID


@dataclass
class DataGeneratorConfig:
    """Configuration for offline dataset generation.

    Attributes:
        n_paths:        Number of trajectory paths N.
        horizon:        Rollout horizon T (steps per path).
        master_seed:    Master seed pair (m0, m1) — must match TrainingConfig.
        n_train_steps:  Used only for SeedSchedule int32 overflow validation.
        cache:          Save datasets to disk (default False = memory only).
        cache_dir:      Directory for disk cache files.
    """
    n_paths:        int             = 5000
    horizon:        int             = 64
    master_seed:    tuple           = field(default_factory=lambda: (20, 26))
    n_train_steps:  Optional[int]   = None
    cache:          bool            = False
    cache_dir:      str             = ".cache/v2_data"


class DataGenerator:
    """Builds fixed, reproducible training/validation/test datasets.

    All RNG is governed by the SeedSchedule derived from master_seed.
    No randomness is introduced outside this class during data generation.

    Args:
        env:    MDPEnvironment instance (must implement exo/endo interface).
        config: DataGeneratorConfig.

    Example::

        config = DataGeneratorConfig(n_paths=5000, horizon=64,
                                     master_seed=(20, 26))
        gen = DataGenerator(env, config)

        train_traj   = gen.get_trajectory_dataset("train")
        train_flat   = gen.get_flattened_dataset("train")
        val_flat     = gen.get_flattened_dataset("val")
    """

    def __init__(self, env, config: DataGeneratorConfig = None):
        self.env = env
        self.config = config or DataGeneratorConfig()

        seed_cfg = SeedScheduleConfig(
            master_seed=self.config.master_seed,
            n_train_steps=self.config.n_train_steps,
        )
        self.seed_schedule = SeedSchedule(seed_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trajectory_dataset(
        self,
        split: Literal["train", "val", "test"] = "train",
        step: int = 1,
    ) -> Dict[str, tf.Tensor]:
        """Build trajectory-format dataset.

        Args:
            split: "train", "val", or "test".
            step:  Seed step index (only used for "train"; ignored for val/test).

        Returns:
            Dict with keys: s_endo_0, z_path, z_fork.
        """
        seeds = self._get_seeds(split, step)
        return self._build_trajectory(seeds)

    def get_flattened_dataset(
        self,
        split: Literal["train", "val", "test"] = "train",
        step: int = 1,
        shuffle: bool = True,
    ) -> Dict[str, tf.Tensor]:
        """Build flattened per-step transition dataset.

        Args:
            split:   "train", "val", or "test".
            step:    Seed step index (only used for "train").
            shuffle: Shuffle the N*T transitions (uses VariableID.SHUFFLE seed).

        Returns:
            Dict with keys: s_endo, z, z_next_main, z_next_fork.
        """
        seeds = self._get_seeds(split, step)
        traj = self._build_trajectory(seeds)
        return self._flatten(traj, seeds, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_seeds(
        self,
        split: Literal["train", "val", "test"],
        step: int,
    ) -> Dict[VariableID, tf.Tensor]:
        """Retrieve all variable seeds for a given split and step."""
        if split == "train":
            return self.seed_schedule.get_train_seeds(steps=[step])[step]
        elif split == "val":
            return self.seed_schedule.get_val_seeds()
        elif split == "test":
            return self.seed_schedule.get_test_seeds()
        else:
            raise ValueError(f"Unknown split '{split}'. Expected train/val/test.")

    def _build_trajectory(
        self,
        seeds: Dict[VariableID, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Core trajectory builder.

        Produces:
            s_endo_0: (N, endo_dim)      — initial endogenous state
            z_path:   (N, T+1, exo_dim)  — main AR(1) chain
            z_fork:   (N, T, exo_dim)    — one-step branches from main path
        """
        N = self.config.n_paths
        T = self.config.horizon
        shock_dim = self.env.shock_dim()

        # --- Initial states ---
        s_endo_0 = self.env.sample_initial_endogenous(N, seeds[VariableID.K0])
        z0       = self.env.sample_initial_exogenous(N, seeds[VariableID.Z0])

        # --- Shocks: (N, T, shock_dim) ---
        eps_path = tf.random.stateless_normal(
            shape=[N, T, shock_dim],
            seed=seeds[VariableID.EPS1],
            dtype=tf.float32,
        )
        eps_fork = tf.random.stateless_normal(
            shape=[N, T, shock_dim],
            seed=seeds[VariableID.EPS2],
            dtype=tf.float32,
        )

        # --- Rollout exogenous state ---
        z_path, z_fork = self._rollout_z(z0, eps_path, eps_fork)

        return {
            "s_endo_0": s_endo_0,   # (N, endo_dim)
            "z_path":   z_path,     # (N, T+1, exo_dim)
            "z_fork":   z_fork,     # (N, T, exo_dim)
        }

    def _rollout_z(
        self,
        z0:       tf.Tensor,   # (N, exo_dim)
        eps_path: tf.Tensor,   # (N, T, shock_dim)
        eps_fork: tf.Tensor,   # (N, T, shock_dim)
    ):
        """Roll out exo state trajectory using env.exogenous_transition.

        Returns:
            z_path: (N, T+1, exo_dim)  main AR(1) trajectory
            z_fork: (N, T, exo_dim)    one-step branches at each t
        """
        z_path_list = [z0]
        z_fork_list = []
        z = z0

        for t in range(self.config.horizon):
            z_next = self.env.exogenous_transition(z, eps_path[:, t, :])
            z_path_list.append(z_next)
            z_fork_list.append(self.env.exogenous_transition(z, eps_fork[:, t, :]))
            z = z_next

        return tf.stack(z_path_list, axis=1), tf.stack(z_fork_list, axis=1)

    def _flatten(
        self,
        traj:    Dict[str, tf.Tensor],
        seeds:   Dict[VariableID, tf.Tensor],
        shuffle: bool = True,
    ) -> Dict[str, tf.Tensor]:
        """Flatten trajectory to per-step i.i.d. transitions.

        Endogenous state (k) is sampled fresh i.i.d. uniform for all N*T
        transitions — independent of the trajectory rollout. This keeps k
        uniformly distributed over the state space (exploration) rather
        than following the path induced by any particular policy.

        Exogenous state (z_curr, z_next_main, z_next_fork) comes from the
        pre-computed trajectory, giving an ergodic-like z distribution.

        Note: k uses the same VariableID.K0 seed as trajectory s_endo_0
        but with shape (N*T, endo_dim) instead of (N, endo_dim), yielding
        independent draws (TF stateless RNG with different shapes).
        """
        N        = self.config.n_paths
        T        = self.config.horizon
        N_total  = N * T
        exo_dim  = self.env.exo_dim()
        endo_dim = self.env.endo_dim()

        z_path = traj["z_path"]   # (N, T+1, exo_dim)
        z_fork = traj["z_fork"]   # (N, T, exo_dim)

        # Extract z transitions
        z_curr      = tf.reshape(z_path[:, :-1, :], [N_total, exo_dim])
        z_next_main = tf.reshape(z_path[:, 1:,  :], [N_total, exo_dim])
        z_next_fork = tf.reshape(z_fork,             [N_total, exo_dim])

        # Fresh i.i.d. endogenous state for each transition
        s_endo = self.env.sample_initial_endogenous(N_total, seeds[VariableID.K0])

        if shuffle:
            shuffle_seed = seeds[VariableID.SHUFFLE]
            indices = tf.range(N_total, dtype=tf.int32)
            idx = tf.random.experimental.stateless_shuffle(indices,
                                                            seed=shuffle_seed)
            s_endo      = tf.gather(s_endo,      idx)
            z_curr      = tf.gather(z_curr,      idx)
            z_next_main = tf.gather(z_next_main, idx)
            z_next_fork = tf.gather(z_next_fork, idx)

        return {
            "s_endo":      s_endo,        # (N*T, endo_dim)
            "z":           z_curr,        # (N*T, exo_dim)
            "z_next_main": z_next_main,   # (N*T, exo_dim)
            "z_next_fork": z_next_fork,   # (N*T, exo_dim)
        }
