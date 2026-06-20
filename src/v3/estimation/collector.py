"""Block-2 dataset collector (DF26 Sec 5.3, Sec 7 collector, serialized).

Draws parameter vectors over the current bounds, refines the network solution
(Block 2), simulates a panel, and computes the 11 moments, building the (raw beta,
moments) dataset the surrogates train on. In production this runs concurrently on
collector GPUs; here it is a serial loop (the expensive step is the per-draw refine).
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.config import ModelParams
from src.v3.simulation.batched import compute_moments_batch, simulate_panel_batch
from src.v3.simulation.moments import compute_moments
from src.v3.simulation.panel import simulate_panel
from src.v3.solver import refine as _refine
from src.v3.solver.batched import refine_batch


def collect_dataset(bundle, bounds, ext, grid_cfg, master_seed, n_rows, *,
                    refine_rounds=6, n_firms=2000, T=120, burn_in=60, max_attempts=None):
    """Return ``(beta_raw [n, 8], moments [n, 11])`` over the current bounds.

    Draws are skipped when the (network-seeded) refined solution yields a degenerate
    panel with no good observations, so the moment vector is non-finite. Keeps drawing
    until ``n_rows`` valid rows or ``max_attempts`` is reached.
    """
    dtype = TF_FLOAT_NUM
    lo = tf.constant(bounds.lower_array(), dtype)
    hi = tf.constant(bounds.upper_array(), dtype)
    max_attempts = max_attempts or 5 * n_rows
    betas, moments, r = [], [], 0
    while len(betas) < n_rows and r < max_attempts:
        u = seeding.uniform([8], master_seed, seeding.Purpose.COLLECT, r, dtype=dtype)
        beta_raw = lo + u * (hi - lo)
        params = ModelParams.from_array(beta_raw.numpy())
        refined = _refine.refine(bundle, params, ext, grid_cfg, bounds, n_rounds=refine_rounds)
        sim_seed = int(seeding.key(master_seed, seeding.Purpose.COLLECT, r, 1)[0])
        panel = simulate_panel(refined, params, ext, grid_cfg, sim_seed,
                               n_firms=n_firms, T=T, burn_in=burn_in)
        m = compute_moments(panel)
        r += 1
        if bool(tf.reduce_all(tf.math.is_finite(m))):
            betas.append(beta_raw)
            moments.append(m)
    return tf.stack(betas), tf.stack(moments)


def collect_dataset_batch(bundle, bounds, ext, grid_cfg, master_seed, n_rows, *,
                          batch_size=16, refine_rounds=6, n_firms=2000, T=120, burn_in=60,
                          max_attempts=None):
    """Batched (GPU-default) collector: same dataset as :func:`collect_dataset`, faster.

    Draws the identical per-row parameter vectors and simulation seeds as the serial
    collector (so ``batch_size=1`` reproduces it row-for-row), but refines and
    simulates ``batch_size`` draws at once via the batched Block-2 path
    (:func:`src.v3.solver.batched.refine_batch`). On a CUDA GPU the batched float64
    solve/einsums give near-``batch_size`` throughput; on CPU the gain is the BLAS
    batching. Non-finite-moment draws are skipped exactly as in the serial collector.
    """
    dtype = TF_FLOAT_NUM
    lo = tf.constant(bounds.lower_array(), dtype)
    hi = tf.constant(bounds.upper_array(), dtype)
    max_attempts = max_attempts or 5 * n_rows
    betas, moments, r = [], [], 0
    while len(betas) < n_rows and r < max_attempts:
        idx = list(range(r, min(r + batch_size, max_attempts)))
        beta_batch = tf.stack([
            lo + seeding.uniform([8], master_seed, seeding.Purpose.COLLECT, j, dtype=dtype) * (hi - lo)
            for j in idx])                                              # [b, 8]
        seeds = [int(seeding.key(master_seed, seeding.Purpose.COLLECT, j, 1)[0]) for j in idx]
        refined = refine_batch(bundle, beta_batch, ext, grid_cfg, bounds, n_rounds=refine_rounds)
        panel = simulate_panel_batch(refined, beta_batch, ext, grid_cfg, seeds,
                                     n_firms=n_firms, T=T, burn_in=burn_in)
        m = compute_moments_batch(panel)                              # [b, 11]
        finite = tf.reduce_all(tf.math.is_finite(m), axis=1)
        for j in range(len(idx)):
            if len(betas) < n_rows and bool(finite[j]):
                betas.append(beta_batch[j])
                moments.append(m[j])
        r += len(idx)
    return tf.stack(betas), tf.stack(moments)
