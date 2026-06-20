"""Monte-Carlo parameter/moment recovery (DF26 Sec 12.2): the critical end-to-end test.

Efficient variant (Sec 12.2): collect one (beta, moments) dataset over the box and
train the surrogates once; then per draw build the "data" from the trusted VFI solver
and recover. For each kept draw:

  beta* ~ Table A1 -> VFI (Howard) -> simulate -> (discard if no defaults) ->
  target moments m* -> W from the panel -> LM estimate beta_hat ->
  refine + simulate at beta_hat -> fitted moments m_hat.

Produces, across draws, the data for Fig V1 (true vs fitted moments) and Fig V2
(true vs fitted parameters). float64.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.config import ModelParams
from src.v3.estimation.collector import collect_dataset, collect_dataset_batch
from src.v3.estimation.estimate import estimate
from src.v3.estimation.surrogate import SurrogateEnsemble
from src.v3.estimation.weighting import weighting_matrix
from src.v3.simulation.moments import compute_moments
from src.v3.simulation.panel import simulate_panel
from src.v3.solver import refine as _refine
from src.v3.validation import vfi as _vfi


def r2(true, fitted):
    """Per-column R^2 of fitted against true on the 45-degree line."""
    true, fitted = np.asarray(true), np.asarray(fitted)
    ss_res = np.sum((fitted - true) ** 2, axis=0)
    ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def run_recovery(bundle, bounds, ext, grid_cfg, master_seed, n_draws, *,
                 collect_rows=200, collect_batch_size=16, surrogate_passes=200,
                 n_firms=2000, T=120, burn_in=60,
                 refine_rounds=6, n_restarts=30, max_redraws=200, verbose=False):
    dtype = TF_FLOAT_NUM
    lo = tf.constant(bounds.lower_array(), dtype)
    hi = tf.constant(bounds.upper_array(), dtype)

    # One collection + surrogate training over the fixed box. The collection is the
    # bottleneck (per-draw refine + simulate); the batched (GPU-default) collector
    # processes ``collect_batch_size`` draws at once. ``collect_batch_size=1`` is the
    # serial path. (Device chosen by precision.configure_devices: GPU on CUDA, CPU on Metal.)
    collect_seed = int(seeding.key(master_seed, seeding.Purpose.COLLECT, 7)[0])
    collect_kw = dict(refine_rounds=refine_rounds, n_firms=n_firms, T=T, burn_in=burn_in)
    if collect_batch_size > 1:
        beta_ds, m_ds = collect_dataset_batch(bundle, bounds, ext, grid_cfg, collect_seed,
                                              collect_rows, batch_size=collect_batch_size,
                                              **collect_kw)
    else:
        beta_ds, m_ds = collect_dataset(bundle, bounds, ext, grid_cfg, collect_seed,
                                        collect_rows, **collect_kw)
    surr = SurrogateEnsemble(master_seed)
    surr.train(beta_ds, m_ds, bounds, master_seed, passes=surrogate_passes)

    true_beta, est_beta, true_m, fit_m = [], [], [], []
    draw = 0
    while len(true_beta) < n_draws and draw < n_draws + max_redraws:
        u = seeding.uniform([8], master_seed, seeding.Purpose.COLLECT, 9, draw, dtype=dtype)
        beta_star = (lo + u * (hi - lo)).numpy()
        params = ModelParams.from_array(beta_star)
        sim_seed = int(seeding.key(master_seed, seeding.Purpose.SIM_INIT, draw)[0])

        vfi_sol = _vfi.solve_vfi(params, ext, grid_cfg, mode="howard", tol=1e-8, max_sweeps=200)
        panel = simulate_panel(vfi_sol, params, ext, grid_cfg, sim_seed,
                               n_firms=n_firms, T=T, burn_in=burn_in)
        draw += 1
        if int(tf.reduce_sum(tf.cast(panel["V"] < 0.0, tf.int32))) == 0:
            continue  # no defaults -> chi unidentified; discard (Sec 12.2 step 4)

        m_star = compute_moments(panel)
        if not bool(tf.reduce_all(tf.math.is_finite(m_star))):
            continue  # degenerate panel (no good observations); discard

        W, _ = weighting_matrix(panel)
        beta_hat = estimate(surr, m_star, W, bounds, master_seed, n_restarts=n_restarts)["beta_hat"]

        fit_params = ModelParams.from_array(beta_hat.numpy())
        refined = _refine.refine(bundle, fit_params, ext, grid_cfg, bounds, n_rounds=refine_rounds)
        panel_fit = simulate_panel(refined, fit_params, ext, grid_cfg, sim_seed,
                                   n_firms=n_firms, T=T, burn_in=burn_in)
        m_fit = compute_moments(panel_fit)
        if not bool(tf.reduce_all(tf.math.is_finite(m_fit))):
            continue  # refined solution at beta_hat degenerate; discard

        true_beta.append(beta_star)
        est_beta.append(beta_hat.numpy())
        true_m.append(m_star.numpy())
        fit_m.append(m_fit.numpy())
        if verbose:
            print(f"  recovery draw {len(true_beta)}/{n_draws} (attempt {draw})")

    out = {k: np.array(v) for k, v in
           dict(true_beta=true_beta, est_beta=est_beta, true_m=true_m, fit_m=fit_m).items()}
    out["moment_r2"] = r2(out["true_m"], out["fit_m"])
    out["param_r2"] = r2(out["true_beta"], out["est_beta"])
    out["surrogate_oos_r2"] = surr.oos_r2().numpy()   # on the trained-on (capped) rows
    return out
