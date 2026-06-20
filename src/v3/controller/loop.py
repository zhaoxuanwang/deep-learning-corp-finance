"""Serial controller loop (DF26 Sec 2 end-to-end schedule, Sec 7 serial mode).

Runs the controller's per-epoch schedule in a single process: train Block 1 a bit, collect
a Block-2 batch into the shared buffer, then (every ``controller_every`` epochs) retrain the
surrogates, run the LM estimation, and -- after the warm-up -- attempt a bound shrink under
the Section-6 guards. Recompute-on-shrink is automatic: every block derives its normalizers
from the current ``bounds`` (a scaler swap), with no network re-initialization.

Section 7 makes clear this serial form is a debugging aid; the production architecture is the
1-trainer / 3-collector asynchronous engine (see :mod:`src.v3.controller.async_engine`). The
serial loop runs the same math, just without the concurrency, so it is what the tests and the
M-series / single-GPU runs use. float64 numerics, float32 networks.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.v3.common import seeding
from src.v3.controller.dataset import DatasetBuffer
from src.v3.controller.min_loss import minimum_loss_profile
from src.v3.controller.shrinkage import shrink_bounds
from src.v3.estimation.collector import collect_dataset_batch
from src.v3.estimation.estimate import estimate
from src.v3.estimation.surrogate import SurrogateEnsemble
from src.v3.solver import trainer


def run_controller(bundle, bounds, ext, profile, master_seed, target_m, W, *,
                   n_epochs, verbose=False):
    """Run the serial controller for ``n_epochs``; returns the final bounds, surrogate, and history.

    ``target_m`` [11] and ``W`` [11,11] are the (data or synthetic) estimation target and
    weighting matrix; ``profile`` is a :class:`src.v3.profiles.Profile`.
    """
    ctrl = profile.controller
    buffer = DatasetBuffer(ctrl.surrogate_max_obs)
    surr = SurrogateEnsemble(master_seed, n_folds=ctrl.n_folds)
    target_np = np.asarray(target_m, np.float64)
    W_np = W.numpy() if tf.is_tensor(W) else np.asarray(W, np.float64)
    history = {"bounds": [], "beta_hat": [], "oos_r2": [], "shrunk": []}
    state = {"beta_hat": None, "oos_r2": None}

    for epoch in range(n_epochs):
        # Block 1: continue training under the current bounds.
        trainer.train_block1(bundle, bounds, ext, profile.grid, profile.train,
                             master_seed=master_seed, n_epochs=1, compile_step=True)

        # Block 2: collect a batch and append to the shared buffer.
        cseed = int(seeding.key(master_seed, seeding.Purpose.COLLECT, 10_000 + epoch)[0])
        beta_b, m_b = collect_dataset_batch(
            bundle, bounds, ext, profile.grid, cseed, profile.collect_batch_size,
            batch_size=profile.collect_batch_size, refine_rounds=profile.refine_rounds,
            n_firms=profile.n_firms, T=profile.T, burn_in=profile.burn_in)
        buffer.add(beta_b, m_b)

        shrunk = False
        if epoch % profile.controller_every == 0 and len(buffer) >= 2 * ctrl.n_folds:
            # End-of-epoch controller step: retrain surrogates, estimate, (maybe) shrink.
            surr.train(buffer.betas, buffer.moments, bounds, master_seed,
                       passes=profile.surrogate_passes, max_obs=ctrl.surrogate_max_obs)
            est = estimate(surr, target_m, W, bounds, master_seed, n_restarts=ctrl.n_restarts)
            state["beta_hat"] = est["beta_hat"]
            state["oos_r2"] = surr.oos_r2()

            if epoch >= ctrl.warmup_epochs:
                prof = minimum_loss_profile(surr, target_m, W, bounds, master_seed, ctrl)
                rb, rm = buffer.recent(ctrl.containment_recent)
                shrink = shrink_bounds(
                    prof, bounds, ctrl, recent_betas=rb.numpy(), recent_moments=rm.numpy(),
                    target_m=target_np, W=W_np, lm_estimates=est["all_betas"].numpy().reshape(-1, 8))
                if shrink.shrunk:
                    bounds = shrink.bounds
                    shrunk = True

        history["bounds"].append(bounds)
        history["beta_hat"].append(state["beta_hat"])
        history["oos_r2"].append(state["oos_r2"])
        history["shrunk"].append(shrunk)
        if verbose:
            r2 = None if state["oos_r2"] is None else round(float(tf.reduce_mean(state["oos_r2"])), 3)
            print(f"  epoch {epoch}: rows={len(buffer)} oos_r2={r2} shrunk={shrunk}")

    return {"bounds": bounds, "surrogate": surr, "buffer": buffer,
            "beta_hat": state["beta_hat"], "history": history}
