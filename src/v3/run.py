"""One-call experiment runners across scale profiles and devices (DF26 Sec 12.2, 6).

A single entry point used identically on an M-series Mac and on a CUDA GPU (Colab): pick
a :mod:`src.v3.profiles` scale preset and a device mode; the code path is the same, only
the sizes and the device change. ``device="auto"`` runs float64 numerics on the GPU under
CUDA and on the CPU on Apple Metal / no-GPU (see :func:`src.v3.common.precision.configure_devices`).

* :func:`train_and_recover` -- Block 1 training + the Monte-Carlo recovery (Figs V1/V2 data).
* :func:`run_adaptive_controller` -- the serial Section-6 controller loop (adaptive shrinkage).

Typical use::

    from src.v3.run import train_and_recover
    out = train_and_recover(profile="MEDIUM")            # CPU-affordable, minutes on M1
    out = train_and_recover(profile="FULL", device="auto")   # paper scale, a CUDA GPU
"""
from __future__ import annotations

import time

from src.v3.common.precision import configure_devices, silence_logging
from src.v3.config import ExternalParams, NetworkConfig, ParamBounds
from src.v3.networks.bundle import NetworkBundle
from src.v3.profiles import get_profile
from src.v3.solver import trainer
from src.v3.validation.recovery import run_recovery


def _setup(profile, device, master_seed, overrides):
    silence_logging()
    mode = configure_devices(device)
    prof = get_profile(profile)
    if overrides:
        prof = prof.with_(**{k: v for k, v in overrides.items() if v is not None})
    bounds = ParamBounds.table_a1()
    ext = ExternalParams()
    bundle = NetworkBundle(NetworkConfig(), master_seed=master_seed)
    return mode, prof, bounds, ext, bundle


def _resume_dir(results_root, prof, master_seed):
    """Isolated, per-(profile, seed) checkpoint dir, separate from the saved run directories."""
    from pathlib import Path
    return Path(results_root) / ".resume" / f"{prof.name}_seed{master_seed}"


def _resume_meta_ok(resume_dir, prof, master_seed):
    """True iff a complete, config-matching insurance checkpoint exists (else start fresh).

    Validates against everything the saved bundle + dataset depend on: profile, seed, grid,
    train_epochs, collect_rows. Any mismatch, missing file, or read error returns False.
    """
    import json
    try:
        meta = json.loads((resume_dir / "meta.json").read_text())
        g = prof.grid
        return (meta.get("profile") == prof.name and meta.get("master_seed") == master_seed
                and meta.get("collect_rows") == prof.collect_rows
                and meta.get("train_epochs") == prof.train_epochs
                and meta.get("grid") == [g.n_z, g.n_k, g.n_b, g.n_kp, g.n_bp, g.n_cp]
                and (resume_dir / "bundle_ckpt.index").exists()
                and (resume_dir / "dataset.npz").exists())
    except Exception:
        return False


def _try_load_resume(resume_dir, bundle, prof, master_seed, verbose):
    """Restore the bundle and return the collected dataset from a valid checkpoint, else None.

    Loads the Block-1 bundle weights in place and returns ``(beta_ds, m_ds)`` so the caller can
    skip training and collection. Any failure falls back to a fresh run (returns None).
    """
    if resume_dir is None or not _resume_meta_ok(resume_dir, prof, master_seed):
        return None
    try:
        import numpy as np
        import tensorflow as tf
        from src.v3.common.precision import TF_FLOAT_NUM
        from src.v3.output.checkpoints import load_bundle
        load_bundle(bundle, str(resume_dir / "bundle_ckpt"))
        d = np.load(resume_dir / "dataset.npz")
        dataset = (tf.constant(d["beta"], TF_FLOAT_NUM), tf.constant(d["moments"], TF_FLOAT_NUM))
        if verbose:
            print(f"[v3] RESUMED from {resume_dir}: loaded Block-1 bundle + dataset "
                  "(skipping train + collection)")
        return dataset
    except Exception as e:
        if verbose:
            print(f"[v3] resume checkpoint unusable ({e}); starting fresh")
        return None


def _save_resume_checkpoint(resume_dir, bundle, beta_ds, m_ds, prof, master_seed):
    """Persist Block-1 bundle + collected dataset BEFORE the recovery loop (crash insurance).

    Wipes the dir first (clean overwrite) and writes ``meta.json`` LAST, so a half-written
    checkpoint never validates. Never raises: a checkpoint failure must not abort a good run.
    """
    import json
    import shutil
    import numpy as np
    from src.v3.output.checkpoints import save_bundle
    try:
        shutil.rmtree(resume_dir, ignore_errors=True)
        resume_dir.mkdir(parents=True, exist_ok=True)
        save_bundle(bundle, str(resume_dir / "bundle_ckpt"))
        np.savez(resume_dir / "dataset.npz", beta=np.asarray(beta_ds), moments=np.asarray(m_ds))
        g = prof.grid
        (resume_dir / "meta.json").write_text(json.dumps({
            "profile": prof.name, "master_seed": master_seed, "collect_rows": prof.collect_rows,
            "train_epochs": prof.train_epochs, "grid": [g.n_z, g.n_k, g.n_b, g.n_kp, g.n_bp, g.n_cp],
            "surrogate_passes": prof.surrogate_passes, "surrogate_hidden": prof.surrogate_hidden}))
    except Exception:
        pass   # insurance is best-effort; never break the run over it


def train_and_recover(profile="MEDIUM", master_seed=20260619, device="auto", *,
                      train_epochs=None, recovery_draws=None, verbose=True,
                      save=True, resume=True, results_root="outputs/v3", run_tag=None):
    """Train Block 1 at the profile's scale, then run the recovery (Sec 12.2). Returns the
    recovery dict (true/est params + moments, R^2s) with ``profile``/``device``/``run_dir`` added.

    When ``save`` (default), every output is persisted under ``results_root`` (arrays, Figs
    V1/V2, R^2 tables, summary.md, manifest, and the network + surrogate checkpoints) so the
    expensive run never has to be repeated just to recover its outputs. On Colab set
    ``results_root`` to a Google Drive path so it survives the runtime.

    Crash insurance (``resume``, on by default when ``save``): before the long per-draw recovery
    loop, the trained Block-1 bundle and the collected dataset are checkpointed to
    ``results_root/.resume/<profile>_seed<seed>/``. If the runtime dies mid-loop (e.g. a Colab
    disconnect), simply re-run: a config-matching checkpoint is detected and reloaded, skipping
    the expensive train + collection (the surrogate is cheap and deterministic, so it is just
    retrained). The checkpoint is isolated from the saved run directories, is overwritten safely
    (meta written last; a partial checkpoint never validates), never aborts the run on failure,
    and is removed once the run finishes cleanly. A change to profile/seed/grid/epochs/collect_rows
    invalidates it; pass ``resume=False`` to force a fresh run."""
    mode, prof, bounds, ext, bundle = _setup(
        profile, device, master_seed,
        {"train_epochs": train_epochs, "recovery_draws": recovery_draws})
    if verbose:
        print(f"[v3] recover | profile={prof.name} device={mode} states={prof.grid.n_states} "
              f"train_epochs={prof.train_epochs} collect_rows={prof.collect_rows} "
              f"batch={prof.collect_batch_size} draws={prof.recovery_draws} "
              f"surrogate={prof.surrogate_passes}x{prof.surrogate_hidden} restarts={prof.n_restarts}")

    # Try to resume from a config-matching insurance checkpoint (skip train + collection).
    resume_dir = _resume_dir(results_root, prof, master_seed) if (save and resume) else None
    dataset = _try_load_resume(resume_dir, bundle, prof, master_seed, verbose)

    if dataset is not None:
        t_train = 0.0
    else:
        _t = time.perf_counter()
        trainer.train_block1(bundle, bounds, ext, prof.grid, prof.train,
                             master_seed=master_seed, n_epochs=prof.train_epochs, compile_step=True)
        t_train = time.perf_counter() - _t

    # On a fresh run, checkpoint bundle + dataset once collection is done, before the draw loop.
    on_ready = None
    if resume_dir is not None and dataset is None:
        on_ready = lambda b, m: _save_resume_checkpoint(resume_dir, bundle, b, m, prof, master_seed)

    out = run_recovery(
        bundle, bounds, ext, prof.grid, master_seed, n_draws=prof.recovery_draws,
        collect_rows=prof.collect_rows, collect_batch_size=prof.collect_batch_size,
        surrogate_passes=prof.surrogate_passes, surrogate_hidden=prof.surrogate_hidden,
        n_firms=prof.n_firms, T=prof.T, burn_in=prof.burn_in, refine_rounds=prof.refine_rounds,
        n_restarts=prof.n_restarts, dataset=dataset, on_dataset_ready=on_ready, verbose=verbose)
    out["profile"], out["device"] = prof.name, mode
    out["bundle"], out["grid"] = bundle, prof.grid   # for in-session Block-1 diagnostics (slices, etc.)
    out["timings"]["train_s"] = t_train               # per-phase wall times (s), to calibrate scale
    if verbose:
        ti = out["timings"]
        print(f"[v3] timings (s): train={ti['train_s']:.0f} collect={ti['collect_s']:.0f} "
              f"surrogate={ti['surrogate_s']:.0f} recovery_loop={ti['recovery_loop_s']:.0f} "
              f"| total={sum(ti.values()):.0f} ({sum(ti.values())/3600:.2f} h)")
    if save:
        from src.v3.output.artifacts import save_recovery
        out["run_dir"] = str(save_recovery(out, bundle, results_root=results_root, run_tag=run_tag))
        if verbose:
            print(f"[v3] artifacts saved to {out['run_dir']}")
        if resume_dir is not None:   # finished cleanly -> drop the insurance checkpoint
            import shutil
            shutil.rmtree(resume_dir, ignore_errors=True)
    return out


def run_adaptive_controller(target_m, W, profile="MEDIUM", master_seed=20260619, device="auto",
                            *, n_epochs=None, verbose=True):
    """Run the serial Section-6 controller loop (train + collect + estimate + adaptive shrink).

    ``target_m`` [11] and ``W`` [11,11] are the estimation target and weighting matrix.
    Returns the controller dict (final bounds, surrogate, beta_hat, history)."""
    from src.v3.controller.loop import run_controller
    mode, prof, bounds, ext, bundle = _setup(profile, device, master_seed, None)
    epochs = n_epochs if n_epochs is not None else prof.train_epochs
    if verbose:
        print(f"[v3] controller | profile={prof.name} device={mode} epochs={epochs} "
              f"warmup={prof.controller.warmup_epochs}")
    out = run_controller(bundle, bounds, ext, prof, master_seed, target_m, W,
                         n_epochs=epochs, verbose=verbose)
    out["profile"], out["device"] = prof.name, mode
    return out
