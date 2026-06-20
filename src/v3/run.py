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


def train_and_recover(profile="MEDIUM", master_seed=20260619, device="auto", *,
                      train_epochs=None, recovery_draws=None, verbose=True,
                      save=True, results_root="outputs/v3", run_tag=None):
    """Train Block 1 at the profile's scale, then run the recovery (Sec 12.2). Returns the
    recovery dict (true/est params + moments, R^2s) with ``profile``/``device``/``run_dir`` added.

    When ``save`` (default), every output is persisted under ``results_root`` (arrays, Figs
    V1/V2, R^2 tables, summary.md, manifest, and the network + surrogate checkpoints) so the
    expensive run never has to be repeated just to recover its outputs. On Colab set
    ``results_root`` to a Google Drive path so it survives the runtime."""
    mode, prof, bounds, ext, bundle = _setup(
        profile, device, master_seed,
        {"train_epochs": train_epochs, "recovery_draws": recovery_draws})
    if verbose:
        print(f"[v3] recover | profile={prof.name} device={mode} states={prof.grid.n_states} "
              f"train_epochs={prof.train_epochs} collect_rows={prof.collect_rows} "
              f"batch={prof.collect_batch_size} draws={prof.recovery_draws} "
              f"surrogate={prof.surrogate_passes}x{prof.surrogate_hidden} restarts={prof.n_restarts}")
    _t = time.perf_counter()
    trainer.train_block1(bundle, bounds, ext, prof.grid, prof.train,
                         master_seed=master_seed, n_epochs=prof.train_epochs, compile_step=True)
    t_train = time.perf_counter() - _t
    out = run_recovery(
        bundle, bounds, ext, prof.grid, master_seed, n_draws=prof.recovery_draws,
        collect_rows=prof.collect_rows, collect_batch_size=prof.collect_batch_size,
        surrogate_passes=prof.surrogate_passes, surrogate_hidden=prof.surrogate_hidden,
        n_firms=prof.n_firms, T=prof.T, burn_in=prof.burn_in, refine_rounds=prof.refine_rounds,
        n_restarts=prof.n_restarts, verbose=verbose)
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
