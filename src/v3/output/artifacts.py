"""Run directory + manifest + figure/array/summary helpers (v2 artifacts pattern).

A run context dict carries the directory layout. Arrays are saved as ``.npz``,
metadata as ``manifest.json``, summaries as CSV; a ``latest`` symlink points at the
most recent run (best-effort).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict


def prepare_run(experiment_name: str, results_root: str = "outputs/v3",
                run_tag: str = "run", save: bool = True) -> Dict:
    """Create (or describe) a run directory and return the run context."""
    root = Path(results_root) / experiment_name
    run_dir = root / run_tag
    ctx = {
        "experiment": experiment_name,
        "run_dir": run_dir,
        "figures_dir": run_dir / "figures",
        "arrays_dir": run_dir / "arrays",
        "manifest_path": run_dir / "manifest.json",
        "save": save,
    }
    if save:
        ctx["figures_dir"].mkdir(parents=True, exist_ok=True)
        ctx["arrays_dir"].mkdir(parents=True, exist_ok=True)
        latest = root / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        except OSError:
            pass  # symlinks may be unavailable; not fatal
    return ctx


def save_manifest(ctx: Dict, **sections) -> None:
    if not ctx.get("save"):
        return
    ctx["manifest_path"].write_text(json.dumps(sections, indent=2, default=str))


def save_figure(ctx: Dict, fig, name: str, dpi: int = 150):
    if not ctx.get("save"):
        return None
    path = ctx["figures_dir"] / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_arrays(ctx: Dict, name: str, **arrays) -> None:
    if not ctx.get("save"):
        return
    import numpy as np
    np.savez(ctx["arrays_dir"] / f"{name}.npz", **arrays)


def save_summary(ctx: Dict, rows, name: str = "summary.csv") -> None:
    if not ctx.get("save") or not rows:
        return
    path = ctx["run_dir"] / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _recovery_markdown(out, gates) -> str:
    import numpy as np
    from src.v3.validation import figures
    mr, pr = out["moment_r2"], out["param_r2"]
    mg, pg = gates
    n_pass = int(np.sum(pr >= pg))
    lines = [
        f"# Recovery summary ({out.get('profile')}, device={out.get('device')})", "",
        f"- draws kept: {out['true_beta'].shape[0]}",
        f"- surrogate OOS R^2 (mean): {float(np.mean(out['surrogate_oos_r2'])):.4f}",
        f"- moment R^2 (mean): {float(np.mean(mr)):.4f}  "
        f"(Fig V1 gate all >= {mg}: {'PASS' if bool(np.all(mr >= mg)) else 'fail'})",
        f"- params with R^2 >= {pg}: {n_pass}/8  "
        f"(Fig V2 gate >= 7/8: {'PASS' if n_pass >= 7 else 'fail'})",
        "", "## Parameter R^2", "", "| param | R^2 |", "|---|---|",
        *[f"| {n} | {pr[j]:.3f} |" for j, n in enumerate(figures.PARAM_NAMES)],
        "", "## Moment R^2", "", "| moment | R^2 |", "|---|---|",
        *[f"| {n} | {mr[j]:.3f} |" for j, n in enumerate(figures.MOMENT_NAMES)],
    ]
    return "\n".join(lines) + "\n"


def save_recovery(out, bundle=None, *, experiment="df26_recovery",
                  results_root="outputs/v3", run_tag=None, save_checkpoints=True,
                  gates=(0.99, 0.95)):
    """Persist a complete recovery run so the outputs never need re-computing.

    Writes, under ``results_root/experiment/<tag>/``: the recovery arrays
    (``arrays/recovery.npz``) and the collected dataset (``arrays/dataset.npz``); Figs
    V1/V2 (png + pdf); per-parameter and per-moment R^2 tables (csv) and ``summary.md``;
    a ``manifest.json`` with the headline numbers and gate pass/fail; and (optionally)
    the trained network and surrogate checkpoints. Returns the run directory.
    """
    import numpy as np
    from datetime import datetime
    from src.v3.validation import figures

    tag = run_tag or f"{out.get('profile', 'run')}_{datetime.now():%Y%m%d_%H%M%S}"
    ctx = prepare_run(experiment, results_root, tag)

    arrays = dict(true_beta=out["true_beta"], est_beta=out["est_beta"],
                  true_m=out["true_m"], fit_m=out["fit_m"],
                  moment_r2=out["moment_r2"], param_r2=out["param_r2"],
                  surrogate_oos_r2=out["surrogate_oos_r2"])
    for key in ("est_beta_folds", "fit_m_se"):   # 95% CI inputs, if present
        if out.get(key) is not None and np.size(out[key]):
            arrays[key] = np.asarray(out[key])
    save_arrays(ctx, "recovery", **arrays)
    if out.get("dataset_beta") is not None:   # the collection (bottleneck) -> retrain w/o recollecting
        save_arrays(ctx, "dataset", beta=np.asarray(out["dataset_beta"]),
                    moments=np.asarray(out["dataset_moments"]))

    import matplotlib.pyplot as plt
    for name, fn in (("fig_v1_moments", figures.plot_recovery_moments),
                     ("fig_v2_params", figures.plot_recovery_params)):
        fig = fn(out)
        save_figure(ctx, fig, name + ".png")
        save_figure(ctx, fig, name + ".pdf")
        plt.close(fig)

    pr, mr = out["param_r2"], out["moment_r2"]
    save_summary(ctx, [{"param": n, "true_mean": float(out["true_beta"][:, j].mean()),
                        "est_mean": float(out["est_beta"][:, j].mean()), "r2": float(pr[j])}
                       for j, n in enumerate(figures.PARAM_NAMES)], "param_r2.csv")
    save_summary(ctx, [{"moment": n, "r2": float(mr[j])}
                       for j, n in enumerate(figures.MOMENT_NAMES)], "moment_r2.csv")
    if ctx.get("save"):
        (ctx["run_dir"] / "summary.md").write_text(_recovery_markdown(out, gates))

    mg, pg = gates
    save_manifest(ctx, profile=out.get("profile"), device=out.get("device"),
                  draws=int(out["true_beta"].shape[0]),
                  moment_r2_mean=float(np.mean(mr)),
                  param_r2=dict(zip(figures.PARAM_NAMES, [float(x) for x in pr])),
                  fig_v1_pass=bool(np.all(mr >= mg)),
                  fig_v2_pass=int(np.sum(pr >= pg)) >= 7)

    if save_checkpoints and ctx.get("save") and bundle is not None:
        from src.v3.output.checkpoints import save_bundle
        save_bundle(bundle, str(ctx["run_dir"] / "bundle_ckpt"))
    if save_checkpoints and ctx.get("save") and out.get("surrogate") is not None:
        import tensorflow as tf
        tf.train.Checkpoint(surrogate=out["surrogate"]).write(str(ctx["run_dir"] / "surrogate_ckpt"))
    return ctx["run_dir"]


def load_recovery(run_dir):
    """Reload a saved recovery into an ``out``-like dict (re-plot/re-tabulate without re-running)."""
    import numpy as np
    d = np.load(Path(run_dir) / "arrays" / "recovery.npz")
    out = {k: d[k] for k in d.files}
    ds = Path(run_dir) / "arrays" / "dataset.npz"
    if ds.exists():
        dd = np.load(ds)
        out["dataset_beta"], out["dataset_moments"] = dd["beta"], dd["moments"]
    return out
