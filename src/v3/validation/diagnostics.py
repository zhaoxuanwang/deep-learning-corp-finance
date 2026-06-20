"""Free post-estimation diagnostics for a saved recovery run (DF26 Sec 12.2 follow-up).

Everything here reads a *saved* recovery (the arrays + the collected dataset written by
:func:`src.v3.output.artifacts.save_recovery`) and, at most, RE-TRAINS the tiny moment
surrogate on the already-collected dataset. It never re-runs Block 1 or the Block-2
collection, so the whole module runs in seconds to a couple of minutes regardless of the
profile that produced the run.

The point is to decide, before paying for a longer (e.g. FULL) run, *why* the surrogate
OOS R^2 is what it is:

* ``surrogate_sweep`` retrains the surrogate at several (passes, hidden) settings on the
  same dataset. If OOS R^2 climbs, the ceiling is surrogate capacity/training (a cheap
  config fix); if it plateaus, the dataset itself is the limit (needs more ``collect_rows``,
  the expensive lever, with 8D-curse diminishing returns).
* ``oracle_check`` evaluates the surrogate at the *true* draws and compares to the true
  moments. A tight oracle with a loose end-to-end R^2 puts the loss in the LM/refine step
  or in identification, not in the surrogate.
* ``identification`` reads the surrogate Jacobian dm/dbeta: per-parameter sensitivity and
  the condition number / weakest-identified direction of the standardized sensitivity
  matrix. Flat directions are weak identification that no amount of data or training fixes.
* ``moment_table`` / ``param_table`` summarize the saved arrays (per-moment variation and
  surrogate fit; per-parameter R^2, bias, RMSE, across-fold CI width).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NUM
from src.v3.config import ParamBounds
from src.v3.estimation.surrogate import SurrogateEnsemble
from src.v3.validation import figures
from src.v3.validation.recovery import r2

DEFAULT_SEED = 20260619


def moment_table(out):
    """Per-moment diagnostics from the saved arrays only (instant, no retraining).

    Columns: surrogate OOS R^2; cross-draw SD of the true and fitted moment; the coefficient
    of variation ``cv`` of the true moment (scale-free spread across the box); the end-to-end
    moment R^2 (Fig V1); the mean fitted-moment clustered SE; and an ``uninformative`` flag.

    A moment is flagged ``uninformative`` if the surrogate cannot predict it from beta
    (OOS R^2 < 0.1, i.e. it is panel noise rather than a smooth function of the parameters) or
    if it barely moves across the box (cv < 0.05). Either way it carries little identifying
    information no matter how good the surrogate. Note that a noisy moment can have a *large*
    cv (mean near zero) yet still be uninformative, so the OOS R^2 condition is the operative one.
    """
    tm, fm = np.asarray(out["true_m"]), np.asarray(out["fit_m"])
    soos, mr = np.asarray(out["surrogate_oos_r2"]), np.asarray(out["moment_r2"])
    se = out.get("fit_m_se")
    se = np.asarray(se).mean(0) if se is not None and np.size(se) else np.full(tm.shape[1], np.nan)
    true_sd = tm.std(0)
    cv = true_sd / (np.abs(tm.mean(0)) + 1e-12)
    return [{"moment": n, "surrogate_oos_r2": round(float(soos[j]), 4),
             "true_sd": round(float(true_sd[j]), 5), "fit_sd": round(float(fm[:, j].std()), 5),
             "cv": round(float(cv[j]), 4), "moment_r2": round(float(mr[j]), 4),
             "mean_fit_se": round(float(se[j]), 5),
             "uninformative": bool(soos[j] < 0.1 or cv[j] < 0.05)}
            for j, n in enumerate(figures.MOMENT_NAMES)]


def param_table(out):
    """Per-parameter diagnostics from the saved arrays only (instant).

    Columns: param R^2 (Fig V2); bias and RMSE (est - true); the mean across-fold SD of the
    estimate and the implied 95% CI half-width; and a ``weak`` flag (R^2 < 0.5).
    """
    tb, eb = np.asarray(out["true_beta"]), np.asarray(out["est_beta"])
    pr = np.asarray(out["param_r2"])
    folds = out.get("est_beta_folds")
    psd = (np.std(np.asarray(folds), axis=1).mean(0) if folds is not None and np.size(folds)
           else np.full(eb.shape[1], np.nan))
    return [{"param": n, "param_r2": round(float(pr[j]), 4),
             "bias": round(float((eb[:, j] - tb[:, j]).mean()), 5),
             "rmse": round(float(np.sqrt(((eb[:, j] - tb[:, j]) ** 2).mean())), 5),
             "across_fold_sd": round(float(psd[j]), 5),
             "ci95_halfwidth": round(float(1.96 * psd[j]), 5),
             "weak": bool(pr[j] < 0.5)}
            for j, n in enumerate(figures.PARAM_NAMES)]


def _train_surrogate(out, master_seed, passes, hidden):
    """Fit a fresh surrogate on the SAVED dataset (no solver/sim). Seconds, not hours."""
    bounds = ParamBounds.table_a1()
    beta = tf.convert_to_tensor(np.asarray(out["dataset_beta"]), TF_FLOAT_NUM)
    m = tf.convert_to_tensor(np.asarray(out["dataset_moments"]), TF_FLOAT_NUM)
    surr = SurrogateEnsemble(master_seed, hidden=hidden)
    surr.train(beta, m, bounds, master_seed, passes=passes)
    return surr


def surrogate_sweep(out, master_seed=DEFAULT_SEED, passes_grid=(200, 400, 800),
                    hidden_grid=(32, 64)):
    """Retrain the surrogate across (passes, hidden) on the saved dataset; report OOS R^2.

    Returns ``(rows, best_surrogate, best_row)`` where ``rows`` is one dict per setting with
    mean / median / min per-moment OOS R^2. The baseline (passes=200, hidden=32) reproduces
    the run's own surrogate. The whole sweep re-uses the collected dataset, so it is free.
    """
    rows, best = [], None
    for h in hidden_grid:
        for p in passes_grid:
            surr = _train_surrogate(out, master_seed, p, h)
            oos = surr.oos_r2().numpy()
            row = {"hidden": h, "passes": p, "mean_oos_r2": round(float(oos.mean()), 4),
                   "median_oos_r2": round(float(np.median(oos)), 4),
                   "min_oos_r2": round(float(oos.min()), 4)}
            rows.append(row)
            if best is None or row["mean_oos_r2"] > best[0]:
                best = (row["mean_oos_r2"], surr, row)
    return rows, best[1], best[2]


def sweep_verdict(rows, baseline=(32, 200)):
    """Classify the surrogate ceiling from the sweep: capacity/training- vs data-limited."""
    bh, bp = baseline
    base = next((r for r in rows if r["hidden"] == bh and r["passes"] == bp), rows[0])
    best = max(rows, key=lambda r: r["mean_oos_r2"])
    gain = best["mean_oos_r2"] - base["mean_oos_r2"]
    if gain > 0.10:
        verdict = ("capacity/training-limited (CHEAP): the saved dataset already supports a "
                   "better fit; raise surrogate passes/hidden, no new collection needed.")
    elif gain < 0.03:
        verdict = ("data/identification-limited (EXPENSIVE): OOS R^2 plateaus on this dataset; "
                   "gains need more collect_rows, but the 8D curse means 4x rows ~ 1.2x per-dim "
                   "density, so expect diminishing returns.")
    else:
        verdict = ("partially capacity-limited: some cheap gain from passes/hidden, but a "
                   "ceiling remains -> more data also needed.")
    return {"baseline_mean_oos_r2": base["mean_oos_r2"], "best_mean_oos_r2": best["mean_oos_r2"],
            "best_config": {"hidden": best["hidden"], "passes": best["passes"]},
            "gain": round(float(gain), 4), "verdict": verdict}


def oracle_check(out, surr):
    """Surrogate evaluated at the TRUE draws vs the true moments (separates surrogate error).

    Per moment: oracle R^2 (squared corr of surrogate(beta*) with m*) and oracle RMSE,
    alongside the end-to-end moment R^2. Oracle >> end-to-end means the loss is downstream
    (LM / refine / identification), not the surrogate.
    """
    scaler = ParamScaler(ParamBounds.table_a1())
    tb = tf.convert_to_tensor(np.asarray(out["true_beta"]), TF_FLOAT_NUM)
    pred = tf.reduce_mean(surr.forward_all(scaler.normalize(tb)), axis=2).numpy()   # [N, M] folds-averaged
    tm = np.asarray(out["true_m"])
    orc, mr = r2(tm, pred), np.asarray(out["moment_r2"])
    rmse = np.sqrt(((tm - pred) ** 2).mean(0))
    return [{"moment": n, "oracle_r2": round(float(orc[j]), 4),
             "oracle_rmse": round(float(rmse[j]), 5),
             "end_to_end_moment_r2": round(float(mr[j]), 4)}
            for j, n in enumerate(figures.MOMENT_NAMES)]


def identification(out, surr):
    """Local identification from the surrogate Jacobian dm/dbeta at the mean draw.

    Standardizes each moment row by its cross-draw SD (so every moment is in units of its
    own variation), then reports: per-parameter sensitivity (column 2-norm), the condition
    number and singular values of the standardized sensitivity matrix, and the weakest-
    identified direction (loadings of the smallest singular vector across parameters).
    A near-flat direction (large condition number) is weak identification.
    """
    scaler = ParamScaler(ParamBounds.table_a1())
    center = np.asarray(out["true_beta"]).mean(0)[None, :]
    bn = tf.Variable(scaler.normalize(tf.convert_to_tensor(center, TF_FLOAT_NUM)))
    with tf.GradientTape() as tape:
        pred = tf.reduce_mean(surr.forward_all(bn), axis=2)        # [1, M] folds-averaged
    J = tf.reshape(tape.jacobian(pred, bn), [surr.M, bn.shape[1]]).numpy()   # dm / d(beta_norm)
    S = J / (np.asarray(out["true_m"]).std(0)[:, None] + 1e-12)   # moment-standardized sensitivity
    col = np.linalg.norm(S, axis=0)
    _, sv, Vt = np.linalg.svd(S, full_matrices=False)
    return {"per_param_sensitivity": [{"param": n, "sensitivity": round(float(col[j]), 4)}
                                      for j, n in enumerate(figures.PARAM_NAMES)],
            "condition_number": round(float(sv[0] / (sv[-1] + 1e-12)), 1),
            "singular_values": [round(float(x), 4) for x in sv],
            "weakest_direction": {n: round(float(Vt[-1, j]), 3)
                                  for j, n in enumerate(figures.PARAM_NAMES)}}


def fixed_weight(out):
    """Diagonal inverse-variance weighting from the collected moments.

    Held FIXED across a before/after surrogate comparison so that the only thing that
    changes is the surrogate. Inverse-variance is a standard GMM weighting and it naturally
    down-weights noisy moments. (This is not the run's own Erickson-Whited per-draw W, which
    is not saved, so the re-estimated levels differ from the saved ones; the controlled
    comparison below holds W identical for old and new, so the *change* is attributable to
    the surrogate alone.)
    """
    v = np.asarray(out["dataset_moments"]).var(0)
    return np.diag(1.0 / (v + 1e-12))


def reestimate_params(out, surr, *, master_seed=DEFAULT_SEED, n_restarts=15, W=None):
    """Re-run ONLY the LM estimation (surrogate + LM, no solver, no simulation) against the
    SAVED target moments. Returns true/est params, across-fold estimates, and param R^2.

    Cheap by construction: it reuses the saved targets (``true_m``) and never re-collects or
    re-simulates. Use it to compare two surrogates on identical inputs (see ``compare_v2``).
    """
    from src.v3.estimation.estimate import estimate
    bounds = ParamBounds.table_a1()
    W = fixed_weight(out) if W is None else W
    tb, tm = np.asarray(out["true_beta"]), np.asarray(out["true_m"])
    eb, folds = [], []
    for d in range(tb.shape[0]):
        est = estimate(surr, tm[d], W, bounds, master_seed, n_restarts=n_restarts)
        eb.append(est["beta_hat"].numpy())
        folds.append(est["fold_betas"].numpy())
    eb, folds = np.asarray(eb), np.asarray(folds)
    return {"true_beta": tb, "est_beta": eb, "est_beta_folds": folds, "param_r2": r2(tb, eb)}


def compare_v2(out, master_seed=DEFAULT_SEED, old=(32, 200), new=(64, 800), n_restarts=15):
    """Controlled before/after of the V2 parameter recovery: same targets, same fixed W, only
    the surrogate (hidden, passes) differs. Returns ``{'old': out_like, 'new': out_like}`` ready
    for :func:`src.v3.validation.figures.plot_recovery_params`.
    """
    W = fixed_weight(out)
    res = {}
    for tag, (h, p) in (("old", old), ("new", new)):
        surr = _train_surrogate(out, master_seed, p, h)
        r = reestimate_params(out, surr, master_seed=master_seed, n_restarts=n_restarts, W=W)
        r["config"] = {"hidden": h, "passes": p}
        res[tag] = r
    return res


def run_all(run_dir, master_seed=DEFAULT_SEED, passes_grid=(200, 400, 800),
            hidden_grid=(32, 64), save=True):
    """Load a saved recovery, compute every diagnostic, optionally persist, return a dict.

    Writes (when ``save``) under ``<run_dir>/diagnostics/``: ``moment_diagnostics.csv``,
    ``param_diagnostics.csv``, ``surrogate_sweep.csv``, ``oracle.csv``, ``identification.csv``,
    and ``diagnostics.json`` (the sweep verdict + identification summary).
    """
    from src.v3.output.artifacts import load_recovery, save_summary

    out = load_recovery(run_dir)
    if out.get("dataset_beta") is None:
        raise ValueError(
            f"{run_dir} has no collected dataset (arrays/dataset.npz); the surrogate sweep, "
            "oracle, and identification need it. Re-run with save=True (it is saved by default).")
    mt, pt = moment_table(out), param_table(out)
    sweep, best_surr, _ = surrogate_sweep(out, master_seed, passes_grid, hidden_grid)
    verdict = sweep_verdict(sweep, baseline=(hidden_grid[0], passes_grid[0]))
    orc, ident = oracle_check(out, best_surr), identification(out, best_surr)
    result = {"run_dir": str(run_dir), "moment_table": mt, "param_table": pt,
              "surrogate_sweep": sweep, "sweep_verdict": verdict, "oracle": orc,
              "identification": ident}
    if save:
        import json
        import os
        from pathlib import Path
        d = Path(run_dir) / "diagnostics"
        d.mkdir(parents=True, exist_ok=True)
        ctx = {"run_dir": d, "save": True}
        save_summary(ctx, mt, "moment_diagnostics.csv")
        save_summary(ctx, pt, "param_diagnostics.csv")
        save_summary(ctx, sweep, "surrogate_sweep.csv")
        save_summary(ctx, orc, "oracle.csv")
        save_summary(ctx, ident["per_param_sensitivity"], "identification.csv")
        (d / "diagnostics.json").write_text(json.dumps(
            {"sweep_verdict": verdict,
             "identification": {k: v for k, v in ident.items() if k != "per_param_sensitivity"}},
            indent=2))
    return result
