# DF26 post-estimation diagnostics and refinement

## What this is

A recovery run gives you two scorecards: Figure V1 (can we recover the moments?) and
Figure V2 (can we recover the parameters?). When the scores are low, those figures do not
tell you **why**, or **what to do next**. This document explains a set of cheap checks that
answer both, and a refinement step that often improves the result without re-running the
expensive parts.

Everything here is "free": it reads a finished run and, at most, retrains the small
surrogate network on data you already collected. It never re-solves the model or
re-simulates the panel. The running example is the LARGE run from 2026-06-20 (A100, 2.94 h:
training 1528 s, collection 5560 s, surrogate 9 s, recovery 3491 s).

## The pipeline in one picture

To read the diagnostics you only need three ideas.

- **The moment map.** The model turns 8 parameters into 11 summary statistics ("moments").
  Call this true function `g: parameters -> moments`. It is smooth and deterministic, but
  each evaluation is expensive (solve the model, simulate a panel, compute moments).
- **The surrogate.** To estimate quickly, we fit a small neural network to copy `g`. We
  feed it many `(parameters, moments)` examples (the "collection") and it learns to predict
  the moments for any parameters, instantly. The surrogate is an approximation of `g`.
- **The estimator.** Given real (or, here, synthetic "true") moments, a search routine asks
  the surrogate: which parameters reproduce these moments? That answer is the estimate.

So the chain is: collect examples -> fit surrogate -> search for parameters. A weak final
result can come from any link, and the diagnostics below isolate which one.

## The five diagnostics

### 1. Which moments actually carry information?

A moment is only useful if it (a) changes when the parameters change, and (b) can be
predicted from the parameters. We check the second directly: the surrogate's out-of-sample
fit per moment (how well it predicts a moment on examples it did not train on).

LARGE, per-moment out-of-sample fit (R^2, 0 = useless, 1 = perfect):

| moment | surrogate R^2 | verdict |
|---|---|---|
| Mean inv rate | 0.62 | informative |
| SD inv rate | 0.78 | informative |
| Mean op income | 0.85 | informative |
| SD op income | 0.93 | informative |
| Autocorr income | 0.04 | **dead** |
| Mean debt | 0.78 | informative |
| SD debt | 0.41 | informative |
| Mean cash | 0.42 | informative |
| SD cash | 0.45 | informative |
| Cash~net debt | 0.48 | informative |
| Cash~income | -0.01 | **dead** |

Intuition: two moments (Autocorr income, Cash~income) cannot be predicted from the
parameters at all. They are regression-slope moments dominated by sampling noise in the
panel, not smooth functions of the parameters. They add noise, not signal.

Why this matters for the headline number: the reported "mean surrogate R^2 = 0.52" is an
average that the two dead moments drag down. The nine informative moments average about
0.64, and their median is higher still. The single mean number hides this; the per-moment
table is the honest view.

### 2. Is the surrogate under-trained, or is the data the limit?

If the surrogate fits poorly, there are two very different causes:

- **Under-trained surrogate (cheap to fix).** The network was too small or trained too
  briefly, so it stopped short of what the data supports. Fix: train it more, on the same
  data. Costs minutes.
- **Data-limited (expensive to fix).** The network already extracts everything in the
  collected examples. Fix: collect more examples. Costs hours.

We tell them apart by retraining the surrogate on the **same saved data** at a few settings
and watching the out-of-sample fit. `passes` is how long we train (SGD epochs over the
examples); `hidden` is the network width (its capacity).

LARGE, retrained on the same 2500 examples:

| setting | mean OOS R^2 | median OOS R^2 |
|---|---|---|
| baseline (hidden 32, passes 200) | 0.524 | 0.484 |
| stronger (hidden 64, passes 800) | 0.617 | **0.722** |

The median jumps from 0.48 to 0.72 on the same data. That is decisive: the LARGE run's
surrogate was **under-trained**. The information was already in the collected examples; the
network simply had not finished learning it. This is a cheap win.

(The mean rises less, from 0.52 to 0.62, because the two dead moments cannot improve no
matter how long we train. Training cannot create signal that is not there.)

### 3. Is the loss in the surrogate, or further downstream?

The "oracle" check asks a clean question: if we hand the surrogate the **true** parameters,
does it return the **true** moments? If yes, the surrogate is faithful and any remaining
error is downstream (in the search step or in identification). If no, the surrogate itself
is the bottleneck.

LARGE, oracle vs end-to-end (per moment):

| moment | oracle R^2 | end-to-end R^2 |
|---|---|---|
| Mean inv rate | 0.91 | 0.64 |
| SD inv rate | 0.82 | 0.41 |
| SD op income | 0.92 | 0.55 |
| Mean cash | 0.19 | 0.33 |
| Mean op income | 0.34 | 0.30 |

Two patterns appear. For the top three, the surrogate reproduces the truth well (0.82-0.92)
yet the end-to-end recovery is worse (0.41-0.64). The gap is downstream: even a near-perfect
surrogate would not close it, so better surrogate training helps only up to a point. For the
bottom two, oracle and end-to-end are both low, so those moments are genuinely hard to
predict (data-limited). The oracle separates "surrogate problem" from "everything else."

### 4. Which parameters are structurally weak?

Some parameters are hard to estimate no matter how good the surrogate, because the moments
barely respond to them. We measure each parameter's "sensitivity": how much the moments move
when that parameter moves (read off the surrogate's slope). Small sensitivity means the data
contains little information about that parameter.

LARGE sensitivity (larger = better identified):

| parameter | sensitivity | reading |
|---|---|---|
| cf | 5.24 | strong |
| theta | 2.84 | strong |
| sigma | 1.87 | good |
| delta | 1.83 | good |
| chi | 0.87 | moderate |
| gamma1 | 0.84 | moderate |
| rho | 0.64 | weak |
| gamma0 | 0.45 | **weak** |

The overall conditioning is moderate (condition number 77.6), and the least-identified
combination is roughly "rho versus gamma0". So gamma0 and rho are genuinely weakly
identified: more data or training will not fix them. This is a property of the model and the
moment set, not a bug.

This sensitivity reading also resolves a puzzle. In the saved recovery, theta has a low
score (R^2 = 0.05) yet here its sensitivity is high (2.84). High sensitivity plus low score
means theta is **estimable but noisily estimated**: its error is approximation noise from
the under-trained surrogate, not weak identification. So theta should improve once the
surrogate is trained properly (diagnostic 2). Contrast gamma0: low sensitivity and low score,
which is true weak identification that no amount of compute fixes.

### 5. Per-parameter accuracy and confidence

Finally, a plain table of each parameter's score, bias, error size, and confidence-interval
width from the saved run (these use the run's own weighting). It is the summary you report.

| parameter | R^2 | bias | RMSE | 95% CI half-width |
|---|---|---|---|---|
| theta | 0.05 | -0.011 | 0.085 | 0.109 |
| rho | 0.12 | 0.027 | 0.116 | 0.168 |
| sigma | 0.17 | 0.017 | 0.051 | 0.047 |
| delta | 0.16 | 0.016 | 0.051 | 0.064 |
| gamma1 | 0.21 | 0.110 | 0.350 | 0.500 |
| gamma0 | 0.01 | 0.035 | 0.092 | 0.135 |
| chi | 0.39 | 0.133 | 0.290 | 0.400 |
| cf | 0.15 | 0.024 | 0.065 | 0.074 |

## The refinement step: retrain the surrogate, re-estimate

The cheap fix is to retrain the surrogate on the data you already collected (stronger
settings) and re-run only the parameter search. No new collection, no new simulation.

### Why this is valid science, not a hack

This is the most important point, so it is worth being precise.

- **There is a real answer to learn.** The surrogate approximates a fixed, deterministic,
  smooth function `g` (parameters to moments). Training the surrogate better makes it a
  closer copy of `g`. It is not tuning toward an arbitrary target; it is converging toward a
  function that genuinely exists.
- **The improvement is measured on held-out data.** Out-of-sample R^2 scores each example
  with a copy of the network that never saw it (cross-validation). If retraining were just
  memorizing the examples (the classic "hack"), the held-out score would fall. Instead it
  rose, from a median of 0.48 to 0.72. Rising held-out accuracy is the textbook signature of
  fixing under-fitting, the opposite of overfitting.
- **A better copy of `g` gives a better estimate.** The search step trusts the surrogate as
  a stand-in for the model. The closer the surrogate is to `g`, the closer the recovered
  parameters are to the truth. This is a direct, principled chain, not a coincidence.
- **The method requires it.** The DF26 design assumes the surrogate is accurate (its own
  success target is a near-perfect Figure V1). Making the surrogate accurate is satisfying a
  precondition of the method, not gaming a score.
- **It is bounded, not magic.** Retraining cannot create signal for the two dead moments,
  and cannot fix the weakly identified parameters (gamma0, rho). It recovers only the part of
  the error that was approximation error. The honest test that we are not fooling ourselves:
  if pushing the network larger ever made the held-out score drop, that would be
  overfitting, and we would stop. Here it climbed, so the original setting was simply too
  small.

In short: the LARGE run under-trained its surrogate, so it left accuracy on the table that
the collected data already supported. Picking it up is correct, and the held-out scores
prove it.

### Evidence: Figure V2 before and after

We re-ran only the parameter search (same true moments, same fixed weighting, only the
surrogate changed) with the old and the stronger surrogate. Holding everything else fixed
isolates the surrogate's effect.

LARGE, parameter R^2 before (old surrogate: hidden 32, passes 200) and after (stronger:
hidden 64, passes 800):

| parameter | R^2 old | R^2 new | identification |
|---|---|---|---|
| theta | 0.10 | **0.23** | strong |
| delta | 0.19 | **0.39** | good |
| cf | 0.40 | **0.53** | strong |
| sigma | 0.30 | **0.34** | good |
| chi | 0.08 | 0.10 | moderate |
| gamma1 | 0.23 | 0.19 | moderate |
| rho | 0.05 | 0.04 | weak |
| gamma0 | 0.02 | 0.02 | weak |
| **mean** | **0.17** | **0.23** | |

Two things stand out, and both confirm the earlier diagnostics.

- **The improvement lands exactly where it should.** The parameters that gained the most
  (theta, delta, cf, sigma) are precisely the ones the identification check flagged as
  strongly identified. The weakly identified ones (rho, gamma0) did not move. A better
  surrogate helps where the data has information and cannot help where it does not. This
  internal consistency is strong evidence the story is right, not luck.
- **The gain is real but bounded.** Mean parameter R^2 rises from 0.17 to 0.23. That is
  smaller than the surrogate's own jump (median moment fit 0.48 to 0.72) because parameter
  recovery is also held back by weak identification and by the downstream gap the oracle
  exposed. The surrogate fix removes the approximation-error part of the loss, not the other
  parts. That is the honest, expected outcome.

(Sanity check: the controlled "old" mean R^2 of 0.17 matches the saved run's own mean of
0.16, even though this comparison uses a simpler fixed weighting. So the controlled setup is
a faithful stand-in, and the 0.17 to 0.23 jump is a genuine effect of the surrogate.)


Figures: `diagnostics/fig_v2_params_old.png` and `diagnostics/fig_v2_params_new.png`.

Note on Figure V1: the moment-fit figure plots moments re-simulated **at the new estimates**,
which needs the model solver. So a refreshed V1 is not free: regenerate it with a short
recovery re-run on a GPU using the stronger surrogate settings (it reuses the collected
dataset, so only the solve-and-simulate per draw is repeated). The V2 figure above is free
because the parameter search uses only the surrogate.

## What to do next: a simple decision guide

1. **Always do the cheap fix first.** Retrain the surrogate on the saved data (more passes,
   wider net) and re-estimate. For LARGE this lifts the typical moment fit from 0.48 to 0.72
   and should help the estimable-but-noisy parameters (theta, sigma, delta, cf). Cost:
   minutes.
2. **Stabilize or drop dead moments.** Autocorr income and Cash~income carry no signal here.
   Bigger panels (more firms, more periods) steady these regression-slope moments more
   efficiently than collecting more parameter draws. Or reconsider whether they belong in the
   moment set.
3. **Accept the structural limits.** gamma0 and rho are weakly identified by the model and
   the moments. No amount of compute fixes that; it is about the moment set, not scale.
4. **Only then scale up.** After the surrogate is properly trained and you understand the
   weak parameters, decide whether a longer run (more collected examples, finer grid) is
   worth it. Much of the LARGE gap is recoverable for free, so do not pay for a day-long run
   to rediscover an under-trained surrogate.

## How to run it

- Notebook: `docs/v3/04_diagnostics_colab.ipynb`. Point it at a finished run and run all
  cells. On a GPU the whole thing is about two minutes; on a CPU the surrogate retraining is
  slower (it runs in float64 on the CPU), so shrink the sweep grids.
- Library: `src/v3/validation/diagnostics.py`. `run_all(run_dir)` computes every diagnostic
  and writes them under `<run_dir>/diagnostics/` (per-moment and per-parameter tables, the
  surrogate sweep, the oracle, the identification summary, and a JSON verdict). `compare_v2`
  produces the before/after parameter figures.

## Scientific caveats

- Out-of-sample R^2 is the honest metric for the surrogate; training R^2 is not. Always read
  the held-out score.
- The before/after V2 uses a fixed inverse-variance weighting (the run's own per-draw
  weighting is not saved), so its levels differ slightly from the saved figure. The
  comparison is still valid because the weighting is identical for old and new, so the change
  is caused by the surrogate alone.
- "R^2" throughout is squared correlation, bounded in [0, 1]. Read bias and error size
  separately; a high R^2 can still hide a bias.
- These diagnostics explain and cheaply improve a run. They do not replace a properly scaled
  run for final, publishable numbers.
