# Development Guidance {#sec:appB-guidance}

#### How to use this document {#sec:how-to-use-this-document}

You are an AI coding agent auditing a scientific estimation codebase. The primary stack is TensorFlow (TF) and TensorFlow Probability (TFP), with NumPy and SciPy used alongside (for example simulated method of moments, value function iteration, and linear programming). Your job is to find **real** defects with a confirmed cause, not to flag every suspicious pattern. For each issue below:

1.  Run the **Diagnostics** in order. Each has a **Confirm** step (evidence the issue is present) and a **Rule out** step (evidence the symptom has a different cause or is a false alarm). Do not report an issue unless Confirm passes and Rule out does not explain it away.

2.  When you report, give the issue ID (e.g. `NUM-1`), the file and line, the evidence from the Confirm step, and the proposed fix.

3.  Prefer the linked **References** over memory. Library behavior changes across versions; verify against the installed versions (`pip show tensorflow tensorflow-probability tf-keras numpy scipy`).

**A few software terms used below**, in case they are unfamiliar:
- **Merge / pull request**: proposing a code change to be added to the shared codebase.
- **CI (continuous integration)**: a service that automatically runs your checks on every proposed change.
- **Quality gate**: an automatic rule that blocks a change from being merged unless the checks pass.
- **Runtime check (assertion)**: a check written inside the code that stops the run if a value is impossible (for example a probability above one, or a `NaN` where a finite number is required).
- **Compiling to a graph (tracing)**: TensorFlow speeds up a function by turning it, on the first call, into a fixed computation plan called a graph. Rebuilding that plan is called retracing.

**Scope and status.** This document is **advisory**: it guides review but does not block anything by itself. The blocking is done by the quality gate (the automatic check on every proposed change). Treat a finding here as something to fix before the gate runs, not as a replacement for the gate.

**Severity legend.** `[``BLOCKER``]` can produce silently wrong results or stop all runs. `[``HIGH``]` likely to cause wrong results or large slowdowns in common cases. `[``MED``]` situational or speed-leaning, but still able to affect a result.

#### Quick audit checklist {#sec:quick-audit-checklist}

Environment and dependencies
- \[ \] `ENV-1` Exact, compatible TF / TFP / Keras versions are pinned and importable.
- \[ \] `ENV-2` One floating-point precision is used end to end; float64 where the numerics are sensitive.

Reproducibility and randomness
- \[ \] `REPRO-1` TF determinism is configured; reproducibility claims match hardware and version reality.
- \[ \] `REPRO-2` One random-number scheme is used across NumPy, SciPy, and TF, with explicit seeded generators.
- \[ \] `REPRO-3` Simulated data used in training or estimation has a fixed, separate seed.

How TensorFlow runs and differentiates your code
- \[ \] `TF-1` A sped-up function is not being rebuilt (retraced) on every call.
- \[ \] `TF-2` No ordinary Python statements (print, list append, counters) are relied on inside a sped-up function.
- \[ \] `TF-3` No gradients silently come back missing; the path from parameter to loss stays inside TensorFlow.

Numerical stability
- \[ \] `NUM-1` Cholesky and matrix factorizations cannot fail silently into `NaN`.
- \[ \] `NUM-2` Likelihoods and products of probabilities are computed in log space.

Optimization and solvers
- \[ \] `OPT-1` When a SciPy/NumPy optimizer wraps a TF objective, gradients and precision are handled.
- \[ \] `OPT-2` Derivative-free optimizers are not trusted on a single "success" flag.
- \[ \] `OPT-3` Value function iteration meets the contraction conditions and a real convergence test.
- \[ \] `OPT-4` Optimizer choice and saved optimizer state are correct for the hardware.

Estimation: moments and Bayesian
- \[ \] `EST-1` SMM uses common random numbers (the random draws are fixed across optimizer iterations).
- \[ \] `EST-2` Method-of-moments Jacobian and weighting matrix are stable and well conditioned.
- \[ \] `EST-3` NUTS cost is understood; an expensive log-density does not run NUTS by default.
- \[ \] `EST-4` Step-size adaptation is wired through a packaged sampler, not bolted onto a bare NUTS kernel.
- \[ \] `EST-5` Sampler output is checked against a known answer and convergence diagnostics.

Neural network training and saving
- \[ \] `NN-1` No information leaks from test to train; data prep is fit on the training part only; time order respected.
- \[ \] `NN-2` A reloaded model reproduces the predictions of the model in memory.

Hardware acceleration
- \[ \] `HW-1` GPU or Apple Metal results are checked against the CPU result.

Data inputs
- \[ \] `DATA-1` Loaded datasets are checked for type, missing values, alignment, and units.

### Environment and dependencies {#sec:environment-and-dependencies}

#### ENV-1 --- TF / TFP / Keras version incompatibility (the Keras 3 break) `[``BLOCKER``]` {#sec:env-1-tf-tfp-keras-version-incompatibility-the-keras-3-break-blocker}

**Issue.** TFP fails to import, or behaves inconsistently, because the installed TensorFlow, TFP, and Keras versions are not a compatible set.

**Cause.** TFP is tied to specific TF versions. From TF 2.16 onward, `tf.keras` means Keras 3, and TFP does not work with Keras 3; it needs Keras 2, which is shipped separately as the `tf-keras` package and imported as `tf_keras`.

**Impact.** A clean install raises an `AttributeError` at `import tensorflow_probability` (often mentioning `__internal__` or a `keras` module), so nothing runs. A reviewer who cannot install the code cannot evaluate it.

**Diagnostics.**
- Confirm: run `python -c "import tensorflow, tensorflow_probability"`. If it raises an `AttributeError` mentioning `keras` or `__internal__`, and `pip list` shows TF 2.16 or later with Keras 3.x present and no `tf-keras`, this is the cause.
- Rule out: if the import succeeds, this is not the issue. An `ImportError` naming an unrelated package is a different dependency problem.

**Known fix.** Pin one tested set of versions (for example TF 2.16.1 with the matching TFP release and `tf-keras`). Either install `tensorflow-probability``[``tf``]`, or set the environment variable `TF_USE_LEGACY_KERAS=1` (with `tf-keras` installed), or `import tf_keras as keras`. Record the tested set in the README.

**References.**
- <https://github.com/tensorflow/probability/releases>
- <https://keras.io/getting_started/>
- <https://github.com/tensorflow/probability/issues/1774>
- <https://github.com/tensorflow/probability/issues/1795>

#### ENV-2 --- Mixed single and double precision (float32 vs float64) `[``HIGH``]` {#sec:env-2-mixed-single-and-double-precision-float32-vs-float64-high}

**Issue.** A precision-mismatch error stops the run, or `NaN` values and divergences appear that disappear at higher precision.

**Cause.** TensorFlow defaults to single precision (float32); NumPy defaults to double precision (float64). TensorFlow requires all inputs to one operation to share the same precision. Sensitive numerics (covariance factorization, HMC step-size handling, log-likelihoods) are fragile in single precision.

**Impact.** Either a hard precision-mismatch error, or a silent loss of accuracy and `NaN` values that corrupt the estimation.

**Diagnostics.**
- Confirm: look for errors like "expected to be a double tensor but is a float tensor". Check the precision of step sizes, states, and log-density inputs. Cast the sensitive path to float64; if the error or the `NaN` values disappear, a precision mismatch was the cause.
- Rule out: if `NaN` values persist in float64, the instability is in the model or algorithm (see `NUM-1`, `NUM-2`), not precision. No error and stable results means do not flag.

**Known fix.** Pick one precision for the numerical core (float64 for filtering, MCMC, and likelihoods) and keep it throughout. Cast constants explicitly. Pass step sizes and acceptance targets as float64.

**References.**
- <https://www.tensorflow.org/api_docs/python/tf/cast>
- <https://www.tensorflow.org/guide/tensor>

### Reproducibility and randomness {#sec:reproducibility-and-randomness}

#### REPRO-1 --- Nondeterminism and false reproducibility claims (TensorFlow) `[``HIGH``]` {#sec:repro-1-nondeterminism-and-false-reproducibility-claims-tensorflow-high}

**Issue.** Re-running the same code gives different numbers, or the code claims exact reproducibility that does not hold.

**Cause.** Unset random seeds, GPU operations that add floating-point numbers in an unpredictable order across threads, and parallel data pipelines. Reproducibility is also not guaranteed across different TF versions or different hardware.

**Impact.** Results cannot be reproduced or audited, and a stated reproducibility guarantee is misleading.

**Diagnostics.**
- Confirm: run the same script twice on the same machine. If outputs differ, set seeds with `tf.keras.utils.set_random_seed(s)` and call `tf.config.experimental.enable_op_determinism()`. If outputs then match run to run, nondeterminism was the cause. A TF random operation that raises `RuntimeError` after you enable determinism means a seed is missing.
- Rule out: differences across different hardware or different TF versions are expected, not a bug. NumPy or SciPy randomness varying is `REPRO-2`, not this.

**Known fix.** Call `set_random_seed` and `enable_op_determinism()` before any random operation. Pin the hardware and software. Avoid `tf.compat.v1.Session` and `ParameterServerStrategy`. State reproducibility as holding only on the same hardware and same versions.

**References.**
- <https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism>
- <https://www.tensorflow.org/guide/random_numbers>
- <https://github.com/tensorflow/community/blob/master/rfcs/20210119-determinism.md>
- <https://github.com/NVIDIA/framework-reproducibility>

#### REPRO-2 --- Random-number generators not coordinated across NumPy, SciPy, and TF `[``HIGH``]` {#sec:repro-2-random-number-generators-not-coordinated-across-numpy-scipy-and-tf-high}

**Issue.** Results are not reproducible, or change when the random-number call is swapped for another, because several generators are uncoordinated.

**Cause.** SciPy draws from NumPy's generator, so NumPy, TF, and any TFP sampler each carry their own state. Within NumPy, the old global `np.random.seed` with `RandomState` and the newer `np.random.default_rng` produce different number streams, and the newer one is not guaranteed to match the old values or to stay identical across NumPy versions. Copying one seeded generator into parallel workers gives every worker the same stream.

**Impact.** Silent irreproducibility, and results that shift for reasons unrelated to the method when the code is refactored or a library is upgraded.

**Diagnostics.**
- Confirm: results change between runs although a TF seed is set; or switching from `np.random.seed`/`RandomState` to `default_rng` changes outputs. Search for the global `np.random.seed` and for mixed random-number calls. Switching to explicit seeded generators plus a TF seed makes runs reproducible, which confirms the cause.
- Rule out: if only TF operations are random (no NumPy or SciPy randomness), this is `REPRO-1`. Stream differences across NumPy versions for the newer generator are expected, not a bug.

**Known fix.** Choose one scheme. Pass an explicit seeded `np.random.default_rng(seed)` into functions instead of relying on the global state; set the TF seed separately; for parallel work, create independent streams (`SeedSequence.spawn`). Log every seed.

**References.**
- <https://numpy.org/doc/stable/reference/random/index.html>
- <https://blog.scientific-python.org/numpy/numpy-rng/>
- <https://www.tensorflow.org/guide/random_numbers>

#### REPRO-3 --- Simulated data not reproducible during training or estimation `[``HIGH``]` {#sec:repro-3-simulated-data-not-reproducible-during-training-or-estimation-high}

**Issue.** When training or estimation uses simulated data, each run uses a different dataset, so results vary for reasons unrelated to the method.

**Cause.** The data-simulation step has no fixed seed, or reuses a generator whose state has already advanced, so the simulated sample changes between runs.

**Impact.** Fitted parameters and trained models are not reproducible, and run-to-run variation is mistaken for instability in the estimator.

**Diagnostics.**
- Confirm: re-running training or estimation gives different fitted parameters, and the simulation step has no explicit seed or shares an advancing generator. Fixing the simulation seed separately makes the dataset and the results stable, confirming the cause.
- Rule out: variation that remains after the simulated data is fixed comes from `REPRO-1` (TF) or optimizer randomness, a different cause.

**Known fix.** Give the data-simulation generator its own explicit seed, separate from the training and optimizer generators. Treat the simulated dataset as a fixed, versioned input and log its seed.

**References.**
- <https://numpy.org/doc/stable/reference/random/index.html>
- <https://www.tensorflow.org/guide/random_numbers>

### How TensorFlow runs and differentiates your code {#sec:how-tensorflow-runs-and-differentiates-your-code}

#### TF-1 --- A sped-up function is rebuilt on every call (retracing) `[``HIGH``]` {#sec:tf-1-a-sped-up-function-is-rebuilt-on-every-call-retracing-high}

**Issue.** A function marked with `@tf.function` is recompiled far more often than expected, making the pipeline slow.

**Cause.** TensorFlow compiles such a function, on its first call, into a fixed computation plan (a graph). Passing plain Python numbers, or arrays whose shape keeps changing, forces it to rebuild that plan each time (this rebuild is called retracing). Looping over Python or NumPy data inside the function also bakes a separate copy of the body into the plan for each iteration.

**Impact.** A huge plan and severe slowdowns, easily mistaken for an algorithmic cost such as sampling time.

**Diagnostics.**
- Confirm: watch for the TF warning that tracing is expensive and repeated. Measure with the function's `experimental_get_tracing_count()`, or put a plain Python `print` inside it (it prints once each time the plan is rebuilt). If the count grows with calls or iterations, the function is being rebuilt; the usual trigger is plain Python-number arguments or changing array shapes.
- Rule out: if the count stays at 1 (or equals a small number of distinct input shapes), rebuilding is not the cause; the slowness is elsewhere (see `EST-3`).

**Known fix.** Pass tensors, not plain Python numbers. Set `reduce_retracing=True` or give an explicit `input_signature`. Do not create `tf.Variable` objects inside the function. Use TensorFlow's own loops and branches (`tf.while_loop`, `tf.cond`, `tf.TensorArray`) instead of Python loops and `if` statements over tensor values.

**References.**
- <https://www.tensorflow.org/guide/function>

#### TF-2 --- Ordinary Python statements inside a sped-up function `[``MED``]` {#sec:tf-2-ordinary-python-statements-inside-a-sped-up-function-med}

**Issue.** Logging, counters, or list building inside a `@tf.function` do not behave the way the code reads.

**Cause.** Once the function is compiled into a graph, ordinary Python statements (print, appending to a list, increasing a Python counter) run only during that first compile, not on later calls.

**Impact.** Misleading diagnostics, lost logs, counters that never advance, and debugging based on values captured once at compile time.

**Diagnostics.**
- Confirm: a Python `print`, list append, or counter inside the function updates only once across many calls. Replace it with `tf.print` or `tf.Variable.assign`; if the behavior corrects, the ordinary Python statement was the cause.
- Rule out: if the function already uses `tf.print` or `tf.Variable.assign` and still misbehaves, the cause is different.

**Known fix.** Use TensorFlow's own versions for anything that must happen on every call: `tf.print`, `tf.summary`, `tf.Variable.assign`, and `tf.TensorArray` to accumulate. Keep ordinary Python statements out of compiled functions.

**References.**
- <https://www.tensorflow.org/guide/function>

#### TF-3 --- Gradients silently come back missing `[``BLOCKER``]` {#sec:tf-3-gradients-silently-come-back-missing-blocker}

**Issue.** The gradient of the loss with respect to some parameters comes back as missing (`None`), and training silently does nothing for those parameters.

**Cause.** TensorFlow computes gradients by recording the operations it sees (with `tf.GradientTape`). If the path from a parameter to the loss leaves TensorFlow at any point (for example a conversion to a NumPy array, a non-TF operation, an integer dtype, or a plain tensor that was never registered for tracking), the recording is broken and the gradient for that parameter is returned as `None`.

**Impact.** The optimizer skips any parameter whose gradient is `None`. Training appears to run while some parameters never update, giving a quietly wrong model.

**Diagnostics.**
- Confirm: after computing gradients, check that none are `None`. If some are, inspect the path for a `.numpy()` call, a `np.` operation, or a `model.predict` between the parameter and the loss. Check that the parameter is a tracked variable (or is passed to `tape.watch`) and that precisions are floating point. Removing the NumPy conversion or adding `tape.watch` brings back a real gradient.
- Rule out: a gradient that is a real zero (not `None`) is a flat-region or vanishing-gradient situation, a different problem. `None` specifically means the path was broken or the parameter was not tracked.

**Known fix.** Keep the whole forward path inside TensorFlow with no conversions to NumPy. Register non-variable inputs with `tape.watch`. Use floating-point precision. Add a permanent check that no returned gradient is `None`.

**References.**
- <https://www.tensorflow.org/guide/autodiff>

### Numerical stability {#sec:numerical-stability}

#### NUM-1 --- Cholesky silently returns `NaN` on a non-positive-definite matrix `[``BLOCKER``]` {#sec:num-1-cholesky-silently-returns-nan-on-a-non-positive-definite-matrix-blocker}

**Issue.** A Cholesky factorization (covariance update, mass matrix, Gaussian log-density) returns `NaN` or zeros instead of raising an error, and the corruption spreads.

**Cause.** Unlike NumPy, which raises an error on a matrix that is not positive definite, `tf.linalg.cholesky` fills the output with `NaN` for a real matrix (and zeros for a complex one) without raising. Rounding in single precision can push a covariance just outside positive-definiteness. The gradient of the Cholesky also fails for nearly singular matrices, which matters when sampling through it.

**Impact.** `NaN` values flow into the sampler or loss as silent garbage rather than a clean error, and gradient-based samplers can crash near singular matrices.

**Diagnostics.**
- Confirm: detect `NaN` (real) or all-zero (complex) output from a Cholesky call with no error raised. Run `numpy.linalg.cholesky` on the same matrix; if NumPy raises an error, the matrix is not positive definite, which confirms the cause. Check symmetry and the smallest eigenvalue.
- Rule out: if `numpy.linalg.cholesky` succeeds but TensorFlow still returns `NaN` or crashes, suspect a backend or GPU-kernel problem, not a non-positive-definite matrix. If the `NaN` starts upstream, the Cholesky is a symptom, not the cause.

**Known fix.** Symmetrize the matrix (average it with its transpose) and add a small value to the diagonal (jitter). Use `tfp.experimental.linalg.simple_robustified_cholesky` and then check its output for `NaN` as its own documentation instructs. For a Kalman-filter covariance, use the Joseph-form update or a square-root filter. Prefer float64 (`ENV-2`). Guard against nearly singular inputs when differentiating through the Cholesky.

**References.**
- <https://github.com/tensorflow/tensorflow/issues/61916>
- <https://github.com/tensorflow/tensorflow/issues/62451>
- <https://github.com/tensorflow/probability/issues/195>
- <https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/linalg/simple_robustified_cholesky>

#### NUM-2 --- Products of probabilities underflow unless computed in log space `[``HIGH``]` {#sec:num-2-products-of-probabilities-underflow-unless-computed-in-log-space-high}

**Issue.** A hand-built likelihood, filter, or particle weight collapses to zero, after which logs and normalizations produce `NaN` or infinity.

**Cause.** Multiplying many small probabilities (a likelihood over many observations, a product of weights, a filter update) underflows to zero in floating point. Exponentiating very large or very small numbers also overflows or underflows.

**Impact.** `NaN` or infinite log-likelihoods that corrupt the optimizer or sampler, and silently wrong normalizing constants.

**Diagnostics.**
- Confirm: the log-likelihood or the weights become `0`, `NaN`, or `-inf` as the sample size or dimension grows, and the code multiplies probabilities or exponentiates before summing. Rewriting in log space with a stable log-sum-exp removes the `NaN`, which confirms underflow.
- Rule out: if the values are already finite and stable in log space, this is not the issue. A `NaN` from a non-positive-definite covariance is `NUM-1`, not underflow.

**Known fix.** Keep likelihoods and weights in log space throughout. Use `scipy.special.logsumexp` (or `numpy.logaddexp` for two terms, or the TensorFlow equivalent) for any sum of exponentials, and exponentiate only at the very last step.

**References.**
- <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html>
- <https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html>

### Optimization and solvers {#sec:optimization-and-solvers}

#### OPT-1 --- A SciPy or NumPy optimizer wrapping a TensorFlow objective `[``HIGH``]` {#sec:opt-1-a-scipy-or-numpy-optimizer-wrapping-a-tensorflow-objective-high}

**Issue.** A SciPy or NumPy optimizer drives a TensorFlow objective, and the handoff silently drops gradients or mismatches precision.

**Cause.** SciPy optimizers work on double-precision NumPy arrays and call a plain Python objective many times. TensorFlow gradients do not pass back through SciPy, so a SciPy run cannot use TFP's automatic gradients unless you supply a gradient yourself; and SciPy's float64 meets TensorFlow's float32 default at every call.

**Impact.** Either a gradient is missing (so the optimizer falls back to noisy finite differences, see `OPT-2`), or precision conversions degrade accuracy, or the cost of crossing between the two libraries on every call dominates the runtime.

**Diagnostics.**
- Confirm: a SciPy optimizer wraps a function that runs TensorFlow operations. Check whether a gradient is supplied and whether the objective returns a plain Python or NumPy float64. Precision errors or float32-to-float64 casts at the handoff, or slow per-call time, confirm the cost.
- Rule out: if a `tfp.optimizer` is used (staying inside TensorFlow), this does not apply. If the slowness is rebuilding the function, that is `TF-1`.

**Known fix.** Decide which library owns the optimization loop. Use a `tfp.optimizer` (for example L-BFGS) to stay inside TensorFlow with gradients, or use SciPy and convert cleanly to double-precision NumPy at the handoff, either supplying a gradient or choosing a derivative-free method on purpose.

**References.**
- <https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer>
- <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>

#### OPT-2 --- Derivative-free optimizers can report success at a non-optimum `[``HIGH``]` {#sec:opt-2-derivative-free-optimizers-can-report-success-at-a-non-optimum-high}

**Issue.** Powell or Nelder-Mead returns "success" at a point that is not actually a minimum.

**Cause.** These methods use only function values and no gradients, so they can stop at a point that is not a true optimum, especially on a noisy or flat objective. Gradient-based methods given no gradient estimate it by finite differences, which is meaningless on a noisy objective.

**Impact.** A confidently reported solution that is wrong, and downstream estimates and standard errors built on it.

**Diagnostics.**
- Confirm: the optimizer reports success, but an independent check fails (the gradient is not near zero where one exists, or restarts and a global method find a lower value). Re-run from several starting points; if the results scatter or a global method beats the reported point, the success flag was not a true optimum.
- Rule out: if multiple starts and a global method all agree on the same point and value, the solution is trustworthy. A genuinely flat objective is an identification problem, not optimizer failure.

**Known fix.** Never trust a single run. Use several starting points, cross-check a global method (`dual_annealing`, `basinhopping`) against a local one, and inspect the objective surface. For SMM, also apply `EST-1`.

**References.**
- <https://docs.scipy.org/doc/scipy/tutorial/optimize.html>
- <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>

#### OPT-3 --- Value function iteration does not converge or converges to the wrong fixed point `[``HIGH``]` {#sec:opt-3-value-function-iteration-does-not-converge-or-converges-to-the-wrong-fixed-point-high}

**Issue.** Value function iteration fails to settle, oscillates, or depends on the starting guess.

**Cause.** Convergence to a single correct fixed point holds only when the Bellman operator is actually a contraction (discount factor below one, bounded returns). A continuous state must be stored on a grid and interpolated, and an approximate or fitted iteration can lose the guarantee and diverge or oscillate.

**Impact.** A value or policy function that is silently wrong, which then feeds every quantity computed from it.

**Diagnostics.**
- Confirm: the largest change between successive value functions does not fall below the tolerance, or it oscillates, or the solution depends on the starting guess. Check that the discount factor is below one and returns are bounded. Vary the grid resolution and the starting guess; a true contraction converges steadily to the same fixed point regardless of the start.
- Rule out: slow but steady convergence that does reach the tolerance is fine. Differences below the chosen tolerance are not non-convergence.

**Known fix.** Confirm the contraction conditions. Require the largest change between iterations to fall below an explicit tolerance before stopping. Test the grid resolution and the interpolation choice, and confirm the solution does not depend on the starting guess.

**References.**
- <https://python.quantecon.org/cake_eating_numerical.html>
- <https://python.quantecon.org/mccall_fitted_vfi.html>

#### OPT-4 --- Optimizer choice and saved optimizer state `[``MED``]` {#sec:opt-4-optimizer-choice-and-saved-optimizer-state-med}

**Issue.** The Keras optimizer is slow on Apple Silicon, or a resumed run does not actually restore the optimizer's state.

**Cause.** The newer Keras optimizers have been reported to run slowly on M1 and M2 hardware, where the legacy optimizer is recommended. The way the optimizer's state is stored changed across Keras versions, so a saved checkpoint can fail to restore it cleanly.

**Impact.** Wasted time on Apple Silicon, or a training resume that silently starts the optimizer from scratch (loss jumps), corrupting a continued run.

**Diagnostics.**
- Confirm: the optimizer is slow specifically on M1 or M2 while plain math is not, or a restored checkpoint shows a loss jump consistent with a reset optimizer. Check which optimizer is used and whether its state was actually restored.
- Rule out: slowness on all hardware is not the Apple-Silicon issue. A loss change explained by a learning-rate schedule is not a restore failure.

**Known fix.** On Apple Silicon, use the legacy optimizer if it benchmarks faster. After loading a checkpoint, check that the optimizer's state was restored before continuing. Pin versions (`ENV-1`) and verify guidance against the installed version.

**References.**
- <https://keras.io/api/optimizers/>
- <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>

### Estimation: moments and Bayesian {#sec:estimation-moments-and-bayesian}

#### EST-1 --- SMM without common random numbers `[``BLOCKER``]` {#sec:est-1-smm-without-common-random-numbers-blocker}

**Issue.** The simulated method of moments objective jitters between nearby parameter guesses, so the optimizer chases simulation noise instead of the parameter direction.

**Cause.** Fresh random draws are taken on every objective evaluation, so the simulated moments wobble even when the parameter barely changes. The objective is then not even zero at the true parameter.

**Impact.** Non-convergence, estimates that depend on the seed, and finite-difference gradients that are meaningless.

**Diagnostics.**
- Confirm: the objective is not near zero at known true parameters when it should be, or it jitters between close guesses, and the simulation redraws shocks each call. Hold the draws fixed across evaluations; if the objective becomes smooth and near zero at the truth, missing common random numbers was the cause.
- Rule out: residual noise from a finite simulation sample that is the same across evaluations is expected, not this bug. A biased objective with fixed draws points to a model or moment problem instead.

**Known fix.** Draw the underlying shocks once and hold them fixed across all parameter evaluations (common random numbers). Redraw only to check simulation error. Coordinate the simulation seed per `REPRO-2`.

**References.**
- <https://scholar.harvard.edu/files/jalali/files/msm_book_chapter.pdf>
- <https://ekw-lectures.readthedocs.io/en/latest/method-of-simulated-moments/notebook.html>
- <https://optimagic.readthedocs.io/en/latest/estimagic/tutorials/msm_overview.html>

#### EST-2 --- Method-of-moments Jacobian and weighting matrix `[``MED``]` {#sec:est-2-method-of-moments-jacobian-and-weighting-matrix-med}

**Issue.** Estimates or standard errors are unstable because the moment Jacobian is noisy or the weighting matrix is ill conditioned.

**Cause.** A finite-difference Jacobian of the moments is noisy, especially on a simulated objective, and a nearly singular or wrongly estimated weighting matrix distorts both the point estimates and the inference.

**Impact.** Unreliable estimates and standard errors that change with implementation details.

**Diagnostics.**
- Confirm: estimates or standard errors are sensitive to the finite-difference step size, or the weighting matrix has a very high condition number. Recompute the Jacobian with a different step and check `numpy.linalg.cond(W)`.
- Rule out: stability across step sizes and a well-conditioned weighting matrix means this is not the issue.

**Known fix.** Use a stable finite-difference step (or an analytic Jacobian where available), check the condition number of the weighting matrix, and check the two-step weighting on a case with a known answer.

**References.**
- <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html>
- <https://optimagic.readthedocs.io/en/latest/estimagic/tutorials/msm_overview.html>

#### EST-3 --- NUTS is slow, especially through an expensive log-density `[``HIGH``]` {#sec:est-3-nuts-is-slow-especially-through-an-expensive-log-density-high}

**Issue.** Sampling with the No-U-Turn Sampler (NUTS) dominates the runtime, sometimes to the point of being impractical.

**Cause.** NUTS builds a variable-length path by doubling, so each draw runs an unpredictable, often large, number of leapfrog steps, and each step needs one full gradient of the log-density. When the log-density runs through a neural-network surrogate, every leapfrog step is a full backprop. NUTS runs about 2 to 5 times slower per sample than fixed-step HMC.

**Impact.** Runtimes of many hours, and a tendency to blame the wrong thing or to give up on Bayesian inference.

**Diagnostics.**
- Confirm: profiling shows the time is in sampling. Inspect the sampler results for path depth or leapfrog counts; high counts with an expensive (surrogate) log-density confirm the cause. Compare time per sample for NUTS against fixed-step HMC; roughly 2 to 5 times slower matches the known behavior.
- Rule out: if the cost per step is constant and unrelated to gradient evaluations, suspect rebuilding the function (`TF-1`) or a large data copy each step. Low path depth with still-slow steps points elsewhere.

**Known fix.** Use `windowed_adaptive_nuts` rather than a hand-built kernel. For an expensive or surrogate log-density, use a gradient-free sampler (Random-Walk Metropolis or Robust Adaptive Metropolis) and keep NUTS for the cheap closed-form path. Lower `max_tree_depth`. Compile the log-density with `tf.function(autograph=False)` and run the chains in parallel. Consider tuned fixed-step HMC.

**References.**
- <https://github.com/tensorflow/probability/issues/728>
- <https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler>
- <https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/windowed_adaptive_nuts>

#### EST-4 --- NUTS step-size adaptation wired incorrectly `[``HIGH``]` {#sec:est-4-nuts-step-size-adaptation-wired-incorrectly-high}

**Issue.** Step-size adaptation behaves wrongly with NUTS: the acceptance rate sits near 1.0 instead of the target, or the kernel raises an attribute error.

**Cause.** Adaptation that works for HMC does not carry over cleanly to a bare NUTS kernel, because the NUTS results have a different structure. Dual-averaging step-size adaptation can badly underestimate the step size with NUTS, and some adapters expect an `accepted_results` field that NUTS results do not provide.

**Impact.** Poor mixing, biased posteriors, or an outright error, often mistaken for a modeling problem.

**Diagnostics.**
- Confirm: the realized acceptance rate is pinned near 1.0 while the target was about 0.75, or the step size is implausibly small, or the run raises `'NUTSKernelResults' object has no attribute 'accepted_results'`. A bare NUTS kernel with `DualAveragingStepSizeAdaptation` matches the known failure.
- Rule out: if the same adapter behaves correctly with an HMC kernel on the same model, that confirms a NUTS-specific wiring problem. If acceptance is reasonable and the effective sample size is healthy, do not flag.

**Known fix.** Use the packaged `windowed_adaptive_nuts`, which wires warmup, step size, and the mass matrix together correctly. If wiring it by hand, pass the correct `log_accept_prob_getter_fn` for the NUTS results.

**References.**
- <https://github.com/tensorflow/probability/issues/983>
- <https://github.com/tensorflow/probability/issues/549>
- <https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/DualAveragingStepSizeAdaptation>

#### EST-5 --- Sampler output not checked for correctness or convergence `[``BLOCKER``]` {#sec:est-5-sampler-output-not-checked-for-correctness-or-convergence-blocker}

**Issue.** Posterior estimates are reported without checking that the chains converged and that the sampler is unbiased.

**Cause.** No check against a known answer, and no convergence diagnostics. NUTS in TFP has had a real bias bug in its history, so trust has to be earned for each model.

**Impact.** Confident but wrong posteriors, with no signal that anything is off.

**Diagnostics.**
- Confirm: compare posterior means and variances against a closed-form posterior (a known answer) or against HMC on the same model. Compute the R-hat statistic (`potential_scale_reduction`) and the effective sample size; an R-hat above about 1.01, a low effective sample size, or reported divergences indicate a problem.
- Rule out: a gap smaller than the Monte Carlo standard error is not bias; divide the error by the Monte Carlo standard error before judging. A high R-hat from too few iterations is non-convergence, not a sampler defect.

**Known fix.** Check against a closed-form posterior where one exists. Run multiple chains and report R-hat and the effective sample size. Treat divergences as a geometry problem (adjust the mass matrix and step size, or reparameterize).

**References.**
- <https://github.com/tensorflow/probability/issues/542>
- <https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/potential_scale_reduction>
- <https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/effective_sample_size>

### Neural network training and saving {#sec:neural-network-training-and-saving}

#### NN-1 --- Information leaks from the test set into training `[``BLOCKER``]` {#sec:nn-1-information-leaks-from-the-test-set-into-training-blocker}

**Issue.** Information from the test set or from the future reaches the model during training, so measured performance is too good and does not hold out of sample.

**Cause.** A data-preparation step (scaling, feature selection, filling missing values) is fit on the whole dataset before the train/test split, or a time series is split without keeping time order, or grouped observations are split across train and test. This is the most common reproducibility failure in machine-learning-based science.

**Impact.** Over-optimistic numbers that collapse out of sample, and conclusions that do not hold.

**Diagnostics.**
- Confirm: test performance is implausibly high or fails to reproduce; a data-preparation step is fit before the split; or a time-based split looks ahead. Move all fitting inside the training part only and re-evaluate. A large drop means the leak was inflating the metric.
- Rule out: a modest, stable gap between train and test, with all preparation fit on the training part only, is normal, not a leak.

**Known fix.** Split first. Fit every preparation step on the training part only and then apply it to the test part. Keep time order for time series; split by group for grouped data. Check the loading step per `DATA-1`.

**References.**
- <https://arxiv.org/abs/2207.07048>
- <https://doi.org/10.1016/j.patter.2023.100804>

#### NN-2 --- Saving and reloading a trained model changes its behavior `[``HIGH``]` {#sec:nn-2-saving-and-reloading-a-trained-model-changes-its-behavior-high}

**Issue.** A model saved to disk and loaded back does not reproduce the model in memory, or fails to load.

**Cause.** Custom layers, losses, or activations need a `get_config` method and registration, or loading raises an unknown-object error or quietly loses behavior. A model loaded from disk may also skip the automatic precision conversion (for example double to single) that an in-memory model does, so the inference inputs must match the precision used in training. The Keras 2 versus Keras 3 split (`ENV-1`) applies to saved files too.

**Impact.** Predictions from the saved model differ from training, silently corrupting any estimation that uses the saved (surrogate) model.

**Diagnostics.**
- Confirm: a reloaded model's predictions differ from the in-memory model by more than rounding, or loading raises an unknown-object error, or the inference precision differs from training. Compare predictions before and after saving and reloading.
- Rule out: tiny rounding-level differences are fine. A load error naming a missing custom object is a registration problem, which confirms saving and loading as the cause and is fixable.

**Known fix.** Register custom objects (`get_config` plus `register_keras_serializable`), pass `custom_objects` on load, check that predictions match (close to equal) after saving and reloading, and set the inference precision to match training.

**References.**
- <https://keras.io/api/models/model_saving_apis/model_saving_and_loading/>
- <https://www.tensorflow.org/tutorials/keras/save_and_load>
- <https://www.tensorflow.org/decision_forests/known_issues>

### Hardware acceleration {#sec:hardware-acceleration}

#### HW-1 --- GPU or Apple Metal results differ from the CPU `[``HIGH``]` {#sec:hw-1-gpu-or-apple-metal-results-differ-from-the-cpu-high}

**Issue.** Results differ between CPU and GPU, or are wrong only on the accelerator, or an operation has no GPU version.

**Cause.** Accelerator backends can differ from the CPU. The Apple `tensorflow-metal` plugin has community-reported cases of wrong results and of CPU-versus-GPU mismatch, and some operations (for example certain random-number operations) have lacked a GPU version, which breaks seeding on that backend. Models made of many tiny operations (state-space filters) are also a poor fit for GPUs.

**Impact.** Silently wrong numbers on the accelerator, or a hard "no registered kernel" error, or no speedup despite using a GPU.

**Diagnostics.**
- Confirm: run the same code with the accelerator hidden (`tf.config.set_visible_devices(``[``]``, 'GPU')`, or the `CUDA_VISIBLE_DEVICES=""` setting) to force the CPU. If the CPU result matches a trusted reference while the GPU or Metal result does not, beyond rounding, that is a backend correctness problem. A "no registered kernel" error names the failing operation directly.
- Rule out: small differences within rounding tolerance are normal, not a bug. If CPU and accelerator agree, do not flag.

**Known fix.** Treat the CPU result as the reference and check any accelerator result against it before trusting it. Pin the plugin version. Avoid or report operations that misbehave. When you need exact reproducibility, run on the CPU (see `REPRO-1`).

**References.**
- <https://developer.apple.com/metal/tensorflow-plugin/>

### Data inputs {#sec:data-inputs}

#### DATA-1 --- Loading a real dataset introduces silent corruption `[``HIGH``]` {#sec:data-1-loading-a-real-dataset-introduces-silent-corruption-high}

**Issue.** Reading a real file corrupts the data through mis-typed columns, mishandled missing values, misaligned joins, or inconsistent units, without raising an error.

**Cause.** Column types guessed wrongly and silently changed, missing values read as text or as unexpected `NaN` that then spread, index or date misalignment when tables are joined, and unit or scale inconsistencies.

**Impact.** Estimates built on quietly wrong inputs, with no error to signal the problem.

**Diagnostics.**
- Confirm: column types are not what you expect after loading, missing values appear as text or unexpected `NaN`, row counts or keys do not line up after a join, or units and scales are inconsistent. Check the expected types, columns, and row counts right after loading; failing checks confirm the corruption is at the loading step.
- Rule out: a clean check that passes, with expected missing-value handling and aligned keys, means loading is not the issue.

**Known fix.** Check on load: confirm the expected columns and types, count and handle missing values explicitly, verify row counts and key alignment after every join, and confirm units and ranges. Keep time order to avoid look-ahead leaks (see `NN-1`).

**References.**
- <https://pandas.pydata.org/docs/user_guide/io.html>
- <https://arxiv.org/abs/2207.07048>

#### Notes for iteration {#sec:notes-for-iteration}

- This list is intentionally general and high-priority. Checks specific to one model or method (the particular estimator, priors, moments, or data) belong in a separate model-level checklist, not here.

- Each test type referred to above, in plain terms: a **known-answer test (oracle)** solves a case you can also solve by hand and checks the code matches; an **always-true test (property-based)** checks an economic or statistical property that must hold for any inputs, such as consumption staying within resources or a covariance staying positive definite; a **direction test (metamorphic)** changes one input and checks the output moves the way theory predicts (a comparative-statics check, for example a tighter prior giving a no-wider posterior); a **snapshot test (golden-master)** checks today's run still matches a run you already validated, under a fixed seed and pinned versions, to catch accidental drift.

- When the quality gate is set up, the checkable items here (the no-missing-gradient check, the `NaN`-after-Cholesky check, the log-space likelihood check, the R-hat and effective-sample-size thresholds, the SMM common-random-numbers test, the save-and-reload prediction check, the checks on load, and the known-answer test) should move from "the agent should check this" to "an automatic test that blocks the merge if it fails." This document then carries the judgment items that are harder to automate.

- Keep each reference pinned to the installed version where the behavior depends on the version.
