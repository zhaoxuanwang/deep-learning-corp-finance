# Validation and Tests {#sec:ch-validation}

The pipeline needs tests that check whether its outputs are correct, not only that it runs. Three
categories cover this:

- Oracle (known-answer): run on a case with a known answer (a trusted alternative method such as VFI,
  or data simulated from known true parameters) and check the pipeline recovers it. Sections[@sec:oracle-test-1-monte-carlo-recovery-of-parameters-and-moments] and
 [@sec:oracle-test-2-block-1-policy-recovery-against-vfi].

- Economic property: check properties that theory says must hold for any input, including parameter
  comparative statics. Section[@sec:economic-property-tests].

- Regression: freeze a validated run under a fixed seed and pinned versions and check that later runs
  still match. This catches drift, not correctness. Section[@sec:regression-tests].

Section[@sec:shared-component-the-vfi-benchmark-solver] builds the shared VFI benchmark the oracle tests rely on.

## Benchmark solver {#sec:shared-component-the-vfi-benchmark-solver}

Both oracle tests need an independent, trusted solution of the model at a fixed parameter vector.
Use value function iteration (VFI) on the discrete grid, built only for validation and never used
in the estimation loop.

- Input: one fixed parameter vector $\boldsymbol{\beta}$.

- Grids and operators: the state grid ($11 \times 15 \times 35$ over $(\log z, k, b)$) and control
  grid ($81 \times 91 \times 71$ over $(k', b', c')$) of Section[@sec:grid-refinement], the 11-state Tauchen
  expectation, the exact (non-smoothed) dividend (Eqs. 7 to 9), and the joint value/bond-price fixed
  point (recompute $q$ from the current $V$ each sweep, as in Section[@sec:grid-refinement] step 3).

- Solve: iterate the Bellman operator from a cold start (set $V$ to the no-default, no-adjustment
  value, or to 0). At each state node, maximize the right-hand side by a global search over the full
  control grid. This differs from the grid refinement in Section[@sec:grid-refinement], which does a local search
  around the network policy. Stop when the sup-norm change in $V$ is below $10^{-8}$ and the argmax
  policy is unchanged between sweeps. Reuse the Section[@sec:grid-refinement] policy-evaluation linear solve for the
  inner step; only the policy-improvement search is global.

- Output: VFI value and policy arrays on the grid. Off-grid values use the same log-$k$ bilinear
  interpolation as Section[@sec:panel-simulation].

VFI is intentionally exact and slow. It is the ground truth the network outputs are checked against.

## Oracle test 1: Monte Carlo recovery of parameters and moments {#sec:oracle-test-1-monte-carlo-recovery-of-parameters-and-moments}

This is the critical end-to-end test. It reproduces DF26 Section 4.1 and its Figures 1 and 2. The
idea: build data from known true parameters with the trusted VFI solver, then check that the full
pipeline recovers both the moments and the parameters.

Each draw is an independent recovery trial: run the full pipeline (Blocks 1 to 3 and the controller,
including adaptive shrinkage toward that draw's target) end to end per draw. The networks, the
collection, and the surrogates depend only on the bounds, not on the target moments, so an efficient
implementation may instead train them once over the fixed Table A1 bounds and repeat only the
target-dependent Levenberg-Marquardt estimation per draw; this skips per-target shrinkage, so use it
only if Figure V1 confirms the full-box surrogate is accurate ($R^2$ near 1).

Procedure:

1.  Draw a true parameter vector $\boldsymbol{\beta}^*$ uniformly over the Table A1 bounds (Section
   [@sec:application-inputs-external-values-bounds-targets-reference-estimates]). Fix the seed.

2.  Solve the model at $\boldsymbol{\beta}^*$ with the VFI benchmark (Section[@sec:shared-component-the-vfi-benchmark-solver]).

3.  Simulate a firm panel from the VFI policy using the Section[@sec:panel-simulation] settings (5,000 firms, 300
    periods, 200 burn-in, Tauchen chain).

4.  Filter: if the panel has no defaulting firms, discard the draw and redraw. The recovery rate
    $\chi$ is not identified without defaults, so DF26 excludes these cases. Keep drawing until you
    have $N$ kept draws. DF26 used $N = 40$; use 40 as the baseline and more for tighter coverage.

5.  Compute the 11 targeted moments (Section[@sec:moment-construction]) from the panel. These are the true targets
    $m^*(\boldsymbol{\beta}^*)$.

6.  Run the Levenberg-Marquardt estimation (Sections[@sec:estimation-objective] to[@sec:multiple-restarts-and-bound-enforcement]) against $m^*(\boldsymbol{\beta}^*)$,
    giving the estimate $\hat{\boldsymbol{\beta}}$.

7.  Refine and simulate at $\hat{\boldsymbol{\beta}}$ (Block 2, Sections[@sec:grid-refinement] to[@sec:moment-construction]) to get the
    fitted moments $\hat{m} = m(\hat{\boldsymbol{\beta}})$.

Repeat steps 1 to 7 for all $N$ kept draws, then produce two figures.

Figure V1 (true versus fitted moments, reproduces DF26 Figure 1). Eleven panels, one per moment in
the Section[@sec:moment-construction] order. In each panel the x-axis is the true target moment $m^*_j$ across draws and
the y-axis is the fitted moment $\hat{m}_j$; draw each draw as a marker with a same-colour capped
vertical spike for its 95% confidence interval ($1.96$ times the moment's firm-clustered standard
error, Section[@sec:weighting-matrix]), over a dashed 45-degree line. Fix both axes to a common range with a square box,
so the 45-degree line is exactly diagonal and coverage is read off the full range, not an auto-zoom.
Title each panel with the moment name and annotate its $R^2$, the standard linear-regression
coefficient of determination (the squared correlation between true and fitted, in $[0,1]$). Pass if
$R^2_j \ge 0.99$ for all 11 moments (DF26 reports 1.00).

Figure V2 (true versus fitted parameters, reproduces DF26 Figure 2). Eight panels, one per parameter
($\theta, \rho, \sigma, \delta, \gamma_1, \gamma_0, \chi, c_f$). In each panel the x-axis is the true
$\beta^*_j$ and the y-axis is the estimate $\hat{\beta}_j$; draw each estimate as a marker with a
same-colour capped vertical spike for its 95% confidence interval ($1.96$ times the across-fold
standard deviation of the estimate), over a dashed 45-degree line. Fix both axes to the parameter's
Table A1 range with a square box, so the 45-degree line is exactly diagonal and coverage and bias are
visible across the full range. Annotate each panel's $R^2$ (the squared correlation between true and
fitted, in $[0,1]$, same definition as Figure V1), and also report the per-parameter bias (mean of
$\hat{\beta}_j - \beta^*_j$) and RMSE, which capture the offset and scale that $R^2$ alone does not.
Pass if $R^2_j \ge 0.95$ for at least seven of the eight parameters (DF26 reports $\ge 0.97$ for seven
of eight). $\chi$ is the expected weak one; a low $R^2$ there signals weak identification, not a bug.

Repeat the whole exercise under several seeds and report the spread of $R^2$ across seeds, since the
draws and the simulation noise vary. A correctly specified data-generating process should give near
perfect recovery; a clear miss flags a pipeline bug rather than a modeling limit.

## Oracle test 2: policy network against benchmark {#sec:oracle-test-2-block-1-policy-recovery-against-vfi}

This checks that the network solution matches the trusted on-grid solution at a fixed parameter
vector. It reproduces and extends DF26 Appendix Figure A1.

Fix $\boldsymbol{\beta}$ at the reference estimates (Section[@sec:application-inputs-external-values-bounds-targets-reference-estimates]), or any interior vector; state which.
Build three solutions at that $\boldsymbol{\beta}$: the raw Block 1 network policy, the refined
network policy (Block 2 grid refinement at $\boldsymbol{\beta}$, Section[@sec:grid-refinement]), and the VFI benchmark
policy (Section[@sec:shared-component-the-vfi-benchmark-solver]).

Reference holding point for one-dimensional slices: median productivity $z = 1$, near-zero net debt
$b = -0.03$, a fixed low capital $k_{\text{ref}}$ (state the value, for example $k_{\text{ref}} = 0.2$), and the held parameters at the reference $\boldsymbol{\beta}$.

Figure V3 (policy slices). A grid of one-dimensional slices. Rows are the policies $i$, $b'$, and
$c'$ (add $V$ if useful). Columns are the 11 arguments: the three states $k$, $b$, $z$ and the eight
parameters $\theta, \rho, \sigma, \delta, \gamma_1, \gamma_0, \chi, c_f$. For each (policy, argument)
panel, vary that one argument over its range (state ranges from Section[@sec:input-normalization-and-bounds]; parameter ranges from
Table A1) on about 50 points, hold every other state and parameter at the reference point, and plot
the policy.

- State-argument panels (columns $k$, $b$, $z$): produce one figure per state axis, each a 1x4 panel
  over $V, i, b', c'$, so the (policy, state) combinations are complete. Overlay three curves: raw VFI
  as a thick dashed grey line, the Block 1 network in blue, and the refined network in bold red. Keep
  the refined curve the most visible, since it is the policy actually used.

- Parameter-argument panels (the eight parameters): produce one 1x3 figure per parameter (columns $i$,
  $b'$, $c'$), saved separately per parameter. Overlay two curves only, the Block 1 network (blue) and
  the refined network (red); raw VFI is omitted, because a single VFI solve fixes $\boldsymbol{\beta}$
  and so cannot trace a parameter axis (see the optional extension).

The $k$ column for rows $i$, $b'$, $c'$ at $z = 1$ and $b = -0.03$ is exactly DF26 Appendix Figure
A1. Reproduce it as the headline panel set, with the same three-curve legend.

Quantitative pass criteria (beyond the visual overlay). Over the state grid in the good region
($V > 0$), compare the refined network policy to VFI: report the maximum and mean absolute deviation
per policy. The refinement targets VFI-level accuracy (Section[@sec:grid-refinement], stop at $10^{-10}$), so the
refined policy should sit on the VFI curve; pass if the maximum relative deviation is below 1 percent
per policy (or an absolute threshold scaled to each policy's range, stated in the output). Report the
raw Block 1 deviation as well for context, with a looser sanity bound (for example within about 10
percent); the raw network is the unrefined approximation and is not required to match VFI tightly.

Optional comprehensive extension (skip at the current stage). To benchmark the parameter-axis panels
against VFI, re-solve VFI at each point along a parameter axis (one VFI solve per grid point of that
parameter, with the others held), producing a VFI curve for the parameter slices. This is expensive,
one full VFI solve per point, but it validates the network's parameter dependence against the trusted
method. Add it once the state-axis checks pass.

## Economic property tests {#sec:economic-property-tests}

These check properties that must hold for any input, not just on a known-answer case. Groups 1 to 5
are always-true properties; Group 6 is parameter comparative statics. Each check is hard or a
diagnostic: a hard check gates the suite (a violation fails the run); a diagnostic is computed and
reported but never fails it. Hard checks are reserved for deterministic or theory-certain properties;
anything expected but not guaranteed is a diagnostic. Shape and slice checks reuse the one-dimensional
slice machinery and reference point of Section[@sec:oracle-test-2-block-1-policy-recovery-against-vfi], evaluated on the VFI and refined-network solutions
unless noted.

**Group 1, correct bounds (all hard; deterministic domain, checked on every evaluation).**

- Bond price $q \in [0,\ 1/(1+r_f(1-\tau))]$; the bracket is a repayment fraction in
  $[\text{recovery},\ 1]$.

- Default probability $P_{\text{def}} \in [0,1]$; recovery fraction $R/(b'k') \in [0,1)$ when the
  gate $g=1$.

- Controls land in their boxes: $i$ keeps $k'=(1+i-\delta)k \in [k_{\min}, k_{\max}]$; $b' \in [0,2]$; $c' \in [0,1]$; net debt $b'-c' \in [-1,2]$. Checking the realized values catches a broken
  Eq. 26 mapping.

- Tauchen transition matrix: rows non-negative and summing to 1.

- Simulated states stay in the grid box; a leak signals an interpolation or bound bug.

**Group 2, correct economics (monotone shape versus states).**

Hard assertions (three): $V$ increasing in $z$; $V$ decreasing in $b$; $i$ decreasing in $k$. Test:
for each, draw about 20 hold configurations (the two held states and $\boldsymbol{\beta}$ sampled
uniformly in the current bounds), sweep the varying state over its grid range on about 50 points,
evaluate the VFI and refined solutions, keep interior points with $V>0$, drop the two endpoints, take
consecutive differences, ignore steps with $|\Delta| < \varepsilon_{\text{mono}} = 10^{-3} \times (\text{slice max} - \text{slice min})$, and require every remaining step to carry the asserted sign.
Pass means zero violations across all configurations on both VFI and the refined network. The raw
Block 1 network is reported, not gated.

Diagnostics (reported, not gated): $V$ versus $k$ (expected up at low $k$, may flatten or reverse at
high $k$, since net debt is per unit of capital so the debt level scales as $b\,k$ under decreasing
returns); $i$ versus $z$ (expected up); $i$ versus $b$ (expected down).

Skipped (no sign asserted): $b'$ and $c'$ versus every state are ambiguous and hump-shaped (Appendix
A1b, A1c); validate their shape through the Section[@sec:oracle-test-2-block-1-policy-recovery-against-vfi] overlays instead.

**Group 3, correct mechanics (optimality and self-consistency).**

Hard checks: the Bellman residual $R = V - D - (1/(1+r_f))\,\mathbb{E}[\max\{V',0\}]$ (Eq. 27) is
small on the refined solution (tight, since refinement converges); the bond-price residual
$|q - q(V)|$ (Eq. 6 at the solved $V$) is small on the refined solution; the accounting identities
hold exactly when recomputed and compared ($k'=(1+i-\delta)k$, next net debt $= b'-c'$, $D$ from
Eqs. 7 to 9 equals the $D$ used, $R$ equals Eq. 5); the weighting matrix $\hat{\Sigma}$ is symmetric
positive definite (Cholesky succeeds); and no value, policy, price, moment, Jacobian, or weighting
entry is NaN or infinite.

Diagnostics: the Bellman and bond residuals on the trained network (looser, since it is approximate);
the Levenberg-Marquardt first-order condition (gradient near zero at $\hat{\boldsymbol{\beta}}$, and
the objective below nearby perturbations, which reads as flat for weakly identified directions like
$\chi$); and the policy-improvement objective (Eq. 29 right-hand side) having plateaued.

**Group 4, corner cases (the deterministic checks above, evaluated at named extremes).**

Hard checks: in the no-default region (low leverage or high $\chi$, so the gate $g=0$), $q$ equals
the risk-free price $1/(1+r_f(1-\tau))$ exactly, and a simulation with zero defaults runs to finite
output without error; under high leverage ($R<b'k'$, $g=1$), $q$ stays in $(0,\ \text{risk-free})$
with recovery below 1; at every Table A1 corner the production constants are finite ($\nu = 1-(1-\alpha)\theta > 0$, so $\xi$ and $A_\pi$ finite), $k_{\min}>0$ so per-capital terms are safe, and
the Tauchen grid still covers the high-$\sigma$, high-$\rho$ case.

Diagnostics: when the optimal control sits at a bound ($c'\to 0$, $b'\to 0$, or $i$ at its limit), the
sigmoid policy saturates near the edge while the grid refinement reaches the true edge, so report
their agreement; and in the no-default case, report the $\chi$ identification flag (Section[@sec:global-identification-diagnostic])
rather than gating on it.

**Group 5, correct consistency (two independent routes agree).**

Hard check: surrogate moments versus simulate-then-measure at the same $\boldsymbol{\beta}$, where the
out-of-sample $R^2$ (Section[@sec:cross-validation]) is at or above its acceptance threshold; a low $R^2$ invalidates the
estimation.

Diagnostics: the training Taylor $P_{\text{def}}$ versus the exact grid $P_{\text{def}}$ (report the
gap, expected small in the interior and largest near the default boundary, consistent with the paper
treating the grid value as the accurate one); the Gauss-Hermite continuation versus the Tauchen
continuation at the same state; and, optionally, normalization invariance (the network output at a
fixed raw $(z,k,b,\boldsymbol{\beta})$ is near-invariant to the active bounds on their overlap), which
catches renormalization bugs.

**Group 6, correct comparative statics (parameter monotonicities; all hard, no diagnostics).**

Assert, holding the other seven parameters and the reference state fixed:

- $V$ decreasing in $c_f$, in $\delta$, in $\gamma_1$, and in $\gamma_0$.

- $V$ increasing in $\chi$.

- $q$ non-decreasing in $\chi$ at fixed state, choices, and $P_{\text{def}}$.

Test: evaluate the Block 1 value network's $V$ at the Section[@sec:oracle-test-2-block-1-policy-recovery-against-vfi] reference state across about 20 hold
configurations (the held state and the other seven parameters sampled uniformly in the current
bounds), sweep the one parameter over its Table A1 range on about 50 points, take consecutive
differences, ignore steps with $|\Delta| < \varepsilon_{\text{mono}} = 10^{-3} \times (\text{sweep max} - \text{sweep min})$, and require every remaining step to carry the asserted sign. For $q$, evaluate
Eq. 6 along the $\chi$ axis at fixed choices and $P_{\text{def}}$.

These are the only certain parameter signs. The value is a maximum, so it inherits any parameter shift
that moves the payoff one way for every policy (a dominance argument needing no closed form), and the
tax enters only through the bond-price debt term, so no profit-tax or depreciation shield offsets the
cost and depreciation signs; $q$ is a direct function of recovery $R = c'k' + \chi(1-\delta)k'$.
Control and moment signs are not asserted, because the controls are the maximizer, with competing
channels and non-convex frictions that leave them ambiguous, and the moments add the stationary
distribution; those are covered by the oracle recovery test (Section[@sec:oracle-test-1-monte-carlo-recovery-of-parameters-and-moments]) and the Section[@sec:oracle-test-2-block-1-policy-recovery-against-vfi]
overlays. The value network evaluates any $\boldsymbol{\beta}$ cheaply, so a violation flags a bug;
VFI along a parameter axis can confirm a few points but is not required.

## Regression tests {#sec:regression-tests}

This is a thin drift guard, not a correctness test; correctness is covered by the oracle and economic
property tests above. It does not re-run training, which is slow and, under the async design, not
reproducible (Section[@sec:seeding-and-reproducibility]). Instead it freezes one validated run and re-checks only the deterministic,
fast pieces.

Freeze once, from a run that passed the oracle and economic property tests, and commit as small
files: the trained network checkpoint (value, three policies, twelve generators) and the surrogate
checkpoint; a handful of fixed evaluation inputs $(z, k, b, \boldsymbol{\beta})$; one parameter vector
$\boldsymbol{\beta}_0$ with its target moments; and the master seed.

The test reloads these and recomputes, comparing each output to its frozen golden:

- the network forward pass at the fixed inputs ($V$ and the three controls), guarding the network,
  FiLM, and normalization code;

- the bond price $q$ and dividend $D$ at the fixed inputs via Eqs. 6 to 9, guarding the economic
  primitives;

- one on-grid refinement sweep from the frozen $(V, q, \text{policy})$, guarding the Tauchen
  expectation, the semismooth-Newton step, and GMRES;

- a short fixed-seed simulation from the frozen policy (a few hundred firms, short horizon) and its
  moments, guarding the simulation and moment code;

- one LM solve on the frozen surrogate against $\boldsymbol{\beta}_0$, guarding the Jacobian, the
  weighting matrix, and the optimizer.

Each piece is a forward evaluation or a single operator application, so the whole test runs in seconds.
Run it on a fixed machine with pinned versions, single-GPU and serial, with op-level determinism on
(Section[@sec:seeding-and-reproducibility]), so the float math is repeatable. Compare floats at rtol $10^{-5}$, atol $10^{-7}$;
integer and structural outputs, such as default counts and the active-set partition, must match
exactly.

When a change or a version bump alters the goldens on purpose, regenerate them under the same seed and
commit them with a one-line note. The test then flags only unintended drift.

Optional: to also guard the training loop, add a short deterministic training slice, a handful of
gradient steps from a fixed init and seed in the serial single-GPU mode, and freeze the loss
trajectory.

## Value and policy accuracy against VFI {#sec:value-and-policy-accuracy-against-vfi}

Figure[@fig:v3slices] reports the Block 1 policy-recovery test (Section[@sec:oracle-test-2-block-1-policy-recovery-against-vfi]): after grid refinement, the network value and policy functions reproduce the VFI benchmark across the state space.

::: {#fig:v3slices}
![](figures/fig_v3_slices_k.png){width=100%}

![](figures/fig_v3_slices_b.png){width=100%}

![](figures/fig_v3_slices_z.png){width=100%}

Value and policy function slices: refined network versus VFI benchmark.
:::

*Notes:* Each row varies one state variable (capital $k$, debt $b$, productivity $z$) with the others held at reference values; columns are the value $V$, investment rate $i$, gross debt $b'$, and cash $c'$. The refined network (used downstream) overlays the grid VFI benchmark; the raw network is shown for comparison.

## Monte Carlo recovery results {#sec:monte-carlo-recovery-results}

Figures[@fig:v1moments] and[@fig:v2params], with Tables[@tbl:moments] and[@tbl:params], report the Monte Carlo recovery test (Section[@sec:oracle-test-1-monte-carlo-recovery-of-parameters-and-moments]): data are simulated from known parameters, re-estimated through the full pipeline, and compared to the truth.

![Moment recovery: fitted versus true moments across replications.](figures/fig_v1_moments.png){#fig:v1moments width=90%}

*Notes:* Each panel plots the fitted (model-implied) value of one of the 11 target moments against its true value across Monte Carlo replications; the dashed line is perfect recovery ($45^\circ$) and $R^2$ in each panel title is the squared correlation. Per-moment values are in Table[@tbl:moments].

![Parameter recovery: estimated versus true parameters across replications.](figures/fig_v2_params.png){#fig:v2params width=100%}

*Notes:* Each panel plots the estimate of one of the eight structural parameters against its true value, with 95% confidence intervals; the dashed line is perfect recovery and $R^2$ in each panel title is the squared correlation. Per-parameter values are in Table[@tbl:params].

| Moment | True | Fitted | 95% CI | $R^2$ |
|:--|--:|--:|:-:|--:|
| Mean inv rate | $-0.051$ | $-0.056$ | $[-0.057,\,-0.056]$ | 0.85 |
| SD inv rate | 0.111 | 0.110 | $[0.109,\,0.111]$ | 0.64 |
| Mean op income | 0.131 | 0.137 | $[0.136,\,0.137]$ | 0.71 |
| SD op income | 0.041 | 0.047 | $[0.047,\,0.048]$ | 0.71 |
| Autocorr income | 0.251 | 0.567 | $[0.558,\,0.577]$ | 0.00 |
| Mean debt | 0.454 | 0.512 | $[0.510,\,0.514]$ | 0.71 |
| SD debt | 0.202 | 0.202 | $[0.201,\,0.203]$ | 0.36 |
| Mean cash | 0.183 | 0.195 | $[0.194,\,0.196]$ | 0.63 |
| SD cash | 0.123 | 0.117 | $[0.117,\,0.118]$ | 0.42 |
| Cash\~net debt | 0.270 | 0.293 | $[0.290,\,0.296]$ | 0.51 |
| Cash\~income | 0.514 | 0.231 | $[0.202,\,0.260]$ | 0.01 |

: Moment recovery summary. {#tbl:moments}

*Notes:* Mean across Monte Carlo replications of each of the 11 target moments: true value, fitted (model-implied) value, 95% confidence interval (firm-clustered standard errors), and $R^2$ (squared correlation between fitted and true). Corresponds to Figure[@fig:v1moments].

| Parameter | True | Estimate | RMSE | 95% CI | $R^2$ |
|:--|--:|--:|--:|:-:|--:|
| $\theta$ (returns to scale) | 0.703 | 0.703 | 0.062 | $[0.610,\,0.796]$ | 0.35 |
| $\rho$ (persistence) | 0.650 | 0.693 | 0.127 | $[0.547,\,0.839]$ | 0.03 |
| $\sigma$ (shock SD) | 0.125 | 0.135 | 0.035 | $[0.097,\,0.173]$ | 0.48 |
| $\delta$ (depreciation) | 0.111 | 0.113 | 0.032 | $[0.076,\,0.151]$ | 0.53 |
| $\gamma_1$ (convex adj. cost) | 0.525 | 0.636 | 0.330 | $[0.278,\,0.994]$ | 0.14 |
| $\gamma_0$ (fixed adj. cost) | 0.104 | 0.123 | 0.079 | $[0.023,\,0.223]$ | 0.07 |
| $\chi$ (recovery rate) | 0.454 | 0.532 | 0.252 | $[0.203,\,0.860]$ | 0.35 |
| $c_f$ (fixed op. cost) | 0.157 | 0.153 | 0.052 | $[0.087,\,0.220]$ | 0.42 |

: Parameter recovery summary. {#tbl:params}

*Notes:* Across Monte Carlo replications for each of the eight structural parameters: true value, mean estimate, root-mean-squared error, 95% confidence interval (across-fold standard deviation), and $R^2$ (squared correlation between estimate and true). Corresponds to Figure[@fig:v2params].
