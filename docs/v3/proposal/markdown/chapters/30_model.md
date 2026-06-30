# Model and Estimation {#sec:ch-model}

This chapter describes the algorithm, methods, technical stack, and implementation details for solving and estimating structural corporate finance models. The methodological reference is Duarte and Fonseca (2026).

## Purpose, scope, and notation {#sec:purpose-scope-and-conventions}

This document is a self-contained build specification for reproducing the DF26 estimation
pipeline end to end. It restates the model, the algorithm, and every implementation choice in
the paper's own notation, and strips out literature review, motivation, and results discussion
except where context is needed to implement a step correctly. DF26 is the single source of
truth here. Where the paper is silent or internally inconsistent, the implementation decision is
recorded in Section[@sec:resolved-design-decisions] (Resolved design decisions).

**Target stack.** The reference implementation in DF26 is written in JAX. This build must be
written in **TensorFlow (TF) and TensorFlow Probability (TFP)**. NumPy is allowed only where it
is the clearly better tool for a one-time setup computation (for example, fixed quadrature
nodes), and even then the result must be converted to a `tf.constant` and consumed natively in
TF. Implementation notes on the JAX-to-TF mapping are in Section[@sec:tensorflow-tfp-implementation-notes]. No code appears in this
document; code is produced by a separate agent.

**Notation.** $\boldsymbol{\beta}$ is the **estimated parameter vector**. The firm's discount
factor is the fixed constant $1/(1+r_f)$ and is written out explicitly wherever it appears;
$\beta$ is reserved for the parameter vector. (DF26 uses $\beta$ for both.)

The canonical order of the estimated parameter vector, used throughout DF26's estimation text, is

$$\boldsymbol{\beta} = (\theta,\ \rho,\ \sigma,\ \delta,\ \gamma_1,\ \gamma_0,\ \chi,\ c_f).$$

Bind each parameter to its bounds and value by name, not by table position; the paper's tables
sometimes print a different order, which is cosmetic.

**Primes.** A prime ($'$) denotes a next-period quantity. Uppercase letters are levels;
lowercase letters are the same quantity expressed per unit of capital. The link is
$C = c\cdot k$, $B = b\cdot k$, $C' = c'\cdot k'$, $B' = b'\cdot k'$.

## Model {#sec:economic-model-partial-equilibrium}

Time is discrete. A single infinitely lived firm combines capital and labor to produce output
under productivity shocks, pays fixed operating costs before production, and finances itself
with current profits, one-period risky debt, external equity, and internal cash. Two frictions
matter: default carries deadweight (recovery) costs, and equity issuance carries fees.

### Parameters {#sec:parameters}

**Estimated parameters** $\boldsymbol{\beta} = (\theta, \rho, \sigma, \delta, \gamma_1, \gamma_0, \chi, c_f)$:

| Symbol     | Meaning                                           |
|:-----------|:--------------------------------------------------|
| $\theta$   | Returns to scale, $\theta \in (0,1)$              |
| $\rho$     | Autocorrelation of log productivity               |
| $\sigma$   | Standard deviation of the productivity innovation |
| $\delta$   | Depreciation rate                                 |
| $\gamma_1$ | Convex capital adjustment cost coefficient        |
| $\gamma_0$ | Fixed capital adjustment cost coefficient         |
| $\chi$     | Default recovery rate                             |
| $c_f$      | Fixed operating cost                              |

**External (fixed) parameters**, set following Gao, Whited, and Zhang (2021):

| Symbol      | Value | Meaning                                           |
|:------------|:------|:--------------------------------------------------|
| $r_f$       | 0.02  | Risk-free rate; discount factor is $1/(1+r_f)$    |
| $\alpha$    | 0.30  | Capital share                                     |
| $\tau$      | 0.20  | Corporate tax rate                                |
| $\lambda_0$ | 0.007 | Fixed equity issuance cost                        |
| $\lambda_1$ | 0.054 | Proportional equity issuance cost                 |
| $r_c$       | 0     | Interest earned on cash; this makes $\iota_c = 1$ |

### State variables and controls {#sec:state-variables-and-controls}

**States:** $(z, k, b)$.

- $z$: productivity. It evolves as an AR(1) in logs (Eq. 2) and enters operating surplus
  linearly (Eq. 3). The $z$ in Eq. 3 is the reduced-form productivity shifter that has absorbed
  the $1/\nu$ exponent from the static labor choice; the spec treats the single state $z$ as that
  shifter throughout (AR(1), Tauchen grid, value function, surplus).

- $k$: capital.

- $b$: net debt, $b = b^{\text{gross}} - c$, where $b^{\text{gross}}$ is gross debt per unit of
  capital and $c$ is cash per unit of capital. Negative $b$ means a net cash position.

**Controls:** $(i, b', c')$.

- $i$: investment **rate**, defined so that $k' = (1+i-\delta)k$ (Eq. 10). Note this is the rate,
  not the level $I$. The level satisfies $I = i\,k = k' - (1-\delta)k$.

- $b'$: next-period **gross** debt per unit of next-period capital $k'$.

- $c'$: next-period cash per unit of next-period capital $k'$.

The next-period net-debt state is therefore $b' - c'$, which is why the continuation value in
the Bellman equation is evaluated at $(z', k', b'-c')$.

### Technology and production {#sec:technology-and-production}

Output:

$$y = z\,\big(k^{\alpha}\,\ell^{\,1-\alpha}\big)^{\theta}, \qquad \alpha \in (0,1),\ \theta \in (0,1). \tag{1}$$

Log productivity follows an AR(1):

$$\log z' = \rho \log z + \sigma \varepsilon', \qquad \varepsilon' \sim \mathcal{N}(0,1). \tag{2}$$

Labor is flexible and hired at competitive wage $w$. Substituting its closed-form optimum gives
the reduced-form operating surplus:

$$\pi(z,k) = z\,A_\pi\,k^{\xi} - c_f, \qquad \xi \equiv \frac{\alpha\theta}{\nu}, \quad \nu \equiv 1-(1-\alpha)\theta. \tag{3}$$

In partial equilibrium the wage is normalized to $w = 1$, so $A_\pi$ is a constant in
$(\alpha, \theta)$:

$$A_\pi = \nu\,\big((1-\alpha)\theta\big)^{\frac{(1-\alpha)\theta}{\nu}}.$$

Labor and the wage do not enter the dynamic problem.

### Investment and adjustment costs {#sec:investment-and-adjustment-costs}

Capital accumulates as

$$k' = (1-\delta)k + I. \tag{4}$$

Investment incurs a convex cost $\tfrac{1}{2}\gamma_1 I^2 / k$ and a fixed cost $\gamma_0 k\, \mathbf{1}\{I>0\}$ that is proportional to capital and is paid whenever investment is positive.

### Debt, default, recovery, and bond pricing {#sec:debt-default-recovery-and-bond-pricing}

The firm issues one-period debt at an endogenous price $q$ that reflects default risk.

**Liquidation recovery.** A defaulting firm is liquidated, and creditors recover

$$R(k', c') = c'k' + \chi(1-\delta)k', \tag{5}$$

its cash plus the salvage value of depreciated capital, where $\chi$ is the recovery rate.

**Default rule.** The firm defaults when (i) its continuation equity value is negative and (ii) its
liquidation proceeds do not cover its outstanding debt. With continuation value $V(z', k', b'-c')$,
recovery $R$ (Eq. 5), and outstanding gross debt $b'k'$, the default indicator is

$$\mathbf{1}_{\text{def}} = \mathbf{1}\{V(z', k', b'-c') < 0\}\cdot\mathbf{1}\{R(k',c') < b'k'\}.$$

Condition (ii), $R < b'k'$ (equivalently $c' + \chi(1-\delta) < b'$), depends only on the controls
and parameters, not on $z'$. It is a deterministic gate: it does not constrain $i$, $b'$, or $c'$,
and it keeps the recovery rate $R/(b'k')$ below 1 whenever default occurs. Condition (i) alone, the
limited-liability walk-away, governs firm exit in the simulation (Section[@sec:panel-simulation]) and the moment filter
(Section[@sec:moment-construction]); (ii) only gates whether creditors take a loss.

**Bond price** (discounted expected repayment):

$$q(z, k', c', b') = \frac{1}{1 + r_f(1-\tau)}\,
\mathbb{E}_{z'\mid z}\!\left[(1 - \mathbf{1}_{\text{def}}) + \mathbf{1}_{\text{def}}\,\frac{R(k',c')}{b'k'}\right]. \tag{6}$$

The gate (ii) and the ratio $R/(b'k')$ are constant across $z'$, so with $g = \mathbf{1}\{R < b'k'\}$
and $P_{\text{def}} = \Pr\big(V(z', k', b'-c') < 0\big)$, Eq. 6 collapses to

$$q = \frac{1}{1+r_f(1-\tau)}\Big[(1 - g\,P_{\text{def}}) + g\,P_{\text{def}}\,\tfrac{R(k',c')}{b'k'}\Big].$$

When $R \ge b'k'$ ($g = 0$) the bond is risk-free; when $R < b'k'$ ($g = 1$) the recovery rate is below 1.

### Dividends and equity issuance (two subperiods) {#sec:dividends-and-equity-issuance-two-subperiods}

Cash flow is split into a preproduction and a postproduction stage.

**Preproduction** (roll over debt, pay fixed operating cost):

$$d_1 = q\cdot b' \cdot k' - b\cdot k - c_f. \tag{7}$$

If $d_1 > 0$, the excess carries forward. If $d_1 < 0$, the firm issues equity at cost
$\lambda_0 k + \lambda_1 |d_1|$ (fixed plus proportional).

**Postproduction** (collect output net of investment and adjustment costs, plus any carried
excess):

$$d_2 = z\,A_\pi\,k^{\xi} + \max\{d_1, 0\} - I - \tfrac{1}{2}\gamma_1 I^2 / k - c'k'\iota_c - \gamma_0 k\,\mathbf{1}\{I>0\}, \tag{8}$$

with $\iota_c = 1/\big(1 + r_c r_f(1-\tau)\big)$. Since $r_c = 0$, $\iota_c = 1$. If $d_2 > 0$ the
firm distributes; if $d_2 < 0$ it issues equity at the same fixed-plus-proportional cost.

**Net payout to shareholders** (issuance costs subtracted wherever issuance occurs):

$$D = d_2 + d_1\cdot\mathbf{1}\{d_1<0\} - (\lambda_0 k + \lambda_1 |d_1|)\cdot\mathbf{1}\{d_1<0\} - (\lambda_0 k + \lambda_1 |d_2|)\cdot\mathbf{1}\{d_2<0\}. \tag{9}$$

Reading of the terms: $d_2$ already includes the rolled-over positive preproduction excess; the
$d_1\cdot\mathbf{1}\{d_1<0\}$ term restores the negative preproduction cash flow that the rollover
left out; the last two terms are the issuance costs in each subperiod where the firm taps equity.

### Bellman equation {#sec:bellman-equation}

Equity value solves

$$V(z,k,b) = \max_{i, b', c'}\Big\{ D(z,k,b,i,b',c') + \tfrac{1}{1+r_f}\,\mathbb{E}_{z'\mid z}\big[\max\{V(z', k', b'-c'),\ 0\}\big]\Big\}, \tag{10}$$

with $k' = (1+i-\delta)k$. The $\max\{V,0\}$ term is limited liability:
shareholders walk away when equity value is negative. Eq. 6 (bond price) and Eq. 10 (value) are
jointly determined, because $q$ depends on $V$ through default and $V$ depends on $q$ through the
dividend. The algorithm breaks this circularity with a target network during training and with
alternation during grid refinement (Sections[@sec:bond-pricing-circularity-target-network] and[@sec:grid-refinement]).

## Estimation method overview {#sec:estimation-method-overview-and-execution-order}

The method solves the model once for all parameter values and then estimates with fast
gradient-based optimization. It has three blocks plus an asynchronous controller.

- **Block 1 (Section[@sec:block-1-value-and-policy-networks]): solve the model for all parameters at once.** Train a value network and
  three policy networks that take states and parameters as inputs, using policy iteration. After
  training they return the solution for any parameter vector instantly.

- **Block 2 (Section[@sec:block-2-grid-refinement-simulation-and-moments]): simulate moments.** For each sampled parameter vector, refine the
  network solution on a grid, simulate a firm panel, and compute the targeted moments. This
  builds a dataset of (parameters, moments) pairs.

- **Block 3 (Section[@sec:block-3-moment-surrogate-networks-and-estimation]): estimate.** Train one differentiable "moment-surrogate" network per
  moment on that dataset, then estimate parameters by minimizing a weighted distance between data
  and model-implied moments using Levenberg-Marquardt with analytical Jacobians from the
  surrogate networks.

- **Adaptive controller (Section[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification]):** profile the objective one parameter at a time to get
  "minimum loss functions," use them to shrink parameter bounds during training, and use them as
  a global identification diagnostic.

The map below shows how the blocks connect and loop.

![End-to-end estimation pipeline and adaptive loop](figures/pipeline_overview.png){width="82%"}

**End-to-end loop (the controller's schedule):**

1.  Initialize parameter bounds (Table A1, Section[@sec:application-inputs-external-values-bounds-targets-reference-estimates]) and the state-bound formulas.

2.  Continuously (Block 1) run value/policy training epochs (500 gradient steps each) drawing
    parameters and states from the current bounds.

3.  Concurrently (Block 2) draw parameter vectors, refine on a grid, simulate panels, compute
    moments, and append (parameter, moment) rows to a shared dataset.

4.  At the end of every epoch: retrain the 110 moment-surrogate networks on the accumulated
    dataset, run the Levenberg-Marquardt estimation to get updated estimates, compute the minimum
    loss functions, and evaluate moment-network out-of-sample $R^2$.

5.  After a 200-epoch warm-up, at the end of every epoch attempt to shrink the parameter bounds
    using the minimum loss functions (subject to the safeguards in Section[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification]). Recompute all
    normalizations whenever bounds change.

6.  Continue until convergence of the estimation loss.

## Block 1: value and policy networks {#sec:block-1-value-and-policy-networks}

### What the networks represent {#sec:what-the-networks-represent}

- One **value network** approximates $V(z,k,b;\boldsymbol{\beta})$, returning a scalar.

- Three **policy networks**, one each for $i$, $b'$, $c'$, each returning a scalar.

All four take the state vector and the parameter vector as inputs. They share the same
architecture (Section[@sec:architecture-film-conditioning]). Each has its own weights.

### Input normalization and bounds {#sec:input-normalization-and-bounds}

Every input is mapped to $[-1,1]$ by

$$\tilde{x} = \frac{2(x - x_{\min})}{x_{\max} - x_{\min}} - 1, \tag{24}$$

using that variable's $[x_{\min}, x_{\max}]$.

**State bounds** depend on parameters and are recomputed per parameter vector:

- Log productivity: $\log z \in \big[-m\,\tfrac{\sigma}{\sqrt{1-\rho^2}},\ +m\,\tfrac{\sigma}{\sqrt{1-\rho^2}}\big]$ with $m = 2.5$. These are the unconditional $\pm 2.5$ standard-deviation bounds of the AR(1). They depend on $\sigma$ and $\rho$.

- Capital: $k \in [k_{ss}(z_{\min}),\ k_{ss}(z_{\max})]$. The frictionless investment Euler equation is
  $$1 = \tfrac{1}{1+r_f}\,\mathbb{E}_{z'\mid z}\big[\,\xi z' A_\pi (k')^{\xi-1} + (1-\delta)\,\big].$$
  $k_{ss}(z)$ is its steady state for productivity held permanently at $z$, where $z' = z$ and the
  expectation collapses to $z$, giving the user-cost condition $\xi z A_\pi k^{\xi-1} = \delta + r_f$ and
  $$k_{ss}(z) = \big[\xi z A_\pi/(\delta+r_f)\big]^{1/(1-\xi)},$$
  evaluated at $z_{\min}$ and $z_{\max}$. This depends on $\delta$, $r_f$, $\theta$, $\alpha$ (through
  $\xi$ and $A_\pi$) and $\rho$, $\sigma$ (through $z_{\min}, z_{\max}$). A one-step target would
  instead use the conditional mean $\mathbb{E}_{z'\mid z}[z'] = z^{\rho} e^{\sigma^2/2}$ (the $Q=5$
  Gauss-Hermite rule of Eq. 30 evaluates it), but that bound is far tighter (it can cut the upper
  bound by about a third) and risks simulated capital leaving the grid, so the permanent-$z$ form is used.

- Net debt: $b \in [-1, 2]$.

Because state bounds depend on parameters, the same raw state maps to different normalized
inputs across parameter vectors.

**Control bounds.** Each control maps to its feasible interval through Eq. 26. The intervals are
functions of the state and parameters only, not of each other, so the three sub-networks stay
independent.

- Investment rate: $i \in \big[k_{\min}/k - (1-\delta),\ k_{\max}/k - (1-\delta)\big]$, a function of
  the current $k$. These are the rates for which next-period capital $k' = (1+i-\delta)k$ stays in
  the capital box.

- Gross debt: $b' \in [0, 2]$. Gross debt is non-negative, and 2 is the net-debt ceiling.

- Cash: $c' \in [0, 1]$. Cash is non-negative. The upper bound is 1 because the next-period net-debt
  state $b' - c'$ must stay in its box $[-1, 2]$: with $b' \ge 0$, the deepest net cash position
  $b' - c' = -1$ is reached at $b' = 0$, which needs $c' \le 1$. With $b' \in [0, 2]$ and
  $c' \in [0, 1]$, $b' - c'$ spans exactly $[-1, 2]$.

**Parameter bounds** are normalized with the same Eq. 24 using the current bounds. The initial
bounds are in Table A1 (Section[@sec:application-inputs-external-values-bounds-targets-reference-estimates]). They narrow over training (Section[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification]). Recompute the
normalization whenever bounds change so inputs always sit in $[-1,1]$.

### Architecture: FiLM conditioning {#sec:architecture-film-conditioning}

States and parameters play different roles, so they enter differently. Feature-wise Linear
Modulation (FiLM, Perez et al. 2018) keeps a compact trunk for the state-dependence and injects the
parameter-dependence through a small generator that modulates each trunk layer, which avoids the large
network a plain concatenation of all 11 inputs would need (DF26 Appendix A.12.1). The **state vector**
$(\log z, k, b)$ (normalized, 3-dimensional) enters the trunk; the **parameter vector** (normalized,
8-dimensional) enters only through FiLM.

![FiLM data flow in one value or policy network](figures/film_data_flow.png){width="\\linewidth"}

One network is one trunk plus three generators, one generator per trunk hidden layer. The state
flows only through the trunk; the parameters flow only through the generators, whose per-layer scale
$\gamma_l$ and shift $\delta_l$ rescale and shift that layer's pre-activation (Eq. 25). Across the
four networks ($V$, $i$, $b'$, $c'$) this is 4 trunks and 12 generators, trained together in one
backward pass. The architecture stays small because the trunk learns the state shape once while the
generators learn only the lower-dimensional map from parameters to per-layer gains and shifts, which
gives a different effective function for each parameter vector.

For each trunk hidden layer $l$, a small auxiliary generator network maps the normalized
parameter vector to a scale $\gamma_l$ and a shift $\delta_l$, each of the same width as the
hidden layer. The layer output is

$$h_{l+1} = \phi\big(\gamma_l \odot (W_l h_l + c_l) + \delta_l\big), \tag{25}$$

where $\odot$ is element-wise multiplication, $W_l, c_l$ are the layer's weight matrix and bias,
and $\phi$ is the activation.

**Trunk.** 3 hidden layers, 128 units each, SiLU activation $\phi(x) = x\,\sigma(x)$ where
$\sigma$ is the logistic sigmoid. The output layer is linear with no activation.

**FiLM generator** (one per trunk hidden layer, so 3 generators per network, with their own
weights): input is the normalized parameter vector; 1 hidden layer of 32 units; output length
$2 \times 128$, split into $\gamma_l$ (128) and $\delta_l$ (128). The generator hidden-layer
activation is SiLU, consistent with the trunk and with Table A2. Initialize the generator output
layer so the modulation starts as the identity, $\gamma_l = 1$ and $\delta_l = 0$: set the output
weights near zero, the $\gamma_l$ bias to 1, and the $\delta_l$ bias to 0. Each FiLM layer is then a
no-op at the start of training, so the trunk keeps its initial behavior and the generator learns
departures from the identity as training proceeds.

**Value network output.** A single scalar interpreted directly as $V$ (no squashing).

**Policy network outputs.** Each of the three control sub-networks outputs one raw scalar $r$,
mapped to a feasible interval by

$$a = a_{\min} + (a_{\max} - a_{\min})\cdot\sigma(r), \tag{26}$$

so $a$ stays strictly inside $[a_{\min}, a_{\max}]$. For gross debt $b'$, subtract 4 from the raw
output first: $b' = b'_{\min} + (b'_{\max} - b'_{\min})\cdot\sigma(r - 4)$. Since
$\sigma(-4)\approx 0.018$, this biases the initial debt policy low, which stabilizes early
training before the bond-pricing function is learned. The feasibility bounds $[a_{\min},a_{\max}]$
for each control are given in Section[@sec:input-normalization-and-bounds].

### Training loop: policy iteration {#sec:training-loop-policy-iteration}

Training alternates policy evaluation and policy improvement. One **epoch** is 500 gradient
steps. Each step does the following in order.

**Step 1 --- sample.** Draw a mini-batch of $N = 8{,}192$ state-parameter pairs. Sample parameter
vectors uniformly from the current bounds. For each parameter vector, compute its
parameter-dependent state bounds (Section[@sec:input-normalization-and-bounds]) and sample $(\log z, k, b)$ uniformly within them.

**Step 2 --- policy evaluation.** Compute the Bellman residual at each sampled point,

$$R_i = V(z_i, k_i, b_i) - D_i - \tfrac{1}{1+r_f}\,\mathbb{E}_{z'\mid z_i}\big[\max\{V(z', k_i', b_i'-c_i'),\ 0\}\big], \tag{27}$$

with $D_i$ the dividend at the current state under the current policy and $k_i' = (1+i_i-\delta)k_i$. Update the value-network weights to reduce the mean squared residual,

$$L_V = \frac{1}{N}\sum_{i=1}^{N} R_i^2, \tag{28}$$

with one Adam step at learning rate $10^{-3}$. The gradient of $L_V$ flows through both the
$V(z_i,k_i,b_i)$ term on the left and the continuation $\mathbb{E}[\max\{V,0\}]$ term on the
right; the policy networks are held fixed in this step.

**Step 3 --- policy improvement.** Using the updated value network, update the three policy
networks to maximize the Bellman right-hand side. The loss is its negative,

$$L_\pi = -\frac{1}{N}\sum_{i=1}^{N}\Big(D_i + \tfrac{1}{1+r_f}\,\mathbb{E}_{z'\mid z_i}\big[\max\{V(z', k_i', b_i'-c_i'),\ 0\}\big]\Big), \tag{29}$$

with one Adam step at learning rate $10^{-3}$. The gradient flows through the policy outputs
$(i_i, b_i', c_i')$, which determine $D_i$ and the next-period states, while the value-network
weights are held fixed.

Then return to Step 1 with a fresh mini-batch. Learning rate decays multiplicatively by 1% per
epoch, floored at $10^{-6}$, for the value and policy networks (Section[@sec:hyperparameters-table-a2-consolidated]).

### Bond-pricing circularity: target network {#sec:bond-pricing-circularity-target-network}

$V$ and $q$ are jointly determined. The circularity is broken with a **target network**: a copy
of the value network whose weights are frozen for all gradient steps within an epoch and reset to
the current value-network weights at the start of the next epoch. The target network is used
**only** to compute the bond price $q$ (through the default probability). The continuation value
$\mathbb{E}[\max\{V,0\}]$ in the Bellman residual uses the **current** value network. This makes
$q$ move slowly relative to $V$, giving $V$ time to adjust before $q$ is recomputed.

**Default probability during training (analytic, no inner quadrature).** Treat
$V' = V(z', k', b'-c')$ as a function of the shock $\varepsilon$ (with $z' = e^{\rho\log z + \sigma\varepsilon}$) using the **target network**. Take a second-order Taylor expansion of
$V'(\varepsilon)$ around $\varepsilon = 0$:

$$V'(\varepsilon) \approx V'(0) + a\,\varepsilon + \tfrac{1}{2}c\,\varepsilon^2, \qquad a = \tfrac{dV'}{d\varepsilon}\Big|_0,\quad c = \tfrac{d^2 V'}{d\varepsilon^2}\Big|_0,$$

with $a$ and $c$ from automatic differentiation. The default region in $\varepsilon$ is the set
where this quadratic is negative. Solve $\tfrac{1}{2}c\,\varepsilon^2 + a\,\varepsilon + V'(0) = 0$
(discriminant $a^2 - 2c\,V'(0)$) and take the region from the curvature: with $c>0$ it is the open
interval between the two real roots, or empty (so $P_{\text{def}}=0$) when the roots are complex;
with $c<0$ it is the two tails outside the real roots, or all of $\mathbb{R}$ (so
$P_{\text{def}}=1$) when the roots are complex; with $c\approx 0$ treat $V'$ as linear and use the
half-line where $V'(0)+a\varepsilon<0$. Integrate the standard normal density over that set with
the standard normal CDF $\Phi$ to get $P_{\text{def}} = \Pr(V'<0)$ in closed form. The bond price
then multiplies this by the deterministic gate $g = \mathbf{1}\{R<b'k'\}$ (Section[@sec:debt-default-recovery-and-bond-pricing]). Use this
approximation only during network training; grid refinement (Section[@sec:grid-refinement]) computes $q$ exactly.

**Gradient management in policy improvement.** In Eq 29 the bond price enters the dividend as a
function of the controls, $q = q(k', b', c')$, through the recovery, the gate, and the default
probability. Let the policy gradient flow through $q$ with respect to the controls: the firm prices
its own default risk, so riskier choices fetch a lower $q$, and the policy must internalize this
exactly as the grid solver does when it prices each candidate control (Section[@sec:grid-refinement]); treating $q$ as
a detached constant would drop this price-impact channel and bias the policy toward over-issuing
risky debt. The only thing detached here is the target network's weights, which are frozen for the
epoch anyway; that freeze does not detach $q$'s dependence on the controls. Compute the
Taylor coefficients $a$ and $c$ with nested forward-mode automatic differentiation (forward
accumulators) rather than nested reverse-mode tapes, so this second derivative stays well-defined and
stable when the outer policy gradient differentiates through it.

### Conditional expectation: Gauss-Hermite quadrature {#sec:conditional-expectation-gauss-hermite-quadrature}

The conditional expectation in the Bellman equation is an integral against the standard normal:

$$\mathbb{E}_{z'\mid z}[f(z')] = \int_{-\infty}^{\infty} f\big(e^{\rho\log z + \sigma\varepsilon}\big)\frac{e^{-\varepsilon^2/2}}{\sqrt{2\pi}}\,d\varepsilon.$$

Gauss-Hermite quadrature with nodes $\{x_q\}$ and weights $\{w_q\}$ (defined for the weight
$e^{-x^2}$) approximates it after the substitution $\varepsilon = \sqrt{2}\,x$:

$$\mathbb{E}_{z'\mid z}[f(z')] \approx \frac{1}{\sqrt{\pi}}\sum_{q=1}^{Q} w_q\, f\big(e^{\rho\log z + \sigma\sqrt{2}\,x_q}\big). \tag{30}$$

Use $Q = 5$ nodes (exact for polynomial integrands up to degree $2Q-1 = 9$). This quadrature is
the expectation used during network training. (Grid refinement and simulation use the discrete
Tauchen chain instead; see Section[@sec:block-2-grid-refinement-simulation-and-moments].)

### Smooth approximation of non-differentiable payoffs {#sec:smooth-approximation-of-non-differentiable-payoffs}

The dividend (Eqs. 8 and 9) is non-differentiable in several places that block gradients during
training: the three step indicators $\mathbf{1}\{d_1<0\}$, $\mathbf{1}\{I>0\}$,
$\mathbf{1}\{d_2<0\}$; the kink $\max\{d_1,0\}$ carried into $d_2$; and the term
$d_1\cdot\mathbf{1}\{d_1<0\}$ (a minimum) in $D$. During training only, replace each with a smooth
version controlled by a temperature $\tau > 0$:

$$\mathbf{1}\{x<0\} \approx \sigma(-x/\tau), \tag{31}$$
$$\max(0,x) \approx \tau\log\big(1 + e^{x/\tau}\big), \tag{32}$$
$$\min(0,x) \approx -\tau\log\big(1 + e^{-x/\tau}\big). \tag{33}$$

Eq. 31 smooths the step indicators (for $\mathbf{1}\{I>0\}$ use the complement $\sigma(I/\tau)$);
Eq. 32 smooths $\max\{d_1,0\}$; Eq. 33 smooths $d_1\cdot\mathbf{1}\{d_1<0\}=\min\{d_1,0\}$; and the
issuance-fee magnitudes $|d_1|,|d_2|$ follow as $\max\{x,0\}-\min\{x,0\}$, which the two softplus
forms reproduce exactly. The Bellman continuation $\max\{V',0\}$ is left in its exact form (its
subgradient is enough for training), and the default indicator $\mathbf{1}\{V'<0\}$ is smoothed
separately through the default probability $P_{\text{def}}$ (Section[@sec:bond-pricing-circularity-target-network]), not here.

As $\tau \to 0$ each converges pointwise to the exact function; the smooth indicator transitions
over a region of width about $4\tau$ around $x=0$, so the gradient is nonzero near the kink. Set
$\tau = 10^{-3}$. Accuracy of the final solution is not very sensitive to $\tau$ because grid
refinement uses the exact, non-smooth dividend.

## Block 2: grid refinement, simulation, and moments {#sec:block-2-grid-refinement-simulation-and-moments}

For each sampled parameter vector: refine the network solution on a grid (Section[@sec:grid-refinement]), simulate
a panel (Section[@sec:panel-simulation]), and compute the moments (Section[@sec:moment-construction]). Output one (parameter, moment) row.

### Grid refinement {#sec:grid-refinement}

This step removes the dependence on the smooth approximations and makes the solution as accurate
as value function iteration (VFI). It uses the exact, non-smooth dividend.

**Grid sizes:**

| Dimension                          | Points |
|:-----------------------------------|:-------|
| Productivity $z$ (state)           | 11     |
| Capital $k$ (state)                | 15     |
| Net debt $b$ (state)               | 35     |
| Next-period capital $k'$ (control) | 81     |
| Gross debt $b'$ (control)          | 91     |
| Cash $c'$ (control)                | 71     |

The grids span the state and control bounds in Section[@sec:input-normalization-and-bounds]. The capital grids ($k$ and the $k'$
control grid) are log-spaced, with more points at low capital where the value function curves most.
The debt and cash grids ($b$, $b'$, $c'$) are uniform, since the net-debt range spans negative and
positive values (so it cannot be log-spaced) and the gross-debt and cash ranges include zero. The
11 productivity points are the Tauchen nodes of the AR(1).

This spacing applies only to the Block 2 refinement grid. Network training (Section[@sec:training-loop-policy-iteration]) instead
samples states uniformly in $(\log z, k, b)$ to cover the whole state-parameter space evenly.

**Procedure for one parameter vector:**

1.  **Initialize from the networks.** Evaluate $V$ and the three policies on the full grid. This
    gives a grid-based value function and policy.

2.  **Policy iteration**, each round consisting of:

    - **(a) Policy improvement.** At each state grid point, search for the control combination
      $(k', b', c')$ that maximizes $D + \tfrac{1}{1+r_f}\,\mathbb{E}_{z'\mid z}[\max\{V', 0\}]$ using the
      exact dividend. Do not search the entire control grid. Build a candidate set per control
      dimension from: the current policy at the grid point, the policies at the immediate
      neighbors in each of the $(z,k,b)$ dimensions, and small $\pm 1$ grid-index perturbations
      around the current policy. This gives 9 candidate values per control dimension, hence
      $9^3 = 729$ combinations to evaluate at each grid point. The network starting point makes
      this local search sufficient.

    - **(b) Policy evaluation.** Given the improved policy, solve for $V$ satisfying the Bellman
      equation under that fixed policy. Because of $\max\{V,0\}$ this is a nonlinear fixed point.
      Solve it with a semismooth Newton method: at each iteration partition grid points by the
      sign of $V$ (set $\max\{V,0\}=V$ where $V>0$ and $0$ where $V\le 0$); with the partition
      fixed the Bellman equation is linear in $V$ and is solved with GMRES (generalized minimal
      residual). Recompute the partition from the new solution and repeat until it stabilizes.

3.  **Recompute bond prices.** With the updated $V$, recompute $q(z,k',c',b')$ exactly on the full
    grid (Eq. 6). This uses the same two-condition default indicator as training,
    $\mathbf{1}\{V'<0\}\cdot\mathbf{1}\{R<b'k'\}$ (Section[@sec:debt-default-recovery-and-bond-pricing]); the only difference is that
    $\Pr(V'<0)$ is evaluated exactly over the Tauchen nodes rather than by the training-time Taylor
    step.

4.  **Repeat.** Return to step 2 with updated bond prices.

Run **six rounds** of steps 2 through 4. Expectations over $z'$ on the grid use the 11-state
Tauchen transition matrix. DF26 does not give the inner-solve tolerances, so use these tight defaults, matching the $10^{-10}$
that DF26 uses for the Levenberg-Marquardt estimation. Semismooth Newton (policy evaluation, step
2b): stop when the active-set partition is unchanged between two consecutive iterations, that is,
when no grid point flips the sign of $V$; as a safeguard, also stop when
$\|V^{j+1}-V^j\|_\infty / (1+\|V^j\|_\infty) < 10^{-10}$, and cap it at 50 iterations. GMRES (the
linear solve inside each Newton step): relative residual $\|b-Ax\|_2/\|b\|_2 \le 10^{-10}$, restart
every 50 Krylov vectors, at most 1000 matrix-vector products, warm-started from the current $V$. The
six policy-iteration rounds are a fixed count, not a convergence loop.

### Panel simulation {#sec:panel-simulation}

For each parameter vector, simulate $N_f = 5{,}000$ firms over $T = 300$ periods and discard the
first $T_0 = 200$ as burn-in. Initialize each firm at $t = 0$ by drawing $z$ from the Tauchen
stationary distribution and $(k, b)$ from random grid points; the burn-in removes any dependence on
this start.

- Productivity evolves on the 11-state Tauchen Markov chain built from Eq. 2. No interpolation is
  needed in $z$ (look up the policy at the current $z$ node).

- For $(k, b)$, which are continuous and generally off-grid, interpolate the grid-based policies
  with **bilinear interpolation** over the four nearest $(k,b)$ grid points (weights proportional
  to proximity).

- Capital updates as $k' = (1+i-\delta)k$. The next-period net-debt state is $b^{\text{gross}\prime} - c'$, where $(i, b^{\text{gross}\prime}, c')$ are the interpolated policy outputs.

- **Default and reseeding.** When a firm's continuation value $V(z,k,b)$ falls below zero, the
  firm walks away (limited liability) and exits. Replace it by drawing a new $(k,b)$ from random grid points while keeping the
  current $z$ (productivity is exogenous and does not reset on default). This keeps the panel at
  $N_f$ active firms and makes the cross-section reflect the ergodic distribution after burn-in.

### Moment construction {#sec:moment-construction}

After burn-in, form the panel as a sequence of consecutive period pairs. For each pair, denote
current-period variables without primes and next-period variables with primes. Define:

$$\text{Investment rate:}\quad i = (K' - (1-\delta)K)/K, \tag{34}$$
$$\text{Operating income:}\quad inc = (z A_\pi K^{\xi} - c_f)/(K+C), \tag{35}$$
$$\text{Debt ratio:}\quad d = B^{\text{gross}\prime}/(K'+C'), \tag{36}$$
$$\text{Cash ratio:}\quad cash = C'/(K'+C'), \tag{37}$$
$$\text{Cash saving:}\quad \Delta c = (C' - C)/(K'+C'), \tag{38}$$
$$\text{Net debt:}\quad net = B/(K+C). \tag{39}$$

All moments are computed over observations where the firm is not in default in either period of
the pair, that is $V > 0$ and $V' > 0$.

**The 11 targeted moments (in this exact order):**

1.  Mean investment rate.

2.  Standard deviation of investment rate.

3.  Mean operating income.

4.  Standard deviation of operating income.

5.  Autocorrelation of operating income: the OLS slope from regressing next-period $inc'$ on
    current-period $inc$, both centered by their sample means over good observations.

6.  Mean debt ratio.

7.  Standard deviation of debt ratio.

8.  Mean cash ratio.

9.  Standard deviation of cash ratio.

10. OLS slope on net debt in the cash-saving regression: a multivariate OLS of cash saving
    $\Delta c$ on net debt $net$ and operating income $inc$, all three centered by their means.

11. OLS slope on operating income in that same cash-saving regression.

The debt, cash, and cash-saving ratios (Eqs. 36-38) use next-period quantities over next-period
total assets $K'+C'$, following the appendix; the investment, operating-income, and net-debt
measures (Eqs. 34, 35, 39) use current-period quantities, following the main text. The debt ratio
uses gross debt, matching the data item DLC+DLTT scaled by total assets.

## Block 3: moment-surrogate networks and estimation {#sec:block-3-moment-surrogate-networks-and-estimation}

### Moment-surrogate networks {#sec:moment-surrogate-networks}

Train one network per moment ($M = 11$). Each is a feedforward network with
**3 hidden layers of 32 units**, SiLU activation. Input is the normalized parameter vector (8-dim,
normalized to $[-1,1]$ with the current bounds). Output is one scalar (the predicted
moment).

Learn $g_j(\boldsymbol{\beta}) \approx \mathbb{E}[m_j \mid \boldsymbol{\beta}]$ by minimizing the
mean squared prediction error on the simulated dataset.

### Cross-validation {#sec:cross-validation}

Use $K = 10$-fold cross-validation. Partition the dataset into 10 roughly equal folds. For each
moment and each fold, train a network on the other 9 folds and hold out the remaining fold for
validation. This produces $M \times K = 110$ networks, a per-moment out-of-sample $R^2$ (monitored
during training), and 10 parameter estimates per moment whose spread measures sensitivity to the
surrogate approximation.

### Training the surrogate networks {#sec:training-the-surrogate-networks}

Train all 110 networks simultaneously and continually: each epoch warm-starts from the current
weights and runs **200 SGD passes** over the accumulated dataset (Adam, learning rate $10^{-4}$,
batch size 256), rather than re-initializing. Cap the dataset at 10,000 observations, keeping the
10,000 most recent. Store each row as the raw parameter vector with its moments, and normalize the
parameters to $[@sec:moment-surrogate-networks]), so a bound change is
absorbed by renormalization and no stored normalization can go stale. The per-network loss is the
mean squared prediction error over that network's training fold:

$$L_{j,k} = \frac{1}{|T_k|}\sum_{i\in T_k}\big(m_{i,j} - g_j(\boldsymbol{\beta}_i)\big)^2. \tag{40}$$

No learning-rate decay for the surrogate networks.

### Estimation objective {#sec:estimation-objective}

For each fold $k$, minimize the weighted distance between data moments $\hat{m}$ and the fold's
surrogate predictions $m_k(\boldsymbol{\beta})$:

$$\hat{\boldsymbol{\beta}}_k = \arg\min_{\boldsymbol{\beta}}\ \big(\hat{m} - m_k(\boldsymbol{\beta})\big)' W \big(\hat{m} - m_k(\boldsymbol{\beta})\big), \tag{41}$$

where $m_k$ stacks the 11 predictions from fold $k$ and $W$ is the weighting matrix
(Section[@sec:weighting-matrix]).

### Levenberg-Marquardt with analytical Jacobian {#sec:levenberg-marquardt-with-analytical-jacobian}

Solve Eq. 41 with Levenberg-Marquardt (LM), using the Jacobian $J = \partial m_k/\partial \boldsymbol{\beta}$ (an $11 \times 8$ matrix). The Jacobian is computed exactly by automatic
differentiation through the surrogate networks. Convergence tolerances are
$10^{-10}$ for function value, step size, and gradient norm, with at most 20 iterations per run.

### Multiple restarts and bound enforcement {#sec:multiple-restarts-and-bound-enforcement}

Run LM from **30 random starting points** per fold and keep the lowest-objective solution. Each
start is drawn from a standard normal in the unconstrained space (below). With $K = 10$ folds and
30 restarts, the estimation step runs 300 LM optimizations.

Enforce bounds by reparametrizing each parameter through a sigmoid:

$$\beta^j = \underline{\beta}^j + (\overline{\beta}^j - \underline{\beta}^j)\cdot\sigma(x^j), \tag{42}$$

with $x^j \in \mathbb{R}$. LM searches over $x = (x^1, \dots, x^8)$. The Jacobian with respect to
$x$ follows from the chain rule, $\partial m/\partial x = (\partial m/\partial \boldsymbol{\beta}) (\partial \boldsymbol{\beta}/\partial x)$, with $\partial \beta^j/\partial x^j = (\overline{\beta}^j - \underline{\beta}^j)\,\sigma(x^j)(1-\sigma(x^j))$. Note the surrogate input is itself the
normalized parameter (Eq. 24), so the full Jacobian also carries the constant normalization
derivative; let automatic differentiation handle the whole chain.

### Weighting matrix {#sec:weighting-matrix}

$W$ is the inverse of the moment variance-covariance matrix, estimated from the data with
influence functions following Erickson and Whited (2002), clustered at the firm level.

For observation $i$ and moment $j$, the influence function $\psi_{ij}$ captures observation $i$'s
contribution to the sampling variability of moment $j$:

- **Means:** $\psi_{ij} = x_{ij} - \hat{\mu}_j$.

- **Standard deviations:** $\psi_{ij} = \big((x_{ij} - \hat{\mu}_j)^2 - \hat{\sigma}_j^2\big)/(2\hat{\sigma}_j)$.

- **Autocorrelation:** $\psi_i = \tilde{x}_i \tilde{\varepsilon}_i / \hat{Q}$, where $\tilde{x}_i$ is the centered lagged profitability, $\tilde{\varepsilon}_i = \tilde{y}_i - \hat{\rho}\tilde{x}_i$ is the OLS residual, and $\hat{Q} = \tfrac{1}{N_\rho}\sum \tilde{x}_i^2$.

- **Regression coefficients:** $\psi_i = \big(\tfrac{1}{N_{\text{reg}}} X'X\big)^{-1} x_i \varepsilon_i$, where $x_i$ is the $2\times 1$ vector of centered regressors and $\varepsilon_i$ is the OLS residual.

Observations that do not contribute to a moment (for example, those without a valid lag for the
autocorrelation) get $\psi_{ij} = 0$.

Stack the influence functions into an $N \times 11$ matrix $\Psi$. To handle within-firm serial
correlation, sum the influence functions across observations belonging to each firm $g$, giving a
$G \times 11$ matrix of firm-level totals. The cluster-robust covariance is

$$\hat{\Sigma}_{jl} = \frac{1}{N_j N_l}\sum_{g=1}^{G}\Big(\sum_{i\in g}\psi_{ij}\Big)\Big(\sum_{i\in g}\psi_{il}\Big), \tag{43}$$

where $N_j, N_l$ are the effective sample sizes for moments $j, l$ (they differ because moments
needing lags use fewer points). Then $W = \hat{\Sigma}^{-1}$.

In practice work with the Cholesky factor $W^{1/2}$ such that $W = (W^{1/2})' W^{1/2}$, and
normalize it by its median column norm. This does not change the estimates but improves
conditioning.

Reproducing $W$ exactly requires the Compustat micro data (Section[@sec:application-inputs-external-values-bounds-targets-reference-estimates]; see Section[@sec:resolved-design-decisions]).

## Minimum loss functions, adaptive shrinkage, and identification {#sec:minimum-loss-functions-adaptive-shrinkage-and-identification}

### Minimum loss functions {#sec:minimum-loss-functions}

For each parameter $\beta^j$, the minimum loss function measures the best achievable fit when
$\beta^j$ is held at a value and all other parameters adjust optimally:

$$L(\beta^j) = \min_{\boldsymbol{\beta}^{-j}}\ \big(\hat{m} - m(\beta^j, \boldsymbol{\beta}^{-j})\big)' W \big(\hat{m} - m(\beta^j, \boldsymbol{\beta}^{-j})\big), \tag{44}$$

where $\boldsymbol{\beta}^{-j}$ are all parameters other than $\beta^j$. Evaluate it at **31 evenly
spaced values** of $\beta^j$ across its current bounds. At each grid point run the full LM
estimation (30 restarts, $K = 10$ folds) over $\boldsymbol{\beta}^{-j}$ and take the **median**
across folds. Also record, at each grid point, the standard deviation across the 10 folds; report
bands of $\pm 2$ pointwise standard deviations whenever displaying these curves.

### Adaptive bound shrinkage {#sec:adaptive-bound-shrinkage}

Use the minimum loss functions to narrow bounds during training, concentrating computation near
the estimates. Begin only after a 200-epoch warm-up. After that, attempt a shrink at the end of
every epoch.

**Per-parameter level rule.** For each $\beta^j$, find $\beta^{j*}$ that minimizes $L(\beta^j)$.
Given a tolerance $\Delta > 0$, set the new interval to the contiguous region around
$\beta^{j*}$ where $L(\beta^j) \le L(\beta^{j*}) + \Delta$. Apply this to each of the 10 folds'
minimum loss functions, and for each parameter take the interval spanning the lowest and highest
endpoints across folds.

**Identification guard (prevents shrinking weakly identified parameters).** Before applying the
level rule to a parameter, compare $L(\beta^{j*})$ to the 90th-percentile loss. If the
90th-percentile loss is not at least **three standard deviations** above the minimum, do not
shrink that parameter. The standard deviation is the median across grid points of the pointwise
across-fold standard deviation; before computing it, recenter each fold's curve around its own
minimum (level differences across folds do not affect which $\beta^j$ minimizes the loss).

**Containment guard (keeps plausible parameters).** A candidate region is admissible only if it
still contains both of: (1) among the 500 most recent sampled parameters, the 50 whose simulated
moments are closest to the targets in the same weighted-GMM norm used to estimate
$\boldsymbol{\beta}$; and (2) every LM estimate produced in the current estimation step (all 30
restarts and 10 folds). If the candidate region would exclude any vector in these two sets, it
fails.

**Choosing the tolerance via volume.** Do not pick $\Delta$ directly; instead pick the **target
volume fraction** $v$: the new region's volume (the
product of interval widths over parameters that pass the identification guard) as a fraction of
the pre-shrink volume. For each candidate $v$, find the $\Delta$ that gives that volume. Start at
$v = 0.80$ (a 20% volume cut) and search for the smallest $v \in [0.05, 1.0]$ that satisfies the
containment guard. If no $v$ in that range satisfies it, leave the bounds unchanged this round.
After shrinking, all subsequent operations use the narrower bounds. Recompute the parameter and
state normalizations from the new bounds so inputs stay in $[-1,1]$; do not re-initialize the value,
policy, or surrogate networks, since continuous training and the per-epoch surrogate updates let
them re-adapt to the renormalized inputs within a few epochs (one reason the shrink is gradual).

### Global identification diagnostic {#sec:global-identification-diagnostic}

To separate weak identification from misspecification, compute the minimum loss functions using
**simulated moments at the estimated parameters** as the target rather than the data moments.
Simulate at $\hat{\boldsymbol{\beta}}$ (Section[@sec:block-2-grid-refinement-simulation-and-moments] process) to get model-implied moments
$\tilde{m}$, then compute

$$L(\beta^j) = \min_{\boldsymbol{\beta}^{-j}}\ \big(\tilde{m} - g(\beta^j, \boldsymbol{\beta}^{-j})\big)' W \big(\tilde{m} - g(\beta^j, \boldsymbol{\beta}^{-j})\big), \tag{45}$$

over 31 grid points of $\beta^j$, with 30 LM restarts over $\boldsymbol{\beta}^{-j}$ and 10 folds.
The diagnostic is twofold: check whether each $L(\beta^j)$ has a unique global minimum (a sharp
minimum means identified; a flat curve means weakly identified), and check whether this second
estimation recovers the parameter vector used to generate $\tilde{m}$.

## Asynchronous execution architecture {#sec:asynchronous-execution-architecture}

This concurrency is the source of the method's speed and is part of the design, not an optional
optimization. Two kinds of process run at the same time on four GPUs and share the network weights.

- **GPU 1 (trainer).** Runs the Block 1 value/policy training loop continuously, cycling through
  epochs of 500 gradient steps. It holds the value- and policy-network weights in `tf.Variable`s
  and updates them in place at every gradient step.

- **GPUs 2 to 4 (collectors).** Each one, at the start of each parameter batch, reads a snapshot of
  the trainer's current weights, then draws parameter vectors from the current bounds, evaluates the
  snapshot networks on the grid, refines the grid solution (Section[@sec:grid-refinement]), simulates panels (Section
 [@sec:panel-simulation]), computes moments (Section[@sec:moment-construction]), and appends (parameter, moment) rows to the shared dataset.

No GPU waits for any other. Because the trainer keeps improving the weights, each collector batch
starts from a better solution than the last, so the dataset improves over time. The grid refinement
(Section[@sec:grid-refinement]) corrects whatever the snapshot network gives, so the moments are accurate at every
stage and only get cheaper to compute as training proceeds; the 10,000-observation cap (Section[@sec:training-the-surrogate-networks])
keeps the surrogates trained on the most recent, best collections.

At the end of every epoch the controller retrains the moment-surrogate networks on the current
dataset, evaluates their held-out $R^2$, runs the Levenberg-Marquardt estimation for updated
estimates, and computes the minimum loss functions. After the 200-epoch warm-up it also attempts a
bound shrink, often declining because of the identification and containment guards (Section[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification]).

Do not serialize this into a train-then-collect alternation between Block 1 and Blocks 2 to 4: that
removes the concurrency the method relies on for its speed. A single-process serial mode is
acceptable only as a debugging aid on one GPU; it runs the same math far more slowly. See Section[@sec:tensorflow-tfp-implementation-notes]
for the TF snapshot mechanism.

## Hyperparameters {#sec:hyperparameters-table-a2-consolidated}

**Value and policy networks**

| Setting                  | Value                                          |
|:-------------------------|:-----------------------------------------------|
| Hidden layers            | 3                                              |
| Units per layer          | 128                                            |
| Activation               | SiLU                                           |
| FiLM generator           | 1 layer, 32 units                              |
| Optimizer                | Adam                                           |
| Learning rate            | $10^{-3}$, decay 1% per epoch, floor $10^{-6}$ |
| Mini-batch size          | 8,192                                          |
| Gradient steps per epoch | 500                                            |

**Moment-surrogate networks**

| Setting                    | Value                    |
|:---------------------------|:-------------------------|
| Hidden layers              | 3                        |
| Units per layer            | 32                       |
| Activation                 | SiLU                     |
| Optimizer                  | Adam                     |
| Learning rate              | $10^{-4}$ (no decay)     |
| Mini-batch size            | 256                      |
| SGD passes per update      | 200                      |
| Cross-validation folds $K$ | 10                       |
| Max training observations  | 10,000 (use most recent) |

**Quadrature and grid**

| Setting | Value |
|:---|:---|
| Gauss-Hermite nodes $Q$ | 5 |
| State grid $n_z \times n_k \times n_b$ | $11 \times 15 \times 35$ |
| Control grid $n_{k'} \times n_{b'} \times n_{c'}$ | $81 \times 91 \times 71$ |
| Tauchen standard deviations $m$ | 2.5 |

**Estimation**

| Setting                    | Value                                 |
|:---------------------------|:--------------------------------------|
| Optimizer                  | Levenberg-Marquardt                   |
| Restarts per fold          | 30                                    |
| Convergence tolerance      | $10^{-10}$ (function, step, gradient) |
| Max iterations per restart | 20                                    |

**Simulation**

| Setting       | Value |
|:--------------|:------|
| Firms $N_f$   | 5,000 |
| Periods $T$   | 300   |
| Burn-in $T_0$ | 200   |

**Minimum loss functions and shrinkage**

| Setting | Value |
|:---|:---|
| Grid points per parameter | 31 |
| Target volume fraction (initial) | 0.80 |
| Warm-up before shrinking | 200 epochs |
| Identification guard threshold | 90th-pct loss must exceed minimum by 3 SD |
| Containment set 1 | 50 closest of 500 most recent sampled parameters |
| Containment set 2 | all 30 restarts $\times$ 10 folds of current LM estimates |

**Smoothing**

| Setting                            | Value     |
|:-----------------------------------|:----------|
| Temperature $\tau$ (Eqs. 31 to 33) | $10^{-3}$ |

## Application inputs: external values, bounds, targets, reference estimates {#sec:application-inputs-external-values-bounds-targets-reference-estimates}

**External parameter values** (Section[@sec:parameters]): $r_f = 0.02$, $\alpha = 0.30$, $\tau = 0.20$,
$\lambda_0 = 0.007$, $\lambda_1 = 0.054$, $r_c = 0$ (so $\iota_c = 1$). The wage is normalized to
$w = 1$ (Section[@sec:technology-and-production]).

**Initial parameter bounds (Table A1).** These are the starting bounds; they narrow during
training. Bound by name (not by the table's print order).

| Parameter                     | Lower | Upper |
|:------------------------------|:------|:------|
| $\theta$ (returns to scale)   | 0.60  | 0.82  |
| $\rho$ (autocorrelation)      | 0.50  | 0.80  |
| $\sigma$ (productivity SD)    | 0.05  | 0.20  |
| $\delta$ (depreciation)       | 0.05  | 0.20  |
| $\gamma_1$ (convex adj. cost) | 0.05  | 1.00  |
| $\gamma_0$ (fixed adj. cost)  | 0.001 | 0.20  |
| $\chi$ (recovery rate)        | 0.001 | 0.90  |
| $c_f$ (fixed operating cost)  | 0.001 | 0.25  |

**Data construction (DF26 Section 3.2.1).** Compustat annual, 1970 to 2019. Keep U.S.-headquartered
firms. Exclude financials (SIC 6000 to 6999), regulated (SIC 4900 to 4999), and quasi-governmental
(SIC 9000+). Drop observations with any missing variable, total assets below \$10 million,
negative sales, or sales or asset growth above 200%. Require at least four consecutive annual
observations per firm. Variable mapping: cash = CH, debt = DLC+DLTT, operating income = OIBDP,
investment = CAPX, each scaled by total assets (AT). Winsorize all ratios at the 1st and 99th
percentiles by year. Because model firms are ex-ante homogeneous, remove firm fixed effects from
the standard-deviation and autocorrelation moments by demeaning at the firm level and adding back
the sample average.

**Data-moment targets $\hat{m}$ (Table 1, Panel B).** These are the values DF26 used. The processing
recipe is fully specified (this section and Section[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification]), so reproducing them exactly requires only
the same Compustat micro data; where that data is unavailable, use these values directly as the
targets. Order matches Section[@sec:moment-construction].

| \#  | Moment                                 | Data target |
|:----|:---------------------------------------|:------------|
| 1   | Mean investment rate                   | 0.065       |
| 2   | SD investment rate                     | 0.045       |
| 3   | Mean operating income                  | 0.095       |
| 4   | SD operating income                    | 0.100       |
| 5   | Serial correlation of operating income | 0.495       |
| 6   | Mean debt ratio                        | 0.271       |
| 7   | SD debt ratio                          | 0.148       |
| 8   | Mean cash ratio                        | 0.096       |
| 9   | SD cash ratio                          | 0.080       |
| 10  | Coefficient of cash on net debt        | 0.094       |
| 11  | Coefficient of cash on income          | 0.039       |

**Reference estimates (for validation only; not inputs).** These let you check that the pipeline
lands in the right region. Standard errors in parentheses.

From Table 1, Panel A:
$\theta = 0.796\,(0.002)$, $\rho = 0.597\,(0.007)$, $\sigma = 0.187\,(0.002)$,
$\delta = 0.066\,(0.000)$, $\gamma_1 = 0.945\,(0.014)$, $\gamma_0 = 0.014\,(0.000)$,
$\chi = 0.003\,(0.024)$, $c_f = 0.028\,(0.001)$. The recovery rate $\chi$ is weakly identified.

## Seeding and reproducibility {#sec:seeding-and-reproducibility}

Goal: given identical inputs and specifications, every random draw in the pipeline is reproducible.
The grid refinement (Section[@sec:block-2-grid-refinement-simulation-and-moments]) is already deterministic. Randomness enters in five seedable places
and two that no seed can control.

Seedable sources:

1.  Weight initialization, for the value and policy networks, the twelve FiLM generators, and the 110
    surrogate networks.

2.  Training-state sampling: the uniform draws of states and parameters in each Block 1 batch, and the
    collector parameter draws in Block 2.

3.  Simulation: the Tauchen chain transitions, the initial $(z, k, b)$, and the reseed after default.

4.  Surrogate training: mini-batch shuffling and the ten cross-validation fold assignments.

5.  LM restarts: the thirty standard-normal starting points per fold.

Not seedable:

1.  Hardware nondeterminism: nondeterministic GPU kernels and reduction order, which differ run to run
    and across machines at the floating-point level.

2.  Async timing: collectors read weight snapshots at varying wall-clock times, so the dataset
    composition depends on timing, not on any seed.

Design. Use stateless RNG everywhere (`tf.random.stateless_*`), never the stateful global generator.
A stateless op is a pure function of an explicit seed and a shape, so its output is identical
regardless of call order, parallelism, or device. That is what makes the draws reproducible in a
batched, multi-GPU, async pipeline, where a stateful counter would shift the whole stream whenever ops
are reordered or run on more threads.

Use one master seed (an integer) per run. Derive every draw's seed from it by folding in a purpose
label and the natural indices, with `tf.random.experimental.stateless_split` and `stateless_fold_in`,
then pass the derived seed to the stateless op. Key each substream by what indexes it:

- init: `(master, "init", layer_id)`

- training samples: `(master, "train", epoch, step)`

- collector parameters: `(master, "collect", collector_id, batch)`

- simulation: `(master, "sim", firm_id, period)`, with separate labels for the initial state and the
  reseed

- surrogate: `(master, "surrogate", epoch, step)` for shuffling, `(master, "folds")` for the split

- LM: `(master, "lm", fold, restart)`

Each draw is then a pure function of `(master, purpose, indices)`, so a rerun reproduces it, and
distinct substreams are independent without reasoning about a shared counter. Keying the simulation by
`(firm, period)` makes the panel identical whether firms are simulated serially, batched, or across
devices.

Reproducibility, in two tiers. Sources 1 to 5 are pinned exactly by the keyed stateless seeds: given
the same master seed, the same inputs, and the same specification, the draws are identical on any
hardware. Sources 6 and 7 are not. The full async production run is therefore not bit-reproducible;
the seeds fix its draws but not the float math or the dataset timing. Exact numerical reproducibility
is reserved for the regression slices (Section[@sec:regression-tests]), which run single-GPU and serial with op-level
determinism on (`tf.config.experimental.enable_op_determinism()` and single-threaded intra-op), on a
fixed machine with pinned library versions. That combination makes the float math repeatable too, so
the golden comparison can use a tight tolerance.

Caveat. Keras initializers read the global stateful generator by default. For the deterministic
training slice, either pass explicit seeds to the initializers or generate the weights with stateless
ops and assign them, so initialization is reproducible as well.

## TensorFlow implementation notes {#sec:tensorflow-tfp-implementation-notes}

DF26 is JAX. Below are the points where the TF/TFP port needs care. No code is given; these are
design directions for the building agent.

**Automatic differentiation.** Use `tf.GradientTape` for all gradients. The Bellman residual loss
(Eqs. 27 to 29) needs gradients of $V$ with respect to its own weights through both the left term
and the continuation term; a single tape covers this. The default-probability quadratic
approximation (Section[@sec:bond-pricing-circularity-target-network]) needs first and second derivatives of $V'$ with respect to the scalar
shock $\varepsilon$; use nested tapes (an inner tape for the first derivative, an outer tape for
the second) or `tape.gradient` applied twice. The estimation Jacobian $\partial m/\partial x$
(Section[@sec:levenberg-marquardt-with-analytical-jacobian]) is best obtained with `GradientTape.jacobian` (or `batch_jacobian` to vectorize over
folds and restarts) through the surrogate networks plus the normalization and bound-reparametrization chain.

**Optimizers.** Use `tf.keras.optimizers.Adam`. For the value/policy learning-rate schedule
(multiply by 0.99 each epoch, floor $10^{-6}$), drive the optimizer's learning rate with a custom
schedule or update it at each epoch boundary; do not use a fixed decay-steps schedule that ignores
the floor. Moment networks use a constant $10^{-4}$.

**Gauss-Hermite nodes.** Computing the 5 fixed nodes and weights once at startup is the one place
NumPy is clearly the right tool (`numpy.polynomial.hermite.hermgauss`). Convert the result to a
`tf.constant` and use it natively in the quadrature sum (Eq. 30). TFP has no direct GH helper.

**Standard normal CDF.** Use `tfp.distributions.Normal(0, 1).cdf` for the default-probability
integration in Section[@sec:bond-pricing-circularity-target-network] and for the Tauchen transition matrix.

**Tauchen discretization.** Build the 11-state chain from each parameter vector's $(\rho, \sigma)$
with the $m = 2.5$ coverage. Because the chain differs per parameter vector during simulation,
vectorize the construction over the batch of parameter vectors using TFP's `Normal.cdf` and TF
array ops. Avoid Python loops over the batch.

**Bilinear interpolation on the (k, b) grid.** The simulation interpolates the grid policies over
$(k, b)$ with bilinear interpolation. Since $k$ is log-spaced and $b$ is uniform, map $k$ to
$\log k$ so both axes are regular, then use `tfp.math.batch_interp_regular_nd_grid`; or implement
gather-based bilinear interpolation directly.

**Levenberg-Marquardt.** TFP has BFGS, L-BFGS, and Nelder-Mead, but no LM. Implement LM natively
in TF: build the damped normal equations $(J' W J + \mu I)\,\Delta = J' W r$ using the analytical
Jacobian from `GradientTape`, solve the linear system with `tf.linalg.solve`, and run the standard
trust-region damping update on $\mu$. Vectorize across the 10 folds and 30 restarts (300
independent small problems) with batched tensors and `tf.linalg.solve` on a leading batch dim.

**Grid-refinement linear solve (policy evaluation).** The active-set partition turns the Bellman
fixed point into a linear system $(I - \tfrac{1}{1+r_f} A)V = D$, where $A$ encodes the policy-implied
next-state interpolation and the Tauchen $z'$-transition. DF26 uses GMRES. The state grid has
$11 \times 15 \times 35 = 5{,}775$ unknowns, so a dense `tf.linalg.solve` is feasible on a GPU and
is the simplest faithful option. If a matrix-free iterative solver is preferred, a GMRES routine
must be implemented (TF has `tf.linalg.experimental.conjugate_gradient` for symmetric positive
definite systems only, which does not fit a nonsymmetric operator). Wrap the semismooth Newton
active-set loop and the six refinement rounds as TF control flow (`tf.while_loop`), and vectorize
over the batch of parameter vectors processed together.

**Cross-validated ensemble of 110 networks.** Train all moment networks together by carrying
leading batch dimensions on the weight tensors (one set of `[``110, ...``]` weights) and using batched
matrix multiplies, or by `tf.vectorized_map` over the ensemble. Keep them on one device since they
are tiny.

**FiLM layers.** Default to computing the FiLM math explicitly (compute each generator's
$(\gamma_l, \delta_l)$ from $\beta$, then apply Eq. 25 in each trunk layer), with all weights held
in a `tf.Module` per network so the trunk and its three generators are created once and
`trainable_variables` collects them for the tape and optimizer. This keeps the flow unified and
avoids hand-managing variables. A subclassed Keras layer also works but adds `build()` and
call-timing complexity and does not cleanly hold the generator-to-trunk coupling, so it is the
fallback, not the default. Either way, never create variables inside a `tf.function`; create them
once in the module and wrap only the step. See Section[@sec:architecture-film-conditioning] for the generator activation and
initialization.

Pitfalls that fail silently (issue: fix):

- Identity init: start $\gamma_l = 1$, $\delta_l = 0$ (near-zero generator output weights, $\gamma$
  bias 1, $\delta$ bias 0). Skipping it distorts the trunk from step one and can diverge.

- Modulation placement: apply $(\gamma_l, \delta_l)$ to the pre-activation $W_l h_l + c_l$ before
  SiLU, exactly as Eq. 25. Applying it after the activation changes the model with no error.

- Per-sample shapes: $\gamma_l, \delta_l$ are $[N, 128]$ and the modulation is element-wise
  $[N,128] \times [N,128]$. Do not collapse $\beta$ to one $(\gamma, \delta)$ for the whole batch,
  which would make every firm share the same modulation.

- Separate generators: keep 12 distinct generators (3 per network, 4 networks). Sharing generator
  weights across layers loses the depth-varying conditioning.

- Exploding or dead units: $\gamma$ scales the pre-activation, so a large $\gamma$ blows activations
  up and $\gamma \approx 0$ kills units. Identity init plus the standard learning rate controls
  this; watch for dead units.

**Multi-GPU asynchrony.** TF supports explicit device placement (`tf.device`) and `tf.distribute`.
In the 1-trainer-plus-3-collectors pattern (Section[@sec:asynchronous-execution-architecture]), the trainer holds the value- and
policy-network weights in `tf.Variable`s on GPU 1 and updates them in place; each collector reads a
snapshot of those weights at the start of every parameter batch, for example with
`tf.Variable.read_value()` into local constants (single process) or a periodic weight copy over a
producer-consumer channel (separate processes). Collectors append rows to a shared,
concurrency-safe dataset buffer that the controller reads each epoch.

**Graph compilation.** Wrap the hot paths (training step, grid refinement, simulation step) in
`tf.function` for XLA-style compilation, mirroring JAX's JIT. Watch for retracing: keep shapes
static, pass the parameter bounds and grids as tensors (not Python scalars that change), and avoid
data-dependent Python control flow inside compiled functions.

**Random number handling.** Use stateless `tf.random.stateless_*` with per-purpose keyed seeds,
never the stateful global generator; the seed design is in Section[@sec:seeding-and-reproducibility]. The TF analogue of the JAX
key-reuse pitfall is reusing one stateless seed across vectorized draws, so give each firm, period,
and parameter vector its own seed path.

### Resolved design decisions {#sec:resolved-design-decisions}

These are implementation decisions this spec makes where DF26 is silent, ambiguous, or internally
inconsistent. DF26 does not specify them. Each is a standard, defensible choice, and each points to
where it is applied in the body.

**Grid spacing (Section[@sec:grid-refinement]).** The capital grids are log-spaced, denser at low capital where the
value function curves most; the net-debt, gross-debt, and cash grids are uniform; network training
samples states uniformly. DF26 is silent on spacing, so this follows common practice.

**Default rule in grid refinement (Section[@sec:debt-default-recovery-and-bond-pricing]).** The bond-price default indicator is
$\mathbf{1}\{V'<0\}\cdot\mathbf{1}\{R<b'k'\}$ (both conditions); firm exit and the moment filter use
$V'<0$ alone. This holds in training, simulation, and grid refinement. Because $\chi$ is small, the
second condition almost always holds for indebted firms, so the rule coincides with $V'<0$ there and
matches DF26's algorithm.

**Debt-ratio moment definition (Section[@sec:moment-construction]).** The debt ratio uses the appendix form, gross debt
over next-period total assets $B^{\text{gross}\prime}/(K'+C')$, which matches the data (the gross item
DLC+DLTT over total assets); the main text's net-debt version does not. The cash and cash-saving
ratios follow the appendix as well; investment, operating income, and net debt follow the main text.
Exact reproduction of the reported targets depends on the micro data (see the data-moments decision
below).

**Expectation rule by step (Section[@sec:grid-refinement]).** Training integrates over continuous $z'$ with 5-node
Gauss-Hermite quadrature; grid refinement uses the 11-state Tauchen transition matrix; the simulation
runs the grid-refined policy and so inherits the Tauchen chain. Each step uses the rule that matches
its representation of $z$.

**Data moments and weighting matrix (Sections[@sec:application-inputs-external-values-bounds-targets-reference-estimates] and[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification]).** The data processing is fully specified, so
the only residual is the raw Compustat panel itself (a data diff from vintage and revisions). Section
[@sec:application-inputs-external-values-bounds-targets-reference-estimates] fixes the sample, filters, variable mapping, winsorization (1st and 99th percentiles by year), and
firm fixed-effect removal; Section[@sec:minimum-loss-functions-adaptive-shrinkage-and-identification] fixes the weighting matrix (inverse of the influence-function
covariance, clustered at the firm level). Given the same micro data these reproduce the targets and
the weighting matrix; where the data is unavailable, use the reported targets in Section[@sec:application-inputs-external-values-bounds-targets-reference-estimates] directly.

**FiLM generator details (Section[@sec:architecture-film-conditioning]).** The generator hidden-layer activation is SiLU, consistent
with the trunk and Table A2, and the scale and shift start at the identity ($\gamma_l = 1$,
$\delta_l = 0$) so each FiLM layer is a no-op at initialization. Both are safe, standard defaults that
DF26 does not state.

**Convergence criteria inside grid refinement (Section[@sec:grid-refinement]).** The semismooth Newton stops when the
active-set partition stabilizes (safeguard: $10^{-10}$ relative value change, at most 50 iterations);
GMRES uses relative residual $10^{-10}$, restart 50, at most 1000 matrix-vector products,
warm-started. DF26 does not specify these; the values match its $10^{-10}$ estimation tolerance and
target VFI-level accuracy.
