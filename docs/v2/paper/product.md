# Business Product Proposal: Conceptual Framework

*Concept-level pitch. Formal model equations and code follow separately.*

## Overview

The project's solution pipeline solves a dynamic structural model of corporate capital structure and estimates its parameters from firm-level data using Bayesian MCMC. The same pipeline supports two products that differ in buyer and output.

| Product | Buyer | Output | Edge / Market Niche |
|---|---|---|---|
| **P1. Buy-side analytics** | Hedge funds and asset managers | Daily "distance-from-optimal" scores and multi-quarter forecasts of leverage, investment, credit spread, and default probability, across thousands of firms, with Bayesian credible intervals | Moody's EDF-X, Bloomberg DRSK, S&P Credit Analytics, and Fitch Connect predict default risk taking capital structure as exogenous input. Ours solves for the optimal structure as a dynamic optimization and gives a prescriptive signal with quantified uncertainty |
| **P2. Corporate decision support** | Corporate CFOs (direct) or internal JPM advisory teams | Recommended leverage, investment, and cash saving with credible intervals; counterfactual value uplift; scenario analysis | Banker advisory (JPM CFA, Goldman, Morgan Stanley, boutiques) is bespoke but covers few clients on annual cycles. FP&A platforms (Anaplan, Workday Adaptive) aggregate scenarios without optimizing. Ours scales structural recommendations across many firms with continuous updating and uncertainty quantification |

The rest of this document elaborates each product and then defines the solution pipeline.

---

## P1. Buy-side analytics

**Buyer.** Long-short equity funds, credit hedge funds, and fixed-income asset managers. Both quantitative and discretionary teams.

**Value created.** Two outputs: the gap between observed and structurally optimal action (across leverage, investment, and cash saving), and the predicted forward path of these variables plus credit spread and default probability. Every output carries a Bayesian credible interval. The signal is built on a structural model expected to be largely orthogonal to standard equity factors (value, momentum, quality, low volatility), so it should provide a new return source. Orthogonality is a hypothesis to verify on data, not yet a result.

**Edge.** Moody's EDF-X and Bloomberg DRSK use static Merton-style structural models. S&P Credit Analytics and Fitch Connect use reduced-form scoring. All treat capital structure as exogenous input and produce a default-risk number. Our product is prescriptive (what the firm should do) rather than predictive (what will happen) and complements existing vendor feeds rather than duplicating them.

**Examples.**

* A publicly listed real estate developer is flagged as 12 percentage points above its model-recommended leverage, with the model predicting credit spread widening over the next 2 quarters at 80% posterior probability. A credit hedge fund takes a short position in the firm's senior bonds.
* A large publicly listed technology firm is flagged as holding cash equal to about 15% of market capitalization above its model-recommended cash holding, with the firm under-leveraged relative to its profitability. The model predicts a buyback or special dividend within 12 months at 75% posterior probability. A long-short equity fund takes a long position ahead of the expected announcement.

---

## P2. Corporate decision support

**Buyer.** Direct: corporate CFOs and treasurers. Internal: JPM Corporate Finance Advisory, Debt Capital Markets (DCM), Equity Capital Markets (ECM), and Private Bank, using the tool to prepare client pitches.

**Value created.** A structural-model recommendation for the firm's three core decisions (leverage, investment, cash saving), with dividend payout residually determined by the budget constraint, plus issuance sequencing over time. Each recommendation carries a credible interval. The product also produces a counterfactual value uplift (expected firm-value gain from moving to the recommended action) and a scenario analysis under recession, interest-rate shock, and tax-regime change.

**Edge.** Banker-led advisory is bespoke and senior-led but covers a narrow client list on annual cycles. FP&A platforms aggregate user-defined scenarios without optimizing. Our product scales structural recommendations across many firms continuously and quantifies confidence. The framing inside JPM is augmentation, not replacement: the model output feeds the banker's pitch, the banker still delivers it.

**Examples.**

* A JPM banker prepares a pitch for a publicly listed industrial firm considering a primary equity offering. The model shows current leverage 8 percentage points above recommended and current investment below recommended capex. An equity raise of about 5% of market capitalization, paired with a planned capex program over four quarters, would close the joint gap with 80% credible interval [3.5%, 7%]. Counterfactual value uplift is 4% per share with credible interval [1.5%, 7%].
* A mid-cap consumer firm with a major senior-note maturity coming due in 18 months evaluates refinancing options. The model recommends refinancing about 60% of the maturity with new 7-year senior notes, repaying the remainder from cash reserves, and holding investment at current levels. Leverage drops 3 percentage points; counterfactual value uplift is 1.5% per share with credible interval [0.5%, 2.5%]. Scenario analysis shows the optimal rollover share rises under a low-rate environment.

---

## The solution pipeline

**Stage 1. Model solve.** Solve the dynamic structural model for the optimal policy function $\pi^*(s; \theta)$ mapping state vector $s$ to action vector $a$.

* State $s$ covers firm-, industry-, and macro-level variables (high-dimensional).
* Action $a$ covers leverage, investment, and cash saving as the three primary controls. Dividend payout is residually determined by the budget constraint, or modeled as a fourth control in extensions.
* Structural parameters $\theta$ include shock persistence and volatility, production curvature, tax rate, bankruptcy cost, and issuance cost.

The solver uses a neural-network approximation in the Maliar et al. (2021) deep-learning style, which scales with state dimension where grid-based VFI fails.

**Stage 2. Bayesian estimation.** For each firm $i$, estimate the posterior $p(\theta_i \mid D_i)$ over structural parameters using hierarchical Bayesian MCMC,
$$\theta_i \sim N(\theta_{\text{ind}}, \Sigma_{\text{ind}}),$$
which pools information across firms within an industry. Data-rich firms drive their own estimates; data-poor firms shrink toward the industry mean. Sampler choice (HMC/NUTS, Metropolis-Hastings, particle MCMC, or Gibbs) depends on model structure. Simulated method of moments (SMM) remains available as a cross-check.

**Stage 3. Posterior summary.** For a target firm at observed state $s^{\text{obs}}$, compute the posterior over the recommended action $\{\pi_\phi(s^{\text{obs}}; \theta_i^{(m)})\}$ and the posterior over firm value at the optimum, then forward-simulate under the optimal policy and stochastic shocks. All user-facing outputs summarize these objects.

**Data feasibility.** Required inputs are covered by standard enterprise subscriptions: Bloomberg, Refinitiv/LSEG, and S&P Capital IQ Pro for global coverage, plus Wind and CSMAR for Greater China.

### Outputs

| User output | Computation |
|---|---|
| Recommended action (leverage, investment, cash saving) with 80% credible interval | Posterior median and 10/90th percentiles of $\{a_i^{*(m)}\}$ per component |
| Distance-from-optimal score | $\Delta a_i = \lVert a_i^{\text{obs}} - \text{posterior median of } a_i^* \rVert$, per component or aggregate |
| Counterfactual value uplift | $\Delta V_i = V(s, a_i^*) - V(s, a_i^{\text{obs}})$, averaged over posterior draws |
| Multi-quarter forecast | Forward simulation under the optimal policy, summarized by quantile bands |
| Cross-sectional rankings | Sort by $\Delta a_i$ or predicted spread change. Feeds the P1 long/short signal |
| Scenario analysis | Re-run Stage 3 at perturbed state vectors; report recommendation deltas |