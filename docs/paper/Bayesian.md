# Bayesian Estimation of the Strebulaev (2012) Models

---

## Overview

### 1.1 Target

Treat the parameter vector $\beta$ as a random variable. Given observed data $y$, the target is the posterior distribution

$$p(\beta \mid y) \;\propto\; \underbrace{p(\beta)}_{\text{prior}} \cdot \underbrace{p(y \mid \beta)}_{\text{likelihood}}.$$

Markov Chain Monte Carlo (MCMC) operates on the **log-target**

$$L(\beta) \;:=\; \log p(\beta) + \log p(y \mid \beta) \;=\; \log p(\beta \mid y) + \log p(y).$$

Since $\log p(y)$ does not depend on $\beta$, it cancels in every Metropolis-Hastings (MH) acceptance ratio and never needs to be computed. The pipeline reduces to two tasks: specify a prior, and evaluate the log-likelihood $\log p(y \mid \beta)$ at any candidate $\beta$.

### 1.2 Generic Algorithm

The pipeline is generic across choices of filter and MCMC sampler.

**Setup.**

- Specify a prior $p(\beta)$.
- Specify a filtering algorithm that, given fixed data $y$ and a candidate $\beta$, returns the scalar $\log p(y \mid \beta)$.
- Fix the number of MCMC iterations $S$ and the number of chains.

**Main loop (per chain, per iteration $s = 1, \ldots, S$).**

1. **Propose** a candidate $\beta'$ by perturbing the current $\beta$. Two baseline schemes:
   - *Random-Walk Metropolis-Hastings (RW-MH):* draw $\beta' = \beta + $ Gaussian noise. Gradient-free.
   - *No-U-Turn Sampler with Hamiltonian Monte Carlo (NUTS-HMC):* use $\nabla_\beta L(\beta)$ to simulate Hamiltonian dynamics on $-L$, then slice-sample a candidate from the trajectory. Gradient-based.
2. **Evaluate** $L(\beta')$:
   - *Model-solve at $\beta'$* — obtain the policy function. This is closed-form here, but in general can be neural-network-based (NN) or by Value/Policy Function Iteration (VFI/PFI). The main computational bottleneck.
   - *Filter at $\beta'$* — plug the policy into the observation equation and run the chosen filter on fixed $y$ to compute $\log p(y \mid \beta')$.
   - *Add prior:* $L(\beta') = \log p(\beta') + \log p(y \mid \beta')$.
3. **Accept or reject.** RW-MH accepts $\beta'$ with probability $\min(1, \exp(L(\beta') - L(\beta)))$ under symmetric proposal. NUTS-HMC uses the joint Hamiltonian ratio.
4. **Record** the current chain position as sample $\beta^{(s)}$.

**Output.** A pooled set of samples $\{\beta^{(s)}\}$ across chains. Their empirical distribution approximates $p(\beta \mid y)$. Posterior means, quantiles, and credible intervals are computed from this set.

**Two principles guide method choice:**

- Filter and sampler are independent components and can be swapped separately.
- The MH accept/reject step guarantees correctness. The proposal rule controls only efficiency.

### 1.3 Method Selection Across Project Models

Two baselines cover all three target models.

| # | Model | Solver | Filter | MCMC sampler |
|---|---|---|---|---|
| 1 | Frictionless basic (Strebulaev 3.1) | Analytical | Kalman (exact) | NUTS-HMC |
| 2 | Full basic with frictions | NN-based, amortized in $\beta$ | Particle (Kalman if linearizable) | NUTS-HMC if NN end-to-end differentiable in $\beta$; else RW-MH |
| 3 | Risky debt with default (Strebulaev 3.6) | VFI/PFI | Particle | RW-MH |

Selection rule:

- **NUTS-HMC** when $L(\beta)$ is differentiable end-to-end in $\beta$.
- **RW-MH** (with adaptive proposal covariance) otherwise. When combined with a particle filter, this is Particle Marginal Metropolis-Hastings (PMMH; Andrieu, Doucet, Holenstein 2010), which targets the exact posterior despite a noisy likelihood estimate.

In the table, "amortized in $\beta$" means a single network is trained once across the prior support of $\beta$, so that the policy at any candidate $\beta$ is one forward pass rather than a full retraining.

---

## 2. Phase 1 — Frictionless Basic Model Validation

The goal is to verify the full Bayesian inference pipeline end-to-end on a model whose solver is exact and analytical. Any inference error is then attributable to the inference machinery rather than to solver approximations. Phase 2 swaps the solver and filter while keeping the rest of the pipeline.

### 2.1 Structural Model

Following Strebulaev and Whited (2012, Section 3.1), each firm $i = 1, \ldots, N$ solves an infinite-horizon investment problem with no adjustment costs and no debt.

$$\pi_{i,t} = z_{i,t} \cdot k_{i,t}^{\alpha}, \qquad \alpha \in (0,1).$$

$$k_{i,t+1} = (1 - \delta)\, k_{i,t} + I_{i,t}, \qquad \delta \in (0,1).$$

The log-productivity follows a first-order autoregressive process (AR(1)):

$$\log z_{i,t+1} = \rho \cdot \log z_{i,t} + \sigma_\varepsilon \cdot \varepsilon_{i,t+1}, \qquad \varepsilon_{i,t+1} \sim \mathcal{N}(0,1), \quad \rho \in (0,1), \quad \sigma_\varepsilon > 0.$$

Shocks $\varepsilon_{i,t+1}$ are independent across firms and time.

**Estimated parameters:** $\beta = (\alpha, \rho, \sigma_\varepsilon, \sigma_\eta)$. The first three are structural; $\sigma_\eta$ is the scale of revenue measurement error (defined in 2.3).

**Calibrated parameters:** $r = 0.04$ (risk-free rate) and $\delta = 0.10$ (depreciation rate), both fixed.

### 2.2 Analytical Policy Function

Without adjustment costs, the firm's Euler equation reduces to $E_t[\alpha \, z_{i,t+1} \, k_{i,t+1}^{\alpha-1}] = r + \delta$. Since $k_{i,t+1}$ is chosen at time $t$,

$$k_{i,t+1}(z_{i,t}; \beta) = \left( \frac{\alpha \cdot E_t[z_{i,t+1}]}{r + \delta} \right)^{\frac{1}{1-\alpha}}.$$

Under log-normal $z$, $E_t[z_{i,t+1}] = \exp(\rho \log z_{i,t} + \sigma_\varepsilon^2 / 2)$. Taking logs:

$$\log k_{i,t+1}(z_{i,t}; \beta) = \frac{\rho}{1-\alpha}\, \log z_{i,t} + \kappa(\beta), \qquad \kappa(\beta) := \frac{1}{1-\alpha}\!\left[\log \alpha + \frac{\sigma_\varepsilon^2}{2} - \log(r + \delta)\right].$$

This is linear in $\log z_{i,t}$, enabling the linear Gaussian state-space form below.

### 2.3 Linear Gaussian State-Space Form

A **Linear Gaussian State-Space Model (LGSSM)** has three properties: state transition and observation equations are linear in the latent state, all shocks and residuals are Gaussian, and an unobserved state evolves over time. Under these properties, the Kalman filter computes the exact likelihood in closed form. Phase 1 keeps the model strictly inside LGSSM.

**Panel data assumed available** for each firm $i = 1, \ldots, N$:

- Operating income $\pi_{i,t}$ for $t = 1, \ldots, T$.
- Capital stock $k_{i,t}$ for $t = 1, \ldots, T$.

Both quantities are treated as observed without error. (Measurement error on capital is deferred; see Future Extensions.)

**Latent state:** $x_{i,t} := \log z_{i,t}$ (scalar).

**State transition:** $x_{i,t+1} = \rho \cdot x_{i,t} + \sigma_\varepsilon \cdot \varepsilon_{i,t+1}$, $\varepsilon_{i,t+1} \sim \mathcal{N}(0,1)$.

**Initial state distribution.** Wide Gaussian prior, fixed independently of $(\rho, \sigma_\varepsilon)$:

$$x_{i,1} \sim \mathcal{N}(0, V_0), \qquad V_0 = 10.$$

Diffuse prior, standard initialization that avoids the divergence of $\sigma_\varepsilon^2 / (1 - \rho^2)$ as $\rho \to 1$ (Hamilton 1994, Ch. 13).

**Observation equation.** The single observable is log-revenue:

$$y_{i,t} := \log \pi_{i,t} = x_{i,t} + \alpha \log k_{i,t} + \eta_{i,t}, \qquad \eta_{i,t} \sim \mathcal{N}(0, \sigma_\eta^2).$$

Residuals $\eta_{i,t}$ are independent across firms and time. The term $\alpha \log k_{i,t}$ enters as a known offset (time-varying input from observed capital). In LGSSM matrix form: $C(\beta) = 1$, $d_{i,t}(\beta) = \alpha \log k_{i,t}$, $R(\beta) = \sigma_\eta^2$.

### 2.4 Likelihood via Kalman Filter
 
Independence across firms gives
 
$$\log p(Y \mid \beta) = \sum_{i=1}^{N} \sum_{t=1}^{T} \log p(y_{i,t} \mid y_{i,1:t-1}, \beta).$$
 
For each firm $i$, the Kalman recursion produces each predictive density $p(y_{i,t} \mid y_{i,1:t-1}, \beta)$ as a univariate Gaussian. Use the notation $m_{t \mid s}$ and $V_{t \mid s}$ for the conditional mean and variance of $x_{i,t}$ given $y_{i,1:s}$.
 
**Initialize** with the prior from 2.3: $m_{1 \mid 0} = 0$, $V_{1 \mid 0} = V_0 = 10$.
 
**For $t = 1, \ldots, T$:**
 
1. **Predict the observation** (uses the observation equation from 2.3):
$$\hat{y}_t = m_{t \mid t-1} + \alpha \log k_{i,t}, \qquad S_t = V_{t \mid t-1} + \sigma_\eta^2.$$
 
Likelihood contribution:
 
$$\log p(y_{i,t} \mid y_{i,1:t-1}, \beta) = -\tfrac{1}{2}\!\left[\log(2\pi S_t) + (y_{i,t} - \hat{y}_t)^2 / S_t\right].$$
 
2. **Update the state** with Kalman gain $K_t = V_{t \mid t-1} / S_t$:
$$m_{t \mid t} = m_{t \mid t-1} + K_t (y_{i,t} - \hat{y}_t), \qquad V_{t \mid t} = (1 - K_t)\, V_{t \mid t-1}.$$
 
3. **Predict next state** (uses the AR(1) state transition from 2.3):
$$m_{t+1 \mid t} = \rho \, m_{t \mid t}, \qquad V_{t+1 \mid t} = \rho^2 V_{t \mid t} + \sigma_\varepsilon^2.$$
 
Sum likelihood contributions across $t$ and $i$ to obtain $\log p(Y \mid \beta)$. Total cost: $O(N \cdot T)$.
 
The structural model enters in exactly two places: $(\alpha, \sigma_\eta, \log k_{i,t})$ in step 1 via the observation equation, and $(\rho, \sigma_\varepsilon)$ in step 3 via the AR(1) state transition. Everything else is generic Gaussian filtering.
 
**TensorFlow Probability (TFP) module:** `tfp.distributions.LinearGaussianStateSpaceModel`. We do not transcribe the recursion above into code. Instead, we feed it the model-specific pieces of the state-space form from 2.3 and TFP runs the recursion internally. Per candidate $\beta$, construct one LGSSM instance with these arguments:

- `transition_matrix=[[ρ]]`, `transition_noise=MVN(scale_diag=[σ_ε])`: the AR(1) state dynamics.
- `observation_matrix=[[1]]`, `observation_noise=MVN(scale_diag=[σ_η])`: the constant slope on the latent state and revenue noise scale.
- `observation_offset=α·log k`: the only batch- and time-varying piece, passed as a tensor of shape `[N, T, 1]`. The leading dim $N$ is the firm batch (the only place batch information enters the LGSSM).
- `initial_state_prior=MVN(loc=[0], scale_diag=[√V_0])`.

Then `.log_prob(Y)` on the observation tensor of shape `[N, T, 1]` returns the per-firm log-likelihoods as a tensor of shape `[N]`; sum to obtain the scalar $\log p(Y \mid \beta)$. The whole call is auto-differentiable in $\beta$, so NUTS-HMC gets $\nabla_\beta \log p(Y \mid \beta)$ without any hand-coded backward pass.
 
### 2.5 Priors

| Parameter | Prior | Support |
|---|---|---|
| $\alpha$ | Beta(2, 2) | $(0, 1)$ |
| $\rho$ | Beta(2, 2) | $(0, 1)$ |
| $\sigma_\varepsilon$ | HalfNormal(0.3) | $(0, \infty)$ |
| $\sigma_\eta$ | HalfNormal(0.1) | $(0, \infty)$ |

Beta is the natural family on $(0, 1)$ and lightly downweights boundary degeneracies. HalfNormal is preferred to Uniform for scale parameters: no arbitrary upper bound and smooth gradients.

NUTS-HMC operates on $\mathbb{R}^4$. Each constrained parameter is transformed to the real line via a TFP bijector (`Sigmoid` for $(0, 1)$, `Exp` for positive scales). TFP applies the Jacobian correction automatically through `tfp.mcmc.TransformedTransitionKernel`.

### 2.6 Sampler Configuration

**Target.** $L(\beta) = \sum_j \log p_j(\beta_j) + \log p(Y \mid \beta)$, auto-differentiable in $\beta$.

**Sampler.** `tfp.experimental.mcmc.windowed_adaptive_nuts`. Jointly adapts step size (dual averaging) and mass matrix (windowed empirical covariance) during warm-up. The mass matrix is essential because $\alpha, \rho, \sigma_\varepsilon, \sigma_\eta$ have different posterior scales on the unconstrained side.

| Setting | Value |
|---|---|
| Number of chains | 4 |
| Warm-up iterations | 1000 |
| Sampling iterations per chain | 2000 |
| Target acceptance rate | 0.80 |

Chains start from independent prior draws. After sampling, map back via inverse bijectors to recover values on the natural scale. Output: 8000 posterior samples $\{\beta^{(s)}\}$.

### 2.7 Convergence Diagnostics

For an MCMC run with $M$ chains and $S$ post-warmup samples each, compute per parameter:

**Split $\hat{R}$** (Vehtari et al. 2021): compares within- and between-chain variance after splitting each chain in half. Converged chains give $\hat{R} \to 1$. **Pass:** $\hat{R} < 1.01$.

**Effective Sample Size (ESS):** number of independent samples the chain is equivalent to,

$$\mathrm{ESS} = \frac{MS}{1 + 2 \sum_{k \geq 1} \rho_k},$$

with $\rho_k$ the lag-$k$ autocorrelation (truncated at the first negative estimate). **Pass:** $\mathrm{ESS} > 400$ (about 5% Monte Carlo error on posterior quantiles).

**Trace plots:** plot $\beta_j^{(s)}$ vs $s$ overlaying chains; visual debug aid.

**TFP modules:** `tfp.mcmc.potential_scale_reduction`, `tfp.mcmc.effective_sample_size`.

If diagnostics fail, increase warm-up iterations, raise target acceptance to 0.95, or revisit priors.

### 2.8 Validation: Coverage Check

Phase 1's deliverable is a verified pipeline. The minimal test is a lightweight coverage check (a coarse version of SBC; full SBC and PPC deferred to Future Extensions).

Procedure:

1. Draw $R = 10$ values $\beta_0$ from the prior (default; raise once timing is known).
2. For each $\beta_0$: generate one synthetic panel of size $(N, T) = (200, 40)$ using the existing simulator (`BasicInvestmentEnv.simulate_smm_panel_data` with `mode="frictionless_analytical"`, `n_panels=1`), which rolls out AR(1) shocks under the analytical policy at $\beta_0$ to produce $(k_{i,t}, z_{i,t})$. Form observations $y_{i,t} = \log z_{i,t} + \alpha \log k_{i,t} + \eta_{i,t}$, $\eta_{i,t} \sim \mathcal{N}(0, \sigma_\eta^2)$ (independent seed). Pass $(y, \log k)$ to the full MCMC pipeline; record the 95% credible interval per parameter (empirical 2.5% and 97.5% quantiles of pooled samples).
3. For each parameter, count the fraction of intervals containing the corresponding component of $\beta_0$.

**Pass:** empirical coverage close to 95% for each parameter (e.g., $\geq 8/10$ at $R = 10$, allowing sampling noise).

This catches gross calibration failures (biased posteriors, mis-calibrated credible intervals) at $R$ MCMC runs rather than the 100–1000 runs needed for full SBC.

### 2.9 Reproducibility

The pipeline reuses the project's existing stateless-seed infrastructure (`src/v2/data/rng.py`, `src/v2/utils/seeding.py`). A single master seed pair `(m0, m1)` controls every RNG-consuming step. Per-replicate child seeds are derived deterministically via `fold_in_seed(master, *tokens)`, where `tokens` are short namespace strings.

For replicate $r \in \{0, \ldots, R-1\}$, derive:

- `rep_seed     = fold_in_seed(master, "replicate", r)`
- `beta0_seed   = fold_in_seed(rep_seed, "beta0")` — prior draw of $\beta_0$.
- `panel_seed   = fold_in_seed(rep_seed, "panel")` — passed to `simulate_smm_panel_data(seed=...)`.
- `noise_seed   = fold_in_seed(rep_seed, "obs_noise")` — `tf.random.stateless_normal` for $\eta$.
- `mcmc_seed    = fold_in_seed(rep_seed, "mcmc")` — passed to `windowed_adaptive_nuts(seed=...)`; covers chain initialization and all internal momentum/slice draws.

Properties:

- Rerunning with the same `master` reproduces every $\beta_0$, every panel, every $\eta$, and every chain trajectory bit-for-bit on the same hardware.
- Token-scoped folding means stages are isolated: re-running just MCMC on a fixed panel only changes `mcmc_seed`'s token, not the panel or $\beta_0$ draws.
- Cross-hardware bit-identical TF op execution is opt-in via `seed_runtime(master, strict_reproducibility=True)`. Default off; on for paper-grade reruns.

No global RNG state is used. All randomness enters through explicit seed arguments.

---

## 3. Future Extensions

Deferred from Phase 1 to keep the baseline minimal:

- **Firm-specific outputs.** Apply the Kalman smoother per posterior draw to obtain per-firm latent productivity estimates $p(x_{i,t} \mid y_{i,1:T}, \beta)$.
- **Capital measurement error.** Putting $\log k$ in the latent state breaks LGSSM and requires a particle filter.
- **Firm-specific structural parameters.** Hierarchical extension $\beta_i \sim p(\cdot \mid \beta_{\text{population}})$.
- **Simulation-Based Calibration (SBC; Talts et al. 2018).** Rank-based formal test of posterior calibration against the prior-implied DGP. Run before real-data deployment.
- **Posterior predictive checks (PPC).** Compare statistics of replicated $Y_{\mathrm{rep}} \sim p(\cdot \mid \beta^{(s)})$ to observed $Y$. Essential for detecting model misspecification on real data.
- **Real-data application.** After synthetic validation passes.

---

## References

- Andrieu, C., A. Doucet, and R. Holenstein (2010). "Particle Markov chain Monte Carlo methods." *Journal of the Royal Statistical Society, Series B* 72, 269–342.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 13 (Kalman filter).
- Hoffman, M. D., and A. Gelman (2014). "The No-U-Turn Sampler." *Journal of Machine Learning Research* 15, 1593–1623.
- Strebulaev, I. A., and T. M. Whited (2012). "Dynamic Models and Structural Estimation in Corporate Finance." *Foundations and Trends in Finance* 6, 1–163.
- Talts, S., M. Betancourt, D. Simpson, A. Vehtari, and A. Gelman (2018). "Validating Bayesian Inference Algorithms with Simulation-Based Calibration." arXiv:1804.06788.
- Vehtari, A., A. Gelman, D. Simpson, B. Carpenter, and P.-C. Bürkner (2021). "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC." *Bayesian Analysis* 16, 667–718.