# Generalized Method of Moments

## General Framework

GMM estimates structural parameters from moment conditions that are known, closed-form functions of observables and parameters. Unlike SMM, GMM does not require solving the model. It applies whenever the model produces structural restrictions (e.g., Euler equations) that can be evaluated directly from data and a candidate $\beta$.

### Notation

| Symbol | Definition |
|---|---|
| $\beta^*$ | True structural parameters. Unknown. |
| $\beta$ | A candidate parameter vector. |
| $\hat{\beta}$ | The GMM estimate: the $\beta$ that minimizes $Q(\beta)$. |
| $K$ | Number of parameters to estimate. |
| $R$ | Total number of moment conditions ($R \geq K$). |
| $N$ | Number of cross-sectional units (e.g., firms). |
| $T$ | Number of time periods. |
| $e_{it}(\beta)$ | Structural residual for observation $i, t$ (e.g., Euler equation error). At the true $\beta^*$, $\mathbb{E}_t[e_{it}(\beta^*)] = 0$. |
| $Z_{it}$ | Vector of instruments: variables known at time $t$, uncorrelated with $e_{it}(\beta^*)$. |
| $g(\beta)$ | $R \times 1$ sample moment vector. At $\beta^*$, $\mathbb{E}[g(\beta^*)] = 0$. |
| $W$ | $R \times R$ positive-definite weighting matrix. |
| $g_{it}(\beta)$ | $R \times 1$ per-observation moment contribution: $e_{it}(\beta) \cdot Z_{it}$. |
| $\hat{\Omega}$ | $R \times R$ long-run variance-covariance matrix of $g_{it}$. |
| $\hat{\Gamma}_l$ | $R \times R$ lag-$l$ autocovariance of $g_{it}$. |
| $L$ | Bandwidth (number of lags) for the HAC estimator. |
| $D$ | $R \times K$ Jacobian matrix: $D_{rk} = \partial g_r / \partial \beta_k\vert_{\hat{\beta}}$. |
| $V$ | $K \times K$ asymptotic variance-covariance matrix of $\hat{\beta}$. |

### Moment conditions

The conditional restriction $\mathbb{E}_t[e_{it}(\beta^*)] = 0$ implies the unconditional restriction $\mathbb{E}[e_{it}(\beta^*) \cdot Z_{it}] = 0$ for any instrument $Z_{it}$ in the time-$t$ information set. Stacking all instrument interactions gives the $R \times 1$ sample moment vector:

$$g(\beta) = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T} e_{it}(\beta) \cdot Z_{it}$$

**Identification requires:** (1) $R \geq K$, and (2) the instruments $Z_{it}$ are first-stage relevant (correlated with the endogenous variables in $e_{it}$) and exclusion restrictions (uncorrelated with $e_{it}(\beta^*)$). Weak instruments cause severe finite-sample bias.

### Estimation

**Objective.** For a given weighting matrix $W$ ($R \times R$, positive definite):

$$Q(\beta) = g(\beta)^\top W\, g(\beta)$$

$$\hat{\beta} = \arg\min_\beta \; Q(\beta)$$

Each evaluation of $Q$ is arithmetic on the data (no model solve), so standard local optimizers suffice (e.g., `scipy.optimize.minimize` with `Powell` or `L-BFGS-B`).

### Optimal weight (two-step)

The optimal $W$ depends on $\beta^*$ because the residuals $e_{it}(\beta)$ change with $\beta$. A two-step procedure is required.

**First step.** Set $W = I_R$ (identity matrix). Minimize $Q(\beta)$ to obtain $\hat{\beta}_1$.

**Construct $\hat{\Omega}$.** At $\hat{\beta}_1$, compute the $R \times 1$ per-observation moment contribution $g_{it}(\hat{\beta}_1) = e_{it}(\hat{\beta}_1) \cdot Z_{it}$. The optimal weighting matrix is $W = \Omega^{-1}$, where $\Omega$ is the long-run variance of the moment vector. There are two estimators for $\hat{\Omega}$, depending on the serial correlation structure.

**i.i.d. estimator.** If $g_{it}$ is serially uncorrelated across $t$ (and independent across $i$):

$$\hat{\Omega}_{\text{iid}} = \frac{1}{NT}\sum_{i,t} g_{it}(\hat{\beta}_1)\, g_{it}(\hat{\beta}_1)^\top$$

**Newey-West estimator.** In dynamic models, the moment contributions $g_{it}$ are serially correlated within each firm because consecutive observations share persistent state variables (today's investment determines tomorrow's capital). The Newey-West HAC estimator accounts for this:

$$\hat{\Omega}_{\text{HAC}} = \hat{\Gamma}_0 + \sum_{l=1}^{L} w(l)\left(\hat{\Gamma}_l + \hat{\Gamma}_l^\top\right)$$

where $\hat{\Gamma}_l$ is the lag-$l$ autocovariance of the moment contributions, computed within each firm and averaged across firms:

$$\hat{\Gamma}_l = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=l+1}^{T} g_{it}(\hat{\beta}_1)\, g_{i,t-l}(\hat{\beta}_1)^\top$$

and $w(l) = 1 - l/(L+1)$ is the Bartlett kernel weight, which ensures $\hat{\Omega}_{\text{HAC}}$ is positive semi-definite. The bandwidth $L = \lfloor T^{1/3} \rfloor$. Cross-sectional independence across firms is assumed. Using $\hat{\Omega}_{\text{iid}}$ when serial correlation is present underestimates the true variance, producing standard errors that are too small and a J-statistic that is too large.

**Second step.** Set $W = \hat{\Omega}^{-1}$, warm start from $\hat{\beta}_1$. Minimize $Q(\beta)$ to obtain $\hat{\beta}$.

### Inference

Done once at the final $\hat{\beta}$.

**Jacobian.** Define the $R \times K$ Jacobian of the moment vector:

$$D_{rk} = \frac{\partial g_r(\beta)}{\partial \beta_k}\bigg|_{\beta = \hat{\beta}}, \qquad r = 1, \ldots, R, \quad k = 1, \ldots, K$$

Since $g$ is a known closed-form function of data and $\beta$, the Jacobian can be computed analytically or via centered finite differences. Each evaluation of $g$ is cheap (no model solve).

**Standard errors.** With the optimal $W = \hat{\Omega}^{-1}$, the $K \times K$ asymptotic variance-covariance matrix is:

$$V = (D^\top \hat{\Omega}^{-1} D)^{-1}$$

$$\text{se}(\hat{\beta}_k) = \sqrt{V_{kk} / (NT)}$$

The t-statistic for testing $H_0: \beta_k^* = \beta_k^0$:

$$t_k = \frac{\hat{\beta}_k - \beta_k^0}{\text{se}(\hat{\beta}_k)}$$

Reject $H_0$ at significance level $\alpha$ if $|t_k| > z_{1-\alpha/2}$.

**Overidentification test.** Requires $R > K$ and $W = \hat{\Omega}^{-1}$.

$$J = NT \cdot Q(\hat{\beta})$$

$$H_0: \quad \exists\; \beta^* \;\text{ such that }\; \mathbb{E}[g(\beta^*)] = 0$$

Under $H_0$: $J \xrightarrow{d} \chi^2(R - K)$.

For significance level $\alpha$, reject $H_0$ if $J > \chi^2_{1-\alpha}(R-K)$. A rejection indicates model misspecification.

---

## Application: Basic Investment Model

This section applies the general GMM framework to the basic investment model with convex adjustment costs in @strebulaev2012 [section 3.1]. The model has a closed-form Euler equation, so GMM is applicable.

### Model

- Production: $\pi(k,z) = zk^\alpha$, where $\alpha \in (0,1)$ is the capital share.
- Convex adjustment cost: $\psi(I,k) = \frac{\psi_1}{2}\frac{I^2}{k}$, where $I_t = k_{t+1} - (1-\delta)k_t$.
- Shock process (zero-drift convention): $\ln z_{t+1} = \rho \ln z_t + \varepsilon_{t+1}$, $\varepsilon \sim N(0, \sigma_\varepsilon^2)$.
- Parameters to estimate: $\beta = (\alpha, \psi_1, \rho, \sigma_\varepsilon)$, $K = 4$.
- Calibrated: $r, \delta$.

### Observables

In data (e.g., Compustat) or a simulated panel, the observable variables for firm $i$ at time $t$ are:

| Symbol | Observable | Definition |
|---|---|---|
| $\pi_{it}$ | Operating income | $z_{it} k_{it}^\alpha$ (a single observed number) |
| $k_{it}$ | Capital stock | Book value of assets |
| $I_{it}$ | Investment | $k_{i,t+1} - (1-\delta)k_{it}$ |

The productivity shock $z_{it}$ is **latent** — not directly observed. It is recovered from observables at a given candidate $\alpha$:

$$\ln z_{it}(\alpha) = \ln \pi_{it} - \alpha \ln k_{it}$$

All residuals and instruments below are known functions of $(\pi_{it}, k_{it}, I_{it})$ and the candidate $\beta$. No model solve is required: each evaluation of $g(\beta)$ is arithmetic on the data.

### Structural residuals

Each residual below is a scalar for observation $(i,t)$. The number of moment conditions from each block equals the dimension of the corresponding instrument vector (see next section).

**Euler equation residual.** The Euler equation, derived by eliminating $V$ via the envelope condition, is:

$$1 + \psi_1 \frac{I_t}{k_t} = \frac{1}{1+r}\mathbb{E}_t\!\left[\alpha\frac{\pi_{t+1}}{k_{t+1}} + \frac{\psi_1}{2}\!\left(\frac{I_{t+1}}{k_{t+1}}\right)^{\!2} + (1-\delta)\!\left(1 + \psi_1 \frac{I_{t+1}}{k_{t+1}}\right)\right]$$

The term $\alpha z_{t+1} k_{t+1}^{\alpha-1} = \alpha \cdot \pi_{t+1} / k_{t+1}$ is the marginal product of capital directly computable from observables $(\pi, k)$ and the candidate $\alpha$. Every term in the residual is a known function of observables and $\beta$, requiring no model solve:

$$e_{it}^u(\beta) = \alpha\frac{\pi_{i,t+1}}{k_{i,t+1}} + \frac{\psi_1}{2}\!\left(\frac{I_{i,t+1}}{k_{i,t+1}}\right)^{\!2} + (1-\delta)\!\left(1 + \psi_1\frac{I_{i,t+1}}{k_{i,t+1}}\right) - (1+r)\!\left(1 + \psi_1\frac{I_{it}}{k_{it}}\right)$$

This residual does not require computing the conditional expectation and instead directly use $I_{t+1}$ and $k_{t+1}$ from the real dataset.
The target parameters to be identified from this residual are $\alpha$ and $\psi_1$.

**Shock process residual.** Using the recovered shock $\ln z_{it}(\alpha) = \ln \pi_{it} - \alpha \ln k_{it}$, the AR(1) residual is:

$$e_{it}^v(\beta) = \ln z_{i,t+1}(\alpha) - \rho \ln z_{it}(\alpha)$$

Note that $\ln z_{it}(\alpha)$ depends on $\alpha$, creating a cross-equation restriction: the same $\alpha$ must fit both the Euler equation and the shock dynamics. This residual identifies $\rho$.

**Variance condition** (identifies $\sigma_\varepsilon$):

$$e_{it}^w(\beta) = (e_{it}^v)^2 - \sigma_\varepsilon^2$$

### Instruments

An instrument $Z_{it}$ is valid if it is (i) known at time $t$ (so uncorrelated with the innovation $\varepsilon_{t+1}$ that drives the residual) and (ii) correlated with next-period outcomes (relevant for identification).

**Lagged-only instrument design.** I restrict instruments to **strictly lagged** variables (time $t-1$ and earlier). Current-period variables ($I_t/k_t$, $\pi_t/k_t$, $\ln z_t$) are excluded because they appear directly in the structural residuals, creating a mechanical channel: the moment product $e_{it} \cdot Z_{it}$ contains terms like $\psi_1 (I_t/k_t)^2$ or $\rho (\ln z_t)^2$ that are precise second moments of the data, producing pathologically small standard errors in finite samples. Lagged instruments avoid this issue because they enter only through the conditional expectation, not through the residual itself.

**For the Euler equation block** ($3 \times 1$ instrument vector), the instruments are once-lagged investment and profitability ratios — all directly observable:

$$Z_{it}^u = \left(1, \;\frac{I_{i,t-1}}{k_{i,t-1}}, \;\frac{\pi_{i,t-1}}{k_{i,t-1}}\right)^\top$$

The constant captures the unconditional restriction. Lagged investment and profitability ratios predict future outcomes through persistence in $z$. All instruments are observed in the data (no dependence on $\beta$). Validity holds because they are known at time $t-1$ and thus orthogonal to innovations at $t$ and $t+1$.

**For the shock process block** ($2 \times 1$ instrument vector), the instrument is the once-lagged recovered shock:

$$Z_{it}^v = \left(1, \; \ln z_{i,t-1}(\alpha)\right)^\top$$

This depends on $\alpha$ through the recovery $\ln z_{it}(\alpha) = \ln \pi_{it} - \alpha \ln k_{it}$, so it is re-evaluated at each candidate $\beta$. Validity holds because $\varepsilon_{t+1}$ is i.i.d. and hence uncorrelated with $z_{t-1}$ by the AR(1) specification.

**For the variance condition**, no instrument is needed beyond the constant ($1 \times 1$).

### Moment count

| Block | Residual | Instruments | Conditions | Identifies |
|---|---|---|---|---|
| Euler equation | $e^u_{it}$ | $Z^u_{it}$ ($3 \times 1$) | 3 | $\alpha, \psi_1$ |
| Shock process | $e^v_{it}$ | $Z^v_{it}$ ($2 \times 1$) | 2 | $\rho$ (and $\alpha$ via $\ln z$) |
| Variance | $e^w_{it}$ | 1 (constant) | 1 | $\sigma_\varepsilon$ |
| **Total** | | | **$R = 6$** | **$K = 4$, overid $= 2$** |

### Stacked moment vector

The $R \times 1$ sample moment vector ($R = 6$) stacks all residual-instrument products:

$$g(\beta) = \frac{1}{NT}\sum_{i,t} \begin{pmatrix} e^u_{it}(\beta) \cdot Z_{it}^u \\ e^v_{it}(\beta) \cdot Z_{it}^v \\ e^w_{it}(\beta) \end{pmatrix} \in \mathbb{R}^{6}$$

Estimation, optimal weight, and inference follow the general framework above with $R = 6$ and $K = 4$. The optimal weight $\hat{\Omega}$ should use the HAC estimator described in the general framework, since the moment contributions $g_{it}$ are serially correlated in this dynamic model.

### Limitations

This GMM approach is available only because the basic investment model with convex adjustment costs has a closed-form Euler equation. Adding any of the following features breaks the Euler equation and requires switching to SMM:

- **Fixed adjustment costs** ($\psi_0 k \cdot \mathbb{1}_{I \neq 0}$): creates a kink in $V(k,z)$ at the inaction boundary, invalidating the envelope condition.
- **Risky debt**: the default option $\max\{0, \cdot\}$ and equity issuance indicator $\mathbb{1}_{e<0}$ introduce non-differentiabilities. The bond price depends on $V$ through the default threshold, preventing elimination of $V$.
- **Any feature that makes $V$ non-differentiable or impossible to eliminate analytically.**