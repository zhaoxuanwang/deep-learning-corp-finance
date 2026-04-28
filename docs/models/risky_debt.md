---
title: Risky Debt Investment Model
bibliography: docs/references.bib
---


# Risky Debt Investment Model

This document implement deep learning methods to solve for the dynamic debt model of @hennessy2007costly and reviewed in @strebulaev2012. 

## 1. Model

The risky debt model extends the basic investment model by allowing firms to borrow at an endogenous risky interest rate, with the option to default. The risky interest rate is determined by the lender's zero-profit condition with rational expectations of default probability. The firm's optimal investment and leverage in turn depend on the endogenous interest rate determined by debt market equilibrium.

### Definitions

**State space.** The state variables are capital $k$, debt $b$, and productivity $z$.
- State vector $s = [s_{\text{endo}} \mid s_{\text{exo}}]$
- Endogenous states $s_{\text{endo}} = (k, b)$ are controlled by policy
- Exogenous states $s_{\text{exo}} = (z)$ follow AR(1) process

State space are bounded: $k \in [k_{\min}, k_{\max}]$, $b \in [b_{\min}, b_{\max}]$, $z \in [z_{\min}, z_{\max}]$.

To determine the bounds, I start with a tax-adjusted frictionless reference capital level 

$$k_{\text{ref}} = \left[\frac{(1-\tau)\alpha \cdot E[z'|z]}{(r+\delta)}\right]^{1/(1-\alpha)}$$

which is derived from the firm's first order condition when there are no frictional costs and no risky debt.

**Action space.** The firm chooses investment $I$ and next-period debt $b'$. I replace the action $I$ with $k'$ because $k'=(1-\delta)k-I$ is directly implied by the action.

- Action vector $a \equiv (k', b')$ 

The main benefit of using $k'$ directly as policy function output is it reduces state-action space and the curse of dimensionality for the Value Function Iteration (VFI) methods.

Following @strebulaev2012, I use a single variable $b'$ to denote both savings when $b'<0$ and debt (borrowing) when $b'>0$. Thus the saving limit is negative and bounded:
$$ -\infty \lt b_{\min} \lt 0$$

The borrowing limit is
$$ b_{\max} = C_b \cdot k_{\max} \gt 0$$

where $C_b$ is a user-specified multiplier and should be set at moderate value.


**State transition.**
$$f(s, a, \epsilon) = \begin{pmatrix} k' \\ b' \\ z' \end{pmatrix} = \begin{pmatrix} (1-\delta)k + I \\ b' \\ \exp\{(1-\rho)\mu + \rho \log z + \sigma \epsilon\} \end{pmatrix}$$
Capital evolves via accumulation; debt is a direct policy choice (passes through); productivity follows the log-AR(1) process (same as basic model).

**Production.** Cobb-Douglas: $\Pi(k,z) = z \cdot k^\alpha$ with $\alpha \in (0,1)$.

**Adjustment cost** (convex only): $\Psi(I,k) = \frac{\phi_0}{2} \cdot \frac{I^2}{k}$.

**Reward (payouts).** The reward is the equity payout net of external financing costs: $$e(k,b,z;\, I, b') - \Omega(e)$$ 

where the cash flow is
$$e = (1-\tau)\Pi(k,z) - \Psi(I,k) - I - b + \frac{b'}{1+\tilde{r}(k',b',z)} + \frac{\tau \cdot \tilde{r}(k',b',z) \cdot b'}{(1+r)(1+\tilde{r}(k',b',z))}$$
Components:
- $(1-\tau)\Pi(k,z)$: after-tax operating revenue
- $\Psi(I,k)$: capital adjustment cost
- $I=k'-(1-\delta)k$: investment expenditure
- $b$: debt repayment (determined from last period)
- $b'$: face value of next-period debt
- $\tilde{r}(k',b',z)$: endogenous interest rate on debt market
- $r$: risk-free interest rate
- $\tau \cdot \tilde{r} \cdot b'/(1+r)(1+\tilde{r})$: present-value tax shield from debt. Firms exploit debt financing to shield profit from taxation.

**External equity cost.** When cash flow is negative, the firm must raise costly external equity:
$$\Omega(e) = \left(\eta_0 + \eta_1 |e|\right)\mathbf{1}\{e < 0\}, \quad \eta_0\geq 0, \eta_1 \geq0$$
where $\eta_0$ is the fixed issuance cost and $\eta_1$ is the proportional cost on the financing shortfall. That is, when $e>0$ it is positive dividend payout from firm to shareholders, but when $e<0$ shareholders must inject $|e|$ to the firm and pay an additional issuance cost $\Omega(e)>0$ to intermediaries. This appear as the negative sum $e-\Omega(e)$ on the Bellman equation. Firm choose $e$ by choosing $(k',b')$, so it will only choose this costly equity injection when the future marginal return to investment and borrowing captured in $V$ is higher than the current period marginal cost determined by $\eta_1$, $\psi_1$, and $\tau$.  

### Endogenous bond pricing

The endogenous risky interest rate $\tilde{r} = \tilde{r}(z, k', b')$ is set so that the lender's (banks) expected return on the risky bond equals the risk-free rate $\bar{r}$. The lender lends $b'$ today and receives a state-contingent payoff next period depending on whether the firm defaults.

**Default decision.** 
@hennessy2007costly [Proposition 6] shows that there exists a critical shock value inducing default. In the Markov formulation used here, next period's continuation/default decision is made from the next-period state $(k', b', z')$. Therefore the default boundary is a boundary in the future shock conditional on the chosen next-period state, which we can write as:
$$ z_d(\cdot) : \mathcal{K} \times \mathcal{B} \to \mathcal{Z}$$
where $z_d(k',b')$ is a critical future-shock value for the next-period state $(k', b')$.

The firm defaults when the next-period shock realization is below the critical shock. The default indicator is defined as:
$$
\mathcal{I}_D(z') = 
\begin{cases}
1, & \text{if } z' < z_d \iff V'\leq 0\\
0, & \text{if } z'> z_d \iff V'\gt 0
\end{cases}
$$

Equivalently, in the discrete-grid algorithm the default set is recovered directly from the next-period value function:
$$\mathcal I_D(k', b', z') = 1\{V(k', b', z') = 0\}.$$

Current productivity $z$ still matters for pricing, but only through the conditional transition law $p(z'|z)$ used by lenders to take expectations over next period outcomes.

Note that this model is much more complex than the standard Trade-Off model considered in @nikolov2021 [section 2.2] where default threshold can be written down analytically.

**Recovery under default.** If the firm defaults, the lender recovers a fraction $c_d$ of the firm's profits and assets: $$R(k', z') = (1-c_d)[(1-\tau)\Pi(k', z') + (1-\delta)k']$$

**Lender's zero-profit condition.** The lender's opportunity cost of lending $b'$ is at risk-free rate $b'(1+r)$. The expected payoff is the recovery in default states plus the full repayment in solvent states:

$$b'(1+\bar{r}) = \int_{z_{\min}}^{z_d} R(k', z')\,dg(z'|z) \;+\; b' (1+\tilde{r}(k',b',z))\!\int_{z_d}^{z_{\max}}\,dg(z'|z)$$

where the default/solvent boundary is determined by the future-state threshold $z_d(k',b')$, or equivalently by the set $\{z' : V(k', b', z') = 0\}$. The endogenous interest rate $\tilde{r}$ can be moved outside the integral because it only depends on the current pricing information $(k',b',z)$.

Re-write this condition as:

$$b'(1+r) = \mathbb{E}_{z'|z} \left[
    R(k', z') \cdot \mathcal{I}_D(z') \;+\; b'(1+\tilde{r}) \cdot (1-\mathcal{I}_D(z')) \right]
$$

The required endogenous interest rate (bond yield) can be solved as:

$$\tilde{r}(k',b',z) =
\frac{(1+r) - \frac{1}{b'}\mathbb{E}_{z'|z}[\mathcal{I}_D(z') \cdot R(k',z')]}
{\mathbb{E}_{z'|z} \left[1-\mathcal{I}_D(z')\right]} - 1\tag{Bond Pricing}$$

This determines $\tilde{r}(z, k', b')$ as a function of the current exogenous state $z$, the firm's action $(k', b')$, and the next-period default boundary induced by $V(k', b', z')$. The current shock $z$ matters because lenders condition on $p(z'|z)$ when forming expectations; the default event itself is still a next-period event determined by $(k', b', z')$.

Note that when $b'<0$, firm must be saving at risk free rate, so $\tilde{r}=r$. 



### Bellman equation
As in equation (3.28) of @strebulaev2012, the firm's problem is written as

$$V(k,b,z) = \max \left\{ 0 , \max_{I,b'}\left\{e(k,b,z;\, I, b') - \Omega(e) + \frac{1}{1+r}\,\mathbb{E}_{z'|z}\!\left[ V(k',b',z') \right]\right\} \right\}$$

The outer $\max(0, \cdot)$ is the limited-liability mechanism: it makes default possible by allowing the firm to exit at zero when there is no $(I,b')$ pair that delivers positive equity value.

In the discrete algorithm, the implementation is therefore:

1. Compute the standard Bellman RHS with fixed outer-loop rates $\tilde r^{(n)}$.
2. Maximize the RHS over the discrete choice set $(k', b')$.
3. Clamp the updated value to enforce limited liability:

$$
V^{(s+1)}(k,b,z)=\max\left\{0,\max_{k',b'} \text{RHS}(k,b,z,k',b';\tilde r^{(n)})\right\}
$$

Thus the solver iterates directly on the nonnegative equity value $V$, and default states are exactly the states where the converged value satisfies $V=0$.

### The nested fixed-point problem

The value $V$ depends on the bond yield $\tilde{r}$ through the current-period cash flow, but $\tilde{r}$ depends on the default probability $\mathbb{E}[\mathcal{I}_D]$ determined by the zero-value default set. This creates a nested fixed point:
$$V \leftrightarrow \tilde{r} \leftrightarrow \mathcal{I}_D\{V = 0\}$$
The main computational challenge is how to break this circularity.

### Effective bounds

VFI is expensive because it evaluates all grid points on the state-action space $(k,b,z)\times(k',b')$. This is wasteful because the economically meaningful grid points are the region centered around 

Set $z_t = z_{\min}$ for all $t$ (worst-case perpetuity):

$$V_{\text{passive}}^{\min} = -b' + \frac{(1-\tau), z_{\min}, k'^{\alpha}}{1 - \beta(1-\delta)^\alpha}$$

Since $V_{\text{actual}} \geq V_{\text{passive}}^{\min}$ (optimal strategy $\geq$ do-nothing; true $z \geq z_{\min}$), setting $V_{\text{passive}}^{\min} = 0$ gives a sufficient condition for solvency:

$$\boxed{b'L(k') = \frac{(1-\tau), z{\min}, k'^{\alpha}}{1 - \beta(1-\delta)^\alpha}}$$

Below $b'L$, the firm is solvent under the worst perpetual shock with zero investment. This is a genuine lower bound: the firm would actually tolerate MORE debt because it can invest and borrow optimally, and $z$ mean-reverts upward from $z{\min}$.

### Defects of the model

There are several defects of the model, I focus on the defects of the model's core economic mechanisms and discuss their theoretical and practical (empirical) implications. I do not discuss critique that are either too board or general, or those that require adding new features and complexity to the model.

To clarify, the core mechanism of the risky debt model is that firm's financing decisions reflect (i) optimal investment under frictional adjustment cost; and (ii) the opportunity to exploit the tax shield benefit of debt (in the form of a one-period corporate bond). Lenders (bank) with rational expectation charge a risk premium on the yields of the corporate bond based on anticipated default probability. Default threshold is determined by the realization of next-period productivity shock conditional on current states and actions, for example, higher debt requires higher realization of future productivity to repay, and thus expand the default set (probability). This is anticipated by the lender and priced into the endogenously-determined risk premium. 

Given this core mechanism, I find three critical defects in the specific risky debt model presented in @strebulaev2012 [section 3.6]:

1. Timing of tax shield benefit create "looting" strategy
2. Wrong-signed equity issuance cost
3. Assumption of perfect managerial information 
4. Computational cost and weak identification

#### Defect 1: Timing of tax benefit of debt

#### Defect 2: Equity issuance cost

#### Defect 3: Unrealistic assumption of perfect managerial information

#### Defect 4: Computational cost and weak identification


---

## 2. Nested VFI (Grid Benchmark)

The discrete benchmark solver uses nested iteration. Starting from a candidate risky-rate schedule $\tilde{r}^{(n)}$, an inner loop solves the firm's Bellman equation by value function iteration and produces the converged outer iterate $V^{(n)}$. The lender zero-profit condition then uses this $V^{(n)}$ to determine the default states and update the pricing schedule to $\tilde{r}^{(n+1)}$.

**Setup: Discrete grids**

- Capital: $k \in \mathcal{K} = \{k_1, \dots, k_{N_k}\}$
- Debt: $b \in \mathcal{B} = \{b_1, \dots, b_{N_b}\}$
- Shock: $z \in \mathcal{Z} = \{z_1, \dots, z_{N_z}\}$ with Markov transition matrix $g_{jl} \equiv p(z_l | z_j)$, where $\sum_{l=1}^{N_z} g_{jl} = 1$

**Objects to solve for:**

$V(k_i, b_m, z_j)$: equity value on the grid

$\tilde{r}(z_j, k'_i, b'_m)$: risky rate for each current shock and next-period choice

### Algorithm

At outer iteration $n$, the pricing schedule $\tilde r^{(n)}$ is treated as fixed. The inner loop solves the Bellman equation under this fixed pricing schedule and produces the converged outer value iterate $V^{(n)}$. This $V^{(n)}$ determines the default states in next period, and those default states imply the updated pricing schedule $\tilde r^{(n+1)}$ through the lender's zero-profit condition. The outer loop therefore searches for a value function that is consistent with the pricing schedule implied by its own default states.

**Step 0. Initialization**

Set iteration counter $n = 0$. Initialize $\tilde{r}^{(0)}(z_j, k'_i, b'_m) = r$ for all $(j, i, m)$.

Set inner-loop counter $s = 0$. Initialize $V^{(s)}$ with all zeros.

**Step 1. Solve the Bellman problem under fixed $\tilde{r}^{(n)}$**

- Take a fixed $\tilde{r}^{(n)}$ as input. For each state $(k_i, b_m, z_j)$, evaluate:

    $$V^{(s+1)}(k_i, b_m, z_j) = \max\left\{0, \; \max_{k'_{i'} \in \mathcal{K}, \; b'_{m'} \in \mathcal{B}} \left[ e^{(n)}(k_i, k'_{i'}, b_m, b'_{m'}, z_j) - \Omega\!\left(e^{(n)}(\cdot)\right) + \frac{1}{1+r}\sum_{l=1}^{N_z} g_{jl} \, V^{(s)}(k'_{i'}, b'_{m'}, z_l) \right]\right\}$$

    where the function uses the **fixed** outer-loop interest rate $\tilde{r}^{(n)}$:

    $$e^{(n)}(k_i, k'_{i'}, b_m, b'_{m'}, z_j) = (1-\tau)\pi(k_i, z_j) - \psi(k'_{i'} - (1-\delta)k_i, \, k_i) - (k'_{i'} - (1-\delta)k_i) + \frac{b'_{m'}}{1 + \tilde{r}^{(n)}(z_j, k'_{i'}, b'_{m'})} - b_m + \frac{\tau \, \tilde{r}^{(n)}(z_j, k'_{i'}, b'_{m'}) \, b'_{m'}}{(1+\tilde{r}^{(n)})(1+r)}$$

- Operationally, the solver first computes the Bellman RHS on the full $(k', b')$ choice grid, then clamps the maximized value to zero if it is negative.

Repeat this value iteration, until $\|V^{(s+1)} - V^{(s)}\| < \epsilon_{\text{inner}}$.

When converged, store the result as $V^{(n)} \equiv V^{(s+1)}$. This is the inner-loop value iterate associated with the fixed pricing schedule $\tilde r^{(n)}$.

**Step 2. Recover the default partition implied by $V^{(n)}$**

For each $(k'_{i'}, b'_{m'})$, partition the **future** shock space:

$$\mathcal{D}(k'_{i'}, b'_{m'}) = \left\{z_l' \in \mathcal{Z} : V^{(n)}(k'_{i'}, b'_{m'}, z_l') = 0 \right\}$$

$$\mathcal{S}(k'_{i'}, b'_{m'}) = \left\{z_l' \in \mathcal{Z} : V^{(n)}(k'_{i'}, b'_{m'}, z_l') > 0 \right\}$$


**Step 3. Update the pricing schedule to $\tilde r^{(n+1)}$**

For each current shock and next-period choice triple $(z_j, k'_{i'}, b'_{m'})$, solve for $\tilde{r}^{(n+1)}$ from the lender's break-even condition:

$$b'_{m'}(1 + r) = \sum_{l \in \mathcal{D}(k'_{i'}, b'_{m'})} g_{jl} \, R(k'_{i'}, z_l) + b'_{m'}(1 + \tilde{r}^{(n+1)}) \sum_{l \in \mathcal{S}(k'_{i'}, b'_{m'})} g_{jl}$$

where $R(k', z') = (1-c_d)\left((1-\tau)\pi(k', z') + (1-\delta)k'\right)$.

Solving for $\tilde{r}^{(n+1)}$ explicitly:

$$\tilde{r}^{(n+1)}(z_j, k'_{i'}, b'_{m'}) = \frac{(1+r) - \frac{1}{b'_{m'}}\sum_{l \in \mathcal{D}} g_{jl} \, R(k'_{i'}, z_l)}{\sum_{l \in \mathcal{S}} g_{jl}} - 1$$

Note: if $\sum_{l \in \mathcal{S}} g_{jl} = 0$ (default in all states), the debt is worthless and set $\tilde{r}^{(n+1)} = \infty$ (or equivalently, the bond price $b'/(1+\tilde{r}) = 0$, meaning no lender will fund this $(k', b')$ pair).

**Step 4. Outer-loop convergence**

Resolve the Bellman problem in Step 1 under $\tilde r^{(n+1)}$ to obtain $V^{(n+1)}$, and stop when consecutive outer value iterates are close:

$$\left\|V^{(n+1)} - V^{(n)}\right\| < \epsilon_{\text{outer}} \quad \text{under} \quad \tilde r^{(n+1)}$$

Outer convergence is checked only after the Bellman solve under $\tilde r^{(n+1)}$ has itself converged. If not converged, set $n \leftarrow n+1$ and go to Step 1 and repeat.

### Remarks
In this method, the endogenous price $\tilde{r}^{(n+1)}$ is solved given the default/solvent partition $\mathcal{D}$ and $\mathcal{S}$, which in turn depends on the last converged $V^{(n)}$ from the inner loop. When both loops converged, the nested fixed point is reached.

The main cons of this method is computational cost. The object $\tilde{r}^{(n)}(z_j, k'_{i'}, b'_{m'})$ is a **three-dimensional array** of size $N_z \times N_k \times N_b$ that must be stored and updated each outer iteration. Each outer iteration triggers a full VFI (many inner iterations). And the inner VFI itself is $O(N_k^2 \times N_b^2 \times N_z)$ per iteration because for each state $(k_i, b_m, z_j)$ we search over all $(k'_{i'}, b'_{m'})$.

---

## Simulated Method of Moments


### Notation Recap

Define the following for observation $i$ at time $t$ with state $(k_{it}, b_{it}, z_{it})$ and choices $(k'_{it}, b'_{it})$:

- $e_{it} \equiv e(k_{it}, k'_{it}, b_{it}, b'_{it}, z_{it})$ — net cash flow 
- $I_{it} \equiv k'_{it} - (1-\delta)k_{it}$ — net investment
- $\psi(I_{it}, k_{it}) \equiv \frac{\psi_1}{2}(I_{it}/k_{it})^2 k_{it}$ — convex capital adjustment cost
- $\tilde{r}_{it} \equiv \tilde{r}(z_{it}, k'_{it}, b'_{it})$ — endogenous risky rate on new debt
- $V_{it} \equiv V(k_{it}, b_{it}, z_{it})$ — equity value
- $y_{it} \equiv z_{it} k_{it}^{\alpha - 1}$ — operating income / book real assets

Convention: $e_{it} > 0$ is distribution, $e_{it} < 0$ is equity issuance. $b'$ is face value; market value is $b'/(1+\tilde{r})$.

### Variables and Parameters

I apply SMM to structurally estimate the risky debt model in @strebulaev2012 [sec 3.6], which is a simplified version of the model in @hennessy2007costly. Following @hennessy2007costly, I use data from COMPUSTAT and choose the same set of variables:

<div align="center">

|Variables| Definition | 
|---|---|
| Investment/Book Real Assets | $[k'-(1-\delta)k]/k$ | 
| Cash Flow/Book Real Assets | $[(1-\tau) zk^\alpha  - b]/k$ |  
| Tobin's $q$ | $\left( V(k,b,z) + b \right)/k$ |
| Operating Income/Book Real Assets| $zk^\alpha / k$ |
| Debt/Market Value Real Assets| $\frac{b'/(1+\tilde{r}(k',b',z))}{V(k,b,z)+b'/(1+\tilde{r})}$  |
| Equity Issuance/Book Real Assets | $-e(\cdot)/k \quad \text{when }e<0$ |
| Dividend Payouts/Book Real Assets | $e(\cdot)/k \quad \text{when }e\geq0$ |

</div>

Note that the Tobin's $q$ and the Debt/Market Value would be defined slightly differently in @strebulaev2012 compared with the original model in @hennessy2007costly. @hennessy2007costly defined $b'$ as the market value of new debt, while in @strebulaev2012 the $b'$ is the face value. 

@hennessy2007costly's Tobin's $q$ as $[V + b \cdot (1+\tilde{r}(k,b,z_{-1})]/k$ using face value $(1+\tilde{r})b$, which makes it dependent on last period's $z_{t-1}$. @strebulaev2012 [p. 94] simplified this assumption explictly: "...firm takes the present value of the interest tax deduction in the period in which it issues debt...this feature greatly simplifies the determination of the risky interest rate. Otherwise, the tax deduction would depend on _last_ period's shock, and the current profits would then depend on _four_ state variables."

From these variables, I calculate moments similar as @hennessy2007costly [Table 1] to estimate these target parameters:

| Parameters | Role |
|---|---|
| $\alpha$ | Production function technology |
| $\eta_0$ | Fixed equity issuance cost |
| $\eta_1$ | Proportional equity issuance cost |
| $\eta_2$ | Quadratic equity issuance cost (optional) |
| $c_{\text{def}}$ | Deadweight default cost |
| $\psi_1$ | Convex capital adjustment cost |
| $\rho$ | Shock persistence |
| $\sigma_\varepsilon$ | Shock std dev |




### Moments

#### Equity issuance moments

To identify equity issuance cost, $\Omega(e)=(\eta_0 + \eta_1 |e| + \eta_2 e^2)\mathcal{I}\{e<0\}$, I attempt to match the following moments. Here $\max\{0,-e_{it}\}$ is a compact way to write equity issuance/distribution. The covariance between investment and equity implies the relative cost and priority of financing choices: when equity issuance is costly, the correlation is weak because firm will finance using internal funds and debt instead of equity issuance unless both are exhasted.

@strebulaev2012 implictly set zero quadratic issuance cost $\eta_2=0$. Although results from @hennessy2007costly supports this choice, I still include it as an exercise.

| # | Moment | Definition | Identifies | Reasons |
|---|---|---|---|---|
| 1 | Avg equity issuance / assets | $\mathbb{E}\bigl[\max(0, -e_{it}) / k_{it}\bigr]$ | $\eta_0$ | Infrequent issuance
| 2 | Var equity issuance / assets | $\text{Var}\bigl[\max(0, -e_{it}) / k_{it}\bigr]$ | $\eta_2$ | (optional)
| 3 | Freq of equity issuance | $\Pr(e_{it} < 0)$ | $\eta_0$ | Infrequent issuance
| 4 | Freq of negative debt | $\Pr(b'_{it} < 0)$ | $\eta_0, \eta_1$ | Precautionary saving
| 5 | Cov(investment, equity iss.) | $\text{Cov}\!\left(\frac{I_{it}}{k_{it}},\; \frac{\max(0,-e_{it})}{k_{it}}\right)$ | $\eta_0, \eta_1$ | Relative financing cost

#### Debt and leverage moments
The average net debt-to-asset ratio is used to estimate the deadweight costs of default. The convariance between investment and leverage is **weak or negative** when default cost is high, in which case firm will rely on other financing sources.  

| # | Moment | Definition | Identifies | Reasons |
|---|---|---|---|---|
| 6 | Avg net debt / assets | $\mathbb{E}\!\left[\frac{b'_{it}/(1+\tilde{r}_{it})}{V_{it} + b'_{it}/(1+\tilde{r}_{it})}\right]$ | $c_{\text{def}}$ | Cost of debt
| 7 | Cov(investment, leverage) | $\text{Cov}\!\left(\frac{I_{it}}{k_{it}},\; \frac{b'_{it}/(1+\tilde{r}_{it})}{V_{it} + b'_{it}/(1+\tilde{r}_{it})}\right)$ | $c_{\text{def}}$ | Relative financing cost

#### Adjustment cost moments
@strebulaev2012 considers a convex cost of capital adjustment and implictly set the fixed cost $\psi_0 \mathcal{I}\{ I\neq 0 \}$ to zero. I use the serial correlation of investments to identify the convex adjustment cost. Higher convex costs make full adjustment in one period too expensive, so the firm does partial adjustment (i.e., slowly moving to target capital level), which creates mechanical positive autocorrelation in $I/k$ that scales monotonically with $\psi_1$.

| # | Moment | Definition | Identifies | Reasons
|---|---|---|---|---|
| 8 | Serial corr of investment | $\text{Corr}\!\left(\frac{I_{it}}{k_{it}},\; \frac{I_{i,t-1}}{k_{i,t-1}}\right)$ | $\psi_1$ | Partial Adjustment


#### Real technology moments
The variance of investment rate directly captures the marginal product to capital. Firms respond less to shocks when the marginal product of capital is low as the production technology $\alpha$ is small. Because the previous moment #8 has only weak indirect effect on this moment, this allow for separate identification of $\alpha$ and $\psi_1$.

The two AR(1) parameters requires running a panel data regression. Denote the income/asset ratio as $y=zk^\alpha/k=zk^{\alpha-1}$. @hennessy2007costly run a first-order panel autoregression on a real panel data of firms indexed by $i$ in time $t$:

$$
\Delta y_{it} = \Delta \delta_t + \beta_1 \Delta y_{i,t-1} + \Delta u_{it}
$$

where $\Delta$ is one-period difference $\Delta y_{it} = y_{it} - y_{i,t-1}$. Because $\Delta y_{i,t-1}$ is serial correlated with $\Delta u_{it}$, I use lagged $y_{i,t-2}$ (or earlier) as instruments for $\Delta y_{i,t-1}$. The identifying assumption is $\mathbb{E}[y_{i,t-2}\cdot u_{it}]=0$ which is by definition consistent with AR(1) process.

The identified $\hat \beta_1$ is used for the AR(1) persistence moment to match $\rho$, and the standard deviation of the regression residual $\hat \sigma_u = std(\hat \mu_{it})$ is used to match $\sigma_\epsilon$. 

Note that the instrument is not needed for simulated data when by construction there is no serial correlation, but for comparibility I still apply the same procedure.

| # | Moment | Definition | Identifies | Reasons
|---|---|---|---|---|
| 9 | Var of investment / assets | $\text{Var}\bigl[I_{it} / k_{it}\bigr]$ |$\alpha, \psi_1$ | Marginal product of capital
| 10 | Serial corr of income / assets | $\hat{\beta}_1$ from panel AR(1) | $\rho$ | Direct
| 11 | Std dev of shock to income | $\hat{\sigma}_u$ (residual std dev from same AR(1)) | $\sigma_\varepsilon$ | Direct

#### Additional moments
@hennessy2007costly assumes nonzero tax on distribution (dividend payouts) and uses mean and variance of payout ratio to identify this parameter. @strebulaev2012's model simplifies this and assumes no dividend tax. I compute these two moments just as diagnostic.

| # | Moment | Definition | Identifies | Reasons
|---|---|---|---|---|
| - | Payout ratio | $\mathbb{E}\bigl[\max(0, e_{it}) / k_{it}\bigr]$ | $\tau_{div}=0$ | (diagnostic) 
| - | Var of distributions | $\text{Var}\bigl[\max(0, e_{it}) / k_{it}\bigr]$ | $\tau_{div}=0$| (diagnostic) 

#### Remarks

- **Parameter count:** Together, I use 11 moment conditions to identify 8 free parameters $(\theta, \psi_1, \eta_0, \eta_1, \eta_2, c_{\text{def}}, \rho, \sigma_\varepsilon)$. 
- **Overidentifying**: Because we have more moments than parameters, this allow for over-identification tests.


### SMM Estimation Diagnostics

This section documents findings from the two-step SMM estimation run on the risky debt model (`docs/06_risky_debt_smm_workflow.ipynb`). The SMM pipeline itself is validated on the frictionless basic-investment model (`docs/05_smm_validation.ipynb`), where bias < 0.003 and RMSE < 0.017 across all parameters. The model solver is benchmarked in `docs/03_risky_debt_vfi_interp.ipynb` and converges to machine-precision zero-profit residuals ($\sim 10^{-14}$). The oracle test at the true parameters passes, confirming that the solver-simulation-moment pipeline produces correct moments at the true $\beta$. Therefore, the issues below are properties of the estimation problem, not implementation bugs.

**Run configuration.** Grid: $N_k = 25, N_b = 50, N_z = 25$ ($N_{z,\text{solve}} = 10$). SMM: $N_{\text{firms}} = 5000$, $T = 25$, $S = 50$ panels, Stage 1 = differential evolution (maxiter = 50, nfev = 5483), Stage 2 = Powell (nfev = 369). Wall time: 175,418s.

#### Issue 1: Ill-conditioned $\hat\Omega$ and loss of the efficient estimator

**Definition.** The sample covariance of panel-level moment errors $\hat\Omega$ has condition number $2.82 \times 10^9$, exceeding the $10^6$ threshold. The solver falls back to $W = I$ for Stage 2, making the J-test and efficient standard errors unavailable.

**Cause.** With $S = 50$ panels and $R = 11$ moments, $\hat\Omega$ is an $11 \times 11$ matrix estimated from 50 observations ($S/R \approx 4.5$). The rule of thumb for stable covariance estimation is $S \gg R$. Additionally, the 11 moments span disparate scales: default frequency is $O(10^{-4})$ while leverage is $O(1)$, creating large eigenvalue spread even in the population covariance.

**Fix.** Increase $S$ to 200+ (recommended $S \geq 50R$) to reduce sampling noise in $\hat\Omega$. Alternatively, use diagonal weighting $W = \text{diag}(\hat\Omega)^{-1}$ as an intermediate step that avoids full matrix inversion. This is a computational budget issue, not a structural limitation.

#### Issue 2: Weak identification of $c_{\text{def}}$

**Definition.** The Jacobian column norm for $c_{\text{def}}$ is $\|D_{c_{\text{def}}}\| = 0.003$, compared to $\|D_{\sigma}\| = 8.77$ for the best-identified parameter. The corresponding singular value is $3 \times 10^{-4}$. The estimated SE is 0.39 on a parameter with true value 0.45 ($t = -0.17$, $p = 0.87$). No moment responds meaningfully to perturbations of $c_{\text{def}}$.

**Cause.** The default haircut $c_{\text{def}}$ affects lender recovery only conditional on default. At the estimated equilibrium, default frequency is $\sim 0.05\%$, so changes in $c_{\text{def}}$ have negligible impact on all 11 moments — including default frequency itself, whose Jacobian entry for $c_{\text{def}}$ is $0.0000$. This is a structural property of the model-moment combination: the equilibrium default rate is too low for the haircut to be identified from the chosen moments.

**Fix.** This is inherent to the model at the calibrated parameter values, not an implementation issue. Options: (a) calibrate $c_{\text{def}}$ externally from observed recovery rates and remove it from estimation; (b) add moments that are directly sensitive to default costs, such as credit spreads or loss-given-default; (c) recalibrate to a regime with higher default frequency where $c_{\text{def}}$ has stronger moment sensitivity.

#### Issue 3: Partial confounding among well-identified parameters

**Definition.** The Jacobian singular values decay sharply: $[11.7, 5.2, 3.1, 1.2, 0.26, 0.11, 0.0003]$. The 5th and 6th values ($0.26, 0.11$) are an order of magnitude below the leading ones, indicating two additional weakly identified parameter directions beyond $c_{\text{def}}$.

Inspecting the Jacobian columns, $\alpha$ and $\psi_1$ produce similar responses on the autocorrelation moments (AC Iss: $2.73$ vs $1.53$; Corr(Lev,Iss): $0.32$ vs $-1.98$), and $\eta_0$, $\eta_1$ both load heavily on the same equity issuance moments. Under $W = I$, the optimizer resolves these trade-offs by overshooting $\sigma_\varepsilon$ ($0.22$ vs true $0.15$) while undershooting $\alpha$ ($0.62$ vs $0.70$) — a compensating distortion across confounded directions.

**Cause.** With 7 estimated parameters and effectively $\sim$5 well-conditioned directions, the equal-weight objective has near-flat ridges that the optimizer can slide along. The resulting parameter estimates are biased along these ridges even if the objective value is near-optimal.

**Fix.** Partially addressable: (a) fixing $c_{\text{def}}$ reduces the problem to 6 parameters and removes the worst-conditioned direction; (b) obtaining a well-conditioned $\hat\Omega^{-1}$ (Issue 1) would reweight moments to better separate confounded parameters; (c) adding moments with orthogonal sensitivity patterns (e.g., moments that load on $\alpha$ but not $\psi_1$) would improve conditioning.

#### Issue 4: Moment fit at the estimate

**Definition.** Several moments are poorly matched at the Stage 2 estimate:

| Moment | Target | Fitted | Relative error |
|---|---|---|---|
| Cond Iss | 0.088 | 0.192 | +118% |
| Std Lev | 0.031 | 0.049 | +57% |
| AR1 $\sigma$ | 0.216 | 0.314 | +45% |
| Avg Iss/k | 0.017 | 0.024 | +40% |
| Var I/k | 0.029 | 0.040 | +38% |

Well-matched moments: AC Iss (+2%), AC I/k (~0%), Def Freq (-20% but at scale $10^{-4}$), AR1 $\beta$ (+7%).

**Cause.** The poorly matched moments cluster into dispersion/volatility moments (Std Lev, Var I/k, AR1 $\sigma$) and equity issuance moments (Cond Iss, Avg Iss/k). This pattern is consistent with the confounding in Issue 3: the overestimated $\sigma_\varepsilon$ inflates all dispersion moments, while the distorted issuance cost parameters ($\eta_0, \eta_1$) shift issuance moments. Under $W = I$, the optimizer sacrifices these moments to better match the correlation and autoregression moments that have larger absolute magnitudes and hence larger squared-error contributions.

**Fix.** Resolving Issues 1–3 (better weighting, fewer free parameters, better conditioning) would improve moment fit as a consequence. The poor fit is a symptom of the identification and weighting problems, not an independent issue.

#### Summary

The estimation difficulties are identification problems inherent to the model-moment combination at the calibrated equilibrium, not pipeline bugs. The oracle test, validated SMM pipeline, and full-rank Jacobian confirm mechanical correctness. The primary actionable step is to fix $c_{\text{def}}$ externally, which simultaneously improves conditioning (removing the worst direction) and reduces the parameter space to a dimension better supported by the available moments. Increasing $S$ for a stable $\hat\Omega$ is the secondary priority.
