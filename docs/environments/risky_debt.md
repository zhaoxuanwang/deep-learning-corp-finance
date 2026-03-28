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

where $C_b$ is a user-specified multiplier and should be set generous enough that it never binds (e.g., $C_b=8$ means firm can borrow up to 8 times of its size).


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
- $\Psi$: capital adjustment cost
- $I=k'-(1-\delta)k$: investment expenditure
- $b$: debt repayment (determined from last period)
- $b'$: face value of next-period debt
- $\tilde{r}(k',b',z)$: endogenous interest rate on debt market
- $r$: risk-free interest rate
- $\tau \cdot \tilde{r} \cdot b'/(1+r)(1+\tilde{r})$: present-value tax shield from debt. Firms exploit debt financing to shield profit from taxation.

**External equity cost.** When cash flow is negative, the firm must raise costly external equity:
$$\Omega(e) = \left(\eta_0 + \eta_1 |e|\right)\mathbf{1}\{e < 0\}, \quad \eta_0\geq 0, \eta_1 \geq0$$
where $\eta_0$ is the fixed issuance cost and $\eta_1$ is the proportional cost on the financing shortfall.


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

**Step 1. Solve the Bellman problem under fixed $\tilde{r}^{(n)}$**

Set inner-loop counter $s = 0$. Initialize $V^{(s)}$ arbitrarily (e.g., all zeros).

For each state $(k_i, b_m, z_j)$:

$$V^{(s+1)}(k_i, b_m, z_j) = \max\left\{0, \; \max_{k'_{i'} \in \mathcal{K}, \; b'_{m'} \in \mathcal{B}} \left[ e^{(n)}(k_i, k'_{i'}, b_m, b'_{m'}, z_j) - \Omega\!\left(e^{(n)}(\cdot)\right) + \frac{1}{1+r}\sum_{l=1}^{N_z} g_{jl} \, V^{(s)}(k'_{i'}, b'_{m'}, z_l) \right]\right\}$$

where the function uses the **fixed** outer-loop interest rate $\tilde{r}^{(n)}$:

$$e^{(n)}(k_i, k'_{i'}, b_m, b'_{m'}, z_j) = (1-\tau)\pi(k_i, z_j) - \psi(k'_{i'} - (1-\delta)k_i, \, k_i) - (k'_{i'} - (1-\delta)k_i) + \frac{b'_{m'}}{1 + \tilde{r}^{(n)}(z_j, k'_{i'}, b'_{m'})} - b_m + \frac{\tau \, \tilde{r}^{(n)}(z_j, k'_{i'}, b'_{m'}) \, b'_{m'}}{(1+\tilde{r}^{(n)})(1+r)}$$

Operationally, the solver first computes the Bellman RHS on the full $(k', b')$ choice grid, then clamps the maximized value to zero if it is negative.

Repeat until $\|V^{(s+1)} - V^{(s)}\| < \epsilon_{\text{inner}}$.

When converged, store the result as $V^{(n)} \equiv V^{(s+1)}$. This is the outer-loop value iterate associated with the fixed pricing schedule $\tilde r^{(n)}$.

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

$$\left\|V^{(n+1)} - V^{(n)}\right\| < \epsilon_{\text{outer}}$$

Outer convergence is checked only after the Bellman solve under $\tilde r^{(n+1)}$ has itself converged. If not converged, set $n \leftarrow n+1$ and go back to the inner loop.

The canonical implementation follows this algorithm literally: it uses the discrete Markov matrix $g_{jl}$ directly in the Bellman expectation and in the zero-profit update, interprets default as a next-period event determined by $(k', b', z')$, and each outer iteration replaces $\tilde r^{(n)}$ with the newly solved $\tilde r^{(n+1)}$ without damping or auxiliary interpolation tricks. Changes in the implied pricing schedule and zero-profit residuals are reported as diagnostics, but the primary outer-loop convergence object is the value function.

### Remarks
In this method, the endogenous price $\tilde{r}^{(n+1)}$ is solved given the default/solvent partition $\mathcal{D}$ and $\mathcal{S}$, which in turn depends on the last converged $V^{(n)}$ from the inner loop. When both loops converged, the nested fixed point is reached.

The main cons of this method is computational cost. The object $\tilde{r}^{(n)}(z_j, k'_{i'}, b'_{m'})$ is a **three-dimensional array** of size $N_z \times N_k \times N_b$ that must be stored and updated each outer iteration. Each outer iteration triggers a full VFI (many inner iterations). And the inner VFI itself is $O(N_k^2 \times N_b^2 \times N_z)$ per iteration because for each state $(k_i, b_m, z_j)$ we search over all $(k'_{i'}, b'_{m'})$.
