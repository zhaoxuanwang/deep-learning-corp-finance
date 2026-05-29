# Nikolov, Schmid, and Steri (2021): Three Theoretical Models

## Technology and Notation

This chapter implements linear programming (LP) methods to solve the three structural corporate finance models in Nikolov21:
- Trade-Off Model (TO)
- Limited Enforcement Model (LE)
- Moral Hazard Model (MH)

All three models share the same production technology, capital accumulation rule, and adjustment cost specification.

**Operating profit (pre-tax):**
$$\pi(k_{it}, z_{it}, \eta_{it}) = (z_{it} + \eta_{it})k_{it}^{\alpha} - f$$

After-tax operating cash flow is $(1-\tau)\pi$. Paper Eq. 1 labels the after-tax expression $(1-\tau)((z+\eta)k^\alpha - f)$ as $\pi$, but every subsequent equation uses $(1-\tau)\pi$, so we treat $\pi$ as pre-tax throughout this doc for consistency.

**Capital accumulation (Eq. 2):**
$$k_{it+1} = (1-\delta)k_{it} + i_{it}$$

**Adjustment cost (Eq. 3):**
$$\Psi(k_{it+1}, k_{it}) = \frac{\psi}{2}\left(\frac{i_{it}}{k_{it}}\right)^2 k_{it} = \frac{\psi}{2}\left(\frac{k_{it+1} - (1-\delta)k_{it}}{k_{it}}\right)^2 k_{it}$$

**Variables and parameters:**
- $k_{it}$: capital stock of firm $i$ at time $t$
- $i_{it}$: investment at $t$
- $z_{it} \in [\underline{z}, \overline{z}]$: persistent profitability shock, transition $Q(z'|z)$
- $\eta_{it} \in \{+\bar\eta, -\bar\eta\}$: i.i.d. disturbance, $P(\eta = +\bar\eta) = \kappa$
- $\tau \in (0, 1)$: corporate tax rate
- $\alpha \in (0, 1)$: capital share (decreasing returns)
- $f > 0$: fixed production cost
- $\delta \in (0, 1)$: capital depreciation rate
- $\psi$: adjustment cost parameter
- $r$: risk-free interest rate
- $\tau\delta k_{it}$: depreciation tax allowance

**Capital cost convention.** In the paper's Bellman and dividend formulas, the investment-related terms appear as $-k_{it} + (1-\delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it}$, which simplifies to $-(1-\tau)\delta k_{it} - \Psi$. The paper treats capital as continuously maintained at level $k_{it}$ (paying $\delta k_{it}$ per period for depreciation replacement), with adjustment cost paid when changing the level. This is non-standard but internally consistent; the LP follows this convention.

### Firm value $W$ and equity value $V$

All three models maximize **firm value** $W(s_{it}, z_{it})$ at the end of period $t$: the PV of future cash flows to equity AND lenders combined, from $t+1$ onward. In schematic form, $W$ satisfies a Bellman recursion of the form
$$W(s, z) = \beta \max_{a}\, E\big[d(s, a, s', z', \eta') + W(s', z')\big]$$
where $s$ is the model-specific state, $a$ is the action, and $d$ is the dividend or cash flow (with adjustments for default in TO).

**Equity value** $V_{it}$ is the PV of dividends to shareholders alone, satisfying
$$V_{it} = \beta\, E\big[d_{z', \eta'} + V_{z', \eta'}\big]$$

In TO and LE, $V$ is not tracked explicitly; firm value and equity value are linked by $W_{it} = V_{it} + V^{\text{debt}}_{it}$ where $V^{\text{debt}}_{it}$ is the value of outstanding debt. In MH, incentive compatibility requires the firm to commit to a specific continuation equity value going forward, so $V$ becomes a state variable in the MH Bellman.

---

## Timing (common across all three models)

Within and across periods, events unfold sequentially as summarized below, following Figure 1 of the paper. The key convention is that $W(s_{it}, z_{it})$ is evaluated at the **end of period $t$**, after period-$t$ shocks, production, repayments/transfers, and default checks have occurred, but before period-$(t+1)$ shocks are realized. Decisions made at the end of period $t$ determine capital and financing/contract terms for period $t+1$.

**State versus realized transfers.** Persistent recursive states are $s=(k,b)$ in TO and LE and $s=(k,V)$ in MH. State-contingent payments such as $p$ in LE and dividends $d$ in MH are realized transfers chosen as part of the previous period's state-contingent contract; they are not persistent state variables except through the continuation balance $b$ or continuation equity value $V$.

| Point in time | Event(s) | TO | LE | MH |
|---|---|---|---|---|
| End of $t-1$ | Firm makes decisions to maximize $W(s_{i,t-1}, z_{i,t-1})$ | Choose capital $k_{it}$ and debt position $b_{it}$ to be carried into period $t$; lender break-even determines the spread attached to this debt contract. | Choose capital $k_{it}$ and a state-contingent contract $\{b_{z_{it},\eta_{it}}, p_{z_{it},\eta_{it}}\}$ for period-$t$ realizations. | Choose capital $k_{it}$ and a state-contingent contract $\{V_{z_{it},\eta_{it}}, d_{z_{it},\eta_{it}}\}$ for period-$t$ realizations. |
| Start/end of $t$ | Shocks $(z_{it}, \eta_{it})$ realize; production and contractual transfers occur; the next recursive state is formed. | Production uses $k_{it}$; the firm repays the debt obligation carried into period $t$ if solvent, otherwise defaults and is liquidated. The next decision state, conditional on survival, is $(k_{it}, b_{it}, z_{it})$. | The realized contract specifies payment $p_{z_{it},\eta_{it}}$ and continuation balance $b_{z_{it},\eta_{it}}$; no default occurs by design. The next decision state is $(k_{it}, b_{z_{it},\eta_{it}}, z_{it})$. | Shareholders observe $\eta_{it}$, lenders do not; the realized contract specifies dividend $d_{z_{it},\eta_{it}}$ and continuation equity value $V_{z_{it},\eta_{it}}$; no default occurs by design. The next decision state is $(k_{it}, V_{z_{it},\eta_{it}}, z_{it})$. |
| End of $t$ | Firm makes decisions to maximize $W(s_{it}, z_{it})$ | Choose capital $k_{i,t+1}$ and debt position $b_{i,t+1}$ to be carried into period $t+1$; lender break-even determines the spread attached to this debt contract. | Choose capital $k_{i,t+1}$ and a state-contingent contract $\{b_{z_{i,t+1},\eta_{i,t+1}}, p_{z_{i,t+1},\eta_{i,t+1}}\}$ for period-$(t+1)$ realizations. | Choose capital $k_{i,t+1}$ and a state-contingent contract $\{V_{z_{i,t+1},\eta_{i,t+1}}, d_{z_{i,t+1},\eta_{i,t+1}}\}$ for period-$(t+1)$ realizations. |
| Start/end of $t+1$ | Shocks $(z_{i,t+1}, \eta_{i,t+1})$ realize; production and contractual transfers occur; the next recursive state is formed. | Production uses $k_{i,t+1}$; the firm repays the debt obligation carried into period $t+1$ if solvent, otherwise defaults and is liquidated. The next decision state, conditional on survival, is $(k_{i,t+1}, b_{i,t+1}, z_{i,t+1})$. | The realized contract specifies payment $p_{z_{i,t+1},\eta_{i,t+1}}$ and continuation balance $b_{z_{i,t+1},\eta_{i,t+1}}$; no default occurs by design. The next decision state is $(k_{i,t+1}, b_{z_{i,t+1},\eta_{i,t+1}}, z_{i,t+1})$. | Shareholders observe $\eta_{i,t+1}$, lenders do not; the realized contract specifies dividend $d_{z_{i,t+1},\eta_{i,t+1}}$ and continuation equity value $V_{z_{i,t+1},\eta_{i,t+1}}$; no default occurs by design. The next decision state is $(k_{i,t+1}, V_{z_{i,t+1},\eta_{i,t+1}}, z_{i,t+1})$. |

The exact TO default condition, LE break-even/collateral constraints, and MH promise-keeping/incentive constraints are defined in the model-specific sections below. The timing table intentionally states the schedule in words rather than duplicating those equations.

---

# Section 1: Trade-off Model

## 1.1 Theoretical Model

In this setup, $\eta_{it}$ is **public information**.

### Financing

- Firms issue one-period bonds: cash inflow $b_{it+1}$ at the beginning of period $t+1$; previously issued bonds $b_{it}$ are due with interest.
- Default premium $\Delta_{it}$ is charged above $r$, so the effective interest rate is $r + \Delta_{it}$.
- Interest payments are tax deductible: effective repayment due in period $t+1$ is $(1 + (1 - \tau)(r + \Delta_{it}))b_{it}$.
- Tax shield: $\tau(r + \Delta_{it})b_{it}$.

**Solvency condition** (firm is solvent iff):

$$(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) + (1 - \delta)k_{it} + \tau\delta k_{it} - (1 + (r + \Delta_{it-1})(1 - \tau))b_{it-1} \geq 0 \quad (4)$$

**Default set**:

$$D_{it} \equiv \{(z_{it}, \eta_{it}, k_{it}, \Delta_{it-1}) \in \overline{Z} \times \overline{N} \times \mathbb{R}^+ \times \mathbb{R}^+ : (4) \text{ does not hold}\}$$

- $\overline{D}_{it}$: set of solvency states (where Eq. 4 holds)
- $\mathcal{I}_{D,it}$: indicator function for default

**Creditor break-even condition** (risk-neutral pricing):

$$E_{t-1}\left[ (1 + r + \Delta_{it-1})(1 - \mathcal{I}_{D,it}) + \frac{\xi(1 - \delta)k_{it}}{b_{it-1}}\mathcal{I}_{D,it} \right] = 1 + r$$

- $\xi$: recovery rate in bankruptcy

**Payouts** (limited liability, seasoned equity precluded):

$$d_{it} \equiv (1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) - k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it} - (1 + (r + \Delta_{it-1})(1 - \tau))b_{it-1} + b_{it} \geq 0$$

### Firm Problem

State variables: $(k_{it-1}, b_{it-1}, z_{it-1})$. Bellman equation:

$$W(k_{it-1}, b_{it-1}, z_{it-1}) \equiv \frac{1}{1 + r} \max_{k_{it}, b_{it}} \Big\{ -k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it}$$

$$+ \tau(r + \Delta_{it-1})b_{it-1}\mathcal{I}_{1-D,it} - ((1 - \xi)(1 - \delta)k_{it} + \tau\delta k_{it})\mathcal{I}_{D,it}$$

$$+ E_{t-1}\big[(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) + W(k_{it}, b_{it}, z_{it})\big] \Big\}$$

subject to:

$$(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) - k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it} - (1 + (r + \Delta_{it-1})(1 - \tau))b_{it-1} + b_{it} \geq 0, \quad \forall z_{it}, \eta_{it}$$

$$E_{t-1}\left[ (1 + r + \Delta_{it-1})(1 - \mathcal{I}_{D,it}) + \frac{\xi(1 - \delta)k_{it}}{b_{it-1}}\mathcal{I}_{D,it} \right] = 1 + r$$

## 1.2 LP Solution Method

The baseline LP solutions (TO, LE, MH) use **NumPy** for array operations, **SciPy sparse matrices** for constraint assembly, and **`scipy.optimize.linprog` with the `'highs'` backend** for the master LP solve. TensorFlow is not used: there is no neural-net training, no autodiff requirement, and no GPU benefit for the modest finite-grid baseline. The workload is dominated by grid construction, finite-action enumeration or action filtering, sparse matrix assembly, and LP solving, all of which NumPy and SciPy handle with mature, well-tested implementations.

**LP state interpretation.** The LP uses the paper's recursive timing. A state $(k,b,z)$ (or $(k,V,z)$ in MH) is the inherited end-of-period state: $k$ is the capital level from the previous choice, $b$ is the outstanding promised balance (or $V$ is promised equity value), and $z$ is the current public persistent shock. At this state the firm chooses the next capital level and financing contract, denoted $(k',b')$ in TO, state-contingent $(b'_{z',\eta'},p_{z',\eta'})$ in LE, and state-contingent $(V'_{z',\eta'},d_{z',\eta'})$ in MH. The Bellman target then evaluates next-period shocks $(z',\eta')$, operating cash flow with the chosen capital $k'$, repayment/contract constraints, and continuation value at the chosen next state.

**Baseline implementation modules.** A coding agent should keep the implementation modular: (i) `grids` for $K$, $B$, $V$, payment/dividend grids, and shock transition matrices; (ii) `primitives` for $\pi$, $\Psi$, discount factors, and model parameters; (iii) `timing`/index helpers that map current state, action, shock realization, and continuation state; (iv) TO `pricing` for $\Delta(k_{\text{choice}},b_{\text{old}},z_{\text{old}})$ and default indicators; (v) LE/MH `action_enumeration` for fixed finite contract menus and feasibility filtering; (vi) `lp_assembly` for sparse Bellman inequalities; (vii) `lp_solve` for HiGHS; (viii) `policy_recovery` for argmax policies; and (ix) `diagnostics` for feasibility counts, LP residuals, pricing residuals, and boundary-hit warnings.

### Primitives and grids

State $S = K \times B \times Z$ with discrete grids $K = \{k_1, \dots, k_{n_k}\}$, $B = \{b_1, \dots, b_{n_b}\}$, $Z = \{z_1, \dots, z_{n_z}\}$. Action $A = K \times B$ (choice of next-period $(k', b')$).

State $(k, b, z)$: capital in place, outstanding debt, current persistent shock. I.i.d. shock $\eta \in \{+\bar\eta, -\bar\eta\}$ with $P(\eta = +\bar\eta) = \kappa$. Persistent transition $Q(z' \mid z)$. Discount $\beta = 1/(1+r)$.

Model functions:
$$\pi(k, z, \eta) = (z + \eta)k^\alpha - f, \qquad \Psi(k', k) = \tfrac{\psi}{2}\big((k' - (1-\delta)k)/k\big)^2 k$$

Parameters: $\tau, \alpha, f, \delta, \psi, \xi, r, \kappa, \bar\eta$.

### Step 1: Pre-compute lender pricing $\Delta(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$

For every candidate chosen capital $k_{\text{choice}} \in K$, inherited debt $b_{\text{old}} \in B$, and inherited persistent shock $z_{\text{old}} \in Z$, compute the risky-debt premium that satisfies the lender break-even condition:

$$\sum_{z'} Q(z' \mid z_{\text{old}}) \sum_{\eta'} P(\eta')\Big[(1+r+\Delta)(1 - \mathcal{I}_D') + \tfrac{\xi(1-\delta)k_{\text{choice}}}{b_{\text{old}}}\mathcal{I}_D'\Big] = 1+r$$

with

$$\mathcal{I}_D' = \mathbf{1}\big\{(1-\tau)\pi(k_{\text{choice}}, z', \eta') + (1-\delta)k_{\text{choice}} + \tau\delta k_{\text{choice}} \;<\; (1+(1-\tau)(r+\Delta))b_{\text{old}}\big\}.$$

**Important timing convention.** In the Bellman constraint for state $(k,b,z)$ and action $(k',b')$, the relevant premium on the outstanding debt $b$ is $\Delta(k',b,z)$, not $\Delta(k,b,z)$. The premium is priced at the time the old debt is issued, using the capital level chosen for the period in which that debt will be repaid.

**Robust finite-shock pricing solver.** Do not rely on naive bisection as the baseline. Because $\mathcal{I}_D'$ changes with $\Delta$ on a discrete shock grid, the lender payoff is piecewise linear with possible jumps. Implement pricing as follows:

1. If $b_{\text{old}}=0$, set $\Delta=0$ and $\mathcal{I}_D'=0$.
2. If $b_{\text{old}}>0$, compute the default-switch threshold for each future shock realization:
   $$\Delta^*(z',\eta') = \frac{(1-\tau)\pi(k_{\text{choice}},z',\eta')+(1-\delta)k_{\text{choice}}+\tau\delta k_{\text{choice}}}{(1-\tau)b_{\text{old}}} - \frac{1}{1-\tau} - r.$$
   A realization defaults when $\Delta > \Delta^*(z',\eta')$.
3. Sort the finite set of thresholds and examine the intervals over which the default set is fixed.
4. On each interval, solve the break-even equation analytically because the default set is fixed and the expected lender payoff is linear in $\Delta$.
5. Keep the lowest nonnegative $\Delta$ whose implied default set is internally consistent and whose expected payoff equals $1+r$ up to tolerance. If discreteness prevents exact equality, use the lowest nonnegative $\Delta$ for which expected lender payoff is weakly at least $1+r$, and record the pricing residual.

**Output:**
- 3D premium table $\Delta(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$ of size $n_k \cdot n_b \cdot n_z$.
- Optional default indicator table $\mathcal{I}_D(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}}, z', \eta')$, evaluated at the stored premium.

In practice, materialize only the 3D premium table unless debugging. Given the stored $\Delta(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$ and the analytic default condition, recompute the indicator on demand to save memory.

### Step 2: LP

**Variables.** $W(k, b, z)$ for every $(k, b, z) \in S$. Count: $n_k n_b n_z$.

**Objective.**
$$\min_W \sum_{(k, b, z) \in S} W(k, b, z)$$

**Bellman constraints.** Following the paper's appendix Eq. (22), for every $(k, b, z) \in S$ and every feasible $(k', b') \in A$:

$$W(k, b, z) \geq \tfrac{1}{1+r}\bigg[\underbrace{-k' + (1-\delta)k' - \Psi(k', k) + \tau\delta k'}_{\text{deterministic, paid regardless of default}} + \sum_{z', \eta'} Q(z' \mid z) P(\eta')\bigg(\underbrace{(1-\tau)\pi(k', z', \eta')}_{\text{profit, always realized}} + \underbrace{\tau(r+\Delta(k', b, z))b\, \mathcal{I}_{1-D}}_{\text{tax shield if solvent}} - \underbrace{(1-\xi)\big((1-\delta)k' + \tau\delta k'\big)\mathcal{I}_D}_{\text{deadweight loss if default}} + \underbrace{(1-\mathcal{I}_D)\, W(k', b', z')}_{\text{continuation if solvent}}\bigg)\bigg]$$

where the default indicator $\mathcal{I}_D = \mathcal{I}_D(k', b, z, z', \eta')$ at the next-shock realization is

$$\mathcal{I}_D = \mathbf{1}\big\{(1-\tau)\pi(k', z', \eta') + (1-\delta)k' + \tau\delta k' \;<\; (1+(1-\tau)(r+\Delta(k', b, z)))b\big\}$$

and $\mathcal{I}_{1-D} = 1 - \mathcal{I}_D$. The tax shield is on the OUTSTANDING bond $b$ with its premium $\Delta(k', b, z)$, per the paper's Section 2.2 financing description and the timing convention above.

**Limited liability (feasibility filter).** A pair $(k', b')$ is feasible at $(k, b, z)$ only if shareholder dividend $d \geq 0$ in every solvent realization $(z', \eta')$:
$$d = (1-\tau)\pi(k', z', \eta') - k' + (1-\delta)k' - \Psi(k', k) + \tau\delta k' - (1+(1-\tau)(r+\Delta(k', b, z)))b + b' \geq 0, \quad \forall (z', \eta') \text{ with } \mathcal{I}_D = 0$$

Drop the Bellman constraint at $(k, b, z, k', b')$ if LL is violated. Default-state dividends are zero by limited liability (firm wiped out, equity holders receive nothing).

Constraint count (post-filter): bounded by $n_k^2 n_b^2 n_z$, sparser after filtering.

### Step 3: Recover the policy

After the LP returns $W^*$:
$$(k', b')^*(k, b, z) = \arg\max_{(k', b') \in A \text{ feasible}} \text{RHS}(k, b, z, k', b'; W^*)$$
where RHS is the same Bellman target as in Step 2, with $W^*$ plugged in for the continuation. Implied premium policy for the chosen next debt is $\Delta^*(k,b,z)=\Delta(k^{\prime *}(k,b,z), b^{\prime *}(k,b,z), z)$; within the Bellman RHS for repayment of inherited debt $b$, use $\Delta(k^{\prime},b,z)$.

### Implementation notes

- **Step 1:** finite-threshold pricing over the $(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$ grid; vectorize threshold construction where possible, but keep explicit consistency checks for default sets and pricing residuals.
- **Step 2 LP construction:** NumPy broadcasting to build the $(n_k, n_b, n_z, n_k, n_b)$ Bellman RHS coefficient tensor and LL feasibility mask; assemble constraints as `scipy.sparse.csc_matrix`.
- **Step 2 LP solve:** `scipy.optimize.linprog` with `method='highs'`.
- **Step 3:** `np.argmax` over the RHS tensor with $W^*$ plugged in.

---

# Section 2: Limited Enforcement Model

## 2.1 Theoretical Model

State-contingent payoffs are allowed. In this context, $\eta_{it}$ is **public information**.

### Financing

- Firms sell a portfolio of securities whose payoffs are contingent on next-period shocks $z_{it+1}$ and $\eta_{it+1}$.
- Selling the portfolio at time $t$ raises:

$$b_{it} \equiv \frac{1}{1 + r}E_t[p_{z_{it+1}, \eta_{it+1}} + b_{z_{it+1}, \eta_{it+1}}]$$

- $p_{z_{it+1}, \eta_{it+1}}$: cash flow transferred to investors contingent on shocks
- $b_{z_{it+1}, \eta_{it+1}}$: residual present value of future promised repayments

Intuitively, the contract operates like a flexible credit line: $b_{it}$ is the outstanding balance today; $p_{z_{it+1}, \eta_{it+1}}$ is the payment the firm makes next period contingent on the realized state; and $b_{z_{it+1}, \eta_{it+1}}$ is the new outstanding balance after that payment, carried forward as the firm's debt at $t+1$.

**Collateral constraint** (state-contingent debt must be fully collateralized):

$$p_{z_{it+1}, \eta_{it+1}} + b_{z_{it+1}, \eta_{it+1}} \leq \theta(1 - \delta)k_{it+1}, \quad \forall z_{it+1}, \eta_{it+1}$$

- $\theta$: fraction of capital that can be pledged as collateral

**Limited liability (payouts)**:

$$d_{it} \equiv (1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) - k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it} + \tau r b_{it-1} - p_{z_{it}, \eta_{it}} \geq 0$$

### Firm Problem

Bellman equation:

$$W(k_{it-1}, b_{it-1}, z_{it-1}) = \frac{1}{1 + r}\max_{k_{it}, b_{z_{it}, \eta_{it}}, p_{z_{it}, \eta_{it}}} \Big\{ -k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it}$$

$$+ \tau r b_{it-1} + E_{t-1}\big[(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) + W(k_{it}, b_{z_{it}, \eta_{it}}, z_{it})\big] \Big\}$$

subject to:

$$b_{it-1} \equiv \frac{1}{1 + r}E_{t-1}[p_{z_{it}, \eta_{it}} + b_{z_{it}, \eta_{it}}] \quad (5)$$

$$(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) - k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it} + \tau r b_{it-1} - p_{z_{it}, \eta_{it}} \geq 0, \quad \forall z_{it}, \eta_{it} \quad (6)$$

$$p_{z_{it}, \eta_{it}} + b_{z_{it}, \eta_{it}} \leq \theta(1 - \delta)k_{it}, \quad \forall z_{it}, \eta_{it} \quad (7)$$

## 2.2 LP Solution Method

### Primitives and grids

State $S = K \times B \times Z$ with the same $K$ and $Z$ grids as 1.2 and a nonnegative debt grid $B = \{b_1,\dots,b_{n_b}\} \subset \mathbb{R}_+$ with $0 \in B$. Additional parameter: $\theta$ (collateral fraction). Shock $\eta$ and persistent shock $z$ are as in 1.2.

State $(k,b,z)$: inherited capital, inherited promised balance, and current persistent shock. There is no default risk and no premium $\Delta$ in LE. Pricing is imposed by the risk-neutral break-even constraint (5).

### Minimal finite-action baseline

To keep the baseline a true LP, use a **finite contract-action menu**. For each current state $(k,b,z)$ and candidate next capital $k' \in K$, a contract action is a fixed collection

$$a = \{b'_{z',\eta'}, p_{z',\eta'}\}_{z' \in Z,\eta' \in N}$$

where each $b'_{z',\eta'} \in B$ and each $p_{z',\eta'} \in P$, with $P=\{p_1,\dots,p_{n_p}\}\subset \mathbb{R}_+$ a nonnegative payment grid. Because the contract components are fixed before a Bellman inequality is added, the continuation terms $W(k',b'_{z',\eta'},z')$ enter linearly with known coefficients.

Do **not** include convex-combination weights such as $\lambda_j W(k',b_j,z')$ as free variables in the master LP. That would be bilinear, not linear. Continuous-state interpolation can be added later only through constraint generation, where interpolation weights are fixed constants when each Bellman inequality is added.

### Step 1: Build feasible contract-action lists

For every $(k,b,z,k')$, enumerate candidate contract actions $a=\{b'_{z',\eta'},p_{z',\eta'}\}$. Keep only actions satisfying all constraints below.

- **Break-even (Eq. 5):**
$$\frac{1}{1+r}\sum_{z', \eta'} Q(z' \mid z) P(\eta')[p_{z', \eta'} + b'_{z', \eta'}] = b.$$

In code, because $P$ and $B$ are discrete, use a tight tolerance `be_tol`:
$$\left|\frac{1}{1+r}\sum_{z', \eta'} Q(z' \mid z) P(\eta')[p_{z', \eta'} + b'_{z', \eta'}] - b\right| \leq \texttt{be\_tol}.$$

- **Limited liability (Eq. 6):** for each $(z',\eta')$,
$$p_{z', \eta'} \leq (1-\tau)\pi(k', z', \eta') - k' + (1-\delta)k' - \Psi(k', k) + \tau\delta k' + \tau r b.$$

- **Collateral (Eq. 7):** for each $(z',\eta')$,
$$p_{z', \eta'} + b'_{z', \eta'} \leq \theta(1-\delta)k'.$$

- **Sign restrictions:**
$$p_{z',\eta'} \geq 0, \qquad b'_{z',\eta'} \geq 0.$$

A tuple $(k,b,z,k')$ contributes no Bellman constraint if no feasible contract action exists for that particular $k'$. However, every state $(k,b,z)$ must have at least one feasible action across all $k'$. If a state has no feasible action, treat this as a grid-design failure: enlarge the payment grid $P$, enlarge/shift the debt grid $B$, relax only the numerical tolerance if the miss is purely rounding error, or remove the state from the admissible state grid. Do not solve the LP with states that have an empty feasible action correspondence.

For the minimal baseline, this brute-force enumeration is intended for small grids only. A faster second-stage implementation can replace enumeration with constraint generation or an auxiliary search routine, but the master LP must still receive only fixed actions.

### Step 2: Master LP

**Variables.** $W(k,b,z)$ for every $(k,b,z)\in S$. Count: $n_k n_b n_z$.

**Objective.**
$$\min_W \sum_{(k,b,z)\in S} W(k,b,z).$$

**Bellman constraints.** For every state $(k,b,z)$ and every feasible fixed action $(k',a)$:

$$W(k,b,z) \geq \frac{1}{1+r}\bigg[-k' + (1-\delta)k' - \Psi(k',k) + \tau\delta k' + \tau r b + \sum_{z',\eta'} Q(z'\mid z)P(\eta')\Big((1-\tau)\pi(k',z',\eta') + W(k',b'_{z',\eta'},z')\Big)\bigg].$$

The entire deterministic term is inside the discount factor $1/(1+r)$, matching the paper's LE Bellman equation.

Constraint count depends on the number of feasible fixed contract actions. This can grow quickly with $n_z n_\eta$, so the minimal implementation should start with small grids.

### Step 3: Recover the policy

After the LP returns $W^*$, choose the feasible fixed action that maximizes the RHS at each state:

$$(k',a)^*(k,b,z)=\arg\max_{(k',a)\in\mathcal{A}_{LE}(k,b,z)} \text{RHS}_{LE}(k,b,z,k',a;W^*).$$

The policy consists of $k'^*(k,b,z)$ and the associated state-contingent contract $\{b'^*_{z',\eta'},p^*_{z',\eta'}\}$.

### Implementation notes

- **Baseline:** enumerate finite contract actions on small $B$ and $P$ grids, filter by break-even, limited liability, collateral, and sign restrictions, then assemble the sparse LP.
- **Correctness condition:** each Bellman inequality must correspond to a fixed feasible action. The continuation coefficient on each $W(k',b_j,z')$ is therefore a known probability weight, not a decision variable.
- **Scaling upgrade:** if enumeration becomes too large, use constraint generation. Given a current value vector $W^{(m)}$, solve a separate action-search problem for each state, freeze the selected action, add the corresponding linear Bellman inequality to the master LP, and repeat until no violated constraints remain.

---

# Section 3: Moral Hazard Model

## 3.1 Theoretical Model

Asymmetric information setup:
- $z_{it}$ follows a Markov chain that is **publicly observable** (also by the lender).
- $\eta_{it}$ is **observable by shareholders but unobservable by lenders**.

A lending contract is a sharing rule splitting firm resources between payments to the lender $p_{it}$ and dividends $d_{it}$, in a fully state-contingent manner.

### State variable choice

- Use equity value of the firm $V_{it}$ as the state variable; debt value recovered from $b_{it} = W_{it} - V_{it}$.
- Tax deductability of interest on debt: $\tau r b_{it} = \tau r(W_{it} - V_{it})$, which yields:
  - Adjusted discount rate for the firm: $1/(1 + (1 - \tau)r)$ instead of $1/(1 + r)$
  - Penalty for foregone tax deductions on debt: $\tau r V_{it}$

### Diversion function

$$\mathcal{D}(k_{it}, z_{it}, \eta_{it}, \hat{\eta}_{it})$$

- $\hat{\eta}_{it}$: shareholders' (potentially misreported) report of $\eta_{it}$
- Most straightforward specification under the pre-tax $\pi$ convention used in this document: $\mathcal{D} = \lambda(1-\tau)\left[\pi(k_{it}, z_{it}, \eta_{it}) - \pi(k_{it}, z_{it}, \hat{\eta}_{it})\right]$.
- $\lambda$: diversion parameter; $1 - \lambda$ captures potential losses in cash flow diversion

### Firm value function

$$W(k_{it-1}, V_{it-1}, z_{it-1}) = \max_{k_{it}, V_{z_{it}, \eta_{it}}, d_{z_{it}, \eta_{it}}} \frac{1}{1 + (1 - \tau)r}$$

$$\times \Big[ -k_{it} - \Psi(k_{it}, k_{it-1}) + (1 - \delta)k_{it} + \tau\delta k_{it} - r\tau V_{it-1}$$

$$+ E_{t-1}\big[(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) + W(k_{it}, V_{z_{it}, \eta_{it}}, z_{it})\big] \Big]$$

subject to:

**Promise-keeping constraint:**

$$V_{it-1} = \frac{1}{1 + r}E_{t-1}[d_{z_{it}, \eta_{it}} + V_{z_{it}, \eta_{it}}] \quad (8)$$

**Incentive compatibility constraints:**

$$d_{z_{it}, \eta_{it}} + V_{z_{it}, \eta_{it}} \geq d_{z_{it}, \hat{\eta}_{it}} + V_{z_{it}, \hat{\eta}_{it}} + \mathcal{D}(k_{it}, z_{it}, \eta_{it}, \hat{\eta}_{it}), \quad \forall z_t, \forall \hat{\eta}_{it} \quad (9)$$

**Limited liability constraints (non-negativity):**

$$d_{z_{it}, \eta_{it}} \geq 0, \quad \forall z_{it}, \forall \eta_{it} \quad (10)$$

$$V_{z_{it}, \eta_{it}} \geq 0, \quad \forall z_{it}, \forall \eta_{it} \quad (11)$$

### Payments to the lender (recovered from resource constraint)

$$p_{it} = -k_{it} - \Psi(k_{it}, k_{it-1}) + (1 - \delta)k_{it} + \tau\delta k_{it} + \tau r(W_{it} - V_{it}) + (1-\tau)\pi(k_{it}, z_{it}, \eta_{it}) - d_{it}$$

Variables (moral hazard model):
- $V_{it-1}$: equity value at the end of period $t-1$
- $d_{z_{it}, \eta_{it}}$: state-contingent dividend payment
- $V_{z_{it}, \eta_{it}}$: state-contingent continuation equity value
- $p_{it}$: state-contingent payment to lender

## 3.2 LP Solution Method

### Primitives and grids

State $S = K \times \mathcal{V} \times Z$ with discrete grids $K = \{k_1, \dots, k_{n_k}\}$, $\mathcal{V} = \{V_1, \dots, V_{n_V}\}\subset\mathbb{R}_+$ with $V_1 = 0$, and $Z = \{z_1, \dots, z_{n_z}\}$. Shock $\eta \in \{+\bar\eta, -\bar\eta\}$ with $P(\eta = +\bar\eta) = \kappa$. Persistent transition $Q(z' \mid z)$. Additional parameter: $\lambda$ (diversion fraction).

State $(k,V,z)$: inherited capital, promised equity value, and current public persistent shock.

Discount in the firm Bellman: $1/(1 + (1-\tau)r)$ (firm-side, with the tax shield on the debt component already embedded). Discount in promise-keeping: $1/(1+r)$ (equity-holder side, no debt tax shield).

Under the pre-tax $\pi$ convention used in this document, the diversion function is

$$\mathcal{D}(k', z', \eta', \hat\eta') = \lambda(1-\tau)\big[\pi(k', z', \eta') - \pi(k', z', \hat\eta')\big].$$

### Minimal finite-action baseline

To keep the baseline a true LP, use a **finite contract-action menu**. For each current state $(k,V,z)$ and candidate next capital $k'\in K$, a contract action is a fixed collection

$$a=\{V'_{z',\eta'},d_{z',\eta'}\}_{z'\in Z,\eta'\in N},$$

where each $V'_{z',\eta'}\in\mathcal{V}$ and each $d_{z',\eta'}\in D$, with $D=\{d_1,\dots,d_{n_d}\}\subset\mathbb{R}_+$ a nonnegative dividend grid. Because the contract components are fixed before a Bellman inequality is added, the continuation terms $W(k',V'_{z',\eta'},z')$ enter linearly with known coefficients.

Do **not** include convex-combination weights such as $\mu_j W(k',V_j,z')$ as free variables in the master LP. That would be bilinear, not linear. Continuous interpolation can be added later only through constraint generation, where interpolation weights are fixed constants when each Bellman inequality is added.

### Step 1: Build feasible contract-action lists

For every $(k,V,z,k')$, enumerate candidate contract actions $a=\{V'_{z',\eta'},d_{z',\eta'}\}$. Keep only actions satisfying all constraints below.

- **Promise keeping (Eq. 8):**
$$V = \frac{1}{1+r}\sum_{z',\eta'} Q(z'\mid z)P(\eta')\big[d_{z',\eta'}+V'_{z',\eta'}\big].$$

In code, because $D$ and $\mathcal{V}$ are discrete, use a tight tolerance `pk_tol`:
$$\left|\frac{1}{1+r}\sum_{z',\eta'} Q(z'\mid z)P(\eta')\big[d_{z',\eta'}+V'_{z',\eta'}\big]-V\right|\leq \texttt{pk\_tol}.$$

- **Incentive compatibility (Eq. 9):** for each $z'$ and each pair $(\eta',\hat\eta')$ with $\eta'\neq\hat\eta'$,
$$d_{z',\eta'}+V'_{z',\eta'} \geq d_{z',\hat\eta'}+V'_{z',\hat\eta'}+\mathcal{D}(k',z',\eta',\hat\eta').$$

With binary $\eta'$, there are two non-trivial IC inequalities per $z'$.

- **Limited liability (Eqs. 10, 11):**
$$d_{z',\eta'}\geq 0, \qquad V'_{z',\eta'}\geq 0.$$

A tuple $(k,V,z,k')$ contributes no Bellman constraint if no feasible contract action exists for that particular $k'$. However, every state $(k,V,z)$ must have at least one feasible action across all $k'$. If a state has no feasible action, treat this as a grid-design failure: enlarge the dividend grid $D$, enlarge/shift the promised-equity grid $\mathcal{V}$, relax only the numerical tolerance if the miss is purely rounding error, or remove the state from the admissible state grid. Do not solve the LP with states that have an empty feasible action correspondence.

The minimal baseline is intended for small grids only. A faster second-stage implementation can replace enumeration with constraint generation or an auxiliary search routine, but the master LP must still receive only fixed actions.

### Step 2: Master LP

**Variables.** $W(k,V,z)$ for every $(k,V,z)\in S$. Count: $n_k n_V n_z$.

**Objective.**
$$\min_W \sum_{(k,V,z)\in S} W(k,V,z).$$

**Bellman constraints.** For every state $(k,V,z)$ and every feasible fixed action $(k',a)$:

$$W(k,V,z) \geq \frac{1}{1+(1-\tau)r}\bigg[-k' - \Psi(k',k) + (1-\delta)k' + \tau\delta k' - r\tau V + \sum_{z',\eta'} Q(z'\mid z)P(\eta')\Big((1-\tau)\pi(k',z',\eta') + W(k',V'_{z',\eta'},z')\Big)\bigg].$$

### Step 3: Recover the policy

After the LP returns $W^*$, choose the feasible fixed action that maximizes the RHS at each state:

$$(k',a)^*(k,V,z)=\arg\max_{(k',a)\in\mathcal{A}_{MH}(k,V,z)} \text{RHS}_{MH}(k,V,z,k',a;W^*).$$

The policy consists of $k'^*(k,V,z)$ and the associated state-contingent contract $\{V'^*_{z',\eta'},d^*_{z',\eta'}\}$.

Payment to the lender is recovered after the LP solve as the residual from the resource constraint. Under the pre-tax $\pi$ convention, for each realized $(z',\eta')$:

$$p^*_{z', \eta'} = -k'^* - \Psi(k'^*, k) + (1-\delta)k'^* + \tau\delta k'^* + \tau r\big(W^*(k'^*, V'^*_{z', \eta'}, z') - V'^*_{z', \eta'}\big) + (1-\tau)\pi(k'^*, z', \eta') - d^*_{z', \eta'}.$$

This recovered payment is for accounting and simulated policy measurement. It should not be reintroduced into the master LP as an additional constraint involving $W-V$, because the Bellman equation already embeds the tax-shield logic through the adjusted discount factor and the $-r\tau V$ term.

### Implementation notes

- **Baseline:** enumerate finite contract actions on small $\mathcal{V}$ and $D$ grids, filter by promise keeping, IC, and limited liability, then assemble the sparse LP.
- **Correctness condition:** each Bellman inequality must correspond to a fixed feasible action. The continuation coefficient on each $W(k',V_j,z')$ is therefore a known probability weight, not a decision variable.
- **Scaling upgrade:** if enumeration becomes too large, use constraint generation. Given a current value vector $W^{(m)}$, solve a separate action-search problem for each state, freeze the selected action, add the corresponding linear Bellman inequality to the master LP, and repeat until no violated constraints remain.

#### Future scaling path: paper-style constraint generation

The current implementation target is a modest local baseline: TO is solved by full finite-action LP, while LE and MH use small finite menus of complete state-contingent contracts. This is sufficient for a clean conceptual implementation and for verifying timing, feasibility, and policy recovery. It is not intended to be paper-scale for LE/MH.

For larger LE/MH grids, follow the paper's LP plus constraint-generation logic. The master problem contains only an active subset of Bellman inequalities, each corresponding to a fixed feasible action. After solving the relaxed master LP, use a separation oracle to search, state by state, for a feasible contract action whose Bellman RHS violates the current value function by more than tolerance. If such an action is found, freeze that complete action (including any interpolation weights, if used), add the resulting linear Bellman inequality to the master LP, and repeat until no violated constraints remain. The paper implements the action-search/separation step with mixed-integer programming because the state-contingent contracting action space is large. A future scalable implementation can therefore add a `constraint_generation` module and an optional `separation_oracle` module, likely using a stronger optimization backend such as Gurobi or CPLEX for the oracle. At the current baseline stage, these modules are documented but not implemented.

#### Why LP instead of NN based methods?

Models like TO, LE, and MH are complex because they (i) do not have closed-form Euler equations, (ii) featured nested fixed points in equilibrium, and (iii) have different equality and inequality constraints. There is generally no mathematical theorem proving that NN-based training can converge to a *unique fixed point*. 

In contrast, grid-based numerical methods like VFI and LP are guaranteed to converge to the unique fixed point of a finite discounted dynamic programming problem under a set of conditions (e.g., contraction mapping). Practically, we need the problem to satisfy:
1. The state grid is finite.
2. Every Bellman inequality corresponds to a fixed feasible action with fixed continuation-state indices and fixed probability weights.
3. The relevant discount factor is strictly below one: $1/(1+r)<1$ for TO/LE and $1/[1+(1-\tau)r]<1$ for MH.
4. The feasible action set is nonempty at every state.
5. TO pricing $\Delta(k_{\text{choice}},b_{\text{old}},z_{\text{old}})$ is well-defined for every grid point, including the $b=0$ case.
6. The resulting LP is feasible and bounded below.

Under these conditions, the Bellman operator for the finite discretized model is a contraction, and the standard LP formulation with objective $\min \sum_s W(s)$ and constraints $W(s)\geq T_a W(s)$ for all feasible state-action pairs recovers the unique fixed point. If continuous interpolation or auxiliary action searches are added later, the master LP remains valid only if each added Bellman inequality freezes the selected action and interpolation weights as constants.

---

## 2. Structural Estimation and Model Comparison

### Policy Functions

The model solutions are the optimal policy function and value function. The **optimal policy function** maps state to  actions and can be written as: 
$$
a_t=\varphi(s_t)
$$
where the action vector is defined as $a_t \equiv\{ k_{t+1}, b_{t+1}, d_{t+1} \}$ and the state vector is $s_{t}\equiv\{k_t, b_t, z_t \}$. 

Because we use a grid-based LP solution, the theoretical policy function is stored as lookup tables over the discrete state grid. For example, at each grid point $(k_t,b_t,z_t)$, the TO and LE solvers store the optimal next capital $k_{t+1}$. The policy table is therefore a discrete numerical approximation to the model policy function.

For empirical estimation and model comparison, the primitive theoretical variables are transformed into observable counterparts. The **empirical policy function** is written as
$$
y_{it}=P(x_{it})+u_{it},
\qquad
E[u_{it}\mid x_{it}]=0,
$$
where the *observable states* are
$$
x_{it}
=
\left( \log k_{it}, \frac{\pi(k_{it},z_{it},\eta_{it})}{k_{it}},\frac{b_{it}}{k_{it}}\right)
$$
and the *observable actions* are
$$
y_{it}
=
\left( \frac{i_{it}}{k_{it}}, \frac{b_{i,t+1}}{k_{i,t+1}}, \frac{d_{i,t+1}}{k_{i,t+1}} \right)
$$

For each action variable $y^n_{it}$ with $n$ indexing each element in vector $y_{it}$, the empirical policy function is estimated semi-parametrically using a series approximation
$$
P^n(x_{it}) \approx \sum_{j=1}^J h^n_j p_j(x_{it}),
$$
where $p_j(x_{it})$ are basis functions and $h^n_j$ are coefficients estimated from regression:
$$
\min_h
\sum_{i,t}
\left(
y^n_{it}
-
\sum_{j=1}^J h^n_j p_j(x_{it})
\right)^2.
$$

To reproduce the **model comparison figure**, I follow these steps:

- For a given model (e.g., TO), solve the raw policy function $\varphi$, use it to simulate a data of firm-year panel
- From the simulated data, construct observable states and actions $(y_{it},x_{it})$
- Fit the series approximation above to obtain empirical policy function $P^n(x_{it})$ and overlay it in twoway plot of $y$ on $x$

For the real-world data, we can directly fit $(y_{it},x_{it})$ to estimate another empirical policy function.


I plot the fitted empirical policy functions $P^n(x_{it})$ as twoway slices of observable actions $y_{it}$ against states $x_{it}$. Each slice plot fix other states at sample median. These six panels can be directly compared to the Figure 2 and 3 in Nikolov21. 

- Investment vs log size
- Future leverage vs current leverage
- Investment vs current leverage
- Payout vs profitability
- Future leverage vs profitability 
- Investment vs profitability

![TO fitted empirical policy functions (panel simulated from the solved TO model)](figures/bonus3-model-hk-data/to/to_empirical_policy_slices.png)

The figure shows the empirical policy function fitted on a panel simulated from the solved TO model: each panel plots one observable action against one observable state, with the red line the fitted (partial-dependence) policy and the points the simulated firm-years. The estimated slopes line up with standard theory and with Nikolov21's TO results: investment falls with firm size and is roughly flat-to-declining in leverage, payout and investment both rise with profitability, future leverage is strongly increasing in current leverage (leverage is persistent), and future leverage falls with profitability (more profitable firms lever less). Because these signs match the theoretical mechanisms and the patterns reported for the TO model in Nikolov21, I treat this as a successful reproduction of the TO model's policy behavior. The figure is shown here as an illustrative example for the TO model; the same fitted coefficients are the auxiliary moments used in the indirect inference below.


### Indirect inference

The empirical policy function estimated above serves as the auxiliary model. The structural model is not estimated by matching individual firm outcomes one by one. Instead, for a candidate structural parameter vector $\beta$, we solve the model, simulate firm panels from the model, estimate the same empirical policy functions on the simulated data, and choose $\beta$ so that the simulated policy functions are close to the empirical policy functions estimated from real data.

Let $ v_{it}\equiv (y_{it},x_{it}) $ denote one observation in the real firm panel, where $i$ indexes firms and $t$ indexes time. The full real-data panel is

$$
\mathcal D_{\text{data}} \equiv \{v_{it}\}_{i,t}.
$$

For a candidate structural parameter vector $\beta$, let

$$
v^{(r)}_{it}(\beta)
\equiv
\left(y^{(r)}_{it}(\beta),x^{(r)}_{it}(\beta)\right)
$$

Denote one observation in simulated panel $r$, where $r=1,\dots,R$ indexes independent simulation replications. The simulated $i,t$ indices are artificial firm and time indices generated by the model. They are used only to construct a simulated panel with the same observable variables as the real data. The full simulated panel in replication $r$ is

$$
\mathcal D^{(r)}_{\text{sim}}(\beta)
\equiv
\{v^{(r)}_{it}(\beta)\}_{i,t}.
$$

The vector $\beta$ contains the structural parameters estimated by indirect inference, including technology parameters, shock-process parameters, adjustment costs, and the model-specific financial-friction parameter. In the paper, the main estimated parameters include

$$
(\alpha, f, \rho_z, \sigma_z, \delta, \psi, \eta),
$$

and the friction parameter is model-specific: $\xi$ (TO), $\theta$ (LE), and  $\lambda$ (MH). Some parameters, such as $r,\tau,\kappa$, are calibrated and fixed outside the inference.

For any panel dataset $\mathcal D$, define $h(\mathcal D)$
as the vector collecting all estimated coefficients from the empirical policy-function regressions. In our notation, $h(\mathcal D)$ stacks the coefficients $h^n_j$ across all action variables $n$ and all basis functions $j$. Therefore, $h(\mathcal D)$ summarizes the estimated mapping from observable states $x_{it}$ to observable actions $y_{it}$ in that panel.

The real-data auxiliary coefficient vector is $h_{\text{data}}=h(\mathcal D_{\text{data}})$. For candidate parameter vector $\beta$, the simulated auxiliary coefficient vector is the average across simulated panels:

$$
h_{\text{sim}}(\beta)
=
\frac{1}{R}
\sum_{r=1}^R
h\left(\mathcal D^{(r)}_{\text{sim}}(\beta)\right).
$$

The indirect-inference moment is the difference between the empirical auxiliary coefficients and the model-implied auxiliary coefficients:

$$
g(\beta)
=
h_{\text{data}}
-
h_{\text{sim}}(\beta).
$$

The structural estimator chooses $\beta$ to minimize the weighted distance between these two coefficient vectors:

$$
\beta^*
=
\arg\min_{\beta}
g(\beta)' W g(\beta),
$$

where $W$ is a positive definite weighting matrix. The paper’s key idea is that a model fits well if, for some $\beta$, its simulated investment, leverage, and payout policies generate auxiliary policy-function coefficients close to those estimated from the real firm panel.

The estimation loop is:

1. Choose a candidate parameter vector $\beta$.
2. Solve the structural model by LP and recover the policy lookup tables.
3. Simulate $R$ firm panels using the recovered policy functions and shock processes.
4. For each simulated panel, transform primitive model variables into the same observable variables $x^{(r)}_{it}(\beta)$ and $y^{(r)}_{it}(\beta)$ used in the real data.
5. Estimate the same semi-parametric empirical policy-function regressions on each simulated panel.
6. Compute $h_{\text{sim}}(\beta)$ by averaging the simulated coefficient vectors across $R$ replications.
7. Compute

$$
g(\beta)=h_{\text{data}}-h_{\text{sim}}(\beta).
$$

8. Set $W$ and search over $\beta$ to minimize

$$
g(\beta)'Wg(\beta).
$$

The same indirect-inference procedure is applied separately to TO, LE, and MH. Because the three models share the same observable policy variables even though their internal financial frictions differ, their fit can be compared by asking which model generates simulated policy-function coefficients closest to the empirical coefficients.

**Variance estimation**
To construct 


### Data and Sample Construction

I construct a sample of listed companies in Hong Kong between 1999 and 2024 from the [Compustat Global database on Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/pages/grid-items/compustat-global-wrds-basics/) subscribed by UBC library.

**Sample construction.** I drop firms in utilities and financial services sector, and those that are higly unbalanced or with key variables missing. The final sample consists of 2,080 firm-years observations, with 116 firms in a balanced panel. I winsorized the top and bottom 1%. When constructing the moment conditions I fit the empirical policy functions on the raw observables and match only their slope and curvature coefficients, dropping each regression's intercept. I drop the intercept because the model and the data differ in level mainly through the deflator (the model normalizes by capital while the data normalizes by total assets), a gap no structural parameter can close, so leaving the intercept out absorbs that level offset and lets the estimator target the comparable shape of the policy functions. 

The table below summarizes the key observables used to fit the empirical policy function and to construct the moment conditions for inference. Note that leverage and future leverage are measured gross, $(\text{DLTT}+\text{DLC})/\text{AT}$, and are therefore non-negative, which matches the model's $b/k \geq 0$ and is the support the structural policies can reproduce. For Hong Kong firms, market value is not available so I do not use market-to-book for estimation.

| Variable | Role in Policy Function | Theory | Data (Compustat) |
|---|---|---| ---|
| Investment rate | Action | $\frac{k_{t+1}-(1-\delta)k_t}{k_t}$ | CAPX/AT 
| Book leverage (current) | State | $b_t/k_t$. | (DLTT+DLC)/AT
| Future leverage | Action | $b_{t+1}/k_{t+1}$. | (DLTT+DLC)/AT
| Profitability | State | $\frac{\pi(k_t,z_t,\eta_t)}{k_t}=\frac{(z_t+\eta_t)k_t^\alpha-f}{k_t}$ | OIBDP/AT
| Dividends payouts | Action | $\frac{d_t}{k_t}$ | (DVT+PSSTKC-PSTKRV)/AT
| Firm size (log) | State | $\log k_t$ | log(PPENT)
| Market-to-book | State | $W_t/k_t$ | (DLTT+DLC+PRCCF $\times$ CHSO)/AT

#### Sample construction summary

| Step | Firm-years | Firms |
|---|---|---|
| Raw quarterly records | 21,706 | 312 |
| Fiscal-Q4 (annual) records | 5,942 | 311 |
| HKD currency | 4,866 | 242 |
| Industrial, consolidated, standard format | 2,561 | 131 |
| Exclude financials and utilities | 2,339 | 122 |
| Positive assets and PP&E | 2,333 | 122 |
| Finite flow variables | 2,205 | 122 |
| Has next-period leverage | 2,083 | 119 |
| At least 3 firm-years per firm | 2,080 | 116 |

The binding cuts are the annual (fiscal-Q4) filter and the industrial/consolidated/standard-format requirement, which together account for almost all of the attrition from 21,706 raw records; the sector exclusions and the positivity/finiteness screens drop comparatively few observations. The final panel is 2,080 firm-years for 116 firms and is close to balanced (mean 18, median 20 annual observations per firm).

#### Missing data and imputation

Debt and payout items are frequently missing in Compustat Global: long-term debt (DLTT) is missing for about 25% of firm-years, current debt (DLC) for 12%, dividends (DVT) for 46%, and repurchases (PRSTKC) for 88%. These fields are imputed to zero, interpreted as no debt or no payout, and flagged, so the corresponding leverage and payout values for those firm-years are zero by construction. The core fields (assets, PP&E, profitability, investment) are essentially complete.

#### Variable distributions

![Hong Kong Compustat policy-variable distributions](figures/bonus3-model-hk-data/nikolov_hkg_variable_distributions.png)

Investment, payout, and leverage are right-skewed with a large mass at zero: many Hong Kong firms invest little, pay nothing, and carry little or no debt. Profitability is roughly symmetric around a low mean and is occasionally negative. Firm size (log capital) is widely dispersed. Orange lines mark medians; the small bars at the right edges are the 1% winsorization caps.

#### Summary Statistics

| Variable | Mean | SD | p1 | Median | p99 |
|---|---|---|---|---|---|
| investment_rate | 0.033 | 0.045 | 0.000 | 0.017 | 0.253 |
| future_leverage | 0.196 | 0.207 | 0.000 | 0.139 | 1.097 |
| payout_rate | 0.015 | 0.024 | 0.000 | 0.003 | 0.125 |
| log_k | 6.258 | 2.463 | 0.052 | 6.253 | 11.850 |
| profitability | 0.039 | 0.105 | -0.438 | 0.040 | 0.316 |
| leverage | 0.194 | 0.204 | 0.000 | 0.141 | 1.090 |

A typical firm invests roughly 2 to 3% of capital per year (median 1.7%), holds modest leverage (median 14%), and pays out little (median 0.3%); profitability is low (median 4%) and can be negative.


### Estimation Results

I now provide results from solving and estimating the three models. The main purpose is to show end-to-end model solve and estimation pipeline. Limited by the CPU capacity (M1), I use smaller grid density than the paper's specification ($[k,b,z]=[21,17,5]$) so the results should be interpreted as rough preliminary estimates. Future work would require better CPU or improving the efficiency of LP solver by adding features such as constraint generation.

#### Trade-off (TO) Model

The table reports the parameter estimates from indirect inference on the TO Model, solved by LP method on grid with density ($[k,b,z]=[15,12,5]$). To conduct the minimization, I use Nelder-Mead as global method to search for the optimal parameter vector. The system is over-identified with 27 moments and 8 parameters. I use identity weight matrix to construct the standard SMM variance. Unfortunately, because simulated moment covariance is near-singular, I'm unable to conduct a valid over-identification (J) test for this run. The full estimation took about 2 hours on Apple M1. Reference Estimate reported in the second column are the initiation value, calibrated to the estimates on Large US public-listed firms reported by Nikolov21.


One notable estimate is the large capital adjustment cost $\hat \psi = 1.036$, which is not obviously realistic. This is consistent with the fact I show earlier that the Compustat-HK firms on average have much lower investment rate compared with Compustat-US firms. The $i/k$ moments are poorly matched (see appendix) and worth examining whether the current measurement for investment need to be refined if, for example, the variable are defined and constructed differently in Hong Kong due to different regulation.


| Parameter | Ref Estimate | Estimate $\hat\beta$ | SE |
|---|---|---|---|
| $\alpha$ (capital share) | 0.75 | 0.613 | 0.089 |
| $f$ (fixed cost) | 0.70 | 0.631 | 0.231 |
| $\rho_z$ (shock persistence) | 0.80 | 0.604 | 0.029 |
| $\sigma_z$ (shock volatility) | 0.30 | 0.296 | 0.073 |
| $\delta$ (depreciation) | 0.15 | 0.175 | 0.072 |
| $\psi$ (adjustment cost) | 0.15 | 1.036 | 0.024 |
| $\bar\eta$ (i.i.d. shock size) | 0.30 | 0.265 | 0.116 |
| $\xi$ (recovery rate) | 0.60 | 0.734 | 0.134 |

The figure below reproduces the empirical policy comparison plot from Nikolov21's Figure 2 and 3. Each panel is a partial-dependence slice: it varies one observable state, holds the others at their sample median, and plots the fitted empirical policy. The black line is fit on the real Hong Kong panel and the red line is fit on a panel simulated from the solved TO model at the estimated parameters. 

![TO indirect-inference policy overlay: the empirical policy fit on the real Hong Kong panel (black) versus the same regressions refit on a panel simulated from the solved TO model at the estimated parameters (red)](figures/bonus3-model-hk-data/to/to_ii_policy_overlay.png)

Because intercepts are dropped, a good match means the two lines share the same slope and curvature, not the same level. Here the lines clearly diverge: the simulated policy sits well above the data in most panels, and the shapes agree only in part (for instance, future leverage rises with current leverage in both, but the model line is much steeper and higher). So the estimator could not bring the model's policy close to the data from the current preliminary results.



**Why the fit is poor.** I rank the likely causes from most to least actionable.

1. **Model solve quality (most likely the main cause).** The reported run solves the TO model on a coarse grid ($k=15$, $b=12$, $z=5$) for tractability, because indirect inference re-solves the model once per candidate parameter vector, hundreds of times in total. A coarse grid turns the policy into a crude step function over a handful of states, so the simulated panel carries little granular variation and the empirical policy refit on it is shaped by discretization rather than by economics. This is plausibly the dominant problem. The clear next step is to refine the grid (denser capital, debt, and shock points), which lowers grid approximation error and lets the simulation reproduce the smooth policy variation the regressions are trying to recover. Only once the solve is accurate can we trust the simulated moments.

2. **Observable and normalization disagreement.** Even with an accurate solve, the model and the data describe firm flows on different scales. The model normalizes every flow by capital $k$, while the Compustat observables normalize by total assets, so the two policies can differ in level for reasons no structural parameter can fix. I already drop each regression's intercept to absorb this level offset and match only slope and curvature. A more careful version is the firm-fixed-effects adjustment used in Nikolov21: estimate firm fixed effects from the real panel, drop the intercept of the simulated empirical policy, then add the real-data firm effects back onto the simulated policy. The purpose is to strip out persistent, firm-specific level differences (size, accounting deflator, unmodeled heterogeneity) that the structural model was never meant to explain, so the estimator compares the part of the policy that is actually comparable, its shape. The problem it solves is keeping a nuisance level gap from looking like a structural misfit.

3. **A genuine model rejection (only after 1 and 2 are ruled out).** If the solve is accurate, the observables are aligned, and the simulated policy still cannot approach the data, then the fit is truly poor and the model is rejected on its own terms. Confirming this needs the formal model-fit test from Nikolov21, the over-identification test on the moment gap. That test is expensive: building its variance requires re-solving the model many times to estimate the moment covariance and the sensitivity of the moments to the parameters. The current run could not even form a valid J statistic (identity weighting plus a near-singular moment covariance), so a formal rejection is not yet warranted. The honest reading is that causes 1 and 2 must be cleared first, and only then is the expensive formal test worth running.

## Appendix 


#### Moment fit: TO Model

The moment vector stacks the slope and curvature coefficients of the three policy regressions (intercepts dropped). Actions are written as $i/k$ (investment rate), $b'/k'$ (future leverage), and $d/k$ (payout); regressors are $\log k$, profitability (prof), and current leverage (lev), together with their squares and interactions. The gap is $g = h_{\text{data}} - h_{\text{sim}}$.

| Action | Regressor | Data $h$ | Sim $h$ | Gap $g$ |
|---|---|---|---|---|
| $i/k$ | $\log k$ | 0.009 | -0.688 | 0.697 |
| $i/k$ | prof | 0.007 | 0.170 | -0.163 |
| $i/k$ | lev | 0.027 | -0.434 | 0.461 |
| $i/k$ | $(\log k)^2$ | -0.001 | 0.091 | -0.092 |
| $i/k$ | $\log k\cdot$ prof | 0.019 | -0.089 | 0.108 |
| $i/k$ | $\log k\cdot$ lev | 0.005 | 0.193 | -0.189 |
| $i/k$ | prof $^2$ | 0.248 | 0.075 | 0.173 |
| $i/k$ | prof $\cdot$ lev | -0.066 | -0.095 | 0.028 |
| $i/k$ | lev $^2$ | -0.042 | 0.236 | -0.278 |
| $b'/k'$ | $\log k$ | -0.011 | -0.210 | 0.199 |
| $b'/k'$ | prof | -0.186 | -0.002 | -0.185 |
| $b'/k'$ | lev | 0.820 | 0.774 | 0.046 |
| $b'/k'$ | $(\log k)^2$ | 0.000 | -0.010 | 0.010 |
| $b'/k'$ | $\log k\cdot$ prof | 0.025 | -0.011 | 0.037 |
| $b'/k'$ | $\log k\cdot$ lev | 0.033 | 0.165 | -0.132 |
| $b'/k'$ | prof $^2$ | -0.083 | 0.018 | -0.101 |
| $b'/k'$ | prof $\cdot$ lev | -0.160 | -0.063 | -0.098 |
| $b'/k'$ | lev $^2$ | -0.227 | -0.245 | 0.017 |
| $d/k$ | $\log k$ | 0.002 | -0.024 | 0.026 |
| $d/k$ | prof | 0.073 | 0.358 | -0.285 |
| $d/k$ | lev | -0.078 | -0.135 | 0.057 |
| $d/k$ | $\log k^2$ | -0.000 | -0.054 | 0.054 |
| $d/k$ | $\log k\cdot$ prof | 0.012 | -0.088 | 0.100 |
| $d/k$ | $\log k\cdot$ lev | 0.003 | 0.091 | -0.088 |
| $d/k$ | prof $^2$ | 0.128 | 0.080 | 0.048 |
| $d/k$ | prof $\cdot$ lev | -0.131 | -0.059 | -0.072 |
| $d/k$ | lev $^2$ | 0.040 | -0.076 | 0.116 |