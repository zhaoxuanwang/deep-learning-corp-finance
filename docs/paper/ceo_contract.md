# Dynamic Principal-Agent Model of CEO Short-Termism

This document summarizes the canonical model of CEO compensation and short-termism, based on Marinovic and Varas (2019). It isolates the fundamental economic mechanics of multi-period agency problems when performance is manipulable. Cronqvist et al. (2024) extends this canonical model to a specific empirical setting: the FAS 123-R regulatory shift in the United States. To do so, they add limited investor attention ($\alpha$) and endogenous risk-taking (where the CEO controls volatility $\sigma_t$).

I choose to implement the Marinovic and Varas (2019) variant because it provides a clean baseline and is applicable to generic empirical settings. This is useful because when applying to APAC market, we may not have quasi-experiments like FAS 123-R to identify the additional limited attension channel.

**What are the key problems that the authors try to illustrate with the model?**

In this model, the CEO privately chooses both productive effort and performance manipulation to boost short-term cash flows at the expense of long-term firm value. Because the board cannot observe these actions directly, it uses the CEO's incentive compatibility constraints as a calibration dial to find the optimal balance between inducing effort and deterring manipulation. The board optimally designs a contract that tolerates some manipulation to maximize net firm value. The mechanics of this dynamic contract are governed by a single state variable representing the duration of the CEO's deferred pay. Crucially, the model captures an endogenous "horizon problem" driven by time. Early in the CEO's tenure, manipulation is naturally deterred because the CEO will personally suffer the future cash-flow reversals while still on the job. However, as retirement approaches, this natural deterrence vanishes. To maintain the CEO's effort without imposing inefficiently high post-retirement risk, the board optimally allows the duration of incentives to drop. This mathematically shifts compensation toward the short term, ensuring the CEO continues working hard but inevitably causing manipulation to escalate in their final years. Ultimately, by solving the principal's dynamic optimization problem, the firm's long-term value is maximized in a reality where short-termism is anticipated, managed, and optimally priced into the contract.

**Whether and how can we extend the basic model and the risky debt model to capture similar behaviors? For example, adding the board-CEO agents to the model.**



**How to estimate the new extended model with the CEO and the board?**


---

## 1. Environment and Actions

The model considers a firm (the principal) hiring a CEO (the agent) in continuous time.

* **Time Horizon**: The model operates in continuous time $t \in [0, T]$, where $T$ is the CEO's deterministic retirement date.


* **Clawback Period**: The firm can dictate post-retirement compensation and tie the CEO's wealth to the firm's performance until time $T + \tau$, where $\tau \ge 0$.


* **Hidden Actions**: At every instant $t$, the CEO privately chooses two costly actions: productive effort $a_t$ and performance manipulation $m_t$.



---

## 2. Cash Flows, Manipulation, and Preferences

The firm cannot observe the CEO's actions directly; it only observes the realized cash flow.

* **Firm Cash Flow**: The observable performance measure evolves according to the stochastic differential equation:

$$dX_t = (a_t + m_t - \theta M_t)dt + \sigma dB_t$$



where $B_t$ is a standard Brownian motion and $\sigma$ is the exogenous cash flow volatility.


* **Stock of Manipulation**: Manipulation borrows from the future. The accumulated stock of manipulation $M_t$ is defined as:

$$M_t = \int_0^t e^{-\kappa(t-s)} m_s ds$$



This stock depreciates at rate $\kappa$, while $\theta$ represents the marginal effect of manipulation on reducing current cash flows. The overall value-destroying effect of manipulation is denoted by $\lambda \equiv \frac{\theta}{r+\kappa} - 1$.


* **CEO Preferences**: The CEO has Constant Absolute Risk Aversion (CARA) preferences with risk aversion $\gamma$, represented by $u(c, a, m) = -e^{-\gamma(c - h(a) - g(m))}/\gamma$.


* **Cost of Actions**: Effort and manipulation are penalized via quadratic cost functions $h(a) = a^2 / 2$ and $g(m) = gm^2 / 2$.


* **Private Savings**: The CEO can unobservably save or borrow at the risk-free rate $r$. This forces the marginal utility of consumption to be a martingale, vastly simplifying the optimal consumption path.



---

## 3. State and Control Variables

Because the CEO can smooth consumption, the principal's dynamic contracting problem can be reduced to choosing two control variables that govern a single state variable.

* **The State Variable ($z_t$)**: The contract is fully summarized by $z_t$, representing the ratio of the CEO's long-term incentives ($p_t$) to their total continuation utility ($W_t$). Formally, $z_t \equiv -p_t / W_t$. Economically, $z_t$ measures the **duration of deferred compensation**.


* **Control Variable 1 ($\beta_t$)**: The short-term **Pay-for-Performance Sensitivity (PPS)**. This dictates how the CEO's continuation utility responds to immediate cash flow shocks ($dB_t$).


* **Control Variable 2 ($\sigma_{zt}$)**: The **performance sensitivity of long-term incentives**. This dictates how the duration of incentives ($z_t$) responds to immediate cash flow shocks ($dB_t$), effectively controlling the rate of performance-based vesting.



---

## 4. Incentive Compatibility (The CEO's Reaction)

The principal cannot dictate $a_t$ or $m_t$. Instead, the principal sets $\beta_t$ and $\sigma_{zt}$, and anticipates the CEO's self-interested response using First-Order Conditions (FOCs).

* **Optimal Effort**: The marginal cost of effort must equal the marginal benefit (the short-term incentive $\beta_t$).



$$r\gamma a_t = \beta_t$$


* **Optimal Manipulation**: The marginal cost of manipulation equals the short-term benefit ($\beta_t$) minus the long-term penalty of future cash flow reversal.



$$g'(m_t) = \frac{\beta_t}{r\gamma} - \phi z_t$$



where $\phi \equiv \frac{\theta}{r\gamma}$.


* **Control Substitution**: Because the principal knows these explicit mappings, the principal's problem is mathematically reformulated to directly choose the target effort $a_t$ (which instantly pins down $\beta_t = r\gamma a_t$) and target manipulation $m_t$, subject to the CEO's FOC constraints.



---

## 5. The Principal's Optimization Problem

The principal maximizes expected firm cash flows net of CEO compensation costs, subject to the evolution of the state variable $z_t$.

* **The Objective**:

$$F(z) = \max_{a_t, \sigma_{zt}} \mathbb{E} \left[ \int_0^T e^{-rt}(a_t - \lambda m_t - h(a_t) - g(m_t))dt - \int_0^{T+\tau} e^{-rt} \frac{\sigma^2 (r\gamma a_t)^2}{2r\gamma} dt \right]$$



Note: The final integral represents the risk premium cost of providing the short-term incentive $\beta_t = r\gamma a_t$.


* **The Constraint (Law of Motion for $z_t$)**:

$$dz_t = \left[ (r+\kappa)z_t + r\gamma a_t(\sigma\sigma_{zt} - 1) \right]dt + \sigma_{zt} dB_t$$



with the terminal condition $z_{T+\tau} = 0$.



---

## 6. The Solution (Value and Policy Functions)

The solution to the dynamic contract is not a closed-form formula over time, but rather a set of **policy functions** characterizing the optimal actions given the current state $z_t$ and time $t$.

* **The Hamilton-Jacobi-Bellman (HJB) Equation**: The principal's problem resolves to the following HJB equation for the value function $F(z,t)$:

$$rF = \max_{a, \sigma_z} \pi(a,z) + F_t + \left[ (r+\kappa)z + a r\gamma(\sigma\sigma_z - 1) \right]F_z + \frac{1}{2}\sigma_z^2 F_{zz}$$



where $\pi(a,z)$ is the simplified flow payoff. The terminal condition is $F(z,T) = -\frac{1}{2}\mathcal{C}z^2$, reflecting the convex cost $\mathcal{C}$ of providing post-retirement incentives.


* **Explicit Policy Functions**: By maximizing the HJB equation with respect to the controls, the model yields semi-explicit optimal policy functions that depend on the value function $F(z,t)$ and its derivatives.


1. **Optimal Vesting Sensitivity**:

$$\sigma_z(z,t) = -r\gamma\sigma a(z,t) \frac{F_z}{F_{zz}}$$



Because $F$ is concave ($F_{zz} < 0$) and $F_z \le 0$, $\sigma_z(z,t) \le 0$. This means positive performance shocks optimally accelerate the vesting of incentives (reduce $z_t$).


2. **Optimal Manipulation**:

$$m(z,t) = \frac{1}{g}(a(z,t) - \phi z)^+$$


3. **Optimal Effort (Interior Solution)**:

$$a(z,t) = \frac{g - \lambda + \phi z - r\gamma g F_z}{1 + g\left(1 + r\gamma\sigma^2 + r^2\gamma^2\sigma^2 \frac{F_z^2}{F_{zz}}\right)}$$



(Assuming manipulation is positive and the constraint is binding).





Because the policy functions depend on the unknown value function $F(z,t)$, the actual path of the contract is found by numerically solving the HJB equation backward from time $T$.


### Algorithm: Implicit Upwind Finite Difference with Howard Policy Iteration

This algorithm solves the Hamilton-Jacobi-Bellman (HJB) equation for the baseline continuous-time dynamic principal-agent model. It utilizes vectorized array operations and sparse matrix linear algebra suited for implementation in Python (`numpy` and `scipy.sparse`).

#### 1. Initialization

* **Grids**:
* State space $z$-grid: 1D array of size $N$ discretizing $[z_{min}, z_{max}]$ with step $\Delta z$.
* Time $t$-grid: 1D array of size $M$ discretizing $[0, T]$ with step $\Delta t$.


* **Value Function ($\mathbf{V}$)**: 1D array of size $N$. Initialize with the terminal condition at $t=T$:



$$\mathbf{V} = -\frac{1}{2}\mathcal{C}\mathbf{z}^2$$


* **Policies ($\mathbf{a}, \mathbf{m}, \mathbf{\sigma_z}$)**: 1D arrays of size $N$. Initialize with steady-state or zero guesses.

#### 2. Finite Difference Operators

Define functions to compute numerical derivatives of $\mathbf{V}$ with respect to $z$ across the grid of size $N$:

* **Forward Difference ($\mathbf{V_z^F}$)**: $\mathbf{V_z^F}[i] = (\mathbf{V}[i+1] - \mathbf{V}[i]) / \Delta z$ (pad $i=N-1$ with $0$).
* **Backward Difference ($\mathbf{V_z^B}$)**: $\mathbf{V_z^B}[i] = (\mathbf{V}[i] - \mathbf{V}[i-1]) / \Delta z$ (pad $i=0$ with $0$).
* **Second Derivative ($\mathbf{V_{zz}}$)**: $\mathbf{V_{zz}}[i] = (\mathbf{V}[i+1] - 2\mathbf{V}[i] + \mathbf{V}[i-1]) / (\Delta z)^2$.
* *Constraint*: Enforce strict concavity to ensure matrix stability. Set $\mathbf{V_{zz}} \leftarrow \min(\mathbf{V_{zz}}, -10^{-6})$.





#### 3. Backward Time-Stepping Solver Loop

```text
FOR t = T down to 0:
    V_old = V
    error = infinity
    
    WHILE error > tolerance (e.g., 1e-6):  // Howard Policy Iteration
        
        // Step 3a: Policy Evaluation (Upwind Scheme)
        1. Calculate drift of z using current policies[cite: 2]:
           mu_z = (r + \kappa)z + a * (r \gamma)(\sigma * \sigma_z - 1)
           
        2. Construct upwind first derivative:
           V_z = V_z_F IF mu_z > 0 ELSE V_z_B
           
        3. Update policies explicitly via FOCs[cite: 2]:
           \sigma_z = -r\gamma\sigma * a * (V_z / V_zz)
           a = (g - \lambda + \phi z - r\gamma g V_z) / (1 + g(1 + r\gamma\sigma^2 + r^2\gamma^2\sigma^2 (V_z^2 / V_zz)))
           m = (1/g) * max(a - \phi z, 0)
           
        // Step 3b: Sparse Transition Matrix (A)
        4. Recalculate true drift mu_z with updated policies.
        5. Construct matrix diagonals (size N):
           c_lower = -min(mu_z, 0) / \Delta z + (\sigma_z^2) / (2 * \Delta z^2)
           c_upper = max(mu_z, 0) / \Delta z + (\sigma_z^2) / (2 * \Delta z^2)
           c_main = -c_lower - c_upper
           // Note: Adjust boundaries i=0 and i=N-1 to ensure rows sum to zero (reflecting boundaries).
           
        6. Assemble sparse tridiagonal matrix A using c_lower, c_main, c_upper.

        // Step 3c: Implicit System Update
        7. Compute flow payoff array[cite: 2]:
           Payoff = a - \lambda m - (1/2)a^2 - (1/2)gm^2 - (\sigma^2 (r\gamma a)^2) / (2r\gamma)
           
        8. Construct implicit system matrix B and RHS vector d:
           B = I + \Delta t (r*I - A)     // I is N x N sparse identity matrix
           d = V_old + Payoff * \Delta t
           
        9. Solve linear system for V_new:
           V_new = solve(B, d)            // via scipy.sparse.linalg.spsolve
           
        // Step 3d: Convergence Check
        10. error = max(|V_new - V_old|)
        11. V_old = V_new
        
    END WHILE
    
    // Step 3e: Advance to next time step
    Store V_new and optimized policies (a, m, \sigma_z) for current time t.
    V = V_new
    
END FOR

```

## Validation Strategy: Reproducing Baseline Policy Functions

To validate the numerical solution of the HJB equation, we will visualize the solved policy functions and compare them against the canonical results. The validation consists of generating 3D surface plots to capture the global state-space dynamics and 2D policy slices to precisely verify the magnitude and monotonicity of the CEO's actions.

### 1. Baseline Parameter Calibration

The HJB solver must be initialized with the exact baseline parameters utilized in the original simulation.

* **Risk-free rate**: r = 0.1


* **CEO risk aversion**: $\gamma$ = 1


* **Manipulation cost parameter**: g = 1


* **Manipulation destruction rate**: $\theta$ = 0.4


* **Manipulation depreciation rate**: $\kappa$ = 0.3


* **Cash flow volatility**: $\sigma$ = 2


* **CEO tenure (retirement)**: T = 10


* **Clawback period**: $\tau$ = 5



### 2. Validation Figure 1: 3D Policy Surfaces

This figure reconstructs the global policy functions over the entire solved grid, mapping the state variable and time to the optimal contract controls.

**Grid Setup:**

* **X-axis (Time)**: CEO Tenure $t$, ranging from 0 to 10.


* **Y-axis (State)**: Long-term incentives $z$, ranging from 0 to 0.25.



**Generated Subplots (Z-axes):**

1. **Optimal Effort Surface ($a(z,t)$)**: Plots the expected productive effort. Validates that effort decreases as time $t$ approaches 10 and increases as long-term incentives $z$ decrease.


2. **Optimal Manipulation Surface ($m(z,t)$)**: Plots the optimal earnings manipulation. Validates that manipulation is precisely zero when both $t$ and $z$ are low, and escalates sharply as $t$ approaches the retirement date 10.


3. **Optimal Vesting Sensitivity Surface ($\sigma_z(z,t)$)**: Plots the stochastic performance-vesting parameter. Validates that the sensitivity is zero (deterministic) early in the CEO's career, but drops negatively as $t$ approaches 10, confirming that positive performance optimally accelerates vesting near retirement.



### 3. Validation Figure 2: 2D Policy Slices

While 3D surfaces verify global geometry, 2D slices provide precise cross-sectional validation by isolating one dimension.

**Slice Type A: Fixed Time (Cross-Sectional State Dynamics)**

* Fix tenure at three distinct stages: Early ($t$ = 1), Mid ($t$ = 5), and Late ($t$ = 9).
* **X-axis**: Long-term incentives $z \in [0, 0.25]$.
* **Y-axis**: Policy values ($a$, $m$, $\sigma_z$).
* **Validation Check**: Confirms exactly at what threshold of $z$ the manipulation $m$ jumps from zero to strictly positive for a given year in the CEO's tenure.

**Slice Type B: Fixed State (Time-Series Dynamics)**

* Fix long-term incentives at three distinct magnitudes: Low ($z$ = 0.05), Medium ($z$ = 0.15), and High ($z$ = 0.25).
* **X-axis**: Tenure $t \in [0, 10]$.
* **Y-axis**: Policy values ($a$, $m$, $\sigma_z$).
* **Validation Check**: Confirms the "Horizon Effect" by showing that, holding the contract structure constant, manipulation $m(z,t)$ strictly increases as the CEO approaches the terminal date $T$.
