# Dynamic Principal-Agent Model of CEO Short-Termism

This document summarizes the canonical model of CEO compensation and short-termism, based on Marinovic and Varas (2019). It isolates the fundamental economic mechanics of multi-period agency problems when performance is manipulable. Cronqvist et al. (2024) extends this canonical model to a specific empirical setting: the FAS 123-R regulatory shift in the United States. To do so, they add limited investor attention ($\alpha$) and endogenous risk-taking (where the CEO controls volatility $\sigma_t$).

I choose to implement the Marinovic and Varas (2019) variant because it provides a clean baseline and is applicable to generic empirical settings. This is useful because when applying to APAC market, we may not have quasi-experiments like FAS 123-R to identify the additional limited attension channel.

*Replication: all results in this chapter are produced by the notebook `docs/14_ceo_contract_pipeline.ipynb`.*

**What are the key problems that the authors try to illustrate with the model?**

In this model, the CEO privately chooses both productive effort and performance manipulation to boost short-term cash flows at the expense of long-term firm value. Because the board cannot observe these actions directly, it uses the CEO's incentive compatibility constraints as a calibration dial to find the optimal balance between inducing effort and deterring manipulation. The board optimally designs a contract that tolerates some manipulation to maximize net firm value. The mechanics of this dynamic contract are governed by a single state variable representing the duration of the CEO's deferred pay. Crucially, the model captures an endogenous "horizon problem" driven by time. Early in the CEO's tenure, manipulation is naturally deterred because the CEO will personally suffer the future cash-flow reversals while still on the job. However, as retirement approaches, this natural deterrence vanishes. To maintain the CEO's effort without imposing inefficiently high post-retirement risk, the board optimally allows the duration of incentives to drop. This mathematically shifts compensation toward the short term, ensuring the CEO continues working hard but inevitably causing manipulation to escalate in their final years. Ultimately, by solving the principal's dynamic optimization problem, the firm's long-term value is maximized in a reality where short-termism is anticipated, managed, and optimally priced into the contract.

**Whether and how can we extend the basic model and the risky debt model to capture similar behaviors? For example, adding the board-CEO agents to the model.**



**How to estimate the new extended model with the CEO and the board?**


## Model of optimal CEO contract

### Environment

The model considers a firm (the principal) hiring a CEO (the agent) in continuous time.

**Time Horizon**: The model operates in continuous time $t \in [0, T]$, where $T$ is the CEO's deterministic retirement date.


**Clawback Period**: The firm can dictate post-retirement compensation and tie the CEO's wealth to the firm's performance until time $T + \tau$, where $\tau \ge 0$.


**Hidden Actions**: At every instant $t$, the CEO privately chooses two costly actions: productive effort $a_t$ and performance manipulation $m_t$.


### Cash Flows and Manipulation

The firm cannot observe the CEO's actions directly; it only observes the realized cash flow.

**Firm Cash Flow**: The observable performance measure evolves according to the stochastic differential equation:

$$dX_t = (a_t + m_t - \theta M_t)dt + \sigma dB_t$$



where $B_t$ is a standard Brownian motion and $\sigma$ is the exogenous cash flow volatility.


**Stock of Manipulation**: Manipulation borrows from the future. The accumulated stock of manipulation $M_t$ is defined as:

$$M_t = \int_0^t e^{-\kappa(t-s)} m_s ds$$



This stock depreciates at rate $\kappa$, while $\theta$ represents the marginal effect of manipulation on reducing current cash flows. The overall value-destroying effect of manipulation is denoted by $\lambda \equiv \frac{\theta}{r+\kappa} - 1$.


**CEO Preferences**: The CEO has Constant Absolute Risk Aversion (CARA) preferences with risk aversion $\gamma$, represented by $u(c, a, m) = -e^{-\gamma(c - h(a) - g(m))}/\gamma$.


**Cost of Actions**: Effort and manipulation are penalized via quadratic cost functions $h(a) = a^2 / 2$ and $g(m) = gm^2 / 2$.


**Private Savings**: The CEO can unobservably save or borrow at the risk-free rate $r$. This forces the marginal utility of consumption to be a martingale, vastly simplifying the optimal consumption path.

### State and Control Variables

Because the CEO can smooth consumption, the principal's dynamic contracting problem can be reduced to choosing two control variables that govern a single state variable.

**Summary State Variable ($z_t$)**: The contract is fully summarized by $z_t$, representing the ratio of the CEO's long-term incentives ($p_t$) to their total continuation utility ($W_t$). Formally, 

$$z_t \equiv -p_t / W_t. $$

This is a key variable of the model that captures the **duration of deferred compensation**. It also makes the model empirically tractable because it reduces both $p$ and $W$ to one single state variable.


**Control Variable 1 ($\beta_t$)**: The short-term **Pay-for-Performance Sensitivity (PPS)**. This dictates how the CEO's continuation utility responds to immediate cash flow shocks ($dB_t$).


**Control Variable 2 ($\sigma_{zt}$)**: The **performance sensitivity of long-term incentives**. This dictates how the duration of incentives ($z_t$) responds to immediate cash flow shocks ($dB_t$), effectively controlling the rate of performance-based vesting.


### Incentive Compatibility (IC) Constraint

The principal cannot dictate $a_t$ or $m_t$. Instead, the principal sets $\beta_t$ and $\sigma_{zt}$, and anticipates the CEO's self-interested response using First-Order Conditions (FOCs).

**1. Optimal Effort**: The marginal cost of effort must equal the marginal benefit (the short-term incentive $\beta_t$).



$$r\gamma a_t = \beta_t$$


**2. Optimal Manipulation**: Denote $\phi \equiv \frac{\theta}{r\gamma}$, the marginal cost of manipulation equals the short-term benefit ($\beta_t$) minus the long-term penalty of future cash flow reversal.

$$g'(m_t) = \frac{\beta_t}{r\gamma} - \phi z_t$$

Because the principal knows these explicit mappings, the principal's problem is mathematically reformulated to directly choose the target effort $a_t$ (which instantly pins down $\beta_t = r\gamma a_t$) and target manipulation $m_t$, subject to the CEO's FOC constraints.

### The Principal's Optimization Problem

The principal maximizes expected firm cash flows net of CEO compensation costs, subject to the evolution of the state variable $z_t$.

* **The Objective**:

$$F(z) = \max_{a_t, \sigma_{zt}} \mathbb{E} \left[ \int_0^T e^{-rt}(a_t - \lambda m_t - h(a_t) - g(m_t))dt - \int_0^{T+\tau} e^{-rt} \frac{\sigma^2 (r\gamma a_t)^2}{2r\gamma} dt \right]$$



Note: The final integral represents the risk premium cost of providing the short-term incentive $\beta_t = r\gamma a_t$.


* **The Constraint (Law of Motion for $z_t$)**:

$$dz_t = \left[ (r+\kappa)z_t + r\gamma a_t(\sigma\sigma_{zt} - 1) \right]dt + \sigma_{zt} dB_t$$



with the terminal condition $z_{T+\tau} = 0$.


**Terminal Value**: At retirement the firm still bears the risk of vesting the CEO's outstanding incentives over the clawback window $[T, T+\tau]$. This gives the value function's terminal condition $F(z, T) = -\frac{1}{2}\mathcal{C}z^2$, a convex penalty on the deferred-pay duration $z$ carried into retirement. The coefficient is closed-form (Marinovic-Varas 2019, Eq. 10),

$$\mathcal{C} = \frac{\sigma^2(r + 2\kappa)}{r\gamma\left(1 - e^{-(r + 2\kappa)\tau}\right)},$$

so a shorter clawback window (smaller $\tau$) makes deferral more expensive.

### Solution to principal's problem

The solution to the dynamic contract is not a closed-form formula over time, but rather a set of **policy functions** characterizing the optimal actions given the current state $z_t$ and time $t$.

**The Hamilton-Jacobi-Bellman (HJB) Equation**: The principal's problem resolves to the following HJB equation for the value function $F(z,t)$:

$$rF = \max_{a, \sigma_z} \pi(a,z) + F_t + \left[ (r+\kappa)z + a r\gamma(\sigma\sigma_z - 1) \right]F_z + \frac{1}{2}\sigma_z^2 F_{zz}$$

where $\pi(a,z)$ is the simplified flow payoff and the terminal condition is $F(z,T) = -\frac{1}{2}\mathcal{C}z^2$ (Section 5).

**Optimal Policy Functions**: By maximizing the HJB equation with respect to the controls, the model yields semi-parametric optimal policy functions that depend on the value function $F(z,t)$ and its derivatives.


**(1) Optimal Vesting Sensitivity**:

$$\sigma_z(z,t) = -r\gamma\sigma a(z,t) \frac{F_z}{F_{zz}}$$



Since $F$ is concave and decreasing in $z$, we have $\sigma_z(z,t) \le 0$: positive performance shocks optimally accelerate vesting (reduce $z_t$).


**(2) Optimal Manipulation**:

$$m(z,t) = \frac{1}{g}(a(z,t) - \phi z)^+$$


**(3) Optimal Effort**: The manipulation floor $m=(a-\phi z)^+/g$ makes the flow payoff kinked in $a$ at $a=\phi z$, so the effort FOC has two regimes. 

- **Interior** ($m>0$, i.e. $a>\phi z$), with the optimal $\sigma_z$ substituted back:

$$a(z,t) = \frac{g - \lambda + \phi z - r\gamma g F_z}{1 + g\left(1 + r\gamma\sigma^2 + r^2\gamma^2\sigma^2 \frac{F_z^2}{F_{zz}}\right)}.$$

- **Boundary** ($m=0$, i.e. $a\le\phi z$), where the manipulation-cost terms drop out of the FOC:

$$a(z,t) = \frac{1 - r\gamma F_z}{1 + r\gamma\sigma^2 + r^2\gamma^2\sigma^2 \frac{F_z^2}{F_{zz}}}.$$

The optimal effort uses the interior root where it exceeds $\phi z$, the boundary root where it falls below $\phi z$, and the kink value $a=\phi z$ in between.

Because the policy functions depend on the unknown value function $F(z,t)$, we solve the HJB equation numerically and then read the policies off the solution. Appendix A summarizes the method.


---

## Numerical Method

I solve the HJB equation with an **implicit finite-difference combined with policy-function iteration (FD-PFI)**. We work in reverse time $s=T-t$ with $f(z,s)=F(z,T-s)$, so the terminal condition becomes the initial condition $f(z,0)=-\tfrac12\mathcal C z^2$.

**Discretization.** Uniform state grid $z_0=0,\dots,z_{N}=z_{\max}$ (step $h$) and time grid $s_0=0,\dots,s_{M}=T$ (step $\Delta s$). For controls $(a,\sigma_z)$ and node $i$, with drift $\mu_i=(r+\kappa)z_i+ar\gamma(\sigma\sigma_z-1)$, the upwind generator is

$$(L^{a,\sigma_z}_h f)_i = \alpha_i\, f_{i+1} + \rho_i\, f_{i-1} - (\alpha_i+\rho_i+r)\,f_i,$$

with positive-coefficient weights $\alpha_i,\rho_i = \tfrac{\sigma_z^2}{2h^2} + \tfrac{1}{h}\,[\,\mu_i\,]^{\pm}$ (diffusion via the central second difference; drift via a forward or backward first difference, chosen so $\alpha_i,\rho_i\ge0$). The controls are restricted to bounded grids: effort uniform on $[0,\bar a]$ and vesting sensitivity $\sigma_z$ refined near $0$.

**Algorithm (FD + PFI).**

- **Input:** parameters, state/time grids, control grids, tolerance $\varepsilon$.
- **Initialize:** $f^0_i = -\tfrac12\mathcal C z_i^2$.
- **For** $n=1,\dots,M$ (backward in calendar time):
  1. Boundary values: $f^n_0=0$; $\;f^n_N=\dfrac{\pi(a_{\max},m_{\max})}{r}\big(1-e^{-rs_n}\big)-e^{-rs_n}\tfrac12\mathcal C z_{\max}^2$, where $a_{\max}=(r+\kappa)z_{\max}/(r\gamma)$ (absorbing edge, $\sigma_z=0$).
  2. Set the PFI iterate $\hat f \leftarrow f^{n-1}$.
  3. **Repeat** (policy-function iteration):
     - *Policy improvement* (grid search per interior node): $\;(a_i,\sigma_{z,i})=\arg\max_{a,\sigma_z}\big\{(L^{a,\sigma_z}_h\hat f)_i+\pi(a,z_i)\big\}$.
     - *Policy evaluation* (implicit step, fixed controls): solve the tridiagonal system $\;\big(I-\Delta s\,L^{a,\sigma_z}_h\big)f' = f^{n-1}+\Delta s\,\pi(a,z)$ with the step-1 Dirichlet rows.
     - Update $\hat f \leftarrow f'$; **until** $\max_i |f'_i-\hat f_i|/\max(1,|f'_i|) < \varepsilon$.
  4. Set $f^n \leftarrow \hat f$ and store $F(\cdot,\,T-s_n)=f^n$.
- **Output:** value function $F(z,t)$; recover policies $a,m,\sigma_z$ by applying the Section-6 closed forms to $F$.

The implicit matrix $I-\Delta s\,L^{a,\sigma_z}_h$ is a diagonally dominant M-matrix, so each evaluation step is monotone and the PFI converges; the scheme is monotone, stable, and consistent, hence it converges to the unique viscosity solution. The bounded control search keeps the policies well behaved where the closed-form FOC for $\sigma_z$ (which divides by $F_{zz}$) would be ill-conditioned.


## Numerical Results

I use finite-difference with policy iteration to numerically solve the model, following the Marinovic and Varas (2019) baseline calibration: risk-free rate $r=0.1$, CEO risk aversion $\gamma=1$, manipulation cost $g=1$ and marginal cash-flow impact $\theta=0.4$, manipulation depreciation $\kappa=0.3$, cash-flow volatility $\sigma=2$, retirement date $T=10$, and clawback window $\tau=5$. These imply the deterrence coefficient $\phi=4$, value-destruction $\lambda=0$. I solve on the duration range $z\in[0,0.30]$ and plot up to the operating range $0.25$.

### Value and policy surfaces

![CEO contract: value and policy surfaces over (z, t)](figures/bonus2-ceo-contract/surfaces_3d.png){#fig-ceo-3d}

@fig-ceo-3d reproduces the optimal value and policy functions over $(z,t)$ space. This closedly replicates figure 1 from the original paper. 

- The value function $F(z,t)$ is concave and decreasing in the deferred-pay duration $z$, anchored at $F(0,t)=0$. 
- Effort $a(z,t)$ rises with $z$: a larger long-term stake makes the CEO work harder. 
- Manipulation $m(z,t)$ is essentially zero early in tenure and at low duration, then escalates sharply as the CEO nears retirement. 
- Vesting sensitivity $\sigma_z(z,t)$ is close to zero early (the contract is almost deterministic) and turns negative near retirement, so good performance accelerates vesting (performance-contingent pay).

### Policy slices

@fig-ceo-2d provides a cleaner plot by slicing the 3D policy and value functions over the $z$ axis and the $t$ axis. This figure is directly comparable to Figure 5 in @marinovic2019ceo. All comparative statics are consistent with economic intuition underlying the model. One of the key insight is that manipulation only start to increase near retirement and is also increasing in the duration of deferred compensation $z$.

![CEO contract: policy slices in z and in t](figures/bonus2-ceo-contract/policy_slices.png){#fig-ceo-2d}

### The horizon problem

The central economic finding is the endogenous horizon problem. Early in the CEO's tenure, manipulation is naturally deterred: borrowing from future cash flows hurts the CEO while still on the job, so it is not worthwhile. As retirement approaches, this self-discipline fades. To keep the CEO exerting effort without loading inefficient risk onto the post-retirement window, the board optimally lets the duration of incentives $z$ fall. This shifts pay toward the short term and, as a by-product, lets manipulation escalate in the final years, an outcome the contract anticipates and prices in rather than eliminates.

### Validation checks

Because the model has no closed-form solution, we confirm the numerical solution two complementary ways. The **HJB residual** substitutes the solved value function and policies back into the HJB equation: a small residual means the solution actually satisfies the equation, so it passes (see notebook). 

The **Monte-Carlo value check** simulates CEO paths under the solved policy and compares the average discounted payoff to $F(z_0,0)$: agreement means the value function is consistent with the policy it implies, and because this check never touches the finite-difference grid, it is an independent confirmation that the solver is right. At the baseline the solution passes both: the residual is small across the state space (largest only at the single manipulation-kink node, where the value function bends sharply), and the simulated value matches $F(z_0,0)$ to about one percent.

The top panel shows that, averaged over the simulated cross-section, both the manipulation flow $m_t$ and its stock $M_t$ rise toward retirement, the horizon effect again. The bottom panel overlays the solved value $F(z_0,0)$ and the simulated mean discounted payoff across initial durations $z_0$; the two curves coincide, so the Monte-Carlo check passes.

![Numerical validation: simulated horizon effect (top) and Monte-Carlo value reconciliation (bottom)](figures/bonus2-ceo-contract/mc_validation.png)

---

## Structural Estimation (Plan)

To estimate the structural parameters of the dynamic CEO-contracting model using the Simulated Method of Moments (SMM), we must map the theoretical state and action variables to observable corporate data, similar to the empirical strategy employed by Cronqvist et al. (2024). However, Cronqvist et al. (2024) rely on the specific quasi-natural experiment of the FAS 123-R regulation to provide identifying variation for their behavioral "limited attention" parameter. Lacking a similar unique regulatory shock to provide exogenous identification, we cannot robustly estimate the attention parameter. This is another reason that I skip the extended "limited attention" model and implement the cleaner, generic baseline model of @marinovic2019ceo.

A critical requirement for SMM is that the observable variables used to construct the target moments must be strongly correlated with the underlying structural parameters. Without this correlation, the estimation lacks the data variation necessary for identification. The table below summarizes these essential variables, their data sources, and the theory-informed economic relationships that link the target moments to the parameters. While the structural model formally dictates these mapping relationships, one must remain cautious, as the actual strength of these identifying variations is ultimately an empirical question.

Currently, I only have access to Compustat for firm-level financial data and lack access to the CEO-specific databases (e.g., ExecuComp, ISS Incentive Lab, BoardEx) required to measure incentive duration and tenure. As a result, the full SMM estimation cannot be executed immediately. The table and subsequent methodology act as a concrete, executable plan that can be deployed as soon as the requisite executive compensation data becomes available.

| Variable | Definition | Source | Target Moments / Identified Parameters |
| :--- | :--- | :--- | :--- |
| **Firm Value (Market-to-Book)** | Market value of equity plus book value of debt, scaled by total assets. | Compustat, CRSP | **Target:** Sensitivity of firm value to incentive duration ($z_t$).<br>**Identifies ($g, \theta$):** Captures the value destroyed by manipulation. A higher sensitivity identifies the CEO's personal cost of manipulation ($g$) and the firm's destruction rate ($\theta$). |
| **Operating Cash Flows ($X_t$)** | Net cash flows from operating activities, scaled by lagged assets. | Compustat | **Target:** Autocorrelation (persistence) of cash flows.<br>**Identifies ($\kappa, \theta$):** Because manipulation artificially boosts current cash flows but reverses later, cash flow persistence identifies the depreciation rate ($\kappa$) and magnitude ($\theta$) of this reversal. |
| **Cash Flow Volatility ($\sigma$)** | Standard deviation of scaled operating cash flows. | Compustat | **Target:** Variance of operating cash flows.<br>**Identifies ($\sigma$):** Directly anchors the fundamental exogenous cash flow risk parameter ($\sigma$) in the stochastic process. |
| **CEO Incentive Duration ($z_t$)** | Weighted-average time to vesting of unvested restricted stock and unexercised options. | ExecuComp, ISS Incentive Lab | **Target:** Average duration and its trend over the CEO's tenure.<br>**Identifies ($g$):** If manipulation is "cheap" for the CEO (low $g$), the firm must rely heavily on long-term deferred pay (high $z_t$). |
| **Vesting Dynamics ($\Delta z_t$)** | Year-over-year change in the CEO's incentive duration. | ExecuComp, ISS Incentive Lab | **Target:** Sensitivity of $\Delta z_t$ to cash flow shocks ($X_t$).<br>**Identifies ($\sigma_z$):** Measures performance-based vesting. Identifies how aggressively the board uses positive shocks to accelerate vesting and reduce the CEO's risk exposure. |
| **CEO Horizon ($T-t$)** | Estimated years until CEO retirement. | ExecuComp, BoardEx | **Target:** Sensitivity of cash flows and duration to the CEO's remaining horizon.<br>**Identifies (Horizon Effect):** Provides the time-series variation to identify the escalating manipulation (and dropping cash flows/firm value) as the CEO approaches retirement. |


## Adding the CEO and the board to risky debt model

I provide a tentative sketch of how we may add the principal-agent problem (the CEO and the board) into a standard corporate model with capital investment and debts. I choose to modify the moral hazard model used in @nikolov2021. This modeling choice, however, face a trade-off that we lose the elegent semi-parametric model implications from the original @marinovic2019ceo and @cronqvist2024. I'll dicuss these details at the end of the section.

**1. Environment and Technology**
Time is discrete and infinite ($t \ge 0$). Both the board (principal) and the CEO (agent) are risk-neutral and discount future cash flows at rate $r$. The firm operates using physical capital $k_t$. The board chooses next period's capital $k_{t+1}$, which implies an investment $i_t = k_{t+1} - (1-\delta)k_t$. Investment incurs a convex adjustment cost $\Psi(k_{t+1}, k_t)$.

The firm's operations are subject to two shocks:

* $z_t$: A persistent productivity shock following a publicly observable Markov process $Q_z(z, z')$.
* $\eta_t$: An i.i.d. transient cash flow shock observed **privately** by the CEO.

**2. The Agency Friction: Hidden Effort and Diversion**
In each period, the CEO makes a hidden effort choice $e_t$, incurring a private cost $c(e_t)$. The firm's true pre-tax operating cash flow is given by a Cobb-Douglas production function augmented by effort and shocks:


$$ \pi(k, z, \eta, e) = z k^\alpha + \eta + e - f $$


where $\alpha \in (0,1)$ is the capital share and $f$ is a fixed cost.

Because the board only observes the final reported cash flow, the CEO can manipulate it. Upon observing the true shock $\eta_t$, the CEO can report a fake shock $\hat{\eta}_t$. The board, expecting the CEO to have exerted the recommended effort $e(z_t, \hat{\eta}_t)$, anticipates a cash flow of $z_t k_t^\alpha + \hat{\eta}_t + e(z_t, \hat{\eta}_t) - f$.

The CEO diverts the difference between the true cash flow generated and the reported cash flow expected by the board:


$$ m_t = \eta_t + e_t - \hat{\eta}_t - e(z_t, \hat{\eta}_t) $$


For every unit of diverted cash flow, the CEO privately pockets a fraction $\lambda \in (0, 1]$, representing the inefficiency of manipulation.

**3. The Principal's Contracting Problem**
The board designs a long-term contract to maximize total firm value $W(k, V, z)$, using the CEO's promised continuation utility $V$ as a state variable. At the end of period $t-1$, given state $(k, V, z)$, the board chooses next period's capital $k'$, and a menu of state-contingent dividends $d(z', \eta')$, continuation values $V(z', \eta')$, and recommended effort levels $e(z', \eta')$.

The Principal's Bellman equation is:


$$ W(k, V, z) = \max_{k', V(\cdot), d(\cdot), e(\cdot)} \frac{1}{1+r} \left[ -k' + (1-\delta)k - \Psi(k', k) + \mathbb{E}_{z', \eta'} \left[ \pi(k', z', \eta', e_{z',\eta'}) - d_{z',\eta'} + W(k', V_{z',\eta'}, z') \right] \right] $$

This maximization is subject to three constraints:

* **Promise Keeping (PK):** The contract must deliver the expected utility $V$ promised to the CEO.

$$ V = \frac{1}{1+r} \mathbb{E}_{z', \eta'} \left[ d_{z',\eta'} - c(e_{z',\eta'}) + V_{z',\eta'} \right] $$


* **Incentive Compatibility (IC):** The CEO must be better off reporting the truth ($\hat{\eta}' = \eta'$) and exerting the recommended effort ($e'$), rather than executing any joint deviation $(\hat{\eta}', \hat{e})$.

$$ d_{z',\eta'} - c(e_{z',\eta'}) + V_{z',\eta'} \ge d_{z',\hat{\eta}'} - c(\hat{e}) + V_{z',\hat{\eta}'} + \lambda \left[ \eta' + \hat{e} - \hat{\eta}' - e_{z',\hat{\eta}'} \right] \quad \forall z', \eta', \hat{\eta}', \hat{e} $$


* **Limited Liability (LL):** Payouts and continuation values must be non-negative.

$$ d_{z',\eta'} \ge 0, \quad V_{z',\eta'} \ge 0 \quad \forall z', \eta' $$



**4. Model Solutions and Implied Debt**
The solution to this dynamic contracting problem yields a value function and a set of policy functions. The **value function** $W(k, V, z)$ maps the firm's current state—its physical capital $k$, the CEO's promised utility $V$, and the economic environment $z$—to the maximum total expected firm value. The **policy functions** map this exact same state $(k, V, z)$ to the optimal decisions the board makes today: the investment policy (choosing $k'$), the payout policy (dividends $d$), the continuation policy (future promised value $V'$), and the recommended effort policy ($e$).

While the firm's debt is not a state variable, the optimal contract dynamically implies a capital structure. The value of the outside investors' claim (debt) can be recovered as a simple residual. It is the total firm value minus the equity value promised to the CEO: $b(k, V, z) = W(k, V, z) - V$.

### Discussion: Model Comparisons and Trade-Offs

**1. Differences from the original model in** @marinovic2019ceo

* **What is Gained:** By explicitly tracking physical capital, we can study real-world corporate investment. The model illustrates how a firm might cut back on expanding or buying equipment—not because a bank refuses to lend, but because the board must keep cash inside the firm to manage the CEO's temptation to steal. It directly connects the physical growth of the firm to the severity of the CEO's moral hazard.
* **What is Lost:** We lose the mathematical simplicity and explicit formulas of the MV19 framework. Because we now track three separate state variables $(k, V, z)$ instead of one, the problem becomes a massive computational grid. We can no longer rely on clean calculus to see how variables interact; we must rely entirely on heavy computer simulations.
* **Economic Limits:** Crucially, we lose the CEO's "horizon" effect. MV19 explicitly models the CEO's timeline to retirement, proving that manipulation escalates as the CEO's departure approaches. By moving to an infinite-horizon setup to accommodate capital accumulation, the CEO never retires. This means we cannot use the model to explain or estimate how age, tenure, or impending retirement impacts corporate fraud. Finally, we lose the smooth compensation structure. In MV19, the CEO receives a steady stream of income; in this risk-neutral model, the CEO is paid nothing for years until their performance crosses a high threshold, at which point they receive a massive cash payout.



**2. Differences from the original model in** @nikolov2021

* **What is Gained:** The original Nikolov model only focuses on the CEO hiding bad performance. By adding an effort choice, we capture a more realistic, dual-agency friction. The board now faces a difficult balancing act: pushing the CEO to work harder increases the firm's actual profits, but it simultaneously increases the CEO's incentive and opportunity to steal those profits. 
* **What is Lost:** We lose computational speed and estimation flexibility. In the original model, the board only had to ensure the CEO wouldn't lie about the numbers. Now, the board must ensure the CEO doesn't simultaneously slack off *and* lie about the numbers to cover it up. The computer must check vastly more "what if" scenarios to ensure the contract is truly manipulation-proof. This heavy computational burden limits how many additional features or structural parameters we can reliably estimate when taking the model to real-world data.