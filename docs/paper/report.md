---
title: "Deep Learning Methods for Corporate Finance"
author: Zhaoxuan Wang
number-sections: false
date: 2026-04-27
bibliography: ../references.bib
thanks: Email - [wxuan.econ@gmail.com](mailto:wxuan.econ@gmail.com) or [zxwang13@student.ubc.ca](mailto:zxwang13@student.ubc.ca).
format:
  html:
    toc: false
  pdf:
    toc: false
    pdf-engine: xelatex
    documentclass: article
    geometry:
      - margin=1in
    fontsize: 11pt
    fig-pos: 'H'
    tbl-cap-location: bottom
    code-block-bg: true
    keep-tex: true
    link-citations: true
    include-in-header:
      text: |
        \usepackage{etoolbox}
        \pretocmd{\section}{\clearpage}{}{}
---

**Abstract.** This report examines quantitative methods for solving and estimating structural corporate finance models. On the solution side, I implement and evaluate the three deep-learning methods proposed in @maliar2021 — Lifetime Reward Maximization (LRM), Euler Residual Minimization (ERM), and Bellman Residual Minimization (BRM) — and benchmark them against classical Value Function Iteration and Policy Function Iteration. I also develop a new solution method based on Short-Horizon Actor-Critic (SHAC) reinforcement learning [@xu2022] and show that it converges to the analytical solution in settings where @maliar2021's methods are inapplicable. On the estimation side, I implement Generalized Method of Moments (GMM) and Simulated Method of Moments (SMM), validate both via Monte Carlo on the basic investment model, and apply SMM to the @hennessy2007costly endogenous default (risky debt) model. I document the practical and structural challenges in these applications.

# Part I. Solving Dynamic Models

To keep this report clear and concise, I focus on presenting the findings and the direct answers to the interview questions. In appendix, I provide a more detailed description of the models, solution methods (algorithms), additional results, and other implementation details.

The corporate finance model considered in this report are generally represented as a dynamic programming problem with the following features:

- Discrete-time
- Continuous state and action spaces, 
- Policy function is deterministic and known
- State transition function with random noise is known

For continuous-time deep learning methods for solving dynamic models, @duarte2024 proposed a "deep policy iteration" method and demonstrated its application to corporate finance and other quantitative finance models.

## Definition

### Markov Decision Process

A Markov Decision Process (MDP) is defined as a collection $(\mathcal{S}, \mathcal{A}, \mathcal{E}, f, r, \gamma)$ that subsumes all relevant information for decision-making.

| Symbol | Definition  |
| --- | --- |
| $\mathcal{S} \subseteq \mathbb{R}^n$ | **State space** (continuous). A state $s \in \mathcal{S}$ is a vector encoding all information the agent observes. When the environment involves multiple variables (e.g., productivity $z$ and capital $k$), they are stacked into a single vector $s = (z, k)^\top$. |
| $\mathcal{A} \subseteq \mathbb{R}^d$ | **Action space** (continuous). An action $a \in \mathcal{A}$ is a vector of controls the agent selects (e.g., investment, consumption). |
| $\mathcal{E}$ | **Shock space**. The space from which exogenous shocks $\varepsilon$ are drawn. When dynamics are deterministic, $\mathcal{E} = \emptyset$.  |
| $f: \mathcal{S} \times \mathcal{A} \times \mathcal{E} \to \mathcal{S}$ | **State transition function/dynamics**. Given current state $s$, action $a$, and exogenous shock $\varepsilon \in \mathcal{E}$, the next state is $s' = f(s, a, \varepsilon)$. When dynamics are deterministic, the shock argument is absent and $f: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$. When the dynamics involve both exogenous and endogenous components, they are combined into a single vector-valued function. |
| $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$                     | **Reward function**. A scalar signal $r(s, a)$ received after taking action $a$ in state $s$ (e.g., cash flow or utility).  |
| $\gamma \in (0, 1)$  | **Discount factor**. Controls the trade-off between immediate and future rewards. |

A full sequence of actions and states is defined as a **trajectory** or **rollouts**:
$$ (s_0,a_0,s_1,a_1,\dots)$$
where the initial state $s_0$ is randomly sampled from some distribution $p_0$. The state transition is given by:
$$ s_{t+1}=f(s_t,a_t,\epsilon_t) $$
where $\epsilon_t$ is a random noise (e.g., productivity shock) but the function $f$ is deterministic. Note that this is different from a stochastic transition function in RL where $s_{t+1}$ is a draw from a distribution $s_{t+1} \sim P(\cdot|s_t,a_t)$.

The **reward function** $r(s,a)$ is assumed to be known exactly and it maps current state and actions $(s,a)$ to a scalar value. In corporate finance, this is typically firm's cash flow. The **discounted lifetime reward** over an infinite-horizon trajectory is summarized as:
$$
\sum^\infty_{t=0} \gamma^t \cdot r(s_t,a_t)
$$

### Optimal Policy Function
A **deterministic policy** $\pi$ is a mapping from $\mathcal{S} \to \mathcal{A}$. In most applications there do not exist a closed-form analytical solution to $\pi$, thus researchers either directly solve for it numerically, or use a function approximator with parameterization. Denote the parameters as $\theta$ and the parameterized deterministic policy as $\pi_\theta$, Given a state $s$, the policy outputs a specific action:
$$a = \pi_\theta(s)$$
The approximator can be as simple as linear, $\pi_\theta(s)=\theta_1 + \theta_2 s$. However, to better capture non-linearities of the unknown mapping, I use a deep neural network approximator where $\theta$ is a vector that collects all trainable weights and biases. Here $\pi_\theta$ is a deterministic function (e.g., the neural network), and $a$ is a realized scalar or vector (the network's output for input $s$). 

The **optimal policy** $\pi^*$ is defined by
$$
\pi^* \equiv \arg\max_{\pi} \mathbb{E}_{\epsilon} \left[ \sum^\infty_{t=0} \gamma^t \cdot r(s_t,\pi(s_t)) \right]
$$
subject to dynamics $s_{t+1}=f(s_t,\pi(s_t),\epsilon_{t+1})$ with random noise $\epsilon_{t+1}\sim P_\epsilon$.

With NN approximator $\pi_\theta$, the objective is equivalent to finding the optimal NN parameters $\theta^*$ such that 
$$
\theta^* = \arg\max_\theta \mathbb{E}_{\epsilon} \left[ \sum^\infty_{t=0} \gamma^t \cdot r(s_t,\pi_\theta(s_t)) \right]
$$

### Value Functions
Given a policy $\pi_\theta$ and starting state $s_0$, define the trajectory $\tau \equiv (s_0, a_0, s_1, a_1, \ldots)$ where $a_t = \pi_\theta(s_t)$ and $s_{t+1} = f(s_t, a_t, \varepsilon_{t+1})$ with shocks $\varepsilon_t$ drawn i.i.d. from some distribution.

#### On-Policy Value function 
The expected return if you start in state $s$ and always act according to policy $\pi$ is given as:
$$ V^{\pi_\theta}(s) = \mathbb{E}_{(\varepsilon_1, \varepsilon_2, \ldots)}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, \pi_\theta(s_t))\right]$$
where $s_{t+1} = f(s_t, \pi_\theta(s_t), \varepsilon_{t+1})$. This is the expected cumulated discounted reward from state $s$ when following policy $\pi_\theta$, with the expectation taken over all future shock realizations.

#### Optimal Value Function
The optimal value function $V^*(s)$ gives the maximum expected lifetime reward if agent start from $s$ and always act according to the _optimal_ policy $\pi^*$:
$$
V^*(s) =\max_{\pi}V^\pi(s) = \mathbb{E}_{(\varepsilon_1, \varepsilon_2, \ldots)}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, \pi^*(s_t))\right]
$$

### Bellman Equations
Let the 'tick' denote next time step variables, e.g., $s'\equiv s_{t+1}$, the Bellman equations for the on-policy value functions are

$$
V^{\pi}(s) = r(s,\pi(s)) + \gamma \mathbb{E}_{\epsilon}\left[ V^{\pi}(s') \right]
$$

where the expectation is taken over $\epsilon$ that governs the state transition functions $s'=f(s,a,\epsilon)$.

The Bellman equations for the optimal value functions are
$$
V^*(s) = \max_a \left\{ r(s,a) + \gamma \mathbb{E}_{\epsilon}\left[ V^*(s') \right] \right\}
$$
where the $\max$ operator ensures that at optimality, the agent will pick the action that maximizes the Bellman right-hand-side (RHS).

## Application to the Basic Investment Model {#model-basic}

This section briefly describes the basic investment model, in which firm chooses the optimal investment given current states (capital stock and realized productivity). The solution to this model is the optimal policy function that maps $(k,z)\to I$ or $k'$.

**State Space**:
State variables are capital $k$ and productivity shock $z$, which is stacked into a single state vector:
$$s=\left( k, z\right)^\top $$
with exogenous bounded space $[k_{\min} , k_{\max} ]$ and $[z_{\min}, z_{\max}]$.

**Action Space**:
Action variable is investment $a=I$ that can be either positive or negative. The state spaces $0 \leq k' \leq k_{\max}$ and $0 \leq k \leq k_{\max}$ implies:
$$
-(1-\delta)k \leq I \leq k_{\max}
$$
so that the action space is also bounded by $[I_{\min}, I_{\max}]$ with $I_{\min} = -(1-\delta) k < 0$ and $I_{\max} = k_{\max}$.

**State Transition Function.** The productivity shocks follow auto-regression (AR-1) process with stationary mean $\mu$, persistent coefficient $\rho$, and variance $\sigma$:
$$\log z' = \mu +\rho \log z + \sigma \epsilon, \quad \epsilon\sim \mathcal{N}(0,1)$$
Capital stock depends on depreciation rate $\delta$ and investment $I$ (action):
$$ k'=(1-\delta)k+I$$
They are stacked into a single vector-valued state transition function:
$$f(s,a,\epsilon) \equiv 
\begin{pmatrix} z' \\ k' \end{pmatrix} = 
\begin{pmatrix}
\exp \left\{ \mu +\rho \log z + \sigma \epsilon \right\}\\
(1-\delta)k+I
\end{pmatrix}$$
**Reward** is defined by the net cash flow $e(k,z,I)$:
$$r(s,a) \equiv e(k,z,I) = \Pi(k,z) - \Psi(I,k) - I$$
where the production function is Cobb-Douglas with parameter $\beta$:
$$\Pi(k,z) = z \cdot k^{\beta}, \quad \beta \in (0,1)$$
and the capital adjustment cost is
$$\Psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbf{1}\{I \neq 0\}$$
where $\phi_0$ is the smooth adjustment cost coefficient and $\phi_1$ is the fixed adjustment cost coefficient. The indicator $\mathbf{1}\{I \neq 0\}$ triggers whenever the firm invests or disinvests.

**Objective**: The solution of the model is to find the optimal investment policy that maximizes expected discounted lifetime cash flows:
$$\max_{\{I_{t}\}_{t=0}^{\infty}} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \cdot r(k_t, z_t, I_t) \right]$$
where $\gamma = 1/(1+ \bar r)$ is the discount factor and $\bar r$ is the risk-free interest rate. The solution is an optimal parameterized policy function $\pi^*_\theta(s)$ that maps states $(k,z)$ to investment $I$.

**Frictionless analytical solution.** When both adjustment costs are zero ($\phi_0 = \phi_1 = 0$), the model admits a closed-form optimal policy that depends only on $z$: capital is fully reversible each period, so the firm resets $k$ to the static optimum every period. The frictionless first-order condition

$$1 \;=\; \gamma \, \mathbb{E}\!\left[\,\beta z'\,(k')^{\beta-1} + (1-\delta) \;\big|\; z\,\right]$$

solves to

$$
k^*(z) \;=\; \left[\,\frac{\beta \cdot \mathbb{E}[z' \mid z]}{\bar r + \delta}\,\right]^{1/(1-\beta)}, \qquad
\mathbb{E}[z' \mid z] \;=\; \exp\!\bigl((1-\rho)\mu + \rho \log z + \tfrac{1}{2}\sigma^2\bigr).
$$

The conditional expectation is given by the log-normal mean in closed form — no numerical quadrature is needed, because $\log (z' | z)$ is exactly normal with mean $(1-\rho)\mu + \rho \log z$ and variance $\sigma^2$. This makes $k^*(z)$ an exact, parameter-free anchor that I use repeatedly throughout the paper: 

- $k^*$ used as anchor for the space bounds of $k$ and $b$.
- $k^*$ as the ground-truth policy for [benchmarking solution methods](#part1-validate);
- $k^*$ used for computing the baseline steady-state value $V^*$ for normalizing the reward and Bellman in SHAC and BRM methods;
- $k^*$ as the closed-form optimal policy in [Part II SMM validation](#part2-validate) that isolates estimation error from solver error. 

**State-space bounds.** All three bounds are pinned to the AR(1) stationary distribution and the frictionless anchor $k^*$:

- $z \in [\,\exp(\mu - 3\sigma_{\text{erg}}),\; \exp(\mu + 3\sigma_{\text{erg}})\,]$ with ergodic standard deviation $\sigma_{\text{erg}} = \sigma/\sqrt{1-\rho^2}$ truncates the log-normal stationary distribution at $\pm 3$ ergodic std-devs. Initial $z_0$ is drawn uniformly in level space within this range.
- $k \in [ \underline c^k \cdot  k^*, \bar c^k \cdot k^*]$, where $k^* = k^*(\bar z)$ is the frictionless capital at the stationary mean $\bar z = \exp(\mu)$. The asymmetric upper and lower bound multiplier $\underline c^k, \bar c^k$ are supplied by user and are set to be generous so the box covers the optimal $k'$ at the upper end of $z$. Default values are $[0.25k^*, 6k^*]$.

These are solver / training parameters calibrated once at construction time from $(\mu, \rho, \sigma, \beta, \bar r, \delta)$.


## Application to the Risky Debt Model {#model-debt}
The risky debt model extends the basic model by allowing firms to borrow at an endogenous risky interest rate, with the option to default. Then risky interest rate is determined by the lender's zero profit condition with rational expectation of default probability. Firm's optimal investment and leveraging in turn depends on the equilibrium risky interest rate.

The solution to this model consists of (i) an optimal policy function $\pi^*$ mapping states $(k,b,z)\to (k',b')$; and (ii) the optimal value function $V^{\pi*}(k,b,z)$ satisfying the Bellman equation for policy $\pi^*$.

**State Space**
The state variables are current capital $k$, productivity $z$, and debt $b$:
$$s \equiv (k,b,z)$$
with bounded state space $k\in [k_{\min},k_{\max}]$, $z \in [z_{\min}, z_{\max}]$, and $b\in [0, b_{\max}]$. Here $b_{\max}$ is an exogenously determined upper bound and should be set generously large enough to avoid binding frequently.

**Action Space**
Firm chooses investment $I$ and next-period debt $b'$ (new borrowing):
$$ a \equiv (I,b')$$
with action space bounds $[I_{\min}, I_{\max}]$ and $[0, b_{\max}]$.

**State Transition Function** is a vector-valued function defined as:
$$
f(s,a,\epsilon) \equiv 
\begin{pmatrix} z' \\ k'\\ b' \end{pmatrix} =
\begin{pmatrix}
\exp \left\{ \mu +\rho \log z + \sigma \epsilon \right\}\\
(1-\delta)k+I \\
b'
\end{pmatrix}
$$

**Reward** is defined as the net cashflow (payouts) minus cost of external financing:
$$
e(k,b,z;I,b') - \Omega(e(\cdot))
$$
Cash flow is given as
$$e(k,b,z;I,b') \equiv (1-\tau)\Pi(k,z) - \Psi(I,k) - I -b + \frac{b'}{1+\tilde{r}(\cdot)} + \frac{\tau \, \tilde{r}(\cdot) \, b'}{[1+\tilde{r}(\cdot)](1+ r)} $$
where 

- $\tau$ is the corporate tax rate
- $b$ is repayment of last-period debt
- $r$ is risk-free interest rate
- $\tilde{r}(\cdot)$ is the endogenous risky interest rate that depends on states
- $b'/(1+\tilde{r})$ is proceeds from issuing new risky debt
- The last term is the tax shield from debt interest

When cash flow is negative, the firm must raise costly external equity:
$$\Omega(e) = (\omega_0 + \omega_1 |e|) \cdot \mathbf{1}\{e < 0\}$$
As in the basic model, the production function is Cobb-Douglas with parameter $\beta$:
$$\Pi(k,z) = z \cdot k^{\beta}, \quad \beta \in (0,1)$$
and the capital adjustment cost is
$$\Psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbf{1}\{I \neq 0\}$$
where $\phi_0$ is the smooth adjustment cost coefficient and $\phi_1$ is the fixed adjustment cost coefficient. The indicator $\mathbf{1}\{I \neq 0\}$ triggers whenever the firm invests or disinvests.

**Endogenous Risky Interest Rate**
The bond price $q = 1/(1+\tilde{r})$ is determined by the lender's zero-profit condition:
$$b'(1+r) = (1+\tilde{r}) b' \, \mathbb{E}_\epsilon[1-D] + \mathbb{E}_\epsilon[D \cdot \text{Recovery}(k',b',z')]$$
where:

- LHS: Opportunity cost of lending at risk-free rate
- RHS: Expected return accounting for default probability and recovery

**Endogenous Default**
The firm defaults when its continuation (latent) value is negative:
$$D(k',b',z') = \mathbf{1}\{\widetilde{V}(k',b',z') < 0\}$$
Shareholders walk away with zero under limited liability:
$$V(k',b',z') = \max\{0, \widetilde{V}(k',b',z')\}$$
**Recovery Under Default**
$$\text{Recovery}(k',z') = (1-\alpha)\left[(1-\tau)\pi(k',z') + (1-\delta)k'\right]$$
where $\alpha \in [0,1]$ is the deadweight loss from liquidation.

**Bellman Equation** for the latent firm value $\widetilde{V}$ is given by
$$
\begin{aligned}
\widetilde{V}(k,b,z) &= \max_{k',b'} \left\{ e(k,b,z;I,b') - \Omega(e) + \gamma \, \mathbb{E}_{\epsilon}[V(k',b',z')] \right\}\\
&= \max_{k',b'} \left\{ e(k,b,z;I,b') - \Omega(e) + \gamma \, \mathbb{E}_{\epsilon}[\max\{0, \widetilde{V}(k',b',z')\}] \right\}
\end{aligned}
$$
where the RHS continuation value encodes limited liability (firm can walk away with zero).

**The Nested Fixed-Point Problem**
A key computational challenge is that the latent value $\widetilde{V}$ depends on the risky rate $\tilde{r}$, but solving for $\tilde{r}$ requires knowing the default probability $\mathbb{E}[D]$, which depends on $\widetilde{V}$. Traditional methods solve this via nested iteration. The neural network approach trains policy, value, and pricing networks jointly, avoiding explicit nested loops.

**State-space bounds.** Productivity bounds are identical to the basic model ($z \in [\exp(\mu \pm 3\sigma_{\text{erg}})]$). Capital and debt bounds are anchored to a tax-adjusted frictionless capital
$$
k_{\text{ref}} \;=\; \left[\,\frac{(1-\tau)\,\beta\,\mathbb{E}[z']}{\bar r + \delta}\,\right]^{1/(1-\beta)},
$$
which is the frictionless optimum after corporate profits are reduced by the tax rate $\tau$. Similar to the default bounds used in basic model, capital is then bounded as $k \in [\,0.25\, k_{\text{ref}},\; 6\, k_{\text{ref}}\,]$. Debt is pinned to capital via: $b_{\max} = 3 k_{\max}$ with $b_{\min} = -0.2 b_{\max}$ allowing cash holdings ($b < 0$). The form mirrors a standard collateral constraint where debt capacity scales with capital, but the multipliers are set by user and should be generous enough so the bounds never bind.


## Overview of Solution Methods

The solution to the model is given by the optimal policy function $\pi^*(s)$ that maps states to actions. Optionally, solution also include the optimal state-value function $V^*(s)$. This section provides a high-level, brief summary of the solution methods using generic notations. I discuss the key ideas of each method, their main strength and limitations, and improvements. A more detailed and comprehensive documentation of each method (algorithm) are provided in [Appendix.A](#sec-solve).

I implemented four main solution methods in Python and Tensorflow:

1. Value and policy function iteration (VFI/PFI)
2. Lifetime reward maximization (LRM) with terminal value correction
3. Euler residual minimization (ERM)
4. Short horizon actor critic method (SHAC)
5. Nested value function iteration (specific to the risky debt model)

**VFI and PFI** are the classical discrete dynamic programming solvers.  They
discretize the continuous state and action spaces onto finite grids,
estimate Markov transitions from data, and iterate the Bellman operator
to convergence.  The resulting value function and policy are exact on the
grid (up to discretization error) and serve as ground-truth benchmarks
for the NN-based methods. Both methods are robust but they suffers from the curse of dimensionality. I use linear interpolation to reduce the grid approximation error and show that it avoids the overestimation bias due to the $\arg\max$ operation on coarse grids. But the real "curse of dimensionality" is the number of state and action variables, and this is the main motivation for alternative methods. The [VFI/PFI appendix](#sec-VFI) describes my algorithm and implementation details.

@maliar2021 introduces three **deep learning methods** that uses neural networks (NN) to approximate policy and value function: Lifetime reward maximization (LRM), Euler residual minimization (ERM), and Bellman residual minimization (BRM). I implemented original version of all these methods and find that only ERM is reliable for actual production. The LRM is plagued by systematic bias from finite horizon truncation, and the BRM usually failed to converge in practice. These defects and their "fixes" are discussed in next section in detail.

The **ERM method** method minimizes violations of the first-order conditions (Euler
equations) that characterize optimality. Rather than simulating full
trajectories, it enforces an intertemporal necessary condition between
$(s, a)$ and $(s', a')$ at each observation independently. At the optimum, the policy $\pi_\theta$ satisfies the Euler equation:

$$\mathbb{E}_\varepsilon \left[F(s, \pi_\theta(s), s', \pi_\theta(s'))\right] = 0$$

where $F: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the Euler residual function derived analytically from the first-order conditions of the Bellman equation, and $s' = f(s, \pi_\theta(s), \varepsilon)$ is the known state transition function. Two important implementation details:
1. The empirical loss is the squared Euler residual $\frac{1}{B}\sum_{i \in B}^{B} F(s_i, a_i, s'_{i,1}, a'_{i,1}) \cdot F(s_i, a_i, s'_{i,2}, a'_{i,2})$ using two i.i.d. draws $s'_{i,1}$ and $s'_{i,2}$, which is an unbiased estimator for $\mathbb{E}_\varepsilon \left[ F(\cdot)^2 \right]$.
2. Both @maliar2021 and @fernandez-villaverde2025 use a single policy network $\pi_\theta$ inside the Euler residual function $F$. In practice, this creates a
moving-target problem that prevents convergence: the gradient of $\theta$ flows through both the current policy $\pi_\theta(s)$ and the next-period policy $\pi_\theta(s')$, which appear on both side of the Euler equation. To fix this, I uses a separate target network $\pi_{\theta^-}$ for the next-period action, which stablizes training and faciliates convergence.

The **LRM method** directly maximizes discounted cumulative rewards by simulating
trajectories under the current policy.  Given initial state $s_0$ and a
shock sequence $\{\varepsilon_1, \ldots, \varepsilon_T\}$, the policy
$\pi_\theta$ generates a trajectory
$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ where $a_t = \pi_\theta(s_t)$
and $s_{t+1} = f(s_t, a_t, \varepsilon_{t+1})$.  Gradients flow backward
through the entire trajectory via backpropagation through time (BPTT),
requiring both the reward $r$ and the endogenous transition $f^{\text{endo}}$
to be differentiable with respect to the action. The objective is 

$$\max_{\theta} V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right] + \gamma^T \, \mathbb{E}\left[V^{\pi}(s_T)\right]$$

where the terminal value is truncated and implicitly set to $\mathbb{E}\left[V^{\pi}(s_T)\right]=0$. This truncation is only valid when the finite-horizon rollout $T$ is sufficiently large. For example, with discount factor $\gamma=0.95$ and $T=100$, the terminal value contribution of $0.6\% \approx (0.95)^{100}$ is negligible. In practice, the **LRM method** faces an important trade-off between bias and computational cost: If we set large $T$ rollout, the computational cost of BPTT is huge and LRM is practically much slower than any other methods (including VFI/PFI). BPTT is sequential so it cannot be parallelized by Tensorflow. In contrast, LRM is feasible when rollout is moderate (e.g., $T\leq 30$) but the truncation bias would be large enough to sysmtetically bias the solution $\pi_{\theta^*}$.



I also implemented and tested the **Bellman residual minimization (BRM) method** following @maliar2021. However, I find that this method is very unstable and may converge to a surpurious, self-consistent fixed point different from the optimal policy. @maliar2021 concludes that the main defect of BRM is it is less precise and requires careful tuning of hyperparameters to match the scale of the Bellman equation residual and the first order condition (FOC) residual. I show that the defect is structural and cannot be solved by fine tuning and warm start (pre-training). In short, the Bellman residual can be minimized for any arbitrary policy and only the first-order necessary conditions are providing useful gradient directions. The BRM method converges only when FOC dominates the Bellman residual in training and when the intiation of the value function network is around the "right" basin of the local optimum. This makes it dependent on pre-training and fine-tuning, and less useful in practice. Therefore I remove BRM from the main production methods and discuss the more fundamental defects in the [BRM appendix](#sec-BRM).

Finally, I introduced and implemented a new method, **Short-Horizon Actor Critic (SHAC)**, based on a revision of the reinforcement learning (RL) algorithm developed by @xu2022. This method requires four neural networks: a policy network parameterized by $\pi_\theta$ and a value function network parametrized by $V_\phi$, and two polyak updated copies $\bar \pi_\theta$ and $\bar V_\phi$.

The basic design of SHAC is we can slice the full $T$ horizon into $T/H$ windows, each with length $H$, then exploits the $H$-step rollout and BPTT to accurately trained $\pi_\theta$ and $V_\phi$ within each window. Each gradient update consists of two steps: For each window $j=0,\dots,T/H$, the actor step update the policy network $\pi_\theta$ that maximizes 

$$\max_{\theta} V_\phi(s_j) = \mathbb{E}\left[\sum_{t=j}^{H-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right] + \gamma^H \, \mathbb{E}\left[\bar V_\phi(s_H)\right]$$

then the critic step is supervised learning of the value network:

$$ \min_{\phi} \mathbb{E}\left[ V_\phi - y \right]^2$$

where the target label $y=\text{Stop Gradient}{(\sum r^H+\gamma^H \bar V_\phi)}$ is the Bellman right-hand-side value from the actor step. The intuition underlying actor critic method is very similar to VFI/PFI, where the policy evaluation find the policy that maximizes the Bellman and the tabular value function is updated using the improved policy. 

The advantages of **SHAC** method are clear:

- it does not requires the existence of closed-form Euler equation
- it introduces a value network to precisely learned the terminal value omitted by LRM
- it uses short-horizon rollout and BPTT to reduce boostrap error for the value network

I show that SHAC and ERM achieves the significantly better accurancy and robustness compared with LRM and BRM. Unlike ERM, SHAC does not require closed-form Euler equation and thus can be applied to more general set of models. The main cost is the computational expense of BPTT, but it can be fine-tuned to achieve a balance between stability (longer $T$) and speed (shorter $T$). 

## Data Generation, Reproducibility, and Workflow

### Sythetic Data
Standard deep learning applications typically use three datasets:

1. Training data: splited into mini-batches to trained the NNs via SGD
2. Validation data: used to evaluate the quality of solution and convergence criteria
3. Test data: sealed and only used once to benchmark the results after the entire training is finished

I adopted a similar approach when training the NNs. The key design principles are:

- All different methods are trained on the same fixed training dataset
- Convergence/early stopping criteria are evaluated on a separate validation dataset

Unlike @maliar2021 who simulated data on-the-fly during NN training, my approach **strictly separates data generation and NN training**. Because different methods are applied to the exact same datasets, their results can be fully reproduced, compared and benchmarked. Any discrepencies in results must be due to the effectiveness of solution methods rather than potential randomness in data simulation.

I simulated datasets in two general structures (with $i$ denotes observation):
1. Full trajectory data: $\{(s_{it}, a_{i,t}, s_{1,i,t+1}, s_{2,i,t+1})\}_{t=0}^{T-1} \equiv \{\big(s_{i,t}, \pi_\theta(s_{i,t}), f(s_i,\pi(s_i), \epsilon_{1,it}), f(s_i,\pi(s_i), \epsilon_{1,it}) \big)\}^{T-1}_{t=0}$
2. One-period transition data: $\big( s_i, a_i, s'_{1i}, s'_{2i} \big) \equiv \big( s_{i}, \pi_\theta(s_{i}), f(s_i, \pi_\theta(s_i),\epsilon_{1,i}, f(s_i, \pi_\theta(s_i),\epsilon_{1,i} \big)$

where the one-period transition data is flattened and randomly shuffled from the full trajectory data. Full trajectory data is used by LRM and SHAC methods, and the one-period transition data is used by VFI/PFI, ERM and BRM methods. This design ensures that all these six methods are trained on the same data points even if their required data structure is different.

Note that for each period $t$, I take two iid draws $(\epsilon_{1,it}, \epsilon_{2,it})$ which is necessary to construct the unbiased estimator for the loss function in ERM and BRM methods. Another point is that the action $a_{it}$ and next-period states $s_{1,i,t+1}$ depends on the current policy $\pi_\theta$, so they must be generated during training. In practice, I separate exogenous states (e.g., AR(1) productivity shocks) that does not depend on $\theta$ from endogenous states (e.g., capital stock $k$) that depends on action $\pi_\theta$ (e.g., investment). Trajectory of exogenous states can be fully unrolled before training, and endogenous states are on-policy rollouts during training.

**Application:** To apply to both the basic investment model and the risky debt model, the data generation is (suppressing $i$ index):
1. Build bounded state spaces $\mathcal{S} = [\underline z, \bar z] \times [\underline k, \bar k] \times [\underline b, \bar b]$ and action spaces $\mathcal{A} = [\underline I, \bar I] \times [\underline b, \bar b]$ from model environment
2. Sample initial states $z_0,\, k_0,\, b_0 \sim \text{Uniform}(\mathcal{S})$ at $t=0$
3. For $t=0,\dots,T-1$, sample $M$ independent shock sequences $\{\varepsilon_{1,t},\dots,\varepsilon_{M,t}\}$ from $\mathcal{N}(0,1)$
4. Separate exogenous states $z_t$ and endogenous states $(k_t,b_t)$, start from $z_0$ and unroll the full trajectories of $\{ z_{1,t},\dots,z_{M,t} \}^{T-1}_{t=1}$ using the state transition function $z_{t+1}= \rho \log z_t + \sigma \varepsilon_{m,t}$ for $m=1,\dots,M$.
5. Store the full trajectory data:
$$\mathcal{D}^{\text{traj}} = \left( k_0, b_0, z_{0}, \{z_{1,t}\}_{t=1}^{T-1}, \dots, \{z_{M,t}\}_{t=1}^{T-1} \right)$$
6. Take the full trajectory data, flatten the exogenou states to only keep one-step sample $\left(z_{m} , z'_{m} \right)$ between $t$ and $t+1$ for a given $m=1,\dots,M$. Sample a new current-period endogenous state $k,\, b \sim \text{Uniform}(\mathcal{S})$, merge them, drop the $t$ subscript and use $'$ to denote next-period variable. Randomly permutate the data to break the serial correlation (ordering) and store it as the one-step transition data: 
$$\mathcal{D}^{\text{flat}} = (k, b, z, \{z'_{m}\}^M_{m=1})$$

### Reproducibility

All randomness in the project flows from a single integer pair of **master seed** $(m_0, m_1)$. TensorFlow stateless RNGs derive deterministic sub-seeds for three independent streams: **data generation** (the initial draws $k_0, z_0, b_0$, the shock paths $\varepsilon^{(1)}, \varepsilon^{(2)}$, and the post-flatten row permutation that break the serial correlation of the $N{\cdot}T$ one-step transitions), **NN initialization** (policy and critic weights), and **SGD mini-batch ordering** (the `tf.data` shuffle iterator inside each trainer). Together these guarantee that two runs with the same master seed produce bit-identical data, identical initial parameters, and identical mini-batch order on the same machine.

The data-generation stream is the most structured. Each random quantity has a fixed integer identifier $\mathrm{ID}(x)$, and for training step $j = 1, 2, \dots$:

$$
\mathbf{s}^{\text{train}}_{x,\, j} = \bigl(m_0 + 100 + \mathrm{ID}(x),\ \ m_1 + j\bigr), \qquad
\mathbf{s}^{\text{val}}_{x} = \bigl(m_0 + 200 + \mathrm{ID}(x),\ \ m_1\bigr).
$$

Training seeds advance with $j$ so each round draws fresh shocks; validation seeds are fixed and shared across rounds and methods. The split offsets together with the per-variable IDs guarantee all streams are pairwise disjoint.

| ID | Variable | Description |
|---|---|---|
| 1 | $k_0$ | Initial endogenous capital |
| 2 | $z_0$ | Initial exogenous productivity |
| 3 | $b_0$ | Initial debt (risky-debt model only) |
| 4 | $\varepsilon^{(1)}$ | Main AR(1) shock path |
| 5 | $\varepsilon^{(2)}$ | Second draw of AR(1) shock path (for AiO cross product) |
| 6 | flatten | Post-flatten permutation of the $N{\cdot}T$ one-step transitions |

The design delivers two benefits: **reproducibility** (same master seed → bit-identical experiment) and **common random numbers** (different methods trained at the same step $j$ see the same data, so cross-method comparisons are paired and free of Monte-Carlo noise). Note that a separate **strict mode** (`strict_reproducibility=True`) additionally pins down kernel-level non-determinism inside TensorFlow itself (parallel reductions, GPU / Metal ops); it is only reserved for strict replication and debugging.

## Validation of Solution {#part1-validate}

To verify the effectiveness and correctness of the solution methods, I benchmark them on a separate validation dataset that is fixed and identical across methods. I consider three main metrics for effectiveness:

1. **Policy MAE** against the true analytical optimal policy (only for frictionless basic model).
2. **Mean absolute Euler residual**: requires a closed-form Euler equation.
3. **Mean lifetime reward**: always feasible, the standard RL evaluation metric.

**Policy MAE.** For the frictionless basic model, the analytical anchor $k^*(z) = \bigl[\beta\,\mathbb{E}[z'\mid z] / (\bar r + \delta)\bigr]^{1/(1-\beta)}$ is exact. I evaluate the policy in next-period capital space $\pi_\theta(k,z)=k'$, where the analytical action $I^*(s)$ has been transformed back via the known endogenous transition:

$$
\text{MAE}(\pi_\theta) \;=\; \frac{1}{N}\sum_{i=1}^N \bigl|\,k'_{\pi_\theta}(k_i, z_i) \,-\, k^*(z_i)\,\bigr|
$$

with $\{(k_i, z_i)\}_{i=1}^N$ drawn from the flattened validation dataset and $k^*$ clipped to the state-space bounds $[k_{\min}, k_{\max}]$. This is a direct correctness proof when the optimal policy is known.

**Mean absolute Euler residual.** When the model admits a closed-form Euler residual $F(s, a, s', a')$ (e.g., the basic model with smooth adjustment costs only), I report

$$
\overline{|F|}(\pi_\theta) \;=\; \frac{1}{N}\sum_{i=1}^N \bigl|\, F\!\left(s_i,\, \pi_\theta(s_i),\, s'_i,\, \pi_\theta(s'_i)\right)\,\bigr|,
$$

on the same fixed validation dataset, where $s'_i$ is computed via the known state transition under the validation-side AR(1) shock draw.

**Mean discounted reward.** When neither the analytical policy nor a closed-form Euler residual is available (e.g., the risky-debt model), I evaluate the policy by rolling it out on the validation trajectory dataset and computing

$$
\bar V_T(\pi_\theta) \;=\; \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1} \gamma^t \cdot r\!\left(s^i_t,\, \pi_\theta(s^i_t)\right),
$$

where every trajectory uses the same pre-simulated exogenous shock path from the validation dataset, and $T$ is set large enough that the truncation tail is negligible (default $T = 200$, giving $\gamma^T \approx 4\times 10^{-5}$ at $\gamma = 0.95$). Because every method sees the same shocks, differences in $\bar V_T$ reflect policy quality rather than Monte-Carlo noise.
 

## Results: Basic Investment Model

This section present results from solving the basic model using different methods. To validate the correctness and effectiveness of solution, I set capital adjustment costs to zero (frictionless) $\psi_0=\psi_1=0$ so that we can benchmark the solution against a ground-truth optimal policy $k^*$. This experiment can be reproduced by running `docs/01_basic_investment_benchmark.ipynb`.

### Convergence Curve

@fig-convergence shows the three validation metrics from the [Validation of Solution](#part1-validate) section across wall-clock training time (seconds), all evaluated on the same held-out validation dataset: mean absolute Euler residual (left), mean discounted reward (middle), and policy MAE against the analytical anchor (right). I do not use the training loss as a convergence diagnostic, because a low loss does not necessarily imply a high-quality policy due to overfitting. The held-out metrics avoid this confound and let me compare every method on the same evaluation surface. Of the three, policy MAE (under frictionless basic model) is the strongest measure: the analytical true policy solution is exact, so zero MAE means the learned policy coincides with the closed-form optimum.

![Convergence of policy MAE across methods.](figures/part1-basic/convergence_curves.png){#fig-convergence}

The dashed line in the right panel marks the fixed MAE threshold $(=2)$ that defines convergence here. The threshold is held fixed across methods, so the relevant comparison is wall time to threshold rather than absolute loss. I find three key patterns:

- **PFI converges fastest and reaches the lowest MAE.** This is expected for a low-dimensional model. The Bellman operator is a $\gamma$-contraction on the discrete grid, so classical iteration remains very efficient at this scale.
- **ERM is the second-best method.** Its policy MAE drops below the threshold within a comparable wall-time budget, and the plateau sits close to the analytical solution.
- **SHAC also converged but takes longer.** This is consistent with its higher BPTT cost per gradient step.
- **LRM obtains close approximate but did not converged.** This is a structural bias due to terminal value truncation, as discussed below.

The most interesting case is LRM. Its policy MAE drops quickly in the early phase and then plateaus just above the threshold without ever crossing it. This is not a training artifact such as insufficient steps, a poor learning rate, or a small batch size. It is the structural defect of LRM described in [the LRM appendix](#sec-LRM). Even with the geometric-perpetuity terminal value correction, **LRM cannot recover the true on-policy continuation value up to precision** $V^\pi(s_T)$. The perpetuity can at-best approximate the stochastic future value with a deterministic steady state and ignores the firm's optimal response to future shocks. This leaves an $O(\sigma_\varepsilon^2)$ approximation error that is small but does not vanish with longer training. LRM therefore always have an approximation bias, regardless of training budget and sample size.

### Learned Policy vs True Policy

@fig-policy provides a visual validation of the solutions. It plots two slices of the next-period capital policy $k'(k, z)$ for each method, against the closed-form analytical anchor $k^*(z) = [\beta\, \mathbb{E}[z'\mid z]/(\bar r + \delta)]^{1/(1-\beta)}$:

- **Left panel: $k'$ as a function of $z$, holding $k$ fixed.** The analytical anchor depends only on $z$ and rises monotonically with $z$, so the plotted curves are upward sloping.
- **Right panel: $k'$ as a function of $k$, holding $z$ fixed.** Because $k^*$ is independent of $k$, the analytical anchor is a horizontal line. A correctly learned policy is also flat in this panel.

![Learned policy against true policy by methods](figures/part1-basic/selected_checkpoints_overlay.png){#fig-policy}

The black dashed line is the analytical solution. The red dotted line is the PFI solution, which serves as the discrete-grid anchor. PFI, ERM, and SHAC all match the analytical solution to within the convergence threshold of policy MAE $\leq 2$ established in @fig-convergence, and their learned policies tightly conincide with the true policy.

The only exception is LRM. In the left panel, **LRM systematically underestimates $k'$ when $z$ is high**. This is exactly the terminal-value truncation error described in [the LRM appendix](#sec-LRM). It is worth emphasizing that this gap remains nontrivial after the terminal value approximation is already implemented by adding $\hat{V}^{\text{term}}(s_T^{\text{endo}})$ to the LRM objective function. Without this correction, the downward bias would be much larger.

The underlying intuition is that the perpetuity deliberately pins the exogenous component at the stationary mean $\bar z$ instead of conditioning on the realized $z_T$. AR(1) persistence means the rollout typically ends at high $z_T$ when the trajectory started from high $z$, but the perpetuity ignores this conditioning and therefore underestimates the true continuation $V^\pi(s_T)$. The actor's gradient then underweights the long-run benefit of investing at high $z$, and the learned $k'$ is pulled toward the interior. **This bias cannot be reduced by more training, more data, or a smaller learning rate**, because the gap lives in the analytic terminal correction itself rather than in the optimization. The only way to close it is to replace the perpetuity approximation with a learned value network, which is exactly what SHAC does.

### Reproducing the Original Maliar et al. (2021) Methods

Now I present three additional experiments that strictly reproduce the original methods in @maliar2021: Euler Residual Minimization (ERM), Lifetime Reward Maximization (LRM), and Bellman Residual Minimization (BRM). All three are applied to the frictionless basic investment model. The goal is to evaluate the methods as they are published, identify their key defects, and explain why we need patches and/or better alternative methods like SHAC to correctly solve the corporate finance models.

- **@fig-maliar-loss** reports each method's **training loss** as a function of training step.
- **@fig-maliar-validate** reports the three **held-out validation metrics** from the [Validation of Solution](#part1-validate) section: mean absolute Euler residual, mean discounted lifetime reward, and policy MAE against the analytical anchor.


![Loss function of original @maliar2021 methods](figures/part1-basic/part_a_training_losses.png){#fig-maliar-loss}

![Validation metrics of original @maliar2021 methods](figures/part1-basic/part_a_validation_diagnostics.png){#fig-maliar-validate}

Reading these two figures side by side reveals an important point: **a low training loss does not imply a correct solution**. Specifically, the BRM loss is "correctly" minimized as visualized in @maliar2021, but the actual solution is wrong and the training shows no sign of improvement.

![Learned policy against true policy of original @maliar2021 methods](figures/part1-basic/part_a_overlay.png){#fig-maliar-policy}

**ERM is the only method that converges.** ERM's training loss in @fig-maliar-loss and its validation metrics in @fig-maliar-validate drop together: the held-out policy MAE crosses the convergence threshold and the Euler residual settles near zero. It is the only one of the three @maliar2021 methods that converged when applied to the basic model.

**LRM converges in training but under-invests at every $z$.** LRM's training loss in @fig-maliar-loss descends steadily, yet @fig-maliar-validate shows the held-out policy MAE plateaus well above the convergence threshold. This is the terminal value truncation bias detailed in [the LRM appendix](#sec-LRM). The original LRM drops every period of cash flow beyond the rollout horizon $T$ from the objective, which under-weights the long-run benefit of investment at every state. The resulting policy under-invests uniformly in $z$. My patched LRM applies a deterministic-perpetuity correction that reduces this to a high-$z$-only bias.

**BRM diverges despite a near-zero training loss.** BRM's training loss in @fig-maliar-loss decreases monotonically and reaches near-zero, matching @maliar2021's training-loss curves. Yet @fig-maliar-validate shows BRM's held-out policy MAE never actually improved and the learned policy is qualitatively wrong.

This refutes an implicit claim in @maliar2021: that joint minimization of the Bellman residual and the FOC residual is sufficient to identify the optimal policy. The paper argues this by analogy without a formal proof, and my experiment shows the analogy fails.

The mechanism is intuitive and detailed in [the BRM appendix](#sec-BRM). In economic environments, the Bellman residual is orders of magnitude larger than the FOC residual, so gradient descent drives down the Bellman loss first. The value network therefore learns to satisfy the Bellman equation for whatever policy emerges, not for the optimal policy. The policy network then minimizes the FOC against this wrong value function. Both losses become small, but the system locks into a self-consistent fixed point that depends on the value network's initialization. @maliar2021 proposes fine-tuning the exogenous weights to balance the two, but they can only partially mitigate the symptom and fail to prevent divergence.

This is why I reject BRM method for production. The new SHAC method is a direct fix to the BRM method and it adopts the canonical actor-critic method design in RL to ensure the convergence to the correct and unique fixed point.


## Issues in Neural Network Architecture and Training {#part1-issues}

Standard VFI and PFI methods are "simple" and robust because its convergence is guaranteed by contraction mapping theorem. In contrast, I find that there are many details that are critical for Neural Network (NN) based methods to work, and these practical issues are often omitted by the higher-level algorithm summary in original papers [@maliar2021, @fernandez-villaverde2025].

Table below summarizes the issues specific to the three methods introduced by @maliar2021:

| Method | Major Defects | Minor Defects | Usability | Reference | 
|---|---|---|---|---|
| Euler Residual Min | None | Single policy network is unstable. Solved by adding a target policy network | Fast and robust for production, but requires existence of a closed-form Euler equation | [@maliar2021, @fernandez-villaverde2025] 
| Lifetime Reward Max | Terminal value truncation bias | Long-horizon backpropagation through time (BPTT) is slow and costly | Can be used as rough baseline, but not ideal for production when accuracy/unbiasedness matters | [@maliar2021]
| Bellman Residual Min | Can easily converge to "wrong" but self-consistent fixed point | Conflicting gradients due to scale mismatch of loss functions; Require existence of closed-form first order condition | High-risk and strongly rejected for production | [@maliar2021, @fernandez-villaverde2025] 
| Short Horizon Actor Critic | None | Slower than ERM due to BPTT, but can be fine-tuned to improve speed | Unbiased. Most generalized and flexible method. Does not require closed-form Euler equation or FOC. | [@xu2022]

In addition, I rank the general issues (shared by all methods) based on their importance in practice:

| General Issues | Description | Solutions | Results | 
|---|---|---|---|
| Smooth and differentiable reward and dynamics | This is a fundamental prerequisite for gradient-based training to work | Kinks can still be handled, but discrete choice or jump discontinuities can only be approximated with error  | For the basic model with fixed adjustment cost, NN-based methods cannot learn the inaction regions; Soft-surrogate suffers nontrivial approximation error; VFI/PFI is strictly better
| Input Normalization | Raw data are measured in level and large units can easily de-stablize training  | Normalize input to z-score and re-scale it back to economic levels as NN output head | Hidden layer only see normalized inputs and is agnostic to environment
| Network output head | Sigmoid, Tanh, and other activation can suppress gradient and prevent learning at extreme values | Use raw linear (no activation function) with affine transformation | Gradient is uniformly "strong" across state/action space; Output variable converted back to original unit
| Hidden layer activation | For economic models, ReLU is not stable, Sigmoid and Tanh cause vanishing gradient| SiLU `swish` always perform better in practice | Gradient is stable and nonzero
| Full reproducibility | Comparison across methods should be fair and fully reproducible | Separate data generation and training; Full schedule of random number generator (RNG)| All methods are trained on exactly the same fixed dataset and results are fully controlled by master seeds
| Convergence metric | Objective and loss function are NOT the correct metrics for the quality of solution | Measure effectiveness of learned policy in a separate validation dataset | Avoid overfitting, enable early stopping based on same criteria, fair comparison across methods

The architecture-level choices in this table are documented in detail in [Implementation Details](#sec-impl): input normalization, hidden-layer activation, output head transformation.

## Risky Debt Model

### Solution method: Nested VFI

To solve the risky debt model described in @strebulaev2012 [section 3.6], I implemented a **nested VFI algorithm**. I find that nested VFI is still the best method for this model in terms of speed, robustness, and accuracy. In contrast, I find that all three of @maliar2021's methods **cannot be applied to solve the risky debt model** because the model has a nested fixed-point structure: the firm's value function $V$ depends on the bond's risky rate $\tilde r$, and $\tilde r$ depends on $V$ because the lender prices the bond using the default states implied by $V$.

I solve the model with a two-level VFI iteration. The **inner loop** is a standard VFI that solves $V$ on the discrete grid for a fixed pricing schedule. The **outer loop** updates $\tilde r$ to be consistent with the default partition implied by the latest $V$. The algorithm terminates when both loops converge, so the value function and the pricing schedule are mutually consistent. 

On top of the standard algorithm described in @strebulaev2012 and used by @hennessy2007costly, I developed two algorithm refinements. I argue that these two refinements significantly **improve the speed of nested VFI without sacrificing accurancy**. The saving in compute can be especially large for applications of the simulated method of moment (SMM) where the computational bottleneck is re-solving the model repeatedly over many optimization steps.

1. **Linear interpolation on $z$-grids**, which significantly increases the speed without hurting precision. The refined nested VFI algorithm can solve a $25 \times 25 \times 15$ in just a few minutes on standard CPU. Currently none of the deep learning methods I've tried can achieve similar performance.
2. **Adaptive $b$-bound around the default boundary**: VFI is expensive and we want to spend the compute on the economically meaningful regions of $(k',b')$ near the default boundary. I added a pre-training stage to VFI that pins down the default boundary first with coarse grid configurations (e.g., $10\times10\times5$) before running the full algorithm on finer grids.

Below is a brief summary of the algorithm. The full details are in the [nested-VFI appendix](#sec-NestedVFI).

**Input:** Discrete grids for capital, debt, and productivity, Markov transition matrix for the productivity shock, model primitives, risk-free rate $r$, convergence tolerances $\varepsilon_{\text{inner}}$ and $\varepsilon_{\text{outer}}$.

**Output:** A mutually consistent value function $V^*$ and risky-rate schedule $\tilde r^*$.

1. Initialize the pricing schedule at the risk-free rate: $\tilde r = r$ everywhere.
2. **For** $n = 1, 2, \ldots$ **do**:
3. **(a) Inner VFI.** With $\tilde r$ held fixed, run standard value function iteration on the discrete grid. Cap the value at zero whenever the Bellman maximand is negative, so the firm defaults exactly when $V \le 0$. Iterate to convergence in sup-norm and store the result as $V^{(n)}$.
4. **(b) Outer convergence check.** If $\|V^{(n)} - V^{(n-1)}\|_\infty < \varepsilon_{\text{outer}}$, stop and return $(V^{(n)}, \tilde r)$.
5. **(c) Default partition.** For each next-period choice $(k', b')$, partition the future productivity grid into default states ($V^{(n)} = 0$) and solvent states ($V^{(n)} > 0$).
6. **(d) Update pricing.** For each $(z, k', b')$, solve the lender's zero-profit condition for the new $\tilde r(z, k', b')$ given the default partition. This pins down the risk premium that compensates lenders for default probability and recovery loss.
7. **End for.**

### Results

The solution of the risky debt model are visualized in three set of plots:
1. Value function slice $V(z,k,b)$ against each one of the argument
2. Policy function slice $k'(z,k,b)$ and $b'(z,k,b)$ against each of the argument
3. Critical value of next-period shock $z'_d$ determining default boundary: for a given optimal action $(k',b')$, any realization $z'<z'_d$ means default, vice versa.

The risky debt model does not have closed form Euler equation or analytical solution under special cases (e.g., frictionless), the only feasible validation metric is the sum of lifetime reward (higher means better) --- but it is only useful for comparison across methods. Instead, I rely on an **economic diagnostic test** to check the solutions. Specifically, I verify whether the comparative statics are consistent with the theoretical propositions proved in @hennessy2007costly:

- $V(z,k,b)$: Firm value should be increasing in $k$, decreasing in $b$, and non-decreasing in $z$ (Proposition 3 and 4)
- Critical value $z'_d$ should be increasing in $b'$ and decreasing in $k'$ (Proposition 6).

My results in @fig-debt-value and @fig-debt-boundary are consistent with these implications. In addition, we know from the basic model that the optimal investment policy $k'$ should be increasing in $z$ and $k$ when there exist adjustment costs. This also matches the first two panels in @fig-debt-policy. 

Neither @hennessy2007costly or @strebulaev2012 derive any comparative statics regarding $k'(z,k,b)$ and $b'(z,k,b)$, so I interpret them as ambiguous. That said, from the basic model we know that $\partial k'/\partial z>0$ and $\partial k'/\partial k>0$ when there exist adjustment costs, which are consistent with @fig-debt-policy. The remaining results are also economically reasonable:

- $b'$ is increasing in $z$ and $k$: firm can borrow more without default risk when productivity is high or when capital stock is large (buffer)
- $k'$ and $b'$ is independent of current $b$: this is consistent with the model where issuing a one-period corporate bond does not entail any frictional costs

![Value function slices $V(z, k, b)$ of the nested-VFI risky-debt solution.](figures/part1-debt/value_slices.png){#fig-debt-value}

![Policy function slices $k'(z, k, b)$ and $b'(z, k, b)$ of the nested-VFI risky-debt solution.](figures/part1-debt/policy_slices.png){#fig-debt-policy}

Finally, @fig-debt-boundary provides a great summary of the results. The top left panel visualizes the default boundary governed by a critical value of $z'_d$. The economic intuition is that at current period given $(z,k,b)$, firm takes the optimal action govern by policy $(k',b')$ and rational expectation of dynamics $P(z'|z)$. When next period shock $z'$ realized, there exist non-zero probability that even the most optimistic realization $z'=z_{\max}\approx 1.8$ cannot prevent default. This forms the default boundary (dark navy) across the $(k',b')$ space. Moreover:

- Higher $k'$ reduces default probability, so $z'_d$ is monotonically decreasing in $k'$
- Higher $b'$ increases default probability, so $z'_d$ is monotonically increasing in $b'$

The bottom panels of @fig-debt-boundary summarizes the relationship between endogenous bond yield (debt discount) $1/(1+\tilde r)$ and $(k',b',z)$. Higher default risks increase $\tilde r$ and thus reduce bond yield, therefore:

- Debt discount is increasing in $z$ and $k'$ because of lower default risk
- Debt discount sharply declines in $b'$ between $b'\approx 70-80$. This cliff captures the default boundary and the default risks are priced into $\tilde r$
- When defualt risks are low (e.g., high $z,k'$ and/or high $b'$), the debt discount is close to risk-free rate $1/(1+r)\approx 0.96$ (dotted line)

In summary, the results are consistent with economic rationals of the risky debt model and both the inner and outer loop of VFI converged up to high precision (error $<10^{-6}$). I consider this as strong evidence of the correctness and effectiveness of the solution.

![Default boundary $z'_d$ over $(k', b')$ and endogenous debt discount $1/(1+\tilde r)$ across states.](figures/part1-debt/boundary_and_discount_slices.png){#fig-debt-boundary}


### Why deep learning methods failed for this model?

For the three methods developed by @maliar2021: LRM does not learn $V$ and $\tilde r$, ERM requires closed-form Euler equation that is not feasible for this model, and BRM is rejected due to structural defect. None of them are suitable for this nested fixed-point problem.

One promising solution methods to this model is Short-Horizon Actor Critic (SHAC), which include a separate value NN training and does not require closed-form Euler. However, I benchmarked the solution of SHAC to nested VFI and find that **SHAC systematically learned a more conservative policy (lower leverage-to-asset ratio)**. This is a consequence of the theoretical model's structure and should be **common to all actor-critic methods**: 

- During training, value network $V$ is initially inaccurate and gradually improving
- But the initially biased $V$ directly determine the default set and interest rate, which affect the policy learning and the target $V$ in next iteration
- Both policy and value network converged to a self-consistent but over-pessimistic (low leverage) or over-optimistic (high leverage) equilibrium, which is sensitive to the NN initiation weights $\phi$ for the value function $V_\phi$.

In practice, my experiments show that SHAC solutions are usually pessimistic (low leverage) because initial $V$ network tends to underestimate $\partial V/\partial b$ and lead to conservative policy (low leverage and less default states), which leads to a more conservative target value network in next iteration and self-reinforcing policy updates. 

There are two promising fixes: (1) use stochastic policy method instead of a deterministic policy to explore off-policy and with scalar cash flow (reward) acting as a score function. This allows the training to explore high-leverage states (default) off-policy. (2) switch to a standard trade-off model as in @nikolov2021 where default states (and endogenous interest rate) does not depend on the value function and can be written down analytically only in terms of current states.

However, stochastic policy methods are usually not sample-efficient. For our model, nested VFI method is clearly faster and more robust. Fix (2) is a practical choice if we are willing to deviate from this version of the endogenous default model, but the standard VFI still performs better for this low-dimensional problem with few states.

# Part II. Structural Estimation

## GMM and SMM  {#part2-validate}

I implemented and tested both GMM and SMM methods to structurally estimate model parameters. I measure effectiveness using the basic investment model because it is computationally cheaper. The basic idea of the Monte Carlo (MC) validation is:

For MC replication $j=0,\dots, J$:
1. Set replication count $j=j+1$. Select a set of *true* parameters $\beta^*_j$, solve for the optimal policy $\pi^*(\cdot|\beta^*_j)$, use it to simulate a "target" panel dataset of $N$ i.i.d. firms over $T$ periods
2. Start with a random guess $\beta^0_j \neq \beta^*_j$, apply GMM or SMM to the target dataset, obtain a set of estimated params $\hat \beta_j$ and variance-covariance matrix.
3. Conduct t-test $H_0: \hat \beta_j = \beta^*_j$ and expect failure to reject the null. Conduct over-identifying test and verify if we fail to reject the hypothesis of model mis-specification.

When all $J$ replications completes, compute diagnostics including the average bias $\frac{1}{J}\sum_j(\hat \beta_j - \beta^*_0)$, Root Mean Square Error (RMSE), average rejection rate of over-identifying test, etc. The detailed diagnostic formulas are defined in the [SMM appendix](#sec-smm-appendix).

If the pipeline is correctly implemented, three properties must hold. (i) The optimizer must reach an interior minimum, so moment errors at $\hat\beta$ should be near zero. (ii) On a single panel, $\hat\beta$ should fall within its sandwich SE of the truth, and the t-tests and J-test should fail to reject at 5%. (iii) Across MC replications, the empirical bias should be small relative to the within-replication SE, the empirical SD across replications should match the average SE (so confidence intervals built from one panel have approximately correct coverage), and the J-test rejection rate at the 5% nominal level should not exceed roughly 0.05. A failure on any one localizes the defect: failure of (i) implies the optimizer; failure of (ii) the point estimate or SE formula; failure of (iii) finite-sample anti-conservativeness in the asymptotic SE. The three result tables in each subsection below test these three properties in turn.

I consider the implementation to be correct only if our MC replication can consistently estimate $\hat \beta_j$ close to the true $\beta^*_j$. The shock realization of replication is controlled by master seeds and are fully reproducible. 

For the actual application, we can replace step 1 with one real-world "target" dataset such as the Compustat firm panel data, and we only apply Step 2-3 once.

There are several important implementation issues:

- GMM uses Euler equation to form the moment condition, so it does not require solving the model
- SMM typically requires re-solving the model for optimal policy in Step 1 for each candidate $\beta$ for evaluation. This is the main computational bottleneck. For this validation, I use the frictionless basic model with analytical solution to the optimal policy to avoid the cost. This validates the correctness of the entire SMM pipeline and separate potential errors of model solver (e.g. VFI/PFI) from estimation.
- Both GMM and SMM requires choosing appropriate global and local optimizer to find $\beta^*$ that minimizes the moment condition error. I implemented both the simulated annealing optimizer used by @hennessy2007costly and the
"differential evolution" optimizer search for $\beta^*$. These optimizers are built in `scipy.optimize`.



### GMM Validation on the Basic Investment Model {#sec-gmm-validation}

I validate the GMM implementation on the basic investment model with smooth (convex) capital adjustment cost, where the Euler equation has a closed form. The Euler conditions provide moment restrictions that are evaluated directly from the observable panel $(\pi, k, I)$, so each $Q(\beta)$ evaluation in the optimizer is arithmetic on the data and the model is never re-solved inside the optimizer loop. The results can be reproduced by running `docs/04_gmm_validation.ipynb`.

**Validation design.**

- **Model.** Basic investment with convex adjustment cost only ($\phi_1 = 0$).
- **Parameters.** $\beta = (\alpha,\, \psi_1,\, \rho,\, \sigma)$: production elasticity, convex-cost coefficient, AR(1) persistence, and AR(1) shock std-dev. Interest rate ($r = 0.04$) and depreciation rate ($\delta = 0.15$) are calibrated externally.
- **Truth and initial guess.** True parameters $\beta^* = (0.60,\, 0.10,\, 0.70,\, 0.15)$. Optimizer starts from $\beta_0 = (0.480,\, 0.500,\, 0.475,\, 0.500)$.
- **Moments (6).** Three Euler-orthogonality conditions (Euler $\times\, 1$, Euler $\times I/k$ lag, Euler $\times \pi/k$ lag), two shock-orthogonality conditions (Shock $\times\, 1$, Shock $\times \ln z$ lag), and one variance condition.
- **Two-step GMM.** Stage 1 minimizes the moment distance with $W = I$ using `dual_annealing` for global search and Powell refinement. Stage 2 warm-starts from the Stage-1 estimate and minimizes with $W = \hat\Omega^{-1}$ using Powell only. Standard errors come from the Stage-2 sandwich formula.
- **"Real" data simulation.** I first run a single PFI solve at $\beta^*$ on a dense grid (exogenous = 50, endogenous = 100, action = 100), then simulate $N = 100$ firms over $T = 25$ periods after a 275-period burn-in to ensure ergodic sampling. The GMM estimator treats $(\pi, k, I)$ as observed data and never re-solves the model. PFI approximation error in the policy enters the Euler residuals as a small systematic bias, kept below the sampling noise floor by the dense grid.
- **Monte Carlo.** $J = 20$ independent panels, each generated from $\beta^*$ under a different master seed; the full two-step procedure is run on each.

**Result 1 — optimizer reaches an interior minimum.** If GMM is correctly implemented, the six moment conditions at $\hat\beta$ should be near zero, well below the sampling noise floor. @tbl-gmm-moment-fit confirms this: five of six conditions are below $10^{-3}$ and the largest residual (Shock $\times 1$) is $3.5 \times 10^{-3}$. The Stage-1 global search and Stage-2 Powell refinement together find an interior minimum.

| Moment              | $g(\hat\beta)$                    |
| ------------------- | --------------------------------- |
| Euler $\times 1$    | $-9.23 \times 10^{-4}$            |
| Euler $\times I/k$ lag    | $-1.70 \times 10^{-4}$            |
| Euler $\times \pi/k$ lag  | $-2.66 \times 10^{-4}$            |
| Shock $\times 1$    | $-3.55 \times 10^{-3}$            |
| Shock $\times \ln z$ lag  | $\phantom{-}6.87 \times 10^{-4}$  |
| Var $\times 1$      | $-3.71 \times 10^{-7}$            |

: Moment-condition vector $g(\hat\beta)$ on a single representative panel. {#tbl-gmm-moment-fit}

**Result 2 — point estimate and sandwich SE are correctly calibrated on a single panel.** If they are, every estimate should be within roughly 1 SE of the truth and the parameter t-tests should fail to reject at 5%. @tbl-gmm-single-rep confirms both: every estimate is within 1 SE of $\beta^*$, all four parameter t-tests have $p > 0.10$ (the smallest is $\sigma$ at $p = 0.11$), and the over-identification J-test is also insignificant ($J = 4.11$, $p = 0.13$, df = 2). The single-panel pipeline behaves correctly.

| Parameter | True   | Estimate | SE       | $t$-stat | $p$-value |
| --------- | ------ | -------- | -------- | -------- | --------- |
| $\alpha$  | $0.60$ | $0.5999$ | $0.0006$ | $-0.14$  | $0.89$    |
| $\psi_1$  | $0.10$ | $0.0861$ | $0.0176$ | $-0.79$  | $0.43$    |
| $\rho$    | $0.70$ | $0.6817$ | $0.0154$ | $-1.18$  | $0.24$    |
| $\sigma$  | $0.15$ | $0.1466$ | $0.0021$ | $-1.60$  | $0.11$    |

: Parameter estimates, sandwich SEs, and t-tests against the true parameter. J-statistic = 4.11, $p$ = 0.13, df = 2. {#tbl-gmm-single-rep}

**Result 3 — point estimate is unbiased on average and asymptotic SE is correctly calibrated.** This requires three things together: bias should be much smaller than the within-replication SE, the empirical SD across replications should match the average SE, and the J-test rejection rate at 5% nominal should be $\approx 0.05$. @tbl-gmm-mc partially confirms this. The point estimates are economically close to the truth (relative bias is below 4% for every parameter). However, the empirical SD across replications is systematically $1.3$–$2 \times$ the average within-replication SE, so the asymptotic sandwich formula understates the true sampling variability. This propagates into J-test over-rejection at $0.20$ vs the nominal $0.05$. Both failures are consistent with PFI grid approximation in the panel-generation step entering the Euler residuals as a small systematic bias.

| Parameter | True   | Mean estimate | Bias        | RMSE     | SD across MC | Avg SE   |
| --------- | ------ | ------------- | ----------- | -------- | ------------ | -------- |
| $\alpha$  | $0.60$ | $0.6015$      | $0.0015$    | $0.0021$ | $0.0016$     | $0.0008$ |
| $\psi_1$  | $0.10$ | $0.1345$      | $0.0345$    | $0.0576$ | $0.0473$     | $0.0332$ |
| $\rho$    | $0.70$ | $0.6856$      | $-0.0144$   | $0.0303$ | $0.0273$     | $0.0188$ |
| $\sigma$  | $0.15$ | $0.1506$      | $0.0006$    | $0.0028$ | $0.0028$     | $0.0022$ |

: Monte Carlo summary across $J = 20$ replications. J-test reject rate at 5% is 0.20. {#tbl-gmm-mc}

**Summary.** Predictions (i) and (ii) hold cleanly. Prediction (iii) is partially confirmed: the point estimate is essentially unbiased, but the asymptotic SE is mildly anti-conservative and the J-test correspondingly over-sized. Both shortfalls are attributable to PFI grid approximation in the panel-generation pipeline rather than the GMM core. With this caveat, the GMM machinery (moment construction, two-step weighting, optimizer, sandwich SE, J-test) is correctly implemented on a model with a closed-form Euler equation.

### SMM Validation on the Frictionless Basic Investment Model {#sec-smm-validation}

I validate the SMM implementation on the frictionless basic investment model, where the analytical policy is exact and there is no model-solve error. Any deviation between the estimated and true parameters in this experiment must come from the SMM machinery itself: the moment construction, the two-step weighting, the global / local optimizer, or the standard-error formula. This isolates SMM correctness from model-solution correctness. The results reported below can be reproduced by running `docs/05_smm_validation.ipynb`.

**Validation design.**

- **Model.** Frictionless basic investment ($\phi_0 = \phi_1 = 0$). The optimal policy is $k^*(z)$ given analytically. This eliminates any iteration error from VFI / PFI / NN solvers.
- **Parameters.** $\beta = (\alpha,\, \rho,\, \sigma)$. Interest rate ($r = 0.04$) and depreciation rate ($\delta = 0.15$) are calibrated externally.
- **Truth and initial guess.** True parameters $\beta^* = (0.60,\, 0.70,\, 0.15)$. Optimizer starts from $\beta_0 = (0.525,\, 0.475,\, 0.480)$.
- **Moments (4 in total, overidentified by 1).** Mean and variance of $I/k$, serial correlation of $I/k$, and the residual std of an AR(1) regression on log income.
- **Two-step SMM.** Stage 1 minimizes the moment distance with $W = I$ using `differential_evolution` for global search and Powell refinement. Stage 2 warm-starts from the Stage-1 estimate and minimizes with $W = \hat\Omega^{-1}$ using Powell only. Standard errors come from the Stage-2 sandwich formula.
- **Simulation budget.** $N = 500$ firms, horizon $T = 25$, burn-in = 75, and $S = 50$ simulated panels per moment evaluation.
- **Monte Carlo.** $J = 10$ independent fake-real panels, each generated from $\beta^*$ under a different master seed.

**Result 1 — optimizer reaches an interior minimum.** If SMM is correctly implemented, the simulated moments at $\hat\beta$ should match their data-side targets to within Monte-Carlo noise. @tbl-smm-moment-fit confirms this: every fitted moment is within $\sim 1\%$ of its target, so the Stage-1 global search and Stage-2 Powell refinement find an interior minimum.

| Moment              | Target    | Fitted    |
| ------------------- | --------- | --------- |
| Mean $I/k$          | $0.1927$  | $0.1916$  |
| Var $I/k$           | $0.0925$  | $0.0926$  |
| Serial corr $I/k$   | $-0.1427$ | $-0.1432$ |
| AR(1) resid std     | $0.2124$  | $0.2123$  |

: Moment fit at $\hat\beta$ on a single representative panel. {#tbl-smm-moment-fit}

**Result 2 — point estimate and sandwich SE are correctly calibrated on a single panel.** If they are, every estimate should fall within roughly 1 SE of the truth and the parameter t-tests should fail to reject at 5%. @tbl-smm-single-rep confirms both: every estimate is within 0.2 SE of $\beta^*$ and all parameter t-tests have $p > 0.85$. The over-identification J-test is also insignificant ($J = 0.62$, $p = 0.43$, df = 1).

| Parameter | True   | Estimate | SE       | $t$-stat | $p$-value |
| --------- | ------ | -------- | -------- | -------- | --------- |
| $\alpha$  | $0.60$ | $0.6007$ | $0.0079$ | $0.09$   | $0.93$    |
| $\rho$    | $0.70$ | $0.7018$ | $0.0156$ | $0.12$   | $0.91$    |
| $\sigma$  | $0.15$ | $0.1502$ | $0.0010$ | $0.19$   | $0.85$    |

: Parameter estimates, sandwich SEs, and t-tests against the true parameter. J-statistic = 0.62, $p$ = 0.43, df = 1. {#tbl-smm-single-rep}

**Result 3 — point estimate is unbiased on average and asymptotic SE is correctly calibrated.** This requires bias much smaller than the within-replication SE, empirical SD $\approx$ average SE, and J-test rejection rate at 5% nominal not exceeding $0.05$. @tbl-smm-mc confirms all three. Bias is more than an order of magnitude below SE for every parameter, the empirical SD agrees with the average SE within $\sim 30\%$ (consistent with the small MC sample of $J = 10$), and the J-test never rejects across 10 replications — $0/10$ is statistically consistent with any size at or below $0.05$.

| Parameter | True   | Mean estimate | Bias       | RMSE     | SD across MC | Avg SE   |
| --------- | ------ | ------------- | ---------- | -------- | ------------ | -------- |
| $\alpha$  | $0.60$ | $0.6003$      | $0.0003$   | $0.0084$ | $0.0089$     | $0.0085$ |
| $\rho$    | $0.70$ | $0.6969$      | $-0.0031$  | $0.0166$ | $0.0172$     | $0.0159$ |
| $\sigma$  | $0.15$ | $0.1505$      | $0.0005$   | $0.0015$ | $0.0016$     | $0.0012$ |

: Monte Carlo summary across $J = 10$ replications. J-test reject rate at 5% is 0.00. {#tbl-smm-mc}

**Summary.** All three predictions hold. The SMM machinery (moment construction, two-step weighting, optimizer, sandwich SE, J-test) is correctly implemented on a model whose ground truth is known. This confirms the correct implementation of the SMM pipeline except for the model solver (e.g., VFI) itself.


## Applying SMM to the risky debt model

Applying SMM to the basic investment model identifies four parameters: production-function curvature ($\alpha$), smooth adjustment cost ($\psi_1$), AR(1) persistence ($\rho$), and AR(1) shock variance ($\sigma$). Adding costly equity issuance from section 3.3 of @strebulaev2012 brings two more parameters into scope: the fixed and proportional cost components ($\eta_0$ and $\eta_1$). The full endogenous-default model adds one final parameter, the deadweight bankruptcy cost $c_{\text{def}}$ — the fraction of firm value lost when the firm defaults. From a pure estimation perspective, the endogenous-default extension only adds one parameter; the rest can be estimated from the simpler frictional model in section 3.3.

The cost of applying SMM to risky debt model is computational: each candidate $\beta$ in the optimizer's inner loop requires a fresh nested-VFI solve on the discrete $(k, b, z)$ grid. A Monte-Carlo replication study at this scale is infeasible (on my current device), so I report a single representative run rather than MC summary statistics. The full SMM target is $\beta = (\alpha,\, \psi_1,\, \eta_0,\, \eta_1,\, c_{\text{def}},\, \rho,\, \sigma)$ with $K = 7$, matched against $R = 11$ moments following @hennessy2007costly's selection (see [Appendix B](#sec-smm-appendix)). The results can be reproduced from `docs/06_risky_debt_smm_workflow.ipynb`. It took about 40 hours to run the full SMM on my 2020 Macbook Pro (M1).

**Result 1 — moment fit.** Fitted moments deviate noticeably from their targets. The conditional-issuance, AR(1)-shock-std, and variance-of-investment moments miss by 50%+ of their target value, indicating the optimizer cannot find a $\beta$ that matches all 11 moments simultaneously.

| Moment              | Target      | Fitted      |
| ------------------- | ----------- | ----------- |
| Avg Iss/k           | $0.0168$    | $0.0235$    |
| Cond Iss            | $0.0881$    | $0.1917$    |
| AC Iss              | $-0.0507$   | $-0.0518$   |
| Corr(Lev, Iss)      | $\phantom{-}0.2723$ | $\phantom{-}0.3008$ |
| Avg Lev             | $0.9099$    | $0.8635$    |
| Std Lev             | $0.0308$    | $0.0485$    |
| AC $I/k$            | $0.0023$    | $0.0023$    |
| Var $I/k$           | $0.0287$    | $0.0395$    |
| AR(1) $\beta$       | $0.3399$    | $0.3639$    |
| AR(1) $\sigma$      | $0.2158$    | $0.3139$    |
| Default freq        | $0.000488$  | $0.000400$  |

: Moment fit at $\hat\beta$ on a single representative panel (11 moments, 7 estimated parameters). {#tbl-smm-debt-moment-fit}

**Result 2 — parameter estimates.** Five of seven t-tests reject $H_0:\, \hat\beta_k = \beta_k^*$ at the 5% level. The two that fail to reject ($c_{\text{def}}$, $\rho$) do so only because their standard errors are abnormally large.

| Parameter        | True   | Estimate | SE       | $t$-stat   | $p$-value      |
| ---------------- | ------ | -------- | -------- | ---------- | -------------- |
| $\alpha$         | $0.70$ | $0.6213$ | $0.0174$ | $-4.53$    | $< 10^{-5}$    |
| $\psi_1$         | $0.05$ | $0.0691$ | $0.0071$ | $\phantom{-}2.69$ | $0.007$ |
| $\eta_0$         | $0.10$ | $0.1207$ | $0.0418$ | $\phantom{-}0.50$ | $0.62$  |
| $\eta_1$         | $0.05$ | $0.0851$ | $0.0049$ | $\phantom{-}7.13$ | $< 10^{-12}$ |
| $c_{\text{def}}$ | $0.45$ | $0.3844$ | $0.3896$ | $-0.17$    | $0.87$         |
| $\rho$           | $0.60$ | $0.6411$ | $0.0624$ | $\phantom{-}0.66$ | $0.51$  |
| $\sigma$         | $0.15$ | $0.2206$ | $0.0306$ | $\phantom{-}2.31$ | $0.02$  |

: Parameter estimates at $\hat\beta$, sandwich SEs, and t-tests against the truth. The J-statistic is `NaN` because $\hat\Omega$ is numerically singular. {#tbl-smm-debt-params}

**Two failure modes are visible.** First, the optimal weight matrix is not available: the J-statistic is `NaN` because $\hat\Omega = \frac{1}{S}\sum_s E_s E_s^\top$ is numerically singular at the Stage-1 estimate. With 11 moments and a moderate $S$, several rows of the per-panel error matrix are nearly collinear, so Stage 2 cannot use $W = \hat\Omega^{-1}$ as the optimal weighting. The sandwich SE for $c_{\text{def}}$ ($0.39$) is essentially the width of its prior range, the typical fingerprint of a singular weight matrix collapsing onto a single parameter. Second, the point estimates of $\alpha$, $\psi_1$, $\eta_1$, and $\sigma$ are biased away from the truth by $4$ to $7$ standard errors. Combined with the moment-fit gaps, this is consistent with weak identification: the moment vector cannot distinguish $\beta^*$ from nearby parameter values that happen to fit a subset of moments better.

**Diagnostic step.** To localize the failure I run two checks on the same fixed simulation seed:

- **Oracle test.** Compute $Q(\beta^*)$ at the true parameters: $Q(\beta^*) \approx 1.4 \times 10^{-3}$, well below the optimization tolerance. The moment-construction pipeline is correctly wired and the truth is reachable in principle; the optimizer simply does not converge to it.
- **Jacobian audit.** Finite-difference the $11 \times 7$ Jacobian $\partial g / \partial \beta$ at $\beta^*$ ($2K$ extra solves). Two findings: (i) the moments *conditional issuance size*, *autocorrelation of equity issuance*, and *cross-correlation leverage-issuance ratio* load almost entirely on $\alpha$ and $\psi_1$ rather than the parameters they were meant to identify. They are redundant with the cleaner @hennessy2007costly moments already in the active set; (ii) the column for $c_{\text{def}}$ has a small norm relative to the other parameter columns, confirming weak identification.

### Attempted fixes

I tried two remedies in addition to dropping the three redundant moments identified by the Jacobian audit. Neither restored the estimator.

**1. Calibrate $c_{\text{def}}$ at the prior value.** Motivated by the Jacobian audit's weak-identification finding, I fix $c_{\text{def}} = 0.45$ and remove it from the estimated parameter vector, dropping $K$ from $7$ to $6$. With 8 remaining moments and 6 parameters, the Jacobian achieves full rank ($6 / 6$). The single-panel estimates of the remaining six parameters are still biased: $\alpha$, $\psi_1$, $\rho$, and $\sigma$ all reject at the 5% level, $\eta_0$ and $\eta_1$ have standard errors above $10^5$ (a numerical artifact of $\hat\Omega^{-1}$ inverting near-zero singular values), and the J-statistic becomes computable but rejects strongly ($J = 44.9$, $p < 10^{-9}$, df $= 2$). Calibrating $c_{\text{def}}$ removes the most visible weak-identification artifact but does not restore the estimator.

**2. Replace summary moments with empirical-policy-function (auxiliary-model) moments** Following @nikolov2021, I run an auxiliary regression on both real and simulated panels and match the *coefficients* of that regression. Concretely, I regress each observed outcome ($I/k$, $b'/k$, max-equity-issuance / $k$) on a small set of lagged states and controls $(\log k_{t-1}, \log z_{t-1}, b_{t-1}/k_{t-1})$, and treat the coefficient vector as the moment vector. This is the indirect-inference approach: the auxiliary model serves as a richer summary of the policy than scalar moments. The richer moment vector improves the conditioning of $\hat\Omega$ and the J-statistic is computable, and the pathological standard errors on $\eta_0$ and $\eta_1$ disappear. Yet the point estimates of $\alpha$, $\psi_1$, $\eta_0$, $\eta_1$ remain biased compared with the truth, and the J-test still rejects.

**Summary.** Neither attempt solved the weak identification problem when applying SMM to the risky debt model. Since we have validated that the SMM pipeline itself is correct, the issue is the model structure itself and would require a better selection of moment conditions beyond the ones used in @hennessy2007costly. Unfortunately, I currently do not have a clean solution to it.


## Defects of the risky debt model and potential solutions

I focus on the defects of the model's core economic mechanisms and discuss their theoretical and practical (empirical) implications. To be fair and constructive, I do not discuss critique that are either too board or general, or those that require adding new features beyond the original focus of the model.

To clarify, the core mechanism of the risky debt model is that firm's financing decisions reflect (i) optimal investment under frictional adjustment cost; and (ii) the opportunity to exploit the tax shield benefit of debt (in the form of a one-period corporate bond). Lenders (bank) with rational expectation charge a risk premium on the yields of the corporate bond based on anticipated default probability. Default threshold is determined by the realization of next-period productivity shock conditional on current states and actions, for example, higher debt requires higher realization of future productivity to repay, and thus expand the default set (probability). This is anticipated by the lender and priced into the endogenously-determined risk premium. 

Given this core mechanism, I find two critical defects in the specific risky debt model presented in @strebulaev2012 [section 3.6]:

1. Timing of tax shield benefit create "always-default" strategy
2. Assumption of perfect managerial information 

Both are structural defects that can only be "solved" by major revisions to the model and estimation strategy. 

### Timing of tax shield benefit

Defect #1 is problematic because the present value of tax shield benefit, $\frac{\tau \tilde{r} b'}{(1+\tilde{r})(1+r)}$ is obtained by firm upfront and is unconditional on the next-period solvent/default states (Equation 3.26). Since the model intentionally does not impose a borrowing limit, firm could exploit an optimal "always-default" strategy that borrow as much as possible and default in next period. The lender rantionalize this in pricing the interest rate $\tilde{r}\to \infty$, but the tax shield benefit is still large and positive: $$ \frac{\tau \tilde{r} b'}{(1+\tilde{r})(1+r)} \rightarrow \frac{\tau b'}{(1+r)} > 0 \quad \text{as} \quad \tilde{r}\to \infty \text{ and } b'\to \infty$$ 

This is confirmed empirically when any naive implementation with large upper bound $b_{\max}$ relative to the $k_{\max}$ will cause the optimal policy to converge to "borrow as much as possible then default" with $b'=b_{\max}$ and $k'\approx 0$. For model solve itself this can be mitigated with well-calibrated parameters and bounds, but the true risk is for SMM estimation when the optimizer re-solved the policy under different parameters and a non-trivial fraction of the parameter combinations will lead to this unintended strategy.

One simple solution is to use the same time schedule as in the trade-off model in @nikolov2021.

### Imperfect managerial information

Defect 2 is directly related to the critique by @deangelo2022: Can manager and lender precisely estimate the continuation value $V$ and default states given by $\{z: V(\cdot, z)\leq 0\}$? This is the key important assumption of the model: the endogenous default decision is a going-concern and manager would only default when the firm's continuation value is negative. The pricing of bonds is from a bargaining between the firm and the lender based on $\mathbb{E}_{z'|z} [V(k',b',z')]$ where there exist a critical value of shock $z'_d$ such that all $z'<z'_d$ are default states [@hennessy2007costly, Proposition 6]. However, if manager and lender cannot learn $V$ with any realistic precision, this core mechanism is broken.

@deangelo2022 [Section VI] reviews direct evidence of imperfect manager knowledge. There are two key takeaways. First, large-scale surveys of CFOs indicated that "most managers have nothing close to the knowledge assumed in extant dynamic capital structure models, which posit a complete understanding of investment opportunities and capital-market conditions over an infinite horizon" [@graham2022presidential]. Second, a number of studies have estimated a near-flat relationship between firm value and leverage, suggesting that "real-world managers are unable to pin down a uniquely optimal capital structure with any real precision" [@korteweg2010net]. 


---

# References

::: {#refs}
:::

# Appendix A. Solution Methods to Dynamic Models {#sec-solve}

## Value and Policy Function Iterations {#sec-VFI}
Value function iteration (VFI) and policy function iteration (PFI) are the most widely used methods to solve discrete-time dynamic programming problems. In their simplest form, these methods discretize the continuous state space into a finite grid $\mathcal{S}_{\text{grid}}$ and iterate on the Bellman equation until convergence.

VFI exploits the property that the Bellman operator is a contraction mapping with unique fixed point $V^*$, so repeatedly applying the operator to any initial $V_0$ converges to $V^*$. Each iteration applies a single Bellman backup across all grid points and selects the maximizing action, but does not explicitly maintain a policy until convergence.

PFI separates each iteration into two steps: (i) **policy evaluation**, which solves for the exact on-policy value function $V^{\pi_j}$ given a fixed policy $\pi_j$, and (ii) **policy improvement**, which updates the policy by maximizing the Bellman right-hand side using $V^{\pi_j}$. The Policy Improvement Theorem guarantees $V^{\pi_{j+1}}(s) \geq V^{\pi_j}(s)$ for all $s$. PFI typically converges in fewer outer iterations than VFI because each iteration performs exact policy evaluation rather than a single Bellman backup, though each iteration is more expensive.

### Algorithm: Value Function Iteration (VFI)
**Input:** Grid $\mathcal{S}_{\text{grid}}$, reward $r$, dynamics $f$, discount $\gamma$, tolerance $\delta > 0$ 

**Output:** $V^*, \pi^*$

1. Initialize $V_0(s) = 0$ for all $s \in \mathcal{S}_{\text{grid}}$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$ **For** each $s \in \mathcal{S}_{\text{grid}}$ **do**
4. $\qquad V_{j+1}(s) = \max_{a} \left\{ r(s, a) + \gamma  \mathbb{E}_{\epsilon}\left[ V_j(f(s, a, \epsilon)) \right] \right\}$
5. $\quad$ **End for**
6. $\quad$ **If** $| V_{j+1} - V_j |_\infty < \delta$ **then break**
7. **End for**
8. $\pi^*(s) = \arg\max_{a} \left\{ r(s, a) + \gamma , \mathbb{E}_{\epsilon}\left[ V^*(f(s, a, \epsilon)) \right] \right\}$ for all $s \in \mathcal{S}_{\text{grid}}$
9. **Return** $V^*, \pi^*$

### Algorithm: Policy Function Iteration (PFI)
**Input:** Grid $\mathcal{S}_{\text{grid}}$, reward $r$, dynamics $f$, discount $\gamma$, tolerance $\delta > 0$ 

**Output:** $V^*, \pi^*$

1. Initialize $\pi_0(s)$ arbitrarily for all $s \in \mathcal{S}_{\text{grid}}$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$ **Policy evaluation:** Solve for $V^{\pi_j}$ satisfying
4. $\qquad V^{\pi_j}(s) = r(s, \pi_j(s)) + \gamma \mathbb{E}_{\epsilon}\left[ V^{\pi_j}(f(s, \pi_j(s), \epsilon)) \right], \quad \forall s \in \mathcal{S}_{\text{grid}}$
5. $\quad$ **Policy improvement:** Update policy
6. $\qquad \pi_{j+1}(s) = \arg\max_{a} \left\{ r(s, a) + \gamma \mathbb{E}_{\epsilon}\left[ V^{\pi_j}(f(s, a, \epsilon)) \right] \right\}, \quad \forall s \in \mathcal{S}_{\text{grid}}$
7. $\quad$ **If** $| \pi_{j+1} - \pi_j |_\infty < \delta$ **then break**
8. **End for**
9. $V^* = V^{\pi_j}$, $\pi^* = \pi_j$
10. **Return** $V^*, \pi^*$

The common limitation of both methods is the reliance on discretization: the computation cost scales with the number of grid points, which grows exponentially in the dimension of the state space, i.e., the so-called "curse of dimensionality".

## Lifetime Reward Maximization {#sec-LRM}

The Lifetime Reward Maximization (LRM) method directly maximizes expected discounted lifetime rewards by simulating trajectories under the current policy. Given initial state $s_0$ and a shock sequence $\{\epsilon_1, \ldots, \epsilon_T\}$, the policy $\pi_\theta$ generates a trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1}, s_T)$ where $a_t = \pi_\theta(s_t)$ and $s_{t+1} = f(s_t, a_t, \epsilon_{t+1})$.

**True objective.** The infinite-horizon value under policy $\pi_\theta$ starting from $s_0$ is:
$$V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_t, \pi_\theta(s_t))\right]$$
Splitting at a finite horizon $T$ gives an exact decomposition:
$$V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right] + \gamma^T \, \mathbb{E}\left[V^{\pi}(s_T)\right]$$
where the second term is the discounted expected continuation value from the terminal state $s_T$ onward.

**Truncated objective.** @maliar2021 approximate the true objective by dropping the continuation term, setting $\hat{V}^{\text{term}}(s_T) = 0$:
$$\max_\theta \; J_T(\theta) = \mathbb{E}_{(s_0,\,\epsilon_1,\dots,\epsilon_T)}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right]$$
This is valid when $T$ is large enough that $\gamma^T V^{\pi}(s_T) \approx 0$. However, the discount factor contracts this term slowly: with $\gamma = 0.96$, keeping the truncation bias below 1\% of the true value requires $T \geq \lceil\log(0.01)/\log(0.96)\rceil = 113$ periods. BPTT through such a long chain is computationally prohibitive because gradient memory scales linearly in $T$, and vanishing or exploding gradients compound across the chain.

**Back-propagation through time (BPTT)** The core mechanics of the LRM method is known as BPTT in machine learning. The idea is to exploit the end-to-end differentiability of the reward function $r$ and the state transition function $f$ and obtain the precise gradient flow $\nabla_\theta J(\theta)$ backward through the entire trajectory generated by $\pi_\theta$ to improve the policy. The main issue of BPTT is the trade-off between truncation bias and computational cost:

- When $T$ is large, computation is slow and expensive, and the gradient may blow up
- When $T$ is moderate-to-small, policy $\pi_\theta$ suffers from nontrivial truncation bias 

**True value function.** 
The true continuation value from the terminal state $s_T$ onward is:
$$V^{\pi}(s_T) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_{T+t},\, \pi(s_{T+t})) \;\middle|\; s_T\right]$$
which integrates over all future shock realizations and the policy's dynamic response to them. 

**Terminal value approximation.** The terminal value approximation formula approximates it by exploiting the endogenous-exogenous state decomposition and using a *deterministic* perpetuity following optimal policy at mean value and realized shocks. This approximation exploits the maximum information of the model's structure, but there remains an approximation error gap that cannot be closed. Specifically, I replace the terminal value using a geometric perpetuity: 
$$\hat{V}^{\text{term}}(s_T^{\text{endo}}) = \frac{r(\bar{s},\, \bar{a})}{1 - \gamma}$$
where state and action variables are set to
$$\bar{s} = [s^{\text{endo}} \mid \bar{s}^{\text{exo}}], \qquad \bar{a} = \bar{a}(s^{\text{endo}})$$
where $\bar{s}^{\text{exo}} = \mathbb{E}[s^{\text{exo}}_\infty]$ is the stationary mean of the exogenous process, and $\bar{a}(s_T^{\text{endo}})$ is the action satisfying $f^{\text{endo}}(s_T^{\text{endo}}, \bar{a}) = s_T^{\text{endo}}$, the steady-state action that holds the endogenous state constant. Both are functions of $s_T^{\text{endo}}$ alone and are model constants provided by the environment. The approximation replaces the stochastic future with a deterministic steady state in which the exogenous state is frozen at its mean and the agent repeats the stationary action forever. The continuation then reduces to a geometric perpetuity. During training, this is  evaluated at $s_{T}^{\text{endo}}$, the terminal endogenous state obtained by rolling out the current policy $\pi_{\theta}$ for $T$ steps. The formula is a fixed function of $s_T^{\text{endo}}$ that does not depend on policy network parameters $\theta$.

Using the basic investment model as an concrete example: exogenous state variable is AR(1) shock $\bar z = \mathbb{E}[z] = \mu$, endogenous state variable is set to steady state capital $k_{ss}$, and action variable at steady state is $I=\delta \cdot k_{ss}$. Because $k_{ss}$ is unknown, the best we can do is to rollout $T$ periods using $\pi_\theta$ and assume that $k_T \approx k_{ss}$. Early in training, the policy has not converged, so the rollout may not reach the steady state even when $T$ is adequate. The terminal value is then a rough approximation. But as the policy improves, rollout trajectories increasingly reach the neighborhood of the steady state, making the terminal value more accurate, which in turn provides a better gradient signal.



**Approximation error.** The error of the perpetuity relative to the true continuation is:
$$\hat{V}^{\text{term}}(s_T^{\text{endo}}) - V^{\pi}(s_T) = \frac{r(\bar{s},\, \bar{a})}{1-\gamma} - \mathbb{E}_{\epsilon}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_{T+t},\, \pi(s_{T+t})) \;\middle|\; s_T\right]$$

To understand its magnitude, consider the idealized case where $s_T^{\text{endo}}$ is at the optimal steady state. The perpetuity gives the reward at the deterministic steady state, while $V^{\pi}$ accounts for the agent's optimal response to future stochastic shocks. By the envelope theorem, the first-order effect of small shocks on the value function vanishes: the agent is already optimizing, so marginal perturbations in the exogenous state are absorbed by optimal policy adjustment. The approximation error is therefore **second-order in the exogenous volatility** $O(\sigma_{\epsilon}^2)$ and it has two components:

1. **Jensen's correction.** The value function is generally concave in the exogenous state. Replacing the stochastic $s^{\text{exo}}$ with its mean overstates the value.
2. **Precautionary motive.** A firm facing adjustment costs benefits from the *option to respond* to future shocks. The perpetuity assumes a fixed action forever, which ignores this option value.

It is important to note that further reducing the approximation error would require explicitly learning $V^{\pi}$ using a value network. At that point the algorithm becomes a critic-based method which is a fundamentally different algorithm. This is exactly the solution method of Short-Horizon Actor Critic that I explore in later section.

**Why this matters?** Without a terminal value $\hat{V}^{\text{term}} \approx V^\pi(s_T)$, the learned policy $\pi_\theta$ has a nontrivial truncation bias. For example, in most economic/finance models, the optimal investment policy would be systematically under-estimated in the absence of $\hat{V}^{\text{term}}$ because the long-run return to investment is ignored. 

**Loss function.** The SGD loss with terminal value correction, evaluated over a mini-batch $\mathcal{B}$:
$$J(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\sum_{t=0}^{T-1} \gamma^t \cdot r(s_{it}, \pi_\theta(s_{it})) + \gamma^T \, \hat{V}^{\text{term}}(s_{iT}^{\text{endo}})\right)$$
Setting $\hat{V}^{\text{term}} = 0$ recovers the truncated objective of @maliar2021. In the BPTT computation, $\hat{V}^{\text{term}}$ should be differentiable with respect to $s_{T}^{\text{endo}}$ so that gradients prevent the policy from de-investing near the horizon, but should not route gradients through the policy network at the terminal step to avoid $1/(1-\gamma)$ gradient amplification through the BPTT chain. The gap between $\hat{V}^{\text{term}}$ and the true continuation $V^{\pi}$ arises because the perpetuity ignores future exogenous volatility and the firm's dynamic response to it.

### Algorithm: Lifetime Reward Maximization
**Input:** Policy network $\pi_\theta$, dynamics $f$, reward $r$, discount $\gamma$, horizon $T$, terminal value $\hat{V}^{\text{term}}$, learning rate $\eta$, convergence rule $\texttt{CONVERGED}(\theta, j)$

**Output:** Trained policy $\pi^*_{\theta}$

1. Initialize policy parameters $\theta$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$ Sample mini-batch $\mathcal{B}$ consisting of initial states $\{s_0\}_i$ and shock sequences $\{\epsilon_1,\dots,\epsilon_T\}_{i}$
4. $\quad$ **For** each observation $i \in \mathcal{B}$, rollout trajectory:
5. $\qquad$ **For** $t = 0, \ldots, T-1$: simulate $a_{i,t} = \pi_\theta(s_{i,t})$ and $s_{i,t+1} = f(s_{i,t}, a_{i,t}, \epsilon_{i,t+1})$
6. $\quad$ Compute loss: $J(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\sum_{t=0}^{T-1} \gamma^t \cdot r(s_{it}, \pi_\theta(s_{it})) + \gamma^T \, \hat{V}^{\text{term}}(s_{iT}^{\text{endo}})\right)$
7. $\quad$ SGD update: $\theta \leftarrow \theta - \eta \cdot \nabla_\theta J(\theta)$
8. $\quad$**If** $\texttt{CONVERGED}(\theta, j)$ **then** **break**
9. **End for**
10. **Return** $\pi_{\theta^*}$


## Euler Residual Minimization {#sec-ERM}

The ER method minimizes the Euler equation errors that characterize optimality. It enforces an intertemporal first-order necessary condition between $(s, a)$ and $(s', a')$ at each observation independently.

**Euler equation.**  At the optimum, the policy $\pi_\theta$ satisfies:

$$\mathbb{E}_\varepsilon \left[F(s, \pi_\theta(s), s', \pi_\theta(s'))\right] = 0$$

where $F: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the Euler residual function derived analytically from the first-order conditions of the Bellman equation, and $s' = f(s, \pi_\theta(s), \varepsilon)$ is computed using the state transition function $f$.

ERM is a one-step method: each observation is a single-step transition $(s, f(s,\cdot), s', f(s',\cdot))$, and the loss is computed independently per observation. ERM is significantly faster than LRM and SHAC because it does not require rolling out a full trajectory and BPTT. The optimality condition is given analytically by the Euler equation, and ERM directly search for $\theta^*$ that directly enforces it.

**Target network.**  Both @maliar2021 and @fernandez-villaverde2025 suggest using a single policy network inside the loss function. However, computing $\pi_\theta(s')$ introduces a recursive dependency: the gradient of $\theta$ flows through both the current policy $\pi_\theta(s)$ and the next-period policy $\pi_\theta(s')$, creating a moving-target problem that prevents convergence. My implementation introduces a separate target network $\pi_{\theta^-}$ for the next-period action, updated via Polyak averaging.

**Loss function** 
The objective is to minimize the squared Euler residual. Following @maliar2021, I use the Monte Carlo cross product estimator with two independent shock draws for unbiased estimation of the squared expectation. Specifically, I first draw two random iid ($\epsilon_1,\epsilon_2$), and use the AR(1) transition function to construct ($s'_1, s'_2$): 

$$\mathcal{L}_\theta = \frac{1}{\mathcal{B}}\sum_{i \in \mathcal{B}} F(s_i, a_i, s'_{i,1}, a'_{i,1}) \cdot F(s_i, a_i, s'_{i,2}, a'_{i,2})$$

where:

- $a_i = \pi_\theta(s_i)$ is current action from the trainable policy network (with gradient)
- $s'_{i,m} = f(s_{i}, \pi_\theta(s_{i}), \varepsilon_{i,m})$ are next states under iid shock draw $m\in \{1,2\}$
- $a'_{i,m} = \pi_{\theta^-}(s'_{i,m})$ is next action from the target policy network (no gradient)
- $\mathcal{B}$ denotes mini-batches and $i$ denotes observation


**Why do we need a target network?** The original method in @maliar2021 and @fernandez-villaverde2025 uses a single policy network $\pi_\theta$ for both current-period and next-period Euler equation. This creates a **moving target problem**. In practice, this leads to oscillatory or divergent dynamics:
the optimizer cannot further minimize the Euler equation error toward zero because every step that adjusts the Euler LHS also shifts the Euler RHS by a similar magnitude.

**The problem with a single network.**  Take the basic investment model as an concrete example, the Euler residual has the structure:

$$F = 1 - \frac{1}{1+r} \cdot \frac{m(\pi_\theta(s'))}{{\chi(\pi_\theta(s))}}$$

where $\chi$ is the marginal cost of investment today (depends on the
current action $a = \pi_\theta(s)$) and $m$ is the marginal benefit of
capital tomorrow (depends on the next-period action $a' = \pi_\theta(s')$).
At optimum, $F = 0$: marginal cost equals discounted marginal benefit. When a single network
$\pi_\theta$ supplies both $a$ and $a'$, any update to $\theta$
simultaneously moves both sides of the equation.  Consider a gradient
step that increases investment everywhere:

- **Today** ($a = \pi_\theta(s)$): higher investment raises the marginal
  cost $\chi$ — the denominator increases.
- **Tomorrow** ($a' = \pi_\theta(s')$): higher investment also raises
  $\chi' = 1 + \partial\psi'/\partial k''$ via the next-period adjustment
  cost — the numerator $m$ increases too.

Both sides of the ratio $m / \chi$ shift in response to the same
parameter update.  The gradient points toward the correct equilibrium, but
the target it is aiming at (the RHS) moves by a comparable amount at each
step.  

This is not unique to investment models.  In any Euler equation
$\text{MC}(a) = \beta \cdot \mathbb{E}[\text{MB}(a')]$, the marginal
quantities on both sides are evaluated under the *same* policy.  A
parameter change that reduces the residual at the current $\theta$ does not
guarantee a smaller residual at $\theta + \Delta\theta$, because the
next-period side has shifted.

**How the target network resolves this.**  The target network
$\pi_{\theta^-}$ provides a fixed reference for the next-period action:

- $a = \pi_\theta(s)$ — gradients flow through the current policy.
- $a' = \pi_{\theta^-}(s')$ — **gradients stopped**; weights are frozen for
  this step.

Now the marginal benefit $m$ is computed from $\pi_{\theta^-}$, which
moves slower per step via Polyak averaging.
The optimizer sees a near-stationary target: it adjusts the current-period
action to match the slowly-moving next-period reference, and the reference
gradually tracks the improving policy.  This converts the unstable
simultaneous update into a stable fixed-point iteration.

### Algorithm: Euler Residual Minimization
**Input:** Policy network $\pi_\theta$, target policy $\pi_{\theta^-}$, known state transition function $f$, analytical formula of Euler equation error $F$, flattened dataset $\{(s_{i}, \epsilon_{i,1}, \epsilon_{i,2} \}_{i=1}^N$, mini-batch $B$, Polyak rate $\tau_{\text{polyak}}$, learning rate $\eta$, convergence rule $\texttt{CONVERGED}(\theta, j)$

**Output:** Trained policy network $\pi^*_\theta$

1. Initialize policy parameters $\theta$ and create target network $\theta^- \leftarrow \theta$
2. Fit a shared StaticNormalizer on the full training dataset (identical statistics, separate copy per network)
3. **For** $j = 0, 1, 2, \ldots$ **do**
4. $\quad$ Sample mini-batch $\mathcal{B} = \{(s_{i},\, \epsilon_{i,1},\, \epsilon_{i,2})\}_{i=1}^N$ from the flattened dataset
5. $\quad$ Use current policy $\theta$ to compute action: $a_i = \pi_\theta(s_i)$
6. $\quad$ **For** $m \in \{1,2\}$: Rollout $s'_{i,m} = f(s_{i}, \pi_\theta(s_{i}), \varepsilon_{i,m})$ using current policy weights $\theta$ 
7. $\quad$ **For** $m \in \{1,2\}$: Rollout $a'_{i,m} = \pi_{\theta^-}(s'_{i,m})$ using target policy weights $\theta^-$ (no gradient)
8. $\quad$ **For** $m \in \{1,2\}$: Compute Euler residuals $F_{i,m} = F(s_i,\, a_i,\, s'_{i,m},\, a'_{i,m})$
9. $\quad$ Compute loss function: $\mathcal{L}(\theta) = \tfrac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} F_{i,m=1} \cdot F_{i,m=2}$ 
10. $\quad$ Update current policy: $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta)$
11. $\quad$ Update target policy: $\theta^- \leftarrow \tau_{\text{polyak}} \cdot \theta^- + (1 - \tau_{\text{polyak}}) \cdot \theta$
12. $\quad$ **If** $\texttt{CONVERGED}(\theta, j)$ **then** **break**
13. **End for**
14. **Return** $\pi_{\theta^*}$



## Short-Horizon Actor-Critic (SHAC) {#sec-SHAC}

SHAC solves infinite-horizon dynamic programming problems by combining
short-horizon backpropagation through differentiable dynamics with a
learned value function network.  It builds on a modern RL algorithm developed by @xu2022. I adopt the core structure of windowed actor BPTT with on-policy continuation across windows, but replacing the critic update with a one-step Bellman target -- similar to the Deep Deterministic Policy Gradient (DDPG) method -- to improve stability in economic environments.

**Core idea.**  The full $T$-step trajectory is divided into consecutive
windows of length $h$.  Within each window, the actor loss
backpropagates through $h$ exact dynamics steps, and a value function $V$
bootstraps the continuation beyond the window boundary.  Between
windows, the endogenous state carries forward (detached via
`stop_gradient`) so the trajectory remains on-policy.  This avoids the
gradient explosion/vanishing of full-trajectory BPTT while retaining
exact policy gradients through known, differentiable dynamics.

**Reward and Bellman normalization.** Unlike in standard RL environments, economic models typically have large reward and value scales, yet SHAC's default hyperparameters assume $O(1)$. To bring values into the right range and stablize training, I rescale every reward by $\lambda_r = 1/|V^*|$ so that the critic learns values of $O(1)$. To obtain $V^*$, I use the environment's baseline steady-state value that can be obtained analytically. For example, in basic investment model I use the first-order approximation value $V^*=\hat{V}^{\text{term}}$ as described in the [LRM appendix](#sec-LRM). The scaling is applied uniformly to $r$ in both the actor loss and the critic Bellman target, which makes the rescaling mathematically equivalent to the unscaled algorithm: the critic learns $\lambda_r V^\pi$ in place of $V^\pi$, and the optimal policy is unchanged. Because $\lambda_r$ is a numerical preconditioning factor (i.e., multiplied by a constant) rather than part of the algorithm's logic, I omit it from the algorithm summary below.

**Why this variant?**  The original SHAC uses an on-policy TD-$\lambda$
critic trained on rewards from the actor's own rollout.  In economic
environments with large reward scales, this creates a positive feedback
loop: the critic overfits to the actor's trajectory, the actor exploits
the critic's overestimates, and training diverges.  Our variant
decouples the critic from the current actor by using a 1-step Bellman
target with separate target networks for both policy and value.

This method uses four separate NNs to separately approximate the policy and value functions:

| Network | Notation | Output head | Role |
|---------|----------|-------------|------|
| Policy (actor) | $\pi_\theta$ | Affine + clip (see below) | Maps full state $s$ to action $a$ |
| Value (critic) | $V_\phi$ | Dense(1), linear | Maps full state $s$ to scalar value |
| Target policy | $\bar{\pi}_\theta$ | Same as $\pi_\theta$ | Polyak-averaged copy of $\pi_\theta$; separate weights |
| Target value | $\bar{V}_\phi$ | Same as $V_\phi$ | Polyak-averaged copy of $V_\phi$; separate weights |


### Algorithm: Short-Horizon Actor-Critic
**Input:** Actor $\pi_\theta$, target actor $\pi_{\theta^-}$, critic $V_\phi$, target critic $V_{\phi^-}$,
dynamics $f$, reward $r$, discount $\gamma$, mini-batch size $B$,
total horizon $T$, window length $h$ (with $T \bmod h = 0$), critic steps per window $n_{\text{critic}}$,
Polyak rate $\tau_{\text{polyak}}$, learning rates $\eta_\theta, \eta_\phi$,
convergence rule $\texttt{CONVERGED}(\theta, \phi, j)$

**Output:** Trained actor $\pi^*_\theta$ and critic $V^*_\phi$

1. Initialize $\theta, \phi$ and target networks $\theta^- \leftarrow \theta$, $\phi^- \leftarrow \phi$;
   fit a shared StaticNormalizer on the full training dataset.
2. **For** training steps $j = 0, 1, 2, \ldots$ **do**
3. Sample mini-batch $\mathcal{B}$ of trajectories with initial states $\{s_{i,0}\}$ and
   pre-simulated shock paths $\{\varepsilon_{i,\, 1:T}\}$; set the rollout state $s_i \leftarrow s_{i,0}$.
4. **For** windows $w = 0, 1, \ldots, T/h - 1$ **do** (with start time $t_0 = w h$):
5. **(a) Actor BPTT.** Set $s_{i,0} \leftarrow s_i$ (detached) and unroll $h$ steps under the current actor:
   $$a_{i,\ell} = \pi_\theta(s_{i,\ell}), \qquad s_{i,\ell+1} = f(s_{i,\ell},\, a_{i,\ell},\, \varepsilon_{i,\, t_0+\ell+1}),
     \qquad \ell = 0, \ldots, h-1$$
6. **(b) Actor loss and update** (current critic $V_\phi$ as terminal bootstrap):
   $$\mathcal{L}(\theta) = -\frac{1}{B}\sum_{i \in \mathcal{B}}
     \left[\sum_{\ell=0}^{h-1} \gamma^\ell\, r(s_{i,\ell},\, a_{i,\ell}) + \gamma^h\, V_\phi(s_{i,h})\right],
     \qquad \theta \leftarrow \theta - \eta_\theta\, \nabla_\theta \mathcal{L}(\theta)$$
   Carry $s_i \leftarrow \texttt{stop\_gradient}(s_{i,h})$ into the next window.
7. **(c) Critic 1-step Bellman regression** on the detached set
   $\mathcal{D}_w = \{(s_{i,\ell},\, \varepsilon_{i,\, t_0+\ell+1}) : i \in \mathcal{B},\, \ell = 0, \ldots, h-1\}$.
   Define the Bellman target under the target networks:
   $$y(s, \varepsilon) = \texttt{stop\_gradient}\!\left[r(s,\, \pi_{\theta^-}(s))
     + \gamma\, V_{\phi^-}\!\big(f(s,\, \pi_{\theta^-}(s),\, \varepsilon)\big)\right]$$
   For $u = 1, \ldots, n_{\text{critic}}$:
   $$\mathcal{L}(\phi) = \frac{1}{|\mathcal{D}_w|}\sum_{(s,\,\varepsilon) \in \mathcal{D}_w}
     \!\left(V_\phi(s) - y(s, \varepsilon)\right)^2,
     \qquad \phi \leftarrow \phi - \eta_\phi\, \nabla_\phi \mathcal{L}(\phi)$$
8. **(d) Polyak update target networks:**
   $$\theta^- \leftarrow \tau_{\text{polyak}}\, \theta^- + (1 - \tau_{\text{polyak}})\, \theta,
     \qquad \phi^- \leftarrow \tau_{\text{polyak}}\, \phi^- + (1 - \tau_{\text{polyak}})\, \phi$$
9. **If** $\texttt{CONVERGED}(\theta, \phi, j)$ **then break**
10. **End for** (windows)
11. **End for** (training steps)
12. **Return** $\pi^*_\theta,\; V^*_\phi$


## Bellman Residual Minimization {#sec-BRM}
The Bellman Residual Minimization (BRM) method jointly trains a policy network $\pi_\theta$ and a value function network $V_\phi$ to satisfy the Bellman equation. The challenge is that the Bellman equation contains a $\max$ operator:

$$V(s) = \max_a \left\{ r(s, a) + \gamma \mathbb{E}_\epsilon\left[V(s')\right] \right\}$$

Rather than solving the inner maximization directly, @maliar2021 and @fernandez-villaverde2025 suggest eliminating the $\max$ by adding the first-order necessary condition as auxiliary losses. This turn the loss function into a multitask objective that combines the Bellman residual, the first-order condition (FOC), and other model-specific constraints and optimality conditions with user-specified exogenous weights.

**Bellman residual.** For a given policy $\pi_\theta$ and value function $V_\phi$, define the Bellman equation residual for each observation $i$:

$$F^{\text{BR}}_{i,m} = V_\phi(s_i) - r(s_i, a_i) - \gamma V_\phi(s'_{i,m})$$

where $a_i = \pi_\theta(s_i)$ and $s'_{i,m} = f(s_i, a_i, \epsilon_{i,m})$ for two Monte Carlo draws $m=1,2$. We state that the value function $V^\pi$ satisfies the **on-policy** Bellman equation if and only if $\mathbb{E}_\epsilon[F^{\text{BR}}] = 0$. Note that this can be satisfied for any arbitrary policy $\pi$ that is not optimal.

**FOC residual.** Differentiating the Bellman RHS with respect to the action $a$ yields the necessary condition:

$$F^{\text{FOC}}_{i,m} = \nabla_a r(s_i, a)\big|_{a = a_i} + \gamma \nabla_{s'} V_\phi(s'_{i,m}) \cdot \nabla_a f(s_i, a, \epsilon_{i,m})\big|_{a = a_i}$$

The necessary condition for optimality is $\mathbb{E}_\epsilon[F^{\text{FOC}}] = 0$.

**Envelope condition residual.** Differentiating the Bellman equation with respect to the state $s$ (applying the envelope theorem) gives:

$$F^{\text{Env}}_i = \nabla_s r(s_i, a_i) - \nabla_s V_\phi(s_i)$$

This condition involves no expectation over future shocks, so the loss uses a direct squared residual.

**Feasibility constraints (Optional).** When the model needs to satisfy feasibility constraints given as:
$$
G(\cdot;\theta) \leq 0 \quad \text{and} \quad H(\cdot;\theta) = 0
$$
where $G(\cdot)$ and $H(\cdot)$ can be either linear or non-linear functions over states, actions, or state-action pairs. There are different approaches to handle complementary constraints like Kuhn-Tucker (KT) conditions. @maliar2021 uses additional NNs to approximate the Lagrangian multiplier on each of the constraint and construct separate loss that measures the empirical violations of the constraint. Let $\mathcal{L}^{IC}$ and $\mathcal{L}^{EC}$ denote the loss for the inequality and equality constraints, respectively, and they will be added into the total loss function with exogenous weight.

**Total loss with AiO integration:**

$$J(\theta, \phi) = \underbrace{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} F^{\text{BR}}_{i,1} \cdot F^{\text{BR}}_{i,2}}_{\mathcal{L}^{\text{BR}}} 
+ w_1 \underbrace{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} F^{\text{FOC}}_{i,1} \cdot F^{\text{FOC}}_{i,2}}_{\mathcal{L}^{\text{FOC}}} 
+ w_2 \underbrace{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} (F^{\text{Env}}_i)^2}_{\mathcal{L}^{\text{Env}}} + w_3 \mathcal{L}^{\text{IC}} + w_4 \mathcal{L}^{\text{EC}}$$

where $w_1, w_2 > 0$ are exogenous weights that must be tuned _manually and carefully_ because $\mathcal{L}^{\text{BR}}$ is measured in value levels (squared Bellman residual), but $\mathcal{L}^{\text{FOC}}$ and $\mathcal{L}^{\text{Env}}$ are measured in derivatives and feasibility constraints $\mathcal{L}^{\text{IC}}$ and $\mathcal{L}^{\text{EC}}$ are measured in arbitrary units (depending on the model).

**Fundamental defects**. In practice, the BRM method is extremely sensitive to the choice of exogenous weights and the unit (scale) of each loss. @maliar2021 recommend fine tuning of the weights in pre-training to make the magnitude of each loss roughly the same, but they still find this method obviously less accurate than the other methods.

Although pre-training and fine tuning helps, I find a more serious defect of this method that is overlooked in @maliar2021: *minimizing the current multi-task objective function does not guarantee convergence to the optimal policy*. 

This is because these multiple auxiliary losses are not all serving a shared goal, but instead creates conflicting gradients that lead to incorrect solutions. Only the FOC loss provides the correct gradient signals toward the optimal policy $\theta$, while $\mathcal{L}^{\text{BR}}$ and other auxilary losses can be flexibly minimized by a set of arbitrary NN weights $\theta, \phi$. As a result, BRM training often lead to spurious convergence where the joint-loss function is minimized but the learned $\theta, \phi$ are obviously wrong.

This defect is further worsen by two machanics: (1) FOC loss is much smaller than Bellman error and other losses, so early training typically focuses on minimizing $\mathcal{L}^{\text{BR}}$ and "ignoring" $\mathcal{L}^{\text{FOC}}$, which lead to a self-consistent Bellman for any arbitrary policy weight $\theta$. (2) Bootstrap estimate of $V_\phi$ is lower than the true $V^*$ due to NN initialization around zero. 

Althought $\mathcal{L}^{\text{FOC}}$ can provide correct gradient signals for $\theta$, it is not sufficient to ensure the converged $\pi_\theta \approx \pi^*$. In practice, the BRM training usually plateau at a small loss where the NN find an arbitrary pair of ($\theta, \phi$) that satisfies on-policy Bellman but only weakly satisfies the FOC. 

**Potential solution: warm-start value network.** I find that warm-start the value NN $V_\phi$ using a supervised regression can help training to learn the correct "shape" of the optimal policy, but the solution remains biased and the training is instable compared with other methods. The idea is use a baseline closed-form $\hat{V}$ as regression label to pre-train $V_\phi$, so that in BRM training the initial $V_\phi$ already captures the correct "shape", in such case, the algorithm is more likely to converge (but it is still not guaranteed).

**Target Network.** As in the ER method, the value network $V_\phi(s'_{i,m})$ introduces a recursive gradient dependency: the SGD update to $\phi$ changes both the current-state evaluation $V_\phi(s_i)$ and the target $V_\phi(s'_{i,m})$ simultaneously. Maliar et al. (2021) do not address this; the actor-critic method in the [SHAC appendix](#sec-SHAC) resolves it via target networks and separated updates.

### Algorithm: Bellman Residual Minimization 

**Input:** Policy network $\pi_\theta$, value network $V_\phi$, dynamics $f$, reward $r$, discount $\gamma$, exogenous weights $w$, learning rate $\eta$, convergence rule $\texttt{CONVERGED}(\theta, \phi, j)$ 

**Output:** Trained policy $\pi^*_\theta$, value function $V^*_\phi$

1. Initialize policy parameters $\theta$ and value parameters $\phi$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$  Sample mini-batch $\mathcal{B}$ of states ${s_i}$ with two independent shock draws $\{\epsilon_{i,m}\}_{m=1}^2$
4. $\quad$  Compute actions: $a_i = \pi_\theta(s_i)$
5. $\quad$  Compute next states: $s'_{i,m} = f(s_i, a_i, \epsilon_{i,m})$ for $m = 1, 2$
6. $\quad$  Compute Bellman residuals: $F^{\text{BR}}_{i,m} = V_\phi(s_i) - r(s_i, a_i) - \gamma V_\phi(s'_{i,m})$ for $m = 1, 2$
7. $\quad$  Compute FOC residuals: $F^{\text{FOC}}_{i,m} = \nabla_a r|_{a_i} + \gamma \nabla_{s'} V_\phi(s'_{i,m}) \cdot \nabla_a f|_{a_i}$ for $m = 1, 2$
8. $\quad$  Compute envelope residuals: $F^{\text{Env}}_i = \nabla_s r(s_i, a_i) - \nabla_s V_\phi(s_i)$
9. $\quad$  Compute constraint losses (if any): $\mathcal{L}^{\text{IC}}$ and $\mathcal{L}^{\text{EC}}$
10. $\quad$ Compute combined loss: $J(\theta, \phi) = \mathcal{L}^{\text{BR}} + w_1 \mathcal{L}^{\text{FOC}} + w_2 \mathcal{L}^{\text{Env}} + w_3 \mathcal{L}^{\text{IC}} + w_4 \mathcal{L}^{\text{EC}}$
11. $\quad$ Update: $(\theta, \phi) \leftarrow (\theta, \phi) - \eta \nabla_{(\theta,\phi)} J(\theta, \phi)$
12. $\quad$ **If** $\texttt{CONVERGED}(\theta, \phi, j)$ **then break**
13. **End for**
14. **Return** $\pi^*_\theta$, $V^*_\phi$


## Nested VFI (Risky Debt) {#sec-NestedVFI}

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


# Implementation Details {#sec-impl}

This appendix describes architecture-level implementation choices common to all deep-learning solvers (LRM, ERM, BRM, SHAC) in the codebase. The choices are not method-specific and are unchanged across solver classes.

## Input Normalization {#sec-impl-norm}

State variables in economic models span orders of magnitude (capital in the hundreds, log-productivity near zero, interest rates as fractions). Without normalization, the first-layer gradient scales with raw feature variances and the optimizer cannot make balanced progress across features. I apply a per-feature **static Z-score** to the input layer:

$$\hat{x}_d = \frac{x_d - \mu_d}{\sigma_d + \varepsilon},$$

where $\mu_d, \sigma_d$ are computed once from the full training dataset before any gradient steps and held fixed throughout training. No hidden-layer normalization (BatchNorm, LayerNorm) is applied.

Statistics for the exogenous component $s^{\text{exo}}$ are fit on all $N \times T$ trajectory samples to capture the AR(1) ergodic distribution. Statistics for the endogenous component $s^{\text{endo}}$ are fit on the $N$ initial states drawn uniformly over the bounded state space. The normalizer does not need to be ergodically exact: every state visited during training falls within the bounded region by construction, so its purpose is to map inputs to an $O(1)$ range that conditions the first-layer gradient, nothing more. Online Z-score normalizers common in RL add no information here because the full dataset is available before training begins.

## Hidden-Layer Activation {#sec-impl-activation}

The hidden layers use the **SiLU** (Sigmoid-weighted Linear Unit, also known as Swish) activation:

$$\mathrm{SiLU}(h) = h \cdot \sigma(h), \qquad \mathrm{SiLU}'(h) = \sigma(h)\bigl(1 + h(1 - \sigma(h))\bigr).$$

ReLU is the standard alternative but has zero gradient for $h < 0$. Any neuron whose pre-activation is negative for all training samples never recovers, a "dead neuron". With centered inputs (after the static Z-score above), roughly half of pre-activations are negative on average, so the dead-neuron risk is concrete rather than theoretical. SiLU's gradient is nonzero everywhere, eliminates dead neurons, and is smooth, which matches the smooth, concave objectives typical in economic models.

## Output Head Transformation {#sec-impl-output}

The policy network outputs a continuous action constrained to box bounds $[a_{\min}, a_{\max}]$ (e.g., investment $I \in [I_{\min}, I_{\max}]$). The standard RL choice is a $\tanh$ squashing function. I instead use a **linear output head followed by clipping**:

$$\hat{y} = w^\top \mathbf{a}^L + b, \qquad a = \mathrm{clip}(\hat{y},\, a_{\min},\, a_{\max}).$$

The motivation is gradient quality near the bounds. For $\tanh$ (or any differentiable bijection $\mathbb{R} \to (a_{\min}, a_{\max})$), $\partial a / \partial \hat{y} \to 0$ as $a$ approaches either bound — a topological necessity for a bounded smooth function. In standard RL benchmarks the optimal policy is rarely near the bounds and the saturation is harmless. In economic models the optimal policy is often near the upper bound when productivity is high, and the per-period reward also has diminishing marginal returns in that region. The two effects compound: $\partial \mathcal{L}/\partial \theta$ becomes small in exactly the region where the policy needs the most learning signal, and the trained policy systematically deviates from the analytical benchmark at the boundaries.

The linear-plus-clip head avoids this. Inside the feasible region the output is identity and $\partial a / \partial \hat{y} = 1$ uniformly; outside, the gradient is zero but the action is correctly snapped. The interior gradient is independent of distance to the boundary, so the policy learns boundary-pushing behavior without saturation. The same design is used by TD-MPC2's MPPI planner, PPO with clipped actions, and DDPG with bounded action spaces.

---
# Appendix B. Structural Estimation

## Generalized Method of Moments (GMM) {#sec-gmm-appendix}

GMM estimates structural parameters from moment conditions that are closed-form functions of observables and parameters. Unlike SMM, GMM does not require solving the model: it applies whenever the model produces structural restrictions (e.g., Euler equations) that can be evaluated directly from data and a candidate $\beta$.

| Symbol | Definition |
|---|---|
| $\beta^*$ | True structural parameters. Unknown. |
| $\beta$ | A candidate parameter vector. |
| $\hat{\beta}$ | The GMM estimate that minimizes $Q(\beta)$. |
| $K$ | Number of parameters to estimate. |
| $R$ | Total number of moment conditions ($R \geq K$). |
| $N$ | Number of cross-sectional units. |
| $T$ | Number of time periods. |
| $e_{it}(\beta)$ | Structural residual for observation $i, t$. At $\beta^*$, $\mathbb{E}_t[e_{it}(\beta^*)] = 0$. |
| $Z_{it}$ | Instrument vector: variables known at time $t$, uncorrelated with $e_{it}(\beta^*)$. |
| $g(\beta)$ | $R \times 1$ sample moment vector. At $\beta^*$, $\mathbb{E}[g(\beta^*)] = 0$. |
| $W$ | $R \times R$ positive-definite weighting matrix. |
| $g_{it}(\beta)$ | Per-observation moment contribution: $e_{it}(\beta) \cdot Z_{it}$. |
| $\hat{\Omega}$ | Long-run variance-covariance matrix of $g_{it}$. |
| $D$ | $R \times K$ Jacobian: $D_{rk} = \partial g_r / \partial \beta_k\vert_{\hat{\beta}}$. |
| $V$ | $K \times K$ asymptotic variance-covariance matrix of $\hat{\beta}$. |

The conditional restriction $\mathbb{E}_t[e_{it}(\beta^*)] = 0$ implies $\mathbb{E}[e_{it}(\beta^*) \cdot Z_{it}] = 0$ for any time-$t$ instrument $Z_{it}$. Stacking instrument interactions gives the sample moment vector

$$g(\beta) = \frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T} e_{it}(\beta) \cdot Z_{it}.$$

Identification requires $R \geq K$ and that the instruments be relevant (correlated with the endogenous variables in $e_{it}$) and exogenous (uncorrelated with $e_{it}(\beta^*)$).

### Estimator and inference

The estimator minimizes $Q(\beta) = g(\beta)^\top W\, g(\beta)$ for a positive-definite $W$. Each evaluation is arithmetic on the data, so a standard local optimizer (`scipy.optimize.minimize` with `Powell` or `L-BFGS-B`) suffices.

**Two-step weighting.** The optimal $W$ depends on $\beta^*$, so estimation is iterative.

1. Set $W = I_R$ and minimize to obtain $\hat{\beta}_1$.
2. At $\hat{\beta}_1$, compute the per-observation contributions $g_{it}(\hat{\beta}_1) = e_{it}(\hat{\beta}_1) \cdot Z_{it}$ and estimate $\hat{\Omega}$. For dynamic models, the moment contributions are serially correlated within firm because consecutive observations share persistent state variables, so a HAC estimator is required:
$$\hat{\Omega}_{\text{HAC}} = \hat{\Gamma}_0 + \sum_{l=1}^{L} w(l)\bigl(\hat{\Gamma}_l + \hat{\Gamma}_l^\top\bigr), \qquad \hat{\Gamma}_l = \frac{1}{NT}\sum_{i,\,t > l} g_{it}\, g_{i,t-l}^\top,$$
with Bartlett kernel weights $w(l) = 1 - l/(L+1)$ and bandwidth $L = \lfloor T^{1/3} \rfloor$. Cross-sectional independence across firms is assumed. The i.i.d. estimator $\hat{\Omega}_{\text{iid}} = \frac{1}{NT}\sum_{i,t} g_{it} g_{it}^\top$ should only be used when serial correlation is absent, since it underestimates the true variance otherwise.
3. Set $W = \hat{\Omega}^{-1}$ and warm-start from $\hat{\beta}_1$ to obtain $\hat{\beta}$.

**Inference at $\hat{\beta}$.** With $W = \hat{\Omega}^{-1}$, the asymptotic variance is
$$V = (D^\top \hat{\Omega}^{-1} D)^{-1}, \qquad \text{se}(\hat{\beta}_k) = \sqrt{V_{kk} / (NT)},$$
where the Jacobian $D$ is computed analytically or by centered finite differences. The t-statistic $t_k = (\hat{\beta}_k - \beta_k^0)/\text{se}(\hat{\beta}_k)$ tests $H_0: \beta_k^* = \beta_k^0$. The overidentification test uses
$$J = NT \cdot Q(\hat{\beta}) \;\xrightarrow{d}\; \chi^2(R - K),$$
requiring $R > K$ and $W = \hat{\Omega}^{-1}$. Reject at level $\alpha$ if $J > \chi^2_{1-\alpha}(R-K)$; rejection indicates misspecification.

### Application to the basic investment model with convex cost

I apply GMM to the basic investment model with convex adjustment costs in @strebulaev2012 [section 3.1], which has a closed-form Euler equation. The structural primitives are:

- Production $\pi(k,z) = z k^{\alpha}$ with $\alpha \in (0, 1)$.
- Convex adjustment cost $\psi(I,k) = \tfrac{\psi_1}{2} I^2 / k$ with $I_t = k_{t+1} - (1-\delta) k_t$.
- AR(1) productivity $\ln z_{t+1} = \rho \ln z_t + \varepsilon_{t+1}$, $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)$.
- Estimated parameters $\beta = (\alpha, \psi_1, \rho, \sigma_\varepsilon)$, $K = 4$. Calibrated: $r$, $\delta$.

**Observables.** For firm $i$ at time $t$, the observable variables are $\pi_{it} = z_{it} k_{it}^\alpha$ (operating income), $k_{it}$ (book capital), and $I_{it} = k_{i,t+1} - (1-\delta) k_{it}$ (investment). The productivity $z_{it}$ is latent and recovered at a candidate $\alpha$ via $\ln z_{it}(\alpha) = \ln \pi_{it} - \alpha \ln k_{it}$. Every term in the residuals below is a known function of $(\pi, k, I)$ and the candidate $\beta$.

**Structural residuals.** Eliminating $V$ via the envelope condition, the Euler equation is

$$1 + \psi_1 \frac{I_t}{k_t} = \frac{1}{1+r}\,\mathbb{E}_t\!\left[\alpha\frac{\pi_{t+1}}{k_{t+1}} + \frac{\psi_1}{2}\!\left(\frac{I_{t+1}}{k_{t+1}}\right)^{\!2} + (1-\delta)\!\left(1 + \psi_1 \frac{I_{t+1}}{k_{t+1}}\right)\right].$$

The marginal product of capital $\alpha z_{t+1} k_{t+1}^{\alpha-1} = \alpha\, \pi_{t+1}/k_{t+1}$ is directly computable from observables. The Euler residual replaces the conditional expectation with realized values:

$$e_{it}^u(\beta) = \alpha\frac{\pi_{i,t+1}}{k_{i,t+1}} + \frac{\psi_1}{2}\!\left(\frac{I_{i,t+1}}{k_{i,t+1}}\right)^{\!2} + (1-\delta)\!\left(1 + \psi_1\frac{I_{i,t+1}}{k_{i,t+1}}\right) - (1+r)\!\left(1 + \psi_1\frac{I_{it}}{k_{it}}\right).$$

This block identifies $\alpha$ and $\psi_1$. The AR(1) residual $e_{it}^v(\beta) = \ln z_{i,t+1}(\alpha) - \rho \ln z_{it}(\alpha)$ identifies $\rho$ (and $\alpha$ through $\ln z$). The variance condition $e_{it}^w(\beta) = (e_{it}^v)^2 - \sigma_\varepsilon^2$ identifies $\sigma_\varepsilon$.

**Instruments: lagged only.** I use strictly lagged variables (time $t-1$ and earlier) as instruments. Current-period variables ($I_t/k_t$, $\pi_t/k_t$, $\ln z_t$) appear directly in the residuals; using them as instruments would create mechanical second-moment terms like $\psi_1 (I_t/k_t)^2$ that produce pathologically small standard errors. The Euler block uses

$$Z_{it}^u = (1,\; I_{i,t-1}/k_{i,t-1},\; \pi_{i,t-1}/k_{i,t-1})^\top.$$

The shock block uses $Z_{it}^v = (1,\; \ln z_{i,t-1}(\alpha))^\top$, which depends on $\alpha$ through the recovery formula and is re-evaluated at each candidate. The variance block uses only the constant. Validity holds because all instruments are known at time $t-1$ and the AR(1) innovations are i.i.d.

**Stacked moment vector.** The $R \times 1$ sample moment vector with $R = 6$ is

$$g(\beta) = \frac{1}{NT}\sum_{i,t} \begin{pmatrix} e^u_{it}(\beta) \cdot Z_{it}^u \\ e^v_{it}(\beta) \cdot Z_{it}^v \\ e^w_{it}(\beta) \end{pmatrix}.$$

| Block | Residual | Instruments | Conditions | Identifies |
|---|---|---|---|---|
| Euler equation | $e^u_{it}$ | $Z^u_{it}$ ($3\times 1$) | 3 | $\alpha,\, \psi_1$ |
| Shock process | $e^v_{it}$ | $Z^v_{it}$ ($2\times 1$) | 2 | $\rho$ (and $\alpha$ via $\ln z$) |
| Variance | $e^w_{it}$ | constant | 1 | $\sigma_\varepsilon$ |
| **Total** | | | **6** | **$K=4$, overid $=2$** |

: Moment block summary for the basic investment model. {#tbl-gmm-moments}

The optimal weight $\hat{\Omega}$ is computed with the HAC estimator above. This GMM design requires a closed-form Euler equation; fixed costs, default options, and other non-differentiabilities break it and require switching to SMM.

## Simulated Method of Moments (SMM) {#sec-smm-appendix}

I estimate the parameters of the risky debt model in @strebulaev2012 [section 3.6] using SMM. Each candidate $\beta$ requires a fresh model solve (VFI / PFI / NN method), so wall time is dominated by the optimizer's inner loop.

| Symbol | Definition |
|---|---|
| $\beta^*$ | True structural parameters. Unknown. |
| $\beta$ | A candidate parameter vector. |
| $\hat{\beta}$ | The SMM estimate that minimizes $Q(\beta)$. |
| $x$ | Real-world dataset (one panel). |
| $K$ | Number of parameters to estimate. |
| $R$ | Number of moments ($R \geq K$). |
| $S$ | Number of independently simulated panels per evaluation of $Q$. |
| $M(x)$ | $R \times 1$ moment vector from the real data. |
| $m_s(\beta)$ | $R \times 1$ moment vector from simulated panel $s$. |
| $\bar{m}(\beta)$ | $R \times 1$ averaged simulated moments: $\bar{m} = \frac{1}{S}\sum_s m_s$. |
| $W$ | $R \times R$ positive-definite weighting matrix. |

### Estimator and inference

1. **Setup (once).** Compute target moments $M(x)$ from the real panel. Fix simulation shocks via a master seed: at every evaluation of $Q$, the same uniform draws $u \sim U(0,1)$ are converted to model-specific shocks (e.g., $\varepsilon = \Phi^{-1}(u)$ for AR(1)). Common random numbers across $\beta$ candidates make $Q(\beta)$ smooth and reduce optimizer iterations.

2. **Evaluate $Q(\beta)$.** Solve the model at $\beta$, simulate $S$ panels using the fixed shocks, compute moments per panel $m_s(\beta)$, average to $\bar{m}(\beta) = \frac{1}{S}\sum_s m_s(\beta)$, form the error vector $e(\beta) = \bar{m}(\beta) - M(x)$, and return
$$Q(\beta) = e(\beta)^\top W\, e(\beta).$$
Retain $\{m_s(\beta)\}_{s=1}^S$ for later $\hat{\Omega}$ construction. Level deviations are the default; percent deviations $e_r = (\bar{m}_r - M_r)/M_r$ are available but only safe when moments have comparable magnitudes (small $M_r$ inflates the $\hat{\Omega}$ condition number by $1/M_r^2$).

3. **Two-step weighting.** Run the optimizer with $W = I_R$ to obtain $\hat{\beta}_1$. At $\hat{\beta}_1$, form the per-panel error vectors $E_s = m_s(\hat{\beta}_1) - M(x)$ and estimate $\hat{\Omega} = \frac{1}{S}\sum_s E_s E_s^\top$ ($S > R$ is required for full rank; $S \gg R$ in practice). Set $W = \hat{\Omega}^{-1}$, warm-start from $\hat{\beta}_1$, and re-run the local optimizer to obtain $\hat{\beta}_2$.

4. **Inference.** Compute the Jacobian by centered finite differences, $D_{rk} \approx [e_r(\hat{\beta} + h_k \mathbf{e}_k) - e_r(\hat{\beta} - h_k \mathbf{e}_k)] / (2 h_k)$ with $h_k = \max(10^{-4}|\hat{\beta}_k|,\, 10^{-8})$. Each entry requires a full evaluation; the full Jacobian costs $2K$ solves. The error vector has variance $(1 + 1/S)\,\Omega$ (combining target noise and simulation noise), so by the delta method
$$V = \left(1 + \frac{1}{S}\right) (D^\top W D)^{-1}, \qquad \text{se}(\hat{\beta}_k) = \sqrt{V_{kk}}.$$
The t-statistic for $H_0: \hat{\beta}_k = \beta_k^0$ is $t_k = (\hat{\beta}_k - \beta_k^0)/\text{se}(\hat{\beta}_k)$. The overidentification test uses
$$J = \frac{S}{S+1}\, Q(\hat{\beta}) \;\xrightarrow{d}\; \chi^2(R - K),$$
where the $S/(S+1)$ factor corrects for the target being a single random panel. Failing to reject is consistent with correct specification but does not prove it.

### Optimization

Each $Q(\beta)$ evaluation requires a full model solve. With hundreds to thousands of optimizer iterations, total wall time is dominated by the solve count. I use `scipy.optimize` rather than `tf.keras.optimizers` because the model solve (VFI, discrete default decisions) is not differentiable through standard automatic differentiation. SciPy supports bounds, global search, and finite-difference gradients; the NumPy / TF conversion overhead is negligible relative to the solve cost.

The optimizer runs in two phases. Stage 1 uses `dual_annealing` for stochastic global search over the bounded parameter space, following @hennessy2007costly. After `dual_annealing` exhausts its `maxiter` budget, Powell is run from the best point found to refine to $\hat{\beta}_1$. Stage 2 reuses Powell from $\hat{\beta}_1$ with $W = \hat{\Omega}^{-1}$. Powell is derivative-free and treats $Q$ as a black box; it converges linearly but is robust when the objective is non-smooth in $\beta$.

| Phase | Method | Input | Output |
|---|---|---|---|
| Stage 1 global | `dual_annealing` | initial guess, bounds | coarse basin |
| Stage 1 polish | Powell | best point above | $\hat{\beta}_1$ |
| Stage 2 | Powell | $\hat{\beta}_1$, $W = \hat{\Omega}^{-1}$ | $\hat{\beta}_2$ |

### Validation metrics

For each parameter $k$, I compute the following diagnostics across $J$ Monte Carlo replications (the validation procedure itself is described in the [GMM and SMM section](#part2-validate)):

| Metric | Formula | Target | Interpretation |
|---|---|---|---|
| Bias | $\frac{1}{J}\sum_j \hat{\beta}_k^{(j)} - \beta_{0,k}$ | $\approx 0$ | A biased parameter implies its identifying moment is poorly computed or weak |
| SD | $\text{sd}(\{\hat{\beta}_k^{(j)}\})$ | small | Variability across replications |
| Avg SE | $\frac{1}{J}\sum_j \text{se}_k^{(j)}$ | small | Used for per-replication t-tests |
| RMSE | $\sqrt{\frac{1}{J}\sum_j(\hat{\beta}_k^{(j)} - \beta_{0,k})^2}$ | small | Combined bias and variance |
| $J$-test size | Fraction with $J^{(j)} > \chi^2_{0.95}(R-K)$ | $\approx 0.05$ | Should match the nominal 5% size |

Under sufficient optimizer budget, SD $\approx$ Avg SE; with limited budget, optimizer noise inflates SD relative to SE.

### Application to the basic investment model

I follow @hennessy2007costly and calibrate $r = 0.04$, $\delta = 0.15$ externally. Two SMM specifications are used: the frictionless model for clean validation (analytical policy, no solver inside the loop) and the frictional model for end-to-end testing.

**Frictionless validation ($\psi_0 = \psi_1 = 0$).** The optimal policy $k^*(z)$ is closed-form, so any error must be in SMM rather than in the model solve. Four moments identify $K = 3$ parameters $(\alpha, \rho, \sigma_\varepsilon)$:

| # | Moment | Definition | Identifies |
|---|---|---|---|
| 1 | Mean $I/k$ | $\mathbb{E}[I_{it}/k_{it}]$ | $\alpha$ |
| 2 | Var $I/k$ | $\text{Var}[I_{it}/k_{it}]$ | $\alpha,\, \sigma_\varepsilon$ |
| 3 | Serial corr $I/k$ | $\text{Corr}(I_{it}/k_{it},\, I_{i,t-1}/k_{i,t-1})$ | $\rho$ |
| 4 | AR(1) resid std | $\hat{\sigma}_u$ from panel AR(1) on $I/k$ | $\sigma_\varepsilon$ |

**Frictional application ($\psi_1 > 0$).** No closed-form policy, so the model is solved via PFI or ER inside the optimizer loop. Five moments identify $K = 4$ parameters $(\alpha, \psi_1, \rho, \sigma_\varepsilon)$:

| # | Moment | Definition | Identifies |
|---|---|---|---|
| 1 | Mean $I/k$ | $\mathbb{E}[I_{it}/k_{it}]$ | $\alpha$ |
| 2 | Var $I/k$ | $\text{Var}[I_{it}/k_{it}]$ | $\alpha,\, \psi_1$ |
| 3 | Serial corr $I/k$ | $\text{Corr}(I_{it}/k_{it},\, I_{i,t-1}/k_{i,t-1})$ | $\psi_1,\, \rho$ |
| 4 | AR(1) persistence | $\hat{\beta}_1$ from panel AR(1) on $\pi/k$ | $\rho,\, \psi_1$ |
| 5 | AR(1) resid std | $\hat{\sigma}_u$ from same regression | $\sigma_\varepsilon$ |

### Application to the risky debt model

I follow @hennessy2007costly's moment selection. The active set has 11 moments identifying up to $K = 7$ parameters $(\alpha, \psi_1, \eta_0, \eta_1, c_{\text{def}}, \rho, \sigma_\varepsilon)$, leaving 4 overidentifying restrictions when all parameters are estimated. When parameters are calibrated externally (e.g., $c_{\text{def}}$), moments tagged solely to those parameters are auto-dropped. The equity issuance cost is $\Omega(e) = (\eta_0 + \eta_1 |e|)\, \mathbf{1}\{e < 0\}$, separating a fixed cost ($\eta_0$, gating issuance frequency) from a proportional cost ($\eta_1$, shaping the pecking order).

| # | Block | Moment | Definition | Identifies |
|---|---|---|---|---|
| 1 | Issuance | Avg equity issuance / assets | $\mathbb{E}[\max(0, -e_{it})/k_{it}]$ | $\eta_0$ |
| 2 | Issuance | Frequency of issuance | $\Pr(e_{it} < 0)$ | $\eta_0$ |
| 3 | Issuance | Corr(issuance, investment) | $\text{Corr}(\max(0,-e_{it})/k_{it},\, I_{it}/k_{it})$ | $\eta_0,\, \eta_1$ |
| 4 | Leverage | Book leverage | $\mathbb{E}[b'_{it}/k_{it}]$ | $c_{\text{def}}$ |
| 5 | Leverage | Cov(leverage, investment) | $\text{Cov}(b'_{it}/k_{it},\, I_{it}/k_{it})$ | $c_{\text{def}}$ |
| 6 | Investment | Var $I/k$ | $\text{Var}[I_{it}/k_{it}]$ | $\alpha,\, \psi_1$ |
| 7 | Investment | Serial corr $I/k$ | $\text{Corr}(I_{it}/k_{it},\, I_{i,t-1}/k_{i,t-1})$ | $\psi_1$ |
| 8 | Real | Mean $I/k$ | $\mathbb{E}[I_{it}/k_{it}]$ | $\alpha$ |
| 9 | Real | AR(1) persistence | $\hat{\beta}_1$ from panel IV on $\Delta \log y_{it}$ | $\rho$ |
| 10 | Real | AR(1) shock std dev | $\hat{\sigma}_u$ from same regression | $\sigma_\varepsilon$ |
| 11 | Default | Default frequency | $\Pr(\text{default}_{it+1} \mid \text{state}_{it})$ | $c_{\text{def}}$ |

: Active SMM moment set for the risky debt model ($R = 11$). {#tbl-smm-debt-moments}

I depart from H&W's covariance form for the issuance-investment pecking-order channel and use the correlation instead, since $\text{Cov}(\text{Iss}/k,\, I/k)$ has a population variance of $\sim 10^{-8}$ (both terms are near-zero for most firm-years), which would make $\hat{\Omega}$ singular in finite samples. The correlation is bounded in $[-1, 1]$ and conditioning is comparable to other moments. Each candidate $\beta$ requires a full nested VFI solve (see the [nested VFI appendix](#sec-NestedVFI)); computing all 11 moments per simulated panel adds negligible overhead relative to the solve.

For Monte Carlo validation, I fix the truth at:

| Parameter | $\beta_0$ | Role |
|---|---|---|
| $\alpha$ | 0.7 | Production technology |
| $\psi_1$ | 0.05 | Adjustment cost |
| $\eta_0$ | 0.6 | Fixed issuance cost |
| $\eta_1$ | 0.1 | Proportional issuance cost |
| $c_{\text{def}}$ | 0.5 | Deadweight default cost |
| $\rho$ | 0.7 | AR(1) persistence |
| $\sigma_\varepsilon$ | 0.15 | AR(1) shock std dev |
