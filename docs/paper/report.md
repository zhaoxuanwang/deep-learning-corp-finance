---
title: "Quantitative Methods for Structural Corporate Finance Models"
author: Zhaoxuan Wang
number-sections: true
top-level-division: chapter
colorlinks: true
date: 2026-05-29
bibliography: ../references.bib
thanks:  Email me at [wxuan.econ@gmail.com](mailto:wxuan.econ@gmail.com)
format:
  html:
    toc: false
  pdf:
    toc: true
    toc-depth: 2
    pdf-engine: xelatex
    documentclass: book
    classoption:
      - oneside
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
        \usepackage{newunicodechar}
        \newunicodechar{μ}{\ensuremath{\mu}}
        \newunicodechar{η}{\ensuremath{\eta}}
        \newunicodechar{ξ}{\ensuremath{\xi}}
        \newunicodechar{σ}{\ensuremath{\sigma}}
        \newunicodechar{α}{\ensuremath{\alpha}}
        \newunicodechar{δ}{\ensuremath{\delta}}
        \providecommand{\gt}{>}
        \providecommand{\lt}{<}
        \usepackage{iftex}
        \ifPDFTeX\else
        \setmainfont{LibertinusSerif}[Extension=.otf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]
        \setmonofont{Inconsolatazi4}[Extension=.otf,UprightFont=*-Regular,BoldFont=*-Bold,Scale=MatchLowercase]
        \setmathfont{LibertinusMath-Regular.otf}
        \fi
        \definecolor{linknavy}{HTML}{1A3E6E}
        \definecolor{codebg}{HTML}{F2F2F2}
        \AtBeginDocument{\hypersetup{colorlinks=true,allcolors=linknavy}}
        \let\oldtexttt\texttt
        \renewcommand{\texttt}[1]{{\setlength{\fboxsep}{1pt}\colorbox{codebg}{\oldtexttt{#1}}}}
        \definecolor{shadecolor}{HTML}{F2F2F2}
        \makeatletter
        \long\def\blthanks#1{\begingroup\renewcommand\thefootnote{}\footnotetext{#1}\endgroup}
        \renewcommand\thanks[1]{\protected@xdef\@thanks{\@thanks\protect\blthanks{#1}}}
        \makeatother
---

# Introduction {.unnumbered}

This report collects my ongoing work on solving and estimating structural corporate finance models. The main objective is to implement and validate a set of solution methods, and test them on different baseline model variants in finance. To reproduce the results, the full codebase can be found at: [https://github.com/zhaoxuanwang/deep-learning-corp-finance](https://github.com/zhaoxuanwang/deep-learning-corp-finance). Below I briefly summarize the coverage, innovation, and limitations of the current project.

**Methods for solving dynamic optimization problems**:

1. Deep learning methods by @maliar2021: lifetime reward maximization, Euler residual minimization, and Bellman residual minimization.
2. Short horizon actor critic (SHAC): a modified version of @xu2022
3. Value function iteration (VFI) and Howard's policy function iteration (PFI)
4. Linear programming (LP) method [@nikolov2021]
5. Finite-difference with policy iteration [@marinovic2019ceo]

**Methods for estimating structural parameters**:

1. Generalized method of moment (GMM) 
2. Simulated method of moment (SMM)
3. Bayesian inference with MCMC sampler and filtering method
4. Indirect inference with auxiliary model

**Application to models in corporate finance**:

- Basic model of optimal investment
- Risky debt model [@hennessy2007costly, @strebulaev2012]
- Trade-off model with risky debt [@nikolov2021, section 2.2]
  - Estimation uses *real data* of HK listed firms (1999-2024) from Compustat Global
- Limited enforcement model with state-contingent securities [@nikolov2021, section 2.3]
- Moral hazard model [@nikolov2021, section 2.4]
- Optimal dynamic compensation contract for CEO [@marinovic2019ceo, @cronqvist2024]

**Neural network (NN) surrogate approach**: I made an engineering improvement that seems promising for future applications. The existing estimation pipeline is a costly loop of 

1. Fix parameter $\beta$, solve model using numerical or NN-based methods;
2. Use solved policy to rollout simulated data at $\beta$, run SMM or MCMC with filtering
3. Propose new $\beta'$, repeat from 1 until global optimizer find $\beta^*$ or posterior $p(\beta^*)$ 

This pipeline typically needs from hundreds to ten thousands of model solve at different $\beta$, which is intractable when a single model solve is moderately costly (e.g., more than 10 minutes).

Instead, my surrogate approach tries to pre-train an optimal policy (either NN or numerical) over the entire state-by-parameter space. Specifically, the extended policy function is $\varphi(S,\beta)$ instead of the standard $\varphi(S)$ over just state space. The gain is huge because we only need to pay the upfront model solve cost *once*, then passing the cached $\varphi(S,\beta)$ for the estimation loop at different $\beta'$ becomes much cheaper. Recent papers have explored similar "surrogate" idea to replace the expensive model solve part inside algorithms. Closely related examples of this approach include @chen2026deep for SMM estimation in finance; and @kase2022estimating for Bayesian estimation of macro HANK model.


**Limitations and next steps**:

- For algorithms/models that are computationally costly, I can only validate them using small grid density or training budget, so their results are rough and imprecise. Future work need to improve algorithm effiency or use better GPU/CPU
- The SHAC method is only validated with basic model, need to be tested on more complex models
- The "neural network surrogate" pipeline is only tested on the basic model and with Random-walk Metropolis-Hastings + Kalman filter. Need to extend applications to more complex models and test it on SMM or other Bayesian methods
- I use real-world data of Hong Kong listed firms to estimate the Trade-off model in @nikolov2021. I'm not able to estimate the remaining two models in @nikolov2021 because the model solve is too costly
- I numerically solved the CEO-contracting model in bonus part 2 but am not able to estimate it due to lack of data on CEOs. I created a concrete plan that can be executed once data is available

**Commercial product**: I consider two broad potential products that could be built on this project. This is summarized in the table below. These are still very preliminary ideas and need to be refined in the future with professional guidance.

| Product | Buyer | Output | Market / Scenario |
|---|---|---|---|
| **Corporate decision support** | Corporate Board, CFOs, or internal JPM advisory teams | Optimal capital structure, leverage, contract, and other decisions; Counterfactual analysis. | Banker advisory (JPM) and consulting services. Evaluate counterfactual outcomes and optimal decisions (e.g., leverage) across scenarios; Design optimal CEO compensation contract  |
| **Buy-side analytics** | Hedge funds and asset managers | Forecasts of leverage, investment, credit spread, default probability, etc. across firms. | Moody's EDF-X, Bloomberg DRSK, S&P Credit Analytics, and Fitch Connect predict default risk. Competitive edge: better forecasting with structural model estimation |


**Last notes**: Each chapter is self-contained and aims to give a high-level summary of the methods and the results. The detailed algorithms and implementations are provided in appendix. Math notation may not be consistent across chapters because it was intended to align with the original referenced paper. Chapter 5 uses a sample of public listed firms in Hong Kong from 1999-2024 accessed via Compustat Global database subscribed by UBC library. All other chapters use simulated data. Most computation uses a Apple M1 CPU on Macbook Pro (2020).



# Deep Learning for solving dynamic models

This chapter examines the validity and practical performance of the three main deep learning methods proposed by @maliar2021 and reviewed in @fernandez-villaverde2025 for solving dynamic programming problems. Compared with traditional numerical methods like VFI and PFI, the main justification for deep learning methods is that neural network approximation functions can scale up efficiently for high-dimensional models where VFI/PFI become intractable due to the "curse of dimensionality".


However, unlike VFI/PFI and other finite-grid methods, there is no mathematical theorem (e.g., contraction mapping) that guarantees the neural network (NN)-based algorithm can converge to the unique fixed point. Moreover, although the *theoretical* efficiency gain of using NN as functional approximator in high-dimensional space is clear [@fernandez-villaverde2024], the *practical* speed of NN-based methods when applied to economic models has not been well-documented. 

To conduct a rigorous comparison of the accuracy across methods, I apply them to the same low-dimensional optimal investment model and benchmarking the @maliar2021 methods against closed-form solution. When closed-form solution does not exist, I benchmark them against VFI solutions. 

Unlike the original algorithms in @maliar2021 that take Monte Carlo draws on-the-fly to simulate the training data and use the objective loss as convergence criteria, I strictly separate simulation from training and use a set of metrics evaluated on held-out validation set as convergence criteria. This ensures that different methods are applied to the same fixed training dataset and evaluated on a set of better validation metrics that are not affected by potential over-fitting.

I have two main findings after testing the three methods in @maliar2021:

- Lifetime reward maximization and Bellman residual minimization methods both have critical defects in algorithm design that cause systematic bias in solution. They are rejected for production use.
- Euler residual minimization method is accurate and fast, but it requires a closed-form Euler equation that can be written analytically in terms of observables. This is infeasible for most models.

This means that, *in practice*, none of the three methods can reliably solve high-dimensional models with no closed-form Euler equation up to good precision.

To address this gap, I make a methodological innovation by introducing a new solution method: Short Horizon Actor Critic (SHAC). This is a modified algorithm based on a canonical method in model-based reinforcement learning (RL) by @xu2022. I tested and validated SHAC and show that it achieves similar precision to VFI/PFI and can be scaled efficiently to higher dimensional space (e.g., $\gt 10$).

In addition, I tested and documented that the canonical Q-learning methods with deterministic policy in RL are not directly applicable to economic and finance model environments. The main root cause is that the value function in most economic and finance models incorporates long-term accumulating signals to current actions, for example, current investment compounds slowly to the value function. Most Q-learning methods struggle to learn this accurately because they rely on a bootstrap estimate of the value function from short-term rewards, and thus for example systematically "under-estimate" the return to current investment. Another root cause is that the reward landscape of most economic models are usually heterogeneous and steep, so that Q-learning method often gets stuck in a degenerate equilibrium.

These findings also reveal why SHAC worked: it rolls out a long-enough horizon (e.g., $t=32$) instead of a one-period bootstrap $t=1$, so that the value function estimates can "see" the long-term rewards to current actions.


Note that all methods here are designed for discrete-time, continuous control problems. For continuous-time counterpart problems, @duarte2024 introduces a "deep policy iteration" method that shares the same actor-critic structure as SHAC, and illustrates its usage in quantitative finance. The idea of using short-horizon rollout to bound the bootstrap error can be adopted to enhance the stability of the @duarte2024 method for continuous time problems.

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

The conditional expectation is given by the log-normal mean in closed form: no numerical quadrature is needed, because $\log (z' | z)$ is exactly normal with mean $(1-\rho)\mu + \rho \log z$ and variance $\sigma^2$. This makes $k^*(z)$ an exact, parameter-free anchor that I use repeatedly throughout the paper: 

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
Production $\Pi(k,z) = z k^{\beta}$ and adjustment cost $\Psi(I,k)$ take the same form as in the [basic model](#model-basic).

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


## Summary of Deep Learning Methods

The solution to the model is given by the optimal policy function $\pi^*(s)$ that maps states to actions. Optionally, solution also include the optimal state-value function $V^*(s)$. This section provides a high-level, brief summary of the solution methods using generic notations. I discuss the key ideas of each method, their main strength and limitations, and improvements. A more detailed and comprehensive documentation of each method (algorithm) are provided in [Appendix.A](#sec-solve).

I implemented five main solution methods in Python and Tensorflow:

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

1. The empirical loss is the squared Euler residual $\frac{1}{|B|}\sum_{i \in B} F(s_i, a_i, s'_{i,1}, a'_{i,1}) \cdot F(s_i, a_i, s'_{i,2}, a'_{i,2})$ using two i.i.d. draws $s'_{i,1}$ and $s'_{i,2}$, which is an unbiased estimator for $\mathbb{E}_\varepsilon \left[ F(\cdot)^2 \right]$.
2. Both @maliar2021 and @fernandez-villaverde2025 use a single policy network $\pi_\theta$ inside the Euler residual function $F$. In practice, this creates a
moving-target problem that prevents convergence: the gradient of $\theta$ flows through both the current policy $\pi_\theta(s)$ and the next-period policy $\pi_\theta(s')$, which appear on both side of the Euler equation. To fix this, I use a separate target network $\pi_{\theta^-}$ for the next-period action, which stabilizes training and facilitates convergence.

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

where the terminal value is truncated and implicitly set to $\mathbb{E}\left[V^{\pi}(s_T)\right]=0$. This truncation is only valid when the finite-horizon rollout $T$ is sufficiently large. For example, with discount factor $\gamma=0.95$ and $T=100$, the terminal value contribution of $0.6\% \approx (0.95)^{100}$ is negligible. In practice, the **LRM method** faces an important trade-off between bias and computational cost: If we set large $T$ rollout, the computational cost of BPTT is huge and LRM is practically much slower than any other methods (including VFI/PFI). BPTT is sequential so it cannot be parallelized by Tensorflow. In contrast, LRM is feasible when rollout is moderate (e.g., $T\leq 30$) but the truncation bias would be large enough to systematically bias the solution $\pi_{\theta^*}$.



I also implemented and tested the **Bellman residual minimization (BRM) method** following @maliar2021. However, I find that this method is very unstable and may converge to a spurious, self-consistent fixed point different from the optimal policy. @maliar2021 concludes that the main defect of BRM is it is less precise and requires careful tuning of hyperparameters to match the scale of the Bellman equation residual and the first order condition (FOC) residual. I show that the defect is structural and cannot be solved by fine tuning and warm start (pre-training). In short, the Bellman residual can be minimized for any arbitrary policy and only the first-order necessary conditions are providing useful gradient directions. The BRM method converges only when FOC dominates the Bellman residual in training and when the initiation of the value function network is around the "right" basin of the local optimum. This makes it dependent on pre-training and fine-tuning, and less useful in practice. Therefore I remove BRM from the main production methods and discuss the more fundamental defects in the [BRM appendix](#sec-BRM).

Finally, I introduced and implemented a new method, **Short-Horizon Actor Critic (SHAC)**, based on a revision of the reinforcement learning (RL) algorithm developed by @xu2022. This method requires four neural networks: a policy network parameterized by $\pi_\theta$ and a value function network parametrized by $V_\phi$, and two polyak updated copies $\bar \pi_\theta$ and $\bar V_\phi$.

The basic design of SHAC is that we slice the full $T$ horizon into $T/H$ windows, each with length $H$, then exploit the $H$-step rollout and BPTT to accurately train $\pi_\theta$ and $V_\phi$ within each window. Each gradient update consists of two steps: For each window $j=0,\dots,T/H$, the actor step updates the policy network $\pi_\theta$ to maximize 

$$\max_{\theta} V_\phi(s_j) = \mathbb{E}\left[\sum_{t=j}^{H-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right] + \gamma^H \, \mathbb{E}\left[\bar V_\phi(s_H)\right]$$

then the critic step is supervised learning of the value network:

$$ \min_{\phi} \mathbb{E}\left[ V_\phi - y \right]^2$$

where the target label $y=\text{Stop Gradient}{(\sum r^H+\gamma^H \bar V_\phi)}$ is the Bellman right-hand-side value from the actor step. The intuition underlying actor critic method is very similar to VFI/PFI, where the policy evaluation find the policy that maximizes the Bellman and the tabular value function is updated using the improved policy. 

The advantages of **SHAC** method are clear:

- it does not requires the existence of closed-form Euler equation
- it introduces a value network to precisely learned the terminal value omitted by LRM
- it uses short-horizon rollout and BPTT to reduce bootstrap error for the value network

I show that SHAC and ERM achieve significantly better accuracy and robustness compared with LRM and BRM. Unlike ERM, SHAC does not require closed-form Euler equation and thus can be applied to more general set of models. The main cost is the computational expense of BPTT, but it can be fine-tuned to achieve a balance between stability (longer $T$) and speed (shorter $T$). 

## Implementation Details

### Synthetic Data
Standard deep learning applications typically use three datasets:

1. Training data: split into mini-batches to train the NNs via SGD
2. Validation data: used to evaluate the quality of solution and convergence criteria
3. Test data: sealed and only used once to benchmark the results after the entire training is finished

I adopted a similar approach when training the NNs. The key design principles are:

- All different methods are trained on the same fixed training dataset
- Convergence/early stopping criteria are evaluated on a separate validation dataset

Unlike @maliar2021 who simulated data on-the-fly during NN training, my approach **strictly separates data generation and NN training**. Because different methods are applied to the exact same datasets, their results can be fully reproduced, compared and benchmarked. Any discrepancies in results must be due to the effectiveness of solution methods rather than potential randomness in data simulation.

I simulated datasets in two general structures (with $i$ denotes observation):
1. Full trajectory data: $\{(s_{it}, a_{i,t}, s_{1,i,t+1}, s_{2,i,t+1})\}_{t=0}^{T-1} \equiv \{\big(s_{i,t}, \pi_\theta(s_{i,t}), f(s_i,\pi(s_i), \epsilon_{1,it}), f(s_i,\pi(s_i), \epsilon_{2,it}) \big)\}^{T-1}_{t=0}$
2. One-period transition data: $\big( s_i, a_i, s'_{1i}, s'_{2i} \big) \equiv \big( s_{i}, \pi_\theta(s_{i}), f(s_i, \pi_\theta(s_i),\epsilon_{1,i}), f(s_i, \pi_\theta(s_i),\epsilon_{2,i}) \big)$

where the one-period transition data is flattened and randomly shuffled from the full trajectory data. Full trajectory data is used by LRM and SHAC methods, and the one-period transition data is used by VFI/PFI, ERM and BRM methods. This design ensures that all these six methods are trained on the same data points even if their required data structure is different.

Note that for each period $t$, I take two iid draws $(\epsilon_{1,it}, \epsilon_{2,it})$ which is necessary to construct the unbiased estimator for the loss function in ERM and BRM methods. Another point is that the action $a_{it}$ and next-period states $s_{1,i,t+1}$ depends on the current policy $\pi_\theta$, so they must be generated during training. In practice, I separate exogenous states (e.g., AR(1) productivity shocks) that does not depend on $\theta$ from endogenous states (e.g., capital stock $k$) that depends on action $\pi_\theta$ (e.g., investment). Trajectory of exogenous states can be fully unrolled before training, and endogenous states are on-policy rollouts during training.

**Application:** To apply to both the basic investment model and the risky debt model, the data generation is (suppressing $i$ index):
1. Build bounded state spaces $\mathcal{S} = [\underline z, \bar z] \times [\underline k, \bar k] \times [\underline b, \bar b]$ and action spaces $\mathcal{A} = [\underline I, \bar I] \times [\underline b, \bar b]$ from model environment
2. Sample initial states $z_0,\, k_0,\, b_0 \sim \text{Uniform}(\mathcal{S})$ at $t=0$
3. For $t=0,\dots,T-1$, sample $M$ independent shock sequences $\{\varepsilon_{1,t},\dots,\varepsilon_{M,t}\}$ from $\mathcal{N}(0,1)$
4. Separate exogenous states $z_t$ and endogenous states $(k_t,b_t)$, start from $z_0$ and unroll the full trajectories of $\{ z_{1,t},\dots,z_{M,t} \}^{T-1}_{t=1}$ using the state transition function $\log z_{t+1}= \mu + \rho \log z_t + \sigma \varepsilon_{m,t}$ for $m=1,\dots,M$.
5. Store the full trajectory data:
$$\mathcal{D}^{\text{traj}} = \left( k_0, b_0, z_{0}, \{z_{1,t}\}_{t=1}^{T-1}, \dots, \{z_{M,t}\}_{t=1}^{T-1} \right)$$
6. Take the full trajectory data, flatten the exogenou states to only keep one-step sample $\left(z_{m} , z'_{m} \right)$ between $t$ and $t+1$ for a given $m=1,\dots,M$. Sample a new current-period endogenous state $k,\, b \sim \text{Uniform}(\mathcal{S})$, merge them, drop the $t$ subscript and use $'$ to denote next-period variable. Randomly permutate the data to break the serial correlation (ordering) and store it as the one-step transition data: 
$$\mathcal{D}^{\text{flat}} = (k, b, z, \{z'_{m}\}^M_{m=1})$$

### Reproducibility

All randomness flows from a single **master seed** pair $(m_0, m_1)$, which seeds three independent stateless-RNG streams: data generation, NN initialization, and SGD mini-batch ordering. This delivers two benefits: **reproducibility** (the same master seed gives a bit-identical experiment on the same machine) and **common random numbers** (different methods trained at the same step see the same data, so cross-method comparisons are paired and free of Monte-Carlo noise). The full seed schedule is in [the implementation appendix](#sec-impl-seeds).

### Validation of Solution {#part1-validate}

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
- **LRM obtains a close approximation but does not converge.** This is a structural bias due to terminal value truncation, as discussed below.

The most interesting case is LRM. Its policy MAE drops quickly in the early phase and then plateaus just above the threshold without ever crossing it. This is not a training artifact such as insufficient steps, a poor learning rate, or a small batch size. It is the structural defect of LRM described in [the LRM appendix](#sec-LRM). Even with the geometric-perpetuity terminal value correction, **LRM cannot recover the true on-policy continuation value up to precision** $V^\pi(s_T)$. The perpetuity can at-best approximate the stochastic future value with a deterministic steady state and ignores the firm's optimal response to future shocks. This leaves an $O(\sigma_\varepsilon^2)$ approximation error that is small but does not vanish with longer training. LRM therefore always has an approximation bias, regardless of training budget and sample size.

### Learned Policy vs True Policy

@fig-policy provides a visual validation of the solutions. It plots two slices of the next-period capital policy $k'(k, z)$ for each method, against the closed-form analytical anchor $k^*(z) = [\beta\, \mathbb{E}[z'\mid z]/(\bar r + \delta)]^{1/(1-\beta)}$:

- **Left panel: $k'$ as a function of $z$, holding $k$ fixed.** The analytical anchor depends only on $z$ and rises monotonically with $z$, so the plotted curves are upward sloping.
- **Right panel: $k'$ as a function of $k$, holding $z$ fixed.** Because $k^*$ is independent of $k$, the analytical anchor is a horizontal line. A correctly learned policy is also flat in this panel.

![Learned policy against true policy by methods](figures/part1-basic/selected_checkpoints_overlay.png){#fig-policy}

The black dashed line is the analytical solution. The red dotted line is the PFI solution, which serves as the discrete-grid anchor. PFI, ERM, and SHAC all match the analytical solution to within the convergence threshold of policy MAE $\leq 2$ established in @fig-convergence, and their learned policies tightly coincide with the true policy.

The only exception is LRM. In the left panel, **LRM systematically underestimates $k'$ when $z$ is high**. This is exactly the terminal-value truncation error described in [the LRM appendix](#sec-LRM). It is worth emphasizing that this gap remains nontrivial after the terminal value approximation is already implemented by adding $\hat{V}^{\text{term}}(s_T^{\text{endo}})$ to the LRM objective function. Without this correction, the downward bias would be much larger.

The intuition is that the perpetuity pins the exogenous state at its stationary mean $\bar z$ instead of the realized $z_T$. Under AR(1) persistence a high-$z$ trajectory typically ends at high $z_T$, so the perpetuity underestimates the true continuation $V^\pi(s_T)$, the actor underweights the long-run benefit of investing at high $z$, and the learned $k'$ is pulled inward. **This bias cannot be reduced by more training, data, or a smaller learning rate**, because it lives in the analytic terminal correction itself; the only fix is to replace the perpetuity with a learned value network, which is exactly what SHAC does.

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


### Issues in Neural Network Architecture and Training {#part1-issues}

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
| Input Normalization | Raw data are measured in level and large units can easily de-stabilize training  | Normalize input to z-score and re-scale it back to economic levels as NN output head | Hidden layer only see normalized inputs and is agnostic to environment
| Network output head | Sigmoid, Tanh, and other activation can suppress gradient and prevent learning at extreme values | Use raw linear (no activation function) with affine transformation | Gradient is uniformly "strong" across state/action space; Output variable converted back to original unit
| Hidden layer activation | For economic models, ReLU is not stable, Sigmoid and Tanh cause vanishing gradient| SiLU `swish` always perform better in practice | Gradient is stable and nonzero
| Full reproducibility | Comparison across methods should be fair and fully reproducible | Separate data generation and training; Full schedule of random number generator (RNG)| All methods are trained on exactly the same fixed dataset and results are fully controlled by master seeds
| Convergence metric | Objective and loss function are NOT the correct metrics for the quality of solution | Measure effectiveness of learned policy in a separate validation dataset | Avoid overfitting, enable early stopping based on same criteria, fair comparison across methods

The architecture-level choices in this table are documented in detail in [Implementation Details](#sec-impl): input normalization, hidden-layer activation, output head transformation.

## Results: Risky Debt Model

This section can be reproduced by running `docs/03_risky_debt_vfi_interp.ipynb`.

### Solution method: Nested VFI

To solve the risky debt model described in @strebulaev2012 [section 3.6], I implemented a **nested VFI algorithm**. I find that nested VFI is still the best method for this model in terms of speed, robustness, and accuracy. In contrast, I find that all three of @maliar2021's methods **cannot be applied to solve the risky debt model** because the model has a nested fixed-point structure: the firm's value function $V$ depends on the bond's risky rate $\tilde r$, and $\tilde r$ depends on $V$ because the lender prices the bond using the default states implied by $V$.

I solve the model with a two-level VFI iteration. The **inner loop** is a standard VFI that solves $V$ on the discrete grid for a fixed pricing schedule. The **outer loop** updates $\tilde r$ to be consistent with the default partition implied by the latest $V$. The algorithm terminates when both loops converge, so the value function and the pricing schedule are mutually consistent. 

On top of the standard algorithm described in @strebulaev2012 and used by @hennessy2007costly, I developed two algorithm refinements. I argue that these two refinements significantly **improve the speed of nested VFI without sacrificing accuracy**. The saving in compute can be especially large for applications of the simulated method of moment (SMM) where the computational bottleneck is re-solving the model repeatedly over many optimization steps.

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
- When default risks are low (e.g., high $z,k'$ and/or high $b'$), the debt discount is close to risk-free rate $1/(1+r)\approx 0.96$ (dotted line)

In summary, the results are consistent with economic rationales of the risky debt model and both the inner and outer loop of VFI converged up to high precision (error $<10^{-6}$). I consider this as strong evidence of the correctness and effectiveness of the solution.

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

# Simulated Method of Moments Estimation {#part2-validate}

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

**Result 1: optimizer reaches an interior minimum.** If GMM is correctly implemented, the six moment conditions at $\hat\beta$ should be near zero, well below the sampling noise floor. @tbl-gmm-moment-fit confirms this: five of six conditions are below $10^{-3}$ and the largest residual (Shock $\times 1$) is $3.5 \times 10^{-3}$. The Stage-1 global search and Stage-2 Powell refinement together find an interior minimum.

| Moment              | $g(\hat\beta)$                    |
| ------------------- | --------------------------------- |
| Euler $\times 1$    | $-9.23 \times 10^{-4}$            |
| Euler $\times I/k$ lag    | $-1.70 \times 10^{-4}$            |
| Euler $\times \pi/k$ lag  | $-2.66 \times 10^{-4}$            |
| Shock $\times 1$    | $-3.55 \times 10^{-3}$            |
| Shock $\times \ln z$ lag  | $\phantom{-}6.87 \times 10^{-4}$  |
| Var $\times 1$      | $-3.71 \times 10^{-7}$            |

: Moment-condition vector $g(\hat\beta)$ on a single representative panel. {#tbl-gmm-moment-fit}

**Result 2: point estimate and sandwich SE are correctly calibrated on a single panel.** If they are, every estimate should be within roughly 1 SE of the truth and the parameter t-tests should fail to reject at 5%. @tbl-gmm-single-rep confirms both: every estimate is within 1 SE of $\beta^*$, all four parameter t-tests have $p > 0.10$ (the smallest is $\sigma$ at $p = 0.11$), and the over-identification J-test is also insignificant ($J = 4.11$, $p = 0.13$, df = 2). The single-panel pipeline behaves correctly.

| Parameter | True   | Estimate | SE       | $t$-stat | $p$-value |
| --------- | ------ | -------- | -------- | -------- | --------- |
| $\alpha$  | $0.60$ | $0.5999$ | $0.0006$ | $-0.14$  | $0.89$    |
| $\psi_1$  | $0.10$ | $0.0861$ | $0.0176$ | $-0.79$  | $0.43$    |
| $\rho$    | $0.70$ | $0.6817$ | $0.0154$ | $-1.18$  | $0.24$    |
| $\sigma$  | $0.15$ | $0.1466$ | $0.0021$ | $-1.60$  | $0.11$    |

: Parameter estimates, sandwich SEs, and t-tests against the true parameter. J-statistic = 4.11, $p$ = 0.13, df = 2. {#tbl-gmm-single-rep}

**Result 3: point estimate is unbiased on average and asymptotic SE is correctly calibrated.** This requires three things together: bias should be much smaller than the within-replication SE, the empirical SD across replications should match the average SE, and the J-test rejection rate at 5% nominal should be $\approx 0.05$. @tbl-gmm-mc partially confirms this. The point estimates are economically close to the truth (relative bias is below 4% for every parameter). However, the empirical SD across replications is systematically $1.3$ to $2\times$ the average within-replication SE, so the asymptotic sandwich formula understates the true sampling variability. This propagates into J-test over-rejection at $0.20$ vs the nominal $0.05$. Both failures are consistent with PFI grid approximation in the panel-generation step entering the Euler residuals as a small systematic bias.

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

**Result 1: optimizer reaches an interior minimum.** If SMM is correctly implemented, the simulated moments at $\hat\beta$ should match their data-side targets to within Monte-Carlo noise. @tbl-smm-moment-fit confirms this: every fitted moment is within $\sim 1\%$ of its target, so the Stage-1 global search and Stage-2 Powell refinement find an interior minimum.

| Moment              | Target    | Fitted    |
| ------------------- | --------- | --------- |
| Mean $I/k$          | $0.1927$  | $0.1916$  |
| Var $I/k$           | $0.0925$  | $0.0926$  |
| Serial corr $I/k$   | $-0.1427$ | $-0.1432$ |
| AR(1) resid std     | $0.2124$  | $0.2123$  |

: Moment fit at $\hat\beta$ on a single representative panel. {#tbl-smm-moment-fit}

**Result 2: point estimate and sandwich SE are correctly calibrated on a single panel.** If they are, every estimate should fall within roughly 1 SE of the truth and the parameter t-tests should fail to reject at 5%. @tbl-smm-single-rep confirms both: every estimate is within 0.2 SE of $\beta^*$ and all parameter t-tests have $p > 0.85$. The over-identification J-test is also insignificant ($J = 0.62$, $p = 0.43$, df = 1).

| Parameter | True   | Estimate | SE       | $t$-stat | $p$-value |
| --------- | ------ | -------- | -------- | -------- | --------- |
| $\alpha$  | $0.60$ | $0.6007$ | $0.0079$ | $0.09$   | $0.93$    |
| $\rho$    | $0.70$ | $0.7018$ | $0.0156$ | $0.12$   | $0.91$    |
| $\sigma$  | $0.15$ | $0.1502$ | $0.0010$ | $0.19$   | $0.85$    |

: Parameter estimates, sandwich SEs, and t-tests against the true parameter. J-statistic = 0.62, $p$ = 0.43, df = 1. {#tbl-smm-single-rep}

**Result 3: point estimate is unbiased on average and asymptotic SE is correctly calibrated.** This requires bias much smaller than the within-replication SE, empirical SD $\approx$ average SE, and J-test rejection rate at 5% nominal not exceeding $0.05$. @tbl-smm-mc confirms all three. Bias is more than an order of magnitude below SE for every parameter, the empirical SD agrees with the average SE within $\sim 30\%$ (consistent with the small MC sample of $J = 10$), and the J-test never rejects across 10 replications; $0/10$ is statistically consistent with any size at or below $0.05$.

| Parameter | True   | Mean estimate | Bias       | RMSE     | SD across MC | Avg SE   |
| --------- | ------ | ------------- | ---------- | -------- | ------------ | -------- |
| $\alpha$  | $0.60$ | $0.6003$      | $0.0003$   | $0.0084$ | $0.0089$     | $0.0085$ |
| $\rho$    | $0.70$ | $0.6969$      | $-0.0031$  | $0.0166$ | $0.0172$     | $0.0159$ |
| $\sigma$  | $0.15$ | $0.1505$      | $0.0005$   | $0.0015$ | $0.0016$     | $0.0012$ |

: Monte Carlo summary across $J = 10$ replications. J-test reject rate at 5% is 0.00. {#tbl-smm-mc}

**Summary.** All three predictions hold. The SMM machinery (moment construction, two-step weighting, optimizer, sandwich SE, J-test) is correctly implemented on a model whose ground truth is known. This confirms the correct implementation of the SMM pipeline except for the model solver (e.g., VFI) itself.


## Applying SMM to the risky debt model

Applying SMM to the basic investment model identifies four parameters: production-function curvature ($\alpha$), smooth adjustment cost ($\psi_1$), AR(1) persistence ($\rho$), and AR(1) shock variance ($\sigma$). Adding costly equity issuance from section 3.3 of @strebulaev2012 brings two more parameters into scope: the fixed and proportional cost components ($\eta_0$ and $\eta_1$). The full endogenous-default model adds one final parameter, the deadweight bankruptcy cost $c_{\text{def}}$, the fraction of firm value lost when the firm defaults. From a pure estimation perspective, the endogenous-default extension only adds one parameter; the rest can be estimated from the simpler frictional model in section 3.3.

The cost of applying SMM to risky debt model is computational: each candidate $\beta$ in the optimizer's inner loop requires a fresh nested-VFI solve on the discrete $(k, b, z)$ grid. A Monte-Carlo replication study at this scale is infeasible (on my current device), so I report a single representative run rather than MC summary statistics. The full SMM target is $\beta = (\alpha,\, \psi_1,\, \eta_0,\, \eta_1,\, c_{\text{def}},\, \rho,\, \sigma)$ with $K = 7$, matched against $R = 11$ moments following @hennessy2007costly's selection (see [Appendix B](#sec-smm-appendix)). The results can be reproduced from `docs/06_risky_debt_smm_workflow.ipynb`. It took about 40 hours to run the full SMM on my 2020 Macbook Pro (M1).

**Result 1: moment fit.** Fitted moments deviate noticeably from their targets. The conditional-issuance, AR(1)-shock-std, and variance-of-investment moments miss by 50%+ of their target value, indicating the optimizer cannot find a $\beta$ that matches all 11 moments simultaneously.

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

**Result 2: parameter estimates.** Five of seven t-tests reject $H_0:\, \hat\beta_k = \beta_k^*$ at the 5% level. The two that fail to reject ($c_{\text{def}}$, $\rho$) do so only because their standard errors are abnormally large.

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

Defect #1 is problematic because the present value of tax shield benefit, $\frac{\tau \tilde{r} b'}{(1+\tilde{r})(1+r)}$ is obtained by firm upfront and is unconditional on the next-period solvent/default states (Equation 3.26). Since the model intentionally does not impose a borrowing limit, firm could exploit an optimal "always-default" strategy that borrow as much as possible and default in next period. The lender rationalize this in pricing the interest rate $\tilde{r}\to \infty$, but the tax shield benefit is still large and positive: $$ \frac{\tau \tilde{r} b'}{(1+\tilde{r})(1+r)} \rightarrow \frac{\tau b'}{(1+r)} > 0 \quad \text{as} \quad \tilde{r}\to \infty \text{ and } b'\to \infty$$ 

This is confirmed empirically when any naive implementation with large upper bound $b_{\max}$ relative to the $k_{\max}$ will cause the optimal policy to converge to "borrow as much as possible then default" with $b'=b_{\max}$ and $k'\approx 0$. For model solve itself this can be mitigated with well-calibrated parameters and bounds, but the true risk is for SMM estimation when the optimizer re-solved the policy under different parameters and a non-trivial fraction of the parameter combinations will lead to this unintended strategy.

One simple solution is to use the same time schedule as in the trade-off model in @nikolov2021.

### Imperfect managerial information

Defect 2 is directly related to the critique by @deangelo2022: Can manager and lender precisely estimate the continuation value $V$ and default states given by $\{z: V(\cdot, z)\leq 0\}$? This is the key important assumption of the model: the endogenous default decision is a going-concern and manager would only default when the firm's continuation value is negative. The pricing of bonds is from a bargaining between the firm and the lender based on $\mathbb{E}_{z'|z} [V(k',b',z')]$ where there exist a critical value of shock $z'_d$ such that all $z'<z'_d$ are default states [@hennessy2007costly, Proposition 6]. However, if manager and lender cannot learn $V$ with any realistic precision, this core mechanism is broken.

@deangelo2022 [Section VI] reviews direct evidence of imperfect manager knowledge. There are two key takeaways. First, large-scale surveys of CFOs indicated that "most managers have nothing close to the knowledge assumed in extant dynamic capital structure models, which posit a complete understanding of investment opportunities and capital-market conditions over an infinite horizon" [@graham2022presidential]. Second, a number of studies have estimated a near-flat relationship between firm value and leverage, suggesting that "real-world managers are unable to pin down a uniquely optimal capital structure with any real precision" [@korteweg2010net].


# Bayesian Estimation

**(a) What are your prior assumptions on the parameters?**
A prior encodes belief about a parameter before seeing data. It is specified as a probability density function of the parameter, $p(\beta)$. Practical specification depends on domain-knowledge. I discuss the specific priors I choose for different parameters in later section.

**(b) What filtering method do you use, and what are its pros and cons vs alternatives?**
By default, I use Kalman Filter when the model can be cast as a Linear Gaussian state-space model (LGSSM). When the likelihood is differentiable but non-linear, I consider Extended Kalman Filter or Unscented Kalman Filter as alternatives. When the likelihood is non-differentiable, I consider the Particle Filter and Random Walk Metropolis-Hastings (RW-MH) as fallback. I discuss specific filtering choice later when applying to different models.

**(c) Which MCMC method (in TFP), and how is it chosen?** I built and tested two main options: (1) No-U-Turns (NUTS) Hamiltonian Monte Carlo with Extended Kalman Filter; (2) Random-Walk Metropolis-Hastings (RW-MH) with Extended Kalman Filter. NUTS are more efficient when gradient info $\nabla L(\beta)$ is available, RW-MH is used when gradient is not available or too computationally costly.

**(d) What tests assess the validity of the estimation method?**
For generic diagnostic metrics, I report split-$\hat{R}$ and effective sample size (ESS), as well as several method-specific metrics such as diverged transitions and leapfrog tree depth (for NUTS-HMC). For posterior analysis, I report visualization of the posterior marginals, summary statistics (median and mean), and trace plot. For more complete tests, I implement a minimal version of coverage check in the spirit of [Simulation-Based Calibration](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html), as well as posterior predictive checks (simulate data to match real data) and prior sensitivity analysis (robustness of result when varying prior).

**(e) How does Bayesian estimation compare to GMM and SMM?**
GMM matches model-implied moments (analytical functions of $\theta$) to sample moments. It requires the optimal policy function itself to be expressible analytically in $\theta$. In this project only the basic model without adjustment cost satisfies that. SMM replaces the analytical moments with simulated ones: one model solve per parameter evaluation, so any model that can be simulated is in scope. SMM gives a point estimate with asymptotic-sandwich standard errors, no full posterior; the analyst must pick the moments (a modeling choice that affects identification); and it is prone to weak identification when the chosen moments don't pin down $\theta$. 

Bayesian estimation uses the full likelihood (no moment selection), delivers the joint posterior distribution, and quantifies parameter uncertainty. It often worked better in identification of high-dimensional models compared with SMM. For models more complicated than the basic model, I introduce a Neural Surrogate approach to cut per-evaluation model solve cost for both Bayesian MCMC and SMM. This is done by pre-training a neural network (NN) to approximate the optimal policy function in state-by-parameter space. Thus, the choice between Bayesian and SMM reduces to a methodological one: full posterior characterization (Bayesian) versus a frequentist point estimate (SMM).

## Overview

### Target

Treat the parameter vector $\beta$ as a random variable. Given observed data $y$, the target is the posterior distribution

$$p(\beta \mid y) \;\propto\; \underbrace{p(\beta)}_{\text{prior}} \cdot \underbrace{p(y \mid \beta)}_{\text{likelihood}}.$$

Markov Chain Monte Carlo (MCMC) operates on the **log-target**

$$L(\beta) \;:=\; \log p(\beta) + \log p(y \mid \beta)$$

Since $\log p(y)$ does not depend on $\beta$, it cancels in every Metropolis-Hastings (MH) acceptance ratio and never needs to be computed. The pipeline reduces to two tasks: specify a prior, and evaluate the log-likelihood $\log p(y \mid \beta)$ at any candidate $\beta$.

### Model specification

The Bayesian pipeline has one upstream modeling choice that shapes everything downstream: Do we assume that observed actions (e.g., investment, leverage) are generated by the firms acting by optimal policy? If so, what structural form shall we impose on the residual gaps?

In principle, there are many ways to cast the structural model into a set of empirical specifications for Bayesian estimation. This specification choice largely depend on (1) the complexity of the model itself (e.g., risky debt); (2) the target parameters to be estimated (e.g., firm's deviation from optimal leverage); and (3) the additional assumptions we are willing to impose on top of the original theoretical model (e.g., Gaussian noise or systematic error).

To illustrate this concretely, I use the simple basic model of optimal investment as example. Consider that we observe a panel data of firms. The basic optimal investment model has a solution defined as an optimal policy function $\varphi$ mapping states to actions. 

**Observed data.** We observe a firm-year panel of capital, leverage, and revenue:
$$
\text{Panel data:} \quad (k_{it}, b_{it}, \Pi_{it}).
$$

**Structural model.** A vector $\beta \equiv (\alpha, \rho, \sigma_\epsilon)$ of structural parameters underlying the data-generating process (DGP):
$$
\begin{aligned}
\text{Cobb-Douglas production:} \quad \log \Pi_{it} &= \log z_{it} + \alpha \cdot \log k_{it} + \eta_{it} \\
\text{Latent AR(1) productivity:} \quad \log z_{it} &= \rho \log z_{i,t-1} + \sigma_\varepsilon \cdot \mathcal{N}(0, 1) \\
\text{Investment policy:} \quad \log k_{it} &= \varphi_k(z_{it}, k_{it}, b_{it} \mid \beta) + \xi^k_{it} \\
\text{Leverage policy:} \quad \log b_{it} &= \varphi_b(z_{it}, k_{it}, b_{it} \mid \beta) + \xi^b_{it}
\end{aligned}
$$
where $z$ is the latent productivity shock, $\varphi_k$ and $\varphi_b$ are the optimal investment and leverage policies implied by the model at $\beta$, and the unconditional mean of $\log z$ is normalized to zero so the production residual carries all level information.

**Residual specification.** The three residuals absorb the gap between model implications and observed data:
$$
\eta_{it} \sim \mathcal{N}(\mu_\eta, \sigma^2_\eta), \quad \xi^k_{it} \sim \mathcal{N}(\mu_{\xi^k}, \sigma^2_{\xi^k}), \quad \xi^b_{it} \sim \mathcal{N}(\mu_{\xi^b}, \sigma^2_{\xi^b}).
$$
The Gaussian shape is a likelihood assumption that can be relaxed (Student-t, mixture). I take Gaussian as the baseline. The means and variances are additional parameters to be estimated.

**Economic interpretation.** Each residual has important economic meanings:

- $\eta$ is production misspecification or measurement error in revenue. A nonzero $\mu_\eta$ indicates the production function omits a systematic factor (e.g., labor input). $\sigma_\eta^2$ measures production-equation fit quality.
- $\xi^k$ and $\xi^b$ are *policy deviations*: the systematic gap between observed investment / leverage and the model-implied optimal choices. A nonzero $\mu_{\xi^k}$ indicates firms invest more or less on average than the model-implied optimal decisions; $\sigma_{\xi^k}^2$ measures dispersion of that deviation. Analogously for $\xi^b$. This is the cross-sectional analog of the investment and labor "wedges" in DSGE business-cycle accounting.

**Identification.** Generally, we need to constraints for identification:

1. The unconditional mean of $\log z$ is normalized to zero. Without this, $E[\log z]$ and $\mu_\eta$ are observationally equivalent in the production equation. The normalization is harmless: any nonzero unconditional mean of latent productivity is absorbed by $\mu_\eta$.
2. The policy intercept of $\varphi_k(\cdot \mid \beta)$ is a deterministic function of $\beta$. A free additive $\mu_{\xi^k}$ is partially collinear with shifts in $\beta$ that move this intercept. Depending on the specific models, identification is enabled by (i) informative priors on $\mu_{\xi^k}$ centered at zero, and (ii) $\beta$ enters $\eta$, $\xi^k$, and $\xi^b$ jointly, so the posterior anchors $\beta$ from production and AR(1) variations, leaving the residual policy gap to $\mu_{\xi^k}$.

These choices capture the argument for identifying firm's deviation from optimal actions: $\mu_{\xi^k}$ captures "policy deviation that cannot be explained by a different $\beta$ within the model class."

**Baseline choice of priors**

I consider the following baseline choice of priors:

| Parameters | Prior | Support | Rationale |
|---|---|---|---|
| $\alpha$ | Beta(2, 2) | $(0, 1)$ | Moderate return to scale |
| $\rho$ | Beta(2, 2) | $(0, 1)$ | Moderate persistence |
| $\sigma_\varepsilon$ | HalfNormal(0.3) | $(0, \infty)$ | More mass on low variance|
| $\sigma_\eta$ | HalfNormal(0.1) | $(0, \infty)$ | More mass on less measurement error |
| $μ_η, μ_{ξ^k}, μ_{ξ^b}$ | Normal(0, 0.1) | $(-\infty, \infty)$ | Center around zero |
| $σ_η, σ_{ξ^k}, σ_{ξ^b}$ | HalfNormal(0.5) | $(0, \infty)$ | More mass on low variance |


**Special case: observation-only inference.** When the only goal is to identify parameters that appear in the production equation $(\alpha, \rho, \sigma_\varepsilon, \mu_\eta, \sigma_\eta)$, the policy residuals can be dropped from the likelihood entirely. Observed $(k, b)$ are then treated as exogenous covariates and no policy solve is needed. This case applies to the frictionless basic model, where the policy adds no extra identifying information beyond what the production equation provides. For every other model in the project (frictional basic, risky debt, extensions), cost and default parameters enter only through the policy, so the policy residuals must be kept in the likelihood. This is the central motivation for the neural-surrogate section below: solving the model at every MCMC step requires the policy to be amortized in $\beta$ so each evaluation is one forward pass.



### Generic Algorithm

The pipeline is generic across choices of filter and MCMC sampler. The Evaluate step branches by approach.

**Setup.**

- Specify priors $p(\cdot)$
- Specify a filtering algorithm that, given fixed data $y$ and a candidate $\beta$, returns the scalar $\log p(y \mid \beta)$.
- Fix the number of MCMC iterations $J$ and the number of chains.

**Pre-training NN surrogate policy**

This step may be skipped if the optimal policy can be written in closed-form formula, or if inference does not need optimal policy to form the observation equations.

- Takes firm panel dataset as input, extract state space bounds (e.g., min and max capital observed)
- Use the extracted state bounds from panel and parameter bounds from prior to simulate a panel of training dataset
- Train NN on the simulated data and obtain the parametrized policy surrogate, $\varphi_\theta(s, \beta)$, which is stored as NN weights and biases. Discard the training and validation set.
   -  Model solve method: Short-horizon Actor Critic (default) and Euler residual minimization (special case)
- Pass the stored NN surrogate to the likelihood estimation in next stage


**Bayesian Inference: MCMC + Filtering Algorithm** 

For each MC chain, per iteration $j = 1, \ldots, J$:

1. **Propose** a candidate $\beta'$ by perturbing the current $\beta$. I built three options:
   - *Random-Walk Metropolis-Hastings (RW-MH):* draw $\beta' = \beta +$ Gaussian noise. Gradient-free default.
   - *Robust Adaptive Metropolis (RAM):* Adaptive version of RW-MH that avoids manual tuning of the search step size.
   - *No-U-Turn Sampler with Hamiltonian Monte Carlo (NUTS-HMC):* use $\nabla_\beta L(\beta)$ to simulate Hamiltonian dynamics on $L$, then slice-sample a candidate from the trajectory. Gradient-based; requires $L$ to be differentiable in $\beta$.
2. **Evaluate** $L(\beta') = \log p(\beta') + \log p(y \mid \beta')$. The likelihood step depends on the approach:
   - *Option 1: Closed-form Policy.* Form the observation equation using closed-form formula of optimal policy. Run the filter. 
   - *Option 2: Neural Surrogate Policy.* Take the pre-trained NN surrogate policy $\varphi_\theta(s,\beta)$, plug into observation and transition equations, run the filter.  

3. **Accept or reject.** MH test accepts $\beta'$ with probability $\min(1, \exp(L(\beta') - L(\beta)))$ under symmetric proposal. NUTS-HMC uses the joint Hamiltonian ratio.
4. **Record** the current chain position as sample $\beta^{(j)}$.

**Output.** A pooled set of samples $\{\beta^{(j)}\}$ across chains. Their empirical distribution approximates $p(\beta \mid y)$. Posterior means, quantiles, and credible intervals are computed from this set.

**Two principles guide method choice:**

- Filter and sampler are independent components and can be swapped separately.
- The MH accept/reject step guarantees correctness. The proposal rule controls only efficiency.

### Validation and Diagnostics

Below are the metrics and tests I use to validate the inference implementation.

**Per-run MCMC convergence.** The following metrics are reported to test whether the sampler converged on this dataset?

- *Split R-hat*. Between- vs within-chain variance ratio after splitting each chain in half. Target $\hat{R} < 1.01$ at production budget; loosened to $\hat{R} < 1.05$ at smoke budget.
- *Effective Sample Size (ESS).* Autocorrelation-corrected count of independent samples,

    $$\mathrm{ESS} = \frac{MS}{1 + 2 \sum_{k \geq 1} \rho_k}$$

    for $M$ chains and $S$ post-warmup samples each, with $\rho_k$ the lag-$k$ autocorrelation truncated at the first negative estimate. Target $\mathrm{ESS} > 400$ (about 5% Monte Carlo error on credible-interval quantiles).
- *NUTS-specific signals.* Zero divergences and no max-tree-depth saturation across post-warmup iterations. Both indicate the sampler is exploring the target geometry without numerical breakdown or premature trajectory termination.
- *Trace plots.* Overlapping chains, no drift, no stuck plateaus. Necessary but not sufficient; complements the numerical diagnostics.

TFP modules: `tfp.mcmc.potential_scale_reduction`, `tfp.mcmc.effective_sample_size`. Divergence counts and per-iteration tree depth are returned in the trace metadata of `tfp.experimental.mcmc.windowed_adaptive_nuts`.

**Calibration of the inference machinery.** Do the credible intervals attain their nominal coverage on data drawn from the model? This validates the implementation against the data-generating process.
- *Coverage check.* Draw $R$ parameter vectors $\beta_0$ from the prior; for each, simulate a panel under the model at $\beta_0$, run the full inference pipeline, and record per-parameter 95% credible intervals. Pass when the empirical hit rate per parameter falls inside the binomial 95% interval around true coverage 0.95 at the chosen $R$.
- Future extension: *Simulation-Based Calibration.* Rank-based formal test at $R \geq 100$ with rank histograms; the rigorous version of the coverage check. Not implemented in current version


**Model-data fit.** The last set of tests aim to examine how the fitted model describe the real data?

- *Posterior predictive checks (PPC).* Compare summary statistics of replicated panels $Y^{\mathrm{rep}} \sim p(\cdot \mid \beta^{(s)})$ to the observed $Y$. Systematic mismatch in a summary that the model should reproduce flags misspecification.
- *Prior sensitivity analysis.* Vary the prior within a defensible range and confirm that posterior summaries move by less than their credible-interval widths. Bounded movement indicates the posterior is data-driven rather than prior-driven on the parameters of interest.


## Bayesian Inference with Neural Surrogate

The goal is to verify the Bayesian inference pipeline end-to-end on the basic model with no adjustment costs. This is a special case of the toy model because the optimal policy has a closed-form formula, which allows me to validate and test the pipeline with ground-truth. 

The main innovation of my implementation is the use of a pre-trained "Neural Surrogate" to approximate the optimal policy function in inference. The benefit is substantial:
- Typical MCMC + Filtering requires at least **thousands of model solve** (via VFI/PFI/NN-based methods) when evaluating different parameter candidates
- My approach only needs **one model solve**: pre-train a NN surrogate over the entire States $\times$ Parameter space, then pass it to the inference pipeline for each evaluation.

I illustrate my pre-training + inference pipeline below using the toy basic investment model.

### Environment: Basic Model

Following @strebulaev2012 [Section 3.1], each firm $i = 1, \ldots, N$ solves an infinite-horizon investment problem with no adjustment costs and no debt.

The log-productivity follows a first-order autoregressive process (AR1):

$$\log z_{i,t+1} = \rho \cdot \log z_{i,t} + \sigma_\varepsilon \cdot \varepsilon_{i,t+1}$$

where $\varepsilon_{i,t+1} \sim \mathcal{N}(0,1)$, $\rho \in (0,1)$, $\sigma_\varepsilon > 0$. Shocks $\varepsilon_{i,t+1}$ are iid across firms and time.

Firm's observed revenue and capital stock is assumed to be generated by:
$$
\begin{aligned}
\log \Pi_{it} &= \log z_{it} + \alpha \cdot \log k_{it} + \eta_{it}  \\
\log k_{it} &= \varphi_k(z_{it}, k_{it} \mid \beta) + \xi^k_{it}
\end{aligned} 
$$

where as before I assume that model specification error are Gaussian:
$$
\eta_{it} \sim N(\mu_\eta, \sigma_\eta), \qquad \xi^k_{it} \sim N(\mu_\xi, \eta_\xi)
$$

Using capital accumulation identity, $k_{i,t+1} = (1 - \delta)\, k_{i,t} + I_{i,t}$, the control (action) variable can be re-written from investment $I$ to $k_{t+1}$, that is, firm directly choose the optimal capital level of next period. 

Without adjustment costs, firm's optimal policy function has closed-form solution:

$$ \begin{aligned}
\varphi_k(z_{it}, k_{it} \mid \beta) &= \frac{\rho}{1-\alpha}\, \log z_{i,t} + \frac{1}{1-\alpha}\!\left[\log \alpha + \frac{\sigma_\varepsilon^2}{2} - \log(r + \delta)\right]\\
&= \frac{\rho}{1-\alpha}\, \log z_{i,t} + \kappa(\alpha,\sigma_\epsilon,r,\delta)
\end{aligned}$$

where we denote the second intercept term as $\kappa(\cdot)$ for simplicity. This formula can be derived easily using the Euler equation $E_t[\alpha \, z_{i,t+1} \, k_{i,t+1}^{\alpha-1}] = r + \delta$ and $E_t[z_{i,t+1}] = \exp(\rho \log z_{i,t} + \sigma_\varepsilon^2 / 2)$ for log-normal variable $z$. 

This means capital is log-linear in $\log z_{i,t}$, enabling the linear Gaussian state-space form.

**Target parameters to be estimated** include economic parameters $\beta \equiv (\alpha, \rho, \sigma_\varepsilon, \sigma_\eta)$ and the mean $(μ_η, μ_{ξ^k})$ and variances $(\sigma_η, \sigma_{ξ^k})$ for the model-specification errors.

**Calibrated parameters:** $r = 0.04$ (risk-free rate) and $\delta = 0.10$ (depreciation rate).

### Pre-training the Neural Surrogate

The pre-training part uses two NN-based methods to solve for firm's optimal policy:
- Default: Short-Horizon Actor Critic (SHAC)
- Optional: Euler Residual Minimization (ER)

These methods have been introduced and tested in part 1. The key point is instead of mapping from state space to action space, the NN surrogate is trained on the higher-dimensional state $\times$ parameter space. 

Concretely, let $\beta \equiv (\alpha, \rho, \sigma_\varepsilon)$ denote the structural economic parameters of the basic model. The pre-trained NN surrogate here is $\varphi(z,k; \beta)$ mapping to the optimal next-period capital $k'$. This is different from pure model solve (Part 1) where parameters are fixed and the NN policy is only 2-dim over $(z,k)$.

This approach is only feasible with NN-based policy approximator. Traditional numerical methods like VFI, PFI, or Linear Programming are grid-based and cannot be solved once over a high-dimensional parameter space. It would require repeatedly solving the model under different parameters, which is intractable even for toy model of this scale (with 8 parameters to be estimated).

To validate the quality of the pre-trained NN surrogate policy, I use the same set of metrics computed on held-out validation dataset. I also plot the NN solution against true analytical solution over $k$ slices. @fig-nb09a-slices shows that SHAC learned a highly precise NN surrogate (orange) with mean absolute error (MAE) lower than 1% when compared against the true closed-form formula (dash). The pre-trained NN surrogate maps $(z,k, \alpha, \rho, \sigma_\varepsilon)$ to the optimal next-period capital $k'$. SHAC is also efficient as this training took about 30min on a CPU (Apple M1). In future, this can be scale up with GPU and more training budget to learn more complex models and to achieve lower mean absolute error. The results can be reproduced in `docs/08a_pretrain_nn_surrogate.ipynb`.

![Training curves: held-out MAE vs SHAC/ER step. Red dashed line marks the best-checkpoint restore; the gap to final-step weights illustrates why best-step restoration matters.](figures/paramNN-validate/training_curve_full.png){#fig-training-curve}

![Pre-trained Neural Surrogate.](figures/bonus1-bayesian-basic/slices_pretrain.png){#fig-nb09a-slices}



### Likelihood: Extended Kalman Filter

I use Extended Kalman Filter (EKF) to compute likelihood.
Independence across firms gives
$$\log p(Y \mid \beta) = \sum_{i=1}^{N} \sum_{t=1}^{T} \log p(y_{i,t} \mid y_{i,1:t-1}, \beta),$$
where $y_{i,t} = (\log \Pi_{i,t}, \log k_{i,t+1})^T$ stacks the two observations at firm-year $(i, t)$ and $\beta$ collects every estimable parameter from the model specification above.

The latent state is scalar, $x_{i,t} \equiv \log z_{i,t}$. Let $m_{t|s}, V_{t|s}$ denote the conditional mean and variance of $x_{i,t}$ given $y_{i,1:s}$ (firm index suppressed).

**Two observation equations.** From the model specification above,
$$
y_{i,t}^{(1)} \;=\; x_{i,t} + \alpha \log k_{i,t} + \mu_\eta + \eta_{i,t}, \qquad \eta \sim \mathcal{N}(0, \sigma_\eta^2),
$$
$$
y_{i,t}^{(2)} \;=\; g(x_{i,t}, k_{i,t}; \beta) + \mu_{\xi^k} + \xi^k_{i,t}, \qquad \xi^k \sim \mathcal{N}(0, \sigma_{\xi^k}^2),
$$
where $g(x, k; \beta) := \log \varphi_k(\exp x, k; \beta)$ is the log-policy prediction for $\log k_{t+1}$. Eq 1 is linear in $x$; Eq 2 is **potentially nonlinear** in $x$ through $g$.

The EKF linearizes the nonlinear policy term $g$ with a first-order Taylor expansion around the predicted latent mean. Two regimes apply. For the **closed-form policy**, $g(x;\beta) = \rho(1-\alpha)^{-1} x + \kappa(\alpha,\sigma_\varepsilon)$ is globally linear in $x$, so the linearization is exact and the EKF reduces to the standard Kalman filter; I use this as the validation ground-truth. For the **neural surrogate**, $g$ is the cached network $\varphi_\theta$ and linearity holds only approximately, with the residual variance $\sigma_{\xi^k}^2$ absorbing the gap. The full predict/update recursion, innovation covariance, and per-step likelihood contribution are in [the EKF appendix](#sec-ekf-appendix); the cost is $O(N\cdot T)$ filter steps per evaluation, each requiring one $g$ evaluation and one Jacobian.

**Implementation.** TFP's `LinearGaussianStateSpaceModel` does not apply because $g$ is nonlinear in $x$ for the NN surrogate, so the EKF is hand-rolled in `src/v2/estimation/bayesian_basic_investment.py` (`_build_ekf_log_likelihood`). The per-step Jacobian $H_2(t)$ is computed inside `tf.GradientTape` (reverse-mode autodiff; at scalar latent state this matches forward-mode efficiency and is XLA-compatible). The whole filter loop is wrapped in `@tf.function(reduce_retracing=True, jit_compile=True)` so NUTS-HMC obtains $\nabla_\beta \log p(Y \mid \beta)$ end-to-end through one compiled graph, with no hand-coded backward pass. The same `_build_ekf_log_likelihood` factory serves both the closed-form path (with `policy_callable = env.analytical_policy`) and the NN-surrogate path (with `policy_callable = policy_nn`); only the policy callable differs.


### MCMC Sampler: NUTS and RW-MH

I implemented two main samplers: NUTS-HMC and RW-MH. I also add an adaptive version of RW-MH known as Robust Adaptive Metropolis (RAM) introduced by @vihola2012robust.

**MCMC Sampler Algorithm**

1. Start at $\beta_0$ (drawn from the prior).
2. At each iteration: 
   - Propose $\beta'$ from a proposal distribution $q(\cdot \mid \beta)$, 
   - Accept it with probability $\min\bigl(1,, \tfrac{p(\beta' \mid y)}{p(\beta \mid y)} \cdot \tfrac{q(\beta \mid \beta')}{q(\beta' \mid \beta)}\bigr)$; otherwise stay at $\beta$.
3. After a warmup phase, the visited $\beta$ values are samples from the posterior $p(\beta \mid y)$.

**Sampler comparison.** I summarize the three samplers by what they do and where each is preferred.

| Sampler | What it does | Use case | Reason |
|---|---|---|---|
| **NUTS-HMC** | Uses the log-target's gradient $\nabla L(\beta)$ to simulate informed trajectories through parameter space. Trajectory length, step size, and per-axis scale are auto-tuned during warmup. | Closed-form policy function | Likelihood is differentiable and the gradient is cheap to compute (e.g., closed-form policy). Highest per-iteration efficiency.
| **RW-MH** | Proposes the next $\beta$ by adding Gaussian random noise to the current value. Uses no gradient or local geometry. Step size is fixed and tuned manually. | NN policy surrogate | Default fallback when the gradient is unavailable or too expensive (e.g., nonlinear NN surrogate). Simple and gradient-free, but needs many more iterations to mix. 

**Why gradient-free is the default for the NN-surrogate pipeline.** NUTS achieves its efficiency by running many small inner steps per iteration, each requiring one gradient of the log-target. With a closed-form policy, this gradient flows through analytical formulas and is cheap to compute. With the NN surrogate, computing the same gradient requires backpropagation through the entire network, which is much more expensive than a single forward evaluation. In our pipeline this gap is dramatic: NUTS with the NN surrogate runs more than 30 hours of wall time on CPU for the basic model, while RW-MH or RAM with the same NN finishes in minutes. The trade-off is that gradient-free samplers need more iterations to reach the same posterior precision, but each iteration is so much cheaper that they remain practical where NUTS+NN does not.

### Implementation Issues

There are several practical issues need to be noted:

- The NN surrogate (SHAC) must be trained over a region that contains the prior's bulk mass. If the prior puts non-negligible probability outside the trained box, MCMC will visit points where the NN extrapolates and the likelihood is junk. At the code level I strictly align the box for parameters between pre-training and inference, and I also let NN training extract the box range of observables (e.g., capital) from the panel data first and align the simulated training data with it.
- In my notebook demo, I slice the full dataset into smaller sub-samples to better control compute cost and wall time (specified by `n_firms` and `horizon`). The full panel should be used in production.
- For this specific model, $\mu_\xi$ is weakly identified because it is highly correlated with $\alpha$. I did not patch this to keep the baseline algorithm simple and minimal. For future implementation, weak identification would need to be handled carefully with re-parameterization tricks or other treatment.


### Reproducibility

The pipeline reuses the project's existing stateless-seed infrastructure (`src/v2/data/rng.py`, `src/v2/utils/seeding.py`). A single master seed pair `(m0, m1)` controls every RNG-consuming step. Per-replicate child seeds are derived deterministically via `fold_in_seed(master, *tokens)`, where `tokens` are short namespace strings. Two key properties:

- Rerunning with the same `master` reproduces every $\beta_0$, every panel, every $\eta$, and every chain trajectory bit-for-bit on the same hardware **and the same TFP version**. The NUTS warmup is delegated to `tfp.experimental.mcmc.windowed_adaptive_nuts`, which derives per-window seeds via `tfp.random.sanitize_seed`. The same stateless-seed model is used throughout TFP / JAX. The per-window split tree can change across TFP releases; posteriors remain statistically equivalent, but individual chain trajectories may differ byte-by-byte.
- Token-scoped folding means stages are isolated: re-running just MCMC on a fixed panel only changes `mcmc_seed`'s token, not the panel or $\beta_0$ draws.

No global RNG state is used. All randomness enters through explicit seed arguments.


## Validation Results

This section reports results by applying Bayesian estimation on simulated panels and verify if it can recover the ground-truth. There are two layers to be validated: 

- Correctness of the inference pipeline itself: MCMC + filtering algorithms
- Whether integrating the NN surrogate (instead of closed-form policy) into the inference pipeline lead to any bias and issues

For the first layer, I already implemented unit tests and integration tests, and the validation here is an additional replication to confirm that we can recover the true parameters (posteriors) of a toy model with closed-form optimal policy `kp = exp((log α + log E[z] - log(r+δ))/(1-α))`. 

For the second layer, the only difference is one-line code change that inject the pretrained and cached neural network `policy_nn` to replace the closed-form policy formula inside the likelihood `_build_ekf_log_likelihood`.

More specifically, the validation has three steps:

1. Fix a set of ground-truth parameters $(\alpha, \rho, \sigma_\varepsilon, \sigma_\eta, μ_η, μ_{ξ^k}, \sigma_η, \sigma_{ξ^k})$, simulate a panel of firms using the observation equations of the model, drop latent variable $z$, the final data include $(k_{it},\Pi_{it})$
2. Extract the range of observables $k_{it}$ from panel, simulate new training and validation set on the same support, pre-train the NN surrogate policy using SHAC, save the learned NN $\varphi_\theta$ after convergence
3. Pass the panel $(k_{it},\Pi_{it})$ and the policy (using either pre-trained $\varphi_\theta$ or closed-form) to MCMC sampler + filtering algorithm. Verify that the posterior median and credible intervals recover the ground-truth.



###  NUTS + Kalman Filter with closed-form policy

This section can be reproduced by running `docs/08c_nuts_closedform_validation.ipynb`.

As the first validation exercise, I use the toy basic investment model to verify the code-level correctness of the inference pipeline for NUTS Sampler + Extended Kalman Filter. The implementation follows these steps:

1. Fix a set of ground-truth parameters $(\alpha, \rho, \sigma_\varepsilon, \sigma_\eta, μ_η, μ_{ξ^k}, \sigma_η, \sigma_{ξ^k})$, simulate a panel of firms using the observation equations of the model, drop latent variable $z$, the final data include $(k_{it},\Pi_{it})$
2. Use closed-form optimal policy to form the LGSSM.
3. Run NUTS-HMC with Kalman filtering on the panel data $(k_{it},\Pi_{it})$, store and verify that the posteriors recover the ground-truth

By default, I use 4 chains, 1000 warmup steps, 500 post-warmup sample, and 0.9 MH test acceptance rate. I slice the panel to a small sample with 50 firms over 20 periods so that the estimation finished in one hour on Apple M1 (2020). 

**Table 1: Posterior summary at ground-truth $\beta$ (single run).**

| Parameter            | True  | Median | 95% CI          | $\hat{R}$ | ESS  |
|----------------------|-------|--------|-----------------|-----------|------|
| $\alpha$             |  0.50 |  0.495 | [0.479, 0.511]  | 1.005     |  404 |
| $\rho$               |  0.50 |  0.507 | [0.494, 0.520]  | 1.002     |  832 |
| $\sigma_\varepsilon$ |  0.24 |  0.232 | [0.221, 0.244]  | 1.002     | 1240 |
| $\mu_\eta$           |  0.00 |  0.014 | [-0.032, 0.065] | 1.004     |  480 |
| $\sigma_\eta$        |  0.05 |  0.051 | [0.032, 0.065]  | 1.000     |  705 |
| $\mu_{\xi^k}$        | -0.10 | -0.048 | [-0.197, 0.100] | 1.006     |  408 |
| $\sigma_{\xi^k}$     |  0.05 |  0.048 | [0.025, 0.062]  | 1.001     |  624 |

All 7 parameters recover truth within the 95% credible interval. As a rule of thumb, all $\hat{R} < 1.01$ and all ESS exceed 400. The three economic parameters $\alpha, \rho, \sigma_\epsilon$ are tightly identified. Out of the 4 remaining parameters, $\mu_{\xi^k}$ is of interest because it captures firm's deviation from the model-implied optimal policy, in this case the true deviation is set to be -10%. The $\mu_{\xi^k}$ posterior correctly covers the -20% to 10% range with median close to -5%, but the CI is wide due to weak identification as $\mu_{\xi^k}$ is partially collinear with $\alpha$. For future production, I will need a larger sample, more training budget, and re-parameterization to solve the weak identification issue.

Split-$\hat{R}$ is the Gelman-Rubin ratio of between-chain to within-chain variance. $\hat{R}$ near 1 mean the chains have mixed to a common distribution. ESS is the autocorrelation-corrected count of effectively independent samples. For all posteriors, our ESS $\gt 400$ pass the minimal threshold. I interpret that both metrics pass the posterior diagnostic checks.

![Plot Posterior marginals: dashed black = true value, red = posterior median. The x-axis is fixed at the posterior median $\pm 4$ standard deviations, expanded to include the true value if it falls outside, clipped to support.](figures/bonus1-bayesian-basic/validate-closedform/marginals.png){#fig-bayes-marginals}

@fig-bayes-marginals plots the posterior marginals for all parameters, where dashed line is ground-truth and the red line is the estimated posterior median. @fig-bayes-trace plots the trace of post-warmup draws. The visual evidence is clean and suggest that the code-level implementation of the Bayesian inference pipeline does not have major defects and bugs.


![Plot Trace: two chains per parameter, overlaid with the truth line. Convergence shows as the two chains exploring the same region with no drift or stuck plateaus.](figures/bonus1-bayesian-basic/validate-closedform/trace.png){#fig-bayes-trace}


**Coverage and Sensitivity Checks**.
To show additional checks, I re-run the estimation and the full post-inference analysis with smaller budget per replication: 2 chains, 500 warmup, 200 post-warmup samples, 15 firms and 10 periods (~90 min on Apple M1). This is obviously not sufficient, so the following results should be interpreted as a *budget-limited demo* under time pressure. Future production will use much larger budget for credible validation results.

![Plot coverage of 95% CI over R=5 replications.](figures/bonus1-bayesian-basic/validate-closedform/coverage_intervals.png){#fig-bayes-coverage}

@fig-bayes-coverage shows whether each replicate's 95% credible interval contains its ground-truth $\beta_0$. A well-calibrated estimator hits truth in roughly 95% of replicates. At the demo budget of $R = 5$ most parameters hit 3 or 4 times out of 5, just below the binomial pass-band and consistent with both small-$R$ noise and modest per-replicate warmup. The pipeline runs end-to-end here. A production calibration claim requires $R \geq 30$ and longer per-rep adaptation.

![Plot coverage of 95% CI over R=5 replications.](figures/bonus1-bayesian-basic/validate-closedform/ppc_distributions.png){#fig-bayes-ppc}

@fig-bayes-ppc compares six summary statistics of the observed panel against their distributions under panels simulated from the posterior. If the model fits the data well, each observed value should sit somewhere in the bulk of its simulated distribution. An extreme tail position flags a feature the model cannot reproduce. All six posterior median estimates sit in the central range. The fitted model reproduces these data features without systematic misspecification.

![Plot coverage of 95% CI over R=5 replications.](figures/bonus1-bayesian-basic/validate-closedform/sensitivity_comparison.png){#fig-bayes-sensitivity}

@fig-bayes-sensitivity overlays the posterior under three variants of the residual standard-deviation priors: tight, baseline, and loose. A posterior that barely moves across variants is data-driven, in contrast, one that tracks the prior is prior-driven. For all seven parameters the medians shift by a fraction of the credible-interval width and the intervals largely overlap. The data carries the identifying information here and the posterior estimates are generally robust.

### RW-MH + Kalman Filter with Neural Surrogate Policy

This section can be reproduced by running `docs/08b_rwmh_three_way_baseline.ipynb`.

The second validation exercise switches the gradient-based NUTS for a gradient-free RW-MH sampler, holding the Kalman filter unchanged. I argue that when using a pre-trained NN surrogate, RW-MH sampler is better than NUTS-HMC because backpropagation through the cached NN is still very slow per leapfrog iteration (>40 hours of wall time on Apple M1).

To make the attribution clean, I run three configurations side-by-side on the same observed panel at the same scalar proposal step size:

- **Closed-form RWMH** (control): RW-MH with the closed-form analytical policy. Isolates the sampler from any NN approximation error.
- **NN-RWMH** (test): same RW-MH, with the cached SHAC NN as the EKF's policy. Difference vs CF-RWMH attributes to NN approximation error.
- **NN-RAM** (adaptive variant): same NN spec, but with the proposal covariance adapted during warmup [@vihola2012robust]. Difference vs NN-RWMH attributes to the adaptive proposal.

All three are run at 4 chains $\times$ (20000 warmup + 5000 samples) on a 100 firms $\times$ 15 periods panel slice. CF finishes in about 30 seconds; the two NN methods take roughly 50 minutes each. The per-step NN evaluation accounts for the entire wall-time gap.


![Posterior marginal densities under CF-RWMH (green, control with no NN), NN-RWMH (blue, baseline), and NN-RAM (orange, adaptive variant). Dashed black is truth; solid vertical lines are per-method medians.](figures/bonus1-bayesian-basic/rwmh-surrogate/density_overlay.png){#fig-rwmh-density}

![Trace plots: three columns (CF-RWMH, NN-RWMH, NN-RAM), one row per parameter, four chains overlaid in each cell. Well-mixed parameters show overlapping chains; ridge parameters show chains stuck in disjoint regions.](figures/bonus1-bayesian-basic/rwmh-surrogate/trace.png){#fig-rwmh-trace}

**Does the evidence validate the pipeline?** Partially. The results suggest the inference pipeline and the NN surrogate approach worked, but the identification is weak for several parameters which is attributed to the structural model specification.

@fig-rwmh-density overlays the marginal posterior density (instead of histogram) of the three configurations. The vertical lines mark the posterior median and the dashed line marks the true parameter value. It shows that:

- **Closed-form RWMH** (green) confirms the inference pipeline itself is correct, with posterior median very close to true value and a tight 95% CI
- **NN-RWMH** (blue) are noisier but the posterior density is still close to **Closed-form-RWMH**. This is important to confirm that the NN surrogate's approximation error does NOT break the inference. It just adds upfront pre-training cost for achieving better precision.
- **NN-RAM** (adaptive variant) rejects adaptive RW-MH as a usable variant at least for the basic model.

On the other hand, however, both the posterior density and the trace plot show failures in mixing. I view it as part of the model's specification issue. In particular, two parameters ($\rho$ and $\mu_\eta$) actively miss truth at the 95% level, consistent with their broken chains and the inflated $\hat{R}$ and insufficient ESS.

The six poorly-mixed parameters all lie along the $\kappa(\alpha) + \mu_{\xi^k}$ ridge already flagged in the identification discussion in previous section: a tightly correlated subspace that a scalar Gaussian random walk cannot traverse efficiently within a practical wall budget. The CF-RWMH control reproduces the same failure pattern, which rules out NN approximation error as the dominant cause and isolates the bottleneck to the sampler class. In other words, **the NN surrogate is validated: it does not inject the mixing failure.** What fails is the gradient-free scalar random walk on a posterior geometry that the previous NUTS+CF baseline (with gradient information) handled cleanly. NN-RAM helps somewhat (max $\hat{R}$ 1670 vs 4138) but does not break the structural ceiling at this budget.

**How to fix weak identification**. Resolving the mixing failure requires two changes, both beyond the scope of the simple baseline reported here. First, the observation equations need to be reparametrized so that the ridge direction aligns with a single sampled quantity. Concretely, $\mu_{\xi^k}$ and $\kappa(\alpha, \sigma_\varepsilon)$ both enter the $\log k$ equation additively as level terms, so sampling the composite offset $\zeta = \mu_{\xi^k} + \kappa(\alpha, \sigma_\varepsilon)$ in place of $\mu_{\xi^k}$ removes the partial collinearity at sampler level; $\mu_{\xi^k}$ is recovered post-hoc as $\zeta - \kappa(\alpha, \sigma_\varepsilon)$ from the joint posterior of $(\alpha, \sigma_\varepsilon, \zeta)$. Second, the MCMC and filtering stack needs either a substantive extension or a replacement, for example particle filter with adaptive Metropolis (PMMH), sequential Monte Carlo with tempering, or full-covariance adaptive proposals that learn the ridge direction during warmup. Both directions are deferred to future implementation and testing.


**Table 2: NN-RWMH posterior summary against truth (baseline NN + gradient-free path).**

| Parameter            | True  | Median | 95% CI            | $\hat{R}$ | ESS  |
|----------------------|-------|--------|-------------------|-----------|------|
| $\alpha$             |  0.50 |  0.547 | [0.534, 0.846]    | 4138      |  23  |
| $\rho$               |  0.50 |  0.481 | [0.103, 0.496]    | 2394      |  25  |
| $\sigma_\varepsilon$ |  0.24 |  0.235 | [0.229, 0.242]    | 1.3       |  65  |
| $\mu_\eta$           |  0.05 | -0.073 | [-0.824, -0.034]  | 2015      |  270 |
| $\sigma_\eta$        |  0.05 |  0.036 | [0.015, 0.052]    | 185       |  29  |
| $\mu_{\xi^k}$        | -0.10 | -0.528 | [-1.219, -0.406]  | 250       |  347 |
| $\sigma_{\xi^k}$     |  0.05 |  0.060 | [0.048, 0.280]    | 3691      |  142 |

### NUTS + Kalman Filter with Neural Surrogate Policy

A single run took over 50 hours on the M1 CPU, so this configuration is impractical. The per-leapfrog backpropagation through the cached NN is the bottleneck, as discussed in the sampler comparison above.




## Future Extensions

I consider the following directions as promising future extensions to better design and implementation of the final commercial product.

- **Neural Likelihood Estimation**. Train a neural network to replace the model-specific likelihood evaluation step, such as the Kalman filter likelihood in a state-space model. The MCMC sampler can remain unchanged, except that each likelihood evaluation inside MCMC is replaced by the neural likelihood surrogate. This is a major extension to our current policy surrogate approach. It shares the same motivation of **amortizing the cost of expensive model solves**: instead of resolving the structural model at every MCMC proposal, we solve/simulate the model many times upfront and train a neural network to approximate the likelihood. Once trained, the surrogate likelihood can be evaluated cheaply inside MCMC, making Bayesian inference feasible for models where repeated likelihood evaluation is computationally prohibitive.

- Benchmark the NN-surrogate inference under alternative filter + sampler pairings, especially **particle filter + Random Walk Metropolis-Hastings (PMMH)**. Although NUTS + Kalman is theoretically more efficient per iteration, the per-leapfrog gradient via autodiff through the NN is a measurable bottleneck on CPU; a gradient-free sampler paired with a non-Gaussian filter trades per-iteration efficiency for cheaper per-evaluation cost and parallelises more naturally on GPU. The cross-comparison would clarify whether the gradient-based pipeline remains the right default for NN-surrogate inference at scale.

- Address **weak-identification via re-parameterisation**. For example, in the current toy model $\mu_{\xi^k}$ (deviation) is weakly identified as it correlate strongly with $\alpha$. Depending on the model and parameters of interest, we may need the re-parameterisation at algorithm level.

- **Hierarchical (multilevel) Bayesian model** for estimating firm-specific posteriors, which is more useful for the commercial product (e.g., firm's deviation from optimal capital structure). In this framework, each firm $i$ has its own parameter vector $\beta_i$, linked through a shared population prior $\beta_i \sim p(\cdot \mid \beta_{\text{pop}})$ with $\beta_{\text{pop}}$ drawn from a hyperprior. This sits between two extremes: full pooling (one $\beta$ shared across all firms in the current setup) and full separation (independent per-firm MCMCs, which discard cross-sectional information). The mechanism is partial pooling: each firm's $\beta_i$ deviates from $\beta_{\text{pop}}$ where its own time series demands, but shrinks toward the population mean for parameters that are weakly identified within a single firm.


# Bonus Question 2: Dynamic Model of Optimal CEO Contract

This chapter summarizes the canonical model of CEO compensation and short-termism, based on @marinovic2019ceo. It isolates the fundamental economic mechanics of multi-period agency problems when performance is manipulable. @cronqvist2024 extends this canonical model to a specific empirical setting: the FAS 123-R regulatory shift in the United States. To do so, they add limited investor attention ($\alpha$) and endogenous risk-taking (where the CEO controls volatility $\sigma_t$).

I choose to implement the @marinovic2019ceo variant because it provides a clean baseline and is applicable to generic empirical settings. This is useful because when applying to APAC market, we may not have quasi-experiments like FAS 123-R to identify the additional limited attention channel.

*Replication: all results in this chapter are produced by the notebook* `docs/14_ceo_contract_pipeline.ipynb`.

**What are the key problems that the authors try to illustrate with the model?**

In this model, the CEO privately chooses both productive effort and performance manipulation to boost short-term cash flows at the expense of long-term firm value. Because the board cannot observe these actions directly, it uses the CEO's incentive compatibility constraints as a calibration dial to find the optimal balance between inducing effort and deterring manipulation. The board optimally designs a contract that tolerates some manipulation to maximize net firm value. The mechanics of this dynamic contract are governed by a single state variable representing the duration of the CEO's deferred pay. Crucially, the model captures an endogenous "horizon problem" driven by time. Early in the CEO's tenure, manipulation is naturally deterred because the CEO will personally suffer the future cash-flow reversals while still on the job. However, as retirement approaches, this natural deterrence vanishes. To maintain the CEO's effort without imposing inefficiently high post-retirement risk, the board optimally allows the duration of incentives to drop. This mathematically shifts compensation toward the short term, ensuring the CEO continues working hard but inevitably causing manipulation to escalate in their final years. Ultimately, by solving the principal's dynamic optimization problem, the firm's long-term value is maximized in a reality where short-termism is anticipated, managed, and optimally priced into the contract.


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


**Terminal Value**: At retirement the firm still bears the risk of vesting the CEO's outstanding incentives over the clawback window $[T, T+\tau]$. This gives the value function's terminal condition $F(z, T) = -\frac{1}{2}\mathcal{C}z^2$, a convex penalty on the deferred-pay duration $z$ carried into retirement. The coefficient is closed-form [@marinovic2019ceo, Eq. 10],

$$\mathcal{C} = \frac{\sigma^2(r + 2\kappa)}{r\gamma\left(1 - e^{-(r + 2\kappa)\tau}\right)},$$

so a shorter clawback window (smaller $\tau$) makes deferral more expensive.

### Solution to principal's problem

The solution to the dynamic contract is not a closed-form formula over time, but rather a set of **policy functions** characterizing the optimal actions given the current state $z_t$ and time $t$.

**The Hamilton-Jacobi-Bellman (HJB) Equation**: The principal's problem resolves to the following HJB equation for the value function $F(z,t)$:

$$rF = \max_{a, \sigma_z} \pi(a,z) + F_t + \left[ (r+\kappa)z + a r\gamma(\sigma\sigma_z - 1) \right]F_z + \frac{1}{2}\sigma_z^2 F_{zz}$$

where $\pi(a,z)$ is the simplified flow payoff and the terminal condition is $F(z,T) = -\frac{1}{2}\mathcal{C}z^2$ (the terminal condition above).

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

Because the policy functions depend on the unknown value function $F(z,t)$, we solve the HJB equation numerically and then read the policies off the solution. The next section summarizes the method.



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


## Solution of the model

I use finite-difference with policy iteration to numerically solve the model, following the @marinovic2019ceo baseline calibration: risk-free rate $r=0.1$, CEO risk aversion $\gamma=1$, manipulation cost $g=1$ and marginal cash-flow impact $\theta=0.4$, manipulation depreciation $\kappa=0.3$, cash-flow volatility $\sigma=2$, retirement date $T=10$, and clawback window $\tau=5$. These imply the deterrence coefficient $\phi=4$, value-destruction $\lambda=0$. I solve on the duration range $z\in[0,0.30]$ and plot up to the operating range $0.25$.

### Value and policy surfaces

![CEO contract: value and policy surfaces over (z, t)](figures/bonus2-ceo-contract/surfaces_3d.png){#fig-ceo-3d}

@fig-ceo-3d reproduces the optimal value and policy functions over $(z,t)$ space. This closely replicates Figure 1 of @marinovic2019ceo. 

- The value function $F(z,t)$ is concave and decreasing in the deferred-pay duration $z$, anchored at $F(0,t)=0$. 
- Effort $a(z,t)$ rises with $z$: a larger long-term stake makes the CEO work harder. 
- Manipulation $m(z,t)$ is essentially zero early in tenure and at low duration, then escalates sharply as the CEO nears retirement. 
- Vesting sensitivity $\sigma_z(z,t)$ is close to zero early (the contract is almost deterministic) and turns negative near retirement, so good performance accelerates vesting (performance-contingent pay).

### Policy slices

@fig-ceo-2d provides a cleaner plot by slicing the 3D policy and value functions over the $z$ axis and the $t$ axis. This figure is directly comparable to Figure 5 in @marinovic2019ceo. All comparative statics are consistent with economic intuition underlying the model. One of the key insights is that manipulation only starts to increase near retirement and is also increasing in the duration of deferred compensation $z$.

![CEO contract: policy slices in z and in t](figures/bonus2-ceo-contract/policy_slices.png){#fig-ceo-2d}

### The horizon problem

The central economic finding is the endogenous horizon problem. Early in the CEO's tenure, manipulation is naturally deterred: borrowing from future cash flows hurts the CEO while still on the job, so it is not worthwhile. As retirement approaches, this self-discipline fades. To keep the CEO exerting effort without loading inefficient risk onto the post-retirement window, the board optimally lets the duration of incentives $z$ fall. This shifts pay toward the short term and, as a by-product, lets manipulation escalate in the final years, an outcome the contract anticipates and prices in rather than eliminates.

### Validation checks

Because the model has no closed-form solution, I confirm the numerical solution in two complementary ways. The **HJB residual** substitutes the solved value function and policies back into the HJB equation: a small residual means the solution actually satisfies the equation, so it passes (see notebook). 

The **Monte-Carlo value check** simulates CEO paths under the solved policy and compares the average discounted payoff to $F(z_0,0)$: agreement means the value function is consistent with the policy it implies, and because this check never touches the finite-difference grid, it is an independent confirmation that the solver is right. At the baseline the solution passes both: the residual is small across the state space (largest only at the single manipulation-kink node, where the value function bends sharply), and the simulated value matches $F(z_0,0)$ to about one percent.

The top panel shows that, averaged over the simulated cross-section, both the manipulation flow $m_t$ and its stock $M_t$ rise toward retirement, the horizon effect again. The bottom panel overlays the solved value $F(z_0,0)$ and the simulated mean discounted payoff across initial durations $z_0$; the two curves coincide, so the Monte-Carlo check passes.

![Numerical validation: simulated horizon effect (top) and Monte-Carlo value reconciliation (bottom)](figures/bonus2-ceo-contract/mc_validation.png)

## Structural Estimation (Plan)

To estimate the structural parameters of the dynamic CEO-contracting model using the Simulated Method of Moments (SMM), we must map the theoretical state and action variables to observable corporate data, similar to the empirical strategy employed by @cronqvist2024. However, @cronqvist2024 rely on the specific quasi-natural experiment of the FAS 123-R regulation to provide identifying variation for their behavioral "limited attention" parameter. Lacking a similar unique regulatory shock to provide exogenous identification, I cannot robustly estimate the attention parameter. This is another reason that I skip the extended "limited attention" model and implement the cleaner, generic baseline model of @marinovic2019ceo.

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

I provide a tentative sketch of how we may add the principal-agent problem (the CEO and the board) into a standard corporate model with capital investment and debts. I choose to modify the moral hazard model used in @nikolov2021. This modeling choice, however, faces a trade-off: we lose the elegant semi-parametric model implications from the original @marinovic2019ceo and @cronqvist2024. I'll discuss these details at the end of the section.

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
The solution to this dynamic contracting problem yields a value function and a set of policy functions. The **value function** $W(k, V, z)$ maps the firm's current state (its physical capital $k$, the CEO's promised utility $V$, and the economic environment $z$) to the maximum total expected firm value. The **policy functions** map this exact same state $(k, V, z)$ to the optimal decisions the board makes today: the investment policy (choosing $k'$), the payout policy (dividends $d$), the continuation policy (future promised value $V'$), and the recommended effort policy ($e$).

While the firm's debt is not a state variable, the optimal contract dynamically implies a capital structure. The value of the outside investors' claim (debt) can be recovered as a simple residual. It is the total firm value minus the equity value promised to the CEO: $b(k, V, z) = W(k, V, z) - V$.

### Discussion: Model Comparisons and Trade-Offs

**1. Differences from the original model in** @marinovic2019ceo

* **What is Gained:** By explicitly tracking physical capital, we can study real-world corporate investment. The model illustrates how a firm might cut back on expanding or buying equipment, not because a bank refuses to lend, but because the board must keep cash inside the firm to manage the CEO's temptation to steal. It directly connects the physical growth of the firm to the severity of the CEO's moral hazard.
* **What is Lost:** We lose the mathematical simplicity and explicit formulas of the @marinovic2019ceo framework. Because we now track three separate state variables $(k, V, z)$ instead of one, the problem becomes a massive computational grid. We can no longer rely on clean calculus to see how variables interact; we must rely entirely on heavy computer simulations.
* **Economic Limits:** Crucially, we lose the CEO's "horizon" effect. @marinovic2019ceo explicitly models the CEO's timeline to retirement, proving that manipulation escalates as the CEO's departure approaches. By moving to an infinite-horizon setup to accommodate capital accumulation, the CEO never retires. This means we cannot use the model to explain or estimate how age, tenure, or impending retirement impacts corporate fraud. Finally, we lose the smooth compensation structure. In @marinovic2019ceo, the CEO receives a steady stream of income; in this risk-neutral model, the CEO is paid nothing for years until their performance crosses a high threshold, at which point they receive a massive cash payout.



**2. Differences from the original model in** @nikolov2021

* **What is Gained:** The original Nikolov model only focuses on the CEO hiding bad performance. By adding an effort choice, we capture a more realistic, dual-agency friction. The board now faces a difficult balancing act: pushing the CEO to work harder increases the firm's actual profits, but it simultaneously increases the CEO's incentive and opportunity to steal those profits. 
* **What is Lost:** We lose computational speed and estimation flexibility. In the original model, the board only had to ensure the CEO wouldn't lie about the numbers. Now, the board must ensure the CEO doesn't simultaneously slack off *and* lie about the numbers to cover it up. The computer must check vastly more "what if" scenarios to ensure the contract is truly manipulation-proof. This heavy computational burden limits how many additional features or structural parameters we can reliably estimate when taking the model to real-world data.


# Bonus Question 3: Three Extended Corporate Models

## Notation and Timing

This chapter implements linear programming (LP) methods to solve the three structural corporate finance models in @nikolov2021:

- Trade-Off Model (TO)
- Limited Enforcement Model (LE)
- Moral Hazard Model (MH)

*Replication: the model solves are in `docs/09_nikolov_models.ipynb`; data cleaning in `docs/11_nikolov_compustat_cleaning.ipynb`; and the TO, LE, and MH estimation pipelines in `docs/10_nikolov_to_policy_pipeline.ipynb`, `docs/12_nikolov_le_policy_pipeline.ipynb`, and `docs/13_nikolov_mh_policy_pipeline.ipynb`.*

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


### Timing (common across all three models)

Within and across periods, events unfold sequentially as summarized below, following Figure 1 of the paper. The key convention is that $W(s_{it}, z_{it})$ is evaluated at the **end of period $t$**, after period-$t$ shocks, production, repayments/transfers, and default checks have occurred, but before period-$(t+1)$ shocks are realized. Decisions made at the end of period $t$ determine capital and financing/contract terms for period $t+1$.

**State versus realized transfers.** Persistent recursive states are $s=(k,b)$ in TO and LE and $s=(k,V)$ in MH. State-contingent payments such as $p$ in LE and dividends $d$ in MH are realized transfers chosen as part of the previous period's state-contingent contract; they are not persistent state variables except through the continuation balance $b$ or continuation equity value $V$. In the table, contract menus are indexed by the next-period realization $(z',\eta')$.

| Point in time | Event(s) | TO | LE | MH |
|---|---|---|---|---|
| End of $t-1$ | Choose period-$t$ capital and financing/contract, maximizing $W(s_{i,t-1}, z_{i,t-1})$ | Choose $k_{it}$ and debt $b_{it}$; lender break-even sets the spread. | Choose $k_{it}$ and a contract $\{b_{z',\eta'}, p_{z',\eta'}\}$. | Choose $k_{it}$ and a contract $\{V_{z',\eta'}, d_{z',\eta'}\}$. |
| Start/end of $t$ | Shocks $(z_{it}, \eta_{it})$ realize; transfers occur; next state formed | Repay debt if solvent, else default and liquidate. Next state $(k_{it}, b_{it}, z_{it})$. | Contract pays $p$, carries balance $b_{z',\eta'}$; no default. Next state $(k_{it}, b_{z',\eta'}, z_{it})$. | Shareholders observe $\eta_{it}$ (lenders do not); contract pays $d$, carries $V_{z',\eta'}$; no default. Next state $(k_{it}, V_{z',\eta'}, z_{it})$. |
| End of $t$ | Choose period-$(t+1)$ capital and financing/contract (schedule repeats) | Choose $k_{i,t+1}$ and debt $b_{i,t+1}$; lender break-even sets the spread. | Choose $k_{i,t+1}$ and a contract $\{b_{z',\eta'}, p_{z',\eta'}\}$. | Choose $k_{i,t+1}$ and a contract $\{V_{z',\eta'}, d_{z',\eta'}\}$. |
| Start/end of $t+1$ | Shocks $(z_{i,t+1}, \eta_{i,t+1})$ realize; as in period $t$, one period ahead | As in period $t$. Next state $(k_{i,t+1}, b_{i,t+1}, z_{i,t+1})$. | As in period $t$. Next state $(k_{i,t+1}, b_{z',\eta'}, z_{i,t+1})$. | As in period $t$. Next state $(k_{i,t+1}, V_{z',\eta'}, z_{i,t+1})$. |

The exact TO default condition, LE break-even/collateral constraints, and MH promise-keeping/incentive constraints are defined in the model-specific sections below. The timing table intentionally states the schedule in words rather than duplicating those equations.

## Solution Method Choice

Models like TO, LE, and MH are complex because they (i) do not have closed-form Euler equations, (ii) feature nested fixed points in equilibrium, and (iii) have different equality and inequality constraints. There is generally no mathematical theorem proving that NN-based training can converge to a *unique fixed point*.

In contrast, grid-based numerical methods like VFI and LP are guaranteed to converge to the unique fixed point of a finite discounted dynamic programming problem under a set of conditions (e.g., contraction mapping). Practically, we need the problem to satisfy:

1. The state grid is finite.
2. Every Bellman inequality corresponds to a fixed feasible action with fixed continuation-state indices and fixed probability weights.
3. The relevant discount factor is strictly below one: $1/(1+r)<1$ for TO/LE and $1/[1+(1-\tau)r]<1$ for MH.
4. The feasible action set is nonempty at every state.
5. TO pricing $\Delta(k_{\text{choice}},b_{\text{old}},z_{\text{old}})$ is well-defined for every grid point, including the $b=0$ case.
6. The resulting LP is feasible and bounded below.

Under these conditions, the Bellman operator for the finite discretized model is a contraction, and the standard LP formulation with objective $\min \sum_s W(s)$ and constraints $W(s)\geq T_a W(s)$ for all feasible state-action pairs recovers the unique fixed point. If continuous interpolation or auxiliary action searches are added later, the master LP remains valid only if each added Bellman inequality freezes the selected action and interpolation weights as constants.


## Trade-off Model

### Theoretical Model

In this setup, $\eta_{it}$ is **public information**.

#### Financing

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

#### Firm Problem

State variables: $(k_{it-1}, b_{it-1}, z_{it-1})$. Bellman equation:

$$W(k_{it-1}, b_{it-1}, z_{it-1}) \equiv \frac{1}{1 + r} \max_{k_{it}, b_{it}} \Big\{ -k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it}$$

$$+ \tau(r + \Delta_{it-1})b_{it-1}\mathcal{I}_{1-D,it} - ((1 - \xi)(1 - \delta)k_{it} + \tau\delta k_{it})\mathcal{I}_{D,it}$$

$$+ E_{t-1}\big[(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) + W(k_{it}, b_{it}, z_{it})\big] \Big\}$$

subject to:

$$(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) - k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it} - (1 + (r + \Delta_{it-1})(1 - \tau))b_{it-1} + b_{it} \geq 0, \quad \forall z_{it}, \eta_{it}$$

$$E_{t-1}\left[ (1 + r + \Delta_{it-1})(1 - \mathcal{I}_{D,it}) + \frac{\xi(1 - \delta)k_{it}}{b_{it-1}}\mathcal{I}_{D,it} \right] = 1 + r$$



## Limited Enforcement Model

### Theoretical Model

State-contingent payoffs are allowed. In this context, $\eta_{it}$ is **public information**.

#### Financing

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

#### Firm Problem

Bellman equation:

$$W(k_{it-1}, b_{it-1}, z_{it-1}) = \frac{1}{1 + r}\max_{k_{it}, b_{z_{it}, \eta_{it}}, p_{z_{it}, \eta_{it}}} \Big\{ -k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it}$$

$$+ \tau r b_{it-1} + E_{t-1}\big[(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) + W(k_{it}, b_{z_{it}, \eta_{it}}, z_{it})\big] \Big\}$$

subject to:

$$b_{it-1} \equiv \frac{1}{1 + r}E_{t-1}[p_{z_{it}, \eta_{it}} + b_{z_{it}, \eta_{it}}] \quad (5)$$

$$(1 - \tau)\pi(k_{it}, z_{it}, \eta_{it}) - k_{it} + (1 - \delta)k_{it} - \Psi(k_{it}, k_{it-1}) + \tau\delta k_{it} + \tau r b_{it-1} - p_{z_{it}, \eta_{it}} \geq 0, \quad \forall z_{it}, \eta_{it} \quad (6)$$

$$p_{z_{it}, \eta_{it}} + b_{z_{it}, \eta_{it}} \leq \theta(1 - \delta)k_{it}, \quad \forall z_{it}, \eta_{it} \quad (7)$$



## Moral Hazard Model

### Theoretical Model

Asymmetric information setup:
- $z_{it}$ follows a Markov chain that is **publicly observable** (also by the lender).
- $\eta_{it}$ is **observable by shareholders but unobservable by lenders**.

A lending contract is a sharing rule splitting firm resources between payments to the lender $p_{it}$ and dividends $d_{it}$, in a fully state-contingent manner.

#### State variable choice

- Use equity value of the firm $V_{it}$ as the state variable; debt value recovered from $b_{it} = W_{it} - V_{it}$.
- Tax deductability of interest on debt: $\tau r b_{it} = \tau r(W_{it} - V_{it})$, which yields:
  - Adjusted discount rate for the firm: $1/(1 + (1 - \tau)r)$ instead of $1/(1 + r)$
  - Penalty for foregone tax deductions on debt: $\tau r V_{it}$

#### Diversion function

$$\mathcal{D}(k_{it}, z_{it}, \eta_{it}, \hat{\eta}_{it})$$

- $\hat{\eta}_{it}$: shareholders' (potentially misreported) report of $\eta_{it}$
- Most straightforward specification under the pre-tax $\pi$ convention used in this document: $\mathcal{D} = \lambda(1-\tau)\left[\pi(k_{it}, z_{it}, \eta_{it}) - \pi(k_{it}, z_{it}, \hat{\eta}_{it})\right]$.
- $\lambda$: diversion parameter; $1 - \lambda$ captures potential losses in cash flow diversion

#### Firm value function

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

#### Payments to the lender (recovered from resource constraint)

$$p_{it} = -k_{it} - \Psi(k_{it}, k_{it-1}) + (1 - \delta)k_{it} + \tau\delta k_{it} + \tau r(W_{it} - V_{it}) + (1-\tau)\pi(k_{it}, z_{it}, \eta_{it}) - d_{it}$$

Variables (moral hazard model):

- $V_{it-1}$: equity value at the end of period $t-1$
- $d_{z_{it}, \eta_{it}}$: state-contingent dividend payment
- $V_{z_{it}, \eta_{it}}$: state-contingent continuation equity value
- $p_{it}$: state-contingent payment to lender


#### Scaling to larger LE/MH grids

The current implementation target is a modest local baseline: TO is solved by full finite-action LP, while LE and MH use small finite menus of complete state-contingent contracts. This is sufficient for a clean conceptual implementation and for verifying timing, feasibility, and policy recovery. It is not intended to be paper-scale for LE/MH.

For larger LE/MH grids, the paper's LP-plus-constraint-generation logic applies. The master problem contains only an active subset of Bellman inequalities, each corresponding to a fixed feasible action. After solving the relaxed master LP, a separation oracle searches, state by state, for a feasible contract action whose Bellman RHS violates the current value function by more than tolerance. When such an action is found, the action is frozen (including any interpolation weights), the resulting linear Bellman inequality is added to the master LP, and the loop repeats until no violated constraints remain. The paper implements the action-search step with mixed-integer programming because the state-contingent action space is large; a scalable extension would add constraint generation and an optional separation oracle, likely using a stronger optimization backend such as Gurobi or CPLEX. These are not implemented in the current baseline.

## Structural Estimation and Model Comparison

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


I plot the fitted empirical policy functions $P^n(x_{it})$ as twoway slices of observable actions $y_{it}$ against states $x_{it}$. Each slice plot fix other states at sample median. These six panels can be directly compared to the Figure 2 and 3 in @nikolov2021. 

- Investment vs log size
- Future leverage vs current leverage
- Investment vs current leverage
- Payout vs profitability
- Future leverage vs profitability 
- Investment vs profitability

![TO fitted empirical policy functions (panel simulated from the solved TO model)](figures/bonus3-model-hk-data/to/to_empirical_policy_slices.png)

The figure shows the empirical policy function fitted on a panel simulated from the solved TO model: each panel plots one observable action against one observable state, with the red line the fitted (partial-dependence) policy and the points the simulated firm-years. The estimated slopes line up with standard theory and with the TO results in @nikolov2021: investment falls with firm size and is roughly flat-to-declining in leverage, payout and investment both rise with profitability, future leverage is strongly increasing in current leverage (leverage is persistent), and future leverage falls with profitability (more profitable firms lever less). Because these signs match the theoretical mechanisms and the patterns reported for the TO model in @nikolov2021, I treat this as a successful reproduction of the TO model's policy behavior. The figure is shown here as an illustrative example for the TO model; the same fitted coefficients are the auxiliary moments used in the indirect inference below.


### Indirect inference

The empirical policy function estimated above serves as the auxiliary model. The structural model is not estimated by matching individual firm outcomes one by one. Instead, for a candidate structural parameter vector $\beta$, we solve the model, simulate firm panels from the model, estimate the same empirical policy functions on the simulated data, and choose $\beta$ so that the simulated policy functions are close to the empirical policy functions estimated from real data.

Let $v_{it}\equiv (y_{it},x_{it})$ denote one observation in the real firm panel, where $i$ indexes firms and $t$ indexes time. The full real-data panel is

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

<!-- TODO: complete the "Variance estimation" paragraph (standard errors / J-test for the indirect-inference estimator). The analogous sandwich-SE and overidentification-test formulas are in the SMM appendix (#sec-smm-appendix). -->


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

The table reports the parameter estimates from indirect inference on the TO Model, solved by LP method on grid with density ($[k,b,z]=[15,12,5]$). To conduct the minimization, I use Nelder-Mead as global method to search for the optimal parameter vector. The system is over-identified with 27 moments and 8 parameters. I use identity weight matrix to construct the standard SMM variance. Unfortunately, because simulated moment covariance is near-singular, I'm unable to conduct a valid over-identification (J) test for this run. The full estimation took about 2 hours on Apple M1. Reference Estimate reported in the second column are the initiation value, calibrated to the estimates on Large US public-listed firms reported by @nikolov2021.


One notable estimate is the large capital adjustment cost $\hat \psi = 1.036$, which is not obviously realistic. This is consistent with the fact, shown earlier, that the Compustat-HK firms on average have a much lower investment rate compared with Compustat-US firms. The $i/k$ moments are poorly matched (see appendix), and it is worth examining whether the current measurement for investment needs to be refined if, for example, the variables are defined and constructed differently in Hong Kong due to different regulation.


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

The figure below reproduces the empirical policy comparison plot from Figure 2 and 3 of @nikolov2021. Each panel is a partial-dependence slice: it varies one observable state, holds the others at their sample median, and plots the fitted empirical policy. The black line is fit on the real Hong Kong panel and the red line is fit on a panel simulated from the solved TO model at the estimated parameters. 

![TO indirect-inference policy overlay: the empirical policy fit on the real Hong Kong panel (black) versus the same regressions refit on a panel simulated from the solved TO model at the estimated parameters (red)](figures/bonus3-model-hk-data/to/to_ii_policy_overlay.png)

Because intercepts are dropped, a good match means the two lines share the same slope and curvature, not the same level. Here the lines clearly diverge: the simulated policy sits well above the data in most panels, and the shapes agree only in part (for instance, future leverage rises with current leverage in both, but the model line is much steeper and higher). So the estimator could not bring the model's policy close to the data from the current preliminary results.



**Why the fit is poor.** I rank the likely causes from most to least actionable.

1. **Model solve quality (most likely the main cause).** The reported run solves the TO model on a coarse grid ($k=15$, $b=12$, $z=5$) for tractability, because indirect inference re-solves the model once per candidate parameter vector, hundreds of times in total. A coarse grid turns the policy into a crude step function over a handful of states, so the simulated panel carries little granular variation and the empirical policy refit on it is shaped by discretization rather than by economics. This is plausibly the dominant problem. The clear next step is to refine the grid (denser capital, debt, and shock points), which lowers grid approximation error and lets the simulation reproduce the smooth policy variation the regressions are trying to recover. Only once the solve is accurate can we trust the simulated moments.

2. **Observable and normalization disagreement.** Even with an accurate solve, the model and the data describe firm flows on different scales. The model normalizes every flow by capital $k$, while the Compustat observables normalize by total assets, so the two policies can differ in level for reasons no structural parameter can fix. I already drop each regression's intercept to absorb this level offset and match only slope and curvature. A more careful version is the firm-fixed-effects adjustment used in @nikolov2021: estimate firm fixed effects from the real panel, drop the intercept of the simulated empirical policy, then add the real-data firm effects back onto the simulated policy. The purpose is to strip out persistent, firm-specific level differences (size, accounting deflator, unmodeled heterogeneity) that the structural model was never meant to explain, so the estimator compares the part of the policy that is actually comparable, its shape. The problem it solves is keeping a nuisance level gap from looking like a structural misfit.

3. **A genuine model rejection (only after 1 and 2 are ruled out).** If the solve is accurate, the observables are aligned, and the simulated policy still cannot approach the data, then the fit is truly poor and the model is rejected on its own terms. Confirming this needs the formal model-fit test from @nikolov2021, the over-identification test on the moment gap. That test is expensive: building its variance requires re-solving the model many times to estimate the moment covariance and the sensitivity of the moments to the parameters. The current run could not even form a valid J statistic (identity weighting plus a near-singular moment covariance), so a formal rejection is not yet warranted. The honest reading is that causes 1 and 2 must be cleared first, and only then is the expensive formal test worth running.


```{=latex}
\appendix
```


# Appendix for Part I {#sec-solve}

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
  cost $\chi$: the denominator increases.
- **Tomorrow** ($a' = \pi_\theta(s')$): higher investment also raises
  $\chi' = 1 + \partial\psi'/\partial k''$ via the next-period adjustment
  cost: the numerator $m$ increases too.

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

- $a = \pi_\theta(s)$: gradients flow through the current policy.
- $a' = \pi_{\theta^-}(s')$: **gradients stopped**; weights are frozen for
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
learned value function network.  It builds on a modern RL algorithm developed by @xu2022. I adopt the core structure of windowed actor BPTT with on-policy continuation across windows, but replace the critic update with a one-step Bellman target (similar to the Deep Deterministic Policy Gradient (DDPG) method) to improve stability in economic environments.

**Core idea.**  The full $T$-step trajectory is divided into consecutive
windows of length $h$.  Within each window, the actor loss
backpropagates through $h$ exact dynamics steps, and a value function $V$
bootstraps the continuation beyond the window boundary.  Between
windows, the endogenous state carries forward (detached via
`stop_gradient`) so the trajectory remains on-policy.  This avoids the
gradient explosion/vanishing of full-trajectory BPTT while retaining
exact policy gradients through known, differentiable dynamics.

**Reward and Bellman normalization.** Unlike in standard RL environments, economic models typically have large reward and value scales, yet SHAC's default hyperparameters assume $O(1)$. To bring values into the right range and stabilize training, I rescale every reward by $\lambda_r = 1/|V^*|$ so that the critic learns values of $O(1)$. To obtain $V^*$, I use the environment's baseline steady-state value that can be obtained analytically. For example, in basic investment model I use the first-order approximation value $V^*=\hat{V}^{\text{term}}$ as described in the [LRM appendix](#sec-LRM). The scaling is applied uniformly to $r$ in both the actor loss and the critic Bellman target, which makes the rescaling mathematically equivalent to the unscaled algorithm: the critic learns $\lambda_r V^\pi$ in place of $V^\pi$, and the optimal policy is unchanged. Because $\lambda_r$ is a numerical preconditioning factor (i.e., multiplied by a constant) rather than part of the algorithm's logic, I omit it from the algorithm summary below.

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

This defect is further worsen by two mechanics: (1) FOC loss is much smaller than Bellman error and other losses, so early training typically focuses on minimizing $\mathcal{L}^{\text{BR}}$ and "ignoring" $\mathcal{L}^{\text{FOC}}$, which lead to a self-consistent Bellman for any arbitrary policy weight $\theta$. (2) Bootstrap estimate of $V_\phi$ is lower than the true $V^*$ due to NN initialization around zero. 

Although $\mathcal{L}^{\text{FOC}}$ can provide correct gradient signals for $\theta$, it is not sufficient to ensure the converged $\pi_\theta \approx \pi^*$. In practice, the BRM training usually plateau at a small loss where the NN find an arbitrary pair of ($\theta, \phi$) that satisfies on-policy Bellman but only weakly satisfies the FOC. 

**Potential solution: warm-start value network.** I find that warm-start the value NN $V_\phi$ using a supervised regression can help training to learn the correct "shape" of the optimal policy, but the solution remains biased and the training is unstable compared with other methods. The idea is use a baseline closed-form $\hat{V}$ as regression label to pre-train $V_\phi$, so that in BRM training the initial $V_\phi$ already captures the correct "shape", in such case, the algorithm is more likely to converge (but it is still not guaranteed).

**Target Network.** As in the ER method, the value network $V_\phi(s'_{i,m})$ introduces a recursive gradient dependency: the SGD update to $\phi$ changes both the current-state evaluation $V_\phi(s_i)$ and the target $V_\phi(s'_{i,m})$ simultaneously. @maliar2021 do not address this; the actor-critic method in the [SHAC appendix](#sec-SHAC) resolves it via target networks and separated updates.

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


## Implementation Details {#sec-impl}

*Supporting material for Chapter 1 (Solving Dynamic Models).*

This appendix describes architecture-level implementation choices common to all deep-learning solvers (LRM, ERM, BRM, SHAC) in the codebase. The choices are not method-specific and are unchanged across solver classes.

### Input Normalization {#sec-impl-norm}

State variables in economic models span orders of magnitude (capital in the hundreds, log-productivity near zero, interest rates as fractions). Without normalization, the first-layer gradient scales with raw feature variances and the optimizer cannot make balanced progress across features. I apply a per-feature **static Z-score** to the input layer:

$$\hat{x}_d = \frac{x_d - \mu_d}{\sigma_d + \varepsilon},$$

where $\mu_d, \sigma_d$ are computed once from the full training dataset before any gradient steps and held fixed throughout training. No hidden-layer normalization (BatchNorm, LayerNorm) is applied.

Statistics for the exogenous component $s^{\text{exo}}$ are fit on all $N \times T$ trajectory samples to capture the AR(1) ergodic distribution. Statistics for the endogenous component $s^{\text{endo}}$ are fit on the $N$ initial states drawn uniformly over the bounded state space. The normalizer does not need to be ergodically exact: every state visited during training falls within the bounded region by construction, so its purpose is to map inputs to an $O(1)$ range that conditions the first-layer gradient, nothing more. Online Z-score normalizers common in RL add no information here because the full dataset is available before training begins.

### Hidden-Layer Activation {#sec-impl-activation}

The hidden layers use the **SiLU** (Sigmoid-weighted Linear Unit, also known as Swish) activation:

$$\mathrm{SiLU}(h) = h \cdot \sigma(h), \qquad \mathrm{SiLU}'(h) = \sigma(h)\bigl(1 + h(1 - \sigma(h))\bigr).$$

ReLU is the standard alternative but has zero gradient for $h < 0$. Any neuron whose pre-activation is negative for all training samples never recovers, a "dead neuron". With centered inputs (after the static Z-score above), roughly half of pre-activations are negative on average, so the dead-neuron risk is concrete rather than theoretical. SiLU's gradient is nonzero everywhere, eliminates dead neurons, and is smooth, which matches the smooth, concave objectives typical in economic models.

### Output Head Transformation {#sec-impl-output}

The policy network outputs a continuous action constrained to box bounds $[a_{\min}, a_{\max}]$ (e.g., investment $I \in [I_{\min}, I_{\max}]$). The standard RL choice is a $\tanh$ squashing function. I instead use a **linear output head followed by clipping**:

$$\hat{y} = w^\top \mathbf{a}^L + b, \qquad a = \mathrm{clip}(\hat{y},\, a_{\min},\, a_{\max}).$$

The motivation is gradient quality near the bounds. For $\tanh$ (or any differentiable bijection $\mathbb{R} \to (a_{\min}, a_{\max})$), $\partial a / \partial \hat{y} \to 0$ as $a$ approaches either bound, a topological necessity for a bounded smooth function. In standard RL benchmarks the optimal policy is rarely near the bounds and the saturation is harmless. In economic models the optimal policy is often near the upper bound when productivity is high, and the per-period reward also has diminishing marginal returns in that region. The two effects compound: $\partial \mathcal{L}/\partial \theta$ becomes small in exactly the region where the policy needs the most learning signal, and the trained policy systematically deviates from the analytical benchmark at the boundaries.

The linear-plus-clip head avoids this. Inside the feasible region the output is identity and $\partial a / \partial \hat{y} = 1$ uniformly; outside, the gradient is zero but the action is correctly snapped. The interior gradient is independent of distance to the boundary, so the policy learns boundary-pushing behavior without saturation. The same design is used by TD-MPC2's MPPI planner, PPO with clipped actions, and DDPG with bounded action spaces.

### Reproducibility and Seeding {#sec-impl-seeds}

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

A separate **strict mode** (`strict_reproducibility=True`) additionally pins down kernel-level non-determinism inside TensorFlow itself (parallel reductions, GPU / Metal ops); it is reserved for strict replication and debugging.

# Appendix for Part II

### Generalized Method of Moments (GMM) {#sec-gmm-appendix}

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

### Simulated Method of Moments (SMM) {#sec-smm-appendix}

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

## Extended Kalman Filter Recursion (Bayesian Likelihood) {#sec-ekf-appendix}

This appendix gives the full EKF recursion for the Bayesian likelihood of the basic model summarized in the Bayesian chapter. The latent state is scalar, $x_{i,t} \equiv \log z_{i,t}$; $m_{t|s}, V_{t|s}$ denote the conditional mean and variance of $x_{i,t}$ given $y_{i,1:s}$ (firm index suppressed). The two observation equations are $y^{(1)}_{i,t} = x_{i,t} + \alpha \log k_{i,t} + \mu_\eta + \eta_{i,t}$ (linear in $x$) and $y^{(2)}_{i,t} = g(x_{i,t}, k_{i,t}; \beta) + \mu_{\xi^k} + \xi^k_{i,t}$ (nonlinear in $x$ through $g$), with $g(x, k; \beta) := \log \varphi_k(\exp x, k; \beta)$.

**When the EKF is appropriate.** The EKF replaces $g(x; \beta)$ with its first-order Taylor expansion around the predicted latent mean,
$$
g(x; \beta) \;\approx\; g(m_{t|t-1}; \beta) + H_2(t) \cdot (x - m_{t|t-1}), \qquad H_2(t) := \frac{\partial g}{\partial x}\bigg|_{x = m_{t|t-1}}.
$$
The Taylor remainder is of order $\tfrac{1}{2} |g''(m_{t|t-1})| \cdot V_{t|t-1}$, so the linearization is accurate when *either* the policy is close to linear in $x$ (small $|g''|$) *or* the predictive uncertainty is small (small $V_{t|t-1}$, i.e. the data anchors the latent state tightly). Two regimes apply here.

- **Closed-form policy.** $g(x; \beta) = \rho (1-\alpha)^{-1} x + \kappa(\alpha, \sigma_\varepsilon)$ is **globally linear in $x$**, so $H_2 = \rho/(1-\alpha)$ is a constant, the Taylor expansion is exact, and EKF reduces to standard Kalman with zero linearization error. I use this as "ground-truth" for validation.
- **Neural surrogate policy.** $g$ is the cached neural network $\varphi_\theta$ that is pre-trained and used to approximate the optimal policy function. Linearity holds only approximately; the surrogate's slope against $\log z$ is checked offline via `diagnose_nn_linear_slope` and the residual variance $\sigma_{\xi^k}^2$ absorbs whatever gap survives.

**Linearized state-space form.** Stack the two equations at the linearization point:
$$
y_{i,t} \;=\; H_t \, x_{i,t} + d_{i,t} + \epsilon_{i,t}, \qquad \epsilon_{i,t} \sim \mathcal{N}(0, R),
$$
with
$$
H_t = \begin{pmatrix} 1 \\ H_2(t) \end{pmatrix}, \quad
d_{i,t} = \begin{pmatrix} \alpha \log k_{i,t} + \mu_\eta \\ g(m_{t|t-1}, k_{i,t}; \beta) - H_2(t)\,m_{t|t-1} + \mu_{\xi^k} \end{pmatrix}, \quad
R = \begin{pmatrix} \sigma_\eta^2 & 0 \\ 0 & \sigma_{\xi^k}^2 \end{pmatrix},
$$
and AR(1) state transition $x_{i,t+1} = \rho \, x_{i,t} + \sigma_\varepsilon \, \nu_{i,t+1}$, $\nu \sim \mathcal{N}(0, 1)$.

**Initialize** $m_{1|0} = 0$, $V_{1|0} = V_0 = 10$.

**For $t = 1, \ldots, T$:**

1. **Predict the state** (AR(1) transition; at $t=1$ use initialization):
$$
m_{t|t-1} = \rho \, m_{t-1|t-1}, \qquad V_{t|t-1} = \rho^2 V_{t-1|t-1} + \sigma_\varepsilon^2.
$$

2. **Evaluate the policy and its Jacobian** at $x = m_{t|t-1}$, $k = k_{i,t}$:
$$
g_t := g(m_{t|t-1}, k_{i,t}; \beta), \qquad H_2(t) := \partial g / \partial x \big|_{x = m_{t|t-1}}.
$$
Closed-form: $H_2(t) = \rho/(1-\alpha)$ analytically. NN surrogate: $H_2(t)$ via autodiff through the network.

3. **Innovation** (observed minus predicted):
$$
\nu_t = \begin{pmatrix} \nu_1 \\ \nu_2 \end{pmatrix} = \begin{pmatrix} y_{i,t}^{(1)} - (m_{t|t-1} + \alpha \log k_{i,t} + \mu_\eta) \\ y_{i,t}^{(2)} - (g_t + \mu_{\xi^k}) \end{pmatrix}.
$$

4. **Innovation covariance** $S_t = H_t V_{t|t-1} H_t^T + R$ (closed-form $2 \times 2$):
$$
S_t = \begin{pmatrix} V_{t|t-1} + \sigma_\eta^2 & H_2(t)\,V_{t|t-1} \\ H_2(t)\,V_{t|t-1} & H_2(t)^2 V_{t|t-1} + \sigma_{\xi^k}^2 \end{pmatrix}.
$$

5. **Likelihood contribution** (bivariate Gaussian):
$$
\log p(y_{i,t} \mid y_{i,1:t-1}, \beta) = -\tfrac{1}{2}\!\left[2 \log(2\pi) + \log |S_t| + \nu_t^T S_t^{-1} \nu_t\right].
$$

6. **Kalman gain and update.** $K_t = V_{t|t-1} H_t^T S_t^{-1}$ (a $1 \times 2$ row); then
$$
m_{t|t} = m_{t|t-1} + K_t \nu_t, \qquad V_{t|t} = V_{t|t-1} (1 - K_t H_t).
$$

Sum likelihood contributions across $t$ and $i$ to obtain $\log p(Y \mid \beta)$.

# Appendix for Bonus Question 3

## LP Method for TO Model

The baseline LP solutions (TO, LE, MH) use **NumPy** for array operations, **SciPy sparse matrices** for constraint assembly, and **`scipy.optimize.linprog` with the `'highs'` backend** for the master LP solve. TensorFlow is not used: there is no neural-net training, no autodiff requirement, and no GPU benefit for the modest finite-grid baseline. The workload is dominated by grid construction, finite-action enumeration or action filtering, sparse matrix assembly, and LP solving, all of which NumPy and SciPy handle with mature, well-tested implementations.

**LP state interpretation.** The LP uses the paper's recursive timing. A state $(k,b,z)$ (or $(k,V,z)$ in MH) is the inherited end-of-period state: $k$ is the capital level from the previous choice, $b$ is the outstanding promised balance (or $V$ is promised equity value), and $z$ is the current public persistent shock. At this state the firm chooses the next capital level and financing contract, denoted $(k',b')$ in TO, state-contingent $(b'_{z',\eta'},p_{z',\eta'})$ in LE, and state-contingent $(V'_{z',\eta'},d_{z',\eta'})$ in MH. The Bellman target then evaluates next-period shocks $(z',\eta')$, operating cash flow with the chosen capital $k'$, repayment/contract constraints, and continuation value at the chosen next state.

#### Primitives and grids

State $S = K \times B \times Z$ with discrete grids $K = \{k_1, \dots, k_{n_k}\}$, $B = \{b_1, \dots, b_{n_b}\}$, $Z = \{z_1, \dots, z_{n_z}\}$. Action $A = K \times B$ (choice of next-period $(k', b')$).

State $(k, b, z)$: capital in place, outstanding debt, current persistent shock. I.i.d. shock $\eta \in \{+\bar\eta, -\bar\eta\}$ with $P(\eta = +\bar\eta) = \kappa$. Persistent transition $Q(z' \mid z)$. Discount $\beta = 1/(1+r)$.

Model functions:
$$\pi(k, z, \eta) = (z + \eta)k^\alpha - f, \qquad \Psi(k', k) = \tfrac{\psi}{2}\big((k' - (1-\delta)k)/k\big)^2 k$$

Parameters: $\tau, \alpha, f, \delta, \psi, \xi, r, \kappa, \bar\eta$.

#### Step 1: Pre-compute lender pricing $\Delta(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$

For every candidate chosen capital $k_{\text{choice}} \in K$, inherited debt $b_{\text{old}} \in B$, and inherited persistent shock $z_{\text{old}} \in Z$, compute the risky-debt premium that satisfies the lender break-even condition:

$$\sum_{z'} Q(z' \mid z_{\text{old}}) \sum_{\eta'} P(\eta')\Big[(1+r+\Delta)(1 - \mathcal{I}_D') + \tfrac{\xi(1-\delta)k_{\text{choice}}}{b_{\text{old}}}\mathcal{I}_D'\Big] = 1+r$$

with

$$\mathcal{I}_D' = \mathbf{1}\big\{(1-\tau)\pi(k_{\text{choice}}, z', \eta') + (1-\delta)k_{\text{choice}} + \tau\delta k_{\text{choice}} \;<\; (1+(1-\tau)(r+\Delta))b_{\text{old}}\big\}.$$

**Important timing convention.** In the Bellman constraint for state $(k,b,z)$ and action $(k',b')$, the relevant premium on the outstanding debt $b$ is $\Delta(k',b,z)$, not $\Delta(k,b,z)$. The premium is priced at the time the old debt is issued, using the capital level chosen for the period in which that debt will be repaid.

**Finite-shock pricing solver.** Because $\mathcal{I}_D'$ changes with $\Delta$ on a discrete shock grid, the lender payoff is piecewise-linear with possible jumps, so naive bisection is unreliable. The solver instead:

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

Only the 3D premium table is stored; the indicator is recomputed on demand from the stored $\Delta(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$ and the analytic default condition, which saves memory.

#### Step 2: LP

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

#### Step 3: Recover the policy

After the LP returns $W^*$:
$$(k', b')^*(k, b, z) = \arg\max_{(k', b') \in A \text{ feasible}} \text{RHS}(k, b, z, k', b'; W^*)$$
where RHS is the same Bellman target as in Step 2, with $W^*$ plugged in for the continuation. Implied premium policy for the chosen next debt is $\Delta^*(k,b,z)=\Delta(k^{\prime *}(k,b,z), b^{\prime *}(k,b,z), z)$; within the Bellman RHS for repayment of inherited debt $b$, use $\Delta(k^{\prime},b,z)$.

#### Implementation notes

- **Step 1:** finite-threshold pricing over the $(k_{\text{choice}}, b_{\text{old}}, z_{\text{old}})$ grid, with explicit consistency checks for default sets and pricing residuals.
- **Step 2 LP construction:** NumPy broadcasting to build the $(n_k, n_b, n_z, n_k, n_b)$ Bellman RHS coefficient tensor and LL feasibility mask; assemble constraints as `scipy.sparse.csc_matrix`.
- **Step 2 LP solve:** `scipy.optimize.linprog` with `method='highs'`.
- **Step 3:** `np.argmax` over the RHS tensor with $W^*$ plugged in.

## LP Method for LE Model

#### Primitives and grids

State $S = K \times B \times Z$ with the same $K$ and $Z$ grids as 1.2 and a nonnegative debt grid $B = \{b_1,\dots,b_{n_b}\} \subset \mathbb{R}_+$ with $0 \in B$. Additional parameter: $\theta$ (collateral fraction). Shock $\eta$ and persistent shock $z$ are as in 1.2.

State $(k,b,z)$: inherited capital, inherited promised balance, and current persistent shock. There is no default risk and no premium $\Delta$ in LE. Pricing is imposed by the risk-neutral break-even constraint (5).

#### Minimal finite-action baseline

To keep the baseline a true LP, use a **finite contract-action menu**. For each current state $(k,b,z)$ and candidate next capital $k' \in K$, a contract action is a fixed collection

$$a = \{b'_{z',\eta'}, p_{z',\eta'}\}_{z' \in Z,\eta' \in N}$$

where each $b'_{z',\eta'} \in B$ and each $p_{z',\eta'} \in P$, with $P=\{p_1,\dots,p_{n_p}\}\subset \mathbb{R}_+$ a nonnegative payment grid. Because the contract components are fixed before a Bellman inequality is added, the continuation terms $W(k',b'_{z',\eta'},z')$ enter linearly with known coefficients.

Do **not** include convex-combination weights such as $\lambda_j W(k',b_j,z')$ as free variables in the master LP. That would be bilinear, not linear. Continuous-state interpolation can be added later only through constraint generation, where interpolation weights are fixed constants when each Bellman inequality is added.

#### Step 1: Build feasible contract-action lists

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

This brute-force enumeration is intended for small grids only. A faster version can replace enumeration with constraint generation or an auxiliary search routine, but the master LP must still receive only fixed actions.

#### Step 2: Master LP

**Variables.** $W(k,b,z)$ for every $(k,b,z)\in S$. Count: $n_k n_b n_z$.

**Objective.**
$$\min_W \sum_{(k,b,z)\in S} W(k,b,z).$$

**Bellman constraints.** For every state $(k,b,z)$ and every feasible fixed action $(k',a)$:

$$W(k,b,z) \geq \frac{1}{1+r}\bigg[-k' + (1-\delta)k' - \Psi(k',k) + \tau\delta k' + \tau r b + \sum_{z',\eta'} Q(z'\mid z)P(\eta')\Big((1-\tau)\pi(k',z',\eta') + W(k',b'_{z',\eta'},z')\Big)\bigg].$$

The entire deterministic term is inside the discount factor $1/(1+r)$, matching the paper's LE Bellman equation.

Constraint count depends on the number of feasible fixed contract actions. This can grow quickly with $n_z n_\eta$, so the minimal implementation should start with small grids.

#### Step 3: Recover the policy

After the LP returns $W^*$, choose the feasible fixed action that maximizes the RHS at each state:

$$(k',a)^*(k,b,z)=\arg\max_{(k',a)\in\mathcal{A}_{LE}(k,b,z)} \text{RHS}_{LE}(k,b,z,k',a;W^*).$$

The policy consists of $k'^*(k,b,z)$ and the associated state-contingent contract $\{b'^*_{z',\eta'},p^*_{z',\eta'}\}$.

#### Implementation notes

- **Baseline:** enumerate finite contract actions on small $B$ and $P$ grids, filter by break-even, limited liability, collateral, and sign restrictions, then assemble the sparse LP.
- **Correctness condition:** each Bellman inequality must correspond to a fixed feasible action. The continuation coefficient on each $W(k',b_j,z')$ is therefore a known probability weight, not a decision variable.
- **Scaling upgrade:** if enumeration becomes too large, use constraint generation. Given a current value vector $W^{(m)}$, solve a separate action-search problem for each state, freeze the selected action, add the corresponding linear Bellman inequality to the master LP, and repeat until no violated constraints remain.

## LP Method for MH Model

#### Primitives and grids

State $S = K \times \mathcal{V} \times Z$ with discrete grids $K = \{k_1, \dots, k_{n_k}\}$, $\mathcal{V} = \{V_1, \dots, V_{n_V}\}\subset\mathbb{R}_+$ with $V_1 = 0$, and $Z = \{z_1, \dots, z_{n_z}\}$. Shock $\eta \in \{+\bar\eta, -\bar\eta\}$ with $P(\eta = +\bar\eta) = \kappa$. Persistent transition $Q(z' \mid z)$. Additional parameter: $\lambda$ (diversion fraction).

State $(k,V,z)$: inherited capital, promised equity value, and current public persistent shock.

Discount in the firm Bellman: $1/(1 + (1-\tau)r)$ (firm-side, with the tax shield on the debt component already embedded). Discount in promise-keeping: $1/(1+r)$ (equity-holder side, no debt tax shield).

Under the pre-tax $\pi$ convention used in this document, the diversion function is

$$\mathcal{D}(k', z', \eta', \hat\eta') = \lambda(1-\tau)\big[\pi(k', z', \eta') - \pi(k', z', \hat\eta')\big].$$

#### Minimal finite-action baseline

To keep the baseline a true LP, use a **finite contract-action menu**. For each current state $(k,V,z)$ and candidate next capital $k'\in K$, a contract action is a fixed collection

$$a=\{V'_{z',\eta'},d_{z',\eta'}\}_{z'\in Z,\eta'\in N},$$

where each $V'_{z',\eta'}\in\mathcal{V}$ and each $d_{z',\eta'}\in D$, with $D=\{d_1,\dots,d_{n_d}\}\subset\mathbb{R}_+$ a nonnegative dividend grid. Because the contract components are fixed before a Bellman inequality is added, the continuation terms $W(k',V'_{z',\eta'},z')$ enter linearly with known coefficients.

Do **not** include convex-combination weights such as $\mu_j W(k',V_j,z')$ as free variables in the master LP. That would be bilinear, not linear. Continuous interpolation can be added later only through constraint generation, where interpolation weights are fixed constants when each Bellman inequality is added.

#### Step 1: Build feasible contract-action lists

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

This minimal baseline is intended for small grids only. A faster version can replace enumeration with constraint generation or an auxiliary search routine, but the master LP must still receive only fixed actions.

#### Step 2: Master LP

**Variables.** $W(k,V,z)$ for every $(k,V,z)\in S$. Count: $n_k n_V n_z$.

**Objective.**
$$\min_W \sum_{(k,V,z)\in S} W(k,V,z).$$

**Bellman constraints.** For every state $(k,V,z)$ and every feasible fixed action $(k',a)$:

$$W(k,V,z) \geq \frac{1}{1+(1-\tau)r}\bigg[-k' - \Psi(k',k) + (1-\delta)k' + \tau\delta k' - r\tau V + \sum_{z',\eta'} Q(z'\mid z)P(\eta')\Big((1-\tau)\pi(k',z',\eta') + W(k',V'_{z',\eta'},z')\Big)\bigg].$$

#### Step 3: Recover the policy

After the LP returns $W^*$, choose the feasible fixed action that maximizes the RHS at each state:

$$(k',a)^*(k,V,z)=\arg\max_{(k',a)\in\mathcal{A}_{MH}(k,V,z)} \text{RHS}_{MH}(k,V,z,k',a;W^*).$$

The policy consists of $k'^*(k,V,z)$ and the associated state-contingent contract $\{V'^*_{z',\eta'},d^*_{z',\eta'}\}$.

Payment to the lender is recovered after the LP solve as the residual from the resource constraint. Under the pre-tax $\pi$ convention, for each realized $(z',\eta')$:

$$p^*_{z', \eta'} = -k'^* - \Psi(k'^*, k) + (1-\delta)k'^* + \tau\delta k'^* + \tau r\big(W^*(k'^*, V'^*_{z', \eta'}, z') - V'^*_{z', \eta'}\big) + (1-\tau)\pi(k'^*, z', \eta') - d^*_{z', \eta'}.$$

This recovered payment is for accounting and simulated policy measurement. It should not be reintroduced into the master LP as an additional constraint involving $W-V$, because the Bellman equation already embeds the tax-shield logic through the adjusted discount factor and the $-r\tau V$ term.

#### Implementation notes

- **Baseline:** enumerate finite contract actions on small $\mathcal{V}$ and $D$ grids, filter by promise keeping, IC, and limited liability, then assemble the sparse LP.
- **Correctness condition:** each Bellman inequality must correspond to a fixed feasible action. The continuation coefficient on each $W(k',V_j,z')$ is therefore a known probability weight, not a decision variable.
- **Scaling upgrade:** if enumeration becomes too large, use constraint generation. Given a current value vector $W^{(m)}$, solve a separate action-search problem for each state, freeze the selected action, add the corresponding linear Bellman inequality to the master LP, and repeat until no violated constraints remain.


# References {.unnumbered}

::: {#refs}
:::
