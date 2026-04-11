---
title: "Quantitative Corporate Finance: Methods and Applications"
number-sections: False
date: 2026-04-20
bibliography: references.bib
thanks: Email - [wxuan.econ@gmail.com](mailto:wxuan.econ@gmail.com) or [zxwang13@student.ubc.ca](mailto:zxwang13@student.ubc.ca).
---

# Abstract
This report examines different quantitative methods to solve and estimate structural corporate finance models. To solve for dynamic models, I implement and evaluate the deep learning methods proposed in @maliar2021, and benchmark their performance against the standard iteration methods and linear programming methods. I also develop a new solution method based on the short-horizon actor critic (SHAC) method from reinforcement learning [@xu2022], and show that SHAC is effective in solving models where @maliar2021's methods are either incorrect or infeasible (e.g., lack of closed-form Euler equation). To estimate the structural parameters from real-world data, I implement generalized method of moments (GMM), simulated method of moments (SMM), and Monte Carlo Markov Chain (MCMC) method. I apply these methods to a set of canonical models of firm's optimal financing structure and discuss my findings [@strebulaev2012, @cronqvist2024, @nikolov2021].

# Introduction

To keep this report clear and concise, I focus on presenting the findings and the direct answers to the interview questions. In appendix, I provide a more detailed description of the models, solution methods (algorithms), additional results, and other implementation details.

# Part I. Solving Dynamic Models

The corporate finance model considered in this report are generally represented as a dynamic programming problem with the following features:
- Discrete-time
- Continuous state and action spaces, 
- Policy function is deterministic and known
- State transition function with random noise is known

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

A full sequence of actions and states is defined as a **trajectory** or **rollouts** $\tau$:
$$ \tau = (s_0,a_0,s_1,a_1,\dots)$$
where the initial state $s_0$ is randomly sampled from some distribution $p_0$. The state transition is given by:
$$ s_{t+1}=f(s_t,a_t,\epsilon_t) $$
where $\epsilon_t$ is a random noise (e.g., productivity shock) but the function $f$ is deterministic. Note that this is different from a stochastic transition function in RL where $s_{t+1}$ is a draw from a distribution $s_{t+1} \sim P(\cdot|s_t,a_t)$.

The **reward function** $r(s,a)$ is assumed to be known exactly and it maps current state and actions $(s,a)$ to a scalar value. In corporate finance, this is typically firm's cash flow. The **discounted lifetime reward** over an infinite-horizon trajectory is summarized as:
$$
R(\tau) = \sum^\infty_{t=0} \gamma^t \cdot r(s_t,a_t)
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

## Solution Methods

The solution to the model is given by the optimal policy function $\pi^*$ that maps states to actions. Optionally, solution also include the optimal state-value function $V^*(s)$.  

I implemented four main solution methods in Python and Tensorflow:
1. Value and policy function iteration (VFI/PFI)
2. Lifetime reward maximization (LRM)
3. Euler residual minimization (ERM)
4. Short horizon actor critic method (SHAC)
5. Nested value function iteration (specific to the risky debt model)

VFI and PFI are the classical discrete dynamic programming solvers.  They
discretize the continuous state and action spaces onto finite grids,
estimate Markov transitions from data, and iterate the Bellman operator
to convergence.  The resulting value function and policy are exact on the
grid (up to discretization error) and serve as ground-truth benchmarks
for the NN-based methods. See @sec-VFI for implementation details.

LRM and ERM are deep learning methods introduced by @maliar2021. My implementation augmented LRM by correcting for the finite-horizon terminal value bias. I also refined the ERM method by adding a separate, slowly-updated target policy network to stablize training.

SHAC is a "new" method I implemented based on a revision of the reinforcement learning (RL) algorithm developed by @xu2022. It inherits the robustness of the LRM method via multiple period rollout of rewards, and it trained a value function network to precisely approximate the terminal value missing in LRM.

In addition, I implemented and tested the Bellman Residual Minimization (BRM) method proposed in @maliar2021 and @fernandez-villaverde2024. I find that this method is generally unstable and convergence is not guaranteed. Detailed disgnostics are discussed in section.

## Validation of Solution

To verify the effectiveness and correctness of these methods, I directly benchmark all methods against an **analytical solution** of optimal policy in the basic investment model (frictionless). I measure the effectiveness of my solution using three measures:

1. Mean Absoluate Error (MAE) against analytical solution (when applicable)
2. Euler equation residual on separate validation dataset (when applicable)
3. Lifetime reward on the on separate validation dataset 

The MAE is a direct proof of correct solutions when anlytical optimal policy is available (e.g., frictionless basic investment model). When the model exist a closed-form Euler equation (e.g., frictional basic investment model), the second-best measure is to compute the mean Euler equation error on a separate fixed validation dataset. Finally, if both #1 and #2 are infeasible, I use the sum of discounted reward over a long enough finite horizon (e.g., 200 periods) to compare across methods. This is a standard evaluation metric used in the RL literature.

## Issues in Neural Network Architecture and Training

Standard VFI and PFI methods are "simple" and robust because its convergence is guaranteed by contraction mapping theorem. In contrast, I find that there are many details that are critical for Neural Network (NN) based methods to work, and these practical issues are often omitted by the higher-level algorithm summary in original papers [@maliar2021].

Table below summarizes the issues specific to the three methods introduced by @maliar2021:

| Method | Major Issue | Minor Issue | Usability 
|---|---|---|---|
| Euler Residual Min | None | Single policy network is unstable. Solved by adding a target policy network | Fast and robust for production, but requires existence of a closed-form Euler equation
| Lifetime Reward Max | Terminal value truncation bias | Long-horizon backpropagation through time is costly | Can be used as rough baseline, but not ideal for production
| Bellman Residual Min | Can easily converge to "wrong" but self-consistent fixed point | Conflicting gradients due to scale mismatch of loss functions; Require existence of closed-form first order condition | High-risk and strongly rejected for production

In addition, I rank the general issues (shared by all methods) based on their importance in practice:

| General Issues | Description | Solutions | Results | 
|---|---|---|---|
| Smooth and differentiable reward and dynamics | This is a fundamental prerequisite for gradient-based training to work | Kinks can still be handled, but discrete choice or jump discontinuities can only be approximated with error  | For the basic model with fixed adjustment cost, NN-based methods cannot learn the inaction regions; Soft-surrogate suffers nontrivial approximation error; VFI/PFI is strictly better
| Input Normalization | Raw data are measured in level and large units can easily de-stablize training  | Normalize input to z-score and re-scale it back to economic levels as NN output head | Hidden layer only see normalized inputs and is agnostic to environment
| Network output head | Sigmoid, Tanh, and other activation can suppress gradient and prevent learning at extreme values | Use raw linear (no activation function) with affine transformation | Gradient is uniformly "strong" across state/action space; Output variable converted back to original unit
| Hidden layer activation | For economic models, ReLU is not stable, Sigmoid and Tanh cause vanishing gradient| SiLU `swish` always perform better in practice | Gradient is stable and nonzero
| Full reproducibility | Comparison across methods should be fair and fully reproducible | Separate data generation and training; Full schedule of random number generator (RNG)| All methods are trained on exactly the same fixed dataset and results are fully controlled by master seeds
| Convergence metric | Objective and loss function are NOT the correct metrics for the quality of solution | Measure effectiveness of learned policy in a separate validation dataset | Avoid overfitting, enable early stopping based on same criteria, fair comparison across methods

## Application to Risky Debt Model

To solve the risky debt model, I implemented a nested VFI algorithm and a NN-based deep learning methods. I find that nested VFI is still the fastest and most robust method for this model, while SHAC is the only deep learning method that could work but is systematically biased due to the model's structure. I present the results and c  in section X.
Below is a summary of my findings.

First, I find that all three of @maliar2021's methods **cannot** be applied to solve the risky debt model described in @strebulaev2012 [section 3.6]. The underlying reason is that the risky debt model involves a "nested fixed point" problem as stated : solving the model requires knowing the value function in order to calculate the default states and thus the interest rate, but one needs to know the interest rate to calculate value function.

More specifically, both the lifetime reward and Euler residual method do not have neural network to approximate the value function. The lifetime reward method drops the value function entirely by design. In principle, Euler residual method allow one to derive the value function from the converged optimal policy, but the risky debt model does not have a closed-form Euler equation so it is infeasible. Lastly, the Bellman residual method does have a separate value network but this method has fundamental defects and is not usable for reasons discussed above. 

One promising solution methods to this model is Short-Horizon Actor Critic (SHAC), which include a separate value network training and does not require closed-form first order conditions. However, I benchmarked the solution of SHAC to nested VFI and find that SHAC systematically learned a more conservative policy (lower leverage-to-asset ratio). This is a consequence of the model's structure and is common to all actor-critic methods because 

- During training, value network $V$ is initially inaccurate and gradually improving
- But the initially biased $V$ directly determine the default set and interest rate, which affect the policy learning and the target $V$ in next iteration
- Both policy and value network converged to a self-consistent but over-pessimistic (low leverage) or over-optimistic (high leverage) equilibrium, depending on the value network initiation

In practice, my experiments show that SHAC solutions are usually pessimistic (low leverage) because initial $V$ network tends to underestimate $\partial V/\partial b$ and lead to conservative policy (low leverage and less default states), which leads to a more conservative target value network in next iteration and self-reinforcing policy updates. 

There are two promising fixes: (1) use stochastic policy method instead of a deterministic policy to explore off-policy and with scalar cash flow (reward) acting as a score function. This allows the training to explore high-leverage states (default) off-policy. (2) switch to a standard trade-off model as in @nikolov2021 where default states (and endogenous interest rate) does not depend on the value function and can be written down analytically only in terms of current states.

However, stochastic policy methods are usually not sample-efficient. For our model, nested VFI method is clearly faster and more robust. Fix (2) is a practical choice if we are willing to deviate from this version of the endogenous default model, but the standard VFI still performs better for this low-dimensional problem with few states.

# Part II. Structural Estimation

## Validation of GMM and SMM

I implemented and tested both GMM and SMM methods to structurally estimate model parameters. I measure effectiveness using the basic investment model because it is computationally cheaper. The basic idea of the Monte Carlo (MC) validation is:

For MC replication $j=0,\dots, J$:
1. Set replication count $j=j+1$. Select a set of *true* parameters $\beta^*_j$, solve for the optimal policy $\pi^*(\cdot|\beta^*_j)$, use it to simulate a "target" panel dataset of $N$ i.i.d. firms over $T$ periods
2. Start with a random guess $\beta^0_j \neq \beta^*_j$, apply GMM or SMM to the target dataset, obtain a set of estimated params $\hat \beta_j$ and variance-covariance matrix.
3. Conduct t-test $H_0: \hat \beta_j = \beta^*_j$ and expect failure to reject the null. Conduct over-identifying test and verify if we fail to reject the hypothesis of model mis-specification.

When all $J$ replications completes, compute diagnostics including the average bias $\frac{1}{J}\sum_j(\hat \beta_j - \beta^*_0)$, Root Mean Square Error (RMSE), average rejection rate of over-identifying test, etc. The detailed diagnostics are defined in section .

I consider the implementation to be correct only if our MC replication can consistently estimate $\hat \beta_j$ close to the true $\beta^*_j$. The shock realization of replication is controlled by master seeds and are fully reproducible. 

For the actual application, we replace step 1 with one real-world "target" dataset such as the Compustat firm panel data, and we only apply Step 2-3 once.

There are two important implementation issues:
- GMM uses Euler equation to form the moment condition, so it does not require solving the model
- SMM typically requires re-solving the model for optimal policy in Step 1 for each candidate $\beta$ for evaluation. This is the main computational bottleneck. For this validation, I use the frictionless basic model with analytical solution to the optimal policy to avoid the cost. This validates the correctness of the entire SMM pipeline and separate potential errors of model solver (e.g. VFI/PFI) from estimation.
- Both GMM and SMM requires choosing appropriate global and local optimizer to find $\beta^*$ that minimizes the moment condition error. I discuss the optimizer choice in detailed.

## Applying SMM to the risky debt model

Applying GMM/SMM to the basic investment model allow us to identify four parameters: production function curvature, capital adjustment cost (smooth), persistence of AR(1) shock, and variance of AR(1) shock. If we further introduce costly external finance to the basic model as in section 3.3 of @strebulaev2012, we can use GMM/SMM to estimate the linear-qudractic cost parameters of equity issuance when cash flow is negative. 

From a pure estimation perspective, applying the full endogenous default model only add one additional parameter to be identified: the deadweight bankruptcy cost, i.e., the fraction of profit and equity that can be recoveried when default. All the other financial friction parameters can be estimated from a simpler model as in section 3.3 of @strebulaev2012. 

The main benefit of estimating the endogenous default model is that it introduces the trade-off between equity and debt, which is richer and more realistic. My SMM application uses the nested VFI as model solver and follows @hennessy2007costly in selecting the target moments. 

## Defects of the risky debt model and potential solutions

I focus on the defects of the model's core economic mechanisms and discuss their theoretical and practical (empirical) implications. To be fair and constructive, I do not discuss critique that are either too board or general, or those that require adding new features beyond the original focus of the model.

To clarify, the core mechanism of the risky debt model is that firm's financing decisions reflect (i) optimal investment under frictional adjustment cost; and (ii) the opportunity to exploit the tax shield benefit of debt (in the form of a one-period corporate bond). Lenders (bank) with rational expectation charge a risk premium on the yields of the corporate bond based on anticipated default probability. Default threshold is determined by the realization of next-period productivity shock conditional on current states and actions, for example, higher debt requires higher realization of future productivity to repay, and thus expand the default set (probability). This is anticipated by the lender and priced into the endogenously-determined risk premium. 

Given this core mechanism, I find four critical defects in the specific risky debt model presented in @strebulaev2012 [section 3.6]:

1. Timing of tax shield benefit create "always-default" strategy
2. Assumption of perfect managerial information 
3. Computational cost and weak identification
4. Wrong-signed equity issuance cost

Only 4 can be easily fixed, and 1-3 are structural defects that can only be "solved" by major revisions to the model and estimation strategy. 

### Timing of tax shield benefit

Defect #1 is problematic because the present value of tax shield benefit, $\frac{\tau \tilde{r} b'}{(1+\tilde{r})(1+r)}$ is obtained by firm upfront and is unconditional on the next-period solvent/default states (Equation 3.26). Since the model intentionally does not impose a borrowing limit, firm could exploit an optimal "always-default" strategy that borrow as much as possible and default in next period. The lender rantionalize this in pricing the interest rate $\tilde{r}\to \infty$, but the tax shield benefit is still large and positive: $$ \frac{\tau \tilde{r} b'}{(1+\tilde{r})(1+r)} \rightarrow \frac{\tau b'}{(1+r)} \gt 0 \quad \text{as} \quad \tilde{r}\to \infty \text{ and } b'\to \infty$$ 

This is confirmed empirically when any naive implementation with large upper bound $b_{\max}$ relative to the $k_{\max}$ will cause the optimal policy to converge to "borrow as much as possible then default" with $b'=b_{\max}$ and $k'\approx 0$. For model solve itself this can be mitigated with well-calibrated parameters and bounds, but the true risk is for SMM estimation when the optimizer re-solved the policy under different parameters and a non-trivial fraction of the parameter combinations will lead to this unintended strategy.

One simple solution is to use the same time schedule as in the trade-off model in @nikolov2021.

### Imperfect managerial information

Defect 2 is directly related to the critique by @deangelo2022: Can manager and lender precisely estimate the continuation value $V$ and default states given by $\{z: V(\cdot, z)\leq 0\}$? This is the key important assumption of the model: the endogenous default decision is a going-concern and manager would only default when the firm's continuation value is negative. The pricing of bonds is from a bargaining between the firm and the lender based on $\mathbb{E}_{z'|z} [V(k',b',z')]$ where there exist a critical value of shock $z'_d$ such that all $z'<z'_d$ are default states [@hennessy2007costly, Proposition 6]. However, if manager and lender cannot learn $V$ with any realistic precision, this core mechanism is broken.

@deangelo2022 [Section VI] reviews direct evidence of imperfect manager knowledge. There are two key takeaways. First, large-scale surveys of CFOs indicated that "most managers have nothing close to the knowledge assumed in extant dynamic capital structure models, which posit a complete understanding of investment opportunities and capital-market conditions over an infinite horizon" [@graham2022presidential]. Second, a number of studies have estimated a near-flat relationship between firm value and leverage, suggesting that "real-world managers are unable to pin down a uniquely optimal capital structure with any real precision" [@korteweg2010net]. 


---


# Solution Method Details

## Value and Policy Function Iterations {#sec-VFI}

Both solvers share the same setup phase and differ only in the iteration
step.

### Setup

Given an `MDPEnvironment` and a flattened training dataset:

1. **Build grids.**  Discretize the endogenous state, exogenous state,
   and action spaces into finite grids.  Each variable can have its own
   spacing rule (linear or log) via `env.grid_spec()`.
   For multi-dimensional problems, the per-variable 1-D grids are
   combined into Cartesian product grids.

2. **Estimate transitions.**  From the dataset's observed $(z, z')$
   pairs, estimate the Markov transition matrix $P(z' \mid z)$ by
   binning observations into the exogenous grid and counting transitions,
   smoothed with a Dirichlet prior. 

3. **Precompute tables.**  Evaluate the environment on every
   (state, action) grid combination to build two tables:
   - **Reward matrix** $R[i_z, i_k, i_a] = r(s, a)$ where
     $s = [k_{i_k},\, z_{i_z}]$ and $a = a_{i_a}$.
   - **Transition map** $T[i_z, i_k, i_a] = i_{k'}$ where $k'$ is
     computed via `env.endogenous_transition()` and snapped to the
     nearest endogenous grid point.

   After this step, the environment is never called again. The Bellman
   iteration operates entirely on precomputed tensors.

### VFI: Value Function Iteration

Initialize $V^{(0)}[i_z, i_k] = 0$ for all grid points.

**Repeat** for $n = 0, 1, 2, \ldots$:

$$V^{(n+1)}[i_z, i_k] = \max_{i_a} \left\{ R[i_z, i_k, i_a] \;+\; \gamma \sum_{j_z} P[i_z, j_z] \cdot V^{(n)}[j_z,\, T[i_z, i_k, i_a]] \right\}$$

**Until** $\|V^{(n+1)} - V^{(n)}\|_\infty < \varepsilon$.

The optimal policy is the argmax:

$$\pi^*[i_z, i_k] = \arg\max_{i_a} \left\{ R[i_z, i_k, i_a] + \gamma \sum_{j_z} P[i_z, j_z] \cdot V^*[j_z,\, T[i_z, i_k, i_a]] \right\}$$

**Convergence rate.**  The Bellman operator is a $\gamma$-contraction in
sup-norm.  After $n$ iterations, $\|V^{(n)} - V^*\|_\infty \leq
\gamma^n \|V^{(0)} - V^*\|_\infty$.  This is linear convergence at
rate $\gamma$.

### PFI: Policy Function Iteration (Howard's Method)

Initialize with one Bellman maximization step to obtain an initial
policy $\pi^{(0)}$.

**Repeat** for $n = 0, 1, 2, \ldots$:

1. **Policy evaluation.**  For the fixed policy $\pi^{(n)}$, compute its
   value by iterating:

   $$V^{(n)}[i_z, i_k] = R[i_z, i_k, \pi^{(n)}[i_z, i_k]] + \gamma \sum_{j_z} P[i_z, j_z] \cdot V^{(n)}[j_z,\, T[i_z, i_k, \pi^{(n)}[i_z, i_k]]]$$

   for a fixed number of evaluation steps (typically 200–400).  This is
   the same Bellman equation but with the max replaced by the fixed
   policy — a linear fixed-point problem.

2. **Policy improvement.**  One full Bellman maximization step:

   $$\pi^{(n+1)}[i_z, i_k] = \arg\max_{i_a} \left\{ R[i_z, i_k, i_a] + \gamma \sum_{j_z} P[i_z, j_z] \cdot V^{(n)}[j_z,\, T[i_z, i_k, i_a]] \right\}$$

**Until** $\pi^{(n+1)} = \pi^{(n)}$ (policy indices unchanged at every
grid point).

**Convergence rate.**  PFI converges quadratically in the number of
outer iterations — typically 5–20 policy updates suffice, versus
hundreds or thousands of VFI iterations.  The cost is the inner
evaluation loop, but this is a fixed linear iteration (no maximization).