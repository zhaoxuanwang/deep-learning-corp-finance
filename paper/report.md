---
title: Deep Learning Methods in Quantitative Corporate Finance
author: Zhaoxuan Wang
affiliation: University of British Columbia
bibliography: references.bib
date: 2026-01-15
---

# The Basic Model

Let's first consider the basic dynamic model of optimal capital investment in [@strebulaev_dynamic_2012, section 3.1]. 

## Model Setup

A risk-neutral manager aims to maximize the value of the firm at current period $t$ by choosing next period capital $k_{t+1}$:

$$
V_t = \max_{k_{t+j}|j=1,\dots,\infty} \mathbb{E}_{t} \left[ \sum^{\infty}_{j=0} \beta^j \cdot u(z_t,k_t,k_{t+1})  \right].
$$

Denote current and next period variables with $x\equiv x_t$ and $x'\equiv x_{t+1}$. The Bellman equation of this dynamic programming problem can be derived as:

$$
V(z,k) = \max_{k'} \left[ u(z,k,k') + \beta \int V(z',k') d g(z'|z)  \right],
$$

where

- Reward function $u(z,k,k')$ is the net cash flow $u(z,k,k') \equiv \pi(z,k) - \psi(I,k) - I$
- Investment is $I = k' - (1-\delta) k$
- Production function is $\pi(z,k) \equiv z k^\theta$ with $\theta \in (0,1)$
- Capital adjustment costs are $\psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbb{1}_{I\neq0}$
- Discount factor is $\beta = 1/(1+r)$
- $r$ is risk-free interest rate

The (log) productivity shock $z$ follows a specific Markov process, AR(1):

$$
\ln (z') = (1 - \rho) \mu + \rho \ln(z) + \sigma \epsilon
$$

where the error term $\epsilon$ is IID and standard normal, the persistence factor $\rho \in (-1,1)$, and the variance $\sigma^2$. For the reminder of the report, I set the long-run mean $\mu = 0$.

## Casting the model into DL functions
Now let's follow @maliar_deep_2021 to cast this basic model into a set of functions in deep learning framework.
The optimal decision rule (or policies) is a function that maps current states $(z,k)$ to next period capital stock $k'$, which is denoted as $h(z,k)$. The reward function depends on both states and actions $(z,k,k')$, and in this basic model it is simply the net future cash flows $u(z,k,k')\equiv e(z,k,k')$.

The first goal is to approximate the optimal policies $h$ using deep neural networks (DNNs). We consider a parametric policy function $h(z,k;\theta)$ with current period state vector $w\equiv (z,k)$ and real hyperparameters $\theta \in \Theta$.

For example, consider a standard fully connected neural network (FcNN) with one hidden layer. Let $w$ be each element in all possible pairs of $(z,k)$, the optimal policy $h(z,k;\theta)$ is parametrized as:
$$
k' = h(z,k;\theta) = \theta^{(2)}_{bias} + \sum^{N_2}_{j=1} \theta^{(2)}_{j} \cdot S\left(\theta^{(1)}_{bias,j} + \sum^{N_1}_{s=1} \theta^{(1)}_{js} w_{j,s} \right),
$$
where $\theta$ include all the biases and weights that are trainable.

After we build a parametric policy function $h(z,k;\theta)$ with neural networks, we will define an objective loss function $\mathcal{L}(\theta)$ that captures the optimality conditions of our model, and train the neural network to find an optimal set of $\theta$ that minimize $\mathcal{L}(\theta)$ and delivers the optimal policies $h(\theta)$.

For more complicated models with larger state spaces, such as the risky debt model in [@strebulaev_dynamic_2012, section 3.6], we can use a similar approach to parameterize the optimal policies and the value function and cast the model into a set of functions in deep learning framework.

## DL Methods

### Method 1: Lifetime Reward Maximization

**Objective**: Maximize the discounted lifetime reward over a finite horizon.

**Network**: Parametrize the **Policy Function** $k' = h(z, k; \theta)$.

The first objective function is the (negative) lifetime reward over a finite horizon:

$$
\Xi_{LR} (\theta) = - \mathbb{E}_{(z_0,k_0,\epsilon_1,\dots,\epsilon_T)}
\left[
\sum_{t=0}^{T} \beta^t u\left( z_t,k_t,h(z_t,k_t;\theta) \right).
\right]
$$

Let $w_i$ denote the $i$-th simulated parallel trajectory vector $(z_0,k_0, \epsilon_1,\dots,\epsilon_T)$, where $z_0$ and $k_0$ are initial productivity and capital stock. A more economically intuitive way is to think about the set of simulated trajectories as $N$ firms or agents with different initial conditions $(z_0,k_0)$ and random exogenous shocks $\{\epsilon_t\}^T_{t=0}$ but follow the same data generating process. For the reminder of the paper, I refer to each trajectory as a **simulated firm** and the number of independent firms $N$ as **batch size**.

Empirically, we approximate the expectation by calculating the sample mean of the lifetime rewards over $N$ simulated firms. For each firm $i \in \{1, \dots, N\}$:

1.  **Initialization:** We initialize the state $(z_0, k_0)$ by sampling from the **ergodic limit**.
    -   **Cold Start**: For the very first batch, sample $(z_0, k_0)$ from the random uniform (or zeros)
    -   **Warm Start**: For all subsequent batches, set initial states $(z_0, k_0)$ to the *terminal states* of the previous batch $(z_T, k_T)$. This ensures sampling from the ergodic set where the solution lives.
2.  **Exogenous Shocks:** We draw a full path of exogenous shocks $\{\epsilon_t\}^T_{t=0}$ from the standard normal distribution.
3.  **State Simulation:** Using the policy function $h(z,k;\theta)$ and the AR(1) process $\ln (z_{t+1}) = \rho \ln(z_t) + \sigma \epsilon_{t+1}$, we can compute the full path of induced states $\{(z_t, k_t)\}^T_{t=0}$ for each firm $i$.

With these simulated data and a specific reward function, we can compute the average lifetime rewards across $N$ firms:

$$
\mathcal{L}_{LR}(\theta) = - \frac{1}{N} \sum^{N}_{i=1} \left( \sum_{t=0}^{T} \beta^t u(w_i ,h(w_i;\theta)) \right),
$$
where $w_i = (z_0, k_0, z_1, k_1, \dots, z_T, k_T)_i$ is the trajectory vector of each simulated firm $i$.

We train the DNN to find the optimal parameters $\theta^*$ and policies $h(z,k;\theta^*)$ that minimize the objective function $\mathcal{L}_{LR}(\theta)$.

**Pros**:

-   Conceptually simple and easy to implement.
-   Robust to non-differentiabilities in reward (e.g., kinks and discontinuities in value function).

**Cons**:

-   High variance in gradients with long horizons.

### Method 2: Euler Residual Minimization

**Objective**: Minimize the residuals of the Euler equation (First Order Conditions).

The second objective function is the Euler equation residuals that measured whether the optimal first-order condition (FOC) are satisfied. Let's begin by setting up the Lagrangian:

$$
\mathbb{E}_{z_{t+1}|z_t} \left[
\sum^\infty_{t=0} \beta^j
\left( u(z_t,k_t,I_t) - \lambda_t (k_{t+1}-k_t (1-\delta) - I_t).
\right)
\right]
$$

It is important to note that the Lagrangian only includes the capital accumulation identity. This is because according to the setting of the basic model [@strebulaev_dynamic_2012, p.67], cash flow $u(z,k,I)$ can be either positive or negative:
- If $u(z,k,I)\geq0$, then firm distribute internal cash flows to shareholders;
- If $u(z,k,I)<0$, then shareholder inject cash into the firm.

In addition, investment $I_t$ can also be either positive or negative, with $I_t<0$ representing the selling of capital, but this de-investment is costly due to the marginal adjustment cost $\psi(z,k,I)$.

Therefore, we don't have to consider inequality constraints like $I\geq 0$ and do not have to enforce Kuhn-Tucker condition with a Fischer-Burmeister function as in [@maliar_deep_2021, p.88]. This simplifies the DL implementation of the Euler-residual minimization approach.

Taking derivatives with respect to $I_t$, $I_{t+1}$, and $k_{t+1}$ yields:

$$
\begin{align}
\lambda_t &= \psi_I(I_t,k_t) + 1 \\
\lambda_{t+1} &= \psi_I(I_{t+1},k_{t+1}) + 1 \\
\lambda_t &= \mathbb{E}_{z_{t+1}|z_t} \left[ \beta \left(
\pi_k (z_{t+1},k_{t+1}) - \psi_k (I_{t+1}, k_{t+1}) + (1 - \delta) \lambda_{t+1}
\right) \right]
\end{align}
$$


which can be combined to form the investment Euler equation:

$$
\mathbb{E}_{z_{t+1}|z_t} \left[ \beta \big(
\pi_k (z_{t+1},k_{t+1}) - \psi_k (I_{t+1}, k_{t+1}) + (1 - \delta) (1 + \psi_I(I_{t+1},k_{t+1})
\big) \right]
= 1 + \psi_I(I_t,k_t).
$$

Given a parametric policy $h(z,k;\theta)$, the unit-free Euler residual is defined as:

$$
\xi_{ER}(\theta) = \frac{ \mathbb{E}_{z'|z} \left[ \beta
\pi_k (z', h ) - \psi_k (I', h ) + (1 - \delta) (1 + \psi_I(I', h)) \right]
}{
1 + \psi_I(I,k)
} - 1,
$$

and the goal is to find the parameters $\theta$ that minimize the Euler residual squares:

$$
\mathbb{E}_{(z',k',I)} \big[ \xi_{ER}(\theta) \big]^{2}
$$

#### Implementation
To implement the Euler-residual minimization method, we follow these steps:

1. **Initialization (Warm Start):** Initialize the current states by sampling from the *ergodic set*. The initial state of the current batch is set to the terminal state of the previous batch. Randomly draw $N$ firms with vector $\{w_i\}^N_{i=1} \equiv \{ (z, k) \}^N_{i=1}$.
2. For each firm $w_i$, use the policy function to compute next period capital $k'=h(z,k;\theta)$, the implied investment $I=h(z,k;\theta)-(1-\delta)k$, and the marginal adjustment cost is $\psi_I(I,k)$
3. For each random draw $w_i$, draw TWO random realization of next period shock $(\epsilon_1,\epsilon_2)$ and compute $(z'_1, z'_2)$ using AR(1) process described before.
4. Compute the implied next period investment $I'=h(z',k';\theta)|_{z'}-(1-\delta)k'$.
5. Given a realized value of $z'$, compute the realized Euler residual, $\xi^{(i)}(w_i, h(\theta))|_{z'}$ for each firm $i$:

$$
\xi^{(i)}(w_i, h(\theta))|_{z'} = \frac{  \beta
\pi_k (z', h ) - \psi_k (I', h ) + (1 - \delta) (1 + \psi_I(I', h))
}{
1 + \psi_I(I,k)
} - 1,
$$

6. Train the DNN and find $\theta^*$ to minimize the empirical loss function across all $N$ firms:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum^N_{i=1} \big[ \xi^{(i)}(\cdot;\theta)|_{z'=z'_1} \times \xi^{(i)}(\cdot;\theta)|_{z'=z'_2} \big]
$$

Note that the theoretical Euler residual squares is the square of an expectation. To compute a consistent estimator of the squared expectation, we use two independent shocks $(z'_1, z'_2)$ to compute the product of the two Euler residuals. This is the trick of the "All-in-One" expectation [@maliar_deep_2021].


**Derivation of Gradients**
An important assumption of this method is the existence of the derivatives of the economic primitives: $\pi_k(z,k)$, $\psi_k(I,k)$, $\psi_I(I,k)$. Recall that our profit function $\pi(z,k)=zk^\theta$ is differentiable with respect to $k$. However, the capital adjustment cost function $\psi(I,k)$ is not differentiable with respect to $I$ at $I=0$ because the fixed investment cost introduces a discontinuity (through the indicator function):

$$
\psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbb{1}_{I\neq0}
$$

Therefore, this method is not applicable to the basic model with fixed adjustment cost, unless we consider the special case where $\phi_1=0$ (or very close to zero).

**Pros**:
-   Checks optimality conditions directly.
-   No need for long simulations.

**Cons**:
-   Requires existence of derivatives for the production function and cost functions.


### Method 3: Bellman Residual Minimization

**Objective**: Minimize the error in the Bellman Equation satisfying optimality conditions.

**Network**:

-   Policy Network: $k' = h(z, k; \theta_1)$
-   Value Network: $V(z, k; \theta_2)$

The Bellman equation is:

$$
V(z,k) = \max_{k'} \{ u(z,k) + \beta \mathbb{E}_{z'|z} [ V(z',k') ] \}
$$

The RHS cannot be directly computed due to the maximization operator. To ensure the policy $h$ is optimal, we impose an additional **Optimality Condition (OC)**. @maliar_deep_2021 discusses three types of OCs: First Order Conditions (of the Bellman), Envelop Condition, and Direct Optimization. 

In this report, I implement the Envelop Condition and the Direct Optimization. The FOC optimality constraint is implemented by @maliar_deep_2021 in a very similar way. I move the Direct Optimization approach to **Method 4** because the algorithm is different despite mathematically the objective function is the same Bellman equation residual.

**Envelope Condition** is obtained by taking derivatives with respect to the endogenous state variable $k$: 

$$
V_k(z,k) = \pi_k(z,k) - \psi_k(I,k)
$$

where $I$ is treated as constant when taking the partial derivative $\psi_k$. Computationally, however, the jump discontinuity of $\psi_k$ at $I=0$ makes the gradient effectively infinite or undefined, and is not applicable to guide the update of the weights $\theta$.

**Objective Function**:
We minimize the weighted sum of the Bellman Residual (BR) and the Optimality Condition (OC):

$$
\mathbb{E}_{(z,k)} \Big[ \xi_{br}^2 \Big] + \nu \cdot \mathbb{E}_{(z,k)} \Big[ \xi_{oc}^2 \Big]
$$

where:
-  Residual of **Bellman Equation**: $\xi_{br} = V(z,k) - u(z,k) - \beta \mathbb{E}_{z'} \left[ V(z',h) \right]$
-  Residual of **Optimality Condition**: $\xi_{oc} = V_k(z,k) - (\pi_k - \psi_k)$
-  $\nu$ is a hyperparameter that controls the relative weight of the two residuals

#### Implementation

1. **Initialization (Warm Start):** Initialize the current states by sampling from the *ergodic set*. The initial state of the current batch is set to the terminal state of the previous batch. Randomly draw $N$ firms with vector $\{w_i\}^N_{i=1} \equiv \{ (z, k) \}^N_{i=1}$. For each firm $i$:
2. Draw TWO realized random shock $(\epsilon_1, \epsilon_2)$ and compute $(z'_1,z'_2)$ based on AR(1) process on realized $(z,\epsilon)$
3. Use the policy function DNN to compute next period capital $k'=h(z,k;\theta_1)$
4. Use the value function DNN to compute $V(z,k;\theta_2)$ and $V(z',k';\theta_2)$
5.  **All-in-One (AiO) Expectation**: To avoid biased gradients when squaring the expectation $\mathbb{E}_{z,k}[\mathbb{E}_{z'|z}\dots]^2$, use the product of two *realized* independent error to form the empirical loss function:

$$
\mathcal{L}_{BR}(\theta) = \frac{1}{N} \sum_{i=1}^N
\Big[ \xi_{br}(z'_1) \cdot \xi_{br}(z'_2) \Big] 
\times 
\Big[ \xi_{oc}(z'_1) \cdot \xi_{oc}(z'_2)\Big]
$$

where the AiO expectation is taken across all firms with their realized $(z,k,z',k')$ and the realized residuals:
- $\xi_{br}(z')=V(z,k;\theta) - u(z,k) - \beta V(z',h;\theta)$
- $\xi_{oc}(z')=V_k(z,k) - \pi_k(z,k) + \psi_k(h,k)$

**Pros**:

-   Learns both Value and Policy explicitly.
-   Directly search for solution characterized by Bellman Equation and Optimality Condition
-   Do not need to impose finite horizon $T$ and simulate full path

**Cons**:

-   FOC and Envelop Condition are not applicable to the fixed adjustment cost model where marginal adjustment cost $\psi_k$ has a jump discontinuity at $I=0$

---

### Method 4: Direct Optimization
A variant of the Bellman Residual Minimization method discussed in @maliar_deep_2021 is the Direct Optimization approach. The main difference is that the Bellman Residual Minimization method imposes an optimality condition (OC) to ensure the policy is optimal, while the Direct Optimization approach directly maximizes the RHS of the Bellman equation through grid search.

**Network**: Parameterize the **Value Function** $V(z, k; \theta)$.

**Target Network**: Uses a slowly updating Target Network $V(z, k; \theta^-)$ to stabilize training.

**Algorithm**:
1.  **Initialization**: Randomly draw current states $(z, k)$ for $N$ firms (from random uniform distribution). For each firm $i$:
2.  **Compute Target**: Search for the optimal $k'$ that maximizes the RHS:
   
    $$
    y_i = \max_{k' \in \text{Grid}} \left\{ u(z, k) + \beta \mathbb{E}_{z'} [ V(z', k'; \theta^-) ] \right\}
    $$

    -   **Grid Search**: The code discretizes the choice set for $k'$ into $N_k$ points.
    -   **Quadrature**: The expectation $\mathbb{E}_{z'|z}$ is computed using Gauss-Hermite Quadrature (5 nodes) for high precision, since the target network is a smooth function.
    -   Note that the target weights $\theta^-$are detached from the gradient (not backpropagated).

3.  **Optimization**: Minimize Mean Squared Error (MSE) between the current network prediction and the target:
    $$
    \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( V(z_i, k_i; \theta) - y_i \right)^2
    $$

4.  **Target Update**: Soft update the parameters with exogenous weight $w$:
    $$
    \theta^- \leftarrow w \theta + (1-w) \theta^-
    $$
    This step is important to stabilizing training because we need to ensure that in step #3 the target parameters $\theta^-$ are detached from gradient and are not updated too fast. Otherwise, whenever $\theta$ is updated to increase/decrease the value $V(z,k;\theta)$, the target parameters $\theta^-$ will also be updated to increase/decrease the target value $V(z,k;\theta^-)$ at similar rate such that the MSE loss may not converge to zero.

**Pros**:
-   **Stability and Robustness**: Converting the problem to a regression is generally stable. Handles non-differentiable reward/cost functions easily.

**Cons**:
-   **Computation Cost**: The grid search and quadrature make each training step computationally expensive compared to derivative-based methods.
-   **Discretization Error**: Policy accuracy is limited by the grid density.

---

# Risky Debt Model
Now let's consider a more complicated model where the firm chooses both the optimal capital investment and the optimal debt level [@strebulaev_dynamic_2012, section 3.6]. 

## Model Setup

**Objective**: Solve for the competitive equilibrium where the firm optimizes capital and debt given bond prices, endogenously choose whether to default, and lenders price bonds given firm default risk in equilibrium. Without specified, all the setup below follows [@strebulaev_dynamic_2012, section 3.6].

**State Space**: $(z, k, b)$ where $z$ is productivity shock, $k$ is capital stock, $b$ is risky debt (bond). 
- Productivity shocks $z$ are exogenous and follow the same AR(1) process as in the baseline model
- Capital stock $k$ is strictly positive
- Debt $b$ can be either positive (borrowing) or negative (cash savings)
- No collateral constraint on $b$. Instead, compute a natural upper bound and lower bound on $b$ based on the firm's equity and the best possible productivity shocks.

**Action Space**: $(k', b')$ where $k'$ is next period capital stock and $b'$ is next period debt

**Price**: Endogenous risky interest rate, $\tilde{r}(z,k',b')$, is determined by lender's zero profit condition in the bond market.

- Consider a zero-coupon bond with face value $b'$ and price $q(z,k',b')\equiv 1/(1+\tilde{r})$
- Firm borrow $q(z,k',b') \cdot b'$ today and repay $b'$ tomorrow 

**Policies**: Define a policy function $h(z,k,b;\theta)$ that maps the state $(z,k,b)$ to the action $(k',b')$

**Rewards** is defined as cash flow net of costs:
$$u(z,k,k',b,b') = e(z,k,k',b,b') - \eta(e(z,k,k',b,b'))$$
where $e(\cdot)$ is the basic net cash flow and $\eta(\cdot)$ is the cost of external financing (equity injection). 

**Net cash flow** with debt: 

$$
e(z,k,k',b,b') = (1-\tau)\pi(z,k) - \psi(I,k) - I - b + \frac{b'}{1+\tilde{r}} + \frac{\tau}{1+r} \left(1 - \frac{1}{1+\tilde{r}}\right) b'
$$ 

where $\tau$ is corporate tax rate, $\psi(I,k)$ is the cost of capital adjustment, $I$ is investment, $b$ is repayment of old debt, $\tilde{r}$ is the endogenous risky interest rate, and the last term is the tax shield on debt.

- Tax shield: firm deduct $\tau \left(1-\frac{1}{1+\tilde{r}}\right) b'$ from its tax liability, which is then discounted at the risk-free rate $r$.

**Cost of external financing** (equity injection) is defined as: 

$$
\eta(e(\cdot))\equiv (\eta_0 + \eta_1 |e(\cdot)|) \cdot \mathbb{1}_{e(\cdot)<0}
$$ 
where $\eta_0\geq0$, $\eta_1\geq0$, and $\mathbb{1}_{e(\cdot)<0}$ is an indicator for negative cash flow.

## Lender's Problem

**Recovery**: When default happens, lenders can recover a fraction $\alpha \in (0,1)$ of the firm's assets:

$$\text{Recovery}(z', k', b') = (1 - \alpha) \left[ (1 - \tau) \pi(z',k',b') + (1-\delta)k' \right]$$

**Bond Yield**: When the firm is solvent, lenders receive the full debt repayment $b'$ and the risky interest rate $\tilde{r}(z,k',b')$:

$$\text{Yield}(z', k', b') = (1+\tilde{r})b'$$

The equilibrium price of risky debt is determined by the zero profit condition of lenders:

$$ b'(1+r) = \mathbb{E}_{z'|z} \left[ \underbrace{(1-D) (1+\tilde{r})b'}_{\text{Solvent}} + \underbrace{D \cdot \text{Recovery}}_{\text{Default}} \right] $$

where the LHS is the opportunity cost and the RHS it the expected payoff of risky debt.

**Bond Price**: Define $q(z,k',b')$ as the price of a zero-coupon bond (risky debt) with face value $b'$.
$$q(z,k',b')\equiv \frac{1}{1+\tilde{r}(z,k',b')}$$

We can thus rewrite lender's zero profit condition as:

$$ b'(1+r) = \frac{b'}{q(z,k',b')} \mathbb{E}_{z'|z} [1-D] + \mathbb{E}_{z'|z}[D \cdot \text{Recovery}] $$

Rearrange terms to isolate $q(z,k',b')$:

$$ 
\begin{equation}
q(z,k',b') = \frac{b' \mathbb{E}_{z'|z}[1 - D]}{ b'(1+r) - \mathbb{E}_{z'|z}[D \cdot \text{Recovery}] }
\end{equation}
$$

Note that when default probability is zero $\mathbb{E}_{z'|z}[D \cdot \text{Recovery}]=0$ and the risky price $q(z,k',b')$ is simply the risk-free price $1/(1+r)$. In contrast, when the default probability is 1, the numerator $\mathbb{E}_{z'|z}[1 - D]=0$ and the risky price $q(z,k',b')$ is 0. Thus the risky price is bounded between $[0,\frac{1}{1+r}]$. 

This is particularly useful in implementation of the DNN with output neuron bounded between $[0,\frac{1}{1+r}]$. Otherwise if we directly target the risky interest rate $\tilde{r}(z,k',b')\in(r,\infty)$, the training may be unstable when $\tilde{r}\to\infty$.

### Endogenous Default
Firm can choose to default if continuation value $\bar{V}(z,k,b)<0$. Formally, the default indicator:

$$ D(z',k',b') = \mathbb{1}_{\bar{V}(z',k',b')<0} $$

and the continuation value is given by:

$$\bar{V}(z,k,b) = \max_{k',b'} \left\{ u(z,k,k',b,b') + \frac{1}{1+r} \mathbb{E}_{z'|z} [ \bar{V}(z', k', b') ] \right\}$$

Firm's can always choose to walk away with zero value (limited liability), so the actual value function is given by:

$$V(z,k,b) = \max \left\{ 0, \bar{V}(z,k,b) \right\}$$

### Residual of Zero Profit Condition
In implementation, the objective function to be minimized is the residual of the bond pricing equation. By randomly sampling $N$ firms and their current states $(z,k,b)$ and TWO independent realized shock $(\epsilon'_1,\epsilon'_2)$, we can compute the realized next period productivity $(z'_1,z'_2)$ using AR(1). To compute the realized bond pricing equation, we also need:

- Paramerized policy function $h(z,k,b;\theta)$ that outputs $(k',b')$
- Paramerized continuation (latent) value function $\bar{V}(z,k,b;\theta)$
- Paramerized risky bond price $q(z,k',b';\theta)$

Together, we use the realized $(z,k,b,z',k',b')$ and the parameterized functions to compute the default indicator $D(z',k',b';\theta)$ for each firm $i$ based on the realized $\bar{V}(z',k',b';\theta)$. Then we compute the sample mean error of the bond pricing equation:

$$ 
\xi_q(\cdot;\theta) = b'(1+r) - \frac{b'}{q(z,k',b';\theta)} (1-D) - D \cdot \text{Recovery}(z',k',b') 
$$

The pricing loss function is defined as the sample mean of the product of the residual:

$$ 
\begin{equation}
\mathcal{L}_q(\theta) = \frac{1}{N} \sum^N_{i=1} \left[ \xi_q(w_i;\theta)|_{\epsilon'_1} \times \xi_q(w_i;\theta)|_{\epsilon'_2} \right] 
\end{equation}
$$

## Firm's Problem

### Lifetime Reward Maximization

Firm's problem is to choose optimal policies $(k',b')$ that maximize lifetime rewards:

$$
\max_{k',b'} \mathbb{E}_{(z_0, k_0, b_0, \epsilon_1, \epsilon_2,\dots,\epsilon_T)} \left[ \sum^T_{t=1}\beta^t u(z_t,k_t,k_{t+1},b_t,b_{t+1}) \right]
$$

Similar to method #1 in the baseline model, we can start with a random intialization $(z_0, k_0, b_0)$ and a full random draw of $(\epsilon_1,\epsilon_2,\dots,\epsilon_T)$, then use the policy network to generate $(k',b')$ at $t=1,\dots,T$. Then we can train the policy network to find the optimal $\theta$ that minimizes the negative expected lifetime reward:

$$
\mathcal{L}_{LR}(\theta) = - \frac{1}{N} \sum^N_{i=1} \left[ \sum^T_{t=1}\beta^t u(w_i;\theta) \right]
$$

where $N$ is the batch size and realized state vector $w_i\equiv (z_0, k_0, b_0, \epsilon_1, \epsilon_2,\dots,\epsilon_T)_{i=1}^{N}$ is the $i$-th firm.

### Bellman Residual Minimization

An alternative presentation of the objective function is the Bellman Equation Residual. First define the Bellman Residual of the continuation value (latent value function):

$$
\begin{equation}
\max_{k',b'} \left\{ u(z,k,k',b,b') + \frac{1}{1+r} \mathbb{E}_{z'|z} [ \bar{V}(z', k', b') ] \right\} - \bar{V}(z,k,b)
\end{equation}
$$

To remove the maximization operator, we impose the Envelop condition with respect to endogenous states $(k,b)$:

$$
\begin{align}
\bar{V}_k(z,k,b) &= \frac{\partial u(z,k,b)}{\partial k} \\
\bar{V}_b(z,k,b) &= \frac{\partial u(z,k,b)}{\partial b} 
\end{align}
$$

To remove the inner expectation operator, I randomly draw $(z,k,b)$ and two realized shock $\epsilon_1,\epsilon_2$ to compute $(z'_1,z'_2)$. With parameterized policy function, actions $(k',b')$ can be computed. Then we can compute the realized residuals for each firm:

$$
\xi_{bellman}(\cdot;\theta)|_{\epsilon'} = u(z,k,k',b,b') + \frac{1}{1+r}  \bar{V}(z', k', b') - \bar{V}(z,k,b)
$$

The AiO expected loss function for the Bellman residual is defined as the sample mean of the product of the residual:

$$
\mathcal{L}_{bellman}(\theta) = \frac{1}{N} \sum^N_{i=1} \left[ \xi_{bellman}(\cdot;\theta)|_{\epsilon'_1} \times \xi_{bellman}(\cdot;\theta)|_{\epsilon'_2} \right]
$$

Given $(z,k,b)$ randomly drawed for each firm $i$, the residuals of the Envelope condition are given by:

$$
\begin{align*}
\xi_{k}(z,k,b) &= \bar{V}_k(z,k,b) - \frac{\partial u(z,k,b)}{\partial k} \\
\xi_{b}(z,k,b) &= \bar{V}_b(z,k,b) - \frac{\partial u(z,k,b)}{\partial b}
\end{align*}
$$

The AiO expected loss function for the Envelope condition is defined as the sample mean of the product of the residual:

$$
\begin{align*}
\mathcal{L}_{k}(\theta) &= \frac{1}{N} \sum^N_{i=1} \left[ \xi_{k}(w_i;\theta) \times \xi_{k}(w_i;\theta) \right] \\
\mathcal{L}_{b}(\theta) &= \frac{1}{N} \sum^N_{i=1} \left[ \xi_{b}(w_i;\theta) \times \xi_{b}(w_i;\theta) \right] \\
\end{align*}
$$

## Total Loss Function
Now we can construct the total loss function by combining the loss functions for the bond price equation, lifetime reward, Bellman residual, and Envelope condition:

$$
\mathcal{L}_{total}(\theta) = \mathcal{L}_{q}(\theta) + \nu_{1} \mathcal{L}_{LR}(\theta) + \nu_{2} \mathcal{L}_{bellman}(\theta) + \nu_{3} \mathcal{L}_{k}(\theta) + \nu_{4} \mathcal{L}_{b}(\theta)
$$

where $\nu$'s are hyperparameters that control the relative importance of each loss function. We train the DNN to find the optimal parameters $\theta$ that minimize the total loss and obtained the optimal policies $h(z,k,b;\theta)$, price function $q(z,k',b';\theta)$ and continuation value functions $\bar{V}(z,k,b;\theta)$. Lastly, we can back out the actual value function $V(z,k,b;\theta)= \max\{0, \bar{V}\}$.


## Implementation

### DNN Architecture

1.  **Policy Network**: $(k', b') = h_{policy}(z, k, b; \theta_{policy})$
    - Output neuron $k'\geq0$ with `softplus` activation
    - Output neuron debt $b'$ with `linear` activation
2.  **Latent Value Network**: $\bar{V}(z, k, b; \theta_{value})$ with `linear` activation
    - Firm default if $\bar{V}(z, k, b)<0$
    - Firm is solvent if $\bar{V}(z, k, b)\geq0$  
3.  **Price Network**: $q(z, k', b') = h_{price}(z, k', b';\theta_{price})$ with `sigmoid` activation


### Algorithm (Simultaneous Training)
We train all networks simultaneously using a random sample with $N$ firms. For each firm $i$:

1.  **Simulation**:
    -   Sample current states $(z, k, b)$
    -   Predict actions: $(k', b') = h_{policy}(z, k, b)$
    -   Predict risky interest rate: $q(z, k', b') = h_{price}(z, k', b')$
    -   Simulate $z'$ using AR(1), given $z$ and the random raw of $\epsilon$

2.  **Lender Step (Pricing)**:
    -   Compute next period firm value $\bar{V}' = \bar{V}(z', k', b')$
    -   Determine Default: $D(z',k',b') = \mathbb{1}_{\bar{V}' \le 0}$
    -   Compute the empirical loss $\mathcal{L}_q(\cdot;\theta)$

3.  **Firm Step (Optimization)**:
    -   Given $(z,k,b,k',b')$, $(z'_1,z'_2)$, and risky bond price $q$, compute the empirical total loss $\mathcal{L}_{total}$
    -   Train the DNN to find the optimal $\theta$ that minimizes $\mathcal{L}_{total}$

#### Recommended Weights
-   $\nu_{Lender} = 1.0$: Ensure the equilibrium pricing condition
-   $\nu_{LR} = 1.0$: Ensure maximization of lifetime reward
-   Set smaller weights for $\nu_{bellman}$ and $\nu_{envelop}$ because they are "redundant" optimality conditions given that the lifetime reward is maximized (?)
