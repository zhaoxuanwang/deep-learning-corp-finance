
# Abstract

The ongoing revolution in deep learning has provided new tools for solving high-dimensional dynamic equilibrium models that were previously considered intractable in economic and finance. Despite their promising theoretical properties, 

# Introduction
In this paper, I explore methods from deep learning and reinforcement learning (RL) which can be used to solve high-dimensional dynamic economic models that were previously considered as intractable. Specifically, I focus on the main toolkit of deep learning methods recently adopted by economists and systematically examine their effectiveness and properties in actual implementations. Drawing insights from the recent RL literature, I introduce significant improvements on the accuracy and performance of the main methods introduced by @maliar21 and @fern26. I build an open-source package in Python and Tensorflow that can easily be used for implementation of any user-specified dynamic models that shares similar generalized structure. I illustrate my proposed method and its software usage using the canonical corporate finance model.

This paper aims to make two key points. First, with just minimal revision, many of the RL methods can be adopted to solve for dynamic problems in finance and quantitative economics. Specifically, I show that the methods introduced in @maliar21 and @fern26 are simplified versions of existing RL methods under the special cases when the policy function is deterministic and the Markov Decision Process (MDP) is known/given. This connection improves transparency and brings fruitful insights because much of the theoretical and empirical properties of the methods have been well-established in the RL literature.

The second point is that the algorithm design is critical. I show that RL-based algorithms significantly outperform the current methods proposed in @maliar21 and @fern26 in terms of the unbiasedness and accuracy of solutions, training stability (convergence), robustness to hyperparameters and neural network architecture, and computational cost. By bridging the RL and economic literature (and harmonizing jargons), I argue that this approach directly addresses the concerns discussed by @fern26: convergence failure, training instability, and lack of diagnostic tools. I show that these issues have been well-studied and many solutions are available off-the-shelf.

# Machine Learning Techniques
This section briefly summarizes the core machine learning concepts and algorithms used in my solution methods. 
## Neural Networks


## Stochastic Gradient Descent
All solution methods in this paper reduce to minimizing a scalar loss function $J(\theta)$ with respect to a vector $\theta$ that collects all trainable parameters indexed by $p$: $\{\theta_1, \theta_2, \dots, \theta_p\}$. The loss takes different forms across methods and for most economic models it can be either of the followings:
- Negative discounted sum of lifetime rewards
- Euler equation residuals
- Bellman equation residuals
- Other objective functions that combine optimality constraints of the economic model
Let $\mathcal{D} = {x_1, \ldots, x_N}$ be a dataset of $N$ observations, where each $x_i$ may represent a state, a state-action pair, or a transition depending on the method. Define the loss as the sample average and we aim to minimize:
$$\min_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell(x_i;\theta)$$
where $\ell(x;\theta)$ is the per-observation loss. Our goal is to optimize the policy $\pi(\theta)$ by gradient descent (ascent):
$$\theta_{j+1} = \theta_j - \eta \cdot \nabla_\theta J(\theta)|_{\theta_j}$$
where $\eta > 0$ is the learning rate (typically small), $j$ indexes the training iteration, and the **policy gradient**  $\nabla_\theta J(\theta)=(\frac{\partial J}{\partial \theta_1}, \ldots, \frac{\partial J}{\partial \theta_p})^\top$ points in the direction of steepest increase. More concretely, consider parameter $\theta_i$​:
- If $\partial J/\partial \theta_i > 0$: increasing $\theta_i$ would increase $J(\theta)$ (bad for minimization), so the gradient update **decreases** $\theta_i$ by subtracting a positive quantity.
- If $\partial J/\partial \theta_i < 0$: increasing $\theta_i$ would decrease $J(\theta)$ (good for minimization), so the update **increases** $\theta_i$ by subtracting a negative quantity.
- If $\partial J/\partial \theta_i = 0$: then $\theta_i$ is at a local flat point and doesn't move.

Full-batch gradient descent updates $\theta$ using the exact gradient $\nabla_\theta J$, which requires evaluating $\nabla_\theta \ell$ at all $N$ observations per iteration. This is computationally expensive when $N$ is large. **Stochastic gradient descent** (SGD) reduces this cost by approximating the gradient with a **mini-batch** of $B \ll N$ observations that are random i.i.d draws from $\mathcal{D}$ at each iteration $j$:
$$\theta_{j+1} = \theta_j - \eta \cdot \frac{1}{B}\sum_{b=1}^{B} \nabla_\theta \, \ell\left(x_{i_b};\theta_j\right) \tag{SGD}$$
where , and ${i_1, \ldots, i_B}$ are indices drawn from ${1, \ldots, N}$. 

The SGD works because the mini-batch gradient at each iteration $j$ is an unbiased estimator of the full gradient $\mathbb{E}\left[ \frac{1}{B}\sum_{b=1}^{B} \nabla_\theta \, \ell\left(x_{i_b};\theta_j\right) \right] =  \nabla_\theta J(\theta_j)$ by the linearity of expectation and for any batch size $B\geq1$. Standard convergence results for SGD (Robbins and Monro, 1951) show that unbiasedness, combined with bounded gradient variance, is sufficient to guarantee that the average squared gradient norm $\frac{1}{K}\sum_j \mathbb{E}[\|\nabla_\theta \mathcal{L}(\theta_j)\|^2]$ converges to a neighborhood of zero as the number of iterations $K$ grows. The radius of this neighborhood is controlled by the learning rate $\eta$, which is why we normally use a small learning rate (e.g., $10^{-3}$) to yield tighter convergence.

**Why SGD over full-batch gradient descent.** Beyond reducing the per-iteration computational cost, mini-batch randomness brings two statistical advantages. First, the gradient noise acts as implicit regularization, biasing SGD toward flatter regions of the loss landscape that tend to generalize better to unseen data. Second, under standard regularity conditions, the noise helps escape saddle points where the full gradient is zero but $\theta$ is not at a local minimum (Ge et al., 2015). Such regions are prevalent in non-convex neural network losses.

**Convergence.** For non-convex losses, standard SGD theory (Robbins and Monro, 1951) guarantees convergence to a stationary point ($\nabla_\theta \mathcal{L} = 0$) under diminishing step sizes, but not to the global optimum. In deep RL, theoretical guarantees are limited to the validity of gradient directions. The Deterministic Policy Gradient theorem (Silver et al., 2014) proves that the policy gradient is a correct ascent (descent) direction, but it does not guarantee that following it will lead to the globally optimal policy. In practice, neural network training reliably finds solutions that perform well empirically.

**Overfitting.** Since SGD repeatedly draws mini-batches from $\mathcal{D}$, the network may eventually memorize individual observations rather than learn the underlying function, degrading performance at unseen states. In online RL methods, where the agent continuously interacts with the environment and adds new transitions to a replay buffer $\mathcal{D}$, this risk is partially mitigated because the training data is regularly refreshed. In offline RL, where the agent learns from a fixed, pre-collected dataset without further environment interaction, overfitting is a more serious concern and typically requires explicit regularization.

# A General Framework
In this section, I use a general framework to represent the dynamic models in terms of deep RL environment and notations. Specifically, I focus on problems under the conditions:
- Discrete-time
- Continuous state and action spaces, 
- Deterministic policy 
- Deterministic dynamics/state transition function with random noise
Most models in economic and finance satisfies these conditions, and are thus directly connected to the Model-Based RL literature and techniques. 
## Markov Decision Process
A Markov Decision Process (MDP) is defined as a collection $(\mathcal{S}, \mathcal{A}, \mathcal{E}, f, r, \gamma)$ that subsumes all relevant information for decision-making.

| Symbol                                                                 | Definition                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{S} \subseteq \mathbb{R}^n$                                   | **State space** (continuous). A state $s \in \mathcal{S}$ is a vector encoding all information the agent observes. When the environment involves multiple variables (e.g., productivity $z$ and capital $k$), they are stacked into a single vector $s = (z, k)^\top$.                                                                                                                                                            |
| $\mathcal{A} \subseteq \mathbb{R}^d$                                   | **Action space** (continuous). An action $a \in \mathcal{A}$ is a vector of controls the agent selects (e.g., investment, consumption).                                                                                                                                                                                                                                                                                           |
| $\mathcal{E}$                                                          | **Shock space**. The space from which exogenous shocks $\varepsilon$ are drawn. When dynamics are deterministic, $\mathcal{E} = \emptyset$.                                                                                                                                                                                                                                                                                       |
| $f: \mathcal{S} \times \mathcal{A} \times \mathcal{E} \to \mathcal{S}$ | **State transition function/dynamics**. Given current state $s$, action $a$, and exogenous shock $\varepsilon \in \mathcal{E}$, the next state is $s' = f(s, a, \varepsilon)$. When dynamics are deterministic, the shock argument is absent and $f: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$. When the dynamics involve both exogenous and endogenous components, they are combined into a single vector-valued function. |
| $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$                     | **Reward function**. A scalar signal $r(s, a)$ received after taking action $a$ in state $s$ (e.g., utility, profit).                                                                                                                                                                                                                                                                                                             |
| $\gamma \in (0, 1)$                                                    | **Discount factor**. Controls the trade-off between immediate and future rewards.                                                                                |

A full sequence of actions and states is defined as a **trajectory** or **rollouts** $\tau$:
$$ \tau = (s_0,a_0,s_1,a_1,\dots)$$
where the initial state $s_0$ is randomly sampled from some distribution $p_0$. The state transitions are **deterministic** according to the transition function $f$:
$$ s_{t+1}=f(s_t,a_t,\epsilon_t) $$
where $\epsilon_t$ is a random noise (e.g., productivity shock) but the function $f$ is deterministic. Note that this is different from a stochastic transition function in RL where $s_{t+1}$ is a draw from a distribution $s_{t+1} \sim P(\cdot|s_t,a_t)$. In most of the economic and finance models, we only work with deterministic dynamics with potential random noise, while the general RL literature also deals with stochastic dynamics.

The **reward function** $r(s,a)$ is assumed to be known exactly and it maps current state and actions $(s,a)$ to a scalar value. Common examples are utility function and firm's profit. The **discounted lifetime reward** over an infinite-horizon trajectory is summarized as:
$$
R(\tau) = \sum^\infty_{t=0} \gamma^t \cdot r(s_t,a_t)
$$
## Optimal Policy Function
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
## Value Functions
Given a policy $\pi_\theta$ and starting state $s_0$, define the trajectory $\tau \equiv (s_0, a_0, s_1, a_1, \ldots)$ where $a_t = \pi_\theta(s_t)$ and $s_{t+1} = f(s_t, a_t, \varepsilon_{t+1})$ with shocks $\varepsilon_t$ drawn i.i.d. from some distribution. There are four value functions commonly used in RL techniques and are also essential to my solutions methods.
### On-Policy Value function 
The expected return if you start in state ![s](https://spinningup.openai.com/en/latest/_images/math/96ac51b9afe79581e48f2f3f0ad3faa0f4402cc7.svg) and always act according to policy $\pi$ is given as:
$$ V^{\pi_\theta}(s) \equiv \mathbb{E}_\epsilon[R(\tau)|s_0=s] = \mathbb{E}_{(\varepsilon_1, \varepsilon_2, \ldots)}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, \pi_\theta(s_t))\right]$$
where $s_{t+1} = f(s_t, \pi_\theta(s_t), \varepsilon_{t+1})$. This is the expected cumulated discounted reward from state $s$ when following policy $\pi_\theta$, with the expectation taken over all future shock realizations. When dynamics are deterministic $s_{t+1}=f(s_t,a_t)$, the expectation over $\epsilon$ is trivial and this reduces to a direct optimization problem.
### On-Policy Action-Value Function 
Following convention in RL, I refer to the state-action value function as the Q function: 
$$
\begin{aligned}
Q^{\pi_\theta}(s, a) &\equiv \mathbb{E}_\epsilon \left[R(\tau)|s_0=s,a_0=a \right] \\
&= r(s_t, a_t) + \gamma  \mathbb{E}_{\varepsilon}\left[V^{\pi_\theta}(s_{t+1})\right]
\end{aligned}
$$
where $s_{t+1}=f(s_t, a_t, \epsilon_{t+1})$. This is the reward from taking a given arbitrary action $a$ (possibly different from $\pi_\theta$) in state $s$ , then following $\pi_\theta$ thereafter, with the expectation over the immediate next shock $\epsilon_{t+1}$.
### Optimal Value Function
The optimal value function $V^*(s)$ gives the maximum expected lifetime reward if agent start from $s$ and always act according to the _optimal_ policy $\pi^*$:
$$
V^*(s) =\max_{\pi}V^\pi(s) = \mathbb{E}_{(\varepsilon_1, \varepsilon_2, \ldots)}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, \pi^*(s_t))\right]
$$
### Optimal Action-Value Function
The optimal action-value function $Q^*(s,a)$ gives the maximum expected lifetime reward if agent start from $s$, take an arbitrary action $a$, and then follow the optimal policy forever:
$$
Q^*(s, a) = r(s_t, a_t) + \gamma \mathbb{E}_\epsilon [ V^*(s_{t+1}) ]
$$
### Why We Need Q Function?
Under a deterministic policy with known dynamics and reward, the Q-function is in principle redundant: it can always be reconstructed from the value function using the definitions above. Differentiating through the known reward $r$ and state transition function $f$ recovers the same gradient information.

However, this construction requires both $r$ and $f$ to be differentiable with respect to the action $a$, since updating the policy involves computing $\nabla_a r$ and $\nabla_a f$ explicitly. Many models in economic and finance violate this condition, for example, utility and profit functions with jumps and kinks all produce rewards that are non-differentiable in $a$. In these cases, the $V^\pi$ function-based policy gradient is undefined. This is a main limitation shared by all the existing deep learning methods introduced in @maliar21 and @fern26.

The Q-function representation resolves this by absorbing $r$ and $f$ into a single learned function approximator $Q_\phi(s, a)$. Because $Q_\phi(s, a)$​ is a neural network, its gradient $\nabla_a Q_\phi$ is always well-defined as the network provides a smooth approximation of the true action-value landscape. The policy gradient can then be computed without ever differentiating $r$ or $f$ directly. Section X develops this approach in detail.
## Bellman Equations
Let the 'tick' denote next time step variables, e.g., $s'\equiv s_{t+1}$, the Bellman equations for the on-policy value functions are
$$
\begin{aligned}
	V^{\pi}(s) &= r(s,\pi(s)) + \gamma \mathbb{E}_{\epsilon}\left[ V^{\pi}(s') \right]\\
	Q^{\pi}(s, a) &=r(s,a) + \gamma \mathbb{E}_{\epsilon}\left[ Q^{\pi}(s',\pi(s'))\right]
\end{aligned}
$$
where the expectation is taken over $\epsilon$ that governs the state transition functions $s'=f(s,a,\epsilon)$.

The Bellman equations for the optimal value functions are
$$
\begin{aligned}
	V^*(s) &= \max_a \left\{ r(s,a) + \gamma \mathbb{E}_{\epsilon}\left[ V^*(s') \right] \right\} \\
	Q^*(s, a) &=r(s,a) + \gamma \max_{a'} \mathbb{E}_{\epsilon}\left[ Q^*(s',a')\right]
\end{aligned}
$$
where the $\max$ operator ensures that at optimality, the agent will pick the action that maximizes the Bellman right-hand-side (RHS).
# Traditional Methods
In this section, I explain the solution methods using the machine learning techniques and the general framework representation. I start with a brief review of three conventional methods: policy function iteration, value function iteration, and projection methods. Then I summarize the deep learning methods proposed by @maliar21 and @fern26 and discuss their advantages and limitations. I also identify the critical limitations of these methods and discuss how they can be mitigated with augmented design and/or more structural assumptions.

Finally, I propose a new solution method based on the canonical Deep Deterministic Policy Gradient algorithm (DDPG) and the Model-Based Value Expansion algorithm (MBVE) from the DRL literature. I argue that the MBVE-DDPG methods can handle problems where the existing method failed (e.g., non-differentiable reward functions) and expand the scope of tractable models in finance and economic applications. 
## Policy and Value Iteration
Value function iteration (VFI) and policy function iteration (PFI) are the most widely used methods to solve discrete-time dynamic programming problems (Bellman, 1957; Howard, 1960). In their simplest form, these methods discretize the continuous state space into a finite grid $\mathcal{S}_{\text{grid}}$ and iterate on the Bellman equation until convergence.

VFI exploits the property that the Bellman operator is a contraction mapping with unique fixed point $V^*$, so repeatedly applying the operator to any initial $V_0$ converges to $V^*$. Each iteration applies a single Bellman backup across all grid points and selects the maximizing action, but does not explicitly maintain a policy until convergence.

PFI separates each iteration into two steps: (i) **policy evaluation**, which solves for the exact on-policy value function $V^{\pi_j}$ given a fixed policy $\pi_j$, and (ii) **policy improvement**, which updates the policy by maximizing the Bellman right-hand side using $V^{\pi_j}$. The Policy Improvement Theorem (Bellman, 1957; Howard, 1960) guarantees $V^{\pi_{j+1}}(s) \geq V^{\pi_j}(s)$ for all $s$. PFI typically converges in fewer outer iterations than VFI because each iteration performs exact policy evaluation rather than a single Bellman backup, though each iteration is more expensive.
### Algorithm 1: Value Function Iteration (VFI)
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
### Algorithm 2: Policy Function Iteration (PFI)
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

::: {.callout-tip title="Implementation"}
I implemented both VFI and PFI in Tensorflow and it can be found under `src/ddp`. The `QuantEcon` library also has a `DiscreteDP` module implemented using Numpy.
:::

## Projection Methods 
An alternative to grid-based iteration is the projection method (Judd, 1992, 1998). Here I consider a variant reviewed in Ljungqvist and Sargent (2018, Chapter 3). The core idea is to replace the step function over a discrete grid with a _continuous approximation_ of the value function using a weighted sum of orthogonal polynomials. The canonical choice is Chebyshev polynomials, defined on $[-1, 1]$ by: $$T_n(x) = \cos(n \arccos x)$$The first few are $T_0(x) = 1$, $T_1(x) = x$, and $T_2(x) = 2x^2 - 1$. To illustrate the method, consider approximating the value function along a single state dimension $s \in [s_{\min}, s_{\max}]$ (when the state is multi-dimensional, the remaining dimensions are either treated separately or handled via tensor products of univariate polynomials). Since Chebyshev polynomials are defined on $[-1, 1]$, we apply a linear change of variable: $$x = \frac{2(s - s_{\min})}{s_{\max} - s_{\min}} - 1$$ The value function is then approximated as a linear combination of $n+1$ basis functions: $$\hat{V}(x;\,\boldsymbol{\theta}) = \sum_{j=0}^{n} \theta_j\, T_j(x)$$where $\boldsymbol{\theta} = (\theta_0, \theta_1, \ldots, \theta_n)$ are the coefficients to be determined. The evaluation points are not chosen uniformly to minimize approximation error, the method uses the zeros of $T_{N_g}(x)$, known as Chebyshev nodes: $$x_g = \cos\!\left(\frac{2g - 1}{2N_g}\,\pi\right), \quad g = 1, \ldots, N_g$$The algorithm iterates on the Bellman equation using this continuous approximation, updating the coefficients $\boldsymbol{\theta}$ at each iteration until convergence. Conceptually, this method maps economic states to Chebyshev nodes: $s \to x$. Then we iterate on the Bellman equation using continuous $T(x)$ as an approximation to the value function, and update the coefficients $(\theta_0, \{\theta_j\}_{j=1}^n)$ to minimize the approximation error (e.g., via regression). Finally, when the coefficients converge, we obtained the optimal mapping: $s \to x \to V^*(s,\mathbf{\theta})$ with fixed coefficients $\mathbf{\theta}^*$ and extract the optimal policy.

Although the projection method provides a continuous approximation by eliminating the step-function of grid-based VFI and PFI, it shares the fundamental curse of dimensionality. For multi-dimensional states, the tensor product of univariate Chebyshev bases requires $O(n^d)$ coefficients where $d$ is the state dimension, and the number of Chebyshev nodes grows correspondingly. This motivates the neural network methods introduced in subsequent sections.
### Algorithm 3: Polynomial Approximation
**Input:** Polynomial degree $n$, number of nodes $N_g \geq n+1$, reward $r$, dynamics $f$, discount $\gamma$, tolerance $\delta > 0$ 

**Output:** Approximate value function $\hat{V}(\cdot;\,\boldsymbol{\theta}^*)$, optimal policy $\pi^*$ 
1. Compute Chebyshev nodes $\{x_g\}_{g=1}^{N_g}$ and map to state-space grid $\{s_g\}_{g=1}^{N_g}$ 
2. Initialize coefficients $\boldsymbol{\theta} = (\theta_0, \ldots, \theta_n)$ (e.g., all zeros) 
3. **For** $j = 0, 1, 2, \ldots$ **do** 
4. $\quad$ **Maximization:** For each node $s_g$, $g = 1, \ldots, N_g$, solve $$\tilde{y}_g = \max_{a}\left\{r(s_g, a) + \gamma\,\mathbb{E}_\epsilon\!\left[\hat{V}(f(s_g, a, \epsilon);\,\boldsymbol{\theta})\right]\right\}$$
5. $\quad$ **Coefficient update:** Compute new coefficients using the discrete orthogonality property of Chebyshev polynomials at their zeros: $$\theta_j^{\text{new}} = \frac{\sum_{g=1}^{N_g} \tilde{y}_g\, T_j(x_g)}{\sum_{g=1}^{N_g} T_j(x_g)^2}, \quad j = 0, 1, \ldots, n$$
6. $\quad$ **If** $\|\boldsymbol{\theta}^{\text{new}} - \boldsymbol{\theta}\| < \delta$ **then break** 
7. $\quad \boldsymbol{\theta^*} \leftarrow \boldsymbol{\theta}^{\text{new}}$ 
8. **End for** 
9. $\pi^*(s) = \arg\max_a\left\{r(s, a) + \gamma\,\mathbb{E}_\epsilon\!\left[\hat{V}(f(s, a, \epsilon);\,\boldsymbol{\theta}^*)\right]\right\}$ 
10. **Return** $\hat{V}(\cdot;\,\boldsymbol{\theta}^*)$, $\pi^*$

---
# Data Pipeline and Training Workflow

This section describes how synthetic training data is generated, the design principles behind the data pipeline, and the full training workflow. The framework is model-agnostic: it applies to any Markov decision process with known dynamics.

## Design Principles {#sec-data-design}

### Separation of simulation and training

Data generation and training are strictly separated. The data generator produces a complete, fixed dataset $\mathcal{D}$ before training begins. The trainer is stateless with respect to simulation: it contains no RNG for data generation, no simulation logic, and no buffer management. Two runs with the same $\mathcal{D}$ and the same optimizer seed produce identical training trajectories.

This provides three concrete benefits. First, when training fails, the cause is unambiguously in the optimizer or the network, not in the data. Second, the loss at any checkpoint is computed on the same distribution, so loss curves reflect optimization progress rather than distributional shift. Third, ablation studies are clean: changing a training hyperparameter changes nothing about the data.

### The distributional mismatch problem

All residual-based methods minimize $J(\theta) = \mathbb{E}_{s \sim \xi}[\ell(s; \theta)^2]$, where $\xi$ is the distribution over evaluation states. The ideal $\xi$ is the ergodic distribution $\xi^*$ of the Markov chain under the optimal policy $\pi^*$, because this concentrates approximation quality where the economy spends time. But $\xi^*$ depends on $\pi^*$, which is the object being solved for. As the policy improves during training, the induced distribution shifts, and evaluation points drawn under earlier policies become stale. @fern26 (Section 2.5) calls this the equilibrium loop.

**Low-dimensional case.** For small state spaces, the problem is sidestepped by sampling uniformly from a bounded domain wide enough to contain the ergodic set. The wasted computation on non-ergodic states is acceptable. @fern26 recommends this for low-dimensional problems.

**Method-dependent asymmetry.** The mismatch affects methods differently. LRM rolls out full trajectories under $\pi_\theta$ during training, so its endogenous states are approximately ergodic under the current policy and update automatically as $\theta$ improves — LRM gets implicit ergodic sampling for free. ER and BR use the flattened format where endogenous states are uniform draws from data generation time, never recomputed under any policy. The exogenous states are ergodic (policy-independent), but the endogenous states are not. For low-dimensional problems this hybrid is adequate; for high-dimensional problems the uniform endogenous states become the bottleneck.

**High-dimensional case: the equilibrium loop.** When the state dimension is large, uniform sampling of endogenous states becomes intractable. The volume of the ergodic set relative to the bounding hypercube shrinks exponentially with dimension (@maliar21, Eq. 18): at $N_s = 10$ about 0.3% of the hypercube is relevant; at $N_s = 30$ the fraction is negligible.

The solution is the equilibrium loop (@fern26, Section 2.5; @maliar21): iterate between simulation and training. (1) Guess $\theta_0$. (2) Simulate under $\pi_{\theta_0}$ to generate evaluation points on the ergodic set. (3) Train to obtain $\theta_1$. (4) Re-simulate under $\pi_{\theta_1}$. (5) Repeat until convergence. Our training workflow implements this as a sequence of **rounds** (defined precisely in the Training Workflow section), where each round generates a completely fresh dataset and trains to convergence. Within each round, the dataset is fixed and training is separated from simulation. For high-dimensional problems, the endogenous initial states at round $k \geq 1$ are harvested from the previous round's learned policy rather than sampled uniformly.

Increasing the data trajectory length $T_{\text{data}}$ is always cheap — it requires only additional shock draws, no gradient computation. Setting $T_{\text{data}}$ much larger than the mixing time of the chain (on the order of 40–100 periods for typical quarterly-calibrated models) ensures that the terminal endogenous states are approximately ergodic.

### Comparison with alternative data strategies

**@maliar21: epoch-based simulation.** Maliar et al. implement the same equilibrium loop: simulate a panel under the current policy, train, re-simulate. Our workflow differs in implementation: we strictly separate data generation from training within each round. Maliar et al. note that their simulated data have serial correlation and recommend training on cross-sections separated in time — our flatten-and-shuffle eliminates this entirely.

**Replay buffer (standard off-policy RL, Duarte 2024).** Standard off-policy reinforcement learning maintains a buffer of transitions from past policies, discarding old transitions as new ones arrive. The buffer's distribution is a path-dependent mixture across policy vintages — it corresponds to no well-defined training objective and introduces non-stationarity. Duarte's DPI uses a buffer of 500,000 states with periodic ergodic refresh. Our design avoids buffer management: within each round the dataset is fixed and stationary, and no tuning of buffer size, eviction policy, or mixing ratio is required. Between rounds, the dataset is regenerated cleanly.

## Dataset Structure

### Setup

Consider a discrete-time MDP with state $s \in \mathcal{S} \subset \mathbb{R}^{N_s}$, action $a \in \mathcal{A} \subset \mathbb{R}^{N_a}$, transition dynamics $s' = f(s, a, \varepsilon)$ with i.i.d. shocks $\varepsilon \sim F$, reward $r(s, a)$, and discount factor $\gamma \in (0,1)$. The policy $\pi_\theta(s)$ and value function $V_\phi(s)$ are parameterized by neural networks.

The state vector $s$ may contain exogenous components $s^{\text{exo}}$ (driven entirely by $\varepsilon$) and endogenous components $s^{\text{endo}}$ (influenced by $a$). This distinction matters for data generation: the exogenous path can be simulated exactly from shock draws alone, while the endogenous path depends on the policy and is computed during training, not during data generation.

Both the state and action spaces are bounded; the rules for setting bounds are described in the state space section. For data generation, we take $\mathcal{S} = [s_{\min}, s_{\max}]$ and $\mathcal{A} = [a_{\min}, a_{\max}]$ as given.

### Trajectory Simulation

The data generator produces $L$ independent trajectories of length $T_{\text{data}}$, where $T_{\text{data}}$ is deliberately set large enough for the endogenous chain to mix (typically $T_{\text{data}} \geq 40\text{--}100$ periods for quarterly-calibrated models). Note that $T_{\text{data}}$ is a data generation parameter that controls trajectory length, distinct from $T_{\text{bptt}}$ which controls the truncation horizon for BPTT in the LRM trainer ($T_{\text{bptt}} \leq T_{\text{data}}$).

The exogenous and endogenous components of the initial state are sampled separately: $s_0^{\text{endo},i}$ and $s_0^{\text{exo},i} \sim \text{Uniform}(\mathcal{S}^{\text{exo}})$, each with its own dedicated RNG stream. The source of $s_0^{\text{endo},i}$ depends on the round and the problem dimension (see Training Workflow). For each trajectory, $M$ independent shock sequences $\varepsilon^{(1)}, \ldots, \varepsilon^{(M)} \sim F$ are drawn and used to roll out the full exogenous state paths. The stored trajectory is:
$$\mathcal{T}^i = \left(s_{0}^{\text{endo},i},\; \{s_t^{\text{exo},i}\}_{t=0}^{T_{\text{data}}},\; \{s_t^{\text{exo},\text{fork},i}\}_{t=1}^{T_{\text{data}}}\right), \quad i = 1, \ldots, L$$

where the exogenous paths are computed at generation time via the policy-independent transition $s^{\text{exo}}_{t+1} = g(s^{\text{exo}}_t, \varepsilon_{t+1})$. Raw shocks are consumed internally by the data generator and are not stored. This is possible because the exogenous and endogenous state transitions are treated separately: $g$ depends only on $s^{\text{exo}}$ and $\varepsilon$ and can be evaluated without the policy, whereas the endogenous transition $h(s_t, \pi_\theta(s_t), \varepsilon_{t+1})$ requires $\pi_\theta$ and is computed during training. The data generator is model-agnostic because each environment specifies $g$ explicitly.

The endogenous path cannot be pre-computed and is handled differently by each method:

- **LRM (trajectory-based):** The trainer rolls out the full endogenous state sequence on-the-fly under the current $\pi_\theta$ at each training step, using the pre-computed $\{s_t^{\text{exo}}\}$ from the dataset as a fixed exogenous backdrop. The BPTT truncation horizon $T_{\text{bptt}}$ controls how many steps gradients flow backward through; this can be much shorter than $T_{\text{data}}$. No additional randomness enters the trainer.

- **ER/BR (one-step methods):** The flattened format pairs each endogenous initial state with a pre-computed exogenous transition. The source of endogenous states depends on the round: uniform draws at round 0, and harvested ergodic states at subsequent rounds for high-dimensional problems (see Training Workflow).

### Main Path and Fork Paths

Multiple shock realizations per trajectory enable Monte Carlo estimation of expectations over future states.

**Main path.** The first shock sequence $\{\varepsilon_t^{(1)}\}$ drives a continuous chain of exogenous states, providing the trajectory for lifetime reward calculations (LRM).

**Fork paths.** At each step $t$, the remaining sequences branch from the main path's current state:
$$s^{\text{exo},\text{fork}(m)}_{t+1} = g(s^{\text{exo}}_t, \varepsilon_{t+1}^{(m)}), \quad m = 2, \ldots, M$$

Each fork is a one-step transition from the main path, not a parallel chain.

```
Main path:   s0 -> s1 -> s2 -> s3 -> ... -> sT
               \     \     \     \
Fork 1:        s1F   s2F   s3F   sTF
Fork 2:        s1F'  s2F'  s3F'  sTF'
  ...
```

With $M = 2$, the cross-product of main and fork transitions provides the AiO variance-reduced estimator (@maliar21) used by ER and BR. Simulating $M > 2$ fork paths yields a lower-variance Monte Carlo estimator for $\mathbb{E}_{\varepsilon}[\cdot \mid s_t]$ (@pascal24_JDEC). This is a pure data generation choice that does not affect the training algorithm.

### Flattened Format

One-step methods (ER, BR, actor-critic) require i.i.d. samples, not full trajectories. The flattened dataset provides each observation as a single-step transition with pre-computed exogenous next states:
$$\text{Obs} = \left(s^{\text{endo}},\; s^{\text{exo}},\; s^{\text{exo},\prime(1)},\; s^{\text{exo},\prime(2)}\right)$$

where $s^{\text{exo}}$ is extracted from the trajectory's main path and $s^{\text{exo},\prime(1)}, s^{\text{exo},\prime(2)}$ are the pre-computed main and fork next states. Because all exogenous transitions are resolved at generation time, the trainer calls no RNG and performs no simulation — it uses $s^{\text{exo},\prime}$ directly. The endogenous states are sampled uniformly from $\mathcal{S}^{\text{endo}}$, independently for each observation — no policy exists at data generation time. After construction, all observations are pooled and shuffled to break the serial correlation in exogenous states. The resulting dataset has $L \times T$ i.i.d. observations.

Both LRM and one-step methods train on the same underlying exogenous trajectories; only the batching and the treatment of endogenous states differ.

### Validation and Test Sets

- **Validation set**: Regenerated each round alongside the training set ($N_{\text{val}} = 10n$ where $n$ is the training batch size), used for early stopping within a round
- **Test set**: Generated once before training begins and sealed ($N_{\text{test}} = 50n$), used for cross-round convergence and final evaluation

The test set uses a separate RNG stream and is never regenerated, providing a stable reference across rounds.

## Training Workflow {#sec-workflow}

Training is organized as a sequence of **rounds**, where each round generates a completely fresh dataset and trains the policy to within-round convergence. This prevents the network from overfitting to specific exogenous shock realizations and ensures that the training distribution remains representative. We adopt the following terminology hierarchy:

- **Round** $k = 0, 1, 2, \ldots$: one complete data-generation-then-training cycle. Each round produces a fresh dataset.
- **Epoch**: one full pass through the dataset within a round (standard ML convention).
- **Step** $j$: one mini-batch gradient update.

### Round Structure

Each round $k$ proceeds in three stages: data generation, training, and evaluation.

**Stage 1: Data generation.** A completely fresh dataset $\mathcal{D}^{(k)}$ is generated at the start of each round. This includes new exogenous initial states, new shock sequences, and new exogenous trajectory rollouts. The endogenous initial states are sourced differently depending on the round and problem dimension (see below). A fresh validation set $\mathcal{D}^{(k)}_{\text{val}}$ is generated alongside.

**Stage 2: Training.** The generic trainer receives the fixed dataset $\mathcal{D}^{(k)}$ and runs SGD with mini-batching. Training proceeds for a number of steps determined by the dataset size and batch size: given $N$ observations and batch size $B$, one epoch (full pass) requires $N/B$ steps; training may run for multiple epochs without replacement until the dataset is exhausted each pass. The input normalizer is warmed up on the training states before gradient steps begin. Early stopping within a round is based on the validation set $\mathcal{D}^{(k)}_{\text{val}}$.

- **LRM:** rolls out endogenous states on-the-fly under $\pi_\theta$ using the pre-computed exogenous paths as a fixed backdrop; BPTT truncation horizon is $T_{\text{bptt}} \leq T_{\text{data}}$.
- **ER/BR:** uses the flattened format with pre-computed exogenous transitions directly — no simulation.

**Stage 3: Cross-round evaluation.** After training completes, the policy $\pi_{\theta_{k+1}}$ is evaluated on the sealed test set $\mathcal{D}_{\text{test}}$. Training terminates when the test metric stabilizes across rounds.

### Endogenous State Initialization

The only difference between low-dimensional and high-dimensional problems is how endogenous initial states are sourced at round $k \geq 1$.

**Round 0** (both cases): All initial states are sampled uniformly. $s_0^{\text{endo}} \sim \text{Uniform}(\mathcal{S}^{\text{endo}})$ and $s_0^{\text{exo}} \sim \text{Uniform}(\mathcal{S}^{\text{exo}})$.

**Round $k \geq 1$, low-dimensional:** Endogenous initial states are sampled uniformly $s_0^{\text{endo}} \sim \text{Uniform}(\mathcal{S}^{\text{endo}})$, same as round 0. The purpose of additional rounds is solely to train on fresh exogenous realizations, preventing overfitting to specific shock paths.

**Round $k \geq 1$, high-dimensional (ergodic harvest):** Before generating the new dataset, a harvest step collects approximately ergodic endogenous states from the previous round. For each trajectory $i$ in $\mathcal{D}^{(k-1)}_{\text{traj}}$, the learned policy $\pi_{\theta_k}$ is rolled out along the stored exogenous path $\{s_t^{\text{exo},i}\}_{t=0}^{T_{\text{data}}}$ to produce the endogenous trajectory, and the terminal state $s_{T_{\text{data}}}^{\text{endo},i}$ is collected. Because $T_{\text{data}}$ exceeds the mixing time of the endogenous chain, these terminal states are approximately ergodic under $\pi_{\theta_k}$. The harvested states are used as $s_0^{\text{endo}}$ for the new dataset. All exogenous components ($s_0^{\text{exo}}$, shock sequences, exogenous paths) are freshly generated.

### Pseudocode

```
Setup:
   Generate sealed test set D_test (fixed across all rounds)

Round 0:
   [Data generation]
   Sample s_endo_0 ~ Uniform(S^endo), s_exo_0 ~ Uniform(S^exo)
   Draw M fresh shock sequences; roll out exogenous paths of length T_data
   Store D_traj_0 = {(s_endo_0, {s_exo_t}_{t=0}^{T_data}, {s_exo_fork_t}_{t=1}^{T_data})}
   Flatten and shuffle into D_flat_0 for one-step methods
   Generate validation set D_val_0 (same procedure, separate RNG)

   [Training]
   Train on D_0 with mini-batching; early stop via D_val_0 → θ_1

Round k (k ≥ 1):
   [Harvest — high-dim only]
   For each trajectory i in D_traj_{k-1}:
     Roll out endogenous states under π_{θ_k} along stored exogenous path
     Collect terminal state s_endo_{T_data, i} as ergodic initial state

   [Data generation]
   Sample s_exo_0 ~ Uniform(S^exo); draw M fresh shock sequences
   Roll out new exogenous paths of length T_data
   Sample s_endo_0:
     Low-dim:  ~ Uniform(S^endo)
     High-dim: from harvested {s_endo_{T_data, i}}
   Store D_traj_k; flatten and shuffle into D_flat_k
   Generate validation set D_val_k (same procedure, separate RNG)

   [Training]
   Train on D_k with mini-batching; early stop via D_val_k → θ_{k+1}

   [Cross-round convergence]
   Evaluate π_{θ_{k+1}} on sealed D_test; stop when test metric stabilizes
```

### Key Invariants

1. **Fresh data each round.** Every round generates completely new exogenous paths and shock sequences. No exogenous data is reused across rounds, preventing overfitting to specific shock realizations.
2. **Stateless trainer.** The trainer receives a fixed dataset dict, contains no RNG for data generation, and performs no simulation. Two runs with the same dataset and optimizer seed produce identical results.
3. **Fixed within round.** Within each round, the dataset is constant. Loss curves reflect optimization progress, not distributional shift. Ablation studies are clean.
4. **Sealed test set.** The test set is generated once at setup and never regenerated, providing the sole stable reference for cross-round convergence.
5. **Single branch point.** The only difference between low-dimensional and high-dimensional workflows is the source of $s_0^{\text{endo}}$ at round $k \geq 1$.

## Application: Corporate Finance Model

The state vector is $s = (k, b, z)$ where $k$ is capital (endogenous), $b$ is debt (endogenous), and $z$ is productivity (exogenous). The action is $a = k'$ (next-period capital). The exogenous transition is $\ln z_{t+1} = (1-\rho)\mu + \rho \ln z_t + \sigma \varepsilon_t$.

Each trajectory sample $i$ at round $k$ contains:
$$\left(k_{0,i},\; b_{0,i},\; \{z_{t,i}\}_{t=0}^{T_{\text{data}}},\; \{z_{t,i}^{\text{fork}}\}_{t=1}^{T_{\text{data}}}\right)$$
where the exogenous initial state $z_0 \sim \text{Uniform}([z_{\min}, z_{\max}])$ (in levels), and the full exogenous trajectories $z_{1:T_{\text{data}}}$ (main path) and $z_{1:T_{\text{data}}}^{\text{fork}}$ (one-step branches for the AiO estimator) are pre-computed from $M = 2$ shock sequences at generation time. Endogenous initial states $(k_0, b_0)$ are sampled uniformly from $\mathcal{S}^{\text{endo}}$ at every round (low-dimensional case). For the one-step ER/BR dataset, the trajectories are flattened into $(k_i, b_i, z_i, z_{i}^{\prime(1)}, z_{i}^{\prime(2)})$ with $(k, b)$ sampled i.i.d. uniform.

The basic model and the risky debt model are trained on the same dataset. The basic model simply ignores $b_0$ and the debt dimension.

## RNG Seed Schedule

Reproducibility requires deterministic random number generation. The framework uses TensorFlow's stateless random functions (`tf.random.stateless_*`) which produce identical outputs given identical seed pairs, regardless of call order.

**Master Seed**

A master seed pair $\mathbf{s}^{\text{master}} = (m_0, m_1)$ anchors all randomness.

**Split Seeds**

Disjoint seeds for each dataset split and round $k$:
$$\mathbf{s}^{\text{train},(k)} = (m_0 + 100, \; m_1 + k), \quad \mathbf{s}^{\text{val},(k)} = (m_0 + 200, \; m_1 + k), \quad \mathbf{s}^{\text{test}} = (m_0 + 300, \; m_1)$$

The test seed has no round index — it is generated once and sealed.

**Variable IDs**

Each random variable has a fixed ID:

| Variable            | ID  |
| ------------------- | --- |
| $k_0$               | 1   |
| $z_0$               | 2   |
| $b_0$               | 3   |
| $\varepsilon^{(1)}$ | 4   |
| $\varepsilon^{(2)}$ | 5   |
| shuffle             | 6   |

**Training Seeds by Round and Variable**

For round $k$ and variable $x$:
$$\mathbf{s}^{\text{train},(k)}_{x} = (m_0 + 100 + \text{VarID}(x), \; m_1 + k)$$

This ensures each round $k$ receives unique, reproducible random draws. Different rounds produce completely independent datasets because the round index $k$ shifts the second seed component.

**Validation Seeds**

Regenerated each round alongside training data:
$$\mathbf{s}^{\text{val},(k)}_{x} = (m_0 + 200 + \text{VarID}(x), \; m_1 + k)$$

**Test Seeds**

Generated once (no round index):
$$\mathbf{s}^{\text{test}}_{x} = (m_0 + 300 + \text{VarID}(x), \; m_1)$$

This schedule guarantees:

1. **Reproducibility**: Identical master seed and round index produce identical data across runs
2. **Fresh data each round**: Different rounds receive independent shock sequences and initial states
3. **Common random numbers**: Different methods trained at the same round $k$ share the same data, enabling fair comparison
4. **No leakage**: Train/validation/test sets use disjoint RNG streams; the test set is independent of the round index


# Deep Learning Methods
This section presents three deep learning solution methods introduced by Maliar et al. (2021) and reviewed by @fern26. All three methods parameterize the policy $\pi_\theta: \mathcal{S} \to \mathcal{A}$ as a neural network and train it by minimizing a loss function $J(\theta)$ via SGD. They differ in which optimality condition defines the loss. For the Bellman residual method, an additional value function network $V_\phi: \mathcal{S} \to \mathbb{R}$ is jointly trained.

Throughout, $\mathcal{D} = \{x_i\}_{i=1}^N$ denotes a replay buffer or a fixed dataset with $N$ i.i.d. observations. Each observation will be defined explicitly and differently depending on the method. For example, $x_i$ may consist of current state $s_i$, current state-action pair $(s_i,a_i)$, or a vector of action-states across periods $(s_i, a_i, s'_i)$.

For each observation $i$, let $\{ \epsilon_{i,m} \}^M_{m=1}$ denote $M$ Monte Carlo i.i.d. draws of the random/stochastic noise from some given distribution and they governs the dynamics through the state transition function $f(s,a,\epsilon)$.

**Monte Carlo (MC) Integration.** @maliar21 proposed an "All-in-One" (AiO) estimator to compute the square of an expectation in theoretical loss. Generally, the loss function is expressed as $\mathbb{E}_s \left[\left(\mathbb{E}_\epsilon[F(x,\epsilon;\theta)]\right)^2\right]$, but the squared expectation $(\mathbb{E}_\epsilon[F])^2$ cannot be estimated with a single shock draw because $\mathbb{E}[F(\epsilon)]^2 \neq \mathbb{E}[F(\epsilon)^2]$. The AiO estimator take two i.i.d draws $\epsilon_1, \epsilon_2$ for each state to form the cross-product:
$$\mathbb{E}_\epsilon[F]^2 = \mathbb{E}_{\epsilon_1}[F(\epsilon_1)] \cdot \mathbb{E}_{\epsilon_2}[F(\epsilon_2)] = \mathbb{E}_{\epsilon_1, \epsilon_2}[F(\epsilon_1) \cdot F(\epsilon_2)]$$
where the equality holds because $\epsilon_1$ and $\epsilon_2$ are independent. In finite-sample, the AiO estimator is computed as:
$$
J^{AiO}(\theta) = \frac{1}{N} \sum_{i=1}^N 
\left[   
F(x_i,\epsilon_{i,1};\theta) \cdot F(x_i,\epsilon_{i,2};\theta)
\right]
$$where $\epsilon_{i,1}$ and $\epsilon_{i,2}$ are two independent draws for observation $i$.

In practice, however, the AiO estimator can be slow and inefficient due to high variance. To address this issue, I adopt the biased-corrected Monte Carlo (MC) estimator analyzed in @Pascal24, which is an unbiased estimator for the squared expectation and with the smallest variance. Specifically, @Pascal24 proved that the AiO estimator is a special case of the MC estimator with $M=2$. Under regularity conditions, the biased-corrected MC estimator is equivalent to the minimum variance unbiased estimator (MVUE) for the loss function and its property has been well-established in the statistics literature:
$$
J^{MC}(\theta) = \frac{1}{N} \frac{2}{M(M-1)} \sum_{i=1}^N  \sum_{1 \leq m \lt k}^M 
\left[   
F(x_i,\epsilon_{i,m};\theta) \cdot F(x_i,\epsilon_{i,k};\theta)
\right]
$$
where $N$ is the number of observations and $M$ is the number of MC draws. When $M=2$ this reduces to the AiO estimator in @maliar21.

## Lifetime Reward Maximization
The Lifetime Reward Maximization (LRM) method directly maximizes expected discounted lifetime rewards by simulating trajectories under the current policy. Given initial state $s_0$ and a shock sequence $\{\epsilon_1, \ldots, \epsilon_T\}$, the policy $\pi_\theta$ generates a trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1}, s_T)$ where $a_t = \pi_\theta(s_t)$ and $s_{t+1} = f(s_t, a_t, \epsilon_{t+1})$.

### Truncated BPTT
**True objective.** The infinite-horizon value under policy $\pi_\theta$ starting from $s_0$ is:
$$V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_t, \pi_\theta(s_t))\right]$$
Splitting at a finite horizon $T$ gives an exact decomposition:
$$V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right] + \gamma^T \, \mathbb{E}\left[V^{\pi}(s_T)\right]$$
where the second term is the discounted expected continuation value from the terminal state $s_T$ onward.

**Truncated objective.** @maliar21 approximate the true objective by dropping the continuation term, setting $\hat{V}^{\text{term}}(s_T) = 0$:
$$\max_\theta \; J_T(\theta) = \mathbb{E}_{(s_0,\,\epsilon_1,\dots,\epsilon_T)}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right]$$
This is valid when $T$ is large enough that $\gamma^T V^{\pi}(s_T) \approx 0$. However, the discount factor contracts this term slowly: with $\gamma = 0.96$, keeping the truncation bias below 1\% of the true value requires $T \geq \lceil\log(0.01)/\log(0.96)\rceil = 113$ periods. BPTT through such a long chain is computationally prohibitive — gradient memory scales linearly in $T$, and vanishing or exploding gradients compound across the chain.

**Key property.** The entire trajectory is generated by composing $\pi_\theta$ and $f$, so the loss is end-to-end differentiable — gradients flow backward through the trajectory via backpropagation through time. This requires $r$ and $f$ to be differentiable with respect to the action.

### Terminal Value Correction
Rather than increasing $T$ to reduce the truncation bias, we approximate the continuation value by exploiting the structure of the MDP. The true continuation value from the terminal state $s_T$ is:
$$V^{\pi}(s_T) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_{T+t},\, \pi(s_{T+t})) \;\middle|\; s_T\right]$$
which integrates over all future shock realizations and the policy's dynamic response to them. We replace this with a closed-form approximation that uses only model primitives.

**Terminal value formula.** Define:
$$\bar{s} = [s^{\text{endo}} \mid \bar{s}^{\text{exo}}], \qquad \bar{a} = \bar{a}(s^{\text{endo}})$$
where $\bar{s}^{\text{exo}} = \mathbb{E}[s^{\text{exo}}_\infty]$ is the stationary mean of the exogenous process, and $\bar{a}(s^{\text{endo}})$ is the action satisfying $f^{\text{endo}}(s^{\text{endo}}, \bar{a}) = s^{\text{endo}}$, the action that holds the endogenous state constant. Both are functions of $s^{\text{endo}}$ alone and are model constants provided by the environment. The approximation replaces the stochastic future with a deterministic steady state in which the exogenous state is frozen at its mean and the agent repeats the stationary action forever. The continuation then reduces to a geometric perpetuity. During training, we evaluate it at $s_{T}^{\text{endo}}$, the terminal endogenous state obtained by rolling out the current policy $\pi_{\theta}$ for $T$ steps:
$$\hat{V}^{\text{term}}(s_T^{\text{endo}}) = \frac{r(\bar{s},\, \bar{a})}{1 - \gamma}$$
The formula is a fixed function of $s^{\text{endo}}$ that does not depend on $\theta$.

**Loss.** The SGD loss with terminal value correction, evaluated over a mini-batch $\mathcal{B}$:
$$J(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\sum_{t=0}^{T-1} \gamma^t \cdot r(s_{it}, \pi_\theta(s_{it})) + \gamma^T \, \hat{V}^{\text{term}}(s_{iT}^{\text{endo}})\right)$$
Setting $\hat{V}^{\text{term}} = 0$ recovers the truncated objective of @maliar21. In the BPTT computation, $\hat{V}^{\text{term}}$ should be differentiable with respect to $s_{T}^{\text{endo}}$ so that gradients prevent the policy from de-investing near the horizon, but should not route gradients through the policy network at the terminal step to avoid $1/(1-\gamma)$ gradient amplification through the BPTT chain.

The gap between $\hat{V}^{\text{term}}$ and the true continuation $V^{\pi}$ arises because the perpetuity ignores future exogenous volatility and the agent's dynamic response to it. We analyze the magnitude and structure of this approximation error below.

### Algorithm 4: Lifetime Reward Maximization
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

### Why the Terminal Value Correction Works

The true continuation value $V^{\pi}(s_T) = \mathbb{E}[\sum_{t \geq T} \gamma^{t-T} r(s_t, \pi(s_t)) \mid s_T]$ integrates over all future shock paths and the agent's dynamic response to them. The perpetuity formula approximates it by exploiting the endogenous-exogenous state decomposition.

**What the correction exploits.** At the terminal step $T$, we treat the two state components differently:

- $s_T^{\text{endo}}$ is taken as given: the actual endogenous state produced by the BPTT rollout.
- $s_T^{\text{exo}}$ is replaced by its stationary mean $\bar{s}^{\text{exo}} = \mathbb{E}[s^{\text{exo}}_\infty]$, computed from the known exogenous process.
- The action is set to $\bar{a}(s^{\text{endo}})$, the unique action satisfying $f^{\text{endo}}(s^{\text{endo}}, \bar{a}) = s^{\text{endo}}$.

All three ingredients (the exogenous mean, the stationarity condition, and the reward function) are model primitives. No learned quantity is needed.

**Approximation gap.** The error of the perpetuity relative to the true continuation is:
$$\hat{V}^{\text{term}}(s_T^{\text{endo}}) - V^{\pi}(s_T) = \frac{r(\bar{s},\, \bar{a})}{1-\gamma} - \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_{T+t},\, \pi(s_{T+t})) \;\middle|\; s_T\right]$$

To understand its magnitude, consider the idealized case where $s_T^{\text{endo}}$ is at the optimal steady state $s^{*,\text{endo}}$. The perpetuity gives the reward at the deterministic steady state, while $V^{\pi}$ accounts for the agent's optimal response to future stochastic shocks. By the **envelope theorem**, the first-order effect of small shocks on the value function vanishes: the agent is already optimizing, so marginal perturbations in the exogenous state are absorbed by optimal policy adjustment. The gap is therefore **second-order in the exogenous volatility** $\sigma$:
$$\hat{V}^{\text{term}}(s^{*,\text{endo}}) - V^{\pi}(s^{*,\text{endo}}, \bar{s}^{\text{exo}}) = O(\sigma^2)$$

The $O(\sigma^2)$ residual has two components:

1. **Jensen's correction.** The value function is generally concave in the exogenous state (due to diminishing returns). Replacing the stochastic $s^{\text{exo}}$ with its mean overstates the value by $O(\text{Var}(s^{\text{exo}}_\infty) \cdot V''_{s^{\text{exo}}})$.
2. **Precautionary motive.** An agent facing adjustment costs benefits from the *option to respond* to future shocks. The perpetuity, which assumes a fixed action forever, ignores this option value.

Both terms depend on the curvature of $V^{\pi}$, specifically how the optimal policy adjusts dynamically to exogenous fluctuations. This is precisely the object that $\pi_\theta$ is being trained to learn. Reducing the residual further would require an explicit approximation of $V^{\pi}$, at which point the algorithm becomes a critic-based method (e.g., actor-critic or Bellman residual minimization), a fundamentally different algorithmic approach. Within the class of corrections that use only known model structure and no learned value function, the perpetuity formula is essentially tight.

**Endogenous convergence.** The analysis above assumes $s_T^{\text{endo}}$ is near the optimal steady state. How quickly this holds depends on the contraction rate of the endogenous dynamics, not on the discount factor $\gamma$:

- When adjustment costs are low, the policy can move $s^{\text{endo}}$ to its target rapidly; even $T = 1$ may suffice.
- When adjustment costs are high, convergence is gradual and a larger $T$ is needed.

In either case, the terminal value correction replaces the discount-rate-dependent horizon requirement ($T \geq 113$ at $\gamma = 0.96$) with a dynamics-dependent one, which is typically far less demanding.

**Self-correcting training dynamics.** Early in training, the policy has not converged, so the rollout may not reach the steady state even when $T$ is adequate. The terminal value is then a rough approximation. However, as the policy improves, rollout trajectories increasingly reach the neighborhood of the steady state, making the terminal value more accurate, which in turn provides a better gradient signal. The approximation is most reliable precisely when it matters most — in the later stages of training when the policy is being fine-tuned near the optimum.

## Euler Residual Minimization
The ER method minimizes violations of the first-order conditions (Euler equations) that characterize optimality. Rather than simulating full trajectories, it enforces an intertemporal necessary condition between $(s,a)$ and $(s',a')$.

**Euler equation.** At the optimum, the policy $\pi_\theta$ satisfies:
$$\mathbb{E}_\epsilon \left[F(s, \pi_\theta(s), s', \pi_\theta(s'))\right] = 0$$
where $F: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the Euler residual function derived analytically from the first-order conditions of the Bellman equation. One time-step transition is calculated with $s' = f(s, \pi_\theta(s), \epsilon)$, and the expectation is over the next-period shock $\epsilon$. The specific form of $F$ depends on the model. For example, firms equating the marginal cost of investment or households equating the marginal utilities of saving across periods.

**Loss function with AiO integration:**
$$J(\theta) = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} 
F\left(s_i, \pi_\theta(s_i), s'_{i,1}, \pi_{\theta^-}(s'_{i,1})\right) \cdot F\left(s_i, \pi_\theta(s_i), s'_{i,2}, \pi_{\theta^-}(s'_{i,2})\right) $$
where the superscript $(m)$ denotes the MC draws, and $s'_{i,m} = f(s_i, \pi_\theta(s_i), \epsilon_{i,m})$ for $m = 1, 2$. Note that $\pi_{\theta^-}(s')$ is a separate _target_ network for next period action, which is different from the _current_ network $\pi_\theta(s)$ for current action.

**Target network.** Both @maliar21 and @fern26 suggest a single policy network inside the loss function construction. However, computing $\pi_\theta(s')$ introduces a recursive dependency: the gradient of $\theta$ flows through both the current policy $\pi(s)$ and next-period policy $\pi(s'$) evaluations. This is a critical issue that prevent training from convergence because the gradient update is chasing a moving target. 

To stabilize training, I introduce an additional target network with parameters $\theta^-$ for the next-period action: $\pi_{\theta^-}(s')$. The target parameters are updated via Polyak averaging after each SGD step: $\theta^- \leftarrow \nu \theta^- + (1 - \nu) \theta$ with $\nu$ close to 1.
### Algorithm 5: Euler Residual Minimization
**Input:** Policy network $\pi_\theta$, dynamics $f$, Euler residual $F$, discount $\gamma$, learning rate $\eta$, Polyak rate $\nu$, convergence rule $\texttt{CONVERGED}(θ, j)$

**Output:** Trained policy $\pi^*_{\theta}$

1. Initialize policy parameters $\theta$, target parameters $\theta^- \leftarrow \theta$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$ Sample mini-batch $\mathcal{B}$ of states $\{s_i\}$ with two independent shock draws $\{\epsilon_{i,m}\}_{m=1,2}$ 
4. $\quad$ Compute current actions: $a_i = \pi_\theta(s_i)$
5. $\quad$ Compute next states: $s'_{i,m} = f(s_i, a_i, \epsilon_{i,m})$ for $m = 1,2$
6. $\quad$ Compute next actions using target network: $a'_{i,m} = \pi_{\theta^-}(s'_{i,m})$
7. $\quad$ Compute Euler residuals: $F_{i,m} = F(s_i, a_i, s'_{i,m}, a'_{i,m})$ for $m = 1, 2$
8. $\quad$ Compute loss: $J(\theta) = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} F_{i,1} \cdot F_{i,2}$
9. $\quad$ Update: $\theta \leftarrow \theta - \eta \nabla_\theta J(\theta)$
10. $\quad$ Polyak update: $\theta^- \leftarrow \nu \theta^- + (1 - \nu) \theta$
11. $\quad$ **If** $\texttt{CONVERGED}(θ, j)$ **then break**
12. **End for**
13. **Return** $\pi_{\theta^*}$

## Multitask Bellman Residual Minimization (BRM)
The BR method jointly trains a policy network $\pi_\theta$ and a value function network $V_\phi$ to satisfy the Bellman equation. The challenge is that the Bellman equation contains a $\max$ operator:
$$V(s) = \max_a \left\{ r(s, a) + \gamma \mathbb{E}_\epsilon\left[V(s')\right] \right\}$$
Rather than solving the inner maximization directly (which requires an actor-critic method, introduced in later sections), @maliar21 and fern26 propose eliminating the $\max$ by enforcing necessary optimality conditions as auxiliary losses. This turn the loss function into a multitask objective that combines the Bellman residual, the first-order condition (FOC), and other model-specific constraints and optimality conditions with user-specified exogenous weights. Throughout the paper, I call it multitask BRM to distinguish from other methods that also minimizes the Bellman residual.

**Bellman residual.** For a given policy $\pi_\theta$ and value function $V_\phi$, define the Bellman equation residual for each observation $i$:

$$F^{\text{BR}}_{i,m} = V_\phi(s_i) - r(s_i, a_i) - \gamma V_\phi(s'_{i,m})$$

where $a_i = \pi_\theta(s_i)$ and $s'_{i,m} = f(s_i, a_i, \epsilon_{i,m})$. At the solution, $\mathbb{E}_\epsilon[F^{\text{BR}}] = 0$.

**FOC residual.** Differentiating the Bellman RHS with respect to the action $a$ yields the necessary condition:

$$F^{\text{FOC}}_{i,m} = \nabla_a r(s_i, a)\big|_{a = a_i} + \gamma \nabla_{s'} V_\phi(s'_{i,m}) \cdot \nabla_a f(s_i, a, \epsilon_{i,m})\big|_{a = a_i}$$

At the optimum, $\mathbb{E}_\epsilon[F^{\text{FOC}}] = 0$.

**Envelope condition residual.** Differentiating the Bellman equation with respect to the state $s$ (applying the envelope theorem) gives:

$$F^{\text{Env}}_i = \nabla_s r(s_i, a_i) - \nabla_s V_\phi(s_i)$$

This condition involves no expectation over future shocks, so the loss uses a direct squared residual.

**Feasibility constraints residual (Optional).** When the model needs to satisfy feasibility constraints given as:
$$
G(\cdot;\theta) \leq 0 \quad \text{and} \quad H(\cdot;\theta) = 0
$$
where $G(\cdot)$ and $H(\cdot)$ can be either linear or non-linear functions over states, actions, or state-action pairs. @maliar21 and @fern26 proposed using two additional network to approximate the Lagrangian multiplier on each of the constraint, and construct separate loss that measures the empirical residual of the constraint. Let $\mathcal{L}^{IC}$ and $\mathcal{L}^{EC}$ denote the loss for the inequality and equality constraints, respectively, and they will be added into the total loss function with exogenous weight. Here I treat them as placeholder and I will provide a formal treatment of the constraint loss in Section [X]. I show that this approach is fragile in implementation and can prevent the method from convergence. Instead, I introduce an alternative Softplus penalty approach that is commonly used in the state-of-the-art RL algorithms.

**Total loss with AiO integration:**

$$J(\theta, \phi) = \underbrace{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} F^{\text{BR}}_{i,1} \cdot F^{\text{BR}}_{i,2}}_{\mathcal{L}^{\text{BR}}} 
+ w_1 \underbrace{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} F^{\text{FOC}}_{i,1} \cdot F^{\text{FOC}}_{i,2}}_{\mathcal{L}^{\text{FOC}}} 
+ w_2 \underbrace{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} (F^{\text{Env}}_i)^2}_{\mathcal{L}^{\text{Env}}} + w_3 \mathcal{L}^{\text{IC}} + w_4 \mathcal{L}^{\text{EC}}$$

where $w_1, w_2 > 0$ are exogenous weights that must be tuned _manually and carefully_ because $\mathcal{L}^{\text{BR}}$ is measured in value levels (squared Bellman residual), but $\mathcal{L}^{\text{FOC}}$ and $\mathcal{L}^{\text{Env}}$ are measured in derivatives and feasibility constraints $\mathcal{L}^{\text{IC}}$ and $\mathcal{L}^{\text{EC}}$ are measured in arbitrary units (depending on the model).

**Fundamental architecture defect**. In practice, the multitask-BR method is extremely sensitive to the choice of exogenous weights and the unit (scale) of each loss. @maliar21 recommend fine tuning of the weights in pre-training to make the magnitude of each loss roughly the same. Nevertheless, @maliar21 still find this method to perform significantly worse than the other methods and does not provide further diagnostics. 

In section X, I show that although pre-training and fine tuning helps, this method is fragile because of a **fundamental architecture error**. The main cause is that the multiple losses are not all serving a shared goal, which allow for the NN to cheaply minimize the total loss for a variety of wrong policies while satisfying all the constraints.

For example, there exist multiple solutions that can minimize the total loss by "ignoring" the FOC residual (which is necessary for optimality) and substitute for the Bellman residual loss (which can be minimized for any arbitrary policy) and other feasibility losses (can be flexibly satisfied by interior points). A simple and straightforward solution is to set a huge positive weight on the FOC residual to make it dominant over other losses, but this trivially reduce to the Euler residual minimization method. A formal discussion of this critical design issue is discussed in section X.

**Target Network.** As in the ER method, the value network $V_\phi(s'_{i,m})$ introduces a recursive gradient dependency: the SGD update to $\phi$ changes both the current-state evaluation $V_\phi(s_i)$ and the target $V_\phi(s'_{i,m})$ simultaneously. Maliar et al. (2021) do not address this; the actor-critic method in Section X resolves it via target networks and separated updates.

### Algorithm 6: Multi-Task Bellman Residual Minimization 

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






# Implementation Details
This section discusses main issues that need to be handled carefully in implementation.

## Input Normalization

Normalization is essential for stable neural network training. Without it, state variables on different scales produce gradients of vastly different magnitudes, causing some parameters to update too aggressively while others stagnate. The goal is a uniform transform that works across economic and financial models without domain-specific tuning. State variables in these applications span orders of magnitude: capital stocks in the hundreds, log-productivity shocks near zero, interest rates as fractions, wealth in the millions. The normalizer must compress all of these to a comparable range.

I only apply observation (input) normalization and do not apply hidden-layer normalization. For the economic models and network architectures considered here, input normalization is sufficient.

### Method: static Z-score fitted from the training dataset

Each feature $x_d$ of the full state $s = (s^{\text{endo}}, s^{\text{exo}})$ is normalized by:

$$\hat{x}_d = \frac{x_d - \mu_d}{\sigma_d + \varepsilon}$$

where $\mu_d$ and $\sigma_d$ are the per-feature mean and standard deviation computed from the full training dataset $\mathcal{D}^{(k)}$ at the start of round $k$, and $\varepsilon$ is a small constant for numerical stability. The statistics are frozen for the entire round and recomputed from fresh data at the start of each subsequent round. No updates occur during gradient training.

Unlike running z-score normalizers common in online RL that estimate statistics incrementally during training, this approach computes exact population statistics from the pre-generated dataset in a single pass before gradient steps begin. Since the full dataset is available before training starts, incremental estimation adds no information and only introduces unnecessary hyperparameters.

### Exogenous and endogenous components

The two state components are fitted from different data sources, reflecting what is available in the dataset without a policy rollout.

**Exogenous states** $s^{\text{exo}}$ (e.g., productivity $z$): the pre-simulated trajectory provides $N \times T_{\text{data}}$ samples of the ergodic exogenous distribution. All $T_{\text{data}}$ steps per trajectory are used for fitting, regardless of the training rollout horizon ($T_{\text{train}}$ for LR, $h$ for SHAC). Using the full $T_{\text{data}}$ ensures the normalizer captures the ergodic distribution of $z$ rather than the narrower initial distribution of $z_0$.

**Endogenous states** $s^{\text{endo}}$ (e.g., capital $k$, debt $b$): only the $N$ initial states $s^{\text{endo}}_0$ are available without a policy rollout, so these are used. At Round 0 with $s^{\text{endo}}_0 \sim \text{Uniform}(\mathcal{S}^{\text{endo}})$, the statistics cover the full state space range. At Round $k \geq 1$ with ergodic harvesting, the initial states approximate the ergodic distribution under $\pi_{\theta_k}$, giving tighter statistics.

In practice, fitting is implemented by pairing each $s^{\text{endo},i}_0$ with all $T_{\text{data}}$ exogenous states from trajectory $i$, producing a single $N \times T_{\text{data}}$ array from which both sets of statistics are extracted in one pass. For one-step methods (ER, BRM), the flattened dataset already provides $N \times T_{\text{data}}$ paired $(s^{\text{endo}}, s^{\text{exo}})$ observations and the normalizer is fitted directly.

The normalizer does not need to be ergodically exact. Since the state space is bounded by construction, all states visited during training fall within a fixed interval after normalization. The purpose is to map inputs to an $O(1)$ range before the first network layer, equalizing gradient scale across features. The optimizer's adaptive learning rates absorb residual scale differences within a few hundred steps.

### Alternatives considered

**Standard logarithm** $\ln(1 + x)$: works well for $x \geq 0$, but undefined for negative values. Variables such as log-productivity, returns, or value functions that take negative values require special-casing, breaking the goal of a uniform transform.

**Hyperbolic tangent** $\tanh(x)$: achieves near-perfect cross-variable equalization but saturates for $|x| > 3$, mapping all large values to $\pm 1$. For a variable like capital with range $[0, 500]$, the network cannot distinguish $k = 100$ from $k = 500$ after $\tanh$ compression.


## Activation and Gradient Saturation

### Hidden-Layer Activation
The hidden-layer activation $\phi$ introduces nonlinearity between layers. Two standard choices are:

$$\mathrm{ReLU}(h) = \max(0, h), \qquad \mathrm{SiLU}(h) = h \cdot \sigma(h)$$

where $\sigma(\cdot)$ is the logistic sigmoid. ReLU is simple and widely used but has zero gradient for $h < 0$. Any neuron whose pre-activation is negative for all training samples receives no gradient and cannot recover. When combined with LayerNorm, which centers pre-activations around zero, roughly half of the neurons fall in the $h < 0$ region at any given sample, making dead neurons a persistent concern.

SiLU avoids this: its gradient $\phi'(h) = \sigma(h)(1 + h(1 - \sigma(h)))$ is nonzero for all $h \in \mathbb{R}$, ensuring that every neuron receives gradient signal regardless of the sign of its pre-activation. SiLU is smooth (infinitely differentiable), which is a better match for the smooth, concave objective functions typical in economic models. I adopt SiLU as the default hidden activation.
### Output Head Transformation
The output head maps the final hidden representation to economic-level variables. Its design has a direct impact on gradient quality and learning stability. I use a single universal output head across all variable types: an identity (linear) mapping followed by variable-specific clipping.
#### The problem with bounded activations
A common approach for enforcing action bounds in RL is to apply a squashing function at the output layer. The standard choice is $\tanh$, which maps $\mathbb{R} \to (-1, 1)$ and is used in SAC, DreamerV3, and other continuous-control algorithms. For a variable with bounds $[a_{\min}, a_{\max}]$, the output is
$$a = a_{\min} + \frac{a_{\max} - a_{\min}}{2}\bigl(\tanh(\hat{y}) + 1\bigr)$$

where $\hat{y}$ is the raw pre-activation. This guarantees $a \in (a_{\min}, a_{\max})$ by construction, but introduces a structural gradient problem.

Any differentiable bijection $g: \mathbb{R} \to (a_{\min}, a_{\max})$ must satisfy $g'(\hat{y}) \to 0$ as $g(\hat{y}) \to a_{\min}$ or $g(\hat{y}) \to a_{\max}$. This is a topological necessity: if the derivative stayed bounded away from zero, the function would be unbounded, contradicting the finite range. For $\tanh$, the gradient is $\partial a / \partial \hat{y} \propto 1 - \tanh^2(\hat{y})$, which vanishes as $|\hat{y}| \to \infty$.

In standard RL benchmarks (MuJoCo, DMControl), this is a minor issue because optimal policies rarely require actions at or near the bounds and the optimal torques and forces are typically interior. In economic and financial models, the situation is different. When a state variable such as productivity is high, the optimal policy (e.g., investment or capital accumulation) can be near the upper end of its feasible range. If the loss function itself also has a decaying gradient in the same region, the two effects compound and lead to weak learning signal near boundaries.

Concretely, in gradient-based training, the parameter update is proportional to $\partial \mathcal{L}/\partial \theta$. By the chain rule, this factors as
$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \theta}$$
where $\partial \mathcal{L}/\partial a$ is the gradient of the loss with respect to the action (determined by the economic model), $\partial a/\partial \hat{y}$ is the gradient of the output activation, and $\partial \hat{y}/\partial \theta$ is the gradient through the hidden layers. The third factor is precisely what the normalization strategy (Symlog + LayerNorm) are designed to keep well-conditioned. But the first two factors act as a scalar multiplier on the entire parameter gradient. When both decay in the same region of the state space --- the economic gradient due to diminishing marginal returns, the activation gradient due to saturation near the bounds --- the parameter update becomes negligibly small for those states, producing a weak learning signal and larger errors near the bounds.

This is not only a theoretical concern. In my experiments with a frictionless capital accumulation model, I find that using the standard tanh squashing function cause the learned policy systematically deviates from the analytical benchmark near the boundaries of the action space, with larger errors for states where the optimal action is in the high range.
#### Solution: linear output with clipping
To address the vanishing gradient issue due to bounded activation function, I instead use a linear output head with clipping:

$$\hat{y} = w^{\mathrm{out}\top} \mathbf{a}^J + b^{\mathrm{out}}$$
For variables with box constraints, feasibility is enforced by clipping:

$$a = \mathrm{clip}(\hat{y},; a_{\min},; a_{\max})$$

where either bound may be $\pm\infty$ (i.e., inactive). The gradient of the clipped output is

$$\frac{\partial a}{\partial \hat{y}} = \begin{cases} 1 & \text{if } a_{\min} < \hat{y} < a_{\max} \\ 0 & \text{if } \hat{y} \leq a_{\min} \text{ or } \hat{y} \geq a_{\max} \end{cases}$$

Inside the feasible region, the gradient is exactly and uniformly one for all states, with no dependence on distance to the boundary. This means that the learning signal is still strong near the corners. This is the critical difference from bounded activations, where the gradient degrades continuously as the output approaches a bound. The zero gradient outside the feasible region is benign: clipped samples contribute the correct action ($a_{\min}$ or $a_{\max}$) even though they provide no gradient, and unclipped samples in the interior provide the learning signal that shapes the function approximation.

This design is used in several continuous-control RL algorithms. In TD-MPC2, the MPPI planner optimizes in unconstrained action space and clips to the environment bounds. In PPO and DDPG, actions are sampled from an unbounded distribution and clipped before being applied. The $\tanh$-squashing convention used in SAC and DreamerV3 exists largely because standard benchmarks normalize actions to $[-1, 1]$, making $\tanh$ a convenient default; it is not motivated by gradient considerations.

#### Special case: non-box constraints
Clipping enforces element-wise box constraints but cannot handle constraints that couple multiple variables or depend on future states. For a general constraint $G(s, a) \leq 0$ or $G(s, a, s') \leq 0$ that is not reducible to per-variable bounds, a soft penalty approach is required. This is described in detail in the section on constraint handling


## Bellman Equation Normalization
For any method that construct loss based on the Bellman equation error, the main source of instability is the lack of normalization. Specifically, Bellman equation is measured in raw economic levels but most gradient-based training only works well within a normalized bounded range. Consider a model where reward is firm's profit and thus the Bellman RHS is the sum of lifetime discounted cash flow in millions, then a Bellman error (LHS-RHS) of about 100 is pretty accurate but it squared to $100^2$ which is still huge for a metric of loss. Any bootstrap error on estimating the RHS also get compounded and can lead to divergence. More severely, when losses of different magnitude are combined into a single loss function as in the multitask-BRM method, the scale mismatch between the losses can easily lead to wrong solutions because one or more loss with larger magnitude can easily dominant others in gradient direction. A typical example is the network learned to prioritize minimizing the Bellman error (which dominate the loss reduction) while ignoring the optimality constraints, eventually lead to self-consistent but wrong policies.

To address this issue, I consider a simple and effective solution that normalized the value function network with a constant $C$. The choice of $C$ depends on the model, but there is a generic approach to approximate the value function without model-specific knowledge.

The critic network outputs normalized values:

$$\tilde{Q}_\phi(s, a) = Q_\phi(s, a) / C$$

The critic loss is computed entirely in normalized space:

$$J_{\text{critic}}(\phi_k) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{1}{T} \sum_{t=0}^{T-1} \left(\tilde{Q}_{\phi_k}(s_{i,t}, a_{i,t}) - \hat{Q}_{i,t} / C \right)^2$$

Since $C$ is a fixed scalar, this rescales the loss by $1/C^2$ with no architectural changes and no running statistics. Wherever level-space Q values are needed (the terminal bootstrap and the actor objective), they are recovered as $Q_{\phi_k^-} = C \cdot \tilde{Q}_{\phi_k^-}$.

### Choosing $C$

$C$ is fixed before any gradient step and held constant for the entire training run. Fixing $C$ at initialization is essential: recomputing $C$ during training changes the loss scale mid-run, which corrupts the second-moment estimates of adaptive optimizers such as Adam and can destabilize training.

The key observation is that the magnitude of $Q$ is determined by the reward scale and discount factor, not by the policy:

$$|Q(s,a)| \leq \frac{\mathbb{E}[|r|]}{1-\gamma}$$

This bound holds for any policy. A bad policy collects bad rewards and a good policy collects good rewards, but both are $\mathcal{O}(\mathbb{E}[|r|]/(1-\gamma))$. This means $C$ can be estimated from the reward function alone, without any policy rollout.

**Default procedure (policy-independent).** Before training, sample $N$ state-action pairs $(s_i, a_i)$ with $s_i$ drawn uniformly from a bounded region of $\mathcal{S}$ and $a_i$ drawn uniformly from $\mathcal{A}$. Evaluate the known reward function at each pair and compute:

$$\hat{\mu}_{|r|} = \frac{1}{N}\sum_{i=1}^N |r(s_i, a_i)|, \qquad C = \frac{\hat{\mu}_{|r|}}{1 - \gamma}$$

The bounded sampling region should be chosen to cover the ergodic distribution of the model — in practice, any reasonable compact subset of the state space suffices, since the ergodic distribution is always concentrated on a bounded region even when $\mathcal{S}$ is unbounded in principle. No dynamics evaluation, no rollout, and no policy are required; only the reward function $r$, which is known exactly by assumption. $N = 1000$ random samples is more than sufficient since $\hat{\mu}_{|r|}$ converges at rate $1/\sqrt{N}$ and order-of-magnitude accuracy is all that is needed.

**Preferred alternative: analytical $V^*$.** When the model admits an analytical or semi-analytical steady state with computable $V^* = Q^*(s^*, \pi^*(s^*))$, set $C = |V^*|$ directly. This gives a tighter estimate than the reward sampling procedure because $V^*$ already accounts for the discount-weighted accumulation of rewards along the optimal path, whereas $\hat{\mu}_{|r|}/(1-\gamma)$ is an upper bound that may overestimate $C$ when the ergodic reward distribution is skewed. An exact computation of $V^*$ is not required; an approximation within the correct order of magnitude is sufficient.


# Applications to Corporate Finance
Now I apply the methods to the canonical corporate finance model. The first step is to cast the model into the MDP.
## Basic Model of Optimal Investment 

### Definitions

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

### Network Architecture

**Policy Network (All methods)** $\quad k' = \pi_\theta \equiv \pi(k, z;; \theta_{\text{policy}})$

- Input: $(k, z)$ in raw levels, normalized internally
- Hidden layers (default): 2 layers with 32 units each, SiLU activation
- Output: linear identity with clip $[k_{\min},; k_{\max}]$

**Critic Network (MVE only)** $\quad Q_\phi \equiv Q(k, z, k'; \phi)$

- Input: $(k, z, k')$ in raw levels, normalized internally
- Hidden layers (default): 2 layers with 32 units each, SiLU activation
- Raw output: $\hat{y}_\phi \approx \operatorname{symlog}(Q)$, linear identity without clip
- Level-space recovery: $Q_\phi = \operatorname{symexp}(\hat{y}_\phi)$, used in actor objective and terminal bootstrap

**Critic Network (Multitask-BR only)** $\quad V_\phi \equiv V(k, z; \phi)$

- Input: $(k, z)$ in raw levels, normalized internally
- Hidden layers (default): 2 layers with 32 units each, SiLU activation
- Raw output: linear identity $V$ without clip 
- Note: I follow the specification from @maliar21 and @fern26 and do not apply value normalization




## Risky Debt Model under Partial Equilibrium
The risky debt model extends the basic model by allowing firms to borrow at an endogenous risky interest rate, with the option to default. Then risky interest rate is determined by the lender's zero profit condition with rational expectation of default probability. Firm's optimal investment and leveraging in turn depends on the equilibrium risky interest rate.

### Definitions
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
r(s,a) \equiv e(k,b,z;I,b') - \Omega(e(\cdot))
$$
Cash flow is given as
$$e(k,b,z;I,b') \equiv (1-\tau)\Pi(k,z) - \Psi(I,k) - I -b + \frac{b'}{1+\tilde{r}(\cdot)} + \frac{\tau \, \tilde{r}(\cdot) \, b'}{[1+\tilde{r}(\cdot)](1+\bar r)} $$
where 
- $\tau$ is the corporate tax rate
- $b$ is repayment of last-period debt
- $\bar r$ is risk-free interest rate
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
$$b'(1+\bar r) = (1+\tilde{r}) b' \, \mathbb{E}_\epsilon[1-D] + \mathbb{E}_\epsilon[D \cdot \text{Recovery}(k',b',z')]$$
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
\widetilde{V}(k,b,z) &= \max_{k',b'} \left\{ r(k,b,z) + \gamma \, \mathbb{E}_{\epsilon}[V(k',b',z')] \right\}\\
&= \max_{k',b'} \left\{ r(k,b,z) + \gamma \, \mathbb{E}_{\epsilon}[\max\{0, \widetilde{V}(k',b',z')\}] \right\}
\end{aligned}
$$
where the RHS continuation value encodes limited liability (firm can walk away with zero).

**The Nested Fixed-Point Problem**
A key computational challenge is that the latent value $\widetilde{V}$ depends on the risky rate $\tilde{r}$, but solving for $\tilde{r}$ requires knowing the default probability $\mathbb{E}[D]$, which depends on $\widetilde{V}$. Traditional methods solve this via nested iteration. The neural network approach trains policy, value, and pricing networks jointly, avoiding explicit nested loops.

### Network Architecture

**Policy Network** $\quad (k', b') = \pi_\theta \equiv \pi_{\text{policy}}(k, b, z; \theta_{\text{policy}})$

- Input: $(k, b, z)$ in raw levels, normalized internally
- Hidden layers: shared trunk with 2 layers, 32 units, SiLU activation
- Output heads (linear identity with variable-specific clips):
    - $k' = \text{clip}(\text{raw}_k,; k_{\min},; k_{\max})$
    - $b' = \text{clip}(\text{raw}_b,; 0,; b_{\max})$

**Price Network** $\quad q = \Gamma_{\text{price}}(k', b', z;; \theta_{\text{price}})$

- Input: $(k', b', z)$ in raw levels, normalized internally
- Hidden layers: 2 layers, 32 units, SiLU activation
- Output: $q = \text{clip}(\text{raw}_q,; 0,; \bar{q})$ where $\bar{q} = 1/(1+r)$

**Critic Network (MVE)** $\quad Q_\phi \equiv Q(k, b, z, k', b';; \phi)$

- Input: $(k, b, z, k', b')$ in raw levels, normalized internally
- Hidden layers: 2 layers, 32 units, SiLU activation
- Raw output: $\hat{y}_\phi \approx \operatorname{symlog}(Q)$, linear identity without clip
- Level-space recovery: $Q_\phi = \operatorname{symexp}(\hat{y}_\phi)$

Two things worth noting. First, the price network now uses identity + clip like everything else, replacing the sigmoid output in earlier version. The economic constraint $q \in [0, \bar{q}]$ is still enforced exactly, but without the vanishing gradient near $\bar{q}$ that sigmoid would cause, which matters because near-risk-free bonds (high $q$) when default risk is high are economically common states. If we use bounded sigmoid output then it lead to underestimation of the bond price even when default risk is high, and push firms to strictly prefer debt over investment. Second, $\widetilde{V}$ can be negative in the default model (triggering default when $\widetilde{V} < 0$), and symlog handles negative values natively, so no special-casing is needed.
