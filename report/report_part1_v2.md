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
| $\gamma \in (0, 1)$                                                    | **Discount factor**. Controls the trade-off between immediate and future rewards.                                                                                                                                                                                                                                                                                                                                                 |
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
The Lifetime Reward Maximization (LRM) method directly maximizes expected discounted lifetime rewards by simulating trajectories under the current policy. Given initial state $s_0$ and a shock sequence ${\epsilon_1, \ldots, \epsilon_T}$, the policy $\pi_\theta$ generates a trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1}, s_T)$ where $a_t = \pi_\theta(s_t)$ and $s_{t+1} = f(s_t, a_t, \epsilon_{t+1})$.

**Objective.** Maximize the discounted sum of rewards over a finite horizon $T$ with a potential terminal value correction:
$$\max_\theta \mathbb{E}_{(s_0,\epsilon_1,\dots,\epsilon_T)} \left[\sum_{t=0}^{T-1} \gamma^t, r(s_t, \pi_\theta(s_t)) + \gamma^T V^{\text{term}}(s_T)\right]$$
@maliar21 implicitly set the terminal value $V^{\text{term}}(s_T)=0$, which is only a valid approximation when $T$ is large and the discount $\gamma$ is moderately small. When we have good reason to believe that the action has converged to a steady state  $\bar{a}=a_T$ by period $T$, I recommend adding the terminal value as $V^{\text{term}}(s_T) = \frac{r(s_T, \bar{a})}{1 - \gamma}$ that explicitly correct for the finite-horizon bias.

**Loss.** The SGD loss is the negated objective evaluated over a mini-batch $\mathcal{B}$:
$$J(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\sum_{t=0}^{T-1} \gamma^t \cdot r(s_{it}, \pi_\theta(s_{it})) + \gamma^T V^{\text{term}}(s_{iT})\right)$$
**Key property.** The entire trajectory is generated by composing $\pi_\theta$ and $f$, so the loss is end-to-end differentiable — gradients flow backward through the trajectory via backpropagation through time. This requires $r$ and $f$ to be differentiable with respect to the action.
### Algorithm 4: Lifetime Reward Maximization
**Input:** Policy network $\pi_\theta$, dynamics $f$, reward $r$, discount $\gamma$, horizon $T$, learning rate $\eta$, convergence rule $\texttt{CONVERGED}(θ, j)$

**Output:** Trained policy $\pi^*_{\theta}$

1. Initialize policy parameters $\theta$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$ Sample mini-batch $\mathcal{B}$ consisting of initial states $\{s_0\}_i$ and shock sequences $\{\epsilon_1,\dots,\epsilon_T\}_{i}$
4. $\quad$ **For** each observation $i \in \mathcal{B}$, rollout trajectory:
5. $\qquad$ **For** $t = 0, \ldots, T-1$: simulate $a_{i,t} = \pi_\theta(s_{i,t})$ and $s_{i,t+1} = f(s_{i,t}, a_{i,t}, \epsilon_{i,t+1})$
6. $\quad$ Compute loss: $J(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\sum_{t=0}^{T-1} \gamma^t \cdot r(s_{it}, \pi_\theta(s_{it})) + \gamma^T V^{\text{term}}(s_{iT})\right)$
7. $\quad$ SGD update: $\theta \leftarrow \theta - \eta \cdot \nabla_\theta J(\theta)$
8. $\quad$**If** $\texttt{CONVERGED}(θ, j)$ **then** **break**
9. **End for**
10. **Return** $\pi_{\theta^*}$

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

## Model-Based Value Expansion (MBVE)
Now I introduce my preferred method based on the Model-Based Value Expansion algorithm by @Feinberg2019. I argue that the MBVE method directly overcomes the main weakness of previous methods and is a robust, generalized method:
- MBVE does not require differentiability of rewards $r$ and dynamics $r$
- MBVE does not require the MC integration trick (AiO estimator)
- MBVE is computationally efficient and avoids the vanishing gradient and backpropagation through time problem faced by the LRM method
Furthermore, MBVE is a well-established method in RL, meaning that we have a full toolkit of diagnostic metrics and can easily extend it to more generalized problems. For example, it can handle models where the dynamics $f$ are not known exactly but noisily approximated ("learned") with $\hat{f}$. Section X provides a formal treatment and comparison of MBVE to the methods from @maliar12 and @fern26.

The MVE method (Feinberg et al., 2018) separates policy optimization from value estimation through an actor-critic architecture. Unlike the BR method, which jointly optimizes $(\theta, \phi)$ on a single combined loss, MVE updates the actor and critic with distinct objectives: the actor maximizes the critic's evaluation of its chosen action, while the critic minimizes Bellman errors on a mixture of real and model-generated states. This separation resolves the co-adaptation problems identified in the BR method (see Section X). 

Another benefit of the actor-critic architecture is despite that the loss is biased (due to the squared expectation), the gradient is unbiased thanks to the use of a slowly updated target network, and it thus completely avoids the Monte Carlo integration (AiO estimator) trick for discrete-time problem by @maliar21 or the expensive finite-difference integration for continuous time problem by @fern26 and @duarte24.

**Neural Networks.** The method maintains four networks: a policy (actor) $\pi_\theta$, an action-value (critic) $Q_\phi$, and their slowly-updated target copies $\pi_{\theta^-}$ and $Q_{\phi^-}$. The critic takes a state-action pair as input, $Q_\phi(s, a)$, rather than a state alone as in the BR method's $V_\phi(s)$. This is required by the deterministic policy gradient theorem (Silver et al., 2014), which computes the actor gradient via $\nabla_a Q_\phi(s, a)|_{a = \pi_\theta(s)}$ and does not require differentiation throught the reward $r$ and dynamics $f$.

**Actor objective.** The actor maximizes the critic's evaluation at its chosen action:
$$J_{\text{actor}}(\theta) = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} Q_\phi(s_i, \pi_\theta(s_i))$$
The negative sign converts maximization to minimization for SGD. The gradient flows through $Q_\phi \to \pi_\theta \to \theta$ only; no dynamics or rollouts are involved.

**MVE target construction.** Starting from a sampled state $s_i$, the algorithm rolls forward $T$ steps using the target policy $\pi_{\theta^-}$ and the known dynamics $f$:
$$s_{i,0} = s_i, \quad a_{i,t} = \pi_{\theta^-}(a_{i,t}), \quad s_{i,t+1} = f(s_{i,t}, a_{i,t}, \epsilon_{i,t+1})$$
for $t = 0, \ldots, T-1$. The MVE target at depth $t$ uses $(T - t)$ steps of model-based rewards plus a terminal bootstrap with the target critic:
$$\hat{Q}_{i,t} = \sum_{j=0}^{T-t-1} \gamma^j r(s_{i,t+j}, a_{i,t+j}) + \gamma^{T-t} Q_{\phi^-}(s_{i,T}, \pi_{\theta^-}(s_{i,T}))$$
The target critic $Q_{\phi^-}$ appears only at the terminal state $s_{i,H}$, reducing bootstrap bias relative to a standard one-step TD target.

**Critic loss (TD-$k$ mixture).** The critic is trained on states at all rollout depths with equal weight:
$$J_{\text{critic}}(\phi) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{1}{T} \sum_{t=0}^{T-1} \left(Q_\phi(s_{i,t}, a_{i,t}) - \hat{Q}_{i,t}\right)^2$$
This TD-$k$ trick ensures the critic is accurate not only at the sampled states but also at states the policy would visit during rollouts, resolving a distribution mismatch problem (discussed in Section X). 

**Relationship to existing methods.** Like the LRM method, the MVE target construction leverages the known dynamic and rewards to rollout $T$ periods accurately and use a bootstrapped $Q_{\phi^-}(s_{i,T}, \pi_{\theta^-}(s_{i,T}))$ to estimate the terminal value. Unlike LRM, the MVE approach use a Q-function approximation to avoid differentiation through $r$ and $f$ and also avoids the vanishing gradient problem of backpropagation through time. Generally, a moderate $T$ (e.g., 10-25) would be accurate enough for MVE, while the LRM method faced the tradeoff between tractability (too large $T$) and accuracy (small $T$ lead to finite-truncation bias). Lastly, setting $T = 1$ recovers standard DDPG with known dynamics in the RL literature.

In contrast with the original MVE-DDPG, the economic models provide exact known dynamics $f$, so the dynamics-learning phase of the original MVE algorithm is unnecessary. The only approximation error in the MVE target comes from the terminal bootstrap $Q_{\phi^-}(s_{i,T}, \pi_{\theta^-}(s_{i,T}))$, whose contribution is discounted by $\gamma^{T-t}$. Furthermore, MVE method does not require the MC integration (AiO estimator) because the gradient is unbiased by the actor-critic design, see section X for detailed discussion.

**Symmetric log transformation of critic prediction.** When the value function spans a wide range across states, the MSE critic loss can be dominated by high-value states, which can prevent convergence and destabilize training. As a refinement, I transform the critic predict in symmetric logarithm space: the raw output is $\hat{y}_\phi(s,a) \approx \operatorname{symlog}(Q(s,a))$, the critic loss is computed as $(\hat{y}_\phi - \operatorname{symlog}(\hat{Q}_{i,t}))^2$, and level-space values are recovered via $\operatorname{symexp}(\hat{y})$ wherever $Q$ values are needed (the terminal bootstrap in step 6 and the actor objective in step 4). This follows DreamerV3, which predicts returns in $\operatorname{symlog}$ space to achieve scale-invariant value learning. See next section for a detailed discussion of the $\operatorname{symlog}$ function.

### Algorithm 7: Model-Based Value Expansion

**Input:** Policy network $\pi_\theta$, critic network $Q_\phi$, dynamics $f$, reward $r$, discount $\gamma$, rollout horizon $T$, learning rate $\eta$, Polyak rate $\nu$, convergence rule $\texttt{CONVERGED}(\theta, \phi, j)$ 
**Output:** Trained policy $\pi^*_\theta$

1. Initialize actor $\theta$, critic $\phi$, targets $\theta^- \leftarrow \theta$, $\phi^- \leftarrow \phi$
2. **For** $j = 0, 1, 2, \ldots$ **do**
3. $\quad$  Sample mini-batch $\mathcal{B}$ of states ${s_i}$ with shock sequences ${\epsilon_{i,t}}_{t=1}^{H}$
4. $\quad$  **Actor update:** $\theta \leftarrow \theta + \eta \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_a Q_\phi(s_i, a)|_{a = \pi_\theta(s_i)} \cdot \nabla_\theta \pi_\theta(s_i)$
5. $\quad$  **Imagination rollout:** For each $i \in \mathcal{B}$, set $s_{i,0} = s_i$ and for $t = 0, \ldots, H-1$: Compute $a_{i,t} = \pi_{\theta^-}(s_{i,t})$, $\quad s_{i,t+1} = f(s_{i,t}, a_{i,t}, \epsilon_{i,t+1})$
6. $\quad$  **Build MVE targets:** $\hat{Q}_{i,t} = \sum_{j=0}^{H-t-1} \gamma^j r(s_{i,t+j}, a_{i,t+j}) + \gamma^{H-t} Q_{\phi^-}(s_{i,H}, \pi_{\theta^-}(s_{i,H}))$
7. $\quad$  **Critic update:** $\phi \leftarrow \phi - \eta \nabla_\phi J_{\text{critic}}(\phi)$
8. $\quad$  **Target updates:** $\theta^- \leftarrow \nu \theta^- + (1 - \nu) \theta$; $\quad \phi^- \leftarrow \nu \phi^- + (1 - \nu) \phi$
9. $\quad$**If** $\texttt{CONVERGED}(\theta, \phi, j)$ **then break**
10. **End for**
11. **Return** $\pi^*_\theta$

# Implementation Details
This section discusses main issues that need to be handled carefully in implementation.
## Normalization
Normalization is essential for stable and efficient neural network training. Without it, variables on different scales produce gradients of vastly different magnitudes, causing some parameters to update too aggressively while others stagnate. In deep RL-type methods, this problem is compounded by non-stationarity: the data distribution shifts as the policy improves, so any normalization that depends on batch statistics must track a moving target.

Modern RL algorithms address this at two levels. First, observations are normalized before entering the network, compressing heterogeneous state variables to comparable scales. Second, hidden-layer normalization (typically LayerNorm or BatchNorm) stabilizes the internal representations between layers, preventing pre-activations from drifting into regions where gradients vanish or explode. For example, the state-of-the-art model-based RL algorithms such as DreamerV3 (Hafner et al., 2023) and TD-MPC2 (Hansen et al., 2024) combine both levels to achieve domain-agnostic performance across environments with very different reward and observation scales.

My goal here is to build a simple and effective normalization strategy that works across a variety of economic and financial models _without requiring domain-specific tuning_. In these applications, state variables can span orders of magnitude in scale and the appropriate scale depends on the model. For example, capital stocks in the hundreds, log-productivity shocks near zero, interest rate and saving rate as fractions, wealth in the millions. My normalization strategy aim to work well across these heterogeneous state and action spaces.
### Observation Normalization
I normalize all observations using the symmetric logarithmic transform (Symlog):
$$\operatorname{symlog}(x) = \operatorname{sign}(x) \cdot \ln(1 + |x|)$$
with inverse
$$\operatorname{symexp}(y) = \operatorname{sign}(y) \cdot \bigl(e^{|y|} - 1\bigr).$$
The transform is applied element-wise to each state variable before the network input layer. It is stateless (no parameters, no running statistics), invertible, and defined for all $x \in \mathbb{R}$.

Symlog has two key properties. For small inputs ($|x| \ll 1$), the Taylor expansion $\ln(1 + |x|) \approx |x|$ gives $\operatorname{symlog}(x) \approx x$, so variables that are already near unit scale pass through unchanged. For large inputs ($|x| \gg 1$), $\operatorname{symlog}(x) \approx \operatorname{sign}(x) \cdot \ln|x|$, compressing large magnitudes logarithmically while preserving sign.

I prefer symlog over three common alternatives:
- **Standard logarithm** $\ln(1 + x)$: identical to symlog for $x \geq 0$, but undefined for negative values. Variables such as log-productivity, returns, or value functions that take negative values require special-casing, breaking the goal of a uniform transform. 
- **Running z-score** $(x - \hat{\mu}) / \hat{\sigma}$: normalizes to zero mean and unit variance using running statistics. This is the default observation normalization in many standard RL implementations, where it works well because these algorithms are typically tuned per environment so the running statistics stabilize as the policy converges within a single task. My setting is different because it requires a single trainer to handle economic models with very different state-variable scales (capital in the hundreds and productivity near one) without per-model tuning. Running z-score would need different warm-up periods, momentum parameters, and clipping thresholds for each model. A fixed transform avoids this configuration burden entirely. As a secondary benefit, the statelessness of symlog ensures that the same input always maps to the same output regardless of training stage, which aids reproducibility and eliminates a potential source of instability.
- **Hyperbolic tangent** $\tanh(x)$: achieves near-perfect cross-variable equalization but saturates for $|x| > 3$, mapping all large values to $\pm 1$. For a variable like capital with range $[0, 500]$, the network cannot distinguish $k = 100$ from $k = 500$ after tanh compression.

This design follows DreamerV3, which applies symlog uniformly to observations, rewards, and value targets to achieve domain-agnostic performance across benchmark environments with a single set of hyperparameters.

### Hidden-Layer Normalization: LayerNorm
After the input transform, residual scale differences across variables may persist (symlog reduces but does not eliminate cross-variable scale ratios). I apply Layer Normalization (Ba et al., 2016) after each hidden-layer affine map to equalize the internal representation.

Given a pre-activation vector $\mathbf{h} = (h_1, \ldots, h_M) \in \mathbb{R}^M$ for a single sample, LayerNorm computes:

$$\mu = \frac{1}{M} \sum_{j=1}^{M} h_j, \qquad \sigma^2 = \frac{1}{M} \sum_{j=1}^{M} (h_j - \mu)^2$$

$$\hat{h}_j = \frac{h_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad j = 1, \ldots, M$$

$$\tilde{h}_j = g_j  \hat{h}_j + d_j$$
where $\epsilon > 0$ is a small constant for numerical stability, and $g, d \in \mathbb{R}^M$ are learnable scale and shift parameters initialized to $g = \mathbf{1}$, $d = \mathbf{0}$. The normalization is computed across neurons within a single sample, not across samples in the batch. Unlike BatchNorm, LayerNorm is well-defined for any batch size (including one, which arises during policy evaluation) and behaves identically at training and inference time, eliminating two common sources of implementation error and instability in RL applications. Because LayerNorm subtracts the mean across neurons, any bias in the preceding Dense layer is redundant; the shift parameter $d$ serves as the effective bias.


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

## Architecture Summary
Let $\mathbf{x} \in \mathbb{R}^N$ denote the raw state vector and $\phi(\cdot)$ the hidden-layer activation function. The network consists of $K$ hidden layers, each with $M_\ell$ units ($\ell = 1, \ldots, K$). The full forward pass is:
$$\tilde{\mathbf{x}} = \operatorname{symlog}(\mathbf{x})$$
$$\mathbf{h}^\ell = W^\ell \mathbf{a}^{\ell-1}, \qquad \hat{\mathbf{h}}^\ell = \operatorname{LN}(\mathbf{h}^\ell), \qquad \mathbf{a}^\ell = \phi(\hat{\mathbf{h}}^\ell), \qquad \ell = 1, \ldots, K$$
$$\hat{y} = w^{\mathrm{out} \top} \mathbf{a}^K + b^{\mathrm{out}}$$
$$ a = \text{clip}(\hat{y}, a_{\min}, a_{\max} ) $$
where $\mathbf{a}^0 \equiv \tilde{\mathbf{x}}$, each $W^\ell \in \mathbb{R}^{M_\ell \times M_{\ell-1}}$ has no bias (absorbed by LayerNorm), and $\operatorname{LN}(\cdot)$ denotes LayerNorm as defined above. The output head directly return the linear identity $\hat{y}$ (no activation). Then I apply a variable/task-specific clip to enforce the action space boundary (exogenously set by the researcher). Note that $⁡a_{\min}$ and $⁡a_{\max}$  can be $-\infty$ or $+\infty$ for unbounded directions. For example: capital $k' \in [k_{\min}, +\infty)$ uses one-sided clipping $a = \max(\hat{y}, k_{\min})$.

Symlog acts once at the input boundary; LayerNorm acts at every hidden layer. Together they ensure that all internal representations remain well-conditioned regardless of the scale or sign conventions of the underlying economic model, without requiring any environment-specific configuration.

The tensors flow through $K$ hidden layers as:
$$\underbrace{\mathbf{x} \xrightarrow{\operatorname{symlog}} \tilde{\mathbf{x}}}_{\text{Input Norm}}  
\underbrace{ \xrightarrow{W^1} 
\mathbf{h}^{1} \xrightarrow{ \mathrm{LN}} \hat{\mathbf{h}}^1 \xrightarrow{\phi} \mathbf{a}^1}_{\text{Hidden Layer 1}} 
\underbrace{
\xrightarrow{W^2} \mathbf{h}^2 \xrightarrow{\mathrm{LN}} \hat{\mathbf{h}}^2 \xrightarrow{\phi} \mathbf{a}^2}_{\text{Hidden Layer 2}} 
\cdots 
\underbrace{\mathbf{a}^{J} \xrightarrow{w^{\mathrm{out}}} \hat{y} \xrightarrow{\text{clip}} a}_{\text{Output}}$$


# Data Generation

This section describes how synthetic training data is generated with emphasis on reproducibility and numerical stability.

### State Space Bounds

The state space bounds can be pre-computed in two ways:
1. **Model-based (recommended)**: Specify bounds "around" a benchmark steady-state capital $k^*$
2. **Direct specification**: Provide bounds directly in arbitrary units

**Productivity Shock Bounds**

Given AR(1) parameters $(\mu, \sigma, \rho)$, the ergodic distribution of $\ln z$ is truncated at $m$ standard deviations:
$$\ln z \in \left[\mu - m \cdot \sigma_{\ln z}, \; \mu + m \cdot \sigma_{\ln z}\right], \quad \sigma_{\ln z} = \frac{\sigma}{\sqrt{1-\rho^2}}$$

The default $m = 3$ covers the mass of the ergodic distribution.

**Capital Stock Bounds**

The frictionless steady-state capital (where marginal product equals user cost) is:
$$k^*(z) = \left(\frac{z \cdot \gamma}{r + \delta}\right)^{\frac{1}{1-\gamma}}$$

evaluated at the stationary mean productivity $z = e^\mu$.

Users specify bounds as multipliers on $k^*$:
$$k_{\min} = k_{\min}^{\text{mult}} \times k^*, \quad k_{\max} = k_{\max}^{\text{mult}} \times k^*$$

For example, multipliers $(0.2, 3.0)$ yield a state space from 20% to 300% of steady-state capital.

**Debt Bounds**

The maximum borrowing (debt) is capped by the collateral constraint defined as:
$$ b' \leq (1-\tau) \pi(k', z_{\min}) + \tau \delta k' + s_{\text{liquid}} \cdot k' $$

where $s_{\text{liquid}}$ is the liquidation fraction between 0 and 1. The RHS is the maximum liquidation value of the firm in the worst state $z_{\min}$. The lower bound is simply $b_{\min} = 0$.

It is important to note that in section 3.6 of @strebulaev_dynamic_2012, the authors remove the collateral constraint and allow for $b'<0$ to denote cash saving. Although this is helpful for theoretical analysis, it introduces serious numerical instability in training because
- firm can easily leverage a huge $b'$ that dominate cash flow and capital investment
- the interest rate switch between a constant $r$ and an endogenously-determined state variable $\tilde{r}$ around default boundary $b'=0$

Therefore, I restored the collateral constraint and restrict training to the borrowing case $b'\geq 0$. It is straightforward to extend the model by introducing a new state variable for cash saving with risk-free rate $r$ (or any other pricing schedule).

**Validation Constraints**

When using model-based bounds, the framework validates:

- $m \in (2, 5)$: Sufficient coverage without extreme outliers
- $k_{\min}^{\text{mult}} \in (0, 0.5)$: Allows starting below steady state
- $k_{\max}^{\text{mult}} \in (1.5, 5)$: Allows starting above steady state

These constraints prevent numerical overflow while permitting economically meaningful variation.

## Dataset Structure

### Training Set

The training set is a stream of batches $\{\mathcal{B}^j\}_{j=1}^{J}$ where each batch contains $n$ i.i.d. samples:
$$\mathcal{B}^j = \left\{ \left(k_{0,i}, b_{0,i}, z_{0,i}, \{\varepsilon_{t,i}^{(1)}, \varepsilon_{t,i}^{(2)}\}_{t=1}^{T} \right) \right\}_{i=1}^{n}$$

Each sample $i$ represents an independent firm with:

- Initial states $(k_0, b_0, z_0)$ drawn uniformly from the state space
- Two independent shock sequences $\varepsilon^{(1)}, \varepsilon^{(2)}$ for the cross-product estimator

The initial debt $b_0$ is used only in the risky debt model.

### Validation and Test Sets

- **Validation set**: Fixed dataset of size $N_{\text{val}} = 10n$, used for model selection and early stopping
- **Test set**: Fixed dataset of size $N_{\text{test}} = 50n$, used only for final evaluation

Both are generated from the same distribution but with different RNG seeds.


## Main Path vs. Fork Path

For each sample, two shock realizations are generated to enable unbiased estimation of squared expectations.

**Main Path (AR(1) Chain)**

The main shock sequence $\{z_t^{(1)}\}$ forms a continuous AR(1) chain:
$$\ln z_{t+1}^{(1)} = (1-\rho)\mu + \rho \ln z_t^{(1)} + \sigma \varepsilon_{t+1}^{(1)}$$

This path is used for lifetime reward calculations in the LR method.

**Fork Path (One-Step Branches)**

The fork sequence $\{z_t^{(2)}\}$ branches from the main path at each period:
$$\ln z_{t+1}^{(2)} = (1-\rho)\mu + \rho \ln z_t^{(1)} + \sigma \varepsilon_{t+1}^{(2)}$$

Each fork $z_{t+1}^{(2)}$ is a one-step transition from the main path $z_t^{(1)}$, not a parallel chain.

```
Main (AR1 Chain):   z0 -> z1 -> z2 -> z3 -> ... -> zT
                      \     \     \     \
Fork (1-step AR1):    z1F   z2F   z3F   zTF
```

**Usage by Method**

- LR: Uses main path only (continuous chain for lifetime rewards)
- ER/BR: Uses cross-product of main and fork at each transition for variance reduction


## Trajectory vs. Flattened Data

The data generator produces two formats optimized for different training methods.

### Trajectory Format (for LR)

The LR method requires full trajectories to compute cumulative discounted rewards:
$$\text{Shape: } (N, T+1) \text{ for shock paths}$$

Each sample preserves the temporal sequence $(z_0, z_1, \ldots, z_T)$.

### Flattened Format (for ER and BR)

The ER and BR methods operate on single-step transitions and require i.i.d. samples for valid stochastic gradient descent. A key design choice is that states $(k, b)$ are sampled **independently** for each transition rather than extracted from simulated trajectories.

**Rationale**: At data generation time, no policy exists to generate the capital/debt sequence $(k_1, k_2, \ldots, k_T)$. Using an arbitrary behavioral policy would introduce bias. Instead, the framework samples $(k, b)$ directly from the state space, treating each draw as from the ergodic distribution. This approach:

1. **Ensures i.i.d. samples**: Each transition is statistically independent, satisfying the assumptions of SGD
2. **Eliminates serial correlation**: Consecutive time steps in a trajectory are correlated; independent sampling removes this dependency
3. **Approximates ergodic coverage**: Uniform sampling over the bounded state space approximates draws from the ergodic distribution that would arise under an optimal policy

The flattened dataset has shape $(N \times T,)$ with each observation:
$$\text{Obs} = (k, b, z, z'_1, z'_2)$$

where $(z'_1, z'_2)$ are the main and fork next-period shocks. After flattening, the dataset is randomly shuffled to further eliminate any residual structure.

![Comparison of trajectory and flattened data formats. Left: trajectory format preserves temporal structure for LR method. Right: flattened format with independent state sampling for ER/BR methods. *Example output from debug-mode training with $T=64$.*](../results/latest/figures/data_format_comparison.png){#fig-data-format width=90%}


## RNG Seed Schedule

Reproducibility requires deterministic random number generation. The framework uses TensorFlow's stateless random functions (`tf.random.stateless_*`) which produce identical outputs given identical seed pairs, regardless of call order.

**Master Seed**

A master seed pair $\mathbf{s}^{\text{master}} = (m_0, m_1)$ anchors all randomness.

**Split Seeds**

Disjoint seeds for each dataset:
$$\mathbf{s}^{\text{train}} = (m_0 + 100, m_1), \quad \mathbf{s}^{\text{val}} = (m_0 + 200, m_1), \quad \mathbf{s}^{\text{test}} = (m_0 + 300, m_1)$$

**Variable IDs**

Each random variable has a fixed ID:

| Variable            | ID  |
| ------------------- | --- |
| $k_0$               | 1   |
| $z_0$               | 2   |
| $b_0$               | 3   |
| $\varepsilon^{(1)}$ | 4   |
| $\varepsilon^{(2)}$ | 5   |

**Training Seeds by Step**

For training step $j$ and variable $x$:
$$\mathbf{s}^{\text{train}}_{j,x} = (m_0 + 100 + \text{VarID}(x), \; m_1 + j)$$

This ensures each batch $j$ receives unique, reproducible random draws.

**Validation/Test Seeds**

These are single fixed datasets (no step index):
$$\mathbf{s}^{\text{val}}_{x} = (m_0 + 200 + \text{VarID}(x), \; m_1)$$
$$\mathbf{s}^{\text{test}}_{x} = (m_0 + 300 + \text{VarID}(x), \; m_1)$$

This schedule guarantees:

1. **Reproducibility**: Identical seeds produce identical data across runs
2. **Common random numbers**: Different methods train on the same data, enabling fair comparison
3. **No leakage**: Train/validation/test sets use disjoint RNG streams


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
