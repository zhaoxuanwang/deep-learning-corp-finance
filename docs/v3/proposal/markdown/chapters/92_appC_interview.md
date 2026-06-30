# Interview Questions and Answers {#sec:appC-interview}

## Interview questions {#sec:interview-questions}

*Here are some areas that we would like to discuss with you in the coming interview:*
Coding:
- Critical issues, proper and correct usage of TF and TFP required.  How can we address this problem?  Can you review your code with codex and/or claude code to ensure the code is correct and proper?  How can you leverage coding agent to help you design and code something that is much more robust?  Are the test coverage enough?

Policy training with Maliar and Maliar type method:
- How should we generate the samples for training?  Should we generate a fixed batch or continue to get a new batch of training samples each time or something in between? What are the mathematical theorems to guarantee the correctness for each time of training regime?  If we use fixed set of data, is there bias in the final result?  Why and why not?  Can you prove your statement one way or another?  Can you run a sample experiment with a simple example to confirm the issue?  E.g. a linear quadratic control problem that has explicit solution? 
- For the Euler residual equations, we need very equations to be satisfied correctly for the AIO method.  How can we ensure that?  Is it reasonable to just use some norm of all the Euler residuals?  Are all the residuals of the same scale?  Even if they are of the same scale, is just some norm as loss function correct? How to handle this problem?
- NUTS for HMC in TFP is probably not good.  Why?  It is very slow, why?  What should we do?  For fixed number of leap frog steps HMC, we need a few things, step size, mass matrix and number of leap frogs.  How to tune it?  

Suppose you have to sell your product to another bank and make your work practical. Which departments of a bank like JP Morgan Chase can benefit from this work?  
- Suppose we would like to apply for your work for the investment banking application of advising the clients on capital structure, what is missing on the current report? What model do we have to build further?
- Can you give a draft project plan to make this a fully useful product?
- What are the likely issues when you try to sell your product?
- What is the price that you think your client is willing to pay? 
- What are the obstacles do you think you are going to have when deploying your product in real life?

## Key updates {#sec:key-updates}

I made a significant update of the project since the last version (May 29). The current report (June 20) now also builds the deep learning methods and entire workflow, proposed in a new working paper posted two weeks ago:

> Victor Duarte and Julia Fonseca, "AI for Structural Estimation," NBER Working Paper 35283 (2026), <https://doi.org/10.3386/w35283>.

The key idea of this paper (DF26) is to train a neural network (NN) surrogate over the high-dimensional state-parameter space only *once*. Then use the pre-trained surrogate to substantially accelerate the estimation (via SMM or Bayesian), which otherwise would requires thousands of model solves for each parameter $\beta$ candidate evaluation. This avoids the main compute bottleneck.

This idea of "train NN surrogate once, re-used in estimation loop" is the direction I pursued in my previous report. DF26 is a more polished version that addressed three main technical obstacles that I did not fully solve:
- **Model solve**: NN surrogate policy (actor-critic method) + VFI refinement on-grid
- **Estimation**: more efficient, gradient-based optimization using moment surrogate NN
- **Asynchronous computing**: model solve (policy NN), simulation (moment NN), and estimation (optimizer) run concurrently on separate GPUs

DF26 only considers SMM and trained separate surrogate NNs to approximate moment functions $g(\beta)$. This surrogate idea might be extended to Bayesian MCMC + filtering methods. My previous report attempted wiring a policy surrogate $\pi(s,\beta)$ directly to the TFP native HMC methods, but it is too slow to be useful.

Instead, the promising extensions to Bayesian inference is:
- Train neural networks to approximate the likelihood $g(\beta)\approx \log P(y \mid \beta)$
- The neural likelihood (NL) can replace the filter inside MCMC loop (particularly when the model is not simply linear-Gaussian)
- Use NL gradient $\nabla g(\beta)$ for efficient MCMC sampler like HMC
I have located the literature in statistics/ML and are directly relevant:
- Papamakarios, Sterratt, Murray (2019), "[Sequential Neural Likelihood](https://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)"
- Cranmer, Brehmer, Louppe (2020), "[The frontier of simulation-based inference](https://www.pnas.org/doi/10.1073/pnas.1912789117)"

The key improvements of current report against previous version (May 29):

+-----------------+---------------------+-----------------+--------------+
| ::: minipage    | ::: minipage        | ::: minipage    | ::: minipage |
| Key Parts       | DF26 & Current      | Previous report | Maliar21     |
| :::             | report              | :::             | :::          |
|                 | :::                 |                 |              |
+:================+:====================+:================+:=============+
| Risky debt      | Yes and fast        | Yes but slow    | No           |
| model           |                     |                 |              |
+-----------------+---------------------+-----------------+--------------+
| Policy inputs   | State x Parameter   | State x         | State        |
|                 |                     | Parameter       |              |
+-----------------+---------------------+-----------------+--------------+
| Model solve     | Actor-critic        | Actor-critic    | Euler res    |
|                 | (one-step)          | (32-step)       |              |
+-----------------+---------------------+-----------------+--------------+
| Discontinuity   | Smooth approx. +    | Smooth approx.  | No           |
|                 | VFI refinement      | with error      |              |
+-----------------+---------------------+-----------------+--------------+
| Compute         | Gaussian Quadrature | GQ and AiO      | AiO          |
| $E_{z'\mid z}$  | (GQ)                | product         |              |
+-----------------+---------------------+-----------------+--------------+
| Default         | Numerical approx.   | Exact from      | No           |
| probability     | from value NN       | nested VFI      |              |
+-----------------+---------------------+-----------------+--------------+
| Nested fixed    | target value NN,    | nested VFI      | No           |
| point           | then VFI refine     |                 |              |
+-----------------+---------------------+-----------------+--------------+
| Moments         | Moment surrogate    | Analytical      | n.a.         |
|                 | maps $\beta$ to     | (mean, SD)      |              |
|                 | $g(\beta)$          |                 |              |
+-----------------+---------------------+-----------------+--------------+
| Optimizer for   | Levenberg-Marquardt | Simulated       | n.a.         |
| finding         | with analytical     | annealing.      |              |
| $\beta^*$       | Jacobian. Efficient | Slower.         |              |
|                 | and faster.         |                 |              |
+-----------------+---------------------+-----------------+--------------+
| Architecture    | Multiple GPUs and   | Sequential,     | n.a.         |
|                 | async. computing    | single CPU/GPU  |              |
+-----------------+---------------------+-----------------+--------------+
| Validation of   | Benchmark against   | VFI itself      | n.a.         |
| model solve     | VFI                 |                 |              |
+-----------------+---------------------+-----------------+--------------+
| Validation of   | Monte Carlo         | Same MC         | n.a.         |
| SMM estimation  | recovery of true    | recovery        |              |
|                 | params              |                 |              |
+-----------------+---------------------+-----------------+--------------+
| Bayesian MCMC   | n.a.                | Policy          | n.a.         |
|                 |                     | surrogate       |              |
+-----------------+---------------------+-----------------+--------------+

## Q1: TF and TFP implementation {#sec:q1-tf-and-tfp-implementation}

*This question is addressed in the main text. See Chapter 2 (Development Methodology and Controls).*

## Q2: Deep Learning Methods {#sec:q2-deep-learning-methods}

#### Data Generation {#sec:data-generation}

> *How should we generate the samples for training?  Should we generate a fixed batch or continue to get a new batch of training samples each time or something in between?*

For **production**, I prefer to adopt DF26's asynchronous training design:
1. **Block 1: Model solve.** Draw new mini-batch each time, used for exactly one gradient step, and discard. No stored training set. For each gradient step $j$:
- Given the current policy $\pi_{\theta_j}$ and value $V_{\phi_j}$
- Draw fresh iid random uniform $(k,b,z, \beta)$ for $N$ obs
- One-step policy rollout $(k',b')=\pi_{\theta_j}(k,b,z,\beta)$
- Actor-critic method to update $\theta_{j+1}$ and $\phi_{j+1}$
- Discard sample $(k',b',k,b,z,\beta)$, go to next step $j+1$
2. **Block 2: Refinement and simulation.** Simulator draws a given parameter vector, evaluates the latest networks at fixed param, refines the solution on the discrete grid, simulates firm panels, and computes moments. Critically, the accumulated data include the pairs of parameter-moments $(\beta_s, m(\beta_s))^S_{s=1}$ over different candidate $\beta_s$
- For a given param value $\beta_s$, plug it into the current $\pi_{\theta_j}$ and $V_{\phi_j}$ from block 1
- Refine the policy and value neural networks on grid (warm-start VFI), obtain the grid-based $\pi^*(\cdot \mid \beta_s)$ and $V^*(\cdot \mid \beta_s)$
- Simulate $N_\beta$ firms over $T=300$ periods. Initial states $(k_0, b_0, z_0)$ are iid uniform draw, then rollout forward with $(k',b')=\pi^*(k,b,z)$ and $z'=\mu \log z + \sigma \epsilon$. Discard the first 200 periods as burn-in, the rest 100 periods reflect ergodic distribution.
- Use simulated data to construct moment $m(\beta_s)$, adds to the current sample of $(\beta_s, m(\beta_s))$, set $s=s+1$, repeat the first step with the next candidate $\beta_{s+1}$

Both model solve (GPU#1) and estimation (GPU#2-4) run concurrently. When GPUs 2--4 each begin evaluating a new batch of parameter vectors, they each read the most recent value and policy network weights from GPU 1.

In summary, there are two separate data generation tasks:
- Solving model generates fresh new sample per gradient update step, then discard it. The draw is uniform (not ergodic) to ensure coverage over the entire state x parameter space.
- Data simulation generates an ergodic trajectory data for each parameter candidate, construct moments, then store the parameter-moment pair into an accumulated dataset. This dataset is growing over training and will be used for estimation.
For both tasks, full reproducibility is ensured using proper RNG and seeding schedule. Except for the final (parameter, moment) data, other samples are discarded after usage.

For **debugging** in early code development, my previous report implements an in-between approach for model solve:
- Simulate forward $T$ periods of exogenous states $z$
- Draw initial endogenous states $(k_0, b_0)$
- Actions are rolled out on-policy over $T$ periods: $(k',b')=\pi(k,b,z)$
So that the part of the training sample is fixed $(k_0, b_0, z)$ and part of it is generated during training and depends on the policy $\pi_\theta$ for each gradient step $\theta$.

The main motivation of using fixed data is it provide cleaner comparison across methods. It also makes it easier to debug by isolating randomness in simulation from the algorithm themselves. But it is only valid as a baseline check, not for the full production pipeline (see discussion below).

> *What are the mathematical theorems to guarantee the correctness for each time of training regime?* 

- **Solving the model** has math guarantees everywhere except the neural network approximation. The NN-based methods in Maliar21, DF24, and DF26 do not have guarantee, but DF26 is more reliable because it uses VFI to refine the NN policy and value approximation before passing it to estimation.

- **Estimation** has the classical asymptotic guarantees, except the DF26's moment surrogate, which again rests on diagnostics rather than a theorem.

In short, the core parts that lack math theorem to guarantee convergence is the NN approximation to policy/value function and to the moment conditions (DF26). This means we need to design "correctness" tests to ensure that the solution is valid.

**Model solving:**

+----------------+-----------------+----------------------+----------------+
| ::: minipage   | ::: minipage    | ::: minipage         | ::: minipage   |
| Method         | Theorem         | Key intuition        | Guarantee      |
| :::            | :::             | :::                  | :::            |
+:===============+:================+:=====================+:===============+
| VFI / PFI      | Banach          | The Bellman operator | Yes            |
| (grid)         | fixed-point     | is a                 |                |
|                | (Blackwell's    | $\beta$-contraction, |                |
|                | conditions)     | so iterating it      |                |
|                |                 | converges to the one |                |
|                |                 | true value function. |                |
+----------------+-----------------+----------------------+----------------+
| Compute        | Gaussian        | n nodes integrate    | Yes (smooth)   |
| expectation    | (Gauss-Hermite) | polynomials up to    |                |
| over           | quadrature      | degree 2n-1 exactly, |                |
| $z' \mid z$    | exactness       | giving precise       |                |
|                |                 | deterministic        |                |
|                |                 | expectations for     |                |
|                |                 | smooth low-dim       |                |
|                |                 | Gaussian shocks.     |                |
+----------------+-----------------+----------------------+----------------+
| Compute        | Law of large    | Sample averages      | Consistent     |
| expectation    | numbers         | converge to the      |                |
| over           |                 | expectation; two     |                |
| $z' \mid z$    |                 | independent draws    |                |
| with Maliar's  |                 | keep the             |                |
| AiO estimator  |                 | squared-residual     |                |
|                |                 | gradient unbiased.   |                |
+----------------+-----------------+----------------------+----------------+
| NN model       | Universal       | The network can      | No             |
| solvers        | Approximation + | represent the        |                |
|                | Robbins-Monro   | solution and SGD     |                |
|                | (SGD)           | reaches a stationary |                |
|                |                 | point, but neither   |                |
|                |                 | guarantee            |                |
|                |                 | convergence to the   |                |
|                |                 | true optimum         |                |
+----------------+-----------------+----------------------+----------------+
| DF26 grid      | Banach          | A few exact          | Yes (up to     |
| refinement     | contraction,    | policy-iteration     | grid)          |
|                | locally per     | steps pull the       |                |
|                | $\beta$         | network policy to    |                |
|                |                 | the grid optimum,    |                |
|                |                 | restoring the        |                |
|                |                 | contraction          |                |
|                |                 | guarantee.           |                |
+----------------+-----------------+----------------------+----------------+
| LP method      | LP formulation  | The value function   | Yes            |
| (Nikolov21)    | of dynamic      | is the unique LP     |                |
|                | programming     | solution (smallest V |                |
|                |                 | with V $\geq$ TV),   |                |
|                |                 | recovered exactly on |                |
|                |                 | the grid.            |                |
+----------------+-----------------+----------------------+----------------+

**Estimation:**

+------------------+----------------------------+---------------------+--------------+
| ::: minipage     | ::: minipage               | ::: minipage        | ::: minipage |
| Method           | Theorem                    | Key intuition       | Guarantee    |
| :::              | :::                        | :::                 | :::          |
+:=================+:===========================+:====================+:=============+
| Simulated        | Ergodic theorem (LLN for   | Long simulated      | Asymptotic   |
| moments          | Markov chains)             | panels after        |              |
|                  |                            | burn-in make sample |              |
|                  |                            | moments converge to |              |
|                  |                            | the model's true    |              |
|                  |                            | stationary moments. |              |
+------------------+----------------------------+---------------------+--------------+
| SMM / GMM        | McFadden 1989,             | Matching simulated  | Asymptotic   |
|                  | Pakes-Pollard 1989 (Hansen | to data moments     |              |
|                  | 1982)                      | gives a consistent, |              |
|                  |                            | asymptotically      |              |
|                  |                            | normal estimator    |              |
|                  |                            | under               |              |
|                  |                            | identification.     |              |
+------------------+----------------------------+---------------------+--------------+
| Moment-surrogate | Universal Approximation +  | The                 | No           |
| net (DF26)       | diagnostics                | parameter-to-moment |              |
|                  |                            | map is              |              |
|                  |                            | representable, but  |              |
|                  |                            | accuracy is         |              |
|                  |                            | certified only by   |              |
|                  |                            | held-out R^2^ and   |              |
|                  |                            | cross-validation.   |              |
+------------------+----------------------------+---------------------+--------------+
| Bayesian MCMC    | MH ergodicity (Tierney     | The chain converges | Asymptotic   |
|                  | 1994) + Bernstein-von      | to the posterior,   |              |
|                  | Mises                      | which is            |              |
|                  |                            | asymptotically      |              |
|                  |                            | normal around the   |              |
|                  |                            | truth.              |              |
+------------------+----------------------------+---------------------+--------------+
| Filtering        | Kalman optimality;         | Kalman is the exact | Mixed        |
| (Kalman / EKF /  | particle-filter LLN        | filter for          |              |
| particle)        |                            | linear-Gaussian;    |              |
|                  |                            | EKF only            |              |
|                  |                            | approximates it;    |              |
|                  |                            | the particle filter |              |
|                  |                            | converges as        |              |
|                  |                            | particles grow.     |              |
+------------------+----------------------------+---------------------+--------------+
| Indirect         | Gourieroux-Monfort-Renault | Matching an         | Asymptotic   |
| inference        | 1993                       | auxiliary model's   |              |
|                  |                            | parameters          |              |
|                  |                            | identifies the      |              |
|                  |                            | structural ones     |              |
|                  |                            | through the binding |              |
|                  |                            | function.           |              |
+------------------+----------------------------+---------------------+--------------+

#### Finite sample bias {#sec:finite-sample-bias}

> *If we use fixed set of data, is there bias in the final result?  Why and why not?  Can you prove your statement one way or another?  Can you run a sample experiment with a simple example to confirm the issue?  E.g. a linear quadratic control problem that has explicit solution?* 

There are two kinds of bias from a fixed training dataset.

The first is **overfitting frozen randomness**. A fixed dataset locks in the particular shock draws that went into it. A flexible model like a neural net will fit those specific draws rather than the true average behavior, so it ends up learning the sample's noise. Fresh data shows new shocks at every step, which forces the model toward the true average instead of memorizing one realization.

The second is the **coverage gap**. A finite dataset only covers finitely many states. Wherever the sample has no points, nothing pins the solution down, so it is simply wrong there. If the simulated economy later visits those states, or if we need the solution across a range of parameters as in DF26, that error flows straight into the moments and the parameter estimates.

Both biases shrink only as the data covers more of the space, not as we train longer. The real fix is to adopt the DF26 approach: **keep drawing fresh states, use them once, and discard them**, which over time covers the whole space and averages out the shocks.
\#### A simple linear quadratic control example

**The problem.** An agent observes a scalar state $s$ and picks a scalar control $u$ each period. The state evolves as $s' = a\,s + b\,u + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,\sigma^2)$, and each period incurs a cost $q\,s^2 + r\,u^2$ (a penalty $q\,s^2$ for being away from the target $s=0$, and a penalty $r\,u^2$ for acting). Because $u$ today also shifts tomorrow's state, the choice is sequential: pick a policy / decision rule for $u$ to minimize the expected discounted total cost,
$$\min_{\{u_t\}}\;\mathbb{E}\sum_{t=0}^{\infty}\beta^t\big(q\,s_t^2 + r\,u_t^2\big)\quad\text{s.t.}\quad s_{t+1}=a\,s_t+b\,u_t+\varepsilon_t.$$
I used the constants $a=0.9$, $b=1.0$, $q=1.0$, $r=0.5$, $\beta=0.95$, $\sigma=0.3$.

**Closed-form solution.** The value function and the optimal policy are known exactly:
$$V^*(s)=P\,s^2+c,\qquad u^*=-K\,s.$$
Here $P$ solves the Riccati equation $P=q+\beta a^2 P-(\beta a b P)^2/(r+\beta b^2 P)$, the gain is $K=\beta a b P/(r+\beta b^2 P)$, and the noise adds $c=\beta P\sigma^2/(1-\beta)$. Numerically these give $P=1.287$, $K=0.639$, $c=2.202$, and a closed-loop state coefficient $a-bK=0.261$.

**The approximator (not a neural net).** I did not use a neural net. I approximated the value function by a flexible model that is linear in $43$ fixed basis functions,
$$\hat V_w(s)=w_0+w_1 s+w_2 s^2+\sum_{j=3}^{42} w_j\,\exp\!\Big(-\frac{(s-\mu_j)^2}{2\ell^2}\Big),$$
with $40$ centers $\mu_j$ spread over $s\in[-3,3]$ and spacing $\ell$. The model has $43$ free weights $\{w_j\}_{j=0}^{42}$, while the truth needs only two of them ($w_2=P$, $w_0=c$). I chose the weights to minimize Bellman residual
$$\text{res}(s)=\hat V_w(s)-\big[q s^2+r(Ks)^2+\beta\,\mathbb{E}_\varepsilon \hat V_w(a_{cl}s+\varepsilon)\big]$$
(with $a_{cl}=a-bK$) to zero at the sampled states. The least square solution is given by
$$\hat{w} = \arg \min_w \frac{1}{N} \sum_{i=1}^N \text{res}(s_i)^2$$
The true $V^*$ makes this residual zero everywhere.

**Fixed vs fresh sample**: the problem is solved twice on two samples with identical budget. Both runs use the same per-step batch size ($N=12$ states) and the same number of gradient steps ($T$), so the per-step sample size and the total compute are held equal. The only difference is:
- **Fixed run** draws $12$ states on $[-3,3]$ once and reuse that same batch at every step.
- **Fresh run** draws $12$ new states on $[-3,3]$ at every step and then discard them.

The figure below shows the solved value function over the two runs in the fair setup (both use batch size 12 and the same 60,000 steps). The black curve is the true value function. The blue dashed curve is the fresh run (12 new states each step), which tracks the truth closely (held-out RMSE 0.06). The red curve is the fixed run (reusing the same 12 states), which is visibly off the truth across the domain (held-out RMSE 2.89), and the red ticks at the bottom mark the 12 states it kept reusing.

**Why using neural network has analogous problems?** Both biases come from one ingredient: a model far more flexible than the true solution, over-fitting to fixed finite data points. A neural net would behaves the same way.

`\imgplaceholder{lq\_value\_function.png}`{=latex}

#### Euler Residual Methods {#sec:euler-residual-methods}

> *For the Euler residual equations, we need very equations to be satisfied correctly for the AIO method.  How can we ensure that?  Is it reasonable to just use some norm of all the Euler residuals?  Are all the residuals of the same scale?  Even if they are of the same scale, is just some norm as loss function correct? How to handle this problem?*

A quick recap of the Euler residual method. Denote $f\equiv f(k',b',z';k,b,z)$ as the closed-form formula such that under optimality the Euler equation holds:
$$E_{z'|z} \left[ f\right]=0$$
Taking $M$ random draws of next-period shock $\{z'_m\}^M_{m=1}$, Maliar21 propose using the All-in-One (AiO) cross-product to form the loss function:
$$L\equiv \frac{1}{N}\sum_i \left[ f(z'_1)\times f(z'_2) \right]$$
which is an unbiased estimator for $E_{z'|z} [f]^2$. Minimizing the loss function enforces the first-order necessary condition to hold.

**1. How do we ensure every equation holds?** For models with closed-form Euler equation, we can write $f$ by re-arranging the Euler equation (LHS-RHS), then everything is written analytically. However, for more complex models like risky debt, a closed-form Euler equation is not available. Specifically, we can still write down the investment FOC for $k'$, but cannot do so for $b'$. To see this, the marginal cost of debt on one side of the debt FOC is $-\gamma E\left[ 1\{\text{solvent}\} V_{b'} \right]$, where the default (solvent) indicator and the derivative of $V$ wrt $b'$ has not analytical formula.

**2. What norm should we use?** Maliar21's cross-product is unbiased and works fine in practice. The main concern is the AiO cross-product fluctuates around zero and can go negative. That oscillation is an artifact of the unbiased estimator, not a bug, since the gradient is still correct on average.

We should NOT naively use norms like MSE, because it would bias the gradient:
$$E_\varepsilon[f(\varepsilon)^2]=(E_\varepsilon[f(\varepsilon)])^2+\operatorname{Var}_\varepsilon(f(\varepsilon)).$$
The extra variance term means a plain squared loss does not target $E_\varepsilon[f]=0$, it also penalizes the residual's variance, so its gradient points the wrong way.

**3. Are the residuals on the same scale?** No. Euler residuals can be written in unit-free (relative) form, for example the investment Euler $1+\psi_1 I/k=\frac{1}{1+r}E[\cdots]$ where both sides are around one, so they sit at a small, comparable scale and a plain sum with unit weights works. My results confirm the Euler residuals can be pushed close to zero this way.

However, for more complex models with additional constraints, such as the inequality budget constraints introduced in Maliar21, this will cause scale mismatch. Maliar21's consumption-saving problem allows for the inequality constraint to be written in unit-free terms so that it is not a concern, but for our corporate finance models, most of such constraints are written in Bellman term (e.g., cash) and cannot be easily normalized, this will caused the scale mismatch issue as documented for the Bellman residual method. The larger residual will dominate the gradient direction and may lead to incorrect solutions. No mathematically theorem can guarantee this, and my empirical result confirms this. Maliar21's proposed fix is to manually tune the weights on losses with different scale, but in my view this is not robust for production.

**Beyond scale and bias.** Fixing both is still not enough when many auxiliary losses are stacked (Bellman residual, first-order condition, envelope condition, constraints). They can pull in different directions: only the FOC term points toward the optimal policy, while the others can be driven to zero by an arbitrary, wrong policy-value pair, so the joint loss looks minimized while the solution is wrong. My report has confirmed this and show that this is why Maliar21's Bellman residual approach failed in practice.

**Summary.** Euler residual approach with AiO estimator works well only when the optimality condition has closed-form formula, which requires the objective to be known analytically, smooth, and differentiable. It is NOT a valid method for more complex models like the risky debt model.

#### NUTS-HMC in TFP {#sec:nuts-hmc-in-tfp}

> *NUTS for HMC in TFP is probably not good. Why? It is very slow, why? What should we do? For fixed number of leap frog steps HMC, we need a few things, step size, mass matrix and number of leap frogs. How to tune it?*  

A quick recap. Our goal is to draw samples from the posterior $P(\beta \mid y) \propto \exp(\ell(\beta))$, where we already know how to compute $\ell(\beta) = \log P(y|\beta) + \log P(\beta)$ using filtering for the log-likelihood and a pre-specified prior $P(\beta)$. The HMC sampler evaluates many different $\beta$ candidates for `num_steps` times, each collecting one sample. Three hyperparameters are set: number of leapfrog steps $L$, step size $\varepsilon$, and mass matrix $M$.

For each iteration up to `num_steps`:
1. Draw a fresh momentum $p \sim N(0, M)$, same dimension as $\beta$.
2. Run $L$ leapfrog steps. Each step repeat:
$$p \leftarrow p + \tfrac{\varepsilon}{2}\nabla\ell(\beta), \qquad \beta \leftarrow \beta + \varepsilon M^{-1} p, \qquad p \leftarrow p + \tfrac{\varepsilon}{2}\nabla\ell(\beta).$$
After $L$ steps we have a proposal $(\beta^*, p^*)$.
3. Accept or reject. Define the total energy $H(\beta, p) = -\ell(\beta) + \tfrac12 p^\top M^{-1} p$. Accept $\beta^*$ with probability $\min\{1, \exp(H(\beta,p) - H(\beta^*,p^*))\}$. This Metropolis correction cancels the small error from finite $\varepsilon$, which is what makes the samples target the exact posterior.
4. Record the current position, the new one if accepted or the old one if not, as one sample. Repeat from step 1.
The compute per iteration is dominated by $L$ gradient evaluations, so it scales as $O(L)$.

**NUTS-HMC is slow in TFP for two reasons.**
**1. Mechanics**: NUTS builds each proposal by doubling a leapfrog trajectory until it detects a U-turn, up to a maximum tree depth (TFP default 10, so up to about $L=1,000$ leapfrog steps per single draw). Every leapfrog step needs one full gradient of the log-posterior. So the cost of one draw is $L$ times (the cost of one gradient). Gradient step is costly if we need to solve the model in the loop, or to auto-diff through a deep neural network (surrogate policy).

**2. TFP implementation** (according to the [doc](https://arxiv.org/pdf/2002.01184).): TFP runs all chains (minimum 4) together as a single batch and steps them through one shared loop. But NUTS builds each proposal by doubling a trajectory until it makes a U-turn, and that length varies from chain to chain.

To fit this dynamic recursion into a batch, TFP unrolls it into one loop that does the same amount of work for every chain at each step. The key is that the shared loop cannot stop until the chain with the longest trajectory finishes (up to the max tree depth). So the whole batch moves at the pace of its deepest chain, and the chains that already turned around get dragged along through extra steps. Since the cost is dominated by gradient evaluations, those extra steps are wasted gradients.

**What to do**
When gradient is expensive (surrogate policy, or any model solve in the loop) so that NUTS is intractable, we have these options:
1. `tfp.mcmc.HamiltonianMonteCarlo` with manually tuned `num_leapfrog_steps` and `step_size`.
2. `tfp.experimental.mcmc.windowed_adaptive_hmc` with manually tuned `num_leapfrog_steps`.
3. Run other gradient-free samplers like Random-Walk Metropolis.

The first two options manually fix the number of leapfrog steps of HMC. As baseline, I'd start with using `windowed_adaptive_hmc` because it adapt the step size $\epsilon$ to a target acceptance rate and estimates a diagonal mass matrix during warmup (see details in [doc](https://mc-stan.org/docs/reference-manual/mcmc.html)) The `num_leapfrog_steps` can be either set manually or adopted from a short NUTS run and adopt the typical $L$ that settled. I summarize the motivation below:

+----------------------+----------------------+-------------------------+
| ::: minipage         | ::: minipage         | ::: minipage            |
| Param                | Impact and Trade-off | How to tune             |
| :::                  | :::                  | :::                     |
+:=====================+:=====================+:========================+
| Step size            | Accuracy v.s. cost.  | Auto-tuned in Option 2  |
| $\varepsilon$        | Too large            | `windowed_adaptive_hmc` |
|                      | $\varepsilon$        |                         |
|                      | increases error, too |                         |
|                      | small $\varepsilon$  |                         |
|                      | waste compute on     |                         |
|                      | gradient eval.       |                         |
+----------------------+----------------------+-------------------------+
| Leapfrog $L$         | Chain mixing v.s.    | Either manually or run  |
|                      | wasted compute. Too  | a small NUTS warmup and |
|                      | few steps is not     | adopt the typical $L$   |
|                      | enough for chain to  |                         |
|                      | mix because $\beta$  |                         |
|                      | barely moved         |                         |
|                      | (auto-correlated),   |                         |
|                      | too large waste      |                         |
|                      | gradient eval cost.  |                         |
+----------------------+----------------------+-------------------------+
| Mass matrix $M$      | Ideal $M$ is the     | Auto-estimated in       |
|                      | posterior covariance | Option 2                |
|                      | matrix for           | `windowed_adaptive_hmc` |
|                      | $P(\beta \mid y)$.   |                         |
|                      | Affect efficiency.   |                         |
+----------------------+----------------------+-------------------------+

## Q3: Commercial Product {#sec:q3-commercial-product}

*The commercial product overview, the three product cases (CFA, DCM, leveraged finance and private credit) with their examples and pricing, and the model-risk-and-compliance discussion are presented in the main text. See Chapter 1 (Commercial Product Plan) and Appendix A.*

### What is missing and what to build further {#sec:what-is-missing-and-what-to-build-further}

> *Suppose we would like to apply your work for the investment banking application of advising the clients on capital structure, what is missing on the current report? What model do we have to build further? Can you give a draft project plan to make this a fully useful product?*

As the baseline model, I'll take the endogenous default model in DF26 (Section 2), which incorporates AR(1) productivity shocks, fixed operating cost, convex capital adjustment cost, external equity cost, and most importantly, endogenous default and market-clearing bond price. This is the same risky debt model in SW12.

I consider four key features that are first-order to make the model a useful product:

**1. Various types of debt.** Real-world firms hold and consider a wide variety of debt financing strategies, including bonds with different maturity, priority and seniority, callable options, etc. The basic model only considers a one-period corporate bond and is far from being realistic. Extensions will follow the dynamic structural models summarized below:
- **Maturity**: He and Milbradt (2016) allows firm to choose its debt maturity structure and default timing dynamically. Chen, Xu, and Yang (2021) links leverage to maturity to capture the tradeoffs between the liquidity discount on long-term debt, the repayment risk of short-term debt, and the value of short-term debt as a commitment device.
- **Priority and seniority**: Hackbarth, Hennessy, and Leland (2012) derives the jointly optimal priority and capital structure when the firm has multiple debt classes and investment. Hackbarth, Hennessy, and Leland (2007) examine the optimal mixture and priority structure of bank and market debt.
- **Callable bond:** Acharya and Carpenter (2002) derives analytically the optimal call and default rules when interest rates and firm value are stochastic. Jarrow, Li, Liu, and Wu (2010) develops an alternative reduced-form approach for valuing callable corporate bonds by empirically characterizing the call probability.
- **Renegotiation, loan-versus-bond**: Hackbarth, Hennessy, and Leland (2007) incorporates the bank-versus-market choice. Mella-Barral and Perraudin (1997) is the canonical strategic-renegotiation pricing model.

**2. Regional regulatory and institutional setting.** Model timing and setup need to be aligned with local tax code, bankruptcy and recovery rule, investment and production schedule, etc.

**3. Systematic risk and credit spread.** The baseline risky debt model assumes no risk premium. The endogenous bond price is defined by the breakeven condition, in which the bond yield only compensates for expected loss, and the lender just earns the risk-free rate in expectation. In reality, lenders demand extra yield as risk premium. The risk premium compensates for systematic risk, the part of default that is correlated across firms with bad aggregate states and is not diversifiable. This is documented in literature that the baseline model's implied debts are "too cheap" and thus over-predict the optimal leverage.
- **Approach**: I will start by adding a stochastic discount factor to the bond pricing equation, similar to Kuehn and Schmid (2014, Eq.15). Further extension should refer to Bhamra, Kuehn, and Strebulaev (2010a) and Chen (2010): both replace the risk-free discounting in the standard SW12/DF26 risky-debt model with a consumption-based stochastic discount factor carrying a countercyclical price of risk, so that debt and equity are priced consistently under systematic macroeconomic risk, which jointly raises credit spreads to realistic levels and lowers optimal leverage toward observed values.

**4. Firm-specific estimation.** The model estimation is based on a full panel of firms, using both cross-sectional and time-series variations. The estimated parameters are thus NOT firm-specific. If our goal is tailor the product to a named client (e.g., Tencent), we should ideally need a higher frequency time series data of the single company's financials.
- **Approach**: calibrate the parameters that are common or weakly identified (e.g., recovery and tax), and estimate the firm-specific technology and cost parameters from the client's own time series.

There are more model features that I view as second-order (for debt financing) but could be useful for general advisory on capital structure:

**5. Cash and liquidity management**. Capital structure advice should also involves cash savings and credit lines (loan) as a buffer. The canonical model is Bolton, Chen, Wang (2011) for dynamic investment, financing, and risk management for financially constrained firms. It highlights the central importance of the endogenous marginal value of liquidity (cash and credit line) for corporate decisions.

**6. Macro factors and business cycles**. The baseline risky debt model in SW12/DF26 only model one idiosyncratic risk. More realistic model should account for aggregate macro risks. There are two macro conditions that affect the optimal leverage choice. The first is the time dimension discussed in point 3: cost of debt is high in bad states and low in good states, so the optimal leverage becomes a function of the business cycle. The second condition is firm's exposure to business cycle, defined as positive covariance between firm's cash flow and macro state. Chen (2010) highlight that a more exposed (procyclical) firm has riskier debt and higher risk premium. Two firms with the same total cash-flow volatility can have different credit spreads due to different exposure to the cycle.
- **Approach**: it's difficult to estimate the full model of Chen (2010). Based on our model, we could first adapt an additional aggregate shock parameter, a reduced-form parameter of firm's exposure, and a pricing kernel to incorporate risk premium in bond price. Another model reference is Bhamra, Kuehn, and Strebulaev (2010b).

**7. Manager-shareholder conflict in leverage choice.** Jointly design the manager contract and optimal capital structure. Morellec, Nikolov, and Schürhoff (2012) develop a dynamic tradeoff model with corporate and personal taxes, refinancing and liquidation costs, and costly renegotiation of debt in distress. Managers own a fraction of the firms' equity, capture part of the free cash flow to equity as private benefits, and have control over financing decisions. The model characterize the optimal leverage decisions of managers.

**Implementation challenges**: Each of these papers typically adds one or two features at a time, because adding all of them at once is what makes the model empirically intractable. So the practical path is to pick one or two features that matter most in the real world for the product, e.g., focusing on priority and maturity. We could also implement the model comparison methods as in Nikolov21 to test which model best explains the data. Model selection is an important phase before we settle the end product.

## Reference {#sec:reference}

Acharya, V. V., and Carpenter, J. N. (2002). Corporate bond valuation and hedging with stochastic interest rates and endogenous bankruptcy. Review of Financial Studies, 15(5), 1355-1383.

Bhamra, H. S., Kuehn, L.-A., and Strebulaev, I. A. (2010a). The levered equity risk premium and credit spreads: A unified framework. Review of Financial Studies, 23(2), 645-703.

Bhamra, H. S., Kuehn, L.-A., and Strebulaev, I. A. (2010b). The aggregate dynamics of capital structure and macroeconomic risk. Review of Financial Studies, 23(12), 4187-4241.

Bolton, P., Chen, H., and Wang, N. (2011). A unified theory of Tobin's q, corporate investment, financing, and risk management. Journal of Finance, 66(5), 1545-1578.

Chen, H. (2010). Macroeconomic conditions and the puzzles of credit spreads and capital structure. Journal of Finance, 65(6), 2171-2212.

Chen, H., Xu, Y., and Yang, J. (2021). Systematic risk, debt maturity, and the term structure of credit spreads. Journal of Financial Economics, 139(3), 770-799.

Hackbarth, D., Hennessy, C. A., and Leland, H. E. (2007). Can the trade-off theory explain debt structure? Review of Financial Studies, 20(5), 1389-1428.

Hackbarth, D., Hennessy, C. A., and Leland, H. E. (2012). Optimal priority structure, capital structure, and investment. (Note: verify exact year and outlet before submission; this work is also widely cited as a Review of Financial Studies contribution in the HHL line. Confirm the precise citation.)

He, Z., and Milbradt, K. (2016). Dynamic debt maturity. Review of Financial Studies, 29(10), 2677-2736.

Jarrow, R., Li, H., Liu, S., and Wu, C. (2010). Reduced-form valuation of callable corporate bonds: Theory and evidence. Journal of Financial Economics, 95(2), 227-248.

Kuehn, L.-A., and Schmid, L. (2014). Investment-based corporate bond pricing. Journal of Finance, 69(6), 2741-2776.

Mella-Barral, P., and Perraudin, W. (1997). Strategic debt service. Journal of Finance, 52(2), 531-556.

Morellec, E., Nikolov, B., and Schürhoff, N. (2012). Corporate governance and capital structure dynamics. Journal of Finance, 67(3), 803-848.
