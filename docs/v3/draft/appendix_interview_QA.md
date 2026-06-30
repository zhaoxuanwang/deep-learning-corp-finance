
# Interview questions
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


## ToC

1. Commercial Product Plan
	- Product#1: CFA
		- Definition; Value; Buyer; Example; Pricing
	- Product#2: DCM
		- Definition; Value; Buyer; Example; Pricing
	- Product#3: leveraged finance and private credit
		- Definition; Value; Buyer; Example; Pricing
	- Model risk and compliance
2. Development plan
	- High-level principles
	- Workflow
	- Use of AI-tools
	- Validation, safety, and monitoring
3. Model and Estimation
	- Purpose, scope, and definitions (DF26)
	- Economic model (risky debt)
	- Method overview
	- Block 1: Solving Model
	- Block 2: Refinement and Data Collector
	- Block 3: Estimation
	- Architecture
	- Reproducibility design
	- Configurations: hyperparameters, inputs, etc
	- Implementation in Tensorflow
	- Model validation
4. User Manual and Examples
	- How to use the current code (notebook demo)
	- How to set or tune configs
	- Hardwares and softwares
- Appendix A for commercial plan
    - US regulation and implication
    - HK regulation and implication
- Appendix B for development
- Appendix C for interview Q&A


---
# Key updates

I made a significant update of the project since the last version (May 29). The current report (June 20) now also builds the  deep learning methods and entire workflow, proposed in a new working paper posted two weeks ago:

> Victor Duarte and Julia Fonseca, "AI for Structural Estimation," NBER Working Paper 35283 (2026), https://doi.org/10.3386/w35283.

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

| Key Parts                       | DF26 & Current report                                               | Previous report              | Maliar21  |
| ------------------------------- | ------------------------------------------------------------------- | ---------------------------- | --------- |
| Risky debt model                | Yes and fast                                                        | Yes but slow                 | No        |
| Policy inputs                   | State x Parameter                                                   | State x Parameter            | State     |
| Model solve                     | Actor-critic (one-step)                                             | Actor-critic (32-step)       | Euler res |
| Discontinuity                   | Smooth approx. + VFI refinement                                     | Smooth approx. with error    | No        |
| Compute $E_{z'\mid z}$          | Gaussian Quadrature (GQ)                                            | GQ and AiO product           | AiO       |
| Default probability             | Numerical approx. from value NN                                     | Exact from nested VFI        | No        |
| Nested fixed point              | target value NN, then VFI refine                                    | nested VFI                   | No        |
| Moments                         | Moment surrogate maps $\beta$ to $g(\beta)$                         | Analytical (mean, SD)        | n.a.      |
| Optimizer for finding $\beta^*$ | Levenberg-Marquardt with analytical Jacobian. Efficient and faster. | Simulated annealing. Slower. | n.a.      |
| Architecture                    | Multiple GPUs and async. computing                                  | Sequential, single CPU/GPU   | n.a.      |
| Validation of model solve       | Benchmark against VFI                                               | VFI itself                   | n.a.      |
| Validation of SMM estimation    | Monte Carlo recovery of true params                                 | Same MC recovery             | n.a.      |
| Bayesian MCMC                   | n.a.                                                                | Policy surrogate             | n.a.      |


---
# Q1: TF and TFP implementation
> *Proper and correct usage of TF and TFP is required. How can we ensure this and address the critical issues? Can you review your code with codex and/or claude code to ensure the code is correct and proper? How can you leverage coding agent to help you design and code something that is much more robust?  Are the test coverage enough?*
## Overview
**Core idea.** Build and enforce a robust system in which human develops the model and the solution methods, specifies implementation details, and designs the test suites and quality checks. Once built, the system evaluates whether code meets a set of criteria in each layer, and automatically block code with defects from progressing. 

**High-level design principles:**
- No black-box or LLM-generated design
- Complete and full documentation
- Full reproducibility and record-keeping
- Model validation with known-answer or credible benchmark

I have implemented this system in the latest version of my report (June 20). 

| Layer                         | Outcomes                                                                                                                                                      | What AI agents do                                                                                                 | What human do                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1. Design and doc             | Full, human-owned design documentation (`method.md`, `model.md`) specifying the algorithm and implementation details; an agent instruction file (`AGENTS.md`) | Drafts the code and environment strictly from your documentation, not AI agent's own assumptions or hallucination | Develop the model, methods, and specs. Write the  documentation and agent instructions.                      |
| 2. Write code                 | Code written to your documentation                                                                                                                            | Writes the code faithfully to the doc                                                                             | Ensure strict enforcement. Understand code.                                                                  |
| 3. Test suite                 | Unit tests, integration tests, known-answer (oracle) tests, and other correctness tests.                                                                      | Writes and runs the unit and integration tests; implements and runs the correctness tests you specify             | Define the tests the pass thresholds; check that asserted values come from the spec, not just current output |
| 4. Code review                | The review checklist (`review.md`), a second independent agent, and your own reading                                                                          | Checks the change against `review.md` and flags issues                                                            | Read the change and resolve flags; never let one agent both write and approve                                |
| 5. Automatic gate             | The quality gate: a rule that runs the checks on every proposed change                                                                                        | Runs formatting, type, and test checks; blocks the merge if any fail                                              | Decide which checks are required; own the rule                                                               |
| 6. Runtime checks and logging | Built-in runtime checks; logs of seeds, versions, and key diagnostics                                                                                         | Stops the run on impossible values; records logs and numbers                                                      | Read the run logs; investigate failures on real data                                                         |
| 7. Results review             | A short results checklist                                                                                                                                     | Summarizes estimates and convergence diagnostics                                                                  | Judge whether results make economic and statistical sense; decide whether to ship                            |

**Test coverage.** I consider two types of "correctness" that the full test suites should address. The first type is **whether each piece and the pipeline wiring are correct**, which is covered by:

- **Unit test**: checks that one function does what it is meant to do (a moment calculation, a transition-matrix builder, a data normalizer), including edge cases and error handling.
- **Integration test**: checks that the parts connect and the whole pipeline runs end to end with the right shapes and types.
- **Regression test:** freeze the output of a run already validated, under a fixed seed and pinned versions, and check that future runs still match it. This catches accidental drift; it does not prove the answer is correct.

The AI agent can mostly write them, because their expected behavior is the code's own contract already specified in design documentation and the code. However, human need to specify the boundary applies to agent-written unit and integration tests in `AGENT.md` to ensure proper implementation. For example, the agent must not assert whatever the code currently outputs.

The second type is the **scientific correctness of the result**. This part requires me (and other human authors) developing the correctness tests carefully: 

- **Known-answer test (oracle)**: run the code on a special case with a known analytical solution, or benchmark against credible results from validated methods (e.g., VFI), or simulate data from known true parameters and check the estimator recovers them.
- **Economic property test**: check properties that economic or statistical theory says must hold for any inputs. Change one input and check the output moves in the theory-predicted way (comparative statics).

These correctness tests are mostly model- or method-specific, so it will be specified by human per `model.md` and `method.md`. 

Take the risky debt model for example, the known-answer test include checking the solved policy and value functions (parameterized by NN) matches the solution from grid-based value function iterations. Then use the validated model solver to simulate data with true parameter values, and verify whether the estimation pipeline (SMM or Bayesian) can correctly recover them. To improve confidence on coverage and robustness, these test need to be repeated for different parameter initialization and simulated batches (RNG seeds).


I have created a `review.md` and will keep maintain and update it. The `review.md` include common high-risk issues, diagnostics, and known fixes that covers:
- TF and TFP usage
- Neural network training
- Estimation
- Data management and matrix operations
- Reproducibility
- Hardware and version compatibility
Note that `review.md` focuses on generic and high-priority issues. Any model- and method-specific issues are covered in `model.md` and `method.md`. 
---
# Q2: Deep Learning Methods

### Data Generation 
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

Both model solve (GPU#1) and estimation (GPU#2-4) run concurrently. When GPUs 2–4 each begin evaluating a new batch of parameter vectors, they each read the most recent value and policy network weights from GPU 1.

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

---
> *What are the mathematical theorems to guarantee the correctness for each time of training regime?*  

- **Solving the model** has math guarantees everywhere except the neural network approximation. The NN-based methods in Maliar21, DF24, and DF26 do not have guarantee, but DF26 is more reliable because it uses VFI to refine the NN policy and value approximation before passing it to estimation.
- **Estimation** has the classical asymptotic guarantees, except the DF26's moment surrogate, which again rests on diagnostics rather than a theorem.

In short, the core parts that lack math theorem to guarantee convergence is the NN approximation to policy/value function and to the moment conditions (DF26). This means we need to design "correctness" tests to ensure that the solution is valid.

**Model solving:**

| Method                                                           | Theorem                                       | Key intuition                                                                                                                          | Guarantee        |
| ---------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| VFI / PFI (grid)                                                 | Banach fixed-point (Blackwell's conditions)   | The Bellman operator is a β-contraction, so iterating it converges to the one true value function.                                     | Yes              |
| Compute expectation over $z' \mid z$                             | Gaussian (Gauss-Hermite) quadrature exactness | n nodes integrate polynomials up to degree 2n−1 exactly, giving precise deterministic expectations for smooth low-dim Gaussian shocks. | Yes (smooth)     |
| Compute expectation over $z' \mid z$ with Maliar's AiO estimator | Law of large numbers                          | Sample averages converge to the expectation; two independent draws keep the squared-residual gradient unbiased.                        | Consistent       |
| NN model solvers                                                 | Universal Approximation + Robbins-Monro (SGD) | The network can represent the solution and SGD reaches a stationary point, but neither guarantee convergence to the true optimum       | No               |
| DF26 grid refinement                                             | Banach contraction, locally per β             | A few exact policy-iteration steps pull the network policy to the grid optimum, restoring the contraction guarantee.                   | Yes (up to grid) |
| LP method (Nikolov21)                                            | LP formulation of dynamic programming         | The value function is the unique LP solution (smallest V with V ≥ TV), recovered exactly on the grid.                                  | Yes              |

**Estimation:**

| Method                              | Theorem                                            | Key intuition                                                                                                              | Guarantee  |
| ----------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Simulated moments                   | Ergodic theorem (LLN for Markov chains)            | Long simulated panels after burn-in make sample moments converge to the model's true stationary moments.                   | Asymptotic |
| SMM / GMM                           | McFadden 1989, Pakes-Pollard 1989 (Hansen 1982)    | Matching simulated to data moments gives a consistent, asymptotically normal estimator under identification.               | Asymptotic |
| Moment-surrogate net (DF26)         | Universal Approximation + diagnostics              | The parameter-to-moment map is representable, but accuracy is certified only by held-out R² and cross-validation.          | No         |
| Bayesian MCMC                       | MH ergodicity (Tierney 1994) + Bernstein-von Mises | The chain converges to the posterior, which is asymptotically normal around the truth.                                     | Asymptotic |
| Filtering (Kalman / EKF / particle) | Kalman optimality; particle-filter LLN             | Kalman is the exact filter for linear-Gaussian; EKF only approximates it; the particle filter converges as particles grow. | Mixed      |
| Indirect inference                  | Gourieroux-Monfort-Renault 1993                    | Matching an auxiliary model's parameters identifies the structural ones through the binding function.                      | Asymptotic |

---
### Finite sample bias
> *If we use fixed set of data, is there bias in the final result?  Why and why not?  Can you prove your statement one way or another?  Can you run a sample experiment with a simple example to confirm the issue?  E.g. a linear quadratic control problem that has explicit solution?* 

There are two kinds of bias from a fixed training dataset.

The first is **overfitting frozen randomness**. A fixed dataset locks in the particular shock draws that went into it. A flexible model like a neural net will fit those specific draws rather than the true average behavior, so it ends up learning the sample's noise. Fresh data shows new shocks at every step, which forces the model toward the true average instead of memorizing one realization.

The second is the **coverage gap**. A finite dataset only covers finitely many states. Wherever the sample has no points, nothing pins the solution down, so it is simply wrong there. If the simulated economy later visits those states, or if we need the solution across a range of parameters as in DF26, that error flows straight into the moments and the parameter estimates.

Both biases shrink only as the data covers more of the space, not as we train longer. The real fix is to adopt the DF26 approach: **keep drawing fresh states, use them once, and discard them**, which over time covers the whole space and averages out the shocks.
#### A simple linear quadratic control example

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
$$ \hat{w} = \arg \min_w \frac{1}{N} \sum_{i=1}^N \text{​res}(s_i​)^2 $$
The true $V^*$ makes this residual zero everywhere. 

**Fixed vs fresh sample**: the problem is solved twice on two samples with identical budget. Both runs use the same per-step batch size ($N=12$ states) and the same number of gradient steps ($T$), so the per-step sample size and the total compute are held equal. The only difference is:
- **Fixed run** draws $12$ states on $[-3,3]$ once and reuse that same batch at every step.
- **Fresh run** draws $12$ new states on $[-3,3]$ at every step and then discard them.

The figure below shows the solved value function over the two runs in the fair setup (both use batch size 12 and the same 60,000 steps). The black curve is the true value function. The blue dashed curve is the fresh run (12 new states each step), which tracks the truth closely (held-out RMSE 0.06). The red curve is the fixed run (reusing the same 12 states), which is visibly off the truth across the domain (held-out RMSE 2.89), and the red ticks at the bottom mark the 12 states it kept reusing.

**Why using neural network has analogous problems?** Both biases come from one ingredient: a model far more flexible than the true solution, over-fitting to fixed finite data points. A neural net would behaves the same way.
![[lq_value_function.png]]

---
### Euler Residual Methods
> *For the Euler residual equations, we need very equations to be satisfied correctly for the AIO method.  How can we ensure that?  Is it reasonable to just use some norm of all the Euler residuals?  Are all the residuals of the same scale?  Even if they are of the same scale, is just some norm as loss function correct? How to handle this problem?*

A quick recap of the Euler residual method. Denote $f\equiv f(k',b',z';k,b,z)$ as the closed-form formula such that under optimality the Euler equation holds:
$$
E_{z'|z} \left[ f\right]=0
$$
Taking $M$ random draws of next-period shock $\{z'_m\}^M_{m=1}$, Maliar21 propose using the All-in-One (AiO) cross-product to form the loss function:
$$
L\equiv \frac{1}{N}\sum_i \left[ f(z'_1)\times f(z'_2) \right]
$$
which is an unbiased estimator for $E_{z'|z} [f]^2$. Minimizing the loss function enforces the first-order necessary condition to hold. 

**1. How do we ensure every equation holds?** For models with closed-form Euler equation, we can write $f$ by re-arranging the Euler equation (LHS-RHS), then everything is written analytically. However, for more complex models like risky debt, a closed-form Euler equation is not available. Specifically, we can still write down the investment FOC for $k'$, but cannot do so for $b'$. To see this, the marginal cost of debt on one side of the debt FOC is $-\gamma E\left[ 1\{\text{solvent}\} V_{b'} \right]$, where the default (solvent) indicator and the derivative of $V$ wrt $b'$ has not analytical formula.

**2. What norm should we use?** Maliar21's cross-product is unbiased and works fine in practice. The main concern is the AiO cross-product fluctuates around zero and can go negative. That oscillation is an artifact of the unbiased estimator, not a bug, since the gradient is still correct on average.

We should NOT naively use norms like MSE, because it would bias the gradient:
$$E_\varepsilon[f(\varepsilon)^2]=(E_\varepsilon[f(\varepsilon)])^2+\operatorname{Var}_\varepsilon(f(\varepsilon)).$$
The extra variance term means a plain squared loss does not target $E_\varepsilon[f]=0$, it also penalizes the residual's variance, so its gradient points the wrong way. 

**3. Are the residuals on the same scale?** No. Euler residuals can be written in unit-free (relative) form, for example the investment Euler $1+\psi_1 I/k=\frac{1}{1+r}E[\cdots]$ where both sides are around one, so they sit at a small, comparable scale and a plain sum with unit weights works. My results confirm the Euler residuals can be pushed close to zero this way. 

However, for more complex models with additional constraints, such as the inequality budget constraints introduced in Maliar21, this will cause scale mismatch. Maliar21's consumption-saving problem allows for the inequality constraint to be written in unit-free terms so that it is not a concern, but for our corporate finance models, most of such constraints are written in Bellman term (e.g., cash) and cannot be easily normalized, this will caused the scale mismatch issue as documented for the Bellman residual method. The larger residual will dominate the gradient direction and may lead to incorrect solutions. No mathematically theorem can guarantee this, and my empirical result confirms this. Maliar21's proposed fix is to manually tune the weights on losses with different scale, but in my view this is not robust for production.

**Beyond scale and bias.** Fixing both is still not enough when many auxiliary losses are stacked (Bellman residual, first-order condition, envelope condition, constraints). They can pull in different directions: only the FOC term points toward the optimal policy, while the others can be driven to zero by an arbitrary, wrong policy-value pair, so the joint loss looks minimized while the solution is wrong. My report has confirmed this and show that this is why Maliar21's Bellman residual approach failed in practice. 

**Summary.** Euler residual approach with AiO estimator works well only when the optimality condition has closed-form formula, which requires the objective to be known analytically, smooth, and differentiable. It is NOT a valid method for more complex models like the risky debt model.

---
### NUTS-HMC in TFP
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

**2. TFP implementation** (according to the [doc](https://arxiv.org/pdf/2002.01184).):  TFP runs all chains (minimum 4) together as a single batch and steps them through one shared loop. But NUTS builds each proposal by doubling a trajectory until it makes a U-turn, and that length varies from chain to chain. 

To fit this dynamic recursion into a batch, TFP unrolls it into one loop that does the same amount of work for every chain at each step. The key is that the shared loop cannot stop until the chain with the longest trajectory finishes (up to the max tree depth). So the whole batch moves at the pace of its deepest chain, and the chains that already turned around get dragged along through extra steps. Since the cost is dominated by gradient evaluations, those extra steps are wasted gradients.

**What to do**
When gradient is expensive (surrogate policy, or any model solve in the loop) so that NUTS is intractable, we have these options:
1. `tfp.mcmc.HamiltonianMonteCarlo` with manually tuned `num_leapfrog_steps` and `step_size`.
2. `tfp.experimental.mcmc.windowed_adaptive_hmc` with manually tuned `num_leapfrog_steps`. 
3. Run other gradient-free samplers like Random-Walk Metropolis.  

The first two options manually fix the number of leapfrog steps of HMC. As baseline, I'd start with using `windowed_adaptive_hmc` because it adapt the step size $\epsilon$ to a target acceptance rate and estimates a diagonal mass matrix during warmup (see details in [doc](https://mc-stan.org/docs/reference-manual/mcmc.html)) The `num_leapfrog_steps` can be either set manually or adopted from a short NUTS run and adopt the typical $L$ that settled. I summarize the motivation below:

| Param                   | Impact and Trade-off                                                                                                                                               | How to tune                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Step size $\varepsilon$ | Accuracy v.s. cost. Too large $\varepsilon$ increases error, too small $\varepsilon$ waste compute on gradient eval.                                               | Auto-tuned in Option 2 `windowed_adaptive_hmc`                       |
| Leapfrog $L$            | Chain mixing v.s. wasted compute. Too few steps is not enough for chain to mix because $\beta$ barely moved (auto-correlated), too large waste gradient eval cost. | Either manually or run a small NUTS warmup and adopt the typical $L$ |
| Mass matrix $M$         | Ideal $M$ is the posterior covariance matrix for $P(\beta \mid y)$. Affect efficiency.                                                                             | Auto-estimated in Option 2 `windowed_adaptive_hmc`                   |

---
# Q3: Commercial Product
> *Suppose you have to sell your product to another bank and make your work practical. Which departments of a bank like JP Morgan Chase can benefit from the project can benefit from this work?*  

**1. Corporate Finance Advisory (CFA)**. This is the primary product case and can directly support [JPMC's CFA teams](https://www.jpmorgan.com/investment-banking/corporate-finance-advisory) on: 
- **Corporate finance solutions**: Provide analysis and recommendations pertaining to capital structure, capital allocation and shareholder distribution policy, including broader market insights
- **Ratings advisory**: Strategic and tactical advice on management of rating agency relationships and optimization of capital structures for desired credit rating objectives
- **Structuring and solutions:** Structure products and alternative financing solutions relating to strategic M&A and capital markets.

**2. Debt Capital Markets (DCM)**. JPMC's [DCM](https://www.jpmorgan.com/investment-banking/capital-markets#accordion-53aabc808f-item-11b1e6f6a7) team can use the tool at key decision points to assists clients on a wide range of debt financing strategies. 
- **Ratings advisory**: collaborate with CFA team to answer key questions like "if we add this much debt, what's the probability of a rating downgrade?"
- **Bond pricing advisory**: provide an independent fair-value anchor based on model estimates from the firm's fundamentals, which can be used to compare the market spread the bank derived from similar-rated bonds trade.
- **Liability management and structuring**: help client deciding whether to refinance, change the debt mix, or pick a tenor and seniority. Our model enables full dynamic counterfactual analysis across alternative capital structures. 

**3. Structural Credit & Leverage Engine**. A fundamentals-based underwriting and monitoring tool for leveraged finance and private credit. Internally it supports JPMorgan's Leveraged Finance group and its direct lending platform in CIB, see [example](https://www.jpmorgan.com/about-us/corporate-news/2025/jpmorgan-increases-direct-lending-commitment-to-50-billion). Externally it is sellable to direct lenders, private credit funds, and leveraged buyout sponsors.
- **Underwriting and leverage sizing**: Give an independent, fundamentals-based estimate of how much debt a borrower can sustainably carry, and its default probability over the life of the loan, as a check on the deal team's comparable-based view before capital is committed.
- **Risk-based pricing**: Produce a fair credit spread for the proposed structure from the borrower's own fundamentals, flagging when a deal underpays the lender for the risk taken.
- **Scenario and covenant stress testing**: Run dynamic counterfactuals across leverage, tenor, and downturn cases, quantifying how default risk and covenant headroom (the leverage and coverage tests) move if earnings fall. This is timely, since 2026 is widely seen as the private credit market's first real stress test.
- **Portfolio monitoring and early warning**: Re-estimate the model each quarter across the existing loan book to flag borrowers whose default risk is rising before it surfaces in the financials, supporting risk governance and investor reporting.

**Example: Corporate Finance Advisory (CFA)**
A company holds $2 billion in surplus cash and wants to return it to shareholders through a buyback, funded partly with new debt. Its board asks one question: does the plan raise value without threatening the company's credit rating? JPMorgan's CFA team advises on the answer. The standard approach relies on peer benchmarks, rating-agency ratio scorecards, stress scenarios, and experienced judgment. These can estimate how the company's ratios map to a likely rating, but they do not value the choice itself, weighing the gain in firm value against the added default risk in a unified, model-based framework.

The model answers the question directly. From the company's own financials, it simulates the buyback and measures two effects together: the change in firm value and the change in default risk. It shows that the full $2 billion buyback raises value only modestly and lifts the chance of a one-notch downgrade to roughly 30%. A smaller $1.5 billion buyback captures most of the value while keeping downgrade risk low. The team brings the board a clear tradeoff, grounded in the company's own numbers rather than peer averages.

The bank benefits in two steps. A recommendation built on the client's fundamentals is more convincing than a benchmark-based one, so it helps win the advisory mandate. It also positions the bank to lead the debt issuance that funds the buyback, where the larger fees are earned. Rival banks, working from comparables alone, cannot answer the board's actual question: what this specific decision does to this specific company.

The value reaches the bank through the mandates it wins, not through a single fee. How to price the tool is therefore better settled in practice than asserted now.

**Example: Debt Capital Markets (DCM)**
A company plans to raise $1 billion by issuing a bond, and JPMorgan's DCM team must advise on the price. The standard method anchors the price to where comparable bonds trade, the issuer's own outstanding debt and similar-rated peers, plus a small new-issue concession. This reflects the market's current read and the issuer's rating, not an independent estimate built from the issuer's own fundamentals.

The model prices the bond from the issuer's fundamentals. It finds the company is stronger than its rating peers and supports a tighter spread. Carrying that independent anchor into the deal, the bank prices the bond about 15 basis points tighter than comparables alone would have set. On a $1 billion ten-year bond, that saves the issuer roughly $1.5 million a year, about $15 million over its life.

The bank captures this value indirectly. The client receives a better result than a peer-based rule would deliver, which strengthens the relationship and the bank's case to win the lead role, where several million dollars of underwriting fees are earned. Competing banks, working only from comparables, cannot produce the same fundamentals-based anchor. The tool does not replace the banker's judgment. It gives that judgment independent support.

As in the advisory case, the value appears in deals won rather than in a direct charge, so the pricing is best left open for now.


**Example: Structural Credit & Leverage Engine:**
A private credit fund must decide whether to lend $200 million to a company that a private equity firm is buying. Its analysts compare the deal to how similar companies were financed and conclude the loan is safe and fairly priced. This peer comparison is the industry standard, and it has one structural weakness. No peer company is this borrower carrying this much new debt. The fund's existing tools, comparison tables and rating-style scorecards, all rest on peers and historical averages. None can model this specific borrower under this specific debt load.

Our structural model closes this gap. It uses data of the borrower's own financials and estimates the probability that the company cannot service the debt. It then computes the interest rate that fairly compensates the fund for that risk. In this case, our model confirms the loan is safe but flags the proposed rate as 50 basis points too low. The fund secures the higher rate. On a $200 million loan, that adds $1 million of income per year, about $5 million over a five-year term.

The larger value lies in the deals that our tool warns the fund to avoid. A single default on a loan this size can cost the fund $80 million or more. Avoiding one such loss may outweighs many years of the subscription.

---
> *What is the price that you think your client is willing to pay?* *Why would they want to purchase the product?*

I use the three product category above as example. The goal here is not to pin down a number (which is not that useful at current stage), but to develop and understand the key sub-questions to be settled for making it a viable commercial product.

**Immediate buyer is the bank, not corporate client.** For all three categories, the immediate buyer is the bank (either internally or externally), not the corporate client. Willingness to pay is anchored to the potential value created to the bank. How can this tool let the bank 
- price a deal better? 
- win more pitches and earn higher advisory fees? 
- reduce the risk of wrong forecast?
- reduce analysis cost? 
Therefore we should anchor pricing to the bank's service, not to corporate client's CFO budget.

**Product value depends on three parties trusting it**. We have to trust it to sell it to a bank. The banker has to trust it enough to put it in a pitch. The CFO and board have to trust it enough to act on it. This "three-party trust" requirement means we should prioritize explainability:
- benchmarking performance against the methods those parties already trust,
- a validation track record, 
before we have a sellable product at all. 

==**1. Price for the CFA tool**.== 
CFA work is firm-specific advice on capital structure, allocation, payout, and ratings. The model produces exactly that counterfactual analysis quantitatively. For a banker, this is a differentiated pitch tool: "our proprietary model, estimated on your fundamentals, says X." This is also a way to standardize work that is currently bespoke and analyst-heavy.

- **Ceiling**: set by how much the bank believes the model lifts mandate wins and fees; 
- **Floor**: set by the analyst hours it displaces.

**Realistic pricing strategy**: platform subscription + per-user license (seats), with the value carried in the sales narrative rather than a success fee.

**Ideal pricing strategy**: causal attribution that pin down the success fee per-deal. Example: our product uplift 10% of the advisory fee. We need to deploy causal inference tools for credible evaluation. For example, start with a randomized experiment that give the tool to different teams internally, collect data on outcomes, and estimate the average treatment effects.

**Reason:** It is hard to pin down the product uplift of wins and fees. CFA is often a relationship and credibility product that supports the broader IB relationship, so part of the product's value is indirect and attribution is weak.

==**2. Price for the DCM tool**.== 
DCM decisions attach to basis points on large events (e.g., issuance, M&A). An independent fair-value anchor that helps price a deal a few bps better, or that strengthens the pitch to win the book-runner role, both has a clear and large per-deal value.

**Example:** A company wants to raise two billion dollars by issuing a 10-year bond. It hires banks to run the sale, and our tool is used by the bank on this deal. There are two places value can come from. 
1. **Winning the mandate:** If the tool's independent fair-value analysis makes the bank's pitch more convincing, the bank is more likely to be chosen as bookrunner. Suppose the bank's fee on a deal this size is a few million dollars.
2. **Better pricing for the client**: Suppose the tool shows the bond can price 5 bps tighter than the comparable-bond view suggested. On two billion dollars, 5 bps is 0.05%, which is one million dollars a year that the borrower saves. This win more future mandates and builds the bank's reputation.

**Pricing strategy**: a share of the value the bank can credibly attribute to the tool across a year of deals it touches, where that value is mostly extra mandate fees and proven cost savings.

---
> *Suppose we would like to apply your work for the investment banking application of advising the clients on capital structure, what is missing on the current report? What model do we have to build further? Can you give a draft project plan to make this a fully useful product?*

As the baseline model, I'll take the endogenous default model in DF26 (Section 2), which incorporates AR(1) productivity shocks, fixed operating cost, convex capital adjustment cost, external equity cost, and most importantly, endogenous default and market-clearing bond price. This is the same risky debt model in SW12. 

I consider four key features that are first-order to make the model a useful product:

**1. Various types of debt.** Real-world firms hold and consider a wide variety of debt financing strategies, including bonds with different maturity, priority and seniority, callable options, etc. The basic model only considers a one-period corporate bond and is far from being realistic. Extensions will follow the dynamic structural models summarized below:
- **Maturity**: He and Milbradt (2016) allows firm to choose its debt maturity structure and default timing dynamically. Chen, Xu, and Yang (2021) links leverage to maturity to capture the tradeoffs between the liquidity discount on long-term debt, the repayment risk of short-term debt, and the value of short-term debt as a commitment device.
- **Priority and seniority**: Hackbarth, Hennessy, and Leland (2012) derives the jointly optimal priority and capital structure when the firm has multiple debt classes and investment. Hackbarth, Hennessy, and Leland (2007) examine the optimal mixture and priority structure of bank and market debt.
- **Callable bond:** Acharya and Carpenter (2002) derives analytically the optimal call and default rules when interest rates and firm value are stochastic. Jarrow, Li, Liu, and Wu (2010) develops an alternative reduced-form approach for valuing callable corporate bonds by empirically characterizing the call probability.
- **Renegotiation, loan-versus-bond**:  Hackbarth, Hennessy, and Leland (2007) incorporates the bank-versus-market choice. Mella-Barral and Perraudin (1997) is the canonical strategic-renegotiation pricing model.

**2. Regional regulatory and institutional setting.** Model timing and setup need to be aligned with local tax code, bankruptcy and recovery rule, investment and production schedule, etc. 

**3. Systematic risk and credit spread.** The baseline risky debt model assumes no risk premium. The endogenous bond price is defined by the breakeven condition, in which the bond yield only compensates for expected loss, and the lender just earns the risk-free rate in expectation. In reality, lenders demand extra yield as risk premium. The risk premium compensates for systematic risk, the part of default that is correlated across firms with bad aggregate states and is not diversifiable. This is documented in literature that the baseline model's implied debts are "too cheap" and thus over-predict the optimal leverage.
- **Approach**: I will start by adding a stochastic discount factor to the bond pricing equation, similar to Kuehn and Schmid (2014, Eq.15). Further extension should refer to Bhamra, Kuehn, and Strebulaev (2010a) and Chen (2010): both replace the risk-free discounting in the standard SW12/DF26 risky-debt model with a consumption-based stochastic discount factor carrying a countercyclical price of risk, so that debt and equity are priced consistently under systematic macroeconomic risk, which jointly raises credit spreads to realistic levels and lowers optimal leverage toward observed values.

**4. Firm-specific estimation.** The model estimation is based on a full panel of firms, using both cross-sectional and time-series variations. The estimated parameters are thus NOT firm-specific. If our goal is tailor the product to a named client (e.g., Tencent), we should ideally need a higher frequency time series data of the single company's financials. 
- **Approach**: calibrate the parameters that are common or weakly identified (e.g., recovery and tax), and estimate the firm-specific technology and cost parameters from the client's own time series.

There are more model features that I view as second-order (for debt financing) but could be useful for general advisory on capital structure:

**5. Cash and liquidity management**. Capital structure advice should also involves cash savings and credit lines (loan) as a buffer. The canonical model is Bolton, Chen, Wang (2011) for dynamic investment, financing, and risk management for financially constrained firms. It highlights the central importance of the endogenous marginal value of liquidity (cash and credit line) for corporate decisions.

**6. Macro factors and business cycles**. The baseline risky debt model in SW12/DF26 only model one idiosyncratic risk. More realistic model should account for aggregate macro risks. There are two macro conditions that affect the optimal leverage choice. The first is the time dimension discussed in point 3: cost of debt is high in bad states and low in good states, so the optimal leverage becomes a function of the business cycle. The second condition is firm's exposure to business cycle, defined as positive covariance between firm's cash flow and macro state. Chen (2010) highlight that a more exposed (procyclical) firm has riskier debt and higher risk premium. Two firms with the same total cash-flow volatility can have different credit spreads due to different exposure to the cycle.
- **Approach**: it's difficult to estimate the full model of Chen (2010). Based on our model, we could first adapt an additional aggregate shock parameter, a reduced-form parameter of firm's exposure, and a pricing kernel to incorporate risk premium in bond price. Another model reference is Bhamra, Kuehn, and Strebulaev (2010b).

**7. Manager-shareholder conflict in leverage choice.** Jointly design the manager contract and optimal capital structure. Morellec, Nikolov, and Schürhoff (2012) develop a dynamic tradeoff model with corporate and personal taxes, refinancing and liquidation costs, and costly renegotiation of debt in distress. Managers own a fraction of the ﬁrms’ equity, capture part of the free cash ﬂow to equity as private beneﬁts, and have control over ﬁnancing decisions. The model characterize the optimal leverage decisions of managers.

**Implementation challenges**: Each of these papers typically adds one or two features at a time, because adding all of them at once is what makes the model empirically intractable. So the practical path is to pick one or two features that matter most in the real world for the product, e.g., focusing on priority and maturity. We could also implement the model comparison methods as in Nikolov21 to test which model best explains the data. Model selection is an important phase before we settle the end product.

---
> *What are the likely issues when you try to sell your product? What are the obstacles do you think you are going to have when deploying your product in real life?*

**Key issues for deployment in real life:**
- Model validation, risk management and compliance
- Show clear value to potential clients, proper pricing and business model
- Ongoing monitoring after launch and performance evaluation

The primary obstacle is **model risk management and compliance**: the product must address the related laws, rules, and supervisory guidance when deployed inside a global bank like JPMorgan. Specifically, the product need to pass the requirements on model validation, risk management, AI-tool-use rules, and data protection policies. 

The second issue is proper **product design and pricing**: a lot of work is needed to understand what are the real needs of client, and how our model can be developed into a viable commercial product to meet the need and create clear value. This requires deeper industry knowledge, professional guidance, and meetings with potential clients.

The third issue is **ongoing monitoring and improvement** after launch. Model design and estimates should be updated with new data and customer feedbacks. We need to develop credible metrics and methods to evaluate the effectiveness of our tool, and quantify its impact.

My plan considers two main jurisdictions: (1) the United States, where JPMorgan is supervised as a US bank, and (2) the Hong Kong SAR, where the bank operates through locally regulated entities. 

**Scope of the product**: For classification purposes, the end product is defined as a quantitative/statistical model that consume input data and configurations, and output statistical estimates. There are several important boundaries:
- The model is NOT a black box and is fully explainable: the deep neural networks are used to approximate an object (e.g., policy, moment) with explicit definition
- The model is NOT a generative model or an agentic system
- NONE of the algorithm and product design are generated by an LLM
- AI-tool-use is restricted to coding implementation and testing, supervised and reviewed by human
Most of the current AI-related obligations center on generative AI and agentic system. Our model is designed such that it is not generative or agentic, and it is largely within the standard quantitative/statistical model category.

**Model risk management and compliance** (as of June 2026):
The table summarizes the key regulations, laws, and policy instruments that I find relevant. The link to the official sources and detailed discussion are left in appendix. 

Shared compliance baseline:
- **A human stays responsible.** Final accountability rests with named people, not with the model. The model is decision support: a banker or officer reviews and signs off, and the model never makes a decision automatically on its own. (US: SR 26-2 governance; HK: HKMA Principle 1, SFC General Principle 9.)
- **No black box.** The model must be explainable to the people who rely on it and to independent reviewers, clearly enough that they can question and challenge it. We have to be able to say, in explicit terms, why the model produced a given number. (US: "effective challenge"; HK: HKMA Principle 3, "no black-box excuse.")
- **Reproducibility and record-keeping.** The model version, spec, parameters, inputs, and the output that informed any advice must be retained for the applicable period and reproducible on demand. Build versioning and an audit log into the product from the start. Baseline validation should emphasize reproducibility control.
- **Independent validation before it goes live.** The model must be tested and approved before it is used, and the builder cannot also be the validator. (US: SR 26-2 conceptual soundness and effective challenge; HK: HKMA Principle 5, SFC Internal Control Guidelines.)
- **Concrete and complete documentation.** The design, assumptions, model choices, data sources, and test results must be documented well enough for a third party to follow and reproduce the work. In practice, if it is not written down, it does not count as done. (US: SR 26-2 conceptual-soundness documentation; HK: across the AI principles and the conduct code.)
- **Out-of-Sample Testing** The model must be checked against actual outcomes on data it was not built on (out-of-sample back-testing), not merely shown to fit its own training data. (US: SR 26-2 outcomes analysis; HK: SPM CA-G-3, validation "should not be limited to back-testing.")
- **Monitoring after launch.** Performance must be monitored over time, and the model re-checked or re-estimated when markets or conditions change. (US: SR 26-2 ongoing monitoring; HK: HKMA AI principles.)
- **Be honest about limitations.** Known weaknesses must be disclosed to users, and use restricted where the model is weak. For us this means reporting ranges rather than false precision where parameters are only weakly identified. (US: SR 26-2 on use limitations; HK: same expectation.)
- **Mind the input data.** The data feeding the model must be appropriate and of good quality, and the data choices documented. (US: SR 26-2 data selection; HK: HKMA Principle 4, data quality.)
- **If sold, give the buyer enough to validate it.** A buyer institution stays accountable for any model it uses, so we must ship a transparency package that lets its own reviewers understand and check the model without us handing over proprietary code. (US: SR 26-2, Section VII; HK: third-party provider principles.)

| #   | Jurisdiction | Instruments                                                                                                            | Issuer                             | Type                                | Binds core product?                     |
| --- | ------------ | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------- | --------------------------------------- |
| A.1 | US           | SR 26-2 / OCC Bulletin 2026-13, Revised MRM Guidance                                                                   | Fed / OCC / FDIC                   | Quantitative-model (model risk)     | **Yes**                                 |
| A.2 | US           | SEC / FINRA supervision, recordkeeping, fair dealing                                                                   | SEC / FINRA                        | Advisory conduct overlay            | Conditional (client-facing advice)      |
| A.3 | US           | Interagency Third-Party Risk Mgmt Guidance (OCC 2023-17)                                                               | Fed / OCC / FDIC                   | Vendor onboarding                   | **Yes, if we are the vendor**           |
| A.4 | US           | Forthcoming AI RFI; EO 14179 federal posture                                                                           | Fed / OCC / FDIC; White House      | AI-specific overlay                 | No (watch only)                         |
| B.1 | HK           | Banking Ordinance Cap. 155 (7th Sch.); SPM IC-1, CG-1, CA-G-4, CA-G-3; HKMA 2019 AI principles; PRA SS1/23 (benchmark) | HKMA; Legislature; PRA (benchmark) | Quantitative-model (model risk)     | **Yes (bank entity)**                   |
| B.2 | HK           | Code of Conduct (GP1, 2, 3, 5, 7, 9); CFA Code (Type 6); Internal Control Guidelines                                   | SFC                                | Advisory conduct + model governance | **Yes (advisory entity)**               |
| B.3 | HK           | PDPO (Cap. 486) + PCPD AI Model Framework                                                                              | Legislature / PCPD                 | Data privacy                        | **Yes, if personal data** (CEO product) |
| B.4 | HK           | SFC GenAI circular (24EC55); HKMA GenAI consumer circular                                                              | SFC; HKMA                          | AI-specific overlay                 | No (generative / customer-facing only)  |

---
# Reference

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