# Outline

## Common Definitions

### Parameters
- Policy net params: $\theta_{\text{policy}}$
- Value net params: $\theta_{\text{value}}$
- Price net params (risky debt only): $\theta_{\text{price}}$

### Notation
- Current period variable: $x$
- Next period variable: $x'$
- Parameterized function: $\Gamma(\cdot; \theta)$
- DNN parameters: $\theta$

### State Variables
- Capital stock: $k$
- Debt: $b$  (borrowing-only in this project version, see Constraints)
- Exogenous productivity shock: $z$

### Shocks
- $\varepsilon' \sim \mathcal{N}(0,1)$ i.i.d.; AR(1): $\ln z' = \rho \ln z + \sigma \varepsilon'$
- Use **two independent draws** $(\varepsilon'_{1},\varepsilon'_{2})$ per state for All-in-One (AiO) losses to avoid nested expectations.

### Mini-batches
- Basic: $\mathcal{B}_n = \{(k_i, z_i, \varepsilon'_{i,1}, \varepsilon'_{i,2})\}_{i=1}^n$
- Risky debt: $\mathcal{B}_n = \{(k_i, b_i, z_i, \varepsilon'_{i,1}, \varepsilon'_{i,2})\}_{i=1}^n$

### Policies and Prices
- Basic policy: $(k,z)\mapsto k'$
- Risky-debt policy: $(k,b,z)\mapsto (k',b')$
- Risky-debt price: $(k',b',z)\mapsto \tilde r$

### Constraints (Project Version)
- Capital stock is positive: $k>0$ and action satisfies $k'>0$
- **Borrowing-only**: restrict state and action to $b \ge 0$ and $b' \ge 0$
  - Note: the paper allows $b<0$ (cash saving). I avoid it here because it can destabilize training and makes the pricing loss irrelevant when $b<0$.
  - If savings is needed later, implement a separate model class with an explicit savings regime and a loss-normalization / masking design.
- Risky interest rate lower bound: $\tilde r \ge r$ where $r>0$ is the risk-free rate
- Risky-debt latent value $\widetilde V(k,b,z)$ can be positive or negative

### Loss Names
- LR: Lifetime Reward $\widehat{\mathcal{L}}_{\text{LR}}$
- ER: Euler Residual $\widehat{\mathcal{L}}_{\text{ER}}$
- BR: Bellman Residual
  - Critic: $\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}(\theta_{\text{value}})$
  - Actor: $\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})$
- Price: lender zero-profit residual $\widehat{\mathcal{L}}^{\text{price}}$

---

## Deep Neural Network Architecture

### Basic Model Networks
1. Policy network:
$$k' = \Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$$
2. Value network (BR only):
$$V(k,z) = \Gamma_{\text{value}}(k,z;\theta_{\text{value}})$$

#### Inputs and outputs
- Inputs to networks: $(\log k, \log z)$
- Outputs (activations):
  - $k' = k_{\min} + \mathrm{softplus}(\cdot)$
  - $V(k,z) = \mathrm{linear}(\cdot)$
- Primitives use **levels**:
  - $k=\exp(\log k)$, $z=\exp(\log z)$
  - Use levels in $\pi(k,z)$, $\psi(I,k)$, $e(\cdot)$, etc.
- In this project version: **do not standardize** network inputs/outputs using running mean/std.

### Risky Debt Model Networks
1. Policy network:
$$ (k',b') = \Gamma_{\text{policy}}(k,b,z;\theta_{\text{policy}}) $$
2. Bond pricing network:
$$\tilde{r}(k',b',z) = \Gamma_{\text{price}}(k',b',z;\theta_{\text{price}})$$
3. Continuation (latent) value network:
$$\widetilde{V}(k,b,z) = \Gamma_{\text{value}}(k,b,z;\theta_{\text{value}})$$

#### Inputs and outputs
- Inputs to networks: $(\log k, b/k, \log z)$
- Outputs (activations):
  - $k' = k_{\min} + \mathrm{softplus}(\cdot)$
  - **Borrowing-only**: $b' = k \cdot \bar{\ell} \cdot \sigma(\cdot)$ (or $k\bar\ell\cdot \mathrm{softplus}(\cdot)$)
    - Ensures $b'/k \in [0,\bar\ell]$ and prevents extreme $b'$ early in training
    - Use **current** $k$ (not $k'$) for scaling
  - $\tilde{r}(k',b',z) = r_{\text{risk-free}} + \mathrm{softplus}(\cdot)$
  - $\widetilde{V}(k,b,z) = \mathrm{linear}(\cdot)$
- Notation tie-down (avoid ambiguity):
  - $\widetilde V(k,b,z)\equiv \Gamma_{\text{value}}(k,b,z;\theta_{\text{value}})$

### General Specification
- Structure: Fully Connected Neural Networks (FCNN)
- Baseline: 2 hidden layers with 16 units each
- User can specify number of hidden layers and neurons per layer
- Input convention:
  - Networks take **transformed features** (e.g., $(\log k,\log z)$ or $(\log k,b/k,\log z)$)
  - Whenever formulas write $(k,b,z)$, these are **economic level variables** used inside primitives
- Primitive convention:
  - Compute primitives using **levels** recovered from transforms, not transformed inputs

### User-provided Hyperparameters
- Training loop: batch size $n$, outer iterations $N_{\text{iter}}$, critic steps per outer loop $N_{\text{critic}}$, learning rates (policy/value/price), optimizer choice, gradient clipping (on/off + threshold)
- LR method: rollout horizon $T$, number of rollouts per update (if different from batch size), discount $\beta$
- Replay / sampling: replay buffer size, replay mixture ratio (box vs replay), percentile $q$ for default-boundary oversampling, replay refresh frequency $M$
- Default smoothing: $\epsilon_{D,0}$, $\epsilon_{D,\min}$, decay $d$, logit clip $u_{\max}$
- Loss weights: $\lambda_{\text{price}}$ (LR), $\lambda_1$ (critic), $\lambda_2$ (actor)
- Action constraints: $k_{\min}$, leverage scale $\bar\ell$ (and borrowing-only enforcement $b'\ge 0$)

---

## Shared Training Rules (Apply to All Methods)

### State Sampling and Replay
- Maintain training distribution $\mu$: mix of
  1) broad box sampling early (e.g., random uniform), and
  2) sampling from an ergodic / replay set later (revisit states recorded in previous batches).
- Risky debt: oversample near the default boundary (small $|\widetilde V|$) using a **percentile rule**:
  - On replay buffer, compute $a_i = |\Gamma_{\text{value}}(k_i,b_i,z_i;\theta_{\text{value}})|$
  - Select states in the bottom $q$ percentile: $\{i:\ a_i \le \text{quantile}_q(a)\}$ (e.g., $q=5\%$)
  - Sample an extra fraction of replay states from this subset
- Rationale: percentile oversampling auto-adapts to value rescaling during training and avoids tuning a fixed threshold.

### Default Smoothing Annealing (Risky Debt Only)
Maintain temperature $\epsilon_D$ for soft default:
$$p^D=\sigma\!\left(-\widetilde V/\epsilon_D\right).$$

- Initialize: $\epsilon_D \leftarrow \epsilon_{D,0}$
- Update once per **outer iteration** (after critic+actor blocks):
  - $\epsilon_D \leftarrow \max(\epsilon_{D,\min},\ d\,\epsilon_D)$ with $d\in(0,1)$
- Outer iteration = sample batch $\to$ critic/price updates $\to$ actor update $\to$ (optional replay refresh)
- Typical stable defaults (placeholders):
  - $\epsilon_{D,0}\in[10^{-1}, 5\times 10^{-1}]$
  - $\epsilon_{D,\min}\in[10^{-4}, 10^{-3}]$
  - $d\in[0.97,0.995]$ if few outer iterations; if many, use $d\in[0.995,0.999]$
- Numerical stability: clip logit $u=-\widetilde V/\epsilon_D$ to $u\in[-u_{\max},u_{\max}]$ before applying $\sigma(\cdot)$.

### Gradient-flow Rules (TF Implementation Intent)
Goal: correct gradients + avoid accidental detachment.

#### Critic / Price block (policy fixed)
- Objective: $\widehat{\mathcal{L}}_{\text{critic}}=\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}+\lambda_1\widehat{\mathcal{L}}^{\text{price}}$
- Update: $\theta_{\text{value}}$, $\theta_{\text{price}}$
- Stop-gradient rule:
  - Compute $(k',b')=\Gamma_{\text{policy}}(\cdot;\theta_{\text{policy}})$, then treat them as constants inside critic/price losses:
    - use $\mathrm{stopgrad}(k'), \mathrm{stopgrad}(b')$
  - Ensures $\nabla_{\theta_{\text{policy}}}\widehat{\mathcal{L}}_{\text{critic}}=0$

#### Actor block (value and price parameters fixed)
- Objective: $\widehat{\mathcal{L}}_{\text{actor}}=\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}+\lambda_2\widehat{\mathcal{L}}^{\text{price}}$
- Update: $\theta_{\text{policy}}$
- TF rule (very important):
  - Freeze parameters $\theta_{\text{value}},\theta_{\text{price}}$ (no optimizer updates to them),
  - but do **not** apply `tf.stop_gradient` to the outputs $\Gamma_{\text{value}}(\cdot)$ or $\Gamma_{\text{price}}(\cdot)$ with respect to $(k',b')$.
  - Gradients must flow from losses through $(k',b')$ back to $\theta_{\text{policy}}$.

### Limited Liability Baseline
- Use strict limited liability in continuation terms: $V=\max\{0,\widetilde V\}$
- Implement as $V=\mathrm{relu}(\widetilde V)$ (TensorFlow: `tf.nn.relu`) to make subgradient behavior explicit.

---

# 1. Basic Model of Optimal Investment (Sec. 3.1)

## 1.1 Model and Primitives
- State: $(k,z)$
- Action: $k'$

### Investment
$$I = k' - (1 - \delta)k$$

### Payout
$$e(k,k',z) = \pi(k,z) - \psi(I,k) - I$$

where profit function is
$$\pi(k,z) = z \cdot k^{\gamma} , \quad \gamma \in (0,1)$$

and capital adjustment cost function is
$$\psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbb{1}_{I\neq0}$$

### Bellman
$$V(k,z) = \max_{k'>0}\{e(k,k',z)+\beta\mathbb{E}[V(k',z')\mid z]\},\qquad \beta=\frac{1}{1+r}$$

---

## 1.2 Method LR (Lifetime Reward)

### Objective
Simulate horizon $T$ under $\Gamma_{\text{policy}}$:
$$k_{t+1}=\Gamma_{\text{policy}}(k_t,z_t;\theta_{\text{policy}})$$
$$J(\theta_{\text{policy}})=\mathbb{E}\sum_{t=0}^{T-1}\beta^t e(k_t,k_{t+1},z_t),\qquad \mathcal{L}_{\text{LR}}=-J$$

### Empirical loss
With $n$ rollouts:
$$\widehat{\mathcal{L}}^{\text{LR}}=-\frac{1}{n}\sum_{i=1}^n\sum_{t=0}^{T-1}\beta^t e(k_t^{(i)},k_{t+1}^{(i)},z_t^{(i)})$$

### Training outline
1. Sample $n$ initial $(k_0,z_0)\sim\mu$
2. Simulate $T$ steps of $k$ using $\Gamma_{\text{policy}}$
3. Simulate $T$ steps of $z$ using AR(1) with $(\rho,\sigma)$
4. Compute $\widehat{\mathcal{L}}^{\text{LR}}$ and update $\theta_{\text{policy}}$

---

## 1.3 Method ER (Euler Residual, AiO)

### Definitions
Let $F_x$ denote the partial derivative of function $F$ with respect to $x$.

Define $\chi(k,z)=1+\psi_I(I,k)$ with $I=k'-(1-\delta)k$ and $k'=\Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$.

Euler equation is derived from the first order conditions w.r.t $I_t$, $I_{t+1}$, $k_t$, $k_{t+1}$:
$$
\mathbb{E}_{z_{t+1}|z_t}\!\left[\beta\Big(
\pi_k(z_{t+1},k_{t+1})-\psi_k(I_{t+1},k_{t+1})+(1-\delta)\big(1+\psi_I(I_{t+1},k_{t+1})\big)
\Big)\right]=1+\psi_I(I_t,k_t).
$$

For a draw $z'$:
$$k''=\Gamma_{\text{policy}}(k',z';\theta_{\text{policy}}),\quad I'=k''-(1-\delta)k'$$
$$m(k',z')=\pi_k(k',z')-\psi_k(I',k')+(1-\delta)\chi(k',z')$$

Euler residual:
$$f(k,z)=\chi(k,z)-\beta\mathbb{E}[m(k',z')\mid z]$$

### AiO loss
$$\mathcal{L}^{\text{ER}}(\theta_{\text{policy}})=\mathbb{E}[f_1 f_2]$$

Empirical:
$$f_{i,1}=\chi(k,z)-\beta\,m(k',z')\big|_{\varepsilon_1},\qquad
\widehat{\mathcal{L}}^{\text{ER}}=\frac{1}{n}\sum_{i=1}^n f_{i,1}f_{i,2}$$

### Training outline
1. Sample $n$ initial $(k,z)\sim\mu$
2. Compute $k'=\Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$
3. Simulate $z'$ using two draws $(\varepsilon_1,\varepsilon_2)$: $\ln z'=\rho\ln z+\sigma\varepsilon$
4. Compute $\widehat{\mathcal{L}}^{\text{ER}}$ and update $\theta_{\text{policy}}$

### ER applicability warning
If $\psi(I,k)$ includes a fixed cost term that creates a kink at $I=0$ (so $\partial\psi/\partial I$ is undefined at $I=0$), autodiff may return unstable/meaningless gradients and ER may be invalid or numerically fragile.
- Coding note: if fixed cost coefficient $\neq 0$, print a warning and disable ER training (or require smoothing before using ER).

---

## 1.4 Method BR (Actor–Critic Bellman)

### Bellman residual definition
$$V(k,z) - \max_{k'}\{e(k,k',z)+\beta\mathbb{E}_{z'}[V(k',z')]\}$$

We use actor–critic:
1) given policy $\theta_{\text{policy}}$, update $\theta_{\text{value}}$ to fit the Bellman equation;
2) given value $\theta_{\text{value}}$, update $\theta_{\text{policy}}$ to maximize the RHS.

### Critic target (two draws, AiO)
$$y_{i,\ell}=e(k_i,k'_i,z_i)+\beta\,\Gamma_{\text{value}}(k'_i,z'_{i,\ell};\theta_{\text{value}}),\qquad \ell\in\{1,2\}$$

**Critic target detachment rule (single source of truth):**
- Compute $\widehat V^{\text{next}}_{i,\ell}=\Gamma_{\text{value}}(k'_i,z'_{i,\ell};\theta_{\text{value}})$,
- then set $\widehat V^{\text{next}}_{i,\ell}\leftarrow \mathrm{stopgrad}(\widehat V^{\text{next}}_{i,\ell})$,
- and form $y_{i,\ell}=e(\cdot)+\beta\,\widehat V^{\text{next}}_{i,\ell}$.
Only the RHS continuation term is detached; the LHS $\Gamma_{\text{value}}(k_i,z_i;\theta_{\text{value}})$ remains trainable.

### Residual and losses
$$\delta_{i,\ell}=\Gamma_{\text{value}}(k_i,z_i;\theta_{\text{value}})-y_{i,\ell}$$

$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}(\theta_{\text{value}})=\frac{1}{n}\sum_{i=1}^n \delta_{i,1}\delta_{i,2}$$

Actor loss (expectation; use variance reduction by averaging two draws):
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})
=-\frac{1}{n}\sum_{i=1}^n\left[e(k_i,k'_i,z_i)+\beta\cdot \frac{1}{2}\sum_{\ell=1}^2 \Gamma_{\text{value}}(k'_i,z'_{i,\ell};\theta_{\text{value}})\right].$$

- Using one draw is unbiased but noisier; averaging two draws reduces variance and remains correct.
- Do not use the cross-product trick for actor: the product form is for squared residual objectives (AiO), not for maximizing an expectation.

### Training outline
1. Sample batch $(k,z)$ and two shocks per sample
2. Compute $k'=\Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$ and $z'_{1},z'_2$
3. Critic update: compute $\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}$ with detached RHS target, update $\theta_{\text{value}}$ (policy fixed)
4. Actor update: compute $\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}$ using averaged continuation values, update $\theta_{\text{policy}}$ (value fixed)

Optional stability upgrade:
- Target-network option: maintain a slowly-updated target copy $\theta_{\text{value}}^{-}$ and compute the continuation term using $\Gamma_{\text{value}}(\cdot;\theta_{\text{value}}^{-})$ (always detached), while updating $\theta_{\text{value}}$ on the current value.

---

# 2. Risky Debt Model (Sec. 3.6)

## 2.1 Model and Primitives
- State: $(k,b,z)$ with $b\ge 0$
- Control: $(k',b')$ with $b'\ge 0$

### Cash flow
$$e=(1-\tau)\pi(k,z)-\psi(I,k)-I+\frac{b'}{1+\tilde r}+\frac{\tau\tilde r\,b'}{(1+\tilde r)(1+r)}-b$$

### External financing cost
$$\eta(e)=(\eta_0+\eta_1|e|)\mathbf{1}_{e<0}$$

### Risky rate
$$\tilde r=\tilde r(z,k',b')$$

### Latent and actual value
$$\widetilde V(k,b,z)=\max_{k',b'}\left\{e(k,b,z;\tilde r)-\eta(e)+\beta\mathbb{E}_{z'}[V(k',b',z')]\right\}$$
$$V(k',b',z')=\max\{0,\widetilde V(k',b',z')\}=\mathrm{relu}(\widetilde V(k',b',z')).$$

---

## 2.2 Risky Debt Pricing

### Lender zero-profit condition
$$ b'(1+r)= (1+\tilde r)b'\,\mathbb{E}_{z'|z}[1-D]+\mathbb{E}_{z'|z}[D\cdot R(k',b',z')] $$

Recovery:
$$R(k',z')=(1-\alpha)\left[(1-\tau)\pi(k',z')+(1-\delta)k'\right],\quad \alpha\in[0,1].$$

Default rule:
$$D(z',k',b')=\mathbb{1}\{\widetilde V(z',k',b')<0\}.$$

### Smooth default probability (used inside pricing loss only)
$$p^D(k',b',z')=\sigma\!\left(-\frac{\widetilde V(k',b',z')}{\epsilon_D}\right),\qquad
p^D\to D \text{ as } \epsilon_D\to 0^+.$$

### Pricing residual (AiO form)
Given $(z'_{1},z'_{2})$ and policy outputs $(k',b')$, compute $p^D_{i,\ell}$ and $R(k'_i,z'_{i,\ell})$, then
$$f^{\text{price}}_{i,\ell}=b'_i(1+r)-\Big[p^D_{i,\ell}R(k'_i,z'_{i,\ell})+(1-p^D_{i,\ell})b'_i(1+\tilde r_i)\Big].$$

Price loss:
$$\widehat{\mathcal{L}}^{\text{price}}=\frac{1}{n}\sum_{i=1}^n f^{\text{price}}_{i,1}f^{\text{price}}_{i,2}.$$

Parametric pricing function:
$$\tilde r=\Gamma_{\text{price}}(z,k',b';\theta_{\text{price}}).$$

---

## 2.3 Lifetime Reward Maximization (Risky Debt)

### Rollout
Simulate horizon $T$:
$$(k_{t+1},b_{t+1})=\Gamma_{\text{policy}}(k_t,b_t,z_t;\theta_{\text{policy}}).$$

### Per-period reward
$$u_t=e(k_t,k_{t+1},b_t,b_{t+1},z_t)-\eta(e(\cdot)).$$

### Losses
$$\mathcal{L}^{\text{LR}}=-\mathbb{E}\left[\sum_{t=0}^{T-1}\beta^t u_t\right]+\lambda_{\text{price}}\mathcal{L}^{\text{price}}$$
$$\widehat{\mathcal{L}}^{\text{LR}}=-\frac{1}{n}\sum_{i=1}^n\sum_{t=0}^{T-1}\beta^t u_t^{(i)}+\lambda_{\text{price}}\widehat{\mathcal{L}}^{\text{price}}.$$

### Training outline
- Update $\theta_{\text{policy}}$ using $\widehat{\mathcal{L}}^{\text{LR}}$
- Update $\theta_{\text{price}}$ using **only** $\widehat{\mathcal{L}}^{\text{price}}$ (pricing equilibrium), not the discounted lifetime reward term

### LR rollout cash flow uses current pricing
During LR simulation, compute per-period cash flow using
$$\tilde r_t=\Gamma_{\text{price}}(z_t,k'_t,b'_t;\theta_{\text{price}}).$$
Default smoothing affects LR only indirectly through how it shapes the trained $\Gamma_{\text{price}}$ via $\widehat{\mathcal{L}}^{\text{price}}$.

---

## 2.4 Bellman Residual Minimization (Risky Debt)

### Critic target (latent value; two draws)
$$y^{\text{lat}}_{i,\ell}=e(k_i,k'_i,b_i,b'_i,z_i)-\eta(e)+\beta\cdot \max\{0,\Gamma_{\text{value}}(k'_i,b'_i,z'_{i,\ell};\theta_{\text{value}})\}.$$

### Critic target detachment rule
In critic update, compute
$$V^{\text{cont}}_{i,\ell}=\max\{0,\Gamma_{\text{value}}(k'_i,b'_i,z'_{i,\ell};\theta_{\text{value}})\},$$
then detach $V^{\text{cont}}_{i,\ell}$ and form
$$y^{\text{lat}}_{i,\ell}=e(\cdot)-\eta(e)+\beta V^{\text{cont}}_{i,\ell}.$$
Only the RHS continuation term is detached.

### Residual and losses
$$\delta^{\text{lat}}_{i,\ell}=\Gamma_{\text{value}}(k_i,b_i,z_i;\theta_{\text{value}})-y^{\text{lat}}_{i,\ell}$$
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}=\frac{1}{n}\sum_{i=1}^n \delta^{\text{lat}}_{i,1}\delta^{\text{lat}}_{i,2}.$$

Actor loss (use average over two draws for variance reduction):
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})
=-\frac{1}{n}\sum_{i=1}^n\left[e(\cdot)-\eta(e)+\beta\cdot \frac{1}{2}\sum_{\ell=1}^2 \max\{0,\Gamma_{\text{value}}(k'_i,b'_i,z'_{i,\ell};\theta_{\text{value}})\}\right].$$

---

## 2.5 Risky Debt Actor–Critic Training Algorithm (Outer Loop)

### Objectives
- Critic objective: $\widehat{\mathcal{L}}_{\text{critic}}=\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}+\lambda_1\widehat{\mathcal{L}}^{\text{price}}$
- Actor objective: $\widehat{\mathcal{L}}_{\text{actor}}=\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}+\lambda_2\widehat{\mathcal{L}}^{\text{price}}$

### Outer loop (repeat for iterations)
1. Sample batch $(k_i,b_i,z_i)\sim\mu_t$, draw two shocks per state
2. Forward pass:
   - $(k'_i,b'_i)=\Gamma_{\text{policy}}(k_i,b_i,z_i;\theta_{\text{policy}})$
   - $\tilde r_i=\Gamma_{\text{price}}(z_i,k'_i,b'_i;\theta_{\text{price}})$
   - Compute rewards $e(\cdot)-\eta(e)$ using $\tilde r_i$

#### Critic / price update block (policy held fixed)
3. Compute $\widehat{\mathcal{L}}_{\text{critic}}$
4. Update $\theta_{\text{value}}$ and $\theta_{\text{price}}$ to minimize $\widehat{\mathcal{L}}_{\text{critic}}$
   - Stop-gradient: treat $(k'_i,b'_i)$ as constants so $\nabla_{\theta_{\text{policy}}}\widehat{\mathcal{L}}_{\text{critic}}=0$
   - Often use $N_{\text{critic}}>1$ critic steps per actor step

#### Actor update block (value and price parameters fixed)
5. Compute $\widehat{\mathcal{L}}_{\text{actor}}$
6. Update $\theta_{\text{policy}}$ to minimize $\widehat{\mathcal{L}}_{\text{actor}}$
   - Freeze parameters $\theta_{\text{value}},\theta_{\text{price}}$ (no optimizer updates to them)
   - Do not detach $\Gamma_{\text{value}}$ or $\Gamma_{\text{price}}$ outputs w.r.t. $(k',b')$

#### Replay refresh (periodic)
7. Every $M$ iterations: refresh replay buffer by simulating trajectories under current $\Gamma_{\text{policy}}$
8. Sample future batches from mixture of replay + box exploration
9. Oversample near default boundary using bottom-$q$ percentile of $|\widetilde V|$ on replay buffer