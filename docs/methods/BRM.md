# Multitask Bellman Residual Minimization (BRM)

## 1. Introduction and Motivation

BRM jointly trains a policy network $\pi_\theta$ and a value function
network $V_\phi$ to satisfy the Bellman equation.  Rather than solving the
inner maximization of the Bellman equation directly (which requires an
actor-critic method), BRM eliminates the $\max$ by enforcing necessary
optimality conditions as auxiliary losses.  The total loss combines two
components:

- **$\mathcal{L}_{\text{BR}}$** (Bellman residual): trains $V_\phi$ so
  that $V(s) \approx r(s, a) + \gamma \mathbb{E}[V(s')]$.
- **$\mathcal{L}_{\text{FOC}}$** (first-order condition): trains
  $\pi_\theta$ so that $\nabla_a r + \gamma (\nabla_a f)^\top \nabla_{s'} V = 0$.

Both losses use the AiO cross-product estimator with two independent shock
draws for unbiased squared-expectation estimation.

**Why two losses?**  The Bellman residual alone can be minimized for *any*
policy — it measures whether $V_\phi$ is consistent with $\pi_\theta$, not
whether $\pi_\theta$ is optimal.  The FOC residual provides the optimality
signal: it drives $\pi_\theta$ toward the action that maximizes the
Bellman RHS.  Without the FOC, the optimizer can cheaply minimize
$\mathcal{L}_{\text{BR}}$ by learning a value function for a suboptimal
policy.

## 2. Objects and Networks

### State decomposition

The state $s = [k,\, z]$ comprises endogenous components $k$ and exogenous
components $z$.  The environment provides:

| Operation | Signature | Description |
|-----------|-----------|-------------|
| Merge | `merge_state(k, z) → s` | Concatenate endogenous and exogenous into full state |
| Split | `split_state(s) → (k, z)` | Extract endogenous and exogenous from full state |
| Endogenous transition | `endogenous_transition(k, a, z) → k'` | Differentiable; on the computation graph |
| Reward | `reward(s, a) → r` | Differentiable; scalar per sample |
| Terminal value | `terminal_value(k) → V_term` | Used for warm-start targets |

### Networks

| Network | Notation | Output head | Role |
|---------|----------|-------------|------|
| Policy | $\pi_\theta$ | Affine + clip | Maps full state $s$ to action $a$ |
| Value (critic) | $V_\phi$ | Dense(1), linear | Maps full state $s$ to scalar value |

Default architecture: $L = 2$ hidden layers, $n = 128$ neurons per layer,
SiLU activation after each hidden layer.

**Input normalization.**  Both networks use a StaticNormalizer fitted once
from the full training dataset before gradient steps.

## 3. Algorithm

### Notation summary

| Symbol | Meaning |
|--------|---------|
| $s = [k,\, z]$ | State: endogenous $k$ + exogenous $z$ |
| $a = \pi_\theta(s)$ | Action from current policy |
| $V_\phi(s)$ | Value network (critic) |
| $\gamma$ | Discount factor |
| $B$ | Mini-batch size |
| $w_{\text{foc}}$ | FOC loss weight |
| $\lambda_{\text{br}}$ | Bellman residual normalizer (scale factor) |
| $n_{\text{steps}}$ | Total training steps |

### Loss functions

**Bellman residual** ($\mathcal{L}_{\text{BR}}$, trains $V_\phi$):

For each observation $i$ and shock draw $m \in \{1, 2\}$:

$$F^{\text{BR}}_{i,m} = \frac{V_\phi(s_i) - r(s_i, a_i) - \gamma \, V_\phi(s'_{i,m})}{\lambda_{\text{br}}}$$

where $a_i = \pi_\theta(s_i)$ and $s'_{i,m} = \text{merge}(f_{\text{endo}}(k_i, a_i, z_i),\, z'_{i,m})$.  The division by $\lambda_{\text{br}}$ normalizes the residual scale (set to $|V^*|$ by default).

$$\mathcal{L}_{\text{BR}} = \frac{1}{B}\sum_{i=1}^{B} F^{\text{BR}}_{i,1} \cdot F^{\text{BR}}_{i,2}$$

**FOC residual** ($\mathcal{L}_{\text{FOC}}$, trains $\pi_\theta$):

The first-order condition is computed via autodiff:

$$F^{\text{FOC}}_{i,m} = \nabla_a r(s_i, a)\big|_{a = a_i} + \gamma \, (\nabla_a s'_{i,m})^\top \nabla_{s'} V_\phi(s'_{i,m})$$

The gradient $\nabla_{s'} V_\phi$ is computed by an inner `GradientTape`,
and the VJP $(\nabla_a s')^\top \nabla_{s'} V$ is computed by a second
tape that watches $a$ and backpropagates through
$s' = \text{merge}(f_{\text{endo}}(k, a, z),\, z')$.

$$\mathcal{L}_{\text{FOC}} = \frac{1}{B}\sum_{i=1}^{B} \sum_{d=1}^{N_a} F^{\text{FOC},d}_{i,1} \cdot F^{\text{FOC},d}_{i,2}$$

where the sum over $d$ accounts for multi-dimensional actions.

**Total loss:**

$$\mathcal{L} = \mathcal{L}_{\text{BR}} + w_{\text{foc}} \cdot \mathcal{L}_{\text{FOC}}$$

### Warm-start: preventing the cold-start failure mode

Before gradient training begins, the value network $V_\phi$ is pre-trained
on analytical terminal-value targets:

$$V_\phi(s) \approx \hat{V}^{\text{term}}(k) = \frac{r(\bar{s},\, \bar{a})}{1 - \gamma}$$

for `warm_start_steps` MSE regression steps (default 200).  This section
explains why warm-start is practically a prerequisite rather than an
optional refinement.

**Default initialization and the FOC.**  TensorFlow's Dense layers
initialize weights with Glorot uniform (zero-mean, scale
$\sim 1/\sqrt{\text{fan}}$) and biases with zeros.  With two hidden
layers of 128 neurons and SiLU activations, the initialized
$V_\phi$ outputs values close to zero with near-zero gradients:
$\partial V_\phi / \partial k \approx 0$ across the state space.

The FOC residual that trains $\pi_\theta$ is:

$$F = \underbrace{\frac{\partial r}{\partial a}}_{\text{marginal cost}} + \;\gamma \frac{\partial k'}{\partial a} \underbrace{\frac{\partial V_\phi}{\partial k'}}_{\approx\, 0 \text{ at init}}$$

When the second term is negligible, the FOC degenerates to
$F \approx \partial r / \partial a$, and the FOC loss
$\mathcal{L}_{\text{FOC}} = \mathbb{E}[F^2]$ drives the policy toward
$\partial r / \partial a = 0$ — the action that maximizes single-period
reward.  For investment models, this means near-zero investment, since
investment is a current-period cost with no perceived future benefit.

**Why the FOC cannot easily self-correct.**  At this point, the FOC
residual $F \approx \partial r / \partial a \approx 0$, so
$\mathcal{L}_{\text{FOC}} \approx 0$.  The loss appears converged — but
this is a **false zero**.  The residual is small not because the
optimality condition is genuinely satisfied ($\partial r / \partial a +
\gamma (\partial k'/\partial a)(\partial V / \partial k') = 0$ with both
terms nonzero and balancing), but because the future-value term has
dropped out entirely.

Can the Bellman residual loss eventually correct $V_\phi$ enough to break
the false zero?  For production-based models, the on-policy value function
$V^\pi$ under *any* feasible policy has $\partial V^\pi / \partial k > 0$.
To see this, consider the myopic policy (near-zero investment) where
$k' \approx (1-\delta)k$:

$$V^\pi(k, z) = \mathbb{E}\!\left[\sum_{t=0}^\infty \gamma^t z_t \cdot ((1-\delta)^t k)^\theta\right] = k^\theta \cdot \underbrace{\sum_{t=0}^\infty \gamma^t (1-\delta)^{t\theta} \mathbb{E}[z_t]}_{\text{positive}}$$

Since $\theta \in (0,1)$: $\partial V^\pi / \partial k = \theta k^{\theta-1}
\cdot (\text{positive sum}) > 0$.  So the BR target is monotone in $k$,
and $V_\phi$ will eventually develop the correct gradient sign.  This
means there is no permanent fixed point at the myopic policy for this
class of models — the system can escape in principle.

However, self-correction faces three compounding obstacles that make it
impractically slow:

1. **False-zero plateau.**  The FOC loss is near zero →
   $\nabla_\theta \mathcal{L}_{\text{FOC}}$ is small → policy updates
   are negligible.  Any correction from the slowly-growing
   $\partial V / \partial k'$ term produces a small perturbation to $F$
   away from zero, but the resulting gradient is proportional to $F$
   itself (from $\nabla_\theta F^2 = 2F \nabla_\theta F$), which is
   small.

2. **Magnitude gap.**  The value $V^\pi$ under the myopic policy has
   a much smaller $\partial V / \partial k$ than $V^*$ under the optimal
   policy, because capital depletes under near-zero investment, shrinking
   the future reward stream.  Even after the BR learns the correct
   gradient sign, the magnitude is too small to generate a FOC correction
   of comparable scale to the $\partial r / \partial a$ term.  The policy
   stays near the myopic optimum, and the BR continues fitting $V_\phi$
   to a near-myopic policy, slowing the growth of
   $\partial V / \partial k'$ toward its correct magnitude.

3. **No target network.**  Without a target value network, the joint
   dynamics — $V_\phi$ chasing $\pi_\theta$ via the BR, $\pi_\theta$
   responding to $V_\phi$ via the FOC — can oscillate rather than
   converge monotonically, further slowing the escape.

In practice, convergence from cold start is so slow that it is
indistinguishable from failure within any reasonable training budget.

**Caveat for general MDPs.**  The $\partial V^\pi / \partial k > 0$
guarantee relies on the production function structure (output increasing
in capital).  For models where the endogenous state includes liabilities
(e.g., debt $b$ where $\partial V / \partial b < 0$), the gradient sign
depends on which state component dominates, and the BR may not develop a
useful signal for all dimensions.  In such models, the cold-start failure
may be a genuine trap rather than a slow transient.

**How warm-start resolves this.**  Pre-training $V_\phi$ on
$r(\bar{s}, \bar{a}) / (1-\gamma)$ produces a value function with two
gradient properties that follow from standard production function
assumptions (diminishing returns, Inada conditions):

1. **$\partial V_\phi / \partial k > 0$**: more capital is valuable.
2. **$\partial^2 V_\phi / \partial k^2 < 0$**: diminishing marginal
   value of capital.

These are sufficient to make the FOC well-behaved from step 1:

- The positive $\partial V_\phi / \partial k'$ provides a
  **counterbalancing force** against the negative $\partial r / \partial a$.
  The agent sees a future benefit to investing, not just the current cost.
  The two terms in the FOC are of comparable magnitude, so the residual
  $F$ reflects a genuine optimality gap rather than a false zero.
- The concavity $\partial^2 V_\phi / \partial k'^2 < 0$ ensures a
  **unique interior solution**: as investment increases, $k'$ increases,
  but $\partial V / \partial k'$ decreases (diminishing returns), so there
  is a well-defined point where marginal cost equals marginal benefit.

This creates a virtuous cycle: reasonable $V_\phi$ → FOC gives meaningful
policy signal → reasonable policy → BR fits accurate $V_\phi$ → cycle
continues.

**No analytical solution required.**  The perpetuity
$r(\bar{s}, \bar{a}) / (1-\gamma)$ uses only model primitives — the
reward function, discount factor, and the stationary action.  It is not
$V^*$; it ignores exogenous volatility and the agent's dynamic response.
The approximation error is $O(\sigma^2)$ (same analysis as the LR
terminal value correction).  What matters is not accuracy but the
**gradient structure**: any approximation with $\partial V / \partial k > 0$
and $\partial^2 V / \partial k^2 < 0$ would serve the same purpose.

**Connection to ER: why Euler residual minimization works without
warm-start.**  The cold-start vulnerability is specific to BRM's autodiff
FOC, not to FOC-based methods in general.  Both BRM's FOC and the ER
method's Euler residual are derived from the same Bellman first-order
condition:

$$\frac{\partial r}{\partial a} + \gamma \, \mathbb{E}\!\left[\frac{\partial V(s')}{\partial s'} \cdot \frac{\partial s'}{\partial a}\right] = 0$$

The two methods differ in how they handle $\nabla_{s'} V$:

- **BRM (autodiff FOC):** retains $\nabla_{s'} V_\phi$ as a learned
  quantity.  The optimality signal is routed through the value network's
  gradient, which starts uninformative.
- **ER (analytical Euler equation):** eliminates $V$ entirely by
  substituting the envelope condition into the Bellman FOC and using the
  model's recursive structure.  The resulting Euler residual —
  e.g., $1 - \beta \cdot m(\pi_\theta(s')) / \chi(\pi_\theta(s))$ in the
  investment model — depends only on $\pi_\theta$ and known model
  derivatives.  There is no learned approximation to be wrong.

Because ER's residual is self-contained, it correctly measures the
optimality gap from step 1 regardless of initialization.  There is no
false-zero mechanism: the marginal cost $\chi$ and marginal benefit $m$
are computed from model primitives (production function, adjustment
costs), not from a learned $V_\phi$, so the residual cannot degenerate
by the future-value term dropping out.

This explains the empirical observation that ER converges reliably from
cold start while BRM requires warm-start.  The trade-off is generality:
ER requires a manually derived, model-specific Euler equation, which may
not exist in closed form for all MDPs.  BRM's autodiff FOC is generic —
it works for any differentiable environment without analytical derivation
— but pays for this generality with the cold-start vulnerability.

### Initialization

1. Initialize $\pi_\theta$ and $V_\phi$ with random weights.
2. Fit StaticNormalizer on the full training dataset (both networks).
3. Warm-start $V_\phi$ on analytical targets (if `warm_start_steps > 0`).
4. No target networks.

### Training loop

The training dataset provides $N$ flattened observations.  A
`build_iterator` wraps this into a shuffled, repeating mini-batch iterator
that yields batches of $B$ observations.

**For each step** $j = 0, 1, \ldots, n_{\text{steps}} - 1$:

**(a) Bellman residual step** (trains $V_\phi$, inside `GradientTape`):

$$
\begin{aligned}
&a = \pi_\theta(s) \quad \text{(no gradient through } \pi_\theta \text{)} \\
&v_s = V_\phi(s) \quad \text{(gradient active)} \\
&r = r(s, a) \\
&k' = f_{\text{endo}}(k, a, z) \\
&s'_{\text{main}} = \text{merge}(k', z'_{\text{main}}), \quad
 s'_{\text{fork}} = \text{merge}(k', z'_{\text{fork}}) \\
&v'_{\text{main}} = V_\phi(s'_{\text{main}}), \quad
 v'_{\text{fork}} = V_\phi(s'_{\text{fork}}) \quad \text{(no gradient)} \\
&F_1 = (v_s - r - \gamma \, v'_{\text{main}}) / \lambda_{\text{br}}, \quad
 F_2 = (v_s - r - \gamma \, v'_{\text{fork}}) / \lambda_{\text{br}} \\
&\mathcal{L}_{\text{BR}} = \text{mean}(F_1 \cdot F_2)
\end{aligned}
$$

Update $V_\phi$ with one Adam step using
$\partial \mathcal{L}_{\text{BR}} / \partial \phi$.

**(b) FOC step** (trains $\pi_\theta$, inside `GradientTape`):

$$
\begin{aligned}
&a = \pi_\theta(s) \\
&k' = f_{\text{endo}}(k, a, z) \\
&s'_{\text{main}} = \text{merge}(k', z'_{\text{main}}), \quad
 s'_{\text{fork}} = \text{merge}(k', z'_{\text{fork}}) \\
&\nabla_{s'} V_{\text{main}} = \nabla_{s'} V_\phi(s'_{\text{main}}), \quad
 \nabla_{s'} V_{\text{fork}} = \nabla_{s'} V_\phi(s'_{\text{fork}}) \\
&F^{\text{FOC}}_1 = \nabla_a r\big|_a + \gamma \cdot (\nabla_a s'_{\text{main}})^\top \nabla_{s'} V_{\text{main}} \\
&F^{\text{FOC}}_2 = \nabla_a r\big|_a + \gamma \cdot (\nabla_a s'_{\text{fork}})^\top \nabla_{s'} V_{\text{fork}} \\
&\mathcal{L}_{\text{FOC}} = \text{mean}\!\left(\sum_d F^{\text{FOC},d}_1 \cdot F^{\text{FOC},d}_2\right)
\end{aligned}
$$

Update $\pi_\theta$ with one Adam step using
$\partial \mathcal{L}_{\text{FOC}} / \partial \theta$.

### Step counting

| Event | Trigger |
|-------|---------|
| One $V_\phi$ gradient update (Bellman residual) | Every step |
| One $\pi_\theta$ gradient update (FOC) | Every step |
| Draw next mini-batch | Every step |

## 4. Code Implementation Details

### Separated gradient tapes for $V_\phi$ and $\pi_\theta$

BRM uses **two separate `GradientTape`** blocks per step — one for the
Bellman residual (updating $V_\phi$ only) and one for the FOC (updating
$\pi_\theta$ only):

1. **$\mathcal{L}_{\text{BR}}$ updates $V_\phi$ only.**  The policy
   $\pi_\theta$ appears in the Bellman residual (to compute $a$, $r$, and
   $s'$), but gradients of $\mathcal{L}_{\text{BR}}$ do not flow into
   $\theta$.  By calling `policy(s, training=False)` inside `tape_v` and
   computing `tape_v.gradient(loss_br, value_net.trainable_variables)`,
   the policy weights are untouched.

2. **$\mathcal{L}_{\text{FOC}}$ updates $\pi_\theta$ only.**  The
   value network $V_\phi$ appears in the FOC (to compute $\nabla_{s'} V$),
   but its weights are frozen for this step.  The tape computes
   `tape_p.gradient(loss_foc, policy.trainable_variables)`.

**Why separation matters.**  The original BRM formulation (Maliar et al.,
2021; Fernández-Villaverde et al., 2026) computes a single joint gradient
on a combined loss:

$$(\theta, \phi) \leftarrow (\theta, \phi) - \eta \cdot \nabla_{(\theta,\phi)} \left[\mathcal{L}_{\text{BR}}(\theta, \phi) + w \cdot \mathcal{L}_{\text{FOC}}(\theta, \phi)\right]$$

This creates a problematic gradient signal for the policy parameters
$\theta$.  The Bellman residual $V_\phi(s) - r(s, \pi_\theta(s)) - \gamma
V_\phi(s')$ depends on $\theta$ through the action $a = \pi_\theta(s)$ and
the next state $s'$.  Its gradient $\nabla_\theta \mathcal{L}_{\text{BR}}$
pushes $\theta$ in the direction that makes $V_\phi$ more consistent with
$\pi_\theta$ — but this gradient carries no optimality information.  The
Bellman equation $V^\pi(s) = r(s, \pi(s)) + \gamma \mathbb{E}[V^\pi(s')]$
holds for *any* policy $\pi$, not just the optimal one.  The gradient
$\nabla_\theta \mathcal{L}_{\text{BR}}$ therefore acts as a parasitic
force on $\theta$: it resists policy changes that would increase the
Bellman residual, even when those changes improve the policy.

Meanwhile, $\nabla_\theta \mathcal{L}_{\text{FOC}}$ provides the actual
optimality signal — it drives $\pi_\theta$ toward the action that
maximizes $r + \gamma V(s')$.  In a joint update, these two forces on
$\theta$ compete at every step.

This competing-gradient mechanism is likely a primary contributor to the
instability of the original formulation, though other factors may also
play a role:

- **Scale mismatch.**  $\mathcal{L}_{\text{BR}}$ is in value-squared
  units, $\mathcal{L}_{\text{FOC}}$ is in derivative units.  A single
  weight $w$ cannot correctly balance these across states, and the
  relative scale shifts as training progresses.
- **Degenerate equilibria.**  The joint objective can be cheaply minimized
  by solutions that are far from optimal.  For example, if $V_\phi$
  becomes nearly flat ($\nabla_s V \approx 0$), then
  $\mathcal{L}_{\text{FOC}} \approx \|\nabla_a r\|^2$, which can be small
  at interior points regardless of whether the policy is optimal.
  Meanwhile, a flat $V$ approximately satisfies $\mathcal{L}_{\text{BR}}$
  for policies that don't move the state much.  The joint optimizer can
  settle into this degenerate basin because it minimizes the total loss.
- **Double-sampling bias.**  $V_\phi$ appears on both sides of the
  Bellman residual without a target network, creating a positive bias.

With separated tapes, each network receives an unambiguous gradient
signal: $V_\phi$ learns to track the current policy (via
$\mathcal{L}_{\text{BR}}$), and $\pi_\theta$ learns to improve given the
current $V_\phi$ (via $\mathcal{L}_{\text{FOC}}$).  No conflicting forces
act on any single set of parameters.  This is analogous to alternating
minimization (Gauss-Seidel) instead of simultaneous minimization (Jacobi):
each step uses the freshest information from the other network.

### The `_autodiff_foc` helper: nested tapes for the FOC

The FOC residual $\nabla_a r + \gamma (\nabla_a s')^\top \nabla_{s'} V$
requires two derivative computations:

1. **$\nabla_a r$:** A `GradientTape` watches $a$ and differentiates
   `env.reward(s, a)` with respect to $a$.

2. **$\gamma (\nabla_a s')^\top \nabla_{s'} V$:** This is a
   vector-Jacobian product (VJP).  A `GradientTape` watches $a$ and
   computes $s' = \text{merge}(f_{\text{endo}}(k, a, z),\, z')$.  The
   `output_gradients` argument passes $\nabla_{s'} V$ as the VJP
   direction, so the tape returns $(\nabla_a s')^\top \nabla_{s'} V$
   directly without materializing the full Jacobian $\nabla_a s'$.

The value gradient $\nabla_{s'} V$ is computed by yet another inner tape
in the main training loop before calling `_autodiff_foc`.

### No target network for $V_\phi$

Unlike SHAC, BRM does not use a target value network.  Both $V_\phi(s)$
(current state) and $V_\phi(s')$ (next state) use the same network.  The
AiO cross-product estimator partially mitigates the double-sampling bias,
and the separate $V_\phi$ / $\pi_\theta$ gradient tapes prevent the
critic from chasing itself within a single step.

The `V_\phi(s')` calls in the Bellman residual tape use
`training=False`, which prevents gradient flow through the next-state
value evaluation — only $V_\phi(s)$ (with `training=True`) contributes
gradients to $\phi$.

### Warm-start implementation

`warm_start_value_net` (from `core.py`) accepts both trajectory and
flattened dataset formats.  It:

1. Fits the normalizer on the dataset (idempotent with later trainer
   fitting).
2. Computes targets: $v_{\text{target}} = \text{env.terminal\_value}(k)$
   for all $k$ in the dataset.
3. Runs Adam MSE regression: $\min_\phi \|V_\phi(s) - v_{\text{target}}\|^2$
   for `warm_start_steps` iterations.

### Data contract

The trainer expects flattened-format data:

| Key | Shape | Description |
|-----|-------|-------------|
| `s_endo` | `(N, endo_dim)` | Endogenous state (i.i.d. uniform draws) |
| `z` | `(N, exo_dim)` | Current exogenous state |
| `z_next_main` | `(N, exo_dim)` | Next exogenous state (main path) |
| `z_next_fork` | `(N, exo_dim)` | Next exogenous state (fork path, for AiO) |

Validation dataset uses the same format.

## 5. Default Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Batch size | $B$ | 256 | |
| Policy learning rate | $\alpha_\theta$ | $10^{-3}$ | Adam with clipnorm=100 |
| Critic learning rate | $\alpha_\phi$ | $10^{-3}$ | Adam with clipnorm=100 |
| FOC weight | $w_{\text{foc}}$ | 1.0 | Relative weight of FOC vs. Bellman residual |
| BR scale | $\lambda_{\text{br}}$ | 1.0 | Set to $|V^*|$ for large-scale problems |
| Loss type | | `crossprod` | AiO estimator; alternative: `mse` |
| Warm-start steps | | 200 | MSE regression on analytical $V^{\text{term}}$; 0 to disable |
| Hidden layers | $L$ | 2 | Shared architecture for $\pi$ and $V$ |
| Hidden neurons | $n$ | 128 | Per layer |
| Activation | | SiLU | |
| Temperature | | $10^{-6}$ | Smooth gate for non-differentiable reward components |

## 6. Differences from Maliar et al. (2021)

### Simplified multitask loss

**Original:** The total loss includes Bellman residual, FOC, envelope
condition, and optional feasibility constraint residuals, each with
separate exogenous weights $w_1, w_2, w_3, w_4$ that must be manually
tuned.

**Ours:** We use only two components — $\mathcal{L}_{\text{BR}}$ and
$\mathcal{L}_{\text{FOC}}$ — with a single weight $w_{\text{foc}}$.

The envelope condition is dropped because it is mathematically redundant
given BR + FOC.  To see this, differentiate the Bellman equation
$V(s) = r(s, \pi(s)) + \gamma \mathbb{E}[V(s')]$ with respect to $s$:

$$\nabla_s V = \frac{\partial r}{\partial s} + \gamma \mathbb{E}\!\left[\nabla_{s'} V \cdot \frac{\partial f}{\partial s}\right] + \nabla_s \pi \cdot \underbrace{\left(\frac{\partial r}{\partial a} + \gamma \mathbb{E}\!\left[\nabla_{s'} V \cdot \frac{\partial f}{\partial a}\right]\right)}_{\text{= 0 when FOC holds}}$$

The underbraced term is exactly the FOC residual.  When both the Bellman
equation (BR = 0) and the first-order condition (FOC = 0) hold, the
envelope condition follows automatically — it is a derived consequence,
not an independent constraint.  Note that the envelope condition is *not*
merely an on-policy consistency condition like the Bellman residual: it
specifically encodes optimality through the FOC.  But since BR + FOC
already imply it, including it as a separate loss adds no information and
only introduces an additional weight to tune.

Feasibility constraints are handled via the environment's reward function
(e.g., soft penalty) rather than separate loss terms.

### Separated optimizer updates

**Original:** A single joint gradient step updates both $\theta$ and
$\phi$ on the combined loss $\mathcal{L}_{\text{BR}} + w \cdot \mathcal{L}_{\text{FOC}}$.

**Ours:** Separate gradient tapes and separate optimizer steps for $V_\phi$
(Bellman residual) and $\pi_\theta$ (FOC).  See Section 4 for a detailed
discussion of why this separation is necessary.

### Warm-start critic

**Original:** Not discussed.

**Ours:** Optional pre-training of $V_\phi$ on analytical terminal-value
targets, providing a reasonable initialization for the critic before
joint training begins.

## 7. Usage Reference

### Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `env` | `MDPEnvironment` | Must implement `merge_state`, `split_state`, `endogenous_transition`, `reward`, `discount`.  Optionally `terminal_value` for warm-start. |
| `policy` | `PolicyNetwork` | Initialized (forward pass called once before training). |
| `value_net` | `StateValueNetwork` | Initialized; same `state_dim` as policy. |
| `train_dataset` | `dict` | Flattened format: `s_endo`, `z`, `z_next_main`, `z_next_fork`. |
| `val_dataset` | `dict` or `None` | Same flattened format.  Used for Euler/Bellman evaluation only. |
| `config` | `BRMConfig` | See Section 5 for defaults. |
| `eval_callback` | callable or `None` | Custom validation metrics.  See [training_infrastructure.md](training_infrastructure.md) §1. |

### Output

See [training_infrastructure.md](training_infrastructure.md) §3 for the
full output schema (history keys, early-stopping metadata, wall time).
BRM records `loss`, `loss_br`, `loss_foc` in history.

### Minimal example

```python
from src.v2.trainers.brm import train_brm
from src.v2.trainers.config import BRMConfig

config = BRMConfig(
    n_steps=3000,
    eval_interval=500,
    br_scale=1.0 / env.reward_scale(),  # normalize BR to O(1)
    warm_start_epochs=1,                # pre-train critic on analytical V
)

result = train_brm(env, policy, value_net,
                   train_flat, val_flat, config=config)
```

**Key config choices:**
- `br_scale` normalizes the Bellman residual to prevent scale mismatch
  with the FOC loss.  Set to `1 / env.reward_scale()` (= |V*|) so that
  the normalized residual is O(1).  Without normalization, the raw BR
  can dominate the FOC and mask policy improvement.
- `warm_start_epochs` pre-trains the critic on the analytical terminal
  value $V^{term}(k) = r(\bar{s}, \bar{a}) / (1-\gamma)$ via MSE
  regression.  This gives the critic $\partial V / \partial k > 0$ so
  that the FOC steers the policy in a meaningful direction from step 0.
  Cold start (0 epochs) risks the critic providing garbage gradients.
- `loss_br` and `loss_foc` should both decrease; if `loss_foc` stalls
  while `loss_br` decreases, the value function is fitting a suboptimal
  policy.
