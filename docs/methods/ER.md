# Euler Residual Minimization (ER)

## 1. Introduction and Motivation

The ER method minimizes violations of the first-order conditions (Euler
equations) that characterize optimality.  Rather than simulating full
trajectories, it enforces an intertemporal necessary condition between
$(s, a)$ and $(s', a')$ at each observation independently.

**Euler equation.**  At the optimum, the policy $\pi_\theta$ satisfies:

$$\mathbb{E}_\varepsilon \left[F(s, \pi_\theta(s), s', \pi_\theta(s'))\right] = 0$$

where $F: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the Euler residual function derived analytically from the first-order conditions of the Bellman equation, and $s' = f(s, \pi_\theta(s), \varepsilon)$.

ER is a one-step method: each observation is a single-step transition
$(s, z, z'_{\text{main}}, z'_{\text{fork}})$, and the loss is computed
independently per observation with no trajectory unrolling.  This makes ER
computationally cheaper per step than LR, at the cost of requiring an
analytically derived Euler residual function from the environment.

**Target network.**  Both Maliar et al. (2021) and Fernández-Villaverde
et al. (2026) suggest a single policy network inside the loss function.
However, computing $\pi_\theta(s')$ introduces a recursive dependency:
the gradient of $\theta$ flows through both the current policy
$\pi_\theta(s)$ and the next-period policy $\pi_\theta(s')$, creating a
moving-target problem that prevents convergence.  Our implementation
introduces a separate target network $\pi_{\theta^-}$ for the next-period
action, updated via Polyak averaging.

## 2. Objects and Networks

### State decomposition

The state $s = [k,\, z]$ comprises endogenous components $k$ and exogenous
components $z$.  The environment provides:

| Operation | Signature | Description |
|-----------|-----------|-------------|
| Merge | `merge_state(k, z) → s` | Concatenate endogenous and exogenous into full state |
| Endogenous transition | `endogenous_transition(k, a, z) → k'` | Differentiable; on the computation graph |
| Euler residual | `euler_residual(s, a, s', a') → F` | Model-specific FOC residual |

Exogenous transitions are pre-simulated offline.  The trainer receives
pre-computed $z'_{\text{main}}$ and $z'_{\text{fork}}$ for each
observation.

### Networks

| Network | Notation | Role |
|---------|----------|------|
| Policy (current) | $\pi_\theta$ | Maps full state $s$ to action $a$; gradients flow through this network |
| Target policy | $\pi_{\theta^-}$ | Polyak-averaged copy of $\pi_\theta$; evaluates next-period action $a'$ with no gradient |

Both networks share the same architecture: StaticNormalizer → [Dense($n$) +
SiLU] $\times$ $L$ → Affine + clip output head.  Default: $L = 2$,
$n = 128$.

**Input normalization.**  Both networks use a StaticNormalizer fitted once
from the full training dataset before gradient steps.  The target network
gets its own copy of normalizer statistics (same values, separate
variables).

## 3. Algorithm

### Notation summary

| Symbol | Meaning |
|--------|---------|
| $s = [k,\, z]$ | State: endogenous $k$ + exogenous $z$ |
| $a = \pi_\theta(s)$ | Action from current policy |
| $a' = \pi_{\theta^-}(s')$ | Next-period action from target policy |
| $F(s, a, s', a')$ | Euler residual function (model-specific) |
| $\gamma$ | Discount factor |
| $B$ | Mini-batch size |
| $\tau_{\text{polyak}}$ | Polyak averaging rate for target network |
| $n_{\text{steps}}$ | Total training steps |

### Loss function

The loss uses the AiO cross-product estimator with two independent shock
draws ($z'_{\text{main}}$ and $z'_{\text{fork}}$) for unbiased estimation
of the squared expectation:

$$\mathcal{L}_\theta = \frac{1}{B}\sum_{i=1}^{B} F(s_i, a_i, s'_{i,1}, a'_{i,1}) \cdot F(s_i, a_i, s'_{i,2}, a'_{i,2})$$

where:
- $a_i = \pi_\theta(s_i)$ — current action from the trainable policy
- $s'_{i,m} = \text{merge}(f_{\text{endo}}(k_i, a_i, z_i),\; z'_{i,m})$ — next state under shock draw $m$
- $a'_{i,m} = \pi_{\theta^-}(s'_{i,m})$ — next action from the target policy (no gradient)

An MSE alternative is available (`loss_type="mse"`):
$\mathcal{L}_\theta = \frac{1}{B}\sum_{i} F(s_i, a_i, s'_{i,1}, a'_{i,1})^2$.

### Initialization

1. Initialize $\pi_\theta$ with random weights.
2. Create target network: $\pi_{\theta^-} \leftarrow \text{copy}(\pi_\theta)$.
3. Fit StaticNormalizer on the full training dataset (both networks get
   identical statistics, separate copies).
4. No warm-start, no critic.

### Training loop

The training dataset provides $N$ flattened observations.  A
`build_iterator` wraps this into a shuffled, repeating mini-batch iterator
that yields batches of $B$ observations.

**For each step** $j = 0, 1, \ldots, n_{\text{steps}} - 1$:

**(a) Gradient step** (inside `GradientTape`):

$$
\begin{aligned}
&s = \text{merge}(k,\, z) \\
&a = \pi_\theta(s) \\
&k' = f_{\text{endo}}(k,\, a,\, z) \\
&s'_{\text{main}} = \text{merge}(k',\, z'_{\text{main}}), \quad
 s'_{\text{fork}} = \text{merge}(k',\, z'_{\text{fork}}) \\
&a'_{\text{main}} = \pi_{\theta^-}(s'_{\text{main}}), \quad
 a'_{\text{fork}} = \pi_{\theta^-}(s'_{\text{fork}}) \\
&F_1 = F(s,\, a,\, s'_{\text{main}},\, a'_{\text{main}}), \quad
 F_2 = F(s,\, a,\, s'_{\text{fork}},\, a'_{\text{fork}}) \\
&\mathcal{L}_\theta = \text{mean}(F_1 \cdot F_2)
\end{aligned}
$$

Compute $\partial \mathcal{L}_\theta / \partial \theta$ via the tape and
update $\pi_\theta$ with one Adam step.

**(b) Polyak update** target network:

$$\theta^- \leftarrow \tau_{\text{polyak}} \cdot \theta^- + (1 - \tau_{\text{polyak}}) \cdot \theta$$

### Step counting

| Event | Trigger |
|-------|---------|
| One policy gradient update | Every step |
| One Polyak update | Every step |
| Draw next mini-batch | Every step |

## 4. Code Implementation Details

### Why a target network is necessary: the moving-target problem

The Euler equation equates marginal cost and marginal benefit of the
agent's action across two adjacent periods.  For example, in the
investment model, the Euler residual has the structure:

$$F = 1 - \beta \cdot \frac{m(\pi_\theta(s'))}{{\chi(\pi_\theta(s))}}$$

where $\chi$ is the marginal cost of investment today (depends on the
current action $a = \pi_\theta(s)$) and $m$ is the marginal benefit of
capital tomorrow (depends on the next-period action $a' = \pi_\theta(s')$).
At optimum, $F = 0$: marginal cost equals discounted marginal benefit.

**The problem with a single network.**  When a single network
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
step.  In practice, this creates oscillatory or divergent dynamics:
the optimizer cannot close the gap because every step that adjusts the
LHS also shifts the RHS by a similar magnitude.

This is not unique to investment models.  In any Euler equation
$\text{MC}(a) = \beta \cdot \mathbb{E}[\text{MB}(a')]$, the marginal
quantities on both sides are evaluated under the *same* policy.  A
parameter change that reduces the residual at the current $\theta$ does not
guarantee a smaller residual at $\theta + \Delta\theta$, because the
next-period side has shifted.

**How the target network resolves this.**  The target network
$\pi_{\theta^-}$ provides a fixed reference for the next-period action:

- $a = \pi_\theta(s)$ — gradients flow through the current policy.
- $a' = \pi_{\theta^-}(s')$ — **no gradients**; weights are frozen for
  this step.

Now the marginal benefit $m$ is computed from $\pi_{\theta^-}$, which
moves only by $O(1 - \tau_{\text{polyak}})$ per step via Polyak averaging.
The optimizer sees a near-stationary target: it adjusts the current-period
action to match the slowly-moving next-period reference, and the reference
gradually tracks the improving policy.  This converts the unstable
simultaneous update into a stable fixed-point iteration.

### Endogenous transition on the computation graph

The call `k' = env.endogenous_transition(k, a, z)` is inside the tape.
Since the current action $a = \pi_\theta(s)$ depends on $\theta$, the
endogenous next state $k'$ is also on the computation graph.  This means
the Euler residual receives gradient contributions through both the direct
action path ($a \to F$) and the indirect transition path
($a \to k' \to s' \to F$).

### No trajectory unrolling

Unlike LR, ER processes each observation independently — there is no
sequential state evolution within a batch.  Each $(k, z, z'_{\text{main}},
z'_{\text{fork}})$ tuple is a self-contained one-step transition.  This
means:

1. No BPTT, no vanishing/exploding gradients across time steps.
2. Observations are truly i.i.d. within a mini-batch.
3. The gradient per step is cheaper (one forward + backward pass through
   the policy, not $T$ passes).

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
| Learning rate | $\alpha_\theta$ | $10^{-3}$ | Adam with clipnorm=100 |
| Polyak rate | $\tau_{\text{polyak}}$ | 0.995 | Controls target network lag |
| Loss type | | `crossprod` | AiO estimator; alternative: `mse` |
| Hidden layers | $L$ | 2 | |
| Hidden neurons | $n$ | 128 | Per layer |
| Activation | | SiLU | |
| Temperature | | $10^{-6}$ | Smooth gate for non-differentiable reward components |

## 6. Usage Reference

### Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `env` | `MDPEnvironment` | Must implement `merge_state`, `endogenous_transition`, `euler_residual`, `discount`. |
| `policy` | `PolicyNetwork` | Initialized (forward pass called once before training). |
| `train_dataset` | `dict` | Flattened format: `s_endo`, `z`, `z_next_main`, `z_next_fork`. |
| `val_dataset` | `dict` or `None` | Same flattened format.  Used for Euler residual evaluation only. |
| `config` | `ERConfig` | See Section 5 for defaults. |

The target network is created internally by the trainer.

### Output

Returns a dict with keys `policy` (updated in-place), `history`, and
`config`.  The `history` dict contains per-evaluation-step lists: `step`,
`loss`, `euler_residual`.

### Minimal example

```python
config = ERConfig(n_steps=3000, eval_interval=500)

result = train_er(env, policy,
                  train_flat, val_flat, config=config)
```

The primary convergence signal is `euler_residual`.  `loss` is the AiO
cross-product of Euler residuals (should converge toward zero).
