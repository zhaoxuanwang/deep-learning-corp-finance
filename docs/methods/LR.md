# Lifetime Reward Maximization (LR)

## 1. Introduction and Motivation

LR directly maximizes discounted cumulative rewards by simulating
trajectories under the current policy.  Given initial state $s_0$ and a
shock sequence $\{\varepsilon_1, \ldots, \varepsilon_T\}$, the policy
$\pi_\theta$ generates a trajectory
$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ where $a_t = \pi_\theta(s_t)$
and $s_{t+1} = f(s_t, a_t, \varepsilon_{t+1})$.  Gradients flow backward
through the entire trajectory via backpropagation through time (BPTT),
requiring both the reward $r$ and the endogenous transition $f^{\text{endo}}$
to be differentiable with respect to the action.

LR is the simplest of the deep learning methods: it needs only a policy
network (no critic), uses no target network, and the loss has a direct
economic interpretation — negative expected lifetime reward.

## 2. Objects and Networks

### State decomposition

The state $s = [k,\, z]$ comprises endogenous components $k$ (e.g.,
capital stock) and exogenous components $z$ (e.g., productivity shocks).
The environment provides:

| Operation | Signature | Description |
|-----------|-----------|-------------|
| Merge | `merge_state(k, z) → s` | Concatenate endogenous and exogenous into full state |
| Split | `split_state(s) → (k, z)` | Extract endogenous and exogenous from full state |
| Endogenous transition | `endogenous_transition(k, a, z) → k'` | Differentiable; on the computation graph |
| Reward | `reward(s, a) → r` | Differentiable; scalar per sample |
| Terminal value | `terminal_value(k) → V_term` | Analytical continuation approximation |

Exogenous transitions $z_{t+1} = f_{\text{exo}}(z_t, \varepsilon_{t+1})$
are pre-simulated offline and stored as paths $z_{0:T}$.  The trainer
never calls the exogenous transition directly.

### Network

A single policy network is used:

| Network | Notation | Output head | Role |
|---------|----------|-------------|------|
| Policy | $\pi_\theta$ | Affine + clip | Maps full state $s$ to action $a$ |

Default architecture: $L = 2$ hidden layers, $n = 128$ neurons per layer,
SiLU activation after each hidden layer.

**Policy output head.**  The raw network output is affine-rescaled before
clipping:

$$a = \text{clip}\!\Big(\, c + \sqrt{h_r} \cdot \text{raw},\; a_{\min},\; a_{\max}\Big)$$

where $c = (a_{\min} + a_{\max}) / 2$ is the action center and
$h_r = (a_{\max} - a_{\min}) / 2$ is the half-range.

**Input normalization.**  The policy uses a StaticNormalizer that z-score
normalizes the input state: $\hat{s} = (s - \mu) / \sigma$.  The
statistics $\mu$ and $\sigma$ are fit once from the full training dataset
before any gradient steps, and remain frozen during training.

## 3. Algorithm

### Notation summary

| Symbol | Meaning |
|--------|---------|
| $s = [k,\, z]$ | State: endogenous $k$ + exogenous $z$ |
| $a = \pi_\theta(s)$ | Action from current policy |
| $r(s, a)$ | Per-period reward |
| $\gamma$ | Discount factor |
| $T$ | BPTT rollout horizon |
| $\hat{V}^{\text{term}}(k)$ | Terminal value correction (analytical) |
| $B$ | Batch size: number of parallel trajectories per mini-batch |
| $n_{\text{steps}}$ | Total training steps |

### Truncated BPTT objective

**True objective.**  The infinite-horizon value under policy $\pi_\theta$
starting from $s_0$ is:

$$V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_t, \pi_\theta(s_t))\right]$$

Splitting at a finite horizon $T$:

$$V^{\pi}(s_0) = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_t, \pi_\theta(s_t))\right] + \gamma^T \, \mathbb{E}\left[V^{\pi}(s_T)\right]$$

### Terminal value correction

Rather than increasing $T$ to make $\gamma^T V^{\pi}(s_T) \approx 0$, we
approximate the continuation value using model primitives.  Define:

$$\bar{s} = [s^{\text{endo}} \mid \bar{s}^{\text{exo}}], \qquad \bar{a} = \bar{a}(s^{\text{endo}})$$

where $\bar{s}^{\text{exo}} = \mathbb{E}[s^{\text{exo}}_\infty]$ is the
stationary mean of the exogenous process, and $\bar{a}(s^{\text{endo}})$ is
the action satisfying $f^{\text{endo}}(s^{\text{endo}}, \bar{a}) = s^{\text{endo}}$.
The terminal value is a geometric perpetuity:

$$\hat{V}^{\text{term}}(s_T^{\text{endo}}) = \frac{r(\bar{s},\, \bar{a})}{1 - \gamma}$$

### Loss function

$$\mathcal{L}_\theta = -\frac{1}{B}\sum_{i=1}^{B}\left[\sum_{t=0}^{T-1} \gamma^t \, r(s_{it}, \pi_\theta(s_{it})) \;+\; \gamma^T \, \hat{V}^{\text{term}}(s_{iT}^{\text{endo}})\right]$$

Setting $\hat{V}^{\text{term}} = 0$ recovers the truncated objective of
Maliar et al. (2021).

### Initialization

1. Initialize $\pi_\theta$ with random weights.
2. Fit StaticNormalizer on the full training dataset (computed once, then
   frozen for the rest of training).
3. No target network, no critic, no warm-start.

### Training loop

The training dataset provides $N$ trajectories, each containing an initial
endogenous state $k_0$ and a pre-simulated exogenous path $z_{0:T}$.  A
`build_iterator` wraps this into a shuffled, repeating mini-batch iterator
that yields batches of $B$ trajectories.

**For each mini-batch** (draw $B$ trajectories: initial states $k_0^{1:B}$
and exogenous paths $z_{0:T}^{1:B}$):

Inside a `GradientTape`, unroll $T$ steps on the computation graph.  All
operations are batched over $B$ trajectories in parallel:

$$
\begin{aligned}
&k \leftarrow s\_\text{endo}\_0 \\
&\textbf{for } t = 0, \ldots, T-1: \\
&\qquad s_t = \text{merge}(k,\; z_t) \\
&\qquad a_t = \pi_\theta(s_t) \\
&\qquad r_t = r(s_t,\, a_t) \\
&\qquad \text{total\_reward} \mathrel{+}= \gamma^t \cdot r_t \\
&\qquad k \leftarrow f_{\text{endo}}(k,\, a_t,\, z_t) \\[6pt]
&\textbf{if terminal\_value enabled}: \\
&\qquad v_{\text{term}} = \hat{V}^{\text{term}}(k) \\
&\qquad \text{total\_reward} \mathrel{+}= \gamma^T \cdot v_{\text{term}}
\end{aligned}
$$

Compute $\mathcal{L}_\theta = -\text{mean}(\text{total\_reward})$.  Compute
$\partial \mathcal{L}_\theta / \partial \theta$ via the tape and update
$\pi_\theta$ with one Adam step.

### Annealing

The environment may provide an annealing schedule (e.g., for smooth-gate
temperature).  The trainer stores the temperature in a `tf.Variable` so
that the `@tf.function`-compiled training step reads the updated value at
each iteration.  The schedule is updated after each gradient step.

## 4. Code Implementation Details

### Gradient flow through the trajectory

The entire trajectory — all $T$ calls to `policy()`, `env.reward()`, and
`env.endogenous_transition()` — is inside a single `GradientTape`.  This
means:

1. Gradients flow from $r_t$ backward through $\pi_\theta(s_t)$ and through
   the chain $k_t \to k_{t-1} \to \cdots \to k_0$ via `endogenous_transition`.
2. The policy at step $t$ receives gradient signal from all future rewards
   $r_{t+1}, \ldots, r_{T-1}$ and the terminal value, not just $r_t$.
3. The terminal value $\hat{V}^{\text{term}}(k_T)$ is differentiable with
   respect to $k_T$, so its gradient flows back through the chain.  However,
   the terminal value function itself does not depend on $\theta$ (no policy
   call at step $T$), avoiding $1/(1-\gamma)$ gradient amplification.

The `policy` is called with `training=False` (no dropout or stochastic
layers), which is standard for deterministic policy gradient methods.

### No target network

Unlike ER and BRM, LR does not use a target network.  This is because LR
has no recursive dependency: the loss at step $j$ depends on the current
$\theta_j$ through the trajectory rollout, and there is no next-period
policy evaluation that would create a moving-target problem.

### Data contract

The trainer expects trajectory-format data:

| Key | Shape | Description |
|-----|-------|-------------|
| `s_endo_0` | `(N, endo_dim)` | Initial endogenous state |
| `z_path` | `(N, T+1, exo_dim)` | Pre-computed exogenous trajectory |
| `z_fork` | `(N, T, exo_dim)` | Unused (present for pipeline compatibility) |

Validation dataset uses flattened format: `{s_endo, z, z_next_main, z_next_fork}`.

## 5. Default Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Rollout horizon | $T$ | 64 | Must be $\leq$ `DataGeneratorConfig.horizon` |
| Batch size | $B$ | 256 | |
| Learning rate | $\alpha_\theta$ | $10^{-3}$ | Adam with clipnorm=100 |
| Terminal value | | True | Analytical $r(\bar{s}, \bar{a}) / (1-\gamma)$ |
| Hidden layers | $L$ | 2 | |
| Hidden neurons | $n$ | 128 | Per layer |
| Activation | | SiLU | |
| Temperature | | $10^{-6}$ | Smooth gate for non-differentiable reward components |

## 6. Usage Reference

### Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `env` | `MDPEnvironment` | Must implement `merge_state`, `split_state`, `endogenous_transition`, `reward`, `discount`.  Optionally `terminal_value` for the correction. |
| `policy` | `PolicyNetwork` | Initialized (forward pass called once before training). |
| `train_dataset` | `dict` | Trajectory format: `s_endo_0` (N, endo_dim), `z_path` (N, T+1, exo_dim).  Requires dataset horizon $\geq$ `LRConfig.horizon`. |
| `val_dataset` | `dict` or `None` | Flattened format: `s_endo`, `z`, `z_next_main`, `z_next_fork`.  Used for Euler residual evaluation only. |
| `config` | `LRConfig` | See Section 5 for defaults. |

### Output

Returns a dict with keys `policy` (updated in-place), `history`, and
`config`.  The `history` dict contains per-evaluation-step lists: `step`,
`loss`, `euler_residual`.

### Minimal example

```python
config = LRConfig(n_steps=3000, horizon=64, eval_interval=500)

result = train_lr(env, policy,
                  train_traj, val_flat, config=config)
```

The primary convergence signal is `euler_residual`.  `loss` is in reward
units (negative expected lifetime reward).
