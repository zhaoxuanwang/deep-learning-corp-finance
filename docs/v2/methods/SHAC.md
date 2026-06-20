# Short-Horizon Actor-Critic (SHAC), DDPG-Style Variant

## 1. Introduction and Motivation

SHAC solves infinite-horizon dynamic programming problems by combining
short-horizon backpropagation through differentiable dynamics with a
learned value function bootstrap.  It builds on
Xu et al. (2022), retaining the core structure of windowed actor BPTT
with on-policy continuation across windows, but replacing the critic
update with a DDPG-style 1-step Bellman target for stability in
economic environments.

**Core idea.**  The full $T$-step trajectory is divided into consecutive
windows of length $h$.  Within each window, the actor loss
backpropagates through $h$ exact dynamics steps, and a value function $V$
bootstraps the continuation beyond the window boundary.  Between
windows, the endogenous state carries forward (detached via
`stop_gradient`) so the trajectory remains on-policy.  This avoids the
gradient explosion/vanishing of full-trajectory BPTT while retaining
exact policy gradients through known, differentiable dynamics.

**Why a variant?**  The original SHAC uses an on-policy TD-$\lambda$
critic trained on rewards from the actor's own rollout.  In economic
environments with large reward scales, this creates a positive feedback
loop: the critic overfits to the actor's trajectory, the actor exploits
the critic's overestimates, and training diverges.  Our variant
decouples the critic from the current actor by using a 1-step Bellman
target with separate target networks for both policy and value.

## 2. Objects and Networks

### State decomposition

The state $s = [k,\, z]$ comprises endogenous components $k$
(e.g., capital stock, controlled by the agent's actions) and exogenous
components $z$ (e.g., productivity shocks, evolving independently of
the agent).  The environment provides four core operations:

| Operation | Signature | Description |
|-----------|-----------|-------------|
| Merge | `merge_state(k, z) → s` | Concatenate endogenous and exogenous into full state |
| Split | `split_state(s) → (k, z)` | Extract endogenous and exogenous from full state |
| Endogenous transition | `endogenous_transition(k, a, z) → k'` | Differentiable; on the computation graph |
| Reward | `reward(s, a) → r` | Differentiable; scalar per sample |

Exogenous transitions $z_{t+1} = f_{\text{exo}}(z_t, \varepsilon_{t+1})$
are pre-simulated offline and stored as paths $z_{0:T}$.  The trainer
never calls the exogenous transition directly.

### Networks

All networks share the architecture:
**StaticNormalizer → [Dense($n$) + SiLU] $\times$ $L$ → output head**.

| Network | Notation | Output head | Role |
|---------|----------|-------------|------|
| Policy (actor) | $\pi_\theta$ | Affine + clip (see below) | Maps full state $s$ to action $a$ |
| Value (critic) | $V_\phi$ | Dense(1), linear | Maps full state $s$ to scalar value |
| Target policy | $\bar{\pi}_\theta$ | Same as $\pi_\theta$ | Polyak-averaged copy of $\pi_\theta$; separate weights |
| Target value | $\bar{V}_\phi$ | Same as $V_\phi$ | Polyak-averaged copy of $V_\phi$; separate weights |

Default architecture: $L = 2$ hidden layers, $n = 128$ neurons per
layer, SiLU activation after each hidden layer.

**Policy output head.**  The raw network output is affine-rescaled
before clipping:

$$a = \text{clip}\!\Big(\, c + \sqrt{h_r} \cdot \text{raw},\; a_{\min},\; a_{\max}\Big)$$

where $c = (a_{\min} + a_{\max}) / 2$ is the action center and
$h_r = (a_{\max} - a_{\min}) / 2$ is the half-range.  The $\sqrt{\cdot}$
scaling provides moderate gradient amplification near the center.  The
output kernel uses Orthogonal initialization with gain $= 0.01$, so the
initial policy outputs $a \approx c$.

**Value output head.**  Linear (Dense(1), no activation).  When reward
normalization is enabled (the default), $V$ operates in scaled space
where $V \approx O(1)$.

**Input normalization.**  All networks use a StaticNormalizer that
z-score normalizes the input state: $\hat{s} = (s - \mu) / \sigma$.
The statistics $\mu$ and $\sigma$ are fit once from the full training
dataset before any gradient steps, and remain frozen during training.
All four networks (policy, value, and both targets) receive identical
normalizer statistics.  Target networks get their own copy (not shared)
because `hard_update` copies only trainable weights, not normalizer
variables.

### Reward normalization

Economic environments typically have reward/value scales of $O(100\text{--}500)$.
SHAC's default hyperparameters (learning rates, gradient clip norms)
assume $O(1)$.  We scale all rewards by
$\lambda_r = 1/|V^*|$, where $V^*$ is the environment's steady-state
value, so that the critic learns values of $O(1)$.  This is enabled by
default (`normalize_rewards=True` in `SHACConfig`).  Every reward $r$ in
both the actor loss and the critic target is multiplied by $\lambda_r$
before use.

## 3. Algorithm

### Notation summary

| Symbol | Meaning |
|--------|---------|
| $s = [k,\, z]$ | State: endogenous $k$ + exogenous $z$ |
| $a = \pi_\theta(s)$ | Action from current policy |
| $\lambda_r \cdot r(s, a)$ | Scaled per-period reward |
| $V_\phi(s)$ | Current value network (critic) |
| $\bar{V}_\phi(s)$ | Target value network (Polyak average of $V_\phi$) |
| $\bar{\pi}_\theta(s)$ | Target policy network (Polyak average of $\pi_\theta$) |
| $\gamma$ | Discount factor (environment-specific; typically 0.95) |
| $T$ | Trajectory horizon; require $T \bmod h = 0$; set large enough that $\gamma^T \approx 0$ |
| $h$ | Window length (short horizon) |
| $B$ | Batch size: number of parallel trajectories per mini-batch |
| $n_{\text{critic}}$ | Number of critic gradient steps per window |
| $\tau_{\text{polyak}}$ | Polyak averaging rate for target networks |
| $n_{\text{steps}}$ | Total training steps (= total windows processed) |

### Initialization

1. Initialize $\pi_\theta$ and $V_\phi$ with random weights.
2. Fit StaticNormalizer on the full training dataset (computed once,
   then frozen for the rest of training).
3. Create target networks: $\bar{\pi}_\theta \leftarrow \text{copy}(\pi_\theta)$,
   $\bar{V}_\phi \leftarrow \text{copy}(V_\phi)$.  Fit normalizer
   statistics on target networks identically (separate copy, same values).
4. No warm-start: both actor and critic learn from scratch (cold start).

### Training loop

The training dataset provides $N$ trajectories, each containing an
initial endogenous state $k_0$ and a pre-simulated exogenous path
$z_{0:T}$.  A `build_iterator` wraps this into a shuffled, repeating
mini-batch iterator that yields batches of $B$ trajectories.

**For each mini-batch** (draw $B$ trajectories: initial states $k_0^{1:B}$
and exogenous paths $z_{0:T}^{1:B}$):

Set $k \leftarrow \text{stop\_gradient}(k_0)$.

**For each window** $w = 0, 1, \ldots, T/h - 1$:

Let $t_0 = w \cdot h$.  Slice $z_{\text{window}} = z_{t_0 : t_0 + h + 1}$
(shape $B \times (h+1) \times d_z$; the extra column provides $z_{t_0+h}$
for the terminal state).

**(a) Actor step** (inside `GradientTape`):

Starting from $k$ (detached from prior windows), unroll $h$ steps on
the computation graph.  All operations below are batched over $B$
trajectories in parallel:

$$
\begin{aligned}
&\textbf{for } \tau = 0, \ldots, h-1: \\
&\qquad s_\tau = \text{merge}(k_{\text{current}},\; z_{t_0+\tau}) \\
&\qquad a_\tau = \pi_\theta(s_\tau) \\
&\qquad r_\tau = \lambda_r \cdot r(s_\tau,\, a_\tau) \\
&\qquad k_{\text{current}} \leftarrow f_{\text{endo}}(k_{\text{current}},\, a_\tau,\, z_{t_0+\tau}) \\[6pt]
&s_h = \text{merge}(k_{\text{current}},\; z_{t_0+h}) \\
&v_{\text{bootstrap}} = V_\phi(s_h) \qquad \textit{(current V, not target } \bar{V}_\phi \textit{)}
\end{aligned}
$$

The actor loss is:

$$\mathcal{L}_\theta = -\frac{1}{B}\sum_{i=1}^{B}\left[\sum_{\tau=0}^{h-1} \gamma^{\tau}\, r_\tau^i \;+\; \gamma^h\, v_{\text{bootstrap}}^i \right]$$

Compute $\partial \mathcal{L}_\theta / \partial \theta$ via the tape
and update $\pi_\theta$ with one Adam step.  The gradient flows through
$\pi_\theta$ and $V_\phi$ into the dynamics chain.

**Important:** the actor update is always full-batch over all $B$
trajectories.  The backward pass requires the intact computation graph,
so no further mini-batching is possible.

Side outputs for critic training (all detached from the actor graph):
- Collected states $\{s_0, \ldots, s_h\}$ (shape $B \times (h+1) \times d_s$,
  with `stop_gradient` applied at collection time)
- Next-step exogenous values $\{z_{t_0+1}, \ldots, z_{t_0+h}\}$
  (shape $B \times h \times d_z$)

Set $k \leftarrow \text{stop\_gradient}(k_{\text{current}})$ for the next
window.

**(b) Critic step** (outside actor tape, $n_{\text{critic}}$ gradient
steps):

Flatten the collected state-pairs from step (a) into $B \cdot h$ pairs
$(s,\, z_{\text{next}})$.  For each critic gradient step, evaluate the
1-step Bellman target at each state independently:

$$
\begin{aligned}
a_{\text{target}} &= \bar{\pi}_\theta(s) \\
r_{\text{target}} &= \lambda_r \cdot r(s,\, a_{\text{target}}) \\
k'_{\text{target}} &= f_{\text{endo}}(k,\, a_{\text{target}},\, z) \\
s'_{\text{target}} &= \text{merge}(k'_{\text{target}},\, z_{\text{next}}) \\
y &= \text{stop\_gradient}\!\Big( r_{\text{target}} + \gamma \cdot \bar{V}_\phi(s'_{\text{target}}) \Big)
\end{aligned}
$$

The critic loss is:

$$\mathcal{L}_\phi = \frac{1}{B \cdot h}\sum_{j=1}^{B \cdot h}\bigl(V_\phi(s_j) - y_j\bigr)^2$$

Compute $\partial \mathcal{L}_\phi / \partial \phi$ and update $V_\phi$
with one Adam step.  Repeat for $n_{\text{critic}}$ total gradient steps
on the same full batch.

Note: the critic re-evaluates each state using target policy
$\bar{\pi}_\theta$ and target value $\bar{V}_\phi$.  It does **not** use
the actor's rollout actions or rewards.  The Bellman target $y$ is fully
detached (`stop_gradient`), so no gradient flows from the critic loss
into $\bar{\pi}_\theta$ or $\bar{V}_\phi$.

**(c) Polyak update** both target networks:

$$\bar{V}_\phi \leftarrow \tau_{\text{polyak}} \cdot \bar{V}_\phi + (1 - \tau_{\text{polyak}}) \cdot V_\phi$$
$$\bar{\pi}_\theta \leftarrow \tau_{\text{polyak}} \cdot \bar{\pi}_\theta + (1 - \tau_{\text{polyak}}) \cdot \pi_\theta$$

**(d) Continue.**  Increment step counter.  The endogenous state $k$
carries into the next window (already detached in step (a)).

### Step counting

| Event | Trigger |
|-------|---------|
| One actor gradient update | Every window (every step) |
| $n_{\text{critic}}$ critic gradient updates | Every window (every step) |
| Draw next mini-batch of $B$ trajectories | Every $T/h$ windows |

1 window = 1 step = 1 actor gradient update.  Each mini-batch of $B$
trajectories produces $T/h$ steps.  The user controls training duration
via $n_{\text{steps}}$ only; mini-batch draws follow mechanically.

## 4. Differences from Xu et al. (2022)

Our implementation retains the core SHAC structure (windowed $h$-step
actor BPTT with on-policy continuation) but modifies the critic update
for stability in economic environments.  The key differences are:

### Critic target: 1-step Bellman with target $\bar{\pi}$ + target $\bar{V}$

**Original (Xu et al.):**  TD-$\lambda$ targets computed from the actor's
own rollout within the window.  The GAE recursion combines $k$-step
returns using the rewards the actor actually received:
$\tilde{V}(s_\tau) = (1-\lambda) \sum_k \lambda^{k-1} G_\tau^k$,
where $G_\tau^k = \sum_{l=0}^{k-1} \gamma^l r_{t_0+\tau+l} + \gamma^k \bar{V}_\phi(s_{t_0+\tau+k})$.

**Ours:**  1-step Bellman target
$y = \lambda_r \cdot r(s, \bar{\pi}(s)) + \gamma \cdot \bar{V}(s')$, where
both the action and the next-state value are evaluated using target
networks.  The critic does not use the actor's rollout rewards; it
independently re-evaluates each state.

**Motivation.**  The original SHAC's critic targets incorporate rewards
$r(s, \pi_\theta(s))$ from the current policy, making them non-stationary
at the rate of policy change.  Our Bellman target uses the slowly-moving
$\bar{\pi}_\theta$, so targets shift by $O(1 - \tau_{\text{polyak}})$ per step
rather than by the full actor learning rate.  When reward magnitudes are
large (as in economic environments where $V \approx O(100\text{--}500)$
before scaling), even small policy perturbations produce large reward
swings that destabilize on-policy targets.

### Actor bootstrap: current $V$ (not target $\bar{V}$)

**Original:**  The actor bootstraps with target $\bar{V}$ (no gradient
through $V$).

**Ours:**  The actor bootstraps with current $V_\phi$, and the gradient
flows through $V_\phi$ into the actor update.

**Motivation.**  This provides a richer gradient signal. The actor
receives gradients not only through $h$ steps of dynamics but also
through $V$'s learned landscape at $s_h$.  This is analogous to DDPG's
actor gradient $\nabla_a Q(s,a)$, except here the signal combines $h$
exact dynamics steps with a $V$ bootstrap.

### Target policy $\bar{\pi}$ (in addition to target $\bar{V}$)

**Original:**  Only $\bar{V}$ is maintained as a target network.

**Ours:**  Both $\bar{\pi}$ and $\bar{V}$ are Polyak-averaged targets.

**Motivation.**  The critic target
$y = r(s, \bar{\pi}(s)) + \gamma \cdot \bar{V}(s')$ requires a stable
action for evaluation.  Without $\bar{\pi}$, using the current
(rapidly changing) policy in the Bellman target would reintroduce the
instability that the target value network is meant to prevent.

### Reward normalization

**Original:**  Not discussed.

**Ours:**  All rewards are scaled by $\lambda_r = 1/|V^*|$ so
$V \approx O(1)$.

**Motivation.**  The paper's default hyperparameters (learning rates,
gradient clip norms) assume rewards and values of $O(1)$.  Economic
environments have $V \approx O(100\text{--}500)$, causing critic
divergence with paper defaults.  Rescaling makes the paper's
hyperparameters appropriate without per-environment tuning.

### Input normalization

**Original:**  Running z-score (updated during training).

**Ours:**  Static z-score (fit once from training data, frozen).

**Motivation.**  Fitting normalizer statistics from the full dataset
before training avoids non-stationarity in input statistics during
gradient steps, and ensures all networks (including targets) see
identically normalized inputs.

## 5. Default Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Window length | $h$ | 32 | See sensitivity discussion below |
| Trajectory horizon | $T$ | 192 | Largest multiple of $h$ where $\gamma^T \approx 0$ ($\gamma = 0.95$) |
| Batch size | $B$ | 64 | Full-batch actor update per window |
| Actor learning rate | $\alpha_\theta$ | $2 \times 10^{-3}$ | See sensitivity discussion below |
| Critic learning rate | $\alpha_\phi$ | $5 \times 10^{-3}$ | Higher than actor to track changing policy |
| Polyak rate | $\tau_{\text{polyak}}$ | 0.995 | Controls target network lag |
| Critic steps per window | $n_{\text{critic}}$ | 16 | Full-batch, repeated $n_{\text{critic}}$ times |
| Gradient clip norm | | 100.0 | Applied to both actor and critic (Adam clipnorm) |
| Reward normalization | | True | Auto-computes $1/|V^*|$ via environment |
| Hidden layers | $L$ | 2 | Shared architecture for $\pi$ and $V$ |
| Hidden neurons | $n$ | 128 | Per layer |
| Activation | | SiLU | Applied after each hidden layer |
| Policy output head | | Affine + clip | $c + \sqrt{h_r} \cdot \text{raw}$, clipped to bounds |
| Value output head | | Dense(1) | Linear, no activation |
| Temperature | | $10^{-6}$ | Smooth gate for non-differentiable reward components |

### Sensitivity notes

**Window length $h = 32$.**  In models where the action's payoff is
spread across many periods (e.g., investment that depreciates slowly),
a short window captures only a fraction of the return while bearing
the full immediate cost.  Concretely, investing one extra unit at step
$\tau = 0$ costs 1 unit immediately but yields only
$\sum_{t=1}^{h-1} \gamma^t \cdot \text{MPK} \cdot (1-\delta)^{t-1}$
in within-window production, which falls short of 1 for small $h$.  The
deficit must be supplied by the critic bootstrap
$\gamma^h V_\phi(s_h)$; if the critic is poorly initialized, the actor
gradient systematically favors under-investment.  With $h = 32$ the
within-window return covers most of the marginal investment cost, making
the actor less dependent on bootstrap quality.

**Actor learning rate $\alpha_\theta = 2 \times 10^{-3}$.**  The SHAC
actor gradient is inherently weaker per step than the full-trajectory LR
gradient, because only $h$ steps of reward and one bootstrap term
contribute to each update (versus $T$ steps in LR).  A learning rate
adequate for LR (e.g., $10^{-3}$) results in negligibly slow policy
updates under SHAC, preventing the actor from escaping its
initialization basin within a practical training budget.

## 6. Usage Reference

### Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `env` | `MDPEnvironment` | Must implement `merge_state`, `split_state`, `endogenous_transition`, `reward`, `discount`, `action_spec`, `compute_reward_scale` |
| `policy` | `PolicyNetwork` | Initialized (forward pass called once before training) |
| `value_net` | `StateValueNetwork` | Initialized; same `state_dim` as policy |
| `train_dataset` | `dict` | Trajectory format: `s_endo_0` (N, endo_dim), `z_path` (N, T+1, exo_dim). Requires T >= `SHACConfig.horizon` |
| `val_dataset` | `dict` or `None` | Flattened format: `s_endo`, `z`, `z_next_main`. Used for Euler/Bellman evaluation only |
| `config` | `SHACConfig` | See Section 5 for defaults |
| `eval_callback` | callable or `None` | Custom validation metrics.  See [training_infrastructure.md](training_infrastructure.md) §1. |

Target networks ($\bar{\pi}_\theta$, $\bar{V}_\phi$) and input
normalizers are created internally by the trainer.

### Output

See [training_infrastructure.md](training_infrastructure.md) §3 for the
full output schema (history keys, early-stopping metadata, wall time).
SHAC records `loss_actor`, `loss_critic` in history.

### Minimal example

```python
from src.v2.trainers.shac import train_shac
from src.v2.trainers.config import SHACConfig

config = SHACConfig(
    n_steps=3000,
    eval_interval=500,
    horizon=192,           # total trajectory length T
    short_horizon=32,      # actor rollout window h (T must be divisible by h)
    # normalize_rewards=True,       # auto reward scaling (default)
    # reward_scale_override=None,   # manual override if needed
)

result = train_shac(env, policy, value_net,
                    train_traj, val_flat, config=config)
```

**Key config choices:**
- `horizon` / `short_horizon` control trajectory slicing: each mini-batch
  of B trajectories is split into T/h windows, each yielding one actor
  gradient step.  Shorter windows (smaller h) reduce BPTT depth but
  increase reliance on the critic bootstrap.
- `normalize_rewards=True` (default) auto-scales rewards by
  `env.reward_scale()` so that value predictions are O(1).  This prevents
  the critic from producing large bootstrap values that destabilize the
  actor.  Set `reward_scale_override` to provide a manual scale.

### Relationship to Xu et al. (2022)

This is a **DDPG-style variant** of the original SHAC algorithm.  The key
modifications (1-step Bellman critic with target networks instead of TD-λ,
reward normalization) are described in Section 4.  The original TD-λ
variant is archived in `src/v2/experimental/shac_vanilla.py` and diverges
in the offline setting due to a positive feedback loop between actor drift
and on-policy critic targets.

### Reference

Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A.,
& Macklin, M. (2022). Accelerated Policy Learning with Parallel
Differentiable Simulation. ICLR 2022.
