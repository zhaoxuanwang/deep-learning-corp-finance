# Value Function Iteration (VFI) and Policy Function Iteration (PFI)

## 1. Introduction and Motivation

VFI and PFI are the classical discrete dynamic programming solvers.  They
discretize the continuous state and action spaces onto finite grids,
estimate Markov transitions from data, and iterate the Bellman operator
to convergence.  The resulting value function and policy are exact on the
grid (up to discretization error) and serve as ground-truth benchmarks
for the neural-network-based methods (LR, ER, BRM, SHAC).

**Why keep both VFI and PFI?**  VFI is simpler and more robust — it
contracts at rate $\gamma$ per iteration regardless of the problem
structure.  PFI (Howard's method) converges in far fewer outer iterations
(quadratic vs linear rate) by solving a linear system at each step, but
each step is more expensive.  Running both and comparing their solutions
is a built-in consistency check: if VFI and PFI disagree beyond
discretization tolerance, something is wrong.

**Why we need NN methods alongside VFI/PFI.**  Grid-based methods suffer
the curse of dimensionality.  A problem with $d$ state variables and $m$
grid points per variable requires $m^d$ grid evaluations per Bellman
step.  For the basic investment model ($d = 2$: capital $k$ and
productivity $z$), grids of 50–100 points per variable are tractable.
For the risky debt model ($d = 3$: $k$, $z$, $b$) the cost is already
substantial.  Beyond 3–4 state variables, grid methods become infeasible
and NN-based function approximation is the only practical approach.

## 2. Algorithm

Both solvers share the same setup phase and differ only in the iteration
step.

### Setup (shared)

Given an `MDPEnvironment` and a flattened training dataset:

1. **Build grids.**  Discretize the endogenous state, exogenous state,
   and action spaces into finite grids.  Each variable can have its own
   spacing rule (linear or log) via `env.grid_spec()`.
   For multi-dimensional problems, the per-variable 1-D grids are
   combined into Cartesian product grids.

2. **Estimate transitions.**  From the dataset's observed $(z, z')$
   pairs, estimate the Markov transition matrix $P(z' \mid z)$ by
   binning observations into the exogenous grid and counting transitions,
   smoothed with a Dirichlet prior.

3. **Precompute tables.**  Evaluate the environment on every
   (state, action) grid combination to build two tables:
   - **Reward matrix** $R[i_z, i_k, i_a] = r(s, a)$ where
     $s = [k_{i_k},\, z_{i_z}]$ and $a = a_{i_a}$.
   - **Transition map** $T[i_z, i_k, i_a] = i_{k'}$ where $k'$ is
     computed via `env.endogenous_transition()` and snapped to the
     nearest endogenous grid point.

   After this step, the environment is never called again — the Bellman
   iteration operates entirely on precomputed tensors.

### VFI: Value Function Iteration

Initialize $V^{(0)}[i_z, i_k] = 0$ for all grid points.

**Repeat** for $n = 0, 1, 2, \ldots$:

$$V^{(n+1)}[i_z, i_k] = \max_{i_a} \left\{ R[i_z, i_k, i_a] \;+\; \gamma \sum_{j_z} P[i_z, j_z] \cdot V^{(n)}[j_z,\, T[i_z, i_k, i_a]] \right\}$$

**Until** $\|V^{(n+1)} - V^{(n)}\|_\infty < \varepsilon$.

The optimal policy is the argmax:

$$\pi^*[i_z, i_k] = \arg\max_{i_a} \left\{ R[i_z, i_k, i_a] + \gamma \sum_{j_z} P[i_z, j_z] \cdot V^*[j_z,\, T[i_z, i_k, i_a]] \right\}$$

**Convergence rate.**  The Bellman operator is a $\gamma$-contraction in
sup-norm.  After $n$ iterations, $\|V^{(n)} - V^*\|_\infty \leq
\gamma^n \|V^{(0)} - V^*\|_\infty$.  This is linear convergence at
rate $\gamma$.

### PFI: Policy Function Iteration (Howard's Method)

Initialize with one Bellman maximization step to obtain an initial
policy $\pi^{(0)}$.

**Repeat** for $n = 0, 1, 2, \ldots$:

1. **Policy evaluation.**  For the fixed policy $\pi^{(n)}$, compute its
   value by iterating:

   $$V^{(n)}[i_z, i_k] = R[i_z, i_k, \pi^{(n)}[i_z, i_k]] + \gamma \sum_{j_z} P[i_z, j_z] \cdot V^{(n)}[j_z,\, T[i_z, i_k, \pi^{(n)}[i_z, i_k]]]$$

   for a fixed number of evaluation steps (typically 200–400).  This is
   the same Bellman equation but with the max replaced by the fixed
   policy — a linear fixed-point problem.

2. **Policy improvement.**  One full Bellman maximization step:

   $$\pi^{(n+1)}[i_z, i_k] = \arg\max_{i_a} \left\{ R[i_z, i_k, i_a] + \gamma \sum_{j_z} P[i_z, j_z] \cdot V^{(n)}[j_z,\, T[i_z, i_k, i_a]] \right\}$$

**Until** $\pi^{(n+1)} = \pi^{(n)}$ (policy indices unchanged at every
grid point).

**Convergence rate.**  PFI converges quadratically in the number of
outer iterations — typically 5–20 policy updates suffice, versus
hundreds or thousands of VFI iterations.  The cost is the inner
evaluation loop, but this is a fixed linear iteration (no maximization).

## 3. Grid Discretization

### Per-variable spacing via `env.grid_spec()`

The environment communicates domain-specific grid structure through
`grid_spec()`, which returns a `GridAxis` specification per variable.
Two spacing options are available:

- **`"linear"`**: `np.linspace(low, high, n)`.  Uniform spacing.
- **`"log"`**: `np.exp(np.linspace(log(low), log(high), n))`.  Denser
  at low values, sparser at high values.  Grid values are always in
  levels, not log-space.

Both options always produce exactly `n` grid points as specified by the
user in `GridConfig`.

| Variable | Spacing | Rationale |
|----------|---------|-----------|
| Capital $k$ | `log` | Dense where curvature is highest (low $k$), sparse where the value function is nearly linear (high $k$). |
| Productivity $z$ | `log` | $z$ follows a log-AR(1), so log-spacing gives uniform resolution in the natural coordinate.  Grid values are in levels (matching v2 convention). |
| Investment $I$ | `linear` | No strong prior on the action distribution. |

Environments that do not override `grid_spec()` fall back to linear
spacing derived from `action_bounds()` and sampled state ranges.

### Transition matrix estimation

The Markov matrix $P(z' \mid z)$ is estimated from the flattened
dataset's `z` and `z_next_main` columns:

1. Map each observation to its nearest exogenous grid bin (nearest
   neighbor in the per-variable grids).
2. Count transitions: $C_{ij}$ = number of times bin $i$ transitions to
   bin $j$.
3. Smooth with a Dirichlet prior: $P_{ij} = (C_{ij} + \alpha / n_z) \,/\, (\sum_j C_{ij} + \alpha)$.
4. Renormalize rows to sum to 1.

The smoothing parameter $\alpha$ (default 1.0) prevents zero-probability
transitions that would create absorbing states.  With typical dataset
sizes ($N \cdot T \geq 10^5$ observations), the choice of $\alpha$ has
negligible impact.

### Multi-dimensional state and action

For problems with `endo_dim > 1` or `action_dim > 1`, per-variable 1-D
grids are combined into Cartesian product grids.  The value function is
indexed by a flat product-grid index (row-major / C-order).  The
endogenous next-state from `env.endogenous_transition()` generally
falls between grid points and is handled according to the
`interpolation` mode (see Section 3a).

This supports moderate dimensionality (e.g., the risky debt model with
2 endogenous variables $k, b$ and 2 action variables $I, b'$), but grid
sizes grow exponentially: $m$ points per variable and $d$ variables
yields $m^d$ product-grid points.

## 3a. Continuation-Value Interpolation

When the endogenous transition $k' = f(k, a, z)$ lands between grid
points, there are two strategies for evaluating the continuation value
$V(z', k')$.  The solver exposes this choice via the `interpolation`
config parameter.

### Snap-to-grid (`interpolation="none"`)

The baseline approach.  $k'$ is rounded to the nearest product-grid
point $k_{\text{near}}$ and the continuation is simply:

$$V_{\text{snap}}(z', k') = V(z', k_{\text{near}})$$

This is exact on the grid but introduces discretization error
proportional to the grid spacing $\Delta k$.  The Bellman max can
exploit this: when two actions map to the same nearest grid point, the
one whose true $k'$ is higher is indistinguishable from the lower one,
creating a systematic upward bias.

### Linear / N-linear interpolation (`interpolation="auto"` or `"linear"`)

Computes the continuation value by interpolating $V$ between the
bracketing grid points.  This eliminates the snap-to-grid bias and
allows coarser grids to achieve accuracy comparable to much finer
snap-to-grid solves.

**1-D endogenous state.**  For a single endogenous variable (e.g.,
capital $k$), the next-state $k'$ falls between two adjacent grid
points $k_{\ell}$ and $k_{\ell+1}$.  Define the interpolation fraction:

$$f = \frac{k' - k_\ell}{k_{\ell+1} - k_\ell}, \quad f \in [0, 1]$$

The interpolated continuation value is:

$$V_{\text{interp}}(z', k') = (1 - f) \cdot V(z', k_\ell) \;+\; f \cdot V(z', k_{\ell+1})$$

This is standard linear interpolation.  It is used whenever
`endo_dim = 1` and `interpolation` is `"auto"` or `"linear"`.

**Multi-D endogenous state (N-linear interpolation).**  For $D \geq 2$
endogenous variables $(x_0, x_1, \ldots, x_{D-1})$, the next state
falls inside a $D$-dimensional hypercube cell.  Each variable $d$ is
independently bracketed:

$$x'_d \in [x_{d,\ell_d},\; x_{d,\ell_d + 1}], \quad
  f_d = \frac{x'_d - x_{d,\ell_d}}{x_{d,\ell_d + 1} - x_{d,\ell_d}} \in [0, 1]$$

The interpolated value is the weighted sum over all $2^D$ corner
vertices of the hypercube:

$$V_{\text{interp}}(z', x') = \sum_{c \,\in\, \{0,1\}^D}
  \left(\prod_{d=0}^{D-1} w_d(c_d)\right) \cdot
  V\!\left(z',\; \text{flat}(i_0(c_0), \ldots, i_{D-1}(c_{D-1}))\right)$$

where for each dimension $d$ and corner bit $c_d$:

$$w_d(c_d) = \begin{cases} 1 - f_d & \text{if } c_d = 0 \\ f_d & \text{if } c_d = 1 \end{cases}, \qquad
  i_d(c_d) = \begin{cases} \ell_d & \text{if } c_d = 0 \\ \ell_d + 1 & \text{if } c_d = 1 \end{cases}$$

The flat product-grid index is computed via row-major strides:

$$\text{flat}(i_0, \ldots, i_{D-1}) = \sum_{d=0}^{D-1} i_d \cdot s_d,
  \quad s_d = \prod_{j=d+1}^{D-1} n_j$$

For $D = 2$ (bilinear), this sums over 4 corners.  For $D = 3$
(trilinear), 8 corners.  The cost per Bellman step grows as $2^D$
relative to snap-to-grid, but for $D \leq 3$ this is negligible
compared to the action-maximization loop.

### Mode selection

| `interpolation` | Behavior |
|-----------------|----------|
| `"auto"` (default) | Use interpolation when per-variable 1-D grids are available; fall back to snap-to-grid otherwise. |
| `"linear"` | Same as `"auto"`. Provided for explicitness. |
| `"none"` | Always snap to nearest grid point (brute-force baseline). |

Both VFI and PFI support the same three modes via `VFIConfig.interpolation`
and `PFIConfig.interpolation`.  The mode applies to:
- VFI: each Bellman maximization step.
- PFI: both the policy evaluation inner loop and the policy improvement step.

### Precomputation

The interpolation data is computed once during table precomputation
(before the iteration loop begins) and stored alongside the reward and
transition tables:

- **1-D:** `trans_lo` (lower bracket index) and `trans_frac`
  (interpolation fraction), each shape `(n_exo, n_endo, n_action)`.
- **Multi-D:** `trans_lo_d` and `trans_frac_d` for each variable $d$,
  same shape; plus `endo_var_sizes = [n_0, n_1, \ldots]`.

When `interpolation="none"`, this precomputed data is present but
ignored — only the snap-to-grid `trans_idx` is used.

## 4. Data Pipeline Integration

VFI and PFI fit into the v2 workflow identically to ER and BRM:

```
Environment ──► DataGenerator ──► flattened dataset ──► solve_vfi / solve_pfi
                                  {s_endo, z, z_next_main, z_next_fork}
```

The solvers consume the flattened dataset format with keys `s_endo`, `z`,
and `z_next_main`.  The `z_next_fork` key is present but unused — the
AiO cross-product estimator is specific to ER/BRM and not needed for
grid-based methods.

The dataset serves a single purpose: estimating $P(z' \mid z)$.  Rewards
are evaluated directly on grid points via `env.reward()`, not read from
the dataset.  The solver is controlled by the same master seed as all
other methods — the `DataGenerator` produces identical datasets for
VFI/PFI and NN trainers.

## 5. Default Hyperparameters

### Grid configuration (`GridConfig`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `exo_sizes` | `[7]` | Per-variable exogenous grid size.  Production: 11–15. |
| `endo_sizes` | `[25]` | Per-variable endogenous grid size.  Production: 50–100. |
| `action_sizes` | `[25]` | Per-variable action grid size.  Production: 50–100. |
| `transition_alpha` | 1.0 | Dirichlet smoothing mass. |

Defaults are deliberately small for fast debugging.  Production runs
should override explicitly.

### VFI configuration (`VFIConfig`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tol` | $10^{-6}$ | Sup-norm convergence tolerance. |
| `max_iter` | 2000 | Sufficient for $\gamma \leq 0.97$ with small grids. |
| `interpolation` | `"auto"` | `"auto"`, `"linear"`, or `"none"`. See Section 3a. |

### PFI configuration (`PFIConfig`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_iter` | 200 | Maximum policy improvement steps. |
| `eval_steps` | 400 | Bellman evaluations per policy evaluation. |
| `interpolation` | `"auto"` | `"auto"`, `"linear"`, or `"none"`. See Section 3a. |

## 6. Usage Reference

### Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `env` | `MDPEnvironment` | Must implement `reward`, `endogenous_transition`, `discount`, `action_bounds`.  Optionally `grid_spec` for non-default spacing. |
| `train_dataset` | `dict` | Flattened format: `z` (N, exo_dim), `z_next_main` (N, exo_dim).  `s_endo` present but unused. |
| `config` | `VFIConfig` or `PFIConfig` | See Section 5 for defaults. |

### Output

Both `solve_vfi` and `solve_pfi` return a dict with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `value` | `(n_exo, n_endo)` | Converged value function on the grid. |
| `policy_action` | `(n_exo, n_endo, action_dim)` | Optimal action in levels at each grid point. |
| `policy_endo` | `(n_exo, n_endo, endo_dim)` | Optimal next endogenous state in levels. |
| `grids` | `dict` | Grid arrays: `exo_grids_1d`, `endo_grids_1d`, `action_grids_1d`, and product grids. |
| `prob_matrix` | `(n_exo, n_exo)` | Estimated Markov transition matrix. |
| `converged` | `bool` | Whether convergence criterion was met. |
| `n_iter` | `int` | Number of iterations used. |
| `history` | `list` | Per-iteration sup-norm diffs (VFI only). |

### Minimal example

```python
from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.data.generator import DataGenerator, DataGeneratorConfig
from src.v2.solvers import solve_vfi, solve_pfi, VFIConfig, PFIConfig, GridConfig

env = BasicInvestmentEnv(econ_params=params, shock_params=shocks)
gen = DataGenerator(env, DataGeneratorConfig(n_paths=5000, horizon=64))
train_flat = gen.get_flattened_dataset("train")

# VFI
gc = GridConfig(exo_sizes=[11], endo_sizes=[50], action_sizes=[50])
result_vfi = solve_vfi(env, train_flat, VFIConfig(grid=gc))

# PFI (same grid)
result_pfi = solve_pfi(env, train_flat, PFIConfig(grid=gc))

# Compare
import tensorflow as tf
value_gap = float(tf.reduce_max(tf.abs(result_vfi["value"] - result_pfi["value"])))
print(f"VFI-PFI value gap: {value_gap:.2e}")
```

**Convergence signals.**  For VFI, monitor the sup-norm diff history —
it should decrease geometrically at rate $\gamma$.  For PFI, convergence
is detected when the policy indices are identical across two consecutive
improvement steps (typically 5–20 steps).

## 7. Limitations

- **Curse of dimensionality.**  Grid size grows as $m^d$ where $d$ is
  the total state + action dimensionality.  Practical limit is
  approximately $d \leq 4$ (e.g., $z, k, b, b'$) with moderate grid
  sizes.

- **Discretization error.**  With `interpolation="none"`, the
  nearest-neighbor snapping of $k'$ to the grid introduces
  approximation error proportional to the grid spacing $\Delta k$.
  With `interpolation="auto"` (default), linear / N-linear
  interpolation of the continuation value eliminates this bias,
  allowing coarser grids to achieve accuracy comparable to much finer
  snap-to-grid solves (see Section 3a).  Log grids further concentrate
  points where the value function has the most curvature.

- **No off-grid function approximation.**  The policy is a lookup table
  on the grid, not a smooth function.  Querying the value or policy at
  arbitrary off-grid states requires external post-processing (e.g.,
  scipy interpolation on the output arrays).  The interpolation mode
  only affects the *solver's internal* continuation value — it does not
  produce a callable interpolant for downstream use.

- **Transition estimation quality.**  The Markov matrix $P(z' \mid z)$
  is estimated from binned observations.  With very few exogenous grid
  points or very small datasets, the estimated transitions may poorly
  approximate the true AR(1) dynamics.  This is rarely a binding
  constraint in practice — datasets of $10^5$+ observations and 7+
  exogenous bins are more than sufficient.

These limitations are precisely why NN-based methods (LR, ER, BRM, SHAC)
exist.  VFI and PFI serve as ground-truth benchmarks for validating those
methods on low-dimensional problems where both approaches are feasible.
