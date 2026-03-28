# Training Infrastructure

Shared infrastructure used by all v2 trainers (LR, ER, BRM, SHAC).
Method-specific algorithm details are in the per-method docs; this document
covers the common API, evaluation, early stopping, and output format.

## 1. Eval Callback

Every trainer accepts an optional `eval_callback` that computes custom
validation metrics at each evaluation checkpoint.

### Signature

```python
def eval_callback(
    step:              int,                # current training step
    env:               MDPEnvironment,     # environment instance
    policy:            PolicyNetwork,      # current policy (updated in-place)
    value_net:         Optional[Model],    # critic, or None (LR/ER have no critic)
    val_dataset:       Optional[dict],     # flattened validation dataset
    train_temperature: float,              # current training temperature
) -> Dict[str, float]:                     # metric_name → scalar value
```

When no callback is provided, the trainer falls back to built-in metrics
computed on `val_dataset`:

| Metric key            | Formula                                         | Available when         |
|-----------------------|-------------------------------------------------|------------------------|
| `euler_residual_val`  | mean |1 − β·m/χ| (one-step Euler condition)      | env implements `euler_residual()` |
| `bellman_residual`    | mean |V(s) − r − γ·V(s')|                         | value_net is not None  |

The callback return value is merged into the history dict.  The `monitor`
field in the config refers to a key from this dict (e.g., `"policy_mae"`,
`"euler_residual_val"`).

### Design notes

- **euler_residual_val**: The gold-standard diagnostic.  Measures how well
  the policy satisfies the Euler equation (first-order optimality condition)
  on held-out states.  Near-zero residual implies near-optimal policy.
  Computed using a single next-period draw (`z_next_main`) from the
  validation dataset.

- **bellman_residual**: Measures self-consistency of the value function:
  V(s) ≈ r(s, a) + γ·V(s').  Low residual does **not** imply the policy
  is good — a constant V = 0 has low residual if rewards are small.
  Useful for diagnosing critic health, not policy quality.

- **lifetime_reward_val**: (Notebook-defined, not built-in.)  Discounted
  cumulative reward from trajectory rollouts under the policy.  Directly
  measures economic value but is noisy for short horizons.

- **policy_mae**: (Notebook-defined, requires analytical solution.)  Mean
  absolute error of k'(policy) vs k'(analytical) on held-out states.
  The most direct quality measure when an analytical solution exists.

## 2. Early Stopping

All trainers use `StopTracker` with two independent rules checked at each
eval checkpoint.  First to fire wins.

### Rule 1: Threshold (convergence)

Fires when the monitored metric satisfies `threshold` for
`threshold_patience` *consecutive* eval checkpoints.
→ `stop_reason = "converged"`

### Rule 2: Plateau (fallback)

Only checked when the threshold rule does not fire.  Fires when the metric
has not improved by at least `max(plateau_min_delta, plateau_rel_delta *
|best|)` for `plateau_patience` consecutive evals.  A threshold hit counts
as progress and resets the plateau counter.
→ `stop_reason = "plateau"`

Set `plateau_patience = None` to disable (default).

### Additional stops

- **max_wall_time_sec**: Wall-clock cap. → `stop_reason = "max_wall_time"`
- **n_steps exhausted**: → `stop_reason = "max_steps"`

All patience counts are in eval checkpoints, not gradient steps.

### Configuration fields

| Field                 | Type             | Default | Notes                                   |
|-----------------------|------------------|---------|-----------------------------------------|
| `monitor`             | `str` or `None`  | `None`  | Key from eval_callback return dict      |
| `mode`                | `"min"` / `"max"`| `"min"` | Optimization direction                  |
| `threshold`           | `float` or `None`| `None`  | Target value for convergence            |
| `threshold_patience`  | `int`            | `2`     | Consecutive evals satisfying threshold  |
| `plateau_patience`    | `int` or `None`  | `None`  | Evals without improvement; None=off     |
| `plateau_min_delta`   | `float`          | `0.0`   | Absolute improvement margin             |
| `plateau_rel_delta`   | `float`          | `0.0`   | Relative improvement margin             |
| `min_steps_before_stop`| `int`           | `0`     | Minimum steps before any stop rule      |
| `max_wall_time_sec`   | `float` or `None`| `None`  | Wall-clock timeout                      |

## 3. Trainer Output

All trainers return a dict with the following structure:

```python
{
    # Models (updated in-place)
    "policy":     policy,
    "value_net":  value_net,       # BRM/SHAC only; absent for LR/ER

    # Training history
    "history": {
        "step":              [int, ...],    # eval checkpoint steps
        "elapsed_sec":       [float, ...],  # wall-clock seconds
        "train_temperature": [float, ...],  # temperature at each eval
        "loss":              [float, ...],  # training loss (method-specific)
        # ... plus any keys from eval_callback (e.g., "euler_residual_val")
    },

    # Config echo
    "config": config,

    # Timing
    "wall_time_sec": float,

    # Early stopping metadata
    "converged":              bool,
    "stop_reason":            str,   # "converged" | "plateau" | "max_steps" | "max_wall_time"
    "stop_step":              int,
    "stop_elapsed_sec":       float,
    "threshold_step":         int | None,   # first eval satisfying threshold
    "threshold_elapsed_sec":  float | None,
    "best_step":              int | None,
    "best_elapsed_sec":       float | None,
    "monitor":                str | None,
    "best_metric_value":      float | None,
}
```

### Method-specific history keys

| Method | `base_scalars` keys                    |
|--------|----------------------------------------|
| LR     | `loss`                                 |
| ER     | `loss`                                 |
| BRM    | `loss`, `loss_br`, `loss_foc`          |
| SHAC   | `loss_actor`, `loss_critic`            |

## 4. Input Normalization

All networks use a `StaticNormalizer` that z-score normalizes input states:
`ŝ = (s − μ) / σ`.  Statistics are fit once from the full training dataset
before any gradient steps, then frozen for the remainder of training (all
forward passes use `training=False`).

For methods with target networks (ER, SHAC), the target gets its own copy
of normalizer variables with identical values.

## 5. Common Config Fields (TrainingConfig base)

| Field               | Type             | Default       | Notes                                    |
|---------------------|------------------|---------------|------------------------------------------|
| `n_steps`           | `int`            | `10000`       | Maximum training steps                   |
| `batch_size`        | `int`            | `256`         | Mini-batch size                          |
| `master_seed`       | `tuple`          | `(20, 26)`    | RNG seed pair                            |
| `polyak_rate`       | `float`          | `0.995`       | Target network EMA rate (ER/SHAC)        |
| `eval_interval`     | `int`            | `500`         | Steps between eval checkpoints           |
| `eval_temperature`  | `float` or `None`| `None`        | Override temperature for eval metrics    |
| `temperature`       | `float`          | `1e-6`        | Smooth-gate temperature                  |
| `network`           | `NetworkConfig`  | `2L × 128N`   | Hidden layers / neurons                  |
| `policy_optimizer`  | `OptimizerConfig`| `Adam(1e-3)`  | Policy optimizer (lr, clipnorm=100)      |
| `critic_optimizer`  | `OptimizerConfig`| `Adam(1e-3)`  | Critic optimizer (BRM/SHAC only)         |
| `strict_reproducibility` | `bool`      | `False`       | Exact-kernel mode for debugging          |
