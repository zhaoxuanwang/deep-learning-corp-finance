"""Core training utilities: target network ops and evaluation metrics.

Shared across all training methods (LR, ER, BRM, SHAC).

Removed in this refactor:
    - ReplayBuffer      (replaced by offline DataGenerator datasets)
    - SeedSchedule      (moved to src/v2/data/rng.py)
    - collect_transitions / generate_eval_dataset  (data is now external)

Evaluation functions use the flattened dataset format:
    {s_endo, z, z_next_main, z_next_fork}
which is produced by DataGenerator.get_flattened_dataset().
"""

from dataclasses import dataclass
import math  # used for ceil in warm_start_value_net and StopTracker
from typing import Callable, Dict, Mapping, Optional

import tensorflow as tf
from src.v2.data.pipeline import fit_normalizer_traj, fit_normalizer_flat
from src.v2.utils.seeding import fold_in_seed


# ---------------------------------------------------------------------------
# Target Network Utilities
# ---------------------------------------------------------------------------

def polyak_update(source: tf.keras.Model, target: tf.keras.Model, tau: float):
    """Soft-update: w_target ← τ·w_target + (1-τ)·w_source."""
    for src_var, tgt_var in zip(source.trainable_variables,
                                target.trainable_variables):
        tgt_var.assign(tau * tgt_var + (1.0 - tau) * src_var)


def hard_update(source: tf.keras.Model, target: tf.keras.Model):
    """Copy source weights to target."""
    for src_var, tgt_var in zip(source.trainable_variables,
                                target.trainable_variables):
        tgt_var.assign(src_var)


def build_target_policy(policy):
    """Create a target policy network with identical architecture.

    Used by ER (stable next-action), BRM, and SHAC (DDPG-style critic
    Bellman targets with target π̄).
    """
    from src.v2.networks.policy import PolicyNetwork
    target_seed = (
        fold_in_seed(policy.seed, "target_policy")
        if getattr(policy, "seed", None) is not None else None
    )
    target = PolicyNetwork(
        state_dim=policy.input_dim,
        action_dim=policy.action_dim,
        action_low=policy.action_low,
        action_high=policy.action_high,
        action_center=policy.action_center,
        action_half_range=policy.action_half_range,
        n_layers=policy.hidden_stack.dense_layers.__len__(),
        n_neurons=policy.hidden_stack.dense_layers[0].units,
        activation=getattr(policy, "activation", "silu"),
        seed=target_seed,
        name="policy_target",
    )
    dummy = tf.zeros((1, policy.input_dim))
    target(dummy)
    hard_update(policy, target)
    return target


def build_target_value(value_net):
    """Create a target value network with identical architecture.

    Used by SHAC for stable critic Bellman targets (target V̄_φ).
    """
    from src.v2.networks.state_value import StateValueNetwork
    target_seed = (
        fold_in_seed(value_net.seed, "target_value")
        if getattr(value_net, "seed", None) is not None else None
    )
    target = StateValueNetwork(
        state_dim=value_net.input_dim,
        n_layers=value_net.hidden_stack.dense_layers.__len__(),
        n_neurons=value_net.hidden_stack.dense_layers[0].units,
        activation=getattr(value_net, "activation", "silu"),
        seed=target_seed,
        name="value_target",
    )
    dummy = tf.zeros((1, value_net.input_dim))
    target(dummy)
    hard_update(value_net, target)
    return target


# ---------------------------------------------------------------------------
# Warm-start utilities
# ---------------------------------------------------------------------------

def warm_start_value_net(env, value_net, train_dataset: dict,
                         n_steps: int = None, n_epochs: int = None,
                         learning_rate: float = 1e-3,
                         batch_size: int = 256, reward_scale: float = 1.0,
                         shuffle_seed: Optional[int] = None):
    """Pre-train value network on analytical terminal-value targets.

    Fits V_φ(s) ≈ λ · V^term(k) = λ · r(k, z̄, δk) / (1-γ), giving the
    critic a reasonable initialization where dV/dk > 0.  This avoids the
    cold-start degenerate equilibrium where an uninformative bootstrap
    causes the actor to learn zero investment.

    When reward_scale (λ) != 1.0, targets are scaled so V_φ is consistent
    with the scaled rewards used during training (e.g. SHAC with reward
    normalization).

    Accepts both dataset formats:
      - Trajectory (SHAC): keys s_endo_0, z_path
      - Flattened  (BRM):  keys s_endo, z

    The normalizer is fit internally (idempotent with later trainer fitting).

    Step count: exactly one of n_steps or n_epochs must be provided.
    n_epochs (preferred) auto-sizes to cover the full dataset proportionally
    regardless of dataset size; n_steps is a legacy exact count.

    Args:
        env:           MDPEnvironment with terminal_value() implemented.
        value_net:     StateValueNetwork to warm-start (modified in-place).
        train_dataset: Trajectory- or flattened-format dataset dict.
        n_steps:       Exact number of MSE regression steps (legacy).
        n_epochs:      Number of full passes through the dataset (preferred).
        learning_rate: Adam learning rate for warm-start.
        batch_size:    Mini-batch size for warm-start.
        reward_scale:  Multiplier for V targets (default 1.0 = no scaling).
        shuffle_seed:  Optional deterministic seed for the warm-start shuffle.
    """
    if n_steps is None and n_epochs is None:
        raise ValueError(
            "warm_start_value_net: specify either n_steps or n_epochs.")
    if n_steps is not None and n_epochs is not None:
        raise ValueError(
            "warm_start_value_net: specify n_steps or n_epochs, not both.")
    # Detect format and build (s_all, v_target) pairs
    if "z_path" in train_dataset:
        # Trajectory format (SHAC, LR)
        fit_normalizer_traj(env, train_dataset, value_net)

        z_path   = train_dataset["z_path"]      # (N, T+1, exo_dim)
        s_endo_0 = train_dataset["s_endo_0"]    # (N, endo_dim)
        T_plus_1 = tf.shape(z_path)[1]
        exo_dim  = env.exo_dim()

        z_flat  = tf.reshape(z_path, [-1, exo_dim])
        k_rep   = tf.repeat(s_endo_0, T_plus_1, axis=0)
        s_all   = env.merge_state(k_rep, z_flat)
        v_target = tf.reshape(env.terminal_value(k_rep), [-1])
    else:
        # Flattened format (BRM)
        fit_normalizer_flat(env, train_dataset, value_net)

        s_endo = train_dataset["s_endo"]        # (N, endo_dim)
        z      = train_dataset["z"]             # (N, exo_dim)
        s_all  = env.merge_state(s_endo, z)
        v_target = tf.reshape(env.terminal_value(s_endo), [-1])

    v_target = v_target * reward_scale

    # Resolve step count: epoch-based (preferred) or exact legacy count.
    n_samples = int(s_all.shape[0])
    if n_epochs is not None:
        effective_steps = math.ceil(n_samples / batch_size) * n_epochs
    else:
        effective_steps = n_steps

    ds = tf.data.Dataset.from_tensor_slices((s_all, v_target))
    ds = ds.shuffle(
        n_samples,
        seed=shuffle_seed,
        reshuffle_each_iteration=True,
    ).batch(batch_size, drop_remainder=True).repeat()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def _step(s_batch, v_batch):
        with tf.GradientTape() as tape:
            v_pred = tf.squeeze(value_net(s_batch, training=True), axis=-1)
            loss = tf.reduce_mean((v_pred - v_batch) ** 2)
        grads = tape.gradient(loss, value_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, value_net.trainable_variables))
        return loss

    for i, (s_b, v_b) in enumerate(ds.take(effective_steps)):
        loss = _step(s_b, v_b)
        if i % 50 == 0 or i == effective_steps - 1:
            print(f"  warm-start step {i:4d} | MSE={float(loss):.2f}")


def resolve_eval_temperature(config, train_temperature: float) -> float:
    """Choose the temperature used for validation metrics.

    By default evaluation mirrors the current training temperature.
    Notebooks can override this with config.eval_temperature to benchmark
    all checkpoints against a fixed target temperature.
    """
    if getattr(config, "eval_temperature", None) is not None:
        return float(config.eval_temperature)
    return float(train_temperature)


EvalCallback = Callable[[int, object, object, Optional[object], Optional[dict], float], Dict[str, float]]


def default_eval_metrics(env, policy, value_net, val_dataset: Optional[dict],
                         temperature: float) -> Dict[str, float]:
    """Default validation metrics used when no eval callback is provided."""
    if val_dataset is None:
        return {}

    metrics = {
        "euler_residual_val": evaluate_euler_residual(
            env, policy, val_dataset, temperature=temperature),
    }
    if value_net is not None:
        metrics["bellman_residual"] = evaluate_bellman_residual(
            env, policy, value_net, val_dataset, temperature=temperature)
    return metrics


def run_eval_callback(step: int,
                      env,
                      policy,
                      value_net,
                      val_dataset: Optional[dict],
                      train_temperature: float,
                      eval_callback: Optional[EvalCallback] = None,
                      eval_temperature: Optional[float] = None) -> Dict[str, float]:
    """Run notebook-supplied eval callback or fall back to built-in metrics."""
    temperature = (
        float(eval_temperature)
        if eval_temperature is not None else float(train_temperature)
    )
    if eval_callback is None:
        return default_eval_metrics(
            env, policy, value_net, val_dataset, temperature=temperature)

    metrics = eval_callback(
        step, env, policy, value_net, val_dataset, float(train_temperature))
    if metrics is None:
        return {}
    if not isinstance(metrics, Mapping):
        raise TypeError(
            "eval_callback must return a mapping of metric_name -> float. "
            f"Got {type(metrics)!r}")
    return {str(k): float(v) for k, v in metrics.items()}


def append_history_row(history: dict,
                       step: int,
                       elapsed_sec: float,
                       train_temperature: float,
                       base_scalars: Optional[Mapping[str, float]] = None,
                       eval_metrics: Optional[Mapping[str, float]] = None) -> None:
    """Append one logged row, creating metric lists lazily when needed."""
    history.setdefault("step", []).append(int(step))
    history.setdefault("elapsed_sec", []).append(float(elapsed_sec))
    history.setdefault("train_temperature", []).append(float(train_temperature))

    for source in (base_scalars or {}, eval_metrics or {}):
        for key, value in source.items():
            history.setdefault(str(key), []).append(float(value))


def capture_checkpoint(step: int,
                       config,
                       *,
                       policy=None,
                       value_net=None) -> None:
    """Capture named model snapshots at an eval checkpoint.

    Supports both the new generic checkpoint_history format and the legacy
    policy-only weight_history format used by older notebooks.
    """
    models = {
        "policy": policy,
        "value_net": value_net,
    }

    checkpoint_history = getattr(config, "checkpoint_history", None)
    snapshot_targets = tuple(getattr(config, "snapshot_targets", ("policy",)))
    if checkpoint_history is not None:
        entry = {"step": int(step), "models": {}}
        for model_name in snapshot_targets:
            model = models.get(model_name)
            if model is None:
                continue
            entry["models"][model_name] = [
                w.numpy() for w in model.trainable_weights
            ]
        checkpoint_history.append(entry)

    weight_history = getattr(config, "weight_history", None)
    if weight_history is not None and policy is not None:
        weight_history.append(
            (int(step), [w.numpy() for w in policy.trainable_weights])
        )


# ---------------------------------------------------------------------------
# Threshold / stopping helpers
# ---------------------------------------------------------------------------

@dataclass
class StopTracker:
    """Tracks two early-stopping rules and best-checkpoint metadata.

    Rules are checked in order at each eval checkpoint.  First to fire wins.

    1. **Threshold** (convergence):
       Fires when the monitored metric satisfies `threshold` for
       `threshold_patience` *consecutive* eval checkpoints.
       → ``stop_reason = "converged"``

    2. **Plateau** (fallback — only checked when threshold fails):
       Fires when the metric has not improved by at least
       ``max(plateau_min_delta, plateau_rel_delta * |best|)`` for
       `plateau_patience` consecutive evals.  A threshold hit counts as
       progress and resets the plateau counter (even if the threshold
       streak is broken later).
       → ``stop_reason = "plateau"``
       Set ``plateau_patience = None`` to disable (default).

    All patience counts are in *eval checkpoints* (not gradient steps).
    With ``eval_interval=100`` and ``threshold_patience=2``, the rule
    fires when two consecutive evals (at steps 100, 200, …) both satisfy
    the criterion.
    """

    # ── Config fields (set once at construction) ──────────────
    monitor: Optional[str] = None
    mode: str = "min"
    threshold: Optional[float] = None
    threshold_patience: int = 2
    plateau_patience: Optional[int] = None
    plateau_min_delta: float = 0.0
    plateau_rel_delta: float = 0.0
    min_steps_before_stop: int = 0

    # ── Mutable state ─────────────────────────────────────────
    converged: bool = False
    stop_reason: Optional[str] = None
    stop_step: Optional[int] = None
    stop_elapsed_sec: Optional[float] = None
    threshold_step: Optional[int] = None
    threshold_elapsed_sec: Optional[float] = None
    best_step: Optional[int] = None
    best_elapsed_sec: Optional[float] = None
    best_metric_value: Optional[float] = None

    _consecutive_hits: int = 0
    _candidate_step: Optional[int] = None
    _candidate_elapsed_sec: Optional[float] = None
    _no_improve_count: int = 0

    def _reset_hits(self) -> None:
        self._consecutive_hits = 0
        self._candidate_step = None
        self._candidate_elapsed_sec = None

    def _reset_no_improve(self) -> None:
        self._no_improve_count = 0

    def _is_strictly_better(self, metric_value: float) -> bool:
        if self.best_metric_value is None:
            return True
        if self.mode == "min":
            return metric_value < self.best_metric_value
        return metric_value > self.best_metric_value

    def _is_meaningful_improvement(self, metric_value: float) -> bool:
        if self.best_metric_value is None:
            return True
        margin = max(self.plateau_min_delta,
                     self.plateau_rel_delta * abs(self.best_metric_value))
        if self.mode == "min":
            return metric_value < self.best_metric_value - margin
        return metric_value > self.best_metric_value + margin

    def _threshold_satisfied(self, metric_value: float) -> bool:
        if self.threshold is None:
            return False
        if self.mode == "min":
            return metric_value <= self.threshold
        return metric_value >= self.threshold

    def record_eval(self, step: int, elapsed_sec: float,
                    metrics: Mapping[str, float]) -> bool:
        """Update tracker from a logged eval point.

        Returns:
            True if a configured stop rule is satisfied.
        """
        if self.stop_reason in {"converged", "plateau"}:
            return True
        if self.stop_reason is not None:
            return False

        if self.monitor is None:
            return False

        metric_value = float(metrics.get(self.monitor, float("nan")))
        if not math.isfinite(metric_value):
            self._reset_hits()
            self._reset_no_improve()
            return False

        # Compute improvement flags BEFORE updating best
        is_meaningful_improvement = self._is_meaningful_improvement(metric_value)

        # Update best-ever tracking
        if self._is_strictly_better(metric_value):
            self.best_metric_value = metric_value
            self.best_step = step
            self.best_elapsed_sec = elapsed_sec

        stop_enabled = step + 1 >= self.min_steps_before_stop
        if not stop_enabled:
            self._reset_hits()
            self._reset_no_improve()
            return False

        # ── Rule 1: Threshold (convergence) ───────────────────
        if self.threshold is not None:
            if self._threshold_satisfied(metric_value):
                # Threshold hit counts as progress → reset plateau
                self._reset_no_improve()
                if self._consecutive_hits == 0:
                    self._candidate_step = step
                    self._candidate_elapsed_sec = elapsed_sec
                self._consecutive_hits += 1
                if self._consecutive_hits >= self.threshold_patience:
                    self.converged = True
                    self.stop_reason = "converged"
                    self.stop_step = step
                    self.stop_elapsed_sec = elapsed_sec
                    self.threshold_step = self._candidate_step
                    self.threshold_elapsed_sec = self._candidate_elapsed_sec
                    return True
                # Threshold hit but streak not long enough → skip plateau
                return False
            else:
                self._reset_hits()

        # ── Rule 2: Plateau (fallback) ────────────────────────
        if self.plateau_patience is not None:
            if is_meaningful_improvement:
                self._reset_no_improve()
            else:
                self._no_improve_count += 1
                if self._no_improve_count >= self.plateau_patience:
                    self.stop_reason = "plateau"
                    self.stop_step = step
                    self.stop_elapsed_sec = elapsed_sec
                    return True

        return False

    def finalize(self, reason: str, step: int, elapsed_sec: float) -> None:
        """Set terminal stop metadata if no earlier stop has been recorded."""
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stop_step = step
            self.stop_elapsed_sec = elapsed_sec

    def result_dict(self) -> dict:
        """Export top-level result metadata."""
        return {
            "converged": self.converged,
            "stop_reason": self.stop_reason or "max_steps",
            "stop_step": self.stop_step,
            "stop_elapsed_sec": self.stop_elapsed_sec,
            "threshold_step": self.threshold_step,
            "threshold_elapsed_sec": self.threshold_elapsed_sec,
            "best_step": self.best_step,
            "best_elapsed_sec": self.best_elapsed_sec,
            "monitor": self.monitor,
            "best_metric_value": self.best_metric_value,
            "best_euler_residual_val": (
                self.best_metric_value
                if self.monitor == "euler_residual_val" else None
            ),
        }


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
# All evaluation functions accept the flattened dataset format:
#     {s_endo: (N, endo_dim), z: (N, exo_dim),
#      z_next_main: (N, exo_dim), z_next_fork: (N, exo_dim)}
# ---------------------------------------------------------------------------

def evaluate_euler_residual(env, policy, val_dataset: dict,
                            temperature: float = 1e-6) -> float:
    """Mean absolute Euler residual on the validation dataset.

    Computes the one-step Euler condition using pre-computed z transitions.
    Returns NaN if the environment does not support ER.

    Args:
        env:         MDPEnvironment instance.
        policy:      Policy network (callable: s -> a).
        val_dataset: Flattened dataset dict with keys:
                     s_endo, z, z_next_main.
        temperature: Smooth-gate temperature.

    Returns:
        float: mean absolute residual, or nan if ER not supported.
    """
    try:
        s_endo = val_dataset["s_endo"]
        z      = val_dataset["z"]
        z_next = val_dataset["z_next_main"]

        s      = env.merge_state(s_endo, z)
        a      = policy(s, training=False)

        k_next = env.endogenous_transition(s_endo, a, z)
        s_next = env.merge_state(k_next, z_next)
        a_next = policy(s_next, training=False)

        residual = env.euler_residual(s, a, s_next, a_next,
                                      temperature=temperature)
        return float(tf.reduce_mean(tf.abs(residual)))
    except NotImplementedError:
        return float("nan")


def evaluate_bellman_residual(env, policy, value_net, val_dataset: dict,
                                temperature: float = 1e-6) -> float:
    """Mean absolute Bellman residual V(s) - r(s,a) - γ·V(s') on val data.

    Args:
        env:         MDPEnvironment instance.
        policy:      Policy network.
        value_net:   State-value network V(s).
        val_dataset: Flattened dataset dict with keys:
                     s_endo, z, z_next_main.
        temperature: Smooth-gate temperature.

    Returns:
        float: mean absolute residual.
    """
    s_endo = val_dataset["s_endo"]
    z      = val_dataset["z"]
    z_next = val_dataset["z_next_main"]

    s = env.merge_state(s_endo, z)
    a = policy(s, training=False)

    r = env.reward(s, a, temperature=temperature)
    r = tf.squeeze(r) if r.shape.rank > 1 else r

    k_next = env.endogenous_transition(s_endo, a, z)
    s_next = env.merge_state(k_next, z_next)

    v_s    = tf.squeeze(value_net(s,      training=False))
    v_next = tf.squeeze(value_net(s_next, training=False))

    residual = v_s - r - env.discount() * v_next
    return float(tf.reduce_mean(tf.abs(residual)))
