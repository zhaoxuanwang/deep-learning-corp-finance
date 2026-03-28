"""Abstract MDP environment interface for v2 generic framework.

Exogenous / Endogenous state decomposition
------------------------------------------
The full state s = [s_endo, s_exo] is partitioned into:

    s_endo  (endo_dim): endogenous state (e.g. capital k, debt b).
            Evolves via the policy through endogenous_transition().
            Not pre-computable without the policy.

    s_exo   (exo_dim):  exogenous state (e.g. productivity z).
            Evolves via known dynamics through exogenous_transition().
            Can be pre-computed by DataGenerator without the policy.

Convention: s = concat([s_endo, s_exo], axis=-1).
    merge_state(s_endo, s_exo) -> s
    split_state(s)             -> (s_endo, s_exo)

Derived defaults
----------------
The following methods have default implementations that compose the
primitive exo/endo methods:

    state_dim()     = endo_dim() + exo_dim()
    merge_state()   = tf.concat([s_endo, s_exo], axis=-1)
    split_state()   = s[..., :endo_dim()], s[..., endo_dim():]
    transition()    = merge_state(endogenous_transition(...),
                                  exogenous_transition(...))
    sample_initial_states() = merge_state(sample_initial_endogenous(),
                                          sample_initial_exogenous())

Implementation note — action consistency
-----------------------------------------
When transition() applies safety constraints (e.g. state floors or
ceilings), reward() must use the *effective* action implied by the
constrained state, not the raw policy output. Subclasses should
centralize constraint logic in a shared private method called by both
reward() and endogenous_transition().
"""

from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf


class MDPEnvironment(ABC):
    """Abstract base for deterministic-dynamics MDPs with exo/endo decomposition."""

    # ------------------------------------------------------------------
    # Dimensions — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def exo_dim(self) -> int:
        """Dimension of the exogenous state (e.g. productivity z)."""

    @abstractmethod
    def endo_dim(self) -> int:
        """Dimension of the endogenous state (e.g. capital k)."""

    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action vector."""

    def state_dim(self) -> int:
        """Total state dimension = endo_dim + exo_dim."""
        return self.endo_dim() + self.exo_dim()

    def shock_dim(self) -> int:
        """Dimension of the exogenous shock vector (default 1)."""
        return 1

    # ------------------------------------------------------------------
    # State composition helpers — default implementations
    # ------------------------------------------------------------------

    def merge_state(self, s_endo: tf.Tensor, s_exo: tf.Tensor) -> tf.Tensor:
        """Combine endogenous and exogenous parts into a full state vector.

        Convention: s = [s_endo | s_exo].

        Args:
            s_endo: shape (..., endo_dim)
            s_exo:  shape (..., exo_dim)

        Returns:
            s: shape (..., state_dim)
        """
        return tf.concat([s_endo, s_exo], axis=-1)

    def split_state(self, s: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Split full state into (s_endo, s_exo).

        Args:
            s: shape (..., state_dim)

        Returns:
            (s_endo, s_exo): shapes (..., endo_dim) and (..., exo_dim)
        """
        d = self.endo_dim()
        return s[..., :d], s[..., d:]

    # ------------------------------------------------------------------
    # Transitions — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def exogenous_transition(
        self, s_exo: tf.Tensor, eps: tf.Tensor
    ) -> tf.Tensor:
        """One-step exogenous state transition: z' = f_exo(z, eps).

        Used by DataGenerator to pre-compute z trajectories without
        running the policy.

        Args:
            s_exo: current exogenous state, shape (batch, exo_dim).
            eps:   shock vector,             shape (batch, shock_dim).

        Returns:
            s_exo_next: shape (batch, exo_dim).
        """

    @abstractmethod
    def endogenous_transition(
        self, s_endo: tf.Tensor, action: tf.Tensor, s_exo: tf.Tensor
    ) -> tf.Tensor:
        """One-step endogenous state transition: k' = f_endo(k, a, z).

        Used inside trainers to evolve the endogenous state under the
        current policy. Gradients flow through this function.

        Args:
            s_endo: current endogenous state, shape (batch, endo_dim).
            action: policy action,            shape (batch, action_dim).
            s_exo:  current exogenous state,  shape (batch, exo_dim).
                    Included for generality; may be unused for models
                    where k' does not depend on z.

        Returns:
            s_endo_next: shape (batch, endo_dim).
        """

    def transition(
        self, s: tf.Tensor, a: tf.Tensor, eps: tf.Tensor
    ) -> tf.Tensor:
        """Full state transition s' = f(s, a, eps).

        Default: compose endogenous_transition and exogenous_transition.
        Used primarily by MVE and for evaluation.

        Args:
            s:   current state, shape (batch, state_dim).
            a:   action,        shape (batch, action_dim).
            eps: shock,         shape (batch, shock_dim).

        Returns:
            s_next: shape (batch, state_dim).
        """
        s_endo, s_exo = self.split_state(s)
        s_endo_next = self.endogenous_transition(s_endo, a, s_exo)
        s_exo_next  = self.exogenous_transition(s_exo, eps)
        return self.merge_state(s_endo_next, s_exo_next)

    # ------------------------------------------------------------------
    # Reward and discount — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def reward(self, s: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        """Scalar reward r(s, a).

        Args:
            s: state tensor, shape (batch, state_dim).
            a: action tensor, shape (batch, action_dim).

        Returns:
            Reward tensor, shape (batch,) or (batch, 1).
        """

    @abstractmethod
    def discount(self) -> float:
        """Discount factor gamma = 1 / (1 + r)."""

    # ------------------------------------------------------------------
    # Action space — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def action_bounds(self) -> tuple:
        """Clip bounds for actions (may be ±inf).

        Returns:
            (action_low, action_high): 1-D tensors of shape (action_dim,).
        """

    def action_scale_reference(self) -> tuple:
        """Reference (center, half_range) for policy output-head scaling.

        Default derives from action_bounds() (requires finite bounds).
        Override when action_bounds() contains infinite values.
        """
        low, high = self.action_bounds()
        if not (tf.reduce_all(tf.math.is_finite(low))
                and tf.reduce_all(tf.math.is_finite(high))):
            raise ValueError(
                f"{self.__class__.__name__}.action_bounds() contains non-finite "
                f"values. Override action_scale_reference().")
        return (low + high) / 2.0, (high - low) / 2.0

    def action_spec(self) -> dict:
        """Full action-space specification for PolicyNetwork construction."""
        low, high = self.action_bounds()
        center, half_range = self.action_scale_reference()
        return dict(
            action_low=low, action_high=high,
            action_center=center, action_half_range=half_range,
        )

    # ------------------------------------------------------------------
    # Initial state sampling — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_initial_endogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Sample n initial endogenous states (e.g. uniform k).

        Args:
            n:    number of samples.
            seed: stateless seed tensor of shape [2], dtype tf.int32.

        Returns:
            s_endo_0: shape (n, endo_dim).
        """

    @abstractmethod
    def sample_initial_exogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Sample n initial exogenous states (e.g. uniform z in level space).

        Args:
            n:    number of samples.
            seed: stateless seed tensor of shape [2], dtype tf.int32.

        Returns:
            s_exo_0: shape (n, exo_dim).
        """

    def sample_initial_states(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Sample n full initial states (derived — merges endo and exo).

        Uses the same seed for both components via a fixed offset for the
        exogenous part. Prefer sample_initial_endogenous / exogenous
        separately when different seeds are available (e.g. via SeedSchedule).

        Args:
            n:    number of samples.
            seed: stateless seed tensor, shape [2], dtype tf.int32.

        Returns:
            s0: shape (n, state_dim).
        """
        seed_exo = tf.stack([seed[0] + 1, seed[1]])
        s_endo = self.sample_initial_endogenous(n, seed)
        s_exo  = self.sample_initial_exogenous(n, seed_exo)
        return self.merge_state(s_endo, s_exo)

    def sample_shocks(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Sample n exogenous shocks (default: N(0,1), shape (n, shock_dim)).

        Used by MVE trainer and reward_scale. Override if the
        shock distribution is non-Gaussian.

        Args:
            n:    number of samples.
            seed: stateless seed tensor, shape [2], dtype tf.int32.

        Returns:
            eps: shape (n, shock_dim).
        """
        return tf.random.stateless_normal(
            [n, self.shock_dim()], seed=seed, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Optional methods — override as needed
    # ------------------------------------------------------------------

    def grid_spec(self):
        """Per-variable grid discretization hints for discrete solvers.

        Returns a dict with keys 'endo', 'exo', 'action', each mapping to
        a list of GridAxis (one per variable in that group).  The solver
        reads this spec and builds grids mechanically — all domain knowledge
        stays inside the environment.

        Common spacing options:
            - "linear": uniform spacing (default for actions).
            - "log":    denser at low values in geometric percentage steps.
            - "geometric": alias for "log", often clearer for capital grids.

        Return None (default) to let the solver fall back to linspace grids
        derived from action_bounds() and sampled state ranges.

        Example override::

            from src.v2.solvers.grid import GridAxis
            def grid_spec(self):
                return {
                    "endo":   [GridAxis(self.k_min, self.k_max, spacing="geometric")],
                    "exo":    [GridAxis(self.z_min, self.z_max, spacing="log")],
                    "action": [GridAxis(-self.k_max, self.k_max)],
                }

        Returns:
            Dict[str, List[GridAxis]] or None.
        """
        return None

    def euler_residual(
        self, s: tf.Tensor, a: tf.Tensor,
        s_next: tf.Tensor, a_next: tf.Tensor,
    ) -> tf.Tensor:
        """Model-specific Euler equation residual for the ER method.

        Raises:
            NotImplementedError: if ER is not available for this model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support the ER method.")

    def validate_nn_training_support(self, trainer_name: str) -> None:
        """Raise when the environment is unsupported by the active NN trainers."""
        del trainer_name

    def analytical_policy(
        self, s: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Closed-form optimal policy a*(s), if available.

        Optional benchmark for eval metrics (policy_mae).  Not required for
        training — trainers must never call this method.

        Args:
            s:        state tensor, shape (batch, state_dim).
            training: included for interface consistency with PolicyNetwork.

        Returns:
            a*: shape (batch, action_dim).

        Raises:
            NotImplementedError: if no analytical policy is available.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement analytical_policy().")

    def reward_scale(
        self, n_samples: int = 1000, seed: tf.Tensor = None
    ) -> float:
        """Reward normalizer λ = (1-γ) / E[|r|].

        Default: MC estimate over random (s, a) pairs.
        Override with an analytical estimate when available.
        If seed is None, a fixed internal seed [0, 0] is used — sufficient
        for computing a normalization constant.
        """
        if seed is None:
            seed = tf.constant([0, 0], dtype=tf.int32)
        s = self.sample_initial_states(n_samples, seed=seed)
        low, high = self.action_bounds()
        a = tf.random.stateless_uniform(
            [n_samples, self.action_dim()],
            seed=tf.stack([seed[0] + 2, seed[1]]),
            minval=low, maxval=high)
        r = self.reward(s, a)
        r = tf.reshape(r, [-1]) if r.shape.rank > 1 else r
        mean_abs_r = max(float(tf.reduce_mean(tf.abs(r))), 1e-8)
        return (1.0 - self.discount()) / mean_abs_r

    def stationary_exo(self) -> tf.Tensor:
        """Stationary mean of the exogenous state, shape (exo_dim,).

        Used by terminal_value (LR method) to construct s̄ = [s_endo | s̄_exo].
        Override in subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stationary_exo().")

    def stationary_action(self, s_endo: tf.Tensor) -> tf.Tensor:
        """Action that holds the endogenous state constant: f_endo(s_endo, ā) = s_endo.

        Args:
            s_endo: shape (batch, endo_dim).

        Returns:
            ā: shape (batch, action_dim).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stationary_action().")

    def continuation_transform(self, v_next: tf.Tensor) -> tf.Tensor:
        """Transform continuation value before use in Bellman/bootstrap.

        Override to apply limited liability (ReLU) or other transforms.
        Default: identity (standard MDP).

        Args:
            v_next: Raw value predictions, shape (batch,) or (batch, 1).

        Returns:
            Transformed values, same shape.
        """
        return v_next

    # ------------------------------------------------------------------
    # Terminal value — optional, for LR method
    # ------------------------------------------------------------------

    def terminal_value(self, s_endo: tf.Tensor) -> tf.Tensor:
        """Analytical terminal value V^term(s_endo) = r(s̄, ā) / (1-γ).

        LR-method specific.  This is an analytical steady-state perpetuity
        used by the Lifetime Reward (LR) trainer to approximate continuation
        value beyond the rollout horizon.  It does NOT involve a learned
        value network — methods that learn V_φ (SHAC, BRM) use the network
        bootstrap instead and do not call this method.

        Constructs s̄ = [s_endo | s̄_exo] and ā = stationary_action(s_endo),
        then computes the steady-state perpetuity.  Gradients flow through
        s_endo but not through the policy network.

        Args:
            s_endo: terminal endogenous state, shape (batch, endo_dim).

        Returns:
            Scalar terminal value per sample, shape (batch,).
        """
        z_bar = self.stationary_exo()
        z_bar = tf.broadcast_to(z_bar, tf.concat([tf.shape(s_endo)[:-1],
                                                   tf.shape(z_bar)], 0))
        s_bar = self.merge_state(s_endo, z_bar)
        a_bar = self.stationary_action(s_endo)
        r = self.reward(s_bar, a_bar)
        r = tf.squeeze(r, axis=-1) if r.shape.rank > 1 else r
        return r / (1.0 - self.discount())
