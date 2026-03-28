"""Basic investment model as an MDPEnvironment.

State decomposition:
    s_endo = (k,)  — capital stock (endogenous, policy-controlled)
    s_exo  = (z,)  — productivity  (exogenous, AR(1) process)
    s      = (k, z)               — endo first, exo second

Action:
    a = (I,)  — investment.  Transition: k' = (1-δ)k + I.

Reward:
    r = profit(k, z) - adjustment_cost(k, k') - I_effective

Constraint consistency:
    _apply_action() is the single source of truth for the capital floor
    k' >= k_min. Both reward() and endogenous_transition() call it so the
    effective investment is always consistent with the constrained next capital.

Capital bounds (multipliers on frictionless k*):
    k_min = k_min_mult * k*   (default 0.25)
    k_max = k_max_mult * k*   (default 6.0)

Exogenous bounds (level space, ±m·σ_erg around mean):
    z_min = exp(μ - m·σ_erg)
    z_max = exp(μ + m·σ_erg)
    z sampled uniformly in [z_min, z_max] (level, not log-space).

Self-contained: this file includes ALL domain knowledge for the basic
investment model (parameters, production, costs).  The rest of the v2
codebase is model-agnostic.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from src.v2.environments.base import MDPEnvironment


# =====================================================================
# Domain-specific parameter containers
# =====================================================================

@dataclass(frozen=True)
class ShockParams:
    """Immutable AR(1) shock process parameters.

    log(z') = (1-ρ)μ + ρ·log(z) + σ·ε,   ε ~ N(0,1).
    """
    rho:   float = 0.7
    sigma: float = 0.15
    mu:    float = 0.0

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0. Got {self.sigma}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1). Got {self.rho}")


def _resolve_legacy_alias(primary_name: str, primary_value: float,
                          legacy_name: str, legacy_value: Optional[float],
                          default_value: float) -> float:
    """Resolve a new parameter name plus an optional legacy alias."""
    if legacy_value is None:
        return primary_value
    if primary_value != default_value:
        if primary_value != legacy_value:
            raise ValueError(
                f"Got conflicting values for {primary_name!r} and legacy alias "
                f"{legacy_name!r}: {primary_value} vs {legacy_value}."
            )
        return primary_value
    return legacy_value


@dataclass(frozen=True, init=False)
class EconomicParams:
    """Immutable economic primitives.

    Contains fields for both the basic and risky-debt models so that a
    single dataclass can be shared across environments.  The basic model
    uses only: interest_rate, depreciation_rate, production_elasticity,
    cost_convex, cost_fixed.

    Attributes:
        interest_rate:      Risk-free interest rate.
        depreciation_rate:  Depreciation rate.
        production_elasticity:
                            Production elasticity (Cobb-Douglas).
        cost_convex:        Convex adjustment cost coefficient (φ₀).
        cost_fixed:         Fixed adjustment cost coefficient (φ₁).
        tax:                Corporate income tax rate.
        cost_default:       Default / bankruptcy cost (α).
        cost_inject_fixed:  Fixed external equity cost (η₀).
        cost_inject_linear: Proportional external equity cost (η₁).
        frac_liquid:        Fraction of capital that can be liquidated.
    """
    # Core production
    interest_rate:         float = 0.04
    depreciation_rate:     float = 0.15
    production_elasticity: float = 0.7
    # Adjustment costs
    cost_convex:  float = 0.0
    cost_fixed:   float = 0.0
    # Risky debt (unused by basic model, kept for reuse by future envs)
    tax:                float = 0.3
    cost_default:       float = 0.4
    cost_inject_fixed:  float = 0.0
    cost_inject_linear: float = 0.0
    frac_liquid:        float = 0.5

    def __init__(
        self,
        interest_rate: float = 0.04,
        depreciation_rate: float = 0.15,
        production_elasticity: float = 0.7,
        cost_convex: float = 0.0,
        cost_fixed: float = 0.0,
        tax: float = 0.3,
        cost_default: float = 0.4,
        cost_inject_fixed: float = 0.0,
        cost_inject_linear: float = 0.0,
        frac_liquid: float = 0.5,
        r_rate: Optional[float] = None,
        delta: Optional[float] = None,
        theta: Optional[float] = None,
    ):
        interest_rate = _resolve_legacy_alias(
            "interest_rate", interest_rate, "r_rate", r_rate, 0.04)
        depreciation_rate = _resolve_legacy_alias(
            "depreciation_rate", depreciation_rate, "delta", delta, 0.15)
        production_elasticity = _resolve_legacy_alias(
            "production_elasticity",
            production_elasticity,
            "theta",
            theta,
            0.7,
        )

        object.__setattr__(self, "interest_rate", interest_rate)
        object.__setattr__(self, "depreciation_rate", depreciation_rate)
        object.__setattr__(self, "production_elasticity", production_elasticity)
        object.__setattr__(self, "cost_convex", cost_convex)
        object.__setattr__(self, "cost_fixed", cost_fixed)
        object.__setattr__(self, "tax", tax)
        object.__setattr__(self, "cost_default", cost_default)
        object.__setattr__(self, "cost_inject_fixed", cost_inject_fixed)
        object.__setattr__(self, "cost_inject_linear", cost_inject_linear)
        object.__setattr__(self, "frac_liquid", frac_liquid)
        self._validate()

    @property
    def r_rate(self) -> float:
        """Legacy alias for backward compatibility."""
        return self.interest_rate

    @property
    def delta(self) -> float:
        """Legacy alias for backward compatibility."""
        return self.depreciation_rate

    @property
    def theta(self) -> float:
        """Legacy alias for backward compatibility."""
        return self.production_elasticity

    def _validate(self):
        if not (0.0 < self.interest_rate < 1.0):
            raise ValueError(
                "interest_rate must be in (0, 1). "
                f"Got {self.interest_rate}")
        if not (0.0 <= self.depreciation_rate <= 1.0):
            raise ValueError(
                "depreciation_rate must be in [0, 1]. "
                f"Got {self.depreciation_rate}")
        if not (0.0 < self.production_elasticity < 1.0):
            raise ValueError(
                "production_elasticity must be in (0, 1). "
                f"Got {self.production_elasticity}")
        if self.cost_convex < 0:
            raise ValueError(f"cost_convex must be >= 0. Got {self.cost_convex}")
        if self.cost_fixed < 0:
            raise ValueError(f"cost_fixed must be >= 0. Got {self.cost_fixed}")
        

# =====================================================================
# Domain-specific economic functions (inlined from v1 economy/logic.py)
# =====================================================================

def _production(k, z, production_elasticity):
    """Cobb-Douglas: y = z · k^θ."""
    return z * (k ** production_elasticity)


def _adjustment_costs(k, investment, params):
    """Total adjustment costs: convex plus exact hard fixed cost."""
    safe_k = tf.maximum(k, 1e-8)
    adj_convex = 0.5 * params.cost_convex * tf.square(investment) / safe_k
    adj_fixed = params.cost_fixed * safe_k * tf.cast(
        tf.not_equal(investment, 0.0), tf.float32
    )
    return adj_convex + adj_fixed


# =====================================================================
# Environment
# =====================================================================

class BasicInvestmentEnv(MDPEnvironment):
    """Corporate finance basic model: optimal capital investment.

    The environment is the canonical economic model. When ``cost_fixed > 0``,
    the reward uses the exact hard indicator ``1{I_eff != 0}`` computed from
    the effective post-constraint investment. This is supported by the
    discrete solvers and intentionally rejected by the active NN trainers.

    Args:
        econ_params:       Economic parameters (interest, depreciation,
                           production elasticity, costs).
        shock_params:      AR(1) shock parameters (rho, sigma, mu).
        k_min_mult:        Lower capital bound as multiplier on k*.
        k_max_mult:        Upper capital bound as multiplier on k*.
        z_sd_mult:         Number of ergodic std devs for z bounds.
    """

    def __init__(
        self,
        econ_params:  EconomicParams = None,
        shock_params: ShockParams    = None,
        k_min_mult:   float          = 0.25,
        k_max_mult:   float          = 6.0,
        z_sd_mult:    float          = 3.0,
    ):
        self.econ   = econ_params  or EconomicParams()
        self.shocks = shock_params or ShockParams()
        self.beta   = 1.0 / (1.0 + self.econ.interest_rate)

        # z bounds in levels
        rho, sigma, mu = self.shocks.rho, self.shocks.sigma, self.shocks.mu
        sigma_ergodic = float(sigma / tf.sqrt(1.0 - rho ** 2))
        self.z_min = float(tf.exp(mu - z_sd_mult * sigma_ergodic))
        self.z_max = float(tf.exp(mu + z_sd_mult * sigma_ergodic))

        # k bounds
        self.k_star = float(compute_frictionless_policy(float(tf.exp(mu)), self.econ, self.shocks))
        self.k_min  = k_min_mult * self.k_star
        self.k_max  = k_max_mult * self.k_star

        # Analytical reference k'(z_max) — for analysis only
        self.k_ref = float(compute_frictionless_policy(self.z_max, self.econ, self.shocks))

        # Feasible investment bounds: k' = (1-δ)k + I with k' ∈ [k_min, k_max]
        self.I_min = self.k_min - (1.0 - self.econ.depreciation_rate) * self.k_max
        self.I_max = self.k_max - (1.0 - self.econ.depreciation_rate) * self.k_min

        # Stationary exogenous mean (level space)
        self._z_bar = tf.constant([float(tf.exp(mu))], dtype=tf.float32)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def exo_dim(self) -> int:
        return 1   # z

    def endo_dim(self) -> int:
        return 1   # k

    def action_dim(self) -> int:
        return 1   # I

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def action_bounds(self) -> tuple:
        return (
            tf.constant([self.I_min], dtype=tf.float32),
            tf.constant([self.I_max], dtype=tf.float32),
        )

    def action_scale_reference(self) -> tuple:
        """Policy output-head scaling: center=0, scale=max(|I_min|, I_max).

        Centering at zero keeps the policy output symmetric around the
        no-adjustment action and works well for the supported smooth model
        with convex costs only.
        """
        half_range = tf.constant([max(abs(self.I_min), self.I_max)],
                                 dtype=tf.float32)
        center     = tf.constant([0.0], dtype=tf.float32)
        return center, half_range

    # ------------------------------------------------------------------
    # Constraint helper — single source of truth
    # ------------------------------------------------------------------

    def _apply_action(self, k: tf.Tensor, a: tf.Tensor):
        """Compute constrained k_next and back-computed I_effective.

        Both reward() and endogenous_transition() call this to ensure
        accounting consistency when the capital floor binds.

        Args:
            k: capital,    shape (batch,) or (batch, 1).
            a: investment, shape (batch, action_dim).

        Returns:
            (k_next, I_effective): constrained capital and effective investment.
        """
        I      = a[..., 0]
        k_next = (1.0 - self.econ.depreciation_rate) * k + I
        k_next = tf.clip_by_value(k_next, self.k_min, self.k_max)
        I_eff  = k_next - (1.0 - self.econ.depreciation_rate) * k
        return k_next, I_eff

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def exogenous_transition(self, s_exo: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        """AR(1) step: z' = exp((1-ρ)μ + ρ·log(z) + σ·ε).

        Args:
            s_exo: shape (batch, 1) — current z.
            eps:   shape (batch, 1) — standard normal shock.

        Returns:
            z_next: shape (batch, 1).
        """
        z       = s_exo[..., 0]
        log_z   = tf.math.log(tf.maximum(z, 1e-8))
        log_z_next = (
            (1.0 - self.shocks.rho) * self.shocks.mu
            + self.shocks.rho * log_z
            + self.shocks.sigma * eps[..., 0]
        )
        return tf.reshape(tf.exp(log_z_next), [-1, 1])

    def endogenous_transition(
        self, s_endo: tf.Tensor, action: tf.Tensor, s_exo: tf.Tensor
    ) -> tf.Tensor:
        """Capital accumulation: k' = max((1-δ)k + I, k_min).

        z (s_exo) is not used for k evolution in this model but is
        accepted for interface consistency.

        Args:
            s_endo: shape (batch, 1) — current k.
            action: shape (batch, 1) — investment I.
            s_exo:  shape (batch, 1) — current z (unused here).

        Returns:
            k_next: shape (batch, 1).
        """
        k      = s_endo[..., 0]
        k_next, _ = self._apply_action(k, action)
        return tf.reshape(k_next, [-1, 1])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, s: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        """Cash flow: profit - adjustment_cost - I_effective.

        s = merge_state(k, z), so s[..., 0] = k, s[..., 1] = z.
        """
        k = s[..., 0]
        z = s[..., 1]
        _, investment = self._apply_action(k, a)

        profit = _production(k, z, self.econ.production_elasticity)
        adj_cost = _adjustment_costs(k, investment, self.econ)
        return profit - adj_cost - investment

    def stationary_exo(self) -> tf.Tensor:
        """Stationary mean of z: exp(μ), shape (1,)."""
        return self._z_bar

    def stationary_action(self, s_endo: tf.Tensor) -> tf.Tensor:
        """Action holding k constant: I = δ·k, shape (batch, 1)."""
        k = s_endo[..., 0:1]
        return self.econ.depreciation_rate * k

    def discount(self) -> float:
        return self.beta

    def validate_nn_training_support(self, trainer_name: str) -> None:
        if self.econ.cost_fixed > 0.0:
            raise ValueError(
                f"{trainer_name} does not support BasicInvestmentEnv with "
                "cost_fixed > 0. Use solve_vfi() or solve_pfi() for the "
                "fixed-cost model."
            )

    # ------------------------------------------------------------------
    # Initial state sampling
    # ------------------------------------------------------------------

    def sample_initial_endogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Uniform k from [k_min, k_max].

        Returns:
            shape (n, 1).
        """
        k = tf.random.stateless_uniform(
            [n], seed=seed, minval=self.k_min, maxval=self.k_max,
            dtype=tf.float32)
        return tf.reshape(k, [n, 1])

    def sample_initial_exogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Uniform z from [z_min, z_max] in level space.

        Sampling in levels (not log-space) gives more weight to the
        upper tail of z, covering extreme states near the boundary.

        Returns:
            shape (n, 1).
        """
        z = tf.random.stateless_uniform(
            [n], seed=seed, minval=self.z_min, maxval=self.z_max,
            dtype=tf.float32)
        return tf.reshape(z, [n, 1])

    # ------------------------------------------------------------------
    # Analytical overrides
    # ------------------------------------------------------------------

    def grid_spec(self):
        """Grid discretization hints for discrete solvers (VFI/PFI).

        Capital (endo): log-spaced grid. Dense where curvature is highest
            (low k) and sparse where the value function is nearly linear
            (high k). Grid size is always exactly the user-specified
            endo_sizes value.

        Productivity (exo): log-spaced grid. z follows a log-AR(1) process,
            so log-spacing gives uniform resolution in the natural (log)
            coordinate of the process. Grid values are in levels (matching
            the v2 convention that all z data is in levels).

        Investment (action): linear grid over [I_min, I_max] where
            I_min = k_min − (1−δ)·k_max and I_max = k_max − (1−δ)·k_min
            are the tightest state-independent feasible bounds.
        """
        from src.v2.solvers.grid import GridAxis
        action_axis = GridAxis(self.I_min, self.I_max, spacing="linear")
        if self.econ.cost_fixed > 0:
            action_axis = GridAxis(
                self.I_min, self.I_max, spacing="zero_power", power=2.0)
        return {
            "endo":   [GridAxis(self.k_min, self.k_max, spacing="geometric")],
            "exo":    [GridAxis(self.z_min, self.z_max, spacing="log")],
            "action": [action_axis],
        }

    def reward_scale(self, **_kwargs) -> float:
        """Analytical λ = 1 / |V*(k*, z_mean)|."""
        s_endo_ss = tf.constant([[self.k_star]])
        abs_v = max(float(tf.abs(self.terminal_value(s_endo_ss)[0])), 1e-8)
        return 1.0 / abs_v

    def euler_residual(
        self, s: tf.Tensor, a: tf.Tensor,
        s_next: tf.Tensor, a_next: tf.Tensor,
    ) -> tf.Tensor:
        """Unit-free Euler residual: 1 - β·m/χ.

        χ = 1 + ∂ψ/∂k'        (marginal cost of current investment)
        m = π_k' - ∂ψ'/∂k' + (1-δ)·χ'  (marginal benefit next period)

        At optimum: E[residual] = 0.
        """
        if self.econ.cost_fixed > 0.0:
            raise NotImplementedError(
                "BasicInvestmentEnv.euler_residual() is only supported when "
                "cost_fixed == 0. The fixed-cost model should be solved with "
                "solve_vfi() or solve_pfi()."
            )

        k      = s[..., 0]
        z_next = s_next[..., 1]
        k_next, _      = self._apply_action(k, a)
        k_next_next, _ = self._apply_action(k_next, a_next)

        # π_k' = θ·z'·k'^(θ-1)
        safe_k = tf.maximum(k_next, 1e-8)
        pi_k = (
            self.econ.production_elasticity
            * z_next
            * tf.pow(safe_k, self.econ.production_elasticity - 1)
        )

        # χ = 1 + ∂ψ(k, k')/∂k'
        with tf.GradientTape() as tape:
            tape.watch(k_next)
            investment = k_next - (1.0 - self.econ.depreciation_rate) * k
            psi = _adjustment_costs(k, investment, self.econ)
        chi = 1.0 + tape.gradient(psi, k_next)

        # m via derivatives of next-period adjustment cost
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(k_next)
            tape.watch(k_next_next)
            investment_next = (
                k_next_next - (1.0 - self.econ.depreciation_rate) * k_next
            )
            psi_next = _adjustment_costs(k_next, investment_next, self.econ)
        dpsi_next_dk = tape.gradient(psi_next, k_next)
        chi_next     = 1.0 + tape.gradient(psi_next, k_next_next)
        del tape

        m = pi_k - dpsi_next_dk + (1.0 - self.econ.depreciation_rate) * chi_next
        return 1.0 - self.beta * m / chi

    def analytical_policy(
        self, s: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Frictionless optimal investment: I*(k, z) = k'*(z) - (1-δ)·k.

        k'*(z) = [θ · E[z'|z] / (r+δ)]^(1/(1-θ)),
        where E[z'|z] = exp((1-ρ)μ + ρ·log(z) + σ²/2).

        Valid only for the frictionless model (cost_fixed == 0, cost_convex == 0).

        Args:
            s:        state tensor, shape (batch, state_dim).
            training: unused, included for interface consistency.

        Returns:
            I*: optimal investment, shape (batch, 1).
        """
        k = s[..., 0]
        z = s[..., 1]
        rho   = tf.constant(self.shocks.rho,   dtype=tf.float32)
        sigma = tf.constant(self.shocks.sigma,  dtype=tf.float32)
        mu    = tf.constant(self.shocks.mu,     dtype=tf.float32)
        theta = tf.constant(self.econ.production_elasticity, dtype=tf.float32)
        r_delta = tf.constant(
            self.econ.interest_rate + self.econ.depreciation_rate, dtype=tf.float32)
        delta = tf.constant(self.econ.depreciation_rate, dtype=tf.float32)

        log_ez = (1.0 - rho) * mu + rho * tf.math.log(tf.maximum(z, 1e-8)) + 0.5 * sigma ** 2
        ez = tf.exp(log_ez)
        kprime = tf.pow(theta * ez / r_delta, 1.0 / (1.0 - theta))
        kprime = tf.clip_by_value(kprime, self.k_min, self.k_max)
        I_star = kprime - (1.0 - delta) * k
        return tf.reshape(I_star, tf.concat([tf.shape(k), [1]], axis=0))


# =====================================================================
# Analytical benchmark (module-level, for notebook validation)
# =====================================================================

def compute_frictionless_policy(z, econ_params, shock_params):
    """Vectorized frictionless optimal k'(z).

    k'(z) = [θ · E[z'|z] / (r + δ)]^(1/(1-θ))

    where E[z'|z] = exp((1-ρ)μ + 0.5σ²) · z^ρ.

    Args:
        z:            Productivity level(s), scalar or array.
        econ_params:  EconomicParams with production_elasticity,
                      interest_rate, depreciation_rate.
        shock_params: ShockParams with rho, sigma, mu.

    Returns:
        k'(z) in levels, same shape as z.
    """
    import numpy as np
    rho, sigma, mu = shock_params.rho, shock_params.sigma, shock_params.mu
    exp_corr = np.exp((1 - rho) * mu + 0.5 * sigma ** 2)
    return (
        (econ_params.production_elasticity * exp_corr * np.asarray(z) ** rho
         / (econ_params.interest_rate + econ_params.depreciation_rate))
        ** (1.0 / (1.0 - econ_params.production_elasticity))
    )
