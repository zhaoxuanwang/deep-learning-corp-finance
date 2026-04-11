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

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from scipy.special import ndtri
import tensorflow as tf

from src.v2.data.generator import DataGenerator, DataGeneratorConfig
from src.v2.data.rng import SeedSchedule, SeedScheduleConfig, VariableID
from src.v2.estimation.gmm import GMMSpec
from src.v2.estimation.smm import (
    SMMPanelMoments,
    SMMRunConfig,
    SMMSpec,
    SMMTargetMoments,
    _panel_iv_first_diff_ar1,
    _panel_serial_correlation,
)
from src.v2.evaluation.policies import (
    build_action_grid_policy,
    restore_selected_snapshot,
)
from src.v2.environments.base import MDPEnvironment
from src.v2.networks.policy import PolicyNetwork
from src.v2.solvers import PFIConfig, solve_pfi
from src.v2.trainers.config import ERConfig, NetworkConfig
from src.v2.trainers.er import train_er
from src.v2.trainers.core import evaluate_euler_residual
from src.v2.utils.seeding import fold_in_seed


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
        self.k_min_mult = float(k_min_mult)
        self.k_max_mult = float(k_max_mult)
        self.z_sd_mult = float(z_sd_mult)

        # z bounds in levels
        rho, sigma, mu = self.shocks.rho, self.shocks.sigma, self.shocks.mu
        sigma_ergodic = float(sigma / tf.sqrt(1.0 - rho ** 2))
        self.z_min = float(tf.exp(mu - self.z_sd_mult * sigma_ergodic))
        self.z_max = float(tf.exp(mu + self.z_sd_mult * sigma_ergodic))

        # k bounds
        self.k_star = float(compute_frictionless_policy(float(tf.exp(mu)), self.econ, self.shocks))
        self.k_min  = self.k_min_mult * self.k_star
        self.k_max  = self.k_max_mult * self.k_star

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

    # ------------------------------------------------------------------
    # SMM helpers
    # ------------------------------------------------------------------

    def smm_parameter_names(
        self,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
    ) -> tuple[str, ...]:
        mode = _resolve_basic_investment_stage(mode)
        if mode == "stage_a":
            return ("alpha", "rho", "sigma_epsilon")
        return ("alpha", "psi1", "rho", "sigma_epsilon")

    def smm_moment_names(
        self,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
    ) -> tuple[str, ...]:
        mode = _resolve_basic_investment_stage(mode)
        if mode == "stage_a":
            # income_ar1_beta is NOT included for Stage A: in the frictionless
            # model, log(income_ratio) = sigma*eps + const (white noise) because
            # instant capital adjustment absorbs all z persistence.  AR1_beta
            # is identically zero regardless of rho and cannot identify it.
            # serial_corr(I/k) replaces it as the ρ-identifying moment.
            return (
                "mean_investment_assets",
                "var_investment_assets",
                "serial_corr_investment",
                "income_ar1_resid_std",
            )
        return (
            "mean_investment_assets",
            "var_investment_assets",
            "serial_corr_investment",
            "income_ar1_beta",
            "income_ar1_resid_std",
        )

    def smm_default_bounds(
        self,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
    ) -> tuple[tuple[float, float], ...]:
        mode = _resolve_basic_investment_stage(mode)
        if mode == "stage_a":
            return (
                (0.10, 0.95),   # alpha
                (-0.95, 0.95),  # rho
                (0.01, 1.0),    # sigma_epsilon
            )
        return (
            (0.10, 0.95),   # alpha
            (0.0, 1.0),     # psi1
            (-0.95, 0.95),  # rho
            (0.01, 1.0),    # sigma_epsilon
        )

    def smm_initial_guess(
        self,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
    ) -> np.ndarray:
        mode = _resolve_basic_investment_stage(mode)
        if mode == "stage_a":
            return np.array(
                [
                    self.econ.production_elasticity,
                    self.shocks.rho,
                    self.shocks.sigma,
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                self.econ.production_elasticity,
                self.econ.cost_convex,
                self.shocks.rho,
                self.shocks.sigma,
            ],
            dtype=np.float64,
        )

    def smm_true_beta(
        self,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
    ) -> np.ndarray:
        """Return the env's current structural parameters as the ground-truth β.

        This is the vector that SMM should recover during Monte Carlo
        validation.  Pass it explicitly to ``validate_smm`` as ``beta_true``.

        Do NOT pass it as ``initial_guess`` to ``make_smm_spec`` in
        validation runs — starting from the true β defeats the test.  The
        default initial guess in ``make_smm_spec`` is the midpoint of the
        parameter bounds, which is the recommended starting point for SMM.
        """
        return self.smm_initial_guess(mode=mode)

    def make_smm_spec(
        self,
        initial_guess: Sequence[float] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
        solver_config: BasicInvestmentSMMSolverConfig | None = None,
    ) -> SMMSpec:
        """Build an env-owned SMM spec for the requested Stage A/B mode.

        Args:
            initial_guess: Starting point for the optimizer.  Defaults to
                           the midpoint of each parameter's bounds, which
                           provides a neutral starting point independent of
                           the true parameter values.  Pass
                           ``env.smm_true_beta()`` explicitly only for
                           oracle debugging; starting from a wrong point is
                           the standard validation test for SMM correctness.
            bounds:        Search region for each parameter.  Defaults to
                           ``smm_default_bounds()``.  Tighten if you have
                           economically motivated prior ranges.
            mode:          ``'stage_a'`` (analytical/frictionless policy)
                           or ``'stage_b'`` (numerical solver in the loop).
            solver_config: Stage-B solver settings.  Ignored for stage_a.
        """
        mode = _resolve_basic_investment_stage(mode)
        if mode == "stage_a":
            _validate_basic_investment_stage_a_support(self)
        else:
            _validate_basic_investment_stage_b_support(self)
            solver_config = solver_config or BasicInvestmentSMMSolverConfig()

        resolved_bounds = (
            self.smm_default_bounds(mode=mode)
            if bounds is None
            else tuple(tuple(map(float, pair)) for pair in bounds)
        )
        if initial_guess is None:
            # Default: midpoint of bounds — neutral start independent of truth.
            guess = np.array(
                [0.5 * (lo + hi) for lo, hi in resolved_bounds],
                dtype=np.float64,
            )
        else:
            guess = np.asarray(initial_guess, dtype=np.float64)
        def _simulate_panel_moments(beta, run_config, seed):
            bundle = None
            if mode == "stage_b":
                bundle = self.solve_smm_policy_bundle(
                    beta=beta,
                    solver_config=solver_config,
                    seed=fold_in_seed(seed, "solver"),
                )
            candidate_env = _basic_investment_env_from_beta(
                self, beta, mode=mode
            )
            return candidate_env.simulate_smm_panel_moments(
                beta=beta,
                run_config=run_config,
                seed=seed,
                mode=mode,
                solver_config=solver_config,
                policy_bundle=bundle,
            )

        return SMMSpec(
            parameter_names=self.smm_parameter_names(mode=mode),
            moment_names=self.smm_moment_names(mode=mode),
            bounds=resolved_bounds,
            initial_guess=guess,
            simulate_panel_moments=_simulate_panel_moments,
            simulate_target_moments=lambda beta, run_config, seed: (
                self.simulate_smm_target_moments(
                    beta=beta,
                    run_config=run_config,
                    seed=seed,
                    mode=mode,
                    solver_config=solver_config,
                )
            ),
        )

    def solve_smm_policy_bundle(
        self,
        beta: Sequence[float],
        solver_config: BasicInvestmentSMMSolverConfig | None,
        seed: tuple[int, int],
        value_init=None,
    ) -> BasicInvestmentSMMSolverBundle:
        """Solve/train one candidate Stage-B policy bundle for SMM rollout.

        Args:
            value_init: Optional warm-start value function from a previous
                        solve at nearby parameters.  Passed through to the
                        PFI/VFI solver.  Ignored for ER method.
        """

        _validate_basic_investment_stage_b_support(self)
        solver_config = solver_config or BasicInvestmentSMMSolverConfig()
        candidate_env = _basic_investment_env_from_beta(
            self,
            beta,
            mode="stage_b",
        )
        if solver_config.method == "ER":
            return _solve_basic_investment_er_bundle(
                candidate_env,
                beta=np.asarray(beta, dtype=np.float64),
                solver_config=solver_config,
                seed=tuple(map(int, seed)),
            )
        if solver_config.method == "PFI":
            return _solve_basic_investment_pfi_bundle(
                candidate_env,
                beta=np.asarray(beta, dtype=np.float64),
                solver_config=solver_config,
                seed=tuple(map(int, seed)),
                value_init=value_init,
            )
        raise ValueError(
            "Unsupported BasicInvestment Stage-B solver method "
            f"{solver_config.method!r}."
        )

    def simulate_smm_panel_data(
        self,
        beta: Sequence[float],
        run_config: SMMRunConfig,
        seed: tuple[int, int],
        *,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
        solver_config: BasicInvestmentSMMSolverConfig | None = None,
        policy_bundle: BasicInvestmentSMMSolverBundle | None = None,
    ) -> BasicInvestmentSMMPanelData:
        """Simulate raw panel data for either the analytical or solve-in-loop path."""

        mode = _resolve_basic_investment_stage(mode)
        candidate_env = _basic_investment_env_from_beta(self, beta, mode=mode)
        panel_seed = tuple(map(int, seed))

        if mode == "stage_a":
            _validate_basic_investment_stage_a_support(candidate_env)
            if policy_bundle is not None:
                raise ValueError(
                    "policy_bundle is only supported for BasicInvestment Stage-B SMM."
                )
            next_capital_policy = _build_analytical_next_capital_policy(candidate_env)
            extra_metadata = {
                "mode": mode,
                "seed": panel_seed,
            }
        else:
            _validate_basic_investment_stage_b_support(candidate_env)
            if solver_config is None and policy_bundle is not None:
                solver_config = BasicInvestmentSMMSolverConfig(
                    method=policy_bundle.method
                )
            solver_config = solver_config or BasicInvestmentSMMSolverConfig()
            if policy_bundle is None:
                policy_bundle = self.solve_smm_policy_bundle(
                    beta=beta,
                    solver_config=solver_config,
                    seed=fold_in_seed(panel_seed, "solver"),
                )
            else:
                _validate_basic_investment_policy_bundle(beta, solver_config, policy_bundle)
            next_capital_policy = policy_bundle.next_capital_policy
            panel_seed = fold_in_seed(panel_seed, "panel")
            extra_metadata = {
                "mode": mode,
                "seed": panel_seed,
                "solver_method": policy_bundle.method,
                "solver_bundle": _summarize_basic_investment_solver_bundle(policy_bundle),
            }

        return _simulate_basic_smm_panel_data(
            candidate_env,
            run_config=run_config,
            n_panels=run_config.n_sim_panels,
            seed=panel_seed,
            next_capital_policy=next_capital_policy,
            metadata=extra_metadata,
        )

    def compute_smm_panel_moments(
        self,
        panel_data: "BasicInvestmentSMMPanelData",
    ) -> dict[str, Any]:
        """Compute env-owned Stage A/B SMM moments from raw panel data."""

        mode = _resolve_basic_investment_stage(
            panel_data.metadata.get("mode", "stage_a")
        )
        panel_moments, diagnostics = _compute_basic_smm_panel_moments(
            self,
            panel_data,
            mode=mode,
        )
        return {
            "moment_names": self.smm_moment_names(mode=mode),
            "panel_moments": panel_moments,
            "average_moments": np.mean(panel_moments, axis=0),
            "diagnostics": diagnostics,
            "n_observations": int(panel_data.n_observations),
            "mode": mode,
        }

    def simulate_smm_panel_moments(
        self,
        beta: Sequence[float],
        run_config: SMMRunConfig,
        seed: tuple[int, int],
        *,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
        solver_config: BasicInvestmentSMMSolverConfig | None = None,
        policy_bundle: BasicInvestmentSMMSolverBundle | None = None,
    ) -> SMMPanelMoments:
        """Simulate per-panel SMM moments under the requested Stage A/B path."""

        panel_data = self.simulate_smm_panel_data(
            beta=beta,
            run_config=run_config,
            seed=seed,
            mode=mode,
            solver_config=solver_config,
            policy_bundle=policy_bundle,
        )
        resolved_mode = _resolve_basic_investment_stage(mode)
        panel_moments, diagnostics = _compute_basic_smm_panel_moments(
            self,
            panel_data,
            mode=resolved_mode,
        )
        return SMMPanelMoments(
            panel_moments=panel_moments,
            n_observations=panel_data.n_observations,
            metadata={
                "diagnostics": diagnostics,
                "seed": tuple(map(int, panel_data.metadata.get("seed", seed))),
                "mode": resolved_mode,
                "solver_method": panel_data.metadata.get("solver_method"),
                "solver_bundle": panel_data.metadata.get("solver_bundle"),
            },
        )

    def simulate_smm_target_moments(
        self,
        beta: Sequence[float],
        run_config: SMMRunConfig,
        seed: tuple[int, int],
        *,
        mode: Literal["stage_a", "stage_b"] = "stage_a",
        solver_config: BasicInvestmentSMMSolverConfig | None = None,
        policy_bundle: BasicInvestmentSMMSolverBundle | None = None,
    ) -> SMMTargetMoments:
        """Simulate one fake-real Stage A/B target-moment vector."""

        mode = _resolve_basic_investment_stage(mode)
        candidate_env = _basic_investment_env_from_beta(self, beta, mode=mode)

        if mode == "stage_a":
            _validate_basic_investment_stage_a_support(candidate_env)
            if policy_bundle is not None:
                raise ValueError(
                    "policy_bundle is only supported for BasicInvestment Stage-B SMM."
                )
            next_capital_policy = _build_analytical_next_capital_policy(candidate_env)
            panel_seed = tuple(map(int, seed))
            extra_metadata = {
                "mode": mode,
                "seed": panel_seed,
            }
        else:
            _validate_basic_investment_stage_b_support(candidate_env)
            if solver_config is None and policy_bundle is not None:
                solver_config = BasicInvestmentSMMSolverConfig(
                    method=policy_bundle.method
                )
            solver_config = solver_config or BasicInvestmentSMMSolverConfig()
            if policy_bundle is None:
                policy_bundle = self.solve_smm_policy_bundle(
                    beta=beta,
                    solver_config=solver_config,
                    seed=fold_in_seed(seed, "solver"),
                )
            else:
                _validate_basic_investment_policy_bundle(beta, solver_config, policy_bundle)
            next_capital_policy = policy_bundle.next_capital_policy
            panel_seed = fold_in_seed(seed, "panel")
            extra_metadata = {
                "mode": mode,
                "seed": panel_seed,
                "solver_method": policy_bundle.method,
                "solver_bundle": _summarize_basic_investment_solver_bundle(policy_bundle),
            }

        panel_data = _simulate_basic_smm_panel_data(
            candidate_env,
            run_config=run_config,
            n_panels=1,
            seed=panel_seed,
            next_capital_policy=next_capital_policy,
            metadata=extra_metadata,
        )
        panel_moments, diagnostics = _compute_basic_smm_panel_moments(
            candidate_env,
            panel_data,
            mode=mode,
        )
        return SMMTargetMoments(
            values=np.asarray(panel_moments[0], dtype=np.float64),
            n_observations=panel_data.n_observations,
            metadata={
                "diagnostics": diagnostics,
                "seed": tuple(map(int, panel_seed)),
                "mode": mode,
                "solver_method": panel_data.metadata.get("solver_method"),
                "solver_bundle": panel_data.metadata.get("solver_bundle"),
            },
        )

    # ------------------------------------------------------------------
    # GMM helpers
    # ------------------------------------------------------------------

    def gmm_parameter_names(self) -> tuple[str, ...]:
        return ("alpha", "psi1", "rho", "sigma_epsilon")

    def gmm_moment_names(self) -> tuple[str, ...]:
        # Lagged-only instruments.  Current-period I_t/k_t and ln z_t are
        # valid IVs (E_t[e_it]=0 implies E[e_it·Z_it]=0) but adding them
        # increases overidentification, which worsens finite-sample J-test
        # size with small NT.  Lagged-only keeps overid=2 for better
        # finite-sample performance.
        return (
            "euler_const",
            "euler_ik_lag",
            "euler_pik_lag",
            "shock_const",
            "shock_lnz_lag",
            "variance_const",
        )

    def gmm_default_bounds(self) -> tuple[tuple[float, float], ...]:
        return (
            (0.10, 0.95),   # alpha
            (0.0, 5.0),     # psi1
            (-0.95, 0.95),  # rho
            (0.01, 1.0),    # sigma_epsilon
        )

    def gmm_true_beta(self) -> np.ndarray:
        return np.array(
            [
                self.econ.production_elasticity,
                self.econ.cost_convex,
                self.shocks.rho,
                self.shocks.sigma,
            ],
            dtype=np.float64,
        )

    def simulate_gmm_panel(
        self,
        seed: tuple[int, int],
        n_firms: int = 256,
        horizon: int = 200,
        burn_in: int = 175,
        solver_config: BasicInvestmentSMMSolverConfig | None = None,
    ) -> "BasicInvestmentGMMPanelData":
        """Simulate a fake-real observed panel for GMM.

        Solves the model at current parameters (Stage B) to obtain the
        optimal policy, then simulates one panel of (pi, k, I).  This
        solve happens once per panel, NOT inside the GMM optimizer loop.
        """

        _validate_basic_investment_stage_b_support(self)
        solver_config = solver_config or BasicInvestmentSMMSolverConfig()
        beta = self.gmm_true_beta()

        policy_bundle = self.solve_smm_policy_bundle(
            beta=beta,
            solver_config=solver_config,
            seed=fold_in_seed(seed, "gmm_solver"),
        )
        next_capital_policy = policy_bundle.next_capital_policy

        tmp_run_config = SMMRunConfig(
            n_firms=n_firms,
            horizon=horizon,
            burn_in=burn_in,
            n_sim_panels=1,
            master_seed=seed,
        )
        panel_data = _simulate_basic_smm_panel_data(
            self,
            run_config=tmp_run_config,
            n_panels=1,
            seed=fold_in_seed(seed, "gmm_panel"),
            next_capital_policy=next_capital_policy,
        )

        k = panel_data.k[0]        # (n_firms, horizon)
        z = panel_data.z[0]
        k_next = panel_data.k_next[0]
        alpha = self.econ.production_elasticity
        delta = self.econ.depreciation_rate

        pi = z * np.power(np.maximum(k, 1e-12), alpha)
        investment = k_next - (1.0 - delta) * k

        return BasicInvestmentGMMPanelData(
            pi=pi,
            k=k,
            investment=investment,
            n_firms=n_firms,
            horizon=horizon,
            calibrated_r=self.econ.interest_rate,
            calibrated_delta=self.econ.depreciation_rate,
        )

    def make_gmm_spec(
        self,
        panel: "BasicInvestmentGMMPanelData",
        initial_guess: Sequence[float] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        solver_config: BasicInvestmentSMMSolverConfig | None = None,
    ) -> GMMSpec:
        """Build a GMMSpec for the basic investment model with friction.

        The returned spec's moment callables are closures over ``panel``.
        """

        resolved_bounds = (
            self.gmm_default_bounds()
            if bounds is None
            else tuple(tuple(map(float, pair)) for pair in bounds)
        )
        if initial_guess is None:
            guess = np.array(
                [0.5 * (lo + hi) for lo, hi in resolved_bounds],
                dtype=np.float64,
            )
        else:
            guess = np.asarray(initial_guess, dtype=np.float64)

        contributions_fn = _make_gmm_contributions_fn(panel)

        def compute_moments(beta: np.ndarray) -> np.ndarray:
            return np.mean(contributions_fn(beta), axis=0)

        # Usable observations: t in [1, T-2] → N * (T-2)
        n_obs = panel.n_firms * (panel.horizon - 2)

        base_env = self
        sc = solver_config or BasicInvestmentSMMSolverConfig()

        def resample_spec(
            beta_true: np.ndarray,
            seed: tuple[int, int],
        ) -> GMMSpec:
            candidate_env = _basic_investment_env_from_beta(
                base_env, beta_true, mode="stage_b",
            )
            new_panel = candidate_env.simulate_gmm_panel(
                seed=seed,
                n_firms=panel.n_firms,
                horizon=panel.horizon,
                solver_config=sc,
            )
            return candidate_env.make_gmm_spec(
                new_panel,
                initial_guess=initial_guess,
                bounds=bounds,
                solver_config=sc,
            )

        # Effective panel dimensions after accounting for lags.
        n_firms = panel.n_firms
        n_periods = panel.horizon - 2  # usable t in [1, T-2]

        return GMMSpec(
            parameter_names=self.gmm_parameter_names(),
            moment_names=self.gmm_moment_names(),
            bounds=resolved_bounds,
            initial_guess=guess,
            n_observations=n_firms * n_periods,
            n_firms=n_firms,
            n_periods=n_periods,
            compute_moments=compute_moments,
            compute_moment_contributions=contributions_fn,
            resample_spec=resample_spec,
        )


# ---------------------------------------------------------------------------
# GMM panel data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasicInvestmentGMMPanelData:
    """Observed panel data for GMM: (pi, k, I) with shape (n_firms, horizon)."""

    pi: np.ndarray
    k: np.ndarray
    investment: np.ndarray
    n_firms: int
    horizon: int
    calibrated_r: float
    calibrated_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        """Effective NT for moment computation (accounts for lags)."""
        return self.n_firms * (self.horizon - 2)


def _make_gmm_contributions_fn(
    panel: "BasicInvestmentGMMPanelData",
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a closure that computes per-observation GMM contributions.

    Returns a callable: beta (4,) -> contributions (n_obs, 6).

    Lagged-only instruments.  Current-period I_t/k_t and ln z_t(alpha)
    are valid IVs (E_t[e_it]=0 implies E[e_it·Z_it]=0 by iterated
    expectations) but adding them increases overidentification, which
    worsens finite-sample J-test size with small NT.

    Instrument layout (R=6, K=4, overid=2):
      Euler  x (1, I_{t-1}/k_{t-1}, pi_{t-1}/k_{t-1})  -> 3 moments
      Shock  x (1, ln z_{t-1})                           -> 2 moments
      Variance x (1)                                     -> 1 moment

    Time indexing: residual at t uses (t, t+1); instruments use (t-1).
    Usable window: t in [1, T-2].
    """

    pi = np.asarray(panel.pi, dtype=np.float64)       # (N, T)
    k = np.asarray(panel.k, dtype=np.float64)
    inv = np.asarray(panel.investment, dtype=np.float64)
    r = float(panel.calibrated_r)
    delta = float(panel.calibrated_delta)

    # Precompute ratios that don't depend on beta
    safe_k = np.maximum(k, 1e-12)
    ik = inv / safe_k                     # (N, T)
    pik = pi / safe_k                     # (N, T)

    # Time slices (all shape (N, T-2)):
    #   t-1 -> [:, :-2],  t -> [:, 1:-1],  t+1 -> [:, 2:]
    ik_prev = ik[:, :-2]
    ik_curr = ik[:, 1:-1]
    ik_next = ik[:, 2:]
    pik_prev = pik[:, :-2]
    pik_next = pik[:, 2:]

    # Precompute log(pi) and log(k) for shock recovery
    log_pi_prev = np.log(np.maximum(pi[:, :-2], 1e-12))
    log_pi_curr = np.log(np.maximum(pi[:, 1:-1], 1e-12))
    log_pi_next = np.log(np.maximum(pi[:, 2:], 1e-12))
    log_k_prev = np.log(safe_k[:, :-2])
    log_k_curr = np.log(safe_k[:, 1:-1])
    log_k_next = np.log(safe_k[:, 2:])

    N, T = pi.shape
    n_obs = N * (T - 2)
    n_moments = 6

    def contributions(beta: np.ndarray) -> np.ndarray:
        alpha, psi1, rho, sigma_eps = (
            float(beta[0]),
            float(beta[1]),
            float(beta[2]),
            float(beta[3]),
        )

        # -- Euler residual e^u (N, T-2) --
        e_euler = (
            alpha * pik_next
            + 0.5 * psi1 * ik_next ** 2
            + (1.0 - delta) * (1.0 + psi1 * ik_next)
            - (1.0 + r) * (1.0 + psi1 * ik_curr)
        )

        # Euler instruments: lagged only (N, T-2, 3)
        ones = np.ones_like(e_euler)
        euler_contrib = np.stack([
            e_euler * ones,
            e_euler * ik_prev,
            e_euler * pik_prev,
        ], axis=-1)

        # -- Shock residual e^v (N, T-2) --
        ln_z_prev = log_pi_prev - alpha * log_k_prev
        ln_z_curr = log_pi_curr - alpha * log_k_curr
        ln_z_next = log_pi_next - alpha * log_k_next
        e_shock = ln_z_next - rho * ln_z_curr

        # Shock instruments: lagged only (N, T-2, 2)
        shock_contrib = np.stack([
            e_shock * ones,
            e_shock * ln_z_prev,
        ], axis=-1)

        # -- Variance residual e^w (N, T-2, 1) --
        e_var = e_shock ** 2 - sigma_eps ** 2
        var_contrib = e_var[..., None]

        # Stack: (N, T-2, 6) -> (n_obs, 6)
        all_contrib = np.concatenate(
            [euler_contrib, shock_contrib, var_contrib], axis=-1
        )
        return all_contrib.reshape(n_obs, n_moments)

    return contributions


@dataclass(frozen=True)
class BasicInvestmentSMMPanelData:
    k: np.ndarray
    z: np.ndarray
    k_next: np.ndarray
    n_observations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_panels(self) -> int:
        return int(self.k.shape[0])

    @property
    def n_firms(self) -> int:
        return int(self.k.shape[1])

    @property
    def horizon(self) -> int:
        return int(self.k.shape[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "z": self.z,
            "k_next": self.k_next,
            "n_observations": self.n_observations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BasicInvestmentSMMSolverConfig:
    """Notebook-facing Stage-B solver settings for solve-in-loop SMM."""

    method: Literal["ER", "PFI"] = "ER"
    dataset_config: DataGeneratorConfig = field(
        default_factory=lambda: DataGeneratorConfig(
            n_paths=128,
            horizon=16,
        )
    )
    er_config: ERConfig = field(
        default_factory=lambda: ERConfig(
            n_steps=50,
            batch_size=128,
            eval_interval=10,
            monitor="euler_residual_val",
            mode="min",
            threshold_patience=2,
            plateau_patience=None,
        )
    )
    er_network_config: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(
            n_layers=2,
            n_neurons=64,
        )
    )
    pfi_config: PFIConfig = field(default_factory=PFIConfig)

    def __post_init__(self):
        if self.method not in {"ER", "PFI"}:
            raise ValueError(
                "BasicInvestmentSMMSolverConfig.method must be 'ER' or 'PFI'. "
                f"Got {self.method!r}."
            )


@dataclass(frozen=True)
class BasicInvestmentSMMSolverBundle:
    """Unified Stage-B policy bundle returned by the env-owned solver resolver."""

    method: Literal["ER", "PFI"]
    beta: np.ndarray
    policy: Any = field(repr=False)
    next_capital_policy: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(repr=False)
    backend_result: dict[str, Any] = field(repr=False, default_factory=dict)
    selected_step: Optional[int] = None
    best_metric_value: Optional[float] = None
    wall_time_sec: float = 0.0
    converged: bool = False
    stop_reason: Optional[str] = None
    seed: tuple[int, int] = (20, 26)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "beta", np.asarray(self.beta, dtype=np.float64))


BasicInvestmentSMMMode = Literal["stage_a", "stage_b"]


def _resolve_basic_investment_stage(mode: BasicInvestmentSMMMode) -> BasicInvestmentSMMMode:
    if mode not in {"stage_a", "stage_b"}:
        raise ValueError(
            "BasicInvestment SMM mode must be 'stage_a' or 'stage_b'. "
            f"Got {mode!r}."
        )
    return mode


def _validate_basic_investment_stage_a_support(env: BasicInvestmentEnv) -> None:
    if abs(float(env.econ.cost_convex)) > 1e-12 or abs(float(env.econ.cost_fixed)) > 1e-12:
        raise ValueError(
            "BasicInvestmentEnv Stage-A analytical SMM requires the frictionless "
            "case with cost_convex == 0 and cost_fixed == 0."
        )


def _validate_basic_investment_stage_b_support(env: BasicInvestmentEnv) -> None:
    if abs(float(env.econ.cost_fixed)) > 1e-12:
        raise ValueError(
            "BasicInvestmentEnv Stage-B SMM currently requires cost_fixed == 0 "
            "because the ER backend does not support fixed adjustment costs."
        )


def _basic_investment_env_from_beta(
    base_env: BasicInvestmentEnv,
    beta: Sequence[float],
    mode: BasicInvestmentSMMMode = "stage_a",
) -> BasicInvestmentEnv:
    mode = _resolve_basic_investment_stage(mode)
    beta = np.asarray(beta, dtype=np.float64)
    if mode == "stage_a":
        if beta.shape != (3,):
            raise ValueError(
                "BasicInvestmentEnv Stage-A SMM expects a 3-parameter vector "
                "(alpha, rho, sigma_epsilon). "
                f"Got shape {beta.shape}."
            )
        production_elasticity = float(beta[0])
        cost_convex = base_env.econ.cost_convex
        rho = float(beta[1])
        sigma = float(beta[2])
    else:
        if beta.shape != (4,):
            raise ValueError(
                "BasicInvestmentEnv Stage-B SMM expects a 4-parameter vector "
                "(alpha, psi1, rho, sigma_epsilon). "
                f"Got shape {beta.shape}."
            )
        production_elasticity = float(beta[0])
        cost_convex = float(beta[1])
        rho = float(beta[2])
        sigma = float(beta[3])

    econ = EconomicParams(
        interest_rate=base_env.econ.interest_rate,
        depreciation_rate=base_env.econ.depreciation_rate,
        production_elasticity=production_elasticity,
        cost_convex=cost_convex,
        cost_fixed=base_env.econ.cost_fixed,
        tax=base_env.econ.tax,
        cost_default=base_env.econ.cost_default,
        cost_inject_fixed=base_env.econ.cost_inject_fixed,
        cost_inject_linear=base_env.econ.cost_inject_linear,
        frac_liquid=base_env.econ.frac_liquid,
    )
    shocks = ShockParams(
        mu=base_env.shocks.mu,
        rho=rho,
        sigma=sigma,
    )
    return BasicInvestmentEnv(
        econ_params=econ,
        shock_params=shocks,
        k_min_mult=base_env.k_min_mult,
        k_max_mult=base_env.k_max_mult,
        z_sd_mult=base_env.z_sd_mult,
    )


def _build_analytical_next_capital_policy(
    env: BasicInvestmentEnv,
):
    def _policy(current_k: np.ndarray, current_z: np.ndarray) -> np.ndarray:
        del current_k
        k_next = compute_frictionless_policy(current_z, env.econ, env.shocks)
        return np.clip(np.asarray(k_next, dtype=np.float64), env.k_min, env.k_max)

    return _policy


def _build_basic_investment_policy_network(
    env: BasicInvestmentEnv,
    network_config: NetworkConfig,
    seed: tuple[int, int],
    name: str = "policy",
) -> PolicyNetwork:
    policy = PolicyNetwork(
        state_dim=env.state_dim(),
        action_dim=env.action_dim(),
        n_layers=network_config.n_layers,
        n_neurons=network_config.n_neurons,
        name=name,
        seed=seed,
        **env.action_spec(),
    )
    sample_state = tf.convert_to_tensor(
        np.array(
            [[env.k_star, float(np.exp(env.shocks.mu))]],
            dtype=np.float32,
        )
    )
    policy(sample_state, training=False)
    return policy


def _make_basic_investment_er_eval_callback(
    env: BasicInvestmentEnv,
    val_dataset: dict[str, tf.Tensor],
):
    def _callback(step, callback_env, policy, value_net, callback_val_dataset):
        del step, callback_env, value_net, callback_val_dataset
        return {
            "euler_residual_val": evaluate_euler_residual(env, policy, val_dataset),
        }

    return _callback


def _select_basic_investment_er_snapshot_step(
    result: dict[str, Any],
    weight_history: Sequence[tuple[int, Sequence[np.ndarray]]],
) -> Optional[int]:
    available_steps = {int(step) for step, _ in weight_history}
    best_step = result.get("best_step")
    if best_step is not None and int(best_step) in available_steps:
        return int(best_step)
    stop_step = result.get("stop_step")
    if stop_step is not None and int(stop_step) in available_steps:
        return int(stop_step)
    if weight_history:
        return int(weight_history[-1][0])
    return None


def _build_next_capital_policy_from_state_policy(
    env: BasicInvestmentEnv,
    state_policy,
):
    def _policy(current_k: np.ndarray, current_z: np.ndarray) -> np.ndarray:
        current_k = np.asarray(current_k, dtype=np.float32).reshape(-1, 1)
        current_z = np.asarray(current_z, dtype=np.float32).reshape(-1, 1)
        state = env.merge_state(
            tf.convert_to_tensor(current_k),
            tf.convert_to_tensor(current_z),
        )
        action = state_policy(state, training=False)
        k_next = env.endogenous_transition(
            tf.convert_to_tensor(current_k),
            action,
            tf.convert_to_tensor(current_z),
        )
        return np.asarray(k_next.numpy()[:, 0], dtype=np.float64)

    return _policy


def _summarize_basic_investment_solver_bundle(
    bundle: BasicInvestmentSMMSolverBundle,
) -> dict[str, Any]:
    return {
        "method": bundle.method,
        "selected_step": bundle.selected_step,
        "best_metric_value": bundle.best_metric_value,
        "wall_time_sec": float(bundle.wall_time_sec),
        "converged": bool(bundle.converged),
        "stop_reason": bundle.stop_reason,
        "seed": tuple(map(int, bundle.seed)),
        **dict(bundle.metadata),
    }


def _validate_basic_investment_policy_bundle(
    beta: Sequence[float],
    solver_config: BasicInvestmentSMMSolverConfig,
    bundle: BasicInvestmentSMMSolverBundle,
) -> None:
    if bundle.method != solver_config.method:
        raise ValueError(
            "BasicInvestment Stage-B policy_bundle.method does not match "
            f"solver_config.method: {bundle.method!r} vs {solver_config.method!r}."
        )
    if bundle.beta.shape != np.asarray(beta, dtype=np.float64).shape or not np.allclose(
        bundle.beta,
        np.asarray(beta, dtype=np.float64),
        atol=1e-12,
    ):
        raise ValueError(
            "BasicInvestment Stage-B policy_bundle.beta does not match the "
            "requested beta."
        )


def _solve_basic_investment_er_bundle(
    env: BasicInvestmentEnv,
    *,
    beta: np.ndarray,
    solver_config: BasicInvestmentSMMSolverConfig,
    seed: tuple[int, int],
) -> BasicInvestmentSMMSolverBundle:
    dataset_seed = fold_in_seed(seed, "dataset")
    policy_seed = fold_in_seed(seed, "policy")
    trainer_seed = fold_in_seed(seed, "trainer")

    generator = DataGenerator(
        env,
        replace(solver_config.dataset_config, master_seed=dataset_seed),
    )
    train_dataset = generator.get_flattened_dataset("train", shuffle=True)
    val_dataset = generator.get_flattened_dataset("val", shuffle=False)

    policy = _build_basic_investment_policy_network(
        env,
        solver_config.er_network_config,
        policy_seed,
        name="policy_smm_er",
    )
    weight_history: list[tuple[int, Sequence[np.ndarray]]] = []
    er_config = replace(
        solver_config.er_config,
        master_seed=trainer_seed,
        network=solver_config.er_network_config,
        monitor="euler_residual_val",
        mode="min",
        weight_history=weight_history,
        checkpoint_history=None,
    )
    start = time.perf_counter()
    result = train_er(
        env,
        policy,
        train_dataset,
        val_dataset=val_dataset,
        config=er_config,
        eval_callback=_make_basic_investment_er_eval_callback(env, val_dataset),
    )
    wall_time_sec = time.perf_counter() - start

    selected_step = _select_basic_investment_er_snapshot_step(result, weight_history)
    if selected_step is not None and weight_history:
        restore_selected_snapshot(policy, weight_history, selected_step)

    return BasicInvestmentSMMSolverBundle(
        method="ER",
        beta=beta,
        policy=policy,
        next_capital_policy=_build_next_capital_policy_from_state_policy(env, policy),
        backend_result=result,
        selected_step=selected_step,
        best_metric_value=result.get("best_metric_value"),
        wall_time_sec=wall_time_sec,
        converged=bool(result.get("converged", False)),
        stop_reason=result.get("stop_reason"),
        seed=seed,
        metadata={
            "best_euler_residual_val": result.get("best_euler_residual_val"),
            "stop_step": result.get("stop_step"),
        },
    )


def _solve_basic_investment_pfi_bundle(
    env: BasicInvestmentEnv,
    *,
    beta: np.ndarray,
    solver_config: BasicInvestmentSMMSolverConfig,
    seed: tuple[int, int],
    value_init=None,
) -> BasicInvestmentSMMSolverBundle:
    dataset_seed = fold_in_seed(seed, "dataset")
    generator = DataGenerator(
        env,
        replace(solver_config.dataset_config, master_seed=dataset_seed),
    )
    train_dataset = generator.get_flattened_dataset("train", shuffle=False)
    val_dataset = generator.get_flattened_dataset("val", shuffle=False)

    start = time.perf_counter()
    result = solve_pfi(
        env,
        train_dataset,
        config=solver_config.pfi_config,
        value_init=value_init,
    )
    wall_time_sec = time.perf_counter() - start

    policy = build_action_grid_policy(result, action_dim=env.action_dim())
    euler_residual_val = evaluate_euler_residual(env, policy, val_dataset)
    return BasicInvestmentSMMSolverBundle(
        method="PFI",
        beta=beta,
        policy=policy,
        next_capital_policy=_build_next_capital_policy_from_state_policy(env, policy),
        backend_result=result,
        selected_step=int(result.get("n_iter", 0)),
        best_metric_value=euler_residual_val,
        wall_time_sec=wall_time_sec,
        converged=bool(result.get("converged", False)),
        stop_reason=result.get("stop_reason"),
        seed=seed,
        metadata={
            "euler_residual_val": euler_residual_val,
            "n_iter": int(result.get("n_iter", 0)),
        },
    )


def _simulate_basic_smm_panel_data(
    env: BasicInvestmentEnv,
    *,
    run_config: SMMRunConfig,
    n_panels: int,
    seed: tuple[int, int],
    next_capital_policy,
    metadata: Optional[dict[str, Any]] = None,
) -> BasicInvestmentSMMPanelData:
    total_units = int(n_panels * run_config.n_firms)
    total_steps = int(run_config.burn_in + run_config.horizon)

    schedule = SeedSchedule(SeedScheduleConfig(master_seed=seed))
    seeds = schedule.get_test_seeds(
        variables=[VariableID.K0, VariableID.Z0, VariableID.EPS1]
    )
    k = env.sample_initial_endogenous(total_units, seeds[VariableID.K0]).numpy()[:, 0]
    z = env.sample_initial_exogenous(total_units, seeds[VariableID.Z0]).numpy()[:, 0]
    uniforms = tf.random.stateless_uniform(
        shape=[total_units, total_steps, env.shock_dim()],
        seed=seeds[VariableID.EPS1],
        dtype=tf.float32,
    ).numpy()
    eps = _uniforms_to_standard_normal(uniforms)

    record_shape = (n_panels, run_config.n_firms, run_config.horizon)
    recorded_k = np.zeros(record_shape, dtype=np.float64)
    recorded_z = np.zeros(record_shape, dtype=np.float64)
    recorded_k_next = np.zeros(record_shape, dtype=np.float64)

    for step in range(total_steps):
        k_next = np.asarray(next_capital_policy(k, z), dtype=np.float64).reshape(-1)
        k_next = np.clip(k_next, env.k_min, env.k_max)

        if step >= run_config.burn_in:
            idx = step - run_config.burn_in
            recorded_k[..., idx] = k.reshape(n_panels, run_config.n_firms)
            recorded_z[..., idx] = z.reshape(n_panels, run_config.n_firms)
            recorded_k_next[..., idx] = k_next.reshape(n_panels, run_config.n_firms)

        z = _exogenous_transition_np(env, z, eps[:, step, 0].astype(np.float64))
        k = k_next

    return BasicInvestmentSMMPanelData(
        k=recorded_k,
        z=recorded_z,
        k_next=recorded_k_next,
        n_observations=int(n_panels * run_config.n_firms * run_config.horizon),
        metadata={
            "seed": tuple(map(int, seed)),
            "n_panels": int(n_panels),
            **(metadata or {}),
        },
    )


def _uniforms_to_standard_normal(uniforms: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(uniforms, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return ndtri(clipped)


def _exogenous_transition_np(
    env: BasicInvestmentEnv,
    z: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    log_z = np.log(np.maximum(np.asarray(z, dtype=np.float64), 1e-8))
    log_z_next = (
        (1.0 - env.shocks.rho) * env.shocks.mu
        + env.shocks.rho * log_z
        + env.shocks.sigma * np.asarray(eps, dtype=np.float64)
    )
    return np.exp(log_z_next)


def _compute_basic_smm_panel_moments(
    env: BasicInvestmentEnv,
    panel_data: BasicInvestmentSMMPanelData,
    mode: BasicInvestmentSMMMode = "stage_a",
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = _resolve_basic_investment_stage(mode)
    safe_k = np.maximum(panel_data.k, 1e-12)
    investment = (
        panel_data.k_next
        - (1.0 - env.econ.depreciation_rate) * panel_data.k
    )
    investment_ratio = investment / safe_k
    # Log income/asset ratio: observable proxy for the AR(1) IV estimator.
    # In real Compustat data z is unobservable, but z·k^(α-1) ≈ operating
    # income / assets is directly measurable.  Log-transforming preserves
    # the AR(1) structure: log(z·k^{α-1}) = log(z) + (α-1)·log(k), so the
    # first-difference IV regression recovers β₁ ≈ ρ and σ_u ≈ σ.
    log_income_ratio = np.log(
        np.maximum(panel_data.z * np.power(safe_k, env.econ.production_elasticity - 1.0), 1e-12)
    )

    n_panels = panel_data.k.shape[0]
    n_moments = 4 if mode == "stage_a" else 5
    panel_moments = np.zeros((n_panels, n_moments), dtype=np.float64)
    mean_investment_assets = np.zeros(n_panels, dtype=np.float64)
    serial_corr_investment = np.zeros(n_panels, dtype=np.float64)

    for idx in range(n_panels):
        invest_panel = investment_ratio[idx]
        income_panel = log_income_ratio[idx]
        ar_beta, ar_sigma = _panel_iv_first_diff_ar1(income_panel)
        serial_corr = _panel_serial_correlation(invest_panel)
        base_moments = [
            float(np.mean(invest_panel)),
            float(np.var(invest_panel)),
            serial_corr,
        ]
        if mode == "stage_b":
            base_moments.append(ar_beta)
        base_moments.append(ar_sigma)
        panel_moments[idx, :] = np.array(
            base_moments,
            dtype=np.float64,
        )
        mean_investment_assets[idx] = float(np.mean(invest_panel))
        serial_corr_investment[idx] = serial_corr

    diagnostics = {
        "mean_investment_assets": mean_investment_assets,
        "serial_corr_investment": serial_corr_investment,
    }
    return panel_moments, diagnostics



# =====================================================================
# Analytical benchmark (module-level, for notebook validation)
# =====================================================================

def compute_frictionless_policy(z, econ_params, shock_params):
    """Vectorized frictionless optimal k'(z).

    k'(z) = [θ · E[z'|z] / (r + δ)]^(1/(1-θ))

    where E[z'|z] = exp((1-ρ)μ + 0.5σ²) · z^ρ.

    Computed in log-space to avoid overflow when α is close to 1 (which
    makes the outer exponent 1/(1-α) large) or when candidate β values
    explored by the SMM optimizer push parameters near their bounds.

    Args:
        z:            Productivity level(s), scalar or array.
        econ_params:  EconomicParams with production_elasticity,
                      interest_rate, depreciation_rate.
        shock_params: ShockParams with rho, sigma, mu.

    Returns:
        k'(z) in levels, same shape as z.
    """
    rho, sigma, mu = shock_params.rho, shock_params.sigma, shock_params.mu
    alpha = econ_params.production_elasticity
    z = np.asarray(z, dtype=np.float64)

    # log E[z'|z] = (1-ρ)μ + 0.5σ² + ρ·log(z)
    log_ez = (1.0 - rho) * mu + 0.5 * sigma ** 2 + rho * np.log(np.maximum(z, 1e-12))
    # log( θ · E[z'|z] / (r+δ) )
    log_bracket = np.log(alpha) + log_ez - np.log(econ_params.interest_rate + econ_params.depreciation_rate)
    # k'(z) = exp( log_bracket / (1-α) )
    return np.exp(log_bracket / (1.0 - alpha))
