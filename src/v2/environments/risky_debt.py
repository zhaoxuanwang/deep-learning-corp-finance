"""Canonical risky-debt environment with explicit risky-rate pricing.

State:
    s_endo = (k, b)
    s_exo  = (z,)
    s      = (k, b, z)

Action:
    a = (I, b')
    k' = (1-delta) k + I
    b' is chosen directly

The reward uses the explicit risky rate r_tilde(z, k', b') as specified
in docs/environments/risky_debt.md. The environment does not solve the
equilibrium internally; instead, callers may install a solved r_tilde
grid for evaluation rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from scipy.special import ndtri

from src.v2.data.rng import SeedSchedule, SeedScheduleConfig, VariableID
from src.v2.estimation.smm import (
    SMMPanelMoments,
    SMMRunConfig,
    SMMSpec,
    SMMTargetMoments,
    _panel_covariance,
    _panel_iv_first_diff_ar1,
    _panel_serial_correlation,
)
from src.v2.environments.base import MDPEnvironment
from src.v2.solvers import (
    NestedVFIConfig,
    RiskyDebtSolverConfig,
    solve_nested_vfi,
    solve_risky_debt,
)

# Type alias for solver callables: fn(env, config=...) -> dict
SolverFn = Any
SolverConfig = NestedVFIConfig | RiskyDebtSolverConfig


@dataclass(frozen=True)
class ShockParams:
    """AR(1) productivity parameters in log space."""

    rho: float = 0.7
    sigma: float = 0.15
    mu: float = 0.0

    def __post_init__(self):
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1). Got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0. Got {self.sigma}")


@dataclass(frozen=True)
class EconomicParams:
    """Economic primitives for the canonical risky-debt model."""

    interest_rate: float = 0.04
    depreciation_rate: float = 0.15
    production_elasticity: float = 0.7
    cost_convex: float = 0.0
    tax: float = 0.3
    default_haircut: float = 0.4
    cost_inject_fixed: float = 0.0
    cost_inject_linear: float = 0.0

    def __post_init__(self):
        if self.interest_rate <= -1.0:
            raise ValueError(
                "interest_rate must be greater than -1. "
                f"Got {self.interest_rate}"
            )
        if not (0.0 <= self.depreciation_rate <= 1.0):
            raise ValueError(
                "depreciation_rate must be in [0, 1]. "
                f"Got {self.depreciation_rate}"
            )
        if not (0.0 < self.production_elasticity < 1.0):
            raise ValueError(
                "production_elasticity must be in (0, 1). "
                f"Got {self.production_elasticity}"
            )
        if self.cost_convex < 0.0:
            raise ValueError(f"cost_convex must be >= 0. Got {self.cost_convex}")
        if not (0.0 <= self.tax < 1.0):
            raise ValueError(f"tax must be in [0, 1). Got {self.tax}")
        if not (0.0 <= self.default_haircut <= 1.0):
            raise ValueError(
                "default_haircut must be in [0, 1]. "
                f"Got {self.default_haircut}"
            )
        if self.cost_inject_fixed < 0.0:
            raise ValueError(
                "cost_inject_fixed must be >= 0. "
                f"Got {self.cost_inject_fixed}"
            )
        if self.cost_inject_linear < 0.0:
            raise ValueError(
                "cost_inject_linear must be >= 0. "
                f"Got {self.cost_inject_linear}"
            )


def _production_np(k, z, alpha):
    return z * np.power(k, alpha)


def _production_tf(k, z, alpha):
    return z * tf.pow(k, alpha)


def _compute_k_ref(econ: EconomicParams, shocks: ShockParams) -> float:
    """Tax-adjusted frictionless reference capital."""

    expected_z_prime = np.exp(shocks.mu + 0.5 * shocks.sigma ** 2)
    return float(
        (
            (1.0 - econ.tax)
            * econ.production_elasticity
            * expected_z_prime
            / (econ.interest_rate + econ.depreciation_rate)
        ) ** (1.0 / (1.0 - econ.production_elasticity))
    )


class RiskyDebtEnv(MDPEnvironment):
    """Canonical risky-debt environment using explicit r_tilde pricing.

    This environment is solver-only in the supported surface. The reward uses
    the true hard issuance-cost kink and explicit risky-rate pricing, so the
    active NN trainers reject it and point users to ``solve_nested_vfi()``.
    """

    def __init__(
        self,
        econ_params: EconomicParams | None = None,
        shock_params: ShockParams | None = None,
        k_min_mult: float = 0.25,
        k_max_mult: float = 6.0,
        b_max_mult: float = 3.0,
        b_min_mult: float = 0.2,
        z_sd_mult: float = 3.0,
        b_min_override: float | None = None,
        b_max_override: float | None = None,
    ):
        if b_min_mult < 0.0:
            raise ValueError(
                f"b_min_mult must be >= 0. Got {b_min_mult}"
            )
        self.econ = econ_params or EconomicParams()
        self.shocks = shock_params or ShockParams()
        self.beta = 1.0 / (1.0 + self.econ.interest_rate)
        self.k_min_mult = float(k_min_mult)
        self.k_max_mult = float(k_max_mult)
        self.b_max_mult = float(b_max_mult)
        self.b_min_mult = float(b_min_mult)
        self.z_sd_mult = float(z_sd_mult)

        rho = self.shocks.rho
        sigma = self.shocks.sigma
        mu = self.shocks.mu
        sigma_ergodic = sigma / np.sqrt(1.0 - rho ** 2)

        self.k_ref = _compute_k_ref(self.econ, self.shocks)
        self.k_min = k_min_mult * self.k_ref
        self.k_max = k_max_mult * self.k_ref
        self.b_max = b_max_mult * self.k_max
        self.b_min = -b_min_mult * self.b_max
        if b_min_override is not None:
            self.b_min = float(b_min_override)
        if b_max_override is not None:
            self.b_max = float(b_max_override)
        self.z_min = float(np.exp(mu - z_sd_mult * sigma_ergodic))
        self.z_max = float(np.exp(mu + z_sd_mult * sigma_ergodic))

        self.I_min = self.k_min - (1.0 - self.econ.depreciation_rate) * self.k_max
        self.I_max = self.k_max - (1.0 - self.econ.depreciation_rate) * self.k_min
        self._z_bar = tf.constant([float(np.exp(mu))], dtype=tf.float32)

        self._installed_r_tilde = None
        self._installed_r_tilde_tf = None
        self._z_grid_tf = None
        self._k_grid_tf = None
        self._b_grid_tf = None

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def exo_dim(self) -> int:
        return 1

    def endo_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 2

    # ------------------------------------------------------------------
    # Action/state helpers
    # ------------------------------------------------------------------

    def action_bounds(self) -> tuple:
        return (
            tf.constant([self.I_min, self.b_min], dtype=tf.float32),
            tf.constant([self.I_max, self.b_max], dtype=tf.float32),
        )

    def action_scale_reference(self) -> tuple:
        debt_center = 0.5 * (self.b_min + self.b_max)
        debt_half_range = 0.5 * (self.b_max - self.b_min)
        center = tf.constant([0.0, debt_center], dtype=tf.float32)
        half_range = tf.constant(
            [max(abs(self.I_min), self.I_max), debt_half_range],
            dtype=tf.float32,
        )
        return center, half_range

    def _apply_action(
        self, k: tf.Tensor, b: tf.Tensor, action: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply action bounds and return (k', b', I_eff)."""

        del b
        investment = action[..., 0]
        b_next = tf.clip_by_value(action[..., 1], self.b_min, self.b_max)
        k_next = (1.0 - self.econ.depreciation_rate) * k + investment
        k_next = tf.clip_by_value(k_next, self.k_min, self.k_max)
        i_eff = k_next - (1.0 - self.econ.depreciation_rate) * k
        return k_next, b_next, i_eff

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def exogenous_transition(self, s_exo: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        z = s_exo[..., 0]
        log_z = tf.math.log(tf.maximum(z, 1e-8))
        log_z_next = (
            (1.0 - self.shocks.rho) * self.shocks.mu
            + self.shocks.rho * log_z
            + self.shocks.sigma * eps[..., 0]
        )
        return tf.reshape(tf.exp(log_z_next), [-1, 1])

    def endogenous_transition(
        self, s_endo: tf.Tensor, action: tf.Tensor, s_exo: tf.Tensor
    ) -> tf.Tensor:
        del s_exo
        k = s_endo[..., 0]
        b = s_endo[..., 1]
        k_next, b_next, _ = self._apply_action(k, b, action)
        return tf.stack([k_next, b_next], axis=-1)

    # ------------------------------------------------------------------
    # Economic primitives
    # ------------------------------------------------------------------

    def recovery_value(self, k_next, z_next):
        """Recovery under default: (1-c_d)[(1-tau)Pi + (1-delta)k']."""

        if tf.is_tensor(k_next) or tf.is_tensor(z_next):
            k_next = tf.convert_to_tensor(k_next, dtype=tf.float32)
            z_next = tf.convert_to_tensor(z_next, dtype=tf.float32)
            profit = _production_tf(k_next, z_next, self.econ.production_elasticity)
            return (1.0 - self.econ.default_haircut) * (
                (1.0 - self.econ.tax) * profit
                + (1.0 - self.econ.depreciation_rate) * k_next
            )

        k_next = np.asarray(k_next, dtype=np.float64)
        z_next = np.asarray(z_next, dtype=np.float64)
        profit = _production_np(k_next, z_next, self.econ.production_elasticity)
        return (1.0 - self.econ.default_haircut) * (
            (1.0 - self.econ.tax) * profit
            + (1.0 - self.econ.depreciation_rate) * k_next
        )

    def debt_discount_factor(self, r_tilde: tf.Tensor) -> tf.Tensor:
        """Compute 1 / (1 + r_tilde), with zero for r_tilde = inf."""

        r_tilde = tf.convert_to_tensor(r_tilde, dtype=tf.float32)
        finite = tf.math.is_finite(r_tilde)
        safe_denominator = tf.maximum(1.0 + r_tilde, 1e-8)
        return tf.where(finite, 1.0 / safe_denominator, tf.zeros_like(r_tilde))

    def cash_flow_from_choice(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_next: tf.Tensor,
        b_next: tf.Tensor,
        r_tilde: tf.Tensor,
    ) -> tf.Tensor:
        """Pre-issuance equity cash flow under explicit risky-rate pricing."""

        investment = k_next - (1.0 - self.econ.depreciation_rate) * k
        safe_k = tf.maximum(k, 1e-8)
        after_tax_profit = (1.0 - self.econ.tax) * _production_tf(
            k, z, self.econ.production_elasticity
        )
        adjustment_cost = (
            0.5 * self.econ.cost_convex * tf.square(investment) / safe_k
        )
        debt_discount = self.debt_discount_factor(r_tilde)
        debt_proceeds = b_next * debt_discount
        tax_shield = (
            self.econ.tax
            * b_next
            * (1.0 - debt_discount)
            / (1.0 + self.econ.interest_rate)
        )
        return (
            after_tax_profit
            - adjustment_cost
            - investment
            - b
            + debt_proceeds
            + tax_shield
        )

    def issuance_cost(self, cash_flow: tf.Tensor) -> tf.Tensor:
        shortfall = tf.nn.relu(-cash_flow)
        fixed_cost = self.econ.cost_inject_fixed * tf.cast(shortfall > 0.0, tf.float32)
        linear_cost = self.econ.cost_inject_linear * shortfall
        return fixed_cost + linear_cost

    def reward_from_choice(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_next: tf.Tensor,
        b_next: tf.Tensor,
        r_tilde: tf.Tensor,
    ) -> tf.Tensor:
        cash_flow = self.cash_flow_from_choice(k, b, z, k_next, b_next, r_tilde)
        return cash_flow - self.issuance_cost(cash_flow)

    # ------------------------------------------------------------------
    # Installed risky-rate schedule
    # ------------------------------------------------------------------

    def install_r_tilde_grid(self, grids: dict, r_tilde_grid) -> None:
        """Install a solved r_tilde(z, k', b') grid for evaluation rollouts."""

        z_grid = np.asarray(grids["exo_grids_1d"][0], dtype=np.float32)
        k_grid = np.asarray(grids["endo_grids_1d"][0], dtype=np.float32)
        b_grid = np.asarray(grids["endo_grids_1d"][1], dtype=np.float32)
        r_tilde_grid = np.asarray(r_tilde_grid, dtype=np.float32)

        expected_shape = (len(z_grid), len(k_grid), len(b_grid))
        if r_tilde_grid.shape != expected_shape:
            raise ValueError(
                "r_tilde_grid shape does not match the supplied grids. "
                f"Expected {expected_shape}, got {r_tilde_grid.shape}."
            )

        self._installed_r_tilde = r_tilde_grid
        self._installed_r_tilde_tf = tf.constant(r_tilde_grid, dtype=tf.float32)
        self._z_grid_tf = tf.constant(z_grid, dtype=tf.float32)
        self._k_grid_tf = tf.constant(k_grid, dtype=tf.float32)
        self._b_grid_tf = tf.constant(b_grid, dtype=tf.float32)

    def clear_r_tilde_grid(self) -> None:
        self._installed_r_tilde = None
        self._installed_r_tilde_tf = None
        self._z_grid_tf = None
        self._k_grid_tf = None
        self._b_grid_tf = None

    def continuation_transform(self, v_next: tf.Tensor) -> tf.Tensor:
        """Equity value under limited liability."""

        return tf.nn.relu(v_next)

    def _lookup_installed_r_tilde(
        self, k_next: tf.Tensor, b_next: tf.Tensor, z: tf.Tensor
    ) -> tf.Tensor:
        if self._installed_r_tilde_tf is None:
            return tf.fill(tf.shape(k_next), tf.cast(self.econ.interest_rate, tf.float32))

        iz = _nearest_idx(z, self._z_grid_tf)
        ik = _nearest_idx(k_next, self._k_grid_tf)
        ib = _nearest_idx(b_next, self._b_grid_tf)
        gather_idx = tf.stack([iz, ik, ib], axis=-1)
        return tf.gather_nd(self._installed_r_tilde_tf, gather_idx)

    def resolve_r_tilde(
        self, k_next: tf.Tensor, b_next: tf.Tensor, z: tf.Tensor
    ) -> tf.Tensor:
        """Resolve r_tilde from the installed solved grid or risk-free fallback."""

        return self._lookup_installed_r_tilde(k_next, b_next, z)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, s: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        k = tf.convert_to_tensor(s[..., 0], dtype=tf.float32)
        b = tf.convert_to_tensor(s[..., 1], dtype=tf.float32)
        z = tf.convert_to_tensor(s[..., 2], dtype=tf.float32)
        k_next, b_next, _ = self._apply_action(k, b, action)
        r_tilde = self.resolve_r_tilde(k_next, b_next, z)
        return self.reward_from_choice(k, b, z, k_next, b_next, r_tilde)

    def validate_nn_training_support(self, trainer_name: str) -> None:
        raise ValueError(
            f"{trainer_name} does not support RiskyDebtEnv in the supported "
            "surface. Use solve_nested_vfi() for the canonical risky-debt "
            "model."
        )

    # ------------------------------------------------------------------
    # Sampling and optional helpers
    # ------------------------------------------------------------------

    def sample_initial_endogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        seed = tf.convert_to_tensor(seed, dtype=tf.int32)
        seed_b = tf.stack([seed[0] + 101, seed[1]])
        k = tf.random.stateless_uniform(
            [n],
            seed=seed,
            minval=self.k_min,
            maxval=self.k_max,
            dtype=tf.float32,
        )
        b = tf.random.stateless_uniform(
            [n],
            seed=seed_b,
            minval=self.b_min,
            maxval=self.b_max,
            dtype=tf.float32,
        )
        return tf.stack([k, b], axis=-1)

    def sample_initial_exogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        z = tf.random.stateless_uniform(
            [n],
            seed=tf.convert_to_tensor(seed, dtype=tf.int32),
            minval=self.z_min,
            maxval=self.z_max,
            dtype=tf.float32,
        )
        return tf.reshape(z, [n, 1])

    def discount(self) -> float:
        return self.beta

    def stationary_exo(self) -> tf.Tensor:
        return self._z_bar

    def stationary_action(self, s_endo: tf.Tensor) -> tf.Tensor:
        k = s_endo[..., 0:1]
        b = s_endo[..., 1:2]
        investment = self.econ.depreciation_rate * k
        return tf.concat([investment, b], axis=-1)

    def grid_spec(self):
        from src.v2.solvers.grid import GridAxis

        return {
            "endo": [
                GridAxis(self.k_min, self.k_max, spacing="geometric"),
                GridAxis(self.b_min, self.b_max, spacing="linear"),
            ],
            "exo": [
                GridAxis(self.z_min, self.z_max, spacing="log"),
            ],
            "action": [
                GridAxis(self.I_min, self.I_max, spacing="linear"),
                GridAxis(self.b_min, self.b_max, spacing="linear"),
            ],
        }

    # ------------------------------------------------------------------
    # SMM helpers
    # ------------------------------------------------------------------

    def clone(
        self,
        *,
        econ_params: EconomicParams | None = None,
        shock_params: ShockParams | None = None,
    ) -> "RiskyDebtEnv":
        """Clone the environment while preserving state-space multipliers."""

        return RiskyDebtEnv(
            econ_params=econ_params or self.econ,
            shock_params=shock_params or self.shocks,
            k_min_mult=self.k_min_mult,
            k_max_mult=self.k_max_mult,
            b_max_mult=self.b_max_mult,
            b_min_mult=self.b_min_mult,
            z_sd_mult=self.z_sd_mult,
        )

    def smm_parameter_names(self) -> tuple[str, ...]:
        return (
            "alpha",
            "psi1",
            "eta0",
            "eta1",
            "c_def",
            "rho",
            "sigma_epsilon",
        )

    def smm_moment_names(self) -> tuple[str, ...]:
        return (
            "avg_equity_issuance_assets",
            "conditional_issuance_size",
            "autocorr_equity_issuance",
            "crosscorr_leverage_issuance",
            "book_leverage",
            "cov_leverage_investment",
            "mean_investment_assets",
            "serial_corr_investment",
            "var_investment_assets",
            "income_ar1_beta",
            "income_ar1_resid_std",
            "default_frequency",
            "frequency_equity_issuance",
            "corr_issuance_investment",
        )

    def smm_default_bounds(self) -> tuple[tuple[float, float], ...]:
        return (
            (0.10, 0.95),   # alpha
            (0.0, 1.0),     # psi1
            (0.0, 2.0),     # eta0
            (0.0, 1.0),     # eta1
            (0.0, 0.95),    # c_def
            (-0.95, 0.95),  # rho
            (0.01, 1.0),    # sigma_epsilon
        )

    def smm_initial_guess(self) -> np.ndarray:
        return np.array(
            [
                self.econ.production_elasticity,
                self.econ.cost_convex,
                self.econ.cost_inject_fixed,
                self.econ.cost_inject_linear,
                self.econ.default_haircut,
                self.shocks.rho,
                self.shocks.sigma,
            ],
            dtype=np.float64,
        )

    def smm_true_beta(
        self,
        estimated_params: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return the env's current structural parameters as the ground-truth β.

        Args:
            estimated_params: When given, returns only the values for the
                named parameters (in the given order).  Defaults to all 7.
        """
        full = self.smm_initial_guess()
        if estimated_params is None:
            return full
        all_names = self.smm_parameter_names()
        indices = [all_names.index(p) for p in estimated_params]
        return full[indices]

    def make_smm_spec(
        self,
        solver_config: SolverConfig | None = None,
        initial_guess: Sequence[float] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        estimated_params: Sequence[str] | None = None,
        solver_fn: SolverFn | None = None,
        disabled_moments: Sequence[str] | None = None,
    ) -> SMMSpec:
        """Build the generic SMM specification owned by this environment.

        Args:
            solver_config: Risky-debt solver settings used inside the optimizer
                           loop.  Passing ``NestedVFIConfig`` selects
                           ``solve_nested_vfi``; passing
                           ``RiskyDebtSolverConfig`` (or leaving this as
                           ``None``) selects ``solve_risky_debt``. Each
                           evaluate_beta call re-solves the model — this is
                           the dominant cost per iteration.
            initial_guess: Starting point for the optimizer.  Defaults to
                           the midpoint of each parameter's bounds.
            bounds:        Search region for each parameter.  Defaults to
                           ``smm_default_bounds()`` sliced to match
                           ``estimated_params``.
            estimated_params:
                           Subset of parameter names to estimate.  Parameters
                           not listed are held fixed at their current env
                           values (calibrated).  Moments are auto-selected:
                           only moments that help identify at least one
                           estimated parameter are included.  Defaults to
                           all 7 parameters (full estimation).
            solver_fn:     Callable ``fn(env, config=...) -> dict`` used to
                           solve the model.  Defaults are inferred from
                           ``solver_config``.
            disabled_moments:
                           Moment names to exclude from the SMM objective
                           even when they would otherwise be auto-selected.
                           These moments remain defined in the registry and
                           are still computed by ``compute_smm_panel_moments``
                           (so they appear in diagnostic outputs), but the
                           optimizer never sees them.  Use to test alternative
                           moment sets without removing code.  Must contain
                           valid moment names from ``smm_moment_names()``.
        """
        # --- Resolve estimated params and select moments ----------------
        if estimated_params is not None:
            est_params = tuple(estimated_params)
            unknown = set(est_params) - set(_ALL_PARAM_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown parameter name(s): {sorted(unknown)}. "
                    f"Valid names: {_ALL_PARAM_NAMES}"
                )
        else:
            est_params = _ALL_PARAM_NAMES

        selected_moments, moment_indices = _select_moments(est_params)

        # --- Apply user-supplied disabled_moments filter ----------------
        if disabled_moments:
            disabled_set = set(disabled_moments)
            unknown_m = disabled_set - set(_ALL_MOMENT_NAMES)
            if unknown_m:
                raise ValueError(
                    f"Unknown moment name(s) in disabled_moments: "
                    f"{sorted(unknown_m)}. "
                    f"Valid names: {_ALL_MOMENT_NAMES}"
                )
            kept = [
                (m, i) for m, i in zip(selected_moments, moment_indices)
                if m not in disabled_set
            ]
            if kept:
                selected_moments, moment_indices = (
                    tuple(m for m, _ in kept),
                    tuple(i for _, i in kept),
                )
            else:
                selected_moments, moment_indices = (), ()

        if len(selected_moments) < len(est_params):
            raise ValueError(
                f"Underidentified: {len(est_params)} estimated parameters "
                f"but only {len(selected_moments)} moments active "
                f"(after applying disabled_moments). "
                f"Need at least as many moments as parameters."
            )

        # --- Bounds -----------------------------------------------------
        if bounds is not None:
            resolved_bounds = tuple(
                tuple(map(float, pair)) for pair in bounds
            )
        else:
            all_defaults = self.smm_default_bounds()
            all_names = self.smm_parameter_names()
            pidx = [all_names.index(p) for p in est_params]
            resolved_bounds = tuple(all_defaults[i] for i in pidx)

        # --- Initial guess ----------------------------------------------
        if initial_guess is None:
            guess = np.array(
                [0.5 * (lo + hi) for lo, hi in resolved_bounds],
                dtype=np.float64,
            )
        else:
            guess = np.asarray(initial_guess, dtype=np.float64)

        solver_config, _solver_fn = _resolve_smm_solver(solver_config, solver_fn)
        midx = list(moment_indices)  # for numpy fancy indexing

        # --- Closures ---------------------------------------------------
        # Build candidate env once per evaluation, call the module-level
        # simulation function directly (avoids the full-beta assumption
        # inside the public simulate_smm_panel_data method).

        def _simulate_panel_moments(beta, run_config, seed):
            candidate_env = _risky_debt_env_from_beta(
                self, beta, est_params
            )
            solved = _solver_fn(candidate_env, config=solver_config)
            panel_data = _simulate_smm_panel_data(
                candidate_env, solved,
                run_config=run_config,
                n_panels=run_config.n_sim_panels,
                seed=seed,
            )
            summary = candidate_env.compute_smm_panel_moments(panel_data)
            return SMMPanelMoments(
                panel_moments=summary["panel_moments"][:, midx],
                n_observations=panel_data.n_observations,
            )

        def _simulate_target_moments(beta, run_config, seed):
            from dataclasses import replace as dc_replace
            candidate_env = _risky_debt_env_from_beta(
                self, beta, est_params
            )
            single_config = dc_replace(run_config, n_sim_panels=1)
            solved = _solver_fn(candidate_env, config=solver_config)
            panel_data = _simulate_smm_panel_data(
                candidate_env, solved,
                run_config=single_config,
                n_panels=1,
                seed=seed,
            )
            summary = candidate_env.compute_smm_panel_moments(panel_data)
            return SMMTargetMoments(
                values=np.asarray(
                    summary["panel_moments"][0, midx], dtype=np.float64
                ),
                n_observations=panel_data.n_observations,
            )

        return SMMSpec(
            parameter_names=est_params,
            moment_names=selected_moments,
            bounds=resolved_bounds,
            initial_guess=guess,
            simulate_panel_moments=_simulate_panel_moments,
            simulate_target_moments=_simulate_target_moments,
        )

    def simulate_smm_panel_data(
        self,
        beta: Sequence[float],
        run_config: SMMRunConfig,
        seed: tuple[int, int],
        solver_config: SolverConfig | None = None,
        solved_result: dict[str, Any] | None = None,
        solver_fn: SolverFn | None = None,
    ) -> "RiskyDebtSMMPanelData":
        """Solve the candidate model and return raw simulated panel data."""

        solver_config, _solver_fn = _resolve_smm_solver(solver_config, solver_fn)
        candidate_env = _risky_debt_env_from_beta(self, beta)
        solved = solved_result
        if solved is None:
            solved = _solver_fn(candidate_env, config=solver_config)
        panel_data = _simulate_smm_panel_data(
            candidate_env,
            solved,
            run_config=run_config,
            n_panels=run_config.n_sim_panels,
            seed=seed,
        )
        return RiskyDebtSMMPanelData(
            k=panel_data.k,
            b=panel_data.b,
            z=panel_data.z,
            value=panel_data.value,
            k_next=panel_data.k_next,
            b_next=panel_data.b_next,
            cash_flow=panel_data.cash_flow,
            debt_discount=panel_data.debt_discount,
            n_observations=panel_data.n_observations,
            metadata={
                "seed": tuple(map(int, seed)),
                "n_panels": int(panel_data.n_panels),
                "solver_reused": bool(solved_result is not None),
                "solver_stop_reason": solved.get("stop_reason"),
                "solver_converged_outer": bool(solved.get("converged_outer", False)),
                "solver_n_outer": int(solved.get("n_outer", 0)),
            },
        )

    def compute_smm_panel_moments(
        self,
        panel_data: "RiskyDebtSMMPanelData",
    ) -> dict[str, Any]:
        """Compute the risky-debt SMM moments from raw panel data."""

        panel_moments, diagnostics = _compute_smm_panel_moments(self, panel_data)
        return {
            "moment_names": self.smm_moment_names(),
            "panel_moments": panel_moments,
            "average_moments": np.mean(panel_moments, axis=0),
            "diagnostics": diagnostics,
            "n_observations": int(panel_data.n_observations),
        }

    def simulate_smm_panel_moments(
        self,
        beta: Sequence[float],
        run_config: SMMRunConfig,
        seed: tuple[int, int],
        solver_config: SolverConfig | None = None,
        solved_result: dict[str, Any] | None = None,
        solver_fn: SolverFn | None = None,
    ) -> SMMPanelMoments:
        """Solve the candidate model and compute per-panel SMM moments.

        Called once per candidate β by the SMM optimizer — re-solves the
        model via nested VFI for each call.  This is the dominant cost per
        optimizer iteration.

        Args:
            solved_result: Optional pre-computed solver output.  When
                           provided, skips the model solve entirely.
            solver_fn:     Callable ``fn(env, config=...) -> dict``.
                           Defaults are inferred from ``solver_config``.
        """
        panel_data = self.simulate_smm_panel_data(
            beta=beta,
            run_config=run_config,
            seed=seed,
            solver_config=solver_config,
            solved_result=solved_result,
            solver_fn=solver_fn,
        )
        summary = self.compute_smm_panel_moments(panel_data)
        return SMMPanelMoments(
            panel_moments=summary["panel_moments"],
            n_observations=panel_data.n_observations,
            metadata={
                "diagnostics": summary["diagnostics"],
                "solver_stop_reason": panel_data.metadata.get("solver_stop_reason"),
                "solver_converged_outer": panel_data.metadata.get("solver_converged_outer"),
                "solver_n_outer": panel_data.metadata.get("solver_n_outer"),
            },
        )

    def simulate_smm_target_moments(
        self,
        beta: Sequence[float],
        run_config: SMMRunConfig,
        seed: tuple[int, int],
        solver_config: SolverConfig | None = None,
        solver_fn: SolverFn | None = None,
    ) -> SMMTargetMoments:
        """Simulate a fake-real target-moment vector from the model.

        Simulates a single panel (n_panels=1) at the given beta and seed.
        The solver result is not cached; call ``simulate_smm_panel_data``
        directly if you need access to the raw panel or wish to reuse a
        pre-solved result.
        """
        from dataclasses import replace as dc_replace
        single_panel_config = dc_replace(run_config, n_sim_panels=1)
        panel_data = self.simulate_smm_panel_data(
            beta=beta,
            run_config=single_panel_config,
            seed=seed,
            solver_config=solver_config,
            solver_fn=solver_fn,
        )
        summary = self.compute_smm_panel_moments(panel_data)
        return SMMTargetMoments(
            values=np.asarray(summary["panel_moments"][0], dtype=np.float64),
            n_observations=panel_data.n_observations,
            metadata={
                "diagnostics": summary["diagnostics"],
                "solver_stop_reason": panel_data.metadata.get("solver_stop_reason"),
                "solver_converged_outer": panel_data.metadata.get("solver_converged_outer"),
                "solver_n_outer": panel_data.metadata.get("solver_n_outer"),
                "seed": tuple(map(int, seed)),
            },
        )

def _nearest_idx(values: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    """Find the nearest grid index for each query value."""

    values = tf.reshape(tf.cast(values, tf.float32), [-1, 1])
    grid = tf.reshape(tf.cast(grid, tf.float32), [1, -1])
    distance = tf.abs(values - grid)
    return tf.argmin(distance, axis=-1, output_type=tf.int32)


@dataclass(frozen=True)
class RiskyDebtSMMPanelData:
    k: np.ndarray
    b: np.ndarray
    z: np.ndarray
    value: np.ndarray
    k_next: np.ndarray
    b_next: np.ndarray
    cash_flow: np.ndarray
    debt_discount: np.ndarray
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
            "b": self.b,
            "z": self.z,
            "value": self.value,
            "k_next": self.k_next,
            "b_next": self.b_next,
            "cash_flow": self.cash_flow,
            "debt_discount": self.debt_discount,
            "n_observations": self.n_observations,
            "metadata": dict(self.metadata),
        }

# ---------------------------------------------------------------------------
# SMM parameter ↔ moment identification mapping
# ---------------------------------------------------------------------------

# Canonical order of all estimable structural parameters.
_ALL_PARAM_NAMES: tuple[str, ...] = (
    "alpha", "psi1", "eta0", "eta1", "c_def", "rho", "sigma_epsilon",
)

# All moment conditions in canonical order (must match _compute_smm_panel_moments).
_ALL_MOMENT_NAMES: tuple[str, ...] = (
    "avg_equity_issuance_assets",
    "conditional_issuance_size",
    "autocorr_equity_issuance",
    "crosscorr_leverage_issuance",
    "book_leverage",
    "cov_leverage_investment",
    "mean_investment_assets",
    "serial_corr_investment",
    "var_investment_assets",
    "income_ar1_beta",
    "income_ar1_resid_std",
    "default_frequency",
    "frequency_equity_issuance",
    "corr_issuance_investment",
)

# Unused moment candidates — defined here for documentation but NOT computed
# by ``_compute_smm_panel_moments`` and NOT exposed via the SMM spec.
#
#   "avg_net_debt_assets"     was  E[debt_market / (V + debt_market)]
#                             (market-value leverage; mislabeled vs H&W's
#                             book "net debt to assets" which is E[b'/k]).
#                             Replaced by ``book_leverage`` = E[b'/k].
#   "std_leverage"            was  Std(market-value leverage).  Replaced by
#                             ``cov_leverage_investment`` = Cov(b'/k, I/k),
#                             H&W's pecking-order moment for c_def.
#   "frequency_negative_debt" was  Pr(b' < 0).  Always zero under
#                             ``solve_risky_debt(adaptive=True)`` because
#                             the adaptive solver shrinks the b-grid to
#                             the default-risk region (b' > 0 only), and
#                             simulation clamps b_next to that tight grid.
#                             A dead moment would create a zero row in
#                             Omega_hat, making it singular.  Removed
#                             entirely since adaptive is the production
#                             default.
_UNUSED_MOMENT_CANDIDATES: tuple[str, ...] = (
    "avg_net_debt_assets",
    "std_leverage",
    "frequency_negative_debt",
)

# For each moment, which structural parameters it primarily helps identify.
# A moment is included in estimation iff at least one of its tags is being
# estimated.  When all tagged parameters are calibrated, the moment loses
# its identifying power and is dropped automatically.
_MOMENT_PARAM_TAGS: dict[str, tuple[str, ...]] = {
    "avg_equity_issuance_assets":  ("eta0", "eta1"),
    "conditional_issuance_size":   ("eta0", "eta1"),
    "autocorr_equity_issuance":    ("eta0", "eta1"),
    "crosscorr_leverage_issuance": ("c_def", "eta0", "eta1"),
    "book_leverage":               ("c_def",),
    "cov_leverage_investment":     ("c_def",),
    "mean_investment_assets":      ("alpha",),
    "serial_corr_investment":      ("rho", "psi1"),
    "var_investment_assets":       ("psi1", "sigma_epsilon"),
    "income_ar1_beta":             ("rho",),
    "income_ar1_resid_std":        ("sigma_epsilon",),
    "default_frequency":           ("c_def",),
    "frequency_equity_issuance":   ("eta0",),
    "corr_issuance_investment":    ("eta0", "eta1"),
}


def _select_moments(
    estimated_params: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Select moments relevant to the estimated parameters.

    Returns (moment_names, moment_indices) where indices refer to positions
    in the full 10-moment vector from ``_compute_smm_panel_moments``.
    """
    est_set = set(estimated_params)
    selected: list[str] = []
    indices: list[int] = []
    for i, m in enumerate(_ALL_MOMENT_NAMES):
        if est_set & set(_MOMENT_PARAM_TAGS[m]):
            selected.append(m)
            indices.append(i)
    return tuple(selected), tuple(indices)


def _resolve_smm_solver(
    solver_config: SolverConfig | None,
    solver_fn: SolverFn | None,
) -> tuple[SolverConfig, SolverFn]:
    """Resolve the risky-debt SMM solver/config pair.

    ``NestedVFIConfig`` is kept for legacy tests and raw-grid diagnostics.
    ``RiskyDebtSolverConfig`` is the production notebook path.
    """

    if solver_fn is None:
        if solver_config is None:
            return RiskyDebtSolverConfig(), solve_risky_debt
        if isinstance(solver_config, NestedVFIConfig):
            return solver_config, solve_nested_vfi
        return solver_config, solve_risky_debt

    if solver_config is None:
        if solver_fn is solve_nested_vfi:
            return NestedVFIConfig(), solver_fn
        return RiskyDebtSolverConfig(), solver_fn
    return solver_config, solver_fn


def _env_param_values(env: RiskyDebtEnv) -> dict[str, float]:
    """Extract all SMM-estimable parameter values from an environment."""
    return {
        "alpha": env.econ.production_elasticity,
        "psi1": env.econ.cost_convex,
        "eta0": env.econ.cost_inject_fixed,
        "eta1": env.econ.cost_inject_linear,
        "c_def": env.econ.default_haircut,
        "rho": env.shocks.rho,
        "sigma_epsilon": env.shocks.sigma,
    }


def _risky_debt_env_from_beta(
    base_env: RiskyDebtEnv,
    beta: Sequence[float],
    estimated_params: tuple[str, ...] = _ALL_PARAM_NAMES,
) -> RiskyDebtEnv:
    """Build a candidate environment by overriding estimated parameters.

    Parameters not in ``estimated_params`` are taken from ``base_env``
    (i.e., treated as calibrated / fixed).
    """
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (len(estimated_params),):
        raise ValueError(
            f"RiskyDebtEnv SMM expects a {len(estimated_params)}-parameter "
            f"vector matching estimated_params={estimated_params}. "
            f"Got shape {beta.shape}."
        )
    values = _env_param_values(base_env)
    for name, val in zip(estimated_params, beta):
        values[name] = float(val)

    econ = EconomicParams(
        interest_rate=base_env.econ.interest_rate,
        depreciation_rate=base_env.econ.depreciation_rate,
        production_elasticity=values["alpha"],
        cost_convex=values["psi1"],
        tax=base_env.econ.tax,
        default_haircut=values["c_def"],
        cost_inject_fixed=values["eta0"],
        cost_inject_linear=values["eta1"],
    )
    shocks = ShockParams(
        mu=base_env.shocks.mu,
        rho=values["rho"],
        sigma=values["sigma_epsilon"],
    )
    return base_env.clone(econ_params=econ, shock_params=shocks)


def _simulate_smm_panel_data(
    env: RiskyDebtEnv,
    solved_result: dict,
    *,
    run_config: SMMRunConfig,
    n_panels: int,
    seed: tuple[int, int],
) -> RiskyDebtSMMPanelData:
    """On-grid panel simulation using the discrete Markov chain.

    All states stay exactly on the solver grid: k' and b' are grid values
    (policy is grid → grid), and z' is drawn from the Tauchen transition
    matrix.  This avoids interpolation entirely — policy lookup is a direct
    array index — and maintains full consistency between the VFI solver
    and the simulated moments.
    """
    # --- Extract grid arrays and solver outputs ---
    grids = solved_result["grids"]
    z_grid = np.asarray(grids["exo_grids_1d"][0], dtype=np.float64)
    k_grid = np.asarray(grids["endo_grids_1d"][0], dtype=np.float64)
    b_grid = np.asarray(grids["endo_grids_1d"][1], dtype=np.float64)
    n_z, n_k, n_b = len(z_grid), len(k_grid), len(b_grid)
    n_state = n_k * n_b

    policy_idx = np.asarray(solved_result["policy_idx"], dtype=np.int64)
    value_grid = np.maximum(
        np.asarray(solved_result["value"], dtype=np.float64), 0.0
    ).reshape(n_z, n_state)

    r_tilde_grid = np.asarray(
        solved_result["r_tilde_grid"], dtype=np.float64
    ).reshape(n_z, n_state)
    debt_discount_grid = np.zeros_like(r_tilde_grid)
    finite = np.isfinite(r_tilde_grid)
    debt_discount_grid[finite] = 1.0 / np.maximum(
        1.0 + r_tilde_grid[finite], 1e-12
    )
    risk_free_discount = 1.0 / (1.0 + env.econ.interest_rate)

    prob_matrix = np.asarray(solved_result["prob_matrix"], dtype=np.float64)
    cumprob = np.cumsum(prob_matrix, axis=1)  # (n_z, n_z)

    # --- Initial states: snap sampled levels to nearest grid indices ---
    total_units = int(n_panels * run_config.n_firms)
    total_steps = int(run_config.burn_in + run_config.horizon)

    schedule = SeedSchedule(SeedScheduleConfig(master_seed=seed))
    seeds = schedule.get_test_seeds(
        variables=[VariableID.K0, VariableID.Z0, VariableID.EPS1]
    )
    s_endo_0 = env.sample_initial_endogenous(
        total_units, seeds[VariableID.K0]
    ).numpy()
    z_0 = env.sample_initial_exogenous(
        total_units, seeds[VariableID.Z0]
    ).numpy()

    k_idx = np.searchsorted(k_grid, s_endo_0[:, 0]).clip(0, n_k - 1)
    b_idx = np.searchsorted(b_grid, s_endo_0[:, 1]).clip(0, n_b - 1)
    z_idx = np.searchsorted(z_grid, z_0[:, 0]).clip(0, n_z - 1)

    # Uniform draws for Markov chain transitions (replacing continuous ε).
    markov_uniforms = tf.random.stateless_uniform(
        shape=[total_units, total_steps],
        seed=seeds[VariableID.EPS1],
        dtype=tf.float64,
    ).numpy()

    # --- Pre-compute economic constants for cash-flow calculation ---
    delta = env.econ.depreciation_rate
    tau = env.econ.tax
    alpha = env.econ.production_elasticity
    psi1 = env.econ.cost_convex
    eta0 = env.econ.cost_inject_fixed
    eta1 = env.econ.cost_inject_linear
    r_rf = env.econ.interest_rate

    # --- Allocate recording arrays ---
    record_shape = (n_panels, run_config.n_firms, run_config.horizon)
    recorded = {
        name: np.zeros(record_shape, dtype=np.float64)
        for name in (
            "k", "b", "z", "value", "k_next", "b_next",
            "cash_flow", "debt_discount",
        )
    }

    # --- Simulation loop (all on-grid, no interpolation) ---
    for step in range(total_steps):
        # Flat state index into (k, b) product grid
        state_flat = k_idx * n_b + b_idx

        # Policy lookup: (k', b') choice index
        choice_flat = policy_idx[z_idx, state_flat]
        k_next_idx = choice_flat // n_b
        b_next_idx = choice_flat % n_b

        # Grid-level values for recording and cash-flow computation
        k_val = k_grid[k_idx]
        b_val = b_grid[b_idx]
        z_val = z_grid[z_idx]
        k_next_val = k_grid[k_next_idx]
        b_next_val = b_grid[b_next_idx]

        # Debt discount at (z, k', b') — pricing is indexed by current z
        choice_flat_for_pricing = k_next_idx * n_b + b_next_idx
        dd = debt_discount_grid[z_idx, choice_flat_for_pricing]
        dd = np.where(b_next_val <= 1e-12, risk_free_discount, dd)

        # Value at current state
        val = value_grid[z_idx, state_flat]

        # Cash flow (pre-issuance-cost)
        investment = k_next_val - (1.0 - delta) * k_val
        safe_k = np.maximum(k_val, 1e-8)
        after_tax_profit = (1.0 - tau) * z_val * np.power(k_val, alpha)
        adj_cost = 0.5 * psi1 * np.square(investment) / safe_k
        debt_proceeds = b_next_val * dd
        tax_shield = tau * b_next_val * (1.0 - dd) / (1.0 + r_rf)
        cash_flow = (
            after_tax_profit - adj_cost - investment - b_val
            + debt_proceeds + tax_shield
        )

        # Record post-burn-in steps
        if step >= run_config.burn_in:
            idx = step - run_config.burn_in
            panel_shape = (n_panels, run_config.n_firms)
            recorded["k"][..., idx] = k_val.reshape(panel_shape)
            recorded["b"][..., idx] = b_val.reshape(panel_shape)
            recorded["z"][..., idx] = z_val.reshape(panel_shape)
            recorded["value"][..., idx] = val.reshape(panel_shape)
            recorded["k_next"][..., idx] = k_next_val.reshape(panel_shape)
            recorded["b_next"][..., idx] = b_next_val.reshape(panel_shape)
            recorded["cash_flow"][..., idx] = cash_flow.reshape(panel_shape)
            recorded["debt_discount"][..., idx] = dd.reshape(panel_shape)

        # Transition: discrete Markov chain for z, grid indices for k, b
        u = markov_uniforms[:, step]
        z_idx = (cumprob[z_idx] <= u[:, None]).sum(axis=1).clip(0, n_z - 1)
        k_idx = k_next_idx
        b_idx = b_next_idx

    n_observations = int(n_panels * run_config.n_firms * run_config.horizon)
    return RiskyDebtSMMPanelData(
        k=recorded["k"],
        b=recorded["b"],
        z=recorded["z"],
        value=recorded["value"],
        k_next=recorded["k_next"],
        b_next=recorded["b_next"],
        cash_flow=recorded["cash_flow"],
        debt_discount=recorded["debt_discount"],
        n_observations=n_observations,
    )


def _uniforms_to_standard_normal(uniforms: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(uniforms, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return ndtri(clipped)


def _compute_smm_panel_moments(
    env: RiskyDebtEnv,
    panel_data: RiskyDebtSMMPanelData,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the 14 risky-debt SMM moments for each simulated panel.

    Moment ordering (must match ``smm_moment_names``):
      0  avg_equity_issuance_assets    E[max(0,-e)/k]
      1  conditional_issuance_size     E[max(0,-e)/k | e<0]
      2  autocorr_equity_issuance      Corr(iss_t, iss_{t-1})
      3  crosscorr_leverage_issuance   Corr(lev_t, iss_{t+1})
      4  book_leverage                 E[b'/k]               [H&W 2007]
      5  cov_leverage_investment       Cov(b'/k, I/k)        [H&W 2007]
      6  mean_investment_assets        E[I/k]                [H&W 2007]
      7  serial_corr_investment        Corr(I/k_t, I/k_{t-1})
      8  var_investment_assets         Var(I/k)
      9  income_ar1_beta               IV first-diff AR(1) beta
     10  income_ar1_resid_std          IV first-diff AR(1) resid std
     11  default_frequency             Pr(V = 0)
     12  frequency_equity_issuance     Pr(e < 0)              [H&W 2007]
     13  corr_issuance_investment      Corr(max(0,-e)/k, I/k) [H&W 2007, scale-stabilised]

    See ``_UNUSED_MOMENT_CANDIDATES`` for moments retained as documentation
    but not computed.
    """
    n_panels = panel_data.k.shape[0]
    panel_moments = np.zeros((n_panels, 14), dtype=np.float64)
    payout_mean = np.zeros(n_panels, dtype=np.float64)
    payout_var = np.zeros(n_panels, dtype=np.float64)

    safe_k = np.maximum(panel_data.k, 1e-12)
    investment = panel_data.k_next - (1.0 - env.econ.depreciation_rate) * panel_data.k
    investment_ratio = investment / safe_k
    equity_issuance_ratio = np.maximum(0.0, -panel_data.cash_flow) / safe_k
    payout_ratio = np.maximum(0.0, panel_data.cash_flow) / safe_k
    debt_market = panel_data.b_next * panel_data.debt_discount
    asset_value = np.maximum(panel_data.value + debt_market, 1e-8)
    leverage = debt_market / asset_value           # market-value leverage; used by
                                                   # crosscorr_leverage_issuance
    book_leverage_arr = panel_data.b_next / safe_k  # H&W "net debt to assets" = b'/k
    log_income_ratio = np.log(
        np.maximum(panel_data.z * np.power(safe_k, env.econ.production_elasticity - 1.0), 1e-12)
    )

    for idx in range(n_panels):
        invest_panel = investment_ratio[idx]    # (n_firms, horizon)
        issue_panel = equity_issuance_ratio[idx]
        leverage_panel = leverage[idx]
        income_panel = log_income_ratio[idx]

        # [0] Avg equity issuance / assets
        avg_iss = float(np.mean(issue_panel))

        # [1] Conditional issuance size: mean issuance GIVEN issuance occurs
        issuing_mask = panel_data.cash_flow[idx] < 0.0
        if np.any(issuing_mask):
            cond_iss = float(np.mean(issue_panel[issuing_mask]))
        else:
            cond_iss = 0.0

        # [2] Autocorrelation of equity issuance
        autocorr_iss = _panel_serial_correlation(issue_panel)

        # [3] Cross-correlation: leverage_t → issuance_{t+1}
        if issue_panel.shape[-1] >= 2:
            lev_lead = leverage_panel[:, :-1].reshape(-1)
            iss_lag = issue_panel[:, 1:].reshape(-1)
            lev_c = lev_lead - np.mean(lev_lead)
            iss_c = iss_lag - np.mean(iss_lag)
            denom = np.sqrt(np.mean(lev_c ** 2) * np.mean(iss_c ** 2))
            crosscorr_lev_iss = float(np.mean(lev_c * iss_c) / denom) if denom > 1e-12 else 0.0
        else:
            crosscorr_lev_iss = 0.0

        # [4] H&W "net debt to assets" — book leverage E[b'/k].  The natural
        # identifier for c_def: high default cost shrinks equilibrium debt.
        book_lev_panel = book_leverage_arr[idx]
        book_lev = float(np.mean(book_lev_panel))

        # [5] H&W pecking-order moment Cov(b'/k, I/k).  Captures how firms
        # use debt to fund investment; large positive cov ⇒ low default
        # cost (debt is cheap), weak cov ⇒ high default cost.
        cov_lev_inv = _panel_covariance(book_lev_panel, invest_panel)

        # [6] Mean investment / assets — H&W 2007 primary identifier for
        # alpha (production curvature scales the equilibrium investment rate).
        mean_inv = float(np.mean(invest_panel))

        # [7] Serial correlation of investment
        serial_corr_inv = _panel_serial_correlation(invest_panel)

        # [8] Var of investment / assets
        var_inv = float(np.var(invest_panel))

        # [8-9] AR(1) on log income/asset ratio (same as basic model)
        ar_beta, ar_sigma = _panel_iv_first_diff_ar1(income_panel)

        # [10] Default frequency: fraction of firm-periods with V = 0
        value_panel = panel_data.value[idx]  # (n_firms, horizon)
        default_freq = float(np.mean(value_panel <= 0.0))

        # [11] Frequency of equity issuance: Pr(e < 0).  H&W 2007's direct
        # identifier for the fixed equity-issuance cost (eta0): a large fixed
        # cost makes flotations infrequent.
        freq_eq_iss = float(np.mean(panel_data.cash_flow[idx] < 0.0))

        # [12] Correlation of equity issuance and investment.  H&W 2007's
        # pecking-order moment; we use the correlation (not covariance) to
        # avoid pathological Omega_hat conditioning — Cov(Iss, I) has
        # intrinsic population variance O(1e-8) because both Iss and I are
        # near-zero for most firm-years, which would make Omega singular
        # regardless of sample size.  Corr is bounded in [-1, 1].
        iss_c = issue_panel - np.mean(issue_panel)
        inv_c = invest_panel - np.mean(invest_panel)
        denom_iss_inv = np.sqrt(np.mean(iss_c ** 2) * np.mean(inv_c ** 2))
        corr_iss_inv = (
            float(np.mean(iss_c * inv_c) / denom_iss_inv)
            if denom_iss_inv > 1e-12 else 0.0
        )

        panel_moments[idx, :] = np.array(
            [
                avg_iss,
                cond_iss,
                autocorr_iss,
                crosscorr_lev_iss,
                book_lev,
                cov_lev_inv,
                mean_inv,
                serial_corr_inv,
                var_inv,
                ar_beta,
                ar_sigma,
                default_freq,
                freq_eq_iss,
                corr_iss_inv,
            ],
            dtype=np.float64,
        )
        payout_mean[idx] = float(np.mean(payout_ratio[idx]))
        payout_var[idx] = float(np.var(payout_ratio[idx]))

    diagnostics = {
        "payout_mean": payout_mean,
        "payout_var": payout_var,
    }
    return panel_moments, diagnostics


# ---------------------------------------------------------------------------
# Empirical policy function regression moments
# ---------------------------------------------------------------------------

def _chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Compute complete Chebyshev polynomial basis up to given degree.

    Args:
        x: (N, D) array of regressors, each column normalized to [-1, 1].
        degree: Maximum total polynomial degree.

    Returns:
        (N, J) basis matrix where J = C(D + degree, degree) is the number
        of terms in a complete polynomial of degree ``degree`` in D variables.
    """
    n, d = x.shape

    # Univariate Chebyshev polynomials T_0, ..., T_degree for each variable.
    # T[var][deg] has shape (N,).
    T = []
    for v in range(d):
        polys = [np.ones(n, dtype=np.float64), x[:, v].copy()]
        for deg in range(2, degree + 1):
            polys.append(2.0 * x[:, v] * polys[-1] - polys[-2])
        T.append(polys)

    # Enumerate all multi-indices (d_0, ..., d_{D-1}) with sum <= degree.
    from itertools import product as _product
    columns = []
    for powers in _product(range(degree + 1), repeat=d):
        if sum(powers) > degree:
            continue
        col = np.ones(n, dtype=np.float64)
        for v, p in enumerate(powers):
            col = col * T[v][p]
        columns.append(col)

    return np.column_stack(columns)


def _normalize_to_chebyshev_domain(
    values: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Map values from [lo, hi] to [-1, 1] per column."""
    span = np.maximum(hi - lo, 1e-12)
    return 2.0 * (values - lo) / span - 1.0


def compute_policy_regression_moments(
    env: "RiskyDebtEnv",
    panel_data: "RiskyDebtSMMPanelData",
    degree: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Empirical policy function regression moments for indirect inference.

    Fits a Chebyshev polynomial basis regression of next-period choices
    (k', b') on current observable state (k, b, y) where y = z*k^(alpha-1)
    is the income/asset ratio.  The estimated coefficients serve as moments
    for SMM estimation.

    Two regressions are run per panel:
        k'_{it} = sum_j theta^k_j * p_j(x_{it}) + u_{it}
        b'_{it} = sum_j theta^b_j * p_j(x_{it}) + v_{it}

    where p_j are Chebyshev polynomial basis functions and
    x = (k_norm, b_norm, y_norm) are states normalized to [-1, 1].

    Args:
        env:        RiskyDebtEnv (uses production_elasticity for y computation).
        panel_data: Simulated panel data with fields k, b, z, k_next, b_next.
        degree:     Maximum total polynomial degree for the basis.

    Returns:
        (panel_moments, moment_names):
            panel_moments: (n_panels, 2*J) array of regression coefficients.
            moment_names:  List of 2*J moment name strings.
    """
    n_panels = panel_data.k.shape[0]
    alpha = env.econ.production_elasticity

    # Compute observable income/asset ratio
    safe_k = np.maximum(panel_data.k, 1e-12)
    income_ratio = panel_data.z * np.power(safe_k, alpha - 1.0)

    # Fixed normalization bounds from the environment grid.
    # Using data-dependent bounds (min/max of the panel) would make the
    # Chebyshev basis different across panels with different random seeds,
    # rendering the regression coefficients incomparable and failing the
    # oracle test.  Fixed bounds ensure identical basis functions everywhere.
    y_lo = env.z_min * max(env.k_min, 1e-12) ** (alpha - 1.0)
    y_hi = env.z_max * env.k_max ** (alpha - 1.0)
    x_lo = np.array([env.k_min, env.b_min, y_lo], dtype=np.float64)
    x_hi = np.array([env.k_max, env.b_max, y_hi], dtype=np.float64)

    # Compute basis dimension from a dummy call
    dummy_x = _normalize_to_chebyshev_domain(
        np.zeros((1, 3), dtype=np.float64), x_lo, x_hi
    )
    n_basis = _chebyshev_basis(dummy_x, degree).shape[1]

    panel_moments = np.zeros((n_panels, 2 * n_basis), dtype=np.float64)

    for idx in range(n_panels):
        k_panel = panel_data.k[idx].reshape(-1)
        b_panel = panel_data.b[idx].reshape(-1)
        y_panel = income_ratio[idx].reshape(-1)
        k_next_panel = panel_data.k_next[idx].reshape(-1)
        b_next_panel = panel_data.b_next[idx].reshape(-1)

        x_raw = np.column_stack([k_panel, b_panel, y_panel])
        x_norm = _normalize_to_chebyshev_domain(x_raw, x_lo, x_hi)
        basis = _chebyshev_basis(x_norm, degree)  # (N_obs, J)

        # OLS: theta = (P'P)^{-1} P' y
        PtP = basis.T @ basis
        reg = max(1e-10 * np.trace(PtP) / n_basis, 1e-12)
        PtP_reg = PtP + reg * np.eye(n_basis)

        theta_k = np.linalg.solve(PtP_reg, basis.T @ k_next_panel)
        theta_b = np.linalg.solve(PtP_reg, basis.T @ b_next_panel)

        panel_moments[idx, :n_basis] = theta_k
        panel_moments[idx, n_basis:] = theta_b

    # Generate moment names
    moment_names = []
    for prefix in ["k_next", "b_next"]:
        for j in range(n_basis):
            moment_names.append(f"policy_reg_{prefix}_c{j}")

    return panel_moments, moment_names


def _exogenous_transition_np(
    env: RiskyDebtEnv,
    z: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    log_z = np.log(np.maximum(np.asarray(z, dtype=np.float64), 1e-8))
    eps = np.asarray(eps, dtype=np.float64)
    log_z_next = (
        (1.0 - env.shocks.rho) * env.shocks.mu
        + env.shocks.rho * log_z
        + env.shocks.sigma * eps
    )
    return np.exp(log_z_next)
