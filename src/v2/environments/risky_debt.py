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

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src.v2.environments.base import MDPEnvironment


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
    ):
        if b_min_mult < 0.0:
            raise ValueError(
                f"b_min_mult must be >= 0. Got {b_min_mult}"
            )
        self.econ = econ_params or EconomicParams()
        self.shocks = shock_params or ShockParams()
        self.beta = 1.0 / (1.0 + self.econ.interest_rate)

        rho = self.shocks.rho
        sigma = self.shocks.sigma
        mu = self.shocks.mu
        sigma_ergodic = sigma / np.sqrt(1.0 - rho ** 2)

        self.k_ref = _compute_k_ref(self.econ, self.shocks)
        self.k_min = k_min_mult * self.k_ref
        self.k_max = k_max_mult * self.k_ref
        self.b_max = b_max_mult * self.k_max
        self.b_min = -b_min_mult * self.b_max
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


def _nearest_idx(values: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    """Find the nearest grid index for each query value."""

    values = tf.reshape(tf.cast(values, tf.float32), [-1, 1])
    grid = tf.reshape(tf.cast(grid, tf.float32), [1, -1])
    distance = tf.abs(values - grid)
    return tf.argmin(distance, axis=-1, output_type=tf.int32)
