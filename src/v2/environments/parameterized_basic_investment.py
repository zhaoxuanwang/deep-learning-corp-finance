"""β-parameterized variant of the basic investment env (frictional).

Used to train a neural surrogate policy φ_NN(k, z, β) over the prior
support of β = (α, ρ, σ_ε, φ_quad, φ_prop). Sibling of
`BasicInvestmentEnv`, not a modification of it — the production env stays
untouched.

End goal: support the Bayesian inference pipeline for the **frictional**
basic investment model (Bayesian.md §1.3 model #2). The frictionless case
(φ_quad = φ_prop = 0) is a validation special case where the
closed-form analytical policy provides a ground-truth reference.

Differences from `BasicInvestmentEnv`:

  * Structural parameters (α, ρ, σ_ε, φ_quad, φ_prop) are NOT stored as
    instance attributes. They are passed per-call as a `beta` tensor of
    shape (batch, 5).
  * Calibrated parameters r, δ, μ remain instance attributes.
  * Adjustment cost ψ(k, I) = 0.5·φ_quad·I²/k + φ_prop·|I| is implemented
    in a self-contained helper (`_friction_cost`) using only β-derived
    coefficients. The existing `EconomicParams` dataclass is NOT touched.
  * State-space bounds (k_min, k_max, z_min, z_max) are derived once from
    nominal (α, ρ, σ_ε) anchors. Frictions do not enter the bound
    calculation (k* in the bound formula is the frictionless steady state).
  * `analytical_policy(s, β)` is only valid at the frictionless slice
    (φ_quad = φ_prop = 0) and raises otherwise — there is no closed form
    once frictions are active.

Method signatures (all require β):

  * exogenous_transition(s_exo, eps, beta)
  * endogenous_transition(s_endo, action, s_exo, beta)
  * reward(s, a, beta)
  * euler_residual(s, a, s_next, a_next, beta)
  * analytical_policy(s, beta)              — frictionless-only

`beta` tensor is shape (batch, 5) ordered (α, ρ, σ_ε, φ_quad, φ_prop).
"""

from __future__ import annotations

from typing import Mapping, Optional

import tensorflow as tf

from src.v2.environments.base import MDPEnvironment
from src.v2.environments.basic_investment import (
    BasicInvestmentEnv,
    EconomicParams,
    ShockParams,
)


_BETA_DIM = 5   # (α, ρ, σ_ε, φ_quad, φ_prop)
_FRICTIONLESS_TOL = 1e-8


def _split_beta(beta: tf.Tensor):
    """Split β tensor into (α, ρ, σ_ε, φ_quad, φ_prop), each shape (..., 1)."""
    alpha    = beta[..., 0:1]
    rho      = beta[..., 1:2]
    sigma    = beta[..., 2:3]
    phi_quad = beta[..., 3:4]
    phi_prop = beta[..., 4:5]
    return alpha, rho, sigma, phi_quad, phi_prop


def _friction_cost(
    k: tf.Tensor, investment: tf.Tensor,
    phi_quad: tf.Tensor, phi_prop: tf.Tensor,
) -> tf.Tensor:
    """Quadratic + proportional adjustment cost ψ(k, I).

        ψ(k, I) = 0.5 · φ_quad · I² / k  +  φ_prop · |I|.

    All inputs broadcast to a common batch shape. Self-contained: does not
    depend on `EconomicParams` so the existing dataclass stays untouched.
    """
    safe_k = tf.maximum(k, 1e-8)
    adj_q  = 0.5 * phi_quad * tf.square(investment) / safe_k
    adj_p  = phi_prop * tf.abs(investment)
    return adj_q + adj_p


class ParameterizedBasicInvestmentEnv(MDPEnvironment):
    """β-conditional basic investment env supporting frictional reward.

    Two ways to specify the state-space box:

    * **Multiplier path** (default): provide ``k_min_mult, k_max_mult,
      z_sd_mult``; the env computes bounds from the nominal frictionless
      ``k_star`` and the stationary-distribution std at nominal β.
      Convenient for prototyping but ties the box to one specific β.
    * **Explicit-bounds path**: provide ``bounds={"k_min", "k_max",
      "z_min", "z_max"}`` (typically computed via
      ``derive_env_bounds_from_panel(...)`` in the inference notebook).
      Use this when you want NN training to align with the (k, z) range
      that real or simulated panel data actually visits.

    The two paths are mutually exclusive — pass one or the other, not both.

    Args:
        nominal_econ:   `EconomicParams` providing the nominal anchor for
                        bounds (uses production_elasticity, interest_rate,
                        depreciation_rate). Cost fields are IGNORED — the
                        new env's adjustment cost comes from β at call time.
        nominal_shocks: `ShockParams` providing nominal anchor for z bounds
                        (uses rho, sigma, mu).
        k_min_mult:     Lower k bound as multiplier on k*(nominal). Multiplier path.
        k_max_mult:     Upper k bound as multiplier on k*(nominal). Multiplier path.
        z_sd_mult:      Number of ergodic std devs for z bounds. Multiplier path.
        bounds:         Optional dict with explicit
                        ``{"k_min", "k_max", "z_min", "z_max"}`` floats.
                        Mutually exclusive with the multiplier args.

    Notes:
        * The nominal anchor is frictionless by construction — frictions do
          not affect k* in the formula. Choose nominal (α, ρ, σ_ε) at the
          prior mean (e.g., α=ρ=0.5, σ_ε≈0.24).
        * Multiplier path: bounds should be wider than for single-β training
          (k_min_mult ≤ 0.1, k_max_mult ≥ 10) to cover the prior support.
        * Explicit-bounds path: ``bounds`` should come from observed-panel
          min/max + margin (k) and prior worst-case stationary distribution
          (z).
    """

    def __init__(
        self,
        nominal_econ:   EconomicParams = None,
        nominal_shocks: ShockParams    = None,
        k_min_mult:     float          = 0.1,
        k_max_mult:     float          = 10.0,
        z_sd_mult:      float          = 4.0,
        bounds:         Optional[Mapping[str, float]] = None,
    ):
        nominal_econ   = nominal_econ   or EconomicParams()
        nominal_shocks = nominal_shocks or ShockParams()

        # Bounds come from a nominal frictionless env (k_star, z_min/max).
        # Strip any cost fields from the nominal so bounds are pure frictionless.
        nominal_frictionless = EconomicParams(
            interest_rate         = nominal_econ.interest_rate,
            depreciation_rate     = nominal_econ.depreciation_rate,
            production_elasticity = nominal_econ.production_elasticity,
            cost_convex           = 0.0,
            cost_fixed            = 0.0,
        )
        # Always build the nominal env at default multipliers — we use it
        # for ``k_star`` (for diagnostics) and for some sampler / action-spec
        # helpers. When the user supplies explicit ``bounds``, we override
        # k_min/max, z_min/max, I_min/max below before exposing them.
        self._nominal = BasicInvestmentEnv(
            econ_params  = nominal_frictionless,
            shock_params = nominal_shocks,
            k_min_mult   = k_min_mult,
            k_max_mult   = k_max_mult,
            z_sd_mult    = z_sd_mult,
        )

        # Calibrated scalars (β-independent)
        self.r_rate     = float(nominal_econ.interest_rate)
        self.delta_rate = float(nominal_econ.depreciation_rate)
        self.mu         = float(nominal_shocks.mu)
        self.gamma      = 1.0 / (1.0 + self.r_rate)

        if bounds is None:
            # Multiplier path — bounds derived from nominal β.
            self.k_min  = float(self._nominal.k_min)
            self.k_max  = float(self._nominal.k_max)
            self.z_min  = float(self._nominal.z_min)
            self.z_max  = float(self._nominal.z_max)
            self.I_min  = float(self._nominal.I_min)
            self.I_max  = float(self._nominal.I_max)
        else:
            # Explicit-bounds path — validate and override.
            required = {"k_min", "k_max", "z_min", "z_max"}
            missing = required - set(bounds.keys())
            if missing:
                raise ValueError(
                    f"bounds dict missing required keys: {sorted(missing)}. "
                    f"Required: {sorted(required)}."
                )
            k_lo, k_hi = float(bounds["k_min"]), float(bounds["k_max"])
            z_lo, z_hi = float(bounds["z_min"]), float(bounds["z_max"])
            if not (k_lo > 0 and k_hi > k_lo):
                raise ValueError(
                    f"Need 0 < k_min < k_max; got k_min={k_lo}, k_max={k_hi}.")
            if not (z_lo > 0 and z_hi > z_lo):
                raise ValueError(
                    f"Need 0 < z_min < z_max; got z_min={z_lo}, z_max={z_hi}.")
            self.k_min = k_lo
            self.k_max = k_hi
            self.z_min = z_lo
            self.z_max = z_hi
            # Action bounds derived consistently from the new k bounds + δ:
            #   I_min = k_min - (1-δ) k_max  (largest possible disinvestment)
            #   I_max = k_max - (1-δ) k_min  (largest possible investment)
            one_minus_d = 1.0 - self.delta_rate
            self.I_min = k_lo - one_minus_d * k_hi
            self.I_max = k_hi - one_minus_d * k_lo
            # Refit the nominal env's bounds-related fields so the initial-
            # state samplers (which delegate to `self._nominal`) draw from
            # the explicit box. The nominal env's k_star is unaffected.
            self._nominal.k_min = k_lo
            self._nominal.k_max = k_hi
            self._nominal.z_min = z_lo
            self._nominal.z_max = z_hi
            self._nominal.I_min = self.I_min
            self._nominal.I_max = self.I_max

        self.k_star = float(self._nominal.k_star)

        # Nominal anchors retained for diagnostics (NOT used in dynamics).
        self.nominal_econ   = nominal_frictionless
        self.nominal_shocks = nominal_shocks

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def exo_dim(self) -> int:
        return 1

    def endo_dim(self) -> int:
        return 1

    def action_dim(self) -> int:
        return 1

    def beta_dim(self) -> int:
        return _BETA_DIM

    # ------------------------------------------------------------------
    # Action space — same bounds as the frictionless nominal
    # ------------------------------------------------------------------

    def action_bounds(self) -> tuple:
        return self._nominal.action_bounds()

    def action_scale_reference(self) -> tuple:
        return self._nominal.action_scale_reference()

    # ------------------------------------------------------------------
    # Initial state sampling — uniform inside fixed bounds
    # ------------------------------------------------------------------

    def sample_initial_endogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        return self._nominal.sample_initial_endogenous(n, seed)

    def sample_initial_exogenous(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        return self._nominal.sample_initial_exogenous(n, seed)

    # ------------------------------------------------------------------
    # Constraint helper (β-independent — only δ matters here)
    # ------------------------------------------------------------------

    def _apply_action(self, k: tf.Tensor, a: tf.Tensor):
        """Compute constrained k_next and back-computed I_effective."""
        I      = a[..., 0]
        k_next = (1.0 - self.delta_rate) * k + I
        k_next = tf.clip_by_value(k_next, self.k_min, self.k_max)
        I_eff  = k_next - (1.0 - self.delta_rate) * k
        return k_next, I_eff

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def exogenous_transition(
        self, s_exo: tf.Tensor, eps: tf.Tensor, beta: tf.Tensor
    ) -> tf.Tensor:
        """AR(1) step under per-sample β: z' = exp((1-ρ)μ + ρ·log z + σ·ε).

        Args:
            s_exo: shape (batch, 1) — current z (levels).
            eps:   shape (batch, 1) — standard normal shock.
            beta:  shape (batch, 5) — (α, ρ, σ_ε, φ_quad, φ_prop).

        Returns:
            z_next: shape (batch, 1).
        """
        _, rho, sigma, _, _ = _split_beta(beta)
        z          = s_exo[..., 0:1]
        log_z      = tf.math.log(tf.maximum(z, 1e-8))
        log_z_next = (1.0 - rho) * self.mu + rho * log_z + sigma * eps[..., 0:1]
        return tf.exp(log_z_next)

    def endogenous_transition(
        self, s_endo: tf.Tensor, action: tf.Tensor, s_exo: tf.Tensor,
        beta: tf.Tensor,
    ) -> tf.Tensor:
        """Capital accumulation k' = clip((1-δ)k + I, k_min, k_max).

        β is not consumed (k' depends only on calibrated δ); kept for API
        uniformity with the other β-aware methods.
        """
        del beta
        k          = s_endo[..., 0]
        k_next, _  = self._apply_action(k, action)
        return tf.reshape(k_next, [-1, 1])

    # ------------------------------------------------------------------
    # Reward — frictional, β-conditional
    # ------------------------------------------------------------------

    def reward(self, s: tf.Tensor, a: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        """Cash flow:  profit(k, z; α) − ψ(k, I; φ_quad, φ_prop) − I_eff.

        Returns shape (batch,).
        """
        alpha, _, _, phi_quad, phi_prop = _split_beta(beta)
        k = s[..., 0:1]
        z = s[..., 1:2]
        _, investment = self._apply_action(k[..., 0], a)
        investment = tf.reshape(investment, [-1, 1])
        safe_k = tf.maximum(k, 1e-8)
        profit = z * tf.pow(safe_k, alpha)
        adj    = _friction_cost(k, investment, phi_quad, phi_prop)
        return (profit - adj - investment)[..., 0]

    # ------------------------------------------------------------------
    # Discount
    # ------------------------------------------------------------------

    def discount(self) -> float:
        return self.gamma

    # ------------------------------------------------------------------
    # Euler residual — general (tape-based), frictional model
    # ------------------------------------------------------------------

    def euler_residual(
        self, s: tf.Tensor, a: tf.Tensor,
        s_next: tf.Tensor, a_next: tf.Tensor, beta: tf.Tensor,
    ) -> tf.Tensor:
        """Unit-free Euler residual under per-sample β.

            χ   = 1 + ∂ψ(k, I)/∂k_next
            m   = π_{k'} − ∂ψ'(k', I')/∂k' + (1−δ) · χ'
            residual = 1 − γ · m / χ

        Frictionless slice (φ = 0): collapses to 1 − γ·(π_{k'} + (1−δ))
        because ψ ≡ 0 forces all friction derivatives to zero.
        """
        alpha, _, _, phi_quad, phi_prop = _split_beta(beta)
        alpha_   = alpha[..., 0]
        phi_q    = phi_quad[..., 0]
        phi_p    = phi_prop[..., 0]

        k        = s[..., 0]
        z_next   = s_next[..., 1]
        k_next, _      = self._apply_action(k, a)
        k_next_next, _ = self._apply_action(k_next, a_next)

        # Marginal product of next-period capital
        safe_k_next = tf.maximum(k_next, 1e-8)
        pi_k = alpha_ * z_next * tf.pow(safe_k_next, alpha_ - 1.0)

        # χ = 1 + ∂ψ(k, I)/∂k_next   where I = k_next - (1-δ)·k
        with tf.GradientTape() as tape:
            tape.watch(k_next)
            investment = k_next - (1.0 - self.delta_rate) * k
            psi = _friction_cost(k, investment, phi_q, phi_p)
        dpsi_dk_next = tape.gradient(
            psi, k_next, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        chi = 1.0 + dpsi_dk_next

        # ∂ψ'/∂k' and χ' = 1 + ∂ψ'/∂k_next_next
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(k_next)
            tape.watch(k_next_next)
            investment_next = k_next_next - (1.0 - self.delta_rate) * k_next
            psi_next = _friction_cost(
                k_next, investment_next, phi_q, phi_p)
        dpsi_next_dk = tape.gradient(
            psi_next, k_next, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        chi_next = 1.0 + tape.gradient(
            psi_next, k_next_next,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape

        m = pi_k - dpsi_next_dk + (1.0 - self.delta_rate) * chi_next
        return 1.0 - self.gamma * m / chi

    # ------------------------------------------------------------------
    # Closed-form analytical reference — frictionless slice only
    # ------------------------------------------------------------------

    def analytical_kprime(self, z: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        """Closed-form k'(z; β) for the frictionless slice.

            k'(z; β) = ( α · E[z'|z] / (r + δ) )^(1/(1-α)),
            E[z'|z] = exp((1-ρ)μ + ρ log z + 0.5 σ²).

        Defined only when (φ_quad, φ_prop) ≈ 0. Raises otherwise.

        Args:
            z:    shape (batch, 1) — current z (levels).
            beta: shape (batch, 5).

        Returns:
            k'(z; β) clipped to [k_min, k_max], shape (batch, 1).
        """
        alpha, rho, sigma, phi_quad, phi_prop = _split_beta(beta)
        _assert_frictionless(phi_quad, phi_prop)
        z       = tf.reshape(z, [-1, 1])
        log_z   = tf.math.log(tf.maximum(z, 1e-8))
        log_ez  = (1.0 - rho) * self.mu + rho * log_z + 0.5 * sigma * sigma
        log_brk = tf.math.log(alpha) + log_ez - tf.math.log(self.r_rate + self.delta_rate)
        kp      = tf.exp(log_brk / (1.0 - alpha))
        return tf.clip_by_value(kp, self.k_min, self.k_max)

    def analytical_policy(
        self, s: tf.Tensor, beta: tf.Tensor, training: bool = False,
    ) -> tf.Tensor:
        """Closed-form optimal investment I*(k, z; β) = k'(z;β) − (1−δ)·k.

        Only defined on the frictionless slice (φ_quad ≈ φ_prop ≈ 0).
        Raises ValueError otherwise.
        """
        del training
        k     = s[..., 0:1]
        z     = s[..., 1:2]
        kp    = self.analytical_kprime(z, beta)
        return kp - (1.0 - self.delta_rate) * k

    # ------------------------------------------------------------------
    # Panel simulator — single source of truth for state-space alignment
    # ------------------------------------------------------------------

    def simulate_panel(
        self,
        beta: tf.Tensor,
        policy_fn,
        n_firms: int,
        horizon: int,
        burn_in: int,
        seed: tuple[int, int],
    ) -> dict[str, "np.ndarray"]:
        """Roll out (k, z) trajectories using this env's transitions.

        Single source of truth for the (state, β, transition) contract that
        the NN saw during training. Both NN training and Bayesian panel
        synthesis go through this method, so they cannot drift apart:

          * Initial states from ``self.sample_initial_endogenous`` /
            ``sample_initial_exogenous`` — the exact distribution the
            trainer warmed the normalizer on.
          * Exogenous step via ``self.exogenous_transition(z, ε, β)``.
          * Endogenous step via ``self._apply_action(k, action)``.
          * Bounds ``self.k_min, self.k_max``.

        The only free choice is the ``policy_fn`` that produces the action
        at each step: pass ``self.analytical_policy`` for the closed-form
        reference, or a trained NN ``policy_nn(s, β)`` to generate panels
        under the surrogate.

        Args:
            beta:      shape ``[5]`` candidate parameter vector
                       ``(α, ρ, σ_ε, φ_quad, φ_prop)``.
            policy_fn: callable ``(state, beta) -> action`` accepting
                       ``state[batch, 2]`` and ``beta[batch, 5]`` and
                       returning ``action[batch, 1]``. Frictionless slice
                       only: ``self.analytical_policy``.
            n_firms:   N — number of firms.
            horizon:   T — recorded periods (after burn-in).
            burn_in:   periods discarded so the panel reaches the policy's
                       stationary distribution.
            seed:      ``(m0, m1)`` deterministic seed pair.

        Returns:
            ``{"k": np.ndarray (N, T), "z": np.ndarray (N, T)}``.
        """
        import numpy as np
        from src.v2.utils.seeding import fold_in_seed

        total_T = int(burn_in) + int(horizon)

        seed_k0  = tf.constant(fold_in_seed(seed, "k0"),  tf.int32)
        seed_z0  = tf.constant(fold_in_seed(seed, "z0"),  tf.int32)
        seed_eps = tf.constant(fold_in_seed(seed, "eps"), tf.int32)

        s_endo = self.sample_initial_endogenous(n_firms, seed=seed_k0)  # [N, 1]
        z      = self.sample_initial_exogenous(n_firms, seed=seed_z0)   # [N, 1]
        k      = s_endo[:, 0:1]

        beta_2d = tf.reshape(tf.cast(beta, tf.float32), [1, 5])
        beta_batch = tf.broadcast_to(beta_2d, [n_firms, 5])

        eps_all = tf.random.stateless_normal(
            shape=[n_firms, total_T, 1],
            seed=seed_eps, dtype=tf.float32,
        )

        k_hist = []
        z_hist = []
        for t_idx in range(total_T):
            state  = tf.concat([k, z], axis=-1)
            action = policy_fn(state, beta_batch)
            k_next, _ = self._apply_action(k[..., 0], action)
            k_next = tf.reshape(k_next, [-1, 1])
            z_next = self.exogenous_transition(z, eps_all[:, t_idx, :], beta_batch)

            k_hist.append(k_next)
            z_hist.append(z_next)
            k, z = k_next, z_next

        k_arr = tf.stack(k_hist, axis=1)[:, int(burn_in):, :].numpy().squeeze(-1)
        z_arr = tf.stack(z_hist, axis=1)[:, int(burn_in):, :].numpy().squeeze(-1)
        return {"k": k_arr, "z": z_arr}

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def stationary_exo(self) -> tf.Tensor:
        return self._nominal.stationary_exo()

    def validate_nn_training_support(self, trainer_name: str) -> None:
        """Frictional model is fully supported by smooth NN trainers.

        Caller must use a β-aware trainer (`train_er_param`, `train_shac_param`).
        We don't gate on trainer name here — the new trainers pass β through,
        the old trainers will fail at call time because they don't pass β.
        """
        del trainer_name


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _assert_frictionless(phi_quad: tf.Tensor, phi_prop: tf.Tensor) -> None:
    """Raise if any sample has a non-trivial friction component.

    Tolerance is _FRICTIONLESS_TOL (1e-8) per coordinate. Use to gate
    `analytical_policy` / `analytical_kprime` which only have closed-form
    expressions at the frictionless slice.
    """
    max_phi_q = float(tf.reduce_max(tf.abs(phi_quad)))
    max_phi_p = float(tf.reduce_max(tf.abs(phi_prop)))
    if max_phi_q > _FRICTIONLESS_TOL or max_phi_p > _FRICTIONLESS_TOL:
        raise ValueError(
            "analytical_policy / analytical_kprime is only defined on the "
            "frictionless slice (φ_quad = φ_prop = 0). Got "
            f"max|φ_quad|={max_phi_q:.3e}, max|φ_prop|={max_phi_p:.3e}."
        )
