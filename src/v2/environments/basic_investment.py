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
"""

import tensorflow as tf
from src.v2.environments.base import MDPEnvironment
from src.economy.parameters import EconomicParams, ShockParams
from src.economy import production_function, adjustment_costs


class BasicInvestmentEnv(MDPEnvironment):
    """Corporate finance basic model: optimal capital investment.

    Args:
        econ_params:  economic parameters (r, delta, theta, costs).
        shock_params: AR(1) shock parameters (rho, sigma, mu).
        k_min_mult:   lower capital bound as multiplier on k*.
        k_max_mult:   upper capital bound as multiplier on k*.
        z_sd_mult:    number of ergodic std devs for z bounds.
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
        self.beta   = 1.0 / (1.0 + self.econ.r_rate)

        # z bounds in levels
        rho, sigma, mu = self.shocks.rho, self.shocks.sigma, self.shocks.mu
        sigma_ergodic = float(sigma / tf.sqrt(1.0 - rho ** 2))
        self.z_min = float(tf.exp(mu - z_sd_mult * sigma_ergodic))
        self.z_max = float(tf.exp(mu + z_sd_mult * sigma_ergodic))

        # k bounds
        self.k_star = self._frictionless_kprime(float(tf.exp(mu)), self.econ, self.shocks)
        self.k_min  = k_min_mult * self.k_star
        self.k_max  = k_max_mult * self.k_star

        # Analytical reference k'(z_max) — for analysis only
        self.k_ref = self._frictionless_kprime(self.z_max, self.econ, self.shocks)

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
            tf.constant([-self.k_max], dtype=tf.float32),
            tf.constant([ self.k_max], dtype=tf.float32),
        )

    def action_scale_reference(self) -> tuple:
        I_ss       = self.econ.delta * self.k_star
        center     = tf.constant([I_ss],       dtype=tf.float32)
        half_range = tf.constant([self.k_max], dtype=tf.float32)
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
        k_next = (1.0 - self.econ.delta) * k + I
        k_next = tf.maximum(k_next, self.k_min)
        I_eff  = k_next - (1.0 - self.econ.delta) * k
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

    def reward(self, s: tf.Tensor, a: tf.Tensor, temperature: float = 1e-6) -> tf.Tensor:
        """Cash flow: profit - adjustment_cost - I_effective.

        s = merge_state(k, z), so s[..., 0] = k, s[..., 1] = z.
        """
        k = s[..., 0]
        z = s[..., 1]
        k_next, I_eff = self._apply_action(k, a)

        profit   = production_function(k, z, self.econ)
        adj_cost = adjustment_costs(k, k_next, self.econ,
                                    temperature=temperature, logit_clip=20.0)
        return profit - adj_cost - I_eff

    def exo_stationary_mean(self) -> tf.Tensor:
        """Stationary mean of z: exp(μ), shape (1,)."""
        return self._z_bar

    def stationary_action(self, s_endo: tf.Tensor) -> tf.Tensor:
        """Action holding k constant: I = δ·k, shape (batch, 1)."""
        k = s_endo[..., 0:1]
        return self.econ.delta * k

    def discount(self) -> float:
        return self.beta

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

    def compute_reward_scale(self, n_samples: int = 1000,
                             seed: tf.Tensor = None) -> float:
        """Analytical λ = 1 / |V*(k*, z_mean)|."""
        s_endo_ss = tf.constant([[self.k_star]])
        abs_v = max(float(tf.abs(self.terminal_value(s_endo_ss)[0])), 1e-8)
        return 1.0 / abs_v

    def euler_residual(
        self, s: tf.Tensor, a: tf.Tensor,
        s_next: tf.Tensor, a_next: tf.Tensor,
        temperature: float = 1e-6,
    ) -> tf.Tensor:
        """Unit-free Euler residual: 1 - β·m/χ.

        χ = 1 + ∂ψ/∂k'        (marginal cost of current investment)
        m = π_k' - ∂ψ'/∂k' + (1-δ)·χ'  (marginal benefit next period)

        At optimum: E[residual] = 0.
        """
        k      = s[..., 0]
        z_next = s_next[..., 1]
        k_next, _      = self._apply_action(k, a)
        k_next_next, _ = self._apply_action(k_next, a_next)

        # π_k' = θ·z'·k'^(θ-1)
        safe_k = tf.maximum(k_next, 1e-8)
        pi_k   = self.econ.theta * z_next * tf.pow(safe_k, self.econ.theta - 1)

        # χ = 1 + ∂ψ(k, k')/∂k'
        with tf.GradientTape() as tape:
            tape.watch(k_next)
            psi = adjustment_costs(k, k_next, self.econ,
                                   temperature=temperature, logit_clip=20.0)
        chi = 1.0 + tape.gradient(psi, k_next)

        # m via derivatives of next-period adjustment cost
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(k_next)
            tape.watch(k_next_next)
            psi_next = adjustment_costs(k_next, k_next_next, self.econ,
                                        temperature=temperature, logit_clip=20.0)
        dpsi_next_dk = tape.gradient(psi_next, k_next)
        chi_next     = 1.0 + tape.gradient(psi_next, k_next_next)
        del tape

        m = pi_k - dpsi_next_dk + (1.0 - self.econ.delta) * chi_next
        return 1.0 - self.beta * m / chi

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frictionless_kprime(z, econ, shocks):
        """Frictionless optimal k'(z) = [θ·E[z'|z] / (r+δ)]^(1/(1-θ))."""
        rho, sigma, mu = shocks.rho, shocks.sigma, shocks.mu
        exp_corr = float(tf.exp((1.0 - rho) * mu + 0.5 * sigma ** 2))
        return float(
            (econ.theta * exp_corr * z ** rho
             / (econ.r_rate + econ.delta)
             ) ** (1.0 / (1.0 - econ.theta))
        )
