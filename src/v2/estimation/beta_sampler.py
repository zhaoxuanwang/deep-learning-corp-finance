"""Training-time sampler for the β-conditional neural surrogate.

Draws β = (α, ρ, σ_ε, φ_quad, φ_prop) **uniformly** from per-coordinate
boxes. The Bayesian.md §2.5 prior (Beta(2, 2), HalfNormal) is 3–4× sparser
at the validation-sweep tails than at the bulk; uniform training gives
the surrogate even accuracy across the support.

Decoupling rationale: the Bayesian-inference pipeline (`bayesian.py`,
`bayesian_basic_investment.py`) defines its own TFP prior at MCMC time
and does NOT use this sampler. The surrogate is just an amortized policy
that needs to be uniformly accurate over the support — training
distribution and inference prior are independent choices.

Reproducibility contract
------------------------
``BetaSampler.sample(B, seed)`` is stateless: same ``seed`` → same tensor
(bit-identical on the same hardware). Inside ``train_er_param`` /
``train_shac_param`` the per-step β draws are seeded by
``fold_in_seed(master_seed, trainer_name, "step", step, "beta")`` so the
entire training run is reproducible from the trainer's ``master_seed``
alone — even though β is drawn per minibatch rather than baked into the
dataset. Pinned by ``test_per_step_beta_seed_is_deterministic`` in
``src/v2/tests/test_er_param.py``.

Usage::

    sampler = BetaSampler()                       # full 5-D uniform
    beta    = sampler.sample(1024, seed=(0, 0))   # → (1024, 5)

    # Frictionless slice for nb 08 (φ_quad, φ_prop forced to 0)
    sampler_frictionless = BetaSampler(freeze_dims=(3, 4))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import tensorflow as tf

from src.v2.utils.seeding import MasterSeed, fold_in_seed


BETA_DIM_NAMES: Tuple[str, ...] = (
    "alpha", "rho", "sigma_epsilon", "phi_quad", "phi_prop",
)
BETA_DIM = len(BETA_DIM_NAMES)


# Default per-coordinate uniform-sampling boxes. Cover the Bayesian.md
# §2.5 prior bulk plus a small margin at the validation-sweep ranges,
# and stay clear of degenerate regions (ρ → 1 unit-root, α → 1 explosive
# k*, σ → 0 degenerate likelihood). φ_quad / φ_prop bounds cover the
# 99th percentile of the HalfNormal priors used by the Bayesian module.
DEFAULT_UNIFORM_BOUNDS: Mapping[str, Tuple[float, float]] = {
    "alpha":          (0.20, 0.85),
    "rho":            (0.10, 0.90),
    "sigma_epsilon":  (0.05, 0.60),
    "phi_quad":       (0.0,  1.5),
    "phi_prop":       (0.0,  0.15),
}


@dataclass(frozen=True)
class BetaSampler:
    """Stateless uniform draw of β.

    Args:
        uniform_bounds: Per-coordinate (low, high) box for each of the 5
                        β dimensions, in the order
                        (α, ρ, σ_ε, φ_quad, φ_prop). None → default box
                        ``DEFAULT_UNIFORM_BOUNDS`` (cf. module docstring).
        freeze_dims:    Tuple of β-dim indices held identically at zero.
                        Defaults to (). Use ``(3, 4)`` for the frictionless
                        slice (nb 08).
    """

    uniform_bounds: Tuple[Tuple[float, float], ...] = None
    freeze_dims:    Tuple[int, ...] = ()

    def __post_init__(self):
        # freeze_dims validation
        seen = set()
        for d in self.freeze_dims:
            if not (0 <= d < BETA_DIM):
                raise ValueError(
                    f"freeze_dims index {d} out of range [0, {BETA_DIM}).")
            if d in seen:
                raise ValueError(f"freeze_dims contains duplicate index {d}.")
            seen.add(d)

        # uniform_bounds fill + validate
        if self.uniform_bounds is None:
            bounds = tuple(
                DEFAULT_UNIFORM_BOUNDS[name] for name in BETA_DIM_NAMES)
            object.__setattr__(self, "uniform_bounds", bounds)
        else:
            bounds = tuple(self.uniform_bounds)
            if len(bounds) != BETA_DIM:
                raise ValueError(
                    f"uniform_bounds must have {BETA_DIM} entries. Got {len(bounds)}.")
            object.__setattr__(self, "uniform_bounds", bounds)
        for name, (lo, hi) in zip(BETA_DIM_NAMES, self.uniform_bounds):
            if lo >= hi:
                raise ValueError(
                    f"uniform_bounds for {name}: low ({lo}) must be < high ({hi}).")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, seed: MasterSeed) -> tf.Tensor:
        """Sample ``batch_size`` i.i.d. β draws.

        Args:
            batch_size: number of β samples to draw.
            seed:       MasterSeed pair (m0, m1).

        Returns:
            Tensor of shape (batch_size, 5), columns ordered
            (α, ρ, σ_ε, φ_quad, φ_prop). Frozen dims are exactly zero.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")

        # Independent stateless sub-seed per coordinate, namespaced by name.
        def _coord(idx: int, name: str) -> tf.Tensor:
            sub_seed = tf.constant(
                fold_in_seed(seed, "beta", name), dtype=tf.int32)
            lo, hi = self.uniform_bounds[idx]
            return tf.random.stateless_uniform(
                [batch_size], seed=sub_seed,
                minval=tf.constant(lo, tf.float32),
                maxval=tf.constant(hi, tf.float32),
                dtype=tf.float32,
            )

        cols = [_coord(i, name) for i, name in enumerate(BETA_DIM_NAMES)]
        beta = tf.stack(cols, axis=-1)

        # Zero out frozen dims.
        if self.freeze_dims:
            mask = tf.constant(
                [0.0 if d in self.freeze_dims else 1.0 for d in range(BETA_DIM)],
                dtype=tf.float32,
            )
            beta = beta * mask
        return beta

    # ------------------------------------------------------------------
    # Anchor / mean point
    # ------------------------------------------------------------------

    def prior_mean(self) -> tf.Tensor:
        """Return the midpoint of each uniform box as a (1, 5) anchor.

        Named ``prior_mean`` for parity with notebook usage; semantically
        it's the box midpoint, not the Bayesian prior mean. Frozen dims
        are zero.
        """
        mean = [0.5 * (lo + hi) for (lo, hi) in self.uniform_bounds]
        for d in self.freeze_dims:
            mean[d] = 0.0
        return tf.constant([mean], dtype=tf.float32)

    @staticmethod
    def dim_names() -> Tuple[str, ...]:
        """Ordered names of the β coordinates."""
        return BETA_DIM_NAMES
