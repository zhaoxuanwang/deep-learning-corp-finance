"""Configuration dataclasses for v2 trainers.

Each method has its own config with method-specific defaults.  The shared
TrainingConfig base provides structural inheritance (field names, types)
but each subclass overrides defaults appropriate to that method.  This
avoids unintended coupling — changing a default for one method cannot
affect another.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    n_layers:  int = 2
    n_neurons: int = 128


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    learning_rate: float           = 1e-3
    clipnorm:      Optional[float] = 100.0   # DreamerV3 default safety net


@dataclass
class TrainingConfig:
    """Core training configuration (base class).

    Subclasses override defaults as needed.  master_seed must match
    DataGeneratorConfig.master_seed so that RNG streams are consistent
    across data generation and training.
    """
    n_steps:     int   = 10000
    batch_size:  int   = 256
    master_seed: tuple = field(default_factory=lambda: (20, 26))

    # Target network
    polyak_rate: float = 0.995

    # Evaluation
    eval_interval: int = 500
    eval_size:     int = 2560       # typically 10 * batch_size

    # Smooth gate temperature for non-differentiable reward components.
    temperature: float = 1e-6

    # Network architecture
    network:          NetworkConfig   = field(default_factory=NetworkConfig)
    policy_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    critic_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class LRConfig(TrainingConfig):
    """Lifetime Reward method configuration.

    Defaults: batch_size=256, policy_lr=1e-3 (inherited from base).
    LR does not use a critic or target network.
    """
    horizon:        int  = 64    # rollout horizon T; must be <= DataGeneratorConfig.horizon
    terminal_value: bool = True  # analytical V_term = r(s̄, ā) / (1-γ); LR-specific


@dataclass
class ERConfig(TrainingConfig):
    """Euler Residual method configuration.

    Defaults: batch_size=256, policy_lr=1e-3 (inherited from base).
    ER uses a target policy network for stable next-action computation.
    """
    loss_type:  str = "crossprod"   # "crossprod" (AiO) or "mse"
    n_mc_draws: int = 2             # informational; AiO uses both z_next_main and z_next_fork


@dataclass
class BRMConfig(TrainingConfig):
    """Bellman Residual Minimization configuration.

    Defaults: batch_size=256, policy_lr=1e-3, critic_lr=1e-3 (inherited).
    L_BR trains V via Bellman residual.  L_FOC trains π via autodiff FOC:
      ∂r/∂a + γ(∂f_endo/∂a)^T ∇V  (two independent shock draws, AiO).
    """
    loss_type:  str   = "crossprod"   # "crossprod" (AiO) or "mse"
    weight_foc: float = 1.0           # FOC loss weight
    br_scale:   float = 1.0           # Bellman residual normalizer (set to |V*|)

    # Warm-start: pre-train critic on analytical V before training
    warm_start_steps: int = 200   # 0 to disable


@dataclass
class SHACConfig(TrainingConfig):
    """Short-Horizon Actor-Critic configuration — DDPG-style variant.

    A variant of Xu et al. (2022) adapted for economic environments.
    Retains SHAC's core structure (h-step actor BPTT with windowed
    continuation), but replaces the critic with a DDPG-style 1-step
    Bellman target using target π̄ + target V̄.  This breaks the
    on-policy critic feedback loop that causes divergence in the
    original algorithm.

    Key differences from vanilla SHAC (Xu et al., 2022):
      - Actor bootstrap: current V (not target V̄)
      - Critic target:   1-step Bellman with target π̄ + target V̄
                          (not TD-λ with on-policy data)
      - Target networks: both π̄ and V̄ (not V̄ only)
      - Cold start only: no warm-start or oracle V dependency

    Reward normalization
    --------------------
    The paper's hyperparameters assume rewards/values of O(1).  When the
    environment has larger reward/value scales (e.g. V ≈ 200-500 for
    BasicInvestment), the critic diverges with paper defaults.  Setting
    normalize_rewards=True (default) auto-computes 1/|V*| via the
    environment so that V ≈ O(1).  Set normalize_rewards=False to
    disable scaling.  For advanced use (e.g. reproducing a specific
    experiment), set reward_scale_override to an explicit float.

    Step counting: 1 window = 1 step = 1 actor gradient update.
    Each mini-batch of B trajectories yields horizon / short_horizon steps.
    Total critic gradient steps per window = n_critic.
    """
    # --- SHAC-specific defaults (override TrainingConfig) ---
    batch_size:    int = 64      # paper default; smaller than LR/ER (256)
    horizon:       int = 192     # total rollout horizon T; γ^192 ≈ 0
    short_horizon: int = 32      # window length h (paper default)
    n_critic:      int = 16      # critic gradient steps per window

    # Reward normalization: scales all rewards by 1/|V*| so V ≈ O(1).
    normalize_rewards:      bool           = True   # auto-compute via env
    reward_scale_override:  Optional[float] = None  # manual override (advanced)

    # SHAC-specific optimizer defaults (higher lr than LR/ER)
    policy_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=2e-3))
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=5e-3))


@dataclass
class SHACVanillaConfig(TrainingConfig):
    """Vanilla SHAC configuration — faithful to Xu et al. (2022).

    Archived for reference.  Uses TD-λ critic targets with on-policy data,
    which diverges in economic environments.  See SHACConfig for the
    production variant.
    """
    batch_size:    int   = 64
    horizon:       int   = 192
    short_horizon: int   = 32
    td_lambda:     float = 0.95
    n_critic:      int   = 16
    n_mb:          int   = 1

    reward_scale: Optional[float] = None
    warm_start_steps: int = 200

    policy_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=2e-3))
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=5e-3))


@dataclass
class MVEConfig(TrainingConfig):
    """MVE-DDPG method configuration."""
    mve_horizon:             int            = 10    # MVE rollout depth
    critic_updates_per_step: int            = 1     # critic updates per actor update
    reward_scale:            Optional[float] = None  # λ; None = auto via env
