"""Configuration dataclasses for v2 trainers."""

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
    """Core training configuration.

    master_seed must match DataGeneratorConfig.master_seed so that the
    RNG streams are consistent across data generation and training.
    """
    n_steps:     int   = 10000
    batch_size:  int   = 256
    master_seed: tuple = field(default_factory=lambda: (20, 26))

    # Normalizer warm-up: number of mini-batches passed through the running
    # z-score normalizer before gradient updates begin. The normalizer is
    # frozen after warm-up and not updated during training.
    warmup_steps: int = 100

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
    """Lifetime Reward method configuration."""
    horizon:        int  = 64    # rollout horizon T; must be <= DataGeneratorConfig.horizon
    terminal_value: bool = True  # V_term = r(s_T, a_T) / (1-γ)


@dataclass
class ERConfig(TrainingConfig):
    """Euler Residual method configuration."""
    loss_type:  str = "crossprod"   # "crossprod" (AiO) or "mse"
    n_mc_draws: int = 2             # informational; AiO uses both z_next_main and z_next_fork


@dataclass
class BRMConfig(TrainingConfig):
    """Bellman Residual Minimization configuration.

    L_BR trains V via Bellman residual.  L_FOC trains π via autodiff FOC:
      ∂r/∂a + γ(∂f_endo/∂a)^T ∇V  (two independent shock draws, AiO).
    """
    loss_type:  str   = "crossprod"   # "crossprod" (AiO) or "mse"
    weight_foc: float = 1.0           # FOC loss weight
    br_scale:   float = 1.0           # Bellman residual normalizer (set to |V*|)


@dataclass
class MVEConfig(TrainingConfig):
    """MVE-DDPG method configuration."""
    mve_horizon:             int            = 10    # MVE rollout depth
    critic_updates_per_step: int            = 1     # critic updates per actor update
    reward_scale:            Optional[float] = None  # λ; None = auto via env
