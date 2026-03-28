"""
src/trainers/config.py

Hierarchical configuration system for DNN experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Literal, Tuple, Union
from src.economy.parameters import EconomicParams, ShockParams


# =============================================================================
# MODULE-LEVEL DEFAULTS (Re-exported from _defaults.py)
# =============================================================================
# All defaults are defined in _defaults.py to avoid circular imports.
# This module re-exports them for backward compatibility.

from src._defaults import (
    # Training loop defaults
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_ITER,
    DEFAULT_LOG_EVERY,
    DEFAULT_N_CRITIC,
    DEFAULT_POLYAK_TAU,
    DEFAULT_WEIGHT_BR,
    # Annealing & numerical defaults
    DEFAULT_TEMPERATURE_INIT,
    DEFAULT_TEMPERATURE_MIN,
    DEFAULT_ANNEAL_DECAY,
    DEFAULT_ANNEAL_BUFFER,
    DEFAULT_LOGIT_CLIP,
    DEFAULT_INDICATOR_THRESHOLD,
    DEFAULT_INDICATOR_LOGIT_CLIP,
    DEFAULT_SAFE_EPSILON,
    # Early stopping / convergence defaults
    DEFAULT_PATIENCE,
    DEFAULT_EVAL_FREQ,
    DEFAULT_LR_EPSILON,
    DEFAULT_LR_WINDOW,
    DEFAULT_MA_WINDOW,
    DEFAULT_ER_EPSILON,
    DEFAULT_BR_CRITIC_EPSILON,
    DEFAULT_BR_ACTOR_EPSILON,
    DEFAULT_DIVISION_EPSILON,
)


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer settings including gradient clipping.

    Gradient clipping helps prevent exploding gradients during training.
    Use clipnorm for global norm clipping (recommended) or clipvalue for
    element-wise clipping.

    Reference:
        Keras Adam optimizer: https://keras.io/api/optimizers/adam/
    """
    optimizer_type: str = "adam"  # Currently only "adam" supported
    clipnorm: Optional[float] = None  # Global norm clipping (e.g., 1.0)
    clipvalue: Optional[float] = None  # Element-wise clipping (e.g., 0.5)


@dataclass
class ObservationNormalizationConfig:
    """
    Observation normalization config applied at model input boundaries.

    Supported schemes:
    - "none": identity
    - "minmax": (x - min) / (max - min)
    - "zscore": (x - mean) / std
    - "log": log transform only (for z: no double-log if already log-space)
    - "log_zscore": log transform then z-score
    """
    # Support-based defaults:
    # - k > 0: log + zscore
    # - z (provided in levels): log + zscore (via z_input_space='level')
    # - b >= 0: zscore
    k_scheme: Literal["none", "minmax", "zscore", "log", "log_zscore"] = "log_zscore"
    z_scheme: Literal["none", "minmax", "zscore", "log", "log_zscore"] = "zscore"
    b_scheme: Literal["none", "minmax", "zscore", "log", "log_zscore"] = "zscore"
    z_input_space: Literal["level", "log"] = "level"
    epsilon: float = DEFAULT_SAFE_EPSILON

    def __post_init__(self):
        valid_schemes = {"none", "minmax", "zscore", "log", "log_zscore"}
        if self.k_scheme not in valid_schemes:
            raise ValueError(
                f"k_scheme must be one of {valid_schemes}, got '{self.k_scheme}'"
            )
        if self.z_scheme not in valid_schemes:
            raise ValueError(
                f"z_scheme must be one of {valid_schemes}, got '{self.z_scheme}'"
            )
        if self.b_scheme not in valid_schemes:
            raise ValueError(
                f"b_scheme must be one of {valid_schemes}, got '{self.b_scheme}'"
            )
        if self.z_input_space not in {"level", "log"}:
            raise ValueError(
                f"z_input_space must be 'level' or 'log', got '{self.z_input_space}'"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")


@dataclass
class OutputClipConfig:
    """
    Inference-only safety clipping for one output channel.

    clip_min / clip_max may be:
    - float (absolute level clip)
    - "k_max" / "b_max" (resolved from model bounds at runtime)
    - None (disabled)
    """
    clip_min: Optional[Union[float, str]] = None
    clip_max: Optional[Union[float, str]] = None

    def __post_init__(self):
        valid_tokens = {"k_max", "b_max"}
        for name, value in (("clip_min", self.clip_min), ("clip_max", self.clip_max)):
            if value is None:
                continue
            if isinstance(value, str):
                if value not in valid_tokens:
                    raise ValueError(
                        f"{name} token must be one of {valid_tokens}, got '{value}'"
                    )
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be float, token string, or None; got "
                    f"{type(value).__name__}"
                )


@dataclass
class InferenceClipConfig:
    """
    Inference-only clipping configuration per output variable.
    """
    basic_policy_k: OutputClipConfig = field(
        default_factory=lambda: OutputClipConfig(clip_max="k_max")
    )
    basic_value: OutputClipConfig = field(default_factory=OutputClipConfig)
    risky_policy_k: OutputClipConfig = field(
        default_factory=lambda: OutputClipConfig(clip_max="k_max")
    )
    risky_policy_b: OutputClipConfig = field(
        default_factory=lambda: OutputClipConfig(clip_min=0.0)
    )
    risky_value: OutputClipConfig = field(default_factory=OutputClipConfig)
    risky_price_q: OutputClipConfig = field(default_factory=OutputClipConfig)

    def __post_init__(self):
        for field_name in (
            "basic_policy_k",
            "basic_value",
            "risky_policy_k",
            "risky_policy_b",
            "risky_value",
            "risky_price_q",
        ):
            value = getattr(self, field_name)
            if isinstance(value, dict):
                setattr(self, field_name, OutputClipConfig(**value))
            elif not isinstance(value, OutputClipConfig):
                raise TypeError(
                    f"{field_name} must be OutputClipConfig or dict, got "
                    f"{type(value).__name__}"
                )


@dataclass
class NetworkConfig:
    """
    Configuration for neural network architectures.

    Defaults follow report_brief.md lines 341-357 (Common Configs).
    """
    n_layers: int = 2
    n_neurons: int = 16
    hidden_activation: str = "relu"

    # Output heads
    basic_policy_head: Literal["bounded_sigmoid", "linear", "affine_exp"] = "affine_exp"
    basic_value_head: Literal["linear"] = "linear"
    risky_policy_k_head: Literal["bounded_sigmoid", "linear", "affine_exp"] = "affine_exp"
    risky_policy_b_head: Literal["bounded_sigmoid", "linear"] = "bounded_sigmoid"
    risky_value_head: Literal["linear"] = "linear"
    risky_price_head: Literal["bounded_sigmoid", "linear"] = "bounded_sigmoid"

    observation_normalization: ObservationNormalizationConfig = field(
        default_factory=ObservationNormalizationConfig
    )
    inference_clips: InferenceClipConfig = field(
        default_factory=InferenceClipConfig
    )

    def __post_init__(self):
        if isinstance(self.observation_normalization, dict):
            self.observation_normalization = ObservationNormalizationConfig(
                **self.observation_normalization
            )
        elif not isinstance(
            self.observation_normalization, ObservationNormalizationConfig
        ):
            raise TypeError(
                "observation_normalization must be ObservationNormalizationConfig "
                f"or dict, got {type(self.observation_normalization).__name__}"
            )
        if isinstance(self.inference_clips, dict):
            self.inference_clips = InferenceClipConfig(**self.inference_clips)
        elif not isinstance(self.inference_clips, InferenceClipConfig):
            raise TypeError(
                "inference_clips must be InferenceClipConfig or dict, "
                f"got {type(self.inference_clips).__name__}"
            )

        valid_heads_by_field = {
            "basic_policy_head": {"bounded_sigmoid", "linear", "affine_exp"},
            "basic_value_head": {"linear"},
            "risky_policy_k_head": {"bounded_sigmoid", "linear", "affine_exp"},
            "risky_policy_b_head": {"bounded_sigmoid", "linear"},
            "risky_value_head": {"linear"},
            "risky_price_head": {"bounded_sigmoid", "linear"},
        }
        for field_name, valid_heads in valid_heads_by_field.items():
            value = getattr(self, field_name)
            if value not in valid_heads:
                raise ValueError(
                    f"{field_name} must be one of {valid_heads}, got '{value}'"
                )


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping / convergence criteria.

    Tolerance Interpretation for Unit-Free Residuals:
        For ER method with unit-free residual f = 1 - β*m/χ:
        - L_ER = E[f²] where f is the fractional Euler deviation
        - er_epsilon = 1e-4 corresponds to ~1% average Euler accuracy
        - er_epsilon = 1e-5 corresponds to ~0.3% average Euler accuracy
        - er_epsilon = 1e-3 corresponds to ~3% average Euler accuracy

    References:
        report_brief.md lines 599-604: Unit-free Euler residual
        report_brief.md lines 723-784: Convergence and Stopping Criteria
    """
    enabled: bool = False  # If False, run exactly n_iter steps (Debug mode)
    patience: int = DEFAULT_PATIENCE  # Consecutive validation checks before stopping
    eval_freq: int = DEFAULT_EVAL_FREQ  # Evaluate validation metrics every N steps

    # LR Method: Relative Improvement Plateau
    lr_epsilon: float = DEFAULT_LR_EPSILON  # Relative improvement threshold
    lr_window: int = DEFAULT_LR_WINDOW  # Window size for improvement evaluation
    ma_window: int = DEFAULT_MA_WINDOW  # Moving average window size for smoothing

    # ER Method: Zero-Tolerance Plateau (unit-free residuals)
    # For f = 1 - β*m/χ: er_epsilon = 1e-4 → ~1% Euler accuracy
    er_epsilon: float = DEFAULT_ER_EPSILON  # Absolute loss threshold (relaxed for unit-free)

    # BR Method: Dual-Condition Convergence
    br_critic_epsilon: float = DEFAULT_BR_CRITIC_EPSILON  # Critic loss threshold
    br_actor_epsilon: float = DEFAULT_BR_ACTOR_EPSILON  # Actor relative improvement threshold


@dataclass
class OptimizationConfig:
    """
    Configuration for training loop, optimizer, and runtime reproducibility.

    Attributes:
        training_seed: Optional runtime seed for model initialization and
            training-time RNG. Accepts either:
            - int: mapped to seed pair (int, 0)
            - Tuple[int, int]: explicit stateless seed pair
            If None, train() falls back to dataset metadata seeds when available.
        deterministic_ops: If True, train() will request deterministic TensorFlow
            kernels when supported by the runtime.
    """
    learning_rate: float = DEFAULT_LEARNING_RATE
    learning_rate_critic: Optional[float] = None  # If None, use learning_rate
    batch_size: int = DEFAULT_BATCH_SIZE
    n_iter: int = DEFAULT_N_ITER
    log_every: int = DEFAULT_LOG_EVERY
    early_stopping: Optional[EarlyStoppingConfig] = None  # None = disabled
    training_seed: Optional[Union[int, Tuple[int, int]]] = None
    deterministic_ops: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)  # Optimizer settings including gradient clipping


@dataclass
class AnnealingConfig:
    """Configuration for temperature annealing (soft gates)."""
    temperature_init: float = DEFAULT_TEMPERATURE_INIT
    temperature_min: float = DEFAULT_TEMPERATURE_MIN
    decay: float = DEFAULT_ANNEAL_DECAY  # Per-step multiplicative decay
    schedule: str = "exponential"  # "exponential" or "linear"
    logit_clip: float = DEFAULT_LOGIT_CLIP


@dataclass
class RiskyDebtConfig:
    """Configuration specific to Risky Debt models."""
    # BR method: Bellman residual weight (price weight is implicitly 1.0)
    # L_critic = weight_br * L_BR + L_price
    # Default 0.1 because BR loss is typically 100x larger than price loss
    weight_br: float = DEFAULT_WEIGHT_BR

    # Loss computation method for critic:
    # - "mse": Mean Squared Error, L = E[f²], biased but stable (default)
    # - "huber": Huber TD loss, robust to outlier residuals
    # - "crossprod": AiO cross-product, L = E[f₁·f₂], unbiased
    loss_type: str = "mse"

    # LR method: adaptive Lagrange multiplier parameters
    lambda_price_init: float = 1.0
    learning_rate_lambda: float = 0.01
    epsilon_price: float = 0.01
    polyak_weight: float = 0.1
    n_value_update_freq: int = 1
    learning_rate_value: Optional[float] = None

    # NOTE: Default probability smoothing (epsilon_D) is now handled by AnnealingConfig
    # Use AnnealingConfig in train_risky_br() for temperature annealing

    def __post_init__(self):
        valid_loss_types = {"mse", "huber", "crossprod"}
        if self.loss_type not in valid_loss_types:
            raise ValueError(
                f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'"
            )


@dataclass
class MethodConfig:
    """
    Configuration for the solution method (Algorithm).

    Attributes:
        name: Method identifier (e.g., "basic_lr", "basic_er",
            "basic_br_actor_critic", "risky_br_actor_critic")
        n_critic: Number of critic updates per actor update (BR methods only)
        polyak_tau: Polyak averaging coefficient for target networks (ER/BR methods)
        loss_type: Loss computation method for ER/BR critic updates:
            - "mse": Mean Squared Error E[f²], biased but stable
            - "huber": Huber loss (BR-family only), robust to outliers
            - "crossprod": AiO cross-product E[f₁·f₂], unbiased (default)
            - "fork_mean_square": BR-multitask only, uses
              E[(mean_k f_k)^2] with O(1/K) positive bias vs. E[f]^2
            For risky debt, configure via RiskyDebtConfig.loss_type instead.
        br_normalization: BR normalization mode for BR-family losses:
            - "frictionless": Normalize by frictionless steady-state value scale (default)
            - "none": Disable BR normalization
            - "custom": Use br_normalizer_value
        br_normalizer_value: Custom normalizer value when br_normalization="custom"
        br_normalizer_epsilon: Lower bound for absolute normalizer value
        br_reg_weight_br: Weight on BR term for basic BR regression method
        br_reg_weight_foc: Weight on FOC regularization term for basic BR regression method
        br_reg_weight_env: Weight on envelope regularization term for basic BR regression method
        br_reg_use_foc: Whether to include FOC term in basic BR regression method
        br_reg_use_env: Whether to include envelope term in basic BR regression method
        br_reg_weighting_mode: Regularizer-weight mode for BR multitask:
            - "fixed": use static BR/FOC/env weights (default)
            - "adaptive": dynamically rebalance active BR/FOC/env weights
              from gradient norms while preserving a total weight budget
        br_reg_adaptive_alpha: Exponent for adaptive weight updates (higher => more aggressive)
        br_reg_adaptive_ema_decay: EMA decay for gradient-norm smoothing in adaptive mode
        br_reg_adaptive_update_every: Update adaptive weights every N train steps
        br_reg_adaptive_warmup_steps: Keep fixed weights for initial steps before adaptation
        br_reg_adaptive_max_step_change: Max relative per-update weight change (e.g., 0.1 = ±10%)
        br_reg_adaptive_grad_floor: Positive floor to avoid division by tiny gradient norms
        br_reg_adaptive_total_weight_budget: Optional total budget used to
            renormalize active adaptive weights (e.g., 2 for BR+FOC).
            If None, uses the sum of initial active weights.
        br_reg_weight_br_min: Lower bound for adaptive BR weight
        br_reg_weight_br_max: Upper bound for adaptive BR weight
        br_reg_weight_foc_min: Lower bound for adaptive FOC weight
        br_reg_weight_foc_max: Upper bound for adaptive FOC weight
        br_reg_weight_env_min: Lower bound for adaptive envelope weight
        br_reg_weight_env_max: Upper bound for adaptive envelope weight
        br_reg_env_derivative_mode: Envelope derivative mode in BR multitask:
            - "total": total derivative d e(k, k'(k,z), z) / d k (current behavior)
            - "partial_stopgrad_policy": partial derivative wrt k holding k' fixed
            - "hardcoded_frictionless": closed-form partial for frictionless baseline only
        br_reg_num_forks: Number of independent z' forks used in BR/FOC losses.
            Use 2 for legacy behavior; >2 uses pairwise unbiased cross-product.
        risky: Optional configuration for risky debt model
    """
    name: str  # e.g., "basic_lr", "basic_er", "risky_br", etc.
    n_critic: int = DEFAULT_N_CRITIC  # Critic updates per actor update (BR methods)
    polyak_tau: float = DEFAULT_POLYAK_TAU  # Polyak averaging coefficient for target networks (ER/BR methods)
    loss_type: str = "mse"  # "mse", "huber", "crossprod", or "fork_mean_square" (BR-multitask only)
    br_normalization: str = "frictionless"  # "frictionless" (default), "none", "custom"
    br_normalizer_value: Optional[float] = None
    br_normalizer_epsilon: float = 1e-8
    br_reg_weight_br: float = 1.0
    br_reg_weight_foc: float = 1.0
    br_reg_weight_env: float = 1.0
    br_reg_use_foc: bool = True
    br_reg_use_env: bool = True
    br_reg_weighting_mode: str = "fixed"  # "fixed" or "adaptive"
    br_reg_adaptive_alpha: float = 1.0
    br_reg_adaptive_ema_decay: float = 0.9
    br_reg_adaptive_update_every: int = 1
    br_reg_adaptive_warmup_steps: int = 0
    br_reg_adaptive_max_step_change: float = 0.1
    br_reg_adaptive_grad_floor: float = 1e-8
    br_reg_adaptive_total_weight_budget: Optional[float] = None
    br_reg_weight_br_min: float = 0.05
    br_reg_weight_br_max: float = 20.0
    br_reg_weight_foc_min: float = 0.05
    br_reg_weight_foc_max: float = 20.0
    br_reg_weight_env_min: float = 0.05
    br_reg_weight_env_max: float = 20.0
    br_reg_env_derivative_mode: str = "total"
    br_reg_num_forks: int = 2
    risky: Optional[RiskyDebtConfig] = None

    def __post_init__(self):
        self.br_normalization = self.br_normalization.strip().lower().replace("-", "_")
        self.br_reg_weighting_mode = self.br_reg_weighting_mode.strip().lower().replace("-", "_")
        self.br_reg_env_derivative_mode = (
            self.br_reg_env_derivative_mode.strip().lower().replace("-", "_")
        )
        valid_loss_types = {"mse", "huber", "crossprod", "fork_mean_square"}
        if self.loss_type not in valid_loss_types:
            raise ValueError(
                f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'"
            )
        if self.loss_type == "fork_mean_square":
            if self.name.strip().lower().replace("-", "_") != "basic_br_multitask":
                raise ValueError(
                    "loss_type='fork_mean_square' is supported only for "
                    "MethodConfig(name='basic_br_multitask')."
                )

        if self.br_reg_weight_br < 0:
            raise ValueError(f"br_reg_weight_br must be >= 0, got {self.br_reg_weight_br}")
        if self.br_reg_weight_foc < 0:
            raise ValueError(f"br_reg_weight_foc must be >= 0, got {self.br_reg_weight_foc}")
        if self.br_reg_weight_env < 0:
            raise ValueError(f"br_reg_weight_env must be >= 0, got {self.br_reg_weight_env}")
        if self.br_reg_weight_br_min < 0:
            raise ValueError(f"br_reg_weight_br_min must be >= 0, got {self.br_reg_weight_br_min}")
        if self.br_reg_weight_foc_min < 0:
            raise ValueError(f"br_reg_weight_foc_min must be >= 0, got {self.br_reg_weight_foc_min}")
        if self.br_reg_weight_env_min < 0:
            raise ValueError(f"br_reg_weight_env_min must be >= 0, got {self.br_reg_weight_env_min}")
        if self.br_reg_weight_br_max < self.br_reg_weight_br_min:
            raise ValueError(
                "br_reg_weight_br_max must be >= br_reg_weight_br_min"
            )
        if self.br_reg_weight_foc_max < self.br_reg_weight_foc_min:
            raise ValueError(
                "br_reg_weight_foc_max must be >= br_reg_weight_foc_min"
            )
        if self.br_reg_weight_env_max < self.br_reg_weight_env_min:
            raise ValueError(
                "br_reg_weight_env_max must be >= br_reg_weight_env_min"
            )
        valid_norm_modes = {"none", "frictionless", "custom"}
        if self.br_normalization not in valid_norm_modes:
            raise ValueError(
                f"br_normalization must be one of {valid_norm_modes}, got '{self.br_normalization}'"
            )
        if self.br_normalization == "custom" and self.br_normalizer_value is None:
            raise ValueError(
                "br_normalization='custom' requires br_normalizer_value."
            )
        if self.br_normalizer_epsilon <= 0:
            raise ValueError(
                f"br_normalizer_epsilon must be > 0, got {self.br_normalizer_epsilon}"
            )
        valid_weight_modes = {"fixed", "adaptive"}
        if self.br_reg_weighting_mode not in valid_weight_modes:
            raise ValueError(
                f"br_reg_weighting_mode must be one of {valid_weight_modes}, got "
                f"'{self.br_reg_weighting_mode}'"
            )
        if self.br_reg_adaptive_alpha <= 0:
            raise ValueError(
                f"br_reg_adaptive_alpha must be > 0, got {self.br_reg_adaptive_alpha}"
            )
        if not (0 <= self.br_reg_adaptive_ema_decay < 1):
            raise ValueError(
                "br_reg_adaptive_ema_decay must be in [0, 1), got "
                f"{self.br_reg_adaptive_ema_decay}"
            )
        if self.br_reg_adaptive_update_every <= 0:
            raise ValueError(
                "br_reg_adaptive_update_every must be >= 1, got "
                f"{self.br_reg_adaptive_update_every}"
            )
        if self.br_reg_adaptive_warmup_steps < 0:
            raise ValueError(
                "br_reg_adaptive_warmup_steps must be >= 0, got "
                f"{self.br_reg_adaptive_warmup_steps}"
            )
        if not (0 < self.br_reg_adaptive_max_step_change <= 1):
            raise ValueError(
                "br_reg_adaptive_max_step_change must be in (0, 1], got "
                f"{self.br_reg_adaptive_max_step_change}"
            )
        if self.br_reg_adaptive_grad_floor <= 0:
            raise ValueError(
                f"br_reg_adaptive_grad_floor must be > 0, got "
                f"{self.br_reg_adaptive_grad_floor}"
            )
        if self.br_reg_adaptive_total_weight_budget is not None and self.br_reg_adaptive_total_weight_budget <= 0:
            raise ValueError(
                "br_reg_adaptive_total_weight_budget must be > 0 when provided, "
                f"got {self.br_reg_adaptive_total_weight_budget}"
            )
        if self.br_reg_weighting_mode == "adaptive":
            if not (self.br_reg_weight_br_min <= self.br_reg_weight_br <= self.br_reg_weight_br_max):
                raise ValueError(
                    "br_reg_weight_br must lie within "
                    "[br_reg_weight_br_min, br_reg_weight_br_max] in adaptive mode."
                )
            if self.br_reg_use_foc and not (
                self.br_reg_weight_foc_min <= self.br_reg_weight_foc <= self.br_reg_weight_foc_max
            ):
                raise ValueError(
                    "br_reg_weight_foc must lie within "
                    "[br_reg_weight_foc_min, br_reg_weight_foc_max] in adaptive mode."
                )
            if self.br_reg_use_env and not (
                self.br_reg_weight_env_min <= self.br_reg_weight_env <= self.br_reg_weight_env_max
            ):
                raise ValueError(
                    "br_reg_weight_env must lie within "
                    "[br_reg_weight_env_min, br_reg_weight_env_max] in adaptive mode."
                )
        valid_env_modes = {
            "total",
            "partial_stopgrad_policy",
            "hardcoded_frictionless",
        }
        if self.br_reg_env_derivative_mode not in valid_env_modes:
            raise ValueError(
                "br_reg_env_derivative_mode must be one of "
                f"{valid_env_modes}, got '{self.br_reg_env_derivative_mode}'"
            )
        if self.br_reg_num_forks < 2:
            raise ValueError(
                f"br_reg_num_forks must be >= 2, got {self.br_reg_num_forks}"
            )

# =============================================================================
# EXPERIMENT CONFIG (Master Configuration)
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Master configuration for a training experiment.

    Groups all sub-configs and provides unified summary display.
    This ensures all parameters (including defaults) are transparent.

    Example:
        >>> config = ExperimentConfig(
        ...     name="Basic Model Demo",
        ...     params=EconomicParams.with_overrides(cost_convex=0.01),
        ...     shock_params=ShockParams(rho=0.7, sigma=0.15),
        ...     network=NetworkConfig(n_layers=2, n_neurons=16),
        ...     optimization=OptimizationConfig(n_iter=500),
        ...     annealing=AnnealingConfig(decay=0.99),
        ...     method=MethodConfig(name="basic_lr")
        ... )
        >>> config.print_summary()
    """
    # Required configs
    params: EconomicParams
    shock_params: ShockParams
    network: NetworkConfig
    optimization: OptimizationConfig
    annealing: AnnealingConfig
    method: MethodConfig

    # Optional metadata
    name: str = "experiment"
    description: str = ""

    def summary(self, style: str = "grouped") -> "pd.DataFrame":
        """
        Return a summary DataFrame of all configuration parameters.

        Args:
            style: "grouped" for hierarchical view, "flat" for single table

        Returns:
            pandas DataFrame with all parameters
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []

        # Economic Parameters
        for k, v in asdict(self.params).items():
            rows.append({"Group": "Economic", "Parameter": k, "Value": v})

        # Shock Parameters
        for k, v in asdict(self.shock_params).items():
            rows.append({"Group": "Shock", "Parameter": k, "Value": v})

        # Network
        for k, v in asdict(self.network).items():
            rows.append({"Group": "Network", "Parameter": k, "Value": v})

        # Optimization (flatten nested configs)
        opt_dict = asdict(self.optimization)
        for k, v in opt_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    rows.append({"Group": "Optimization", "Parameter": f"{k}.{k2}", "Value": v2})
            else:
                rows.append({"Group": "Optimization", "Parameter": k, "Value": v})

        # Annealing
        for k, v in asdict(self.annealing).items():
            rows.append({"Group": "Annealing", "Parameter": k, "Value": v})

        # Method (skip None values and nested dicts for cleaner display)
        method_dict = asdict(self.method)
        for k, v in method_dict.items():
            if v is not None and not isinstance(v, dict):
                rows.append({"Group": "Method", "Parameter": k, "Value": v})
            elif isinstance(v, dict):
                # Flatten risky config if present
                for k2, v2 in v.items():
                    rows.append({"Group": "Method", "Parameter": f"risky.{k2}", "Value": v2})

        df = pd.DataFrame(rows)

        if style == "grouped":
            return df.set_index(["Group", "Parameter"])
        return df

    def print_summary(self, width: int = 60) -> None:
        """Print a formatted summary to console."""
        print(f"\n{'='*width}")
        print(f"EXPERIMENT: {self.name}")
        if self.description:
            print(f"{self.description}")
        print(f"{'='*width}")
        print(self.summary().to_string())
        print(f"{'='*width}\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to a serializable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "params": asdict(self.params),
            "shock_params": asdict(self.shock_params),
            "network": asdict(self.network),
            "optimization": asdict(self.optimization),
            "annealing": asdict(self.annealing),
            "method": asdict(self.method)
        }


# =============================================================================
# DATA CONFIG (Data Generation Configuration)
# =============================================================================

@dataclass
class DataConfig:
    """
    Configuration for data generation with economically-anchored bounds.

    This config supports TWO modes for specifying bounds:

    1. MODEL-BASED AUTO-COMPUTATION (default, recommended for economic models):
       - Specify bounds as multipliers on steady-state k* (e.g., k_min=0.2, k_max=3.0)
       - Bounds are converted to LEVELS internally
       - Networks receive level values directly; no post-hoc conversion needed

    2. DIRECT SPECIFICATION (for custom data or arbitrary units):
       - Set auto_compute_bounds=False
       - Pass bounds directly via custom_bounds in whatever units your data uses
       - Framework is completely unit-agnostic

    IMPORTANT: All bounds are ultimately expressed in LEVELS (actual values).
    The multiplier notation is just a convenient way to specify bounds relative
    to economic steady-state. Observation normalization is configured separately.

    Bounds Constraints (for auto-compute mode):
        - std_dev_multiplier (m): Must be in (2, 5)
        - k_min_multiplier: Must be in (0, 0.5)
        - k_max_multiplier: Must be in (1.5, 5)

    Usage Examples:
        1. Model-based (bounds computed from economic parameters):
           >>> data_config = DataConfig(
           ...     master_seed=(42, 0),
           ...     std_dev_multiplier=3.0,  # m for log_z bounds
           ...     k_min_multiplier=0.2,    # 20% of steady-state k*
           ...     k_max_multiplier=3.0     # 300% of steady-state k*
           ... )
           # Result: k_bounds might be (15.4, 231.7) in levels for k*=77.24

        2. Custom bounds (for arbitrary data units):
           >>> data_config = DataConfig(
           ...     master_seed=(42, 0),
           ...     auto_compute_bounds=False,
           ...     custom_bounds={'k': (5000, 10000), 'log_z': (-0.5, 0.5), 'b': (0, 20000)}
           ... )
           # Result: Bounds used exactly as specified
    """
    # === Required ===
    master_seed: tuple  # (m0, m1) for RNG reproducibility

    # === Simulation Dimensions ===
    T: int = 64                    # Rollout horizon
    sim_batch_size: int = 128      # Trajectories per batch
    n_sim_batches: int = 256       # Number of batches
    N_val: Optional[int] = None    # Validation size (default: 10 * sim_batch_size)
    N_test: Optional[int] = None   # Test size (default: 50 * sim_batch_size)

    # === Bounds Configuration (Auto-Compute Mode) ===
    auto_compute_bounds: bool = True
    std_dev_multiplier: float = 3.0   # m for log_z: μ ± m*σ_ergodic (must be in 2-5)
    k_min_multiplier: float = 0.2     # k_min as fraction of k* (must be in 0-0.5)
    k_max_multiplier: float = 3.0     # k_max as multiple of k* (must be in 1.5-5)
    k_star_override: Optional[float] = None  # Override k* if provided

    # === Bounds Configuration (Custom Mode) ===
    # Use when auto_compute_bounds=False
    custom_bounds: Optional[Dict[str, tuple]] = None  # {'k': (min, max), 'log_z': (...), 'b': (...)}

    # === Cache Settings ===
    cache_dir: Optional[str] = None  # None = PROJECT_ROOT/data
    save_to_disk: bool = True

    def __post_init__(self):
        """Validate bounds constraints."""
        if self.auto_compute_bounds:
            if not (2.0 < self.std_dev_multiplier < 5.0):
                raise ValueError(
                    f"std_dev_multiplier (m) must be in (2, 5), got {self.std_dev_multiplier}"
                )
            if not (0.0 < self.k_min_multiplier < 0.5):
                raise ValueError(
                    f"k_min_multiplier must be in (0, 0.5), got {self.k_min_multiplier}"
                )
            if not (1.5 < self.k_max_multiplier < 5.0):
                raise ValueError(
                    f"k_max_multiplier must be in (1.5, 5), got {self.k_max_multiplier}"
                )

    def summary(self) -> "pd.DataFrame":
        """Return a summary DataFrame of all data configuration parameters."""
        import pandas as pd

        rows: List[Dict[str, Any]] = []

        # Simulation dimensions
        rows.append({"Group": "Simulation", "Parameter": "master_seed", "Value": self.master_seed})
        rows.append({"Group": "Simulation", "Parameter": "T (horizon)", "Value": self.T})
        rows.append({"Group": "Simulation", "Parameter": "sim_batch_size", "Value": self.sim_batch_size})
        rows.append({"Group": "Simulation", "Parameter": "n_sim_batches", "Value": self.n_sim_batches})
        rows.append({"Group": "Simulation", "Parameter": "total_samples", "Value": self.sim_batch_size * self.n_sim_batches})
        rows.append({"Group": "Simulation", "Parameter": "N_val", "Value": self.N_val or f"{10 * self.sim_batch_size} (default)"})
        rows.append({"Group": "Simulation", "Parameter": "N_test", "Value": self.N_test or f"{50 * self.sim_batch_size} (default)"})

        # Bounds configuration
        rows.append({"Group": "Bounds", "Parameter": "auto_compute_bounds", "Value": self.auto_compute_bounds})
        if self.auto_compute_bounds:
            rows.append({"Group": "Bounds", "Parameter": "std_dev_multiplier (m)", "Value": self.std_dev_multiplier})
            rows.append({"Group": "Bounds", "Parameter": "k_min_multiplier", "Value": self.k_min_multiplier})
            rows.append({"Group": "Bounds", "Parameter": "k_max_multiplier", "Value": self.k_max_multiplier})
            if self.k_star_override:
                rows.append({"Group": "Bounds", "Parameter": "k_star_override", "Value": self.k_star_override})
        else:
            if self.custom_bounds:
                for key, val in self.custom_bounds.items():
                    rows.append({"Group": "Bounds", "Parameter": f"custom_{key}", "Value": val})

        # Cache settings
        rows.append({"Group": "Cache", "Parameter": "cache_dir", "Value": self.cache_dir or "PROJECT_ROOT/data"})
        rows.append({"Group": "Cache", "Parameter": "save_to_disk", "Value": self.save_to_disk})

        df = pd.DataFrame(rows)
        return df.set_index(["Group", "Parameter"])

    def print_summary(self, width: int = 60) -> None:
        """Print a formatted summary to console."""
        print(f"\n{'='*width}")
        print("DATA CONFIGURATION")
        print(f"{'='*width}")
        print(self.summary().to_string())
        print(f"{'='*width}\n")

    def create_generator(
        self,
        params: "EconomicParams",
        shock_params: "ShockParams",
        verbose: bool = True
    ):
        """
        Create a DataGenerator from this configuration.

        This is a convenience method that calls create_data_generator() with
        all fields from this DataConfig, avoiding verbose parameter passing.

        Args:
            params: Economic parameters (for bounds computation)
            shock_params: Shock process parameters
            verbose: Print configuration summary (default: True)

        Returns:
            Tuple of (DataGenerator, ShockParams, bounds_dict)

        Example:
            >>> data_config = DataConfig(master_seed=(42, 0), save_to_disk=False)
            >>> generator, shock_params, bounds = data_config.create_generator(
            ...     params=EconomicParams(),
            ...     shock_params=ShockParams()
            ... )
        """
        from src.economy.data_generator import create_data_generator

        # Determine bounds to pass
        bounds = self.custom_bounds if not self.auto_compute_bounds else None

        return create_data_generator(
            master_seed=self.master_seed,
            T=self.T,
            sim_batch_size=self.sim_batch_size,
            n_sim_batches=self.n_sim_batches,
            N_val=self.N_val,
            N_test=self.N_test,
            theta=params.theta,
            r=params.r_rate,
            delta=params.delta,
            shock_params=shock_params,
            bounds=bounds,
            auto_compute_bounds=self.auto_compute_bounds,
            std_dev_multiplier=self.std_dev_multiplier,
            k_min_multiplier=self.k_min_multiplier,
            k_max_multiplier=self.k_max_multiplier,
            k_star_override=self.k_star_override,
            cache_dir=self.cache_dir,
            save_to_disk=self.save_to_disk,
            verbose=verbose
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_optimizer(
    learning_rate: float,
    optimizer_config: Optional[OptimizerConfig] = None
):
    """
    Create a Keras optimizer with optional gradient clipping.

    Args:
        learning_rate: Learning rate for the optimizer.
        optimizer_config: Optional OptimizerConfig with clipping settings.
            If None, creates Adam with no clipping.

    Returns:
        tf.keras.optimizers.Optimizer: Configured optimizer instance.

    Example:
        >>> opt_cfg = OptimizerConfig(clipnorm=1.0)
        >>> optimizer = create_optimizer(1e-3, opt_cfg)
    """
    import tensorflow as tf

    if optimizer_config is None:
        optimizer_config = OptimizerConfig()

    # Build optimizer kwargs
    kwargs = {"learning_rate": learning_rate}

    if optimizer_config.clipnorm is not None:
        kwargs["clipnorm"] = optimizer_config.clipnorm

    if optimizer_config.clipvalue is not None:
        kwargs["clipvalue"] = optimizer_config.clipvalue

    # Currently only Adam is supported
    if optimizer_config.optimizer_type.lower() == "adam":
        return tf.keras.optimizers.Adam(**kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config.optimizer_type}")
