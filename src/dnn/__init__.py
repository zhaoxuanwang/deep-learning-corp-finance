"""
src/dnn/__init__.py

DNN module for Deep Learning solution of corporate finance models.
"""

# Networks
from src.dnn.networks import (
    BasicPolicyNetwork,
    BasicValueNetwork,
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    apply_limited_liability,
    build_basic_networks,
    build_risky_networks
)

# Losses
from src.dnn.losses import (
    compute_lr_loss,
    compute_lr_loss_risky,
    compute_er_loss_aio,
    compute_br_critic_loss_aio,
    compute_br_actor_loss,
    compute_br_actor_loss_risky,
    compute_price_loss_aio,
    compute_critic_objective,
    compute_actor_objective
)

# Re-export Euler primitives from economy (for backward compatibility)
from src.economy.logic import (
    euler_chi,
    euler_m,
    pricing_residual_zero_profit
)

# Annealing Schedule (single source of truth)
from src.dnn.annealing import (
    AnnealingSchedule,
    smooth_default_prob
)

# Sampling
from src.dnn.sampling import (
    AdaptiveBounds,
    SamplingBounds,
    ReplayBuffer,
    sample_box,
    sample_box_basic,
    sample_box_risky,
    sample_mixture,
    sample_with_oversampling,
    draw_shocks
)

# Trainers - Basic
from src.dnn.trainer_basic import (
    BasicTrainerLR,
    BasicTrainerER,
    BasicTrainerBR
)

# Trainers - Risky Debt
from src.dnn.trainer_risky import (
    RiskyDebtTrainerLR,
    RiskyDebtTrainerBR
)

# Experiments & Notebook Utilities
from src.dnn.experiments import (
    TrainingConfig,
    EconomicScenario,
    train_basic_lr,
    train_basic_er,
    train_basic_br,
    train_risky_lr,
    train_risky_br,
    train
)

from src.dnn.evaluation.common import (
    get_eval_grids,
    compute_moments,
    compare_moments,
    find_steady_state_k,
)
from src.dnn.evaluation.wrappers import (
    evaluate_basic_policy,
    evaluate_basic_value,
    evaluate_risky_policy,
    evaluate_risky_value,
)
from src.dnn.evaluation.simulation import (
    simulate_policy_path,
    evaluate_policy_return,
)
from src.dnn.evaluation.residuals import (
    eval_euler_residual_basic,
)

from src.dnn.plotting import (
    plot_loss_curve,
    plot_loss_comparison,
    plot_run_registry,
    plot_policy_slice,
    plot_scenario_comparison,
    plot_2d_heatmap,
    display_moments_table,
    quick_plot
)

__all__ = [
    # Networks
    "BasicPolicyNetwork",
    "BasicValueNetwork", 
    "RiskyPolicyNetwork",
    "RiskyValueNetwork",
    "RiskyPriceNetwork",
    "apply_limited_liability",
    "build_basic_networks",
    "build_risky_networks",
    # Losses
    "compute_lr_loss",
    "compute_lr_loss_risky",
    "compute_er_loss_aio",
    "euler_chi",
    "euler_m",
    "compute_br_critic_loss_aio",
    "compute_br_actor_loss",
    "compute_br_actor_loss_risky",
    "compute_price_loss_aio",
    "pricing_residual_zero_profit",
    "compute_critic_objective",
    "compute_actor_objective",
    # Annealing Schedule
    "AnnealingSchedule",
    "smooth_default_prob",
    # Sampling
    "AdaptiveBounds",
    "ReplayBuffer",
    "sample_box",
    "sample_box_basic",
    "sample_box_risky",
    "sample_mixture",
    "sample_with_oversampling",
    "draw_shocks",
    # Trainers
    "BasicTrainerLR",
    "BasicTrainerER",
    "BasicTrainerBR",
    "RiskyDebtTrainerLR",
    "RiskyDebtTrainerBR",
    # Experiments
    "TrainingConfig",
    "EconomicScenario",
    "train_basic_lr",
    "train_basic_er",
    "train_basic_br",
    "train_risky_lr",
    "train_risky_br",
    "train",
    # Evaluation
    "get_eval_grids",
    "evaluate_basic_policy",
    "evaluate_basic_value",
    "evaluate_risky_policy",
    "evaluate_risky_value",
    "compute_moments",
    "compare_moments",
    "find_steady_state_k",
    "simulate_policy_path",
    "evaluate_policy_return",
    "eval_euler_residual_basic",
    # Plotting
    "plot_loss_curve",
    "plot_loss_comparison",
    "plot_run_registry",
    "plot_policy_slice",
    "plot_scenario_comparison",
    "plot_2d_heatmap",
    "display_moments_table",
    "quick_plot",
]

