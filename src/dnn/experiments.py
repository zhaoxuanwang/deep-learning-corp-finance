"""
src/dnn/experiments.py

Experiment utilities for notebook-friendly training of DNN models.
Provides config dataclasses and training wrappers with history logging.

REFACTORED:
- EconomicScenario now references EconomicParams (single source of truth)
- Removed _MockParams adapter — trainers use scenario.params directly
- Added from_overrides() factory for explicit parameter changes
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

from src.economy.parameters import EconomicParams
from src.dnn.sampling import SamplingBounds, sample_box_basic, sample_box_risky, TrainingContext, ReplayBuffer
from src.dnn.networks import (
    BasicPolicyNetwork,
    BasicValueNetwork,
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    build_basic_networks,
    build_risky_networks
)
from src.dnn.trainer_basic import BasicTrainerLR, BasicTrainerER, BasicTrainerBR
from src.dnn.trainer_risky import RiskyDebtTrainerLR, RiskyDebtTrainerBR
from src.dnn.default_smoothing import DefaultSmoothingSchedule


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration for DNN experiments.
    
    Attributes:
        n_layers: Number of hidden layers in networks
        n_neurons: Neurons per hidden layer
        activation: Hidden layer activation function
        learning_rate: Optimizer learning rate
        learning_rate_critic: Learning rate for critic (if different)
        batch_size: Training batch size
        n_iter: Number of outer training iterations
        n_critic: Critic updates per actor update (BR methods)
        T: Rollout horizon (LR methods)
        seed: Random seed for reproducibility
        log_every: Log metrics every N iterations
    """
    n_layers: int = 2
    n_neurons: int = 16
    activation: str = "tanh"
    learning_rate: float = 1e-3
    learning_rate_critic: Optional[float] = None
    batch_size: int = 128
    n_iter: int = 1000
    n_critic: int = 1
    T: int = 32
    seed: int = 42
    log_every: int = 10
    
    # Replay settings for ergodic sampling
    replay_ratio: float = 0.5        # Fraction of batch from replay buffer
    buffer_capacity: int = 10000     # Max states in replay buffer
    warmup_iters: int = 100          # Iterations before mixing (all fresh)
    percentile_q: float = 0.05
    
    # Risky debt specific
    lambda_1: float = 1.0  # Price weight for critic
    lambda_2: float = 1.0  # Price weight for actor
    lambda_price: float = 1.0  # Price weight for LR
    epsilon_D_0: float = 0.1
    epsilon_D_min: float = 1e-4
    decay_d: float = 0.99
    u_max: float = 10.0
    
    # Network constraints
    k_min: float = 1e-4
    leverage_scale: float = 1.5


@dataclass
class EconomicScenario:
    """
    Named scenario for DNN experiments.
    
    REFACTORED: No longer duplicates economic parameters.
    References EconomicParams (single source of truth) + SamplingBounds.
    
    Attributes:
        name: Scenario label for plotting/logging
        params: Economic parameters (EconomicParams instance)
        sampling: Sampling bounds for DNN training (SamplingBounds instance)
    
    Example:
        # Default scenario
        scenario = EconomicScenario()
        
        # Scenario with parameter overrides
        scenario = EconomicScenario.from_overrides("high_cost", cost_convex=1.0)
        
        # Pass to trainer
        trainer = BasicTrainerLR(policy_net, params=scenario.params, ...)
    """
    name: str = "baseline"
    params: EconomicParams = field(default_factory=EconomicParams)
    sampling: SamplingBounds = field(default_factory=SamplingBounds)
    
    @classmethod
    def from_overrides(
        cls,
        name: str,
        sampling: Optional[SamplingBounds] = None,
        **param_overrides
    ) -> "EconomicScenario":
        """
        Create scenario with explicit parameter overrides.
        
        Args:
            name: Scenario name
            sampling: Optional custom SamplingBounds
            **param_overrides: Fields to override in EconomicParams
        
        Returns:
            New EconomicScenario with overrides applied
        
        Example:
            scenario = EconomicScenario.from_overrides(
                "high_cost",
                cost_convex=1.0,
                cost_fixed=0.0
            )
        """
        params = EconomicParams.with_overrides(**param_overrides)
        return cls(
            name=name,
            params=params,
            sampling=sampling or SamplingBounds()
        )


# =============================================================================
# TRAINING WRAPPERS
# =============================================================================

def _set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _sample_basic(scenario: EconomicScenario, n: int, rng: np.random.Generator):
    """Sample (k, z) for Basic model."""
    k, z = sample_box_basic(
        n=n,
        k_bounds=scenario.sampling.k_bounds,
        log_z_bounds=scenario.sampling.log_z_bounds,
        rng=rng
    )
    return tf.constant(k, dtype=tf.float32), tf.constant(z, dtype=tf.float32)


def _sample_risky(scenario: EconomicScenario, n: int, rng: np.random.Generator):
    """Sample (k, b, z) for Risky Debt model."""
    k, b, z = sample_box_risky(
        n=n,
        k_bounds=scenario.sampling.k_bounds,
        b_bounds=scenario.sampling.b_bounds,
        log_z_bounds=scenario.sampling.log_z_bounds,
        rng=rng
    )
    return (
        tf.constant(k, dtype=tf.float32),
        tf.constant(b, dtype=tf.float32),
        tf.constant(z, dtype=tf.float32)
    )


def train_basic_lr(
    scenario: EconomicScenario,
    config: TrainingConfig
) -> Dict[str, List]:
    """
    Train Basic model with Lifetime Reward method.
    
    Args:
        scenario: Economic parameters and sampling bounds
        config: Training configuration
    
    Returns:
        History dict with keys: iteration, loss_LR, mean_reward
    """
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    
    # Build network
    policy_net = BasicPolicyNetwork(
        k_min=config.k_min,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    
    # Build trainer — use scenario.params directly (no adapter needed)
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    
    trainer = BasicTrainerLR(
        policy_net=policy_net,
        params=scenario.params,  # Direct use of EconomicParams
        optimizer=optimizer,
        T=config.T,
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Training context for ergodic sampling
    ctx = TrainingContext(
        scenario=scenario,
        buffer=ReplayBuffer(capacity=config.buffer_capacity, state_dim=2),
        replay_ratio=config.replay_ratio,
        warmup_iters=config.warmup_iters,
        state_keys=("k", "z"),
        rng=rng
    )
    
    # Training loop
    history = {"iteration": [], "loss_LR": [], "mean_reward": []}
    
    for i in range(config.n_iter):
        # Sample with ergodic replay
        states = ctx.sample(config.batch_size)
        k = tf.constant(states["k"], dtype=tf.float32)
        z = tf.constant(states["z"], dtype=tf.float32)
        
        metrics = trainer.train_step(k, z)
        
        # Feed back terminal states for next iteration
        ctx.update(metrics.get("terminal_states"))
        
        if i % config.log_every == 0:
            history["iteration"].append(i)
            history["loss_LR"].append(metrics["loss"])
            history["mean_reward"].append(metrics["mean_reward"])
    
    # Store trained networks
    history["_policy_net"] = policy_net
    history["_config"] = config
    history["_scenario"] = scenario
    
    return history


def train_basic_er(
    scenario: EconomicScenario,
    config: TrainingConfig
) -> Dict[str, List]:
    """
    Train Basic model with Euler Residual method.
    
    Args:
        scenario: Economic parameters and sampling bounds
        config: Training configuration
    
    Returns:
        History dict with keys: iteration, loss_ER
    """
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    
    policy_net = BasicPolicyNetwork(
        k_min=config.k_min,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    
    trainer = BasicTrainerER(
        policy_net=policy_net,
        params=scenario.params,
        optimizer=optimizer,
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Training context for ergodic sampling
    ctx = TrainingContext(
        scenario=scenario,
        buffer=ReplayBuffer(capacity=config.buffer_capacity, state_dim=2),
        replay_ratio=config.replay_ratio,
        warmup_iters=config.warmup_iters,
        state_keys=("k", "z"),
        rng=rng
    )
    
    history = {"iteration": [], "loss_ER": []}
    
    for i in range(config.n_iter):
        # Sample with ergodic replay
        states = ctx.sample(config.batch_size)
        k = tf.constant(states["k"], dtype=tf.float32)
        z = tf.constant(states["z"], dtype=tf.float32)
        
        metrics = trainer.train_step(k, z)
        
        # Feed back terminal states
        ctx.update(metrics.get("terminal_states"))
        
        if i % config.log_every == 0:
            history["iteration"].append(i)
            history["loss_ER"].append(metrics["loss"])
    
    history["_policy_net"] = policy_net
    history["_config"] = config
    history["_scenario"] = scenario
    
    return history


def train_basic_br(
    scenario: EconomicScenario,
    config: TrainingConfig
) -> Dict[str, List]:
    """
    Train Basic model with Bellman Residual (Actor-Critic) method.
    
    Args:
        scenario: Economic parameters and sampling bounds
        config: Training configuration
    
    Returns:
        History dict with keys: iteration, loss_BR_critic, loss_BR_actor,
                                mse_proxy, mae_proxy (diagnostic metrics)
    """
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    
    policy_net = BasicPolicyNetwork(
        k_min=config.k_min,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    value_net = BasicValueNetwork(
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    
    optimizer_policy = tf.keras.optimizers.Adam(config.learning_rate)
    lr_critic = config.learning_rate_critic or config.learning_rate
    optimizer_value = tf.keras.optimizers.Adam(lr_critic)
    
    trainer = BasicTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        params=scenario.params,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        batch_size=config.batch_size,
        n_critic_steps=config.n_critic,
        seed=config.seed
    )
    
    # Training context for ergodic sampling
    ctx = TrainingContext(
        scenario=scenario,
        buffer=ReplayBuffer(capacity=config.buffer_capacity, state_dim=2),
        replay_ratio=config.replay_ratio,
        warmup_iters=config.warmup_iters,
        state_keys=("k", "z"),
        rng=rng
    )
    
    history = {
        "iteration": [], 
        "loss_BR_critic": [], 
        "loss_BR_actor": [],
        "mse_proxy": [],
        "mae_proxy": [],
    }
    
    for i in range(config.n_iter):
        # Sample with ergodic replay
        states = ctx.sample(config.batch_size)
        k = tf.constant(states["k"], dtype=tf.float32)
        z = tf.constant(states["z"], dtype=tf.float32)
        
        metrics = trainer.train_step(k, z)
        
        # Feed back terminal states
        ctx.update(metrics.get("terminal_states"))
        
        if i % config.log_every == 0:
            history["iteration"].append(i)
            history["loss_BR_critic"].append(metrics["loss_critic"])
            history["loss_BR_actor"].append(metrics["loss_actor"])
            # Diagnostics (may not always be present)
            history["mse_proxy"].append(metrics.get("mse_proxy", metrics["loss_critic"]))
            history["mae_proxy"].append(metrics.get("mae_proxy", 0.0))
    
    history["_policy_net"] = policy_net
    history["_value_net"] = value_net
    history["_config"] = config
    history["_scenario"] = scenario
    
    return history


def train_risky_lr(
    scenario: EconomicScenario,
    config: TrainingConfig
) -> Dict[str, List]:
    """
    Train Risky Debt model with Lifetime Reward + Price method.
    
    Args:
        scenario: Economic parameters and sampling bounds
        config: Training configuration
    
    Returns:
        History dict with keys: iteration, loss_LR, loss_price, mean_utility
    """
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    
    policy_net = RiskyPolicyNetwork(
        k_min=config.k_min,
        leverage_scale=config.leverage_scale,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    price_net = RiskyPriceNetwork(
        r_risk_free=scenario.params.r_rate,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    
    smoothing = DefaultSmoothingSchedule(
        epsilon_D_0=config.epsilon_D_0,
        epsilon_D_min=config.epsilon_D_min,
        decay_d=config.decay_d,
        u_max=config.u_max
    )
    
    optimizer_policy = tf.keras.optimizers.Adam(config.learning_rate)
    optimizer_price = tf.keras.optimizers.Adam(config.learning_rate)
    
    trainer = RiskyDebtTrainerLR(
        policy_net=policy_net,
        price_net=price_net,
        params=scenario.params,
        optimizer_policy=optimizer_policy,
        optimizer_price=optimizer_price,
        T=config.T,
        batch_size=config.batch_size,
        lambda_price=config.lambda_price,
        smoothing=smoothing,
        seed=config.seed
    )
    
    history = {"iteration": [], "loss_LR": [], "loss_price": [], "mean_utility": []}
    
    for i in range(config.n_iter):
        k, b, z = _sample_risky(scenario, config.batch_size, rng)
        metrics = trainer.train_step(k, b, z)
        
        if i % config.log_every == 0:
            history["iteration"].append(i)
            history["loss_LR"].append(metrics["loss_lr"])
            history["loss_price"].append(metrics["loss_price"])
            history["mean_utility"].append(metrics["mean_utility"])
    
    history["_policy_net"] = policy_net
    history["_price_net"] = price_net
    history["_config"] = config
    history["_scenario"] = scenario
    
    return history


def train_risky_br(
    scenario: EconomicScenario,
    config: TrainingConfig
) -> Dict[str, List]:
    """
    Train Risky Debt model with Bellman Residual + Price (Actor-Critic).
    
    Args:
        scenario: Economic parameters and sampling bounds
        config: Training configuration
    
    Returns:
        History dict with keys: iteration, loss_critic, loss_actor, loss_price, epsilon_D
    """
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    
    policy_net = RiskyPolicyNetwork(
        k_min=config.k_min,
        leverage_scale=config.leverage_scale,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    value_net = RiskyValueNetwork(
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    price_net = RiskyPriceNetwork(
        r_risk_free=scenario.params.r_rate,
        n_layers=config.n_layers,
        n_neurons=config.n_neurons,
        activation=config.activation
    )
    
    smoothing = DefaultSmoothingSchedule(
        epsilon_D_0=config.epsilon_D_0,
        epsilon_D_min=config.epsilon_D_min,
        decay_d=config.decay_d,
        u_max=config.u_max
    )
    
    optimizer_policy = tf.keras.optimizers.Adam(config.learning_rate)
    lr_critic = config.learning_rate_critic or config.learning_rate
    optimizer_critic = tf.keras.optimizers.Adam(lr_critic)
    
    trainer = RiskyDebtTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
        params=scenario.params,
        optimizer_policy=optimizer_policy,
        optimizer_critic=optimizer_critic,
        batch_size=config.batch_size,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        n_critic_steps=config.n_critic,
        smoothing=smoothing,
        seed=config.seed,
        collect_diagnostics=True  # Always collect for debugging
    )
    
    history = {
        "iteration": [],
        "loss_critic": [],
        "loss_actor": [],
        "loss_price": [],
        "epsilon_D": [],
        # Gradient flow diagnostics
        "grad_norm_policy": [],
        "share_v_tilde_negative": [],
        "mean_v_tilde": [],
        "mean_leverage": [],
    }
    
    for i in range(config.n_iter):
        k, b, z = _sample_risky(scenario, config.batch_size, rng)
        metrics = trainer.train_step(k, b, z, update_smoothing=True)
        
        if i % config.log_every == 0:
            history["iteration"].append(i)
            history["loss_critic"].append(metrics["loss_critic"])
            history["loss_actor"].append(metrics["loss_actor"])
            history["loss_price"].append(metrics["loss_price_critic"])
            history["epsilon_D"].append(metrics["epsilon_D"])
            # Log diagnostics
            history["grad_norm_policy"].append(metrics.get("grad_norm_policy", 0.0))
            history["share_v_tilde_negative"].append(metrics.get("share_v_tilde_negative", 0.0))
            history["mean_v_tilde"].append(metrics.get("mean_v_tilde", 0.0))
            history["mean_leverage"].append(metrics.get("mean_leverage", 0.0))
    
    history["_policy_net"] = policy_net
    history["_value_net"] = value_net
    history["_price_net"] = price_net
    history["_config"] = config
    history["_scenario"] = scenario
    
    return history


# =============================================================================
# CONVENIENCE: UNIFIED TRAINING API
# =============================================================================

def train(
    method: Literal["basic_lr", "basic_er", "basic_br", "risky_lr", "risky_br"],
    scenario: EconomicScenario,
    config: TrainingConfig
) -> Dict[str, List]:
    """
    Unified training API.
    
    Args:
        method: Training method name
        scenario: Economic parameters and sampling bounds
        config: Training configuration
    
    Returns:
        History dict with method-specific metrics
    """
    dispatch = {
        "basic_lr": train_basic_lr,
        "basic_er": train_basic_er,
        "basic_br": train_basic_br,
        "risky_lr": train_risky_lr,
        "risky_br": train_risky_br,
    }
    
    if method not in dispatch:
        raise ValueError(f"Unknown method: {method}. Choose from {list(dispatch.keys())}")
    
    return dispatch[method](scenario, config)
