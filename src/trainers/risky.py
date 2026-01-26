"""
src/trainers/risky.py

Trainers and high-level training entry points for the Risky Debt Model (Sec. 2).
Implements LR+price and BR (actor-critic)+price training methods.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Optional, Any, Iterator, Tuple

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.logic import (
    cash_flow_risky_debt,
    external_financing_cost,
    recovery_value,
    pricing_residual_zero_profit,
)
from src.networks.network_risky import (
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    build_risky_networks,
    apply_limited_liability
)
from src.trainers.losses import (
    compute_lr_loss,
    compute_price_loss_aio,
    compute_br_critic_loss_aio,
    compute_br_actor_loss_risky,
    compute_critic_objective,
    compute_actor_objective
)
from src.utils.annealing import AnnealingSchedule, smooth_default_prob
from src.trainers.config import NetworkConfig, OptimizationConfig, AnnealingConfig, MethodConfig
from src.trainers.core import execute_training_loop

logger = logging.getLogger(__name__)


class RiskyDebtTrainerLR:
    """
    LR Trainer for Risky Debt.
    """
    def __init__(
        self,
        policy_net: RiskyPolicyNetwork,
        price_net: RiskyPriceNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        optimizer_policy: Optional[tf.keras.optimizers.Optimizer] = None,
        optimizer_price: Optional[tf.keras.optimizers.Optimizer] = None,
        T: int = 32,
        batch_size: int = 128,
        lambda_price: float = 1.0,
        smoothing: Optional[AnnealingSchedule] = None,
        logit_clip: float = 20.0,
        value_net_for_default: Optional[RiskyValueNetwork] = None
    ):
        self.policy_net = policy_net
        self.price_net = price_net
        self.params = params
        self.shock_params = shock_params
        self.optimizer_policy = optimizer_policy or tf.keras.optimizers.Adam(1e-3)
        self.optimizer_price = optimizer_price or tf.keras.optimizers.Adam(1e-3)
        self.T = T
        self.batch_size = batch_size
        self.lambda_price = lambda_price
        self.smoothing = smoothing or AnnealingSchedule(init_temp=0.1, min=1e-4, decay=0.99)
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        # LR needs value estimate for default prob p^D
        self._value_net = value_net_for_default
    
    def train_step(self, k: tf.Tensor, b: tf.Tensor, z_path: tf.Tensor, z_fork: Optional[tf.Tensor] = None, temperature: float = 0.1) -> Dict[str, float]:
        """
        Executes one training step for the Lifetime Reward (LR) + Price Consistency method.

        Args:
            k (tf.Tensor): Initial capital batch. Shape: (Batch, 1) or (Batch,).
            b (tf.Tensor): Initial debt batch. Shape: (Batch, 1) or (Batch,).
            z_path (tf.Tensor): Productivity paths. Shape: (Batch, T+1).
            z_fork (Optional[tf.Tensor]): Forked productivity paths for pricing expectation. Shape: (Batch, T).
            temperature (float): Annealing temperature for smooth gates.

        Returns:
            Dict[str, float]: Dictionary of losses and metrics.
        """
        n = tf.shape(k)[0]
        utilities_list = []
        
        # 1. Rollout for LR
        with tf.GradientTape(persistent=True) as tape:
            k = tf.reshape(k, [-1, 1])
            b = tf.reshape(b, [-1, 1])
            z = tf.reshape(z_path[:, 0], [-1, 1])
            
            # 2. Price Loss (Lifetime Consistency)
            # Accumulate price loss at every step t using deterministic next states from dataset.
            # z_next_main: from z_path (t -> t+1) used for R_1
            # z_next_fork: from z_fork (t) used for R_2 (alternative realization for expectation)
            loss_price_accum = 0.0
            
            for t in range(self.T):
                z_curr = tf.reshape(z_path[:, t], [-1, 1])
                z_next_main = tf.reshape(z_path[:, t+1], [-1, 1])
                
                # Strict usage of z_fork implies we must have it for pricing
                if z_fork is None:
                    raise ValueError("z_fork is required for Risky Debt pricing but was not provided.")
                z_next_fork = tf.reshape(z_fork[:, t], [-1, 1]) 

                # Policy Step: (k, b, z) -> (k', b')
                k_next, b_next = self.policy_net(k, b, z_curr)
                
                # Pricing Step: (k', b', z) -> r_tilde
                r_tilde = self.price_net(k_next, b_next, z_curr)
                
                # Cash Flow Calculation
                e = cash_flow_risky_debt(
                    k, k_next, b, b_next, z_curr, 
                    r_tilde, self.params, temperature=temperature, logit_clip=self.logit_clip
                )
                eta = external_financing_cost(
                    e, self.params, temperature=temperature, logit_clip=self.logit_clip
                )
                utilities_list.append(e - eta)
                
                # --- Price Loss (Deterministic via Data) ---
                if self._value_net is not None:
                    V_tilde_1 = self._value_net(k_next, b_next, z_next_main)
                    V_tilde_2 = self._value_net(k_next, b_next, z_next_fork)
                    p_D_1 = smooth_default_prob(V_tilde_1, self.smoothing, logit_clip=self.logit_clip)
                    p_D_2 = smooth_default_prob(V_tilde_2, self.smoothing, logit_clip=self.logit_clip)
                else:
                    p_D_1 = tf.zeros_like(k_next)
                    p_D_2 = tf.zeros_like(k_next)
                
                R_1 = recovery_value(k_next, z_next_main, self.params)
                R_2 = recovery_value(k_next, z_next_fork, self.params)
                
                f_p1 = pricing_residual_zero_profit(b_next, self.params.r_rate, r_tilde, p_D_1, R_1)
                f_p2 = pricing_residual_zero_profit(b_next, self.params.r_rate, r_tilde, p_D_2, R_2)
                l_p = compute_price_loss_aio(f_p1, f_p2)
                loss_price_accum += l_p
                # -------------------------------------------

                k = k_next
                b = b_next
            
            utilities = tf.concat(utilities_list, axis=1)
            loss_lr = compute_lr_loss(utilities, self.beta)
            loss_price = loss_price_accum / tf.cast(self.T, tf.float32)
            
            loss_policy = loss_lr + self.lambda_price * loss_price
            
        grads_policy = tape.gradient(loss_policy, self.policy_net.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_policy, self.policy_net.trainable_variables))
        
        grads_price = tape.gradient(loss_price, self.price_net.trainable_variables)
        self.optimizer_price.apply_gradients(zip(grads_price, self.price_net.trainable_variables))
        
        del tape
        return {
            "loss_lr": float(loss_lr),
            "loss_price": float(loss_price),
            "mean_utility": float(tf.reduce_mean(utilities))
        }


class RiskyDebtTrainerBR:
    """
    BR Trainer for Risky Debt.
    """
    def __init__(
        self,
        policy_net: RiskyPolicyNetwork,
        value_net: RiskyValueNetwork,
        price_net: RiskyPriceNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        batch_size: int = 128,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        n_critic_steps: int = 1,
        smoothing: Optional[AnnealingSchedule] = None,
        logit_clip: float = 20.0,
        leaky_actor: bool = False
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.price_net = price_net
        self.params = params
        self.shock_params = shock_params
        self.batch_size = batch_size
        self.optimizer_policy = tf.keras.optimizers.Adam(actor_lr)
        self.optimizer_critic = tf.keras.optimizers.Adam(critic_lr)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_critic_steps = n_critic_steps
        self.smoothing = smoothing or AnnealingSchedule(init_temp=0.1, min=1e-4, decay=0.99)
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)
        self.leaky_actor = leaky_actor

    def _critic_step(self, k_init, b_init, z_path, z_fork, temperature):
        """
        Performs one gradient descent step for the Critic (Value + Price networks).

        Updates:
        - Value Network: Minimizes Bellman Residuals using All-in-One (AiO) logic.
        - Price Network: Minimizes equilibrium pricing residuals.

        Args:
            k_init (tf.Tensor): Initial capital. Shape: (Batch, 1).
            b_init (tf.Tensor): Initial debt. Shape: (Batch, 1).
            z_path (tf.Tensor): Productivity trajectory. Shape: (Batch, T_horizon+1).
            z_fork (tf.Tensor): Alternative next-step productivity. Shape: (Batch, T_horizon).
            temperature (float): Gate temperature.

        Returns:
            Tuple[float, float]: (loss_critic_accum, loss_price_accum)
        """
        T_horizon = tf.shape(z_path)[1] - 1
        with tf.GradientTape() as tape:
            loss_critic_accum = 0.0
            loss_price_accum = 0.0
            
            k_curr, b_curr = k_init, b_init
            
            for t in range(T_horizon):
                z = tf.reshape(z_path[:, t], [-1, 1])
                z_next_main = tf.reshape(z_path[:, t+1], [-1, 1])
                
                if z_fork is None:
                    raise ValueError("z_fork is required for Risky Debt pricing but was not provided.")
                z_next_fork = tf.reshape(z_fork[:, t], [-1, 1])
                
                # Policy (Detached)
                k_next, b_next = self.policy_net(k_curr, b_curr, z)
                k_next = tf.stop_gradient(k_next)
                b_next = tf.stop_gradient(b_next)
                
                r_tilde = self.price_net(k_next, b_next, z)
                
                e = cash_flow_risky_debt(k_curr, k_next, b_curr, b_next, z, r_tilde, self.params, temperature=temperature, logit_clip=self.logit_clip)
                eta = external_financing_cost(e, self.params, temperature=temperature, logit_clip=self.logit_clip)
                
                # Continuation
                V_tilde_next_1 = self.value_net(k_next, b_next, z_next_main)
                V_tilde_next_2 = self.value_net(k_next, b_next, z_next_fork)
                V_next_1 = apply_limited_liability(V_tilde_next_1)
                V_next_2 = apply_limited_liability(V_tilde_next_2)
                
                # Target Construction (Detached)
                y1 = e - eta + self.beta * tf.stop_gradient(V_next_1)
                y2 = e - eta + self.beta * tf.stop_gradient(V_next_2)
                
                V_tilde_curr = self.value_net(k_curr, b_curr, z)
                l_br = compute_br_critic_loss_aio(V_tilde_curr, y1, y2)
                loss_critic_accum += l_br
                
                # Price Loss
                p_D_1 = smooth_default_prob(V_tilde_next_1, self.smoothing, logit_clip=self.logit_clip)
                p_D_2 = smooth_default_prob(V_tilde_next_2, self.smoothing, logit_clip=self.logit_clip)
                R_1 = recovery_value(k_next, z_next_main, self.params)
                R_2 = recovery_value(k_next, z_next_fork, self.params)
                
                f_p1 = pricing_residual_zero_profit(b_next, self.params.r_rate, r_tilde, p_D_1, R_1)
                f_p2 = pricing_residual_zero_profit(b_next, self.params.r_rate, r_tilde, p_D_2, R_2)
                l_p = compute_price_loss_aio(f_p1, f_p2)
                loss_price_accum += l_p
                
                k_curr, b_curr = k_next, b_next
            
            loss_critic_accum /= tf.cast(T_horizon, tf.float32)
            loss_price_accum /= tf.cast(T_horizon, tf.float32)
            
            total_loss = compute_critic_objective(loss_critic_accum, loss_price_accum, self.lambda_1)
            
        vars_critic = self.value_net.trainable_variables + self.price_net.trainable_variables
        grads = tape.gradient(total_loss, vars_critic)
        self.optimizer_critic.apply_gradients(zip(grads, vars_critic))
        return loss_critic_accum, loss_price_accum

    def _actor_step(self, k_init, b_init, z_path, z_fork, temperature):
        """
        Performs one gradient descent step for the Actor (Policy network).

        Maximizes:
        - Expected Lifetime Value (via Bellman maximization)
        - Subtracts a penalty for pricing violations involved in the trajectory.

        Args:
            k_init (tf.Tensor): Initial capital. Shape: (Batch, 1).
            b_init (tf.Tensor): Initial debt. Shape: (Batch, 1).
            z_path (tf.Tensor): Productivity trajectory. Shape: (Batch, T_horizon+1).
            z_fork (tf.Tensor): Alternative next-step productivity. Shape: (Batch, T_horizon).
            temperature (float): Gate temperature.

        Returns:
            Tuple[float, float]: (loss_actor_accum, loss_price_accum)
        """
        T_horizon = tf.shape(z_path)[1] - 1
        with tf.GradientTape() as tape:
            loss_actor_accum = 0.0
            loss_price_accum = 0.0
            
            k_curr, b_curr = k_init, b_init
            
            for t in range(T_horizon):
                z = tf.reshape(z_path[:, t], [-1, 1])
                z_next_main = tf.reshape(z_path[:, t+1], [-1, 1])
                
                if z_fork is None:
                    raise ValueError("z_fork is required for Risky Debt pricing but was not provided.")
                z_next_fork = tf.reshape(z_fork[:, t], [-1, 1])
                
                # Policy Rollout (Gradients Flow)
                k_next, b_next = self.policy_net(k_curr, b_curr, z)
                
                r_tilde = self.price_net(k_next, b_next, z)
                e = cash_flow_risky_debt(k_curr, k_next, b_curr, b_next, z, r_tilde, self.params, temperature=temperature, logit_clip=self.logit_clip)
                eta = external_financing_cost(e, self.params, temperature=temperature, logit_clip=self.logit_clip)
                
                # Continuation (Stop Gradient on Value)
                V_tilde_next_1 = self.value_net(k_next, b_next, z_next_main)
                V_tilde_next_2 = self.value_net(k_next, b_next, z_next_fork)
                V_next_1 = apply_limited_liability(V_tilde_next_1, leaky=self.leaky_actor)
                V_next_2 = apply_limited_liability(V_tilde_next_2, leaky=self.leaky_actor)
                
                # Actor Loss
                l_actor = compute_br_actor_loss_risky(e, eta, V_next_1, V_next_2, self.beta)
                loss_actor_accum += l_actor
                
                # Price Constraint
                p_D_1 = smooth_default_prob(V_tilde_next_1, self.smoothing, logit_clip=self.logit_clip)
                p_D_2 = smooth_default_prob(V_tilde_next_2, self.smoothing, logit_clip=self.logit_clip)
                R_1 = recovery_value(k_next, z_next_main, self.params)
                R_2 = recovery_value(k_next, z_next_fork, self.params)
                
                f_p1 = pricing_residual_zero_profit(b_next, self.params.r_rate, r_tilde, p_D_1, R_1)
                f_p2 = pricing_residual_zero_profit(b_next, self.params.r_rate, r_tilde, p_D_2, R_2)
                l_p = compute_price_loss_aio(f_p1, f_p2)
                loss_price_accum += l_p
                
                k_curr, b_curr = k_next, b_next
            
            loss_actor_accum /= tf.cast(T_horizon, tf.float32)
            loss_price_accum /= tf.cast(T_horizon, tf.float32)
            
            total_loss = compute_actor_objective(loss_actor_accum, loss_price_accum, self.lambda_2)
            
        grads = tape.gradient(total_loss, self.policy_net.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        return loss_actor_accum, loss_price_accum

    def train_step(self, k, b, z_path, z_fork, temperature=0.1):
        critic_l, price_l_c = [], []
        
        # Critic specific loop if n_critic_steps > 1
        # Since helpers now run the full T loop, this is "n_critic_epochs" essentially.
        for _ in range(self.n_critic_steps):
            l_c, l_p = self._critic_step(k, b, z_path, z_fork, temperature)
            critic_l.append(float(l_c))
            price_l_c.append(float(l_p))
            
        loss_actor, loss_price_actor = self._actor_step(k, b, z_path, z_fork, temperature)
        self.smoothing.update()
        
        return {
            "loss_critic": np.mean(critic_l),
            "loss_actor": float(loss_actor),
            "loss_price_critic": np.mean(price_l_c),
            "loss_price_actor": float(loss_price_actor),
            "epsilon_D": self.smoothing.value
        }


# =============================================================================
# HIGH-LEVEL ENTRY POINTS
# =============================================================================

def train_risky_lr(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """Train Risky Debt Model using LR + Price."""
    k_bounds = bounds['k']
    b_bounds = bounds['b']
    
    # Check for risky config
    if method_config.risky is None:
        raise ValueError("RiskyDebtConfig required in method_config.risky for train_risky_lr")
    risky_cfg = method_config.risky

    # 0. Data Setup
    if 'z_path' in dataset:
        T = dataset['z_path'].shape[1] - 1
    else:
        raise ValueError("Cannot infer Horizon T from dataset.")
        
    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k0'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    policy_net, value_net, price_net = build_risky_networks(
        k_min=k_bounds[0], k_max=k_bounds[1],
        b_min=b_bounds[0], b_max=b_bounds[1],
        r_risk_free=params.r_rate,
        n_layers=net_config.n_layers, n_neurons=net_config.n_neurons, activation=net_config.activation
    )
    
    # Pre-build value net for default prob? LR usually doesn't train Value.
    # We pass None or strict separation. Here we pass None to replicate original behavior unless we add value training.
    # Original trainer_risky used `self._value_net` but didn't train it in LR mode.
    
    trainer = RiskyDebtTrainerLR(
        policy_net=policy_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        T=T,
        batch_size=opt_config.batch_size,
        lambda_price=risky_cfg.lambda_price,
        logit_clip=anneal_config.logit_clip,
        value_net_for_default=value_net # Passed but not trained here
    )
    
    history = execute_training_loop(
        trainer, 
        data_iter, 
        opt_config,
        anneal_config, 
        method_name="risky_lr"
    )
    
    return {
        "history": history,
        "_policy_net": policy_net, "_price_net": price_net, "_value_net": value_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        }, 
        "_params": params
    }


def train_risky_br(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """Train Risky Debt Model using BR (Actor-Critic)."""
    k_bounds = bounds['k']
    b_bounds = bounds['b']
    
    # Check for risky config
    if method_config.risky is None:
        raise ValueError("RiskyDebtConfig required in method_config.risky for train_risky_br")
    risky_cfg = method_config.risky

    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k0'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    policy_net, value_net, price_net = build_risky_networks(
        k_min=k_bounds[0], k_max=k_bounds[1],
        b_min=b_bounds[0], b_max=b_bounds[1],
        r_risk_free=params.r_rate,
        n_layers=net_config.n_layers, n_neurons=net_config.n_neurons, activation=net_config.activation
    )
    
    trainer = RiskyDebtTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        batch_size=opt_config.batch_size,
        actor_lr=opt_config.learning_rate,
        # Default critic LR to actor LR if not specified
        critic_lr=opt_config.learning_rate_critic or opt_config.learning_rate,
        lambda_1=risky_cfg.lambda_1,
        lambda_2=risky_cfg.lambda_2,
        n_critic_steps=method_config.n_critic,
        logit_clip=anneal_config.logit_clip
    )
    
    history = execute_training_loop(
        trainer, 
        data_iter, 
        opt_config,
        anneal_config, 
        method_name="risky_br"
    )
    
    return {
        "history": history,
        "_policy_net": policy_net, "_value_net": value_net, "_price_net": price_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        },
        "_params": params
    }
