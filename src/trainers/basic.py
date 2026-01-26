"""
src/trainers/basic.py

Trainers and high-level training entry points for the Basic Model (Sec. 1).
Implements LR, ER, and BR (actor-critic) training methods.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Optional, Any, Iterator, List, Tuple

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.logic import compute_cash_flow_basic, euler_chi, euler_m

from src.networks.network_basic import BasicPolicyNetwork, BasicValueNetwork, build_basic_networks
from src.trainers.losses import (
    compute_lr_loss,
    compute_er_loss_aio,
    compute_br_critic_loss_aio,
    compute_br_critic_diagnostics,
    compute_br_actor_loss
)
from src.trainers.config import NetworkConfig, OptimizationConfig, AnnealingConfig, MethodConfig
from src.trainers.core import execute_training_loop

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINER CLASSES (Low-Level Logic)
# =============================================================================

class BasicTrainerLR:
    """
    Method 1: Lifetime Reward Maximization for Basic model.
    Trains policy network by maximizing expected discounted rewards.
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        T: int,
        logit_clip: float,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    ):
        self.policy_net = policy_net
        self.params = params
        self.shock_params = shock_params
        self.optimizer = optimizer or tf.keras.optimizers.Adam(1e-3)
        self.T = T
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)
    
    def train_step(
        self,
        k: tf.Tensor,
        z_path: tf.Tensor,
        temperature: float,
    ) -> Dict[str, float]:
        """
        Executes one training step via Lifetime Reward Maximization.

        Args:
            k (tf.Tensor): Initial capital batch. Shape: (Batch, 1).
            z_path (tf.Tensor): Productivity trajectory. Shape: (Batch, T+1).
            temperature (float): Gate temperature.

        Returns:
            Dict[str, float]: Loss and metrics.
        """
        rewards_list = []
        with tf.GradientTape() as tape:
            k = tf.reshape(k, [-1, 1])
            # Use pre-computed path. z_path is (Batch, T+1)
            # This ensures determinism and uses the exact shock realizations from data generation.
            
            for t in range(self.T):
                z_curr = tf.reshape(z_path[:, t], [-1, 1])
                
                k_next = self.policy_net(k, z_curr)
                reward = compute_cash_flow_basic(k, k_next, z_curr, self.params, temperature=temperature, logit_clip=self.logit_clip)
                rewards_list.append(reward)
                
                k = k_next
            
            rewards = tf.concat(rewards_list, axis=1)
            loss = compute_lr_loss(rewards, self.beta)
        
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        
        z_terminal = tf.reshape(z_path[:, self.T], [-1, 1])
        terminal_states = tf.concat([k, z_terminal], axis=1).numpy()
        return {
            "loss_LR": float(loss), 
            "mean_reward": float(tf.reduce_mean(rewards)),
            "terminal_states": terminal_states
        }


class BasicTrainerER:
    """
    Method 2: Euler Residual Minimization for Basic model.
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    ):
        self.policy_net = policy_net
        self.params = params
        self.shock_params = shock_params
        self.optimizer = optimizer or tf.keras.optimizers.Adam(1e-3)
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        if params.cost_fixed > 0:
            logger.warning("ER with cost_fixed > 0: FOC undefined at I=0. Approximated with smooth function.")
    
    def train_step(self, k: tf.Tensor, z_path: tf.Tensor, z_fork: tf.Tensor) -> Dict[str, float]:
        """
        Executes one training step via Euler Residual Minimization.

        Uses explicit 'All-in-One' (AiO) method where a single batch contains pairs of shocks
        to estimate conditional expectations without nested integration loops.

        Args:
            k (tf.Tensor): Initial capital. Shape: (Batch, 1).
            z_path (tf.Tensor): Main productivity path. Shape: (Batch, T+1).
            z_fork (tf.Tensor): Forked productivity path for expectation. Shape: (Batch, T).

        Returns:
            Dict[str, float]: Loss metrics.
        """
        with tf.GradientTape() as tape:
            # Initialize single path
            k_curr = tf.reshape(k, [-1, 1])
            
            # Infer T from z_path (shape: [Batch, T+1])
            T = tf.shape(z_path)[1] - 1
            
            losses = []
            
            # Unroll loop over time
            for t in range(T):
                # Data for this step
                z_curr = tf.reshape(z_path[:, t], [-1, 1])
                z_next_main = tf.reshape(z_path[:, t+1], [-1, 1])
                z_next_fork = tf.reshape(z_fork[:, t], [-1, 1])
                
                # Policy Step
                k_next = self.policy_net(k_curr, z_curr)
                
                # 1. Current Euler LHS (Chi)
                chi = euler_chi(k_curr, k_next, self.params)
                
                # 2. Future Euler RHS (Expected beta * m)
                # Branch 1: Main Path
                k_next_next_main = self.policy_net(k_next, z_next_main)
                m_main = euler_m(k_next, k_next_next_main, z_next_main, self.params)
                f_main = chi - self.beta * m_main
                
                # Branch 2: Fork Path
                k_next_next_fork = self.policy_net(k_next, z_next_fork)
                m_fork = euler_m(k_next, k_next_next_fork, z_next_fork, self.params)
                f_fork = chi - self.beta * m_fork
                
                # Compute Stepwise AiO Loss (Minimize E[f] using two samples)
                step_loss = compute_er_loss_aio(f_main, f_fork)
                losses.append(step_loss)
                
                # Update state
                k_curr = k_next
            
            # Aggregate Lifetime Loss (Mean over time)
            loss = tf.reduce_mean(losses)
        
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        if grads[0] is not None:
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
            
        return {"loss_ER": float(loss)}


class BasicTrainerBR:
    """
    Method 3: Bellman Residual (Actor-Critic).
    
    Refactored to use Lifetime Trajectories and Data-Driven Shocks.
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        value_net: BasicValueNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        n_critic_steps: int = 20,
        logit_clip: float = 20.0
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.params = params
        self.shock_params = shock_params
        self.optimizer_policy = tf.keras.optimizers.Adam(actor_lr)
        self.optimizer_value = tf.keras.optimizers.Adam(critic_lr)
        self.n_critic_steps = n_critic_steps
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        # Target Network (Not used in simple BR, but kept if needed later)
        self.value_net_target = None


    def _compute_critic_loss(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor,
        k_next: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float
    ) -> Tuple[tf.Tensor, float]:
        """
        Compute AiO Critic Loss for a single transition batch.
        Everything here uses STOP_GRADIENT on targets or assumes detached inputs if from fixed path.
        """
        # Current Value Estimate
        V_curr = self.value_net(k_curr, z_curr)
        
        # Continuation Values (Detached)
        V_next_main = tf.stop_gradient(self.value_net(k_next, z_next_main))
        V_next_fork = tf.stop_gradient(self.value_net(k_next, z_next_fork))
        
        # Compute Cash Flow (Target is Bellman RHS)
        e = compute_cash_flow_basic(
            k_curr, k_next, z_curr, 
            self.params, temperature=temperature, logit_clip=self.logit_clip
        )
        
        y1 = e + self.beta * V_next_main
        y2 = e + self.beta * V_next_fork
        
        # AiO Loss
        loss_critic = compute_br_critic_loss_aio(V_curr, y1, y2)
        
        # Diagnostic MSE Proxy
        mse_proxy = compute_br_critic_diagnostics(V_curr, y1, y2)["mse_proxy"]
        
        return loss_critic, mse_proxy

    def _compute_actor_loss(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor,
        k_next: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float
    ) -> tf.Tensor:
        """
        Compute AiO Actor Loss for a single transition batch.
        Gradients MUST flow through k_next to policy.
        """
        # Re-Compute Cash Flow (Attached to Tape)
        e = compute_cash_flow_basic(
            k_curr, k_next, z_curr, 
            self.params, temperature=temperature, logit_clip=self.logit_clip
        )
        
        # Re-Compute V_next (Attached to Tape, but V weights fixed)
        V_next_main = self.value_net(k_next, z_next_main)
        V_next_fork = self.value_net(k_next, z_next_fork)
        
        # Actor Loss
        loss_actor = compute_br_actor_loss(e, V_next_main, V_next_fork, self.beta)
        
        return loss_actor

    def train_step(
        self, 
        k: tf.Tensor, 
        z_path: tf.Tensor, 
        z_fork: tf.Tensor,
        temperature: float
    ) -> Dict[str, float]:
        """
        Perform a training step using Lifetime Bellman Residual Minimization.
        Refactored to use Pre-computed Forks from DataGenerator.
        """
        # 1. Data Prep
        k0_curr = tf.reshape(k, [-1, 1])
        T = tf.shape(z_path)[1] - 1
        
        # z_path is (B, T+1), z_fork is (B, T, 1)
        
        # 2. Policy Rollout (Generate Trajectory of k)
        
        k_seq = [k0_curr]
        k_curr = k0_curr
        for t in range(T):
            z_curr = tf.reshape(z_path[:, t], [-1, 1])
            k_next = self.policy_net(k_curr, z_curr)
            k_seq.append(k_next)
            k_curr = k_next
        k_path_stack = tf.stack(k_seq, axis=1) # (2B, T+1, 1)
        
        # Detach for Critic
        k_path_fixed = tf.stop_gradient(k_path_stack)
        
        # 3. Critic Training Loop
        critic_losses = []
        mses = []
        
        for _ in range(self.n_critic_steps):
            with tf.GradientTape() as tape_critic:
                step_losses = []
                
                # Loop over time T
                for t in range(T):
                    # Slices
                    k_t = tf.reshape(k_path_fixed[:, t, :], [-1, 1])
                    k_tp1 = tf.reshape(k_path_fixed[:, t+1, :], [-1, 1])
                    z_t = tf.reshape(z_path[:, t], [-1, 1])
                    z_tp1_main = tf.reshape(z_path[:, t+1], [-1, 1])
                    z_tp1_fork = tf.reshape(z_fork[:, t], [-1, 1])
                    
                    sloss, smse = self._compute_critic_loss(
                        k_t, z_t, k_tp1, z_tp1_main, z_tp1_fork, temperature
                    )
                    step_losses.append(sloss)
                    
                    if _ == self.n_critic_steps - 1:
                        mses.append(smse) # track last iter
                
                # Average over Time
                total_loss_critic = tf.reduce_mean(step_losses)
            
            # Update Critic
            grads_crit = tape_critic.gradient(total_loss_critic, self.value_net.trainable_variables)
            self.optimizer_value.apply_gradients(zip(grads_crit, self.value_net.trainable_variables))
            critic_losses.append(total_loss_critic)

        # 4. Actor Update
        # Re-Rollout with Tape to get gradients
        with tf.GradientTape() as tape_actor:
            k_curr = k0_curr
            step_losses_actor = []
            
            for t in range(T):
                z_curr = tf.reshape(z_path[:, t], [-1, 1])
                k_next = self.policy_net(k_curr, z_curr)
                
                # Next states for Loss
                z_tp1_main = tf.reshape(z_path[:, t+1], [-1, 1])
                z_tp1_fork = tf.reshape(z_fork[:, t], [-1, 1])
                
                sloss = self._compute_actor_loss(
                    k_curr, z_curr, k_next, z_tp1_main, z_tp1_fork, temperature
                )
                step_losses_actor.append(sloss)
                
                # Update State
                k_curr = k_next
                
            # Average Actor Loss
            total_loss_actor = tf.reduce_mean(step_losses_actor)
            
        # Update Actor
        grads_actor = tape_actor.gradient(total_loss_actor, self.policy_net.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_actor, self.policy_net.trainable_variables))
        
        return {
            "loss_critic": float(np.mean(critic_losses)),
            "loss_actor": float(total_loss_actor),
            "mse_proxy": float(np.mean(mses))
        }


# =============================================================================
# HIGH-LEVEL ENTRY POINTS
# =============================================================================

def train_basic_lr(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Train Basic Model using Lifetime Reward (LR).
    """
    k_bounds = bounds['k']
    
    # 0. Data Setup
    if 'z_path' in dataset:
        T = dataset['z_path'].shape[1] - 1
    else:
        raise ValueError("Cannot infer Horizon T from dataset (missing 'z_path').")
        
    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k0'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    # 1. Build Networks
    policy_net, _ = build_basic_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        activation=net_config.activation
    )
    
    # 2. Setup Trainer
    optimizer = tf.keras.optimizers.Adam(learning_rate=opt_config.learning_rate)
    trainer = BasicTrainerLR(
        policy_net=policy_net,
        params=params,
        shock_params=shock_params,
        optimizer=optimizer,
        T=T,
        logit_clip=anneal_config.logit_clip
    )
    
    # 3. Execute Loop
    history = execute_training_loop(
        trainer, 
        data_iter, 
        opt_config,
        anneal_config, 
        method_name="basic_lr"
    )
    
    return {
        "history": history,
        "_policy_net": policy_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        },
        "_params": params
    }


def train_basic_er(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Train Basic Model using Euler Residuals (ER).
    """
    k_bounds = bounds['k']
    
    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k0'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    policy_net, _ = build_basic_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        activation=net_config.activation
    )
    
    trainer = BasicTrainerER(
        policy_net=policy_net,
        params=params,
        shock_params=shock_params,
        optimizer=tf.keras.optimizers.Adam(opt_config.learning_rate)
    )
    
    history = execute_training_loop(
        trainer, 
        data_iter, 
        opt_config,
        anneal_config, 
        method_name="basic_er"
    )
    
    return {
        "history": history,
        "_policy_net": policy_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        },
        "_params": params
    }


def train_basic_br(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Train Basic Model using Bellman Residuals (Actor-Critic).
    """
    k_bounds = bounds['k']
    
    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k0'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    policy_net, value_net = build_basic_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        activation=net_config.activation
    )
    
    trainer = BasicTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        params=params,
        shock_params=shock_params,
        actor_lr=opt_config.learning_rate,
        # Default critic LR to actor LR if not specified
        critic_lr=opt_config.learning_rate_critic or opt_config.learning_rate,
        n_critic_steps=method_config.n_critic,
        logit_clip=anneal_config.logit_clip
    )
    
    history = execute_training_loop(
        trainer, 
        data_iter, 
        opt_config,
        anneal_config, 
        method_name="basic_br"
    )
    
    return {
        "history": history,
        "_policy_net": policy_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        },
        "_params": params
    }
