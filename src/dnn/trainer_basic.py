"""
src/dnn/trainer_basic.py

Training orchestrators for the Basic Model (Sec. 1).
Implements LR, ER, and BR (actor-critic) training methods.

Note: Economic logic (cash flow, Euler primitives) is delegated to src/economy.
Trainers handle: sampling, forward passes, loss formation, gradient rules, optimization.

Reference: outline_v2.md Sections 1.2, 1.3, 1.4
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple

from src.dnn.networks import BasicPolicyNetwork, BasicValueNetwork
from src.dnn.losses import (
    compute_lr_loss,
    compute_er_loss_aio,
    compute_br_critic_loss_aio,
    compute_br_critic_diagnostics,
    compute_br_actor_loss
)
from src.dnn.sampling import draw_shocks, step_ar1_tf
from src.economy.logic import (
    compute_cash_flow_basic,
    euler_chi,
    euler_m,
)


# =============================================================================
# LIFETIME REWARD TRAINER (Sec. 1.2)
# =============================================================================

class BasicTrainerLR:
    """
    Method 1: Lifetime Reward Maximization for Basic model.
    
    Trains policy network by maximizing expected discounted rewards:
        J(theta) = E[sum_t beta^t * e(k_t, k_{t+1}, z_t)]
        L = -J
    
    Args:
        policy_net: Policy network (k, z) -> k'
        params: Economic parameters
        optimizer: TensorFlow optimizer
        T: Rollout horizon
        batch_size: Number of parallel trajectories
        seed: Random seed for reproducibility
    
    Reference: outline_v2.md lines 189-199
    """
    
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params,  # ModelParameters
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        T: int = 32,
        batch_size: int = 128,
        seed: Optional[int] = None
    ):
        self.policy_net = policy_net
        self.params = params
        self.optimizer = optimizer or tf.keras.optimizers.Adam(1e-3)
        self.T = T
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        
        self.beta = 1.0 / (1.0 + params.r_rate)
    
    def train_step(
        self,
        k_init: tf.Tensor,
        z_init: tf.Tensor,
        temperature: float = 0.1
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            k_init: Initial capital (batch_size,)
            z_init: Initial productivity (batch_size,)
            temperature: Annealing temperature for smooth gates
        
        Returns:
            Dictionary with loss value
        """
        rewards_list = []
        
        with tf.GradientTape() as tape:
            k = tf.reshape(k_init, [-1, 1])
            z = tf.reshape(z_init, [-1, 1])
            
            for t in range(self.T):
                # Policy: k' = Gamma_policy(k, z)
                k_next = self.policy_net(k, z)
                
                # Reward - delegated to economy module
                reward = compute_cash_flow_basic(k, k_next, z, self.params, temperature=temperature)
                rewards_list.append(reward)
                
                # Transition: z' from AR(1) - use canonical implementation
                z = step_ar1_tf(z, self.params.rho, self.params.sigma, self.params.mu)
                k = k_next
            
            # Stack rewards: (batch, T)
            rewards = tf.concat(rewards_list, axis=1)
            
            # Loss = -mean(sum_t beta^t * e_t)
            loss = compute_lr_loss(rewards, self.beta)
        
        # Update policy
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.policy_net.trainable_variables)
        )
        
        # Return terminal states for ergodic sampling
        # k, z are the terminal states after T-step rollout
        terminal_states = tf.concat([k, z], axis=1).numpy()
        
        return {
            "loss": float(loss), 
            "mean_reward": float(tf.reduce_mean(rewards)),
            "terminal_states": terminal_states
        }


# =============================================================================
# EULER RESIDUAL TRAINER (Sec. 1.3)
# =============================================================================

class BasicTrainerER:
    """
    Method 2: Euler Residual Minimization for Basic model (AiO form).
    
    Trains policy network by minimizing Euler equation residuals:
        f = chi(k,z) - beta * E[m(k', z')]
        L = E[f_1 * f_2]  (AiO unbiased estimator)
    
    WARNING: If cost_fixed > 0, FOC is undefined at I=0.
    
    Args:
        policy_net: Policy network (k, z) -> k'
        params: Economic parameters
        optimizer: TensorFlow optimizer
        batch_size: Batch size
        seed: Random seed
    
    Reference: outline_v2.md lines 203-239
    """
    
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        batch_size: int = 128,
        seed: Optional[int] = None
    ):
        self.policy_net = policy_net
        self.params = params
        self.optimizer = optimizer or tf.keras.optimizers.Adam(1e-3)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        # Warning for fixed costs (per spec line 238-239)
        if params.cost_fixed > 0:
            import warnings
            warnings.warn(
                "Euler Residual method with cost_fixed > 0: "
                "FOC is undefined at I=0. Consider using LR or BR methods."
            )
    
    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            k: Current capital (batch_size,)
            z: Current productivity (batch_size,)
        
        Returns:
            Dictionary with loss value
        """
        n = tf.shape(k)[0]
        
        with tf.GradientTape() as tape:
            k = tf.reshape(k, [-1, 1])
            z = tf.reshape(z, [-1, 1])
            
            # Current policy
            k_next = self.policy_net(k, z)
            
            # chi(k, z) = 1 + psi_I(I, k)
            chi = euler_chi(k, k_next, self.params)
            
            # Two independent z' draws for AiO
            z_next_1, z_next_2 = draw_shocks(
                n, z, self.params.rho, self.params.sigma, self.params.mu
            )
            
            # Policy at (k', z'_1) and (k', z'_2)
            k_next_next_1 = self.policy_net(k_next, z_next_1)
            k_next_next_2 = self.policy_net(k_next, z_next_2)
            
            # m(k', z'_1) and m(k', z'_2)
            m1 = euler_m(k_next, k_next_next_1, z_next_1, self.params)
            m2 = euler_m(k_next, k_next_next_2, z_next_2, self.params)
            
            # Euler residuals
            f1 = chi - self.beta * m1
            f2 = chi - self.beta * m2
            
            # AiO loss
            loss = compute_er_loss_aio(f1, f2)
        
        # Update policy
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        if grads[0] is not None:
            self.optimizer.apply_gradients(
                zip(grads, self.policy_net.trainable_variables)
            )
        
        # Return terminal states for ergodic sampling
        # Use (k', z'_1) as one-step lookahead for next iteration
        terminal_states = tf.concat([k_next, z_next_1], axis=1).numpy()
        
        return {"loss": float(loss), "terminal_states": terminal_states}


# =============================================================================
# BELLMAN RESIDUAL TRAINER (Sec. 1.4)
# =============================================================================

class BasicTrainerBR:
    """
    Method 3: Bellman Residual (Actor-Critic) for Basic model.
    
    Trains both policy and value networks:
    - Critic: Minimize L_critic = E[(V - y_1)(V - y_2)] with detached RHS
    - Actor: Minimize L_actor = -E[e + beta * (V'_1 + V'_2)/2]
    
    Gradient-flow rules:
    - Critic: stop-gradient on RHS continuation term only
    - Actor: freeze value params (no update), but allow gradients through V(k')
    
    Target Value Network (optional):
    - When use_value_target=True, V_next in critic targets is computed from a 
      slowly-updated target network (Polyak averaging), reducing moving-target
      bootstrap instability.
    - Actor step still uses online value_net so gradients flow through V(k').
    
    Args:
        policy_net: Policy network
        value_net: Value network
        params: Economic parameters
        actor_lr: Learning rate for policy optimizer (default: 1e-3)
        critic_lr: Learning rate for value optimizer (default: 1e-3)
        optimizer_policy: Custom optimizer for policy (overrides actor_lr if provided)
        optimizer_value: Custom optimizer for value (overrides critic_lr if provided)
        batch_size: Batch size
        n_critic_steps: Critic updates per actor update
        seed: Random seed
        use_value_target: If True, use target network for RHS continuation
        value_target_mix: Polyak averaging coefficient (0,1], higher = faster update
    
    Reference: outline_v2.md lines 243-280
    """
    
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        value_net: BasicValueNetwork,
        params,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        optimizer_policy: Optional[tf.keras.optimizers.Optimizer] = None,
        optimizer_value: Optional[tf.keras.optimizers.Optimizer] = None,
        batch_size: int = 128,
        n_critic_steps: int = 20,
        seed: Optional[int] = None,
        use_value_target: bool = True,  # For stability
        value_target_mix: float = 0.01
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.params = params
        # Use provided optimizer or create from LR
        self.optimizer_policy = optimizer_policy or tf.keras.optimizers.Adam(actor_lr)
        self.optimizer_value = optimizer_value or tf.keras.optimizers.Adam(critic_lr)
        self.batch_size = batch_size
        self.n_critic_steps = n_critic_steps
        self._rng = np.random.default_rng(seed)
        
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        # Target value network (optional)
        self.use_value_target = use_value_target
        self.value_target_mix = value_target_mix
        
        if use_value_target:
            # Clone architecture by inferring from value_net
            n_layers = len(value_net.hidden_layers)
            n_neurons = value_net.hidden_layers[0].units
            # Get activation name from first hidden layer
            activation = value_net.hidden_layers[0].activation.__name__
            
            self.value_net_target = BasicValueNetwork(
                n_layers=n_layers,
                n_neurons=n_neurons,
                activation=activation
            )
            # Build both networks with dummy input
            dummy_k = tf.zeros((1, 1))
            dummy_z = tf.zeros((1, 1))
            self.value_net(dummy_k, dummy_z)  # build source if not built
            self.value_net_target(dummy_k, dummy_z)  # build target
            # Copy weights
            self.value_net_target.set_weights(self.value_net.get_weights())
        else:
            self.value_net_target = None
    
    def _polyak_update_value_target(self):
        """Update target network weights via Polyak/EMA averaging."""
        if not self.use_value_target or self.value_net_target is None:
            return
        
        mix = self.value_target_mix
        target_weights = self.value_net_target.get_weights()
        online_weights = self.value_net.get_weights()
        
        new_weights = [
            (1.0 - mix) * tw + mix * ow 
            for tw, ow in zip(target_weights, online_weights)
        ]
        self.value_net_target.set_weights(new_weights)
    
    def _critic_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        temperature: float = 0.1
    ) -> tf.Tensor:
        """
        Perform one critic update.
        
        Stop-gradient on k' (policy fixed) and on RHS continuation.
        
        Returns:
            Critic loss
        """
        n = tf.shape(k)[0]
        
        with tf.GradientTape() as tape:
            # Policy output (detached for critic)
            k_next = self.policy_net(k, z)
            k_next_sg = tf.stop_gradient(k_next)
            
            # Cash flow (uses detached k') - delegated to economy
            e = compute_cash_flow_basic(k, k_next_sg, z, self.params, temperature=temperature)
            
            # Two z' draws
            z_next_1, z_next_2 = draw_shocks(
                n, z, self.params.rho, self.params.sigma, self.params.mu
            )
            
            # Continuation values - use target net if enabled
            if self.use_value_target and self.value_net_target is not None:
                V_next_1 = self.value_net_target(k_next_sg, z_next_1)
                V_next_2 = self.value_net_target(k_next_sg, z_next_2)
            else:
                V_next_1 = self.value_net(k_next_sg, z_next_1)
                V_next_2 = self.value_net(k_next_sg, z_next_2)
            
            # Always detach RHS continuation
            V_next_1_sg = tf.stop_gradient(V_next_1)
            V_next_2_sg = tf.stop_gradient(V_next_2)
            
            # Targets
            y1 = e + self.beta * V_next_1_sg
            y2 = e + self.beta * V_next_2_sg
            
            # Current value (trainable LHS)
            V_curr = self.value_net(k, z)
            
            # Critic loss
            loss = compute_br_critic_loss_aio(V_curr, y1, y2)
            
            # Compute diagnostics (outside gradient tape)
            diagnostics = compute_br_critic_diagnostics(V_curr, y1, y2)
        
        # Update value network only
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        self.optimizer_value.apply_gradients(
            zip(grads, self.value_net.trainable_variables)
        )
        
        return loss, diagnostics
    
    def _actor_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        temperature: float = 0.1
    ) -> tf.Tensor:
        """
        Perform one actor update.
        
        Value params frozen (no update), but gradients flow through V(k').
        
        Returns:
            Actor loss
        """
        n = tf.shape(k)[0]
        
        with tf.GradientTape() as tape:
            # Policy output (NOT detached - gradients flow through)
            k_next = self.policy_net(k, z)
            
            # Cash flow - delegated to economy
            e = compute_cash_flow_basic(k, k_next, z, self.params, temperature=temperature)
            
            # Two z' draws
            z_next_1, z_next_2 = draw_shocks(
                n, z, self.params.rho, self.params.sigma, self.params.mu
            )
            
            # Continuation values (NOT detached - gradients flow to policy via k')
            V_next_1 = self.value_net(k_next, z_next_1)
            V_next_2 = self.value_net(k_next, z_next_2)
            
            # Actor loss with averaged continuation
            loss = compute_br_actor_loss(e, V_next_1, V_next_2, self.beta)
        
        # Update policy network ONLY (value is frozen)
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer_policy.apply_gradients(
            zip(grads, self.policy_net.trainable_variables)
        )
        
        return loss
    
    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        temperature: float = 0.1
    ) -> Dict[str, float]:
        """
        Perform one outer training step.
        
        1. N_critic critic updates
        2. 1 actor update
        
        Args:
            k: Current capital (batch_size,)
            z: Current productivity (batch_size,)
            temperature: Annealing temperature
        
        Returns:
            Dictionary with losses
        """
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        
        # Critic updates
        critic_losses = []
        mse_proxies = []
        mae_proxies = []
        for _ in range(self.n_critic_steps):
            loss_c, diag = self._critic_step(k, z, temperature=temperature)
            critic_losses.append(float(loss_c))
            mse_proxies.append(diag["mse_proxy"])
            mae_proxies.append(diag["mae_proxy"])
        
        # Polyak update target network (if enabled)
        self._polyak_update_value_target()
        
        # Actor update
        loss_a = self._actor_step(k, z, temperature=temperature)
        
        # Compute terminal states for ergodic sampling (k', z')
        n = tf.shape(k)[0]
        k_next = self.policy_net(k, z)
        z_next, _ = draw_shocks(n, z, self.params.rho, self.params.sigma, self.params.mu)
        terminal_states = tf.concat([k_next, z_next], axis=1).numpy()
        
        return {
            "loss_critic": np.mean(critic_losses),
            "loss_actor": float(loss_a),
            "mse_proxy": np.mean(mse_proxies),
            "mae_proxy": np.mean(mae_proxies),
            "terminal_states": terminal_states,
        }
