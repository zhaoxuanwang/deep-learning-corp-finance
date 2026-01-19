"""
src/dnn/trainer_risky.py

Training orchestrators for the Risky Debt Model (Sec. 2).
Implements LR+price and BR (actor-critic)+price training methods.

Note: Economic logic (cash flow, recovery, pricing) is delegated to src/economy.
Trainers handle: sampling, forward passes, loss formation, gradient rules, optimization.

Reference: outline_v2.md Sections 2.3, 2.4, 2.5
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.dnn.networks import (
    RiskyPolicyNetwork,
    RiskyValueNetwork,
    RiskyPriceNetwork,
    apply_limited_liability
)
from src.dnn.losses import (
    compute_lr_loss_risky,
    compute_br_critic_loss_aio,
    compute_br_actor_loss_risky,
    compute_price_loss_aio,
    compute_critic_objective,
    compute_actor_objective
)
from src.dnn.default_smoothing import DefaultSmoothingSchedule
from src.dnn.sampling import ReplayBuffer, draw_shocks, step_ar1_tf
from src.economy.logic import (
    cash_flow_risky_debt,
    external_financing_cost,
    recovery_value,
    pricing_residual_zero_profit,
)


# =============================================================================
# TRAINING DIAGNOSTICS
# =============================================================================

@dataclass
class TrainingDiagnostics:
    """
    Diagnostic metrics for debugging gradient flow in BR training.
    
    Attributes:
        grad_norm_policy: L2 norm of gradients w.r.t. policy params
        grad_norm_value: L2 norm of gradients w.r.t. value params (critic step)
        grad_norm_price: L2 norm of gradients w.r.t. price params (critic step)
        share_v_tilde_negative: Fraction of samples with V_tilde < 0
        mean_v_tilde: Mean of V_tilde across batch
        sigmoid_saturation_rate: Fraction of |sigmoid preact| > 0.95 (for b')
        mean_leverage: Mean b'/k' ratio
    """
    grad_norm_policy: float = 0.0
    grad_norm_value: float = 0.0
    grad_norm_price: float = 0.0
    share_v_tilde_negative: float = 0.0
    mean_v_tilde: float = 0.0
    sigmoid_saturation_rate: float = 0.0
    mean_leverage: float = 0.0


def _compute_grad_norm(grads) -> float:
    """Compute L2 norm of gradients."""
    if grads is None:
        return 0.0
    total = 0.0
    for g in grads:
        if g is not None:
            total += tf.reduce_sum(tf.square(g))
    return float(tf.sqrt(total))

# =============================================================================
# LIFETIME REWARD TRAINER FOR RISKY DEBT (Sec. 2.3)
# =============================================================================

class RiskyDebtTrainerLR:
    """
    Risky Debt LR Training (Sec. 2.3).
    
    - Policy trained on: L_LR = -J + lambda_price * L_price
    - Price trained on: L_price only
    
    Args:
        policy_net: Policy network
        price_net: Price network
        params: Economic parameters
        optimizer_policy: Optimizer for policy
        optimizer_price: Optimizer for price
        T: Rollout horizon
        batch_size: Batch size
        lambda_price: Weight on price loss for policy
        smoothing: Default smoothing schedule
        seed: Random seed
    
    Reference: outline_v2.md lines 332-352
    """
    
    def __init__(
        self,
        policy_net: RiskyPolicyNetwork,
        price_net: RiskyPriceNetwork,
        params,
        optimizer_policy: Optional[tf.keras.optimizers.Optimizer] = None,
        optimizer_price: Optional[tf.keras.optimizers.Optimizer] = None,
        T: int = 32,
        batch_size: int = 128,
        lambda_price: float = 1.0,
        smoothing: Optional[DefaultSmoothingSchedule] = None,
        seed: Optional[int] = None
    ):
        self.policy_net = policy_net
        self.price_net = price_net
        self.params = params
        self.optimizer_policy = optimizer_policy or tf.keras.optimizers.Adam(1e-3)
        self.optimizer_price = optimizer_price or tf.keras.optimizers.Adam(1e-3)
        self.T = T
        self.batch_size = batch_size
        self.lambda_price = lambda_price
        self.smoothing = smoothing or DefaultSmoothingSchedule()
        self._rng = np.random.default_rng(seed)
        
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        # Note: LR method doesn't use value network, but price loss uses p^D
        # which needs V_tilde. For pure LR, we'd need a separate value network
        # or use a simpler pricing approach.
        # For now, we implement price loss using a temporary value estimate.
        self._value_net: Optional[RiskyValueNetwork] = None
    
    def set_value_net(self, value_net: RiskyValueNetwork):
        """Set value network for computing p^D in price loss."""
        self._value_net = value_net
    
    def train_step(
        self,
        k_init: tf.Tensor,
        b_init: tf.Tensor,
        z_init: tf.Tensor
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        1. Simulate T-step rollout to compute L_LR
        2. Compute L_price
        3. Update policy on L_LR + lambda_price * L_price
        4. Update price on L_price only
        """
        n = self.batch_size
        utilities_list = []
        
        # --- Rollout for LR ---
        with tf.GradientTape(persistent=True) as tape:
            k = tf.reshape(k_init, [-1, 1])
            b = tf.reshape(b_init, [-1, 1])
            z = tf.reshape(z_init, [-1, 1])
            
            for t in range(self.T):
                # Policy
                k_next, b_next = self.policy_net(k, b, z)
                
                # Price
                r_tilde = self.price_net(k_next, b_next, z)
                
                # Cash flow - delegated to economy
                e = cash_flow_risky_debt(k, k_next, b, b_next, z, r_tilde, self.params)
                eta = external_financing_cost(e, self.params)
                utility = e - eta
                utilities_list.append(utility)
                
                # Transition - use canonical AR(1) implementation
                z = step_ar1_tf(z, self.params.rho, self.params.sigma, self.params.mu)
                k = k_next
                b = b_next
            
            # LR component
            utilities = tf.concat(utilities_list, axis=1)
            loss_lr = compute_lr_loss(utilities, self.beta)
            
            # --- Price Loss (AiO) ---
            # Use terminal state for price loss computation
            k_term = k
            b_term = b
            z_term = z
            
            k_next_term, b_next_term = self.policy_net(k_term, b_term, z_term)
            r_tilde_term = self.price_net(k_next_term, b_next_term, z_term)
            
            # Two z' draws
            z_next_1, z_next_2 = draw_shocks(
                n, z_term, self.params.rho, self.params.sigma, self.params.mu
            )
            
            # Default probability from smooth formula
            if self._value_net is not None:
                V_tilde_1 = self._value_net(k_next_term, b_next_term, z_next_1)
                V_tilde_2 = self._value_net(k_next_term, b_next_term, z_next_2)
                p_D_1 = self.smoothing.compute_default_prob(V_tilde_1)
                p_D_2 = self.smoothing.compute_default_prob(V_tilde_2)
            else:
                # Fall back to zero default probability if no value net
                p_D_1 = tf.zeros_like(k_next_term)
                p_D_2 = tf.zeros_like(k_next_term)
            
            # Recovery - delegated to economy
            R_1 = recovery_value(k_next_term, z_next_1, self.params)
            R_2 = recovery_value(k_next_term, z_next_2, self.params)
            
            # Price residuals - delegated to economy
            f_price_1 = pricing_residual_zero_profit(
                b_next_term, self.params.r_rate, r_tilde_term, p_D_1, R_1
            )
            f_price_2 = pricing_residual_zero_profit(
                b_next_term, self.params.r_rate, r_tilde_term, p_D_2, R_2
            )
            
            loss_price = compute_price_loss_aio(f_price_1, f_price_2)
            
            # Combined policy loss
            loss_policy = loss_lr + self.lambda_price * loss_price
        
        # Update policy
        grads_policy = tape.gradient(loss_policy, self.policy_net.trainable_variables)
        self.optimizer_policy.apply_gradients(
            zip(grads_policy, self.policy_net.trainable_variables)
        )
        
        # Update price on L_price only
        grads_price = tape.gradient(loss_price, self.price_net.trainable_variables)
        self.optimizer_price.apply_gradients(
            zip(grads_price, self.price_net.trainable_variables)
        )
        
        del tape
        
        return {
            "loss_lr": float(loss_lr),
            "loss_price": float(loss_price),
            "mean_utility": float(tf.reduce_mean(utilities))
        }


# =============================================================================
# BELLMAN RESIDUAL TRAINER FOR RISKY DEBT (Sec. 2.4, 2.5)
# =============================================================================

class RiskyDebtTrainerBR:
    """
    Risky Debt BR Training (Sec. 2.4, 2.5).
    
    Outer loop per iteration:
    1. Sample batch with two shocks
    2. Forward pass: policy -> (k', b'), price -> r_tilde
    3. Critic/Price update:
       - L_critic = L_BR_critic + lambda_1 * L_price
       - Update theta_value, theta_price
       - stop_gradient on (k', b') so grad wrt theta_policy = 0
    4. Actor update:
       - L_actor = L_BR_actor + lambda_2 * L_price
       - Update theta_policy only
       - Freeze theta_value, theta_price (no optimizer update)
       - Do NOT detach V/price outputs - gradients flow through (k', b')
    5. Optionally update smoothing schedule
    
    Args:
        policy_net: Policy network
        value_net: Value network
        price_net: Price network
        params: Economic parameters
        optimizer_policy: Optimizer for policy
        optimizer_critic: Optimizer for value and price
        batch_size: Batch size
        lambda_1: Price weight for critic objective
        lambda_2: Price weight for actor objective
        n_critic_steps: Critic updates per actor update
        smoothing: Default smoothing schedule
        seed: Random seed
        leaky_actor: If True, use leaky ReLU in actor step (default False for economic correctness)
        collect_diagnostics: If True, collect gradient/saturation diagnostics
    
    Reference: outline_v2.md lines 377-405
    """
    
    def __init__(
        self,
        policy_net: RiskyPolicyNetwork,
        value_net: RiskyValueNetwork,
        price_net: RiskyPriceNetwork,
        params,
        optimizer_policy: Optional[tf.keras.optimizers.Optimizer] = None,
        optimizer_critic: Optional[tf.keras.optimizers.Optimizer] = None,
        batch_size: int = 128,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        n_critic_steps: int = 1,
        smoothing: Optional[DefaultSmoothingSchedule] = None,
        seed: Optional[int] = None,
        leaky_actor: bool = False,
        collect_diagnostics: bool = False
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.price_net = price_net
        self.params = params
        self.optimizer_policy = optimizer_policy or tf.keras.optimizers.Adam(1e-3)
        self.optimizer_critic = optimizer_critic or tf.keras.optimizers.Adam(1e-3)
        self.batch_size = batch_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_critic_steps = n_critic_steps
        self.smoothing = smoothing or DefaultSmoothingSchedule()
        self._rng = np.random.default_rng(seed)
        
        self.beta = 1.0 / (1.0 + params.r_rate)
        
        # Gradient flow options
        self.leaky_actor = leaky_actor
        self.collect_diagnostics = collect_diagnostics
        self._last_diagnostics: Optional[TrainingDiagnostics] = None
    
    @property
    def last_diagnostics(self) -> Optional[TrainingDiagnostics]:
        """Get diagnostics from last train_step (if collect_diagnostics=True)."""
        return self._last_diagnostics

    
    def _critic_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform one critic/price update.
        
        Gradient rules:
        - stop_gradient on (k', b') so no grad wrt theta_policy
        - stop_gradient on RHS continuation term
        
        Returns:
            (loss_br_critic, loss_price)
        """
        n = tf.shape(k)[0]
        
        with tf.GradientTape() as tape:
            # Policy output (DETACHED for critic block)
            k_next, b_next = self.policy_net(k, b, z)
            k_next_sg = tf.stop_gradient(k_next)
            b_next_sg = tf.stop_gradient(b_next)
            
            # Price
            r_tilde = self.price_net(k_next_sg, b_next_sg, z)
            
            # Cash flow (uses detached policy outputs) - delegated to economy
            e = cash_flow_risky_debt(
                k, k_next_sg, b, b_next_sg, z, r_tilde, self.params
            )
            eta = external_financing_cost(e, self.params)
            
            # Two z' draws
            z_next_1, z_next_2 = draw_shocks(
                n, z, self.params.rho, self.params.sigma, self.params.mu
            )
            
            # --- BR Critic Loss ---
            # Continuation values (DETACHED - RHS target)
            V_tilde_next_1 = self.value_net(k_next_sg, b_next_sg, z_next_1)
            V_tilde_next_2 = self.value_net(k_next_sg, b_next_sg, z_next_2)
            
            # Apply limited liability
            V_next_1 = apply_limited_liability(V_tilde_next_1)
            V_next_2 = apply_limited_liability(V_tilde_next_2)
            
            # Detach for RHS target
            V_next_1_sg = tf.stop_gradient(V_next_1)
            V_next_2_sg = tf.stop_gradient(V_next_2)
            
            # Targets
            y1 = e - eta + self.beta * V_next_1_sg
            y2 = e - eta + self.beta * V_next_2_sg
            
            # Current latent value (trainable LHS)
            V_tilde_curr = self.value_net(k, b, z)
            
            loss_br_critic = compute_br_critic_loss_aio(V_tilde_curr, y1, y2)
            
            # --- Price Loss ---
            # Default probabilities
            p_D_1 = self.smoothing.compute_default_prob(V_tilde_next_1)
            p_D_2 = self.smoothing.compute_default_prob(V_tilde_next_2)
            
            # Recovery - delegated to economy
            R_1 = recovery_value(k_next_sg, z_next_1, self.params)
            R_2 = recovery_value(k_next_sg, z_next_2, self.params)
            
            # Price residuals - delegated to economy
            f_price_1 = pricing_residual_zero_profit(
                b_next_sg, self.params.r_rate, r_tilde, p_D_1, R_1
            )
            f_price_2 = pricing_residual_zero_profit(
                b_next_sg, self.params.r_rate, r_tilde, p_D_2, R_2
            )
            loss_price = compute_price_loss_aio(f_price_1, f_price_2)
            
            # Combined critic objective
            loss_critic = compute_critic_objective(loss_br_critic, loss_price, self.lambda_1)
        
        # Update value and price networks
        critic_vars = (
            self.value_net.trainable_variables + 
            self.price_net.trainable_variables
        )
        grads = tape.gradient(loss_critic, critic_vars)
        self.optimizer_critic.apply_gradients(zip(grads, critic_vars))
        
        return loss_br_critic, loss_price
    
    def _actor_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[TrainingDiagnostics]]:
        """
        Perform one actor update.
        
        Gradient rules:
        - Freeze theta_value, theta_price (no optimizer update)
        - Do NOT detach V/price outputs - gradients flow through (k', b')
        
        Returns:
            (loss_br_actor, loss_price, diagnostics)
        """
        n = tf.shape(k)[0]
        diagnostics = None
        
        with tf.GradientTape() as tape:
            # Policy output (NOT detached - gradients flow)
            k_next, b_next = self.policy_net(k, b, z)
            
            # Price (gradients flow through k', b')
            r_tilde = self.price_net(k_next, b_next, z)
            
            # Cash flow - delegated to economy
            e = cash_flow_risky_debt(
                k, k_next, b, b_next, z, r_tilde, self.params
            )
            eta = external_financing_cost(e, self.params)
            
            # Two z' draws
            z_next_1, z_next_2 = draw_shocks(
                n, z, self.params.rho, self.params.sigma, self.params.mu
            )
            
            # --- BR Actor Loss (averaged continuation, not cross-product) ---
            V_tilde_next_1 = self.value_net(k_next, b_next, z_next_1)
            V_tilde_next_2 = self.value_net(k_next, b_next, z_next_2)
            
            # Apply limited liability (optionally leaky for gradient flow)
            V_next_1 = apply_limited_liability(V_tilde_next_1, leaky=self.leaky_actor)
            V_next_2 = apply_limited_liability(V_tilde_next_2, leaky=self.leaky_actor)
            
            loss_br_actor = compute_br_actor_loss_risky(
                e, eta, V_next_1, V_next_2, self.beta
            )
            
            # --- Price Loss ---
            p_D_1 = self.smoothing.compute_default_prob(V_tilde_next_1)
            p_D_2 = self.smoothing.compute_default_prob(V_tilde_next_2)
            
            # Recovery - delegated to economy
            R_1 = recovery_value(k_next, z_next_1, self.params)
            R_2 = recovery_value(k_next, z_next_2, self.params)
            
            # Price residuals - delegated to economy
            f_price_1 = pricing_residual_zero_profit(
                b_next, self.params.r_rate, r_tilde, p_D_1, R_1
            )
            f_price_2 = pricing_residual_zero_profit(
                b_next, self.params.r_rate, r_tilde, p_D_2, R_2
            )
            loss_price = compute_price_loss_aio(f_price_1, f_price_2)
            
            # Combined actor objective
            loss_actor = compute_actor_objective(loss_br_actor, loss_price, self.lambda_2)
        
        # Compute gradients
        grads = tape.gradient(loss_actor, self.policy_net.trainable_variables)
        
        # Collect diagnostics before applying gradients
        if self.collect_diagnostics:
            # Compute V_tilde stats
            V_tilde_all = tf.concat([V_tilde_next_1, V_tilde_next_2], axis=0)
            share_negative = float(tf.reduce_mean(tf.cast(V_tilde_all < 0, tf.float32)))
            mean_v_tilde = float(tf.reduce_mean(V_tilde_all))
            
            # Compute leverage stats
            safe_k_next = tf.maximum(k_next, 1e-8)
            leverage = b_next / safe_k_next
            mean_leverage = float(tf.reduce_mean(leverage))
            
            # Gradient norm
            grad_norm = _compute_grad_norm(grads)
            
            diagnostics = TrainingDiagnostics(
                grad_norm_policy=grad_norm,
                share_v_tilde_negative=share_negative,
                mean_v_tilde=mean_v_tilde,
                mean_leverage=mean_leverage,
            )
        
        # Apply gradients
        self.optimizer_policy.apply_gradients(
            zip(grads, self.policy_net.trainable_variables)
        )
        
        return loss_br_actor, loss_price, diagnostics
    
    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        update_smoothing: bool = True
    ) -> Dict[str, float]:
        """
        Perform one outer training step.
        
        1. N_critic critic/price updates
        2. 1 actor update
        3. Optionally update smoothing schedule
        
        Args:
            k: Current capital (batch_size,)
            b: Current debt (batch_size,)
            z: Current productivity (batch_size,)
            update_smoothing: Whether to update epsilon_D
        
        Returns:
            Dictionary with losses (and diagnostics if collect_diagnostics=True)
        """
        k = tf.reshape(k, [-1, 1])
        b = tf.reshape(b, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        
        # Critic updates
        critic_losses = []
        price_losses_critic = []
        for _ in range(self.n_critic_steps):
            loss_c, loss_p = self._critic_step(k, b, z)
            critic_losses.append(float(loss_c))
            price_losses_critic.append(float(loss_p))
        
        # Actor update
        loss_a, loss_p_a, diagnostics = self._actor_step(k, b, z)
        
        # Store last diagnostics
        if diagnostics is not None:
            self._last_diagnostics = diagnostics
        
        # Update smoothing schedule
        if update_smoothing:
            self.smoothing.update()
        
        result = {
            "loss_critic": np.mean(critic_losses),
            "loss_actor": float(loss_a),
            "loss_price_critic": np.mean(price_losses_critic),
            "loss_price_actor": float(loss_p_a),
            "epsilon_D": self.smoothing.epsilon_D
        }
        
        # Add diagnostics to result if collecting
        if diagnostics is not None:
            result["grad_norm_policy"] = diagnostics.grad_norm_policy
            result["share_v_tilde_negative"] = diagnostics.share_v_tilde_negative
            result["mean_v_tilde"] = diagnostics.mean_v_tilde
            result["mean_leverage"] = diagnostics.mean_leverage
        
        return result


# =============================================================================
# HELPER: Compute loss function for LR
# =============================================================================

def compute_lr_loss(utilities: tf.Tensor, beta: float) -> tf.Tensor:
    """Wrapper matching signature from losses.py for utilities."""
    T = tf.shape(utilities)[1]
    t_indices = tf.cast(tf.range(T), tf.float32)
    discount_factors = tf.pow(beta, t_indices)
    discounted = utilities * discount_factors[None, :]
    total = tf.reduce_sum(discounted, axis=1)
    return -tf.reduce_mean(total)
