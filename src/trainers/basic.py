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
from src.economy.logic import compute_cash_flow_basic, euler_chi, euler_m, compute_terminal_value

from src.networks.network_basic import BasicPolicyNetwork, BasicValueNetwork, build_basic_networks
from src.trainers.losses import (
    compute_lr_loss,
    compute_er_loss_aio,
    compute_br_critic_loss_aio,
    compute_br_critic_diagnostics,
    compute_br_actor_loss
)
from src.trainers.config import NetworkConfig, OptimizationConfig, AnnealingConfig, MethodConfig, EarlyStoppingConfig
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

        Includes terminal value correction for finite horizon truncation.

        Reference:
            report_brief.md lines 499-514: LR loss with terminal value

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

            # Compute terminal value correction (report lines 503-514)
            # V^term(k_T, z_T) = e(k_SS, k_SS, z_T) / (1 - β)
            z_terminal = tf.reshape(z_path[:, self.T], [-1, 1])
            v_terminal = compute_terminal_value(
                k, z_terminal, self.params, self.beta,
                temperature=temperature,
                logit_clip=self.logit_clip
            )

            # Loss includes terminal value weighted by β^T
            loss = compute_lr_loss(rewards, self.beta, terminal_value=v_terminal)

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        terminal_states = tf.concat([k, z_terminal], axis=1).numpy()
        return {
            "loss_LR": float(loss),
            "mean_reward": float(tf.reduce_mean(rewards)),
            "terminal_states": terminal_states
        }

    def evaluate(
        self,
        k: tf.Tensor,
        z_path: tf.Tensor,
        temperature: float,
    ) -> Dict[str, float]:
        """
        Evaluate on data without updating weights.

        Used for validation set evaluation during early stopping.

        Args:
            k (tf.Tensor): Initial capital batch. Shape: (Batch, 1).
            z_path (tf.Tensor): Productivity trajectory. Shape: (Batch, T+1).
            temperature (float): Gate temperature.

        Returns:
            Dict[str, float]: Loss and metrics.
        """
        rewards_list = []
        k = tf.reshape(k, [-1, 1])

        for t in range(self.T):
            z_curr = tf.reshape(z_path[:, t], [-1, 1])
            k_next = self.policy_net(k, z_curr)
            reward = compute_cash_flow_basic(k, k_next, z_curr, self.params, temperature=temperature, logit_clip=self.logit_clip)
            rewards_list.append(reward)
            k = k_next

        rewards = tf.concat(rewards_list, axis=1)

        # Compute terminal value
        z_terminal = tf.reshape(z_path[:, self.T], [-1, 1])
        v_terminal = compute_terminal_value(
            k, z_terminal, self.params, self.beta,
            temperature=temperature,
            logit_clip=self.logit_clip
        )

        loss = compute_lr_loss(rewards, self.beta, terminal_value=v_terminal)

        return {
            "loss_LR": float(loss),
            "mean_reward": float(tf.reduce_mean(rewards)),
        }


class BasicTrainerER:
    """
    Method 2: Euler Residual Minimization for Basic model.

    Uses flattened i.i.d. transitions and target policy network for stability.
    Implements the algorithm from report_brief.md lines 486-520.

    Key Features:
    - Operates on individual transitions (k, z) -> (k', z') from flattened dataset
    - Uses target policy network for computing k'' (two-step lookahead)
    - Polyak averaging to update target network
    - No time loops - batch-based one-step optimization

    Reference:
        report_brief.md lines 486-520: "Algorithm Summary: ER Method"
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        polyak_tau: float = 0.995  # Polyak averaging coefficient
    ):
        self.policy_net = policy_net
        self.params = params
        self.shock_params = shock_params
        self.optimizer = optimizer or tf.keras.optimizers.Adam(1e-3)
        self.beta = 1.0 / (1.0 + params.r_rate)
        self.polyak_tau = polyak_tau

        # Create target policy network (frozen copy of policy_net)
        # Reference: report_brief.md line 491 "Initiate target policy"
        self.target_policy_net = tf.keras.models.clone_model(policy_net)

        # Build target network by calling it with dummy data
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        _ = self.target_policy_net(dummy_k, dummy_z)

        # Now set weights from source network
        self.target_policy_net.set_weights(policy_net.get_weights())

        if params.cost_fixed > 0:
            logger.warning("ER with cost_fixed > 0: FOC undefined at I=0. Approximated with smooth function.")

    def _update_target_network(self) -> float:
        """
        Update target policy network using Polyak averaging.

        θ⁻_policy ← ν θ⁻_policy + (1-ν) θ_policy

        Reference: report_brief.md line 519 "Polyak Averaging to update target policy"

        Returns:
            Average magnitude of parameter updates (for monitoring)
        """
        update_magnitudes = []

        for target_var, source_var in zip(
            self.target_policy_net.trainable_variables,
            self.policy_net.trainable_variables
        ):
            # Compute update magnitude before applying
            delta = (1.0 - self.polyak_tau) * (source_var - target_var)
            update_mag = tf.reduce_mean(tf.abs(delta))
            update_magnitudes.append(float(update_mag))

            # Apply Polyak update
            target_var.assign(
                self.polyak_tau * target_var + (1.0 - self.polyak_tau) * source_var
            )

        # Return average update magnitude
        return float(np.mean(update_magnitudes))

    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor
    ) -> Dict[str, float]:
        """
        Execute one training step via Euler Residual Minimization.

        Uses flattened i.i.d. transitions with All-in-One (AiO) estimator.
        Computes two-step lookahead k'' using TARGET policy for stability.

        Algorithm (report_brief.md lines 494-519):
        1. Current Step (Trainable): k' = π(k, z; θ) and χ = 1 + ψ_I(I, k)
        2. Future Step (Target): k'' = π(k', z'; θ⁻) for both forks
        3. Compute residuals: f = χ - β * m where m = π_k - ψ_k + (1-δ)χ'
        4. AiO Loss: L_ER = mean(f_main * f_fork)
        5. Update θ and then update θ⁻ with Polyak averaging

        Args:
            k: Current capital (batch_size,) - independent samples from ergodic distribution
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)

        Returns:
            Dict with metrics:
                - loss_ER: Euler residual loss (AiO form)
                - target_policy_update: Average magnitude of target network updates

        Reference:
            report_brief.md lines 494-519: ER training loop with target policy
        """
        with tf.GradientTape() as tape:
            # Reshape inputs
            k = tf.reshape(k, [-1, 1])
            z = tf.reshape(z, [-1, 1])
            z_next_main = tf.reshape(z_next_main, [-1, 1])
            z_next_fork = tf.reshape(z_next_fork, [-1, 1])

            # === CURRENT STEP (Trainable) ===
            # Compute k' using current policy
            k_next = self.policy_net(k, z)

            # Compute Euler LHS: χ = 1 + ψ_I(I, k)
            chi = euler_chi(k, k_next, self.params)

            # === FUTURE STEP - MAIN FORK (Target Policy) ===
            # Compute k'' using TARGET policy for stability (DDPG-style)
            k_next_next_main = self.target_policy_net(k_next, z_next_main)
            m_main = euler_m(k_next, k_next_next_main, z_next_main, self.params)
            f_main = chi - self.beta * m_main

            # === FUTURE STEP - FORK PATH (Target Policy) ===
            k_next_next_fork = self.target_policy_net(k_next, z_next_fork)
            m_fork = euler_m(k_next, k_next_next_fork, z_next_fork, self.params)
            f_fork = chi - self.beta * m_fork

            # === AiO LOSS ===
            # L_ER = mean(f_main * f_fork)
            # Reference: report_brief.md line 516
            loss = compute_er_loss_aio(f_main, f_fork)

        # Update current policy
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        if grads[0] is not None:
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        # Update target policy with Polyak averaging
        # Reference: report_brief.md line 519
        target_update_mag = self._update_target_network()

        return {
            "loss_ER": float(loss),
            "target_policy_update": target_update_mag
        }

    def evaluate(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate on data without updating weights.

        Used for validation set evaluation during early stopping.

        Args:
            k: Current capital (batch_size,)
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)

        Returns:
            Dict with loss_ER metric.
        """
        # Reshape inputs
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        # Current step
        k_next = self.policy_net(k, z)
        chi = euler_chi(k, k_next, self.params)

        # Future step (target policy)
        k_next_next_main = self.target_policy_net(k_next, z_next_main)
        m_main = euler_m(k_next, k_next_next_main, z_next_main, self.params)
        f_main = chi - self.beta * m_main

        k_next_next_fork = self.target_policy_net(k_next, z_next_fork)
        m_fork = euler_m(k_next, k_next_next_fork, z_next_fork, self.params)
        f_fork = chi - self.beta * m_fork

        loss = compute_er_loss_aio(f_main, f_fork)

        return {"loss_ER": float(loss)}


class BasicTrainerBR:
    """
    Method 3: Bellman Residual (Actor-Critic) for Basic model.

    Uses flattened i.i.d. transitions with target networks for stability (DDPG-style).
    Implements the algorithm from report_brief.md lines 585-644.

    Key Features:
    - Operates on individual transitions (k, z) -> (k', z') from flattened dataset
    - Target policy network for computing actions in critic targets
    - Target value network for computing continuation values in critic targets
    - Polyak averaging to update both target networks
    - No time loops - batch-based one-step optimization
    - Critic trains value function, Actor trains policy

    Architecture (DDPG-style):
    - Critic uses TARGET networks for computing targets (stability)
    - Actor uses CURRENT networks but freezes value weights during update

    Reference:
        report_brief.md lines 585-644: "Algorithm Summary: BR Method"
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
        logit_clip: float = 20.0,
        polyak_tau: float = 0.995  # Polyak averaging coefficient
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
        self.polyak_tau = polyak_tau

        # Create target networks (frozen copies)
        # Reference: report_brief.md line 590 "Initiate target networks"
        self.target_policy_net = tf.keras.models.clone_model(policy_net)
        self.target_value_net = tf.keras.models.clone_model(value_net)

        # Build target networks by calling them with dummy data
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        _ = self.target_policy_net(dummy_k, dummy_z)
        _ = self.target_value_net(dummy_k, dummy_z)

        # Now set weights from source networks
        self.target_policy_net.set_weights(policy_net.get_weights())
        self.target_value_net.set_weights(value_net.get_weights())

    def _update_target_value(self) -> float:
        """
        Update target value network using Polyak averaging.

        θ⁻_value ← ν θ⁻_value + (1-ν) θ_value

        Reference:
            report_brief.md line 614: "Update Target Value"

        Returns:
            Average magnitude of parameter updates (for monitoring)
        """
        update_magnitudes = []

        for target_var, source_var in zip(
            self.target_value_net.trainable_variables,
            self.value_net.trainable_variables
        ):
            # Compute update magnitude before applying
            delta = (1.0 - self.polyak_tau) * (source_var - target_var)
            update_mag = tf.reduce_mean(tf.abs(delta))
            update_magnitudes.append(float(update_mag))

            # Apply Polyak update
            target_var.assign(
                self.polyak_tau * target_var + (1.0 - self.polyak_tau) * source_var
            )

        return float(np.mean(update_magnitudes))

    def _update_target_policy(self) -> float:
        """
        Update target policy network using Polyak averaging.

        θ⁻_policy ← ν θ⁻_policy + (1-ν) θ_policy

        Reference:
            report_brief.md line 633: "Update Target Policy"

        Returns:
            Average magnitude of parameter updates (for monitoring)
        """
        update_magnitudes = []

        for target_var, source_var in zip(
            self.target_policy_net.trainable_variables,
            self.policy_net.trainable_variables
        ):
            # Compute update magnitude before applying
            delta = (1.0 - self.polyak_tau) * (source_var - target_var)
            update_mag = tf.reduce_mean(tf.abs(delta))
            update_magnitudes.append(float(update_mag))

            # Apply Polyak update
            target_var.assign(
                self.polyak_tau * target_var + (1.0 - self.polyak_tau) * source_var
            )

        return float(np.mean(update_magnitudes))

    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float
    ) -> Dict[str, float]:
        """
        Execute one training step via Bellman Residual Minimization (Actor-Critic).

        Uses flattened i.i.d. transitions with All-in-One (AiO) estimator.
        Implements DDPG-style updates with target networks.

        Algorithm (report_brief.md lines 593-643):
        A. Critic Update (repeat n_critic_steps times):
           1. Compute action using TARGET policy: k' = π(k, z; θ⁻_policy)
           2. Compute continuation values using TARGET value network
           3. Compute critic targets (detached): y = e + β * V_target
           4. AiO Loss: L_critic = mean((V_curr - y_1) * (V_curr - y_2))
           5. Update value network θ_value
           6. Update target value network with Polyak averaging

        B. Actor Update (once per train_step):
           1. Compute action using CURRENT policy: k' = π(k, z; θ_policy)
           2. Compute continuation value using CURRENT value network (freeze weights)
           3. Actor Loss: L_actor = -mean(e + β * V_next)
           4. Update policy network θ_policy
           5. Update target policy network with Polyak averaging

        Args:
            k: Current capital (batch_size,) - independent samples
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)
            temperature: Annealing temperature for smooth approximations

        Returns:
            Dict with metrics:
                - loss_critic: Critic loss (last of n_critic_steps)
                - loss_actor: Actor loss
                - mse_proxy: Diagnostic MSE proxy for critic
                - target_value_update: Average magnitude of target value network updates
                - target_policy_update: Average magnitude of target policy network updates

        Reference:
            report_brief.md lines 593-643: BR training loop with target networks
        """
        # Reshape inputs
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        # ===================================================================
        # A. CRITIC UPDATE (Multiple steps per actor update)
        # ===================================================================
        critic_losses = []
        mses = []

        for critic_step in range(self.n_critic_steps):
            with tf.GradientTape() as tape_critic:
                # === Compute Action using TARGET Policy ===
                # Reference: report_brief.md line 600
                # "Compute next action using the Target Policy"
                k_next = self.target_policy_net(k, z)

                # === Compute Continuation Values using TARGET Value Network ===
                # Reference: report_brief.md line 602-603
                V_next_main = self.target_value_net(k_next, z_next_main)
                V_next_fork = self.target_value_net(k_next, z_next_fork)

                # === Compute Critic Targets (Detached) ===
                # Reference: report_brief.md line 605-606
                # "Compute the fixed Bellman target (detach gradients)"
                e = compute_cash_flow_basic(
                    k, k_next, z,
                    self.params,
                    temperature=temperature,
                    logit_clip=self.logit_clip
                )

                y1 = tf.stop_gradient(e + self.beta * V_next_main)
                y2 = tf.stop_gradient(e + self.beta * V_next_fork)

                # === Compute Current Value (Trainable) ===
                # Reference: report_brief.md line 609
                V_curr = self.value_net(k, z)

                # === AiO Critic Loss ===
                # Reference: report_brief.md line 611-612
                # "Calculate residuals and compute AiO Loss"
                loss_critic = compute_br_critic_loss_aio(V_curr, y1, y2)

            # Update critic (value network)
            # Reference: report_brief.md line 613
            grads_critic = tape_critic.gradient(loss_critic, self.value_net.trainable_variables)
            self.optimizer_value.apply_gradients(zip(grads_critic, self.value_net.trainable_variables))

            # Track metrics
            critic_losses.append(float(loss_critic))

            # Compute diagnostic on last iteration
            if critic_step == self.n_critic_steps - 1:
                diagnostics = compute_br_critic_diagnostics(V_curr, y1, y2)
                mses.append(diagnostics["mse_proxy"])

        # Update target value network ONCE per train_step (industry standard)
        # Reference: report_brief.md line 614
        # Optimization: Update once instead of n_critic_steps times
        target_value_update = self._update_target_value()

        # ===================================================================
        # B. ACTOR UPDATE (Once per train_step)
        # ===================================================================
        with tf.GradientTape() as tape_actor:
            # === Compute Action using CURRENT Policy (gradients flow) ===
            # Reference: report_brief.md line 622-623
            # "Compute next action using the Current Policy"
            k_next = self.policy_net(k, z)

            # === Compute Continuation Value using CURRENT Value Network ===
            # Reference: report_brief.md line 626-627
            # "Predict continuation value using the Current Value network"
            # Note: We freeze value weights by not including them in gradients
            V_next_main = self.value_net(k_next, z_next_main)

            # === Compute Cash Flow ===
            e = compute_cash_flow_basic(
                k, k_next, z,
                self.params,
                temperature=temperature,
                logit_clip=self.logit_clip
            )

            # === Actor Loss ===
            # Reference: report_brief.md line 628-629
            # "Define Loss (negative expected value of Bellman RHS)"
            # Note: We use only main fork for actor (not AiO form)
            loss_actor = -tf.reduce_mean(e + self.beta * V_next_main)

        # Update actor (policy network)
        # Reference: report_brief.md line 632
        grads_actor = tape_actor.gradient(loss_actor, self.policy_net.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_actor, self.policy_net.trainable_variables))

        # Update target policy network with Polyak averaging
        # Reference: report_brief.md line 633
        target_policy_update = self._update_target_policy()

        return {
            "loss_critic": float(np.mean(critic_losses)),
            "loss_actor": float(loss_actor),
            "mse_proxy": float(np.mean(mses)) if mses else 0.0,
            "target_value_update": target_value_update,
            "target_policy_update": target_policy_update
        }

    def evaluate(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float
    ) -> Dict[str, float]:
        """
        Evaluate on data without updating weights.

        Used for validation set evaluation during early stopping.

        Args:
            k: Current capital (batch_size,)
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)
            temperature: Annealing temperature

        Returns:
            Dict with loss_critic and loss_actor metrics.
        """
        # Reshape inputs
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        # Critic evaluation (using target networks)
        k_next = self.target_policy_net(k, z)
        V_next_main = self.target_value_net(k_next, z_next_main)
        V_next_fork = self.target_value_net(k_next, z_next_fork)

        e = compute_cash_flow_basic(k, k_next, z, self.params, temperature=temperature, logit_clip=self.logit_clip)

        y1 = e + self.beta * V_next_main
        y2 = e + self.beta * V_next_fork

        V_curr = self.value_net(k, z)
        loss_critic = compute_br_critic_loss_aio(V_curr, y1, y2)

        # Actor evaluation (using current policy)
        k_next_actor = self.policy_net(k, z)
        V_next_actor = self.value_net(k_next_actor, z_next_main)
        e_actor = compute_cash_flow_basic(k, k_next_actor, z, self.params, temperature=temperature, logit_clip=self.logit_clip)
        loss_actor = -tf.reduce_mean(e_actor + self.beta * V_next_actor)

        return {
            "loss_critic": float(loss_critic),
            "loss_actor": float(loss_actor)
        }


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _make_validation_fn_lr(trainer: BasicTrainerLR):
    """Create validation function for LR method."""
    def validation_fn(trainer, batch, temperature):
        k = batch.get('k0', batch.get('k'))
        z_path = batch['z_path']
        return trainer.evaluate(k, z_path, temperature)
    return validation_fn


def _make_validation_fn_er(trainer: BasicTrainerER):
    """Create validation function for ER method."""
    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(
            batch['k'], batch['z'],
            batch['z_next_main'], batch['z_next_fork']
        )
    return validation_fn


def _make_validation_fn_br(trainer: BasicTrainerBR):
    """Create validation function for BR method."""
    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(
            batch['k'], batch['z'],
            batch['z_next_main'], batch['z_next_fork'],
            temperature
        )
    return validation_fn


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
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None
) -> Dict[str, Any]:
    """
    Train Basic Model using Lifetime Reward (LR).

    Args:
        dataset: Training dataset with keys 'k0', 'z_path'
        net_config: Network architecture configuration
        opt_config: Optimization configuration (includes early_stopping)
        method_config: Method-specific configuration
        anneal_config: Annealing configuration
        params: Economic parameters
        shock_params: Shock parameters
        bounds: State bounds dictionary
        validation_data: Optional validation dataset for early stopping

    Returns:
        Dict with history, trained networks, and configs
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

    # 3. Setup validation function (for early stopping)
    validation_fn = _make_validation_fn_lr(trainer) if validation_data is not None else None

    # 4. Execute Loop
    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="basic_lr",
        validation_data=validation_data,
        validation_fn=validation_fn
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
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None
) -> Dict[str, Any]:
    """
    Train Basic Model using Euler Residuals (ER).

    IMPORTANT: This method requires FLATTENED dataset format.
    Use DataGenerator.get_flattened_training_dataset() instead of get_training_dataset().

    The dataset must contain:
        - 'k': Current capital (N*T,) - independent samples
        - 'z': Current productivity (N*T,)
        - 'z_next_main': Next productivity, main fork (N*T,)
        - 'z_next_fork': Next productivity, second fork (N*T,)

    Reference:
        report_brief.md lines 157-167: "Flatten Data for ER and BR"
        report_brief.md lines 456-510: "ER Method"

    Args:
        dataset: FLATTENED training dataset from DataGenerator.get_flattened_training_dataset()
        net_config: Network architecture configuration
        opt_config: Optimization configuration
        method_config: Method-specific configuration
        anneal_config: Annealing configuration
        params: Economic parameters
        shock_params: Shock parameters
        bounds: State bounds dictionary
        validation_data: Optional FLATTENED validation dataset for early stopping

    Returns:
        Dict containing:
            - history: Training metrics over time
            - _policy_net: Trained policy network
            - _configs: Configuration objects
            - _params: Economic parameters
    """
    k_bounds = bounds['k']

    # Validate dataset format
    required_keys = {'k', 'z', 'z_next_main', 'z_next_fork'}
    if not required_keys.issubset(dataset.keys()):
        raise ValueError(
            f"ER method requires FLATTENED dataset with keys {required_keys}. "
            f"Got keys: {set(dataset.keys())}. "
            f"Use DataGenerator.get_flattened_training_dataset() instead of get_training_dataset()."
        )

    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k'].shape[0]).batch(opt_config.batch_size).repeat()
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
        optimizer=tf.keras.optimizers.Adam(opt_config.learning_rate),
        polyak_tau=method_config.polyak_tau if hasattr(method_config, 'polyak_tau') else 0.995
    )

    # Setup validation function (for early stopping)
    validation_fn = _make_validation_fn_er(trainer) if validation_data is not None else None

    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="basic_er",
        validation_data=validation_data,
        validation_fn=validation_fn
    )

    return {
        "history": history,
        "_policy_net": policy_net,
        "_target_policy_net": trainer.target_policy_net,
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
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None
) -> Dict[str, Any]:
    """
    Train Basic Model using Bellman Residuals (Actor-Critic).

    IMPORTANT: This method requires FLATTENED dataset format.
    Use DataGenerator.get_flattened_training_dataset() instead of get_training_dataset().

    The dataset must contain:
        - 'k': Current capital (N*T,) - independent samples
        - 'z': Current productivity (N*T,)
        - 'z_next_main': Next productivity, main fork (N*T,)
        - 'z_next_fork': Next productivity, second fork (N*T,)

    Implements DDPG-style actor-critic with target networks and Polyak averaging.

    Reference:
        report_brief.md lines 157-167: "Flatten Data for ER and BR"
        report_brief.md lines 585-644: "Algorithm Summary: BR Method"

    Args:
        dataset: FLATTENED training dataset from DataGenerator.get_flattened_training_dataset()
        net_config: Network architecture configuration
        opt_config: Optimization configuration
        method_config: Method-specific configuration (must include n_critic)
        anneal_config: Annealing configuration
        params: Economic parameters
        shock_params: Shock parameters
        bounds: State bounds dictionary
        validation_data: Optional FLATTENED validation dataset for early stopping

    Returns:
        Dict containing:
            - history: Training metrics over time
            - _policy_net: Trained policy network
            - _value_net: Trained value network
            - _target_policy_net: Target policy network
            - _target_value_net: Target value network
            - _configs: Configuration objects
            - _params: Economic parameters
    """
    k_bounds = bounds['k']

    # Validate dataset format
    required_keys = {'k', 'z', 'z_next_main', 'z_next_fork'}
    if not required_keys.issubset(dataset.keys()):
        raise ValueError(
            f"BR method requires FLATTENED dataset with keys {required_keys}. "
            f"Got keys: {set(dataset.keys())}. "
            f"Use DataGenerator.get_flattened_training_dataset() instead of get_training_dataset()."
        )

    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k'].shape[0]).batch(opt_config.batch_size).repeat()
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
        critic_lr=opt_config.learning_rate_critic if hasattr(opt_config, 'learning_rate_critic') and opt_config.learning_rate_critic else opt_config.learning_rate,
        n_critic_steps=method_config.n_critic,
        logit_clip=anneal_config.logit_clip,
        polyak_tau=method_config.polyak_tau if hasattr(method_config, 'polyak_tau') else 0.995
    )

    # Setup validation function (for early stopping)
    validation_fn = _make_validation_fn_br(trainer) if validation_data is not None else None

    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="basic_br",
        validation_data=validation_data,
        validation_fn=validation_fn
    )

    return {
        "history": history,
        "_policy_net": policy_net,
        "_value_net": value_net,
        "_target_policy_net": trainer.target_policy_net,
        "_target_value_net": trainer.target_value_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        },
        "_params": params
    }
