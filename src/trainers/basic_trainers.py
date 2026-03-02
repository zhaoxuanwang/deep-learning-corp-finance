"""
Trainer implementations for the Basic Model (Sec. 1).
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

from src.economy.parameters import EconomicParams, ShockParams
from src.economy.logic import compute_cash_flow_basic, euler_chi, euler_m, compute_terminal_value

from src.networks.network_basic import BasicPolicyNetwork, BasicValueNetwork
from src.trainers.io_transforms import (
    forward_basic_policy_levels,
    forward_basic_value_levels,
    compute_k_clip_diagnostics,
    build_legacy_basic_transform_spec_from_networks,
)
from src.trainers.losses import (
    compute_lr_loss,
    compute_er_loss_aio,
    compute_er_loss_huber,
    compute_er_loss_mse,
    compute_br_critic_loss_aio,
    compute_br_critic_loss_huber,
    compute_br_critic_loss_mse,
    compute_br_critic_diagnostics,
)

logger = logging.getLogger(__name__)


def _inference_clip_k(
    k_preclip: tf.Tensor,
    *,
    transform_spec: Dict[str, Any],
) -> tf.Tensor:
    cfg = transform_spec["outputs"]["policy_k"]
    out = k_preclip
    clip_min = cfg.get("clip_min")
    clip_max = cfg.get("clip_max")
    if clip_min is not None:
        out = tf.maximum(out, tf.constant(float(clip_min), dtype=out.dtype))
    if clip_max is not None:
        out = tf.minimum(out, tf.constant(float(clip_max), dtype=out.dtype))
    return out


def _ensure_finite_scalar(name: str, value: tf.Tensor) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(value))):
        raise FloatingPointError(f"Non-finite scalar encountered for '{name}'.")


def _ensure_finite_gradients(name: str, grads: List[Optional[tf.Tensor]]) -> None:
    bad_indices = []
    for idx, grad in enumerate(grads):
        if grad is None:
            continue
        if not bool(tf.reduce_all(tf.math.is_finite(grad))):
            bad_indices.append(idx)
    if bad_indices:
        raise FloatingPointError(
            f"Non-finite gradients encountered for '{name}' at indices {bad_indices}."
        )


# =============================================================================
# TRAINER CLASSES (Low-Level Logic)
# =============================================================================

class BasicTrainerLR:
    """
    Method 1: Lifetime Reward Maximization for Basic model.
    Trains policy network by maximizing expected discounted rewards.

    Note: Network and data operate in LEVELS. The network's internal
    normalization (bounded sigmoid) handles numerical stability.
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        T: int,
        logit_clip: float,
        optimizer: tf.keras.optimizers.Optimizer,
        transform_spec: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LR trainer.

        Args:
            policy_net: Policy network to train.
            params: Economic parameters.
            shock_params: Shock process parameters.
            T: Time horizon.
            logit_clip: Logit clipping bound for smooth indicators.
            optimizer: Configured optimizer (use create_optimizer() from config).
        """
        self.policy_net = policy_net
        self.transform_spec = transform_spec or build_legacy_basic_transform_spec_from_networks(
            policy_net=policy_net
        )
        self.params = params
        self.shock_params = shock_params
        self.optimizer = optimizer
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
        k_preclip_list = []
        with tf.GradientTape() as tape:
            k = tf.reshape(k, [-1, 1])
            # Use pre-computed path. z_path is (Batch, T+1)
            # This ensures determinism and uses the exact shock realizations from data generation.

            for t in range(self.T):
                z_curr = tf.reshape(z_path[:, t], [-1, 1])

                k_next, k_preclip = forward_basic_policy_levels(
                    policy_net=self.policy_net,
                    k=k,
                    z=z_curr,
                    transform_spec=self.transform_spec,
                    training=True,
                    apply_output_clips=False,
                    return_preclip=True,
                )
                k_preclip_list.append(k_preclip)

                # k and k_next are in LEVELS - pass directly to economic functions
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

        k_preclip_all = tf.concat(k_preclip_list, axis=0)
        k_postclip_all = _inference_clip_k(
            k_preclip_all,
            transform_spec=self.transform_spec,
        )
        clip_diag = compute_k_clip_diagnostics(
            k_postclip=k_postclip_all,
            k_preclip=k_preclip_all,
        )

        terminal_states = tf.concat([k, z_terminal], axis=1).numpy()
        return {
            "loss_LR": float(loss),
            "mean_reward": float(tf.reduce_mean(rewards)),
            "terminal_states": terminal_states,
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
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
        k_preclip_list = []
        k = tf.reshape(k, [-1, 1])

        for t in range(self.T):
            z_curr = tf.reshape(z_path[:, t], [-1, 1])
            k_next, k_preclip = forward_basic_policy_levels(
                policy_net=self.policy_net,
                k=k,
                z=z_curr,
                transform_spec=self.transform_spec,
                training=False,
                apply_output_clips=False,
                return_preclip=True,
            )
            k_preclip_list.append(k_preclip)

            # k and k_next are in LEVELS - pass directly to economic functions
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

        k_preclip_all = tf.concat(k_preclip_list, axis=0)
        k_postclip_all = _inference_clip_k(
            k_preclip_all,
            transform_spec=self.transform_spec,
        )
        clip_diag = compute_k_clip_diagnostics(
            k_postclip=k_postclip_all,
            k_preclip=k_preclip_all,
        )

        return {
            "loss_LR": float(loss),
            "mean_reward": float(tf.reduce_mean(rewards)),
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
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

    Note: Network and data operate in LEVELS. The network's internal
    normalization (bounded sigmoid) handles numerical stability.

    Reference:
        report_brief.md lines 486-520: "Algorithm Summary: ER Method"
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        optimizer: tf.keras.optimizers.Optimizer,
        polyak_tau: float,
        loss_type: str = "crossprod",
        transform_spec: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ER trainer.

        Args:
            policy_net: Policy network to train.
            params: Economic parameters.
            shock_params: Shock process parameters.
            optimizer: Configured optimizer (use create_optimizer() from config).
            polyak_tau: Polyak averaging coefficient for target network updates.
                Use MethodConfig.polyak_tau (default: DEFAULT_POLYAK_TAU = 0.995).
            loss_type: Loss computation method for Euler residual:
                - "mse": Mean Squared Error E[f²], biased but stable
                - "huber": Huber loss, robust to outlier residuals
                - "crossprod": AiO cross-product E[f₁·f₂], unbiased (default)
                Use MethodConfig.loss_type (default: "crossprod").
        """
        self.policy_net = policy_net
        self.transform_spec = transform_spec or build_legacy_basic_transform_spec_from_networks(
            policy_net=policy_net
        )
        self.params = params
        self.shock_params = shock_params
        self.optimizer = optimizer
        self.beta = 1.0 / (1.0 + params.r_rate)
        self.polyak_tau = polyak_tau
        self.loss_type = loss_type

        # Validate loss_type
        valid_loss_types = {"mse", "huber", "crossprod", "fork_mean_square"}
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{loss_type}'")

        # Create target policy network (frozen copy of policy_net)
        # Reference: report_brief.md line 491 "Initiate target policy"
        self.target_policy_net = tf.keras.models.clone_model(policy_net)

        # Build target network by calling it with normalized dummy features.
        dummy_x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        _ = self.target_policy_net(dummy_x, training=False)

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
            self.target_policy_net.variables,
            self.policy_net.variables
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

        Algorithm (report_brief.md lines 605-642):
        1. Current Step (Trainable): k' = π(k, z; θ) and χ = 1 + ψ_I(I, k)
        2. Future Step (Target): k'' = π(k', z'; θ⁻) for both forks
        3. Compute unit-free residuals: f = 1 - β * m / χ
           where m = π_k - ψ_k + (1-δ)χ'
        4. AiO Loss: L_ER = mean(f_main * f_fork)
        5. Update θ and then update θ⁻ with Polyak averaging

        Reference:
            report_brief.md lines 601-604: Unit-free Euler residual formula
            f_{i,ℓ} = 1 - β * m(k', k'', z') / (1 + ψ_I(I, k))

        Args:
            k: Current capital (batch_size,) - independent samples from ergodic distribution
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)

        Returns:
            Dict with metrics:
                - loss_ER: Euler residual loss (AiO form)
                - target_policy_update: Average magnitude of target network updates

        """
        with tf.GradientTape() as tape:
            # Reshape inputs
            k = tf.reshape(k, [-1, 1])
            z = tf.reshape(z, [-1, 1])
            z_next_main = tf.reshape(z_next_main, [-1, 1])
            z_next_fork = tf.reshape(z_next_fork, [-1, 1])

            # === CURRENT STEP (Trainable) ===
            # Compute k' using current policy
            k_next, k_preclip = forward_basic_policy_levels(
                policy_net=self.policy_net,
                k=k,
                z=z,
                transform_spec=self.transform_spec,
                training=True,
                apply_output_clips=False,
                return_preclip=True,
            )

            # k and k_next are in LEVELS - pass directly to Euler primitives
            chi = euler_chi(k, k_next, self.params)

            # === FUTURE STEP - MAIN FORK (Target Policy) ===
            # Compute k'' using TARGET policy for stability (DDPG-style)
            k_next_next_main = forward_basic_policy_levels(
                policy_net=self.target_policy_net,
                k=k_next,
                z=z_next_main,
                transform_spec=self.transform_spec,
                training=False,
                apply_output_clips=False,
            )
            m_main = euler_m(k_next, k_next_next_main, z_next_main, self.params)

            # === FUTURE STEP - FORK PATH (Target Policy) ===
            k_next_next_fork = forward_basic_policy_levels(
                policy_net=self.target_policy_net,
                k=k_next,
                z=z_next_fork,
                transform_spec=self.transform_spec,
                training=False,
                apply_output_clips=False,
            )
            m_fork = euler_m(k_next, k_next_next_fork, z_next_fork, self.params)

            # === UNIT-FREE EULER RESIDUALS ===
            # Reference: report_brief.md lines 601-604
            # f = 1 - β * m / χ (unit-free form)
            safe_chi = tf.maximum(chi, 1e-8)  # Avoid division by zero
            f_main = 1.0 - self.beta * m_main / safe_chi
            f_fork = 1.0 - self.beta * m_fork / safe_chi

            # === ER LOSS ===
            # Reference: report_brief.md lines 599-600
            if self.loss_type == "mse":
                loss = compute_er_loss_mse(f_main, f_fork)
            elif self.loss_type == "huber":
                loss = compute_er_loss_huber(f_main, f_fork)
            else:  # "crossprod"
                loss = compute_er_loss_aio(f_main, f_fork)

        # Update current policy
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        if grads[0] is not None:
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        # Update target policy with Polyak averaging
        # Reference: report_brief.md line 519
        target_update_mag = self._update_target_network()

        k_postclip = _inference_clip_k(k_preclip, transform_spec=self.transform_spec)
        clip_diag = compute_k_clip_diagnostics(
            k_postclip=k_postclip,
            k_preclip=k_preclip,
        )

        return {
            "loss_ER": float(loss),
            "target_policy_update": target_update_mag,
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
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
        k_next, k_preclip = forward_basic_policy_levels(
            policy_net=self.policy_net,
            k=k,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
            return_preclip=True,
        )

        # k and k_next are in LEVELS - pass directly to Euler primitives
        chi = euler_chi(k, k_next, self.params)

        # Future step (target policy)
        k_next_next_main = forward_basic_policy_levels(
            policy_net=self.target_policy_net,
            k=k_next,
            z=z_next_main,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )
        m_main = euler_m(k_next, k_next_next_main, z_next_main, self.params)

        k_next_next_fork = forward_basic_policy_levels(
            policy_net=self.target_policy_net,
            k=k_next,
            z=z_next_fork,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )
        m_fork = euler_m(k_next, k_next_next_fork, z_next_fork, self.params)

        # Unit-free Euler residuals: f = 1 - β * m / χ
        safe_chi = tf.maximum(chi, 1e-8)
        f_main = 1.0 - self.beta * m_main / safe_chi
        f_fork = 1.0 - self.beta * m_fork / safe_chi

        if self.loss_type == "mse":
            loss = compute_er_loss_mse(f_main, f_fork)
        elif self.loss_type == "huber":
            loss = compute_er_loss_huber(f_main, f_fork)
        else:  # "crossprod"
            loss = compute_er_loss_aio(f_main, f_fork)

        k_postclip = _inference_clip_k(k_preclip, transform_spec=self.transform_spec)
        clip_diag = compute_k_clip_diagnostics(
            k_postclip=k_postclip,
            k_preclip=k_preclip,
        )
        return {
            "loss_ER": float(loss),
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
        }


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

    Note: Network and data operate in LEVELS. The network's internal
    normalization (bounded sigmoid) handles numerical stability.
    """
    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        value_net: BasicValueNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        optimizer_actor: tf.keras.optimizers.Optimizer,
        optimizer_value: tf.keras.optimizers.Optimizer,
        n_critic_steps: int,
        logit_clip: float,
        polyak_tau: float,
        loss_type: str = "crossprod",
        br_scale: float = 1.0,
        transform_spec: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize BR (Actor-Critic) trainer.

        Args:
            policy_net: Policy (actor) network to train.
            value_net: Value (critic) network to train.
            params: Economic parameters.
            shock_params: Shock process parameters.
            optimizer_actor: Configured optimizer for actor (use create_optimizer()).
            optimizer_value: Configured optimizer for critic (use create_optimizer()).
            n_critic_steps: Number of critic updates per actor update.
                Use MethodConfig.n_critic (default: DEFAULT_N_CRITIC = 5).
            logit_clip: Logit clipping bound for smooth indicators.
                Use AnnealingConfig.logit_clip (default: DEFAULT_LOGIT_CLIP = 20.0).
            polyak_tau: Polyak averaging coefficient for target network updates.
                Use MethodConfig.polyak_tau (default: DEFAULT_POLYAK_TAU = 0.995).
            loss_type: Loss computation method for Bellman residual:
                - "mse": Mean Squared Error E[f²], biased but stable
                - "huber": Huber TD loss, robust to outlier residuals
                - "crossprod": AiO cross-product E[f₁·f₂], unbiased (default)
                Use MethodConfig.loss_type (default: "crossprod").
            br_scale: Positive scalar used to normalize BR residuals before
                computing critic loss. Set to 1.0 to disable normalization.
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.transform_spec = transform_spec or build_legacy_basic_transform_spec_from_networks(
            policy_net=policy_net,
            value_net=value_net,
        )
        self.params = params
        self.shock_params = shock_params
        self.optimizer_policy = optimizer_actor
        self.optimizer_value = optimizer_value
        self.n_critic_steps = n_critic_steps
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)
        self.polyak_tau = polyak_tau
        self.loss_type = loss_type
        self.br_scale = float(br_scale)

        # Validate loss_type
        valid_loss_types = {"mse", "huber", "crossprod", "fork_mean_square"}
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{loss_type}'")
        if self.br_scale <= 0:
            raise ValueError(f"br_scale must be > 0, got {self.br_scale}")

        # Create target networks (frozen copies)
        # Reference: report_brief.md line 590 "Initiate target networks"
        self.target_policy_net = tf.keras.models.clone_model(policy_net)
        self.target_value_net = tf.keras.models.clone_model(value_net)

        # Build target networks by calling them with normalized dummy features.
        dummy_x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        _ = self.target_policy_net(dummy_x, training=False)
        _ = self.target_value_net(dummy_x, training=False)

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
            self.target_value_net.variables,
            self.value_net.variables
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
            self.target_policy_net.variables,
            self.policy_net.variables
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
        last_diagnostics = {}  # Store diagnostics from last critic step

        for critic_step in range(self.n_critic_steps):
            with tf.GradientTape() as tape_critic:
                # === Compute Action using TARGET Policy ===
                # Reference: report_brief.md line 600
                # "Compute next action using the Target Policy"
                k_next = forward_basic_policy_levels(
                    policy_net=self.target_policy_net,
                    k=k,
                    z=z,
                    transform_spec=self.transform_spec,
                    training=False,
                    apply_output_clips=False,
                )

                # === Compute Continuation Values using TARGET Value Network ===
                # Reference: report_brief.md line 602-603
                V_next_main = forward_basic_value_levels(
                    value_net=self.target_value_net,
                    k=k_next,
                    z=z_next_main,
                    transform_spec=self.transform_spec,
                    training=False,
                    apply_output_clips=False,
                )
                V_next_fork = forward_basic_value_levels(
                    value_net=self.target_value_net,
                    k=k_next,
                    z=z_next_fork,
                    transform_spec=self.transform_spec,
                    training=False,
                    apply_output_clips=False,
                )

                # === Compute Critic Targets (Detached) ===
                # Reference: report_brief.md line 605-606
                # "Compute the fixed Bellman target (detach gradients)"
                # k and k_next are already in LEVELS - pass directly to economic functions
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
                V_curr = forward_basic_value_levels(
                    value_net=self.value_net,
                    k=k,
                    z=z,
                    transform_spec=self.transform_spec,
                    training=True,
                    apply_output_clips=False,
                )
                V_curr_norm = V_curr / self.br_scale
                y1_norm = y1 / self.br_scale
                y2_norm = y2 / self.br_scale

                # === Critic Loss ===
                # Reference: report_brief.md line 611-612
                if self.loss_type == "mse":
                    loss_critic = compute_br_critic_loss_mse(V_curr_norm, y1_norm, y2_norm)
                elif self.loss_type == "huber":
                    loss_critic = compute_br_critic_loss_huber(V_curr_norm, y1_norm, y2_norm)
                else:  # "crossprod"
                    loss_critic = compute_br_critic_loss_aio(V_curr_norm, y1_norm, y2_norm)
                _ensure_finite_scalar("basic_br/loss_critic", loss_critic)

            # Update critic (value network)
            # Reference: report_brief.md line 613
            grads_critic = tape_critic.gradient(loss_critic, self.value_net.trainable_variables)
            _ensure_finite_gradients("basic_br/critic", grads_critic)
            self.optimizer_value.apply_gradients(zip(grads_critic, self.value_net.trainable_variables))

            # Track metrics
            critic_losses.append(float(loss_critic))

            # Compute diagnostic on last iteration
            if critic_step == self.n_critic_steps - 1:
                diagnostics = compute_br_critic_diagnostics(V_curr, y1, y2)
                # Store all diagnostics for return
                last_diagnostics = diagnostics

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
            k_next, k_preclip = forward_basic_policy_levels(
                policy_net=self.policy_net,
                k=k,
                z=z,
                transform_spec=self.transform_spec,
                training=True,
                apply_output_clips=False,
                return_preclip=True,
            )

            # === Compute Continuation Value using CURRENT Value Network ===
            # Reference: report_brief.md line 626-627
            # "Predict continuation value using the Current Value network"
            # Note: We freeze value weights by not including them in gradients
            V_next_main = forward_basic_value_levels(
                value_net=self.value_net,
                k=k_next,
                z=z_next_main,
                transform_spec=self.transform_spec,
                training=False,
                apply_output_clips=False,
            )

            # === Compute Cash Flow ===
            # k and k_next are already in LEVELS - pass directly to economic functions
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
            _ensure_finite_scalar("basic_br/loss_actor", loss_actor)

        # Update actor (policy network)
        # Reference: report_brief.md line 632
        grads_actor = tape_actor.gradient(loss_actor, self.policy_net.trainable_variables)
        _ensure_finite_gradients("basic_br/actor", grads_actor)
        self.optimizer_policy.apply_gradients(zip(grads_actor, self.policy_net.trainable_variables))

        # Update target policy network with Polyak averaging
        # Reference: report_brief.md line 633
        target_policy_update = self._update_target_policy()

        k_postclip = _inference_clip_k(k_preclip, transform_spec=self.transform_spec)
        clip_diag = compute_k_clip_diagnostics(
            k_postclip=k_postclip,
            k_preclip=k_preclip,
        )

        return {
            "loss_critic": float(np.mean(critic_losses)),
            "loss_actor": float(loss_actor),
            "mse_proxy": last_diagnostics.get("mse_proxy", 0.0),
            "rel_mse": last_diagnostics.get("rel_mse", 0.0),
            "rel_mae": last_diagnostics.get("rel_mae", 0.0),
            "mean_value_scale": last_diagnostics.get("mean_value_scale", 0.0),
            "target_value_update": target_value_update,
            "target_policy_update": target_policy_update,
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
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
        k_next = forward_basic_policy_levels(
            policy_net=self.target_policy_net,
            k=k,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )
        V_next_main = forward_basic_value_levels(
            value_net=self.target_value_net,
            k=k_next,
            z=z_next_main,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )
        V_next_fork = forward_basic_value_levels(
            value_net=self.target_value_net,
            k=k_next,
            z=z_next_fork,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )

        # k and k_next are already in LEVELS - pass directly to economic functions
        e = compute_cash_flow_basic(k, k_next, z, self.params, temperature=temperature, logit_clip=self.logit_clip)

        y1 = e + self.beta * V_next_main
        y2 = e + self.beta * V_next_fork

        V_curr = forward_basic_value_levels(
            value_net=self.value_net,
            k=k,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )
        V_curr_norm = V_curr / self.br_scale
        y1_norm = y1 / self.br_scale
        y2_norm = y2 / self.br_scale
        if self.loss_type == "mse":
            loss_critic = compute_br_critic_loss_mse(V_curr_norm, y1_norm, y2_norm)
        elif self.loss_type == "huber":
            loss_critic = compute_br_critic_loss_huber(V_curr_norm, y1_norm, y2_norm)
        else:  # "crossprod"
            loss_critic = compute_br_critic_loss_aio(V_curr_norm, y1_norm, y2_norm)

        # Actor evaluation (using current policy)
        k_next_actor, k_preclip = forward_basic_policy_levels(
            policy_net=self.policy_net,
            k=k,
            z=z,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
            return_preclip=True,
        )
        V_next_actor = forward_basic_value_levels(
            value_net=self.value_net,
            k=k_next_actor,
            z=z_next_main,
            transform_spec=self.transform_spec,
            training=False,
            apply_output_clips=False,
        )
        # k and k_next_actor are already in LEVELS - pass directly to economic functions
        e_actor = compute_cash_flow_basic(k, k_next_actor, z, self.params, temperature=temperature, logit_clip=self.logit_clip)
        loss_actor = -tf.reduce_mean(e_actor + self.beta * V_next_actor)

        k_postclip = _inference_clip_k(k_preclip, transform_spec=self.transform_spec)
        clip_diag = compute_k_clip_diagnostics(
            k_postclip=k_postclip,
            k_preclip=k_preclip,
        )

        return {
            "loss_critic": float(loss_critic),
            "loss_actor": float(loss_actor),
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
        }


# BasicTrainerBRRegression has been moved to src/experimental/br_multitask.py.
# The BR-multitask formulation has a structural identification failure
# (see report/br_multitask_structural_issues.md). Use ER for production.
#
# To use the experimental trainer:
#   from src.experimental.br_multitask import BasicTrainerBRRegression
_BR_MULTITASK_MOVED = True  # sentinel for grep/search
