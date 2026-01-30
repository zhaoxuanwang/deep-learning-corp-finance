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
    cash_flow_risky_debt_q,
    external_financing_cost,
    recovery_value,
    pricing_residual_zero_profit,
    pricing_residual_bond_price,
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
)
from src.utils.annealing import AnnealingSchedule, indicator_default
from src.trainers.config import NetworkConfig, OptimizationConfig, AnnealingConfig, MethodConfig
from src.trainers.core import execute_training_loop

logger = logging.getLogger(__name__)



class RiskyDebtTrainerBR:
    """
    Bellman Residual (Actor-Critic) trainer for Risky Debt model.

    Implements the algorithm from report_brief.md lines 917-1082.

    Key Features:
    - Operates on flattened i.i.d. transitions (k, b, z) -> (k', b', z')
    - Three networks: Policy, Value, Price (with corresponding target networks)
    - Target networks updated via Polyak averaging for stability (DDPG-style)
    - Uses Gumbel-Sigmoid for smooth default probability approximation
    - Jointly minimizes Bellman residual and pricing residual in Critic step
    - Actor maximizes expected firm value using main shock only

    Reference:
        report_brief.md lines 917-1082: "BR Method" for Risky Debt
    """

    def __init__(
        self,
        policy_net: RiskyPolicyNetwork,
        value_net: RiskyValueNetwork,
        price_net: RiskyPriceNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        weight_br: float = 0.1,
        n_critic_steps: int = 5,
        polyak_tau: float = 0.995,
        smoothing: Optional[AnnealingSchedule] = None,
        logit_clip: float = 20.0
    ):
        """
        Initialize the BR trainer with target networks.

        Args:
            policy_net: Policy network (k, b, z) -> (k', b')
            value_net: Value network (k, b, z) -> V_tilde
            price_net: Price network (k', b', z) -> q (bond price)
            params: Economic parameters
            shock_params: Shock process parameters
            actor_lr: Learning rate for policy network
            critic_lr: Learning rate for value and price networks
            weight_br: Weight on BR loss in critic objective (price weight = 1.0 implicit)
                       L_critic = weight_br * L_BR + L_price
                       Default 0.1 because BR loss is typically 100x larger than price loss
            n_critic_steps: Number of critic updates per actor update
            polyak_tau: Polyak averaging coefficient for target networks
            smoothing: Annealing schedule for default probability temperature
            logit_clip: Clipping bound for logits in smooth indicators
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.price_net = price_net
        self.params = params
        self.shock_params = shock_params
        self.weight_br = weight_br
        self.n_critic_steps = n_critic_steps
        self.polyak_tau = polyak_tau
        self.smoothing = smoothing or AnnealingSchedule(init_temp=1.0, min_temp=1e-4, decay_rate=0.99)
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)

        # Separate optimizers for actor and critic
        self.optimizer_actor = tf.keras.optimizers.Adam(actor_lr)
        self.optimizer_critic = tf.keras.optimizers.Adam(critic_lr)

        # Build original networks first (required before cloning/copying weights)
        dummy_k = tf.constant([[1.0]], dtype=tf.float32)
        dummy_b = tf.constant([[0.5]], dtype=tf.float32)
        dummy_z = tf.constant([[1.0]], dtype=tf.float32)
        _ = policy_net(dummy_k, dummy_b, dummy_z)
        _ = value_net(dummy_k, dummy_b, dummy_z)
        _ = price_net(dummy_k, dummy_b, dummy_z)

        # Create target networks (frozen copies)
        # Reference: report_brief.md line 671: "Initiate target networks"
        self.target_policy_net = tf.keras.models.clone_model(policy_net)
        self.target_value_net = tf.keras.models.clone_model(value_net)
        self.target_price_net = tf.keras.models.clone_model(price_net)

        # Build target networks with dummy data
        _ = self.target_policy_net(dummy_k, dummy_b, dummy_z)
        _ = self.target_value_net(dummy_k, dummy_b, dummy_z)
        _ = self.target_price_net(dummy_k, dummy_b, dummy_z)

        # Initialize target networks with current network weights
        self.target_policy_net.set_weights(policy_net.get_weights())
        self.target_value_net.set_weights(value_net.get_weights())
        self.target_price_net.set_weights(price_net.get_weights())

    def _update_target_networks(self) -> Dict[str, float]:
        """
        Update all target networks using Polyak averaging.

        θ⁻ ← ν θ⁻ + (1-ν) θ

        Reference:
            report_brief.md line 1065: "Polyak update θ⁻_value, θ⁻_price"
            report_brief.md line 1076: "Polyak update θ⁻_policy"

        Returns:
            Dict with update magnitudes for monitoring
        """
        update_mags = {}

        for name, (target_net, source_net) in [
            ("policy", (self.target_policy_net, self.policy_net)),
            ("value", (self.target_value_net, self.value_net)),
            ("price", (self.target_price_net, self.price_net))
        ]:
            mags = []
            for target_var, source_var in zip(
                target_net.trainable_variables,
                source_net.trainable_variables
            ):
                delta = (1.0 - self.polyak_tau) * (source_var - target_var)
                mags.append(float(tf.reduce_mean(tf.abs(delta))))
                target_var.assign(self.polyak_tau * target_var + (1.0 - self.polyak_tau) * source_var)
            update_mags[f"target_{name}_update"] = float(np.mean(mags)) if mags else 0.0

        return update_mags

    def _critic_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float
    ) -> Dict[str, float]:
        """
        Execute one critic update step.

        Updates value and price networks to minimize:
        L_critic = w1 * L_BR + w2 * L_price

        Uses TARGET networks for computing Bellman targets (DDPG-style).

        Reference:
            report_brief.md lines 1041-1065: "A. Critic Update"

        Args:
            k: Current capital (batch_size,)
            b: Current debt (batch_size,)
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)
            temperature: Annealing temperature for smooth indicators

        Returns:
            Dict with loss_br and loss_price
        """
        # Import here to avoid circular imports
        from src.networks.network_risky import compute_effective_value
        from src.economy.logic import cash_flow_risky_debt_q, pricing_residual_bond_price, recovery_value

        with tf.GradientTape() as tape:
            # === Compute Actions using TARGET Policy (Detached) ===
            # Reference: report_brief.md line 1044: "Get next actions"
            k_next, b_next = self.target_policy_net(k, b, z)

            # === Compute Bond Price using CURRENT Price Network (Trainable) ===
            q = self.price_net(k_next, b_next, z)

            # === Compute Cash Flow ===
            e = cash_flow_risky_debt_q(k, k_next, b, b_next, z, q, self.params,
                                        temperature=temperature, logit_clip=self.logit_clip)
            eta = external_financing_cost(e, self.params, temperature=temperature, logit_clip=self.logit_clip)

            # === Compute Continuation Values using TARGET Value Network ===
            # Reference: report_brief.md lines 1045-1047
            V_tilde_next_1 = self.target_value_net(k_next, b_next, z_next_main)
            V_tilde_next_2 = self.target_value_net(k_next, b_next, z_next_fork)

            # Apply smooth limited liability: V_eff = (1-p) * V
            # Reference: report_brief.md lines 930-941
            V_eff_1, p_D_1 = compute_effective_value(V_tilde_next_1, k_next, temperature, self.logit_clip, noise=True)
            V_eff_2, p_D_2 = compute_effective_value(V_tilde_next_2, k_next, temperature, self.logit_clip, noise=True)

            # === Compute Bellman Targets (Detached) ===
            # Reference: report_brief.md line 1047: "y = e - η + β·V_eff"
            y1 = tf.stop_gradient(e - eta + self.beta * V_eff_1)
            y2 = tf.stop_gradient(e - eta + self.beta * V_eff_2)

            # === Compute Current Value (Trainable) ===
            V_curr = self.value_net(k, b, z)

            # === Bellman Residual Loss (AiO) ===
            # Reference: report_brief.md lines 1056-1057
            loss_br = compute_br_critic_loss_aio(V_curr, y1, y2)

            # === Pricing Residual Loss (AiO) ===
            # Reference: report_brief.md lines 1048-1062
            R_1 = recovery_value(k_next, z_next_main, self.params)
            R_2 = recovery_value(k_next, z_next_fork, self.params)

            # Pricing residual: f = q·b'·(1+r) - β·[(1-p)·b' + p·R]
            f_p1 = pricing_residual_bond_price(q, b_next, self.params.r_rate, p_D_1, R_1)
            f_p2 = pricing_residual_bond_price(q, b_next, self.params.r_rate, p_D_2, R_2)
            loss_price = compute_price_loss_aio(f_p1, f_p2)

            # === Combined Critic Loss ===
            # L_critic = weight_br * L_BR + L_price (price weight = 1.0 implicit)
            # Normalize to price loss for better numerical stability (BR loss is ~100x larger)
            # Reference: report_brief.md lines 1062-1063
            total_loss = self.weight_br * loss_br + loss_price

        # Update critic networks (value + price)
        critic_vars = self.value_net.trainable_variables + self.price_net.trainable_variables
        grads = tape.gradient(total_loss, critic_vars)
        self.optimizer_critic.apply_gradients(zip(grads, critic_vars))

        return {
            "loss_br": float(loss_br),
            "loss_price": float(loss_price),
            "mean_p_default": float(tf.reduce_mean(0.5 * (p_D_1 + p_D_2)))
        }

    def _actor_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        temperature: float
    ) -> Dict[str, float]:
        """
        Execute one actor update step.

        Updates policy network to maximize expected firm value (Bellman RHS).
        Uses CURRENT networks with gradients flowing through policy.

        IMPORTANT: Uses only main shock z'_1, not AiO cross-product.

        Reference:
            report_brief.md lines 1067-1077: "B. Actor Update"
            report_brief.md lines 992-993: "Use the main shock z'_{i,1}"

        Args:
            k: Current capital (batch_size,)
            b: Current debt (batch_size,)
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork only (batch_size,)
            temperature: Annealing temperature

        Returns:
            Dict with loss_actor
        """
        from src.networks.network_risky import compute_effective_value
        from src.economy.logic import cash_flow_risky_debt_q

        with tf.GradientTape() as tape:
            # === Compute Actions using CURRENT Policy (Gradients Flow) ===
            # Reference: report_brief.md line 1069
            k_next, b_next = self.policy_net(k, b, z)

            # === Evaluate Price using CURRENT Price Network ===
            # Reference: report_brief.md line 1071
            q = self.price_net(k_next, b_next, z)

            # === Compute Cash Flow ===
            e = cash_flow_risky_debt_q(k, k_next, b, b_next, z, q, self.params,
                                        temperature=temperature, logit_clip=self.logit_clip)
            eta = external_financing_cost(e, self.params, temperature=temperature, logit_clip=self.logit_clip)

            # === Compute Continuation Value using CURRENT Value Network ===
            # Reference: report_brief.md line 1072: "Evaluate Value"
            # NOTE: We do NOT freeze value weights - gradients flow through k', b'
            V_tilde_next = self.value_net(k_next, b_next, z_next_main)

            # Apply smooth limited liability: V_eff = (1-p) * V
            # This avoids dying ReLU problem in default region
            # Reference: report_brief.md lines 994-1002
            V_eff, p_D = compute_effective_value(V_tilde_next, k_next, temperature, self.logit_clip, noise=False)

            # === Actor Loss ===
            # Maximize: E[e - η + β·V_eff] => Minimize: -E[...]
            # Reference: report_brief.md lines 1073-1074
            bellman_rhs = e - eta + self.beta * V_eff
            loss_actor = -tf.reduce_mean(bellman_rhs)

        # Update policy network only
        grads = tape.gradient(loss_actor, self.policy_net.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        return {
            "loss_actor": float(loss_actor),
            "mean_bellman_rhs": float(tf.reduce_mean(bellman_rhs)),
            "mean_p_default_actor": float(tf.reduce_mean(p_D))
        }

    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float = 0.1
    ) -> Dict[str, float]:
        """
        Execute one complete training step (critic + actor updates).

        Algorithm:
        1. Run n_critic_steps critic updates (value + price networks)
        2. Run 1 actor update (policy network)
        3. Polyak update all target networks
        4. Update annealing schedule

        Reference:
            report_brief.md lines 1036-1082: "Algorithm Summary"

        Args:
            k: Current capital (batch_size,) - flattened i.i.d. samples
            b: Current debt (batch_size,)
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)
            temperature: Annealing temperature (overridden by smoothing.value if provided)

        Returns:
            Dict with training metrics
        """
        # Reshape inputs
        k = tf.reshape(k, [-1, 1])
        b = tf.reshape(b, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        # Use annealing schedule temperature
        temp = self.smoothing.value

        # === A. Critic Updates (Multiple per actor) ===
        critic_losses = []
        price_losses = []
        p_defaults = []

        for _ in range(self.n_critic_steps):
            critic_metrics = self._critic_step(k, b, z, z_next_main, z_next_fork, temp)
            critic_losses.append(critic_metrics["loss_br"])
            price_losses.append(critic_metrics["loss_price"])
            p_defaults.append(critic_metrics["mean_p_default"])

        # === B. Actor Update (Once) ===
        actor_metrics = self._actor_step(k, b, z, z_next_main, temp)

        # === Update Target Networks ===
        target_updates = self._update_target_networks()

        # === Update Annealing Schedule ===
        self.smoothing.update()

        return {
            "loss_critic": float(np.mean(critic_losses)),
            "loss_actor": actor_metrics["loss_actor"],
            "loss_price": float(np.mean(price_losses)),
            "mean_p_default": float(np.mean(p_defaults)),
            "mean_bellman_rhs": actor_metrics["mean_bellman_rhs"],
            "temperature": temp,
            **target_updates
        }

    def evaluate(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
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
            b: Current debt (batch_size,)
            z: Current productivity (batch_size,)
            z_next_main: Next productivity, main fork (batch_size,)
            z_next_fork: Next productivity, second fork (batch_size,)
            temperature: Annealing temperature

        Returns:
            Dict with loss_critic and loss_actor metrics.
        """
        from src.networks.network_risky import compute_effective_value
        from src.economy.logic import cash_flow_risky_debt_q, pricing_residual_bond_price, recovery_value

        # Reshape inputs
        k = tf.reshape(k, [-1, 1])
        b = tf.reshape(b, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        # Critic evaluation (using target networks)
        k_next, b_next = self.target_policy_net(k, b, z)
        q = self.price_net(k_next, b_next, z)

        e = cash_flow_risky_debt_q(k, k_next, b, b_next, z, q, self.params,
                                    temperature=temperature, logit_clip=self.logit_clip)
        eta = external_financing_cost(e, self.params, temperature=temperature, logit_clip=self.logit_clip)

        V_tilde_next_1 = self.target_value_net(k_next, b_next, z_next_main)
        V_tilde_next_2 = self.target_value_net(k_next, b_next, z_next_fork)

        # Use deterministic sigmoid for evaluation (no Gumbel noise)
        V_eff_1, p_D_1 = compute_effective_value(V_tilde_next_1, k_next, temperature, self.logit_clip, noise=False)
        V_eff_2, p_D_2 = compute_effective_value(V_tilde_next_2, k_next, temperature, self.logit_clip, noise=False)

        y1 = e - eta + self.beta * V_eff_1
        y2 = e - eta + self.beta * V_eff_2

        V_curr = self.value_net(k, b, z)
        loss_critic = compute_br_critic_loss_aio(V_curr, y1, y2)

        # Actor evaluation
        k_next_actor, b_next_actor = self.policy_net(k, b, z)
        q_actor = self.price_net(k_next_actor, b_next_actor, z)
        e_actor = cash_flow_risky_debt_q(k, k_next_actor, b, b_next_actor, z, q_actor, self.params,
                                          temperature=temperature, logit_clip=self.logit_clip)
        eta_actor = external_financing_cost(e_actor, self.params, temperature=temperature, logit_clip=self.logit_clip)
        V_tilde_next_actor = self.value_net(k_next_actor, b_next_actor, z_next_main)
        V_eff_actor, _ = compute_effective_value(V_tilde_next_actor, k_next_actor, temperature, self.logit_clip, noise=False)
        loss_actor = -tf.reduce_mean(e_actor - eta_actor + self.beta * V_eff_actor)

        return {
            "loss_critic": float(loss_critic),
            "loss_actor": float(loss_actor)
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
    """
    Train Risky Debt model via constrained lifetime reward maximization.

    Uses adaptive Lagrange multiplier to enforce zero-profit bond pricing
    constraint while maximizing firm value. The value network is trained
    jointly to provide default probability estimates required for pricing.

    Args:
        dataset: Training data with trajectories
        net_config: Network architecture configuration
        opt_config: Optimization hyperparameters
        method_config: Method-specific configuration (must include risky field)
        anneal_config: Annealing schedule for smooth gates
        params: Economic parameters
        shock_params: Shock process parameters
        bounds: State space bounds (k, b, log_z)

    Returns:
        Dictionary with training history and trained networks
    """
    k_bounds = bounds['k']
    b_bounds = bounds['b']

    # Check for risky config
    if method_config.risky is None:
        raise ValueError("RiskyDebtConfig required in method_config.risky for train_risky_lr")
    risky_cfg = method_config.risky

    # Data Setup
    if 'z_path' in dataset:
        T = dataset['z_path'].shape[1] - 1
    else:
        raise ValueError("Cannot infer Horizon T from dataset.")

    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k0'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    # Build all three networks
    policy_net, value_net, price_net = build_risky_networks(
        k_min=k_bounds[0], k_max=k_bounds[1],
        b_min=b_bounds[0], b_max=b_bounds[1],
        r_risk_free=params.r_rate,
        n_layers=net_config.n_layers, n_neurons=net_config.n_neurons, activation=net_config.activation
    )

    # Setup annealing schedule for default probability
    smoothing = AnnealingSchedule(
        init_temp=risky_cfg.epsilon_D_0,
        min=risky_cfg.epsilon_D_min,
        decay=risky_cfg.decay_d,
        schedule="exponential"
    )

    # Determine value network learning rate
    lr_value = risky_cfg.learning_rate_value if risky_cfg.learning_rate_value is not None else opt_config.learning_rate

    # Create trainer with adaptive Lagrange multiplier
    trainer = RiskyDebtTrainerLR(
        policy_net=policy_net,
        price_net=price_net,
        value_net=value_net,  # Now trained jointly
        params=params,
        shock_params=shock_params,
        optimizer_policy=tf.keras.optimizers.Adam(opt_config.learning_rate),
        optimizer_price=tf.keras.optimizers.Adam(opt_config.learning_rate),
        optimizer_value=tf.keras.optimizers.Adam(lr_value),
        T=T,
        batch_size=opt_config.batch_size,
        lambda_price_init=risky_cfg.lambda_price_init,
        learning_rate_lambda=risky_cfg.learning_rate_lambda,
        epsilon_price=risky_cfg.epsilon_price,
        polyak_weight=risky_cfg.polyak_weight,
        n_value_update_freq=risky_cfg.n_value_update_freq,
        smoothing=smoothing,
        logit_clip=anneal_config.logit_clip
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
        "_policy_net": policy_net,
        "_price_net": price_net,
        "_value_net": value_net,
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
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None
) -> Dict[str, Any]:
    """
    Train Risky Debt Model using BR (Actor-Critic) method.

    IMPORTANT: This method requires FLATTENED dataset format.
    Use DataGenerator.get_flattened_training_dataset() with include_debt=True.

    The dataset must contain:
        - 'k': Current capital (N*T,) - independent samples
        - 'b': Current debt (N*T,)
        - 'z': Current productivity (N*T,)
        - 'z_next_main': Next productivity, main fork (N*T,)
        - 'z_next_fork': Next productivity, second fork (N*T,)

    Reference:
        report_brief.md lines 157-167: "Flatten Data for ER and BR"
        report_brief.md lines 917-1082: "BR Method" for Risky Debt

    Args:
        dataset: FLATTENED training dataset
        net_config: Network architecture configuration
        opt_config: Optimization configuration
        method_config: Method-specific configuration (must include risky field)
        anneal_config: Annealing configuration
        params: Economic parameters
        shock_params: Shock process parameters
        bounds: State space bounds (k, b)
        validation_data: Optional FLATTENED validation dataset for early stopping

    Returns:
        Dict containing:
            - history: Training metrics over time
            - _policy_net: Trained policy network
            - _value_net: Trained value network
            - _price_net: Trained price network
            - _target_*: Target networks
            - _configs: Configuration objects
            - _params: Economic parameters
    """
    k_bounds = bounds['k']
    b_bounds = bounds['b']

    # Check for risky config
    if method_config.risky is None:
        raise ValueError("RiskyDebtConfig required in method_config.risky for train_risky_br")
    risky_cfg = method_config.risky

    # Validate dataset format - must be flattened
    required_keys = {'k', 'b', 'z', 'z_next_main', 'z_next_fork'}
    if not required_keys.issubset(dataset.keys()):
        raise ValueError(
            f"Risky BR method requires FLATTENED dataset with keys {required_keys}. "
            f"Got keys: {set(dataset.keys())}. "
            f"Use DataGenerator.get_flattened_training_dataset(include_debt=True) for Risky BR method."
        )

    # Create batched iterator
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=dataset['k'].shape[0]).batch(opt_config.batch_size).repeat()
    data_iter = iter(tf_dataset)

    # Build all three networks
    policy_net, value_net, price_net = build_risky_networks(
        k_min=k_bounds[0], k_max=k_bounds[1],
        b_min=b_bounds[0], b_max=b_bounds[1],
        r_risk_free=params.r_rate,
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        activation=net_config.activation
    )

    # Setup annealing schedule for default probability smoothing (from AnnealingConfig)
    smoothing = AnnealingSchedule(
        init_temp=anneal_config.temperature_init,
        min_temp=anneal_config.temperature_min,
        decay_rate=anneal_config.decay
    )

    # Determine learning rates
    actor_lr = opt_config.learning_rate
    critic_lr = opt_config.learning_rate_critic if opt_config.learning_rate_critic else opt_config.learning_rate

    # Create trainer with target networks
    trainer = RiskyDebtTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        weight_br=risky_cfg.weight_br,
        n_critic_steps=method_config.n_critic,
        polyak_tau=method_config.polyak_tau,
        smoothing=smoothing,
        logit_clip=anneal_config.logit_clip
    )

    # Setup validation function (for early stopping)
    validation_fn = _make_validation_fn_risky_br(trainer) if validation_data is not None else None

    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="risky_br",
        validation_data=validation_data,
        validation_fn=validation_fn
    )

    return {
        "history": history,
        "_policy_net": policy_net,
        "_value_net": value_net,
        "_price_net": price_net,
        "_target_policy_net": trainer.target_policy_net,
        "_target_value_net": trainer.target_value_net,
        "_target_price_net": trainer.target_price_net,
        "_configs": {
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config
        },
        "_params": params
    }


def _make_validation_fn_risky_br(trainer: RiskyDebtTrainerBR):
    """Create validation function for Risky BR method."""
    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(
            batch['k'], batch['b'], batch['z'],
            batch['z_next_main'], batch['z_next_fork'],
            temperature
        )
    return validation_fn
