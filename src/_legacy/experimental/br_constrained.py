"""
Research-only BR constrained trainer.

WARNING:
This module is experimental and intentionally excluded from the production
trainer API, pipeline, and notebooks. Use at your own risk.
"""

from __future__ import annotations

import warnings
from typing import Dict

import tensorflow as tf

from src.networks.network_basic import BasicPolicyNetwork, BasicValueNetwork
from src.economy.parameters import EconomicParams, ShockParams
from src.experimental.br_multitask import BasicTrainerBRRegression


class BasicTrainerBRConstrainedExperimental(BasicTrainerBRRegression):
    """
    Experimental constrained BR regression via adaptive Lagrangian weighting.

    WARNING:
    This class is kept for research experiments only and is not part of the
    supported trainer interface.
    """

    def __init__(
        self,
        policy_net: BasicPolicyNetwork,
        value_net: BasicValueNetwork,
        params: EconomicParams,
        shock_params: ShockParams,
        optimizer_policy: tf.keras.optimizers.Optimizer,
        optimizer_value: tf.keras.optimizers.Optimizer,
        logit_clip: float,
        loss_type: str = "crossprod",
        br_scale: float = 1.0,
        use_foc: bool = True,
        use_env: bool = True,
        weight_env: float = 0.0,
        mu_init: float = 1.0,
        mu_max: float = 50.0,
        rho: float = 0.5,
    ):
        warnings.warn(
            "BasicTrainerBRConstrainedExperimental is research-only and not "
            "supported in production APIs.",
            RuntimeWarning,
            stacklevel=2,
        )
        super().__init__(
            policy_net=policy_net,
            value_net=value_net,
            params=params,
            shock_params=shock_params,
            optimizer_policy=optimizer_policy,
            optimizer_value=optimizer_value,
            logit_clip=logit_clip,
            loss_type=loss_type,
            br_scale=br_scale,
            weight_foc=0.0,
            weight_env=weight_env,
            use_foc=use_foc,
            use_env=use_env,
        )
        self.mu = float(mu_init)
        self.mu_max = float(mu_max)
        self.rho = float(rho)
        if self.mu < 0:
            raise ValueError(f"mu_init must be >= 0, got {mu_init}")
        if self.mu_max <= 0:
            raise ValueError(f"mu_max must be > 0, got {mu_max}")
        if self.mu > self.mu_max:
            raise ValueError("mu_init cannot exceed mu_max")
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")

    def _compute_objective_terms(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float,
    ) -> Dict[str, tf.Tensor]:
        base_terms = super()._compute_objective_terms(
            k=k,
            z=z,
            z_next_main=z_next_main,
            z_next_fork=z_next_fork,
            temperature=temperature,
        )
        loss_br = base_terms["loss_br"]
        loss_foc = base_terms["loss_foc"]
        loss_env = base_terms["loss_env"]
        loss_total = (
            loss_br
            + tf.cast(self.mu, tf.float32) * loss_foc
            + tf.cast(self.weight_env, tf.float32) * loss_env
        )
        out = dict(base_terms)
        out["loss_total"] = loss_total
        return out

    def _update_multiplier(self, loss_foc_value: float) -> None:
        clipped_foc = max(0.0, float(loss_foc_value))
        updated_mu = self.mu + self.rho * clipped_foc
        self.mu = max(0.0, min(self.mu_max, updated_mu))

    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float,
    ) -> Dict[str, float]:
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        with tf.GradientTape(persistent=True) as tape_outer:
            terms = self._compute_objective_terms(
                k, z, z_next_main, z_next_fork, temperature
            )
            loss_total = terms["loss_total"]

        grads_policy = tape_outer.gradient(loss_total, self.policy_net.trainable_variables)
        grads_value = tape_outer.gradient(loss_total, self.value_net.trainable_variables)
        del tape_outer

        self._apply_gradients(self.optimizer_policy, grads_policy, self.policy_net.trainable_variables)
        self._apply_gradients(self.optimizer_value, grads_value, self.value_net.trainable_variables)

        loss_foc_value = float(terms["loss_foc"])
        self._update_multiplier(loss_foc_value)

        return {
            "loss_BR_constrained": float(terms["loss_total"]),
            "loss_BR_reg": float(terms["loss_total"]),
            "loss_BR": float(terms["loss_br"]),
            "loss_FOC": loss_foc_value,
            "loss_Env": float(terms["loss_env"]),
            "mse_BR_proxy": float(terms["mse_br_proxy"]),
            "mse_FOC_proxy": float(terms["mse_foc_proxy"]),
            "mse_Env_proxy": float(terms["mse_env_proxy"]),
            "mean_abs_foc": float(terms["mean_abs_foc"]),
            "mean_abs_env": float(terms["mean_abs_env"]),
            "lagrange_mu": float(self.mu),
        }

    def evaluate(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float,
    ) -> Dict[str, float]:
        k = tf.reshape(k, [-1, 1])
        z = tf.reshape(z, [-1, 1])
        z_next_main = tf.reshape(z_next_main, [-1, 1])
        z_next_fork = tf.reshape(z_next_fork, [-1, 1])

        terms = self._compute_objective_terms(
            k, z, z_next_main, z_next_fork, temperature
        )
        return {
            "loss_BR_constrained": float(terms["loss_total"]),
            "loss_BR_reg": float(terms["loss_total"]),
            "loss_BR": float(terms["loss_br"]),
            "loss_FOC": float(terms["loss_foc"]),
            "loss_Env": float(terms["loss_env"]),
            "mse_BR_proxy": float(terms["mse_br_proxy"]),
            "mse_FOC_proxy": float(terms["mse_foc_proxy"]),
            "mse_Env_proxy": float(terms["mse_env_proxy"]),
            "mean_abs_foc": float(terms["mean_abs_foc"]),
            "mean_abs_env": float(terms["mean_abs_env"]),
            "lagrange_mu": float(self.mu),
        }

