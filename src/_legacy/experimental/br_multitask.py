"""
Experimental BR-Multitask trainer (Bellman Residual + FOC/Envelope regularization).

WARNING:
This module is experimental and intentionally excluded from the production
trainer API, pipeline, and notebooks. The BR-multitask formulation has a
structural identification failure: the joint (V, π) parameterization admits
infinitely many zero-loss pairs (see report/br_multitask_structural_issues.md).

Use ER (BasicTrainerER) for production policy learning.
This module is kept for research reproducibility and ablation experiments only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.economy.logic import compute_cash_flow_basic
from src.economy.parameters import EconomicParams, ShockParams
from src.networks.network_basic import BasicPolicyNetwork, BasicValueNetwork
from src.trainers.io_transforms import (
    build_legacy_basic_transform_spec_from_networks,
    compute_k_clip_diagnostics,
    forward_basic_policy_levels,
    forward_basic_value_levels,
)
from src.trainers.losses import (
    compute_br_critic_loss_aio,
    compute_er_loss_aio,
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


class BasicTrainerBRRegression:
    """
    Experimental: Bellman Residual Regression with FOC/Envelope regularization.

    Minimal fixed-weight form:
        L = w_br * L_BR + w_foc * L_FOC + w_env * L_Env

    WARNING: This method has a structural identification failure.
    The joint (V_θ, π_φ) parameterization admits infinitely many zero-loss
    pairs. See report/br_multitask_structural_issues.md for details.
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
        weight_br: float = 1.0,
        weight_foc: float = 1.0,
        weight_env: float = 1.0,
        use_foc: bool = True,
        use_env: bool = True,
        transform_spec: Optional[Dict[str, Any]] = None,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.transform_spec = transform_spec or build_legacy_basic_transform_spec_from_networks(
            policy_net=policy_net,
            value_net=value_net,
        )
        self.params = params
        self.shock_params = shock_params
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.logit_clip = logit_clip
        self.beta = 1.0 / (1.0 + params.r_rate)
        self.loss_type = loss_type
        self.br_scale = float(br_scale)
        self.weight_br = float(weight_br)
        self.weight_foc = float(weight_foc)
        self.weight_env = float(weight_env)
        self.use_foc = use_foc
        self.use_env = use_env

        valid_loss_types = {"mse", "crossprod"}
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{loss_type}'")
        if self.br_scale <= 0:
            raise ValueError(f"br_scale must be > 0, got {self.br_scale}")
        if self.weight_br < 0:
            raise ValueError(f"weight_br must be >= 0, got {weight_br}")
        if self.weight_foc < 0:
            raise ValueError(f"weight_foc must be >= 0, got {weight_foc}")
        if self.weight_env < 0:
            raise ValueError(f"weight_env must be >= 0, got {weight_env}")
        if (self.weight_br <= 0) and (not use_foc) and (not use_env):
            raise ValueError(
                "At least one objective component must be active: "
                "weight_br > 0 or use_foc=True or use_env=True."
            )

        # Fixed weights (no adaptive mode)
        self.current_weight_br = float(self.weight_br)
        self.current_weight_foc = float(self.weight_foc)
        self.current_weight_env = float(self.weight_env)

    @staticmethod
    def _safe_gradient(
        tape: tf.GradientTape,
        target: tf.Tensor,
        source: tf.Tensor,
    ) -> tf.Tensor:
        grad = tape.gradient(target, source)
        if grad is None:
            return tf.zeros_like(source)
        return grad

    @staticmethod
    def _apply_gradients(
        optimizer: tf.keras.optimizers.Optimizer,
        grads: List[tf.Tensor],
        vars_: List[tf.Variable],
    ) -> None:
        pairs = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if pairs:
            optimizer.apply_gradients(pairs)

    def _compute_objective_terms(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        z_next_main: tf.Tensor,
        z_next_fork: tf.Tensor,
        temperature: float,
    ) -> Dict[str, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape_inner:
            k_next, k_preclip = forward_basic_policy_levels(
                policy_net=self.policy_net,
                k=k,
                z=z,
                transform_spec=self.transform_spec,
                training=True,
                apply_output_clips=False,
                return_preclip=True,
            )
            tape_inner.watch(k)
            tape_inner.watch(k_next)

            e = compute_cash_flow_basic(
                k, k_next, z, self.params,
                temperature=temperature,
                logit_clip=self.logit_clip,
            )
            V_curr = forward_basic_value_levels(
                value_net=self.value_net, k=k, z=z,
                transform_spec=self.transform_spec,
                training=True, apply_output_clips=False,
            )
            V_next_main = forward_basic_value_levels(
                value_net=self.value_net, k=k_next, z=z_next_main,
                transform_spec=self.transform_spec,
                training=True, apply_output_clips=False,
            )
            V_next_fork = forward_basic_value_levels(
                value_net=self.value_net, k=k_next, z=z_next_fork,
                transform_spec=self.transform_spec,
                training=True, apply_output_clips=False,
            )

        # BR residuals
        y_main = e + self.beta * V_next_main
        y_fork = e + self.beta * V_next_fork
        V_curr_norm = V_curr / self.br_scale
        y_main_norm = y_main / self.br_scale
        y_fork_norm = y_fork / self.br_scale
        res_main = V_curr_norm - y_main_norm
        res_fork = V_curr_norm - y_fork_norm

        if self.loss_type == "mse":
            loss_br = tf.reduce_mean(0.5 * (tf.square(res_main) + tf.square(res_fork)))
        else:  # crossprod
            loss_br = compute_br_critic_loss_aio(V_curr_norm, y_main_norm, y_fork_norm)

        # FOC residual: de/dk' + β * dV/dk' = 0
        de_dk_next = self._safe_gradient(tape_inner, e, k_next)
        dV_main_dk_next = self._safe_gradient(tape_inner, V_next_main, k_next)
        dV_fork_dk_next = self._safe_gradient(tape_inner, V_next_fork, k_next)
        foc_main = de_dk_next + self.beta * dV_main_dk_next
        foc_fork = de_dk_next + self.beta * dV_fork_dk_next

        if self.use_foc:
            if self.loss_type == "mse":
                loss_foc = tf.reduce_mean(0.5 * (tf.square(foc_main) + tf.square(foc_fork)))
            else:
                loss_foc = compute_er_loss_aio(foc_main, foc_fork)
        else:
            loss_foc = tf.constant(0.0, dtype=tf.float32)

        # Envelope residual: de/dk - dV/dk = 0
        de_dk = self._safe_gradient(tape_inner, e, k)
        dV_curr_dk = self._safe_gradient(tape_inner, V_curr, k)
        env_residual = de_dk - dV_curr_dk
        if self.use_env:
            loss_env = tf.reduce_mean(tf.square(env_residual))
        else:
            loss_env = tf.constant(0.0, dtype=tf.float32)

        del tape_inner

        # Diagnostics
        bellman_res_main = V_curr - y_main
        bellman_res_fork = V_curr - y_fork
        mse_br_proxy = tf.reduce_mean(
            0.5 * (tf.square(bellman_res_main) + tf.square(bellman_res_fork))
        )
        mse_foc_proxy = tf.reduce_mean(0.5 * (tf.square(foc_main) + tf.square(foc_fork)))
        mse_env_proxy = tf.reduce_mean(tf.square(env_residual))

        return {
            "loss_br": loss_br,
            "loss_foc": loss_foc,
            "loss_env": loss_env,
            "mse_br_proxy": mse_br_proxy,
            "mse_foc_proxy": mse_foc_proxy,
            "mse_env_proxy": mse_env_proxy,
            "mean_abs_foc": tf.reduce_mean(0.5 * (tf.abs(foc_main) + tf.abs(foc_fork))),
            "mean_abs_env": tf.reduce_mean(tf.abs(env_residual)),
            "k_preclip": k_preclip,
        }

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

        w_br = self.current_weight_br
        w_foc = self.current_weight_foc if self.use_foc else 0.0
        w_env = self.current_weight_env if self.use_env else 0.0

        with tf.GradientTape(persistent=True) as tape_outer:
            terms = self._compute_objective_terms(
                k, z, z_next_main, z_next_fork, temperature
            )
            loss_total = (
                tf.constant(w_br, dtype=tf.float32) * terms["loss_br"]
                + tf.constant(w_foc, dtype=tf.float32) * terms["loss_foc"]
                + tf.constant(w_env, dtype=tf.float32) * terms["loss_env"]
            )

        grads_policy = tape_outer.gradient(loss_total, self.policy_net.trainable_variables)
        grads_value = tape_outer.gradient(loss_total, self.value_net.trainable_variables)
        del tape_outer

        self._apply_gradients(self.optimizer_policy, grads_policy, self.policy_net.trainable_variables)
        self._apply_gradients(self.optimizer_value, grads_value, self.value_net.trainable_variables)

        k_postclip = _inference_clip_k(terms["k_preclip"], transform_spec=self.transform_spec)
        clip_diag = compute_k_clip_diagnostics(k_postclip=k_postclip, k_preclip=terms["k_preclip"])

        return {
            "loss_BR_reg": float(loss_total),
            "loss_BR": float(terms["loss_br"]),
            "loss_FOC": float(terms["loss_foc"]),
            "loss_Env": float(terms["loss_env"]),
            "weight_br_used": float(w_br),
            "weight_foc_used": float(w_foc),
            "weight_env_used": float(w_env),
            "mse_BR_proxy": float(terms["mse_br_proxy"]),
            "mse_FOC_proxy": float(terms["mse_foc_proxy"]),
            "mse_Env_proxy": float(terms["mse_env_proxy"]),
            "mean_abs_foc": float(terms["mean_abs_foc"]),
            "mean_abs_env": float(terms["mean_abs_env"]),
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
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
        w_br = self.current_weight_br
        w_foc = self.current_weight_foc if self.use_foc else 0.0
        w_env = self.current_weight_env if self.use_env else 0.0
        loss_total = (
            tf.constant(w_br, dtype=tf.float32) * terms["loss_br"]
            + tf.constant(w_foc, dtype=tf.float32) * terms["loss_foc"]
            + tf.constant(w_env, dtype=tf.float32) * terms["loss_env"]
        )
        k_postclip = _inference_clip_k(terms["k_preclip"], transform_spec=self.transform_spec)
        clip_diag = compute_k_clip_diagnostics(k_postclip=k_postclip, k_preclip=terms["k_preclip"])
        return {
            "loss_BR_reg": float(loss_total),
            "loss_BR": float(terms["loss_br"]),
            "loss_FOC": float(terms["loss_foc"]),
            "loss_Env": float(terms["loss_env"]),
            "weight_br_used": float(w_br),
            "weight_foc_used": float(w_foc),
            "weight_env_used": float(w_env),
            "mse_BR_proxy": float(terms["mse_br_proxy"]),
            "mse_FOC_proxy": float(terms["mse_foc_proxy"]),
            "mse_Env_proxy": float(terms["mse_env_proxy"]),
            "mean_abs_foc": float(terms["mean_abs_foc"]),
            "mean_abs_env": float(terms["mean_abs_env"]),
            "clip_fraction_k": clip_diag["clip_fraction_k"],
            "preclip_max_k": clip_diag["preclip_max_k"],
        }
