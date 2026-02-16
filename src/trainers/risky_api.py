"""
High-level training entrypoints for the Risky Debt model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from src.economy.parameters import EconomicParams, ShockParams
from src.networks.network_risky import build_risky_networks
from src.trainers.config import (
    NetworkConfig,
    OptimizationConfig,
    AnnealingConfig,
    MethodConfig,
    create_optimizer,
)
from src.trainers.core import execute_training_loop
from src.trainers.data_pipeline import validate_dataset_keys, build_training_iterator
from src.trainers.method_specs import get_method_spec
from src.trainers.method_names import validate_method_config_name
from src.trainers.results import TrainingResult
from src.trainers.risky_trainers import RiskyDebtTrainerBR
from src.trainers.br_normalization import resolve_br_normalization_scale


def _make_validation_fn_risky_br(trainer: RiskyDebtTrainerBR):
    """Create validation function for Risky BR method."""

    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(
            batch["k"],
            batch["b"],
            batch["z"],
            batch["z_next_main"],
            batch["z_next_fork"],
            temperature,
        )

    return validation_fn


def _batch_adapter_risky_br(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Map flattened risky batch to trainer inputs."""
    return {
        "k": batch["k"],
        "b": batch["b"],
        "z": batch["z"],
        "z_next_main": batch["z_next_main"],
        "z_next_fork": batch["z_next_fork"],
    }


def _coerce_method_config(
    method_config: MethodConfig,
    *,
    expected_name: str,
) -> MethodConfig:
    canonical = validate_method_config_name(
        method_config.name,
        expected_name=expected_name,
        model="risky",
    )
    if method_config.name != canonical:
        return replace(method_config, name=canonical)
    return method_config


def train_risky_br_actor_critic(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
) -> Union[Dict[str, Any], TrainingResult]:
    """
    Train Risky Debt Model using BR (Actor-Critic) method.
    """
    method_config = _coerce_method_config(
        method_config, expected_name="risky_br_actor_critic"
    )
    method_spec = get_method_spec("risky_br_actor_critic")
    k_bounds = bounds["k"]
    b_bounds = bounds["b"]

    if method_config.risky is None:
        raise ValueError("RiskyDebtConfig required in method_config.risky for train_risky_br")
    risky_cfg = method_config.risky

    validate_dataset_keys(
        dataset,
        method_spec.required_keys,
        method_name="risky_br_actor_critic",
    )

    logz_bounds = bounds["log_z"]
    data_iter = build_training_iterator(
        dataset,
        batch_size=opt_config.batch_size,
        shuffle_key=method_spec.shuffle_key or "k",
    )

    policy_net, value_net, price_net = build_risky_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        b_min=b_bounds[0],
        b_max=b_bounds[1],
        logz_min=logz_bounds[0],
        logz_max=logz_bounds[1],
        r_risk_free=params.r_rate,
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        activation=net_config.activation,
    )

    actor_lr = opt_config.learning_rate
    critic_lr = (
        opt_config.learning_rate_critic
        if opt_config.learning_rate_critic
        else opt_config.learning_rate
    )
    optimizer_actor = create_optimizer(actor_lr, opt_config.optimizer)
    optimizer_critic = create_optimizer(critic_lr, opt_config.optimizer)
    br_norm = resolve_br_normalization_scale(
        mode=method_config.br_normalization,
        custom_value=method_config.br_normalizer_value,
        epsilon=method_config.br_normalizer_epsilon,
        params=params,
        shock_params=shock_params,
    )

    trainer = RiskyDebtTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        price_net=price_net,
        params=params,
        shock_params=shock_params,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        weight_br=risky_cfg.weight_br,
        n_critic_steps=method_config.n_critic,
        polyak_tau=method_config.polyak_tau,
        smoothing=None,
        logit_clip=anneal_config.logit_clip,
        b_max=b_bounds[1],
        loss_type=risky_cfg.loss_type,
        br_scale=br_norm.scale,
    )

    validation_fn = _make_validation_fn_risky_br(trainer) if validation_data is not None else None
    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="risky_br_actor_critic",
        validation_data=validation_data,
        validation_fn=validation_fn,
        required_batch_keys=set(method_spec.required_keys),
        batch_adapter=_batch_adapter_risky_br,
    )

    result = TrainingResult(
        history=history,
        artifacts={
            "policy_net": policy_net,
            "value_net": value_net,
            "price_net": price_net,
            "target_policy_net": trainer.target_policy_net,
            "target_value_net": trainer.target_value_net,
            "target_price_net": trainer.target_price_net,
        },
        config={
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config,
        },
        params=params,
        meta={"model": "risky", "method": "risky_br_actor_critic", "br_scale": br_norm.scale},
    )
    return result.to_legacy_dict()


def train_risky_br(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
) -> Union[Dict[str, Any], TrainingResult]:
    """Backward-compatible alias for risky BR actor-critic."""
    return train_risky_br_actor_critic(
        dataset=dataset,
        net_config=net_config,
        opt_config=opt_config,
        method_config=method_config,
        anneal_config=anneal_config,
        params=params,
        shock_params=shock_params,
        bounds=bounds,
        validation_data=validation_data,
    )
