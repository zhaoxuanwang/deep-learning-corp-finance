"""
High-level training entrypoints for the Basic model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from src.economy.parameters import EconomicParams, ShockParams
from src.networks.network_basic import build_basic_networks
from src.networks.observation_normalization import build_observation_normalizer
from src.trainers.io_transforms import build_basic_transform_spec
from src.trainers.basic_trainers import (
    BasicTrainerLR,
    BasicTrainerER,
    BasicTrainerBR,
)
from src.trainers.config import (
    NetworkConfig,
    OptimizationConfig,
    AnnealingConfig,
    MethodConfig,
    create_optimizer,
)
from src.trainers.core import execute_training_loop
from src.trainers.data_pipeline import build_training_iterator, validate_dataset_keys
from src.trainers.method_specs import get_method_spec
from src.trainers.method_names import validate_method_config_name
from src.trainers.results import TrainingResult
from src.trainers.br_normalization import resolve_br_normalization_scale
from src.trainers.warm_start import copy_policy_weights, describe_policy_source


def _batch_adapter_lr(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    return {
        "k": batch.get("k0", batch.get("k")),
        "z_path": batch["z_path"],
    }


def _batch_adapter_er(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    return {
        "k": batch["k"],
        "z": batch["z"],
        "z_next_main": batch["z_next_main"],
        "z_next_fork": batch["z_next_fork"],
    }


def _batch_adapter_br(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    return {
        "k": batch["k"],
        "z": batch["z"],
        "z_next_main": batch["z_next_main"],
        "z_next_fork": batch["z_next_fork"],
    }


def _make_validation_fn_lr(trainer: BasicTrainerLR):
    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(batch["k0"], batch["z_path"], temperature)

    return validation_fn


def _make_validation_fn_er(trainer: BasicTrainerER):
    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(
            batch["k"],
            batch["z"],
            batch["z_next_main"],
            batch["z_next_fork"],
        )

    return validation_fn


def _make_validation_fn_br(trainer: BasicTrainerBR):
    def validation_fn(trainer, batch, temperature):
        return trainer.evaluate(
            batch["k"],
            batch["z"],
            batch["z_next_main"],
            batch["z_next_fork"],
            temperature,
        )

    return validation_fn


def _coerce_method_config(
    method_config: MethodConfig,
    *,
    expected_name: str,
) -> MethodConfig:
    canonical = validate_method_config_name(
        method_config.name,
        expected_name=expected_name,
        model="basic",
    )
    if method_config.name != canonical:
        return replace(method_config, name=canonical)
    return method_config


def _resolve_br_scale(
    method_config: MethodConfig,
    params: EconomicParams,
    shock_params: ShockParams,
) -> float:
    info = resolve_br_normalization_scale(
        mode=method_config.br_normalization,
        custom_value=method_config.br_normalizer_value,
        epsilon=method_config.br_normalizer_epsilon,
        params=params,
        shock_params=shock_params,
    )
    return info.scale


def _maybe_apply_policy_warm_start(
    policy_net: tf.keras.Model,
    warm_start_policy: Optional[Any],
) -> Dict[str, Any]:
    """
    Optionally copy policy weights from a warm-start source into policy_net.
    """
    if warm_start_policy is None:
        return {"warm_start_applied": False, "warm_start_source": None}

    source_policy = copy_policy_weights(warm_start_policy, policy_net)
    source_label = describe_policy_source(warm_start_policy)
    return {
        "warm_start_applied": True,
        "warm_start_source": f"{source_label}:{type(source_policy).__name__}",
    }


def _build_basic_observation_normalizer(
    *,
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    # TODO: upgrade to one shared calibration sample across methods for
    # fully consistent LR/ER/BR normalization stats.
    obs_cfg = net_config.observation_normalization
    return build_observation_normalizer(
        dataset=dataset,
        bounds=bounds,
        k_scheme=obs_cfg.k_scheme,
        z_scheme=obs_cfg.z_scheme,
        b_scheme=obs_cfg.b_scheme,
        z_input_space=obs_cfg.z_input_space,
        epsilon=obs_cfg.epsilon,
        include_debt=False,
    )


def train_basic_lr(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    training_seed: Optional[Tuple[int, int]] = None,
) -> Union[Dict[str, Any], TrainingResult]:
    method_config = _coerce_method_config(method_config, expected_name="basic_lr")
    spec = get_method_spec("basic_lr")
    validate_dataset_keys(dataset, spec.required_keys, method_name="basic_lr")

    if "z_path" not in dataset:
        raise ValueError("Cannot infer horizon T from dataset (missing 'z_path').")
    horizon = dataset["z_path"].shape[1] - 1

    data_iter = build_training_iterator(
        dataset,
        batch_size=opt_config.batch_size,
        batch_order=spec.batch_order,
        permutation_seed=training_seed if spec.batch_order == "fixed_permutation" else None,
    )

    k_bounds = bounds["k"]
    logz_bounds = bounds["log_z"]
    observation_normalizer = _build_basic_observation_normalizer(
        dataset=dataset,
        net_config=net_config,
        bounds=bounds,
    )
    clip_cfg = net_config.inference_clips
    transform_spec = build_basic_transform_spec(
        normalizer=observation_normalizer,
        k_bounds=k_bounds,
        policy_head=net_config.basic_policy_head,
        value_head=net_config.basic_value_head,
        clip_policy_k_min=clip_cfg.basic_policy_k.clip_min,
        clip_policy_k_max=clip_cfg.basic_policy_k.clip_max,
        clip_value_min=clip_cfg.basic_value.clip_min,
        clip_value_max=clip_cfg.basic_value.clip_max,
    )
    policy_net, _ = build_basic_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        logz_min=logz_bounds[0],
        logz_max=logz_bounds[1],
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        hidden_activation=net_config.hidden_activation,
    )

    optimizer = create_optimizer(opt_config.learning_rate, opt_config.optimizer)
    trainer = BasicTrainerLR(
        policy_net=policy_net,
        transform_spec=transform_spec,
        params=params,
        shock_params=shock_params,
        optimizer=optimizer,
        T=horizon,
        logit_clip=anneal_config.logit_clip,
    )

    validation_fn = _make_validation_fn_lr(trainer) if validation_data is not None else None
    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="basic_lr",
        validation_data=validation_data,
        validation_fn=validation_fn,
        required_batch_keys=set(spec.required_keys),
        batch_adapter=_batch_adapter_lr,
    )

    result = TrainingResult(
        history=history,
        artifacts={
            "policy_net": policy_net,
        },
        config={
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config,
        },
        params=params,
        meta={
            "model": "basic",
            "method": "basic_lr",
            "io_transforms": transform_spec,
            "training_seed": list(training_seed) if training_seed is not None else None,
        },
    )
    return result.to_legacy_dict()


def train_basic_er(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    training_seed: Optional[Tuple[int, int]] = None,
) -> Union[Dict[str, Any], TrainingResult]:
    method_config = _coerce_method_config(method_config, expected_name="basic_er")
    spec = get_method_spec("basic_er")
    validate_dataset_keys(dataset, spec.required_keys, method_name="basic_er")

    data_iter = build_training_iterator(
        dataset,
        batch_size=opt_config.batch_size,
        batch_order=spec.batch_order,
        permutation_seed=training_seed if spec.batch_order == "fixed_permutation" else None,
    )

    k_bounds = bounds["k"]
    logz_bounds = bounds["log_z"]
    observation_normalizer = _build_basic_observation_normalizer(
        dataset=dataset,
        net_config=net_config,
        bounds=bounds,
    )
    clip_cfg = net_config.inference_clips
    transform_spec = build_basic_transform_spec(
        normalizer=observation_normalizer,
        k_bounds=k_bounds,
        policy_head=net_config.basic_policy_head,
        value_head=net_config.basic_value_head,
        clip_policy_k_min=clip_cfg.basic_policy_k.clip_min,
        clip_policy_k_max=clip_cfg.basic_policy_k.clip_max,
        clip_value_min=clip_cfg.basic_value.clip_min,
        clip_value_max=clip_cfg.basic_value.clip_max,
    )
    policy_net, _ = build_basic_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        logz_min=logz_bounds[0],
        logz_max=logz_bounds[1],
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        hidden_activation=net_config.hidden_activation,
    )

    optimizer = create_optimizer(opt_config.learning_rate, opt_config.optimizer)
    trainer = BasicTrainerER(
        policy_net=policy_net,
        transform_spec=transform_spec,
        params=params,
        shock_params=shock_params,
        optimizer=optimizer,
        polyak_tau=method_config.polyak_tau,
        loss_type=method_config.loss_type,
    )

    validation_fn = _make_validation_fn_er(trainer) if validation_data is not None else None
    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="basic_er",
        validation_data=validation_data,
        validation_fn=validation_fn,
        required_batch_keys=set(spec.required_keys),
        batch_adapter=_batch_adapter_er,
    )

    result = TrainingResult(
        history=history,
        artifacts={
            "policy_net": policy_net,
            "target_policy_net": trainer.target_policy_net,
        },
        config={
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config,
        },
        params=params,
        meta={
            "model": "basic",
            "method": "basic_er",
            "io_transforms": transform_spec,
            "training_seed": list(training_seed) if training_seed is not None else None,
        },
    )
    return result.to_legacy_dict()


def train_basic_br_actor_critic(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    warm_start_policy: Optional[Any] = None,
    training_seed: Optional[Tuple[int, int]] = None,
) -> Union[Dict[str, Any], TrainingResult]:
    method_config = _coerce_method_config(
        method_config, expected_name="basic_br_actor_critic"
    )
    spec = get_method_spec("basic_br_actor_critic")
    validate_dataset_keys(dataset, spec.required_keys, method_name="basic_br_actor_critic")

    data_iter = build_training_iterator(
        dataset,
        batch_size=opt_config.batch_size,
        batch_order=spec.batch_order,
        permutation_seed=training_seed if spec.batch_order == "fixed_permutation" else None,
    )

    k_bounds = bounds["k"]
    logz_bounds = bounds["log_z"]
    observation_normalizer = _build_basic_observation_normalizer(
        dataset=dataset,
        net_config=net_config,
        bounds=bounds,
    )
    clip_cfg = net_config.inference_clips
    transform_spec = build_basic_transform_spec(
        normalizer=observation_normalizer,
        k_bounds=k_bounds,
        policy_head=net_config.basic_policy_head,
        value_head=net_config.basic_value_head,
        clip_policy_k_min=clip_cfg.basic_policy_k.clip_min,
        clip_policy_k_max=clip_cfg.basic_policy_k.clip_max,
        clip_value_min=clip_cfg.basic_value.clip_min,
        clip_value_max=clip_cfg.basic_value.clip_max,
    )
    policy_net, value_net = build_basic_networks(
        k_min=k_bounds[0],
        k_max=k_bounds[1],
        logz_min=logz_bounds[0],
        logz_max=logz_bounds[1],
        n_layers=net_config.n_layers,
        n_neurons=net_config.n_neurons,
        hidden_activation=net_config.hidden_activation,
    )
    warm_start_meta = _maybe_apply_policy_warm_start(policy_net, warm_start_policy)

    actor_lr = opt_config.learning_rate
    critic_lr = (
        opt_config.learning_rate_critic
        if opt_config.learning_rate_critic
        else opt_config.learning_rate
    )
    optimizer_actor = create_optimizer(actor_lr, opt_config.optimizer)
    optimizer_value = create_optimizer(critic_lr, opt_config.optimizer)
    br_scale = _resolve_br_scale(method_config, params, shock_params)

    trainer = BasicTrainerBR(
        policy_net=policy_net,
        value_net=value_net,
        transform_spec=transform_spec,
        params=params,
        shock_params=shock_params,
        optimizer_actor=optimizer_actor,
        optimizer_value=optimizer_value,
        n_critic_steps=method_config.n_critic,
        logit_clip=anneal_config.logit_clip,
        polyak_tau=method_config.polyak_tau,
        loss_type=method_config.loss_type,
        br_scale=br_scale,
    )

    validation_fn = _make_validation_fn_br(trainer) if validation_data is not None else None
    history = execute_training_loop(
        trainer,
        data_iter,
        opt_config,
        anneal_config,
        method_name="basic_br_actor_critic",
        validation_data=validation_data,
        validation_fn=validation_fn,
        required_batch_keys=set(spec.required_keys),
        batch_adapter=_batch_adapter_br,
    )

    meta = {
        "model": "basic",
        "method": "basic_br_actor_critic",
        "br_scale": br_scale,
        "warm_start_applied": warm_start_meta["warm_start_applied"],
        "io_transforms": transform_spec,
        "training_seed": list(training_seed) if training_seed is not None else None,
    }
    if warm_start_meta["warm_start_source"] is not None:
        meta["warm_start_source"] = warm_start_meta["warm_start_source"]

    result = TrainingResult(
        history=history,
        artifacts={
            "policy_net": policy_net,
            "value_net": value_net,
            "target_policy_net": trainer.target_policy_net,
            "target_value_net": trainer.target_value_net,
        },
        config={
            "network": net_config,
            "optimization": opt_config,
            "method": method_config,
            "annealing": anneal_config,
        },
        params=params,
        meta=meta,
    )
    return result.to_legacy_dict()


def train_basic_br(
    dataset: Dict[str, tf.Tensor],
    net_config: NetworkConfig,
    opt_config: OptimizationConfig,
    method_config: MethodConfig,
    anneal_config: AnnealingConfig,
    params: EconomicParams,
    shock_params: ShockParams,
    bounds: Dict[str, Tuple[float, float]],
    validation_data: Optional[Dict[str, tf.Tensor]] = None,
    warm_start_policy: Optional[Any] = None,
    training_seed: Optional[Tuple[int, int]] = None,
) -> Union[Dict[str, Any], TrainingResult]:
    """Backward-compatible alias for BR actor-critic."""
    return train_basic_br_actor_critic(
        dataset=dataset,
        net_config=net_config,
        opt_config=opt_config,
        method_config=method_config,
        anneal_config=anneal_config,
        params=params,
        shock_params=shock_params,
        bounds=bounds,
        validation_data=validation_data,
        warm_start_policy=warm_start_policy,
        training_seed=training_seed,
    )


