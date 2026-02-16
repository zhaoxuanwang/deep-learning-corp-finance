import tensorflow as tf
import pytest

from src.economy.parameters import EconomicParams, ShockParams
from src.trainers import api as trainer_api
from src.trainers import basic_api, risky_api
from src.trainers.api import train
from src.trainers.config import (
    AnnealingConfig,
    ExperimentConfig,
    MethodConfig,
    NetworkConfig,
    OptimizationConfig,
    RiskyDebtConfig,
)
from src.trainers.results import TrainingResult


def _make_experiment_config(method_name: str, *, risky: bool = False) -> ExperimentConfig:
    method = MethodConfig(
        name=method_name,
        risky=RiskyDebtConfig() if risky else None,
    )
    return ExperimentConfig(
        params=EconomicParams(),
        shock_params=ShockParams(),
        network=NetworkConfig(),
        optimization=OptimizationConfig(n_iter=1, batch_size=2, log_every=1),
        annealing=AnnealingConfig(temperature_init=1.0, temperature_min=1e-4, decay=0.5),
        method=method,
        name="test",
    )


def test_train_dispatch_returns_training_result(monkeypatch):
    cfg = _make_experiment_config("basic_br")
    warm_start_token = object()
    captured = {}

    def fake_train_basic_br(**kwargs):
        captured["warm_start_policy"] = kwargs.get("warm_start_policy")
        return {
            "history": {"loss_critic": [1.0]},
            "_policy_net": "policy",
            "_value_net": "value",
            "_configs": {"method": kwargs["method_config"]},
            "_params": kwargs["params"],
        }

    monkeypatch.setattr(trainer_api, "train_basic_br", fake_train_basic_br)

    result = train(
        model="basic",
        method="br",
        config=cfg,
        train_data={},
        bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
        warm_start_policy=warm_start_token,
    )

    assert isinstance(result, TrainingResult)
    assert result.history["loss_critic"] == [1.0]
    assert result.artifacts["policy_net"] == "policy"
    assert result.artifacts["value_net"] == "value"
    assert captured["warm_start_policy"] is warm_start_token


def test_train_dispatch_basic_br_reg_returns_training_result(monkeypatch):
    cfg = _make_experiment_config("basic_br_reg")

    def fake_train_basic_br_reg(**kwargs):
        return {
            "history": {"loss_BR_reg": [0.5]},
            "_policy_net": "policy",
            "_value_net": "value",
            "_configs": {"method": kwargs["method_config"]},
            "_params": kwargs["params"],
        }

    monkeypatch.setattr(trainer_api, "train_basic_br_reg", fake_train_basic_br_reg)

    result = train(
        model="basic",
        method="br_reg",
        config=cfg,
        train_data={},
        bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
    )

    assert isinstance(result, TrainingResult)
    assert result.history["loss_BR_reg"] == [0.5]
    assert result.artifacts["policy_net"] == "policy"
    assert result.artifacts["value_net"] == "value"


def test_train_can_return_legacy_dict(monkeypatch):
    cfg = _make_experiment_config("basic_lr")

    def fake_train_basic_lr(**kwargs):
        return {
            "history": {"loss_LR": [1.0]},
            "_policy_net": "policy",
            "_configs": {"method": kwargs["method_config"]},
            "_params": kwargs["params"],
        }

    monkeypatch.setattr(trainer_api, "train_basic_lr", fake_train_basic_lr)

    result = train(
        model="basic",
        method="lr",
        config=cfg,
        train_data={},
        bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
        return_legacy_dict=True,
    )

    assert isinstance(result, dict)
    assert "history" in result
    assert "_policy_net" in result


def test_train_method_arg_must_match_config_method():
    cfg = _make_experiment_config("basic_er")
    with pytest.raises(ValueError, match="Method mismatch"):
        train(
            model="basic",
            method="br",
            config=cfg,
            train_data={},
            bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
        )


def test_train_rejects_warm_start_for_non_br_method(monkeypatch):
    cfg = _make_experiment_config("basic_lr")

    def fake_train_basic_lr(**kwargs):
        return {
            "history": {"loss_LR": [1.0]},
            "_policy_net": "policy",
            "_configs": {"method": kwargs["method_config"]},
            "_params": kwargs["params"],
        }

    monkeypatch.setattr(trainer_api, "train_basic_lr", fake_train_basic_lr)

    with pytest.raises(ValueError, match="warm_start_policy is only supported for basic BR methods"):
        train(
            model="basic",
            method="lr",
            config=cfg,
            train_data={},
            bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
            warm_start_policy=object(),
        )


def test_basic_wrapper_rejects_method_config_mismatch():
    with pytest.raises(ValueError, match="does not match requested method"):
        basic_api.train_basic_br(
            dataset={},
            net_config=NetworkConfig(),
            opt_config=OptimizationConfig(n_iter=1, batch_size=2),
            method_config=MethodConfig(name="basic_er"),
            anneal_config=AnnealingConfig(),
            params=EconomicParams(),
            shock_params=ShockParams(),
            bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
        )


def test_basic_br_reg_wrapper_rejects_method_config_mismatch():
    with pytest.raises(ValueError, match="does not match requested method"):
        basic_api.train_basic_br_reg(
            dataset={},
            net_config=NetworkConfig(),
            opt_config=OptimizationConfig(n_iter=1, batch_size=2),
            method_config=MethodConfig(name="basic_br"),
            anneal_config=AnnealingConfig(),
            params=EconomicParams(),
            shock_params=ShockParams(),
            bounds={"k": (0.1, 1.0), "log_z": (-0.1, 0.1)},
        )


def test_risky_wrapper_rejects_method_config_mismatch():
    with pytest.raises(ValueError, match="does not match requested method"):
        risky_api.train_risky_br(
            dataset={},
            net_config=NetworkConfig(),
            opt_config=OptimizationConfig(n_iter=1, batch_size=2),
            method_config=MethodConfig(name="basic_br", risky=RiskyDebtConfig()),
            anneal_config=AnnealingConfig(),
            params=EconomicParams(),
            shock_params=ShockParams(),
            bounds={"k": (0.1, 1.0), "b": (0.0, 1.0), "log_z": (-0.1, 0.1)},
        )


def test_risky_api_uses_core_owned_annealing(monkeypatch):
    captured = {}

    class FakeRiskyTrainer:
        def __init__(self, **kwargs):
            captured["smoothing"] = kwargs.get("smoothing")
            self.target_policy_net = "tp"
            self.target_value_net = "tv"
            self.target_price_net = "tpr"

    def fake_build_risky_networks(**kwargs):
        return "policy", "value", "price"

    def fake_execute_training_loop(*args, **kwargs):
        return {"temperature": [1.0]}

    monkeypatch.setattr(risky_api, "RiskyDebtTrainerBR", FakeRiskyTrainer)
    monkeypatch.setattr(risky_api, "build_risky_networks", fake_build_risky_networks)
    monkeypatch.setattr(risky_api, "execute_training_loop", fake_execute_training_loop)

    dataset = {
        "k": tf.constant([0.5, 0.6], dtype=tf.float32),
        "b": tf.constant([0.1, 0.2], dtype=tf.float32),
        "z": tf.constant([1.0, 1.1], dtype=tf.float32),
        "z_next_main": tf.constant([1.0, 1.2], dtype=tf.float32),
        "z_next_fork": tf.constant([0.9, 1.1], dtype=tf.float32),
    }

    risky_api.train_risky_br(
        dataset=dataset,
        net_config=NetworkConfig(),
        opt_config=OptimizationConfig(n_iter=1, batch_size=2),
        method_config=MethodConfig(name="risky_br", risky=RiskyDebtConfig()),
        anneal_config=AnnealingConfig(),
        params=EconomicParams(),
        shock_params=ShockParams(),
        bounds={"k": (0.1, 1.0), "b": (0.0, 1.0), "log_z": (-0.1, 0.1)},
    )

    assert captured["smoothing"] is None
