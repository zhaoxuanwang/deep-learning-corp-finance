import numpy as np
import pytest
import tensorflow as tf

from src.networks.network_basic import build_basic_networks
from src.trainers.results import TrainingResult
from src.trainers.warm_start import copy_policy_weights, extract_policy_model


def _build_basic_policy_value(
    *,
    k_min: float = 0.1,
    k_max: float = 10.0,
    logz_min: float = -0.5,
    logz_max: float = 0.5,
    n_layers: int = 2,
    n_neurons: int = 16,
):
    return build_basic_networks(
        k_min=k_min,
        k_max=k_max,
        logz_min=logz_min,
        logz_max=logz_max,
        n_layers=n_layers,
        n_neurons=n_neurons,
        activation="swish",
    )


def test_extract_policy_model_from_training_result():
    policy_net, _ = _build_basic_policy_value()
    result = TrainingResult(history={}, artifacts={"policy_net": policy_net})
    extracted = extract_policy_model(result)
    assert extracted is policy_net


def test_extract_policy_model_from_legacy_dict():
    policy_net, _ = _build_basic_policy_value()
    extracted = extract_policy_model({"_policy_net": policy_net})
    assert extracted is policy_net


def test_copy_policy_weights_success():
    source_policy, _ = _build_basic_policy_value()
    target_policy, _ = _build_basic_policy_value()

    source_weights = [w + 0.123 for w in source_policy.get_weights()]
    source_policy.set_weights(source_weights)

    copy_policy_weights(source_policy, target_policy)

    for src_w, tgt_w in zip(source_policy.get_weights(), target_policy.get_weights()):
        assert np.allclose(src_w, tgt_w)


def test_copy_policy_weights_reports_class_mismatch():
    source_policy, source_value = _build_basic_policy_value()
    target_policy, _ = _build_basic_policy_value()
    _ = source_policy

    with pytest.raises(ValueError, match="model class mismatch"):
        copy_policy_weights(source_value, target_policy)


def test_copy_policy_weights_reports_bounds_mismatch():
    source_policy, _ = _build_basic_policy_value(k_min=0.1, k_max=10.0)
    target_policy, _ = _build_basic_policy_value(k_min=0.2, k_max=10.0)

    with pytest.raises(ValueError) as exc_info:
        copy_policy_weights(source_policy, target_policy)

    message = str(exc_info.value)
    assert "config mismatch 'k_min'" in message


def test_copy_policy_weights_reports_shape_mismatch():
    source_policy, _ = _build_basic_policy_value(n_neurons=8)
    target_policy, _ = _build_basic_policy_value(n_neurons=16)

    with pytest.raises(ValueError) as exc_info:
        copy_policy_weights(source_policy, target_policy)

    message = str(exc_info.value)
    assert "weight shape mismatch" in message or "config mismatch 'n_neurons'" in message

