"""
Neural network builders for the Basic model.

Refactor note:
- Networks are plain FcNN blocks over normalized inputs.
- Economic input/output transforms are handled in trainers/inference helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf


_LEGACY_CALL_ERROR = (
    "Direct level-space model calls are deprecated. "
    "This raw network now expects a single normalized feature tensor "
    "with shape (batch, 2). Use trainer io transforms/inference helper for "
    "level-space inputs and outputs."
)


class _HiddenStackMixin:
    def _init_hidden_stack(
        self,
        *,
        n_layers: int,
        n_neurons: int,
        hidden_activation: str,
    ) -> None:
        self.hidden_activation_fn = tf.keras.activations.get(hidden_activation)
        self.hidden_dense_layers: List[tf.keras.layers.Layer] = []
        for _ in range(n_layers):
            self.hidden_dense_layers.append(
                tf.keras.layers.Dense(
                    n_neurons,
                    activation=None,
                    use_bias=True,
                )
            )

    def _forward_hidden(self, x: tf.Tensor) -> tf.Tensor:
        for dense in self.hidden_dense_layers:
            x = dense(x)
            x = self.hidden_activation_fn(x)
        return x


class _RawNetworkBase(tf.keras.Model):
    input_dim: int = 0

    @staticmethod
    def _coerce_extra_inputs(
        positional_extra_inputs: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> tuple[Any, ...]:
        kw_extra = kwargs.pop("extra_inputs", ())
        kwargs.pop("mask", None)

        if kw_extra is None:
            kw_tuple: tuple[Any, ...] = ()
        elif isinstance(kw_extra, tuple):
            kw_tuple = kw_extra
        elif isinstance(kw_extra, list):
            kw_tuple = tuple(kw_extra)
        else:
            kw_tuple = (kw_extra,)

        return tuple(positional_extra_inputs) + kw_tuple

    def _expect_normalized_features(
        self,
        inputs: Any,
        extra_inputs: tuple[Any, ...],
    ) -> tf.Tensor:
        if extra_inputs:
            raise RuntimeError(_LEGACY_CALL_ERROR)

        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if x.shape.rank == 1:
            x = tf.reshape(x, [-1, self.input_dim])
        if x.shape.rank != 2:
            raise ValueError(
                f"Expected normalized features of rank 2, got rank={x.shape.rank}."
            )
        if x.shape[-1] is not None and int(x.shape[-1]) != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {x.shape[-1]}."
            )
        return x


class BasicPolicyNetwork(_HiddenStackMixin, _RawNetworkBase):
    """
    Raw policy network for the Basic model.

    Input: normalized (k, z) feature tensor of shape (batch, 2)
    Output: raw scalar tensor for k' head, shape (batch, 1)
    """

    input_dim = 2

    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        hidden_activation: str,
        **legacy_config: Any,
    ):
        super().__init__()
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._hidden_activation = hidden_activation
        self._legacy_config = dict(legacy_config)

        self._init_hidden_stack(
            n_layers=n_layers,
            n_neurons=n_neurons,
            hidden_activation=hidden_activation,
        )
        self.output_layer = tf.keras.layers.Dense(1, activation=None, name="policy_raw")

    def call(
        self,
        inputs: Any,
        *extra_inputs: Any,
        training: bool = False,
        **kwargs: Any,
    ) -> tf.Tensor:
        del training  # No BN/dropout in raw network.
        extra_inputs = self._coerce_extra_inputs(extra_inputs, kwargs)
        x = self._expect_normalized_features(inputs, extra_inputs)
        x = self._forward_hidden(x)
        return self.output_layer(x)

    def get_config(self):
        cfg = {
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
            "hidden_activation": self._hidden_activation,
        }
        cfg.update(self._legacy_config)
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_from_config(self, config):
        del config
        self(tf.constant([[0.0, 0.0]], dtype=tf.float32), training=False)

    def get_build_config(self):
        return self.get_config()


class BasicValueNetwork(_HiddenStackMixin, _RawNetworkBase):
    """
    Raw value network for the Basic model.

    Input: normalized (k, z) feature tensor of shape (batch, 2)
    Output: raw scalar tensor for value head, shape (batch, 1)
    """

    input_dim = 2

    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        hidden_activation: str,
        **legacy_config: Any,
    ):
        super().__init__()
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._hidden_activation = hidden_activation
        self._legacy_config = dict(legacy_config)

        self._init_hidden_stack(
            n_layers=n_layers,
            n_neurons=n_neurons,
            hidden_activation=hidden_activation,
        )
        self.output_layer = tf.keras.layers.Dense(1, activation=None, name="value_raw")

    def call(
        self,
        inputs: Any,
        *extra_inputs: Any,
        training: bool = False,
        **kwargs: Any,
    ) -> tf.Tensor:
        del training
        extra_inputs = self._coerce_extra_inputs(extra_inputs, kwargs)
        x = self._expect_normalized_features(inputs, extra_inputs)
        x = self._forward_hidden(x)
        return self.output_layer(x)

    def get_config(self):
        cfg = {
            "n_layers": self._n_layers,
            "n_neurons": self._n_neurons,
            "hidden_activation": self._hidden_activation,
        }
        cfg.update(self._legacy_config)
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_from_config(self, config):
        del config
        self(tf.constant([[0.0, 0.0]], dtype=tf.float32), training=False)

    def get_build_config(self):
        return self.get_config()


def build_basic_networks(
    k_min: float,
    k_max: float,
    logz_min: float,
    logz_max: float,
    n_layers: int,
    n_neurons: int,
    hidden_activation: str,
    use_batch_norm: Optional[bool] = None,
    policy_head: Optional[str] = None,
    value_head: Optional[str] = None,
    allow_unsafe_heads: Optional[bool] = None,
    observation_normalizer: Optional[Dict[str, Any]] = None,
) -> Tuple[BasicPolicyNetwork, BasicValueNetwork]:
    """
    Factory to build Basic model policy/value networks.

    Legacy parameters are accepted for checkpoint compatibility but ignored by
    the raw model architecture.
    """
    legacy = {
        "k_min": k_min,
        "k_max": k_max,
        "logz_min": logz_min,
        "logz_max": logz_max,
        "use_batch_norm": use_batch_norm,
        "policy_head": policy_head,
        "value_head": value_head,
        "allow_unsafe_heads": allow_unsafe_heads,
        "observation_normalizer": observation_normalizer,
    }

    policy = BasicPolicyNetwork(
        n_layers=n_layers,
        n_neurons=n_neurons,
        hidden_activation=hidden_activation,
        **legacy,
    )
    value = BasicValueNetwork(
        n_layers=n_layers,
        n_neurons=n_neurons,
        hidden_activation=hidden_activation,
        **legacy,
    )

    dummy_x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    _ = policy(dummy_x, training=False)
    _ = value(dummy_x, training=False)
    return policy, value
