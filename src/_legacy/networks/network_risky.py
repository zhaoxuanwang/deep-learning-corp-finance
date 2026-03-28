"""
Neural network builders for the Risky Debt model.

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
    "with shape (batch, 3). Use trainer io transforms/inference helper for "
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


class RiskyPolicyNetwork(_HiddenStackMixin, _RawNetworkBase):
    """
    Raw policy network for Risky Debt model.

    Input: normalized (k, b, z) feature tensor of shape (batch, 3)
    Output: (raw_k, raw_b), each shape (batch, 1)
    """

    input_dim = 3

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
        self.k_head = tf.keras.layers.Dense(1, activation=None, name="policy_k_raw")
        self.b_head = tf.keras.layers.Dense(1, activation=None, name="policy_b_raw")

    def call(
        self,
        inputs: Any,
        *extra_inputs: Any,
        training: bool = False,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        del training
        extra_inputs = self._coerce_extra_inputs(extra_inputs, kwargs)
        x = self._expect_normalized_features(inputs, extra_inputs)
        x = self._forward_hidden(x)
        return self.k_head(x), self.b_head(x)

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
        self(tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32), training=False)

    def get_build_config(self):
        return self.get_config()


class RiskyValueNetwork(_HiddenStackMixin, _RawNetworkBase):
    """
    Raw value network for Risky Debt model.

    Input: normalized (k, b, z) feature tensor of shape (batch, 3)
    Output: raw scalar tensor for value head, shape (batch, 1)
    """

    input_dim = 3

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
        self(tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32), training=False)

    def get_build_config(self):
        return self.get_config()


class RiskyPriceNetwork(_HiddenStackMixin, _RawNetworkBase):
    """
    Raw bond-price network for Risky Debt model.

    Input: normalized (k', b', z) feature tensor of shape (batch, 3)
    Output: raw scalar tensor for q head, shape (batch, 1)
    """

    input_dim = 3

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
        self.output_layer = tf.keras.layers.Dense(1, activation=None, name="price_raw")

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
        self(tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32), training=False)

    def get_build_config(self):
        return self.get_config()


def build_risky_networks(
    k_min: float,
    k_max: float,
    b_min: float,
    b_max: float,
    logz_min: float,
    logz_max: float,
    r_risk_free: float,
    n_layers: int,
    n_neurons: int,
    hidden_activation: str,
    use_batch_norm: Optional[bool] = None,
    policy_k_head: Optional[str] = None,
    policy_b_head: Optional[str] = None,
    value_head: Optional[str] = None,
    price_head: Optional[str] = None,
    allow_unsafe_heads: Optional[bool] = None,
    observation_normalizer: Optional[Dict[str, Any]] = None,
) -> Tuple[RiskyPolicyNetwork, RiskyValueNetwork, RiskyPriceNetwork]:
    """
    Factory to build Risky Debt model networks.

    Legacy parameters are accepted for checkpoint compatibility but ignored by
    the raw model architecture.
    """
    legacy = {
        "k_min": k_min,
        "k_max": k_max,
        "b_min": b_min,
        "b_max": b_max,
        "logz_min": logz_min,
        "logz_max": logz_max,
        "r_risk_free": r_risk_free,
        "use_batch_norm": use_batch_norm,
        "policy_k_head": policy_k_head,
        "policy_b_head": policy_b_head,
        "value_head": value_head,
        "price_head": price_head,
        "allow_unsafe_heads": allow_unsafe_heads,
        "observation_normalizer": observation_normalizer,
    }

    policy = RiskyPolicyNetwork(
        n_layers=n_layers,
        n_neurons=n_neurons,
        hidden_activation=hidden_activation,
        **legacy,
    )
    value = RiskyValueNetwork(
        n_layers=n_layers,
        n_neurons=n_neurons,
        hidden_activation=hidden_activation,
        **legacy,
    )
    price = RiskyPriceNetwork(
        n_layers=n_layers,
        n_neurons=n_neurons,
        hidden_activation=hidden_activation,
        **legacy,
    )

    dummy_x = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    _ = policy(dummy_x, training=False)
    _ = value(dummy_x, training=False)
    _ = price(dummy_x, training=False)

    return policy, value, price


def apply_limited_liability(
    V_tilde: tf.Tensor,
    leaky: bool = False,
    alpha: float = 0.01
) -> tf.Tensor:
    """
    Apply limited liability: V = max{0, V_tilde}.

    By default uses ReLU. Optionally use leaky ReLU to preserve
    non-zero gradients when V_tilde < 0 (useful for actor training).

    Note: For training, prefer compute_effective_value() which uses
    smooth Gumbel-Sigmoid approximation.
    """
    if leaky:
        return tf.nn.leaky_relu(V_tilde, alpha=alpha)
    return tf.nn.relu(V_tilde)


def compute_effective_value(
    V_tilde: tf.Tensor,
    k: tf.Tensor,
    temperature: float,
    logit_clip: float = 20.0,
    noise: bool = True,
    noise_seed: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute effective continuation value with smooth default probability.
    """
    from src.utils.annealing import indicator_default

    safe_k = tf.maximum(k, 1e-8)
    V_norm = V_tilde / safe_k

    p_default = indicator_default(
        V_norm,
        temperature,
        logit_clip=logit_clip,
        noise=noise,
        noise_seed=noise_seed
    )

    V_eff = (1.0 - p_default) * V_tilde
    return V_eff, p_default
