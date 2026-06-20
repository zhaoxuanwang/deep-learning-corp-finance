"""FiLM network: a state trunk modulated by parameter-driven generators (Sec 3.3).

One network = one trunk (3 x 128 SiLU, linear head) + one FiLM generator per trunk
hidden layer (1 x 32 SiLU -> 2 x trunk_units, split into scale gamma and shift
delta). The modulation is applied to the PRE-activation (Eq 25):

    h_{l+1} = SiLU(gamma_l * (W_l h_l + c_l) + delta_l).

Identity initialization (Sec 3.3, Sec 11 pitfall): generator output weights are
zero and the output bias is ``[1...1, 0...0]``, so gamma_l = 1, delta_l = 0 at the
start and each FiLM layer is a no-op, leaving the trunk's initial behavior intact.
"""
from __future__ import annotations

import math

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.precision import TF_FLOAT_NET
from src.v3.config import NetworkConfig


def silu(x):
    """SiLU activation phi(x) = x * sigmoid(x) (Sec 3.3)."""
    return x * tf.sigmoid(x)


def _glorot(shape, master, net_id, slot, layer, dtype):
    """Seeded Glorot-uniform initializer as a ``tf.Variable`` (stateless seed)."""
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    init = seeding.uniform(shape, master, seeding.Purpose.INIT, net_id, slot, layer,
                           minval=-limit, maxval=limit, dtype=dtype)
    return tf.Variable(init)


# Integer slot codes (keep stable; they key the init seed).
_TRUNK_W, _TRUNK_OUT, _GEN_W1, _GEN_W2 = 0, 1, 2, 3


class FiLMNetwork(tf.Module):
    """Trunk + per-layer FiLM generators, returning a raw scalar [N]."""

    def __init__(self, state_dim: int, param_dim: int, cfg: NetworkConfig,
                 master_seed: int, net_id: int, dtype=TF_FLOAT_NET, name=None):
        super().__init__(name=name)
        self.L = cfg.trunk_layers
        U = cfg.trunk_units
        H = cfg.film_hidden
        self.dtype_ = dtype

        # Trunk: state_dim -> U -> ... -> U, then linear head U -> 1.
        in_dims = [state_dim] + [U] * (self.L - 1)
        self.trunk_W = [_glorot([in_dims[l], U], master_seed, net_id, _TRUNK_W, l, dtype)
                        for l in range(self.L)]
        self.trunk_c = [tf.Variable(tf.zeros([U], dtype)) for _ in range(self.L)]
        self.out_W = _glorot([U, 1], master_seed, net_id, _TRUNK_OUT, 0, dtype)
        self.out_c = tf.Variable(tf.zeros([1], dtype))

        # One generator per trunk hidden layer: param_dim -> H (SiLU) -> 2U.
        self.gen_W1 = [_glorot([param_dim, H], master_seed, net_id, _GEN_W1, l, dtype)
                       for l in range(self.L)]
        self.gen_c1 = [tf.Variable(tf.zeros([H], dtype)) for _ in range(self.L)]
        # Identity init: zero output weights, bias = [1...1 (U), 0...0 (U)].
        self.gen_W2 = [tf.Variable(tf.zeros([H, 2 * U], dtype)) for _ in range(self.L)]
        id_bias = tf.concat([tf.ones([U], dtype), tf.zeros([U], dtype)], axis=0)
        self.gen_c2 = [tf.Variable(tf.identity(id_bias)) for _ in range(self.L)]

    def film(self, param_norm, layer: int):
        """Return per-sample (gamma, delta), each ``[N, U]`` (Sec 3.3)."""
        g_hidden = silu(tf.matmul(param_norm, self.gen_W1[layer]) + self.gen_c1[layer])
        g_out = tf.matmul(g_hidden, self.gen_W2[layer]) + self.gen_c2[layer]
        U = self.trunk_c[layer].shape[0]
        return g_out[..., :U], g_out[..., U:]

    def __call__(self, state_norm, param_norm):
        h = state_norm
        for l in range(self.L):
            pre = tf.matmul(h, self.trunk_W[l]) + self.trunk_c[l]
            gamma, delta = self.film(param_norm, l)
            h = silu(gamma * pre + delta)            # modulate pre-activation (Eq 25)
        return tf.squeeze(tf.matmul(h, self.out_W) + self.out_c, axis=-1)  # [N]
