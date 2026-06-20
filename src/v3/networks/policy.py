"""Policy network: a FiLM network whose raw scalar is squashed to a feasible box.

Each control sub-network outputs a raw scalar ``r`` mapped by Eq 26:

    a = a_min + (a_max - a_min) * sigmoid(r + raw_bias),

so ``a`` stays strictly inside ``[a_min, a_max]``. The gross-debt network uses
``raw_bias = -4`` to bias the initial debt policy low (Sec 3.3). The feasible
bounds ``[a_min, a_max]`` are passed in at call time because the investment-rate
bounds depend on the current capital (Sec 3.2).
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.config import NetworkConfig
from src.v3.networks.film import FiLMNetwork
from src.v3.networks.value import PARAM_DIM, STATE_DIM


def squash(raw, a_min, a_max, raw_bias=0.0):
    """Map a raw scalar to ``[a_min, a_max]`` via a biased sigmoid (Eq 26)."""
    return a_min + (a_max - a_min) * tf.sigmoid(raw + raw_bias)


class PolicyNetwork(tf.Module):
    def __init__(self, cfg: NetworkConfig, master_seed: int, net_id: int,
                 raw_bias: float = 0.0, name="policy"):
        super().__init__(name=name)
        self.raw_bias = raw_bias
        self.net = FiLMNetwork(STATE_DIM, PARAM_DIM, cfg, master_seed, net_id, name=name)

    def raw(self, state_norm, param_norm):
        return self.net(state_norm, param_norm)

    def __call__(self, state_norm, param_norm, a_min, a_max):
        """Return the control in ``[a_min, a_max]`` (Eq 26)."""
        return squash(self.net(state_norm, param_norm), a_min, a_max, self.raw_bias)
