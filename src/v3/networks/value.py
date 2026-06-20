"""Value network: a FiLM network whose raw scalar output is V (no squashing, Sec 3.1)."""
from __future__ import annotations

import tensorflow as tf

from src.v3.config import NetworkConfig
from src.v3.networks.film import FiLMNetwork

STATE_DIM = 3   # (log z, k, b)
PARAM_DIM = 8   # the estimated parameter vector


class ValueNetwork(tf.Module):
    def __init__(self, cfg: NetworkConfig, master_seed: int, net_id: int = 0, name="value"):
        super().__init__(name=name)
        self.net = FiLMNetwork(STATE_DIM, PARAM_DIM, cfg, master_seed, net_id, name=name)

    def __call__(self, state_norm, param_norm):
        """Return V [N] for normalized states and parameters."""
        return self.net(state_norm, param_norm)
