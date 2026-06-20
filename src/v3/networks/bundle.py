"""Bundle of the four FiLM networks plus a target value network (Sec 3.1, 3.5).

Holds V and the three policy networks (i, b', c'), plus a frozen target copy of V
used only for the bond-price default probability (Sec 3.5). ``sync_target`` resets
the target to the current value weights (done at each epoch boundary).
``snapshot``/``restore`` support the WeightStore seam for the collector boundary
(Sec 7).
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.config import NetworkConfig
from src.v3.networks.policy import PolicyNetwork
from src.v3.networks.value import ValueNetwork

# Stable per-network ids (key the weight-init seeds).
_ID_VALUE, _ID_I, _ID_BP, _ID_CP, _ID_TARGET = 0, 1, 2, 3, 4


class NetworkBundle(tf.Module):
    def __init__(self, cfg: NetworkConfig, master_seed: int,
                 bprime_init_bias: float = -4.0, name="bundle"):
        super().__init__(name=name)
        self.value = ValueNetwork(cfg, master_seed, net_id=_ID_VALUE)
        self.policy_i = PolicyNetwork(cfg, master_seed, net_id=_ID_I, raw_bias=0.0)
        self.policy_bp = PolicyNetwork(cfg, master_seed, net_id=_ID_BP, raw_bias=bprime_init_bias)
        self.policy_cp = PolicyNetwork(cfg, master_seed, net_id=_ID_CP, raw_bias=0.0)
        self.target = ValueNetwork(cfg, master_seed, net_id=_ID_TARGET)
        self.sync_target()

    def sync_target(self) -> None:
        """Copy the current value-network weights into the target network."""
        for t, v in zip(self.target.variables, self.value.variables):
            t.assign(v)

    @property
    def value_variables(self):
        return list(self.value.trainable_variables)

    @property
    def policy_variables(self):
        return (list(self.policy_i.trainable_variables)
                + list(self.policy_bp.trainable_variables)
                + list(self.policy_cp.trainable_variables))

    def snapshot(self):
        """Read-only copies of all weights (for a collector snapshot)."""
        return [v.read_value() for v in self.variables]

    def restore(self, snap) -> None:
        for v, s in zip(self.variables, snap):
            v.assign(s)
