"""Checkpoint save/reload reproduces predictions (DF26 Sec 12.5; review NN-2)."""
import numpy as np
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.networks.bundle import NetworkBundle
from src.v3.networks.value import PARAM_DIM, STATE_DIM
from src.v3.output import checkpoints

NETCFG = cfg.NetworkConfig()


def test_save_reload_reproduces_predictions(tmp_path):
    b1 = NetworkBundle(NETCFG, master_seed=1)
    # Move off identity init so weights are non-trivial.
    for i, w in enumerate(b1.value.net.gen_W2):
        w.assign(tf.random.stateless_normal(w.shape, seed=[5 + i, 6 + i]) * 0.1)

    s = tf.random.stateless_normal([8, STATE_DIM], seed=[3, 3])
    p = tf.random.stateless_normal([8, PARAM_DIM], seed=[4, 4])
    a_min, a_max = tf.fill([8], 0.0), tf.fill([8], 2.0)
    v_before = b1.value(s, p).numpy()
    bp_before = b1.policy_bp(s, p, a_min, a_max).numpy()

    path = str(tmp_path / "ckpt")
    checkpoints.save_bundle(b1, path)

    b2 = NetworkBundle(NETCFG, master_seed=999)  # different init
    assert not np.allclose(b2.value(s, p).numpy(), v_before, atol=1e-4)

    checkpoints.load_bundle(b2, path)
    np.testing.assert_allclose(b2.value(s, p).numpy(), v_before, atol=1e-6)
    np.testing.assert_allclose(b2.policy_bp(s, p, a_min, a_max).numpy(), bp_before, atol=1e-6)
