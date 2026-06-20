"""FiLM network tests (DF26 Sec 3.3; review TF-3 / Sec 11 silent-failure pitfalls)."""
import numpy as np
import tensorflow as tf

from src.v3 import config as cfg
from src.v3.networks import film as F
from src.v3.networks.bundle import NetworkBundle
from src.v3.networks.policy import PolicyNetwork
from src.v3.networks.value import PARAM_DIM, STATE_DIM

NETCFG = cfg.NetworkConfig()
SEED = 20260619


def _inputs(n=16):
    s = tf.random.stateless_normal([n, STATE_DIM], seed=[1, 1])
    p = tf.random.stateless_normal([n, PARAM_DIM], seed=[2, 2])
    return s, p


def test_identity_init_output_invariant_to_params():
    # At identity init (gamma=1, delta=0 for any param), output ignores params.
    net = F.FiLMNetwork(STATE_DIM, PARAM_DIM, NETCFG, SEED, net_id=0)
    s = tf.random.stateless_normal([16, STATE_DIM], seed=[1, 1])
    p1 = tf.random.stateless_normal([16, PARAM_DIM], seed=[2, 2])
    p2 = tf.random.stateless_normal([16, PARAM_DIM], seed=[3, 3])
    np.testing.assert_allclose(net(s, p1).numpy(), net(s, p2).numpy(), atol=1e-6)


def test_identity_init_gamma_one_delta_zero():
    net = F.FiLMNetwork(STATE_DIM, PARAM_DIM, NETCFG, SEED, net_id=0)
    _, p = _inputs()
    for l in range(NETCFG.trunk_layers):
        gamma, delta = net.film(p, l)
        assert tuple(gamma.shape) == (16, NETCFG.trunk_units)
        np.testing.assert_allclose(gamma.numpy(), 1.0, atol=1e-6)
        np.testing.assert_allclose(delta.numpy(), 0.0, atol=1e-6)


def test_per_sample_modulation_after_breaking_identity():
    net = F.FiLMNetwork(STATE_DIM, PARAM_DIM, NETCFG, SEED, net_id=0)
    for i, w in enumerate(net.gen_W2):
        w.assign(tf.random.stateless_normal(w.shape, seed=[10 + i, 11 + i]) * 0.1)
    s, p = _inputs()
    # gamma now varies across samples (not collapsed to one modulation).
    gamma, _ = net.film(p, 0)
    row_std = tf.math.reduce_std(gamma, axis=0)
    assert float(tf.reduce_max(row_std)) > 1e-4
    # output now depends on params.
    p2 = tf.random.stateless_normal([16, PARAM_DIM], seed=[4, 4])
    assert not np.allclose(net(s, p).numpy(), net(s, p2).numpy(), atol=1e-4)


def test_gradients_not_none(_break_identity=True):
    # review TF-3: every trainable variable must receive a gradient (not None).
    net = F.FiLMNetwork(STATE_DIM, PARAM_DIM, NETCFG, SEED, net_id=0)
    for i, w in enumerate(net.gen_W2):
        w.assign(tf.random.stateless_normal(w.shape, seed=[20 + i, 21 + i]) * 0.1)
    s, p = _inputs()
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(net(s, p))
    grads = tape.gradient(loss, net.trainable_variables)
    assert all(g is not None for g in grads)
    assert any(float(tf.reduce_max(tf.abs(g))) > 0 for g in grads)


def test_gradient_flows_to_params():
    net = F.FiLMNetwork(STATE_DIM, PARAM_DIM, NETCFG, SEED, net_id=0)
    for i, w in enumerate(net.gen_W2):
        w.assign(tf.random.stateless_normal(w.shape, seed=[30 + i, 31 + i]) * 0.1)
    s, p = _inputs()
    with tf.GradientTape() as tape:
        tape.watch(p)
        loss = tf.reduce_sum(net(s, p))
    gp = tape.gradient(loss, p)
    assert gp is not None
    assert float(tf.reduce_max(tf.abs(gp))) > 0


def test_policy_output_in_bounds():
    pol = PolicyNetwork(NETCFG, SEED, net_id=1)
    s, p = _inputs()
    a_min = tf.fill([16], -0.5)
    a_max = tf.fill([16], 1.5)
    out = pol(s, p, a_min, a_max).numpy()
    assert out.min() >= -0.5 - 1e-6 and out.max() <= 1.5 + 1e-6


def test_bprime_bias_lowers_initial_policy():
    s, p = _inputs()
    a_min, a_max = tf.fill([16], 0.0), tf.fill([16], 2.0)
    biased = PolicyNetwork(NETCFG, SEED, net_id=2, raw_bias=-4.0)
    plain = PolicyNetwork(NETCFG, SEED, net_id=2, raw_bias=0.0)
    assert float(tf.reduce_mean(biased(s, p, a_min, a_max))) < \
           float(tf.reduce_mean(plain(s, p, a_min, a_max)))


def test_bundle_has_twelve_distinct_generators():
    b = NetworkBundle(NETCFG, SEED)
    gens = []
    for net in (b.value.net, b.policy_i.net, b.policy_bp.net, b.policy_cp.net):
        gens.extend(net.gen_W1)
        gens.extend(net.gen_W2)
    # 4 networks x 3 layers x 2 generator matrices = 24 distinct variables.
    assert len({id(g) for g in gens}) == 24
    # 12 generators (3 per network x 4 networks).
    n_gen = sum(net.L for net in (b.value.net, b.policy_i.net, b.policy_bp.net, b.policy_cp.net))
    assert n_gen == 12


def test_bundle_target_sync_and_independent():
    b = NetworkBundle(NETCFG, SEED)
    s, p = _inputs()
    # target starts equal to value.
    np.testing.assert_allclose(b.value(s, p).numpy(), b.target(s, p).numpy(), atol=1e-6)
    # perturb value, target unchanged until synced.
    b.value.net.out_c.assign(b.value.net.out_c + 1.0)
    assert not np.allclose(b.value(s, p).numpy(), b.target(s, p).numpy(), atol=1e-3)
    b.sync_target()
    np.testing.assert_allclose(b.value(s, p).numpy(), b.target(s, p).numpy(), atol=1e-6)
