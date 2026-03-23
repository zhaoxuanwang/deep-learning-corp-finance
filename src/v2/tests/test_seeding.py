"""Tests for reproducibility helpers and seeded training components."""

import numpy as np
import tensorflow as tf

from src.v2.data.generator import DataGenerator, DataGeneratorConfig
from src.v2.data.pipeline import build_iterator
from src.v2.environments.basic_investment import BasicInvestmentEnv
from src.v2.networks.policy import PolicyNetwork
from src.v2.networks.state_value import StateValueNetwork
from src.v2.trainers.core import warm_start_value_net
from src.v2.utils.seeding import fold_in_seed, make_seed_int


def _flatten_weights(model):
    return np.concatenate([
        w.numpy().reshape(-1) for w in model.trainable_weights
    ])


class TestSeedUtils:

    def test_fold_in_seed_is_deterministic(self):
        seed_1 = fold_in_seed((20, 26), "policy", 1)
        seed_2 = fold_in_seed((20, 26), "policy", 1)
        assert seed_1 == seed_2

    def test_fold_in_seed_changes_across_namespaces(self):
        seed_1 = fold_in_seed((20, 26), "policy")
        seed_2 = fold_in_seed((20, 26), "value")
        assert seed_1 != seed_2

    def test_make_seed_int_is_positive(self):
        value = make_seed_int((20, 26), "iterator")
        assert isinstance(value, int)
        assert value > 0


class TestSeededInitialization:

    def test_policy_seed_reproduces_initial_weights(self):
        seed = fold_in_seed((20, 26), "policy_init")
        net_1 = PolicyNetwork(
            state_dim=2, action_dim=1,
            action_low=tf.constant([-5.0]),
            action_high=tf.constant([5.0]),
            n_layers=2, n_neurons=16, seed=seed,
        )
        net_2 = PolicyNetwork(
            state_dim=2, action_dim=1,
            action_low=tf.constant([-5.0]),
            action_high=tf.constant([5.0]),
            n_layers=2, n_neurons=16, seed=seed,
        )
        dummy = tf.zeros((1, 2))
        net_1(dummy)
        net_2(dummy)
        np.testing.assert_allclose(
            _flatten_weights(net_1), _flatten_weights(net_2), atol=1e-7)

    def test_value_seed_reproduces_initial_weights(self):
        seed = fold_in_seed((20, 26), "value_init")
        net_1 = StateValueNetwork(
            state_dim=2, n_layers=2, n_neurons=16, seed=seed)
        net_2 = StateValueNetwork(
            state_dim=2, n_layers=2, n_neurons=16, seed=seed)
        dummy = tf.zeros((1, 2))
        net_1(dummy)
        net_2(dummy)
        np.testing.assert_allclose(
            _flatten_weights(net_1), _flatten_weights(net_2), atol=1e-7)


class TestSeededDataPipeline:

    def test_build_iterator_reproducible_with_shuffle_seed(self):
        dataset = {
            "x": tf.reshape(tf.range(32, dtype=tf.float32), [-1, 1]),
            "y": tf.range(32, dtype=tf.int32),
        }
        seed = make_seed_int((20, 26), "iterator")
        ds_1 = build_iterator(dataset, batch_size=8, shuffle_seed=seed)
        ds_2 = build_iterator(dataset, batch_size=8, shuffle_seed=seed)
        it_1 = iter(ds_1)
        it_2 = iter(ds_2)

        batches_1 = [next(it_1) for _ in range(3)]
        batches_2 = [next(it_2) for _ in range(3)]

        for b1, b2 in zip(batches_1, batches_2):
            np.testing.assert_array_equal(b1["y"].numpy(), b2["y"].numpy())


class TestSeededWarmStart:

    def test_warm_start_reproducible_with_explicit_seed(self):
        env = BasicInvestmentEnv()
        gen = DataGenerator(env, DataGeneratorConfig(n_paths=16, horizon=8))
        flat_dataset = gen.get_flattened_dataset("train")
        net_seed = fold_in_seed((20, 26), "warm_start", "value")
        shuffle_seed = make_seed_int((20, 26), "warm_start", "shuffle")

        value_1 = StateValueNetwork(
            state_dim=env.state_dim(), n_layers=2, n_neurons=16, seed=net_seed)
        value_2 = StateValueNetwork(
            state_dim=env.state_dim(), n_layers=2, n_neurons=16, seed=net_seed)
        dummy = tf.zeros((1, env.state_dim()))
        value_1(dummy)
        value_2(dummy)

        warm_start_value_net(
            env, value_1, flat_dataset,
            n_steps=5, batch_size=8, shuffle_seed=shuffle_seed)
        warm_start_value_net(
            env, value_2, flat_dataset,
            n_steps=5, batch_size=8, shuffle_seed=shuffle_seed)

        np.testing.assert_allclose(
            _flatten_weights(value_1), _flatten_weights(value_2), atol=1e-7)
