"""Tests for the supported evaluation helpers."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.v2.evaluation import (
    build_action_grid_policy,
    load_method_bundle,
    load_plot_inputs,
    load_solver_bundle,
    prepare_evaluation_run,
    restore_selected_models,
    restore_selected_snapshot,
    save_manifest_sections,
    save_method_bundle,
    save_plot_inputs,
    save_solver_bundle,
)


class TestEvaluationRunArtifacts:
    def test_prepare_evaluation_run_creates_dirs_and_latest(self, tmp_path):
        ctx = prepare_evaluation_run(
            "demo_eval",
            save_run=True,
            results_root=str(tmp_path),
            run_tag="smoke",
        )
        assert ctx["run_dir"].exists()
        assert ctx["figures_dir"].exists()
        assert ctx["solver_dir"].exists()
        latest = Path(tmp_path) / "demo_eval" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == ctx["run_dir"]

    def test_plot_inputs_round_trip(self, tmp_path):
        ctx = prepare_evaluation_run("demo_eval", save_run=True, results_root=str(tmp_path))
        save_plot_inputs(ctx, "slice_grids", {"z_grid": [1.0, 2.0], "k_grid": [3.0, 4.0]})
        loaded = load_plot_inputs(ctx, "slice_grids")
        np.testing.assert_allclose(loaded["z_grid"], [1.0, 2.0])
        np.testing.assert_allclose(loaded["k_grid"], [3.0, 4.0])

    def test_solver_bundle_round_trip(self, tmp_path):
        ctx = prepare_evaluation_run("demo_eval", save_run=True, results_root=str(tmp_path))
        result = {
            "value": tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32),
            "policy_action": tf.constant(np.arange(6).reshape(2, 3, 1), dtype=tf.float32),
            "policy_endo": tf.constant(np.arange(6, 12).reshape(2, 3, 1), dtype=tf.float32),
            "prob_matrix": tf.constant(np.eye(2), dtype=tf.float32),
            "grids": {
                "exo_grids_1d": [np.array([1.0, 2.0])],
                "endo_grids_1d": [np.array([10.0, 20.0, 30.0])],
                "action_grids_1d": [np.array([-1.0, 0.0, 1.0])],
            },
            "converged": True,
            "n_iter": 4,
        }
        save_solver_bundle(ctx, result, summary={"wall_time_sec": 12.5})
        loaded = load_solver_bundle(ctx)
        np.testing.assert_allclose(
            loaded["result"]["policy_action"].numpy(),
            result["policy_action"].numpy(),
        )
        assert loaded["summary"]["n_iter"] == 4
        assert loaded["summary"]["wall_time_sec"] == 12.5

    def test_named_solver_bundle_round_trip(self, tmp_path):
        ctx = prepare_evaluation_run("demo_eval", save_run=True, results_root=str(tmp_path))
        result = {
            "value": tf.constant(np.arange(4).reshape(2, 2), dtype=tf.float32),
            "policy_action": tf.constant(np.arange(4).reshape(2, 2, 1), dtype=tf.float32),
            "policy_endo": tf.constant(np.arange(4, 8).reshape(2, 2, 1), dtype=tf.float32),
            "prob_matrix": tf.constant(np.eye(2), dtype=tf.float32),
            "grids": {
                "exo_grids_1d": [np.array([1.0, 2.0])],
                "endo_grids_1d": [np.array([10.0, 20.0])],
                "action_grids_1d": [np.array([-1.0, 1.0])],
            },
            "converged": True,
            "n_iter": 3,
        }
        save_solver_bundle(ctx, result, summary={"tau": 1e-3}, name="soft_tau_1e-3")
        loaded = load_solver_bundle(ctx, name="soft_tau_1e-3")
        np.testing.assert_allclose(
            loaded["result"]["policy_endo"].numpy(),
            result["policy_endo"].numpy(),
        )
        assert loaded["summary"]["tau"] == 1e-3

    def test_method_bundle_round_trip(self, tmp_path):
        ctx = prepare_evaluation_run("demo_eval", save_run=True, results_root=str(tmp_path))
        result = {
            "history": {"step": [0, 5], "loss": [1.0, 0.5], "elapsed_sec": [0.1, 0.2]},
            "config": {"n_steps": 10, "master_seed": [20, 26]},
            "converged": False,
            "best_step": 5,
            "wall_time_sec": 0.2,
        }
        checkpoint_history = [
            {
                "step": 0,
                "models": {
                    "policy": [
                        np.array([[1.0], [2.0]], dtype=np.float32),
                        np.array([0.1], dtype=np.float32),
                    ]
                },
            },
            {
                "step": 5,
                "models": {
                    "policy": [
                        np.array([[3.0], [4.0]], dtype=np.float32),
                        np.array([0.2], dtype=np.float32),
                    ],
                    "value_net": [
                        np.array([[7.0], [8.0]], dtype=np.float32),
                        np.array([0.3], dtype=np.float32),
                    ],
                },
            },
        ]
        checkpoint_metrics = {
            "step": [0, 5],
            "elapsed_sec": [0.1, 0.2],
            "euler_residual": [0.3, 0.2],
        }
        selected = {"selected_step": 5, "status": "best-within-cap"}
        save_method_bundle(
            ctx,
            "LR",
            result=result,
            checkpoint_history=checkpoint_history,
            checkpoint_metrics=checkpoint_metrics,
            selected=selected,
            extra_sections={"note": "roundtrip"},
        )
        loaded = load_method_bundle(ctx, "LR")
        assert loaded["result"]["best_step"] == 5
        assert loaded["result"]["note"] == "roundtrip"
        assert loaded["selected"]["selected_step"] == 5
        assert len(loaded["checkpoint_history"]) == 2
        assert "value_net" in loaded["checkpoint_history"][1]["models"]

    def test_manifest_sections_merge(self, tmp_path):
        ctx = prepare_evaluation_run("demo_eval", save_run=True, results_root=str(tmp_path))
        save_manifest_sections(ctx, setup={"profile": "BALANCED"})
        save_manifest_sections(ctx, solver={"grid": [200, 200, 2000]})
        with open(ctx["manifest_path"]) as f:
            manifest = json.load(f)
        assert manifest["setup"]["profile"] == "BALANCED"
        assert manifest["solver"]["grid"] == [200, 200, 2000]


class TestEvaluationPolicies:
    def test_build_action_grid_policy_interpolates(self):
        result = {
            "policy_action": tf.constant(
                [[[11.0], [21.0]], [[12.0], [22.0]]],
                dtype=tf.float32,
            ),
            "grids": {
                "exo_grids_1d": [np.array([1.0, 2.0])],
                "endo_grids_1d": [np.array([10.0, 20.0])],
            },
        }
        policy = build_action_grid_policy(result, action_dim=1)
        out = policy(tf.constant([[15.0, 1.5]], dtype=tf.float32)).numpy().reshape(-1)
        np.testing.assert_allclose(out, [16.5], atol=1e-6)

    def test_restore_selected_snapshot(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
        model(tf.zeros((1, 2)))
        weight_history = [
            (0, [np.array([[1.0], [2.0]], dtype=np.float32), np.array([0.1], dtype=np.float32)]),
            (5, [np.array([[3.0], [4.0]], dtype=np.float32), np.array([0.2], dtype=np.float32)]),
        ]
        restore_selected_snapshot(model, weight_history, selected_step=5)
        np.testing.assert_allclose(model.get_weights()[0], [[3.0], [4.0]])

    def test_restore_selected_models_from_checkpoint(self):
        policy = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
        value_net = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
        policy(tf.zeros((1, 2)))
        value_net(tf.zeros((1, 2)))

        checkpoint_history = [
            {
                "step": 5,
                "models": {
                    "policy": [
                        np.array([[3.0], [4.0]], dtype=np.float32),
                        np.array([0.2], dtype=np.float32),
                    ],
                    "value_net": [
                        np.array([[7.0], [8.0]], dtype=np.float32),
                        np.array([0.3], dtype=np.float32),
                    ],
                },
            },
        ]
        restore_selected_models(
            {"policy": policy, "value_net": value_net},
            checkpoint_history,
            selected_step=5,
        )
        np.testing.assert_allclose(policy.get_weights()[0], [[3.0], [4.0]])
        np.testing.assert_allclose(value_net.get_weights()[0], [[7.0], [8.0]])
