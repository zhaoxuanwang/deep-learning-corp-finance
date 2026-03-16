"""Tests for v2 checkpointing and dataset storage."""

import json
import os
import tempfile

import numpy as np
import tensorflow as tf
import pytest

from src.v2.utils.checkpointing import TrainingResult, save_run, load_run
from src.v2.data.storage import save_dataset, load_dataset, load_manifest


# =====================================================================
# TrainingResult + save_run / load_run
# =====================================================================

class TestTrainingResult:
    """Tests for TrainingResult construction."""

    def test_from_trainer_output(self):
        """from_trainer_output populates method, config, history."""
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            lr: float = 0.001
            steps: int = 100

        output = {
            "policy": None,
            "history": {"loss": [1.0, 0.5, 0.1]},
            "config": FakeConfig(),
        }
        result = TrainingResult.from_trainer_output(output, method="lr",
                                                     wall_time=42.5)
        assert result.method == "lr"
        assert result.config["lr"] == 0.001
        assert result.history["loss"] == [1.0, 0.5, 0.1]
        assert result.metadata["wall_time_seconds"] == 42.5
        assert "timestamp" in result.metadata

    def test_from_trainer_output_no_wall_time(self):
        """wall_time is optional."""
        from dataclasses import dataclass

        @dataclass
        class Cfg:
            x: int = 1

        output = {"policy": None, "history": {"a": [1]}, "config": Cfg()}
        result = TrainingResult.from_trainer_output(output, method="er")
        assert "wall_time_seconds" not in result.metadata


class TestSaveLoadRun:
    """Tests for save_run / load_run round-trip."""

    def test_round_trip_without_weights(self):
        """Config, history, metadata survive save/load."""
        result = TrainingResult(
            method="brm",
            config={"lr": 0.001, "batch_size": 256},
            history={"loss": [1.0, 0.8, 0.6], "step": [0, 100, 200]},
            metadata={"method": "brm", "timestamp": "2026-03-14T00:00:00"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "test_run")
            save_run(result, run_dir)

            loaded = load_run(run_dir)
            assert loaded.method == "brm"
            assert loaded.config["lr"] == 0.001
            assert loaded.history["loss"] == [1.0, 0.8, 0.6]
            assert loaded.history["step"] == [0, 100, 200]
            assert loaded.run_dir == run_dir

    def test_saved_files_exist(self):
        """save_run creates config.json, metadata.json, history.npz."""
        result = TrainingResult(
            method="lr",
            config={"a": 1},
            history={"loss": [0.5]},
            metadata={"method": "lr"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            save_run(result, run_dir)

            assert os.path.isfile(os.path.join(run_dir, "config.json"))
            assert os.path.isfile(os.path.join(run_dir, "metadata.json"))
            assert os.path.isfile(os.path.join(run_dir, "history.npz"))
            assert os.path.isdir(os.path.join(run_dir, "weights"))

    def test_config_is_human_readable_json(self):
        """config.json should be valid, indented JSON."""
        result = TrainingResult(
            method="er",
            config={"lr": 1e-3, "nested": {"a": 1}},
            history={"loss": [1.0]},
            metadata={"method": "er"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            save_run(result, run_dir)

            with open(os.path.join(run_dir, "config.json")) as f:
                raw = f.read()
            assert "\n" in raw  # indented
            parsed = json.loads(raw)
            assert parsed["lr"] == 1e-3

    def test_round_trip_with_weights(self):
        """Model weights survive save/load round-trip."""
        # Build a tiny model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, input_shape=(2,)),
            tf.keras.layers.Dense(1),
        ])
        model(tf.zeros((1, 2)))  # build

        original_weights = model.get_weights()

        result = TrainingResult(
            method="lr",
            config={"x": 1},
            history={"loss": [0.1]},
            metadata={"method": "lr"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            save_run(result, run_dir, policy=model)

            # Scramble weights
            model.set_weights([np.random.randn(*w.shape).astype(np.float32)
                               for w in original_weights])

            loaded = load_run(run_dir)
            loaded.load_weights(policy=model)

            restored_weights = model.get_weights()
            for orig, rest in zip(original_weights, restored_weights):
                np.testing.assert_allclose(orig, rest, atol=1e-6)

    def test_load_weights_without_run_dir_raises(self):
        """load_weights raises if run_dir not set."""
        result = TrainingResult(method="lr", config={}, history={})
        with pytest.raises(FileNotFoundError, match="run_dir"):
            result.load_weights(policy=tf.keras.Sequential())


# =====================================================================
# Dataset storage
# =====================================================================

class TestDatasetStorage:
    """Tests for save_dataset / load_dataset."""

    def test_trajectory_round_trip(self):
        """Trajectory dataset survives save/load."""
        data = {
            "s_endo_0": tf.constant(np.random.randn(10, 1).astype(np.float32)),
            "z_path":   tf.constant(np.random.randn(10, 5, 1).astype(np.float32)),
            "z_fork":   tf.constant(np.random.randn(10, 4, 1).astype(np.float32)),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = os.path.join(tmpdir, "ds")
            save_dataset(data, ds_dir, fmt="trajectory",
                         env_config={"theta": 0.7})

            loaded = load_dataset(ds_dir, fmt="trajectory")
            for key in data:
                np.testing.assert_allclose(
                    loaded[key].numpy(), data[key].numpy(), atol=1e-6)

    def test_flat_round_trip(self):
        """Flat dataset survives save/load."""
        data = {
            "s_endo":      tf.random.normal((20, 1)),
            "z":           tf.random.normal((20, 1)),
            "z_next_main": tf.random.normal((20, 1)),
            "z_next_fork": tf.random.normal((20, 1)),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = os.path.join(tmpdir, "ds")
            save_dataset(data, ds_dir, fmt="flat")

            loaded = load_dataset(ds_dir, fmt="flat")
            assert set(loaded.keys()) == set(data.keys())
            for key in data:
                np.testing.assert_allclose(
                    loaded[key].numpy(), data[key].numpy(), atol=1e-6)

    def test_manifest_created(self):
        """save_dataset creates manifest.json with shapes and keys."""
        data = {"x": tf.constant([[1.0, 2.0], [3.0, 4.0]])}
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = os.path.join(tmpdir, "ds")
            save_dataset(data, ds_dir, fmt="flat",
                         env_config={"model": "basic"})

            manifest = load_manifest(ds_dir)
            assert "flat" in manifest
            assert manifest["flat"]["keys"] == ["x"]
            assert manifest["flat"]["shapes"]["x"] == [2, 2]
            assert manifest["env_config"]["model"] == "basic"

    def test_both_formats_in_same_dir(self):
        """Can save both trajectory and flat in the same directory."""
        traj = {"s_endo_0": tf.constant([[1.0]])}
        flat = {"s_endo": tf.constant([[2.0]])}
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = os.path.join(tmpdir, "ds")
            save_dataset(traj, ds_dir, fmt="trajectory")
            save_dataset(flat, ds_dir, fmt="flat")

            manifest = load_manifest(ds_dir)
            assert "trajectory" in manifest
            assert "flat" in manifest

            loaded_traj = load_dataset(ds_dir, fmt="trajectory")
            loaded_flat = load_dataset(ds_dir, fmt="flat")
            np.testing.assert_allclose(
                loaded_traj["s_endo_0"].numpy(), [[1.0]])
            np.testing.assert_allclose(
                loaded_flat["s_endo"].numpy(), [[2.0]])

    def test_invalid_fmt_raises(self):
        """Invalid format string raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="fmt must be"):
                save_dataset({}, tmpdir, fmt="invalid")

    def test_load_missing_file_raises(self):
        """Loading a format that wasn't saved raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_dataset(tmpdir, fmt="trajectory")
