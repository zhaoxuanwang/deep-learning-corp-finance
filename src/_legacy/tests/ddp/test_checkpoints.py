"""
Tests for DDP checkpoint save/load round-trip integrity.
"""
import json

import numpy as np
import pytest

from src.ddp.checkpoints import save_ddp_solution, load_ddp_solution


def test_save_load_round_trip(tmp_path):
    """Saved arrays and metadata survive a save/load round trip."""
    value = np.random.randn(5, 10).astype(np.float32)
    policy_k = np.random.randn(5, 10).astype(np.float32)
    policy_b = np.random.randn(5, 10, 8).astype(np.float32)

    ckpt_dir = save_ddp_solution(
        save_dir=str(tmp_path),
        model_name="basic",
        scenario_name="test",
        solver_name="vfi",
        value=value,
        policy_k=policy_k,
        policy_b=policy_b,
        metrics={"iterations": 100, "max_diff": 1e-6},
        verbose=False,
    )

    loaded = load_ddp_solution(ckpt_dir)
    np.testing.assert_array_equal(loaded["arrays"]["value"], value)
    np.testing.assert_array_equal(loaded["arrays"]["policy_k"], policy_k)
    np.testing.assert_array_equal(loaded["arrays"]["policy_b"], policy_b)
    assert loaded["metadata"]["model_name"] == "basic"
    assert loaded["metadata"]["metrics"]["iterations"] == 100


def test_save_refuses_overwrite_by_default(tmp_path):
    """Second save to same path raises FileExistsError."""
    kwargs = dict(
        save_dir=str(tmp_path),
        model_name="basic",
        scenario_name="test",
        solver_name="vfi",
        value=np.zeros((3, 3)),
        policy_k=np.zeros((3, 3)),
        verbose=False,
    )
    save_ddp_solution(**kwargs)
    with pytest.raises(FileExistsError):
        save_ddp_solution(**kwargs)


def test_save_overwrite_allowed(tmp_path):
    """overwrite=True replaces existing checkpoint."""
    kwargs = dict(
        save_dir=str(tmp_path),
        model_name="basic",
        scenario_name="test",
        solver_name="vfi",
        value=np.ones((3, 3)),
        policy_k=np.ones((3, 3)),
        verbose=False,
    )
    save_ddp_solution(**kwargs)
    save_ddp_solution(**kwargs, overwrite=True)  # Should not raise
    loaded = load_ddp_solution(
        str(tmp_path / "ddp" / "basic" / "test" / "vfi")
    )
    np.testing.assert_array_equal(loaded["arrays"]["value"], np.ones((3, 3)))


def test_load_missing_raises(tmp_path):
    """Loading from nonexistent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_ddp_solution(str(tmp_path / "nonexistent"))
