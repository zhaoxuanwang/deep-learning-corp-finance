"""Run-artifact management and checkpointing (mirrors the v2 evaluation pattern).

Writes a per-run directory under ``outputs/v3/<experiment>/<tag>/`` with a
manifest, figures, arrays, and summary tables, plus version-robust network
checkpoints via ``tf.train.Checkpoint``.
"""
