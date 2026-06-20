"""v3: TensorFlow port of the Duarte and Fonseca (2026) structural-estimation pipeline.

This package is fully self-contained and never imports from ``src.v2`` or
``src._legacy`` (enforced by ``src/v3/tests/test_isolation.py``). It depends only on
TensorFlow + NumPy (no TensorFlow-Probability), so it runs on a modern stack (recent
TF, NumPy 2, CUDA / Colab) without version pinning, and on Apple Silicon.

Legacy Keras is an Apple-Silicon-only performance choice; the flag is set there before
any TensorFlow import. See ``src/v3/common/precision.py``.
"""
import os
import platform

# Apple-Silicon only (legacy Adam perf, OPT-4); native Keras elsewhere. Set before TF import.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

__all__ = ["__version__"]
__version__ = "0.0.1"
