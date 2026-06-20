"""v3: TensorFlow/TFP port of the Duarte and Fonseca (2026) structural-estimation pipeline.

This package is fully self-contained and never imports from ``src.v2`` or
``src._legacy`` (enforced by ``src/v3/tests/test_isolation.py``).

The legacy-Keras flag is pinned at import time, before any TensorFlow import, so
TensorFlow-Probability (which needs Keras 2 semantics) works on a stack where
Keras 3 is installed alongside TensorFlow 2.16. See ``src/v3/common/precision.py``.
"""
import os

# Must be set before TensorFlow is first imported anywhere in the process.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

__all__ = ["__version__"]
__version__ = "0.0.1"
