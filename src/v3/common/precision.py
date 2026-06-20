"""Central precision, Keras, and device policy for v3.

Import this module before any heavy TensorFlow use so the legacy-Keras flag is
pinned. v3 runs on a stack where Keras 3 is installed alongside TensorFlow 2.16;
TensorFlow-Probability needs Keras 2 semantics, so ``TF_USE_LEGACY_KERAS=1`` is
forced here (and in ``src/v3/__init__.py``) before TensorFlow is imported.

Two-tier precision policy (see the v3 plan):

* Network / surrogate training runs in float32 (:data:`TF_FLOAT_NET`) on the GPU
  (Apple Metal).
* The numerically sensitive parts (VFI, grid-refinement linear solve,
  Levenberg-Marquardt, weighting-matrix Cholesky) run in float64
  (:data:`TF_FLOAT_NUM`) on the CPU, because Apple Metal has no float64 GPU
  kernels.

Cast at every boundary; never let the two tiers mix silently (review ENV-2).
"""
from __future__ import annotations

import os

# Pin legacy Keras BEFORE TensorFlow is imported (TFP + Keras 3 are incompatible,
# review ENV-1). ``setdefault`` so an explicit caller choice is respected.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import platform  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# Two-tier dtypes.
TF_FLOAT_NET = tf.float32  # network training (GPU/Metal friendly)
TF_FLOAT_NUM = tf.float64  # VFI / linear solves / LM / Cholesky (CPU)
NP_FLOAT_NET = np.float32
NP_FLOAT_NUM = np.float64

CPU = "/CPU:0"


def as_float(x, dtype):
    """Convert ``x`` to ``dtype`` without an accidental float32 detour.

    ``tf.cast(0.6, tf.float64)`` first turns the Python float into a float32 tensor
    (TF's default), then casts, so the float64 value is only float32-accurate
    (~2e-8 error). For Python/list scalars we build the constant directly in the
    target dtype; existing tensors/arrays are cast as usual.
    """
    if isinstance(x, (bool, int, float, list, tuple)):
        return tf.constant(x, dtype=dtype)
    return tf.cast(x, dtype)


def as_net_float(x):
    """Cast ``x`` to the network-training dtype (float32)."""
    return tf.cast(x, TF_FLOAT_NET)


def as_num_float(x):
    """Cast ``x`` to the numeric dtype (float64)."""
    return tf.cast(x, TF_FLOAT_NUM)


def cpu_device():
    """Context manager placing ops on the CPU (required for float64 on Metal)."""
    return tf.device(CPU)


def is_apple_silicon() -> bool:
    """True on Apple M-series hardware (where float64 is CPU-only)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def make_adam(learning_rate):
    """Adam optimizer, preferring the legacy implementation.

    The post-2.11 Keras Adam runs slowly on Apple M1/M2 (review OPT-4); the
    legacy optimizer is recommended there and is stable under
    ``TF_USE_LEGACY_KERAS``. Falls back to the standard Adam if legacy is absent.
    """
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    except (AttributeError, ImportError):  # pragma: no cover - environment dependent
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def assert_all_finite(x, name: str = "tensor"):
    """Raise if ``x`` contains any NaN/Inf (review NUM-1 guard); returns ``x``."""
    x = tf.convert_to_tensor(x)
    tf.debugging.assert_all_finite(x, message=f"{name} has non-finite values")
    return x


def use_cpu_only() -> bool:
    """Hide the GPU so all ops run on the CPU. Returns True on success.

    On Apple Metal, float64 ops can silently lose precision (errors ~1e-8 instead
    of ~1e-16, review HW-1), and TF's placement heuristic sometimes routes batched
    float64 ops to Metal even under a ``tf.device('/CPU:0')`` context. The whole
    float64 numeric tier (Tauchen, VFI, grid refinement, Levenberg-Marquardt,
    weighting Cholesky) therefore runs CPU-only for correctness. Must be called
    before any op initializes the GPU.
    """
    return configure_devices("cpu")


def configure_devices(mode: str = "auto") -> str:
    """Set the device policy for the float64 numeric tier. Returns the chosen mode.

    The code is device-agnostic; only GPU *visibility* changes. float64 numeric ops
    (VFI, grid refinement, LM, weighting) run on whatever device is visible:

    * ``"gpu"`` (default on CUDA): keep the GPU visible; float64 runs on the GPU.
    * ``"cpu"``: hide the GPU; everything runs on the CPU. Required on Apple Metal,
      which has no reliable float64 GPU kernels (review HW-1), and for the
      deterministic regression slice.
    * ``"auto"``: CUDA GPU -> ``"gpu"``; Apple Metal or no GPU -> ``"cpu"``.

    Must be called before any op initializes the GPU.
    """
    has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
    if mode == "auto":
        mode = "gpu" if (has_gpu and not is_apple_silicon()) else "cpu"
    try:
        if mode == "cpu":
            tf.config.set_visible_devices([], "GPU")
        else:  # "gpu": leave the GPU visible; enable memory growth where supported.
            for dev in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(dev, True)
                except (RuntimeError, ValueError):
                    pass
        return mode
    except RuntimeError:
        # Devices already initialized; cannot change visibility now.
        return mode


def configure_determinism(disable_gpu: bool = True) -> None:
    """Pin deterministic, single-threaded CPU execution for golden tests.

    Metal RNG is nondeterministic and some ops lack GPU kernels (review
    HW-1/REPRO-1), so exact-reproducibility slices run CPU-only. Call before any
    GPU is initialized.
    """
    if disable_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass  # devices already initialized
    tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # threading already configured (shared process); op-determinism still applies


def silence_logging() -> None:
    """Quiet TensorFlow's C++ and Python loggers."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
